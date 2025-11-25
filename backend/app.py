import cycls
import os
import sys
# add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
from dotenv import load_dotenv
from parse_requirements import get_requirements_list
load_dotenv( "backend/.env")

agent = cycls.Agent(
    pip=get_requirements_list(),
    api_key=os.getenv("API_KEY"),
    copy=[
        "chroma_db",
        "backend/.env",
        "backend/indexing",
        "backend/logger"
    ]
)


# RAG retrieval function
async def retrieve_context(query: str, n_results: int = 10):
    """Retrieve relevant chunks from ChromaDB"""
    from backend.indexing.pipeline.chroma_storage import ChromaStorage
    from dotenv import load_dotenv
    load_dotenv("backend/.env")
    # Initialize ChromaDB
    chroma_storage = ChromaStorage(
        collection_name="procurement_kb",
        persist_directory="chroma_db",
        embedding_model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    results = chroma_storage.query(
        query_text=query,
        n_results=n_results
    )

    # Format context
    context_parts = []
    if results['documents'][0]:
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            context_parts.append(
                f" المصدر: {metadata['source']} | الصفحة: {metadata['page_number']}\n\n{doc}"
            )

    return "\n\n---\n\n".join(context_parts)


# LLM streaming function
async def llm(messages):
    """Call Groq with streaming"""
    from groq import AsyncGroq
    from dotenv import load_dotenv
    load_dotenv("backend/.env")

    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    response = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=True
    )

    async def event_stream():
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return event_stream()



@agent("agent", auth=False)
async def rag_agent(context):
    """
    RAG-powered conversational agent for procurement documents
    """
    # Get user's latest message
    user_message = context.messages[-1]["content"]

    # Retrieve relevant context
    retrieved_context = await retrieve_context(user_message, n_results=3)

    # Build RAG prompt in Arabic
    system_prompt = {
        "role": "system",
        "content": """أنت مساعد متخصص في سياسات المشتريات والعقود.
استخدم السياق المقدم للإجابة على الأسئلة بدقة.
إذا لم يحتوي السياق على الإجابة، قل ذلك بوضوح.
عند الإجابة، يجب عليك الاستشهاد بالمصدر والصفحة من السياق المقدم.
استخدم التنسيق التالي للاستشهاد: (المصدر: اسم_الملف، الصفحة: رقم_الصفحة)
قدم إجابة مفصلة وواضحة بالعربية مع ذكر المصادر.
"""
    }

    # Add context to user message
    rag_user_message = {
        "role": "user",
        "content": f"""السياق من قاعدة المعرفة:
{retrieved_context}

---

سؤال المستخدم: {user_message}

الرجاء الإجابة بناءً على السياق أعلاه."""
    }

    messages = [system_prompt]

    if len(context.messages) > 1:
        messages.extend(context.messages[:-1])

    messages.append(rag_user_message)

    return await llm(messages)


if __name__ == "__main__":
    agent.local()
