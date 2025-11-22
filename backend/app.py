import cycls
import os
from pathlib import Path
from dotenv import load_dotenv

# Get script directory
script_dir = Path(__file__).parent
root_dir = script_dir.parent

load_dotenv(script_dir / ".env")

agent = cycls.Agent()


# RAG retrieval function
async def retrieve_context(query: str, n_results: int = 10):
    """Retrieve relevant chunks from ChromaDB"""
    from indexing.pipeline.chroma_storage import ChromaStorage
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
                f"[Source: {metadata['source']}, Page: {metadata['page_number']}]\n{doc}"
            )

    return "\n\n---\n\n".join(context_parts)


# LLM streaming function
async def llm(messages):
    """Call OpenAI with streaming"""
    import openai

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=messages,
        temperature=0.3,
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

    # Build RAG prompt
    system_prompt = {
        "role": "system",
        "content": """You are a helpful assistant for procurement and contracting policies.
Use the provided context to answer questions accurately.
If the context doesn't contain the answer, say so clearly.
Always cite the source and page number when referencing information.
"""
    }

    # Add context to user message
    rag_user_message = {
        "role": "user",
        "content": f"""Context from knowledge base:
{retrieved_context}

---

User Question: {user_message}

Please answer based on the context above. Cite sources with [Source: filename, Page: X]."""
    }

    messages = [system_prompt]

    if len(context.messages) > 1:
        messages.extend(context.messages[:-1])

    messages.append(rag_user_message)

    return await llm(messages)


if __name__ == "__main__":
    agent.local()
