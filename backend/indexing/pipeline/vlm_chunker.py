import base64
from pathlib import Path
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
import json
import uuid
from schemas.page_chunk import Page_chunks
from schemas.knowledge_base import KnowledgeBase
from logger import logger


class VLMSemanticChunker:
    """Uses VLM to create semantic chunks from document pages"""

    def __init__(
        self,
        model: str = "gpt-4.1",
        api_key: Optional[str] = None,
        output_folder: str = "output",
        timeout: int = 30,
        max_tokens: int = 15000
    ):
        self.model = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=2
        )
        self.parser = PydanticOutputParser(pydantic_object=Page_chunks)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_chunks_from_page(
        self,
        page_text: str,
        image_path: str,
        page_number: int,
        source_file: str
    ) -> Page_chunks:
        """
        Use VLM to create semantic chunks from a page

        Args:
            page_text: Extracted text from the page
            image_path: Path to the page image
            page_number: Actual page number
            source_file: Source filename

        Returns:
            Page_chunks object with semantic chunks
        """
        # Encode image
        base64_image = self.encode_image(image_path)

        # Create prompt for VLM - Optimized for RAG
        prompt = f"""You are an expert document analyst specializing in creating semantic chunks for a Retrieval-Augmented Generation (RAG) system.
### **INPUTS PROVIDED**
* **EXTRACTED TEXT:**
  {page_text}
* **PAGE IMAGE:**
---
## **YOUR TASK**
Analyze **both** the page image **and** the extracted text to produce **high-quality semantic chunks** optimized for knowledge retrieval.
---
## **CORE REQUIREMENTS**
### **1. FULL PAGE SUMMARY (IMAGE + TEXT)**
* Create a complete and accurate understanding of the page using **both** the image content and extracted text.
* NO new information may be added.
* Summaries must reflect only what appears on the page.
* Do Not omit any important details that are visible in the image but missing from the text.
* Do not Make alot of Chunks if the page is small and simple allways make sure one chunk cover a good part of the page.
---
### **2. SEMANTIC CHUNKING**
Each chunk must:
* Be **clean, human-readable, and immediately useful** for an LLM.
* Be **standalone**, so it can be retrieved independently by a RAG system.
* Not mix multiple unrelated ideas in the same chunk.
* Do NOT create chunks for meaningless or low-value text, such as:Repeated headers,Confidential labels,Page numbers,Empty sections,Decorative elements
* Make the chunks have a good level of detail, avoiding being too broad or too narrow.
* Make sure that the chunks cover all important aspects of the page without redundancy.
---
### **3. LANGUAGE PRESERVATION (STRICT)**
* **Preserve the language exactly as it appears in the document.**
* If the text is **Arabic**, all chunks **must be in Arabic**.
* If the text is **English**, all chunks **must be in English**.
* If the document is **mixed**, each chunk must remain in **its original language segment**.
* If the test is **Arbic** the metadata short_note must also be in **Arabic**.
* **NO translation, NO rewriting into another language, and NO language mixing.**
---
## **OUTPUT FORMAT**
{self.parser.get_format_instructions()}
"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        )

        # Get VLM response
        response = self.model.invoke([message])

        # Parse response to Pydantic model
        try:
            chunks = self.parser.parse(response.content)

            chunks.page_number = page_number

            # Update metadata and generate unique IDs for each chunk
            for chunk in chunks.chunks:
                chunk.metadata.page_number = page_number
                chunk.metadata.source = source_file
                # Generate unique UUID for each chunk
                chunk.chunk_id = str(uuid.uuid4())

            return chunks
        except Exception as e:
            logger.error(f"Error parsing VLM response for page {image_path}: {e}")
            return None 
            
    def process_document_pages(
        self,
        pages_data: List[dict],
        output_filename: str
    ) -> str:
        """
        Process multiple pages and save incrementally to knowledge base JSON

        Args:
            pages_data: List of dicts with 'page_number', 'text', 'image_path', 'source'
            output_filename: Name for the output JSON file (without extension)

        Returns:
            Path to the saved knowledge base JSON file
        """
        output_path = self.output_folder / f"{output_filename}.json"

        # Initialize empty knowledge base
        knowledge_base = KnowledgeBase(all_chunks=[])

        # Save initial structure
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Processing {len(pages_data)} pages")

        # Process each page and save incrementally
        total_chunks = 0
        for page_data in pages_data:
            page_num = page_data['page_number']
            logger.debug(f"Processing page {page_num}/{len(pages_data)}")

            try:
                page_chunks = self.create_chunks_from_page(
                    page_text=page_data['text'],
                    image_path=page_data['image_path'],
                    page_number=page_data['page_number'],
                    source_file=page_data['source']
                )

                if page_chunks:
                    knowledge_base.all_chunks.append(page_chunks)
                    total_chunks += len(page_chunks.chunks)

                    # Save after each page
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(knowledge_base.model_dump(), f, indent=2, ensure_ascii=False)

                    logger.debug(f"Page {page_num}: {len(page_chunks.chunks)} chunks saved")
                else:
                    logger.warning(f"Page {page_num}: Failed to create chunks")

            except Exception as e:
                logger.error(f"Error on page {page_num}: {e}")
                continue

        logger.info(f"Complete: {total_chunks} chunks from {len(knowledge_base.all_chunks)} pages")
        logger.info(f"Saved to: {output_path}")

        return str(output_path)
