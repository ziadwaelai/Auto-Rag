from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

from knowledge_base_pipeline import KnowledgeBasePipeline
from embedding_generator import EmbeddingGenerator
from chroma_storage import ChromaStorage
from logger import logger
# Load environment variables
load_dotenv()


class CompleteRAGPipeline:
    """Complete end-to-end RAG pipeline"""

    def __init__(
        self,
        data_folder: str,
        output_folder: str = "output",
        temp_images_folder: str = "temp_images",
        chroma_db_folder: str = "chroma_db",
        collection_name: str = "knowledge_base",
        vlm_model: str = "gpt-4.1-nano-2025-04-14",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize complete pipeline

        Args:
            data_folder: Folder containing PDF files
            output_folder: Folder for JSON outputs
            temp_images_folder: Folder for temporary images
            chroma_db_folder: Folder for ChromaDB storage
            collection_name: Name for ChromaDB collection
            vlm_model: VLM model for chunking
            embedding_model: OpenAI embedding model
            api_key: OpenAI API key
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Initialize components
        self.kb_pipeline = KnowledgeBasePipeline(
            data_folder=data_folder,
            output_folder=output_folder,
            temp_images_folder=temp_images_folder,
            vlm_model=vlm_model,
            api_key=self.api_key
        )

        self.embedding_gen = EmbeddingGenerator(
            model=embedding_model,
            api_key=self.api_key
        )

        self.chroma_storage = ChromaStorage(
            collection_name=collection_name,
            persist_directory=chroma_db_folder,
            embedding_model=embedding_model,
            api_key=self.api_key
        )

        self.output_folder = Path(output_folder)

    def process_single_pdf(
        self,
        pdf_path: str,
        cleanup_images: bool = True
    ) -> str:
        """
        Process single PDF through complete pipeline

        Args:
            pdf_path: Path to PDF file
            cleanup_images: Whether to cleanup temporary images

        Returns:
            Status message
        """
        logger.info("COMPLETE RAG PIPELINE - SINGLE PDF")

        # Step 1: Create chunks with VLM
        logger.info("[1/3] Creating knowledge base chunks...")
        json_path = self.kb_pipeline.process_single_document(pdf_path)

        if not json_path:
            return "Failed to create knowledge base"

        # Step 2: Generate embeddings
        logger.info("[2/3] Generating embeddings...")
        chunks_with_embeddings = self.embedding_gen.process_knowledge_base(json_path)

        # Step 3: Store in ChromaDB
        logger.info("[3/3] Storing in ChromaDB...")
        self.chroma_storage.add_chunks(chunks_with_embeddings)

        # Cleanup
        if cleanup_images:
            self.kb_pipeline.image_converter.cleanup()

        # Summary
        info = self.chroma_storage.get_collection_info()
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Knowledge Base: {json_path}")
        logger.info(f"Chunks in ChromaDB: {info['count']}")
        logger.info(f"ChromaDB Location: {info['persist_directory']}")
        return "Success"

    def process_all_pdfs(
        self,
        cleanup_images: bool = True,
        clear_existing: bool = False
    ) -> str:
        """
        Process all PDFs in data folder

        Args:
            cleanup_images: Whether to cleanup temporary images
            clear_existing: Whether to clear existing ChromaDB data

        Returns:
            Status message
        """
        logger.info("COMPLETE RAG PIPELINE - ALL PDFs")

        # Clear existing data if requested
        if clear_existing:
            logger.info("Clearing existing ChromaDB collection...")
            self.chroma_storage.clear_collection()

        # Step 1: Create knowledge bases
        logger.info("[1/3] Creating knowledge base chunks for all PDFs...")
        kb_results = self.kb_pipeline.process_all_documents(
            cleanup_images=False,  # Don't cleanup yet
            save_combined=False
        )

        if not kb_results:
            return "No PDFs processed"

        # Step 2 & 3: Process each KB JSON
        total_chunks = 0
        for doc_name, json_path in kb_results.items():
            logger.info(f"[2-3/3] Processing: {doc_name}")

            # Generate embeddings
            chunks_with_embeddings = self.embedding_gen.process_knowledge_base(json_path)

            # Store in ChromaDB
            self.chroma_storage.add_chunks(chunks_with_embeddings)

            total_chunks += len(chunks_with_embeddings)

        # Cleanup
        if cleanup_images:
            self.kb_pipeline.image_converter.cleanup()

        # Summary
        info = self.chroma_storage.get_collection_info()
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Processed PDFs: {len(kb_results)}")
        logger.info(f"Total Chunks: {total_chunks}")
        logger.info(f"ChromaDB Collection: {info['name']}")
        logger.info(f"Chunks in DB: {info['count']}")
        logger.info(f"ChromaDB Location: {info['persist_directory']}")

        return "Success"

    def query(self, query_text: str, n_results: int = 5):
        """
        Query the knowledge base

        Args:
            query_text: Search query
            n_results: Number of results

        Returns:
            Query results
        """
        logger.info(f"Searching for: '{query_text}'")
        results = self.chroma_storage.query(
            query_text=query_text,
            n_results=n_results
        )

        logger.info(f"Found {len(results['documents'][0])} results:")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            logger.info(f"[{i}] Source: {metadata['source']} | Page: {metadata['page_number']} | Type: {metadata['type']}")
            logger.info(f"Content: {doc[:200]}..." if len(doc) > 200 else f"Content: {doc}")

        return results

