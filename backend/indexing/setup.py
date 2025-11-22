import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from logger import logger
# Load environment variables
load_dotenv()

from pipeline.embedding_generator import EmbeddingGenerator
from pipeline.chroma_storage import ChromaStorage




def process_single_pdf_worker(args):
    """Worker function to process a single PDF"""
    pdf_path, kb_pipeline = args

    try:
        json_path = kb_pipeline.process_single_document(str(pdf_path))
        if json_path:
            return {"status": "success", "pdf": pdf_path.name, "json_path": json_path}
        else:
            return {"status": "failed", "pdf": pdf_path.name, "error": "Chunking failed"}
    except Exception as e:
        return {"status": "failed", "pdf": pdf_path.name, "error": str(e)}


def setup_rag_system():
    """Setup complete RAG system from PDFs to ChromaDB"""

    # Configuration
    data_folder = Path("D:/Cycls_workingspace/RAG/data/clean")
    output_folder = "D:/Cycls_workingspace/RAG/output"
    temp_images_folder = "D:/Cycls_workingspace/RAG/temp_images"
    chroma_db_folder = "D:/Cycls_workingspace/RAG/chroma_db"
    collection_name = "procurement_kb"
    api_key = os.getenv("OPENAI_API_KEY")

    logger.info("Starting RAG system setup with parallel processing")

    # Get all PDF files
    pdf_files = list(data_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in data folder")
        return

    logger.info(f"Found {len(pdf_files)} PDF files | Max workers: 3")

    # Initialize knowledge base pipeline
    from pipeline.knowledge_base_pipeline import KnowledgeBasePipeline

    kb_pipeline = KnowledgeBasePipeline(
        data_folder=str(data_folder),
        output_folder=output_folder,
        temp_images_folder=temp_images_folder,
        vlm_model="gpt-4.1-mini-2025-04-14",
        api_key=api_key
    )

    # Step 1: Create chunks with VLM (parallel)
    logger.info("Step 1/3: Creating semantic chunks with VLM")
    successful_jsons = []
    failed_pdfs = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_pdf = {
            executor.submit(process_single_pdf_worker, (pdf_file, kb_pipeline)): pdf_file
            for pdf_file in pdf_files
        }

        for future in as_completed(future_to_pdf):
            result = future.result()
            if result["status"] == "success":
                successful_jsons.append(result["json_path"])
                logger.info(f"Processed: {result['pdf']}")
            else:
                failed_pdfs.append(result["pdf"])
                logger.error(f"Failed: {result['pdf']} - {result['error']}")

    logger.info(f"Chunking complete: {len(successful_jsons)}/{len(pdf_files)} PDFs")

    if not successful_jsons:
        logger.error("No PDFs processed successfully")
        return

    # Step 2: Initialize components
    logger.info("Step 2/3: Generating embeddings")

    embedding_gen = EmbeddingGenerator(
        model="text-embedding-3-small",
        api_key=api_key
    )

    chroma_storage = ChromaStorage(
        collection_name=collection_name,
        persist_directory=chroma_db_folder,
        embedding_model="text-embedding-3-small",
        api_key=api_key
    )

    # Clear existing data
    logger.info("Clearing existing ChromaDB collection")
    chroma_storage.clear_collection()

    # Step 3: Generate embeddings and store
    logger.info("Step 3/3: Storing chunks in ChromaDB")
    total_chunks = 0

    for i, json_path in enumerate(successful_jsons, 1):
        doc_name = Path(json_path).stem
        try:
            chunks_with_embeddings = embedding_gen.process_knowledge_base(json_path)
            chroma_storage.add_chunks(chunks_with_embeddings)
            total_chunks += len(chunks_with_embeddings)
            logger.info(f"[{i}/{len(successful_jsons)}] {doc_name}: {len(chunks_with_embeddings)} chunks")
        except Exception as e:
            logger.error(f"[{i}/{len(successful_jsons)}] {doc_name}: {str(e)}")
            continue

    # Cleanup
    logger.info("Cleaning up temporary images")
    kb_pipeline.image_converter.cleanup()

    # Summary
    info = chroma_storage.get_collection_info()
    logger.info("RAG SYSTEM READY")
    logger.info(f"PDFs processed: {len(successful_jsons)}/{len(pdf_files)}")
    if failed_pdfs:
        logger.warning(f"Failed PDFs: {', '.join(failed_pdfs)}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"ChromaDB collection: {info['name']}")
    logger.info(f"Chunks in database: {info['count']}")
    logger.info(f"Database location: {info['persist_directory']}")


if __name__ == "__main__":
    setup_rag_system()
