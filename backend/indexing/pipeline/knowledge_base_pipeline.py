from pathlib import Path
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

from indexing.pipeline.ingestor import FileIngestor
from indexing.pipeline.document_to_image import DocumentToImageConverter
from indexing.pipeline.vlm_chunker import VLMSemanticChunker
from indexing.pipeline.json_handler import ChunkJSONHandler

from logger import logger

# Load environment variables
load_dotenv()


class KnowledgeBasePipeline:
    """Complete pipeline for building knowledge base from PDF documents"""

    def __init__(
        self,
        data_folder: str,
        output_folder: str = "output",
        temp_images_folder: str = "temp_images",
        vlm_model: str = "gpt-4.1-nano-2025-04-14",
        api_key: Optional[str] = None
    ):
        """
        Initialize the pipeline

        Args:
            data_folder: Folder containing source documents
            output_folder: Folder to save JSON outputs
            temp_images_folder: Folder for temporary images
            vlm_model: VLM model to use
            api_key: API key for VLM (if not in environment)
        """
        self.data_folder = data_folder
        self.output_folder = output_folder

        # Initialize components
        self.ingestor = FileIngestor(data_folder)
        self.image_converter = DocumentToImageConverter(temp_images_folder)
        self.vlm_chunker = VLMSemanticChunker(
            model=vlm_model,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.json_handler = ChunkJSONHandler(output_folder)

        logger.debug(f"Pipeline initialized | Data: {data_folder} | Output: {output_folder} | VLM: {vlm_model}")

    def process_single_document(self, file_path: str) -> Optional[str]:
        """
        Process a single document through the pipeline

        Args:
            file_path: Path to the document

        Returns:
            Path to saved JSON file, or None if processing failed
        """
        file_path_obj = Path(file_path)
        logger.info(f"Processing: {file_path_obj.name}")

        # Step 1: Load and extract text
        logger.debug("Loading document and extracting text")
        documents = self.ingestor.load_file(file_path_obj)
        if not documents:
            logger.error(f"Failed to load {file_path}")
            return None

        # Step 2: Convert to images
        logger.debug("Converting document to images")
        images_data = self.image_converter.convert_document(str(file_path_obj))
        if not images_data:
            logger.error(f"Failed to convert {file_path} to images")
            return None

        # Step 3: Prepare pages data for VLM
        logger.debug("Preparing pages for VLM processing")
        pages_data = []

        for i, (doc, img_data) in enumerate(zip(documents, images_data)):
            pages_data.append({
                'page_number': img_data['page_number'],
                'text': doc.page_content,
                'image_path': img_data['image_path'],
                'source': file_path_obj.name
            })

        # Step 4: VLM chunking and save (incremental saving happens inside)
        logger.debug("Creating semantic chunks with VLM")
        output_filename = file_path_obj.stem
        json_path = self.vlm_chunker.process_document_pages(
            pages_data=pages_data,
            output_filename=output_filename
        )

        logger.info(f"Successfully processed {file_path_obj.name}")
        return json_path

    def process_all_documents(
        self,
        cleanup_images: bool = True,
        save_combined: bool = True
    ) -> Dict[str, str]:
        """
        Process all documents in the data folder

        Args:
            cleanup_images: Whether to delete temporary images after processing
            save_combined: Whether to save a combined JSON with all documents

        Returns:
            Dictionary mapping document names to their JSON output paths
        """
        logger.info("Processing all documents")

        # Get file count
        file_count = self.ingestor.get_file_count()
        logger.info(f"Found {file_count} PDF files")

        # Load all documents
        logger.debug("Loading all documents")
        all_documents = self.ingestor.load_all_files()

        results = {}
        all_chunks = {}

        # Process each document
        for doc_name, documents in all_documents.items():
            file_path = None

            # Find the file path
            for file_path_obj in Path(self.data_folder).rglob('*'):
                if file_path_obj.name == doc_name:
                    file_path = str(file_path_obj)
                    break

            if file_path:
                json_path = self.process_single_document(file_path)
                if json_path:
                    results[doc_name] = json_path

                    # Load chunks for combined file
                    if save_combined:
                        chunks = self.json_handler.load_chunks(json_path)
                        all_chunks[doc_name] = chunks

        # Save combined file
        if save_combined and all_chunks:
            logger.debug("Saving combined JSON file")
            combined_path = self.json_handler.save_all_documents(
                all_chunks,
                combined_filename="knowledge_base_all_chunks"
            )
            results["__combined__"] = combined_path

        # Cleanup
        if cleanup_images:
            logger.debug("Cleaning up temporary images")
            self.image_converter.cleanup()

        # Summary
        logger.info(f"Pipeline complete: {len(results)} documents processed")
        logger.info(f"Output saved to: {self.output_folder}")

        return results

    def get_pipeline_statistics(self) -> Dict:
        """Get statistics about the pipeline output"""
        stats = {
            "output_folder": str(self.output_folder),
            "json_files": []
        }

        for json_file in Path(self.output_folder).glob("*.json"):
            file_stats = self.json_handler.get_statistics(str(json_file))
            stats["json_files"].append({
                "filename": json_file.name,
                "stats": file_stats
            })

        return stats


