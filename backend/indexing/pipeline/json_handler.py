import json
from pathlib import Path
from typing import List, Dict
from indexing.schemas.page_chunk import Page_chunks


class ChunkJSONHandler:
    """Handles saving and loading chunks to/from JSON files"""

    def __init__(self, output_folder: str = "output"):
        """
        Initialize JSON handler

        Args:
            output_folder: Folder to save JSON files
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def save_chunks(
        self,
        chunks: List[Page_chunks],
        filename: str,
        source_document: str
    ) -> str:
        """
        Save chunks to a JSON file

        Args:
            chunks: List of Page_chunks objects
            filename: Name for the output JSON file
            source_document: Original document name

        Returns:
            Path to saved JSON file
        """
        # Convert Pydantic models to dict
        chunks_data = {
            "source_document": source_document,
            "total_pages": len(chunks),
            "total_chunks": sum(len(page.chunks) for page in chunks),
            "pages": [page.model_dump() for page in chunks]
        }

        # Save to JSON
        output_path = self.output_folder / f"{filename}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        print(f"Saved chunks to {output_path}")
        return str(output_path)

    def load_chunks(self, json_path: str) -> List[Page_chunks]:
        """
        Load chunks from a JSON file

        Args:
            json_path: Path to JSON file

        Returns:
            List of Page_chunks objects
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert back to Pydantic models
        chunks = [Page_chunks(**page_data) for page_data in data['pages']]

        return chunks

    def save_all_documents(
        self,
        documents_chunks: Dict[str, List[Page_chunks]],
        combined_filename: str = "all_chunks"
    ) -> str:
        """
        Save chunks from all documents to a single JSON file

        Args:
            documents_chunks: Dictionary mapping document names to their chunks
            combined_filename: Name for the combined output file

        Returns:
            Path to saved JSON file
        """
        all_data = {
            "total_documents": len(documents_chunks),
            "documents": {}
        }

        for doc_name, chunks in documents_chunks.items():
            all_data["documents"][doc_name] = {
                "total_pages": len(chunks),
                "total_chunks": sum(len(page.chunks) for page in chunks),
                "pages": [page.model_dump() for page in chunks]
            }

        # Save to JSON
        output_path = self.output_folder / f"{combined_filename}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        print(f"Saved all documents to {output_path}")
        return str(output_path)

    def get_statistics(self, json_path: str) -> Dict:
        """
        Get statistics from a saved JSON file

        Args:
            json_path: Path to JSON file

        Returns:
            Dictionary with statistics
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "documents" in data:
            # Combined file
            stats = {
                "total_documents": data["total_documents"],
                "documents": {}
            }
            for doc_name, doc_data in data["documents"].items():
                stats["documents"][doc_name] = {
                    "total_pages": doc_data["total_pages"],
                    "total_chunks": doc_data["total_chunks"]
                }
        else:
            # Single document file
            stats = {
                "source_document": data["source_document"],
                "total_pages": data["total_pages"],
                "total_chunks": data["total_chunks"]
            }

        return stats
