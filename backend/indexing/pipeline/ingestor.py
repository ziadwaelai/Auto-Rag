from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from logger import logger

class FileIngestor:
    """Ingests PDF files from a directory"""

    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
    }

    def __init__(self, data_folder: str):
        """
        Initialize the file ingestor

        Args:
            data_folder: Path to the folder containing documents
        """
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            raise ValueError(f"Data folder does not exist: {data_folder}")

    def load_file(self, file_path: Path) -> List[Document]:
        """
        Load a single PDF file

        Args:
            file_path: Path to the PDF file

        Returns:
            List of LangChain Document objects (one per page)
        """
        if file_path.suffix.lower() != '.pdf':
            logger.info(f"Skipping non-PDF file: {file_path}")
            return []

        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []

    def load_all_files(self) -> Dict[str, List[Document]]:
        """
        Load all PDF files from the data folder

        Returns:
            Dictionary mapping PDF file names to their documents
        """
        all_documents = {}

        for file_path in self.data_folder.rglob('*.pdf'):
            if file_path.is_file():
                logger.info(f"Loading: {file_path.name}")
                documents = self.load_file(file_path)
                if documents:
                    all_documents[file_path.name] = documents

        return all_documents

    def get_file_count(self) -> int:
        """Get count of PDF files in the data folder"""
        count = len(list(self.data_folder.rglob('*.pdf')))
        return count
