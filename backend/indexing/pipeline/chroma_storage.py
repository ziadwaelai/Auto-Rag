from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
import sys
from backend.logger import logger


class ChromaStorage:
    """Manages ChromaDB storage for knowledge base"""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: str = "chroma_db",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize ChromaDB storage

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: OpenAI embedding model for queries
            api_key: OpenAI API key
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI embeddings for queries
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Get or create collection (without embedding function to allow manual embeddings)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG knowledge base chunks with embeddings"}
        )

        self.collection_name = collection_name

    def add_chunks(self, chunks_with_embeddings: List[Dict]):
        """
        Add chunks with embeddings to ChromaDB

        Args:
            chunks_with_embeddings: List of dicts with id, content, metadata, embedding
        """
        if not chunks_with_embeddings:
            logger.warning("No chunks to add")
            return

        logger.debug(f"Adding {len(chunks_with_embeddings)} chunks to ChromaDB")

        # Prepare data for ChromaDB
        ids = [chunk['id'] for chunk in chunks_with_embeddings]
        documents = [chunk['content'] for chunk in chunks_with_embeddings]
        embeddings = [chunk['embedding'] for chunk in chunks_with_embeddings]
        metadatas = [chunk['metadata'] for chunk in chunks_with_embeddings]

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.debug(f"Successfully added {len(ids)} chunks to collection '{self.collection_name}'")

    def query(
        self,
        query_text: str = None,
        query_embedding: List[float] = None,
        n_results: int = 5,
        filter_metadata: Dict = None
    ):
        """
        Query the ChromaDB collection

        Args:
            query_text: Text to search for (will be embedded using OpenAI)
            query_embedding: Pre-computed embedding vector
            n_results: Number of results to return
            filter_metadata: Metadata filters (e.g., {'source': 'doc.pdf'})

        Returns:
            Query results
        """
        if query_embedding:
            # Use pre-computed embedding
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
        elif query_text:
            # Generate embedding using OpenAI (same model as stored embeddings)
            query_embedding = self.embeddings.embed_query(query_text)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
        else:
            raise ValueError("Either query_text or query_embedding must be provided")

        return results

    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": str(self.persist_directory)
        }

    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection_name)
        logger.debug(f"Deleted collection '{self.collection_name}'")

    def clear_collection(self):
        """Clear all data from collection"""
        # Delete and recreate
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG knowledge base chunks with embeddings"}
        )
        logger.debug(f"Cleared collection '{self.collection_name}'")
