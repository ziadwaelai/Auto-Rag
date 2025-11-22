import json
from typing import List, Dict
from schemas.knowledge_base import KnowledgeBase
from langchain_openai import OpenAIEmbeddings
from logger import logger


class EmbeddingGenerator:
    """Generates embeddings for knowledge base chunks"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None
    ):
        """
        Initialize embedding generator

        Args:
            model: OpenAI embedding model to use
            api_key: OpenAI API key
        """
        self.embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key
        )
        self.model_name = model

    def load_knowledge_base(self, json_path: str) -> KnowledgeBase:
        """Load knowledge base from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return KnowledgeBase(**data)

    def prepare_chunks_for_embedding(self, kb: KnowledgeBase) -> List[Dict]:
        """
        Prepare chunks with metadata for embedding

        Args:
            kb: KnowledgeBase object

        Returns:
            List of dicts with chunk info
        """
        chunks_data = []

        for page in kb.all_chunks:
            for chunk in page.chunks:
                chunks_data.append({
                    'id': chunk.chunk_id,
                    'content': chunk.content,
                    'metadata': {
                        'page_number': chunk.metadata.page_number,
                        'source': chunk.metadata.source,
                        'type': chunk.metadata.type,
                        'chunk_id': chunk.chunk_id
                    }
                })

        return chunks_data

    def generate_embeddings(self, chunks_data: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for all chunks

        Args:
            chunks_data: List of chunk dictionaries

        Returns:
            List of chunks with embeddings added
        """
        logger.debug(f"Generating embeddings for {len(chunks_data)} chunks")

        # Extract texts
        texts = [chunk['content'] for chunk in chunks_data]

        # Generate embeddings in batch
        embeddings = self.embeddings.embed_documents(texts)

        # Add embeddings to chunks
        for i, chunk in enumerate(chunks_data):
            chunk['embedding'] = embeddings[i]

        logger.debug(f"Generated {len(embeddings)} embeddings")
        return chunks_data

    def process_knowledge_base(self, json_path: str) -> List[Dict]:
        """
        Complete process: load KB, prepare chunks, generate embeddings

        Args:
            json_path: Path to knowledge base JSON

        Returns:
            List of chunks with embeddings
        """
        logger.debug(f"Processing knowledge base: {json_path}")

        # Load KB
        kb = self.load_knowledge_base(json_path)
        logger.debug(f"Loaded KB with {len(kb.all_chunks)} pages")

        # Prepare chunks
        chunks_data = self.prepare_chunks_for_embedding(kb)
        logger.debug(f"Prepared {len(chunks_data)} chunks")

        # Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks_data)

        return chunks_with_embeddings
