from pydantic import Field, BaseModel
from .page_chunk import Page_chunks


class KnowledgeBase(BaseModel):
    """Complete knowledge base containing all page chunks from a document"""
    all_chunks: list[Page_chunks] = Field(..., description="List of all page chunks in the knowledge base")