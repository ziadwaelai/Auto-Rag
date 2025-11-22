from pydantic import Field, BaseModel


class Metadata(BaseModel):
    """Metadata information for a document chunk"""
    page_number: int = Field(..., description="The page number where this chunk appears in the source document")
    source: str = Field(..., description="The source file name of the document")
    type: str = Field(..., description="The type of content (e.g., 'text', 'table', 'header', 'footer','summary','rule')")
    short_note: str = Field(..., description="A brief note about the chunk content")

class Chunks(BaseModel):
    """Individual text chunk extracted from a document"""
    chunk_id: str = Field(..., description="Unique identifier for this chunk like (page_number_chunkIndex)")
    content: str = Field(..., description="The actual text content of the chunk")
    metadata: Metadata = Field(..., description="Associated metadata for this chunk")


class Page_chunks(BaseModel):
    """Collection of all chunks from a single page"""
    page_number: int = Field(..., description="The page number for all chunks in this collection")
    chunks: list[Chunks] = Field(..., description="List of all text chunks extracted from this page")


