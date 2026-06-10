from pydantic import BaseModel
from typing import Any, Optional


class ChunkModel(BaseModel):
    chunk_id: str
    document_id: str
    parent_chunk_id: Optional[str] = None
    text: str
    metadata: dict[str, Any] = {}
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: Optional[float] = None
    section: Optional[str] = None
    document_type: Optional[str] = None
