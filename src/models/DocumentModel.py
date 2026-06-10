from pydantic import BaseModel
from typing import Any, Optional


class DocumentModel(BaseModel):
    id: str
    filename: str
    source: str
    file_type: str
    document_type: str = "general"
    upload_date: str
    checksum: str
    chunk_count: int
    parent_chunk_count: int = 0
    metadata: dict[str, Any] = {}
