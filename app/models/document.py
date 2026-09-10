from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=3, ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
