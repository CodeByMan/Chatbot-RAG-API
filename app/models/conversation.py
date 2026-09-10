from pydantic import BaseModel, Field
from typing import List, Optional

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ConversationRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[str] = Field(default=None, max_length=100)
    top_k: Optional[int] = Field(default=3, ge=1, le=20)

class ConversationResponse(BaseModel):
    answer: str
    sources: List[dict]
    conversation_id: str
