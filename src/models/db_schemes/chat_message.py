from pydantic import BaseModel, Field
from typing import Optional, Any
from bson.objectid import ObjectId
from datetime import datetime


class ChatMessage(BaseModel):
    """A single chat turn (user question or assistant answer) tied to a student + project."""
    id: Optional[Any] = Field(None, alias="_id")
    student_id: str
    project_id: str
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
