from pydantic import BaseModel, Field
from typing import Optional, List
from bson.objectid import ObjectId
from datetime import datetime

class StudentProfile(BaseModel):
    """Student profile for tracking learning progress"""
    id: Optional[ObjectId] = Field(None, alias="_id")
    user_id: str  # From auth system (email, username, etc.)
    full_name: str
    grade: int  # Current grade level (1-12)
    current_level: str = "BEGINNER"  # BEGINNER, INTERMEDIATE, ADVANCED
    enrolled_subjects: List[str] = []  # ["Mathematics", "Biology"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True
