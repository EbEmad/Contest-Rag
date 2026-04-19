from pydantic import BaseModel, Field
from typing import Optional, List, Any
from bson.objectid import ObjectId
from datetime import datetime


class LearningRoadmap(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    student_id: str
    weak_topic_names: List[str]
    llm_explanation: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True