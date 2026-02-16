from pydantic import BaseModel, Field
from typing import Optional, List
from bson.objectid import ObjectId
from datetime import datetime


class TopicPerformance(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    student_id: ObjectId
    topic_id: ObjectId
    accuracy: float = 0.0
    total_attempts: int = 0
    correct_answers: int = 0
    trend: str = "STABLE"  # IMPROVING, DECLINING, STABLE
    class Config:
        arbitrary_types_allowed = True



class LearningRoadmap(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    student_id:ObjectId
    weak_topic_ids: List[ObjectId]
    llm_explanation: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
    