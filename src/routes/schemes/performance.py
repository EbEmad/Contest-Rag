from pydantic import BaseModel, Field
from typing import List, Optional


class RoadmapRequest(BaseModel):
    student_id: str
    student_name: str
    grade: int = Field(ge=1, le=12)
    student_level: str = "BEGINNER"
    weak_topic_names: List[str]
