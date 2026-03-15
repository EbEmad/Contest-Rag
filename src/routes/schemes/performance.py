from pydantic import BaseModel
from typing import List, Optional


class RoadmapRequest(BaseModel):
    student_id: str
    student_name: Optional[str] = None
    grade: Optional[int] = None
    student_level: str = "BEGINNER"
    weak_topic_names: Optional[List[str]] = None
    weak_topic_ids: Optional[List[str]] = None


class TeacherDashboardRequest(BaseModel):
    topic_ids: List[str]
