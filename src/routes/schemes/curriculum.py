from pydantic import BaseModel, Field
from typing import Optional


class SubjectCreateRequest(BaseModel):
    name: str
    grade: int = Field(ge=1, le=12)


class ChapterCreateRequest(BaseModel):
    subject_id: str
    name: str
    order: int = Field(ge=1)


class TopicCreateRequest(BaseModel):
    chapter_id: str
    subject_id: str
    name: str
    order: int = Field(ge=1)
