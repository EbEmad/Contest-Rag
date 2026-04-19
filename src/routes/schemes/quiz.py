from pydantic import BaseModel, Field
from typing import List, Optional
from models.db_schemes import QuestionType, DifficultyLevel


class QuizGenerateRequest(BaseModel):
    topic_id: str
    topic_name: str
    num_questions: int = Field(default=5, ge=1, le=20)
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    question_types: Optional[List[QuestionType]] = None
    student_id: str
    grade: Optional[int] = Field(default=None, ge=1, le=12)
    subject: Optional[str] = None


class QuizSubmitRequest(BaseModel):
    answers: dict  # {question_id: answer_str}
    topic_id: str
    student_id: str
