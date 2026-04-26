from pydantic import BaseModel,Field
from typing import Optional, List, Dict, Any
from bson.objectid import ObjectId
from datetime import datetime
from enum import Enum

class QuestionType(str, Enum):
    MCQ = "MCQ"
    TRUE_FALSE = "TRUE_FALSE"
    SHORT_ANSWER = "SHORT_ANSWER"

class DifficultyLevel(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"

class Question(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    topic_id: Any
    question_type:QuestionType
    question_text:str
    options: Optional[List[str]] = None
    correct_answer: str
    difficulty: DifficultyLevel
    generated_by: str = "AI"
    source_chunks: List[Any] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True

class QuizAttempt(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    student_id: Any
    topic_id: Any
    questions: List[Any]
    answers: Dict[str, str] = {}
    score: Optional[float] = None
    max_score: float
    completed_at: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True

        