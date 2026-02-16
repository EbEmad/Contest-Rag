from pydantic import BaseModel
from typing import Optional, List

class ProcessRequest(BaseModel):
    file_id: str = None
    chunk_size: Optional[int] = 100
    overlap_size: Optional[int] = 20
    do_reset: Optional[int] = 0
    
    # Optional curriculum metadata
    grade: Optional[int] = None
    subject: Optional[str] = None
    chapter_name: Optional[str] = None  # NEW: Chapter name
    topic_names: Optional[List[str]] = None  # NEW: List of topic names this book covers