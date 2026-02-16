from pydantic import BaseModel,Field
from typing import Optional
from bson.objectid import ObjectId

class Subject(BaseModel):
    id :Optional[ObjectId] = Field(None, alias="_id")
    name:str  # ex:: "Mathematics", "Biology"
    grade:int # 1-12

    class Config:
        arbitrary_types_allowed = True

class Chapter(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    subject_id: ObjectId
    name: str
    order: int
    
    class Config:
        arbitrary_types_allowed = True

class Topic(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    chapter_id: ObjectId
    subject_id: ObjectId
    name: str
    order: int
    
    class Config:
        arbitrary_types_allowed = True