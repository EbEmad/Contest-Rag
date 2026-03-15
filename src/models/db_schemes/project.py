from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId
import re

class Project(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    project_id: str = Field(..., min_length=1)

    @validator('project_id')
    def validate_project_id(cls, value):
        if not re.match(r'^[a-zA-Z0-9_]+$', value):
            raise ValueError('project_id must be alphanumeric or contain underscores')
        
        return value

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls):

        return [
            {
                "key": [
                    ("project_id", 1)
                ],
                "name": "project_id_index_1",
                "unique": True
            }
        ]