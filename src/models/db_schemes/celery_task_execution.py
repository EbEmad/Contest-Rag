from pydantic import BaseModel, Field
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime
from uuid import UUID

class CeleryTaskExecution(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    
    task_name: str = Field(..., max_length=255)
    task_args_hash: str = Field(..., max_length=64)  # SHA-256 hash of task arguments
    celery_task_id: Optional[str] = None
    
    status: str = Field(default='PENDING', max_length=20)
    
    task_args: Optional[dict] = None
    result: Optional[dict] = None
    
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            UUID: str
        }

    @classmethod
    def get_indexes(cls):
        """
        Define MongoDB indexes 
        """
        return [
            {
                "key": [
                    ("task_name", 1),
                    ("task_args_hash", 1),
                    ("celery_task_id", 1)
                ],
                "name": "ixz_task_name_args_celery_hash",
                "unique": True
            },
            {
                "key": [("status", 1)],
                "name": "ixz_task_execution_status",
                "unique": False
            },
            {
                "key": [("created_at", 1)],
                "name": "ixz_task_execution_created_at",
                "unique": False
            },
            {
                "key": [("celery_task_id", 1)],
                "name": "ixz_celery_task_id",
                "unique": False
            }
        ]
