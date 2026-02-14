from enum import Enum

class DataBaseEnum(Enum):
    COLLECTION_PROJECT_NAME= "projects"
    COLLECTION_CHUNK_NAME= "chunks"
    COLLECTION_ASSET_NAME= "assets"
    COLLECTION_CELERY_TASK_EXECUTION_NAME= "celery_task_executions"