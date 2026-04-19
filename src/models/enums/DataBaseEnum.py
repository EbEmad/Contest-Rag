from enum import Enum

class DataBaseEnum(Enum):
    COLLECTION_PROJECT_NAME = "projects"
    COLLECTION_CHUNK_NAME = "chunks"
    COLLECTION_ASSET_NAME = "assets"
    COLLECTION_CELERY_TASK_EXECUTION_NAME = "celery_task_executions"
    COLLECTION_QUESTION_NAME = "questions"
    COLLECTION_QUIZ_ATTEMPT_NAME = "quiz_attempts"
    COLLECTION_LEARNING_ROADMAP_NAME = "learning_roadmaps"
    COLLECTION_CHAT_MESSAGE_NAME = "chat_messages"