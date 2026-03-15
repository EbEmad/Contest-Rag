from enum import Enum

class DataBaseEnum(Enum):
    COLLECTION_PROJECT_NAME= "projects"
    COLLECTION_CHUNK_NAME= "chunks"
    COLLECTION_ASSET_NAME= "assets"
    COLLECTION_CELERY_TASK_EXECUTION_NAME= "celery_task_executions"
    COLLECTION_USER_NAME="users"
    COLLECTION_STUDENT_NAME = "students"
    COLLECTION_QUESTION_NAME = "questions"
    COLLECTION_QUIZ_ATTEMPT_NAME = "quiz_attempts"
    COLLECTION_TOPIC_PERFORMANCE_NAME = "topic_performances"
    COLLECTION_LEARNING_ROADMAP_NAME = "learning_roadmaps"
    COLLECTION_SUBJECT_NAME = "subjects"
    COLLECTION_CHAPTER_NAME = "chapters"
    COLLECTION_TOPIC_NAME = "topics"
    COLLECTION_CHAT_MESSAGE_NAME = "chat_messages"