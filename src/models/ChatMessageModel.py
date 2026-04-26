from .BaseDataModel import BaseDataModel
from .db_schemes import ChatMessage
from .enums.DataBaseEnum import DataBaseEnum
from typing import List
import pymongo


class ChatMessageModel(BaseDataModel):
    """Persists per-student chat turns (user/assistant) in MongoDB."""

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_MESSAGE_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_CHAT_MESSAGE_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_MESSAGE_NAME.value]
            # Compound index for fast per-student lookups sorted by time
            await self.collection.create_index(
                [("student_id", 1), ("project_id", 1), ("created_at", -1)],
                name="idx_student_project_time",
            )

    async def add_message(self, student_id: str, project_id: str,
                          role: str, content: str) -> ChatMessage:
        """Insert a single chat turn."""
        msg = ChatMessage(
            student_id=student_id,
            project_id=project_id,
            role=role,
            content=content,
        )
        result = await self.collection.insert_one(
            msg.dict(by_alias=True, exclude_none=True)
        )
        msg.id = result.inserted_id
        return msg

    async def get_recent_messages(self, student_id: str, project_id: str,
                                  limit: int = 10) -> List[ChatMessage]:
        """Return the last *limit* messages for this student+project, oldest-first."""
        cursor = (
            self.collection
            .find({"student_id": student_id, "project_id": project_id})
            .sort("created_at", pymongo.DESCENDING)
            .limit(limit)
        )
        messages = []
        async for doc in cursor:
            messages.append(ChatMessage(**doc))
        messages.reverse()  # oldest first so the LLM reads chronologically
        return messages

    async def clear_history(self, student_id: str, project_id: str) -> int:
        """Delete all chat history for a student+project. Returns deleted count."""
        result = await self.collection.delete_many(
            {"student_id": student_id, "project_id": project_id}
        )
        return result.deleted_count
