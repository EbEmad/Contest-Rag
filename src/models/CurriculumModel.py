from .BaseDataModel import BaseDataModel
from .db_schemes import Subject, Chapter, Topic
from .enums.DataBaseEnum import DataBaseEnum
from bson.objectid import ObjectId
from typing import List, Optional


class CurriculumModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.subjects = self.db_client[DataBaseEnum.COLLECTION_SUBJECT_NAME.value]
        self.chapters = self.db_client[DataBaseEnum.COLLECTION_CHAPTER_NAME.value]
        self.topics = self.db_client[DataBaseEnum.COLLECTION_TOPIC_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_SUBJECT_NAME.value not in all_collections:
            await self.subjects.create_index(
                [("name", 1), ("grade", 1)],
                name="idx_subject_name_grade",
                unique=True,
            )
        if DataBaseEnum.COLLECTION_CHAPTER_NAME.value not in all_collections:
            await self.chapters.create_index(
                [("subject_id", 1), ("order", 1)],
                name="idx_chapter_subject_order",
                unique=False,
            )
        if DataBaseEnum.COLLECTION_TOPIC_NAME.value not in all_collections:
            await self.topics.create_index(
                [("chapter_id", 1), ("order", 1)],
                name="idx_topic_chapter_order",
                unique=False,
            )

    # ── Subjects ───────────────────────────────────────────────────────────────

    async def create_subject(self, subject: Subject) -> Subject:
        result = await self.subjects.insert_one(subject.dict(by_alias=True, exclude_unset=True))
        subject.id = result.inserted_id
        return subject

    async def get_subject(self, subject_id: str) -> Optional[Subject]:
        record = await self.subjects.find_one({"_id": ObjectId(subject_id)})
        return Subject(**record) if record else None

    async def get_subjects_by_grade(self, grade: int) -> List[Subject]:
        cursor = self.subjects.find({"grade": grade}).sort("name", 1)
        return [Subject(**doc) async for doc in cursor]

    async def get_all_subjects(self) -> List[Subject]:
        cursor = self.subjects.find().sort([("grade", 1), ("name", 1)])
        return [Subject(**doc) async for doc in cursor]

    # ── Chapters ───────────────────────────────────────────────────────────────

    async def create_chapter(self, chapter: Chapter) -> Chapter:
        result = await self.chapters.insert_one(chapter.dict(by_alias=True, exclude_unset=True))
        chapter.id = result.inserted_id
        return chapter

    async def get_chapter(self, chapter_id: str) -> Optional[Chapter]:
        record = await self.chapters.find_one({"_id": ObjectId(chapter_id)})
        return Chapter(**record) if record else None

    async def get_chapters_by_subject(self, subject_id: str) -> List[Chapter]:
        cursor = self.chapters.find({"subject_id": ObjectId(subject_id)}).sort("order", 1)
        return [Chapter(**doc) async for doc in cursor]

    # ── Topics ─────────────────────────────────────────────────────────────────

    async def create_topic(self, topic: Topic) -> Topic:
        result = await self.topics.insert_one(topic.dict(by_alias=True, exclude_unset=True))
        topic.id = result.inserted_id
        return topic

    async def get_topic(self, topic_id: str) -> Optional[Topic]:
        record = await self.topics.find_one({"_id": ObjectId(topic_id)})
        return Topic(**record) if record else None

    async def get_topics_by_chapter(self, chapter_id: str) -> List[Topic]:
        cursor = self.topics.find({"chapter_id": ObjectId(chapter_id)}).sort("order", 1)
        return [Topic(**doc) async for doc in cursor]

    async def get_topics_by_subject(self, subject_id: str) -> List[Topic]:
        cursor = self.topics.find({"subject_id": ObjectId(subject_id)}).sort("order", 1)
        return [Topic(**doc) async for doc in cursor]
