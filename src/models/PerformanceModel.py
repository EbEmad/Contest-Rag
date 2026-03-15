from .BaseDataModel import BaseDataModel
from .db_schemes import TopicPerformance, LearningRoadmap
from .enums.DataBaseEnum import DataBaseEnum
from bson.objectid import ObjectId
from typing import List, Optional


class PerformanceModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.performances = self.db_client[DataBaseEnum.COLLECTION_TOPIC_PERFORMANCE_NAME.value]
        self.roadmaps = self.db_client[DataBaseEnum.COLLECTION_LEARNING_ROADMAP_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_TOPIC_PERFORMANCE_NAME.value not in all_collections:
            await self.performances.create_index(
                [("student_id", 1), ("topic_id", 1)],
                name="idx_perf_student_topic",
                unique=True,
            )
        if DataBaseEnum.COLLECTION_LEARNING_ROADMAP_NAME.value not in all_collections:
            await self.roadmaps.create_index(
                [("student_id", 1)], name="idx_roadmap_student_id", unique=False
            )

    # ── TopicPerformance ───────────────────────────────────────────────────────

    async def upsert_topic_performance(
        self,
        student_id: str,
        topic_id: str,
        correct: int,
        total: int,
    ) -> TopicPerformance:
        oid_student = self.to_object_id(student_id)
        oid_topic = self.to_object_id(topic_id)

        existing = await self.performances.find_one(
            {"student_id": oid_student, "topic_id": oid_topic}
        )

        if existing:
            prev_correct = existing.get("correct_answers", 0)
            prev_total = existing.get("total_attempts", 0)
            new_correct = prev_correct + correct
            new_total = prev_total + total
            new_accuracy = (new_correct / new_total * 100) if new_total > 0 else 0.0

            prev_accuracy = existing.get("accuracy", 0.0)
            if new_accuracy > prev_accuracy + 5:
                trend = "IMPROVING"
            elif new_accuracy < prev_accuracy - 5:
                trend = "DECLINING"
            else:
                trend = "STABLE"

            await self.performances.update_one(
                {"student_id": oid_student, "topic_id": oid_topic},
                {
                    "$set": {
                        "accuracy": new_accuracy,
                        "total_attempts": new_total,
                        "correct_answers": new_correct,
                        "trend": trend,
                    }
                },
            )
        else:
            accuracy = (correct / total * 100) if total > 0 else 0.0
            doc = TopicPerformance(
                student_id=oid_student,
                topic_id=oid_topic,
                accuracy=accuracy,
                total_attempts=total,
                correct_answers=correct,
                trend="STABLE",
            )
            result = await self.performances.insert_one(
                doc.dict(by_alias=True, exclude_unset=True)
            )
            doc.id = result.inserted_id

        record = await self.performances.find_one(
            {"student_id": oid_student, "topic_id": oid_topic}
        )
        return TopicPerformance(**record)

    async def get_student_performances(self, student_id: str) -> List[TopicPerformance]:
        cursor = self.performances.find({"student_id": self.to_object_id(student_id)})
        return [TopicPerformance(**doc) async for doc in cursor]

    async def get_weak_topics(self, student_id: str, threshold: float = 60.0) -> List[TopicPerformance]:
        """Return topics where accuracy is below threshold (default 60%)."""
        cursor = self.performances.find(
            {"student_id": self.to_object_id(student_id), "accuracy": {"$lt": threshold}}
        ).sort("accuracy", 1)
        return [TopicPerformance(**doc) async for doc in cursor]

    async def get_topic_class_performance(self, topic_id: str) -> List[TopicPerformance]:
        """All student performances for a given topic (for teacher view)."""
        cursor = self.performances.find({"topic_id": self.to_object_id(topic_id)})
        return [TopicPerformance(**doc) async for doc in cursor]

    # ── LearningRoadmap ────────────────────────────────────────────────────────

    async def save_roadmap(self, roadmap: LearningRoadmap) -> LearningRoadmap:
        result = await self.roadmaps.insert_one(
            roadmap.dict(by_alias=True, exclude_unset=True)
        )
        roadmap.id = result.inserted_id
        return roadmap

    async def get_latest_roadmap(self, student_id: str) -> Optional[LearningRoadmap]:
        record = await self.roadmaps.find_one(
            {"student_id": self.to_object_id(student_id)},
            sort=[("generated_at", -1)],
        )
        return LearningRoadmap(**record) if record else None

    async def get_all_roadmaps(self, student_id: str) -> List[LearningRoadmap]:
        cursor = self.roadmaps.find(
            {"student_id": self.to_object_id(student_id)}
        ).sort("generated_at", -1)
        return [LearningRoadmap(**doc) async for doc in cursor]
