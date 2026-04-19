from .BaseDataModel import BaseDataModel
from .db_schemes import LearningRoadmap
from .enums.DataBaseEnum import DataBaseEnum
from typing import List, Optional


class PerformanceModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.roadmaps = self.db_client[DataBaseEnum.COLLECTION_LEARNING_ROADMAP_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_LEARNING_ROADMAP_NAME.value not in all_collections:
            await self.roadmaps.create_index(
                [("student_id", 1)], name="idx_roadmap_student_id", unique=False
            )

    # ── LearningRoadmap ────────────────────────────────────────────────────────

    async def save_roadmap(self, roadmap: LearningRoadmap) -> LearningRoadmap:
        result = await self.roadmaps.insert_one(
            roadmap.dict(by_alias=True, exclude_unset=True)
        )
        roadmap.id = result.inserted_id
        return roadmap

    async def get_latest_roadmap(self, student_id: str) -> Optional[LearningRoadmap]:
        record = await self.roadmaps.find_one(
            {"student_id": student_id},
            sort=[("generated_at", -1)],
        )
        return LearningRoadmap(**record) if record else None

    async def get_all_roadmaps(self, student_id: str) -> List[LearningRoadmap]:
        cursor = self.roadmaps.find(
            {"student_id": student_id}
        ).sort("generated_at", -1)
        return [LearningRoadmap(**doc) async for doc in cursor]
