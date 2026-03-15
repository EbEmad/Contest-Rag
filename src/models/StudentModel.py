from .BaseDataModel import BaseDataModel
from .db_schemes import StudentProfile
from .enums.DataBaseEnum import DataBaseEnum
from bson.objectid import ObjectId
from typing import Optional


class StudentModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_STUDENT_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_STUDENT_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_STUDENT_NAME.value]
            await self.collection.create_index(
                [("user_id", 1)],
                name="idx_student_user_id",
                unique=True
            )

    async def create_student(self, student: StudentProfile) -> StudentProfile:
        result = await self.collection.insert_one(
            student.dict(by_alias=True, exclude_unset=True)
        )
        student.id = result.inserted_id
        return student

    async def get_student_by_user_id(self, user_id: str) -> Optional[StudentProfile]:
        record = await self.collection.find_one({"user_id": user_id})
        if record is None:
            return None
        return StudentProfile(**record)

    async def get_student_by_id(self, student_id: str) -> Optional[StudentProfile]:
        record = await self.collection.find_one({"_id": self.to_object_id(student_id)})
        if record is None:
            return None
        return StudentProfile(**record)

    async def get_or_create_student(self, user_id: str, full_name: str = "Unknown", grade: int = 1) -> StudentProfile:
        existing = await self.get_student_by_user_id(user_id)
        if existing:
            return existing
        student = StudentProfile(user_id=user_id, full_name=full_name, grade=grade)
        return await self.create_student(student)

    async def update_student_level(self, student_id: str, new_level: str) -> bool:
        result = await self.collection.update_one(
            {"_id": self.to_object_id(student_id)},
            {"$set": {"current_level": new_level}}
        )
        return result.modified_count > 0

    async def get_all_students(self, page: int = 1, page_size: int = 20):
        total = await self.collection.count_documents({})
        total_pages = (total + page_size - 1) // page_size
        cursor = self.collection.find().skip((page - 1) * page_size).limit(page_size)
        students = []
        async for doc in cursor:
            students.append(StudentProfile(**doc))
        return students, total_pages
