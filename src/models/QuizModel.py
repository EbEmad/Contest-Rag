from .BaseDataModel import BaseDataModel
from .db_schemes import Question, QuizAttempt, DifficultyLevel, QuestionType
from .enums.DataBaseEnum import DataBaseEnum
from bson.objectid import ObjectId
from typing import List, Dict, Optional
from datetime import datetime


class QuizModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.questions = self.db_client[DataBaseEnum.COLLECTION_QUESTION_NAME.value]
        self.attempts = self.db_client[DataBaseEnum.COLLECTION_QUIZ_ATTEMPT_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_QUESTION_NAME.value not in all_collections:
            await self.questions.create_index(
                [("topic_id", 1)], name="idx_question_topic_id", unique=False
            )
        if DataBaseEnum.COLLECTION_QUIZ_ATTEMPT_NAME.value not in all_collections:
            await self.attempts.create_index(
                [("student_id", 1)], name="idx_attempt_student_id", unique=False
            )

    # ── Questions ──────────────────────────────────────────────────────────────

    async def save_question(self, question: Question) -> Question:
        result = await self.questions.insert_one(
            question.dict(by_alias=True, exclude_unset=True)
        )
        question.id = result.inserted_id
        return question

    async def save_questions_bulk(self, questions: List[Question]) -> List[Question]:
        docs = [q.dict(by_alias=True, exclude_unset=True) for q in questions]
        result = await self.questions.insert_many(docs)
        for q, oid in zip(questions, result.inserted_ids):
            q.id = oid
        return questions

    async def get_question(self, question_id: str) -> Optional[Question]:
        record = await self.questions.find_one({"_id": self.to_object_id(question_id)})
        return Question(**record) if record else None

    async def get_questions_by_topic(
        self,
        topic_id: str,
        difficulty: Optional[DifficultyLevel] = None,
        limit: int = 20,
    ) -> List[Question]:
        filt: Dict = {"topic_id": self.to_object_id(topic_id)}
        if difficulty:
            filt["difficulty"] = difficulty.value
        cursor = self.questions.find(filt).limit(limit)
        return [Question(**doc) async for doc in cursor]

    async def get_questions_by_ids(self, question_ids: List[str]) -> List[Question]:
        oids = [self.to_object_id(qid) for qid in question_ids]
        cursor = self.questions.find({"_id": {"$in": oids}})
        return [Question(**doc) async for doc in cursor]

    # ── Quiz Attempts ──────────────────────────────────────────────────────────

    async def create_attempt(self, attempt: QuizAttempt) -> QuizAttempt:
        result = await self.attempts.insert_one(
            attempt.dict(by_alias=True, exclude_unset=True)
        )
        attempt.id = result.inserted_id
        return attempt

    async def get_attempt(self, attempt_id: str) -> Optional[QuizAttempt]:
        record = await self.attempts.find_one({"_id": self.to_object_id(attempt_id)})
        return QuizAttempt(**record) if record else None

    async def submit_attempt(
        self,
        attempt_id: str,
        answers: Dict[str, str],
        score: float,
    ) -> bool:
        result = await self.attempts.update_one(
            {"_id": self.to_object_id(attempt_id)},
            {
                "$set": {
                    "answers": answers,
                    "score": score,
                    "completed_at": datetime.utcnow(),
                }
            },
        )
        return result.modified_count > 0

    async def get_student_attempts(
        self, student_id: str, page: int = 1, page_size: int = 20
    ):
        filt = {"student_id": self.to_object_id(student_id)}
        total = await self.attempts.count_documents(filt)
        total_pages = (total + page_size - 1) // page_size
        cursor = (
            self.attempts.find(filt)
            .sort("completed_at", -1)
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        attempts = [QuizAttempt(**doc) async for doc in cursor]
        return attempts, total_pages
