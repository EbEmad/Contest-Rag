import logging
from typing import List, Optional
from bson.objectid import ObjectId

from controllers.BaseController import BaseController
from models.PerformanceModel import PerformanceModel
from models.QuizModel import QuizModel
from models.db_schemes import LearningRoadmap


class PerformanceController(BaseController):
    """
    Handles performance tracking, dashboards (student/teacher/parent),
    and LLM-generated study roadmaps.
    """

    def __init__(
        self,
        performance_model: PerformanceModel,
        quiz_model: QuizModel,
        student_model,
        curriculum_model,
        generation_client,
        template_parser,
    ):
        super().__init__()
        self.performance_model = performance_model
        self.quiz_model = quiz_model
        self.student_model = student_model
        self.curriculum_model = curriculum_model
        self.generation_client = generation_client
        self.template_parser = template_parser
        self.logger = logging.getLogger(__name__)

    # ── Performance Update ─────────────────────────────────────────────────────

    async def update_after_attempt(self, grade_result: dict, topic_id: str, student_id: str):
        """
        Called right after quiz grading. Updates TopicPerformance for the topic.
        grade_result is the dict returned by QuizController.grade_quiz_attempt.
        """
        correct = grade_result.get("score", 0)
        total = grade_result.get("max_score", 0)
        return await self.performance_model.upsert_topic_performance(
            student_id=student_id,
            topic_id=topic_id,
            correct=correct,
            total=total,
        )

    # ── Student Dashboard ─────────────────────────────────────────────────────

    async def get_student_dashboard(self, student_id: str) -> dict:
        """
        Returns aggregated performance data for a student.
        Suitable for students and parents.
        """
        performances = await self.performance_model.get_student_performances(student_id)
        weak = [p for p in performances if p.accuracy < 60.0]
        strong = [p for p in performances if p.accuracy >= 80.0]

        overall_accuracy = (
            sum(p.accuracy for p in performances) / len(performances)
            if performances else 0.0
        )

        return {
            "student_id": student_id,
            "overall_accuracy": round(overall_accuracy, 2),
            "total_topics_attempted": len(performances),
            "weak_topics": [
                {
                    "topic_id": str(p.topic_id),
                    "accuracy": p.accuracy,
                    "trend": p.trend,
                    "total_attempts": p.total_attempts,
                }
                for p in weak
            ],
            "strong_topics": [
                {
                    "topic_id": str(p.topic_id),
                    "accuracy": p.accuracy,
                    "trend": p.trend,
                }
                for p in strong
            ],
            "all_topics": [
                {
                    "topic_id": str(p.topic_id),
                    "accuracy": p.accuracy,
                    "trend": p.trend,
                    "total_attempts": p.total_attempts,
                    "correct_answers": p.correct_answers,
                }
                for p in performances
            ],
        }

    # ── Parent Dashboard ──────────────────────────────────────────────────────

    async def get_parent_dashboard(self, student_id: str) -> dict:
        """
        Simplified progress summary for parents.
        """
        data = await self.get_student_dashboard(student_id)
        weak_count = len(data["weak_topics"])
        strong_count = len(data["strong_topics"])
        total = data["total_topics_attempted"]

        if data["overall_accuracy"] >= 80:
            overall_status = "Excellent — your child is performing very well!"
        elif data["overall_accuracy"] >= 60:
            overall_status = "Good — some topics need more practice."
        else:
            overall_status = "Needs improvement — consider additional study sessions."

        return {
            "student_id": student_id,
            "overall_accuracy": data["overall_accuracy"],
            "overall_status": overall_status,
            "topics_mastered": strong_count,
            "topics_needing_attention": weak_count,
            "total_topics_attempted": total,
            "areas_to_focus": [
                {"topic_id": t["topic_id"], "accuracy": t["accuracy"]}
                for t in data["weak_topics"]
            ],
        }

    # ── Teacher Dashboard ─────────────────────────────────────────────────────

    async def get_teacher_dashboard(self, topic_ids: List[str]) -> dict:
        """
        Aggregated per-topic class performance for a teacher.
        topic_ids: list of topic ObjectId strings to analyze.
        """
        topic_summaries = []
        for topic_id in topic_ids:
            perfs = await self.performance_model.get_topic_class_performance(topic_id)
            if not perfs:
                continue
            avg_accuracy = sum(p.accuracy for p in perfs) / len(perfs)
            struggling = [p for p in perfs if p.accuracy < 60.0]
            topic_summaries.append({
                "topic_id": topic_id,
                "num_students": len(perfs),
                "avg_accuracy": round(avg_accuracy, 2),
                "struggling_students": len(struggling),
                "needs_attention": avg_accuracy < 60.0,
            })

        # Sort by weakest topics first
        topic_summaries.sort(key=lambda x: x["avg_accuracy"])

        return {
            "topics_analyzed": len(topic_summaries),
            "topics": topic_summaries,
            "flagged_topics": [t for t in topic_summaries if t["needs_attention"]],
        }

    # ── Roadmap Generation ────────────────────────────────────────────────────

    async def generate_roadmap(
        self,
        student_id: str,
        student_name: Optional[str] = None,
        grade: Optional[int] = None,
        student_level: Optional[str] = None,
        weak_topic_names: Optional[List[str]] = None,
        weak_topic_ids: Optional[List[str]] = None,
    ) -> LearningRoadmap:
        """
        Ask the LLM to generate a 2-week study roadmap based on the student's weak topics.
        Saves and returns the LearningRoadmap.
        """
        # 1. Fetch Student Profile if missing
        if not student_name or grade is None or not student_level:
            student_profile = await self.student_model.get_student_by_id(student_id)
            if student_profile:
                student_name = student_name or student_profile.full_name
                grade = grade if grade is not None else student_profile.grade
                student_level = student_level or student_profile.current_level
            else:
                student_name = student_name or "Student"
                grade = grade if grade is not None else 10
                student_level = student_level or "BEGINNER"

        # 2. Fetch Weak Topics if missing
        if not weak_topic_names:
            weak_perfs = await self.performance_model.get_weak_topics(student_id, threshold=60.0)
            if weak_perfs:
                # Limit to top 5 weakest for prompt clarity
                weak_perfs = weak_perfs[:5]
                weak_topic_names = []
                weak_topic_ids = weak_topic_ids or []
                
                for perf in weak_perfs:
                    topic = await self.curriculum_model.get_topic(str(perf.topic_id))
                    if topic:
                        weak_topic_names.append(topic.name)
                        if str(perf.topic_id) not in weak_topic_ids:
                            weak_topic_ids.append(str(perf.topic_id))
            
            if not weak_topic_names:
                weak_topic_names = ["General review"]

        weak_topics_formatted = "\n".join(
            f"- {name}" for name in weak_topic_names
        )

        system_prompt = self.template_parser.get("roadmap", "system_prompt", {
            "student_name": student_name,
            "grade": grade,
            "student_level": student_level,
            "weak_topics_list": weak_topics_formatted,
        })
        footer_prompt = self.template_parser.get("roadmap", "footer_prompt", {
            "student_name": student_name,
        })

        full_prompt = "\n\n".join([system_prompt, footer_prompt])

        llm_text = await self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=[],
        )

        if not llm_text:
            llm_text = "No roadmap could be generated at this time. Please try again later."

        # Validate and handle IDs
        safe_student_id = ObjectId(student_id) if ObjectId.is_valid(student_id) else ObjectId()
        
        oid_weak_topics = []
        for tid in (weak_topic_ids or []):
            if ObjectId.is_valid(tid):
                oid_weak_topics.append(ObjectId(tid))

        roadmap = LearningRoadmap(
            student_id=safe_student_id,
            weak_topic_ids=oid_weak_topics,
            llm_explanation=llm_text,
        )

        saved = await self.performance_model.save_roadmap(roadmap)
        self.logger.info(f"Roadmap generated for student {student_id}")
        return saved

    # ── Latest Roadmap ────────────────────────────────────────────────────────

    async def get_latest_roadmap(self, student_id: str):
        return await self.performance_model.get_latest_roadmap(student_id)
