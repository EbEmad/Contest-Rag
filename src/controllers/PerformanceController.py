import logging
from typing import List, Optional

from controllers.BaseController import BaseController
from models.PerformanceModel import PerformanceModel
from models.db_schemes import LearningRoadmap


class PerformanceController(BaseController):
    """
    Handles LLM-generated study roadmaps for students.
    All student data (name, grade, level, weak topics) is provided by the caller.
    """

    def __init__(
        self,
        performance_model: PerformanceModel,
        generation_client,
        template_parser,
    ):
        super().__init__()
        self.performance_model = performance_model
        self.generation_client = generation_client
        self.template_parser = template_parser
        self.logger = logging.getLogger(__name__)

    # ── Roadmap Generation ────────────────────────────────────────────────────

    async def generate_roadmap(
        self,
        student_id: str,
        student_name: str,
        grade: int,
        student_level: str,
        weak_topic_names: List[str],
    ) -> Optional[LearningRoadmap]:
        """
        Generate a 2-week LLM study roadmap based on weak topics.
        All parameters are required — provided by the calling backend.
        """
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
            self.logger.error(f"LLM returned empty roadmap for student {student_id}")
            return None

        roadmap = LearningRoadmap(
            student_id=student_id,
            weak_topic_names=weak_topic_names,
            llm_explanation=llm_text,
        )

        saved = await self.performance_model.save_roadmap(roadmap)
        self.logger.info(f"Roadmap generated for student {student_id}")
        return saved

    # ── Latest Roadmap ────────────────────────────────────────────────────────

    async def get_latest_roadmap(self, student_id: str):
        return await self.performance_model.get_latest_roadmap(student_id)
