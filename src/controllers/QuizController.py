import json
import logging
from typing import List, Optional
from bson.objectid import ObjectId

from controllers.BaseController import BaseController
from models.QuizModel import QuizModel
from models.db_schemes import Question, QuizAttempt, QuestionType, DifficultyLevel


class QuizController(BaseController):
    """
    Handles quiz generation (via LLM) and quiz grading.
    """

    def __init__(self, quiz_model: QuizModel, generation_client, template_parser):
        super().__init__()
        self.quiz_model = quiz_model
        self.generation_client = generation_client
        self.template_parser = template_parser
        self.logger = logging.getLogger(__name__)

    # ── Quiz Generation ────────────────────────────────────────────────────────

    async def generate_quiz(
        self,
        topic_id: str,
        topic_name: str,
        context_documents: List,           # list of RetrievedDocument from RAG search
        num_questions: int = 5,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        question_types: Optional[List[QuestionType]] = None,
        student_level: str = "BEGINNER",
    ) -> List[Question]:
        """
        Generate quiz questions for a topic using retrieved RAG context and LLM.
        Returns saved Question objects.
        """
        if question_types is None:
            question_types = [QuestionType.MCQ, QuestionType.TRUE_FALSE, QuestionType.SHORT_ANSWER]

        # Build prompt
        system_prompt = self.template_parser.get("quiz_generation", "system_prompt", {
            "num_questions": num_questions,
            "topic_name": topic_name,
            "student_level": student_level,
            "difficulty": difficulty.value if hasattr(difficulty, "value") else difficulty,
            "question_types": ", ".join(
                [qt.value if hasattr(qt, "value") else qt for qt in question_types]
            ),
        })

        docs_prompt = "\n".join([
            self.template_parser.get("quiz_generation", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": doc.text,
            })
            for idx, doc in enumerate(context_documents)
        ])

        footer_prompt = self.template_parser.get("quiz_generation", "footer_prompt", {
            "num_questions": num_questions,
            "topic_name": topic_name,
        })

        full_prompt = "\n\n".join([system_prompt, "### Context Documents:", docs_prompt, footer_prompt])

        # Call LLM
        raw_answer = await self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=[],
        )

        if not raw_answer:
            self.logger.error("LLM returned empty response for quiz generation")
            return []

        # Parse JSON response
        questions = self._parse_llm_questions(
            raw_answer=raw_answer,
            topic_id=topic_id,
            context_documents=context_documents,
        )

        if not questions:
            self.logger.warning("No valid questions parsed from LLM response")
            return []

        # Persist to DB
        saved = await self.quiz_model.save_questions_bulk(questions)
        self.logger.info(f"Saved {len(saved)} questions for topic {topic_id}")
        return saved

    def _parse_llm_questions(
        self,
        raw_answer: str,
        topic_id: str,
        context_documents: List,
    ) -> List[Question]:
        """Parse the LLM JSON response into Question objects."""
        questions = []
        try:
            # Strip markdown code fences if present
            clean = raw_answer.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean

            data = json.loads(clean)
            if not isinstance(data, list):
                data = [data]

            chunk_ids = [doc.id for doc in context_documents if hasattr(doc, "id")]

            for item in data:
                try:
                    q = Question(
                        topic_id=ObjectId(topic_id),
                        question_type=item.get("question_type", QuestionType.MCQ),
                        question_text=item.get("question_text", ""),
                        options=item.get("options"),
                        correct_answer=item.get("correct_answer", ""),
                        difficulty=item.get("difficulty", DifficultyLevel.MEDIUM),
                        generated_by="AI",
                        source_chunks=chunk_ids,
                    )
                    questions.append(q)
                except Exception as e:
                    self.logger.warning(f"Skipped malformed question: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM JSON response: {e}\nRaw: {raw_answer[:500]}")
        return questions

    # ── Quiz Grading ───────────────────────────────────────────────────────────

    async def grade_quiz_attempt(
        self,
        attempt_id: str,
        student_answers: dict,  # {question_id: answer_str}
    ) -> dict:
        """
        Grade a quiz attempt.
        Returns { score, max_score, percentage, results: [{question_id, correct, expected}] }
        """
        attempt = await self.quiz_model.get_attempt(attempt_id)
        if not attempt:
            return None

        question_ids = [str(qid) for qid in attempt.questions]
        questions = await self.quiz_model.get_questions_by_ids(question_ids)

        correct_count = 0
        results = []
        for q in questions:
            qid_str = str(q.id)
            student_ans = student_answers.get(qid_str, "").strip().lower()
            expected_ans = q.correct_answer.strip().lower()
            is_correct = student_ans == expected_ans
            if is_correct:
                correct_count += 1
            results.append({
                "question_id": qid_str,
                "correct": is_correct,
                "expected": q.correct_answer,
                "student_answer": student_answers.get(qid_str, ""),
            })

        max_score = len(questions)
        score = correct_count
        percentage = (score / max_score * 100) if max_score > 0 else 0.0

        await self.quiz_model.submit_attempt(
            attempt_id=attempt_id,
            answers=student_answers,
            score=score,
        )

        return {
            "attempt_id": attempt_id,
            "score": score,
            "max_score": max_score,
            "percentage": round(percentage, 2),
            "results": results,
        }

    # ── History ────────────────────────────────────────────────────────────────

    async def get_quiz_history(self, student_id: str, page: int = 1, page_size: int = 20):
        attempts, total_pages = await self.quiz_model.get_student_attempts(
            student_id=student_id, page=page, page_size=page_size
        )
        return attempts, total_pages
