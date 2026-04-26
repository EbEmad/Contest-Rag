import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from routes.schemes.quiz import QuizGenerateRequest, QuizSubmitRequest
from models.db_schemes import QuizAttempt, DifficultyLevel, QuestionType
from bson.objectid import ObjectId

logger = logging.getLogger("uvicorn.error")

quiz_router = APIRouter(
    prefix="/api/v1/quiz",
    tags=["api_v1", "quiz"],
)


@quiz_router.post("/generate")
async def generate_quiz(request: Request, quiz_request: QuizGenerateRequest):
    """
    Generate a quiz for a topic using RAG context + LLM.
    Requires: topic_id, topic_name, student_id.
    Optional: num_questions, difficulty, question_types, grade, subject.
    """
    nlp_controller = request.app.nlp_controller
    quiz_controller = request.app.quiz_controller
    project_model = request.app.project_model

    if not ObjectId.is_valid(quiz_request.topic_id):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": "INVALID_ID",
                "detail": f"topic_id '{quiz_request.topic_id}' is not a valid ObjectId."
            },
        )

    project = await project_model.get_project_or_create_one(
        project_id=quiz_request.student_id
    )

    # Retrieve context documents via RAG pipeline
    context_docs = await nlp_controller.search_by_curriculum(
        project=project,
        query=quiz_request.topic_name,
        grade=quiz_request.grade,
        subject=quiz_request.subject,
        limit=10,
    )

    if not context_docs:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": "NO_CONTEXT_FOUND", "detail": "No relevant content found for this topic."},
        )

    questions = await quiz_controller.generate_quiz(
        topic_id=quiz_request.topic_id,
        topic_name=quiz_request.topic_name,
        context_documents=context_docs,
        num_questions=quiz_request.num_questions,
        difficulty=quiz_request.difficulty,
        question_types=quiz_request.question_types,
        student_level="BEGINNER",
    )

    if not questions:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": "QUIZ_GENERATION_FAILED"},
        )

    # Create an attempt record
    quiz_model = request.app.quiz_model
    attempt = QuizAttempt(
        student_id=ObjectId(quiz_request.student_id) if ObjectId.is_valid(quiz_request.student_id) else ObjectId(),
        topic_id=ObjectId(quiz_request.topic_id),
        questions=[q.id for q in questions],
        max_score=float(len(questions)),
    )
    saved_attempt = await quiz_model.create_attempt(attempt)

    return JSONResponse(
        content={
            "signal": "QUIZ_GENERATED",
            "attempt_id": str(saved_attempt.id),
            "questions": [
                {
                    "id": str(q.id),
                    "question_type": q.question_type,
                    "question_text": q.question_text,
                    "options": q.options,
                    "difficulty": q.difficulty,
                }
                for q in questions
            ],
        }
    )


@quiz_router.post("/submit/{attempt_id}")
async def submit_quiz(request: Request, attempt_id: str, submit_request: QuizSubmitRequest):
    """
    Submit answers for a quiz attempt. Returns score and per-question results.
    """
    quiz_controller = request.app.quiz_controller

    result = await quiz_controller.grade_quiz_attempt(
        attempt_id=attempt_id,
        student_answers=submit_request.answers,
    )

    if result is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": "ATTEMPT_NOT_FOUND"},
        )

    return JSONResponse(content={"signal": "QUIZ_GRADED", **result})


@quiz_router.get("/history/{student_id}")
async def get_quiz_history(request: Request, student_id: str, page: int = 1, page_size: int = 20):
    """List all past quiz attempts for a student."""
    quiz_controller = request.app.quiz_controller
    attempts, total_pages = await quiz_controller.get_quiz_history(
        student_id=student_id, page=page, page_size=page_size
    )
    return JSONResponse(
        content={
            "signal": "HISTORY_RETRIEVED",
            "total_pages": total_pages,
            "attempts": [
                {
                    "attempt_id": str(a.id),
                    "topic_id": str(a.topic_id),
                    "score": a.score,
                    "max_score": a.max_score,
                    "completed_at": str(a.completed_at) if a.completed_at else None,
                }
                for a in attempts
            ],
        }
    )


@quiz_router.get("/{question_id}")
async def get_question(request: Request, question_id: str):
    """Fetch a single question by ID."""
    quiz_model = request.app.quiz_model
    question = await quiz_model.get_question(question_id)
    if not question:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": "QUESTION_NOT_FOUND"},
        )
    return JSONResponse(
        content={
            "signal": "QUESTION_RETRIEVED",
            "question": {
                "id": str(question.id),
                "question_type": question.question_type,
                "question_text": question.question_text,
                "options": question.options,
                "difficulty": question.difficulty,
            },
        }
    )
