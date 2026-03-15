import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from routes.schemes.curriculum import SubjectCreateRequest, ChapterCreateRequest, TopicCreateRequest
from models.db_schemes import Subject, Chapter, Topic
from bson.objectid import ObjectId

logger = logging.getLogger("uvicorn.error")

curriculum_router = APIRouter(
    prefix="/api/v1/curriculum",
    tags=["api_v1", "curriculum"],
)

# ── Subjects ───────────────────────────────────────────────────────────────────

@curriculum_router.get("/subjects")
async def list_subjects(request: Request, grade: int = None):
    """List all subjects. Optionally filter by grade."""
    curriculum_model = request.app.curriculum_model
    if grade is not None:
        subjects = await curriculum_model.get_subjects_by_grade(grade)
    else:
        subjects = await curriculum_model.get_all_subjects()

    return JSONResponse(
        content={
            "signal": "SUBJECTS_RETRIEVED",
            "subjects": [
                {"id": str(s.id), "name": s.name, "grade": s.grade}
                for s in subjects
            ],
        }
    )


@curriculum_router.post("/subjects")
async def create_subject(request: Request, body: SubjectCreateRequest):
    """Create a new subject."""
    curriculum_model = request.app.curriculum_model
    subject = Subject(name=body.name, grade=body.grade)
    saved = await curriculum_model.create_subject(subject)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"signal": "SUBJECT_CREATED", "id": str(saved.id)},
    )


# ── Chapters ───────────────────────────────────────────────────────────────────

@curriculum_router.get("/chapters/{subject_id}")
async def list_chapters(request: Request, subject_id: str):
    """List all chapters for a subject, ordered by chapter order."""
    curriculum_model = request.app.curriculum_model
    chapters = await curriculum_model.get_chapters_by_subject(subject_id)
    return JSONResponse(
        content={
            "signal": "CHAPTERS_RETRIEVED",
            "chapters": [
                {"id": str(c.id), "name": c.name, "order": c.order, "subject_id": str(c.subject_id)}
                for c in chapters
            ],
        }
    )


@curriculum_router.post("/chapters")
async def create_chapter(request: Request, body: ChapterCreateRequest):
    """Create a new chapter under a subject."""
    curriculum_model = request.app.curriculum_model
    chapter = Chapter(
        subject_id=ObjectId(body.subject_id),
        name=body.name,
        order=body.order,
    )
    saved = await curriculum_model.create_chapter(chapter)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"signal": "CHAPTER_CREATED", "id": str(saved.id)},
    )


# ── Topics ─────────────────────────────────────────────────────────────────────

@curriculum_router.get("/topics/{chapter_id}")
async def list_topics(request: Request, chapter_id: str):
    """List all topics in a chapter, ordered by topic order."""
    curriculum_model = request.app.curriculum_model
    topics = await curriculum_model.get_topics_by_chapter(chapter_id)
    return JSONResponse(
        content={
            "signal": "TOPICS_RETRIEVED",
            "topics": [
                {
                    "id": str(t.id),
                    "name": t.name,
                    "order": t.order,
                    "chapter_id": str(t.chapter_id),
                    "subject_id": str(t.subject_id),
                }
                for t in topics
            ],
        }
    )


@curriculum_router.post("/topics")
async def create_topic(request: Request, body: TopicCreateRequest):
    """Create a new topic under a chapter."""
    curriculum_model = request.app.curriculum_model
    topic = Topic(
        chapter_id=ObjectId(body.chapter_id),
        subject_id=ObjectId(body.subject_id),
        name=body.name,
        order=body.order,
    )
    saved = await curriculum_model.create_topic(topic)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"signal": "TOPIC_CREATED", "id": str(saved.id)},
    )
