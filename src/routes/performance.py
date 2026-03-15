import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from routes.schemes.performance import RoadmapRequest, TeacherDashboardRequest

logger = logging.getLogger("uvicorn.error")

performance_router = APIRouter(
    prefix="/api/v1/performance",
    tags=["api_v1", "performance"],
)


@performance_router.get("/student/{student_id}")
async def student_dashboard(request: Request, student_id: str):
    """
    Student dashboard: overall accuracy, weak and strong topics, all topic details.
    """
    performance_controller = request.app.performance_controller
    dashboard = await performance_controller.get_student_dashboard(student_id)
    return JSONResponse(content={"signal": "STUDENT_DASHBOARD_RETRIEVED", **dashboard})


@performance_router.get("/parent/{student_id}")
async def parent_dashboard(request: Request, student_id: str):
    """
    Parent-facing summary: overall status message, topics mastered vs needing attention.
    """
    performance_controller = request.app.performance_controller
    dashboard = await performance_controller.get_parent_dashboard(student_id)
    return JSONResponse(content={"signal": "PARENT_DASHBOARD_RETRIEVED", **dashboard})


@performance_router.post("/teacher")
async def teacher_dashboard(request: Request, teacher_request: TeacherDashboardRequest):
    """
    Teacher dashboard: per-topic class averages, struggling students count, flagged weak topics.
    """
    performance_controller = request.app.performance_controller
    dashboard = await performance_controller.get_teacher_dashboard(teacher_request.topic_ids)
    return JSONResponse(content={"signal": "TEACHER_DASHBOARD_RETRIEVED", **dashboard})


@performance_router.post("/roadmap")
async def generate_roadmap(request: Request, roadmap_request: RoadmapRequest):
    """
    Generate a personalized 2-week LLM study roadmap for a student based on weak topics.
    """
    performance_controller = request.app.performance_controller
    roadmap = await performance_controller.generate_roadmap(
        student_id=roadmap_request.student_id,
        student_name=roadmap_request.student_name,
        grade=roadmap_request.grade,
        student_level=roadmap_request.student_level,
        weak_topic_names=roadmap_request.weak_topic_names,
        weak_topic_ids=roadmap_request.weak_topic_ids,
    )

    if not roadmap:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": "ROADMAP_GENERATION_FAILED"},
        )

    return JSONResponse(
        content={
            "signal": "ROADMAP_GENERATED",
            "roadmap_id": str(roadmap.id),
            "student_id": roadmap_request.student_id,
            "llm_explanation": roadmap.llm_explanation,
            "generated_at": str(roadmap.generated_at),
        }
    )


@performance_router.get("/roadmap/{student_id}")
async def get_latest_roadmap(request: Request, student_id: str):
    """Retrieve the most recent roadmap for a student."""
    performance_controller = request.app.performance_controller
    roadmap = await performance_controller.get_latest_roadmap(student_id)

    if not roadmap:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": "ROADMAP_NOT_FOUND"},
        )

    return JSONResponse(
        content={
            "signal": "ROADMAP_RETRIEVED",
            "roadmap_id": str(roadmap.id),
            "student_id": student_id,
            "llm_explanation": roadmap.llm_explanation,
            "weak_topic_ids": [str(tid) for tid in roadmap.weak_topic_ids],
            "generated_at": str(roadmap.generated_at),
        }
    )
