import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from routes.schemes.performance import RoadmapRequest

logger = logging.getLogger("uvicorn.error")

performance_router = APIRouter(
    prefix="/api/v1/performance",
    tags=["api_v1", "performance"],
)


@performance_router.post("/roadmap")
async def generate_roadmap(request: Request, roadmap_request: RoadmapRequest):
    """
    Generate a personalized 2-week LLM study roadmap for a student.
    All student data is provided by the calling backend.
    """
    performance_controller = request.app.performance_controller
    roadmap = await performance_controller.generate_roadmap(
        student_id=roadmap_request.student_id,
        student_name=roadmap_request.student_name,
        grade=roadmap_request.grade,
        student_level=roadmap_request.student_level,
        weak_topic_names=roadmap_request.weak_topic_names,
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
            "weak_topic_names": roadmap.weak_topic_names,
            "generated_at": str(roadmap.generated_at),
        }
    )
