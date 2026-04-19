from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import SearchRequest
from models import ResponseSignal

import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)


async def _build_chat_history(nlp_controller, chat_message_model, student_id: str, project_id: str) -> list:
    """Build chat history from stored messages for a student+project pair."""
    if not student_id or not chat_message_model:
        return []

    past_messages = await chat_message_model.get_recent_messages(
        student_id=student_id,
        project_id=project_id,
        limit=10,
    )

    chat_history = []
    for msg in past_messages:
        if msg.role in ("model", "assistant"):
            role = nlp_controller.generation_client.enums.ASSISTANT.value
        else:
            role = nlp_controller.generation_client.enums.USER.value
        formatted_msg = await nlp_controller.generation_client.construct_prompt(prompt=msg.content, role=role)
        chat_history.append(formatted_msg)

    return chat_history


async def _persist_chat_turn(chat_message_model, student_id: str, project_id: str, question: str, answer: str):
    """Save a user question + model answer to chat history."""
    if not student_id or not chat_message_model:
        return
    await chat_message_model.add_message(
        student_id=student_id, project_id=project_id, role="user", content=question,
    )
    await chat_message_model.add_message(
        student_id=student_id, project_id=project_id, role="model", content=answer,
    )


@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):
    
    project_model = request.app.project_model

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = request.app.nlp_controller

    collection_info = nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):
    
    project_model = request.app.project_model

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = request.app.nlp_controller

    results = await nlp_controller.search_by_curriculum(
        project=project,
        query=search_request.text,
        grade=search_request.grade,
        subject=search_request.subject,
        limit=search_request.limit
    )

    if not results:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value
                }
            )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [ result.dict()  for result in results ]
        }
    )

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: str, search_request: SearchRequest):
    
    project_model = request.app.project_model
    project = await project_model.get_project_or_create_one(project_id=project_id)
    nlp_controller = request.app.nlp_controller
    chat_message_model = getattr(request.app, "chat_message_model", None)

    chat_history = await _build_chat_history(
        nlp_controller, chat_message_model,
        search_request.student_id, project_id,
    )

    answer, full_prompt, _ = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
        grade=search_request.grade,
        subject=search_request.subject,
        chat_history=chat_history,
    )

    if not answer:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.RAG_ANSWER_ERROR.value
                }
        )

    await _persist_chat_turn(
        chat_message_model, search_request.student_id,
        project_id, search_request.text, answer,
    )

    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
        }
    )

@nlp_router.post("/index/answer_stream/{project_id}")
async def answer_rag_stream(request: Request, project_id: str, search_request: SearchRequest):
    from fastapi.responses import StreamingResponse
    
    project_model = request.app.project_model
    project = await project_model.get_project_or_create_one(project_id=project_id)
    nlp_controller = request.app.nlp_controller
    chat_message_model = getattr(request.app, "chat_message_model", None)

    chat_history = await _build_chat_history(
        nlp_controller, chat_message_model,
        search_request.student_id, project_id,
    )

    async def event_generator():
        full_answer_parts = []
        async for chunk in nlp_controller.answer_rag_question_stream(
            project=project,
            query=search_request.text,
            limit=search_request.limit,
            grade=search_request.grade,
            subject=search_request.subject,
            chat_history=chat_history,
        ):
            if await request.is_disconnected():
                break
            full_answer_parts.append(chunk)
            formatted_chunk = chunk.replace("\n", "\ndata: ")
            yield f"data: {formatted_chunk}\n\n"

        if full_answer_parts:
            full_answer = "".join(full_answer_parts)
            await _persist_chat_turn(
                chat_message_model, search_request.student_id,
                project_id, search_request.text, full_answer,
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")
