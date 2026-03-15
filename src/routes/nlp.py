from fastapi import  APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest
from models import ResponseSignal
from tasks.data_indexing import index_data_content

import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):

    task=index_data_content.delay(
        project_id=project_id,
        do_reset=push_request.do_reset
    )
    return JSONResponse(
        content={
            "signal":ResponseSignal.DATA_PUSH_TASK_READY.value,
            "task_id":task.id
        }
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

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = request.app.nlp_controller

    # ── Build chat history from memory (if student_id provided) ──
    chat_history = []
    chat_message_model = getattr(request.app, "chat_message_model", None)

    if search_request.student_id and chat_message_model:
        past_messages = await chat_message_model.get_recent_messages(
            student_id=search_request.student_id,
            project_id=project_id,
            limit=10,
        )
        chat_history = []
        for msg in past_messages:
            # Map database roles ("user", "model") to provider-specific enums
            if msg.role == "model" or msg.role == "assistant":
                role = nlp_controller.generation_client.enums.ASSISTANT.value
            else:
                role = nlp_controller.generation_client.enums.USER.value
            formatted_msg = await nlp_controller.generation_client.construct_prompt(prompt=msg.content, role=role)
            chat_history.append(formatted_msg)

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

    # ── Persist conversation turn ──
    if search_request.student_id and chat_message_model:
        await chat_message_model.add_message(
            student_id=search_request.student_id,
            project_id=project_id,
            role="user",
            content=search_request.text,
        )
        await chat_message_model.add_message(
            student_id=search_request.student_id,
            project_id=project_id,
            role="model",
            content=answer,
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

    # ── Build chat history from memory ──
    chat_history = []
    if search_request.student_id and chat_message_model:
        past_messages = await chat_message_model.get_recent_messages(
            student_id=search_request.student_id,
            project_id=project_id,
            limit=10,
        )
        chat_history = []
        for msg in past_messages:
            if msg.role == "model" or msg.role == "assistant":
                role = nlp_controller.generation_client.enums.ASSISTANT.value
            else:
                role = nlp_controller.generation_client.enums.USER.value
            formatted_msg = await nlp_controller.generation_client.construct_prompt(prompt=msg.content, role=role)
            chat_history.append(formatted_msg)

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

        # ── Persist conversation turn after stream completes ──
        if search_request.student_id and chat_message_model and full_answer_parts:
            full_answer = "".join(full_answer_parts)
            await chat_message_model.add_message(
                student_id=search_request.student_id,
                project_id=project_id,
                role="user",
                content=search_request.text,
            )
            await chat_message_model.add_message(
                student_id=search_request.student_id,
                project_id=project_id,
                role="model",
                content=full_answer,
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")

