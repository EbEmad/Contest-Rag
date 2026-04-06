from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController
import aiofiles
from models import ResponseSignal
import logging
from .schemes.data import ProcessRequest
from models.db_schemes import  Asset
from models.enums.AssetTypeEnum import AssetTypeEnum
# from tasks.file_processing import process_project_files
from tasks.process_workflow import process_and_push_workflow
logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: str, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
        
    
    project_model=request.app.project_model

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    # validate the file properties
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )


    asset_model=request.app.asset_model

    asset_resource = Asset(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path)
    )

    asset_record = await asset_model.create_asset(asset=asset_resource)

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                "file_id": str(asset_record.id),
            }
        )

# @data_router.post("/process/{project_id}")
# async def process_endpoint(request: Request, project_id: str, process_request: ProcessRequest):
#     """
#     Process uploaded files and optionally tag with curriculum metadata.
    
#     If curriculum metadata (grade, subject) is provided,
#     chunks will be tagged for curriculum-aware RAG filtering.
#     """
    
#     chunk_size = process_request.chunk_size
#     overlap_size = process_request.overlap_size
#     do_reset = process_request.do_reset
    
#     # Build curriculum metadata dict (only if provided)
#     curriculum_metadata = None
#     if process_request.grade is not None:
#         curriculum_metadata = {
#             "grade": process_request.grade,
#             "subject": process_request.subject,
#             "chapter_name": process_request.chapter_name,
#             "topic_names": process_request.topic_names
#         }

#     task=process_project_files.delay(
#         project_id=project_id,
#         file_id=process_request.file_id,
#         chunk_size=chunk_size,
#         overlap_size=overlap_size,
#         do_reset=do_reset,
#         curriculum_metadata=curriculum_metadata  # ← Pass curriculum metadata
#     )

#     return JSONResponse(
#         content={
#             "signal": ResponseSignal.PROCESSING_SUCCESS.value,
#             "task_id": task.id,
#             "curriculum_metadata": curriculum_metadata  # Return what was set
#         }
#     )




@data_router.post("/process_and_push/{project_id}")
async def process_and_push_endpoint(request: Request, project_id: str, process_request: ProcessRequest):
    """
    Process files and push to vector DB, optionally with curriculum metadata.
    """
    
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset
    
    # Build curriculum metadata dict (only if provided)
    curriculum_metadata = None
    if process_request.grade is not None:
        curriculum_metadata = {
            "grade": process_request.grade,
            "subject": process_request.subject,
            "chapter_name": process_request.chapter_name,
            "topic_names": process_request.topic_names
        }

    workflow_task=process_and_push_workflow.delay(
        project_id=project_id,
        file_id=process_request.file_id,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        do_reset=do_reset,
        curriculum_metadata=curriculum_metadata  # ← Pass curriculum metadata
    )

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROCESS_AND_PUSH_WORKFLOW_READY.value,
            "workflow_task_id": workflow_task.id,
            "curriculum_metadata": curriculum_metadata
        }
    )


    