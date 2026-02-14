from celery_app import celery_app,get_setup_utils
from helpers.config import get_settings
from controllers import ProcessController
from models.db_schemes import DataChunk
from fastapi.responses import JSONResponse
from models import ResponseSignal
from models.enums.AssetTypeEnum import AssetTypeEnum
from utils.idempotency_manager import IdempotencyManager
import asyncio
import logging
logger = logging.getLogger(__name__)

@celery_app.task(bind=True,name="tasks.file_processing.process_project_files",autoretry_for=(Exception,),retry_kwargs={"max_retries":3,"countdown":60})
def process_project_files(self,project_id,file_id:int,chunk_size:int,overlap_size:int,do_reset:int):
    return asyncio.run(
        _process_project_files(self, project_id, file_id, chunk_size,
                               overlap_size, do_reset)
    )

async def _process_project_files(task_instance, project_id: int, 
                                 file_id: int, chunk_size: int,
                                 overlap_size: int, do_reset: int):
    
    mongo_conn,vectordb_client=None,None
    try:
        (
            mongo_conn,
            db_client,
            llm_provider_factory,
            vectordb_provider_factory,
            generation_client,
            embedding_client,
            vectordb_client,
            template_parser,
            project_model,
            chunk_model,
            asset_model,
            nlp_controller
        ) = await get_setup_utils()

        # Create idempotency manager
        idempotency_manager = IdempotencyManager(db_client)
        task_args = {
            "project_id": project_id,
            "file_id": file_id,
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "do_reset": do_reset
        }
        
        task_name = "tasks.file_processing.process_project_files"

        settings=get_settings()

        should_execute,existing_task=await idempotency_manager.should_execute_task(
            task_name=task_name,
            task_args=task_args,
            celery_task_id=task_instance.request.id,
            task_time_limit=settings.CELERY_TASK_TIME_LIMIT
        )

        if not should_execute:
            logger.warning(f"Can not handle th task | status: {existing_task.status}")
            return existing_task.result

        task_record=None
        if existing_task:
            # Update existing task with new celery task ID
            await idempotency_manager.update_task_status(
                execution_id=existing_task.id,
                status='PENDING'
            )
            task_record = existing_task
        else:
            # Create new task record
            task_record=await idempotency_manager.create_task_record(
                task_name=task_name,
                task_args=task_args,
                celery_task_id=task_instance.request.id
            )

        # Update status to STARTED
        await idempotency_manager.update_task_status(
            execution_id=task_record.id,
            status='STARTED'
        )
       

        project = await project_model.get_project_or_create_one(
            project_id=project_id
        )

        # asset_model = await AssetModel.create_instance(
        #         db_client=db_client
        #     )

        project_files_ids = {}
        if file_id:
            asset_record = await asset_model.get_asset_record(
                asset_project_id=project.id,
                asset_name=file_id
            )

            if asset_record is None:
                task_instance.update_state(
                        state="FAILURE",
                        meta={
                            "signal": ResponseSignal.FILE_ID_ERROR.value,
                        }
                    )
                # Update task status to FAILURE
                await idempotency_manager.update_task_status(
                    execution_id=task_record.id,
                    status='FAILURE',
                    result={"signal": ResponseSignal.FILE_ID_ERROR.value}
                )
                raise Exception(f"File with ID {file_id} ")
                # return JSONResponse(
                #     status_code=status.HTTP_400_BAD_REQUEST,
                #     content={
                #         "signal": ResponseSignal.FILE_ID_ERROR.value,
                #     }
                # )

            project_files_ids = {
                asset_record.id: asset_record.asset_name
            }
        
        else:
            

            project_files = await asset_model.get_all_project_assets(
                asset_project_id=project.id,
                asset_type=AssetTypeEnum.FILE.value,
            )

            project_files_ids = {
                record.id: record.asset_name
                for record in project_files
            }

        if len(project_files_ids) == 0:

            task_instance.update_state(
                        state="FAILURE",
                        meta={
                            "signal": ResponseSignal.NO_FILES_ERROR.value,
                        }
                )
            # Update task status to FAILURE
            await idempotency_manager.update_task_status(
                execution_id=task_record.id,
                status='FAILURE',
                result={"signal": ResponseSignal.NO_FILES_ERROR.value}
            )
            raise Exception(f"No files found in project {project.project_id}")
            
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.NO_FILES_ERROR.value,
                }
            )
        
        process_controller = ProcessController(project_id=project_id)

        no_records = 0
        no_files = 0

        # chunk_model = await ChunkModel.create_instance(
        #                     db_client=db_client
        #                 )

        if do_reset == 1:
            _ = await chunk_model.delete_chunks_by_project_id(
                project_id=project.id
            )

        async def process_single_file(asset_id, file_id, process_controller, chunk_model, project, chunk_size, overlap_size):
            """Process a single file."""
            file_content = await process_controller.get_file_content(file_id=file_id)
            
            if file_content is None:
                logger.error(f"Error while processing file: {file_id}")
                return 0
            
            file_chunks = await process_controller.process_file_content_semantic(
                file_content=file_content,
                file_id=file_id,
                #chunk_size=chunk_size,
                #overlap_size=overlap_size
            )
            
            if file_chunks is None or len(file_chunks) == 0:
                logger.warning(f"No chunks created for file: {file_id}")
                pass 
            
            # Skip empty chunks; DataChunk requires chunk_text min_length=1
            non_empty = [c for c in file_chunks if (c.page_content or "").strip()]
            file_chunks_records = [
                DataChunk(
                    chunk_text=chunk.page_content.strip(),
                    chunk_metadata=chunk.metadata,
                    chunk_order=i + 1,
                    chunk_project_id=project.id,
                    chunk_asset_id=asset_id
                )
                for i, chunk in enumerate(non_empty)
            ]
            
            return await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        # Process all files concurrently

        tasks = [
            process_single_file(asset_id, file_id, process_controller, chunk_model, project, chunk_size, overlap_size)
            for asset_id, file_id in project_files_ids.items()
        ]
        results = await asyncio.gather(*tasks)
        no_records = sum(results)
        no_files = len([r for r in results if r > 0])

        task_instance.update_state(
                state="SUCCESS",
                meta={
                    "signal": ResponseSignal.PROCESSING_SUCCESS.value,
                }
            )
        
        # Update task status to FAILURE
        await idempotency_manager.update_task_status(
            execution_id=task_record.id,
            status='SUCCESS',
            result={"signal": ResponseSignal.PROCESSING_SUCCESS.value}
        )
        
        return {
                "signal": ResponseSignal.PROCESSING_SUCCESS.value,
                "inserted_chunks": no_records,
                "processed_files": no_files,
                "project_id":project_id,
                "do_reset":do_reset
            }
    except Exception as e:
        logger.error(f"task failed: {str(e)}")
        raise
    finally:
        try:
            if mongo_conn:
                mongo_conn.close()
            if vectordb_client:
                vectordb_client.disconnect()
        except Exception as e:
            logger.error(f"Task failed while cleaning: {str(e)}")
