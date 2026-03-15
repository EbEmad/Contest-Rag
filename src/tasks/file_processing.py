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
# Add a file handler to capture worker logs since we can't see the worker terminal
file_handler = logging.FileHandler("/home/ebemad/a/Contest-Rag/worker_debug.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

@celery_app.task(bind=True,name="tasks.file_processing.process_project_files",autoretry_for=(Exception,),retry_kwargs={"max_retries":3,"countdown":60})
def process_project_files(self,project_id,file_id:int,chunk_size:int,overlap_size:int,do_reset:int,curriculum_metadata:dict=None):
    logger.info(f"STARTING TASK: process_project_files for project {project_id}, file {file_id}")
    return asyncio.run(
        _process_project_files(self, project_id, file_id, chunk_size,
                               overlap_size, do_reset, curriculum_metadata)
    )

async def _process_project_files(task_instance, project_id: str, 
                                 file_id: str, chunk_size: int,
                                 overlap_size: int, do_reset: int,
                                 curriculum_metadata: dict = None):

    
    mongo_conn,vectordb_client=None,None
    try:
        logger.info(f"Loading setup utils for task...")
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
        logger.info("Setup utils loaded successfully.")

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
            logger.warning(f"Task already executed or running | status: {existing_task.status}")
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
        logger.info(f"Task record {task_record.id} set to STARTED.")

        project = await project_model.get_project_or_create_one(
            project_id=project_id
        )
        logger.info(f"Using project {project_id} (ID: {project.id})")

        project_files_ids = {}
        if file_id:
            asset_record = None
            
            # Try lookup by ObjectId first (since API returns IDs)
            if isinstance(file_id, str) and len(file_id) == 24:
                try:
                    asset_record = await asset_model.get_asset_by_id(file_id)
                except:
                    pass
            
            # Fallback to lookup by name
            if not asset_record:
                asset_record = await asset_model.get_asset_record(
                    asset_project_id=project.id,
                    asset_name=file_id
                )

            if asset_record is None:
                logger.error(f"Asset not found for file_id: {file_id}")
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
                raise Exception(f"File with ID {file_id} not found")
            
            logger.info(f"Found asset {asset_record.asset_name} (ID: {asset_record.id})")
            project_files_ids = {
                asset_record.id: asset_record.asset_name
            }
        
        else:
            logger.info("No file_id provided, processing all project files.")
            project_files = await asset_model.get_all_project_assets(
                asset_project_id=project.id,
                asset_type=AssetTypeEnum.FILE.value,
            )

            project_files_ids = {
                record.id: record.asset_name
                for record in project_files
            }

        if len(project_files_ids) == 0:
            logger.warning("No files found to process.")
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
        
        process_controller = ProcessController(project_id=project_id)

        if do_reset == 1:
            logger.info("Resetting project chunks (do_reset=1)")
            _ = await chunk_model.delete_chunks_by_project_id(
                project_id=project.id
            )

        
        # Process all files concurrently

        tasks = [
            _process_single_file(asset_id, file_id, process_controller, chunk_model, project, chunk_size, overlap_size,curriculum_metadata)
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

async def _process_single_file(asset_id, filename, process_controller, chunk_model, project, chunk_size, overlap_size,curriculum_metadata):
            """Process a single file."""
            logger.info(f"Processing file: {filename}")
            file_content = await process_controller.get_file_content(file_id=filename)
            
            if file_content is None:
                logger.error(f"Error while loading content for file: {filename}")
                return 0
            
            # Temporarily switch to standard chunking for debugging
            logger.info("Using standard RecursiveCharacterTextSplitter for processing.")
            file_chunks = await process_controller.process_file_content(
                file_content=file_content,
                file_id=filename,
                chunk_size=chunk_size,
                overlap_size=overlap_size
            )
            
            if file_chunks is None or len(file_chunks) == 0:
                logger.warning(f"No chunks created for file: {filename}")
                return 0
            
            logger.info(f"Created {len(file_chunks)} chunks for file: {filename}")
            
            # Skip empty chunks; DataChunk requires chunk_text min_length=1
            non_empty = [c for c in file_chunks if (c.page_content or "").strip()]
            
            # Build chunk metadata
            file_chunks_records = []
            for i, chunk in enumerate(non_empty):
                # Start with original chunk metadata
                chunk_meta = chunk.metadata.copy() if chunk.metadata else {}
                
                # Add curriculum metadata if provided
                if curriculum_metadata:
                    chunk_meta.update({
                        "grade": curriculum_metadata.get("grade"),
                        "subject": curriculum_metadata.get("subject"),
                        "chapter_name": curriculum_metadata.get("chapter_name"),
                        "topic_names": curriculum_metadata.get("topic_names")
                    })
                
                file_chunks_records.append(
                    DataChunk(
                        chunk_text=chunk.page_content.strip(),
                        chunk_metadata=chunk_meta,
                        chunk_order=i + 1,
                        chunk_project_id=project.id,
                        chunk_asset_id=asset_id
                    )
                )

            logger.info(f"Inserting {len(file_chunks_records)} chunk records into DB...")
            return await chunk_model.insert_many_chunks(chunks=file_chunks_records)