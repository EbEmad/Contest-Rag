from celery_app import celery_app,get_setup_utils
from helpers.config import get_settings
from fastapi.responses import JSONResponse
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController
from models import ResponseSignal
import asyncio
import logging
logger = logging.getLogger(__name__)

@celery_app.task(
        bind=True, name="tasks.data_indexing.index_data_content",
        autoretry_for=(Exception,),
        retry_kwargs={'max_retries': 3, 'countdown': 60}
    )
def index_data_content(self,project_id:int, do_reset:int):

    return asyncio.run(
        _index_data_content(self,project_id, do_reset)
    )

async def _index_data_content(task_instance,project_id:int, do_reset:int):
    
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
        
        logger.info("Setup utils were loaded!")
  
        project = await project_model.get_project_or_create_one(
            project_id=project_id
        )

        if not project:
            task_instance.update_state(
                state='FAILURE',
                meta={
                    "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
                }
            )
            raise Exception("Project not found for project_id: {project_id}")
        
        has_records = True
        page_no = 1
        inserted_items_count = 0
        idx = 0

        while has_records:
            page_chunks = await chunk_model.get_poject_chunks(project_id=project.id, page_no=page_no)
            if len(page_chunks):
                page_no += 1
            
            if not page_chunks or len(page_chunks) == 0:
                has_records = False
                break

            chunks_ids =  list(range(idx, idx + len(page_chunks)))
            idx += len(page_chunks)
            
            is_inserted = await nlp_controller.index_into_vector_db(
                project=project,
                chunks=page_chunks,
                do_reset=do_reset,
                chunks_ids=chunks_ids
            )

            if not is_inserted:
                task_instance.update_state(
                    state='FAILURE',
                    meta={
                        "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
                    }
                )
                raise Exception("Project not found for project_id: {project_id}")


                # return JSONResponse(
                #     status_code=status.HTTP_400_BAD_REQUEST,
                #     content={
                #         "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
                #     }
                # )
            
            inserted_items_count += len(page_chunks)
        task_instance.update_state(
                    state='SUCCESS',
                    meta={
                        "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value
                    }
            )
        return {
                "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
                "inserted_items_count": inserted_items_count
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
