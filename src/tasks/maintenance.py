from celery_app import celery_app,get_setup_utils
from helpers.config import get_settings
from utils.idempotency_manager import IdempotencyManager


import asyncio
import logging
logger = logging.getLogger(__name__)


@celery_app.task(
                 bind=True, name="tasks.maintenance.clean_celery_executions_table",
                 autoretry_for=(Exception,),
                 retry_kwargs={'max_retries': 3, 'countdown': 60}
                )
def clean_celery_executions_table(self):
    return asyncio.run(
        _clean_celery_executions_table(self)
    )

async def _clean_celery_executions_table(task_instance):
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
        _=await idempotency_manager.cleanup_old_tasks()
        return True
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