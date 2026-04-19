from celery import Celery
from helpers.config import get_settings
from AI.llm.LLMProviderFactory import LLMProviderFactory
from AI.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from AI.llm.templates.template_parser import TemplateParser
from controllers import NLPController
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from helpers.cache import CacheManager
from motor.motor_asyncio import AsyncIOMotorClient
settings = get_settings()


async def get_setup_utils():
    settings = get_settings()
    mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    db_client = mongo_conn[settings.MONGODB_DATABASE]

   
    cache_manager = CacheManager(
        redis_url=settings.REDIS_URL,
        ttl=getattr(settings, "CACHE_TTL", 3600),
    )

    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(settings)
    
    
    project_model=await  ProjectModel.create_instance(db_client=db_client)

  
    chunk_model=await ChunkModel.create_instance(db_client=db_client)
    
    
    asset_model= await AssetModel.create_instance(db_client=db_client)

    
    generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)

   
    embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)
    
   
    vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    vectordb_client.connect()

    template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

   
    nlp_controller=NLPController(
        vectordb_client=vectordb_client,
        generation_client=generation_client,
        embedding_client=embedding_client,
        template_parser=template_parser,
    )
   
    nlp_controller.cache = cache_manager
    
    nlp_controller.db = db_client

    return (
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
    )


celery_app = Celery(
    settings.APP_NAME,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.file_processing",
        "tasks.data_indexing",
        "tasks.process_workflow",
        "tasks.maintenance"
    ]
)

celery_app.conf.update(
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_TASK_SERIALIZER,
    accept_content=[
        settings.CELERY_TASK_SERIALIZER
    ],
    task_acks_late=settings.CELERY_TASK_ACKS_LATE,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,

    task_ignore_result=False,
    result_expires=3600,

    worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,

    # Connection settings for better reliability
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    worker_cancel_long_running_tasks_on_connection_loss=True,

    task_routes={
        "tasks.file_processing.process_project_files": {"queue": "file_processing"},
        "tasks.data_indexing.index_data_content": {"queue": "data_indexing"},
        "tasks.process_workflow.process_and_push_workflow":{"queue": "file_processing"},
        "tasks.maintenance.clean_celery_executions_table": {"queue": "default"},
    },

    beat_schedule={
        'cleanup-old-task-records':{
            'task':"tasks.maintenance.clean_celery_executions_table",
            'schedule':10,
            'args':()
        }
    },

    timezone='UTC',
)
celery_app.conf.task_default_queue = "default"