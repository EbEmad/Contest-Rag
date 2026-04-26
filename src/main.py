from fastapi import FastAPI
from routes import base, data, nlp
from routes import quiz, performance
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from helpers.cache import CacheManager
from AI.llm.LLMProviderFactory import LLMProviderFactory
from AI.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from AI.llm.templates.template_parser import TemplateParser
from controllers import NLPController, QuizController, PerformanceController
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models.QuizModel import QuizModel
from models.PerformanceModel import PerformanceModel
from models.ChatMessageModel import ChatMessageModel

app = FastAPI()

async def startup_span():
    settings = get_settings()
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
    app.db = app.db_client

    
    app.cache_manager = CacheManager(
        redis_url=settings.REDIS_URL,
        ttl=getattr(settings, "CACHE_TTL", 3600),
    )

    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(settings)
    
    
    app.project_model = await ProjectModel.create_instance(db_client=app.db_client)
    app.chunk_model = await ChunkModel.create_instance(db_client=app.db_client)
    app.asset_model = await AssetModel.create_instance(db_client=app.db_client)

    
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )
    
   
    app.vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    app.vectordb_client.connect()

    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )


    app.nlp_controller = NLPController(
        vectordb_client=app.vectordb_client,
        generation_client=app.generation_client,
        embedding_client=app.embedding_client,
        template_parser=app.template_parser,
    )
    app.nlp_controller.cache = app.cache_manager
    app.nlp_controller.db = app.db_client

    
    app.quiz_model = await QuizModel.create_instance(db_client=app.db_client)
    app.quiz_controller = QuizController(
        quiz_model=app.quiz_model,
        generation_client=app.generation_client,
        template_parser=app.template_parser,
    )

 
    app.performance_model = await PerformanceModel.create_instance(db_client=app.db_client)
    app.performance_controller = PerformanceController(
        performance_model=app.performance_model,
        generation_client=app.generation_client,
        template_parser=app.template_parser,
    )

   
    app.chat_message_model = await ChatMessageModel.create_instance(db_client=app.db_client)


async def shutdown_span():
    app.mongo_conn.close()
    app.vectordb_client.disconnect()

app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(quiz.quiz_router)
app.include_router(performance.performance_router)