from .BaseController import BaseController
from .ProjectController import ProjectController
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import ProcessingEnum
from langchain_experimental.text_splitter import SemanticChunker
from AI.llm.LLMProviderFactory import LLMProviderFactory
from helpers.config import get_settings
class ProcessController(BaseController):
    
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.settings = get_settings()
        self.factory = LLMProviderFactory(self.settings)
    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]
    async def get_file_loader(self, file_id: str):
        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(self.project_path, file_id)
        if not os.path.exists(file_path):
            return None
        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")
        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)
        
        return None
    async def get_file_content(self, file_id: str):
        """Load file content asynchronously using thread pool."""
        loader = await self.get_file_loader(file_id=file_id)
        if loader:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, loader.load)
        return None
    async def process_file_content(self, file_content: list, file_id: str,
                                   chunk_size: int=1000, overlap_size: int=200):
        """Process file content asynchronously."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
        )
        file_content_texts = [rec.page_content for rec in file_content]
        file_content_metadata = [rec.metadata for rec in file_content]
        # Run chunking in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            self.executor,
            text_splitter.create_documents,
            file_content_texts,
            file_content_metadata
        )
        
        return chunks
    async def process_file_content_semantic(self, file_content: list, file_id: str,
                                        embedding_model: str = None):
        """Use semantic chunking instead of fixed-size chunks."""
        
        # Simple and clean: delegate embedding creation to the factory
        embeddings = self.factory.create_embeddings(model_id=embedding_model)

        if not embeddings:
            self.logger.error("Failed to create embedding provider from factory")
            return None

        text_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
        )

        file_content_texts = [rec.page_content for rec in file_content]
        file_content_metadata = [rec.metadata for rec in file_content]

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            self.executor,
            text_splitter.create_documents,
            file_content_texts,
            file_content_metadata,
        )

        return chunks