from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import PDFMinerLoader,PyPDFLoader,csv_loader,TextLoader
from models import ProcessingEnum
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):

        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(
            self.project_path,
            file_id
        )
        if not os.path.exists(file_path):
            return None

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")  

        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)
        
        if file_ext == ProcessingEnum.CSV.value:
            return csv_loader(file_path,encoding="utf-8")
        
        if file_ext == ProcessingEnum.DOCX.value:
            return PDFMinerLoader(file_path)
        
        if file_ext == ProcessingEnum.TEXT.value:
            return TextLoader(file_path, encoding="utf-8")
        
        return None

    def get_file_content(self, file_id: str):

        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()
        return None

    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int=100, overlap_size: int=20):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
        )

        file_content_texts = [
            rec.page_content
            for rec in file_content
        ]

        file_content_metadata = [
            rec.metadata
            for rec in file_content
        ]

        chunks = text_splitter.create_documents(
            file_content_texts,
            metadatas=file_content_metadata
        )

        return chunks
        
