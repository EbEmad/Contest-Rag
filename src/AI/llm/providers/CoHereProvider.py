from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
from cohere import AsyncClient
import logging
from typing import List

class CoHereProvider(LLMInterface):

    def __init__(self, api_key: str,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):
        
        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = AsyncClient(api_key=self.api_key)

        self.enums = CoHereEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    async def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    async def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):

        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for CoHere was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        processed_prompt = await self.process_text(prompt)
        response = await self.client.chat(
            model = self.generation_model_id,
            chat_history = chat_history,
            message = processed_prompt,
            temperature = temperature,
            max_tokens = max_output_tokens
        )

        if not response or not response.text:
            self.logger.error("Error while generating text with CoHere")
            return None
        
        return response.text
    
    async def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model for CoHere was not set")
            return None
        
        input_type = CoHereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnum.QUERY.value:
            input_type = CoHereEnums.QUERY.value

        processed_text = await self.process_text(text)
        response = await self.client.embed(
            model = self.embedding_model_id,
            texts = [processed_text],
            input_type = input_type,
            embedding_types=['float'],
        )

        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("Error while embedding text with CoHere")
            return None
        
        return response.embeddings.float[0]

    async def embed_texts_batch(self, texts: List[str], document_type: str = None, batch_size: int = 100) -> List[List[float]]:
        if not self.client or not self.embedding_model_id:
            self.logger.error("CoHere client or embedding model not set")
            return []

        input_type = CoHereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnum.QUERY.value:
            input_type = CoHereEnums.QUERY.value

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            processed_batch = [await self.process_text(text) for text in batch]
            try:
                response = await self.client.embed(
                    model=self.embedding_model_id,
                    texts=processed_batch,
                    input_type=input_type,
                    embedding_types=['float'],
                )
                if response and response.embeddings and response.embeddings.float:
                    all_embeddings.extend(response.embeddings.float)
                else:
                    self.logger.error("CoHere batch embedding returned empty result")
                    return []
            except Exception as e:
                self.logger.error(f"CoHere batch embedding error: {e}")
                return []

        return all_embeddings
    
    async def generate_text_stream(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                                   temperature: float = None):
        if not self.client:
            self.logger.error("CoHere client was not set")
            return

        if not self.generation_model_id:
            self.logger.error("Generation model for CoHere was not set")
            return

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        processed_prompt = await self.process_text(prompt)
        response = await self.client.chat_stream(
            model=self.generation_model_id,
            chat_history=chat_history,
            message=processed_prompt,
            temperature=temperature,
            max_tokens=max_output_tokens
        )

        async for event in response:
            if event.event_type == "text-generation":
                yield event.text

    async def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "text": await  self.process_text(prompt)
        }