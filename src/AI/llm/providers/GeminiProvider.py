from ..LLMInterface import LLMInterface
from ..LLMEnums import GeminiEnums, DocumentTypeEnum
from google import genai
from google.genai.types import EmbedContentConfig, GenerateContentConfig
import asyncio
import logging
from typing import List

class GeminiProvider(LLMInterface):
    def __init__(self,api_key:str,api_url:str,
                default_input_max_characters: int=1000,
                default_generation_max_output_tokens: int=1000,
                default_generation_temperature: float=0.7):
    
        self.api_key = api_key
        self.api_url = api_url
        
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client=genai.Client(
            api_key=self.api_key
        )

        self.enums=GeminiEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self,model_id:str):
        self.generation_model_id=model_id

    def set_embedding_model(self,model_id:str,embedding_size:int):
        self.embedding_model_id=model_id
        self.embedding_size=embedding_size

    async def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    async def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        if self.client is None:
            self.logger.error("Gemini client is not initialized.")
            return None
        
        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None
        
        try:
            # Extract system instruction from chat_history if present
            system_instruction = None
            clean_history = []
            if chat_history:
                for msg in chat_history:
                    if msg.get("role") == "system":
                        # Extract text from parts
                        parts = msg.get("parts", [])
                        if parts and isinstance(parts[0], dict):
                            system_instruction = parts[0].get("text")
                        elif parts and isinstance(parts[0], str):
                             system_instruction = parts[0]
                    else:
                        clean_history.append(msg)

            config=GenerateContentConfig(
                temperature=temperature or self.default_generation_temperature,
                max_output_tokens=max_output_tokens or self.default_generation_max_output_tokens,
                system_instruction=system_instruction
            )
            # create chat session with CLEAN history (no system role)
            chat = self.client.aio.chats.create(model=self.generation_model_id, history=clean_history)
            
            # send the user message (prompt)
            # define content directly or use construct_prompt? 
            # construct_prompt creates a dict {"role":..., "parts":...} which is for history.
            # send_message takes string or parts.
            
            response= await  chat.send_message(
                message=prompt,
                config=config,
            )

            if not response or not response.text:
                self.logger.error("Gemini response is empty")
                return None
            
            return response.text

        except Exception as e:
            self.logger.error(f"Error generating text with Gemini: {str(e)}")
            return None
    async def embed_text(self, text: str, document_type: str = None):
        if self.client is None:
            self.logger.error("Gemini client is not initialized.")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None
        
        try:
            task_type=self.enums.DOCUMENT.value
            if document_type == DocumentTypeEnum.QUERY.value:
                task_type = self.enums.QUERY.value

            config=EmbedContentConfig(task_type=task_type,output_dimensionality=self.embedding_size)
            results = await self.client.aio.models.embed_content(
                model=self.embedding_model_id,
                contents=text,
                config=config
            )

            if not results:
                self.logger.error("Error while embedding text with Gemini")
                return None

            return results.embeddings[0].values

        except Exception as e:
            self.logger.error(f"Error embedding text with Gemini: {str(e)}")
            return None
    async def embed_texts_batch(self, texts: List[str], document_type: str = None, batch_size: int = 100) -> List[List[float]]:
        """Batch embed texts using Gemini API (concurrent embed_content per batch)."""
        if not self.client or not self.embedding_model_id:
            self.logger.error("Gemini client or embedding model not set")
            return []

        task_type = self.enums.DOCUMENT.value
        if document_type == DocumentTypeEnum.QUERY.value:
            task_type = self.enums.QUERY.value
        config = EmbedContentConfig(task_type=task_type, output_dimensionality=self.embedding_size)

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_results = await asyncio.gather(*[
                    self.client.aio.models.embed_content(
                        model=self.embedding_model_id,
                        contents=text,
                        config=config,
                    )
                    for text in batch
                ])
                for results in batch_results:
                    if results and results.embeddings:
                        all_embeddings.append(results.embeddings[0].values)
                    else:
                        self.logger.error("Gemini batch embedding returned empty result for one item")
                        return []
            except Exception as e:
                self.logger.error(f"Batch embedding error: {e}")
                return []

        return all_embeddings
    async def generate_text_stream(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                                   temperature: float = None):
        if self.client is None:
            self.logger.error("Gemini client is not initialized.")
            return

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return

        try:
            # Extract system instruction from chat_history if present
            system_instruction = None
            clean_history = []
            if chat_history:
                for msg in chat_history:
                    if msg.get("role") == "system":
                        # Extract text from parts
                        parts = msg.get("parts", [])
                        if parts and isinstance(parts[0], dict):
                            system_instruction = parts[0].get("text")
                        elif parts and isinstance(parts[0], str):
                             system_instruction = parts[0]
                    else:
                        clean_history.append(msg)
            
            config = GenerateContentConfig(
                temperature=temperature or self.default_generation_temperature,
                max_output_tokens=max_output_tokens or self.default_generation_max_output_tokens,
                system_instruction=system_instruction
            )
            # Add user prompt to history (Gemini SDK handles history via the chat session)
            # However, our interface expects us to manage it or at least handle the current message.
            # The Gemini SDK `aio.chats.create` can take history.
            
            chat = self.client.aio.chats.create(model=self.generation_model_id, history=clean_history)
            
            response = await chat.send_message_stream(
                message=prompt,
                config=config,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            self.logger.error(f"Error generating text stream with Gemini: {str(e)}")
            return

    async def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "parts": [{"text": await self.process_text(prompt)}]
        }

        

        

    

