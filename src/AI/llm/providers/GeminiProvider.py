from ..LLMInterface import LLMInterface
from ..LLMEnums import GeminiEnums, DocumentTypeEnum
from google import genai
from google.genai.types import EmbedContentConfig, GenerateContentConfig
import logging


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
            config=GenerateContentConfig(
                temperature=temperature or self.default_generation_temperature,
                max_output_tokens=max_output_tokens or self.default_generation_max_output_tokens,
            )
            # create chat session
            chat = self.client.aio.chats.create(model=self.generation_model_id)
            # add user prompt to history
            chat_history.append(
                await self.construct_prompt(prompt=prompt,role=GeminiEnums.USER.value)
            )
            # send a message with a full history
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
            results =  self.client.aio.models.embed_content(
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
    async def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "parts": [await self.process_text(prompt)]
        }

        

        

    

