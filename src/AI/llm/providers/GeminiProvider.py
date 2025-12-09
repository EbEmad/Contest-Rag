from ..LLMInterface import LLMInterface
from ..LLMEnums import GeminiEnums, DocumentTypeEnum
from google import genai
from google.genai.types import EmbedContentConfig, GenerateContentConfig


class GeminiProvider(LLMInterface):
    def __init__(self,api_key:str,
                default_max_input_characters: int=1000,
                default_max_output_tokens: int=1000,
                default_temperature: float=0.7):
    
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)
        
        self.default_max_input_characters = default_max_input_characters
        self.default_max_output_tokens = default_max_output_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None


        self.enums=GeminiEnums
        self.logger = logging.get_logger(__name__)


    def set_generation_model(self,model_id:str):
        self.generation_model_id=model_id

    def set_embedding_model(self,model_id:str,embedding_size:int):
        self.embedding_model_id=model_id
        self.embedding_size=embedding_size
    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        if self.client is None:
            self.logger.error("Gemini client is not initialized.")
            return None
        
        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None
        
        try:
            config=GenerateContentConfig(
                system_instruction=prompt,
                temperature=temperature or self.default_temperature,
                max_output_tokens=max_output_tokens or self.default_max_output_tokens,
            )

            chat = self.client.aio.chats.create(model=self.generation_model_id)
            response= chat.send_message(
                message=chat_history,
                config=config,
            )

            if not response or not response.text:
                self.logger.error("Gemini response is empty")
                return None
            
            return response.text

        except Exception as e:
            self.logger.error(f"Error generating text with Gemini: {str(e)}")
            return None
    def embed(self, text: str, document_type: str = None):
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
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "parts": [self.process_text(prompt)]
        }

        

        

    

