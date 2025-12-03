from ..LLMInterface import LLMInterface
from ..LLMEnums import CohereEnums,DocumentTypeEnum
import cohere
import logging

class CoHereProvider(LLMInterface):

    def __init__(self,api_key:str,
                    default_input_max_characters:int=1000,
                    default_generation_max_output_tokens:int= 1000,
                    temperature:float=0.0):
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = temperature
        self.generation_model_id=None

        self.embedding_model_id=None
        self.embedding_size=None


        self.client=cohere.client(api_key=self.api_key)

        self.logger=logging.getLogger(__name__)

    def set_generation_model(self,model_id:str):
        self.generation_model_id=model_id

    def set_embedding_model(self,model_id:str,embdding_size:int):
        self.embedding_model_id=model_id
        self.embdding_size=embdding_size

    def process_text(self,text:str):        
        return text[:self.default_input_max_characters].strip()


    def generate_text(self,prompt:str,
                        chat_history:list=[],
                        max_output_tokens:int=None,
                        temperature:float=None):
        if not self.client:
            self.logger.error("CoHere client is not initialized")
            return None
        
        if not self.generation_model_id:
            self.logger.error("CoHere generation model is not initialized")
            return None
        
        max_output_tokens=max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature=temperature if temperature else self.default_generation_temperature

        response=self.client.chat(
            model=self.generation_model_id,
            chat_history=chat_history,
            message=self.process_text(prompt),
            
        )

        if not response or not response.text:
            self.logger.error("CoHere response is not valid")
            return None

        return response.text


    def embed_text(self,text:str,documnet_type:str=None):
        if not self.client:
            self.logger.error("CoHere client is not initialized")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("CoHere embedding model is not initialized")
            return None

        input_type=CohereEnums.DOCUMENT.value

        if document_type==DocumentTypeEnum.QUERY:
            input_type=CohereEnums.QUERY.value

        response=self.client.embed(
            model=self.embedding_model_id,
            text=[self.process_text(text)],
            input_type=input_type,
            embedding_types=["float"],
        )

        if not response or not response.embeddings or response.embeddings.float:
            self.logger.error("CoHere response is not valid")
            return None

        return response.embeddings.float[0]

    def construct_prompt(self,prompt:str,role:str):
        return {
            "role":role,
            "text": self.process_text(prompt)
        }


