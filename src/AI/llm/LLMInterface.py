from abc import ABC, abstractmethod
from typing import List
class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id: str):
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        pass

    @abstractmethod
    async def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        pass

    @abstractmethod
    async def embed_text(self, text: str, document_type: str = None):
        pass

    @abstractmethod
    async def construct_prompt(self, prompt: str, role: str):
        pass

    @abstractmethod
    async def generate_text_stream(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                                   temperature: float = None):
        """Generate text in a streaming fashion. Yields text chunks."""
        pass

    async def embed_texts_batch(self, texts: List[str], document_type: str = None, batch_size: int = 100) -> List[List[float]]:
        """Embed multiple texts in batches. Override for provider-specific batching."""
        pass