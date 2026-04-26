from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, CoHereProvider, GeminiProvider
import asyncio

class LangChainEmbeddingWrapper:
    """Bridges our async LLM providers with LangChain's sync-based Embedding interface."""
    def __init__(self, provider_instance):
        self.provider = provider_instance

    def embed_documents(self, texts):
        return asyncio.run(self.provider.embed_texts_batch(texts))

    def embed_query(self, text):
        return asyncio.run(self.provider.embed_text(text))

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key = self.config.OPENAI_API_KEY,
                api_url = self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        if provider == LLMEnums.COHERE.value:
            return CoHereProvider(
                api_key = self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        if provider == LLMEnums.GEMINI.value:
            return GeminiProvider(
                api_key=self.config.GEMINI_API_KEY,
                api_url=self.config.GEMINI_API_URL,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        return None

    def create_embeddings(self, provider: str = None, model_id: str = None):
        """Create a LangChain-compatible embedding object based on provided or default settings."""
        target_provider = provider or self.config.EMBEDDING_BACKEND
        instance = self.create(target_provider)
        if not instance:
            return None
        
        # Use provided model_id or configuration default
        target_model = model_id or self.config.EMBEDDING_MODEL_ID
        instance.set_embedding_model(target_model, self.config.EMBEDDING_MODEL_SIZE)
        
        return LangChainEmbeddingWrapper(instance)