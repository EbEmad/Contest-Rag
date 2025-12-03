from enum import Enum


class LLMEnums(Enum):
    OPENAI="OPENAI"
    COHERE="COHERE"
    OLLAMA="OLLAMA"
    GEMINI="GEMINI"

class OpenAIEnums(Enum):
    SYSTEM= "system"
    USER= "user"
    ASSISTANT= "assistant"

class CohereEnums(Enum):
    SYSTEM= "SYSTEM"
    USER= "USER"
    ASSISTANT= "CHATBOT"
    DOCUMENT="search_document"
    QUERY="search_query"
    
class DocumentTypeEnum(Enum):
    DOCUMENT="document"
    QUERY="query"