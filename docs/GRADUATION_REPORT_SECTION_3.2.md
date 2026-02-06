# 3.2 RAG Backend Implementation

This section outlines the architectural design and technical implementation of the Contest-RAG backend application, which serves as the core service for a Retrieval-Augmented Generation (RAG) system. The system enables users to upload documents, automatically chunk and index them into a vector store, and perform context-aware question answering using large language models (LLMs). Designed for scalability and provider flexibility, the backend supports multiple embedding and generation providers (OpenAI, Gemini, Cohere), semantic and hybrid search, optional reranking, and localized prompt templates, delivering a robust, maintainable API suitable for integration with web or mobile clients.

In addition to the core RAG pipeline, the application provides document lifecycle management (upload, processing, indexing), project-scoped collections, configurable chunking strategies (fixed-size, semantic, and line-based), and optional response caching. These features support reliable ingestion of educational or organizational content and consistent, document-grounded answers for end users.

The backend is implemented using modern Python technologies with an emphasis on async I/O, clear separation of concerns, and testability. This implementation ensures high throughput for embedding and indexing workloads, minimal latency for search and answer endpoints, and straightforward extension to new LLM or vector database providers.

---

## 3.2.1 Framework: FastAPI

FastAPI is a high-performance, asynchronous web framework for building APIs in Python. The choice of FastAPI for the Contest-RAG backend was driven by several strategic advantages:

- **Asynchronous Request Handling**: FastAPI is built on Starlette and supports native `async`/`await`, enabling non-blocking I/O for database access (MongoDB via Motor), embedding and LLM API calls, and file operations. This is critical for handling concurrent uploads, batch indexing, and multiple simultaneous RAG queries without blocking the event loop.

- **Automatic OpenAPI Documentation**: FastAPI generates interactive API documentation (Swagger UI and ReDoc) from type hints and route definitions. This simplifies integration for frontend or mobile clients and reduces the need for separate API specification documents.

- **Request and Response Validation**: Integration with Pydantic provides automatic validation of request bodies and path/query parameters. Request schemes (e.g. `ProcessRequest`, `SearchRequest`) ensure that chunk sizes, limits, and query text conform to expected types and constraints before reaching business logic, reducing runtime errors and improving security.

- **Dependency Injection and Application State**: The application uses FastAPI’s request-scoped dependency injection and attaches shared resources (MongoDB client, vector DB client, LLM providers, template parser, NLP controller) to the application instance during startup. Route handlers access these via `request.app`, avoiding global singletons and simplifying testing and configuration.

- **Performance and Standards**: FastAPI is one of the fastest Python web frameworks in benchmarks and supports HTTP/1.1, JSON serialization, and CORS configuration out of the box, making it suitable for production deployment behind a reverse proxy or load balancer.

---

## 3.2.2 Architecture

To ensure maintainability and clear separation of concerns, the Contest-RAG backend follows a layered, modular architecture. Business logic is decoupled from HTTP handling and from concrete implementations of vector storage and LLM providers, enabling easier testing, scalability, and long-term maintenance.

The project structure is organized into the following layers:

- **Routes Layer**: Defines REST API endpoints under `/api/v1/data` and `/api/v1/nlp`. Routes handle request parsing, response formatting, and HTTP status codes. They delegate all business logic to controllers and do not contain embedding, chunking, or LLM logic. Request and response shapes are defined in Pydantic schemes under `routes/schemes`.

- **Controllers Layer**: Implements the core application workflows. The `DataController` validates uploaded files and manages file paths; the `ProcessController` loads documents (PDF/TXT via LangChain loaders), runs chunking (RecursiveCharacterTextSplitter, SemanticChunker, or line-based splitting), and returns chunk lists. The `NLPController` orchestrates indexing (embedding chunks and inserting into the vector DB), semantic/hybrid search, optional reranking with a cross-encoder, RAG prompt construction from templates, and LLM-based answer generation. Controllers depend on abstractions (vector DB interface, LLM interface) rather than concrete providers.

- **AI Layer**: Provides pluggable implementations for embeddings, text generation, and vector storage. The `AI/llm` module defines an `LLMInterface` (embed_text, embed_texts_batch, generate_text, construct_prompt) and provider implementations (OpenAI, Gemini, Cohere) plus a factory for runtime selection. The `AI/vectordb` module defines a `VectorDBInterface` (create_collection, insert_many, search_by_vector, hybrid_search) with a Qdrant-based implementation. RAG prompt templates are stored under `AI/llm/templates/locales` (e.g. English and Arabic) and resolved via a `TemplateParser` according to configuration.

- **Models Layer**: Contains Pydantic data models and database schemas for projects, assets, and chunks (MongoDB), plus enums for asset types, processing options, and response signals. Data access is encapsulated in model classes (e.g. `ProjectModel`, `ChunkModel`, `AssetModel`) that interact with the MongoDB client.

- **Helpers Layer**: Centralizes configuration (environment-based settings via Pydantic Settings), optional caching (e.g. Redis), and shared utilities. Configuration includes API keys, model IDs, vector DB path and distance method, and primary/default language for templates.

This architecture keeps HTTP, business logic, and external services (LLMs, vector DB, document store) clearly separated and allows swapping providers or adding new chunking strategies without changing route or controller interfaces.

---

## 3.2.3 Document Ingestion and Chunking

Document ingestion is the first stage of the RAG pipeline. Uploaded files are stored on disk and registered as assets in MongoDB; processing produces text chunks that are persisted as chunk records and later embedded and indexed into the vector store.

- **File Upload and Validation**: The data upload endpoint accepts multipart form data (PDF or TXT). A `DataController` validates file type and size against application settings. Files are saved under a project-specific directory with unique generated filenames to avoid collisions. An `Asset` record is created in MongoDB linking the project, file identifier, and file size for later processing and indexing.

- **Chunking Strategies**: The system supports multiple chunking strategies to accommodate different document structures and use cases. **RecursiveCharacterTextSplitter** (LangChain) splits text by configurable chunk size and overlap (e.g. 1000 characters with 200 overlap), respecting paragraph and sentence boundaries. **SemanticChunker** (LangChain Experimental) uses an embedding model to compute similarity between adjacent sentences and splits at semantic boundaries, producing more coherent chunks for retrieval. **Line-based chunking** (split_by_lines) creates one chunk per non-empty line, which is suitable for structured content (e.g. FAQs or per-item descriptions) so that queries like “who is X” retrieve only the relevant line. Empty or whitespace-only chunks are filtered out before persistence to satisfy schema constraints (e.g. `chunk_text` minimum length).

- **Asynchronous Processing**: File loading (PyMuPDF for PDF, TextLoader for TXT) and CPU-bound chunking run inside a thread pool executor to avoid blocking the async event loop. Processed chunks are written to MongoDB with project and asset references and a sequential order, enabling paginated reads during indexing.

- **Idempotency and Reset**: Processing can optionally reset existing chunks for a project (do_reset) before inserting new ones, ensuring a clean state when re-processing documents with updated chunk sizes or strategies.

---

## 3.2.4 Embedding and Vector Store

The system converts text chunks into dense vector representations and stores them in a dedicated vector database for fast similarity search. This layer is abstracted behind an interface so that the storage backend can be replaced or extended (e.g. different distance metrics or scaling strategies).

- **Embedding Provider Abstraction**: All embedding calls go through the `LLMInterface`: single-text `embed_text` and batch `embed_texts_batch` with configurable batch size. Providers (OpenAI, Gemini, Cohere) implement this interface. Document-type and query-type embeddings can be distinguished (e.g. for models that support different embedding modes) to improve retrieval quality.

- **Batch Embedding and Fallback**: During indexing, chunks are embedded in batches (e.g. 100 texts per request) to reduce API round-trips. If a provider does not implement `embed_texts_batch` or returns fewer vectors than input texts (e.g. due to rate limits or errors), the controller falls back to per-text embedding via `asyncio.gather`. Only after obtaining a full, aligned list of vectors is data sent to the vector store, avoiding index corruption or validation errors.

- **Vector Database (Qdrant)**: The Qdrant provider implements collection creation with configurable vector dimension and distance metric (cosine or dot product). Vectors, chunk text, and metadata are stored as points with integer IDs. Insertions are batched for efficiency. The provider validates that the number of vectors matches the number of texts before insertion and logs clear errors on mismatch.

- **Project-Scoped Collections**: Each project has a dedicated Qdrant collection (e.g. `collection_{project_id}`). This isolates data per project and allows resetting or rebuilding the index for one project without affecting others. Collection info (e.g. point count) is exposed via an info endpoint for monitoring and debugging.

---

## 3.2.5 Retrieval and Reranking

When a user submits a question, the system retrieves the most relevant chunks from the vector store and optionally reranks them before passing context to the LLM. This two-stage approach improves answer quality by prioritizing the most relevant passages.

- **Semantic and Hybrid Search**: The query string is embedded using the same embedding model and document-type configuration as the indexed chunks. The vector DB client supports **vector-only search** (similarity between query vector and stored vectors) and **hybrid search** (combination of vector similarity and keyword matching where available). Results are returned as a list of retrieved documents (text and similarity score), ordered by relevance.

- **Optional Score Threshold**: Callers can supply a score threshold so that only chunks above a minimum relevance score are included in the RAG context. This reduces noise when the top-k retrieval returns marginally relevant items and helps keep the prompt focused on highly relevant content.

- **Reranking with Cross-Encoder**: After initial retrieval, the system can rerank candidates using a cross-encoder model (e.g. ms-marco-MiniLM-L-6-v2 from the sentence-transformers library). Query–document pairs are scored in batch; documents are reordered by this score and the top-k are selected. The cross-encoder is loaded lazily and reused across requests to avoid repeated model load times and to suppress verbose library logs during startup. Reranking improves precision when the initial vector search returns many candidates of similar score.

- **Stable Pipeline**: If the vector store or hybrid search returns no results, the RAG pipeline exits early and returns an appropriate error or message without calling the LLM, ensuring predictable behavior and avoiding unnecessary API costs.

---

## 3.2.6 LLM Integration and RAG Prompting

Answer generation is implemented in a provider-agnostic way: the same RAG flow works with OpenAI, Gemini, or Cohere by swapping the generation client. Prompt construction is template-based and localizable.

- **Multi-Provider Support**: The application uses a factory to instantiate the configured LLM provider at startup. Each provider implements a common interface: set generation and embedding models, generate text given a prompt and chat history, and construct prompt objects in the format expected by that provider (e.g. message roles and content shape). This allows deployment with different backends (e.g. OpenAI for generation and Gemini for embeddings) without code changes in the controller.

- **Template-Based RAG Prompts**: RAG prompts are built from templates stored under `AI/llm/templates/locales`. A system prompt defines the assistant’s role (e.g. study tutor that answers only from provided documents). A document prompt formats each retrieved chunk with a number and content. A footer prompt states the user question and asks for an answer. The active locale (e.g. English or Arabic) is determined by configuration, supporting multilingual deployments and consistent behavior with the user’s language.

- **Prompt Assembly and Generation**: The NLP controller concatenates the system prompt, formatted document chunks, and footer (with the user query) into a single user-facing prompt. The generation client is called with this prompt and the system role in chat history. The generated answer is returned to the client; optional caching (e.g. by query and project) can be added to avoid redundant LLM calls for repeated questions.

- **Error Handling and Fallbacks**: Failed embedding or generation calls are logged; the controller returns structured error signals (e.g. RAG_ANSWER_ERROR) so clients can display appropriate messages. Embedding fallback (batch to single-text) ensures indexing can complete even when batch APIs are unavailable or misbehave.

---

## 3.2.7 Configuration and Deployment

The application is designed to run in different environments (development, staging, production) with configuration driven by environment variables and optional `.env` files.

- **Centralized Settings**: Pydantic Settings loads configuration such as MongoDB URL and database name, OpenAI/Cohere/Gemini API keys and base URLs, generation and embedding model IDs and dimensions, vector DB backend and path, distance method, primary and default language, and optional Redis URL and cache TTL. Type checking and default values reduce configuration errors. Settings are cached (e.g. via `lru_cache`) so that repeated access does not re-read the environment.

- **Lifecycle Management**: FastAPI startup creates the MongoDB connection, initializes project/chunk/asset models, builds the LLM and vector DB clients from the factory, configures the template parser, and instantiates the NLP controller. Shutdown closes the MongoDB connection and disconnects the vector DB client, ensuring clean resource release.

- **Docker Support**: The repository includes a Dockerfile and docker-compose configuration for running the API and its dependencies (e.g. MongoDB, Qdrant, Redis if used). This supports consistent deployment and local development with minimal setup.

---

## 3.2.8 API Design and Usage

The REST API is organized around two main areas: data management (upload and process) and NLP (index, search, answer). All responses use a consistent signal-based format for success and error cases.

- **Data Endpoints**: **Upload** (`POST /api/v1/data/upload/{project_id}`) accepts a file and returns a file identifier. **Process** (`POST /api/v1/data/process/{project_id}`) accepts chunk size, overlap, reset flag, and optional line-based splitting, runs the selected chunking strategy, and persists chunks to MongoDB. Response signals indicate success or failure (e.g. processing failed, no files).

- **NLP Endpoints**: **Index push** (`POST /api/v1/nlp/index/push/{project_id}`) reads chunks for the project in pages, embeds them, and inserts them into the project’s Qdrant collection; it supports reset on first run. **Index info** (`GET /api/v1/nlp/index/info/{project_id}`) returns collection metadata. **Search** (`POST /api/v1/nlp/index/search/{project_id}`) accepts query text and limit, performs vector (or hybrid) search, and returns ranked chunks with scores. **Answer** (`POST /api/v1/nlp/index/answer/{project_id}`) accepts query text and limit, retrieves and optionally reranks chunks, builds the RAG prompt, calls the LLM, and returns the generated answer along with a success signal.

- **Request and Response Schemes**: Request bodies use Pydantic models (e.g. `ProcessRequest`, `SearchRequest`) so that invalid or missing fields are rejected before reaching the controller. Response bodies include a signal (e.g. `rag_answer_success`, `processing_success`) and relevant data (answer, inserted count, search results), enabling clients to handle outcomes consistently and to display user-friendly messages based on the signal.

---

## Summary

The Contest-RAG backend implements a complete RAG pipeline: document upload and validation, flexible chunking (fixed-size, semantic, and line-based), batch embedding with fallback, project-scoped vector indexing in Qdrant, semantic and hybrid retrieval, optional cross-encoder reranking, and template-based, multi-provider LLM answer generation. The architecture separates routes, controllers, AI providers, and data models, and uses interfaces for vector storage and LLMs to keep the system testable and extensible. Configuration is environment-driven, and the API is designed for clear success/error signaling and straightforward integration with a web or mobile frontend for a graduation project or production deployment.
