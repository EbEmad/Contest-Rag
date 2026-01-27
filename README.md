# Contest-Rag

A robust Retrieval-Augmented Generation (RAG) system built with FastAPI, MongoDB, and Qdrant. This application allows users to upload documents, automatically chunk and index them, and perform context-aware Q&A using advanced LLMs.

## Architecture

The system follows a modern RAG pipeline, separating data ingestion from the retrieval-augmented generation flow.
```mermaid
graph LR
    A[Doc Upload] -->|Processing & Chunking| B[(MongoDB<br>Raw Data & Chunks)]
    B -->|Embedding Model| C[(Qdrant<br>Vector Store)]
    
    subgraph "RAG Flow"
        User[User Query] -->|API Request| App[FastAPI]
        App -->|Vector Search| C
        C -->|Retrieved Context| App
        App -->|"Context + Prompt"| LLM["LLM<br>(OpenAI / Gemini)"]
    end

    App -->|Generated Answer| User

    classDef db fill:#249edc,stroke:#fff,stroke-width:2px,color:#fff
    class B,C db
```

## How It Works

Here is a short and simple explanation of each part of the project:

### 1. The Web Server (FastAPI)
Think of this as the front desk. It receives your requests (like "upload this file" or "answer this question") and directs them to the right department.

### 2. The Database (MongoDB)
This is the filing cabinet. It stores your uploaded files and keeps track of all the small text pieces (chunks) we make from them.

### 3. The Vector Store (Qdrant)
This is the smart index. It doesn't just store words; it stores the *meaning* of the text as numbers (vectors). This allows the system to find relevant information even if the exact keywords don't match.

### 4. The Brains (LLM)
This is the intelligent part (like OpenAI or Gemini). It reads the relevant information found by Qdrant and writes a clear answer to your question.

### 5. The Workflow
1.  **Ingestion**: You upload a PDF. We chop it into small pieces (chunks) and save them to MongoDB.
2.  **Indexing**: We turn those chunks into "vectors" (meaning-numbers) and save them in Qdrant.
3.  **Search**: You ask a question. We turn your question into a vector and find the most similar chunks in Qdrant.
4.  **Answer**: We give those chunks to the LLM and say "Answer this question using these notes."

## Tech Stack

-   **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) - High-performance async web framework.
-   **Database**:
    -   [MongoDB](https://www.mongodb.com/) (via [Motor](https://motor.readthedocs.io/)) - Stores raw document assets and metadata.
    -   [Qdrant](https://qdrant.tech/) - Vector database for efficient semantic search.
-   **AI & LLM**:
    -   [LangChain](https://www.langchain.com/) - Orchestration framework.
    -   **LLMs**: Supports OpenAI, Gemini, and Cohere.
-   **Processing**:
    -   [PyMuPDF](https://pymupdf.readthedocs.io/) - efficient PDF processing.

## Key Features

-   **Document Ingestion**: Upload PDF documents via REST API.
-   **Automatic Indexing**: Documents are automatically processed, chunked, and embedded into the vector store.
-   **Context-Aware QA**: Ask questions about your documents and receive answers grounded in the uploaded content.
-   **Hybrid Storage**: Combines MongoDB for document management with Qdrant for vector search.

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd Contest-Rag
    ```

2.  **Set up environment variables**:
    Copy `.env.example` to `.env` and configure your API keys (OpenAI/Gemini) and database credentials.

3.  **Run with Docker**:
    ```bash
    docker-compose up -d
    ```

4.  **Access the API**:
    Navigate to `http://localhost:8000/docs` to view the interactive API documentation.

## API Usage

Here are the essential `curl` commands to interact with the API.

### 1. Check Health
```bash
curl -X GET "http://localhost:8000/api/v1/"
```

### 2. Upload Document
Upload a PDF file to the system.
```bash
curl -X POST "http://localhost:8000/api/v1/data/upload/my_project" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```

### 3. Process Document
Chunk the uploaded document.
```bash
curl -X POST "http://localhost:8000/api/v1/data/process/my_project" \
     -H "Content-Type: application/json" \
     -d '{
           "chunk_size": 100,
           "overlap_size": 20,
           "do_reset": 0
         }'
```

### 4. Index Data
Push processed chunks to Qdrant vector store.
```bash
curl -X POST "http://localhost:8000/api/v1/nlp/index/push/my_project" \
     -H "Content-Type: application/json" \
     -d '{"do_reset": 0}'
```

### 5. Search Index
Semantic search for relevant context.
```bash
curl -X POST "http://localhost:8000/api/v1/nlp/index/search/my_project" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "What is the summary of the document?",
           "limit": 5
         }'
```

### 6. RAG Answer
Ask a question and get an AI-generated answer.
```bash
curl -X POST "http://localhost:8000/api/v1/nlp/index/answer/my_project" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Explain the key findings.",
           "limit": 5
         }'
```
