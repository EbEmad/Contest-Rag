# Contest-Rag

A robust Retrieval-Augmented Generation (RAG) system built with FastAPI, MongoDB, and Qdrant. This application allows users to upload documents, automatically chunk and index them, and perform context-aware Q&A using advanced LLMs.

## System Architecture

This platform transforms a basic RAG system into a complete AI-powered educational ecosystem serving **students**, **teachers**, and **parents**.

### High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        StudentUI[Student Dashboard]
        TeacherUI[Teacher Dashboard]
        ParentUI[Parent Dashboard]
    end
    
    subgraph "API Gateway - FastAPI"
        AuthAPI[Authentication]
        StudentAPI[Student Routes]
        TeacherAPI[Teacher Routes]
        ParentAPI[Parent Routes]
        CurriculumAPI[Curriculum Routes]
    end
    
    subgraph "Controller Layer"
        StudentCtrl[Student Controller]
        TeacherCtrl[Teacher Controller]
        ParentCtrl[Parent Controller]
        AnalyticsCtrl[Analytics Controller]
        NLPCtrl["RAG Controller (Existing)"]
    end
    
    subgraph "AI Agent Layer"
        AdaptiveAgent["🤖 Adaptive Engine<br/>(Difficulty & Roadmap)"]
        QuizAgent["🤖 Quiz Generator<br/>(Question Creation)"]
        ReportAgent["🤖 Report Generator<br/>(Teacher/Parent Reports)"]
    end
    
    subgraph "Core RAG Engine - Your Existing System"
        RAG["RAG Pipeline<br/>(NLPController)"]
        Embed[Embedding Model]
        Rerank[Cross-Encoder Reranking]
    end
    
    subgraph "Data Storage"
        MongoDB[(MongoDB)]
        Qdrant[(Qdrant<br/>Vector Store)]
        Redis[(Redis Cache)]
    end
    
    subgraph "LLM Providers"
        OpenAI[OpenAI GPT-4]
        Gemini[Google Gemini]
        Cohere[Cohere]
    end
    
    subgraph "Background Tasks"
        Celery[Celery Workers]
        Analytics[Analytics Tasks]
        Ranking[Ranking Updates]
    end
    
    StudentUI --> AuthAPI --> StudentAPI --> StudentCtrl
    TeacherUI --> AuthAPI --> TeacherAPI --> TeacherCtrl
    ParentUI --> AuthAPI --> ParentAPI --> ParentCtrl
    
    StudentCtrl --> AdaptiveAgent
    StudentCtrl --> QuizAgent
    StudentCtrl --> NLPCtrl
    
    TeacherCtrl --> ReportAgent
    TeacherCtrl --> AnalyticsCtrl
    
    ParentCtrl --> ReportAgent
    ParentCtrl --> AnalyticsCtrl
    
    AdaptiveAgent --> RAG
    QuizAgent --> RAG
    ReportAgent --> AnalyticsCtrl
    
    RAG --> Embed --> Qdrant
    RAG --> Rerank
    
    AdaptiveAgent --> OpenAI
    QuizAgent --> Gemini
    ReportAgent --> OpenAI
    RAG --> OpenAI
    RAG --> Gemini
    RAG --> Cohere
    
    StudentCtrl --> MongoDB
    TeacherCtrl --> MongoDB
    AnalyticsCtrl --> MongoDB
    NLPCtrl --> MongoDB
    
    AnalyticsCtrl --> Redis
    
    Celery --> Analytics
    Celery --> Ranking
    Analytics --> MongoDB
    
    classDef agent fill:#9C27B0,stroke:#fff,stroke-width:2px,color:#fff
    classDef db fill:#249edc,stroke:#fff,stroke-width:2px,color:#fff
    classDef rag fill:#FF9800,stroke:#fff,stroke-width:2px,color:#fff
    classDef llm fill:#4CAF50,stroke:#fff,stroke-width:2px,color:#fff
    
    class AdaptiveAgent,QuizAgent,ReportAgent agent
    class MongoDB,Qdrant,Redis db
    class RAG,Embed,Rerank,NLPCtrl rag
    class OpenAI,Gemini,Cohere llm
```

### 🔍 Role of Your Existing RAG System

Your current RAG pipeline (`NLPController`) is the **foundation** - it doesn't get replaced, it gets **enhanced and utilized by AI agents**:

| Component | Role in New System |
|-----------|-------------------|
| **Document Upload & Chunking** | Still used - teachers upload curriculum books |
| **Vector Store (Qdrant)** | Core semantic search engine for all features |
| **RAG Q&A** | Powers student explanations + used by quiz generator |
| **Cross-Encoder Reranking** | Ensures high-quality context for all AI operations |
| **LLM Integration** | Shared by RAG answers + AI agents |

**Key Enhancement:** RAG now includes **curriculum hierarchy filtering** (Grade → Subject → Chapter → Topic) instead of flat project-level search.

---

## User Flow Diagrams

### 👨‍🎓 Student Learning Flow

```mermaid
sequenceDiagram
    actor Student
    participant API as Student API
    participant Ctrl as Student Controller
    participant Adaptive as Adaptive Engine
    participant Quiz as Quiz Generator
    participant RAG as RAG Engine
    participant VDB as Qdrant Vector DB
    participant LLM as LLM Provider
    participant DB as MongoDB
    
    Note over Student,DB: 1️⃣ Student Requests Explanation
    Student->>API: "Explain Linear Equations"
    API->>Ctrl: Get explanation (topic_id)
    Ctrl->>RAG: Search by topic + query
    RAG->>VDB: Vector search with filters<br/>(Grade=10, Subject=Math, Topic=Linear Eq)
    VDB-->>RAG: Top 5 relevant chunks
    RAG->>RAG: Rerank with cross-encoder
    RAG->>LLM: Prompt + context chunks
    LLM-->>RAG: Natural language answer
    RAG-->>Student: Explanation based on curriculum
    
    Note over Student,DB: 2️⃣ Student Requests Quiz
    Student->>API: "Give me a quiz on Linear Equations"
    API->>Ctrl: Generate quiz (topic_id)
    Ctrl->>Adaptive: Determine difficulty for student
    Adaptive->>DB: Get student's past performance
    DB-->>Adaptive: Last 5 quiz scores, trends
    Adaptive->>LLM: "Analyze performance, suggest difficulty"
    LLM-->>Adaptive: "MEDIUM difficulty"
    Adaptive-->>Ctrl: MEDIUM
    
    Ctrl->>Quiz: Generate 5 MEDIUM questions
    Quiz->>RAG: Get curriculum content for topic
    RAG->>VDB: Retrieve chunks
    VDB-->>Quiz: Content chunks
    Quiz->>LLM: "Create 5 MCQ questions from content"
    LLM-->>Quiz: Generated questions JSON
    Quiz->>DB: Save questions with source_chunks
    Quiz-->>Student: Quiz with 5 questions
    
    Note over Student,DB: 3️⃣ Student Submits Quiz
    Student->>API: Submit answers
    API->>Ctrl: Process submission
    Ctrl->>DB: Calculate score, update performance
    Ctrl->>Celery: Trigger analytics update (async)
    Ctrl-->>Student: Results + score
    
    Note over Student,DB: 4️⃣ Student Requests Roadmap
    Student->>API: "Show my learning plan"
    API->>Ctrl: Get roadmap
    Ctrl->>Adaptive: Generate personalized roadmap
    Adaptive->>DB: Analyze all topic performances
    Adaptive->>LLM: "Create 2-week study plan for weak topics"
    LLM-->>Adaptive: Detailed roadmap markdown
    Adaptive->>DB: Save roadmap
    Adaptive-->>Student: Personalized learning plan
```

### 👩‍🏫 Teacher Analytics Flow

```mermaid
sequenceDiagram
    actor Teacher
    participant API as Teacher API
    participant Ctrl as Teacher Controller
    participant Analytics as Analytics Controller
    participant Report as Report Agent
    participant LLM as LLM Provider
    participant DB as MongoDB
    
    Note over Teacher,DB: Teacher Views Class Analytics
    Teacher->>API: "Show Class Dashboard"
    API->>Ctrl: Get class analytics (class_id)
    Ctrl->>Analytics: Aggregate class performance
    Analytics->>DB: Query all students in class
    DB-->>Analytics: Student performance data
    Analytics->>Analytics: Calculate:<br/>- Avg score per topic<br/>- Weak topic distribution<br/>- Student rankings
    Analytics-->>Teacher: Class analytics dashboard
    
    Note over Teacher,DB: Teacher Requests AI Report
    Teacher->>API: "Generate class report"
    API->>Ctrl: Generate report (class_id)
    Ctrl->>Report: Create teacher report
    Report->>DB: Get class data
    DB-->>Report: Performance, trends, weak topics
    Report->>LLM: "Analyze class performance, suggest interventions"
    LLM-->>Report: Natural language insights
    Report->>DB: Save report
    Report-->>Teacher: AI-generated insights & recommendations
```

### 👨‍👩‍👦 Parent Progress Flow

```mermaid
sequenceDiagram
    actor Parent
    participant API as Parent API
    participant Ctrl as Parent Controller
    participant Analytics as Analytics Controller
    participant Report as Report Agent
    participant LLM as LLM Provider
    participant DB as MongoDB
    
    Parent->>API: "Show my child's progress"
    API->>Ctrl: Get child progress
    Ctrl->>Analytics: Get student summary
    Analytics->>DB: Query student performance
    DB-->>Analytics: Quiz results, weak topics, trends
    Analytics-->>Parent: Performance dashboard
    
    Parent->>API: "How can I help?"
    API->>Ctrl: Generate parent report
    Ctrl->>Report: Create parent suggestions
    Report->>DB: Get child's data
    Report->>LLM: "Explain progress to parent, suggest how to help"
    LLM-->>Report: Simple, actionable advice
    Report-->>Parent: Parent-friendly report + tips
```

---

## Data Flow: How RAG Powers Everything

```mermaid
graph LR
    subgraph "Content Ingestion (Teachers)"
        T1[Teacher Uploads<br/>Grade 10 Math Book]
        T2[System Processes PDF]
        T3[Chunks Tagged with<br/>Grade/Subject/Chapter/Topic]
        T4[Embedded to Vectors]
    end
    
    subgraph "Existing RAG Pipeline"
        R1[(MongoDB<br/>Chunks + Metadata)]
        R2[(Qdrant<br/>Vector Store)]
        R3[Semantic Search<br/>with Filters]
    end
    
    subgraph "AI Agents Consume RAG"
        A1["Quiz Generator:<br/>Retrieve content → Generate questions"]
        A2["Adaptive Engine:<br/>Context-aware difficulty decisions"]
        A3["Student Explanations:<br/>Direct RAG Q&A"]
    end
    
    T1 --> T2 --> T3 --> T4
    T3 --> R1
    T4 --> R2
    
    R1 --> R3
    R2 --> R3
    
    R3 --> A1
    R3 --> A2
    R3 --> A3
    
    A1 --> S1[Student Gets Quiz]
    A2 --> S2[Next Quiz Adapts]
    A3 --> S3[Student Gets Answer]
    
    style R3 fill:#FF9800,stroke:#fff,stroke-width:3px,color:#fff
```

---

## Key Enhancements to Existing RAG

### Before (Current System)
```python
# Simple project-level search
search_result = nlp_controller.search_vector_db_collection(
    project="my_project",
    text="What is photosynthesis?",
    limit=5
)
```

### After (Enhanced with Curriculum)
```python
# Hierarchical filtered search
search_result = nlp_controller.search_by_curriculum(
    grade=10,
    subject="Biology",
    chapter="Plant Biology",
    topic="Photosynthesis",
    text="What is photosynthesis?",
    limit=5
)
# Returns ONLY chunks from Grade 10 Biology > Plant Biology > Photosynthesis
```

**Impact:** More accurate, relevant results for educational content.


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

## AI Roles in the System

This platform leverages AI to empower all stakeholders in the educational ecosystem:

### 1. For Students: The Intelligent Learning Companion
*   **Context-Aware Explanations**: Provides detailed explanations based specifically on the materials provided by the teacher, ensuring accuracy and relevance.
*   **Smart Question Banks**: Automatically generates quizzes and practice questions indexed by historical exam patterns to help students prepare effectively.
*   **Performance Analytics**: Measures the student's level and tracks contest performance to identify specific gaps.
*   **Personalized Future Plans**: Generates a tailored roadmap to address weak points and improve overall learning outcomes.

### 2. For Teachers & Schools: Proactive Educational Management
*   **Student Progress Monitoring**: Offers high-level insights into student performance trends across different subjects.
*   **Problem Identification**: Pinpoints specific areas where multiple students are struggling, suggesting potential curriculum adjustments or focused review sessions.
*   **Curriculum Alignment**: Ensures that the AI-generated content remains strictly within the bounds of the provided educational resources.

### 3. For Parents: Transparent Progress Tracking
*   **Automated Reporting**: Generates periodic reports detailing the child's strengths, weaknesses, and improvement over time.
*   **Guided Support**: Helps parents understand exactly what their child needs to work on, making home-based support more effective.

---

## Why this Platform? (vs. ChatGPT/NotebookLM)

While general tools like ChatGPT or specialized ones like NotebookLM are powerful, this platform offers a tailored ecosystem for institutional learning:

| Feature | Contest-Rag Platform | General LLMs (ChatGPT) | NotebookLM |
| :--- | :--- | :--- | :--- |
| **Data Privacy** | Full control over source documents | Data may be used for training | Limited to Google ecosystem |
| **Contest Integration** | Built-in contest platform & ranking | None | None |
| **Stakeholder Reports** | Custom reports for parents/teachers | None | Personal use only |
| **Structured Analytics** | In-depth weak point analysis | Chat-based only | Document-based only |
| **Learning Roadmap** | AI-generated plans based on history | Generic advice | None |

This platform is not just a chatbot; it's a comprehensive **Educational Management System** powered by RAG technology.

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
