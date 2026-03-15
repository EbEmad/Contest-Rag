# Contest-Rag

A robust Retrieval-Augmented Generation (RAG) system built with FastAPI, MongoDB, and Qdrant. This application allows users to upload documents, automatically chunk and index them, and perform context-aware Q&A using advanced LLMs.

## System Architecture

This platform transforms a basic RAG system into a complete AI-powered educational ecosystem serving **students**, **teachers**, and **parents**.

### High-Level Architecture
### simple Rag
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

### We need in our system

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

### 🛡️ Detailed Roadmap Generation (Automated)

This diagram shows how the system automatically fetches student and performance data to inform the AI, without requiring manual input of topic names.

```mermaid
sequenceDiagram
    autonumber
    participant API as performance.py (Route)
    participant CTRL as PerformanceController.py (Controller)
    participant SM as StudentModel.py (Model)
    participant PM as PerformanceModel.py (Model)
    participant CM as CurriculumModel.py (Model)
    participant AI as LLM Client

    Note over API, AI: Automated Roadmap Generation Flow

    API->>CTRL: generate_roadmap(student_id)
    
    rect rgb(240, 240, 240)
        Note right of CTRL: Step 1: Identify Student
        CTRL->>SM: get_student_by_id(student_id)
        SM-->>CTRL: {full_name, grade, current_level}
    end

    rect rgb(240, 240, 240)
        Note right of CTRL: Step 2: Identify Weakness
        CTRL->>PM: get_weak_topics(student_id)
        PM-->>CTRL: List[TopicPerformance] (Topic IDs)
    end

    rect rgb(240, 240, 240)
        Note right of CTRL: Step 3: Translate to Names
        loop for each Weak Topic ID
            CTRL->>CM: get_topic(topic_id)
            CM-->>CTRL: Topic Name (e.g. "Newton Laws")
        end
    end

    rect rgb(240, 240, 240)
        Note right of CTRL: Step 4: Generate LLM study plan
        CTRL->>AI: generate_text(Prompt with Names & Grade)
        AI-->>CTRL: "Your 2-week plan for Physics..."
    end

    rect rgb(240, 240, 240)
        Note right of CTRL: Step 5: Save & Return
        CTRL->>PM: save_roadmap(LearningRoadmap)
        CTRL-->>API: JSON Response (roadmap_id, llm_explanation)
    end
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

