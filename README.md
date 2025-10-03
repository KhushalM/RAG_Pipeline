# RAG Pipeline with Mistral AI

A Retrieval-Augmented Generation (RAG) pipeline for PDF documents, featuring hybrid search (Semantic + BM25), semantic chunking, LLM reranking, hallucination detection, and query safety policies.

## Features

- **PDF Ingestion**: Upload and process multiple PDF files with semantic chunking
- **Hybrid Search**: Combines semantic (vector similarity) and lexical (BM25) search
- **Custom Vector Database**: No external vector DB dependencies
- **Query Intelligence**:
  - Intent detection and query routing (Checks if query needs retrieval)
  - Query transformation for better retrieval
  - PII detection with hard refusal
  - Legal/medical disclaimers
- **Answer Quality**:
  - LLM-based reranking
  - Chat memory support
  - Hallucination detection (LLM based)
  - Insufficient evidence handling 
  - Answer shaping (lists, steps, comparisons) (Transforming query to request the intended)
- **Modern UI**: React-based chat interface with source citations

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11 or higher** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18+ and npm** - [Download Node.js](https://nodejs.org/)
- **uv** (Python package manager) - [Install uv](https://github.com/astral-sh/uv)
- **Mistral AI API Key** - [Get API Key](https://docs.mistral.ai/)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG_Pipeline
```

### 2. Backend Setup

#### Step 2.1: Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Step 2.2: Navigate to Backend Directory

```bash
cd backend
```

#### Step 2.3: Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

#### Step 2.4: Configure Environment Variables

Create a `.env` file in the `backend` directory:

```bash
cd backend
cp app/env.example .env
```

Edit the `.env` file and add your Mistral API key:

```env
mistral_api_key="your_mistral_api_key_here"
```
### 3. Frontend Setup

#### Step 3.1: Navigate to Frontend Directory

```bash
cd ../frontend  # From backend directory
# OR
cd frontend     # From root directory
```

#### Step 3.2: Install Dependencies

```bash
npm install
```

## Running the Application

You'll need **two terminal windows** - one for the backend and one for the frontend.

### Terminal 1: Start the Backend Server

```bash
cd backend
source .venv/bin/activate  # Activate virtual environment (macOS/Linux)
# OR
.venv\Scripts\activate     # Windows

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

### Terminal 2: Start the Frontend Server

```bash
cd frontend
npm run dev
```

The frontend UI will be available at `http://localhost:5173`

## Usage

### 1. Upload PDF Files

1. Open your browser to `http://localhost:5173`
2. Click the **"Select PDFs"** button
3. Choose one or more PDF files to upload
4. Wait for the files to be processed and chunked

### 2. Ask Questions

1. Once PDFs are uploaded, type your question in the input field
2. Press **Enter** or click **"Ask"**
3. The system will:
   - Analyze your query intent
   - Transform the query with added context like chat memory
   - Search through the knowledge base (if needed)
   - Generate an answer with source citations
   - Check for hallucinations and point out unverified claims
   - List, define or compare as intended

### Example Queries

- **Factual**: "What are the main findings in the document?"
- **List-based**: "What are the benefits of this approach?" (automatically formatted as a list)
- **Step-by-step**: "How do I implement this process?" (formatted as steps)
- **Conversational**: "Hello, how are you?" (doesn't trigger retrieval)

## Project Structure

```
RAG_Pipeline/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application and endpoints
│   │   ├── mistral.py              # Mistral AI integration (embeddings & LLM)
│   │   ├── ingestion/
│   │   │   ├── pdf_extract.py      # PDF text extraction
│   │   │   └── chunk.py            # Semantic chunking algorithm
│   │   ├── storage/
│   │   │   └── custom_vector_db.py # Hybrid vector database (BM25 + semantic)
│   │   ├── retrieval/
│   │   │   └── rerank.py           # LLM-based reranking
│   │   └── tools/
│   │       ├── query_router.py     # Intent detection & query transformation
│   │       ├── query_refusal.py    # PII detection & legal/medical disclaimers
│   │       └── hallucination_check.py # Answer verification
│   ├── pyproject.toml              # Python dependencies
│   └── pdf_files/                  # Uploaded PDFs storage
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # React application
│   │   ├── main.jsx                # Entry point
│   │   └── styles.css              # Styling
│   └── package.json                # Node dependencies
└── README.md
```

## Libraries Used

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[Mistral AI](https://docs.mistral.ai/)** - Embeddings and LLM
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - PDF text extraction
- **[NumPy](https://numpy.org/)** - Vector operations
- **[Uvicorn](https://www.uvicorn.org/)** - ASGI server

### Frontend
- **[React 18](https://react.dev/)** - UI framework
- **[Vite](https://vite.dev/)** - Build tool
- **[React Markdown](https://github.com/remarkjs/react-markdown)** - Markdown rendering

## Architecture Diagrams

### PDF Upload Pipeline

The following diagram illustrates the complete workflow of the `/pdf_upload` endpoint, from receiving a PDF file to storing it in the vector database:

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI<br/>(main.py)
    participant PDFExtract as PDF Extract<br/>(pdf_extract.py)
    participant SemanticChunker as Semantic Chunker<br/>(chunk.py)
    participant MistralEmbed as Mistral Embeddings<br/>(mistral.py)
    participant VectorDB as Hybrid Vector DB<br/>(custom_vector_db.py)

    Client->>FastAPI: POST /pdf_upload (PDF files)
    
    alt File Already Uploaded
        FastAPI->>FastAPI: Check if filename in uploaded_files set
        FastAPI-->>Client: Return "already uploaded" message
    else File Not Uploaded
        FastAPI->>FastAPI: Save PDF to pdf_files/ directory
        
        FastAPI->>PDFExtract: extract_text_from_pdf(file_path)
        PDFExtract->>PDFExtract: Open PDF with PyMuPDF
        PDFExtract->>PDFExtract: Extract text from all pages
        PDFExtract-->>FastAPI: Return concatenated text
        
        FastAPI->>SemanticChunker: chunking(text)
        
        SemanticChunker->>SemanticChunker: sentence_chunks()<br/>Split text into sentences using regex
        Note over SemanticChunker: Regex: (?<=[.!?])\s+(?=[A-Z])
        
        SemanticChunker->>MistralEmbed: embed_documents(sentences)
        MistralEmbed->>MistralEmbed: Batch sentences (32 per batch)
        MistralEmbed->>MistralEmbed: Call Mistral API with<br/>model='mistral-embed'
        MistralEmbed-->>SemanticChunker: Return sentence embeddings
        
        SemanticChunker->>SemanticChunker: Calculate cosine similarity<br/>between consecutive sentences
        Note over SemanticChunker: similarity = dot(v1,v2) / (||v1|| * ||v2||)
        
        SemanticChunker->>SemanticChunker: Glue sentences together<br/>if similarity > 0.7 AND<br/>combined_size <= max_chunk_size
        
        SemanticChunker->>SemanticChunker: filter_chunks()<br/>Merge small chunks<br/>Split large chunks
        Note over SemanticChunker: min_chunk_size=100<br/>max_chunk_size=2000
        
        SemanticChunker-->>FastAPI: Return final chunks
        
        FastAPI->>MistralEmbed: embed_documents(chunks)
        MistralEmbed->>MistralEmbed: Batch chunks (32 per batch)
        MistralEmbed->>MistralEmbed: Call Mistral API with<br/>model='mistral-embed'
        MistralEmbed-->>FastAPI: Return chunk embeddings (vectors)
        
        FastAPI->>FastAPI: Create metadata for each chunk<br/>{filename, chunk_index,<br/>chunk_size, file_path}
        
        FastAPI->>FastAPI: Generate UUID for each chunk
        
        FastAPI->>VectorDB: add(vectors, chunks, metadatas, ids)
        
        VectorDB->>VectorDB: Normalize vectors<br/>(v / ||v||)
        
        VectorDB->>VectorDB: Extend internal storage<br/>(vectors, texts, metadata, ids)
        
        VectorDB->>VectorDB: _build_bm25_index()
        Note over VectorDB: Tokenize all texts<br/>Build word frequency map<br/>Calculate doc lengths & avg
        
        VectorDB-->>FastAPI: Storage complete
        
        FastAPI->>FastAPI: Mark filename as uploaded
        
        FastAPI-->>Client: Return success response<br/>{status, per_file_chunks,<br/>number_of_chunks, chunk_info}
    end
```

### Query Processing Pipeline

The following diagram shows the complete RAG query processing flow with safety checks, intelligent routing, and hallucination detection:

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI<br/>(main.py)
    participant QueryRefusal as Query Refusal<br/>(query_refusal.py)
    participant QueryRouter as Query Router<br/>(query_router.py)
    participant MistralLLM as Mistral LLM<br/>(mistral.py)
    participant MistralEmbed as Mistral Embeddings<br/>(mistral.py)
    participant VectorDB as Hybrid Vector DB<br/>(custom_vector_db.py)
    participant Reranker as LLM Reranker<br/>(rerank.py)
    participant HallucinationCheck as Hallucination Check<br/>(hallucination_check.py)

    Client->>FastAPI: POST /query_processing<br/>{query, session_id, retrieval_mode}
    
    FastAPI->>FastAPI: Initialize chat_memories[session_id]<br/>if not exists
    
    alt No Vector DB
        FastAPI->>FastAPI: Check if vector_db.vectors empty
        FastAPI-->>Client: Error: "No VectorDB initialized,<br/>please upload at least one PDF"
    else Vector DB Exists
        FastAPI->>QueryRefusal: should_refuse_query(query)
        QueryRefusal->>QueryRefusal: Check for PII patterns<br/>(SSN, credit cards, emails, etc.)
        QueryRefusal->>QueryRefusal: Check for legal/medical keywords
        
        alt PII Detected
            QueryRefusal-->>FastAPI: action="REFUSE"
            FastAPI-->>Client: Refuse with message
        else Legal/Medical Query
            QueryRefusal-->>FastAPI: action="DISCLAIMER", message
            FastAPI->>FastAPI: Store disclaimer for later
        else Safe Query
            QueryRefusal-->>FastAPI: action="ALLOW"
        end
        
        FastAPI->>QueryRouter: analyze_and_transform(query, history)
        QueryRouter->>QueryRouter: Detect if KB retrieval needed<br/>(factual vs conversational)
        QueryRouter->>QueryRouter: Transform query with context<br/>& output format hints
        QueryRouter-->>FastAPI: (need_retrieval, transformed_query)
        
        alt No Retrieval Needed
            FastAPI->>MistralLLM: generate_response(query)
            Note over MistralLLM: Model: mistral-small-2503
            MistralLLM-->>FastAPI: Direct answer
            FastAPI->>FastAPI: Prepend disclaimer if needed
            FastAPI->>FastAPI: Add to chat_memories[session_id]
            FastAPI-->>Client: {answer, sources=[], processing_time}
            
        else Retrieval Required
            FastAPI->>FastAPI: Use transformed_query for retrieval
            
            FastAPI->>MistralEmbed: embed_query(transformed_query)
            MistralEmbed->>MistralEmbed: Call Mistral API<br/>model='mistral-embed'
            MistralEmbed-->>FastAPI: query_vector
            
            alt Hybrid Search (65% semantic + 35% lexical)
                FastAPI->>VectorDB: hybrid_search(query, query_vector)
                VectorDB->>VectorDB: semantic_search(query_vector)<br/>Cosine similarity with normalized vectors
                VectorDB->>VectorDB: lexical_search(query)<br/>BM25 scoring
                VectorDB->>VectorDB: Combine scores with weights<br/>Filter by threshold (0.5)
                Note over VectorDB: combined_score = 0.65*semantic + 0.35*lexical<br/>Return top-k if >= 0.5 threshold
            else Semantic Only
                FastAPI->>VectorDB: semantic_search(query_vector)
            else Lexical Only  
                FastAPI->>VectorDB: lexical_search(query)
            end
            
            alt No Results or Below Threshold
                VectorDB-->>FastAPI: Empty results
                FastAPI->>FastAPI: Prepend disclaimer if needed
                FastAPI-->>Client: "Insufficient evidence:<br/>Not enough reliable information<br/>to answer confidently"
                
            else Valid Results
                VectorDB-->>FastAPI: Retrieved documents
                
                FastAPI->>Reranker: rerank(query, results)
                Reranker->>Reranker: Use LLM to score relevance<br/>of each document to query
                Reranker-->>FastAPI: Reranked top documents
                
                FastAPI->>MistralLLM: create_rag_prompt(query, reranked_results)
                FastAPI->>MistralLLM: generate_response(prompt, temp, max_tokens)
                Note over MistralLLM: Model: mistral-large-latest<br/>Context: Retrieved documents + query
                MistralLLM-->>FastAPI: Generated answer
                
                FastAPI->>HallucinationCheck: check_hallucination(query, answer, sources)
                HallucinationCheck->>HallucinationCheck: Extract claims from answer
                HallucinationCheck->>HallucinationCheck: Verify each claim against sources
                HallucinationCheck-->>FastAPI: (unverified_claims, report)
                
                FastAPI->>FastAPI: Prepend disclaimer if needed
                FastAPI->>FastAPI: Add to chat_memories[session_id]
                FastAPI-->>Client: {answer, sources, unverified_claims,<br/>processing_time}
            end
        end
    end
```

## Key Design Decisions

### Semantic Chunking
- Splits text into sentences, measures cosine similarity between consecutive sentences
- Merges similar sentences (threshold > 0.7) into coherent chunks
- Maintains chunk size: 100-2000 characters for optimal retrieval

### Hybrid Search
- **65% semantic** (vector similarity) + **35% lexical** (BM25)
- Minimum similarity threshold: 0.5 to ensure quality results
