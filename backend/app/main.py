from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
import os
import uuid
import logging
import time

from .ingestion.chunk import SemanticChunks
from .ingestion.pdf_extract import extract_text_from_pdf
from .storage.custom_vector_db import HybridVectorDB
from .mistral import MistralEmbeddings, MistralLLM
from .retrieval.rerank import LLMReranker
from .tools.retrieval_need import RetrievalNeed
from .tools.query_transformation import QueryTransformation
from .tools.hallucination_check import HallucinationCheck
from .tools.answer_shaping import AnswerShaping
from .tools.query_refusal import QueryRefusal

app = FastAPI(
    title="RAG Pipeline API",
    description="RAG Pipeline for PDF and Text files",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRequest(BaseModel):
    query: str
    max_context_chunks: Optional[int] = 3
    retrieval_mode: Optional[str] = "hybrid"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    hallucination_report: Optional[Dict[str, Any]] = None
    evidence_status: Optional[str] = "Insufficient"

# In-memory storage (replace with actual database in production)
documents_store = {}
uploaded_files = set()  # Track uploaded filenames to prevent duplicates
vector_db = HybridVectorDB()
embedder = MistralEmbeddings()
mistral = MistralLLM()

#Defining all tools (feature-specific not necessarily LLM tools)
query_refusal = QueryRefusal()
retrieval_need = RetrievalNeed()
query_transformation = QueryTransformation()
reranker = LLMReranker()
hallucination_check = HallucinationCheck()
answer_shaping = AnswerShaping()

import os
import dotenv
dotenv.load_dotenv()
mistral_api_key = os.getenv('mistral_api_key')
logger.info(f"mistral_api_key: {mistral_api_key}")


@app.post("/pdf_upload")
async def pdf_upload(pdf_files: list[UploadFile] = File(...)):

    """Upload and process a document for RAG"""
    if not pdf_files or any(f.filename is None or f.filename == "" for f in pdf_files):
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save pdf files and extract text for each
    os.makedirs("pdf_files", exist_ok=True)
    per_file_chunks = {}
    all_chunks = []
    all_vectors = []
    all_metadatas = []
    all_ids = []

    chunker = SemanticChunks()
    #Step 1: Extract text from pdf files
    for upload in pdf_files:
        # Skip if file already uploaded
        if upload.filename in uploaded_files:
            logger.info(f"Skipping {upload.filename} - already uploaded")
            return {
                "status": "All files already uploaded",
                "per_file_chunks": {},
                "skipped_files": [upload.filename],
                "number_of_chunks": 0,
                "message": f"Skipped {upload.filename} - already uploaded"
            }
            continue
            
        file_path = f"pdf_files/{upload.filename}"
        with open(file_path, "wb") as f:
            f.write(await upload.read())

        text = extract_text_from_pdf(file_path)
        chunks = chunker.chunking(text)

        if not chunks:
            continue
    
        #Step 2: Embeddings
        embeddings = embedder.embed_documents(chunks)
        vectors = [np.array(emb.embedding) for emb in embeddings]  # type: ignore

        #Step 3: Metadata
        metadatas = []
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            metadatas.append({
                "filename": upload.filename,
                "chunk_index": i,
                "chunk_size": len(chunk),
                "file_path": file_path
            })
            ids.append(chunk_id)

        #Step 4: Accumulate and mark as uploaded
        per_file_chunks[upload.filename] = len(chunks)
        uploaded_files.add(upload.filename)
        all_chunks.extend(chunks)
        all_vectors.extend(vectors)
        all_metadatas.extend(metadatas)
        all_ids.extend(ids)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from uploaded PDFs")
    #Step 5: Add to vector database
    vector_db.add(all_vectors, all_chunks, all_metadatas, all_ids)
    
    return {
        "status": "PDF(s) uploaded and processed successfully",
        "per_file_chunks": per_file_chunks,
        "number_of_chunks": len(all_chunks),
        "chunk_info": chunker.get_chunk_info(all_chunks)
    }

@app.post("/query_processing")
async def query_processing(request: RAGRequest):
    """Query processing endpoint"""
    start_time = time.time()


    if not vector_db.vectors:
        logger.error("No VectorDB initialized")
        raise HTTPException(status_code=400, detail="No VectorDB initialized")
    
    #Step 0: Check if query should be refused (PII) or needs disclaimer (Legal/Medical)
    action, message = query_refusal.should_refuse_query(request.query)
    logger.info(f"Query refusal action: {action}")
    logger.info(f"Query refusal message: {message}")
    
    if action == "REFUSE":
        logger.warning(f"Query refused: {message}")
        return {
            "query": request.query,
            "answer": message,
            "sources": [],
            "processing_time": time.time() - start_time
        }
    
    # Store disclaimer if needed, will be prepended to final answer
    disclaimer = message if action == "DISCLAIMER" else None
    
    #Step 1: Determine if the query requires retrieval
    need_to_retrieve = retrieval_need.need_to_retrieve(request.query)
    if not need_to_retrieve:
        results = mistral.generate_response(request.query)
        final_answer = f"{disclaimer}\n\n{results}" if disclaimer else results
        return {
            "query": request.query,
            "answer": final_answer,
            "sources": ["No retrieval required"],
            "processing_time": time.time() - start_time
        }
    
    #Step 2: Transform query
    request.query = query_transformation.transform_query(request.query)
    logger.info(f"Transformed query: {request.query}")
        
    #Step 3: Embed query
    query_vector = embedder.embed_query(request.query)
    query_vector = np.array(query_vector.embedding)

    #Step 4: Retrieve results based on retrieval mode
    if request.retrieval_mode == "hybrid":
        results = vector_db.hybrid_search(request.query, query_vector)
    elif request.retrieval_mode == "semantic":
        results = vector_db.semantic_search(query_vector)
    elif request.retrieval_mode == "lexical":
        results = vector_db.lexical_search(request.query)
    else:
        raise HTTPException(status_code=400, detail="Invalid retrieval mode")

    if not results:
        logger.info("Insufficient evidence to answer the query")
        insufficient_msg = "Insufficient evidence: I don't have enough reliable information in my knowledge base to answer this question confidently."
        final_answer = f"{disclaimer}\n\n{insufficient_msg}" if disclaimer else insufficient_msg
        return {
            "query": request.query,
            "answer": final_answer,
            "sources": [],
            "processing_time": time.time() - start_time
        }

    #Step 5: Reranking
    #Apply reranking steps here
    reranked_results = reranker.rerank(request.query, results)

    #Step 6: Generate answer
    generate_prompt = mistral.create_rag_prompt(request.query, reranked_results)
    if request.temperature and request.max_tokens:
        answer = mistral.generate_response(generate_prompt, request.temperature, request.max_tokens)
    else:
        answer = mistral.generate_response(generate_prompt)
    
    #Step 7: Check for hallucination
    #unverified_answer_list, hallucination_report = hallucination_check.check_hallucination(request.query, answer, reranked_results)
    #logger.info(f"Unverified answer list: {unverified_answer_list}")
    
    #Step 8: Shaping the final answer based on the intent
    shaped_answer = answer_shaping.shape_answer(request.query, answer)
    
    #Step 9: Prepend disclaimer if needed
    final_answer = f"{disclaimer}\n\n{shaped_answer}" if disclaimer else shaped_answer

    return {
        "query": request.query,
        "answer": final_answer,
        "sources": reranked_results,
        "processing_time": time.time() - start_time
    }
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents_count": len(vector_db.vectors),
        "db_stats": vector_db.get_stats()
    }

@app.post("/reset")
async def reset_database():
    """Reset the vector database and uploaded files tracking"""
    global vector_db, uploaded_files
    vector_db = HybridVectorDB()
    uploaded_files.clear()
    return {"status": "Database reset successfully"}


