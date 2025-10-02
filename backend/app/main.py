from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import pymupdf
import numpy as np
import os
import uuid
import logging

from .ingestion.chunk import SemanticChunks
from .ingestion.pdf_extract import extract_text_from_pdf
from .storage.custom_vector_db import HybridVectorDB
from .mistral import MistralEmbeddings

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
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    use_reranking: Optional[bool] = True
    use_llm_reranking: Optional[bool] = False

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float

# In-memory storage (replace with actual database in production)
documents_store = {}
vector_db = HybridVectorDB()


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
    embedder = MistralEmbeddings()

    for upload in pdf_files:
        file_path = f"pdf_files/{upload.filename}"
        with open(file_path, "wb") as f:
            f.write(await upload.read())

        text = extract_text_from_pdf(file_path)
        chunks = chunker.chunking(text)

        if not chunks:
            continue
    
        # Embeddings
        embeddings = embedder.embed_documents(chunks)
        vectors = [np.array(emb.embedding) for emb in embeddings]  # type: ignore

        # Metadata
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

        # Accumulate
        per_file_chunks[upload.filename] = len(chunks)
        all_chunks.extend(chunks)
        all_vectors.extend(vectors)
        all_metadatas.extend(metadatas)
        all_ids.extend(ids)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from uploaded PDFs")

    vector_db.add(all_vectors, all_chunks, all_metadatas, all_ids)
    
    return {
        "status": "PDF(s) uploaded and processed successfully",
        "per_file_chunks": per_file_chunks,
        "number_of_chunks": len(all_chunks),
        "chunk_info": chunker.get_chunk_info(all_chunks)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents_count": len(vector_db.vectors),
        "db_stats": vector_db.get_stats()
    }

