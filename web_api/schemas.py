
"""

Pydantic Schemas - API Request/Response DTOs

core/entities'deki modelleri baz alan, Pydantic compatible

"""

from pydantic import BaseModel, Field

from typing import List, Optional

from datetime import datetime
 
 
# ============ REQUEST SCHEMAS ============
 
class RAGQueryRequest(BaseModel):

    """RAG Sorgu Request"""

    query: str = Field(..., min_length=1, max_length=5000, description="Sorgu metni")

    top_k: int = Field(default=10, ge=1, le=20, description="En iyi k sonuç")

    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Yaratıcılık (0-1)")
 
    class Config:

        example = {

            "query": "eğitim prosedürü nedir?",

            "top_k": 15,

            "temperature": 0.7

        }
 
 
# ============ RESPONSE SCHEMAS ============
 
class DocumentChunkResponse(BaseModel):

    """Doküman Chunk Response"""

    id: Optional[int] = None

    filename: str

    chunk_index: int

    content: str = Field(..., description="İlk 200 karakteri gösterilir")

    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)

    metadata: dict = Field(default_factory=dict)

    created_at: Optional[datetime] = None
 
    class Config:

        example = {

            "id": 1,

            "filename": "egitim_proseduru.docx",

            "chunk_index": 0,

            "content": "Bu prosedür, Bankamızın strateji ve hedefleri çerçevesinde...",

            "similarity_score": 0.85,

            "metadata": {"file_type": "docx", "chunk_size": 1000}

        }
 
 
class RAGQueryResponse(BaseModel):

    """RAG Sorgu Yanıtı"""

    question: str

    answer: str

    sources: List[DocumentChunkResponse] = Field(default_factory=list)

    model: str

    timestamp: datetime
 
    class Config:

        example = {

            "question": "eğitim prosedürü nedir?",

            "answer": "Eğitim prosedürü, bankanın tüm eğitim faaliyetlerini...",

            "sources": [

                {

                    "filename": "egitim_proseduru.docx",

                    "chunk_index": 0,

                    "similarity_score": 0.85

                }

            ],

            "model": "openai/gpt-oss-120b",

            "timestamp": "2026-02-06T12:00:00"

        }
 
 
class DocumentIngestionResponse(BaseModel):

    """Doküman Yükleme Yanıtı"""

    status: str = Field(..., description="success or error")

    filename: str

    chunks_ingested: int

    total_tokens: int

    timestamp: datetime

    error: Optional[str] = None
 
    class Config:

        example = {

            "status": "success",

            "filename": "egitim_proseduru.docx",

            "chunks_ingested": 25,

            "total_tokens": 6189,

            "timestamp": "2026-02-06T12:00:00"

        }
 
 
class HealthCheckResponse(BaseModel):

    """Health Check Yanıtı"""

    status: str = Field(default="healthy")

    jina_available: bool

    postgres_available: bool

    vllm_available: bool

    timestamp: datetime
 
    class Config:

        example = {

            "status": "healthy",

            "jina_available": True,

            "postgres_available": True,

            "vllm_available": True,

            "timestamp": "2026-02-06T12:00:00"

        }
 
 
class ErrorResponse(BaseModel):

    """Error Response"""

    status: str = "error"

    detail: str

    timestamp: datetime
 
    class Config:

        example = {

            "status": "error",

            "detail": "Invalid query",

            "timestamp": "2026-02-06T12:00:00"

        }
 
