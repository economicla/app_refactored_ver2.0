"""

FastAPI Routes - API Endpoints

Dependency Injection ile Use Case'leri tetikler

"""
 
import logging

import hashlib

import time

import uuid

from typing import AsyncGenerator, Optional, List, Dict, Any

from datetime import datetime

from functools import lru_cache
 
from fastapi import (

    APIRouter, 

    File, 

    UploadFile, 

    Depends, 

    HTTPException, 

    Request, 

    Header, 

    status

)

from fastapi.responses import StreamingResponse

import aiofiles

import tempfile

import os
 
from app_refactored.web_api.schemas import (

    RAGQueryRequest,

    RAGQueryResponse,

    DocumentIngestionResponse,

    DocumentChunkResponse,

    HealthCheckResponse,

    ErrorResponse

)

from app_refactored.di import DIContainer

from app_refactored.core.entities import RAGQuery, DocumentChunk

from app_refactored.infra.redis_client import redis_client
 
logger = logging.getLogger(__name__)
 
# Global DIContainer (app startup'ta set edilecek)

_di_container: DIContainer = None
 
# ============ CONTAINER MANAGEMENT ============
 
def get_di_container() -> DIContainer:

    """Dependency Injection - DIContainer'ı döndür"""

    if _di_container is None:

        raise RuntimeError("DIContainer not initialized")

    return _di_container
 
 
def set_di_container(container: DIContainer):

    """DIContainer'ı set et (main.py'de çağrılır)"""

    global _di_container

    _di_container = container
 
 
# ============ SECURITY & AUTHENTICATION LAYER ============
 
async def get_user_id(

    request: Request,

    x_user_id: Optional[str] = Header(None)

) -> str:

    """Extract user_id from X-User-ID header or query param"""

    if x_user_id:

        return x_user_id

    user_id = request.query_params.get("user_id")

    if user_id:

        return user_id

    return "anonymous"
 
 
async def verify_api_key(

    api_key: str = Header(None, alias="X-API-Key"),

    container: DIContainer = Depends(get_di_container)

) -> str:

    """Verify API Key for n8n & integrations"""

    if not api_key:

        raise HTTPException(

            status_code=status.HTTP_401_UNAUTHORIZED,

            detail="API Key required"

        )

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # TODO: Verify key_hash in database via repository

    return api_key
 
 
async def check_rate_limit(

    user_id: str = Depends(get_user_id),

    api_key: Optional[str] = Header(None, alias="X-API-Key"),

    limit: int = 100,

    window_seconds: int = 3600

) -> bool:

    """

    Redis-based rate limiting (atomic INCR + EXPIRE)

    Flow:

    1. Use Redis INCR to atomically increment counter

    2. Set EXPIRE to auto-reset after window_seconds

    3. Check if count exceeds limit

    """

    identifier = api_key or user_id

    try:

        # Check Redis rate limit

        within_limit = await redis_client.check_rate_limit(

            identifier=identifier,

            limit=limit,

            window_seconds=window_seconds

        )

        if not within_limit:

            current_count = await redis_client.get_rate_limit_count(identifier)

            raise HTTPException(

                status_code=status.HTTP_429_TOO_MANY_REQUESTS,

                detail=f"Rate limit exceeded: {current_count}/{limit} requests in {window_seconds}s"

            )

        return True

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Rate limit check error: {e}")

        # Fail open - allow on error

        return True
 
 
async def get_request_id(request: Request) -> str:

    """Extract or generate request ID for idempotency & audit trail"""

    req_id = request.headers.get("X-Request-ID")

    if req_id:

        return req_id

    return str(uuid.uuid4())
 
 
# ============ API ENDPOINTS ============

router = APIRouter(prefix="/api/v2", tags=["RAG API"])
 
# ============ HEALTH CHECK ============
 
@router.get("/health", response_model=HealthCheckResponse)

async def health_check(container: DIContainer = Depends(get_di_container)):

    """Sistem sağlığını kontrol et"""

    try:

        jina_ok = await container.get_embedding_service().is_available()

        vllm_ok = await container.get_llm_service().is_available()

        # PostgreSQL kontrol etmek için basit bir count query

        postgres_ok = True

        try:

            await container.get_document_repository().count()

        except:

            postgres_ok = False

        return HealthCheckResponse(

            status="healthy" if (jina_ok and vllm_ok and postgres_ok) else "degraded",

            jina_available=jina_ok,

            postgres_available=postgres_ok,

            vllm_available=vllm_ok,

            timestamp=datetime.now()

        )

    except Exception as e:

        logger.error(f"Health check failed: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))
 
 
# ============ RAG QUERY ============
 
@router.post("/query", response_model=RAGQueryResponse)

async def query_documents(

    request: RAGQueryRequest,

    container: DIContainer = Depends(get_di_container)

):

    """

    RAG Sorgusu - Dokümanlardan cevap bul

    - Sorguyu embedding'e çevir

    - Benzer dokümanları ara

    - LLM'den cevap al

    """
    import time 
    start_time = time.time() # TIMER BAŞLA

    try:

        logger.info(f"📥 RAG Query: {request.query[:50]}...")

        # Use Case'i al

        rag_use_case = container.get_rag_query_use_case()

        # Query object oluştur

        query = RAGQuery(

            query=request.query,

            top_k=request.top_k,

            temperature=request.temperature

        )

        # Execute use case

        result = await rag_use_case.execute(query)

        # Convert to response

        sources = [

            DocumentChunkResponse(

                id=doc.chunk_index,

                filename=doc.filename,

                chunk_index=doc.chunk_index,

                content=doc.content_preview,

                similarity_score=doc.similarity_score,

                metadata={},

                created_at=None

            )

            for doc in result.sources

        ]

        response = RAGQueryResponse(

            question=result.question,

            answer=result.answer,

            sources=sources,

            model=result.model,

            timestamp=result.timestamp

        )

        response_time_ms = int((time.time() - start_time) *1000)
        top_source = result.sources[0].filename if result.sources else "N/A"

        try:
            await container.get_document_repository().log_query(
                query_text=request.query,
                answer_preview=result.answer[:200],
                response_time_ms=response_time_ms,
                chunks_retrieved=len(result.sources),
                top_source=top_source
            )
            logger.info(f"✅ Query logged: {response_time_ms}ms, {len(result.sources)} chunks")
        except Exception as log_error:
            logger.warning(f"⚠️ Query logging failed (non-critical): {log_error}")

        return response

    except Exception as e:

        logger.error(f"❌ Query failed: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))
 
 
# ============ RAG QUERY STREAM ============
 
async def stream_query_generator(

    query_text: str,

    top_k: int,

    temperature: float,

    container: DIContainer,

    start_time: float

) -> AsyncGenerator[str, None]:

    """Query stream generator"""

    import time
    import json
    
    full_answer = ""
    

    try:

        rag_use_case = container.get_rag_query_use_case()

        query = RAGQuery(

            query=query_text,

            top_k=top_k,

            temperature=temperature

        )

        result = await rag_use_case.execute(query)

        yield f"data: {json.dumps({'answer': result.answer})}\n\n"
        full_answer = result.answer

        if result.sources:
            sources_data = [
                {
                    "filename": doc.filename,
                    "chunk_index": doc.chunk_index,
                    "header": doc.header,
                    "similarity_score": doc.similarity_score,
                    "content_preview": doc.content_preview
                }
                for doc in result.sources
            ]
            yield f"data: {json.dumps({'sources': sources_data})}\n\n"
        response_time_ms = int((time.time() - start_time) * 1000)

        try:
            await container.get_document_repository().log_query(
                query_text=query_text,
                answer_preview=full_answer[:200],
                response_time_ms=response_time_ms,
                chunks_retrieved=len(result.sources) if result.sources else 0,
                top_source=result.sources[0].filename if result.sources else "N/A"
            )
            logger.info(f"✅ Stream query logged: {response_time_ms}ms")
        except Exception as e:
            logger.warning(f"⚠️ Stream query logging failed: {e}")

    except Exception as e:

        logger.error(f"❌ Stream query failed: {str(e)}")

        yield f"data: {json.dumps({'error': str(e)})}\n\n"
 

@router.post("/query/filtered")

async def query_filtered(

    request: RAGQueryRequest,

    document_filter: Optional[str] = None,

    container: DIContainer = Depends(get_di_container)

):

    """

    Belirli bir dokümanda arama yap (metadata filtering)

    Query params:

    - document_filter: İsteğe bağlı, sadece bu dosyada ara (filename)

    Örnek: POST /api/v2/query/filtered?document_filter=my_doc.pdf

    """

    try:

        import time

        start_time = time.time()

        repository = container.get_document_repository()
        embedding_service = container.get_embedding_service()

        #Query text'i embed et (Jina kullanarak)
        query_embedding = await embedding_service.embed_text(request.query)
        
        # Filtered search

        search_result = await repository.search_similar_filtered(

            embedding=query_embedding,

            document_id=document_filter,

            top_k=request.top_k

        )

        response_time = int((time.time() - start_time) * 1000)

        # Analytics logging

        answer_preview = search_result.documents[0].content[:500] if search_result.documents else "No results"

        await repository.log_query(

            query_text=request.query,

            answer_preview=answer_preview,

            response_time_ms=response_time,

            chunks_retrieved=len(search_result.documents),

            top_source=search_result.documents[0].filename if search_result.documents else "N/A"

        )

        return {

            "status": "success",

            "query": request.query,

            "filter": document_filter or "global",

            "results": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "chunk_index": doc.chunk_index,
                    "content": doc.content[:500],
                    "similarity_score": doc.similarity_score,
                    "metadata": doc.metadata
                }
                for doc in search_result.documents
            ],           

            "response_time_ms": response_time

        }

    except Exception as e:

        logger.error(f"Filtered query failed: {e}")

        raise HTTPException(status_code=500, detail=str(e))
 


@router.post("/query/stream")

async def query_documents_stream(

    request: RAGQueryRequest,

    container: DIContainer = Depends(get_di_container)

):

    """

    RAG Sorgusu - Streaming Response

    Real-time cevap almak için

    """
    import time
    start_time = time.time()

    try:

        logger.info(f"📥 Stream Query: {request.query[:50]}...")

        return StreamingResponse(

            stream_query_generator(

                query_text=request.query,

                top_k=request.top_k,

                temperature=request.temperature,

                container=container,

                start_time=start_time

            ),

            media_type="text/event-stream"

        )

    except Exception as e:

        logger.error(f"❌ Stream query endpoint failed: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))
 
 
# ============ DOCUMENT INGESTION ============
 
@router.post("/ingest", response_model=DocumentIngestionResponse)

async def ingest_document(

    file: UploadFile = File(...),

    container: DIContainer = Depends(get_di_container)

):

    """

    Doküman Yükleme

    Desteklenen formatlar: PDF, DOCX, TXT, XLSX, PPTX

    """

    temp_file_path = None

    try:

        logger.info(f"📥 Ingesting file: {file.filename}")

        # Geçici dosya oluştur

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:

            temp_file_path = tmp.name

            # Dosyayı geçici konuma yaz

            content = await file.read()

            with open(temp_file_path, 'wb') as f:

                f.write(content)

        # Use Case'i al

        ingestion_use_case = container.get_document_ingestion_use_case()

        # Execute use case

        result = await ingestion_use_case.execute(temp_file_path, file.filename)

        return DocumentIngestionResponse(

            status=result.status,

            filename=result.filename,

            chunks_ingested=result.chunks_ingested,

            total_tokens=result.total_tokens,

            timestamp=result.timestamp,

            error=result.error

        )

    except Exception as e:

        logger.error(f"❌ Ingestion failed: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))

    finally:

        # Geçici dosyayı sil

        if temp_file_path and os.path.exists(temp_file_path):

            try:

                os.remove(temp_file_path)

            except:

                pass
 
@router.post("/ingest-with-metadata")

async def ingest_with_metadata(

    file: UploadFile = File(...),

    upload_date: Optional[str] = None,

    pages: Optional[str] = None,  # JSON: {"chunk_0": [1,2], "chunk_1": [3,4]}

    container: DIContainer = Depends(get_di_container)

):

    """

    Metadata ile dokument ingest et

    upload_date: ISO format (2025-02-09T10:30:00)

    pages: JSON string örneği: '{"chunk_0": [1, 2], "chunk_1": [3]}'

    """

    try:

        import json
        from pathlib import Path

        repository = container.get_document_repository()

        filename = file.filename

        file_content = await file.read()

        file_ext = Path(filename).suffix.lower()

        text = ""

        if file_ext == ".docx":
            # DOCX
            from docx import Document
            import io
            doc = Document(io.BytesIO(file_content))
            text = "\n".join([para.text for para in doc.paragraphs])

        elif file_ext == ".pdf":
            #PDF
            from pypdf import PdfReader
            import io
            pdf = PdfReader(io.BytesIO(file_content))
            text = "\n".join([page.extract_text() for page in pdf.pages])

        elif file_ext == ".txt":
            # TXT - UTF-8 ile decode et
            text = file_content.decode("utf-8", errors="replace")
        elif file_ext in (".html", ".htm"):
            # HTML - BeautifulSoup ile temiz metin çıkar
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(file_content, "html.parser")

            #Script ve style etiketlerini kaldır
            for tag in soup (["script", "style", "meta", "link"]):
                tag.decompose()

            #Tablo yapısını koruyarak metin çıkar
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
                    row.replace_width(" | ".join(cells) + "\n")

            text = soup.get_text(separator="\n", strip=True)
        else:
            #Default - UTF-8 decode et

            text = file_content.decode("utf-8", errors="replace")

        # Metadata oluştur

        metadata = {

            "filename": filename,

            "file_type": file_ext,

            "upload_date": upload_date or datetime.utcnow().isoformat(),

            "file_size": len(file_content),

            "content_length": len(text)

        }

        # Pages var mı?

        if pages:

            try:

                page_map = json.loads(pages)

                metadata["pages"] = page_map

            except json.JSONDecodeError:

                logger.warning("Pages JSON parse failed, skipping")

        from ..core.entities import DocumentChunk

        # Document oluştur

        doc = DocumentChunk(

            filename=filename,

            chunk_index=0,

            content=text,

            embedding=None,

            metadata={}

        )

        # Metadata ile kaydet

        saved = await repository.save_with_metadata(doc, metadata)

        return {

            "status": "success",

            "filename": filename,

            "file_type": file_ext,

            "metadata": saved.metadata,

            "content_length": len(text)

        }

    except Exception as e:

        logger.error(f"Ingest with metadata failed: {e}")

        raise HTTPException(status_code=500, detail=str(e))
 


# ============ DOCUMENT LIST ============

@router.get("/documents")

async def list_documents(container: DIContainer = Depends(get_di_container)):

    try:

        repo = container.get_document_repository()

        # Yeni eklediğimiz veya adaptörde olan uygun fonksiyonu çağırıyoruz

        documents = await repo.get_all_chunks() 

        return {

            "total": len(documents),

            "results": documents

        }

    except Exception as e:

        logger.error(f"❌ List failed: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))
 
 
@router.get("/documents/count")

async def count_documents(container: DIContainer = Depends(get_di_container)):

    """Toplam dokuman sayısını döndür"""

    try:

        repo = container.get_document_repository()

        count = await repo.count()

        return {"total_documents": count}

    except Exception as e:

        logger.error(f"❌ Count failed: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics", response_model=dict)

async def analytics_dashboard(container: DIContainer = Depends(get_di_container)):

    """

    Analytics dashboard - System statistics

    Returns:

        - total_documents: Unique file count

        - total_chunks: Total chunks

        - total_chars: Total characters

        - avg_chunk_size: Average chunk size

        - total_queries: Total logged queries

        - avg_query_response_time_ms: Average response time

        - documents_ingested_today: Today's ingested docs

        - queries_today: Today's queries

        - timestamp: ISO timestamp

    """

    try:

        logger.info("📊 Analytics dashboard requested")

        repo = container.get_document_repository()

        analytics = await repo.get_analytics()

        return analytics

    except Exception as e:

        logger.error(f"❌ Analytics failed: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{doc_id}")

async def delete_document(

    doc_id: int,

    container: DIContainer = Depends(get_di_container)

):

    """

    Dokümantı sil (ID'ye göre)

    Args:

        doc_id: Silinecek dokümantın ID'si

    Returns:

        Silme sonucu

    """

    try:

        repo = container.get_document_repository()

        deleted = await repo.delete(doc_id)

        if deleted:

            logger.info(f"✅ Document deleted: {doc_id}")

            return {"status": "deleted", "id": doc_id}

        else:

            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    except Exception as e:

        logger.error(f"❌ Delete failed: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))
 

@router.get("/info/system", response_model=dict)

async def system_information(container: DIContainer = Depends(get_di_container)):

    """

    System Information - Transparency for C-level demo

    Returns:

        - api_version: API version

        - database: PostgreSQL & pgvector info

        - embedding_service: Jina configuration

        - llm_service: vLLM configuration

        - last_ingestion: Last document ingestion time

        - timestamp: ISO timestamp

    """

    try:

        logger.info("ℹ️ System info requested")

        repo = container.get_document_repository()

        system_info = await repo.get_system_info()

        return system_info

    except Exception as e:

        logger.error(f"❌ System info failed: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))
 
 
 
def set_di_container(container: DIContainer):

    """DIContainer'ı global olarak set et (app startup'ta çağrılacak)"""

    global _di_container

    _di_container = container

    logger.info("✅ DIContainer set for routes")
 
