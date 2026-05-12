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

    Form,

    UploadFile, 

    Depends, 

    HTTPException,

    Query,

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

    ScopedRAGQueryRequest,

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

        postgres_ok = True
        try:
            await container.get_document_repository().count()
        except Exception as db_err:
            logger.warning(f"Health check: PostgreSQL probe failed: {db_err}")
            postgres_ok = False

        vlm_ok = False
        try:
            import httpx as _httpx
            vlm_cfg = container.config['vlm']
            base = f"{vlm_cfg['host']}:{vlm_cfg['port']}" if vlm_cfg['port'] else vlm_cfg['host']
            async with _httpx.AsyncClient(timeout=_httpx.Timeout(5)) as c:
                r = await c.get(f"{base}/v1/models")
                vlm_ok = r.status_code == 200
        except Exception as vlm_err:
            logger.warning(f"Health check: VLM probe failed: {vlm_err}")

        all_ok = jina_ok and vllm_ok and postgres_ok and vlm_ok
        return HealthCheckResponse(
            status="healthy" if all_ok else "degraded",
            jina_available=jina_ok,
            postgres_available=postgres_ok,
            vllm_available=vllm_ok,
            vlm_available=vlm_ok,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
# ============ RAG QUERY ============

async def _execute_rag_query(
    request: RAGQueryRequest,
    container: DIContainer,
    unit: Optional[str] = None,
    collection: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> RAGQueryResponse:
    """Ortak RAG çalıştırma (global, filename/filenames veya collection ile kapsamlı)."""
    import time

    start_time = time.time()
    logger.info(f"📥 RAG Query: {request.query[:50]}... (unit={unit}, collection={collection})")

    rag_use_case = container.get_rag_query_use_case()
    query = RAGQuery(
        query=request.query,
        top_k=request.top_k,
        temperature=request.temperature,
        filename=getattr(request, "filename", None),
        filenames=request.filenames,
        unit=unit,
        collection=collection,
        system_prompt=system_prompt,
    )
    result = await rag_use_case.execute(query)

    sources = [
        DocumentChunkResponse(
            id=doc.chunk_index,
            filename=doc.filename,
            unit=getattr(doc, "unit", None),
            chunk_index=doc.chunk_index,
            content=doc.content_preview,
            similarity_score=doc.similarity_score,
            metadata={},
            created_at=None,
            source_pages=getattr(doc, "source_pages", None),
        )
        for doc in result.sources
    ]

    response = RAGQueryResponse(
        question=result.question,
        answer=result.answer,
        sources=sources,
        model=result.model,
        timestamp=result.timestamp,
        debug_info=result.debug_info,
    )

    response_time_ms = int((time.time() - start_time) * 1000)
    top_source = result.sources[0].filename if result.sources else "N/A"

    try:
        await container.get_document_repository().log_query(
            query_text=request.query,
            answer_preview=result.answer[:200],
            response_time_ms=response_time_ms,
            chunks_retrieved=len(result.sources),
            top_source=top_source,
        )
        logger.info(f"✅ Query logged: {response_time_ms}ms, {len(result.sources)} chunks")
    except Exception as log_error:
        logger.warning(f"⚠️ Query logging failed (non-critical): {log_error}")

    return response

@router.post("/query/scoped", response_model=RAGQueryResponse)

async def query_documents_scoped(

    request: ScopedRAGQueryRequest,

    container: DIContainer = Depends(get_di_container),

):

    """

    Kapsamlı RAG — birim, dosya adları ve/veya koleksiyon ile filtrelenmiş arama.

    unit, collection veya filenames'den en az biri verilmelidir.

    """
    if not request.filenames and not request.collection and not getattr(request, "unit", None):
        raise HTTPException(
            status_code=422,
            detail="filenames, collection veya unit parametrelerinden en az biri gereklidir.",
        )
    try:
        unified = RAGQueryRequest(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            filename=None,
            filenames=request.filenames,
        )
        return await _execute_rag_query(
            unified,
            container,
            unit=getattr(request, "unit", None),
            collection=getattr(request, "collection", None),
            system_prompt=getattr(request, "system_prompt", None),
        )
    except Exception as e:
        logger.error(f"❌ Scoped query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
# ============ RAG QUERY STREAM ============
 
async def stream_query_generator(

    query_text: str,

    top_k: int,

    temperature: float,

    container: DIContainer,

    start_time: float,

    filename: Optional[str] = None,

    filenames: Optional[List[str]] = None,

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

            temperature=temperature,

            filename=filename,

            filenames=filenames,

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
                    "content_preview": doc.content_preview,
                    "source_pages": getattr(doc, "source_pages", None),
                }
                for doc in result.sources
            ]
            yield f"data: {json.dumps({'sources': sources_data})}\n\n"

        if result.debug_info:
            yield f"data: {json.dumps({'debug_info': result.debug_info})}\n\n"
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

                start_time=start_time,

                filename=request.filename,

                filenames=request.filenames,

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

    unit: Optional[str] = Form(default=None, description="Birim/departman adı (ör. 'krediler', 'egitim')"),

    collection: Optional[str] = Form(default=None, description="Koleksiyon adı (ör. 'kredi', 'egitim')"),

    container: DIContainer = Depends(get_di_container)

):

    """

    Doküman Yükleme

    Desteklenen formatlar: PDF, DOCX, TXT, XLSX, PPTX

    Opsiyonel: unit ve collection parametreleri ile dokümanı birim/koleksiyona ata

    """

    temp_file_path = None

    try:

        logger.info(f"📥 Ingesting file: {file.filename} (unit={unit}, collection={collection})")

        # Geçici dosya oluştur

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:

            temp_file_path = tmp.name

            # Dosyayı geçici konuma yaz

            content = await file.read()

            with open(temp_file_path, 'wb') as f:

                f.write(content)

        ingestion_use_case = container.get_document_ingestion_use_case()

        result = await ingestion_use_case.execute(
            temp_file_path,
            file.filename,
            collection=collection,
            unit=unit,
        )

        return DocumentIngestionResponse(

            status=result.status,

            filename=result.filename,

            chunks_ingested=result.chunks_ingested,

            total_tokens=result.total_tokens,

            timestamp=result.timestamp,

            error=result.error

        )

    except HTTPException:

        raise

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
    """
    Tüm chunk'ları listele. Her öğede size_chars (içerik uzunluğu) bulunur;
    ön yüz/pipe bunu kullanarak toplam karakter sayısını hesaplar.
    """
    try:
        repo = container.get_document_repository()
        documents = await repo.get_all_chunks()

        results = []
        for doc in documents:
            created = doc.created_at.isoformat() if getattr(doc, "created_at", None) else None
            content = getattr(doc, "content", "") or ""
            results.append({
                "id": doc.id,
                "filename": doc.filename,
                "unit": getattr(doc, "unit", None),
                "collection": getattr(doc, "collection", None),
                "chunk_index": getattr(doc, "chunk_index", 0),
                "created_at": created,
                "metadata": getattr(doc, "metadata", None) or {},
                "size_chars": len(content),
            })

        return {"total": len(results), "results": results}
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

@router.delete("/documents/by-filename/{filename}")
async def delete_document_by_filename(
    filename: str,
    unit: Optional[str] = Query(default=None, description="Birim filtresi (opsiyonel)"),
    collection: Optional[str] = Query(default=None, description="Collection filtresi (opsiyonel)"),
    container: DIContainer = Depends(get_di_container)
):
    """
    Dosya adına göre chunk'ları sil. unit/collection verilirse sadece o scope silinir.
    Örn: DELETE /api/v2/documents/by-filename/topbas-performans.html?unit=krediler&collection=karesi
    """
    try:
        repo = container.get_document_repository()
        deleted_count = await repo.delete_by_filename(filename, unit=unit, collection=collection)
        if deleted_count > 0:
            logger.info(f"✅ Deleted {deleted_count} chunks for {filename} (unit={unit}, collection={collection})")
            return {
                "status": "deleted",
                "filename": filename,
                "unit": unit,
                "collection": collection,
                "chunks_deleted": deleted_count,
            }
        else:
            raise HTTPException(status_code=404, detail=f"No documents found for '{filename}'")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete by filename failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def set_di_container(container: DIContainer):

    """DIContainer'ı global olarak set et (app startup'ta çağrılacak)"""

    global _di_container

    _di_container = container

    logger.info("✅ DIContainer set for routes")
 
