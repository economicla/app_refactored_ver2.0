"""
PostgresDocumentAdapter - PostgreSQL'i IDocumentRepository interface'ine adapt et
Async SQLAlchemy ile connection pooling
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, func, select, text, delete, Index, Boolean, Text
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
import logging

from app_refactored.core.interfaces import IDocumentRepository
from app_refactored.core.entities import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)
Base = declarative_base()

class APIKeyModel(Base):
    """API Key management for n8n & external integrations"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    integration_type = Column(String(50), nullable=False) # "n8n", "frontend", "custom"

    # Rate limiting
    rate_limit_requests = Column(Integer, default=1000)
    rate_limit_window_seconds = Column(Integer, default=3600)

    #Permissions
    allowed_operations = Column(JSON, default=["query"])
    allowed_endpoints = Column(JSON, default=[])

    #Status
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

class DocumentModel(Base):
    """Documents table with user isolation and audit trail"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    document_id = Column(String(255), unique=True, nullable=False)
    filename = Column(String(255), nullable=False, index=True)
    file_type = Column(String(50), default="pdf")
    file_size = Column(Integer, default=0)
    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=True)
    source = Column(String(255), nullable=True)
    content_hash = Column(String(64), nullable=True)
    embedding = Column(Vector(2048))  # pgvector
    doc_metadata = Column(JSON, default={})
    is_encrypted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)


class AuditLogModel(Base):

    """Comprehensive audit trail for banking compliance"""

    __tablename__ = "audit_logs"
 
    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(String(255), nullable=False, index=True)

    request_id = Column(String(36), unique=True, nullable=False)  # Idempotency

    # Query details

    query_text = Column(String(2000), nullable=False)

    query_language = Column(String(50), default="tr")

    # Response details

    generated_response = Column(Text, nullable=True)

    response_summary = Column(String(500), nullable=True)

    used_documents = Column(JSON, default=[])  # [{id, filename, similarity}, ...]

    # Token accounting

    input_tokens = Column(Integer, default=0)

    output_tokens = Column(Integer, default=0)

    total_tokens = Column(Integer, default=0)

    # Performance metrics

    response_time_ms = Column(Integer, nullable=False)

    vector_search_time_ms = Column(Integer, default=0)

    # Request context

    ip_address = Column(String(45), nullable=False, index=True)

    user_agent = Column(String(255), nullable=True)

    api_key_hash = Column(String(64), nullable=True)  # Authentication audit

    # Status tracking

    status = Column(String(50), default="success", index=True)

    error_code = Column(String(50), nullable=True)

    error_message = Column(String(500), nullable=True)

    # Webhook/Integration

    integration_type = Column(String(50), nullable=True)  # "frontend", "n8n", "api"

    webhook_delivery_status = Column(String(50), nullable=True)

    # Compliance

    data_classification = Column(String(50), default="public")  # public, internal, confidential

    is_pii_detected = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (

        Index('idx_audit_user_created', 'user_id', 'created_at'),

        Index('idx_audit_request_id', 'request_id'),

    )
 

class QueryLogModel(Base):
    """SQLAlchemy ORM Model - PostgreSQL'deki query_logs tablosu"""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(String(1000), nullable=False)
    answer_preview = Column(String(500), nullable=True)
    response_time_ms = Column(Integer, nullable=False)
    chunks_retrieved = Column(Integer, default=0)
    top_source = Column(String(255), nullable=True)
    user_ip = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DocumentChunkModel(Base):

    """SQLAlchemy ORM Model - PostgreSQL'deki document_chunks tablosu"""

    __tablename__ = "document_chunks"
 
    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(String, nullable=False)

    chunk_index = Column(Integer)

    content = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)
 
 
class EmbeddingModel(Base):

    """SQLAlchemy ORM Model - PostgreSQL'deki embeddings tablosu"""

    __tablename__ = "embeddings"
 
    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(String, nullable=False)

    embedding = Column(Vector(2048))

    created_at = Column(DateTime, default=datetime.utcnow)

class PostgresDocumentAdapter(IDocumentRepository):
    """PostgreSQL async adapter with async SQLAlchemy and connection pooling"""

    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_timeout: int = 30
    ):
        """
        Initialize PostgreSQL adapter
        
        Args:
            database_url: Async PostgreSQL URL (postgresql+asyncpg://...)
            pool_size: Connection pool size
            max_overflow: Maksimum overflow connections
            pool_timeout: Pool timeout (saniye)
        """
        self.database_url = database_url
        
        # Async engine with connection pooling
        self.engine = create_async_engine(
            database_url,
            echo=False,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=True  # Connection health check
        )
        
        # Async session factory
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def create_tables(self):

        """Modelleri veritabanında tablo olarak oluşturur (Startup'ta çağrılmalı)"""

        async with self.engine.begin() as conn:

            try:

                # 1. pgvector uzantısını aktif et

                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                # 2. Tüm tabloları (Base'den türeyenler) oluştur

                # Not: run_sync asenkron akışta senkron SQLAlchemy komutlarını çalıştırmak içindir

                await conn.run_sync(Base.metadata.create_all)

                logger.info("✅ Veritabanı şeması başarıyla doğrulandı ve tablolar oluşturuldu.")

            except Exception as e:

                logger.error(f"❌ Tablo oluşturma sırasında hata: {str(e)}")

                # Banka ortamında burada hata alırsan muhtemelen 'superuser' yetkin yoktur.

                # Öyle bir durumda 'CREATE EXTENSION' satırını yorum satırı yapabilirsin.

    async def _get_session(self) -> AsyncSession:
        """Get async session from pool"""
        return self.async_session_maker()

    async def save(self, document: DocumentChunk) -> DocumentChunk:
        """Bir doküman chunk'ını kaydet"""
        async with await self._get_session() as session:
            try:
                db_doc = DocumentModel(
                    filename=document.filename,
                    chunk_index=document.chunk_index,
                    content=document.content,
                    embedding=document.embedding,
                    doc_metadata=document.metadata,
                    created_at=datetime.utcnow()
                )
                session.add(db_doc)
                await session.commit()
                await session.refresh(db_doc)
                
                document.id = db_doc.id
                document.created_at = db_doc.created_at
                
                logger.info(f"✅ Saved: {document.filename} chunk {document.chunk_index}")
                return document
                
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Save failed: {str(e)}")
                raise

    async def save_batch(self, documents: List[DocumentChunk]) -> List[DocumentChunk]:
        """Birden fazla chunk'ı batch olarak kaydet"""
        async with await self._get_session() as session:
            try:
                db_docs = [
                    DocumentModel(
                        filename=doc.filename,
                        chunk_index=doc.chunk_index,
                        content=doc.content,
                        embedding=doc.embedding,
                        doc_metadata=doc.metadata,
                        created_at=datetime.utcnow()
                    )
                    for doc in documents
                ]
                
                session.add_all(db_docs)
                await session.commit()
                
                # Refresh ve ID'leri ata
                for doc, db_doc in zip(documents, db_docs):
                    await session.refresh(db_doc)
                    doc.id = db_doc.id
                    doc.created_at = db_doc.created_at
                
                logger.info(f"✅ Batch saved: {len(documents)} chunks")
                return documents
                
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Batch save failed: {str(e)}")
                raise

    async def save_with_metadata(

        self,

        document: DocumentChunk,

        metadata: Dict[str, Any]

    ) -> DocumentChunk:

        """Metadata ile kaydet (upload_date, pages vb)"""

        async with await self._get_session() as session:

            try:

                # Metadata'ı birleştir

                full_metadata = document.metadata or {}

                full_metadata.update(metadata)

                db_doc = DocumentModel(

                    filename=document.filename,

                    chunk_index=document.chunk_index,

                    content=document.content,

                    embedding=document.embedding,

                    doc_metadata=full_metadata,

                    created_at=datetime.utcnow()

                )

                session.add(db_doc)

                await session.commit()

                await session.refresh(db_doc)

                document.id = db_doc.id

                document.metadata = full_metadata

                document.created_at = db_doc.created_at

                logger.info(f"✅ Saved with metadata: {document.filename}")

                logger.info(f"   Metadata: {full_metadata}")

                return document

            except Exception as e:

                await session.rollback()

                logger.error(f"❌ Save with metadata failed: {str(e)}")

                raise
 
    
    async def search_similar(
        self,
        embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> SearchResult:
        """pgvector L2 distance ile benzer dokümanları ara"""
        async with await self._get_session() as session:
            try:
                # pgvector L2 distance query
                distance = DocumentModel.embedding.cosine_distance(embedding)
                
                query = (
                    select(DocumentModel, distance.label('distance'))
                    .where(DocumentModel.deleted_at.is_(None))
                    .where(
                        text(
                            "(doc_metadata->>'is_dictionary') IS DISTINCT FROM 'true'"
                            " AND lower(filename) NOT LIKE '%sozluk%'"
                            " AND lower(filename) NOT LIKE '%sözlük%'"
                            " AND lower(filename) NOT LIKE '%dictionary%'"
                        )
                    )
                    .order_by(distance)
                    .limit(top_k)
                )
                
                result = await session.execute(query)
                rows = result.fetchall()
                
                documents = []
                for db_doc, distance_val in rows:
                    similarity_score = 1.0 - distance_val  # Convert distance to similarity
                    
                    doc = DocumentChunk(
                        id=db_doc.id,
                        filename=db_doc.filename,
                        chunk_index=db_doc.chunk_index,
                        content=db_doc.content,
                        embedding=db_doc.embedding,
                        similarity_score=float(similarity_score),
                        metadata=db_doc.doc_metadata or {},
                        created_at=db_doc.created_at
                    )
                    documents.append(doc)
                
                logger.info(f"🔍 Found {len(documents)} similar documents")
                
                return SearchResult(
                    documents=documents,
                    search_time_ms=0.0
                )
                
            except Exception as e:
                logger.error(f"❌ Search failed: {str(e)}")
                raise

    async def search_similar_filtered(
        self,
        embedding: List[float],
        document_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        top_k: int = 5
    ) -> SearchResult:
        """Document ID veya doc_type'a göre filtered search"""
        async with await self._get_session() as session:
            try:
                distance = DocumentModel.embedding.cosine_distance(embedding)

                query = (
                    select(DocumentModel, distance.label('distance'))
                    .where(DocumentModel.deleted_at.is_(None))
                    .where(
                        text(
                            "(doc_metadata->>'is_dictionary') IS DISTINCT FROM 'true'"
                            " AND lower(filename) NOT LIKE '%sozluk%'"
                            " AND lower(filename) NOT LIKE '%sözlük%'"
                            " AND lower(filename) NOT LIKE '%dictionary%'"
                        )
                    )
                )

                if document_id:
                    query = query.where(DocumentModel.filename == document_id)
                    logger.info(f"🔍 Filtered search in document: {document_id}")
                elif doc_type:
                    query = query.where(
                        text("doc_metadata->>'doc_type' = :filter_doc_type").bindparams(
                            filter_doc_type=doc_type
                        )
                    )
                    logger.info(f"🔍 Filtered search by doc_type: {doc_type}")
                else:
                    logger.info("🔍 Global search (no filter)")

                query = query.order_by(distance).limit(top_k)

                result = await session.execute(query)
                rows = result.all()

                documents = []
                for db_doc, distance_val in rows:
                    similarity_score = 1.0 - distance_val

                    doc = DocumentChunk(
                        id=db_doc.id,
                        filename=db_doc.filename,
                        chunk_index=db_doc.chunk_index,
                        content=db_doc.content,
                        embedding=db_doc.embedding,
                        similarity_score=float(similarity_score),
                        metadata=db_doc.doc_metadata or {},
                        created_at=db_doc.created_at
                    )
                    documents.append(doc)
                logger.info(f"Found {len(documents)} results")

                return SearchResult(
                    documents=documents,
                    search_time_ms=0.0
                )
            except Exception as e:
                logger.error(f"❌ Filtered search failed: {str(e)}")
                raise
            
    async def search_dictionary(
        self,
        embedding: List[float],
        top_k: int = 5
    ) -> SearchResult:
        """
        Sadece veri sözlüğü (is_dictionary=true) chunk'larını ara.
        Sözlük dokümanı yoksa boş SearchResult döndür.
        """
        async with await self._get_session() as session:
            try:
                distance = DocumentModel.embedding.cosine_distance(embedding)

                query = (
                    select(DocumentModel, distance.label('distance'))
                    .where(DocumentModel.deleted_at.is_(None))
                    .where(
                        text(
                            "(doc_metadata->>'is_dictionary' = 'true'"
                            " OR lower(filename) LIKE '%sozluk%'"
                            " OR lower(filename) LIKE '%sözlük%'"
                            " OR lower(filename) LIKE '%dictionary%')"
                        )
                    )
                    .order_by(distance)
                    .limit(top_k)
                )

                result = await session.execute(query)
                rows = result.fetchall()

                documents = []
                for db_doc, distance_val in rows:
                    similarity_score = 1.0 - distance_val
                    doc = DocumentChunk(
                        id=db_doc.id,
                        filename=db_doc.filename,
                        chunk_index=db_doc.chunk_index,
                        content=db_doc.content,
                        embedding=db_doc.embedding,
                        similarity_score=float(similarity_score),
                        metadata=db_doc.doc_metadata or {},
                        created_at=db_doc.created_at
                    )
                    documents.append(doc)

                if documents:
                    logger.info(f"📖 Dictionary search: {len(documents)} sözlük chunk'ı bulundu")
                return SearchResult(documents=documents, search_time_ms=0.0)

            except Exception as e:
                logger.debug(f"📖 Dictionary search not available: {e}")
                return SearchResult(documents=[], search_time_ms=0.0)

    async def get_by_filename(self, filename: str) -> List[DocumentChunk]:
        """Dosya adına göre tüm chunk'ları getir"""
        async with await self._get_session() as session:
            try:
                query = (
                    select(DocumentModel)
                    .where(DocumentModel.filename == filename)
                    .order_by(DocumentModel.chunk_index)
                )
                
                result = await session.execute(query)
                db_docs = result.scalars().all()
                
                documents = [
                    DocumentChunk(
                        id=db_doc.id,
                        filename=db_doc.filename,
                        chunk_index=db_doc.chunk_index,
                        content=db_doc.content,
                        embedding=db_doc.embedding,
                        metadata=db_doc.doc_metadata or {},
                        created_at=db_doc.created_at
                    )
                    for db_doc in db_docs
                ]
                
                logger.info(f"✅ Retrieved {len(documents)} chunks for {filename}")
                return documents
                
            except Exception as e:
                logger.error(f"❌ Get by filename failed: {str(e)}")
                raise

    async def get_all_chunks(self) -> List[DocumentChunk]:

        """Veritabanındaki tüm döküman parçalarını getir (JSON uyumlu)"""

        from sqlalchemy import select

        async with (await self._get_session()) as session:

            try:

                result = await session.execute(select(DocumentModel))

                db_docs = result.scalars().all()

                documents = []

                for db_doc in db_docs:

                    doc = DocumentChunk(

                        id=db_doc.id,

                        filename=db_doc.filename,

                        chunk_index=db_doc.chunk_index,

                        content=db_doc.content,

                        # HATA BURADAN KAYNAKLANIYORDU: 

                        # Vektör verisini listeye dahil etmiyoruz (None yapıyoruz)

                        embedding=None, 

                        metadata=db_doc.doc_metadata or {}, 

                        created_at=db_doc.created_at

                    )

                    documents.append(doc)

                logger.info(f"✅ {len(documents)} döküman parçası JSON uyumlu hale getirildi")

                return documents

            except Exception as e:

                logger.error(f"❌ Serialization hazırlığı başarısız: {str(e)}")

                raise
 
 
 
    
    async def get_by_id(self, chunk_id: int) -> Optional[DocumentChunk]:
        """ID'ye göre tek chunk getir"""
        async with await self._get_session() as session:
            try:
                query = select(DocumentModel).where(DocumentModel.id == chunk_id)
                result = await session.execute(query)
                db_doc = result.scalar_one_or_none()
                
                if not db_doc:
                    return None
                
                return DocumentChunk(
                    id=db_doc.id,
                    filename=db_doc.filename,
                    chunk_index=db_doc.chunk_index,
                    content=db_doc.content,
                    embedding=db_doc.embedding,
                    metadata=db_doc.doc_metadata or {},
                    created_at=db_doc.created_at
                )
                
            except Exception as e:
                logger.error(f"❌ Get by ID failed: {str(e)}")
                raise

    async def delete(self, chunk_id: int) -> bool:

        """ID'ye göre document ve ilişkili data'yı sil (cascade)"""

        async with await self._get_session() as session:

            try:

                query = select(DocumentModel).where(DocumentModel.id == chunk_id)

                result = await session.execute(query)

                db_doc = result.scalar_one_or_none()

                if not db_doc:

                    return False

                doc_filename = db_doc.filename

                # Document'ı doğrudan sil

                await session.delete(db_doc)

                await session.commit()

                logger.info(f"✅ Deleted: {db_doc.filename}")

                return True

            except Exception as e:

                await session.rollback()

                logger.error(f"❌ Delete failed: {str(e)}")

                raise
 
    async def delete_by_filename(self, filename: str) -> int:

        """Dosya adına göre tüm data'yı sil (cascade)"""

        async with await self._get_session() as session:

            try:

                query = select(DocumentModel).where(DocumentModel.filename == filename)

                result = await session.execute(query)

                db_docs = result.scalars().all()

                if not db_docs:

                    return 0

                count = len(db_docs)

                # Documents sil

                for db_doc in db_docs:

                    await session.delete(db_doc)

                await session.commit()

                logger.info(f"✅ Deleted {count} docs for {filename}")

                return count

            except Exception as e:

                await session.rollback()

                logger.error(f"❌ Delete by filename failed: {str(e)}")

                raise
 
 

    async def count(self) -> int:
        """Toplam chunk sayısını döndür"""
        async with await self._get_session() as session:
            try:
                query = select(func.count(DocumentModel.id))
                result = await session.execute(query)
                count = result.scalar()
                
                return count or 0
                
            except Exception as e:
                logger.error(f"❌ Count failed: {str(e)}")
                raise

    async def log_query(
        self,
        query_text: str,
        answer_preview: str,
        response_time_ms: int,
        chunks_retrieved: int,
        top_source: str = "N/A",
        user_ip: str = None
    ) -> int:
        """
        Bir sorguyu analytics için veritabanına kaydet

        Args:
            query_text: Orijinal sorgu
            answer_preview: Yanıtın ilk 500 karakteri
            response_time_ms: Milisaniye cinsinden yanıt süresi
            chunks_retrieved: Kullanılan chunk sayısı
            top_source: En uygun kaynak dosyasının adı
            user_ip: İsteğe bağlı kullanıcı IP adresi

        Returns:
            Log entry ID
        """
        try:
            async with await self._get_session() as session:

                log_entry = QueryLogModel(
                    query_text=query_text[:1000],
                    answer_preview=answer_preview[:500],
                    response_time_ms=response_time_ms,
                    chunks_retrieved=chunks_retrieved,
                    top_source=top_source,
                    user_ip=user_ip
                )
                session.add(log_entry)
                await session.commit()
                logger.info(f"✅ Query logged: {log_entry.id}")
                return log_entry.id
        except Exception as e:
            logger.error(f"❌ Query logging failed: {str(e)}")
            return -1

    async def get_analytics(self) -> dict:
        """Analytics dashboard için istatistikler topla

        Returns:
            Dictionary: toplam document, chunk, char, query stats
        """
        async with await self._get_session() as session:
            try:
                # Toplam document sayısı (unique filenames)
                doc_count_query = select(func.count(func.distinct(DocumentModel.filename)))
                doc_result = await session.execute(doc_count_query)
                total_documents = doc_result.scalar() or 0

                #Toplam chunk sayısı
                chunk_count_query = select(func.count(DocumentModel.id))
                chunk_result = await session.execute(chunk_count_query)
                total_chunks = chunk_result.scalar() or 0

                # Toplam karakter sayısı (func.length ile)
                char_sum_query = select(
                    func.sum(func.length(DocumentModel.content.cast(String)))
                )
                char_result = await session.execute(char_sum_query)
                total_chars = char_result.scalar() or 0
                avg_chunk_size = int(total_chars // total_chunks) if total_chunks > 0 else 0

                # Toplam log edilmiş sorgu
                query_count_query = select(func.count(QueryLogModel.id))
                query_result = await session.execute(query_count_query)
                total_queries = query_result.scalar() or 0

                # Ortalama yanıt süresi
                avg_time_query = select(func.avg(QueryLogModel.response_time_ms))
                avg_time_result = await session.execute(avg_time_query)
                avg_response_time = int(avg_time_result.scalar() or 0)

                #Bugünün soruları
                today_queries_query = select(func.count(QueryLogModel.id)).where(
                    func.date(QueryLogModel.created_at) == func.current_date()
                )
                today_queries_result = await session.execute(today_queries_query)
                queries_today = today_queries_result.scalar() or 0

                #Bugün yüklenen document
                today_docs_query = select(func.count(func.distinct(DocumentModel.filename))).where(
                    func.date(DocumentModel.created_at) == func.current_date()
                )
                today_docs_result = await session.execute(today_docs_query)
                documents_ingested_today = today_docs_result.scalar() or 0

                logger.info("✅ Analytics data collected")

                return {
                    "total_documents": total_documents,
                    "total_chunks": total_chunks,
                    "total_chars": total_chars,
                    "avg_chunk_size": avg_chunk_size,
                    "total_queries": total_queries,
                    "avg_query_response_time_ms": avg_response_time,
                    "documents_ingested_today": documents_ingested_today,
                    "queries_today": queries_today,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"❌ Analytics query failed: {str(e)}")
                return {"error": str(e)}


    async def get_system_info(self) -> dict:
        """
        Sistem şeffaflığı için bilgiler (C-level demo için)

        Returns:
            Dictionary: PostgreSQL, pgvector, ve servis bilgileri
        """
        async with await self._get_session() as session:
            try:
                from sqlalchemy import text

                #PostgreSQL sürümü
                version_result = await session.execute(text("SELECT version()"))
                pg_version_full = version_result.scalar() or "Unknown"
                #"PostgreSQL 16.1 on..." şeklinden 16.1 çıkart
                pg_version = pg_version_full.split(',')[0].replace('PostgreSQL ','')

                #pgvector extension kontrol
                ext_result = await session.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname='vector'")
                )
                pgvector_version = ext_result.scalar() or "Not installed"
                pgvector_enabled = pgvector_version != "Not installed"

                #Son yükleme zamanı
                last_ingest_query = select(func.max(DocumentModel.created_at))
                last_ingest_result = await session.execute(last_ingest_query)
                last_ingestion = last_ingest_result.scalar()
                last_ingestion_str = last_ingestion.isoformat() if last_ingestion else "Never"

                logger.info("✅ System info retrieved")

                return{
                    "api_version": "2.0",
                    "database": {
                        "type": "PostgreSQL",
                        "version": pg_version,
                        "pgvector_enabled": pgvector_enabled,
                        "pgvector_version": pgvector_version
                    },
                    "embedding_service": {
                        "provider": "Jina",
                        "model": "jina-embeddings-v3",
                        "dimensions": 2048,
                        "status": "configured"
                    },
                    "llm_service": {
                        "provider": "vLLM",
                        "model": "gpt-oss120b",
                        "hardware": "H100",
                        "status": "configured"
                    },
                    "last_ingestion": last_ingestion_str,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"❌ System info query failed: {str(e)}")
                return{"error": str(e)}


    async def close(self):
        """Veritabanı bağlantısını kapat"""
        await self.engine.dispose()
        logger.info("✅ Database connection closed")

    async def __aenter__(self):
        """Context manager support"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        await self.close()
