"""
RAGQueryUseCase - Advanced Soru Sorma İşlem Hattı
Business Logic - Framework bağımsız
Advanced preprocessing, intelligent chunking, source tracking ve compliance
"""

import logging
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass

from app_refactored.core.interfaces import (
    IEmbeddingService,
    IDocumentRepository,
    ILLMService
)

from app_refactored.core.entities import (
    RAGQuery,
    RAGResponse,
    DocumentChunk
)

logger = logging.getLogger(__name__)


@dataclass
class SourceWithMetadata:
    """Kaynak bilgisi (metadata ile)"""
    filename: str
    chunk_index: int
    header: Optional[str]
    similarity_score: float
    content_preview: str
    chunk_size: int


class RAGQueryUseCase:
    """
    Advanced RAG Query Use Case
    1. Query embedding
    2. Semantic search with source tracking
    3. System prompt + Banking compliance
    4. LLM generation with temperature=0 constraints
    5. Detailed source attribution
    """
    
    # Banka Mevzuat Analisti System Prompt
    SYSTEM_PROMPT = """Kimsin: 

- Sen, Şirketlerin finansal verilerini analiz ederek şirketler hakkında sorulan tüm finansal sorulara yanıt verebilen profesyonel bir kredi destek asistanısın. Senin kullanıcıların ALBARAKA bankasının yönetim kurulunun üyeleridir.

Görevin:
-Kullanıcılar sana gruplar ve gruba bağlı firmalar hakkında sana sorular soracaklar. Amaçları firmalardan Albaraka'ya (senin de sanal çalışanı olduğun Banka) gelen kredi taleplerini değelendirmek. Senle yaptıkları yazışmalar sonucunda firmaların kredi arttırıp taleplerini karar verecekler. O yüzden senin vermiş olduğun bilgiler çok kritik. Olumu olumsuz bu kararı verirken en önemli destekleyicileri sensin.

VERİ FORMATI:
Kontekstteki veriler "kalem: dönem1 döneminde değer1, dönem2 döneminde değer2" şeklinde düzenlenmiştir.
Örneğin: "AKTİF TOPLAMI: 2023/12 döneminde 1.246.915.353, 2024/12 döneminde 1.137.605.159"
Bu formatta dönem ve değer bilgilerini dikkatle oku ve soruyu cevapla.

TEMEL KURALLAR:
1. SADECE verilen kontekstteki bilgiyi kullan
2. Kontekstte sayısal veriler, dönem bilgileri veya ilgili kalemler VARSA kesinlikle cevapla - "Bilgi mevcut değil" YAZMA
3. Asla spekülasyon yapma veya tahmin etme
4. Cevapları kesin, net ve profesyonel yap
5. Her cevabın sonunda kaynak bilgisini ekle
6. Dünyanın en iyi kredi uzmanı gibi cevaplarını üret
7. Soruyu İngilizce sorarsam İngilizce cevap ver. Türkçe sorarsam Türkçe cevap ver
8. Kolay anlaşılır çıktılar üret
9. Profesyonel bir biçimde yanıt ver, kullandığın dil resmi bir dil olsun
10. Dönemler arası karşılaştırma istendiğinde, değerleri tablo veya liste halinde sun

ÇIKTI FORMATI:
CEVAP: [Detaylı ve kesin yanıt]
KAYNAKLAR: 
  - [Dosya: dosya_adı]
  - [Bölüm: başlık]
  - [Güven: Yüksek/Orta/Düşük]

UYARI: SADECE kontekstte soruyla hiç ilgili veri bulunmadığında "Bilgi mevcut değil" yaz. Eğer kontekstte herhangi bir sayısal veri veya dönem bilgisi varsa, onu kullanarak mutlaka cevap ver."""

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        document_repository: IDocumentRepository,
        llm_service: ILLMService
    ):
        """
        Initialize Advanced RAG use case with dependencies
        
        Args:
            embedding_service: IEmbeddingService implementation
            document_repository: IDocumentRepository implementation
            llm_service: ILLMService implementation
        """
        self.embedding_service = embedding_service
        self.document_repository = document_repository
        self.llm_service = llm_service

    async def execute(self, query: RAGQuery) -> RAGResponse:
        """
        Execute advanced RAG query pipeline with source tracking
        
        Args:
            query: RAGQuery nesnesi (sorgu, top_k, temperature, user_id)
        
        Returns:
            RAGResponse: Detaylı cevap + kaynak atıflaması
        """
        
        try:
            logger.info(f"🔍 Processing query: {query.query[:50]}... [User: {query.user_id}]")

            # Step 1: Sorguyu embedding'e çevir
            logger.info("📊 Embedding query...")
            query_embedding = await self.embedding_service.embed_text(query.query)
            
            if not query_embedding:
                raise Exception("Embedding oluşturulamadı")

            # Step 2: Benzer dokümantasyonu ara (semantic search)
            logger.info(f"🔎 Searching similar documents (top_k={query.top_k})...")
            search_result = await self.document_repository.search_similar(
                embedding=query_embedding,
                top_k=query.top_k,
                threshold=0.0
            )
            
            # Sonuç yoksa
            if not search_result.documents:
                logger.warning("⚠️ No similar documents found")
                return RAGResponse(
                    question=query.query,
                    answer="Sorgunuzla ilgili döküman bulunamadı.",
                    sources=[],
                    model=await self.llm_service.get_model_name(),
                    timestamp=datetime.utcnow(),
                    user_id=query.user_id
                )

            # Step 3: Kontekst oluştur + Kaynak izle (source tracking)
            logger.info(f"📝 Building context from {len(search_result.documents)} chunks...")
            context_parts = []
            sources_with_metadata: List[SourceWithMetadata] = []
            
            for idx, doc in enumerate(search_result.documents):
                # Metadata'dan başlık bilgisini al
                header = None
                if hasattr(doc, 'metadata') and doc.metadata:
                    header = doc.metadata.get('header')
                
                # DEBUG: İlk chunk'ın içeriğini logla
                if idx == 0:
                    logger.info(f"📋 Top chunk [{doc.filename}] header={header} "
                                f"sim={getattr(doc, 'similarity_score', 0):.3f} "
                                f"len={len(doc.content)} "
                                f"preview={doc.content[:300]}")
                
                # Context'e ekle
                header_text = f" [{header}]" if header else ""
                context_parts.append(
                    f"[Kaynak {idx+1}: {doc.filename}{header_text}]\n{doc.content}"
                )
                
                # Detaylı kaynak bilgisi
                similarity_score = getattr(doc, 'similarity_score', 0)
                content_preview = doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
                
                sources_with_metadata.append(
                    SourceWithMetadata(
                        filename=doc.filename,
                        chunk_index=doc.chunk_index,
                        header=header,
                        similarity_score=similarity_score,
                        content_preview=content_preview,
                        chunk_size=len(doc.content)
                    )
                )
            
            context = "\n\n---\n\n".join(context_parts)

            # Step 4: Prompt oluştur (System prompt + Constraints)
            logger.info("📝 Building prompt with banking compliance constraints...")
            prompt = f"""KONTEXT (Aşağıdaki finansal verileri dikkatlice oku ve soruyu cevapla):
{context}

SORU: {query.query}

YANIT (kesin, kaynaklı ve profesyonel):"""

            # Step 5: LLM'den yanıt al (Temperature=0 - kesin yanıtlar)
            logger.info("🤖 Generating response from LLM (temperature=0)...")
            answer = await self.llm_service.generate_response(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0,  # KESIN YANITLAR (query.temperature'den bağımsız)
                max_tokens=2000
            )

            logger.info(f"✅ Advanced RAG query successful [User: {query.user_id}]")

            # Yanıt döndür (source tracking ile)
            return RAGResponse(
                question=query.query,
                answer=answer.strip(),
                sources=sources_with_metadata,
                model=await self.llm_service.get_model_name(),
                timestamp=datetime.utcnow(),
                user_id=query.user_id
            )

        except Exception as e:
            logger.error(f"❌ Advanced RAG query failed: {str(e)}")
            raise

    async def stream_query(self, query: RAGQuery):
        """
        Execute advanced RAG query with streaming response + source tracking
        Real-time cevap almak için
        
        Yields:
            str: Akış halinde cevap parçaları
        """
        
        try:
            logger.info(f"🔍 Processing stream query: {query.query[:50]}... [User: {query.user_id}]")

            # Step 1: Sorguyu embedding'e çevir
            query_embedding = await self.embedding_service.embed_text(query.query)
            
            if not query_embedding:
                raise Exception("Embedding oluşturulamadı")

            # Step 2: Benzer dokümantasyonu ara
            logger.info(f"🔎 Searching similar documents (top_k={query.top_k})...")
            search_result = await self.document_repository.search_similar(
                embedding=query_embedding,
                top_k=query.top_k,
                threshold=0.0
            )
            
            if not search_result.documents:
                logger.warning("⚠️ No similar documents found for stream")
                yield "Sorgunuzla ilgili döküman bulunamadı."
                return

            # Step 3: Kontekst oluştur + Kaynak izle
            logger.info(f"📝 Building context from {len(search_result.documents)} chunks...")
            context_parts = []
            sources_with_metadata: List[SourceWithMetadata] = []
            
            for idx, doc in enumerate(search_result.documents):
                # Metadata'dan başlık bilgisini al
                header = None
                if hasattr(doc, 'metadata') and doc.metadata:
                    header = doc.metadata.get('header')
                
                # Context'e ekle
                header_text = f" [{header}]" if header else ""
                context_parts.append(
                    f"[Kaynak {idx+1}: {doc.filename}{header_text}]\n{doc.content}"
                )
                
                # Detaylı kaynak bilgisi
                similarity_score = getattr(doc, 'similarity_score', 0)
                content_preview = doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
                
                sources_with_metadata.append(
                    SourceWithMetadata(
                        filename=doc.filename,
                        chunk_index=doc.chunk_index,
                        header=header,
                        similarity_score=similarity_score,
                        content_preview=content_preview,
                        chunk_size=len(doc.content)
                    )
                )
            
            context = "\n\n---\n\n".join(context_parts)

            # Step 4: Prompt oluştur
            prompt = f"""KONTEXT (Aşağıdaki finansal verileri dikkatlice oku ve soruyu cevapla):
{context}

SORU: {query.query}

YANIT (kesin, kaynaklı ve profesyonel):"""

            # Step 5: LLM'den stream yanıt al (Temperature=0)
            logger.info("🤖 Generating stream response from LLM (temperature=0)...")
            
            # Kaynak bilgisini ilk olarak gönder
            yield f"📚 KAYNAKLAR ({len(sources_with_metadata)} chunk):\n"
            for source in sources_with_metadata:
                header_info = f" - {source.header}" if source.header else ""
                yield f"  • {source.filename}{header_info} (Similarity: {source.similarity_score:.2f})\n"
            yield "\n" + "="*70 + "\n\n"
            
            # Stream cevapları gönder
            async for chunk in self.llm_service.stream_response(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0,  # KESIN YANITLAR
                max_tokens=2000
            ):
                yield chunk

            logger.info(f"✅ Advanced stream RAG query successful [User: {query.user_id}]")

        except Exception as e:
            logger.error(f"❌ Stream RAG query failed: {str(e)}")
            raise
