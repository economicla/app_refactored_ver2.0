"""
RAGQueryUseCase - Advanced Soru Sorma İşlem Hattı
Business Logic - Framework bağımsız
Advanced preprocessing, intelligent chunking, source tracking ve compliance
"""

import logging
import re
from typing import Optional, List, Dict, Tuple
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

DOKÜMAN TÜRLERİ VE YÖNLENDİRME:
Kontekstteki her chunk [Doküman Türü: X] etiketi taşır. Soruyu yanıtlarken en uygun doküman türünden yararlan:
- "Teklif Özeti": Albaraka'nın kendi kredi teklifi — risk tablosu, limit bilgileri, teminat koşulları, rating, kefiller, ortaklık yapısı. Risk, limit, teminat gibi GENEL sorularda önce bunu kullan.
- "İstihbarat Raporu": Grubun DİĞER bankalardaki risk durumları, harici limitler, harici teminatlar. "Farklı/diğer bankalardaki" ifadesi geçen sorularda bunu kullan.
- "Performans Değerlendirmesi": İş hacmi, karlılık, verimlilik göstergeleri.
- "Mali Veri Tabloları": Bilanço, gelir tablosu, mali oranlar, net satışlar, aktif toplamı, özkaynaklar.
- "Genel Doküman": Yukarıdakilere uymayan dokümanlar.

Birden fazla doküman türünden chunk varsa, sorunun amacına en uygun türü tercih et ve kaynağını belirt.

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
11. Soruda geçen dönem kontekstte yoksa, "Bilgi mevcut değil" YAZMA, bunun yerine kontekstteki mevcut dönemleri belirt ve o dönemlerin verilerini sun. Örneğin: "Soruda belirtilen 2024/6 dönemi verilerde bulunmamaktadır. Mevcut dönemler: 2023/12, 2024/12, 2025/6. Bu dönemlere ait veriler şöyledir: ..."

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

    # ================================================================
    # SORGU NİYETİ TESPİTİ VE RE-RANKING
    # ================================================================

    # Anahtar kelime → tercih edilen doküman türü eşlemesi
    # Tuple: (keywords, preferred_doc_type, boost_score)
    QUERY_INTENT_RULES: List[Tuple[List[str], str, float]] = [
        # Diğer bankalar → İstihbarat Raporu
        (["diğer banka", "farklı banka", "harici risk", "harici limit",
          "bankacılık risk", "sektör riski", "istihbarat", "diğer bankalardaki"],
         "İstihbarat Raporu", 0.25),

        # Genel risk, limit, teminat → Teklif Özeti
        (["genel risk", "toplam risk", "grubun riski", "risk tablosu",
          "limit bilgi", "teminat koşul", "rating", "kefil", "ortaklık yapı",
          "teklif", "komite", "şube teklif", "kredi müdürlüğü"],
         "Teklif Özeti", 0.25),

        # Mali veriler → Mali Veri Tabloları
        (["net satış", "aktif toplam", "özkaynak", "bilanço", "bilançe",
          "gelir tablosu", "mali oran", "kısa vadeli", "uzun vadeli",
          "dönen varlık", "duran varlık", "brüt kar", "faaliyet karı",
          "esas faaliyet", "amortisman", "finansman gideri"],
         "Mali Veri Tabloları", 0.20),

        # Performans → Performans Değerlendirmesi
        (["performans", "iş hacmi", "karlılık", "verimlilik", "büyüme oranı"],
         "Performans Değerlendirmesi", 0.20),
    ]

    def _detect_query_intent(self, query: str) -> Optional[str]:
        """
        Sorgu metninden tercih edilen doküman türünü belirle.
        Returns: Tercih edilen doküman türü veya None
        """
        q_lower = query.lower()
        for keywords, doc_type, _ in self.QUERY_INTENT_RULES:
            for kw in keywords:
                if kw in q_lower:
                    logger.info(f"🎯 Sorgu niyeti tespit edildi: '{kw}' → {doc_type}")
                    return doc_type
        return None

    def _get_boost_for_type(self, doc_type: str) -> float:
        """Doküman türü için boost değerini getir."""
        for _, dtype, boost in self.QUERY_INTENT_RULES:
            if dtype == doc_type:
                return boost
        return 0.0

    def _rerank_chunks(self, documents: list, query: str, final_k: int) -> list:
        """
        Sorgu niyetine göre chunk'ları yeniden sırala.
        
        Strateji:
        1. Sorgu niyetini tespit et (hangi doküman türü tercih edilmeli)
        2. Tercih edilen türdeki chunk'ların skorunu boost et
        3. Çeşitlilik sağla: her doküman türünden en az 1 chunk al
        4. Final top_k kadar döndür
        """
        if not documents:
            return documents

        preferred_type = self._detect_query_intent(query)
        
        if not preferred_type:
            # Niyet tespit edilemedi → orijinal sıralama
            logger.info("🔀 Sorgu niyeti tespit edilemedi, orijinal sıralama korunuyor")
            return documents[:final_k]

        boost = self._get_boost_for_type(preferred_type)

        # Her chunk'ın doküman türünü metadata veya content'ten belirle
        scored_docs = []
        for doc in documents:
            sim = getattr(doc, 'similarity_score', 0)
            doc_type = ""
            
            # 1. Metadata'dan
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_type = doc.metadata.get('doc_type', '')
            
            # 2. Content prefix'inden  [Doküman Türü: X]
            if not doc_type and doc.content:
                m = re.search(r'Doküman Türü:\s*([^\]\n]+)', doc.content)
                if m:
                    doc_type = m.group(1).strip()

            # Boost uygula
            boosted_sim = sim + boost if doc_type == preferred_type else sim
            scored_docs.append((boosted_sim, doc_type, doc))

        # Boost'lu skora göre sırala
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Çeşitlilik: tercih edilen türden en az yarısı, diğerlerinden de temsil
        preferred_chunks = [x for x in scored_docs if x[1] == preferred_type]
        other_chunks = [x for x in scored_docs if x[1] != preferred_type]

        # Tercih edilen türden en fazla ceil(final_k * 0.6), kalanı diğerlerinden
        max_preferred = max(1, int(final_k * 0.6))
        result = preferred_chunks[:max_preferred]
        remaining = final_k - len(result)
        result.extend(other_chunks[:remaining])

        # Eğer hala yetmiyorsa, tercih edilenden kalan ekle
        if len(result) < final_k:
            result.extend(preferred_chunks[max_preferred:final_k - len(result)])

        # Boost'lu skora göre son sıralama
        result.sort(key=lambda x: x[0], reverse=True)

        final_docs = [x[2] for x in result[:final_k]]
        
        # Loglama
        type_counts = {}
        for _, dt, _ in result[:final_k]:
            type_counts[dt] = type_counts.get(dt, 0) + 1
        logger.info(f"🔀 Re-ranking: preferred={preferred_type}, boost={boost}, "
                     f"distribution={type_counts}")

        return final_docs

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

            # Step 2: Benzer dokümantasyonu ara (geniş aday havuzu)
            candidate_k = query.top_k * 4  # 4x fazla aday çek
            logger.info(f"🔎 Searching similar documents (candidate_k={candidate_k}, final_k={query.top_k})...")
            search_result = await self.document_repository.search_similar(
                embedding=query_embedding,
                top_k=candidate_k,
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

            # Step 2.5: Akıllı re-ranking (sorgu niyetine göre)
            reranked_docs = self._rerank_chunks(
                search_result.documents, query.query, query.top_k
            )

            # Step 3: Kontekst oluştur + Kaynak izle (source tracking)
            logger.info(f"📝 Building context from {len(reranked_docs)} chunks...")
            context_parts = []
            sources_with_metadata: List[SourceWithMetadata] = []
            
            for idx, doc in enumerate(reranked_docs):
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

            # Step 2: Benzer dokümantasyonu ara (geniş aday havuzu)
            candidate_k = query.top_k * 4
            logger.info(f"🔎 Searching similar documents (candidate_k={candidate_k}, final_k={query.top_k})...")
            search_result = await self.document_repository.search_similar(
                embedding=query_embedding,
                top_k=candidate_k,
                threshold=0.0
            )
            
            if not search_result.documents:
                logger.warning("⚠️ No similar documents found for stream")
                yield "Sorgunuzla ilgili döküman bulunamadı."
                return

            # Step 2.5: Akıllı re-ranking
            reranked_docs = self._rerank_chunks(
                search_result.documents, query.query, query.top_k
            )

            # Step 3: Kontekst oluştur + Kaynak izle
            logger.info(f"📝 Building context from {len(reranked_docs)} chunks...")
            context_parts = []
            sources_with_metadata: List[SourceWithMetadata] = []
            
            for idx, doc in enumerate(reranked_docs):
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
