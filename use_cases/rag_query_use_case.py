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
          "bankacılık risk", "sektör riski", "istihbarat", "diğer bankalardaki",
          "dış risk", "harici teminat", "diğer bankalar"],
         "İstihbarat Raporu", 0.30),

        # Genel risk, limit, teminat → Teklif Özeti (öncelikli, genel risk sorguları)
        (["genel risk", "toplam risk", "grubun riski", "grubun risk",
          "risk tablosu", "risk nedir", "riski nedir", "risk durumu",
          "mevcut risk", "güncel risk", "risk yapısı", "kredi riski",
          "limit bilgi", "limit nedir", "teminat koşul", "teminat nedir",
          "rating", "kefil", "ortaklık yapı",
          "teklif", "komite", "şube teklif", "kredi müdürlüğü"],
         "Teklif Özeti", 0.35),

        # Mali veriler → Mali Veri Tabloları
        (["net satış", "aktif toplam", "özkaynak", "bilanço", "bilançe",
          "gelir tablosu", "mali oran", "kısa vadeli", "uzun vadeli",
          "dönen varlık", "duran varlık", "brüt kar", "faaliyet karı",
          "esas faaliyet", "amortisman", "finansman gideri",
          "mali veri", "finansal tablo"],
         "Mali Veri Tabloları", 0.25),

        # Performans → Performans Değerlendirmesi
        (["performans", "iş hacmi", "karlılık", "verimlilik", "büyüme oranı"],
         "Performans Değerlendirmesi", 0.25),
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

    # Rekabet eden doküman türleri — tercih edilen tür seçildiğinde,
    # rakip türlere negatif boost (ceza) uygulanır
    COMPETING_TYPES: Dict[str, List[str]] = {
        "Teklif Özeti": ["İstihbarat Raporu"],
        "İstihbarat Raporu": ["Teklif Özeti"],
    }

    # ================================================================
    # QUERY ENHANCEMENT — Sorgu Genişletme ve Bölüm Hedefleme
    # ================================================================
    # Sorgu anahtar kelimeleri → hedef bölüm başlıkları + genişletme terimleri
    # Tuple: (keywords, target_section_headers, expansion_terms, section_boost)
    QUERY_SECTION_RULES: List[Tuple[List[str], List[str], List[str], float]] = [
        # Risk/limit sorguları → Limit Bilgileri / Risk Tablosu
        (["genel risk", "toplam risk", "grubun riski", "grubun risk",
          "risk tablosu", "risk nedir", "riski nedir", "risk durumu",
          "mevcut risk", "güncel risk", "nakdi risk", "gayrinakdi risk",
          "limit bilgi", "limit nedir", "toplam limit", "kredi riski"],
         ["LİMİT BİLGİLERİ", "GRUP RİSK TABLOSU", "RİSK TABLOSU", "LIMIT"],
         ["limit bilgileri", "nakdi risk", "gayrinakdi risk", "toplam limit",
          "mevcut limit", "risk tablosu", "umumi", "firma risk", "grup risk"],
         0.25),

        # Teminat → Teminat Koşulları
        (["teminat koşul", "teminat nedir", "teminat bilgi", "ipotek", "rehin",
          "teminat durumu", "teminat yapısı"],
         ["TEMİNAT KOŞULLARI", "TEMİNAT"],
         ["teminat koşulları", "ipotek", "rehin", "teminat tutarı", "teminat oranı"],
         0.25),

        # Rating → Rating Değerleri
        (["rating", "kredi notu", "derecelendirme", "nota sahip"],
         ["RATING DEĞERLERİ", "RATING"],
         ["rating değeri", "kredi notu", "derecelendirme", "rating skoru"],
         0.25),

        # Kefiller → Kefiller bölümü
        (["kefil", "kefalet", "müşterek borçlu", "kefaleti"],
         ["KEFİLLER", "KEFALET"],
         ["kefil bilgileri", "kefalet", "müşterek borçlu", "müteselsil kefil"],
         0.25),

        # Ortaklık yapısı → Genel Değerlendirme
        (["ortaklık", "ortak yapı", "hisse", "pay oranı", "şirket yapısı"],
         ["GENEL DEĞERLENDİRME VE ORTAKLIK YAPISI", "ORTAKLIK"],
         ["ortaklık yapısı", "hisse oranı", "pay sahipleri", "ortak bilgileri"],
         0.25),

        # Karlılık / Performans oranları
        (["karlılık", "kar marjı", "brüt kar", "net kar", "favök",
          "ROA", "ROE", "kârlılık"],
         ["Karlılık Oranları", "Profitability Ratios", "KARLILIK"],
         ["brüt kar marjı", "net kar marjı", "ROA", "ROE", "FAVÖK marjı"],
         0.25),

        # Bilanço verileri
        (["aktif toplam", "özkaynak", "bilanço", "borç yapı",
          "yabancı kaynak", "kısa vadeli", "uzun vadeli"],
         ["Bilanço", "BALANCE SHEET", "BİLANÇO", "AKTİF"],
         ["aktif toplamı", "özkaynak", "kısa vadeli borç", "uzun vadeli borç",
          "dönen varlık", "duran varlık", "yabancı kaynak"],
         0.25),

        # Gelir tablosu
        (["net satış", "gelir tablosu", "faaliyet karı", "esas faaliyet",
          "amortisman", "finansman gideri", "satış geliri"],
         ["Gelir Tablosu", "INCOME STATEMENT", "GELİR TABLOSU"],
         ["net satışlar", "faaliyet karı", "esas faaliyet geliri", "amortisman",
          "finansman gideri", "brüt kar"],
         0.25),

        # Şube teklifi / komite görüşü
        (["şube teklif", "komite görüş", "kredi müdürlüğü", "müdürlük görüş"],
         ["ŞUBE TEKLİFİ VE KREDİ MÜDÜRLÜĞÜ GÖRÜŞÜ", "ŞUBE TEKLİFİ", "KOMİTE"],
         ["şube teklifi", "kredi müdürlüğü görüşü", "komite kararı"],
         0.25),
    ]

    def _detect_target_sections(self, query: str) -> Tuple[List[str], List[str], float]:
        """
        Sorgudan hedef bölüm başlıklarını ve genişletme terimlerini tespit et.
        
        Returns:
            (target_headers, expansion_terms, section_boost)
        """
        q_lower = query.lower()
        all_headers: List[str] = []
        all_terms: List[str] = []
        max_boost = 0.0

        for keywords, headers, terms, boost in self.QUERY_SECTION_RULES:
            for kw in keywords:
                if kw in q_lower:
                    all_headers.extend(headers)
                    all_terms.extend(terms)
                    max_boost = max(max_boost, boost)
                    logger.info(f"🎯 Bölüm hedefi tespit: '{kw}' → {headers[0]}")
                    break  # Bu kural eşleşti, sonraki kurala geç

        return (list(set(all_headers)), list(set(all_terms)), max_boost)

    def _enhance_query(self, query: str) -> str:
        """
        Sorguyu hedef bölüm terimleriyle genişlet.
        Embedding aramasının doğru bölümleri yakalamasını sağlar.
        
        Örnek:
          'grubun genel riski nedir?' →
          'grubun genel riski nedir? (limit bilgileri, nakdi risk, gayrinakdi risk, ...)
        """
        _, expansion_terms, _ = self._detect_target_sections(query)
        if not expansion_terms:
            return query

        # Orijinal sorgu + genişletme terimleri (parantez içinde, embedding'i yönlendirir)
        enhanced = f"{query} ({', '.join(expansion_terms)})"
        logger.info(f"📝 Sorgu genişletildi: '{query}' → '{enhanced}'")
        return enhanced

    def _get_chunk_header(self, doc) -> str:
        """
        Chunk'ın bölüm başlığını metadata veya content'ten çıkar.
        """
        header = ""
        # 1. Metadata'dan
        if hasattr(doc, 'metadata') and doc.metadata:
            header = doc.metadata.get('header', '')
        
        # 2. Content'ten markdown header (## veya ### ile başlayan ilk satır)
        if not header and doc.content:
            m = re.match(r'^#{1,4}\s+(.+)', doc.content.strip())
            if m:
                header = m.group(1).strip()
        
        return header

    def _calc_section_boost(self, chunk_header: str, target_headers: List[str],
                            section_boost: float) -> float:
        """
        Chunk'ın başlığı hedef bölümlerden biriyle eşleşiyorsa boost döndür.
        Kısmi eşleşme de kabul edilir (alt bölüm hiyerarşisi için).
        """
        if not chunk_header or not target_headers:
            return 0.0

        h_upper = chunk_header.upper()
        for th in target_headers:
            if th.upper() in h_upper:
                return section_boost
        return 0.0

    @staticmethod
    def _detect_type_from_filename(filename: str) -> str:
        """Dosya adından doküman türünü tespit et (son fallback)."""
        fname = filename.lower()
        if "istihbarat" in fname:
            return "İstihbarat Raporu"
        if "teklif" in fname:
            return "Teklif Özeti"
        if any(kw in fname for kw in ("performans", "değerlendirme")):
            return "Performans Değerlendirmesi"
        if any(kw in fname for kw in ("mali", "solo", "konsolide", "bilanco", "bilanço")):
            return "Mali Veri Tabloları"
        return ""

    def _rerank_chunks(self, documents: list, query: str, final_k: int) -> list:
        """
        Sorgu niyetine göre chunk'ları yeniden sırala.
        
        Strateji:
        1. Sorgu niyetini tespit et (hangi doküman türü tercih edilmeli)
        2. Tercih edilen türdeki chunk'ların skorunu boost et
        3. Rakip doküman türüne ceza (penalty) uygula
        4. Bölüm başlığı hedef bölümle eşleşiyorsa ek boost (section boost)
        5. Çeşitlilik: tercih edilen türden %75, diğerlerinden %25
        6. Final top_k kadar döndür
        """
        if not documents:
            return documents

        preferred_type = self._detect_query_intent(query)
        
        if not preferred_type:
            # Niyet tespit edilemedi → sadece section boost dene
            target_headers, _, sec_boost = self._detect_target_sections(query)
            if not target_headers:
                logger.info("🔀 Sorgu niyeti tespit edilemedi, orijinal sıralama korunuyor")
                return documents[:final_k]
            # Section boost varsa uygula (doc_type boost olmadan)
            scored_docs = []
            for doc in documents:
                sim = getattr(doc, 'similarity_score', 0)
                chunk_header = self._get_chunk_header(doc)
                h_boost = self._calc_section_boost(chunk_header, target_headers, sec_boost)
                scored_docs.append((sim + h_boost, "", chunk_header, doc))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            final_docs = [x[3] for x in scored_docs[:final_k]]
            logger.info(f"🔀 Section-only re-ranking: headers={target_headers}, boost={sec_boost}")
            return final_docs

        boost = self._get_boost_for_type(preferred_type)
        competing_types = self.COMPETING_TYPES.get(preferred_type, [])
        penalty = boost * 0.5  # Rakip tür cezası (boost'un yarısı)

        # Section header hedeflerini al
        target_headers, _, sec_boost = self._detect_target_sections(query)

        # Her chunk'ın doküman türünü ve bölüm başlığını belirle
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

            # 3. Dosya adından (son fallback)
            if not doc_type and doc.filename:
                doc_type = self._detect_type_from_filename(doc.filename)

            # === Doküman türü Boost / Penalty ===
            if doc_type == preferred_type:
                type_adj = boost
            elif doc_type in competing_types:
                type_adj = -penalty
            else:
                type_adj = 0.0

            # === Bölüm başlığı Boost ===
            chunk_header = self._get_chunk_header(doc)
            header_adj = self._calc_section_boost(chunk_header, target_headers, sec_boost)

            boosted_sim = sim + type_adj + header_adj

            scored_docs.append((boosted_sim, doc_type, chunk_header, doc))

        # Boost'lu skora göre sırala
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Çeşitlilik: tercih edilen türden %75, diğerlerinden %25
        preferred_chunks = [x for x in scored_docs if x[1] == preferred_type]
        other_chunks = [x for x in scored_docs if x[1] != preferred_type]

        max_preferred = max(1, int(final_k * 0.75))
        result = preferred_chunks[:max_preferred]
        remaining = final_k - len(result)
        result.extend(other_chunks[:remaining])

        # Eğer hala yetmiyorsa, tercih edilenden kalan ekle
        if len(result) < final_k:
            result.extend(preferred_chunks[max_preferred:final_k - len(result)])

        # Boost'lu skora göre son sıralama
        result.sort(key=lambda x: x[0], reverse=True)

        final_docs = [x[3] for x in result[:final_k]]
        
        # Detaylı loglama
        type_counts: Dict[str, int] = {}
        header_counts: Dict[str, int] = {}
        for _, dt, hdr, _ in result[:final_k]:
            type_counts[dt] = type_counts.get(dt, 0) + 1
            if hdr:
                hdr_short = hdr[:40]
                header_counts[hdr_short] = header_counts.get(hdr_short, 0) + 1
        logger.info(f"🔀 Re-ranking: preferred={preferred_type}, type_boost={boost}, "
                     f"penalty={penalty}, section_boost={sec_boost}, "
                     f"target_headers={target_headers}")
        logger.info(f"🔀 Distribution: types={type_counts}, headers={header_counts}")

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

            # Step 1: Sorguyu genişlet ve embedding'e çevir
            enhanced_query = self._enhance_query(query.query)
            logger.info(f"📊 Embedding query (enhanced={enhanced_query != query.query})...")
            query_embedding = await self.embedding_service.embed_text(enhanced_query)
            
            if not query_embedding:
                raise Exception("Embedding oluşturulamadı")

            # Step 2: Benzer dokümantasyonu ara (geniş aday havuzu)
            candidate_k = query.top_k * 6  # 6x fazla aday çek (daha geniş aday havuzu)
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

            # Step 1: Sorguyu genişlet ve embedding'e çevir
            enhanced_query = self._enhance_query(query.query)
            logger.info(f"📊 Embedding query (enhanced={enhanced_query != query.query})...")
            query_embedding = await self.embedding_service.embed_text(enhanced_query)
            
            if not query_embedding:
                raise Exception("Embedding oluşturulamadı")

            # Step 2: Benzer dokümantasyonu ara (geniş aday havuzu)
            candidate_k = query.top_k * 6
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
