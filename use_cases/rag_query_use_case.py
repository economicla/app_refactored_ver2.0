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

BANKA İSTİHBARATI VE LİMİT RİSK TABLOLARI:
- "Banka istihbaratına göre", "diğer bankalarda limit/risk" gibi sorularda kontekstteki TÜM kaynakları (Kaynak 1, 2, 3...) tara. Sadece "Banka: ... | Firma: ... | Genel Limit: [sayı] | Nakit Risk: [sayı]" biçiminde tam sayısal limit ve risk verisi olan satırları cevap tablosuna ekle. Limit veya risk sayısı olmayan, Özet İstihbarat veya başka bölümlerdeki eksik satırları tabloya ekleme. Aynı banka farklı firmada (Kuveyt Türk–AKTÜL KAĞIT, Kuveyt Türk–MKS MARMARA, Türkiye Finans–AKTÜL KAĞIT, Türkiye Finans–MKS MARMARA) tam veriyle geçiyorsa hepsini ayrı satırda ver. Banka adlarını K1, K2 ile değiştirme.
- "Limit Risk Bilgileri (Bin TL)" veya "Kaynak Bazında Detay" tablolarından veri kullanacaksan: Ya tablonun TAMAMINI başlıkla ver, ya da hiç alma. Kısmen gösterme.
- Cevabında "**Banka İstihbaratı bölümünden:**" altında sadece tam limit/risk verisi olan banka listesi; diğer bölümleri ayrı başlıkla belirt.

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
    # PRIORITY-ORDERED ROUTING RULES
    # Evaluated in list order (highest priority first); first match wins.
    # This guarantees deterministic, explainable routing.
    # ================================================================

    ROUTING_RULES: List[Dict] = [
        # P100: External / specific bank signal → İstihbarat Raporu
        # MUST be evaluated BEFORE risk/limit/teminat rules to prevent
        # "Emlak Bankası'ndaki teminat şartları" from routing to Teklif Özeti
        {
            "rule_id": "EXTERNAL_BANK",
            "rule_name": "Harici/Spesifik Banka Sinyali",
            "priority": 100,
            "doc_type": "İstihbarat Raporu",
            "keywords": [
                # Named/specific bank suffixes (Turkish morphology)
                "bankası", "bankasındaki", "bankasında", "bankasından", "bankasına",
                "bankası'ndaki", "bankası'nda", "bankası'ndan",
                # General external bank context
                "bankadaki", "bankada", "bankadan",
                "banka nezdinde", "nezdindeki", "nezdinde",
                "banka bazında", "bankalara göre", "bankalar nezdinde",
                # Existing İstihbarat keywords
                "diğer banka", "diğer bankalar", "diğer bankalardaki",
                "farklı banka", "farklı bankalardaki",
                "harici risk", "harici limit", "harici teminat",
                "dış risk", "istihbarat",
                "bankacılık risk", "sektör riski",
            ],
            "patterns": [
                r'\w+\s+bankası',
            ],
            "boost_score": 0.30,
        },
        # P50: General risk / limit / teminat → Teklif Özeti
        {
            "rule_id": "TEKLIF_OZETI",
            "rule_name": "Genel Risk/Limit/Teminat",
            "priority": 50,
            "doc_type": "Teklif Özeti",
            "keywords": [
                "genel risk", "toplam risk", "grubun riski", "grubun risk",
                "nakdi risk", "nakdi riski", "nakit risk", "nakit riski",
                "gayrinakdi risk", "gayrinakdi riski", "gayri nakdi risk", "gayri nakdi riski",
                "gayrinakdi", "gayri nakdi",
                "nakdi limit", "nakdi limiti", "gayrinakdi limit", "gayrinakdi limiti",
                "grubun nakdi", "grubun gayrinakdi", "nakdi ve gayrinakdi",
                "risk tablosu", "risk nedir", "riski nedir", "risk durumu",
                "riski ne kadar", "risk ne kadar", "ne kadar risk",
                "mevcut risk", "güncel risk", "risk yapısı", "kredi riski",
                "limit bilgi", "limit nedir", "limiti nedir", "limiti ne kadar",
                "limit türü", "limit türleri", "limit yapısı",
                "teminat koşul", "teminat koşulları", "teminat şart", "teminat şartları",
                "teminat nedir", "teminat yapısı", "teminata bağlı",
                "tahsis edil", "tahsis edilecek", "tahsis edilen",
                "önerilen limit", "önerilen risk", "önerilen teminat",
                "teklif edilen", "teklif edilen limit", "teklif edilen risk",
                "rating", "kefil", "kefalet", "kefiller", "ortaklık yapı",
                "teklif", "komite", "şube teklif", "kredi müdürlüğü",
                "teklif özet", "kredi teklif", "kredi tahsis",
            ],
            "patterns": [],
            "boost_score": 0.35,
        },
        # P40: Mali veriler → Mali Veri Tabloları
        {
            "rule_id": "MALI_VERI",
            "rule_name": "Mali Veriler",
            "priority": 40,
            "doc_type": "Mali Veri Tabloları",
            "keywords": [
                "net satış", "aktif toplam", "özkaynak", "bilanço", "bilançe",
                "gelir tablosu", "mali oran", "kısa vadeli", "uzun vadeli",
                "dönen varlık", "duran varlık", "brüt kar", "faaliyet karı",
                "esas faaliyet", "amortisman", "finansman gideri",
                "mali veri", "finansal tablo",
            ],
            "patterns": [],
            "boost_score": 0.25,
        },
        # P40: Performans → Performans Değerlendirmesi
        {
            "rule_id": "PERFORMANS",
            "rule_name": "Performans Değerlendirmesi",
            "priority": 40,
            "doc_type": "Performans Değerlendirmesi",
            "keywords": [
                "performans", "iş hacmi", "karlılık", "verimlilik", "büyüme oranı",
            ],
            "patterns": [],
            "boost_score": 0.25,
        },
    ]

    def _detect_query_intent(self, query: str) -> Optional[Dict]:
        """
        Priority-ordered deterministic intent detection.
        Rules are pre-sorted by priority (desc); first match wins.

        Returns:
            Dict with doc_type, matched_rule_id, matched_rule_name,
            matched_keywords, rule_priority, boost_score.
            None if no rule matches.
        """
        q_lower = query.lower()

        for rule in self.ROUTING_RULES:
            matched = []

            for kw in rule["keywords"]:
                if kw in q_lower:
                    matched.append(kw)

            for pat in rule.get("patterns", []):
                m = re.search(pat, q_lower)
                if m:
                    matched.append(f"pattern:{m.group()}")

            if matched:
                result = {
                    "doc_type": rule["doc_type"],
                    "matched_rule_id": rule["rule_id"],
                    "matched_rule_name": rule["rule_name"],
                    "matched_keywords": matched,
                    "rule_priority": rule["priority"],
                    "boost_score": rule["boost_score"],
                }
                logger.info(
                    f"🎯 Routing: rule={rule['rule_id']} (P{rule['priority']}) → "
                    f"{rule['doc_type']} | matched={matched[:5]}"
                )
                return result

        return None

    def _get_boost_for_type(self, doc_type: str) -> float:
        """Doküman türü için boost değerini getir."""
        for rule in self.ROUTING_RULES:
            if rule["doc_type"] == doc_type:
                return rule["boost_score"]
        return 0.0

    # Rekabet eden doküman türleri — tercih edilen tür seçildiğinde,
    # rakip türlere negatif boost (ceza) uygulanır
    COMPETING_TYPES: Dict[str, List[str]] = {
        "Teklif Özeti": ["İstihbarat Raporu"],
        "İstihbarat Raporu": ["Teklif Özeti"],
    }

    # ================================================================
    # GENEL İÇERİK ANALİZİ — Otomatik, kural gerektirmeyen sinyaller
    # ================================================================

    @staticmethod
    def _query_wants_numbers(query_lower: str) -> bool:
        """
        Sorgunun sayısal/tablo veri isteyip istemediğini otomatik tespit eder.
        Hardcoded bölüm kuralı gerektirmez — sorgu dilinden çıkarım yapar.
        """
        indicators = [
            "kaç", "ne kadar", "tutarı", "toplam", "miktarı", "değeri",
            "oranı", "yüzdesi", "%", "rakam", "sayısal",
            "nedir", "ne dir",
            "risk", "limit", "teminat", "kredi", "borç",
            "aktif", "pasif", "özkaynak", "satış", "kar", "zarar",
            "bilanço", "gelir tablosu", "mevcut", "güncel",
        ]
        return any(ind in query_lower for ind in indicators)

    @staticmethod
    def _calc_numericity(content: str) -> float:
        """
        İçerikteki sayısal yoğunluğu hesapla (0.0 - 1.0).
        Sayısal veri ağırlıklı chunk'lar daha yüksek skor alır.
        """
        if not content:
            return 0.0
        numbers = re.findall(r'\d[\d.,]+', content)
        total_chars = len(content)
        num_chars = sum(len(n) for n in numbers)
        return min(1.0, (num_chars / total_chars) * 10)

    @staticmethod
    def _has_tabular_data(content: str) -> bool:
        """
        İçerikte tablo/yapısal veri var mı?
        Para birimleri, büyük sayılar, key:value çiftleri.
        """
        currency_data = re.findall(r'\d[\d.,]+\s*(TRY|USD|EUR|TL)', content)
        big_numbers = re.findall(r'\d{1,3}(?:[.,]\d{3})+', content)
        kv_pairs = re.findall(r'\w+\s*:\s*\d', content)
        return len(currency_data) >= 2 or len(big_numbers) >= 3 or len(kv_pairs) >= 3

    def _get_chunk_header(self, doc) -> str:
        """Chunk'ın bölüm başlığını metadata veya content'ten çıkar."""
        header = ""
        if hasattr(doc, 'metadata') and doc.metadata:
            header = doc.metadata.get('header', '')
        if not header and doc.content:
            m = re.match(r'^#{1,4}\s+(.+)', doc.content.strip())
            if m:
                header = m.group(1).strip()
        return header

    def _score_chunk_relevance(self, doc, query: str) -> float:
        """
        Genel içerik uygunluk skoru — hardcoded kural kullanmaz.

        3 otomatik sinyal:
        1. Sayısallık: Sorgu sayı istiyorsa, sayısal chunk'lar tercih edilir
        2. Tablo verisi: Yapısal veri içeren chunk'lara ek puan
        3. Sorgu-içerik kelime örtüşmesi: Sorgu kelimeleri içerikte geçiyorsa ek puan

        Toplam max boost: ~0.35
        """
        score = 0.0
        content = doc.content
        q_lower = query.lower()

        # 1. Sayısallık sinyali
        wants_nums = self._query_wants_numbers(q_lower)
        if wants_nums:
            num_density = self._calc_numericity(content)
            score += num_density * 0.15  # max +0.15

        # 2. Tablo verisi sinyali
        if wants_nums and self._has_tabular_data(content):
            score += 0.10  # +0.10

        # 3. Kelime örtüşmesi
        query_tokens = set(re.findall(r'[a-züöçşığA-ZÜÖÇŞİĞ]{3,}', q_lower))
        if query_tokens:
            content_lower = content.lower()
            overlap = sum(1 for t in query_tokens if t in content_lower)
            score += (overlap / len(query_tokens)) * 0.10  # max +0.10

        return score

    # ================================================================
    # SÖZLÜK DESTEKLİ SORGU GENİŞLETME (Veri Sözlüğü dokümanları ile)
    # ================================================================

    async def _enhance_query_with_dictionary(
        self, query: str, query_embedding: List[float]
    ) -> Tuple[str, List[str]]:
        """
        Veri sözlüğü chunk'larını arayarak sorguyu OTOMATİK genişlet.
        Sözlük dokümanları yoksa sessizce atlar — sistem normal çalışır.

        Returns:
            (enhanced_query, target_section_headers)
        """
        try:
            dict_result = await self.document_repository.search_dictionary(
                embedding=query_embedding,
                top_k=3
            )

            if not dict_result.documents:
                return query, []

            target_headers: List[str] = []
            expansion_terms: List[str] = []

            for doc in dict_result.documents:
                header = self._get_chunk_header(doc)
                if header:
                    target_headers.append(header)

                # İçerikten anahtar terimleri çıkar
                snippet = doc.content[:500]
                words = re.findall(r'[A-ZÜÖÇŞİĞa-züöçşığ]{4,}', snippet)
                # En sık geçen terimleri al (Counter olmadan)
                freq: Dict[str, int] = {}
                for w in words:
                    wl = w.lower()
                    freq[wl] = freq.get(wl, 0) + 1
                top = sorted(freq, key=freq.get, reverse=True)[:5]
                expansion_terms.extend(top)

            unique_terms = list(dict.fromkeys(expansion_terms))[:8]
            if unique_terms:
                enhanced = f"{query} ({', '.join(unique_terms)})"
                logger.info(f"📖 Sözlük ile sorgu genişletildi: '{query}' → '{enhanced}'")
                logger.info(f"📖 Hedef bölümler: {target_headers}")
            else:
                enhanced = query

            return enhanced, target_headers

        except Exception as e:
            logger.debug(f"📖 Sözlük araması kullanılamıyor (normal): {e}")
            return query, []

    def _calc_header_similarity(self, chunk_header: str, target_headers: List[str]) -> float:
        """
        Chunk başlığı ile sözlüğün önerdiği hedef bölümler arasındaki benzerlik.
        Tam eşleşme veya kelime örtüşmesi kullanır (embedding gerektirmez).
        """
        if not chunk_header or not target_headers:
            return 0.0

        h_upper = chunk_header.upper()
        best = 0.0
        for th in target_headers:
            th_upper = th.upper()
            # Tam eşleşme
            if th_upper in h_upper or h_upper in th_upper:
                return 0.25
            # Kelime örtüşmesi
            th_words = set(re.findall(r'[A-ZÜÖÇŞİĞa-züöçşığ]{3,}', th_upper))
            h_words = set(re.findall(r'[A-ZÜÖÇŞİĞa-züöçşığ]{3,}', h_upper))
            if th_words and h_words:
                overlap = len(th_words & h_words) / max(len(th_words), len(h_words))
                best = max(best, overlap * 0.25)

        return best

    # Canonical doc_type enum — maps any lowercase/trimmed variant to proper form
    DOC_TYPE_ENUM: Dict[str, str] = {
        "teklif özeti": "Teklif Özeti",
        "teklif ozeti": "Teklif Özeti",
        "istihbarat raporu": "İstihbarat Raporu",
        "istihbarat": "İstihbarat Raporu",
        "performans değerlendirmesi": "Performans Değerlendirmesi",
        "performans degerlendirmesi": "Performans Değerlendirmesi",
        "mali veri tabloları": "Mali Veri Tabloları",
        "mali veri tablolari": "Mali Veri Tabloları",
        "genel doküman": "Genel Doküman",
        "genel dokuman": "Genel Doküman",
    }

    @classmethod
    def _normalize_doc_type(cls, raw: str) -> str:
        """Normalize doc_type: trim + lowercase → canonical enum value."""
        if not raw:
            return "Genel Doküman"
        key = raw.strip().lower()
        return cls.DOC_TYPE_ENUM.get(key, raw.strip())

    def _resolve_doc_type(self, doc) -> str:
        """Chunk'ın doküman türünü metadata, içerik veya dosya adından tespit et ve normalize et."""
        doc_type = ""
        if hasattr(doc, 'metadata') and doc.metadata:
            doc_type = doc.metadata.get('doc_type', '')
        if not doc_type and doc.content:
            m = re.search(r'Doküman Türü:\s*([^\]\n]+)', doc.content)
            if m:
                doc_type = m.group(1).strip()
        if not doc_type and doc.filename:
            doc_type = self._detect_type_from_filename(doc.filename)
        return self._normalize_doc_type(doc_type)

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

    # ================================================================
    # BANK NAME NORMALIZATION (Entity Resolution)
    # Uses İstihbarat Raporu content as the source-of-truth dictionary.
    # No hardcoded bank lists — all names come from the report itself.
    # ================================================================

    _BANK_NAME_STOP_WORDS = frozenset({
        "türkiye", "t.c.", "tc", "cumhuriyeti",
        "bankası", "bankasi", "bank", "bankas",
        "a.ş.", "a.ş", "aş", "a.s.", "as",
        "t.a.ş.", "t.a.ş", "taş",
        "ve",
    })

    _OFFICIAL_BANK_RE = re.compile(
        r'([A-ZÜÖÇŞİĞT]'
        r'[A-ZÜÖÇŞİĞa-züöçşığ\s\.\'\-&]{2,60}?'
        r'(?:[Bb]ankası|[Bb]ankas[ıi]|[Bb]ank)'
        r'(?:\s*(?:A\.?\s?Ş\.?|T\.?\s?A\.?\s?Ş\.?))?)',
        re.UNICODE
    )

    _QUERY_BANK_RE = re.compile(
        r'([A-ZÜÖÇŞİĞa-züöçşığ]+'
        r'(?:\s+[A-ZÜÖÇŞİĞa-züöçşığ]+){0,3}?)'
        r"\s+(?:bankası|bankas[ıi]|bankası')",
        re.IGNORECASE | re.UNICODE
    )

    _BANK_MENTION_NOISE_WORDS = frozenset({
        "grubun", "grup", "firmanın", "firmanin", "firma",
        "şirketin", "sirketin", "şirket", "sirket",
        "müşterinin", "musterinin", "müşteri", "musteri",
        "bizim", "onların", "onlarin", "bu", "şu", "o",
        "tüm", "tum", "bütün", "butun", "herhangi",
        "mevcut", "yeni", "eski", "diğer", "diger",
    })

    @classmethod
    def _tokenize_bank_name(cls, name: str) -> set:
        """Extract meaningful tokens, ignoring stop words and punctuation."""
        raw = re.findall(r'[a-züöçşığA-ZÜÖÇŞİĞ]{2,}', name.lower())
        return {t for t in raw if t not in cls._BANK_NAME_STOP_WORDS}

    _BANKA_SECTION_RE = re.compile(
        r'##\s*Banka\s+[İi]stihbarat[ıi]', re.IGNORECASE
    )
    _BANKA_LINE_RE = re.compile(
        r'^Banka:\s*(.+?)(?:\s*\|)', re.MULTILINE
    )

    def _extract_bank_index_from_chunks(self, chunks: list) -> List[str]:
        """
        Build bank index from the structured extractor's rendered text.

        Looks for the ``## Banka İstihbaratı`` heading inside chunks
        (both in content AND in metadata header), then extracts official
        bank names from ``Banka: <name> |`` lines.
        Falls back to regex scan if no structured section found.
        """
        seen: set = set()
        bank_names: List[str] = []

        for doc in chunks:
            content = getattr(doc, 'content', '')
            header = ''
            if hasattr(doc, 'metadata') and doc.metadata:
                header = (doc.metadata.get('header', '') or '')

            is_banka_section = (
                self._BANKA_SECTION_RE.search(content)
                or 'banka istihbarat' in header.lower().replace('İ', 'i').replace('ı', 'i')
            )
            if not is_banka_section:
                continue
            for m in self._BANKA_LINE_RE.finditer(content):
                raw = m.group(1).strip().rstrip('.')
                if len(raw) > 4:
                    key = raw.lower()
                    if key not in seen:
                        seen.add(key)
                        bank_names.append(raw)

        if not bank_names:
            for doc in chunks:
                content = getattr(doc, 'content', '')
                for m in self._OFFICIAL_BANK_RE.finditer(content):
                    raw = m.group(1).strip().rstrip('.')
                    if len(raw) > 5:
                        key = raw.lower()
                        if key not in seen:
                            seen.add(key)
                            bank_names.append(raw)

        logger.info(f"🏦 Bank index: {len(bank_names)} names from {len(chunks)} chunks")
        for name in bank_names[:10]:
            logger.debug(f"  🏦 {name}")
        return bank_names

    @classmethod
    def _extract_bank_mention(cls, query: str) -> Optional[str]:
        """
        Extract the informal bank name mention from a user query.
        "grubun emlak bankasındaki limit..." → "emlak Bankası"
        Strips noise words (grubun, firmanın, etc.) from the captured prefix.
        """
        m = cls._QUERY_BANK_RE.search(query)
        if m:
            prefix = m.group(1).strip()
            cleaned_parts = [
                w for w in prefix.split()
                if w.lower() not in cls._BANK_MENTION_NOISE_WORDS
            ]
            cleaned = " ".join(cleaned_parts)
            if len(cleaned) >= 2:
                return f"{cleaned} Bankası"
            if len(prefix) >= 2:
                return f"{prefix} Bankası"
        return None

    # ================================================================
    # ENTITY / FIRM NAME EXTRACTION & CHUNK FOCUSING
    # ================================================================

    _ENTITY_STOP_WORDS = frozenset({
        "grubun", "gruptaki", "grup", "grubundaki", "grubu",
        "firmanın", "firmanin", "firma", "firması",
        "şirketin", "sirketin", "şirket", "sirket", "şirketinin",
        "müşterinin", "musterinin", "müşteri", "musteri",
        "nedir", "nelerdir", "neler", "neden", "nasıl", "kaç",
        "kayıtları", "kayıtları", "kayitlari", "kayıtlar",
        "bilgileri", "bilgisi", "bilgi", "durumu", "durumları",
        "limit", "risk", "teminat", "haciz", "kredi", "banka",
        "ne", "bu", "şu", "o", "ve", "ile", "için", "icin",
        "olan", "var", "yok", "mı", "mi", "mu", "mü",
        "bir", "tüm", "tum", "bütün", "butun", "herhangi",
        "mevcut", "yeni", "eski", "diğer", "diger",
        "hangi", "kadar", "gibi", "daha", "çok", "az",
    })

    _TR_SUFFIX_RE = re.compile(
        r"(?:'?(?:nın|nin|nun|nün|ın|in|un|ün|ının|inin|"
        r"daki|deki|taki|teki|ndaki|ndeki|"
        r"dan|den|tan|ten|ndan|nden|"
        r"ları|leri|lar|ler|"
        r"ığın|iğin|uğun|üğün|"
        r"ğın|ğin|ğun|ğün))+$",
        re.IGNORECASE | re.UNICODE,
    )

    @classmethod
    def _extract_entity_keywords(cls, query: str) -> List[str]:
        """
        Extract potential entity/firm name tokens from the query.
        Strips Turkish suffixes and stop words, returns stems >= 3 chars.
        """
        tokens = re.findall(r'[a-züöçşığA-ZÜÖÇŞİĞ]+', query)
        keywords: List[str] = []
        for tok in tokens:
            low = tok.lower()
            if low in cls._ENTITY_STOP_WORDS:
                continue
            stem = cls._TR_SUFFIX_RE.sub('', low)
            if not stem:
                stem = low
            if len(stem) >= 3:
                keywords.append(stem)
        return keywords

    @staticmethod
    def _focus_entity_chunks(
        docs: list,
        entity_keywords: List[str],
        min_entity_matches: int = 1,
    ) -> Tuple[list, Optional[str]]:
        """
        Reorder docs: chunks containing entity keywords come first.
        Returns (reordered_docs, matched_entity_display_name).
        """
        if not entity_keywords:
            return docs, None

        entity_docs = []
        other_docs = []
        matched_name: Optional[str] = None

        for doc in docs:
            content_lower = getattr(doc, 'content', '').lower()
            match_count = sum(1 for kw in entity_keywords if kw in content_lower)
            if match_count >= min_entity_matches:
                entity_docs.append(doc)
                if not matched_name:
                    for kw in entity_keywords:
                        idx = content_lower.find(kw)
                        if idx >= 0:
                            end = content_lower.find('|', idx)
                            if end < 0:
                                end = min(idx + 60, len(content_lower))
                            snippet = getattr(doc, 'content', '')[idx:end].strip()
                            first_line = snippet.split('\n')[0].strip().rstrip('|').strip()
                            if len(first_line) > len(kw):
                                matched_name = first_line
                                break
            else:
                other_docs.append(doc)

        if entity_docs:
            logger.info(
                f"🎯 Entity focus: {len(entity_docs)}/{len(docs)} chunks "
                f"match keywords {entity_keywords}, entity='{matched_name}'"
            )
            return entity_docs + other_docs, matched_name

        return docs, None

    def _normalize_bank_name(
        self,
        mention: str,
        bank_index: List[str]
    ) -> Dict:
        """
        Fuzzy-match a user's bank mention against the official bank index.
        Score = fraction of mention tokens found in official name tokens.
        """
        if not mention or not bank_index:
            return {
                "normalized_bank_name": None,
                "original_bank_mention": mention,
                "match_confidence": "none",
                "match_score": 0.0,
            }

        mention_tokens = self._tokenize_bank_name(mention)
        if not mention_tokens:
            return {
                "normalized_bank_name": None,
                "original_bank_mention": mention,
                "match_confidence": "none",
                "match_score": 0.0,
            }

        best_name: Optional[str] = None
        best_score = 0.0

        for official in bank_index:
            official_tokens = self._tokenize_bank_name(official)
            if not official_tokens:
                continue
            overlap = len(mention_tokens & official_tokens)
            score = overlap / len(mention_tokens)
            if score > best_score:
                best_score = score
                best_name = official

        if best_score >= 0.8:
            confidence = "high"
        elif best_score >= 0.5:
            confidence = "medium"
        elif best_score > 0:
            confidence = "low"
        else:
            confidence = "none"
            best_name = None

        if confidence == "none":
            best_name = None

        result = {
            "normalized_bank_name": best_name,
            "original_bank_mention": mention,
            "match_confidence": confidence,
            "match_score": round(best_score, 2),
        }

        if best_name:
            logger.info(
                f"🏦 Bank normalized: '{mention}' → '{best_name}' "
                f"(confidence={confidence}, score={best_score:.2f})"
            )
        else:
            logger.info(
                f"🏦 Bank not matched: '{mention}' (best_score={best_score:.2f})"
            )
        return result

    def _resolve_bank_entity(
        self,
        query: str,
        chunks: list
    ) -> Optional[Dict]:
        """
        Full bank entity resolution pipeline:
        1. Extract bank mention from query (regex)
        2. If regex fails, reverse-match query tokens against bank index
        3. Build bank index from İstihbarat chunks
        4. Fuzzy match mention → official name

        Returns normalization result dict, or None if no bank mention found.
        """
        mention = self._extract_bank_mention(query)

        bank_index = self._extract_bank_index_from_chunks(chunks)

        if not mention and bank_index:
            mention = self._reverse_match_bank(query, bank_index)
            if mention:
                logger.info(f"🏦 Reverse bank match: '{mention}' from query tokens")

        if not mention:
            return None

        if not bank_index:
            return {
                "normalized_bank_name": None,
                "original_bank_mention": mention,
                "match_confidence": "none",
                "match_score": 0.0,
                "bank_index_size": 0,
            }

        result = self._normalize_bank_name(mention, bank_index)
        result["bank_index_size"] = len(bank_index)
        return result

    def _reverse_match_bank(self, query: str, bank_index: List[str]) -> Optional[str]:
        """
        When regex-based bank extraction fails, check if any bank name
        from the index matches tokens in the query.
        e.g. "kuveytteki" → stem "kuveyt" → matches "Kuveyt Türk Katılım Bankası"
        """
        query_stems = set(self._extract_entity_keywords(query))
        if not query_stems:
            return None

        best_bank: Optional[str] = None
        best_overlap = 0

        for official in bank_index:
            bank_tokens = self._tokenize_bank_name(official)
            if not bank_tokens:
                continue
            overlap = len(query_stems & bank_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_bank = official

        if best_bank and best_overlap >= 1:
            return best_bank

        return None

    # ================================================================
    # BANK-SCOPE GROUNDING GUARDRAIL
    # Prevents hallucination by ensuring answers reference verified bank data.
    # ================================================================

    BANK_NOT_FOUND_ANSWER = (
        "Belirtilen banka, raporun 'Banka istihbaratı' bölümündeki banka listesinde "
        "bulunamadı. Bu nedenle başka tablolara ait sayısal veriler paylaşılmamaktadır.\n\n"
        "Lütfen resmi banka adını paylaşır mısınız? "
        "(Örnek: 'Türkiye Emlak Katılım Bankası A.Ş.')"
    )

    BANK_NO_DATA_ANSWER_TEMPLATE = (
        "'{bank_name}' raporda tanınmaktadır; ancak mevcut bağlam parçalarında "
        "bu bankaya ait spesifik veri bulunamamıştır. "
        "Lütfen sorunuzu daraltarak veya farklı bir ifadeyle tekrar deneyiniz."
    )

    BANK_GUARDRAIL_CONFIDENCE_THRESHOLD = ("medium", "high")

    @classmethod
    def _bank_search_key(cls, official_name: str) -> str:
        """Strip common prefix/suffix to produce a substring search key."""
        key = official_name.strip()
        low = key.lower()
        for p in ("türkiye ", "t.c. "):
            if low.startswith(p):
                key = key[len(p):]
                low = key.lower()
                break
        for s in (" a.ş.", " a.ş", " a.s.", " t.a.ş.", " t.a.ş"):
            if low.endswith(s):
                key = key[:len(key) - len(s)]
                break
        return key.strip().lower()

    def _apply_bank_guardrail(
        self,
        bank_resolution: Dict,
        reranked_docs: list,
        all_candidates: list = None,
    ) -> Dict:
        """
        Bank-scope grounding guardrail — 4-tier scoping strategy.

        Searches ALL candidates (not just reranked top-K) so that bank-specific
        chunks that ranked lower in semantic similarity are still discoverable.

        Tier 1 — ``substring_scoped``:
            Bank name substring found in chunks → use only those chunks.
        Tier 2 — ``fallback_banka_istihbarati_only``:
            Bank name not found, but chunks with "Banka İstihbaratı" header
            exist → use only those (LLM still runs, but with narrower context).
        Tier 3 — ``soft_fallback``:
            Bank was resolved but not found in any chunk text →
            pass through reranked docs with a warning (LLM still runs).
        Tier 4 — ``blocked``:
            Confidence too low (bank not resolved at all) → safe_answer, no LLM.
        """
        norm_name = bank_resolution.get("normalized_bank_name")
        confidence = bank_resolution.get("match_confidence", "none")

        # Gate: confidence too low → blocked (Tier 4)
        if not norm_name or confidence not in self.BANK_GUARDRAIL_CONFIDENCE_THRESHOLD:
            logger.info(
                f"🛡️ Bank guardrail BLOCKED: norm={norm_name}, "
                f"confidence={confidence} (below threshold)"
            )
            return {
                "passed": False,
                "reason": "bank_not_resolved",
                "scoping_strategy": "blocked",
                "safe_answer": self.BANK_NOT_FOUND_ANSWER,
                "bank_scoped_chunks_used": [],
                "scoped_count": 0,
            }

        search_pool = all_candidates if all_candidates else reranked_docs
        search_key = self._bank_search_key(norm_name)

        # Also build alternative search keys from the original mention
        mention = bank_resolution.get("original_bank_mention", "")
        alt_keys: List[str] = []
        if mention:
            mention_tokens = self._tokenize_bank_name(mention)
            for t in mention_tokens:
                if len(t) >= 4:
                    alt_keys.append(t)

        # Tier 1: substring match on the bank name — search ALL candidates
        scoped: list = []
        scoped_info: List[Dict] = []

        for doc in search_pool:
            content_lower = getattr(doc, 'content', '').lower()
            header = ''
            if hasattr(doc, 'metadata') and doc.metadata:
                header = (doc.metadata.get('header', '') or '').lower()

            found = False
            if search_key and search_key in content_lower:
                found = True
            elif search_key and search_key in header:
                found = True
            else:
                for ak in alt_keys:
                    if ak in content_lower and 'banka' in content_lower:
                        found = True
                        break

            if found:
                scoped.append(doc)
                scoped_info.append({
                    "filename": doc.filename,
                    "chunk_index": getattr(doc, 'chunk_index', None),
                })

        if scoped:
            logger.info(
                f"🛡️ Bank guardrail PASSED (substring_scoped): "
                f"'{norm_name}' (key='{search_key}') "
                f"found in {len(scoped)}/{len(search_pool)} chunks"
            )
            return {
                "passed": True,
                "reason": "ok",
                "scoping_strategy": "substring_scoped",
                "safe_answer": None,
                "bank_scoped_chunks_used": scoped_info,
                "scoped_count": len(scoped),
                "scoped_docs": scoped,
            }

        # Tier 2: fall back to chunks with "Banka İstihbaratı" header — search ALL candidates
        banka_section_docs: list = []
        banka_section_info: List[Dict] = []

        for doc in search_pool:
            content = getattr(doc, 'content', '')
            header = ''
            if hasattr(doc, 'metadata') and doc.metadata:
                header = (doc.metadata.get('header', '') or '').lower()

            is_banka_section = (
                self._BANKA_SECTION_RE.search(content)
                or 'banka istihbarat' in content.lower().replace('İ', 'i').replace('ı', 'i')
                or 'banka istihbarat' in header.replace('İ', 'i').replace('ı', 'i')
            )
            if is_banka_section:
                banka_section_docs.append(doc)
                banka_section_info.append({
                    "filename": doc.filename,
                    "chunk_index": getattr(doc, 'chunk_index', None),
                })

        if banka_section_docs:
            logger.info(
                f"🛡️ Bank guardrail PASSED (fallback_banka_istihbarati_only): "
                f"'{norm_name}' not in text, but {len(banka_section_docs)} "
                f"'Banka İstihbaratı' chunks available in candidates"
            )
            return {
                "passed": True,
                "reason": "bank_not_in_text_but_section_available",
                "scoping_strategy": "fallback_banka_istihbarati_only",
                "safe_answer": None,
                "bank_scoped_chunks_used": banka_section_info,
                "scoped_count": len(banka_section_docs),
                "scoped_docs": banka_section_docs,
            }

        # Tier 3: bank was resolved but not in any chunk text →
        # soft fallback: let LLM try with original reranked docs
        if confidence in self.BANK_GUARDRAIL_CONFIDENCE_THRESHOLD:
            logger.info(
                f"🛡️ Bank guardrail SOFT FALLBACK: '{norm_name}' "
                f"(confidence={confidence}) resolved but not in chunk text. "
                f"Passing through {len(reranked_docs)} reranked docs to LLM."
            )
            return {
                "passed": True,
                "reason": "bank_resolved_but_not_in_text_soft_fallback",
                "scoping_strategy": "soft_fallback",
                "safe_answer": None,
                "bank_scoped_chunks_used": [],
                "scoped_count": len(reranked_docs),
                "scoped_docs": reranked_docs,
            }

        # Tier 4: nothing found → blocked
        logger.info(
            f"🛡️ Bank guardrail BLOCKED: '{norm_name}' (key='{search_key}') "
            f"not found, no Banka İstihbaratı chunks either"
        )
        return {
            "passed": False,
            "reason": "bank_not_in_context",
            "scoping_strategy": "blocked",
            "safe_answer": self.BANK_NO_DATA_ANSWER_TEMPLATE.format(bank_name=norm_name),
            "bank_scoped_chunks_used": [],
            "scoped_count": 0,
        }

    # ================================================================
    # DOCUMENT-TYPE-AWARE STRICT RETRIEVAL
    # ================================================================

    async def _retrieve_documents(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int,
        dict_headers: List[str]
    ) -> Tuple[list, Dict]:
        """
        Document-type-aware strict filtered retrieval.

        Strategy:
        - If preferred_type detected:
            1. Run filtered search by doc_type (normalized)
            2. filtered_count >= 1 → STRICT_FILTERED: use ONLY filtered, no global
            3. filtered_count == 0 → FILTERED_FALLBACK: run global as last resort
        - If no preferred_type → GLOBAL (current behavior)

        Returns:
            (reranked_docs, debug_info_dict)
        """
        routing = self._detect_query_intent(query_text)
        preferred_type = self._normalize_doc_type(routing["doc_type"]) if routing else None
        q_lower = query_text.lower()
        is_banka_istihbarati_query = (
            "banka istihbarat" in q_lower or "diğer bankalarda" in q_lower
            or "diğer bankalar" in q_lower
        )
        candidate_k = top_k * 10 if is_banka_istihbarati_query else top_k * 6

        debug: Dict = {
            "preferred_type": preferred_type,
            "retrieval_mode": "GLOBAL",
            "filtered_count": 0,
            "top3_chunks": [],
            "routing_decision": routing,
        }

        if not preferred_type:
            # No intent → global retrieval
            search_result = await self.document_repository.search_similar(
                embedding=query_embedding,
                top_k=candidate_k,
                threshold=0.0
            )
            candidates = search_result.documents
            debug["retrieval_mode"] = "GLOBAL"
            logger.info(
                f"📊 Retrieval mode=GLOBAL | no preferred_type | "
                f"candidates={len(candidates)}"
            )
        else:
            # Filtered retrieval by normalized doc_type
            try:
                filtered_result = await self.document_repository.search_similar_filtered(
                    embedding=query_embedding,
                    doc_type=preferred_type,
                    top_k=candidate_k
                )
                filtered_docs = filtered_result.documents
            except Exception as e:
                logger.warning(f"⚠️ Filtered search failed, falling back to global: {e}")
                filtered_docs = []

            filtered_count = len(filtered_docs)
            debug["filtered_count"] = filtered_count

            if filtered_count >= 1:
                # STRICT: use ONLY filtered results — no global mixing
                candidates = filtered_docs
                debug["retrieval_mode"] = "STRICT_FILTERED"
                top_sim = getattr(filtered_docs[0], 'similarity_score', 0)
                logger.info(
                    f"📊 Retrieval mode=STRICT_FILTERED | "
                    f"preferred_type={preferred_type} | "
                    f"filtered_count={filtered_count} | top_sim={top_sim:.3f}"
                )
            else:
                # filtered_count == 0 → fallback to global
                global_result = await self.document_repository.search_similar(
                    embedding=query_embedding,
                    top_k=candidate_k,
                    threshold=0.0
                )
                candidates = global_result.documents
                debug["retrieval_mode"] = "FILTERED_FALLBACK"
                logger.info(
                    f"📊 Retrieval mode=FILTERED_FALLBACK | "
                    f"preferred_type={preferred_type} | "
                    f"filtered_count=0 | global_count={len(candidates)}"
                )

        if not candidates:
            return [], debug

        # Re-rank candidates through existing pipeline
        reranked = self._rerank_chunks(
            candidates, query_text, top_k, dict_headers=dict_headers
        )

        # Build top3_chunks debug info
        top3 = []
        for i, doc in enumerate(reranked[:3]):
            doc_type = self._resolve_doc_type(doc)
            sim = getattr(doc, 'similarity_score', 0)
            top3.append({
                "rank": i + 1,
                "filename": doc.filename,
                "doc_type": doc_type,
                "similarity_score": round(sim, 4),
            })
            logger.info(
                f"📊 Selected Top-{i+1}: {doc.filename} | "
                f"doc_type={doc_type} | sim={sim:.3f}"
            )
        debug["top3_chunks"] = top3

        # Bank entity resolution + guardrail
        # Runs when EXTERNAL_BANK routing fires OR reverse bank match succeeds
        bank_resolution = self._resolve_bank_entity(query_text, candidates)
        if bank_resolution is not None:
            debug["bank_normalization"] = bank_resolution

            if routing and routing.get("matched_rule_id") != "EXTERNAL_BANK":
                debug["routing_decision"] = routing or {}
                debug.setdefault("routing_decision", {})["bank_override"] = True

            guardrail = self._apply_bank_guardrail(
                bank_resolution, reranked, all_candidates=candidates
            )
            scoped_docs = guardrail.pop("scoped_docs", None)
            debug["bank_guardrail"] = guardrail

            if not guardrail["passed"]:
                return [], debug

            if scoped_docs:
                reranked = scoped_docs

        return reranked, debug

    _BANKA_ISTIHBARATI_HEADER_BOOST = 0.40

    def _rerank_chunks(self, documents: list, query: str, final_k: int,
                       dict_headers: List[str] = None) -> list:
        """
        Chunk'ları çoklu OTOMATİK sinyal ile yeniden sırala.
        "Banka istihbaratı" / "diğer bankalarda" sorgularında Banka İstihbaratı
        başlıklı chunk'lar öne alınır (7 banka kaydının tamamı kontekste gelsin).
        """
        if not documents:
            return documents

        q_lower = query.lower()
        is_banka_istihbarati_query = (
            "banka istihbarat" in q_lower or "diğer bankalarda" in q_lower
            or "diğer bankalar" in q_lower
        )

        def _is_banka_istihbarati_chunk(doc, chunk_header: Optional[str]) -> bool:
            if chunk_header and "banka istihbarat" in chunk_header.lower().replace("ı", "i"):
                return True
            content = getattr(doc, "content", "") or ""
            if "## Banka İstihbaratı" in content or "Banka İstihbaratı" in content[:500]:
                return True
            return False

        routing = self._detect_query_intent(query)
        preferred_type = self._normalize_doc_type(routing["doc_type"]) if routing else None

        if not preferred_type:
            scored_docs = []
            for doc in documents:
                sim = getattr(doc, 'similarity_score', 0)
                content_adj = self._score_chunk_relevance(doc, query)
                chunk_header = self._get_chunk_header(doc)
                header_adj = self._calc_header_similarity(chunk_header, dict_headers or [])
                banka_adj = (
                    self._BANKA_ISTIHBARATI_HEADER_BOOST
                    if is_banka_istihbarati_query and _is_banka_istihbarati_chunk(doc, chunk_header)
                    else 0.0
                )
                scored_docs.append((sim + content_adj + header_adj + banka_adj, "", chunk_header, doc))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            if is_banka_istihbarati_query:
                bi_count = sum(1 for _, _, h, d in scored_docs[:final_k]
                              if _is_banka_istihbarati_chunk(d, h))
                logger.info(f"🔀 Banka İstihbaratı önceliği: top-{final_k} içinde {bi_count} BI chunk")
            logger.info(f"🔀 Niyet yok, genel skorlama uygulandı (dict_headers={bool(dict_headers)})")
            return [x[3] for x in scored_docs[:final_k]]

        boost = self._get_boost_for_type(preferred_type)
        competing_types = self.COMPETING_TYPES.get(preferred_type, [])
        penalty = boost * 0.5

        scored_docs = []
        for doc in documents:
            sim = getattr(doc, 'similarity_score', 0)
            doc_type = self._resolve_doc_type(doc)

            # 1. Doküman türü boost/penalty
            if doc_type == preferred_type:
                type_adj = boost
            elif doc_type in competing_types:
                type_adj = -penalty
            else:
                type_adj = 0.0

            # 2. Genel içerik skoru (otomatik)
            content_adj = self._score_chunk_relevance(doc, query)

            # 3. Sözlük kaynaklı bölüm başlığı boost
            chunk_header = self._get_chunk_header(doc)
            header_adj = self._calc_header_similarity(chunk_header, dict_headers or [])

            # 4. "Banka istihbaratı" sorgularında Banka İstihbaratı bölümü öne alınır
            banka_adj = (
                self._BANKA_ISTIHBARATI_HEADER_BOOST
                if is_banka_istihbarati_query and _is_banka_istihbarati_chunk(doc, chunk_header)
                else 0.0
            )

            boosted_sim = sim + type_adj + content_adj + header_adj + banka_adj
            scored_docs.append((boosted_sim, doc_type, chunk_header, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Çeşitlilik: tercih edilen türden %75
        preferred_chunks = [x for x in scored_docs if x[1] == preferred_type]
        other_chunks = [x for x in scored_docs if x[1] != preferred_type]

        max_preferred = max(1, int(final_k * 0.75))
        result = preferred_chunks[:max_preferred]
        remaining = final_k - len(result)
        result.extend(other_chunks[:remaining])

        if len(result) < final_k:
            result.extend(preferred_chunks[max_preferred:final_k - len(result)])

        result.sort(key=lambda x: x[0], reverse=True)
        final_docs = [x[3] for x in result[:final_k]]

        # Detaylı loglama
        type_counts: Dict[str, int] = {}
        header_counts: Dict[str, int] = {}
        for _, dt, hdr, _ in result[:final_k]:
            type_counts[dt] = type_counts.get(dt, 0) + 1
            if hdr:
                header_counts[hdr[:40]] = header_counts.get(hdr[:40], 0) + 1
        logger.info(f"🔀 Re-ranking: preferred={preferred_type}, type_boost={boost}, "
                     f"content_scoring=auto, dict_headers={dict_headers is not None}")
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

            # Step 1: Sorguyu embedding'e çevir
            logger.info("📊 Embedding query...")
            query_embedding = await self.embedding_service.embed_text(query.query)

            if not query_embedding:
                raise Exception("Embedding oluşturulamadı")

            # Step 1.5: Sözlük-destekli sorgu genişletme (sözlük varsa otomatik)
            enhanced_query, dict_headers = await self._enhance_query_with_dictionary(
                query.query, query_embedding
            )
            if enhanced_query != query.query:
                query_embedding = await self.embedding_service.embed_text(enhanced_query)

            # Step 2: Document-type-aware strict retrieval
            logger.info(f"🔎 Retrieving documents (final_k={query.top_k})...")
            reranked_docs, debug_info = await self._retrieve_documents(
                query_embedding=query_embedding,
                query_text=query.query,
                top_k=query.top_k,
                dict_headers=dict_headers
            )

            # Bank-scope guardrail — deterministic safe answer, no LLM call
            bank_guardrail = debug_info.get("bank_guardrail")
            if bank_guardrail and not bank_guardrail.get("passed", True):
                safe_answer = bank_guardrail.get("safe_answer",
                                                  "Sorgunuzla ilgili döküman bulunamadı.")
                logger.info(
                    f"🛡️ Bank guardrail blocked LLM call: "
                    f"{bank_guardrail.get('reason')}"
                )
                return RAGResponse(
                    question=query.query,
                    answer=safe_answer,
                    sources=[],
                    model=await self.llm_service.get_model_name(),
                    timestamp=datetime.utcnow(),
                    user_id=query.user_id,
                    debug_info=debug_info
                )

            if not reranked_docs:
                logger.warning("⚠️ No similar documents found")
                return RAGResponse(
                    question=query.query,
                    answer="Sorgunuzla ilgili döküman bulunamadı.",
                    sources=[],
                    model=await self.llm_service.get_model_name(),
                    timestamp=datetime.utcnow(),
                    user_id=query.user_id,
                    debug_info=debug_info
                )

            # Step 2.5: Entity-aware chunk focusing
            entity_keywords = self._extract_entity_keywords(query.query)
            focused_docs, entity_display = self._focus_entity_chunks(
                reranked_docs, entity_keywords
            )
            if entity_display:
                debug_info["entity_focus"] = {
                    "keywords": entity_keywords,
                    "matched_entity": entity_display,
                    "focused_chunks": sum(
                        1 for d in focused_docs
                        if any(kw in getattr(d, 'content', '').lower() for kw in entity_keywords)
                    ),
                }
                reranked_docs = focused_docs

            retrieval_mode = debug_info["retrieval_mode"]

            # Step 3: Kontekst oluştur + Kaynak izle (source tracking)
            logger.info(f"📝 Building context from {len(reranked_docs)} chunks (retrieval={retrieval_mode})...")
            context_parts = []
            sources_with_metadata: List[SourceWithMetadata] = []
            
            for idx, doc in enumerate(reranked_docs):
                header = None
                if hasattr(doc, 'metadata') and doc.metadata:
                    header = doc.metadata.get('header')
                
                doc_type = self._resolve_doc_type(doc)

                if idx == 0:
                    logger.info(f"📋 Top chunk [{doc.filename}] header={header} "
                                f"type={doc_type} "
                                f"sim={getattr(doc, 'similarity_score', 0):.3f} "
                                f"len={len(doc.content)} "
                                f"preview={doc.content[:300]}")
                
                header_text = f" [{header}]" if header else ""
                context_parts.append(
                    f"[Kaynak {idx+1}: {doc.filename}{header_text}] [Doküman Türü: {doc_type}]\n{doc.content}"
                )
                
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

            # Step 4: Prompt oluştur (System prompt + Constraints + Entity/Bank hint)
            logger.info("📝 Building prompt with banking compliance constraints...")
            bank_norm = debug_info.get("bank_normalization")
            resolved_bank = bank_norm.get("normalized_bank_name") if bank_norm else None

            hint = ""
            q_lower = query.query.lower()
            is_banka_istihbarati_genel = (
                "banka istihbarat" in q_lower or "diğer bankalarda" in q_lower
                or "diğer bankalar" in q_lower
            ) and not resolved_bank

            if resolved_bank:
                hint = (
                    f"\n\nÖNEMLİ İPUCU: Soruda '{resolved_bank}' bankası hakkında bilgi "
                    f"istenmektedir. 'Banka İstihbaratı' bölümündeki bu bankaya ait TÜM "
                    f"firma bazlı kayıtları (limit, nakit risk, gayrinakdi risk, teminat "
                    f"şartları, revize tarihi, durum) detaylı olarak listeleyin. "
                    f"Her firma için ayrı ayrı bilgi verin."
                )
            elif is_banka_istihbarati_genel:
                # Kontekstte tam limit/risk verisi olan kaç banka satırı var say (özet/eksik karışmasın)
                full_bank_rows = [
                    line for line in context.split("\n")
                    if "Banka:" in line and "Genel Limit:" in line
                    and re.search(r"Genel Limit:\s*[\d.,]+", line)
                ]
                n_full = len(full_bank_rows)
                hint = (
                    "\n\nİPUCU: Kontekstteki her kaynağı (Kaynak 1, 2, 3...) tara. Sadece 'Genel Limit: [sayı]' ve "
                    "'Nakit Risk: [sayı]' geçen banka–firma satırlarını **Banka İstihbaratı bölümünden:** tablosuna yaz. "
                    "Limit/risk sayısı olmayan veya Özet rapordaki eksik satırları ekleme. "
                )
                if n_full > 0:
                    hint += f"Kontekstte bu formatta {n_full} banka satırı var; hepsini tabloya yaz (Kuveyt Türk–MKS MARMARA, Türkiye Finans–AKTÜL/MKS dahil)."
                else:
                    hint += "Tam verisi olan tüm satırları tek tek ekle."
            elif entity_display:
                hint = (
                    f"\n\nÖNEMLİ İPUCU: Soruda '{entity_display}' "
                    f"hakkında bilgi istenmektedir. Kontekstte bu firma/kuruluş "
                    f"adını dikkatlice arayın ve bulduğunuz tüm verileri raporlayın."
                )

            prompt = f"""KONTEXT (Aşağıdaki finansal verileri dikkatlice oku ve soruyu cevapla):
{context}{hint}

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

            # Yanıt döndür (source tracking + debug_info ile)
            return RAGResponse(
                question=query.query,
                answer=answer.strip(),
                sources=sources_with_metadata,
                model=await self.llm_service.get_model_name(),
                timestamp=datetime.utcnow(),
                user_id=query.user_id,
                debug_info=debug_info
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
            logger.info("📊 Embedding query...")
            query_embedding = await self.embedding_service.embed_text(query.query)

            if not query_embedding:
                raise Exception("Embedding oluşturulamadı")

            # Step 1.5: Sözlük-destekli sorgu genişletme (sözlük varsa otomatik)
            enhanced_query, dict_headers = await self._enhance_query_with_dictionary(
                query.query, query_embedding
            )
            if enhanced_query != query.query:
                query_embedding = await self.embedding_service.embed_text(enhanced_query)

            # Step 2: Document-type-aware strict retrieval
            logger.info(f"🔎 Retrieving documents (final_k={query.top_k})...")
            reranked_docs, debug_info = await self._retrieve_documents(
                query_embedding=query_embedding,
                query_text=query.query,
                top_k=query.top_k,
                dict_headers=dict_headers
            )

            # Bank-scope guardrail — deterministic safe answer, no LLM call
            bank_guardrail = debug_info.get("bank_guardrail")
            if bank_guardrail and not bank_guardrail.get("passed", True):
                safe_answer = bank_guardrail.get("safe_answer",
                                                  "Sorgunuzla ilgili döküman bulunamadı.")
                logger.info(
                    f"🛡️ Bank guardrail blocked stream: "
                    f"{bank_guardrail.get('reason')}"
                )
                yield safe_answer
                return

            if not reranked_docs:
                logger.warning("⚠️ No similar documents found for stream")
                yield "Sorgunuzla ilgili döküman bulunamadı."
                return

            # Step 2.5: Entity-aware chunk focusing (stream)
            entity_keywords = self._extract_entity_keywords(query.query)
            focused_docs, entity_display = self._focus_entity_chunks(
                reranked_docs, entity_keywords
            )
            if entity_display:
                reranked_docs = focused_docs

            retrieval_mode = debug_info["retrieval_mode"]

            # Step 3: Kontekst oluştur + Kaynak izle
            logger.info(f"📝 Building context from {len(reranked_docs)} chunks (retrieval={retrieval_mode})...")
            context_parts = []
            sources_with_metadata: List[SourceWithMetadata] = []
            
            for idx, doc in enumerate(reranked_docs):
                header = None
                if hasattr(doc, 'metadata') and doc.metadata:
                    header = doc.metadata.get('header')
                
                doc_type = self._resolve_doc_type(doc)

                header_text = f" [{header}]" if header else ""
                context_parts.append(
                    f"[Kaynak {idx+1}: {doc.filename}{header_text}] [Doküman Türü: {doc_type}]\n{doc.content}"
                )
                
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
            bank_norm = debug_info.get("bank_normalization")
            resolved_bank = bank_norm.get("normalized_bank_name") if bank_norm else None

            hint = ""
            q_lower = query.query.lower()
            is_banka_istihbarati_genel = (
                "banka istihbarat" in q_lower or "diğer bankalarda" in q_lower
                or "diğer bankalar" in q_lower
            ) and not resolved_bank

            if resolved_bank:
                hint = (
                    f"\n\nÖNEMLİ İPUCU: Soruda '{resolved_bank}' bankası hakkında bilgi "
                    f"istenmektedir. 'Banka İstihbaratı' bölümündeki bu bankaya ait TÜM "
                    f"firma bazlı kayıtları (limit, nakit risk, gayrinakdi risk, teminat "
                    f"şartları, revize tarihi, durum) detaylı olarak listeleyin. "
                    f"Her firma için ayrı ayrı bilgi verin."
                )
            elif is_banka_istihbarati_genel:
                full_bank_rows = [
                    line for line in context.split("\n")
                    if "Banka:" in line and "Genel Limit:" in line
                    and re.search(r"Genel Limit:\s*[\d.,]+", line)
                ]
                n_full = len(full_bank_rows)
                hint = (
                    "\n\nİPUCU: Kontekstteki her kaynağı tara. Sadece 'Genel Limit: [sayı]' ve 'Nakit Risk: [sayı]' "
                    "geçen banka–firma satırlarını **Banka İstihbaratı bölümünden:** tablosuna yaz. Limit/risk olmayan "
                    "veya Özet rapordaki eksik satırları ekleme. "
                )
                if n_full > 0:
                    hint += f"Kontekstte bu formatta {n_full} banka satırı var; hepsini tabloya yaz."
                else:
                    hint += "Tam verisi olan tüm satırları tek tek ekle."
            elif entity_display:
                hint = (
                    f"\n\nÖNEMLİ İPUCU: Soruda '{entity_display}' "
                    f"hakkında bilgi istenmektedir. Kontekstte bu firma/kuruluş "
                    f"adını dikkatlice arayın ve bulduğunuz tüm verileri raporlayın."
                )

            prompt = f"""KONTEXT (Aşağıdaki finansal verileri dikkatlice oku ve soruyu cevapla):
{context}{hint}

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
