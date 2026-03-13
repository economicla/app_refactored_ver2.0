"""
RAGQueryUseCase - Advanced Soru Sorma İşlem Hattı
Business Logic - Framework bağımsız
Advanced preprocessing, intelligent chunking, source tracking ve compliance
"""

import json
import logging
import re
from typing import Optional, List, Dict, Tuple, Any
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

İSTİHBARAT RAPORU BÖLÜM REHBERİ (soru–bölüm eşleştirmesi):
İstihbarat Raporu kontekstte yer alıyorsa, rapordaki bölümleri şöyle yorumla. Soru hangi bölümle ilgiliyse cevabını SADECE o bölümden ver; diğer bölümlerdeki veriyi karıştırma veya başka bölümden örnek verme.
- "Özet / Genel Bilgiler": Rapor tarihi, şube, müşteri adı/no, VKN/TCKN, kuruluş, grup, sektör, segment, skorlar, erken uyarı durumu, özet bayraklar.
- "E-Haciz Tarihçesi": Haciz/araştırma kayıtları (sicil no, ödenen/ödenmeyen adet ve tutar).
- "Erken Uyarı Tarihçesi": Erken uyarı durumları, tarih, açıklama.
- "Piyasa İstihbaratı": Bilgi veren firma, kişi, ünvan, tarih, görüş, açıklama (MarketInformation).
- "Banka İstihbaratı": Diğer bankalardaki limit/risk/teminat (banka adı, firma, genel limit, nakit risk, g.nakdi risk).
- "Memzuç Bilgileri / Doluluk Oranları": Kredi grubu memzuçları, doluluk oranları, dönem bazlı limit/risk.
- "Konsolide Memzuç / KKB Riski": Konsolide memzuç, KKB risk bilgileri.
- "Çek Performansı": Karşılıksız çek, protestolu senet, ihale yasaklısı.
- "KKB Raporları": Son sorgu tarihi, firma/müşteri bilgileri.
- "Limit Risk Bilgileri": Kaynak bazında limit/risk tablosu (Bin TL).
Soru "piyasa" diyorsa sadece Piyasa İstihbaratı bölümünü, "memzuc" diyorsa sadece Memzuç bölümünü, "banka limit/risk" diyorsa Banka İstihbaratı veya Limit Risk bölümünü kullan.

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
        # P90: Piyasa İstihbaratı (MarketInformation) → İstihbarat Raporu, deterministik cevap
        # MEMZUC'tan önce eşleşmeli ki "piyasa istihbaratı" memzuc chunk'larına düşmesin
        {
            "rule_id": "PIYASA_ISTIHBARATI",
            "rule_name": "Piyasa İstihbaratı / Market Bilgisi",
            "priority": 90,
            "doc_type": "İstihbarat Raporu",
            "keywords": [
                "piyasa istihbaratı", "piyasa istihbarat", "piyasa bilgisi", "piyasa görüşü",
                "piyasa değerlendirmesi", "market intelligence", "bilgi veren firma",
                "piyasadan bilgi", "piyasa bilgileri", "market information",
            ],
            "patterns": [],
            "boost_score": 0.35,
        },
        # P55: Memzuç (Kredi Grubu Firma Memzuçları) → İstihbarat Raporu (K.V. risk, doluluk vb. bu tabloda)
        {
            "rule_id": "MEMZUC",
            "rule_name": "Memzuç / Kredi Grubu Memzuçları",
            "priority": 55,
            "doc_type": "İstihbarat Raporu",
            "keywords": [
                "memzuç", "memzuc", "memzuculuk", "memzuçlar", "memzucular",
                "kredi grubu memzuc", "kredi grubu memzuç", "memzuç veri",
                "memzuç tablosu", "memzuç dönem",
            ],
            "patterns": [],
            "boost_score": 0.35,
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

    @staticmethod
    def _is_likely_bank_name(name: str) -> bool:
        """Sadece gerçek banka adlarını kabul et; GNAkdi, Leasing, Toplam vb. tablo satırlarını elene."""
        if not name or len(name.strip()) < 5:
            return False
        n = name.strip().upper()
        if "BANKASI" in n or "BANK" in n or "KATILIM" in n:
            return True
        if any(x in n for x in ("TOPLAM", "GNAKDI", "LEASING", "AKREDİTİF", "UMUMI", "NAKDİ", "GN ", "TL", "YP", "KOD")):
            return False
        return False

    # DB'den çekilen BI içeriklerinden satır parse et (deterministik tablo için).
    # Tüm para birimlerini kabul eder (TRY, ATS, EUR, USD, BHD, CNY, XAG, AUD vb.)
    # böylece XML/PDF'deki tüm banka kayıtları tabloya girer.
    _BI_ROW_RE = re.compile(
        r"Banka:\s*(?P<bank>[^|]+)\s*\|\s*Firma:\s*(?P<firm>[^|]*)\s*\|\s*Genel Limit:\s*(?P<genel>[0-9,\-\s]+?)\s*(?P<cur>[A-Z]{2,5})\s*\|\s*Nakit Risk:\s*(?P<nakit>[0-9,\-\s]+?)\s*(?:[A-Z]{2,5})\s*\|\s*G\.Nakdi Risk:\s*(?P<gn>[0-9,\-\s]+?)\s*(?:[A-Z]{2,5})",
        re.IGNORECASE,
    )

    def _parse_bi_rows_from_contents(
        self, contents: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        get_bi_lines ile alınan content listesinden BI satırlarını regex ile çıkar.
        Tüm para birimleri kabul edilir; bank+firm uniq yapılır.
        """
        seen: set = set()
        rows: List[Dict] = []
        for item in contents:
            content = item.get("content") or ""
            for m in self._BI_ROW_RE.finditer(content):
                bank = (m.group("bank") or "").strip()
                firm = (m.group("firm") or "").strip()
                genel_s = (m.group("genel") or "").strip().replace(" ", "")
                nakit_s = (m.group("nakit") or "").strip().replace(" ", "")
                gn_s = (m.group("gn") or "").strip().replace(" ", "")
                currency = (m.group("cur") or "TRY").strip().upper()

                def _to_int(s: str) -> int:
                    if not s or s == "-":
                        return 0
                    try:
                        return int(s.replace(",", ""))
                    except ValueError:
                        return 0

                genel = _to_int(genel_s)
                nakit = _to_int(nakit_s)
                gn = _to_int(gn_s)
                key = (bank, firm)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "bank": bank,
                    "firm": firm,
                    "currency": currency,
                    "genel_limit": genel,
                    "nakit_risk": nakit,
                    "gn_risk": gn,
                })
        return rows

    def _bi_rows_from_structured_list(
        self, structured: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ingest sırasında metadata'ya kaydedilen banka_istihbarati listesini
        deterministik tablo satır formatına çevirir (tüm bankaların gösterilmesi için).
        """
        rows: List[Dict[str, Any]] = []
        for r in structured:
            gl = r.get("genel_limit") or {}
            nr = r.get("nakit_risk") or {}
            gnr = r.get("gn_risk") or {}
            cur = (
                (gl.get("currency") or nr.get("currency") or gnr.get("currency") or "TRY")
            )
            if isinstance(cur, str):
                cur = cur.strip().upper()
            else:
                cur = "TRY"
            def _val(x: Any) -> int:
                v = x.get("value") if isinstance(x, dict) else None
                if v is None:
                    return 0
                try:
                    return int(float(v))
                except (TypeError, ValueError):
                    return 0
            rows.append({
                "bank": (r.get("bank_name") or "").strip(),
                "firm": (r.get("group_or_firm") or "").strip(),
                "currency": cur,
                "genel_limit": _val(gl),
                "nakit_risk": _val(nr),
                "gn_risk": _val(gnr),
            })
        return rows

    def _format_bi_table_deterministic(
        self, rows: List[Dict], include_totals: bool = True
    ) -> str:
        """BI satırlarından markdown tablo + toplamlar üret. Tüm para birimleri gösterilir."""
        if not rows:
            return ""
        lines = [
            "| BANKA | FİRMA | PARA BİRİMİ | GENEL LİMİT | NAKİT RİSK | G.NAKDİ RİSK |",
            "|-------|-------|-------------|-------------|------------|---------------|",
        ]
        total_genel_try = total_nakit_try = total_gn_try = 0
        for r in rows:
            g = r.get("genel_limit", 0) or 0
            n = r.get("nakit_risk", 0) or 0
            gn = r.get("gn_risk", 0) or 0
            cur = r.get("currency") or "TRY"
            if cur == "TRY":
                total_genel_try += g
                total_nakit_try += n
                total_gn_try += gn
            lines.append(
                f"| {r.get('bank', '')} | {r.get('firm', '')} | {cur} | {g:,} | {n:,} | {gn:,} |"
            )
        table = "\n".join(lines)
        if include_totals and rows:
            if total_genel_try or total_nakit_try or total_gn_try:
                table += (
                    f"\n\n**Toplam (sadece TRY):** Genel Limit {total_genel_try:,} TRY  |  "
                    f"Nakit Risk {total_nakit_try:,} TRY  |  G.Nakdi Risk {total_gn_try:,} TRY"
                )
            if any((r.get("currency") or "TRY").upper() != "TRY" for r in rows):
                table += "\n\n*Tablo tüm banka istihbaratı kayıtlarını (tüm para birimleri) içerir.*"
        return table

    # Memzuc Doluluk Oranı: dönem bazlı parse (deterministik cevap)
    _MEMZUC_PERIOD_RE = re.compile(r"(?P<period>20\d{2}\s*/\s*\d{2})")
    _MEMZUC_ROW_NAMES = (
        "Umumi Limit",
        "Toplam Nakdi Kredi",
        "Toplam GN Kredi",
        "TOPLAM",
    )

    _MEMZUC_DOLULUK_LINE_RE = re.compile(
        r"MEMZUC_DOLULUK\s*\|\s*dönem:\s*(\S+)\s*\|\s*kalem:\s*([^|]+?)\s*\|\s*doluluk:\s*(\d+)",
        re.IGNORECASE,
    )
    # Extractor ile aynı: sadece bu section_type grup tablosu; diğer tablolar (ortak, KKB vb.) karışmaz
    _MEMZUC_SECTION_TYPE_GRUP = "kredi_grubu_firma_memzuclari"

    def _parse_memzuc_structured_json_from_contents(
        self, contents: List[Dict[str, str]]
    ) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Chunk'lardan MEMZUC_STRUCTURED_JSON bloğunu bulur, sadece grup tablosu (section_type
        kredi_grubu_firma_memzuclari) için data_by_period döner. Böylece ortak/KKB/tek firma
        tabloları hiç karışmaz.
        Returns: {"2025/12": {"Umumi Limit": 20, ...}, "2024/12": {...}} veya None.
        """
        combined = "\n".join((item.get("content") or "").replace("\r", "\n") for item in contents)
        idx = combined.find("MEMZUC_STRUCTURED_JSON")
        if idx == -1:
            return None
        rest = combined[idx:]
        newline = rest.find("\n")
        if newline == -1:
            first_line = rest.strip()
        else:
            json_str = rest[newline + 1 :].strip()
            first_line = json_str.split("\n")[0].strip()
        # Tek satır: "[{...}]" veya "MEMZUC_STRUCTURED_JSON=[{...}]" (chunk aranabilir olsun diye)
        if first_line.startswith("MEMZUC_STRUCTURED_JSON="):
            first_line = first_line[len("MEMZUC_STRUCTURED_JSON=") :].strip()
        if not first_line.startswith("["):
            return None
        try:
            payload = json.loads(first_line)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, list):
            return None
        for item in payload:
            if not isinstance(item, dict):
                continue
            if item.get("section_type") != self._MEMZUC_SECTION_TYPE_GRUP:
                continue
            data = item.get("data_by_period")
            if isinstance(data, dict):
                return data
        return None

    def _normalize_memzuc_kalem(self, kalem: str) -> Optional[str]:
        """Kalem adını canonical isme çevirir; üst özet tablodan gelen Umumi Limit / TOPLAM da dahil edilir."""
        k = (kalem or "").strip()
        if not k:
            return None
        k_lo = k.lower()
        if "umumi" in k_lo and "limit" in k_lo:
            return "Umumi Limit"
        if "toplam" in k_lo and "nakdi" in k_lo and "kredi" in k_lo:
            return "Toplam Nakdi Kredi"
        if "toplam" in k_lo and "gn" in k_lo and "kredi" in k_lo:
            return "Toplam GN Kredi"
        if k_lo == "toplam":
            return "TOPLAM"
        # Tam eşleşme
        for row_name in self._MEMZUC_ROW_NAMES:
            if row_name.lower() == k_lo:
                return row_name
        return None

    def _merge_memzuc_value(
        self, result: Dict[str, Dict[str, int]], period: str, kalem: str, pct: int
    ) -> None:
        """Aynı (dönem, kalem) için birden fazla değer gelirse max ile merge et (üst özet vs ana blok)."""
        norm = self._normalize_memzuc_kalem(kalem)
        if not norm:
            return
        existing = result.setdefault(period, {}).get(norm)
        result[period][norm] = max(existing, pct) if existing is not None else pct

    def _parse_memzuc_doluluk_from_contents(
        self, contents: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, int]]:
        """
        get_memzuc_lines ile alınan content'lerden dönem bazlı doluluk oranlarını çıkar.
        Aynı (dönem, kalem) için birden fazla değer gelirse max merge (üst özet + ana firma dahil).
        Önce MEMZUC_DOLULUK satırlarını parse eder; yoksa dönem bloğu fallback.
        Returns: { "2025/12": {"Umumi Limit": 20, "Toplam Nakdi Kredi": 25, ...}, ... }
        """
        combined = "\n".join((item.get("content") or "").replace("\r", "\n") for item in contents)
        if not combined.strip():
            return {}
        result: Dict[str, Dict[str, int]] = {}
        # 1) Structured format: MEMZUC_DOLULUK | dönem: 2025/12 | kalem: Umumi Limit | doluluk: 20
        for m in self._MEMZUC_DOLULUK_LINE_RE.finditer(combined):
            period_norm = (m.group(1) or "").strip().replace(" ", "")
            kalem = (m.group(2) or "").strip()
            try:
                pct = int((m.group(3) or "").strip())
            except ValueError:
                continue
            if period_norm and kalem:
                self._merge_memzuc_value(result, period_norm, kalem, pct)
        if result:
            return result
        # 2) Fallback: dönem başlığı + blok içinde satır adı + satır sonu yüzde (merge ile)
        period_positions = list(self._MEMZUC_PERIOD_RE.finditer(combined))
        for i, m in enumerate(period_positions):
            period_raw = (m.group("period") or "").strip()
            period_norm = period_raw.replace(" ", "")
            start = m.end()
            end = period_positions[i + 1].start() if i + 1 < len(period_positions) else len(combined)
            block = combined[start:end]
            for line in block.splitlines():
                line_clean = line.strip()
                if not line_clean:
                    continue
                line_lower = line_clean.lower()
                for row_name in self._MEMZUC_ROW_NAMES:
                    if row_name.lower() in line_lower:
                        pct_m = re.search(r"\d{1,2}\s*$", line_clean)
                        if pct_m:
                            try:
                                pct_val = int(pct_m.group(0).strip())
                                self._merge_memzuc_value(result, period_norm, row_name, pct_val)
                            except ValueError:
                                pass
                        break
        return result

    def _extract_requested_period_from_query(self, query: str) -> Optional[str]:
        """Sorgudan dönem çıkar (2025/10, 2024/12 vb.); normalize '2025/12'."""
        m = self._MEMZUC_PERIOD_RE.search(query)
        if not m:
            return None
        return (m.group("period") or "").replace(" ", "").strip()

    def _select_closest_period(
        self, requested: Optional[str], periods_found: List[str]
    ) -> Tuple[Optional[str], bool]:
        """
        İstenen dönem raporda varsa onu seç; yoksa en güncel dönemi seç (max YYYY/MM).
        Returns: (selected_period, exact_match).
        """
        if not periods_found:
            return None, False
        if requested and requested in periods_found:
            return requested, True
        # İstenen dönem yoksa en güncel dönem (2025/12 varken 2024/12 seçilmesin)
        return max(periods_found), False

    def _format_memzuc_doluluk_response(
        self,
        period_data: Dict[str, int],
        period_label: str,
        requested_period: Optional[str],
        exact_match: bool,
    ) -> str:
        """Memzuc doluluk oranları: 4 kalem (Umumi Limit, Toplam Nakdi Kredi, Toplam GN Kredi, TOPLAM)."""
        if not period_data:
            return "Raporda bu dönem için doluluk oranı verisi bulunamadı."
        lines = [
            f"**{period_label} dönemi — Doluluk Oranları (%)**",
            "",
            "| Kalem | Doluluk Oranı (%) |",
            "|-------|-------------------|",
        ]
        for row_name in self._MEMZUC_ROW_NAMES:
            val = period_data.get(row_name)
            lines.append(f"| {row_name} | {val if val is not None else '-'} |")
        body = "\n".join(lines)
        if not exact_match and requested_period:
            body = (
                f"*Raporda {requested_period} dönemi bulunamadı; en güncel dönem olan **{period_label}** gösteriliyor.*\n\n"
                + body
            )
        return body

    def _extract_bank_table_from_context(self, context: str) -> Tuple[List[Dict], str]:
        """
        Kontekstten sadece 'Banka: ... | Firma: ... | Genel Limit: ...' formatındaki
        gerçek banka satırlarını parse et. Limit Risk / Kaynak Bazında Detay tablolarındaki
        satırlar (GNAkdi-TL, Leasing vb.) elenir.
        """
        rows: List[Dict] = []
        parts = context.split("Banka:")
        for part in parts[1:]:
            row: Dict = {"banka": "", "firma": "", "genel_limit": "", "nakit_risk": "", "gn_risk": ""}
            segs = [s.strip() for s in part.split("|")]
            for seg in segs:
                if not seg:
                    continue
                if ":" in seg:
                    k, v = seg.split(":", 1)
                    k, v = k.strip().lower(), v.strip()
                    if k == "firma":
                        row["firma"] = v
                    elif "genel limit" in k or k == "genel limit":
                        row["genel_limit"] = (v.split()[0] if v else "").strip()
                    elif "nakit risk" in k or k == "nakit risk":
                        row["nakit_risk"] = (v.split()[0] if v else "").strip()
                    elif "g.nakdi" in k or "gnakdi" in k or "gayri" in k:
                        row["gn_risk"] = (v.split()[0] if v else "").strip()
                else:
                    if not row["banka"] and len(seg) > 2 and "Firma" not in seg and "Kaynak" not in seg:
                        row["banka"] = seg.strip()
            if not row["banka"] and segs:
                cand = segs[0].strip()
                if cand and "Kaynak" not in cand and len(cand) > 3:
                    row["banka"] = cand
            if row.get("banka"):
                row["banka"] = re.sub(r"\[Kaynak\s*\d+:.*?\]", "", row["banka"]).strip().replace("\n", " ")
            if re.search(r"[\d.,]+", row.get("genel_limit", "")) and self._is_likely_bank_name(row.get("banka", "")):
                rows.append(row)

        genel_limit_matches = list(re.finditer(r"Genel Limit:\s*([\d.,]+)", context))
        logger.info(f"📊 Banka tablosu: kontekstte {len(genel_limit_matches)} 'Genel Limit', parse {len(rows)} banka satırı")
        if len(genel_limit_matches) > len(rows):
            fallback_rows: List[Dict] = []
            for m in genel_limit_matches:
                start = m.start()
                block = context[max(0, start - 800) : start + 200]
                if "Banka:" not in block:
                    continue
                row = {"banka": "", "firma": "", "genel_limit": (m.group(1) or "").strip(), "nakit_risk": "", "gn_risk": ""}
                banka_m = re.search(r"Banka:\s*([^|\[\]]+?)(?:\s*\|\s*Firma:|\s*$)", block, re.DOTALL)
                if banka_m:
                    row["banka"] = re.sub(r"\[Kaynak\s*\d+:.*?\]", "", banka_m.group(1)).strip().replace("\n", " ")[:120]
                if not self._is_likely_bank_name(row["banka"]):
                    continue
                firma_m = re.search(r"Firma:\s*([^|]+)", block)
                if firma_m:
                    row["firma"] = firma_m.group(1).strip()
                nakit_m = re.search(r"Nakit Risk:\s*([\d.,]+)", block)
                if nakit_m:
                    row["nakit_risk"] = nakit_m.group(1).strip()
                gn_m = re.search(r"G\.?Nakdi Risk:\s*([\d.,]+)", block, re.IGNORECASE)
                if gn_m:
                    row["gn_risk"] = gn_m.group(1).strip()
                if row["genel_limit"]:
                    fallback_rows.append(row)
            if fallback_rows:
                rows = fallback_rows
                logger.info(f"📊 Banka tablosu fallback: {len(rows)} banka satırı (yalnızca Banka: adı geçen bloklar)")

        rows = [r for r in rows if self._is_likely_bank_name(r.get("banka", ""))]
        if not rows:
            return [], ""
        lines = [
            "| BANKA | FİRMA | GENEL LİMİT (TRY) | NAKİT RİSK (TRY) | G.NAKDİ RİSK (TRY) |",
            "|-------|-------|-------------------|------------------|---------------------|",
        ]
        for r in rows:
            lines.append(
                f"| {r.get('banka', '')} | {r.get('firma', '')} | {r.get('genel_limit', '')} | "
                f"{r.get('nakit_risk', '')} | {r.get('gn_risk', '')} |"
            )
        return rows, "\n".join(lines)

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

    # Sorudan / routing'den hangi bölümün kastedildiğini çıkar (LLM kontekstinde doğru chunk'lar öne gelsin)
    _SECTION_KEYWORDS: List[tuple] = [
        ("piyasa", "Piyasa İstihbaratı"),
        ("memzuç", "Memzuç Bilgileri"),
        ("memzuc", "Memzuç Bilgileri"),
        ("doluluk oran", "Memzuc Doluluk"),
        ("erken uyarı", "Erken Uyarı"),
        ("kkb risk", "KKB Riski"),
        ("kkb rapor", "KKB Raporları"),
        ("özet", "Özet"),
        ("genel bilgi", "Özet"),
        ("e-haciz", "E-Haciz"),
        ("haciz tarih", "E-Haciz"),
        ("çek performans", "Çek Performansı"),
        ("karşılıksız çek", "Çek Performansı"),
        ("limit risk bilgi", "Limit Risk"),
        ("kaynak bazında", "Limit Risk"),
        ("banka istihbarat", "Banka İstihbaratı"),
    ]

    def _infer_section_from_query(
        self, query_lower: str, routing: Optional[Dict] = None
    ) -> Optional[str]:
        """Soru veya routing'e göre İstihbarat Raporu bölüm adı döndürür (kontekst sıralaması için)."""
        if routing and routing.get("matched_rule_id") == "PIYASA_ISTIHBARATI":
            return "Piyasa İstihbaratı"
        if routing and routing.get("matched_rule_id") == "MEMZUC":
            return "Memzuç"
        for kw, section in self._SECTION_KEYWORDS:
            if kw in query_lower:
                return section
        return None

    def _order_chunks_by_section(
        self,
        docs: List[Any],
        query_lower: str,
        routing: Optional[Dict] = None,
    ) -> List[Any]:
        """İstihbarat Raporu için chunk'ları soruyla ilgili bölüme göre öne alır (LLM doğru bölümü görsün)."""
        section = self._infer_section_from_query(query_lower, routing)
        if not section or not docs:
            return docs
        content_key = "content" if hasattr(docs[0], "content") else None
        header_key = "metadata"
        matched, rest = [], []
        for d in docs:
            text = ""
            if content_key:
                text = (getattr(d, content_key, "") or "").lower()
            if hasattr(d, header_key) and getattr(d, header_key):
                meta = getattr(d, header_key) or {}
                text += " " + (meta.get("header") or "").lower()
            if section.lower() in text or ("piyasa" in section.lower() and "bilgi veren firma" in text):
                matched.append(d)
            else:
                rest.append(d)
        reordered = matched + rest
        if matched:
            logger.info(f"📌 Bölüm-duyarlı sıralama: '{section}' içeren {len(matched)} chunk öne alındı")
        return reordered

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

            # Step 2: Document-type-aware strict retrieval (banka istihbaratı için daha fazla chunk)
            q_lo = query.query.lower()
            is_bi = "banka istihbarat" in q_lo or "diğer bankalarda" in q_lo or "diğer bankalar" in q_lo
            effective_k = min(25, max(query.top_k, 18)) if is_bi else query.top_k
            logger.info(f"🔎 Retrieving documents (final_k={effective_k})...")
            reranked_docs, debug_info = await self._retrieve_documents(
                query_embedding=query_embedding,
                query_text=query.query,
                top_k=effective_k,
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

            # Deterministik BI tablosu: diğer bankalarda limit/risk veya EXTERNAL_BANK ise DB'den BI satırlarını topla
            routing = debug_info.get("routing_decision") or {}
            is_external_bank = routing.get("matched_rule_id") == "EXTERNAL_BANK"
            use_deterministic_bi = (
                (is_bi or is_external_bank)
                and hasattr(self.document_repository, "get_bi_lines")
            )
            if use_deterministic_bi:
                try:
                    preferred_filename = getattr(reranked_docs[0], "filename", None) if reranked_docs else None
                    bi_rows = None
                    get_structured = getattr(
                        self.document_repository, "get_banka_istihbarati_structured", None
                    )
                    if get_structured and preferred_filename:
                        structured_bi = await get_structured(preferred_filename)
                        if structured_bi:
                            bi_rows = self._bi_rows_from_structured_list(structured_bi)
                    if not bi_rows:
                        get_bi_lines = getattr(self.document_repository, "get_bi_lines")
                        bi_chunks = await get_bi_lines(
                            filename=preferred_filename,
                            doc_type="İstihbarat Raporu",
                        )
                        if not bi_chunks and preferred_filename is None:
                            bi_chunks = await get_bi_lines(filename=None, doc_type="İstihbarat Raporu")
                        bi_rows = self._parse_bi_rows_from_contents(bi_chunks)
                    if bi_rows:
                        table_md = self._format_bi_table_deterministic(bi_rows)
                        answer = "**Banka İstihbaratı bölümünden:**\n\n" + table_md
                        debug_info["bi_deterministic"] = {
                            "rows": len(bi_rows),
                            "filename": preferred_filename,
                        }
                        sources_bi = [
                            SourceWithMetadata(
                                filename=getattr(d, "filename", ""),
                                chunk_index=getattr(d, "chunk_index", 0),
                                header=None,
                                similarity_score=getattr(d, "similarity_score", 0),
                                content_preview=(getattr(d, "content", "") or "")[:150],
                                chunk_size=len(getattr(d, "content", "") or ""),
                            )
                            for d in reranked_docs[:5]
                        ]
                        logger.info(f"📊 Deterministik BI tablosu: {len(bi_rows)} satır (LLM atlandı)")
                        return RAGResponse(
                            question=query.query,
                            answer=answer,
                            sources=sources_bi,
                            model=await self.llm_service.get_model_name(),
                            timestamp=datetime.utcnow(),
                            user_id=query.user_id,
                            debug_info=debug_info,
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Deterministik BI toplama hatası, LLM yoluna düşülüyor: {e}")
                    debug_info.setdefault("bi_deterministic", {})["error"] = str(e)

            # Deterministik Piyasa İstihbaratı: piyasa istihbaratı sorulduğunda sadece MarketInformation verisi dön (memzuc/LLM karışmasın)
            is_piyasa_route = routing.get("matched_rule_id") == "PIYASA_ISTIHBARATI"
            if is_piyasa_route and reranked_docs:
                try:
                    get_piyasa = getattr(
                        self.document_repository, "get_piyasa_istihbarati_structured", None
                    )
                    preferred_filename = getattr(reranked_docs[0], "filename", None)
                    if get_piyasa and preferred_filename:
                        piyasa_data = await get_piyasa(preferred_filename)
                        if piyasa_data and isinstance(piyasa_data, dict):
                            lines = ["**Piyasa İstihbaratı (MarketInformation):**\n"]
                            for key, value in piyasa_data.items():
                                if key and value and not key.startswith("_"):
                                    lines.append(f"- **{key}:** {value}")
                            if len(lines) > 1:
                                answer = "\n".join(lines)
                                debug_info["piyasa_deterministic"] = {
                                    "entries": len(piyasa_data),
                                    "filename": preferred_filename,
                                }
                                sources_piyasa = [
                                    SourceWithMetadata(
                                        filename=getattr(d, "filename", ""),
                                        chunk_index=getattr(d, "chunk_index", 0),
                                        header=None,
                                        similarity_score=getattr(d, "similarity_score", 0),
                                        content_preview=(getattr(d, "content", "") or "")[:150],
                                        chunk_size=len(getattr(d, "content", "") or ""),
                                    )
                                    for d in reranked_docs[:5]
                                ]
                                logger.info(f"📊 Deterministik Piyasa İstihbaratı: {len(piyasa_data)} kayıt (LLM atlandı)")
                                return RAGResponse(
                                    question=query.query,
                                    answer=answer,
                                    sources=sources_piyasa,
                                    model=await self.llm_service.get_model_name(),
                                    timestamp=datetime.utcnow(),
                                    user_id=query.user_id,
                                    debug_info=debug_info,
                                )
                except Exception as e:
                    logger.warning(f"⚠️ Deterministik Piyasa toplama hatası, LLM yoluna düşülüyor: {e}")
                    debug_info.setdefault("piyasa_deterministic", {})["error"] = str(e)

            # Deterministik Memzuc Doluluk: sorguda memzuc/doluluk geçiyorsa doluluk tablosu dön; ama kısa/orta/uzun vadeli risk veya toplam risk soruluyorsa normal RAG ile tablonun tam metninden cevaplansın
            routing_m = debug_info.get("routing_decision") or {}
            q_lo = query.query.lower()
            _MEMZUC_TRIGGER_KEYWORDS = (
                "memzuç", "memzuc", "doluluk", "kredi grubu memzuculuk", "memzuculuk",
            )
            _MEMZUC_RISK_KEYWORDS = (
                "kısa vadeli risk", "k.v. risk", "kv risk", "orta vadeli risk", "o.v. risk", "ov risk",
                "uzun vadeli risk", "u.v. risk", "uv risk", "toplam risk", "kısa vadeli riski",
                "kısa vadeli risk oranı", "k.v risk", "temerrüt", "faiz reeskont",
            )
            # Tek firma soruluyorsa deterministik doluluk (grup tablosu) kullanma; normal RAG ile o firma chunk'ından cevaplansın
            _MEMZUC_SINGLE_FIRMA_KEYWORDS = (
                "aktül kağıt", "aktul kagit", "bahariye mensucat", "bahariye tekstil",
                "mks marmara", "mustafa latif topbaş", "mustafa latif topbas",
            )
            is_memzuc_query = (
                any(kw in q_lo for kw in _MEMZUC_TRIGGER_KEYWORDS)
                or routing_m.get("matched_rule_id") == "MEMZUC"
            )
            asks_for_single_firma = any(kw in q_lo for kw in _MEMZUC_SINGLE_FIRMA_KEYWORDS)
            asks_for_risk_or_limit = any(kw in q_lo for kw in _MEMZUC_RISK_KEYWORDS)
            use_memzuc_doluluk_only = (
                is_memzuc_query and not asks_for_risk_or_limit and not asks_for_single_firma
            )
            if use_memzuc_doluluk_only and hasattr(self.document_repository, "get_memzuc_lines"):
                try:
                    get_memzuc_lines = getattr(self.document_repository, "get_memzuc_lines")
                    preferred_filename = getattr(reranked_docs[0], "filename", None) if reranked_docs else None
                    memzuc_chunks = await get_memzuc_lines(
                        filename=preferred_filename,
                        doc_type="İstihbarat Raporu",
                    )
                    if not memzuc_chunks and preferred_filename:
                        memzuc_chunks = await get_memzuc_lines(filename=None, doc_type="İstihbarat Raporu")
                    # Önce yapılandırılmış JSON'dan oku (sadece grup tablosu; ortak/KKB/tek firma karışmaz)
                    period_to_data = self._parse_memzuc_structured_json_from_contents(memzuc_chunks)
                    if not period_to_data:
                        period_to_data = self._parse_memzuc_doluluk_from_contents(memzuc_chunks)
                    periods_found = sorted(period_to_data.keys()) if period_to_data else []
                    requested_period = self._extract_requested_period_from_query(query.query)
                    selected_period, exact_match = self._select_closest_period(
                        requested_period, periods_found
                    )
                    if selected_period and period_to_data.get(selected_period):
                        period_data = period_to_data[selected_period]
                        answer = self._format_memzuc_doluluk_response(
                            period_data,
                            selected_period,
                            requested_period,
                            exact_match,
                        )
                        row_count = len(period_data)
                        debug_info["memzuc_deterministic"] = True
                        debug_info["memzuc_periods_found"] = periods_found
                        debug_info["memzuc_requested_period"] = requested_period
                        debug_info["memzuc_selected_period"] = selected_period
                        debug_info["memzuc_row_count"] = row_count
                        sources_memzuc = [
                            SourceWithMetadata(
                                filename=getattr(d, "filename", ""),
                                chunk_index=getattr(d, "chunk_index", 0),
                                header=None,
                                similarity_score=getattr(d, "similarity_score", 0),
                                content_preview=(getattr(d, "content", "") or "")[:150],
                                chunk_size=len(getattr(d, "content", "") or ""),
                            )
                            for d in reranked_docs[:5]
                        ]
                        logger.info(
                            f"📊 Deterministik Memzuc doluluk: selected_period={selected_period}, "
                            f"periods_found={periods_found}, row_count={row_count} (LLM atlandı)"
                        )
                        return RAGResponse(
                            question=query.query,
                            answer=answer,
                            sources=sources_memzuc,
                            model=await self.llm_service.get_model_name(),
                            timestamp=datetime.utcnow(),
                            user_id=query.user_id,
                            debug_info=debug_info,
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Deterministik Memzuc toplama hatası, LLM yoluna düşülüyor: {e}"
                    )
                    debug_info["memzuc_error"] = str(e)
                    debug_info.setdefault("memzuc_deterministic", {})["error"] = str(e)

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

            # Bölüm-duyarlı sıralama: soru hangi bölümü soruyorsa o bölümün chunk'ları kontekste önce gelsin (model doğru veriyi görsün)
            routing = debug_info.get("routing_decision") or {}
            reranked_docs = self._order_chunks_by_section(
                reranked_docs, query.query.lower(), routing
            )

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
                # Kontekstten banka tablosunu kodla çıkar; LLM'e hazır tablo ver, eksik kalmasın
                bank_rows, bank_table_md = self._extract_bank_table_from_context(context)
                n_full = len(bank_rows)
                logger.info(f"📊 Banka istihbaratı: kontekstten {n_full} banka kaydı çıkarıldı (tablo enjekte)")
                if bank_table_md:
                    hint = (
                        "\n\nZORUNLU: Aşağıda kontekstten çıkarılmış TÜM banka kayıtları tablosu verilmiştir. "
                        "Cevabında **Banka İstihbaratı bölümünden:** başlığı altında bu tabloyu AYNEN kullan, "
                        "satır ekleme/çıkarma yapma. Ardından kısa toplam özeti yaz.\n\n"
                        "ÇIKARILAN TABLO:\n" + bank_table_md
                    )
                else:
                    n_fallback = min(len(re.findall(r"Genel Limit:\s*[\d.,]+", context)), 15)
                    hint = (
                        "\n\nİPUCU: Kontekstteki her kaynağı tara. 'Genel Limit: [sayı]' ve 'Nakit Risk: [sayı]' "
                        "geçen tüm banka–firma satırlarını **Banka İstihbaratı bölümünden:** tablosunda listele. "
                    )
                    if n_fallback > 0:
                        hint += f"Kontekstte {n_fallback} tam kayıt var; hepsini tabloya yaz."
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

            # Step 2: Document-type-aware strict retrieval (banka istihbaratı için daha fazla chunk)
            q_lo = query.query.lower()
            is_bi = "banka istihbarat" in q_lo or "diğer bankalarda" in q_lo or "diğer bankalar" in q_lo
            effective_k = min(25, max(query.top_k, 18)) if is_bi else query.top_k
            logger.info(f"🔎 Retrieving documents (final_k={effective_k})...")
            reranked_docs, debug_info = await self._retrieve_documents(
                query_embedding=query_embedding,
                query_text=query.query,
                top_k=effective_k,
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

            # Bölüm-duyarlı sıralama (stream path)
            routing = debug_info.get("routing_decision") or {}
            reranked_docs = self._order_chunks_by_section(
                reranked_docs, query.query.lower(), routing
            )

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
                bank_rows, bank_table_md = self._extract_bank_table_from_context(context)
                n_full = len(bank_rows)
                logger.info(f"📊 Banka istihbaratı (stream): kontekstten {n_full} banka kaydı çıkarıldı (tablo enjekte)")
                if bank_table_md:
                    hint = (
                        "\n\nZORUNLU: Aşağıda kontekstten çıkarılmış TÜM banka kayıtları tablosu verilmiştir. "
                        "Cevabında **Banka İstihbaratı bölümünden:** başlığı altında bu tabloyu AYNEN kullan.\n\n"
                        "ÇIKARILAN TABLO:\n" + bank_table_md
                    )
                else:
                    n_fallback = min(len(re.findall(r"Genel Limit:\s*[\d.,]+", context)), 15)
                    hint = (
                        "\n\nİPUCU: Kontekstteki her kaynağı tara. Tam limit/risk verisi olan tüm banka satırlarını listele. "
                    )
                    if n_fallback > 0:
                        hint += f"Cevap tablosunda {n_fallback} satır olmalı."
                    else:
                        hint += "Tam verisi olan tüm satırları ekle."
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
