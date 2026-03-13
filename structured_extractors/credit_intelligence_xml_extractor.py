"""
CreditIntelligenceXMLExtractor — Structured extractor for "Kredi İstihbarat
Raporu" XML files (IntelligenceReportAdobe schema).

Produces the **exact same JSON schema** as CreditIntelligencePDFExtractor so
that render_structured_text(), IntelligentChunker, RAGQueryUseCase (routing,
bank guardrail, deterministic BI/memzuc tables) work without any changes.

Tolerant parsing: pre-sanitizes broken XML (unclosed tags, malformed
self-closing tags) before feeding to the parser.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML sanitizer — fixes common malformations before strict parse
# ---------------------------------------------------------------------------

_BROKEN_SELF_CLOSE_RE = re.compile(rb"<(\w+)\s+/(?!>)")

_FALSE_SELF_CLOSE_RE = re.compile(
    rb"<(\w+)\s*/>\s*([^<]+)</\1>", re.DOTALL
)

_OPEN_TAG_NO_CLOSE_RE = re.compile(rb"^(\s*)<(\w+)>([^<]+)$")
_SIBLING_TAG_RE = re.compile(rb"^\s*<\w+")


def _sanitize_xml(raw: bytes) -> bytes:
    """Best-effort fix of known XML issues so ElementTree can parse."""
    raw = _BROKEN_SELF_CLOSE_RE.sub(rb"<\1 />", raw)
    raw = _FALSE_SELF_CLOSE_RE.sub(rb"<\1>\2</\1>", raw)

    lines = raw.split(b"\n")
    fixed: list = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _OPEN_TAG_NO_CLOSE_RE.match(line.rstrip())
        if m and m.group(3).strip():
            tag = m.group(2)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].lstrip()
                is_sibling = (
                    _SIBLING_TAG_RE.match(nxt)
                    and not nxt.startswith(b"</" + tag + b">")
                )
                if is_sibling:
                    indent = m.group(1)
                    value = m.group(3).rstrip()
                    fixed.append(
                        indent + b"<" + tag + b">" + value + b"</" + tag + b">"
                    )
                    i += 1
                    continue
        fixed.append(line)
        i += 1

    return b"\n".join(fixed)


def _parse_xml(file_path: str) -> Tuple[Optional[ET.Element], List[str]]:
    """Parse XML with sanitization fallback.  Returns (root, warnings)."""
    warnings: List[str] = []
    path = Path(file_path)

    raw = path.read_bytes()
    try:
        return ET.fromstring(raw), warnings
    except ET.ParseError as e:
        warnings.append(f"Initial parse failed ({e}); applying sanitizer")

    sanitized = _sanitize_xml(raw)
    try:
        return ET.fromstring(sanitized), warnings
    except ET.ParseError as e2:
        warnings.append(f"Sanitized parse also failed ({e2}); trying lxml recover")

    try:
        from lxml import etree as lxml_etree
        parser = lxml_etree.XMLParser(recover=True, encoding="utf-8")
        tree = lxml_etree.fromstring(sanitized, parser=parser)
        std_xml = ET.fromstring(lxml_etree.tostring(tree))
        warnings.append("Parsed via lxml recover mode")
        return std_xml, warnings
    except Exception as e3:
        warnings.append(f"lxml recover failed ({e3}); extraction may be partial")

    try:
        cleaned = re.sub(rb"[^\x09\x0A\x0D\x20-\xFF]", b"", sanitized)
        return ET.fromstring(cleaned), warnings
    except ET.ParseError as e4:
        warnings.append(f"Final parse attempt failed ({e4})")
        return None, warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(el: Optional[ET.Element], tag: str, default: str = "") -> str:
    if el is None:
        return default
    child = el.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _text_direct(el: Optional[ET.Element], default: str = "") -> str:
    if el is None or el.text is None:
        return default
    return el.text.strip()


def _num(el: Optional[ET.Element], tag: str) -> Optional[float]:
    raw = _text(el, tag)
    if not raw:
        return None
    try:
        return float(raw.replace(",", "."))
    except (ValueError, TypeError):
        return None


def _int_val(el: Optional[ET.Element], tag: str) -> Optional[int]:
    v = _num(el, tag)
    if v is None:
        return None
    return int(v)


def _money(value: Optional[float], currency: str = "TRY") -> Dict:
    int_val = None
    if value is not None:
        int_val = int(value) if float(value) == int(value) else value
    return {"value": int_val, "currency": currency}


_FILL_RATE_SENTINEL = 99999.0
_DOLULUK_ROW_NAMES = ("Umumi Limit", "Toplam Nakdi Kredi", "Toplam GN Kredi", "TOPLAM")

_MEMZUC_SECTION_TYPE_GRUP = "kredi_grubu_firma_memzuclari"

_MAINLABEL_TO_SECTION_TYPE: List[Tuple[str, str, str]] = [
    ("KREDİ GRUBU FİRMA MEMZUÇLARI", _MEMZUC_SECTION_TYPE_GRUP, "grup"),
    ("KREDİ GRUBU FİRMA MEMZUCULARI", _MEMZUC_SECTION_TYPE_GRUP, "grup"),
    ("MEMZUÇLARI (ANA FİRMA DAHİL)", _MEMZUC_SECTION_TYPE_GRUP, "grup"),
    ("GRUP ANA FİRMA MEMZUCU", "grup_ana_firma_memzucu", ""),
    ("ORTAĞI OLDUĞU ŞİRKETLERİN MEMZUÇLARI", "ortagi_oldugu_sirketlerin_memzuclari", "ortak_sirketler"),
    ("KKB EKRANI İLİŞKİLİ FİRMA MEMZUÇLARI", "kkb_ekrani_iliskili_firma_memzuclari", ""),
    ("GRUP FİRMALARI VE İLİŞKİ", "grup_firmalari_ve_iliskili_memzuclari", "grup_ve_iliskili"),
    ("KONSOLİDE MEMZUÇ/KRM", "konsolide_memzuc_krm", "konsolide"),
]


def _classify_memzuc_label(label: str) -> Tuple[str, str]:
    """Map MainLabel text to (section_type, default_entity)."""
    upper = (label or "").upper()
    for pattern, stype, entity in _MAINLABEL_TO_SECTION_TYPE:
        if pattern.upper() in upper:
            return stype, entity
    return "grup_ana_firma_memzucu", ""


def _normalize_doluluk_row(risk_code: str) -> Optional[str]:
    rc = (risk_code or "").strip()
    low = rc.lower()
    if "umumi" in low and "limit" in low:
        return "Umumi Limit"
    if "toplam" in low and "nakdi" in low and "kredi" in low:
        return "Toplam Nakdi Kredi"
    if "toplam" in low and "gn" in low and "kredi" in low:
        return "Toplam GN Kredi"
    if low == "toplam":
        return "TOPLAM"
    for name in _DOLULUK_ROW_NAMES:
        if name.lower() == low:
            return name
    return None


def _map_currency(xml_currency: str) -> str:
    """Map XML CurrencyType to standard code."""
    c = (xml_currency or "").strip().upper()
    if c in ("TRY", "TL", ""):
        return "TRY"
    if c in ("USD", "$"):
        return "USD"
    if c in ("EUR", "€"):
        return "EUR"
    return c if c else "TRY"


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class CreditIntelligenceXMLExtractor:
    """
    Extract structured JSON from a Kredi İstihbarat Raporu XML file.

    Output schema is identical to CreditIntelligencePDFExtractor.extract()
    so that render_structured_text() and the full RAG pipeline work as-is.

    Usage::

        extractor = CreditIntelligenceXMLExtractor()
        data = extractor.extract("/path/to/report.xml")
    """

    def __init__(self, *, default_currency: str = "TRY"):
        self._default_currency = default_currency

    # ---------------------------------------------------------------- public
    def extract(self, file_path: str) -> Dict:
        path = Path(file_path)
        root, warnings = _parse_xml(file_path)

        if root is None:
            logger.error("XML parse failed completely for %s", path.name)
            return {
                "doc_type": "İstihbarat Raporu",
                "meta": {
                    "source_file": path.name,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "pages": 0,
                },
                "sections": {},
                "debug": {"warnings": warnings},
            }

        sections: Dict[str, Any] = {}

        sections["ozet_genel_bilgiler"] = self._build_ozet(root, warnings)
        sections["e_haciz_tarihcesi"] = self._build_e_haciz(root, warnings)
        sections["erken_uyari_tarihcesi"] = self._build_erken_uyari(root, warnings)
        sections["piyasa_istihbarati"] = self._build_piyasa(root, warnings)
        sections["banka_istihbarati"] = self._build_banka_istihbarati(root, warnings)

        memzuc = self._build_memzuc_all(root, warnings)
        sections["memzuc_bilgileri"] = memzuc["memzuc_bilgileri"]
        if memzuc["doluluk_fallback"]:
            sections["memzuc_doluluk_fallback"] = memzuc["doluluk_fallback"]
        if memzuc["structured"]:
            sections["memzuc_structured"] = memzuc["structured"]

        sections["konsolide_memzuc"] = self._build_konsolide(root, warnings)
        sections["cek_performansi"] = self._build_cek_performansi(root, warnings)
        sections["kkb_riski"] = self._build_kkb_riski(root, warnings)
        sections["kkb_raporlari"] = self._build_kkb_raporlari(root, warnings)
        sections["limit_risk_bilgileri"] = self._build_limit_risk(root, warnings)

        logger.info(
            "XML extraction complete: %s — %d sections, %d warnings",
            path.name,
            sum(1 for v in sections.values() if v),
            len(warnings),
        )

        return {
            "doc_type": "İstihbarat Raporu",
            "meta": {
                "source_file": path.name,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "pages": 1,
            },
            "sections": sections,
            "debug": {"warnings": warnings},
        }

    # ----------------------------------------------------- section builders

    def _build_ozet(self, root: ET.Element, warnings: List[str]) -> Dict:
        """GeneralInformation + SummaryReport → ozet_genel_bilgiler."""
        result: Dict[str, Any] = {}

        gi = root.find("GeneralInformation")
        if gi is not None:
            field_map = {
                "ReportDate": "Rapor Tarihi",
                "BranchName": "Şube",
                "CustomerName": "Müşteri Adı",
                "CustomerNumber": "Müşteri No",
                "IdentityNumber": "VKN / TCKN",
                "EstablishmentDate": "Kuruluş Tarihi",
                "GroupCode": "Grup Kodu",
                "GroupName": "Grup Adı",
                "GroupFirmCount": "Grup Firma Sayısı",
                "Sector": "Sektör",
                "SegmentName": "Segment",
                "CustomerScore": "Müşteri Skoru",
                "CustomerTradeIndebtnessIndex": "Ticari Borçluluk Endeksi",
                "RatingNote": "Rating Notu",
                "IntelligenceScore": "İstihbarat Skoru",
                "KkbChequeScore": "KKB Çek Skoru",
                "EarlyWarningStatus": "Erken Uyarı Durumu",
            }
            for xml_tag, display_name in field_map.items():
                val = _text(gi, xml_tag)
                if val:
                    result[display_name] = val

        sr = root.find("SummaryReport")
        if sr is not None:
            flag_map = {
                "IsTenderForbidden": "İhale Yasaklı",
                "IsNegativeMarket": "Olumsuz Piyasa",
                "IsNegativeBank": "Olumsuz Banka",
                "IsDudeCheque": "Karşılıksız Çek",
                "IsTaxSgkDebtor": "Vergi/SGK Borçlusu",
                "IsDistraint": "Haciz",
                "IsBankingFactoring": "Bankacılık Factoring",
                "IsMarketFactoring": "Piyasa Factoring",
                "IsBankruptcyConcordat": "İflas/Konkordato",
                "IsBlock": "Bloke",
                "IsGmi": "GMİ",
                "IsCheque": "Çek",
                "IsFollow": "Takip",
                "IsOpen": "Açık",
                "IsBail": "Kefalet",
            }
            flags = {}
            for xml_tag, display_name in flag_map.items():
                val = _text(sr, xml_tag)
                if val:
                    flags[display_name] = val
            if flags:
                result["Özet Bayraklar"] = flags

        main_name = _text(root, "MainCustomerName")
        main_no = _text(root, "MainCustomerNumber")
        if main_name and "Müşteri Adı" not in result:
            result["Müşteri Adı"] = main_name
        if main_no and "Müşteri No" not in result:
            result["Müşteri No"] = main_no

        result["source_pages"] = [1]
        return result

    def _build_e_haciz(self, root: ET.Element, warnings: List[str]) -> List[Dict]:
        """AutomaticInformationSummary (HI) + CompanyMoralityDistraints."""
        rows: List[Dict] = []

        for auto_sum in root.iter("AutomaticInformationDetail"):
            type_code = _text(auto_sum, "TypeCode")
            if type_code != "HI":
                continue
            rows.append({
                "sicil_no": _text(auto_sum, "RegisterNo"),
                "unvan": _text(auto_sum, "CustomerName"),
                "yil": _text(auto_sum, "Year"),
                "odenen_adet": _int_val(auto_sum, "Count") if _text(auto_sum, "Status") in ("Cancelled", "K") else 0,
                "odenen_tutar": _num(auto_sum, "TotalAmountTRY") if _text(auto_sum, "Status") in ("Cancelled", "K") else 0,
                "odenmeyen_adet": _int_val(auto_sum, "Count") if _text(auto_sum, "Status") not in ("Cancelled", "K", "") else 0,
                "odenmeyen_tutar": _num(auto_sum, "TotalAmountTRY") if _text(auto_sum, "Status") not in ("Cancelled", "K", "") else 0,
                "source_page": 1,
            })

        for dist in root.iter("CompanyMoralityDistraint"):
            rows.append({
                "sicil_no": _text(dist, "RegisterNo", ""),
                "unvan": _text(dist, "CustomerName"),
                "yil": _text(dist, "Year"),
                "odenen_adet": _int_val(dist, "PaidCount") or 0,
                "odenen_tutar": _num(dist, "PaidTotalAmount") or 0,
                "odenmeyen_adet": _int_val(dist, "NonPaidCount") or 0,
                "odenmeyen_tutar": _num(dist, "NonPaidTotalAmount") or 0,
                "source_page": 1,
            })

        return rows

    def _build_erken_uyari(self, root: ET.Element, warnings: List[str]) -> List[Dict]:
        """CompanyMoralities > EarlyWarningHistories."""
        rows: List[Dict] = []
        for ew in root.iter("EarlyWarningHistory"):
            rows.append({
                "Müşteri No": _text(ew, "EwsCustomerNumber"),
                "Müşteri Adı": _text(ew, "EwsCustomerName"),
                "Tarih": _text(ew, "EwsCodeUpdateDate"),
                "Durum": _text(ew, "EarlyWarningStatusCode"),
                "Açıklama": _text(ew, "EwsReason"),
                "source_page": 1,
            })
        return rows

    def _build_piyasa(self, root: ET.Element, warnings: List[str]) -> Dict:
        """MarketInfo > MarketInformation → key-value dict."""
        result: Dict[str, Any] = {}
        records: List[str] = []

        for idx, mi in enumerate(root.iter("MarketInformation"), 1):
            firm = _text(mi, "InformantFirm")
            person = _text(mi, "InformantPerson")
            title = _text(mi, "InformantTitle")
            explanation = _text(mi, "InformantExplanation")
            date = _text(mi, "MarketInfoDate")
            opinion = _text(mi, "IsMarketIntelligenceNegative")

            if date and "T" in date:
                date = date.split("T")[0]

            record_line = (
                f"Bilgi Veren Firma: {firm} | Kişi: {person} | "
                f"Ünvan: {title} | Tarih: {date} | "
                f"Görüş: {opinion} | Açıklama: {explanation}"
            )
            records.append(record_line)
            result[f"Piyasa İstihbaratı #{idx}"] = record_line

        result["source_pages"] = [1]
        return result

    def _build_banka_istihbarati(
        self, root: ET.Element, warnings: List[str]
    ) -> List[Dict]:
        """BankingIntelligence > BankRiskCautions → banka_istihbarati records.

        Output format is identical to CreditIntelligencePDFExtractor so that
        render_structured_text() produces ``Banka: X | Firma: Y | Genel Limit: Z``
        lines and _BI_ROW_RE in RAGQueryUseCase can parse them.
        """
        records: List[Dict] = []
        seen_keys: set = set()

        for bi in root.iter("BankingIntelligence"):
            for cautions in bi.iter("BankRiskCautions"):
                bank_name = _text(cautions, "BankName")
                if not bank_name:
                    continue

                caution = cautions.find("BankRiskCaution")
                if caution is None:
                    continue

                currency = _map_currency(_text(caution, "CurrencyType"))
                customer = _text(caution, "CustomerName")
                istih_date = _text(caution, "InformationDate")
                revize_date = _text(caution, "RevisionDate")

                opinion = _text(caution, "IntelOpinion")
                if not opinion:
                    opinion = _text(caution, "IntelOpinionBlack") or _text(caution, "IntelOpinionRed")

                guarantee = cautions.find("BankRiskGuarantee")
                teminat = _text(guarantee, "GeneralInfoTeminat") if guarantee is not None else ""
                alinan_tem = _text(guarantee, "ReceivingGuarantee") if guarantee is not None else ""
                alinan_kefil = _text(guarantee, "ReceivingGuarantor") if guarantee is not None else ""

                general_info = _text(caution, "GeneralInfo")
                notes = [general_info] if general_info else []

                dedup_key = (bank_name, customer, istih_date, revize_date)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                records.append({
                    "bank_name": bank_name,
                    "group_or_firm": customer,
                    "genel_limit": _money(_num(caution, "GeneralLimit"), currency),
                    "nakit_risk": _money(_num(caution, "CashRisk"), currency),
                    "gn_risk": _money(_num(caution, "NonCashRisk"), currency),
                    "d_kd": _money(None, currency),
                    "istih_tarihi": istih_date,
                    "revize_tarihi": revize_date,
                    "status": opinion,
                    "teminat_sarti": teminat,
                    "alinan_teminat": alinan_tem,
                    "alinan_kefil": alinan_kefil,
                    "notes": notes,
                    "source_pages": [1],
                })

        logger.info("XML banka_istihbarati: %d records extracted", len(records))
        return records

    # --------------------------------------------------------- memzuc

    def _extract_period_from_label(self, label: str) -> str:
        m = re.search(r"(20\d{2}\s*/\s*\d{1,2})", label or "")
        if m:
            return m.group(1).replace(" ", "")
        return ""

    def _build_memzuc_all(
        self, root: ET.Element, warnings: List[str]
    ) -> Dict[str, Any]:
        """Build memzuc_bilgileri, memzuc_doluluk_fallback, memzuc_structured."""

        memzuc_bilgileri: Dict[str, List] = {
            "grup_ana_firma_memzucu": [],
            "kredi_grubu_firma_memzuclari": [],
            "ortagi_oldugu_sirketlerin_memzuclari": [],
            "kkb_ekrani_iliskili_firma_memzuclari": [],
        }
        doluluk_fallback: List[str] = []
        structured: List[Dict[str, Any]] = []

        all_fill_rate_groups = list(root.iter("MemzucFillRateGroupeds"))
        all_memzuc_grouped = list(root.iter("MemzucGrouped"))

        sources = all_fill_rate_groups if all_fill_rate_groups else all_memzuc_grouped

        for group_el in sources:
            main_label = _text(group_el, "MainLabel")
            period_label = _text(group_el, "PeriodLabel")
            customer_name = _text(group_el, "CustomerName")
            bank_count_label = _text(group_el, "BankCountLabel")

            period = self._extract_period_from_label(period_label)
            if not period:
                period = self._extract_period_from_label(main_label)

            section_type, default_entity = _classify_memzuc_label(main_label)
            entity = default_entity or customer_name or "bilinmiyor"

            data_by_period: Dict[str, Dict[str, int]] = {}
            detail_rows: List[Dict] = []

            for detail in group_el.iter("MemzucGroupedDetail"):
                risk_code = _text(detail, "RiskCode")
                fill_rate_raw = _num(detail, "FillRate")
                limit_val = _num(detail, "Limit")
                kv_risk = _num(detail, "KVRisk")
                ov_risk = _num(detail, "OVRisk")
                uv_risk = _num(detail, "UVRisk")
                total_risk = _num(detail, "TotalRisk")

                row_entry = {
                    "RiskCode": risk_code,
                    "Limit": limit_val,
                    "KVRisk": kv_risk,
                    "OVRisk": ov_risk,
                    "UVRisk": uv_risk,
                    "TotalRisk": total_risk,
                    "FillRate": fill_rate_raw,
                    "source_page": 1,
                }
                detail_rows.append(row_entry)

                doluluk_name = _normalize_doluluk_row(risk_code)
                if doluluk_name and fill_rate_raw is not None and fill_rate_raw != _FILL_RATE_SENTINEL:
                    doluluk_int = round(fill_rate_raw)
                    if period:
                        data_by_period.setdefault(period, {})[doluluk_name] = doluluk_int
                        doluluk_fallback.append(
                            f"MEMZUC_DOLULUK | dönem: {period} | kalem: {doluluk_name} | doluluk: {doluluk_int}"
                        )

            sub_key = section_type if section_type in memzuc_bilgileri else "grup_ana_firma_memzucu"
            memzuc_bilgileri.setdefault(sub_key, []).extend(detail_rows)

            if data_by_period:
                structured.append({
                    "section_type": section_type,
                    "entity": entity,
                    "data_by_period": data_by_period,
                })

            for all_el in group_el.iter("MemzucGroupeAll"):
                inner_bank = _text(all_el, "BankCountLabel")
                inner_period_label = _text(all_el, "PeriodLabel")
                inner_period = self._extract_period_from_label(inner_period_label)
                if not inner_period:
                    inner_period = period

                for detail in all_el.iter("MemzucGroupedDetail"):
                    risk_code = _text(detail, "RiskCode")
                    fill_rate_raw = _num(detail, "FillRate")
                    total_risk = _num(detail, "TotalRisk")

                    detail_rows.append({
                        "RiskCode": risk_code,
                        "TotalRisk": total_risk,
                        "FillRate": fill_rate_raw,
                        "source_page": 1,
                    })

                    doluluk_name = _normalize_doluluk_row(risk_code)
                    if doluluk_name and fill_rate_raw is not None and fill_rate_raw != _FILL_RATE_SENTINEL:
                        doluluk_int = round(fill_rate_raw)
                        p = inner_period or period
                        if p:
                            existing = data_by_period.get(p, {}).get(doluluk_name)
                            if existing is None or doluluk_int > existing:
                                data_by_period.setdefault(p, {})[doluluk_name] = doluluk_int

        self._add_summary_memzuc(root, memzuc_bilgileri, warnings)

        logger.info(
            "XML memzuc: %d doluluk lines, %d structured sections",
            len(doluluk_fallback),
            len(structured),
        )

        return {
            "memzuc_bilgileri": memzuc_bilgileri,
            "doluluk_fallback": doluluk_fallback,
            "structured": structured,
        }

    def _add_summary_memzuc(
        self, root: ET.Element, memzuc_bilgileri: Dict, warnings: List[str]
    ) -> None:
        """SummaryMainFirmMemzuc / SummaryGroupFirmMemzuc → summary ratios."""
        for tag, label in [
            ("SummaryMainFirmMemzuc", "Ana Firma Memzuç Özeti"),
            ("SummaryGroupFirmMemzuc", "Grup Firma Memzuç Özeti"),
        ]:
            el = root.find(tag)
            if el is None:
                continue
            summary = {"label": label}
            for child in el:
                if child.text and child.text.strip():
                    summary[child.tag] = child.text.strip()
            if len(summary) > 1:
                memzuc_bilgileri.setdefault("grup_ana_firma_memzucu", []).append(summary)

    # --------------------------------------------------------- konsolide

    def _build_konsolide(self, root: ET.Element, warnings: List[str]) -> Dict:
        result: Dict[str, Any] = {}
        for idx, cm in enumerate(root.iter("ConsolidateMemzuc"), 1):
            customer = _text(cm, "CustomerName")
            term = _text(cm, "Term")
            kv = _text(cm, "KVRisk")
            ov = _text(cm, "OVRisk")
            uv = _text(cm, "UVRisk")
            total = _text(cm, "Total")
            key = f"Konsolide #{idx}"
            if customer:
                key = f"{customer} ({term})" if term else customer
            result[key] = f"KV: {kv} | OV: {ov} | UV: {uv} | Toplam: {total}"

        result["source_pages"] = [1]
        return result

    # --------------------------------------------------------- çek performansı

    def _build_cek_performansi(self, root: ET.Element, warnings: List[str]) -> Dict:
        result: Dict[str, Any] = {}

        for idx, dc in enumerate(root.iter("CompanyMoralityDudeCheque"), 1):
            customer = _text(dc, "CustomerName")
            year = _text(dc, "Year")
            paid_count = _text(dc, "PaidCount")
            paid_amount = _text(dc, "PaidTotalAmount")
            non_paid_count = _text(dc, "NonPaidCount")
            non_paid_amount = _text(dc, "NonPaidTotalAmount")
            result[f"Karşılıksız Çek #{idx} ({year})"] = (
                f"Firma: {customer} | Ödenen: {paid_count} adet / {paid_amount} TL | "
                f"Ödenmeyen: {non_paid_count} adet / {non_paid_amount} TL"
            )

        for idx, pb in enumerate(root.iter("CompanyMoralityProtestedBill"), 1):
            customer = _text(pb, "CustomerName")
            year = _text(pb, "Year")
            paid_count = _text(pb, "PaidCount")
            non_paid_count = _text(pb, "NonPaidCount")
            result[f"Protestolu Senet #{idx} ({year})"] = (
                f"Firma: {customer} | Ödenen: {paid_count} | Ödenmeyen: {non_paid_count}"
            )

        for idx, tf in enumerate(root.iter("CompanyMoralityTenderForbidden"), 1):
            customer = _text(tf, "CustomerName")
            year = _text(tf, "Year")
            result[f"İhale Yasaklısı #{idx} ({year})"] = f"Firma: {customer}"

        result["source_pages"] = [1]
        return result

    # --------------------------------------------------------- KKB

    def _build_kkb_riski(self, root: ET.Element, warnings: List[str]) -> List[Dict]:
        rows: List[Dict] = []

        for info in root.iter("IndividualKkbRecordInfo"):
            rows.append({
                "Müşteri No": _text(info, "CustomerNumber"),
                "Müşteri": _text(info, "Customer"),
                "KKB Toplam Limit": _text(info, "KKBTotalLimit"),
                "Kredi Toplam Limit": _text(info, "TotalCreditsLimit"),
                "Toplam KMH Limit": _text(info, "TotalKMHLimit"),
                "Diğer Banka": _text(info, "IndOtherBank"),
                "Kendi Banka": _text(info, "IndOwnBank"),
                "Gecikmiş Bakiye": _text(info, "TotalOverdueBalance"),
                "Dava Verisi": _text(info, "IsLitigationData"),
                "source_page": 1,
            })

        return rows

    def _build_kkb_raporlari(self, root: ET.Element, warnings: List[str]) -> Dict:
        result: Dict[str, Any] = {}

        corp = root.find(".//KkbCorporateReport")
        if corp is not None:
            result["Son Sorgu Tarihi"] = _text(corp, "LastQueryDate")

            summary = corp.find("CorporateSummaryInformation")
            if summary is not None:
                result["Firma Adı"] = _text(summary, "FirmName")
                result["Müşteri No"] = _text(summary, "CustomerNumber")
                result["Vergi No"] = _text(summary, "TaxNumber")

        result["source_pages"] = [1]
        return result

    def _build_limit_risk(self, root: ET.Element, warnings: List[str]) -> Dict:
        """LimitRiskCautionEntity → limit_risk_bilgileri."""
        table: List[Dict] = []

        for lr in root.iter("LimitRiskCautionEntity"):
            table.append({
                "kaynak": _text(lr, "LineSource"),
                "grup_limit": _num(lr, "GroupLimit") or 0,
                "top_limit": _num(lr, "TotalLimit") or 0,
                "nak_limit": _num(lr, "CashLimit") or 0,
                "gn_limit": _num(lr, "NonCashLimit") or 0,
                "grup_risk": _num(lr, "GroupRisk") or 0,
                "top_risk": _num(lr, "TotalRisk") or 0,
                "n_risk": _num(lr, "CashRisk") or 0,
                "gn_risk": _num(lr, "NonCashRisk") or 0,
                "revize": _text(lr, "GeneralReviseDate"),
                "gec_hes_say": _int_val(lr, "LateAccountCount") or 0,
                "gec_hes_top": _num(lr, "LateAccountTotalAmount") or 0,
                "source_pages": [1],
            })

        return {"table": table}
