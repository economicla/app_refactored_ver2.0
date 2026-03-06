"""
CreditIntelligencePDFExtractor — Production-grade structured extractor
for "Kredi İstihbarat Raporu" PDFs.

Extracts section-aware, schema-validated JSON from the specific PDF layout
used by Turkish credit intelligence reports. Handles page spills, table
normalization, and bank-name verification guardrails.

Primary library: pdfplumber (tables).  PyMuPDF fallback for free text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section registry — canonical section IDs, header patterns, display names
# Order matters: first match wins for ambiguous headers.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SectionDef:
    key: str
    display: str
    patterns: Tuple[str, ...]


_SECTION_DEFS: List[_SectionDef] = [
    _SectionDef("ozet_genel_bilgiler",       "Özet / Genel Bilgiler",
                ("özet", "genel bilgi", "rapor özet")),
    _SectionDef("e_haciz_tarihcesi",         "E-Haciz Tarihçesi",
                ("e-haciz", "e haciz", "haciz tarih")),
    _SectionDef("erken_uyari_tarihcesi",     "Erken Uyarı Tarihçesi",
                ("erken uyarı", "erken uyari")),
    _SectionDef("piyasa_istihbarati",        "Piyasa İstihbaratı",
                ("piyasa istihbarat", "piyasa istih")),
    _SectionDef("banka_istihbarati",
                "Banka İstihbaratı",
                ("banka istihbarat", "banka istih",
                "genel limit", "nakit risk", "nakdi risk",
                "g.nakdi", "gayrinakdi", "teminat", "revize")),
    _SectionDef("konsolide_memzuc",          "Konsolide Memzuç",
                ("konsolide memzuç", "konsolide memzuc")),
    _SectionDef("memzuc_bilgileri",          "Memzuç Bilgileri",
                ("memzuç bilgi", "memzuc bilgi", "memzuç", "memzuc")),
    _SectionDef("kkb_riski",                 "KKB Riski",
                ("kkb riski", "kkb risk")),
    _SectionDef("cek_performansi",           "Çek Performansı",
                ("çek performans", "cek performans")),
    _SectionDef("kkb_raporlari",             "KKB Raporları",
                ("kkb rapor",)),
    _SectionDef("kaynak_bazinda_detay",      "Kaynak Bazında Detay (Bin TL)",
                ("kaynak bazında detay", "kaynak bazinda detay")),
    _SectionDef("limit_risk_bilgileri",      "Limit Risk Bilgileri (Bin TL)",
                ("limit risk bilgi", "limit-risk bilgi")),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THOUSAND_SEP_RE = re.compile(r'^-?\d{1,3}(\.\d{3})+(,\d+)?$')
_DECIMAL_RE = re.compile(r'^-?\d+(,\d+)$')
_INTEGER_RE = re.compile(r'^-?\d+$')

_BI_FALLBACK_HEADERS = [
    "banka adı",
    "grup/firma",
    "genel limit",
    "nakit risk",
    "g.n.risk",
    "d.kd.",
    "istih. tarihi",
    "rvz. tar.",
    "statü",
]

def _bi_headers_for_cols(n: int) -> List[str]:
    base = _BI_FALLBACK_HEADERS[:]
    if n <= len(base):
        return base[:n]
    return base + [f"col_{i}" for i in range(len(base), n)]


def _tr_lower(s: str) -> str:
    """Turkish-aware lowercase: İ→i, I→i, ı→i, Ğ→ğ, etc."""
    s = s.replace('İ', 'i').replace('ı', 'i').replace('I', 'i')
    return s.lower()

_BANK_NAME_RE = re.compile(
    r'([A-ZÜÖÇŞİĞa-züöçşığ]'
    r'[A-ZÜÖÇŞİĞa-züöçşığ\s\.\'\-&]{2,80}?'
    r'(?:[Bb]ankası|[Bb]ankas[ıi]|[Bb]ank)'
    r'(?:\s*(?:A\.?\s?Ş\.?|T\.?\s?A\.?\s?Ş\.?))?)',
    re.UNICODE,
)


def _parse_number(raw: Optional[str]) -> Optional[int | float]:
    """Parse a Turkish-locale number cell to int/float. Returns None on failure."""
    if raw is None:
        return None

    s = str(raw).strip()
    if not s:
        return None

    # normalize common "empty" cells
    s = s.replace('\n', '').replace(' ', '')
    if not s or s == '-' or s.lower() == 'none':
        return None

    # handle accounting negatives e.g. "(1.234,56)"
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]

    # remove currency symbols/letters and any non numeric separators
    # keeps digits, ., , and leading minus
    s = re.sub(r'[^0-9\-\.,]', '', s)

    if not s or s == '-' or s == ',' or s == '.':
        return None

    if _THOUSAND_SEP_RE.match(s):
        s = s.replace('.', '').replace(',', '.')
        return _safe_num(s)

    if _DECIMAL_RE.match(s):
        s = s.replace(',', '.')
        return _safe_num(s)

    if _INTEGER_RE.match(s):
        try:
            return int(s)
        except ValueError:
            return None

    # last resort: remove thousand dots, convert decimal comma
    s_clean = s.replace('.', '').replace(',', '.')
    return _safe_num(s_clean)


def _safe_num(s: str) -> Optional[int | float]:
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except (ValueError, OverflowError):
        return None


def _clean_cell(cell: Any) -> str:
    """Normalize a single table cell to a stripped string."""
    if cell is None:
        return ""
    return str(cell).strip().replace('\n', ' ')


def _normalize_bank_name(name: str) -> str:
   if not name:
       return ""
   # newline ve tabları space yap
   name = name.replace("\n", " ").replace("\t", " ")
   # çoklu boşlukları teke indir
   name = re.sub(r"\s+", " ", name)
   # baş/son boşlukları sil
   name = name.strip()
   # son noktayı kaldır
   name = name.rstrip(".")
   # A.Ş varyasyonlarını normalize et
   # (A.Ş., A.Ş, A.S., A S vb. gibi kirli varyasyonlar olabilir)
   name = re.sub(r"\bA\s*\.?\s*Ş\.?\b", "A.Ş", name, flags=re.IGNORECASE)
   return name


def _detect_section(text: str) -> Optional[str]:
    """Return section key if *text* matches a known section header."""
    low = _tr_lower(text).strip()
    for sd in _SECTION_DEFS:
        for pat in sd.patterns:
            if pat in low:
                return sd.key
    return None


def _detect_currency(cell: str) -> str:
    """Guess currency from cell text. Default TRY."""
    up = cell.upper()
    if 'USD' in up or '$' in up:
        return 'USD'
    if 'EUR' in up or '€' in up:
        return 'EUR'
    return 'TRY'


def _money(value: Optional[int | float], currency: str = 'TRY') -> Dict:
    # None -> None kalsın; render tarafında "-" olarak gösterilecek
    return {"value": value, "currency": currency}


_DATE_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b")


def _money_cells_from_row(row: List[str]) -> List[float]:
    """Satırdan para birimi sayılarını topla; tarih ve TRY/TL atlanır (kolon kayması fallback için)."""
    vals: List[float] = []
    for c in row:
        if not c or not str(c).strip():
            continue
        s = str(c).strip()
        if _DATE_RE.fullmatch(s):
            continue
        if s.upper() in ("TRY", "TL"):
            continue
        v = _parse_number(s)
        if v is not None:
            vals.append(float(v))
    return vals


# Text fallback: sayfa tabloları boş/bozuk olduğunda free_text/page_text'ten banka satırı parse et
_BI_TEXT_NUMBER_RE = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?")
_BI_FIRMA_KNOWN = ("MKS MARMARA", "AKTÜL KAĞIT", "BAHARİYE", "GROUP", "MKS Marmara", "Aktül Kağıt")

# Satır formatı: <bank> <firm> <genel> <nakit> <gn> TRY <istih> <rvz> <status> (extract_text örnekleri)
_BI_LINE_RE = re.compile(
    r"^\s*(.+?)\s+(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+)\s+(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+)\s+(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+)\s+TRY\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+(OLUMLU|OLUMSUZ)\s*$",
    re.UNICODE,
)
# Alternative split: bank segment BANKASI veya BANKASI A.Ş. ile biter; başta OLUMLU/OLUMSUZ varsa atla
_BI_ALTERNATIVE_BANK_RE = re.compile(
    r"^(?:\s*(?:OLUMLU|OLUMSUZ)\s+)?(?P<bank>.+?BANKASI(?:\s+A\.Ş\.)?)\s+(?P<rest>.+)$",
    re.IGNORECASE | re.UNICODE,
)
_BI_REST_RE = re.compile(
    r"^\s*(.+?)\s+(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+)\s+(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+)\s+(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+)\s+TRY\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+(OLUMLU|OLUMSUZ)\s*$",
    re.UNICODE,
)
# Bilinen bankalar (bank_name doğrulaması; substring/eq ile eşleşir)
_BI_KNOWN_BANKS = (
    "TÜRKİYE FİNANS",
    "KUVEYT TÜRK",
    "TÜRKİYE EMLAK",
    "ZİRAAT KATILIM",
    "VAKIF KATILIM",
)
# prefix bu kelimelerle başlıyorsa doğrudan alternative split kullan (bank_name OLUMLU ile başlamasın)
_BI_BAD_PREFIXES = ("OLUMLU", "OLUMSUZ", "DURUM", "REVIZE", "TEMINAT")


def _is_valid_bi_bank_name(name: str) -> bool:
    """BANKASI/BANKASI A.Ş. ile bitsin veya bilinen bankalar listesinde olsun; OLUMLU/OLUMSUZ ile başlamamalı."""
    if not name or not name.strip():
        return False
    if _bi_prefix_requires_alternative(name):
        return False
    n = _tr_lower(name).strip()
    if n.endswith("bankasi") or n.endswith("bankasi a.ş."):
        return True
    for b in _BI_KNOWN_BANKS:
        bn = _tr_lower(b).strip()
        if bn in n or n in bn:
            return True
    return False


def _bi_prefix_requires_alternative(prefix: str) -> bool:
    """Prefix OLUMLU/OLUMSUZ/DURUM/REVIZE/TEMINAT ile başlıyorsa alternative split zorunlu."""
    p = prefix.strip().upper()
    return any(p.startswith(bad) for bad in _BI_BAD_PREFIXES)


def _parse_bi_line_alternative(line: str) -> Optional[Dict]:
    """Alternative split: bank segmentini BANKASI (opsiyonel A.Ş.) regex ile bul, kalanı firm + sayılar olarak parse et."""
    line = line.strip()
    m = _BI_ALTERNATIVE_BANK_RE.match(line)
    if not m:
        return None
    bank = m.group("bank").strip()
    rest = m.group("rest").strip()
    m2 = _BI_REST_RE.match(rest)
    if not m2:
        return None
    firm = m2.group(1).strip()
    genel_s, nakit_s, gn_s = m2.group(2), m2.group(3), m2.group(4)
    istih_tarihi, revize_tarihi, status = m2.group(5), m2.group(6), m2.group(7)
    genel_v = _parse_number(genel_s)
    nakit_v = _parse_number(nakit_s)
    gn_v = _parse_number(gn_s)
    genel_v = int(genel_v) if genel_v is not None else 0
    nakit_v = int(nakit_v) if nakit_v is not None else 0
    gn_v = int(gn_v) if gn_v is not None else 0
    bank_name = _normalize_bank_name(bank)
    if not bank_name:
        bank_name = bank
    # bank_name OLUMLU/OLUMSUZ ile başlamamalı (regex yanlış segment yakalasa bile reddet)
    if _bi_prefix_requires_alternative(bank_name):
        return None
    return {
        "bank_name": bank_name,
        "group_or_firm": firm,
        "genel_limit": _money(genel_v, "TRY"),
        "nakit_risk": _money(nakit_v, "TRY"),
        "gn_risk": _money(gn_v, "TRY"),
        "d_kd": _money(None, "TRY"),
        "istih_tarihi": istih_tarihi,
        "revize_tarihi": revize_tarihi,
        "status": status,
        "teminat_sarti": "",
        "notes": [],
    }


def _parse_bi_line(line: str) -> Optional[Dict]:
    """
    Tek satırı parse et: bank firm genel_limit nakit_risk gn_risk TRY istih rvz status.
    bank_name doğrulaması: BANKASI/BANKASI A.Ş. ile bitmeli veya bilinen bankalar listesinde olmalı;
    aksi veya prefix OLUMLU/OLUMSUZ vb. ise alternative split (BANKASI regex ile bank segmenti) uygulanır.
    """
    raw = line.strip()
    m = _BI_LINE_RE.match(raw)
    if not m:
        return None
    prefix = m.group(1).strip()
    # Ek güvenlik: prefix OLUMLU/OLUMSUZ/DURUM/REVIZE/TEMINAT ile başlıyorsa doğrudan alternative
    if _bi_prefix_requires_alternative(prefix):
        return _parse_bi_line_alternative(raw)
    genel_s, nakit_s, gn_s = m.group(2), m.group(3), m.group(4)
    istih_tarihi, revize_tarihi, status = m.group(5), m.group(6), m.group(7)
    genel_v = _parse_number(genel_s)
    nakit_v = _parse_number(nakit_s)
    gn_v = _parse_number(gn_s)
    genel_v = int(genel_v) if genel_v is not None else 0
    nakit_v = int(nakit_v) if nakit_v is not None else 0
    gn_v = int(gn_v) if gn_v is not None else 0
    firm = ""
    bank = prefix
    for known in _BI_FIRMA_KNOWN:
        idx = prefix.rfind(known)
        if idx >= 0:
            firm = known
            bank = prefix[:idx].strip()
            # Birleştirilmiş satırda firm sonrası banka soneki varsa bank'a ekle (TÜRKİYE FİNANS KATILIM BANKASI A.Ş.)
            after_firm = prefix[idx + len(known) :].strip()
            if after_firm:
                alow = _tr_lower(after_firm)
                if alow.endswith("bankasi a.ş.") or alow.endswith("katilim bankasi"):
                    bank = (bank + " " + after_firm).strip()
            break
    bank_name = _normalize_bank_name(bank) if bank else ""
    if not bank_name and prefix:
        bank_name = prefix.strip()
    # bank_name doğrulaması veya OLUMLU/OLUMSUZ ile başlıyorsa alternative split dene
    if not _is_valid_bi_bank_name(bank_name) or _bi_prefix_requires_alternative(bank_name):
        alt = _parse_bi_line_alternative(raw)
        return alt if alt else None
    return {
        "bank_name": bank_name,
        "group_or_firm": firm,
        "genel_limit": _money(genel_v, "TRY"),
        "nakit_risk": _money(nakit_v, "TRY"),
        "gn_risk": _money(gn_v, "TRY"),
        "d_kd": _money(None, "TRY"),
        "istih_tarihi": istih_tarihi,
        "revize_tarihi": revize_tarihi,
        "status": status,
        "teminat_sarti": "",
        "notes": [],
    }


def _bi_record_key(rec: Dict) -> tuple:
    """Duplicate key: bank_name, group_or_firm, istih_tarihi, revize_tarihi (Türkiye Finans AKTÜL vs MKS ayrı kalır)."""
    return (
        rec.get("bank_name", ""),
        rec.get("group_or_firm", ""),
        rec.get("istih_tarihi", ""),
        rec.get("revize_tarihi", ""),
    )


def _is_bi_bank_suffix_line(line: str) -> bool:
    """Sonraki satır sadece banka adı devamı mı (KATILIM BANKASI A.Ş. / BANKASI A.Ş. ile biter)."""
    s = line.strip()
    if not s or len(s) > 60:
        return False
    n = _tr_lower(s)
    return (
        n.endswith("katilim bankasi a.ş.")
        or n.endswith("bankasi a.ş.")
        or n.endswith("katilim bankasi")
    )


def _parse_banka_istihbarati_lines(text: str) -> List[Dict]:
    """
    Metni satır satır tara; BI satır formatına uyan her satırdan bir record üret (duplicate yok).
    Bank adı bir sonraki satıra kaymışsa (örn. 'KATILIM BANKASI A.Ş.') birleştirip tek satır gibi parse et.
    """
    seen: set = set()
    records: List[Dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Sonraki satır banka soneki mi (TÜRKİYE FİNANS ... + KATILIM BANKASI A.Ş.)
        if i + 1 < len(lines) and _is_bi_bank_suffix_line(lines[i + 1]):
            next_line = lines[i + 1].strip()
            m = _BI_LINE_RE.match(stripped)
            if m:
                prefix, g2, g3, g4, g5, g6, g7 = (
                    m.group(1), m.group(2), m.group(3), m.group(4),
                    m.group(5), m.group(6), m.group(7),
                )
                new_prefix = prefix.strip() + " " + next_line
                new_line = f"{new_prefix} {g2} {g3} {g4} TRY {g5} {g6} {g7}"
                rec = _parse_bi_line(new_line)
                if rec:
                    key = _bi_record_key(rec)
                    if key not in seen:
                        seen.add(key)
                        records.append(rec)
                i += 2
                continue
        rec = _parse_bi_line(stripped)
        if not rec:
            i += 1
            continue
        key = _bi_record_key(rec)
        if key in seen:
            i += 1
            continue
        seen.add(key)
        records.append(rec)
        i += 1
    return records


# Normalize edilmiş metinde banka adı (TÜRKİYE FİNANS KATILIM BANKASI A.Ş. → _tr_lower ile eşleşir)
_BI_BANK_NAME_NORM_RE = re.compile(
    r"([a-züöçşığ][a-züöçşığ\s.\'\-\&]{2,79}?bankasi(?:\s*a\.?\s*ş\.?)?)",
    re.UNICODE,
)
# Etiket bazlı sayı: "Genel Limit", "Nakit Risk", "G.N.Risk" vb. sonrası sayı
_BI_LABEL_GENEL = re.compile(
    r"(?:genel\s+limit|Genel\s+Limit)\s*[:\s]*(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)",
    re.IGNORECASE,
)
_BI_LABEL_NAKIT = re.compile(
    r"(?:nakit\s+risk|Nakit\s+Risk)\s*[:\s]*(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)",
    re.IGNORECASE,
)
_BI_LABEL_GN = re.compile(
    r"(?:g\.?\s*n\.?\s*risk|G\.?\s*N\.?\s*Risk|gn\s+risk)\s*[:\s]*(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)",
    re.IGNORECASE,
)


def _ensure_bank_name_norm_matches_turkiye_finans() -> None:
    """TÜRKİYE FİNANS KATILIM BANKASI A.Ş. gibi isimler normalize metinde yakalanmalı."""
    sample = "TÜRKİYE FİNANS KATILIM BANKASI A.Ş."
    norm = _tr_lower(sample)
    if not _BI_BANK_NAME_NORM_RE.search(norm):
        raise ValueError(
            "BI bank name norm regex must match normalized 'TÜRKİYE FİNANS KATILIM BANKASI A.Ş.'; "
            f"normalized={norm!r}"
        )


_ensure_bank_name_norm_matches_turkiye_finans()


def _parse_banka_istihbarati_from_text(text: str, page_num: int) -> List[Dict]:
    """
    Page text'ten (extract_text) Banka İstihbaratı satırlarını çıkar.
    Sadece satır bazlı parser kullanılır; blok parser kapalı (OLUMLU KATILIM BANKASI vb. hatalı kayıt engellenir).
    """
    if not text or not text.strip():
        return []
    text_clean = text.replace("\r", "\n").strip()
    records = _parse_banka_istihbarati_lines(text_clean)
    preview = [
        {
            "bank_name": r["bank_name"],
            "group_or_firm": r.get("group_or_firm"),
            "genel_limit": r["genel_limit"],
            "nakit_risk": r["nakit_risk"],
            "gn_risk": r["gn_risk"],
        }
        for r in records[:2]
    ]
    logger.info(
        "BI text fallback parse: page=%s, records_produced=%s, preview=%s",
        page_num,
        len(records),
        preview,
    )
    return records


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class CreditIntelligencePDFExtractor:
    """
    Extract structured JSON from a Kredi İstihbarat Raporu PDF.

    Usage::

        extractor = CreditIntelligencePDFExtractor()
        data = extractor.extract("/path/to/report.pdf")
    """

    def __init__(self, *, default_currency: str = "TRY"):
        self._default_currency = default_currency

    # ------------------------------------------------------------------ public
    def extract(self, file_path: str) -> Dict:
        """
        Main entry point.

        Returns a dict conforming to the İstihbarat Raporu JSON schema
        (doc_type, meta, sections, debug).
        """
        import pdfplumber

        file_path = str(file_path)
        path = Path(file_path)
        warnings: List[str] = []
        section_hits: List[Dict] = []
        tables_per_page: List[Dict] = []
        spill_merges: List[Dict] = []

        pages_data: List[_PageData] = []

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    pd = self._process_page(page, page_num, warnings)
                    pages_data.append(pd)
                    tables_per_page.append({"page": page_num, "count": pd.table_count})
        except Exception as exc:
            logger.error(f"pdfplumber failed on {path.name}: {exc}")
            warnings.append(f"pdfplumber open failed: {exc}")
            total_pages = 0

        # Assign sections to pages, handling spill merges
        assigned = self._assign_sections(pages_data, section_hits, spill_merges, warnings)

        # Build each section from the assigned data
        sections = self._build_sections(assigned, warnings)

        return {
            "doc_type": "İstihbarat Raporu",
            "meta": {
                "source_file": path.name,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "pages": total_pages,
            },
            "sections": sections,
            "debug": {
                "section_hits": section_hits,
                "tables_found_per_page": tables_per_page,
                "spill_merges": spill_merges,
                "warnings": warnings,
            },
        }

    # --------------------------------------------------------- page processing

    def _process_page(
        self, page, page_num: int, warnings: List[str]
    ) -> "_PageData":
        """Extract text + tables from a single pdfplumber page."""
        tables_raw: List[List[List[str]]] = []
        free_text = ""

        try:
            raw_tables = page.extract_tables() or []
            for tbl in raw_tables:
                cleaned = [[_clean_cell(c) for c in row] for row in tbl if row]
                if cleaned:
                    tables_raw.append(cleaned)
        except Exception as exc:
            warnings.append(f"page {page_num}: table extraction error: {exc}")

        try:
            free_text = (page.extract_text() or "").strip()
        except Exception as exc:
            warnings.append(f"page {page_num}: text extraction error: {exc}")

        detected_header = None
        for line in free_text.split('\n')[:15]:
            sec = _detect_section(line)
            if sec:
                detected_header = sec
                break

        if detected_header is None and tables_raw:
            for tbl in tables_raw[:2]:
                for row in tbl[:30]:
                    combined = ' '.join(row)
                    sec = _detect_section(combined)
                    if sec:
                        detected_header = sec
                        break
                if detected_header:
                    break

        return _PageData(
            page_num=page_num,
            section_header=detected_header,
            tables=tables_raw,
            free_text=free_text,
            table_count=len(tables_raw),
        )

    def _looks_like_banka_istihbarati(self, pd: "_PageData") -> bool:
        needles = ("genel limit", "nakit risk", "nakdi risk", "g.nakdi", "gayrinakdi", "teminat", "revize")
        hay = _tr_lower(pd.free_text or "")
        for tbl in pd.tables[:3]:
            for row in tbl[:50]:
                hay += " " + _tr_lower(" ".join(row))
        score = sum(1 for n in needles if n in hay)
        return score >= 2

    def _looks_like_banka_istihbarati_continuation(self, pd: "_PageData") -> bool:
        """
        Devam sayfası: başlık satırı yok, sadece veri satırları var (banka adı + sayılar).
        En az bir satırda banka adı regex + en az bir sayısal hücre varsa True.
        """
        for tbl in pd.tables:
            for row in tbl:
                if len(row) < 2:
                    continue
                joined = " ".join(str(c) for c in row if c)
                if not _BANK_NAME_RE.search(joined):
                    continue
                has_number = any(_parse_number(c) is not None for c in row if c)
                if has_number:
                    return True
        return False

    def _has_banka_istihbarati_table(self, pd: "_PageData") -> bool:
        """
        High-confidence check: page contains Banka İstihbaratı data.
        Searches ALL rows because pdfplumber may merge Erken Uyarı and
        Banka İstihbaratı into a single table.
        """
        for tbl in pd.tables:
            for row in tbl:
                combined = _tr_lower(' '.join(c for c in row if c))
                if 'banka istihbarat' in combined:
                    return True
                has_bank_col = ('banka ad' in combined or 'banka adi' in combined)
                has_finance_col = ('limit' in combined or 'risk' in combined)
                if has_bank_col and has_finance_col:
                    return True
            for row in tbl:
                joined = ' '.join(c for c in row if c)
                if _BANK_NAME_RE.search(joined):
                    other = _tr_lower(joined)
                    if any(kw in other for kw in ("limit", "risk", "genel limit", "nakit")):
                        return True
        return False

    # ------------------------------------------------------- section assignment

    def _assign_sections(
        self,
        pages: List["_PageData"],
        section_hits: List[Dict],
        spill_merges: List[Dict],
        warnings: List[str],
    ) -> Dict[str, List["_PageData"]]:
        """
        Walk pages in order, assign each to a section.
        If a page has no header, it spills into the previous section.
        Banka İstihbaratı content-based override takes priority when a page
        contains a bank table even if the text header says otherwise.
        """
        assigned: Dict[str, List[_PageData]] = {}
        current_section: Optional[str] = None

        for pd in pages:
            if pd.section_header:
                if (pd.section_header != "banka_istihbarati"
                        and self._has_banka_istihbarati_table(pd)):
                    current_section = "banka_istihbarati"
                    section_hits.append({
                        "page": pd.page_num,
                        "section": current_section,
                        "override_from": pd.section_header,
                    })
                    logger.info(
                        f"📑 Page {pd.page_num}: banka table override "
                        f"'{pd.section_header}' -> 'banka_istihbarati'"
                    )
                    assigned.setdefault(current_section, []).append(pd)
                    continue
                # Devam sayfası: önceki sayfa Banka İstihbaratı idi, bu sayfada da aynı tablo yapısı var
                # (başlıksız devam tablosu = banka adı + sayısal hücre satırları da kabul)
                if (current_section == "banka_istihbarati"
                        and (self._has_banka_istihbarati_table(pd)
                             or self._looks_like_banka_istihbarati(pd)
                             or self._looks_like_banka_istihbarati_continuation(pd))):
                    section_hits.append({
                        "page": pd.page_num,
                        "section": "banka_istihbarati",
                        "continuation": True,
                        "page_header": pd.section_header,
                    })
                    logger.info(
                        f"📑 Page {pd.page_num}: banka_istihbarati devamı (sayfa başlığı: {pd.section_header})"
                    )
                    assigned.setdefault("banka_istihbarati", []).append(pd)
                    continue
                current_section = pd.section_header

            elif self._looks_like_banka_istihbarati(pd) or (
                    current_section == "banka_istihbarati"
                    and self._looks_like_banka_istihbarati_continuation(pd)):
                current_section = "banka_istihbarati"
                section_hits.append({"page": pd.page_num, "section": current_section})
                logger.info(f"📑 Page {pd.page_num}: banka override / devam -> '{current_section}'")
                assigned.setdefault(current_section, []).append(pd)
                continue
            else:
                if current_section:
                    spill_merges.append({
                        "from_page": pd.page_num - 1,
                        "to_page": pd.page_num,
                        "section": current_section,
                    })
                    logger.info(
                        f"📑 Page {pd.page_num}: spill-merge into '{current_section}'"
                    )
                else:
                    current_section = "ozet_genel_bilgiler"
                    warnings.append(
                        f"page {pd.page_num}: no section detected, defaulting to ozet"
                    )

            assigned.setdefault(current_section, []).append(pd)

        return assigned

    # --------------------------------------------------------- section builders

    def _build_sections(
        self,
        assigned: Dict[str, List["_PageData"]],
        warnings: List[str],
    ) -> Dict:
        sections: Dict = {}

        builder_map = {
            "ozet_genel_bilgiler": self._build_kv_section,
            "e_haciz_tarihcesi": self._build_e_haciz,
            "erken_uyari_tarihcesi": self._build_list_section,
            "piyasa_istihbarati": self._build_kv_section,
            "banka_istihbarati": self._build_banka_istihbarati,
            "memzuc_bilgileri": self._build_memzuc,
            "konsolide_memzuc": self._build_kv_section,
            "kkb_riski": self._build_list_section,
            "cek_performansi": self._build_kv_section,
            "kkb_raporlari": self._build_kv_section,
            "limit_risk_bilgileri": self._build_limit_risk,
            "kaynak_bazinda_detay": self._build_kaynak_bazinda,
        }

        for sec_key, pages in assigned.items():
            builder = builder_map.get(sec_key, self._build_kv_section)
            try:
                sections[sec_key] = builder(pages, warnings)
            except Exception as exc:
                warnings.append(f"section '{sec_key}' build error: {exc}")
                sections[sec_key] = {"error": str(exc)}

        return sections

    # ---- generic builders ---

    def _build_kv_section(
        self, pages: List["_PageData"], warnings: List[str]
    ) -> Dict:
        """Key-value extraction from tables + free text."""
        result: Dict[str, Any] = {}
        for pd in pages:
            for tbl in pd.tables:
                for row in tbl:
                    non_empty = [c for c in row if c]
                    if len(non_empty) == 2:
                        result[non_empty[0]] = non_empty[1]
                    elif len(non_empty) > 2:
                        result[non_empty[0]] = non_empty[1:]
            if not pd.tables and pd.free_text:
                for line in pd.free_text.split('\n'):
                    if ':' in line:
                        k, _, v = line.partition(':')
                        result[k.strip()] = v.strip()
        result["source_pages"] = [p.page_num for p in pages]
        return result

    def _build_list_section(
        self, pages: List["_PageData"], warnings: List[str]
    ) -> List[Dict]:
        """Return table rows as list of dicts (first row = headers)."""
        all_rows: List[Dict] = []
        for pd in pages:
            for tbl in pd.tables:
                headers = self._detect_header_row(tbl)
                data_rows = tbl[1:] if headers else tbl
                if not headers:
                    headers = [f"col_{i}" for i in range(len(tbl[0]))]
                for row in data_rows:
                    entry: Dict[str, Any] = {}
                    for i, h in enumerate(headers):
                        val = row[i] if i < len(row) else ""
                        num = _parse_number(val)
                        entry[h] = num if num is not None else val
                    entry["source_page"] = pd.page_num
                    all_rows.append(entry)
        return all_rows

    # ---- specialized builders ---

    _E_HACIZ_KNOWN_HEADERS = [
        "sicil_no", "unvan", "yil", "odenen_adet", "odenen_tutar",
        "odenmeyen_adet", "odenmeyen_tutar",
    ]

    def _build_e_haciz(
        self, pages: List["_PageData"], warnings: List[str]
    ) -> List[Dict]:
        """
        Parse E-Haciz Tarihçesi with known column semantics.
        Falls back to generic list if column count doesn't match.
        """
        all_rows: List[Dict] = []
        for pd in pages:
            for tbl in pd.tables:
                detected_headers = self._detect_header_row(tbl)
                if detected_headers:
                    data_rows = tbl[1:]
                    headers = detected_headers
                else:
                    first_row = tbl[0] if tbl else []
                    col_count = len(first_row)

                    if col_count <= len(self._E_HACIZ_KNOWN_HEADERS) + 1:
                        headers = self._E_HACIZ_KNOWN_HEADERS[:col_count]
                        looks_like_header = (
                            first_row
                            and any(
                                kw in ' '.join(first_row).lower()
                                for kw in ('unvan', 'ünvan', 'adet', 'tutar', 'firma')
                            )
                        )
                        data_rows = tbl[1:] if looks_like_header else tbl
                    else:
                        headers = [f"col_{i}" for i in range(col_count)]
                        data_rows = tbl

                for row in data_rows:
                    if all(not c for c in row):
                        continue
                    entry: Dict[str, Any] = {}
                    for i, h in enumerate(headers):
                        val = row[i] if i < len(row) else ""
                        num = _parse_number(val)
                        entry[h] = num if num is not None else val
                    entry["source_page"] = pd.page_num
                    all_rows.append(entry)
        return all_rows

    @staticmethod
    def _find_bi_subtable(tbl: List[List[str]]) -> List[List[str]]:
        """
        If a table spans multiple sections (e.g. Erken Uyarı + Banka İstihbaratı),
        find where the Banka İstihbaratı sub-table starts and return only that part.
        """
        for i, row in enumerate(tbl):
            combined = _tr_lower(' '.join(c for c in row if c))
            if 'banka istihbarat' in combined and len(combined.strip()) < 60:
                return tbl[i + 1:]
            if ('banka ad' in combined or 'banka adi' in combined):
                if 'limit' in combined or 'risk' in combined:
                    return tbl[i:]
        return tbl

    def _build_banka_istihbarati(
        self, pages: List["_PageData"], warnings: List[str]
    ) -> List[Dict]:
        """
        Parse the Banka İstihbaratı section: one record per bank.
        BI için sadece text fallback kullanılır (table parse devre dışı — bozuk kayıt/duplicate önlenir).
        """
        records: List[Dict] = []
        source_pages = [p.page_num for p in pages]
        existing_keys: set = set()

        for pd in pages:
            # Page text: free_text (pdfplumber extract_text) veya page_text/raw_text
            text = (
                getattr(pd, "free_text", None)
                or getattr(pd, "raw_text", None)
                or getattr(pd, "page_text", None)
            )
            if not (text and str(text).strip()):
                logger.warning(
                    "BI text fallback: page %s has no free_text/raw_text/page_text — skip",
                    getattr(pd, "page_num", "?"),
                )
                continue
            text_recs = _parse_banka_istihbarati_from_text(text, pd.page_num)
            for rec in text_recs:
                key = _bi_record_key(rec)
                if key not in existing_keys:
                    rec["source_pages"] = source_pages
                    records.append(rec)
                    existing_keys.add(key)

        return records

    @staticmethod
    def _scan_sub_row_cells(rec: Dict, row: List[str]) -> None:
        """
        Scan each cell of a non-bank row for teminat, kefil, and note data.
        Works cell-by-cell to handle pdfplumber's cell splitting.
        """
        matched_any = False

        for cell in row:
            if not cell or not cell.strip():
                continue
            c = cell.strip()
            c_lower = _tr_lower(c)

            if 'teminat' in c_lower and ('sart' in c_lower or 'şart' in c_lower):
                parts = re.split(r'[:\s]+', c, maxsplit=1)
                if len(parts) > 1:
                    val = parts[1].strip().rstrip('.')
                else:
                    val = c
                m = re.search(r'(?:teminat\s*(?:[şs]art[ıi]?)?)\s*[:\s]+\s*(.+)',
                              c, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip('.')
                if val and len(val) > 1:
                    existing = rec.get("teminat_sarti", "")
                    rec["teminat_sarti"] = f"{existing}; {val}" if existing else val
                    matched_any = True

            elif 'alinan teminat' in c_lower or 'alınan teminat' in c_lower:
                m = re.search(r'al[ıi]nan\s+teminat\s*[:\s]+\s*(.*)',
                              c, re.IGNORECASE)
                if m:
                    val = m.group(1).strip()
                    if val:
                        rec["alinan_teminat"] = val
                        matched_any = True

            elif 'alinan kefil' in c_lower or 'alınan kefil' in c_lower:
                m = re.search(r'al[ıi]nan\s+kefil\s*[:\s]+\s*(.*)',
                              c, re.IGNORECASE)
                if m:
                    val = m.group(1).strip()
                    if val:
                        rec["alinan_kefil"] = val
                        matched_any = True

        if not matched_any:
            joined = ' '.join(c for c in row if c and c.strip()).strip()
            if joined and len(joined) > 5:
                rec.setdefault("notes", [])
                rec["notes"].append(joined)

    def _parse_bank_row(
        self,
        row: List[str],
        h_lower: List[str],
        headers: List[str],
        page_num: int,
    ) -> Optional[Dict]:
        """Map a single table row to a banka_istihbarati record."""

        def _col(names: Tuple[str, ...]) -> Optional[str]:
            for n in names:
                for i, h in enumerate(h_lower):
                    if n in h and i < len(row):
                        return row[i]
            return None

        raw_name = _col(("banka adı", "banka adi", "banka"))
        if not raw_name:
            joined = " ".join(row)
            m = _BANK_NAME_RE.search(joined)
            if m:
                raw_name = m.group(1)
            else:
                return None

        bank_name = _normalize_bank_name(raw_name)
        if len(bank_name) < 4:
            return None

        cur = self._default_currency

        rec = {
            "bank_name": bank_name,
            "group_or_firm": _col(("grup", "firma")) or "",
            "genel_limit": _money(_parse_number(_col(("genel limit",))), cur),
            "nakit_risk": _money(
                _parse_number(_col(("nakit risk", "nakdi risk", "n.risk", "n risk"))),
                cur,
            ),
            "gn_risk": _money(
                _parse_number(_col((
                    "g.n.risk", "g.n. risk", "g.n risk",
                    "gn risk", "gayrinakdi risk", "g.nakdi",
                    "gayri nakdi",
                ))),
                cur,
            ),
            "d_kd": _money(_parse_number(_col(("döviz", "d.kd", "d kd", "d.kd."))), cur),
            "istih_tarihi": _col(("istih tarihi", "istihbarat tarihi", "istih.", "istih")) or "",
            "revize_tarihi": _col(("rvz", "revize", "revize tarihi", "rvz.")) or "",
            "status": _col(("durum", "status", "statü", "stat")) or "",
            "teminat_sarti": _col(("teminat", "teminat şartı", "teminat sarti")) or "",
            "notes": [],
        }

        # FALLBACK: kolon kayması / header yoksa satırdan sayıları topla (sayfa 5 vb.)
        if rec["genel_limit"].get("value") is None and rec["nakit_risk"].get("value") is None:
            compact = [str(x).strip() for x in row if x and str(x).strip()]
            nums = _money_cells_from_row(compact)
            if len(nums) >= 2:
                rec["genel_limit"]["value"] = nums[0]
                rec["nakit_risk"]["value"] = nums[1]
            if len(nums) >= 3 and rec["gn_risk"].get("value") is None:
                rec["gn_risk"]["value"] = nums[2]
            if not rec["group_or_firm"] and len(compact) >= 2:
                rec["group_or_firm"] = compact[1]
            if not rec["status"]:
                for t in compact:
                    if str(t).upper() in ("OLUMLU", "OLUMSUZ"):
                        rec["status"] = str(t).upper()
                        break

        return rec

    @staticmethod
    def _verify_bank_in_page(
        bank_name: str, pages: List["_PageData"], expected_page: int
    ) -> bool:
        """Guardrail: verify bank_name appears in the raw text of its source page."""
        search_key = _tr_lower(bank_name)
        search_tokens = set(re.findall(r'[a-züöçşığ]{3,}', search_key))
        search_tokens -= {"bankasi", "banka", "bank", "turkiye", "t.c", "tc", "a", "s", "as", "tas"}

        for pd in pages:
            if pd.page_num == expected_page:
                text_lower = _tr_lower(pd.free_text)
                # tablo metnini de ekle
                for tbl in pd.tables[:3]:
                    for row in tbl[:50]:
                        text_lower += " " + _tr_lower(" ".join(row))
                        
                if search_key in text_lower:
                    return True
                matched = sum(1 for t in search_tokens if t in text_lower)
                return len(search_tokens) > 0 and matched >= len(search_tokens) * 0.6
        return False

    def _build_memzuc(
        self, pages: List["_PageData"], warnings: List[str]
    ) -> Dict:
        """
        Memzuç has sub-sections: grup/ana firma, kredi grubu, ortağı olduğu, KKB ilişkili.
        Detect sub-headers inside the tables and split accordingly.
        """
        sub_keys = {
            "grup_ana_firma_memzucu": ("grup", "ana firma"),
            "kredi_grubu_firma_memzuclari": ("kredi grubu",),
            "ortagi_oldugu_sirketlerin_memzuclari": ("ortağı olduğu", "ortagi oldugu"),
            "kkb_ekrani_iliskili_firma_memzuclari": ("kkb ekranı", "kkb ekrani", "ilişkili"),
        }
        result: Dict[str, List] = {k: [] for k in sub_keys}
        current_sub: Optional[str] = None

        for pd in pages:
            for tbl in pd.tables:
                headers_detected = False
                effective_headers: Optional[List[str]] = None

                for row in tbl:
                    combined = ' '.join(row).lower()

                    matched_sub = None
                    for sk, patterns in sub_keys.items():
                        if any(p in combined for p in patterns):
                            matched_sub = sk
                            break

                    if matched_sub:
                        current_sub = matched_sub
                        headers_detected = False
                        effective_headers = None
                        continue

                    if not current_sub:
                        current_sub = "grup_ana_firma_memzucu"

                    if not headers_detected:
                        non_numeric = sum(
                            1 for c in row if c and _parse_number(c) is None
                        )
                        if non_numeric >= len(row) * 0.5 and len(row) >= 2:
                            effective_headers = [c if c else f"col_{i}" for i, c in enumerate(row)]
                            headers_detected = True
                            continue

                    if all(not c for c in row):
                        continue

                    entry: Dict[str, Any] = {}
                    if effective_headers:
                        for i, h in enumerate(effective_headers):
                            val = row[i] if i < len(row) else ""
                            num = _parse_number(val)
                            entry[h] = num if num is not None else val
                    else:
                        for i, c in enumerate(row):
                            num = _parse_number(c)
                            entry[f"col_{i}"] = num if num is not None else c

                    entry["source_page"] = pd.page_num
                    result[current_sub].append(entry)

        return result

    def _build_limit_risk(
        self, pages: List["_PageData"], warnings: List[str]
    ) -> Dict:
        """Parse the Limit Risk Bilgileri (Bin TL) table."""
        table_rows: List[Dict] = []
        target_cols = (
            "kaynak", "grup_limit", "top_limit", "nak_limit", "gn_limit",
            "grup_risk", "top_risk", "n_risk", "gn_risk",
            "revize", "gec_hes_say", "gec_hes_top",
        )

        for pd in pages:
            for tbl in pd.tables:
                headers = self._detect_header_row(tbl)
                data_start = 1 if headers else 0

                for row in tbl[data_start:]:
                    if all(not c for c in row):
                        continue
                    entry: Dict[str, Any] = {}
                    for i, col_name in enumerate(target_cols):
                        raw = row[i] if i < len(row) else ""
                        if col_name in ("kaynak", "revize"):
                            entry[col_name] = raw
                        else:
                            entry[col_name] = _parse_number(raw) or 0
                    entry["source_pages"] = [pd.page_num]
                    table_rows.append(entry)

        return {"table": table_rows}

    def _build_kaynak_bazinda(
        self, pages: List["_PageData"], warnings: List[str]
    ) -> List[Dict]:
        """Parse Kaynak Bazında Detay (Bin TL) section."""
        records: List[Dict] = []

        for pd in pages:
            for tbl in pd.tables:
                headers = self._detect_header_row(tbl)
                data_start = 1 if headers else 0

                for row in tbl[data_start:]:
                    if all(not c for c in row):
                        continue

                    kaynak = row[0] if row else ""
                    bank_name_raw = row[1] if len(row) > 1 else ""

                    bank_name: Optional[str] = None
                    m = _BANK_NAME_RE.search(bank_name_raw)
                    if m:
                        bank_name = _normalize_bank_name(m.group(1))

                    num_cols = (
                        "grup_limit", "toplam_limit", "nakit_limit",
                        "g_nakdi_limit", "grup_risk", "toplam_risk",
                        "nakit_risk", "g_nakdi_risk",
                    )
                    entry: Dict[str, Any] = {
                        "kaynak": kaynak,
                        "bank_name": bank_name,
                    }
                    offset = 2
                    for j, col_name in enumerate(num_cols):
                        idx = offset + j
                        raw = row[idx] if idx < len(row) else ""
                        entry[col_name] = _parse_number(raw) or 0

                    revize_idx = offset + len(num_cols)
                    entry["revize_tarihi"] = row[revize_idx] if revize_idx < len(row) else ""

                    teminat_keys = (
                        "ipotek", "cek", "kefalet", "genel_toplam",
                        "diger_kefalet", "kambiyo_senetleri", "nakit_rehni",
                    )
                    teminat: Dict[str, int | float] = {}
                    t_offset = revize_idx + 1
                    for k, tk in enumerate(teminat_keys):
                        tidx = t_offset + k
                        raw = row[tidx] if tidx < len(row) else ""
                        teminat[tk] = _parse_number(raw) or 0
                    entry["teminat_kirilimi"] = teminat
                    entry["source_pages"] = [pd.page_num]
                    records.append(entry)

        return records

    # ------------------------------------------------------------- table utils

    @staticmethod
    def _detect_header_row(table: List[List[str]]) -> Optional[List[str]]:
        """
        Heuristic: if the first row's non-empty cells are predominantly
        non-numeric text, treat it as the header.
        Ignores empty cells in the ratio calculation so sparse PDF tables
        (merged cells) are correctly detected.
        """
        if not table:
            return None
        first = table[0]
        non_empty = [c for c in first if c and c.strip()]
        if not non_empty or len(first) < 2:
            return None
        non_numeric = sum(1 for c in non_empty if _parse_number(c) is None)
        if non_numeric >= len(non_empty) * 0.5:
            return [c if c else f"col_{i}" for i, c in enumerate(first)]
        return None


# ---------------------------------------------------------------------------
# Internal page-data container
# ---------------------------------------------------------------------------

@dataclass
class _PageData:
    page_num: int
    section_header: Optional[str]
    tables: List[List[List[str]]]
    free_text: str
    table_count: int


# ---------------------------------------------------------------------------
# render_structured_text — converts JSON → RAG-ready controlled text
# ---------------------------------------------------------------------------

def render_structured_text(data: Dict) -> str:
    """
    Convert a CreditIntelligencePDFExtractor output dict into a flat,
    section-headed text suitable for RAG chunking.

    Each section gets a markdown heading. Tables become key:value lines.
    Bank names are explicit. No information is invented.
    """
    from typing import Optional

    def _fmt_money(val: Optional[int | float]) -> str:
        if val is None:
            return "-"
        try:
            if isinstance(val, int):
                return f"{val:,}"
            if isinstance(val, float):
                if val.is_integer():
                    return f"{int(val):,}"
                return f"{val:,.2f}"
            return str(val)
        except Exception:
            return str(val)

    parts: List[str] = []
    sections = data.get("sections", {})
    meta = data.get("meta", {})

    parts.append(f"# İstihbarat Raporu — {meta.get('source_file', '?')}")
    parts.append(f"Sayfa sayısı: {meta.get('pages', '?')}\n")

    # Özet / Genel Bilgiler
    ozet = sections.get("ozet_genel_bilgiler", {})
    if ozet:
        parts.append("## Özet / Genel Bilgiler")
        for k, v in ozet.items():
            if k == "source_pages":
                continue
            parts.append(f"{k}: {v}")
        parts.append("")

    # E-Haciz — firma bazlı okunabilir format
    e_haciz = sections.get("e_haciz_tarihcesi", [])
    if e_haciz and isinstance(e_haciz, list):
        parts.append("## E-Haciz Tarihçesi")
        for entry in e_haciz:
            unvan = entry.get("unvan", entry.get("col_1", "?"))
            yil = entry.get("yil", entry.get("col_2", "?"))
            od_adet = entry.get("odenen_adet", entry.get("col_3", "-"))
            od_tutar = entry.get("odenen_tutar", entry.get("col_4", "-"))
            odn_adet = entry.get("odenmeyen_adet", entry.get("col_5", "-"))
            odn_tutar = entry.get("odenmeyen_tutar", entry.get("col_6", "-"))
            parts.append(
                f"Firma: {unvan} | Yıl: {yil} | "
                f"Ödenen Haciz: {od_adet} adet, {_fmt_money(od_tutar) if isinstance(od_tutar, (int, float)) else od_tutar} TL | "
                f"Ödenmeyen Haciz: {odn_adet} adet, {_fmt_money(odn_tutar) if isinstance(odn_tutar, (int, float)) else odn_tutar} TL"
            )
        parts.append("")

    # Erken Uyarı
    _render_table_section(parts, sections, "erken_uyari_tarihcesi", "Erken Uyarı Tarihçesi")

    # Piyasa İstihbaratı
    piyasa = sections.get("piyasa_istihbarati", {})
    if piyasa:
        parts.append("## Piyasa İstihbaratı")
        for k, v in piyasa.items():
            if k == "source_pages":
                continue
            parts.append(f"{k}: {v}")
        parts.append("")

    # Banka İstihbaratı — line per bank
    banka_list = (
        sections.get("banka_istihbarati")
        or sections.get("banka_istihbaratı")
        or sections.get("bankaIstihbarati")
        or sections.get("bank_istihbarati")
        or []
    )
    if banka_list:
        parts.append("## Banka İstihbaratı")
        for rec in banka_list:
            name = rec.get("bank_name", "?")

            gl_obj = rec.get("genel_limit", {}) or {}
            nr_obj = rec.get("nakit_risk", {}) or {}
            gnr_obj = rec.get("gn_risk", {}) or {}

            gl = gl_obj.get("value", None)
            nr = nr_obj.get("value", None)
            gnr = gnr_obj.get("value", None)

            cur = gl_obj.get("currency", "TRY")

            teminat = rec.get('teminat_sarti', '-') or '-'
            alinan_tem = rec.get('alinan_teminat', '')
            alinan_kefil = rec.get('alinan_kefil', '')
            notes = rec.get('notes', [])

            firma = rec.get('group_or_firm', '').strip()
            firma_part = f" | Firma: {firma}" if firma else ""
            status = rec.get('status', '').strip()
            status_part = f" | Durum: {status}" if status else ""

            parts.append(
                f"Banka: {name}{firma_part} | Genel Limit: {_fmt_money(gl)} {cur} | "
                f"Nakit Risk: {_fmt_money(nr)} {cur} | G.Nakdi Risk: {_fmt_money(gnr)} {cur} | "
                f"Teminat Şartı: {teminat} | "
                f"Alınan Teminat: {alinan_tem if alinan_tem else '-'} | "
                f"Alınan Kefil: {alinan_kefil if alinan_kefil else '-'} | "
                f"Revize: {rec.get('revize_tarihi', '-')}{status_part}"
            )
            if notes:
                for note in notes:
                    parts.append(f"  Not: {note}")
        parts.append("")

    # Memzuç
    memzuc = sections.get("memzuc_bilgileri", {})
    if memzuc and isinstance(memzuc, dict):
        parts.append("## Memzuç Bilgileri")
        for sub_key, sub_list in memzuc.items():
            if not isinstance(sub_list, list) or not sub_list:
                continue
            parts.append(f"### {sub_key}")
            for entry in sub_list:
                line = " | ".join(
                    f"{k}: {v}" for k, v in entry.items() if k != "source_page"
                )
                parts.append(line)
        parts.append("")

    # Konsolide Memzuç
    kons = sections.get("konsolide_memzuc", {})
    if kons:
        parts.append("## Konsolide Memzuç / KKB Riski")
        for k, v in kons.items():
            if k == "source_pages":
                continue
            parts.append(f"{k}: {v}")
        parts.append("")

    # KKB Riski
    _render_table_section(parts, sections, "kkb_riski", "KKB Riski")

    # Çek Performansı
    cek = sections.get("cek_performansi", {})
    if cek:
        parts.append("## Çek Performansı")
        for k, v in cek.items():
            if k == "source_pages":
                continue
            parts.append(f"{k}: {v}")
        parts.append("")

    # KKB Raporları
    kkb_rap = sections.get("kkb_raporlari", {})
    if kkb_rap:
        parts.append("## KKB Raporları")
        for k, v in kkb_rap.items():
            if k == "source_pages":
                continue
            parts.append(f"{k}: {v}")
        parts.append("")

    # Limit Risk Bilgileri
    lr = sections.get("limit_risk_bilgileri", {})
    if lr and lr.get("table"):
        parts.append("## Limit Risk Bilgileri (Bin TL)")
        for row in lr["table"]:
            line = " | ".join(
                f"{k}: {v}" for k, v in row.items() if k != "source_pages"
            )
            parts.append(line)
        parts.append("")

    # Kaynak Bazında Detay
    kbd = sections.get("kaynak_bazinda_detay", [])
    if kbd:
        parts.append("## Kaynak Bazında Detay (Bin TL)")
        for row in kbd:
            teminat = row.get("teminat_kirilimi", {})
            base = " | ".join(
                f"{k}: {v}" for k, v in row.items()
                if k not in ("source_pages", "teminat_kirilimi")
            )
            tem_str = ", ".join(f"{tk}: {tv}" for tk, tv in teminat.items())
            parts.append(f"{base} | Teminat: [{tem_str}]")
        parts.append("")

    return "\n".join(parts)


def _render_table_section(
    parts: List[str], sections: Dict, key: str, title: str
) -> None:
    """Helper: render a list-of-dicts section as text lines."""
    data = sections.get(key, [])
    if not data or not isinstance(data, list):
        return
    parts.append(f"## {title}")
    for entry in data:
        line = " | ".join(
            f"{k}: {v}" for k, v in entry.items() if k != "source_page"
        )
        parts.append(line)
    parts.append("")
