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
# Memzuc Doluluk Oranı: extract_words (koordinat) öncelikli, text fallback yedek
# ---------------------------------------------------------------------------

_MEMZUC_DOLULUK_SIGNALS = (
    "KREDİ GRUBU FİRMA MEMZUÇ",
    "KREDİ GRUBU FİRMA MEMZUC",
    "MEMZUÇ DOLULUK",
    "MEMZUC DOLULUK",
    "Doluluk Oranı",
)
_MEMZUC_DOLULUK_PERIOD_RE = re.compile(r"(20\d{2}\s*/\s*\d{1,2})")
_MEMZUC_LINE_RE = re.compile(
    r"MEMZUC_DOLULUK\s*\|\s*dönem:\s*(\S+)\s*\|\s*kalem:\s*([^|]+?)\s*\|\s*doluluk:\s*(\d+)",
    re.IGNORECASE,
)
_MEMZUC_DOLULUK_ROW_NAMES = (
    "Umumi Limit",
    "Toplam Nakdi Kredi",
    "Toplam GN Kredi",
    "TOPLAM",
)
# Satır gruplama: aynı y'ye yakın kelimeler aynı satır (tolerance pt)
_MEMZUC_ROW_Y_TOLERANCE = 4
# Crop A: ANA FİRMA DAHİL tablosu — tüm doluluk satırları (Toplam Nakdi/GN, TOPLAM, Umumi Limit) burada
_MEMZUC_CROP_A_HEIGHT = 580
# Crop B: üst özet bandı max sayfa yüksekliğinin %30'u ile sınırlı
_MEMZUC_CROP_B_MAX_FRAC = 0.30
_MEMZUC_CROP_B_ABOVE_HEADER = 10
# Crop B: doluluk sütunu sağ bant width * ratio (0.70 test; 0.85 hâlâ 2/7 veriyorsa bbox sorunu)
_CROP_B_DOLULUK_BAND_RATIO = 0.70
# True ise Crop B/A header_top'tan bağımsız: B = üst %25, A = %25..%70 (yanlış header_top için)
_MEMZUC_USE_PERCENTAGE_CROPS = False
_MEMZUC_CROP_B_TOP_FRAC = 0.25
_MEMZUC_CROP_A_TOP_FRAC = 0.25
_MEMZUC_CROP_A_BOTTOM_FRAC = 0.70
# Başlık: tam metin "KREDİ GRUBU FİRMA MEMZUÇLARI (ANA FİRMA DAHİL) (BankaSayisi: 9)" — bölünmüş satırda da bulunsun diye ANA FİRMA DAHİL de aranır
_MEMZUC_HEADER_FOR_CROP = (
    "KREDİ GRUBU FİRMA MEMZUCULARI",
    "KREDİ GRUBU FİRMA MEMZUÇLARI",
    "MEMZUÇLARI (ANA FİRMA DAHİL)",
    "MEMZUCULARI (ANA FİRMA DAHİL)",
    "ANA FİRMA DAHİL",
)
_MEMZUC_HEADER_PATTERNS = (
    "KREDİ GRUBU FİRMA MEMZUCULARI",
    "KREDİ GRUBU FİRMA MEMZUÇLARI",
    "ANA FİRMA DAHİL",
)
# Crop A: ana tablodan dört kalemin hepsi (doğru Doluluk Oranı sütunu burada)
_CROP_A_ROW_NAMES = ("Umumi Limit", "Toplam Nakdi Kredi", "Toplam GN Kredi", "TOPLAM")
# Crop B: üst özet bandı — aynı kalemler farklı sayılara sahip; Crop A öncelikli (merge'de max alınır)
_CROP_B_ROW_NAMES = ("TOPLAM", "Umumi Limit")
# Devam sayfası: tablo sayfa sonunda kesilirse TOPLAM/Umumi Limit sonraki sayfanın üstünde olabilir
_MEMZUC_CONTINUATION_CROP_HEIGHT = 280
_DEFAULT_MEMZUC_PERIOD = "2025/12"


def _find_memzuc_header_top(words: List[Dict]) -> Optional[float]:
    """
    Sayfa kelimelerinde 'KREDİ GRUBU FİRMA MEMZUCULARI' başlık satırını bulur,
    o satırın top değerini döndürür (Crop A/B ayırımı için).
    """
    tol = _MEMZUC_ROW_Y_TOLERANCE
    by_row: Dict[float, List[Dict]] = {}
    for w in words:
        text = (w.get("text") or "").strip()
        if not text:
            continue
        top = float(w.get("top", 0))
        row_key = round(top / tol) * tol
        by_row.setdefault(row_key, []).append(w)
    for rk in sorted(by_row.keys()):
        row_words = by_row[rk]
        line_text = " ".join((w.get("text") or "").strip() for w in row_words).upper()
        for pat in _MEMZUC_HEADER_FOR_CROP:
            if pat in line_text or pat.replace("İ", "I").replace("Ç", "C") in line_text:
                return min(float(w.get("top", 0)) for w in row_words)
    return None


def _period_from_page_text(page) -> str:
    """
    Memzuc sayfasında dönemi crop içinden aramıyoruz; sayfa full_text'inden max(20xx/yy) alınır.
    Böylece 10 ve 13 gibi başka dönemin değerleri yanlış döneme yazılmaz. Bulunamazsa 2025/12.
    """
    try:
        text = page.extract_text() or ""
    except Exception:
        return _DEFAULT_MEMZUC_PERIOD
    periods = [
        (m.group(1) or "").replace(" ", "").strip()
        for m in _MEMZUC_DOLULUK_PERIOD_RE.finditer(text)
    ]
    periods = [p for p in periods if p]
    return max(periods) if periods else _DEFAULT_MEMZUC_PERIOD


def _find_period_positions_from_words(
    words: List[Dict], min_top: float, max_top: float
) -> List[Tuple[float, str]]:
    """
    Kelimeler arasında 20xx/yy formatındaki dönem etiketlerini bulur; (top, period) listesi döner (top'a göre sıralı).
    Aynı dönem birden fazla kelimede geçerse en üstteki (min top) kullanılır.
    """
    seen: Dict[str, float] = {}
    for w in words:
        top = float(w.get("top", 0))
        if not (min_top <= top <= max_top):
            continue
        txt = (w.get("text") or "").strip().replace(" ", "")
        m = _MEMZUC_DOLULUK_PERIOD_RE.match(txt) or _MEMZUC_DOLULUK_PERIOD_RE.search(txt)
        if not m:
            continue
        period = (m.group(1) or "").replace(" ", "").strip()
        if not period or len(period) < 6:
            continue
        if period not in seen or top < seen[period]:
            seen[period] = top
    out = [(top, p) for p, top in seen.items()]
    out.sort(key=lambda x: x[0])
    return out


def _merge_memzuc_lines_across_pages(lines: List[str]) -> List[str]:
    """
    Birden fazla sayfadan gelen MEMZUC_DOLULUK satırlarını (dönem, kalem) bazında birleştirir;
    aynı (dönem, kalem) için doluluk = max(doluluk). Böylece chunk'lara dağılan 2/7/0/28/39
    yerine tek satırda 20/22 döner.
    """
    merged: Dict[Tuple[str, str], int] = {}
    for line in lines:
        line = (line or "").strip()
        m = _MEMZUC_LINE_RE.search(line)
        if not m:
            continue
        period = (m.group(1) or "").strip()
        kalem = (m.group(2) or "").strip()
        try:
            doluluk = int((m.group(3) or "").strip())
        except ValueError:
            continue
        if not period or not kalem:
            continue
        key = (period, kalem)
        merged[key] = max(merged.get(key, 0), doluluk)
    return [
        f"MEMZUC_DOLULUK | dönem: {p} | kalem: {rn} | doluluk: {d}"
        for (p, rn), d in merged.items()
    ]


def _find_doluluk_column_x0(words: List[Dict]) -> Optional[float]:
    """
    extract_words içinde 'Doluluk' / 'Doluluk Oranı' başlığını bulur.
    Sayfada iki tablo (2024/12, 2025/12) varsa en sağdaki tablonun sütununu al: max(x0).
    Böylece 20/22 doğru sütundan okunur (sol tablodaki 2/7 değil).
    """
    x0_candidates: List[float] = []
    for w in words:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue
        low = _tr_lower(txt)
        if "doluluk" in low:
            x0_candidates.append(float(w.get("x0", 0)))
    return max(x0_candidates) if x0_candidates else None


def _doluluk_from_row_words_in_column(
    row_words_sorted: List[Dict], doluluk_x0: Optional[float], margin: float = 10.0
) -> Optional[int]:
    """
    Satırda sadece Doluluk Oranı sütunundaki (x0 >= doluluk_x0 - margin) numeric token'ları aday alır,
    adaylardan 0-100 arası doluluk değerini seçer (birleştirerek: 2+0 -> 20).
    doluluk_x0 yoksa sağdaki sayıya fallback.
    """
    if doluluk_x0 is not None:
        row_words_sorted = [
            w for w in row_words_sorted
            if float(w.get("x0", 0)) >= doluluk_x0 - margin
        ]
    if not row_words_sorted:
        return None
    return _rightmost_doluluk_from_row_words(row_words_sorted)


def _rightmost_doluluk_from_row_words(
    row_words_sorted: List[Dict],
) -> Optional[int]:
    """
    Satırdaki kelimelerden en sağdaki doluluk oranını (0-100) al.
    PDF'te '20' bazen '2' ve '0' iki kelime gelir; sağdan bitişik rakam kelimelerini birleştirir.
    """
    for i in range(len(row_words_sorted) - 1, -1, -1):
        w = row_words_sorted[i]
        t = (w.get("text") or "").strip().replace(" ", "")
        if not t:
            continue
        try:
            n = int(t)
        except ValueError:
            continue
        if not (0 <= n <= 100):
            continue
        # Zaten iki+ haneli sayı (20, 22, 25 vb.) -> doğrudan kullan
        if n >= 10:
            return n
        # Tek rakam: sola doğru bitişik tek rakam kelimelerini topla (örn. "2","0" -> 20)
        digits = [n]
        for j in range(i - 1, -1, -1):
            tw = (row_words_sorted[j].get("text") or "").strip().replace(" ", "")
            try:
                d = int(tw)
                if 0 <= d <= 9:
                    digits.append(d)
                    if len(digits) >= 3:
                        break
                else:
                    break
            except ValueError:
                break
        if len(digits) == 2:
            combined = digits[1] * 10 + digits[0]
            if 0 <= combined <= 100:
                return combined
        return n
    return None


def _normalize_memzuc_row_name(left_text: str) -> Optional[str]:
    """Soldaki kelimelerden canonical satır adını döndür (üst özet + ana blok aynı kalemler)."""
    t = (left_text or "").strip()
    if not t:
        return None
    low = _tr_lower(t)
    if "umumi" in low and "limit" in low:
        return "Umumi Limit"
    if "toplam" in low and "nakdi" in low and "kredi" in low:
        return "Toplam Nakdi Kredi"
    if "toplam" in low and "gn" in low and "kredi" in low:
        return "Toplam GN Kredi"
    # TOPLAM: sadece "toplam" veya satır sonu "toplam" (Toplam Nakdi/GN ile karışmasın)
    if low.strip() == "toplam" or low.rstrip().endswith("toplam"):
        return "TOPLAM"
    if re.search(r"\btoplam\b", low) and "nakdi" not in low and "gn" not in low:
        return "TOPLAM"
    for rn in _MEMZUC_DOLULUK_ROW_NAMES:
        if rn.lower() in low or low in rn.lower():
            return rn
    return None


def _parse_memzuc_crop(
    words: List[Dict],
    allowed_row_names: Tuple[str, ...],
    period: str,
    crop_label: str,
    doluluk_band_x0: Optional[float] = None,
) -> List[Tuple[str, str, int]]:
    """
    Tek crop'tan (words) sadece allowed_row_names kalemlerini çıkarır; her biri için (period, row_name, doluluk).
    doluluk_band_x0 verilirse (Crop B): doluluk_x0 arama yok, sadece x0 >= doluluk_band_x0 olan numeric
    tokenlar aday; 0-100 arası doluluk seçilir (Umumi 20, TOPLAM 22). Verilmezse (Crop A) mevcut doluluk_x0.
    """
    tol = _MEMZUC_ROW_Y_TOLERANCE
    if doluluk_band_x0 is not None:
        doluluk_x0 = doluluk_band_x0
        margin = 0.0
    else:
        doluluk_x0 = _find_doluluk_column_x0(words)
        margin = 10.0
    by_row: Dict[float, List[Dict]] = {}
    for w in words:
        text = (w.get("text") or "").strip()
        if not text:
            continue
        top = float(w.get("top", 0))
        row_key = round(top / tol) * tol
        by_row.setdefault(row_key, []).append(w)

    out: List[Tuple[str, str, int]] = []
    for rk in sorted(by_row.keys()):
        row_words = by_row[rk]
        row_words_sorted = sorted(row_words, key=lambda w: float(w.get("x0", 0)))
        left_parts = []
        for w in row_words_sorted[:20]:
            t = (w.get("text") or "").strip()
            if t and not t.isdigit():
                left_parts.append(t)
        left_text = " ".join(left_parts)
        row_name = _normalize_memzuc_row_name(left_text)
        if not row_name or row_name not in allowed_row_names:
            continue
        doluluk = _doluluk_from_row_words_in_column(row_words_sorted, doluluk_x0, margin=margin)
        if doluluk is None:
            continue
        out.append((period, row_name, doluluk))
    return out


def _build_cropb_debug_line(
    words_b: List[Dict],
    header_top: float,
    page_width: float,
    crop_b_bottom: float,
    band_x0: float,
) -> str:
    """Crop B içinde x0 >= band_x0 olan numeric (0-100) tokenların ilk 20'si; bbox/band ayrımı için."""
    tokens: List[str] = []
    for w in sorted(words_b, key=lambda x: float(x.get("x0", 0))):
        x0 = float(w.get("x0", 0))
        if x0 < band_x0:
            continue
        t = (w.get("text") or "").strip().replace(" ", "")
        try:
            n = int(t)
            if 0 <= n <= 100:
                tokens.append(str(n))
                if len(tokens) >= 20:
                    break
        except ValueError:
            continue
    return (
        f"MEMZUC_CROPB_DEBUG | header_top={header_top:.1f} | bboxB=(0,0,{page_width:.0f},{crop_b_bottom:.1f}) | "
        f"band_x0={band_x0:.1f} | tokens={','.join(tokens)}"
    )


def _memzuc_doluluk_from_tables(
    page, crop_bbox: Tuple[float, float, float, float], period: str
) -> List[Tuple[str, str, int]]:
    """
    Crop bölgesindeki pdfplumber tablolarından Doluluk Oranı sütununu oku.
    Words ile bulunamayan kalemler için yedek (Umumi Limit, TOPLAM vb.).
    """
    out: List[Tuple[str, str, int]] = []
    try:
        crop = page.crop(crop_bbox)
        tables = crop.extract_tables() or []
    except Exception:
        return out
    for tbl in tables:
        if not tbl or len(tbl) < 2:
            continue
        # Başlık satırında "Doluluk" veya "Doluluk Oranı" sütununu bul
        header = tbl[0]
        doluluk_col: Optional[int] = None
        for i, cell in enumerate(header or []):
            if not cell:
                continue
            c = (cell or "").strip().lower()
            if "doluluk" in c:
                doluluk_col = i
                break
        if doluluk_col is None:
            continue
        for row in tbl[1:]:
            if not row or doluluk_col >= len(row):
                continue
            left_text = " ".join(str(c or "").strip() for c in (row[: max(doluluk_col, 3)] or []))
            row_name = _normalize_memzuc_row_name(left_text)
            if not row_name or row_name not in _MEMZUC_DOLULUK_ROW_NAMES:
                continue
            raw = (row[doluluk_col] or "").strip().replace(" ", "")
            try:
                d = int(raw)
            except ValueError:
                continue
            if 0 <= d <= 100:
                out.append((period, row_name, d))
    return out


def _extract_memzuc_continuation_page(
    page, period: str, page_num: int
) -> List[str]:
    """
    Tablo önceki sayfada kesildiyse (header ~sayfa sonunda) TOPLAM ve Umumi Limit
    sonraki sayfanın üstünde olabilir. Bu sayfadan sadece bu iki kalemi oku.
    """
    lines_out: List[str] = []
    try:
        words = page.extract_words() or []
    except Exception:
        return lines_out
    if not words:
        return lines_out
    page_width = getattr(page, "width", 612)
    crop = (0, 0, page_width, _MEMZUC_CONTINUATION_CROP_HEIGHT)
    try:
        cropped = page.crop(crop)
        words = cropped.extract_words() or []
    except Exception:
        return lines_out
    for _p, rn, doluluk in _parse_memzuc_crop(
        words, _CROP_B_ROW_NAMES, period, "continuation", doluluk_band_x0=None
    ):
        lines_out.append(f"MEMZUC_DOLULUK | dönem: {period} | kalem: {rn} | doluluk: {doluluk}")
    return lines_out


def _extract_memzuc_doluluk_from_page_words(
    page, page_num: int
) -> Tuple[List[str], Dict[str, Any], Optional[str]]:
    """
    MEMZUC_DOLULUK üretimini iki crop alanına kısıtlar. Dönem sayfa full_text max(20xx/yy).
    Crop A: header_top+320 veya (h*0.25..h*0.70). Crop B: üst band veya h*0.25.
    Returns: (lines_out, debug_out, cropb_debug_line).
    """
    empty_debug: Dict[str, Any] = {
        "doluluk_x0": None,
        "periods_found": [],
        "extracted_rows_count": 0,
        "memzuc_bbox_top": None,
        "memzuc_bbox_bottom": None,
    }
    try:
        full_words = page.extract_words() or []
    except Exception as e:
        logger.debug("Memzuc words extract page %s: %s", page_num, e)
        return [], empty_debug, None
    if not full_words:
        return [], empty_debug, None

    header_top = _find_memzuc_header_top(full_words)
    page_width = getattr(page, "width", 612)
    page_height = getattr(page, "height", 792)
    default_period = _period_from_page_text(page)
    all_results: List[Tuple[str, str, int]] = []
    cropb_debug_line: Optional[str] = None

    crop_b_bottom_val: float = 0.0
    if _MEMZUC_USE_PERCENTAGE_CROPS:
        # Crop B = üst %25, Crop A = %25..%70 (header_top yanlışsa bu mod)
        crop_b_bottom_val = page_height * _MEMZUC_CROP_B_TOP_FRAC
        crop_a_top = page_height * _MEMZUC_CROP_A_TOP_FRAC
        crop_a_bottom = page_height * _MEMZUC_CROP_A_BOTTOM_FRAC
        try:
            crop_b = page.crop((0, 0, page_width, crop_b_bottom_val))
            words_b = crop_b.extract_words() or []
        except Exception as e:
            logger.debug("Memzuc crop B (pct) page %s: %s", page_num, e)
            words_b = []
        try:
            crop_a = page.crop((0, crop_a_top, page_width, crop_a_bottom))
            words_a = crop_a.extract_words() or []
        except Exception as e:
            logger.debug("Memzuc crop A (pct) page %s: %s", page_num, e)
            words_a = []
        end_crop_a = crop_a_bottom
    else:
        if header_top is None:
            logger.debug("Memzuc header KREDİ GRUBU FİRMA MEMZUCULARI not found on page %s", page_num)
            return [], empty_debug, None
        # Crop A: ana tablo (TOPLAM, Umumi Limit dahil); sayfa sonunu aşmayacak şekilde
        crop_a_bottom = min(header_top + _MEMZUC_CROP_A_HEIGHT, page_height)
        end_crop_a = crop_a_bottom
        try:
            crop_a = page.crop((0, header_top, page_width, crop_a_bottom))
            words_a = crop_a.extract_words() or []
        except Exception as e:
            logger.debug("Memzuc crop A page %s: %s", page_num, e)
            words_a = []
        # Crop B: (0, 0, w, min(header_top-10, h*0.30))
        crop_b_bottom_val = min(
            max(0.0, header_top - _MEMZUC_CROP_B_ABOVE_HEADER),
            page_height * _MEMZUC_CROP_B_MAX_FRAC,
        )
        try:
            crop_b = page.crop((0, 0, page_width, crop_b_bottom_val))
            words_b = crop_b.extract_words() or []
        except Exception as e:
            logger.debug("Memzuc crop B page %s: %s", page_num, e)
            words_b = []

    crop_a_results: List[Tuple[str, str, int]] = []
    periods_found_list: List[str] = []
    if not _MEMZUC_USE_PERCENTAGE_CROPS and header_top is not None and full_words:
        period_positions = _find_period_positions_from_words(full_words, header_top, end_crop_a)
        if period_positions:
            for i, (seg_top, period) in enumerate(period_positions):
                seg_bottom = period_positions[i + 1][0] if i + 1 < len(period_positions) else end_crop_a
                try:
                    seg_crop = page.crop((0, seg_top, page_width, seg_bottom))
                    words_seg = seg_crop.extract_words() or []
                except Exception:
                    words_seg = []
                if words_seg:
                    for t in _parse_memzuc_crop(words_seg, _CROP_A_ROW_NAMES, period, "A"):
                        crop_a_results.append(t)
                        all_results.append(t)
                        if period not in periods_found_list:
                            periods_found_list.append(period)
    if not periods_found_list and words_a:
        for t in _parse_memzuc_crop(words_a, _CROP_A_ROW_NAMES, default_period, "A"):
            crop_a_results.append(t)
            all_results.append(t)
        periods_found_list = [default_period]

    # Crop B (üst özet): sadece Crop A'nın bulamadığı kalemler için fallback
    doluluk_band_x0_b = page_width * _CROP_B_DOLULUK_BAND_RATIO
    found_kalem = {rn for (_, rn, _) in crop_a_results}
    if words_b:
        for t in _parse_memzuc_crop(
            words_b, _CROP_B_ROW_NAMES, default_period, "B", doluluk_band_x0=None
        ):
            _, rn, _ = t
            if rn not in found_kalem:
                all_results.append(t)
                found_kalem.add(rn)
    if words_b:
        cropb_debug_line = _build_cropb_debug_line(
            words_b,
            header_top if header_top is not None else 0.0,
            page_width,
            crop_b_bottom_val,
            doluluk_band_x0_b,
        )

    merged: Dict[Tuple[str, str], int] = {}
    for p, row_name, doluluk in all_results:
        key = (p, row_name)
        merged[key] = max(merged.get(key, 0), doluluk)

    # Words/crop 4 kalem bulamadıysa tablo yedek dene (Doluluk Oranı sütunlu tablo)
    for p in (periods_found_list or [default_period]):
        if len([k for k in merged if k[0] == p]) >= 4:
            continue
        if header_top is not None:
            crop_bbox = (0, header_top, page_width, end_crop_a)
            table_results = _memzuc_doluluk_from_tables(page, crop_bbox, p)
            for _p, row_name, doluluk in table_results:
                key = (_p, row_name)
                if key not in merged:
                    merged[key] = doluluk

    if not periods_found_list and merged:
        periods_found_list = sorted(set(k[0] for k in merged))

    lines_out = [
        f"MEMZUC_DOLULUK | dönem: {p} | kalem: {rn} | doluluk: {d}"
        for (p, rn), d in merged.items()
    ]
    debug_out = {
        "doluluk_x0": None,
        "periods_found": periods_found_list or [default_period],
        "extracted_rows_count": len(lines_out),
        "memzuc_bbox_top": header_top,
        "memzuc_bbox_bottom": end_crop_a,
        "rows": [{"kalem": rn, "doluluk": d} for (_, rn), d in merged.items()],
        "words_a_count": len(words_a) if words_a else 0,
    }
    return lines_out, debug_out, cropb_debug_line


def _page_has_memzuc_signal(free_text: Optional[str]) -> bool:
    """
    Memzuc words parser sadece bu başlık bulunursa çalışsın; yoksa text fallback.
    True: sayfa metninde 'KREDİ GRUBU FİRMA MEMZUCULARI' veya 'ANA FİRMA DAHİL' geçiyorsa.
    Sinyal gevşetme kaldırıldı (Doluluk / 20xx/yy tek başına yeterli değil).
    """
    if not free_text or not str(free_text).strip():
        return False
    t = str(free_text).strip()
    hay_upper = t.upper()
    if "KREDİ GRUBU FİRMA MEMZUCULARI" in hay_upper or "KREDİ GRUBU FİRMA MEMZUÇLARI" in hay_upper:
        return True
    if "ANA FİRMA DAHİL" in hay_upper:
        return True
    return False


def _parse_memzuc_doluluk_from_text(text: str) -> List[str]:
    """
    Metinde dönem (20xx/yy) bloklarını bulur; her blokta her kalem için **son** eşleşen satırdaki
    son 1-2 haneli sayıyı doluluk oranı olarak alır (özet tablosu yerine ana tablo satırı seçilir).
    Returns: ["MEMZUC_DOLULUK | dönem: 2025/12 | kalem: Umumi Limit | doluluk: 20", ...]
    """
    lines_out: List[str] = []
    normalized = text.replace("\r", "\n")
    period_matches = list(_MEMZUC_DOLULUK_PERIOD_RE.finditer(normalized))
    for i, m in enumerate(period_matches):
        period_raw = (m.group(1) or "").strip()
        period_norm = period_raw.replace(" ", "")
        start = m.end()
        end = period_matches[i + 1].start() if i + 1 < len(period_matches) else len(normalized)
        block = normalized[start:end]
        block_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        # Her kalem için bloktaki son eşleşen satırı al (özet satırı değil, ana tablo satırı)
        for row_name in _MEMZUC_DOLULUK_ROW_NAMES:
            row_lower = row_name.lower()
            last_match_line: Optional[str] = None
            for line in block_lines:
                if row_lower in line.lower():
                    last_match_line = line
            if last_match_line:
                nums = re.findall(r"\b\d{1,2}\b", last_match_line)
                if nums:
                    pct = nums[-1]
                    lines_out.append(
                        f"MEMZUC_DOLULUK | dönem: {period_norm} | kalem: {row_name} | doluluk: {pct}"
                    )
    return lines_out


def _collect_memzuc_doluluk_from_pages(pages: List["_PageData"]) -> List[str]:
    """
    Tüm sayfalardan free_text/page_text alır; memzuc doluluk sinyali geçen sayfalarda
    _parse_memzuc_doluluk_from_text çalıştırıp satırları toplar. DB'ye yazılacak format.
    """
    all_lines: List[str] = []
    for pd in pages:
        text = (
            getattr(pd, "free_text", None)
            or getattr(pd, "page_text", None)
            or getattr(pd, "raw_text", None)
        )
        if not (text and str(text).strip()):
            continue
        t = str(text).strip()
        hay = t.upper()
        if not any(
            sig in hay or sig in t
            for sig in _MEMZUC_DOLULUK_SIGNALS
        ):
            continue
        page_lines = _parse_memzuc_doluluk_from_text(t)
        all_lines.extend(page_lines)
    return all_lines


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
        memzuc_lines_from_words: List[str] = []
        memzuc_words_debug: Dict[str, Any] = {}
        last_cropb_debug_line: Optional[str] = None

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    pd = self._process_page(page, page_num, warnings)
                    pages_data.append(pd)
                    tables_per_page.append({"page": page_num, "count": pd.table_count})
                    # Memzuc doluluk: başlık sinyali ile words parser (Crop A/B)
                    if _page_has_memzuc_signal(pd.free_text):
                        word_lines, page_debug, cropb_debug_line = _extract_memzuc_doluluk_from_page_words(
                            page, page_num
                        )
                        if word_lines:
                            memzuc_lines_from_words.extend(word_lines)
                            if cropb_debug_line:
                                last_cropb_debug_line = cropb_debug_line
                            # Tablo sayfa sonunda kesildiyse (bbox_bottom ~ sayfa yüksekliği) TOPLAM/Umumi Limit sonraki sayfada olabilir
                            if (
                                len(word_lines) < 4
                                and page_num < total_pages
                                and page_debug.get("memzuc_bbox_bottom") is not None
                            ):
                                try:
                                    next_page = pdf.pages[page_num]
                                    period = (
                                        (page_debug.get("periods_found") or [_DEFAULT_MEMZUC_PERIOD])[0]
                                        if page_debug.get("periods_found")
                                        else _DEFAULT_MEMZUC_PERIOD
                                    )
                                    cont_lines = _extract_memzuc_continuation_page(
                                        next_page, period, page_num + 1
                                    )
                                    if cont_lines:
                                        memzuc_lines_from_words.extend(cont_lines)
                                        logger.info(
                                            "Memzuc devam sayfası %s: +%s satır (TOPLAM/Umumi Limit)",
                                            page_num + 1,
                                            len(cont_lines),
                                        )
                                except Exception as e:
                                    logger.debug("Memzuc devam sayfası %s: %s", page_num + 1, e)
                            memzuc_words_debug = {
                                "words_parser_used": True,
                                "doluluk_x0": page_debug.get("doluluk_x0"),
                                "periods_found": page_debug.get("periods_found", []),
                                "extracted_rows_count": len(memzuc_lines_from_words),
                                "memzuc_bbox_top": page_debug.get("memzuc_bbox_top"),
                                "memzuc_bbox_bottom": page_debug.get("memzuc_bbox_bottom"),
                                "rows": page_debug.get("rows", []),
                            }
                            logger.info(
                                "Memzuc words parser page %s: bbox_top=%s bbox_bottom=%s doluluk_x0=%s rows=%s",
                                page_num,
                                page_debug.get("memzuc_bbox_top"),
                                page_debug.get("memzuc_bbox_bottom"),
                                page_debug.get("doluluk_x0"),
                                len(word_lines),
                            )
                        else:
                            memzuc_lines_from_words.extend(
                                _parse_memzuc_doluluk_from_text(pd.free_text or "")
                            )
        except Exception as exc:
            logger.error(f"pdfplumber failed on {path.name}: {exc}")
            warnings.append(f"pdfplumber open failed: {exc}")
            total_pages = 0

        # Assign sections to pages, handling spill merges
        assigned = self._assign_sections(pages_data, section_hits, spill_merges, warnings)

        # Build each section from the assigned data
        sections = self._build_sections(assigned, warnings)

        # Memzuc doluluk: words-based (öncelik) veya tüm sayfalardan text fallback
        memzuc_fallback = memzuc_lines_from_words
        if not memzuc_fallback:
            memzuc_fallback = _collect_memzuc_doluluk_from_pages(pages_data)
        # Aynı (dönem, kalem) birden fazla sayfa/chunk'ta farklı değerle gelirse tek satırda max al
        if memzuc_fallback:
            before_merge = len(memzuc_fallback)
            memzuc_fallback = _merge_memzuc_lines_across_pages(memzuc_fallback)
            if before_merge != len(memzuc_fallback):
                logger.info(
                    "Memzuc sayfalar arası merge: %s -> %s satır",
                    before_merge,
                    len(memzuc_fallback),
                )
            if last_cropb_debug_line:
                memzuc_fallback.append(last_cropb_debug_line)
            sections["memzuc_doluluk_fallback"] = memzuc_fallback
            if memzuc_words_debug:
                memzuc_words_debug["extracted_rows_count"] = len(memzuc_fallback)
            logger.info(
                "Memzuc doluluk: %s satır (words_parser_used=%s)",
                len(memzuc_fallback),
                memzuc_words_debug.get("words_parser_used", False),
            )

        debug_payload: Dict[str, Any] = {
            "section_hits": section_hits,
            "tables_found_per_page": tables_per_page,
            "spill_merges": spill_merges,
            "warnings": warnings,
        }
        if memzuc_lines_from_words and memzuc_words_debug:
            debug_payload["memzuc_words_parser"] = memzuc_words_debug

        return {
            "doc_type": "İstihbarat Raporu",
            "meta": {
                "source_file": path.name,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "pages": total_pages,
            },
            "sections": sections,
            "debug": debug_payload,
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

    # Memzuc Doluluk Oranları (text fallback — DB'de MEMZUC_DOLULUK araması için)
    memzuc_doluluk = sections.get("memzuc_doluluk_fallback", [])
    if memzuc_doluluk and isinstance(memzuc_doluluk, list):
        parts.append("## Memzuc Doluluk Oranları")
        for line in memzuc_doluluk:
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
