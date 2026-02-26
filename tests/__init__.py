"""
Unit tests for CreditIntelligencePDFExtractor — section detection, spill merge,
bank name normalization, number parsing, guardrail, render_structured_text.

Run:  python -m pytest tests/test_credit_intelligence_pdf_extractor.py -v
"""

from typing import Dict, List, Optional

import pytest

from structured_extractors.credit_intelligence_pdf_extractor import (
    CreditIntelligencePDFExtractor,
    _clean_cell,
    _detect_section,
    _normalize_bank_name,
    _parse_number,
    _PageData,
    render_structured_text,
)


# ===================================================================
# Section detection
# ===================================================================

class TestSectionDetection:
    """_detect_section must recognize known section headers."""

    @pytest.mark.parametrize("text, expected", [
        ("BANKA İSTİHBARATI", "banka_istihbarati"),
        ("Banka İstihbaratı", "banka_istihbarati"),
        ("banka istihbaratı bilgileri", "banka_istihbarati"),
        ("MEMZUÇ BİLGİLERİ", "memzuc_bilgileri"),
        ("Memzuc Bilgileri", "memzuc_bilgileri"),
        ("LİMİT RİSK BİLGİLERİ (BİN TL)", "limit_risk_bilgileri"),
        ("Limit Risk Bilgileri", "limit_risk_bilgileri"),
        ("KAYNAK BAZINDA DETAY (BİN TL)", "kaynak_bazinda_detay"),
        ("Kaynak Bazında Detay", "kaynak_bazinda_detay"),
        ("E-HACİZ TARİHÇESİ", "e_haciz_tarihcesi"),
        ("ERKEN UYARI TARİHÇESİ", "erken_uyari_tarihcesi"),
        ("PİYASA İSTİHBARATI", "piyasa_istihbarati"),
        ("ÖZET / GENEL BİLGİLER", "ozet_genel_bilgiler"),
        ("ÇEK PERFORMANSI", "cek_performansi"),
        ("KKB RAPORLARI", "kkb_raporlari"),
        ("KKB RİSKİ", "kkb_riski"),
        ("KONSOLİDE MEMZUÇ", "konsolide_memzuc"),
    ])
    def test_known_headers(self, text: str, expected: str):
        assert _detect_section(text) == expected

    def test_unknown_header_returns_none(self):
        assert _detect_section("Bilinmeyen Başlık") is None
        assert _detect_section("Sayfa 4") is None


# ===================================================================
# Bank name normalization
# ===================================================================

class TestBankNameNormalization:
    """_normalize_bank_name should clean whitespace and punctuation."""

    def test_basic_normalization(self):
        assert _normalize_bank_name("  TÜRKİYE  EMLAK   KATILIM  BANKASI  A.Ş. ") == \
               "TÜRKİYE EMLAK KATILIM BANKASI A.Ş"

    def test_trailing_dot(self):
        assert _normalize_bank_name("Ziraat Bankası A.Ş.") == "Ziraat Bankası A.Ş"

    def test_newline_in_name(self):
        raw = "TÜRKİYE\nEMLAK KATILIM\nBANKASI A.Ş."
        result = _normalize_bank_name(raw)
        assert "\n" not in result
        assert "TÜRKİYE EMLAK KATILIM BANKASI A.Ş" == result


# ===================================================================
# Number parsing (Turkish locale: . = thousand, , = decimal)
# ===================================================================

class TestParseNumber:
    @pytest.mark.parametrize("raw, expected", [
        ("1.234.567", 1234567),
        ("1.234,56", 1234.56),
        ("42", 42),
        ("-100", -100),
        ("0", 0),
        ("", None),
        ("-", None),
        (None, None),
        ("abc", None),
        ("1.234.567,89", 1234567.89),
    ])
    def test_parse(self, raw, expected):
        result = _parse_number(raw)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


# ===================================================================
# Cell cleaning
# ===================================================================

class TestCleanCell:
    def test_none(self):
        assert _clean_cell(None) == ""

    def test_whitespace(self):
        assert _clean_cell("  hello \n world  ") == "hello   world"

    def test_simple_strip(self):
        assert _clean_cell("  abc  ") == "abc"

    def test_number_passthrough(self):
        assert _clean_cell(42) == "42"


# ===================================================================
# Spill merge logic
# ===================================================================

class TestSpillMerge:
    """
    If a page has no section header, it should merge into the previous section.
    """

    def test_spill_merge(self):
        extractor = CreditIntelligencePDFExtractor()
        pages = [
            _PageData(page_num=1, section_header="ozet_genel_bilgiler",
                      tables=[], free_text="Özet", table_count=0),
            _PageData(page_num=2, section_header="banka_istihbarati",
                      tables=[], free_text="Banka list page 2", table_count=1),
            _PageData(page_num=3, section_header=None,
                      tables=[], free_text="continued bank data", table_count=1),
            _PageData(page_num=4, section_header="memzuc_bilgileri",
                      tables=[], free_text="Memzuç", table_count=1),
        ]

        section_hits: list = []
        spill_merges: list = []
        warnings: list = []

        assigned = extractor._assign_sections(pages, section_hits, spill_merges, warnings)

        assert "banka_istihbarati" in assigned
        assert len(assigned["banka_istihbarati"]) == 2
        assert assigned["banka_istihbarati"][0].page_num == 2
        assert assigned["banka_istihbarati"][1].page_num == 3

        assert len(spill_merges) == 1
        assert spill_merges[0]["section"] == "banka_istihbarati"
        assert spill_merges[0]["from_page"] == 2
        assert spill_merges[0]["to_page"] == 3

    def test_no_header_first_page_defaults_to_ozet(self):
        extractor = CreditIntelligencePDFExtractor()
        pages = [
            _PageData(page_num=1, section_header=None,
                      tables=[], free_text="Some intro", table_count=0),
        ]
        section_hits: list = []
        spill_merges: list = []
        warnings: list = []

        assigned = extractor._assign_sections(pages, section_hits, spill_merges, warnings)

        assert "ozet_genel_bilgiler" in assigned
        assert len(warnings) == 1


# ===================================================================
# Bank verification guardrail
# ===================================================================

class TestBankVerificationGuardrail:
    def test_bank_present_in_page(self):
        pages = [
            _PageData(
                page_num=4,
                section_header="banka_istihbarati",
                tables=[],
                free_text="Banka: TÜRKİYE EMLAK KATILIM BANKASI A.Ş. Genel Limit: 50.000",
                table_count=1,
            )
        ]
        assert CreditIntelligencePDFExtractor._verify_bank_in_page(
            "TÜRKİYE EMLAK KATILIM BANKASI A.Ş", pages, 4
        ) is True

    def test_bank_not_present_in_page(self):
        pages = [
            _PageData(
                page_num=4,
                section_header="banka_istihbarati",
                tables=[],
                free_text="Some other content without the bank name",
                table_count=1,
            )
        ]
        assert CreditIntelligencePDFExtractor._verify_bank_in_page(
            "AKBANK T.A.Ş", pages, 4
        ) is False

    def test_bank_partial_token_match(self):
        pages = [
            _PageData(
                page_num=5,
                section_header=None,
                tables=[],
                free_text="EMLAK KATILIM BANKASI verileri şöyledir...",
                table_count=0,
            )
        ]
        assert CreditIntelligencePDFExtractor._verify_bank_in_page(
            "TÜRKİYE EMLAK KATILIM BANKASI A.Ş", pages, 5
        ) is True


# ===================================================================
# Header row detection
# ===================================================================

class TestHeaderDetection:
    def test_text_header(self):
        table = [
            ["Banka Adı", "Genel Limit", "Nakit Risk"],
            ["Ziraat", "1.000", "500"],
        ]
        headers = CreditIntelligencePDFExtractor._detect_header_row(table)
        assert headers is not None
        assert headers[0] == "Banka Adı"

    def test_numeric_first_row_no_header(self):
        table = [
            ["1.000", "2.000", "3.000"],
            ["4.000", "5.000", "6.000"],
        ]
        headers = CreditIntelligencePDFExtractor._detect_header_row(table)
        assert headers is None

    def test_empty_table(self):
        assert CreditIntelligencePDFExtractor._detect_header_row([]) is None


# ===================================================================
# render_structured_text
# ===================================================================

class TestRenderStructuredText:
    def test_basic_render(self):
        data = {
            "doc_type": "İstihbarat Raporu",
            "meta": {"source_file": "test.pdf", "pages": 5},
            "sections": {
                "banka_istihbarati": [
                    {
                        "bank_name": "TÜRKİYE EMLAK KATILIM BANKASI A.Ş",
                        "genel_limit": {"value": 50000, "currency": "TRY"},
                        "nakit_risk": {"value": 25000, "currency": "TRY"},
                        "gn_risk": {"value": 10000, "currency": "TRY"},
                        "teminat_sarti": "İpotek",
                        "revize_tarihi": "11.2025",
                    }
                ],
            },
            "debug": {},
        }
        text = render_structured_text(data)
        assert "TÜRKİYE EMLAK KATILIM BANKASI A.Ş" in text
        assert "50,000 TRY" in text
        assert "## Banka İstihbaratı" in text

    def test_empty_sections(self):
        data = {
            "doc_type": "İstihbarat Raporu",
            "meta": {"source_file": "empty.pdf", "pages": 0},
            "sections": {},
            "debug": {},
        }
        text = render_structured_text(data)
        assert "İstihbarat Raporu" in text


# ===================================================================
# _bank_search_key (RAG guardrail helper) — standalone reimplementation
# to avoid heavy app_refactored import chain in unit test context
# ===================================================================

def _bank_search_key_standalone(official_name: str) -> str:
    """Mirror of RAGQueryUseCase._bank_search_key for unit testing."""
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


class TestBankSearchKey:
    """Test the search key derivation used by the RAG guardrail."""

    def test_strip_turkiye_and_as(self):
        key = _bank_search_key_standalone("Türkiye Emlak Katılım Bankası A.Ş.")
        assert key == "emlak katılım bankası"

    def test_no_prefix(self):
        key = _bank_search_key_standalone("Akbank T.A.Ş.")
        assert key == "akbank"

    def test_tc_prefix(self):
        key = _bank_search_key_standalone("T.C. Ziraat Bankası A.Ş.")
        assert key == "ziraat bankası"


# ===================================================================
# Regression: bank index from structured text (Banka: lines)
# ===================================================================

# Standalone reimplementation of the core logic so tests don't need
# the full app_refactored import chain.

import re

_BANKA_SECTION_RE_TEST = re.compile(
    r'##\s*Banka\s+[İi]stihbarat[ıi]', re.IGNORECASE
)
_BANKA_LINE_RE_TEST = re.compile(
    r'^Banka:\s*(.+?)(?:\s*\|)', re.MULTILINE
)


def _extract_bank_index_standalone(chunks_content: List[str]) -> List[str]:
    """Mirror of RAGQueryUseCase._extract_bank_index_from_chunks logic."""
    seen: set = set()
    names: List[str] = []
    for content in chunks_content:
        if not _BANKA_SECTION_RE_TEST.search(content):
            continue
        for m in _BANKA_LINE_RE_TEST.finditer(content):
            raw = m.group(1).strip().rstrip('.')
            if len(raw) > 4:
                key = raw.lower()
                if key not in seen:
                    seen.add(key)
                    names.append(raw)
    return names


class TestBankIndexFromStructuredText:
    """Regression: bank index must come from 'Banka:' lines, not regex on random tables."""

    STRUCTURED_CHUNK = (
        "## Banka İstihbaratı\n"
        "Banka: TÜRKİYE EMLAK KATILIM BANKASI A.Ş | Genel Limit: 50,000 TRY | "
        "Nakit Risk: 25,000 TRY | G.Nakdi Risk: 10,000 TRY | Teminat: İpotek | Revize: 11.2025\n"
        "Banka: T.C. ZİRAAT BANKASI A.Ş | Genel Limit: 100,000 TRY | "
        "Nakit Risk: 80,000 TRY | G.Nakdi Risk: 5,000 TRY | Teminat: Kefalet | Revize: 06.2025\n"
    )

    KAYNAK_CHUNK = (
        "## Kaynak Bazında Detay (Bin TL)\n"
        "kaynak: K12 | bank_name: None | grup_limit: 500 | toplam_limit: 400\n"
    )

    def test_extracts_from_banka_section(self):
        names = _extract_bank_index_standalone([self.STRUCTURED_CHUNK])
        assert len(names) == 2
        assert "TÜRKİYE EMLAK KATILIM BANKASI A.Ş" in names
        assert "T.C. ZİRAAT BANKASI A.Ş" in names

    def test_does_not_extract_from_kaynak_section(self):
        names = _extract_bank_index_standalone([self.KAYNAK_CHUNK])
        assert len(names) == 0

    def test_ignores_non_banka_chunks(self):
        other = "## Limit Risk Bilgileri (Bin TL)\nkaynak: K1 | grup_limit: 100"
        names = _extract_bank_index_standalone([other, self.STRUCTURED_CHUNK])
        assert len(names) == 2

    def test_deduplication(self):
        double = self.STRUCTURED_CHUNK + "\n" + self.STRUCTURED_CHUNK
        names = _extract_bank_index_standalone([double])
        assert len(names) == 2


# ===================================================================
# Regression: 3-tier bank guardrail scoping strategy
# ===================================================================

# Standalone reimplementation of the 3-tier guardrail logic.

def _apply_guardrail_standalone(
    norm_name: Optional[str],
    confidence: str,
    chunks: List[Dict],
) -> Dict:
    """
    Simplified mirror of RAGQueryUseCase._apply_bank_guardrail.
    chunks: list of {"content": str, "filename": str}
    """
    THRESHOLD = ("medium", "high")
    if not norm_name or confidence not in THRESHOLD:
        return {"passed": False, "scoping_strategy": "blocked"}

    # search key
    key = _bank_search_key_standalone(norm_name)

    # Tier 1: substring
    scoped = [c for c in chunks if key and key in c["content"].lower()]
    if scoped:
        return {
            "passed": True,
            "scoping_strategy": "substring_scoped",
            "scoped_count": len(scoped),
        }

    # Tier 2: Banka İstihbaratı section chunks
    banka_chunks = [
        c for c in chunks
        if _BANKA_SECTION_RE_TEST.search(c["content"])
        or "banka istihbarat" in c["content"].lower()
    ]
    if banka_chunks:
        return {
            "passed": True,
            "scoping_strategy": "fallback_banka_istihbarati_only",
            "scoped_count": len(banka_chunks),
        }

    # Tier 3: blocked
    return {"passed": False, "scoping_strategy": "blocked"}


class TestBankGuardrailScopingStrategy:
    """
    Regression tests for the 3-tier guardrail:
    - substring_scoped: bank name in chunk text
    - fallback_banka_istihbarati_only: bank name NOT in text, but section header present
    - blocked: nothing relevant
    """

    BANKA_CHUNK_WITH_EMLAK = {
        "content": (
            "## Banka İstihbaratı\n"
            "Banka: TÜRKİYE EMLAK KATILIM BANKASI A.Ş | Genel Limit: 50,000 TRY\n"
            "Banka: T.C. ZİRAAT BANKASI A.Ş | Genel Limit: 100,000 TRY\n"
        ),
        "filename": "istihbarat.pdf",
    }

    KAYNAK_CHUNK = {
        "content": "## Kaynak Bazında Detay (Bin TL)\nkaynak: K12 | bank_name: None",
        "filename": "istihbarat.pdf",
    }

    def test_bank_in_banka_line_substring_scoped(self):
        """Bank appears only in Banka: lines → substring_scoped."""
        result = _apply_guardrail_standalone(
            "TÜRKİYE EMLAK KATILIM BANKASI A.Ş", "high",
            [self.BANKA_CHUNK_WITH_EMLAK, self.KAYNAK_CHUNK],
        )
        assert result["passed"] is True
        assert result["scoping_strategy"] == "substring_scoped"

    def test_bank_not_in_text_but_section_exists(self):
        """Bank resolved but not in any chunk text; Banka İstihbaratı header
        exists → fallback_banka_istihbarati_only (LLM still runs)."""
        result = _apply_guardrail_standalone(
            "AKBANK T.A.Ş", "high",
            [self.BANKA_CHUNK_WITH_EMLAK, self.KAYNAK_CHUNK],
        )
        assert result["passed"] is True
        assert result["scoping_strategy"] == "fallback_banka_istihbarati_only"

    def test_wrong_bank_mention_blocked(self):
        """Confidence too low → blocked, safe_answer."""
        result = _apply_guardrail_standalone(
            None, "low",
            [self.BANKA_CHUNK_WITH_EMLAK],
        )
        assert result["passed"] is False
        assert result["scoping_strategy"] == "blocked"

    def test_no_relevant_chunks_blocked(self):
        """Bank resolved, but no Banka İstihbaratı chunks at all → blocked."""
        result = _apply_guardrail_standalone(
            "AKBANK T.A.Ş", "high",
            [self.KAYNAK_CHUNK],
        )
        assert result["passed"] is False
        assert result["scoping_strategy"] == "blocked"

    def test_unresolved_bank_blocked(self):
        """No normalized_bank_name → blocked."""
        result = _apply_guardrail_standalone(
            None, "none",
            [self.BANKA_CHUNK_WITH_EMLAK],
        )
        assert result["passed"] is False
        assert result["scoping_strategy"] == "blocked"
