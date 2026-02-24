"""
HTML Structured Converter
Bankacılık HTML dokümanlarını yapısal olarak parse edip
RAG sistemi için optimize edilmiş doğal dil metnine dönüştürür.

Her veri noktası kendi bağlamıyla birlikte saklanır:
  - Tablo değerleri para birimi ile: "Risk 1.346.045.880 TRY"
  - Her satır bağımsız ve kendi kendine yeterli
  - Bölüm başlıkları korunur

Desteklenen 3 format:
  1. Teklif:     <tbody id="..."> iç içe yapı
  2. Performans: <div class="dl-bold"> + tablo yapısı
  3. Mali Veri:  düz/iç içe tablo yapısı

Temel özellik:
  PCN (Para Cinsi Notu) sütunlarını değer sütunlarıyla otomatik ilişkilendirir.
  Örn: Risk=1.346.045.880 + PCN=TRY → "Risk 1.346.045.880 TRY"
"""

import logging
import re
from typing import List, Optional, Dict, Set, Tuple

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


class HTMLStructuredConverter:
    """
    Bankacılık HTML dokümanlarını parse edip
    LLM/RAG sistemi için optimize edilmiş doğal dil metnine dönüştürür.
    """

    # PCN (Para Cinsi Notu) olarak kabul edilen sütun başlıkları
    PCN_KEYWORDS: Set[str] = {
        "PCN", "P.C.N", "PARA CİNSİ", "DÖVİZ", "DÖVİZ CİNSİ",
        "CUR", "CURRENCY", "P.CİNSİ", "PARA BIRIMI", "PARA BİRİMİ",
    }

    # Bilinen para birimi kodları
    CURRENCY_CODES: Set[str] = {"TRY", "USD", "EUR", "GBP", "CHF", "JPY", "TL"}

    # Teklif section ID → insan-okunur Türkçe başlık eşlemesi
    TEKLIF_SECTION_NAMES: Dict[str, str] = {
        "Header": "BAŞLIK BİLGİLERİ",
        "CommitteeName": "KOMİTE ADI",
        "Limits": "LİMİT BİLGİLERİ",
        "RatingValues": "RATING DEĞERLERİ",
        "CautionConditions": "TEMİNAT KOŞULLARI",
        "Guarantors": "KEFİLLER",
        "Constant": "ŞUBE TEKLİFİ VE KREDİ MÜDÜRLÜĞÜ GÖRÜŞÜ",
        "TeklifDetailInfo": "TEKLİF DETAY BİLGİLERİ",
        "OtherInfo": "GENEL DEĞERLENDİRME VE ORTAKLIK YAPISI",
    }

    # ================================================================
    # ANA DÖNÜŞTÜRME METODU
    # ================================================================

    def convert(self, html_content: str) -> str:
        """
        HTML içeriğini RAG-optimized doğal dil metnine dönüştürür.

        Args:
            html_content: Ham HTML string
        Returns:
            RAG için optimize edilmiş doğal dil metni
        """
        soup = BeautifulSoup(html_content, "html.parser")
        self._clean_soup(soup)

        format_type = self._detect_format(soup)
        self.detected_format = format_type
        logger.info(f"📋 HTML format tespit edildi: {format_type}")

        if format_type == "teklif":
            sections = self._parse_teklif(soup)
        elif format_type == "performans":
            sections = self._parse_performans(soup)
        else:
            sections = self._parse_mali_veri(soup)

        # Boş section'ları filtrele
        sections = [s for s in sections if s and s.strip()]

        # Bölümler zaten ## header ile ayrılıyor, --- ayracı chunk'ları bozuyor
        result = "\n\n".join(sections)
        logger.info(f"✅ HTML conversion: {len(result)} chars, {len(sections)} sections")
        return result

    # ================================================================
    # ORTAK YARDIMCI METODLAR
    # ================================================================

    def _clean_soup(self, soup: BeautifulSoup) -> None:
        """Script, style ve meta etiketlerini temizle, <br> etiketlerini koru."""
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()
        # <br> etiketlerini satır sonuna dönüştür (açıklama metinlerinde korunması için)
        for br in soup.find_all("br"):
            br.replace_with("\n")

    def _detect_format(self, soup: BeautifulSoup) -> str:
        """HTML formatını otomatik tespit et."""
        # Format 1: Teklif → tbody id yapısı
        tbodies_with_id = soup.find_all("tbody", id=True)
        if tbodies_with_id:
            return "teklif"

        # Format 2: Performans → div dl-bold yapısı
        if soup.find("div", class_=lambda c: c and "dl-bold" in c):
            return "performans"

        # Format 3: Mali Veri
        return "mali_veri"

    def _belongs_to_section(self, element: Tag, section_tbody: Tag) -> bool:
        """
        Element doğrudan bu section'a mı ait, yoksa iç içe bir alt section'a mı?
        DOM ağacında yukarı yürür: section_tbody'e ulaşırsa True,
        başka bir id'li tbody'e önce ulaşırsa False.
        """
        for parent in element.parents:
            if parent == section_tbody:
                return True
            if (isinstance(parent, Tag)
                    and parent.name == "tbody"
                    and parent.get("id")
                    and parent != section_tbody):
                return False
        return False

    def _is_bold_cell(self, cell: Optional[Tag]) -> bool:
        """Hücrenin bold olup olmadığını kontrol et (style, <b>, <strong>, background)."""
        if not cell:
            return False
        style = cell.get("style", "")
        if "font-weight" in style and "bold" in style:
            return True
        if cell.find("b") or cell.find("strong"):
            return True
        if "background-color" in style:
            return True
        # Class bazlı bold kontrolü
        cls = cell.get("class", [])
        if isinstance(cls, list):
            cls = " ".join(cls)
        if "bold" in str(cls).lower():
            return True
        return False

    # ================================================================
    # AKILLI TABLO PARSE (PCN-AWARE)
    # ================================================================

    def _extract_row_values(self, tr: Tag) -> List[str]:
        """
        TR'den hücre değerlerini çıkar.

        Standart <td>/<th> hücreleri varsa onları kullan.
        Tek hücreli satırlarda inner <div>/<span> elementlerini sanal hücre olarak kullan.
        Bu sayede CSS-tablo yapılarını (tek <td> + çok <div>) da parse edebiliriz.

        Örnek:
          <tr><td>                                     → 1 hücre
            <div>Label</div><div>val1</div><div>val2</div>
          </td></tr>
          → Standart: ["Labelval1val2"]  (tek hücre, birleşik)
          → Bu metod: ["Label", "val1", "val2"]  (sanal hücreler)
        """
        cells = tr.find_all(["td", "th"], recursive=False)
        texts = [c.get_text(strip=True) for c in cells]

        # 2+ gerçek hücre varsa → standart tablo, doğrudan dön
        if len(texts) >= 2:
            return texts

        # Tek hücre varsa → inner elementlerden sanal hücre çıkar
        if len(cells) == 1:
            cell = cells[0]

            # 1) Doğrudan child div/span elementlerini dene
            inner = cell.find_all(["div", "span"], recursive=False)
            if len(inner) >= 3:
                inner_texts = [el.get_text(strip=True) for el in inner]
                non_empty = [t for t in inner_texts if t.strip()]
                if len(non_empty) >= 3:
                    return inner_texts

            # 2) Herhangi bir child Tag'i dene (p, a, label, b vs.)
            children = [c for c in cell.children if isinstance(c, Tag)]
            if len(children) >= 3:
                child_texts = [c.get_text(strip=True) for c in children]
                non_empty = [t for t in child_texts if t.strip()]
                if len(non_empty) >= 3:
                    return child_texts

        return texts

    def _build_pcn_map(self, headers: List[str]) -> Dict[int, int]:
        """
        PCN sütunlarını tespit et ve her değer sütununu ilgili PCN sütunuyla eşleştir.

        Strateji: Soldan sağa tara. Bir PCN sütununa ulaştığında,
        önceki (son PCN'den sonraki) tüm değer sütunlarını bu PCN'e ata.

        Returns:
            {değer_sütun_index: pcn_sütun_index} eşlemesi
        """
        pcn_indices: List[int] = []
        for i, h in enumerate(headers):
            h_normalized = h.upper().strip().rstrip(".")
            if h_normalized in self.PCN_KEYWORDS:
                pcn_indices.append(i)

        col_to_pcn: Dict[int, int] = {}
        if pcn_indices:
            prev_boundary = 0
            for pcn_idx in pcn_indices:
                for col_idx in range(prev_boundary, pcn_idx):
                    if col_idx != 0:  # 0. sütun = satır etiketi
                        col_to_pcn[col_idx] = pcn_idx
                prev_boundary = pcn_idx + 1

        return col_to_pcn

    def _find_header_row(self, trs: List[Tag]) -> Tuple[int, List[str]]:
        """
        TR listesinde gerçek header satırını bul.
        Tek hücreli title/colspan satırlarını (ör: "LİMİT BİLGİLERİ") atlar,
        3+ hücreli ilk satırı header olarak döndürür.
        Div-tabloları da destekler (tek <td> + çok <div>).

        Returns:
            (header_index, header_texts) — bulunamazsa (-1, [])
        """
        for idx, tr in enumerate(trs):
            texts = self._extract_row_values(tr)
            # 3+ hücre (gerçek veya sanal) olan ilk satır = header
            if len(texts) >= 3:
                return idx, texts
        return -1, []

    def _parse_trs_smart(self, trs: List[Tag], section_title: str = "") -> str:
        """
        TR listesini PCN-aware olarak parse edip doğal dil metnine dönüştürür.

        Tek hücreli title satırlarını otomatik atlar ve gerçek header satırını bulur.
        Her satır bağımsız, kendi içinde tam bilgi içerir:
          "UMUMI: Risk 1.346.045.880 TRY, Mevcut Limit 2.025.000.000 TRY"

        Dönem sütunları varsa (2023/12, 2024/6 gibi) değerlerle otomatik eşleştirir:
          "Net Satışlar: 2024/6 döneminde 1.684.070.108, 2023/12 döneminde 1.142.758.961"

        Args:
            trs: <tr> Tag listesi
            section_title: Bölüm başlığı (opsiyonel)
        """
        if not trs:
            return ""

        lines: List[str] = []
        if section_title:
            lines.append(f"## {section_title}")

        # ── Header satırını bul (tek hücreli title satırlarını atla) ──
        header_idx, headers = self._find_header_row(trs)

        if header_idx < 0:
            # Hiç multi-cell satır yok → basit parse
            return self._parse_simple_trs(trs, section_title)

        # Header öncesi title satırlarını ekle
        for tr in trs[:header_idx]:
            title_text = tr.get_text(strip=True)
            if title_text:
                lines.append(f"### {title_text}")

        # ── PCN eşlemesini oluştur ──
        pcn_indices: Set[int] = set()
        for i, h in enumerate(headers):
            if h.upper().strip().rstrip(".") in self.PCN_KEYWORDS:
                pcn_indices.add(i)

        col_to_pcn = self._build_pcn_map(headers)

        # ── Dönem sütunlarını tespit et ──
        period_cols: Dict[int, str] = {}
        for i, h in enumerate(headers):
            period_match = re.search(r'(\d{4}[/\-]\d{1,2})', h)
            if period_match:
                period_cols[i] = period_match.group(1)

        # ── Data satırlarını işle (header'dan sonrası) ──
        for tr in trs[header_idx + 1:]:
            values = self._extract_row_values(tr)

            if not values or not any(v.strip() for v in values):
                continue

            # Tek hücreli satır → ara başlık/ayraç olabilir
            if len(values) == 1:
                if values[0].strip():
                    lines.append(f"\n### {values[0].strip()}")
                continue

            label = values[0] if values else ""
            parts: List[str] = []

            for i in range(1, len(headers)):
                # PCN sütununu atla (değerlerle birleştirildi)
                if i in pcn_indices:
                    continue
                if i >= len(values):
                    continue

                value = values[i].strip()
                if not value:
                    continue

                header_name = headers[i]

                # PCN değerini al ve değere ekle
                pcn = ""
                if i in col_to_pcn:
                    pcn_col = col_to_pcn[i]
                    if pcn_col < len(values):
                        pcn = values[pcn_col].strip()

                # Dönem + PCN birlikte
                if i in period_cols:
                    if pcn:
                        parts.append(f"{period_cols[i]} döneminde {value} {pcn}")
                    else:
                        parts.append(f"{period_cols[i]} döneminde {value}")
                elif pcn:
                    parts.append(f"{header_name} {value} {pcn}")
                else:
                    parts.append(f"{header_name}: {value}")

            if parts:
                lines.append(f"{label}: {', '.join(parts)}")
            elif label:
                lines.append(label)

        return "\n".join(lines)

    def _parse_table_smart(self, table: Tag, section_title: str = "") -> str:
        """
        Bir <table> elementini PCN-aware olarak parse eder.
        _parse_trs_smart'ın table-level wrapper'ı.
        """
        rows = table.find_all("tr")
        if not rows:
            return ""
        return self._parse_trs_smart(rows, section_title)

    def _parse_simple_trs(self, trs: List[Tag], section_title: str = "") -> str:
        """Tek/iki sütunlu basit tabloyu doğal dil metnine dönüştür.
        3+ sütunlu satırlar için currency-aware eşleştirme yapar.
        Div-tabloları da destekler (tek <td> + çok <div>)."""
        lines: List[str] = []
        if section_title:
            lines.append(f"## {section_title}")

        for tr in trs:
            texts = self._extract_row_values(tr)
            texts = [t for t in texts if t]

            if not texts:
                continue
            if len(texts) == 1:
                lines.append(texts[0])
            elif len(texts) == 2:
                lines.append(f"{texts[0]}: {texts[1]}")
            else:
                # 3+ sütun: currency-aware eşleştirme
                lines.append(self._flatten_with_currency(texts))

        return "\n".join(lines)

    def _flatten_with_currency(self, texts: List[str]) -> str:
        """
        Header olmadan bile para birimi kodlarını önceki değerlerle eşleştirir.
        Giriş: ["ÇEK KARNESİ", "0", "2.000.000", "TRY", "5.000.000", "TRY"]
        Çıkış: "ÇEK KARNESİ: 0 TRY, 2.000.000 TRY, 5.000.000 TRY"
        """
        if not texts:
            return ""

        label = texts[0]
        parts: List[str] = []
        i = 1
        while i < len(texts):
            val = texts[i].strip()
            # Sonraki hücre para birimi kodu mu?
            if (i + 1 < len(texts)
                    and texts[i + 1].strip().upper() in self.CURRENCY_CODES):
                parts.append(f"{val} {texts[i + 1].strip()}")
                i += 2
            else:
                parts.append(val)
                i += 1

        if parts:
            return f"{label}: {', '.join(parts)}"
        return label

    # ================================================================
    # FORMAT 1: TEKLİF  (<tbody id="..."> iç içe yapı)
    # ================================================================

    def _find_inner_tables(self, section_tbody: Tag) -> List[Tag]:
        """
        Section tbody içindeki iç içe tabloları bul.
        Sadece bu section'a ait tabloları döndürür (alt section'lardakileri değil).
        """
        tables = []
        for table in section_tbody.find_all("table"):
            if self._belongs_to_section(table, section_tbody):
                # En dıştaki tabloyu al, iç içe tabloların child'larını değil
                if not table.find_parent("table") or not self._belongs_to_section(
                    table.find_parent("table"), section_tbody
                ):
                    tables.append(table)
        return tables

    def _parse_teklif(self, soup: BeautifulSoup) -> List[str]:
        """
        Teklif HTML'ini bölüm bölüm parse eder.
        İç içe tbody yapısında her bölümün sadece kendi içeriğini alır.
        İç içe table yapılarını tespit edip doğrudan inner table'ı parse eder.
        """
        sections: List[str] = []

        for tbody in soup.find_all("tbody", id=True):
            section_id = tbody.get("id", "")
            display_name = self.TEKLIF_SECTION_NAMES.get(section_id, section_id)
            section_lines: List[str] = [f"## {display_name}"]

            # ── 1. İç içe tabloları bul ve parse et ──
            inner_tables = self._find_inner_tables(tbody)
            tables_parsed = False

            for table in inner_tables:
                # Inner table'daki TR'leri al
                inner_trs = table.find_all("tr")
                if not inner_trs:
                    continue

                # İlk TR'nin hücre sayısını kontrol et
                test_cells = inner_trs[0].find_all(["td", "th"], recursive=False)
                if len(test_cells) >= 2:
                    # Gerçek veri tablosu → akıllı parse
                    table_text = self._parse_trs_smart(inner_trs)
                    if table_text:
                        section_lines.append(table_text)
                        tables_parsed = True

            # ── 2. Inner table bulunamadıysa, doğrudan TR'leri dene ──
            if not tables_parsed:
                own_trs = [
                    tr for tr in tbody.find_all("tr", recursive=False)
                    if self._belongs_to_section(tr, tbody)
                ]
                # Bir TR'nin içinde nested table var mı kontrol et
                for tr in own_trs:
                    nested_table = tr.find("table")
                    if nested_table:
                        nested_trs = nested_table.find_all("tr")
                        if nested_trs:
                            test_cells = nested_trs[0].find_all(["td", "th"], recursive=False)
                            if len(test_cells) >= 2:
                                table_text = self._parse_trs_smart(nested_trs)
                                if table_text:
                                    section_lines.append(table_text)
                                    tables_parsed = True
                                    break

            # ── 3. Hâlâ tablo bulanamadıysa own_trs ile dene ──
            if not tables_parsed:
                own_trs = [
                    tr for tr in tbody.find_all("tr")
                    if self._belongs_to_section(tr, tbody)
                ]
                if own_trs:
                    table_text = self._parse_trs_smart(own_trs)
                    if table_text:
                        section_lines.append(table_text)

            # ── 2. Bu section'a ait div'leri parse et ──
            own_divs = [
                d for d in tbody.find_all("div")
                if self._belongs_to_section(d, tbody)
            ]

            for div in own_divs:
                div_text = div.get_text(strip=True)
                if not div_text or len(div_text) <= 10:
                    continue

                div_id = div.get("id", "")

                # Bilinen başlık id'leri
                if div_id in (
                    "SM_KTO_SubeTeklifi",
                    "SM_KTO_KrediMudurluguGorusu",
                    "KunyeSummary",
                    "PartnershipStructures",
                ):
                    section_lines.append(f"\n### {div_text}")
                elif div_id and "KTO_" in div_id:
                    # Alt bölüm detay metni
                    section_lines.append(div_text)
                elif div_text[0].isdigit() and ")" in div_text[:5]:
                    # Numaralı bölüm: "1) FİRMA TANITIMI"
                    section_lines.append(f"\n### {div_text}")
                elif div_text.startswith("-"):
                    # Madde imi
                    section_lines.append(div_text)
                elif self._looks_like_amount_block(div_text):
                    # Tutar satırı: "123.020.100 TRYa) 103.021.067 ..."
                    parsed = self._parse_amount_block(div_text)
                    section_lines.append(parsed)
                else:
                    section_lines.append(div_text)

            # Sadece başlıktan ibaret değilse ekle
            if len(section_lines) > 1:
                sections.append("\n".join(section_lines))

        return sections

    def _looks_like_amount_block(self, text: str) -> bool:
        """Metnin birleşik tutar bloğu olup olmadığını kontrol et."""
        # "123.020.100 TRYa)" veya "1.539.013 TRY" pattern
        return bool(re.search(r'[\d.]{5,}\s*(?:TRY|USD|EUR|GBP|TL)', text))

    def _parse_amount_block(self, text: str) -> str:
        """
        Birleşik tutar div'lerini ayrıştırıp okunabilir hale getirir.

        Giriş:  "123.020.100 TRYa) 103.021.067 TRY  İşletme Kredileri:67.000."
        Çıktı:  "Toplam: 123.020.100 TRY\n  a) 103.021.067 TRY - İşletme Kredileri: 67.000"
        """
        # Harf+) ile (a), b), c)...) alt kalemlere böl
        # Önce toplam tutarı ayır
        total_match = re.match(
            r'^([\d.]+)\s*(TRY|USD|EUR|GBP|TL)\s*',
            text
        )

        result_parts: List[str] = []

        if total_match:
            total_amount = total_match.group(1)
            total_currency = total_match.group(2)
            result_parts.append(f"Toplam: {total_amount} {total_currency}")
            remainder = text[total_match.end():]
        else:
            remainder = text

        # Alt kalemleri bul: a) ... b) ... c) ...
        sub_items = re.split(r'(?=\b[a-zğüşıöç]\)\s*)', remainder, flags=re.IGNORECASE)

        for item in sub_items:
            item = item.strip()
            if not item:
                continue
            # Temizle: fazla boşlukları düzelt
            item = re.sub(r'\s{2,}', ' ', item)
            # İçindeki tutar+açıklama yapısını düzenle
            item = re.sub(
                r'([\d.]+)\s*(TRY|USD|EUR|GBP|TL)\s*',
                r'\1 \2 - ',
                item
            )
            # Sondaki gereksiz tire/boşluk
            item = item.rstrip(' -')
            if item:
                result_parts.append(f"  {item}")

        return "\n".join(result_parts) if result_parts else text

    # ================================================================
    # FORMAT 2: PERFORMANS  (<div class="dl-bold"> + tablo yapısı)
    # ================================================================

    def _parse_performans(self, soup: BeautifulSoup) -> List[str]:
        """Performans raporu formatını parse et."""
        sections: List[str] = []

        # Ana başlık (dl-center + dl-bold, AMA dl-bg1 OLMAYAN)
        main_title = soup.find(
            "div",
            class_=lambda c: (
                c and "dl-bold" in c
                and "dl-center" in c
                and "dl-bg1" not in c
            ),
        )
        if main_title:
            sections.append(f"# {main_title.get_text(strip=True)}")

        # Bölüm başlıkları (dl-bold + dl-bg1)
        section_divs = soup.find_all(
            "div",
            class_=lambda c: c and "dl-bold" in c and "dl-bg1" in c,
        )

        for div in section_divs:
            title = div.get_text(strip=True)
            if not title:
                continue

            # Başlıktan sonraki ilk tabloyu bul
            table = div.find_next_sibling("table")
            if not table:
                table = div.find_next("table")

            if table:
                table_text = self._parse_table_smart(table, section_title=title)
                if table_text:
                    sections.append(table_text)
            else:
                sections.append(f"## {title}")

        return sections

    # ================================================================
    # FORMAT 3: MALİ VERİ  (düz/iç içe tablo yapısı)
    # ================================================================

    def _parse_mali_veri(self, soup: BeautifulSoup) -> List[str]:
        """
        Mali veri / bilanço / performans (tablo-tabanlı) formatını parse et.

        İç içe tablo yapısını (table > tr > td > table) destekler:
          - Wrapper TR'ler (tek <td> + nested table) otomatik atlanır
          - Dönem sütunları bölümler arası taşınır (shared period context)
          - Başlık bilgi tabloları (firma adı, para birimi vs.) ayrı yakalanır
          - Her bölüme firma adı eklenir (chunk'larda bağlam korunması için)
        """
        sections: List[str] = []

        top_tables = [
            t for t in soup.find_all("table")
            if not t.find_parent("table")
        ]

        # ── Pre-pass: Firma adını tespit et ──
        # Başlık tablolarından firma/grup adını çıkar
        firm_name = ""
        for table in top_tables:
            trs = table.find_all("tr", recursive=False)
            if not trs:
                continue
            all_single = all(
                len(tr.find_all(["td", "th"], recursive=False)) <= 1
                for tr in trs
            )
            if all_single:
                texts = [
                    tr.get_text(strip=True) for tr in trs
                    if tr.get_text(strip=True)
                ]
                if texts and len(texts) >= 2:
                    sections.append("\n".join(texts))
                    # Firma adını tespit et (ilk başlık tablosundaki 3. satır genelde firma/grup adı)
                    if not firm_name:
                        for t in texts:
                            # "584316-BAHARIYE GRUBU 2025-6" veya "584325-AKTÜL + MKS 2025-6" formatı
                            if re.search(r'\d{4,}-', t) and not t.startswith('1') and len(t) < 200:
                                # Koddan sonraki firma adını al
                                match = re.match(r'\d+-(.+?)(?:\s+\d{4}[-/]\d+.*)?$', t.strip())
                                if match:
                                    firm_name = match.group(1).strip()
                                    break

        if firm_name:
            logger.info(f"🏢 Firma adı tespit edildi: {firm_name}")

        # ── Shared period context across tables/sections ──
        shared_periods: Dict[int, str] = {}

        for table in top_tables:
            all_trs = table.find_all("tr")
            if not all_trs:
                continue

            current_title = ""
            current_data_trs: List[Tag] = []

            for tr in all_trs:
                cells = tr.find_all(["td", "th"], recursive=False)
                cell_texts = [c.get_text().strip() for c in cells]
                non_empty_texts = [t for t in cell_texts if t]

                if len(cells) <= 1:
                    # ── Wrapper cell (nested table içerir) → atla ──
                    # İç içe tablo TRleri zaten all_trs'te find_all ile bulunur
                    cell = cells[0] if cells else None
                    if cell and cell.find("table"):
                        continue

                    text = non_empty_texts[0] if non_empty_texts else ""
                    if text and len(text) > 3:
                        # Önceki bölümü kaydet
                        if current_data_trs:
                            section_text, detected = self._process_mali_section(
                                current_title, current_data_trs, shared_periods, firm_name
                            )
                            if detected:
                                shared_periods = detected
                            if section_text:
                                sections.append(section_text)

                        current_title = text
                        current_data_trs = []
                elif len(cells) >= 2:
                    current_data_trs.append(tr)

            # Son bölümü kaydet
            if current_data_trs:
                section_text, detected = self._process_mali_section(
                    current_title, current_data_trs, shared_periods, firm_name
                )
                if detected:
                    shared_periods = detected
                if section_text:
                    sections.append(section_text)

        return sections

    def _process_mali_section(
        self, title: str, data_trs: List[Tag],
        shared_periods: Optional[Dict[int, str]] = None,
        firm_name: str = ""
    ) -> Tuple[str, Dict[int, str]]:
        """
        Mali veri bölümünü dönem-aware olarak doğal dil metnine dönüştürür.

        Dönem sütunlarını tespit eder ve her değeri dönemiyle birlikte yazar:
          "Toplam Aktifler: 2023/12 döneminde 1.234.567, 2024/06 döneminde 2.345.678"

        Kendi dönem bilgisi yoksa shared_periods kullanır (önceki bölümlerden taşınan).

        Returns:
            (section_text, detected_periods)
            detected_periods: Bu bölümde tespit edilen dönem eşlemesi (boş olabilir)
        """
        if not data_trs:
            return "", {}

        # İlk çok-sütunlu satırı header olarak dene
        first_cells = data_trs[0].find_all(["td", "th"], recursive=False)
        first_texts = [c.get_text().strip() for c in first_cells]

        # Dönem sütunlarını tespit et
        period_cols: Dict[int, str] = {}
        is_first_row_header = False

        for i, h in enumerate(first_texts):
            period_match = re.search(r'(\d{4}[/\-]\d{1,2})', h)
            if period_match:
                period_cols[i] = period_match.group(1)
                is_first_row_header = True

        # Dönem/Period kelimesi var mı kontrol et
        if not is_first_row_header:
            for h in first_texts:
                if re.search(r'dönem|period', h, re.IGNORECASE):
                    is_first_row_header = True
                    break

        # Bu bölümde tespit edilen dönemler (dışarıya iletilecek)
        detected_periods = period_cols.copy() if period_cols else {}

        # Kendi dönem bilgisi yoksa shared context kullan
        if not period_cols and shared_periods:
            period_cols = shared_periods

        # Shared periods kullanılıyorsa, veri sütun sayısı uyumluluğunu doğrula
        # Açıklama tabloları (2 sütunlu) dönem verileriyle (4+ sütunlu) uyuşmaz
        if period_cols and not detected_periods and data_trs:
            max_period_idx = max(period_cols.keys())
            # İlk veri satırının sütun sayısını kontrol et
            ref_cells = data_trs[0].find_all(["td", "th"], recursive=False)
            if len(ref_cells) <= max_period_idx:
                period_cols = {}

        headers = first_texts if is_first_row_header else []
        data_start = 1 if is_first_row_header else 0

        # PCN sütunları da kontrol et (mali veride de olabilir)
        pcn_indices: Set[int] = set()
        col_to_pcn: Dict[int, int] = {}
        if headers:
            for i, h in enumerate(headers):
                if h.upper().strip().rstrip(".") in self.PCN_KEYWORDS:
                    pcn_indices.add(i)
            col_to_pcn = self._build_pcn_map(headers)

        lines: List[str] = []
        if title:
            lines.append(f"## {title}")
            if firm_name:
                lines.append(f"Firma/Grup: {firm_name}")

        for tr in data_trs[data_start:]:
            cells = tr.find_all(["td", "th"], recursive=False)
            texts = [c.get_text().strip() for c in cells]

            if not texts or not any(t.strip() for t in texts):
                continue

            label = texts[0]
            is_bold = self._is_bold_cell(cells[0]) if cells else False

            # Bold tek-sütun → alt bölüm başlığı
            if is_bold and len([t for t in texts if t.strip()]) <= 1:
                lines.append(f"\n### {label}")
                continue

            parts: List[str] = []
            for i in range(1, len(texts)):
                if i in pcn_indices:
                    continue

                val = texts[i].strip()
                if not val:
                    continue

                # PCN varsa değere ekle
                pcn = ""
                if i in col_to_pcn:
                    pcn_col = col_to_pcn[i]
                    if pcn_col < len(texts):
                        pcn = texts[pcn_col].strip()

                if pcn:
                    if i in period_cols:
                        parts.append(f"{period_cols[i]} döneminde {val} {pcn}")
                    elif headers and i < len(headers):
                        parts.append(f"{headers[i]} {val} {pcn}")
                    else:
                        parts.append(f"{val} {pcn}")
                elif i in period_cols:
                    parts.append(f"{period_cols[i]} döneminde {val}")
                elif headers and i < len(headers) and headers[i]:
                    parts.append(f"{headers[i]}: {val}")
                else:
                    parts.append(val)

            if parts:
                lines.append(f"{label}: {', '.join(parts)}")
            elif label:
                lines.append(label)

        return "\n".join(lines), detected_periods
