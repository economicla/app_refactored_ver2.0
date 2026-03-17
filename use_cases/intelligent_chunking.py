"""

Intelligent Text Chunking

RecursiveCharacterTextSplitter + Markdown headers  (hierarchy-aware)

"""
 
import logging

from typing import List, Tuple, Optional, Dict

import re
 
logger = logging.getLogger(__name__)
 
 
class IntelligentChunker:

    """

    Anlamsal bütünlüğü koruyan chunker

    Markdown başlık hiyerarşisini korur:
      ## Ana Bölüm  →  ### Alt Bölüm  hiyerarşisinde
      alt bölüm chunk'ları "Ana Bölüm > Alt Bölüm" header bilgisi taşır.

    """

    # Kayıt bazlı bölümleme: chunk ortasında tablo/satır kesilmesin
    BANKA_RECORDS_PER_CHUNK = 10
    E_HACIZ_RECORDS_PER_CHUNK = 8
    LIMIT_RISK_OR_KAYNAK_LINES_PER_CHUNK = 12

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

    def find_sections(self, text: str) -> List[Tuple[str, int, int, int]]:

        """

        Markdown başlıklarını bul

        Returns: [(title, level, start_pos, end_pos), ...]

        """

        sections = []

        header_pattern = r'^(#{1,6})\s+(.+)$'

        for match in re.finditer(header_pattern, text, re.MULTILINE):

            level = len(match.group(1))

            title = match.group(2).strip()

            sections.append((title, level, match.start(), match.end()))

        return sections

    def chunk_by_headers(self, text: str) -> List[dict]:

        """

        Markdown başlıklarına göre hiyerarşik parçala.
        Üst başlık (##) bilgisi alt başlık (###) chunk'larına taşınır.

        """

        sections = self.find_sections(text)

        if not sections:

            # Başlık yoksa paragraflar halinde böl

            return self._chunk_by_paragraphs(text)

        # ── Üst başlık hiyerarşisi: her seviye için en son görülen başlık ──
        parent_headers: Dict[int, str] = {}  # level → title
        chunks = []

        for i, (title, level, start, end) in enumerate(sections):

            # Bir sonraki başlığa kadar metni al

            next_start = sections[i + 1][2] if i + 1 < len(sections) else len(text)

            section_text = text[start:next_start].strip()

            # Hiyerarşiyi güncelle
            parent_headers[level] = title
            # Bu seviyeden düşük seviyeleri temizle (yeni üst bölüme geçildi)
            for lv in list(parent_headers.keys()):
                if lv > level:
                    del parent_headers[lv]

            # Birleşik header oluştur (## Ana > ### Alt)
            composite_parts = []
            for lv in sorted(parent_headers.keys()):
                if lv < level:
                    composite_parts.append(parent_headers[lv])
            composite_parts.append(title)
            composite_header = " > ".join(composite_parts) if len(composite_parts) > 1 else title

            if section_text:

                chunks.append({

                    'content': section_text,

                    'header': composite_header,

                    'level': level,

                    'position': i

                })

        return chunks

    def _chunk_by_paragraphs(self, text: str, header: Optional[str] = None) -> List[dict]:

        """

        Paragraflar halinde böl

        """

        paragraphs = text.split('\n\n')

        #Eğer tek parça kaldıysa \n ile böl
        if len(paragraphs) <= 1:
            paragraphs = text.split('\n')

        chunks = []

        current_chunk = ""

        for para in paragraphs:

            if len(current_chunk) + len(para) < self.chunk_size:

                current_chunk += para + "\n\n"

            else:

                if current_chunk:

                    chunks.append({

                        'content': current_chunk.strip(),

                        'header': header,

                        'level': 0,

                        'position': len(chunks)

                    })

                current_chunk = para + "\n\n"

        if current_chunk:

            chunks.append({

                'content': current_chunk.strip(),

                'header': header,

                'level': 0,

                'position': len(chunks)

            })

        return chunks

    def _chunk_banka_istihbarati_by_records(self, content: str, section_header: str) -> List[dict]:
        """
        Banka İstihbaratı bölümünü kayıt bazlı böl; satır ortasından kesme.
        Her kayıt 'Banka: ...' ile başlar; '  Not:' satırları önceki kayda ait.
        """
        lines = content.splitlines()
        records: List[List[str]] = []
        current: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    records.append(current)
                    current = []
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("Banka:"):
                if current:
                    records.append(current)
                current = [line]
            elif stripped.startswith("Not:") and current:
                current.append(line)
            elif line.startswith("  ") and current:
                current.append(line)
            else:
                if current:
                    records.append(current)
                    current = []

        if current:
            records.append(current)

        n = self.BANKA_RECORDS_PER_CHUNK
        out = []
        for i in range(0, len(records), n):
            group = records[i : i + n]
            block = "\n".join(line for rec in group for line in rec)
            if not block.strip():
                continue
            out.append({"content": "## Banka İstihbaratı\n\n" + block})
        return out if out else [{"content": content}]

    def _chunk_section_by_lines(
        self,
        content: str,
        section_title: str,
        line_prefix: str,
        lines_per_chunk: int,
    ) -> List[dict]:
        """
        Bölümü satır bazlı böl; her chunk'a başlık + isteğe bağlı prefix (örn. Bin TL) ekle.
        Limit Risk Bilgileri ve Kaynak Bazında Detay için tablo ortasından kesilmez.
        """
        lines = content.splitlines()
        data_lines: List[str] = []
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("Tüm tutarlar Bin TL"):
                continue
            data_lines.append(line)
        if not data_lines:
            return [{"content": content}]
        out = []
        for i in range(0, len(data_lines), lines_per_chunk):
            group = data_lines[i : i + lines_per_chunk]
            block = "\n".join(group)
            header_block = f"## {section_title}\n\n{line_prefix}\n\n" if line_prefix else f"## {section_title}\n\n"
            out.append({"content": header_block + block})
        return out if out else [{"content": content}]

    def chunk(self, text: str) -> List[dict]:

        """

        Ana chunking metodu

        """

        logger.info(f"🔪 Chunking text (size={self.chunk_size}, overlap={self.chunk_overlap})...")

        chunks = self.chunk_by_headers(text)
        final_chunks = []

        for c in chunks:
            content = c['content']
            header = c.get('header') or ''
            if len(content.strip()) < 50:
                continue
            # Banka İstihbaratı: kayıt bazlı böl (satır ortasından kesme; her chunk'ta N banka kaydı)
            if 'Banka İstihbaratı' in header or 'banka_istihbarati' in header.lower():
                sub_chunks = self._chunk_banka_istihbarati_by_records(content, header)
                for sub in sub_chunks:
                    sub['header'] = c['header']
                    sub['level'] = c['level']
                    sub['position'] = c.get('position', 0)
                    final_chunks.append(sub)
                continue
            # E-Haciz Tarihçesi: satır = kayıt (Firma | Yıl | Ödendi Adet | Ödendi Tutar | Ödenmedi Adet | Ödenmedi Tutar)
            if 'E-Haciz' in header or 'e_haciz' in header.lower():
                sub_chunks = self._chunk_section_by_lines(
                    content,
                    "E-Haciz Tarihçesi",
                    "",
                    self.E_HACIZ_RECORDS_PER_CHUNK,
                )
                for sub in sub_chunks:
                    sub['header'] = c['header']
                    sub['level'] = c['level']
                    sub['position'] = c.get('position', 0)
                    final_chunks.append(sub)
                continue
            # Limit Risk Bilgileri / Kaynak Bazında Detay: Bin TL birimi + satır bazlı böl (tablo ortasından kesme)
            if 'Limit Risk' in header or 'Kaynak Bazında' in header or 'limit_risk' in header.lower() or 'kaynak_bazinda' in header.lower():
                section_title = "Limit Risk Bilgileri (Bin TL)" if 'Limit Risk' in header or 'limit_risk' in header.lower() else "Kaynak Bazında Detay (Bin TL)"
                sub_chunks = self._chunk_section_by_lines(
                    content,
                    section_title,
                    "Tüm tutarlar Bin TL'dir.",
                    self.LIMIT_RISK_OR_KAYNAK_LINES_PER_CHUNK,
                )
                for sub in sub_chunks:
                    sub['header'] = c['header']
                    sub['level'] = c['level']
                    sub['position'] = c.get('position', 0)
                    final_chunks.append(sub)
                continue
            if len(content) > self.chunk_size * 1.2:
                sub_chunks = self._chunk_by_paragraphs(content, header=c['header'])
                for sub in sub_chunks:
                    if len(sub['content'].strip()) < 50:
                        continue
                    sub['header'] = c['header']
                    sub['level'] = c['level']
                    final_chunks.append(sub)
            else:
                final_chunks.append(c)

        logger.info(f"✅ Created {len(final_chunks)} chunks")

        return final_chunks
 
