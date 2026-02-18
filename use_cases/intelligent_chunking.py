"""

Intelligent Text Chunking

RecursiveCharacterTextSplitter + Markdown headers

"""
 
import logging

from typing import List, Tuple

import re
 
logger = logging.getLogger(__name__)
 
 
class IntelligentChunker:

    """

    Anlamsal bütünlüğü koruyan chunker

    Markdown başlıkları ve paragraflon temel alır

    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

    def find_sections(self, text: str) -> List[Tuple[str, int, int]]:

        """

        Markdown başlıklarını bul

        Returns: [(header, start_pos, end_pos), ...]

        """

        sections = []

        header_pattern = r'^(#{1,6})\s+(.+)$'

        for match in re.finditer(header_pattern, text, re.MULTILINE):

            level = len(match.group(1))

            title = match.group(2)

            sections.append((title, level, match.start(), match.end()))

        return sections

    def chunk_by_headers(self, text: str) -> List[dict]:

        """

        Markdown başlıklarına göre parçala

        """

        sections = self.find_sections(text)

        if not sections:

            # Başlık yoksa paragraflar halinde böl

            return self._chunk_by_paragraphs(text)

        chunks = []

        for i, (title, level, start, end) in enumerate(sections):

            # Bir sonraki başlığa kadar metni al

            next_start = sections[i + 1][2] if i + 1 < len(sections) else len(text)

            section_text = text[start:next_start].strip()

            if section_text:

                chunks.append({

                    'content': section_text,

                    'header': title,

                    'level': level,

                    'position': i

                })

        return chunks

    def _chunk_by_paragraphs(self, text: str) -> List[dict]:

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

                        'header': None,

                        'level': 0,

                        'position': len(chunks)

                    })

                current_chunk = para + "\n\n"

        if current_chunk:

            chunks.append({

                'content': current_chunk.strip(),

                'header': None,

                'level': 0,

                'position': len(chunks)

            })

        return chunks

    def chunk(self, text: str) -> List[dict]:

        """

        Ana chunking metodu

        """

        logger.info(f"🔪 Chunking text (size={self.chunk_size}, overlap={self.chunk_overlap})...")

        chunks = self.chunk_by_headers(text)
        final_chunks = []

        for c in chunks:
            content = c['content']
            #Çok kısa chunk'ları atla
            if len(content.strip()) < 50:
                continue
            if len(content) > self.chunk_size * 1.2:
                #Büyük chunk'ı tekrar böl
                sub_chunks = self._chunk_by_paragraphs(content)
                for sub in sub_chunks:
                    #Header bilgisini koru
                    sub['header'] = c['header']
                    sub['level'] = c['level']
                    final_chunks.append(sub)
            else:
                final_chunks.append(c)

        logger.info(f"✅ Created {len(final_chunks)} chunks")

        return final_chunks
 
