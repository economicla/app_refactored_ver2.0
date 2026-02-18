"""

Advanced Document Preprocessing

HTML, Markdown, PDF, DOCX temizliği ve normalizasyonu

"""
 
import logging

import re

from typing import Optional

from bs4 import BeautifulSoup

import markdown
 
logger = logging.getLogger(__name__)
 
 
class DocumentPreprocessor:

    """Production-grade document preprocessing"""

    @staticmethod

    def clean_html(html_content: str) -> str:

        """

        HTML'den anlamlı veriyi çıkar

        Sadece: h1-h6, p, li, table etiketlerini al

        """

        try:

            soup = BeautifulSoup(html_content, 'html.parser')

            # Script ve style bloklarını kaldır

            for script in soup(["script", "style"]):

                script.decompose()

            # İlgili etiketleri bul

            relevant_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'table'])

            text_parts = []

            for tag in relevant_tags:

                if tag.name.startswith('h'):

                    # Başlık olarak işaretle

                    level = int(tag.name[1])

                    text_parts.append(f"{'#' * level} {tag.get_text(strip=True)}")

                elif tag.name == 'p':

                    text_parts.append(tag.get_text(strip=True))

                elif tag.name == 'li':

                    text_parts.append(f"- {tag.get_text(strip=True)}")

                elif tag.name == 'table':

                    # Tablo metnini satır satır al

                    rows = tag.find_all('tr')

                    for row in rows:

                        cols = row.find_all(['td', 'th'])

                        row_text = " | ".join([col.get_text(strip=True) for col in cols])

                        text_parts.append(row_text)

            return "\n\n".join(text_parts)

        except Exception as e:

            logger.warning(f"HTML cleaning failed: {str(e)}")

            return html_content

    @staticmethod

    def to_markdown(text: str, source_format: str = "text") -> str:

        """

        Metni Markdown formatına çevir

        PDF/HTML → Markdown (başlık yapısını koru)

        """

        try:

            # Eğer zaten Markdown ise döndür

            if source_format == "md":

                return text

            # HTML ise önce temizle

            if source_format == "html":

                text = DocumentPreprocessor.clean_html(text)

            # Başlık kombinasyonlarını Markdown'a çevir

            # "BAŞLIK" veya "BAŞLIK\n---" → "# BAŞLIK"

            text = re.sub(r'^([A-ZÇĞIŞÖÜÂÎÛ][A-Za-z0-9\s\.,;:!?\-ÇĞİŞÖÜâîû]+)\n[=\-]+\s*$', 

                         r'# \1', text, flags=re.MULTILINE)

            # Listeleri normalize et

            text = re.sub(r'^\s*[•✓*]\s+', '- ', text, flags=re.MULTILINE)

            return text

        except Exception as e:

            logger.warning(f"Markdown conversion failed: {str(e)}")

            return text

    @staticmethod

    def clean_text(text: str) -> str:

        """

        Gereksiz boşluklar ve karakterleri temizle

        """

        # Fazla boşlukları kaldır

        text = re.sub(r'[^\S\n]+', ' ', text)       # Boşlukları temizle ama \n'leri koru

        text = re.sub(r'\n{3,}', '\n\n', text)       # 3+ satır sonunu 2'ye düşür
 

        # Satır başındaki ve sonundaki boşlukları kaldır

        text = re.sub(r'^[^\S\n]+|[^\S\n]+$', '', text, flags=re.MULTILINE)

        # Tekrarlayan noktalama işaretlerini kaldır

        text = re.sub(r'([.?!]){2,}', r'\1', text)

        # Kontrol karakterlerini kaldır

        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

        return text.strip()

    @staticmethod

    def preprocess(

        content: str,

        file_type: str = "txt",

        clean: bool = True,

        to_markdown: bool = True

    ) -> str:

        """

        Tüm preprocessing adımlarını çalıştır

        Args:

            content: Dokuman metni

            file_type: pdf, docx, html, txt, md

            clean: Temizlik yapılsın mı

            to_markdown: Markdown'a çevrilsin mi

        Returns:

            İşlenmiş ve temizlenmiş metin

        """

        logger.info(f"📝 Preprocessing {file_type} document...")

        # HTML ise temizle

        if file_type.lower() == "html":

            content = DocumentPreprocessor.clean_html(content)

        # Markdown'a çevir

        if to_markdown:

            content = DocumentPreprocessor.to_markdown(content, source_format=file_type)

        # Final temizliği yap

        if clean:

            content = DocumentPreprocessor.clean_text(content)

        logger.info(f"✅ Preprocessing complete. Length: {len(content)} chars")

        return content
 
