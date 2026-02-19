
"""

DocumentIngestionUseCase - Doküman yükleme işlem hattı (Business Logic)

PDF, DOCX, TXT, XLSX, PPTX desteği

"""

import logging

from datetime import datetime

from pathlib import Path

from typing import List
 
from app_refactored.core.interfaces import (

    IEmbeddingService,

    IDocumentRepository

)

from app_refactored.core.entities import (

    DocumentChunk,

    DocumentIngestionResult

)

from .document_preprocessing import DocumentPreprocessor
from .intelligent_chunking import IntelligentChunker
from .html_structured_converter import HTMLStructuredConverter
 
logger = logging.getLogger(__name__)
 
 
class DocumentIngestionUseCase:

    """Doküman yükleme use case - Dependency Injection ile"""
 
    def __init__(

        self,

        embedding_service: IEmbeddingService,

        document_repository: IDocumentRepository,

        chunk_size: int = 1000,

        chunk_overlap: int = 200

    ):

        """

        Initialize document ingestion use case

        Args:

            embedding_service: IEmbeddingService implementation

            document_repository: IDocumentRepository implementation

            chunk_size: Chunk boyutu (karakterler)

            chunk_overlap: Chunk'lar arası overlap

        """

        self.embedding_service = embedding_service

        self.document_repository = document_repository

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap
 
    def _extract_text(self, file_path: str) -> str:

        """Dosya türüne göre metni çıkart"""

        path = Path(file_path)

        file_type = path.suffix.lower()
 
        try:

            if file_type == '.pdf':

                return self._extract_pdf(file_path)

            elif file_type == '.docx':

                return self._extract_docx(file_path)

            elif file_type == '.txt':

                return self._extract_txt(file_path)

            elif file_type in ('.html', '.html'):
                return self._extract_html(file_path)

            elif file_type == '.xlsx':

                return self._extract_xlsx(file_path)

            elif file_type == '.pptx':

                return self._extract_pptx(file_path)

            else:

                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:

            logger.error(f"❌ Text extraction failed: {str(e)}")

            raise
 
    def _extract_pdf(self, file_path: str) -> str:

        """PDF'den metin çıkart - pdfplumber (tablo destekli) + PyMuPDF fallback"""

        try:

            import pdfplumber

            pages_text = []

            with pdfplumber.open(file_path) as pdf:

                for page_num, page in enumerate(pdf.pages):

                    page_parts = []

                    page_parts.append(f"## Sayfa {page_num + 1}")

                    # Tabloları çıkar

                    tables = page.extract_tables()

                    if tables:

                        for table in tables:

                            for row in table:

                                cells = [str(c).strip() if c else "" for c in row]

                                non_empty = [c for c in cells if c]
                                #Key:Value çiftlerini tespit et
                                i = 0
                                while i < len(non_empty):
                                    cell = non_empty[i]
                                    if cell.endswith(':') and i + 1 < len(non_empty):
                                        page_parts.append(f"{cell} {non_empty[i+1]}")
                                        i += 2
                                    elif ':' not in cell and i == 0 and len(non_empty) == 2:
                                        page_parts.append(f"{non_empty[0]}: {non_empty[1]}")
                                        i = len(non_empty)
                                    else:
                                        page_parts.append(cell)
                                        i += 1

                                
                    else:

                        text = page.extract_text()

                        if text:

                            page_parts.append(text.strip())

                    if len(page_parts) > 1:

                        pages_text.append("\n".join(page_parts))

            full_text = "\n\n".join(pages_text)

            if len(full_text.strip()) > 100:

                logger.info(f"✅ PDF pdfplumber extraction: {len(full_text)} chars, {len(pages_text)} pages")

                return full_text

            # pdfplumber yetersizse PyMuPDF fallback

            logger.warning(f"⚠️ pdfplumber yetersiz ({len(full_text)} chars), PyMuPDF deneniyor...")

            import fitz

            doc = fitz.open(file_path)

            pages_text = []

            for page_num, page in enumerate(doc):

                text = page.get_text("text")

                if text and text.strip():

                    pages_text.append(f"## Sayfa {page_num + 1}\n\n{text.strip()}")

            doc.close()

            full_text = "\n\n".join(pages_text)

            logger.info(f"✅ PDF PyMuPDF fallback: {len(full_text)} chars")

            return full_text

        except Exception as e:

            logger.error(f"❌ PDF extraction failed: {str(e)}")

            raise
 
    def _extract_docx(self, file_path: str) -> str:

        """DOCX'den metin çıkart"""

        try:

            from docx import Document

            doc = Document(file_path)

            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

            logger.info(f"✅ DOCX extraction: {len(text)} chars")

            return text

        except Exception as e:

            logger.error(f"❌ DOCX extraction failed: {str(e)}")

            raise
 
    def _extract_txt(self, file_path: str) -> str:

        """TXT'den metin çıkart"""

        try:

            with open(file_path, 'r', encoding='utf-8') as file:

                text = file.read()

            logger.info(f"✅ TXT extraction: {len(text)} chars")

            return text

        except Exception as e:

            logger.error(f"❌ TXT extraction failed: {str(e)}")

            raise
 
    def _extract_xlsx(self, file_path: str) -> str:

        """XLSX'den metin çıkart"""

        try:

            import openpyxl

            wb = openpyxl.load_workbook(file_path)

            text = ""

            for sheet in wb.sheetnames:

                ws = wb[sheet]

                for row in ws.iter_rows(values_only=True):

                    text += " ".join(str(cell) if cell else "" for cell in row) + "\n"

            logger.info(f"✅ XLSX extraction: {len(text)} chars")

            return text

        except Exception as e:

            logger.error(f"❌ XLSX extraction failed: {str(e)}")

            raise
 
    def _extract_pptx(self, file_path: str) -> str:

        """PPTX'den metin çıkart"""

        try:

            from pptx import Presentation

            prs = Presentation(file_path)

            text = ""

            for slide in prs.slides:

                for shape in slide.shapes:

                    if hasattr(shape, "text"):

                        text += shape.text + "\n"

            logger.info(f"✅ PPTX extraction: {len(text)} chars")

            return text

        except Exception as e:

            logger.error(f"❌ PPTX extraction failed: {str(e)}")

            raise

    def _extract_html(self, file_path: str) -> str:

        """HTML'den metin çıkart - HTMLStructuredConverter ile yapısal parse"""

        try:

            html_content = None

            for enc in ('utf-8', 'windows-1254', 'latin-1', 'iso-8859-9'):

                try:

                    with open(file_path, 'r', encoding=enc) as file:

                        html_content = file.read()

                    logger.info(f"📄 HTML decoded with encoding: {enc}")

                    break

                except UnicodeDecodeError:

                    continue

            if html_content is None:

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:

                    html_content = file.read()

            converter = HTMLStructuredConverter()

            result = converter.convert(html_content)

            logger.info(f"✅ HTML extraction: {len(result)} chars")

            return result

        except Exception as e:

            logger.error(f"❌ HTML extraction failed: {str(e)}")

            raise
 
    
    def _chunk_text(self, text: str) -> List[str]:

        """Metni chunk'lara böl"""

        chunks = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk = text[start:end]

            chunks.append(chunk)

            start += (self.chunk_size - self.chunk_overlap)

        logger.info(f"✅ Text chunked: {len(chunks)} chunks")

        return chunks
 
    async def execute(self, file_path: str, filename: str) -> DocumentIngestionResult:

        """

        Execute document ingestion pipeline

        Args:

            file_path: Dosyanın tam yolu

            filename: Orijinal dosya adı

        Returns:

            DocumentIngestionResult

        """

        try:

            logger.info(f"📥 Ingesting: {filename}")

            # Step 0: Aynı dosyanın eski chunk'larını temizle (re-upload desteği)
            try:
                deleted_count = await self.document_repository.delete_by_filename(filename)
                if deleted_count > 0:
                    logger.info(f"🗑️ Deleted {deleted_count} old chunks for {filename}")
            except Exception as del_err:
                logger.warning(f"⚠️ Old chunk cleanup failed (continuing): {del_err}")
 
            # Step 1: Metni çıkart

            logger.info(f"📄 Extracting text from {Path(filename).suffix}...")

            text = self._extract_text(file_path)

            original_length = len(text)

            logger.info("🧹 Preprocessing text...")
            preprocessor = DocumentPreprocessor()
            raw_type = Path(filename).suffix.lstrip('.')
            # HTML zaten HTMLStructuredConverter ile markdown'a dönüştürüldü
            # "md" olarak işle → to_markdown() tekrar çalışmaz, sadece clean_text() çalışır
            preprocess_type = "md" if raw_type in ("html", "htm") else raw_type
            text = preprocessor.preprocess(
                text,
                file_type=preprocess_type
            )
 
            # Step 2: Chunk'la

            logger.info("✂️ Intelligent chunking...")

            chunker = IntelligentChunker(chunk_size=1000, chunk_overlap=200)
            chunk_objects = chunker.chunk(text)
            chunks = [c['content'] for c in chunk_objects]
            #Kaynak prefix ekle
            doc_label = filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()
            chunks = [f"[Kaynak: {doc_label}]\n\n{c}" for c in chunks]
 
            # Step 3: Embedding'ler oluştur

            logger.info("📊 Generating embeddings...")

            embeddings = await self.embedding_service.embed_batch(chunks)
 
            if len(embeddings) != len(chunks):

                raise Exception(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")
 
            # Step 4: Veritabanına kaydet

            logger.info("💾 Saving to database...")

            documents = [

                DocumentChunk(

                    filename=filename,

                    chunk_index=idx,

                    content=chunk,

                    embedding=embedding,

                    metadata={

                        "file_type": Path(filename).suffix,

                        "chunk_size": len(chunk),

                        "original_length": original_length,

                        "header": chunk_objects[idx].get('header'),

                        "source": filename,

                        "chunk_position": chunk_objects[idx].get('position')

                    }

                )

                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))

            ]
 
            saved_docs = await self.document_repository.save_batch(documents)
 
            # Token sayısı (rough estimate: 1 token ≈ 4 chars)

            total_tokens = original_length // 4
 
            logger.info(f"✅ Ingestion completed: {len(saved_docs)} chunks, {total_tokens} tokens")
 
            return DocumentIngestionResult(

                status="success",

                filename=filename,

                chunks_ingested=len(saved_docs),

                total_tokens=total_tokens,

                timestamp=datetime.now()

            )
 
        except Exception as e:

            logger.error(f"❌ Document ingestion failed: {str(e)}")

            return DocumentIngestionResult(

                status="error",

                filename=filename,

                chunks_ingested=0,

                total_tokens=0,

                timestamp=datetime.now(),

                error=str(e)

            )
 
