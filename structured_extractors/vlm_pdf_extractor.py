"""
VLM-based PDF Extractor — Qwen3-VL ile sayfa görüntülerinden structured markdown üretir.

Pipeline:  PDF → sayfa görselleri → VLM (Qwen3-VL) → structured markdown → chunking → Vector DB

pdfplumber/regex tabanlı extraction yerine, Vision-Language Model doğrudan
sayfa görüntüsünü okuyup tablo ve metni anlar.
"""

import asyncio
import base64
import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


def _postprocess_markdown(text: str) -> str:
    """VLM çıktısındaki bilinen artefaktları temizle."""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</?(?:span|div|p|font|b|i|u|em|strong)[^>]*>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


SYSTEM_PROMPT = """\
Sen bir bankacılık dokümanı analiz uzmanısın. Sana verilen sayfa görüntüsünü \
dikkatle oku ve içeriği aşağıdaki kurallara uygun şekilde **Türkçe markdown** \
olarak üret.

KURALLAR:
1. Sayfadaki tüm metin ve tabloları eksiksiz aktar.
2. Bölüm başlıklarını `##` markdown başlığı olarak yaz.
3. Tabloları **markdown tablo** formatında yaz (| başlık1 | başlık2 | ... |).
4. Birleşik (merged) tablo başlıkları varsa bunları düz sütun başlıklarına aç. \
Örneğin "Ödendi" altında "Adet" ve "Tutar" varsa → "Ödendi Adet | Ödendi Tutar" yaz.
5. Sayısal değerleri, tarihleri, banka adlarını **olduğu gibi** koru — yuvarlama veya tahmin yapma.
6. Eğer sayfada grafik veya logo varsa, kısa bir açıklama yaz: `[Grafik: ...]`.
7. Para birimi varsa (TL, TRY, USD, EUR) sayının yanına yaz.
8. Boş veya anlamsız satırları atla.
9. HTML tagları (<br>, <span> vb.) KULLANMA — sadece saf markdown yaz.
10. Çıktında yalnızca markdown olsun — açıklama, yorum veya ek bilgi ekleme.
"""

PAGE_PROMPT_TEMPLATE = """\
Bu sayfa bir Kredi İstihbarat Raporu'nun {page_num}. sayfasıdır (toplam {total_pages} sayfa).

Sayfadaki tüm içeriği (metin, tablolar, başlıklar) kurallarıma uygun şekilde \
markdown olarak yaz.\
"""


def _pdf_to_images(pdf_path: str, dpi: int = 200) -> List[str]:
    """Convert PDF pages to temporary PNG files. Returns list of file paths."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) gerekli: pip install PyMuPDF")

    tmp_dir = tempfile.mkdtemp(prefix="vlm_pdf_")
    image_paths: List[str] = []
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(matrix=mat)
        out_path = str(Path(tmp_dir) / f"page_{page_num + 1:03d}.png")
        pix.save(out_path)
        image_paths.append(out_path)
        logger.debug(f"Page {page_num + 1} → {out_path} ({pix.width}x{pix.height})")

    doc.close()
    logger.info(f"📄 PDF → {len(image_paths)} page images (dpi={dpi})")
    return image_paths


def _load_image_folder(folder_path: str) -> List[str]:
    """Load pre-existing page images from a folder (Page1.png, Page2.png, ...)."""
    folder = Path(folder_path)
    candidates = sorted(
        folder.glob("Page*.png"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )
    if not candidates:
        candidates = sorted(
            folder.glob("page_*.png"),
            key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
        )
    paths = [str(p) for p in candidates]
    logger.info(f"📂 Loaded {len(paths)} pre-existing page images from {folder_path}")
    return paths


class VLMPDFExtractor:
    """Vision-Language Model tabanlı PDF okuyucu.

    Kullanım:
        extractor = VLMPDFExtractor(host="http://vllm-model.example.local", model="model-name")
        result = await extractor.extract("rapor.pdf")
        markdown_text = result["markdown"]
    """

    def __init__(
        self,
        host: str,
        port: int = 0,
        model: str = "",
        timeout: int = 600,
        max_concurrent: int = 1,
        dpi: int = 150,
    ):
        self.host = host.rstrip("/")
        self.port = port
        self.model = model
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.dpi = dpi
        base = f"{self.host}:{port}" if port else self.host
        self.api_url = f"{base}/v1/chat/completions"

    async def extract(
        self,
        pdf_path: str,
        *,
        image_folder: Optional[str] = None,
        source_display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structured markdown from a PDF using VLM.

        Args:
            pdf_path: Path to the PDF file.
            image_folder: Optional path to pre-rendered page images.
                          If provided, skips PDF→image conversion.
            source_display_name: Orijinal yükleme dosya adı (örn. rapor.pdf). Verilmezse
                pdf_path'in basename'i kullanılır; tempfile adları chunk başlıklarında kötü görünür.

        Returns:
            Dict with keys: markdown, meta, pages (per-page results).
        """
        if image_folder:
            image_paths = _load_image_folder(image_folder)
        else:
            image_paths = _pdf_to_images(pdf_path, dpi=self.dpi)

        if not image_paths:
            raise ValueError(f"No page images found for {pdf_path}")

        total = len(image_paths)
        logger.info(f"🚀 VLM extraction starting: {total} pages, model={self.model}")

        semaphore = asyncio.Semaphore(self.max_concurrent)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=self.max_concurrent + 2, max_keepalive_connections=self.max_concurrent),
        ) as client:
            tasks = [
                self._extract_page(client, img_path, idx + 1, total, semaphore)
                for idx, img_path in enumerate(image_paths)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        pages: List[Dict[str, Any]] = []
        markdown_parts: List[str] = []
        errors: List[str] = []

        raw_label = (source_display_name or "").strip() or Path(pdf_path).name
        source_file = Path(raw_label).name
        markdown_parts.append(f"# İstihbarat Raporu — {source_file}\n")
        markdown_parts.append(f"Sayfa sayısı: {total}\n")

        for idx, res in enumerate(results):
            page_num = idx + 1
            if isinstance(res, Exception):
                err_msg = f"Sayfa {page_num} hata: {res}"
                logger.error(f"❌ {err_msg}")
                errors.append(err_msg)
                pages.append({"page": page_num, "status": "error", "error": str(res)})
            else:
                pages.append({"page": page_num, "status": "ok", "char_count": len(res)})
                markdown_parts.append(f"\n---\n## Sayfa {page_num}\n")
                markdown_parts.append(res)

        full_markdown = "\n".join(markdown_parts)

        logger.info(
            f"✅ VLM extraction done: {total} pages, "
            f"{len(errors)} errors, {len(full_markdown)} chars total"
        )

        return {
            "markdown": full_markdown,
            "meta": {
                "source_file": source_file,
                "pages": total,
                "model": self.model,
                "errors": len(errors),
            },
            "pages": pages,
            "errors": errors,
        }

    async def extract_single_page(
        self,
        image_path: str,
        page_num: int = 1,
        total_pages: int = 1,
    ) -> str:
        """Extract a single page image — useful for testing."""
        semaphore = asyncio.Semaphore(1)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
        ) as client:
            return await self._extract_page(client, image_path, page_num, total_pages, semaphore)

    async def _extract_page(
        self,
        client: httpx.AsyncClient,
        image_path: str,
        page_num: int,
        total_pages: int,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Send one page image to the VLM and return markdown."""
        async with semaphore:
            img_file = Path(image_path)
            raw = img_file.read_bytes()
            img_size_kb = len(raw) / 1024
            b64 = base64.b64encode(raw).decode("utf-8")
            suffix = img_file.suffix.lower().lstrip(".")
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")

            logger.info(f"📤 VLM sayfa {page_num}/{total_pages}: {img_file.name}, {img_size_kb:.0f} KB")

            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": PAGE_PROMPT_TEMPLATE.format(page_num=page_num, total_pages=total_pages),
                },
            ]

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0,
                "max_tokens": 8192,
                "top_p": 0.9,
                "stream": False,
            }

            t0 = time.monotonic()
            logger.debug(f"VLM POST → {self.api_url} (payload ~{len(str(payload)) / 1024 / 1024:.1f} MB)")
            response = await client.post(self.api_url, json=payload)
            response.raise_for_status()
            elapsed_ms = (time.monotonic() - t0) * 1000
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            answer = _postprocess_markdown(answer)

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            is_refusal = any(kw in answer[:100] for kw in ["erişemiyorum", "görüntüsünü paylaş", "Lütfen sayfa"])
            if is_refusal:
                logger.warning(f"⚠️ VLM sayfa {page_num}: Model görüntüyü göremedi! İlk 100 char: {answer[:100]}")
            else:
                logger.info(
                    f"✅ VLM sayfa {page_num}: {len(answer)} chars, "
                    f"{elapsed_ms:.0f}ms, tokens={prompt_tokens}+{completion_tokens}"
                )

            return answer
