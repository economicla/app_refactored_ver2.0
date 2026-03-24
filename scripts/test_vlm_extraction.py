"""
VLM PDF Extraction — Tek sayfa test scripti.

Kullanım (sunucuda):
    python -m scripts.test_vlm_extraction

Veya doğrudan:
    python scripts/test_vlm_extraction.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structured_extractors.vlm_pdf_extractor import VLMPDFExtractor

# ── Ayarlar ─────────────────────────────────────────────────────────
VLLM_HOST = "http://10.144.100.204"
VLLM_PORT = 8814
VLLM_MODEL = "RedHatAI/Qwen3-VL-32B-Instruct-NVFP4"

IMAGES_DIR = Path(__file__).resolve().parent.parent / "PDF - images"
# ────────────────────────────────────────────────────────────────────


async def test_single_page(page_num: int = 1):
    """Tek bir sayfa görüntüsünü VLM'e gönder ve çıktıyı göster."""
    img_path = IMAGES_DIR / f"Page{page_num}.png"
    if not img_path.exists():
        print(f"❌ Görüntü bulunamadı: {img_path}")
        return

    print(f"📤 Sayfa {page_num} gönderiliyor → {VLLM_MODEL}")
    print(f"   Görüntü: {img_path} ({img_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("-" * 60)

    extractor = VLMPDFExtractor(
        host=VLLM_HOST,
        port=VLLM_PORT,
        model=VLLM_MODEL,
    )

    result = await extractor.extract_single_page(
        image_path=str(img_path),
        page_num=page_num,
        total_pages=13,
    )

    print(result)
    print("-" * 60)
    print(f"✅ Çıktı uzunluğu: {len(result)} karakter")


async def test_all_pages():
    """Tüm sayfa görüntülerini VLM ile oku."""
    if not IMAGES_DIR.exists():
        print(f"❌ Görüntü klasörü bulunamadı: {IMAGES_DIR}")
        return

    extractor = VLMPDFExtractor(
        host=VLLM_HOST,
        port=VLLM_PORT,
        model=VLLM_MODEL,
        max_concurrent=2,
    )

    result = await extractor.extract(
        pdf_path="rapor.pdf",  # meta için kullanılır
        image_folder=str(IMAGES_DIR),
    )

    print(result["markdown"])
    print("=" * 60)
    print(f"✅ Toplam: {result['meta']['pages']} sayfa, {len(result['markdown'])} karakter")
    if result["errors"]:
        print(f"⚠️ Hatalar: {result['errors']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLM PDF Extraction Test")
    parser.add_argument("--page", type=int, default=0, help="Tek sayfa test (1-13). 0=tüm sayfalar.")
    args = parser.parse_args()

    if args.page > 0:
        asyncio.run(test_single_page(args.page))
    else:
        asyncio.run(test_all_pages())
