"""
VLM PDF Extraction — Test scripti.

Kullanım (sunucuda):

  # Doğrudan PDF dosyasından (otomatik sayfa görseline çevirir):
    python scripts/test_vlm_extraction.py --pdf /path/to/rapor.pdf

  # Tek sayfa test:
    python scripts/test_vlm_extraction.py --pdf /path/to/rapor.pdf --page 1

  # Hazır görsel klasöründen:
    python scripts/test_vlm_extraction.py --images-dir /path/to/images/

  # Tek bir görsel dosyasından:
    python scripts/test_vlm_extraction.py --image /path/to/Page1.png
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structured_extractors.vlm_pdf_extractor import VLMPDFExtractor

# ── Varsayılan ayarlar (environment.env ile uyumlu) ─────────────────
VLLM_HOST = "http://10.144.100.204"
VLLM_PORT = 8814
VLLM_MODEL = "RedHatAI/Qwen3-VL-32B-Instruct-NVFP4"
# ────────────────────────────────────────────────────────────────────


def build_extractor(args) -> VLMPDFExtractor:
    return VLMPDFExtractor(
        host=args.host,
        port=args.port,
        model=args.model,
        max_concurrent=args.concurrent,
    )


async def test_single_image(args, image_path: str, page_num: int = 1, total: int = 1):
    p = Path(image_path)
    if not p.exists():
        print(f"❌ Görüntü bulunamadı: {p}")
        return
    print(f"📤 Sayfa {page_num} gönderiliyor → {args.model}")
    print(f"   Görüntü: {p} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    print("-" * 60)

    extractor = build_extractor(args)
    result = await extractor.extract_single_page(str(p), page_num, total)
    print(result)
    print("-" * 60)
    print(f"✅ Çıktı: {len(result)} karakter")


async def test_from_pdf(args):
    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"❌ PDF bulunamadı: {pdf}")
        return

    extractor = build_extractor(args)

    if args.page > 0:
        from structured_extractors.vlm_pdf_extractor import _pdf_to_images
        print(f"📄 PDF → sayfa görselleri çıkarılıyor...")
        images = _pdf_to_images(str(pdf))
        total = len(images)
        if args.page > total:
            print(f"❌ Sayfa {args.page} yok (toplam {total} sayfa)")
            return
        img = images[args.page - 1]
        await test_single_image(args, img, args.page, total)
    else:
        print(f"📄 Tüm sayfalar işleniyor: {pdf}")
        result = await extractor.extract(str(pdf))
        print(result["markdown"])
        print("=" * 60)
        meta = result["meta"]
        print(f"✅ Toplam: {meta['pages']} sayfa, {len(result['markdown'])} karakter, {meta['errors']} hata")
        if result["errors"]:
            for e in result["errors"]:
                print(f"   ⚠️ {e}")


async def test_from_images_dir(args):
    d = Path(args.images_dir)
    if not d.exists():
        print(f"❌ Klasör bulunamadı: {d}")
        return

    extractor = build_extractor(args)

    if args.page > 0:
        candidates = sorted(d.glob("Page*.png"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
        if not candidates:
            candidates = sorted(d.glob("page_*.png"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
        if args.page > len(candidates):
            print(f"❌ Sayfa {args.page} yok (toplam {len(candidates)} görsel)")
            return
        await test_single_image(args, str(candidates[args.page - 1]), args.page, len(candidates))
    else:
        result = await extractor.extract(pdf_path="rapor.pdf", image_folder=str(d))
        print(result["markdown"])
        print("=" * 60)
        meta = result["meta"]
        print(f"✅ Toplam: {meta['pages']} sayfa, {len(result['markdown'])} karakter, {meta['errors']} hata")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLM PDF Extraction Test")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=str, help="PDF dosya yolu")
    source.add_argument("--images-dir", type=str, help="Sayfa görselleri klasörü")
    source.add_argument("--image", type=str, help="Tek görsel dosyası")

    parser.add_argument("--page", type=int, default=0, help="Tek sayfa test (1,2,...). 0=tüm sayfalar.")
    parser.add_argument("--host", type=str, default=VLLM_HOST, help="vLLM host")
    parser.add_argument("--port", type=int, default=VLLM_PORT, help="vLLM port")
    parser.add_argument("--model", type=str, default=VLLM_MODEL, help="VLM model adı")
    parser.add_argument("--concurrent", type=int, default=2, help="Eşzamanlı sayfa sayısı")

    args = parser.parse_args()

    if args.image:
        asyncio.run(test_single_image(args, args.image))
    elif args.pdf:
        asyncio.run(test_from_pdf(args))
    elif args.images_dir:
        asyncio.run(test_from_images_dir(args))
