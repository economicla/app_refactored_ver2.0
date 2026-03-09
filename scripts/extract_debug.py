#!/usr/bin/env python3
"""
İstihbarat raporu PDF'ini extract edip debug çıktısını (memzuc dahil) yazdırır.
Kullanım (proje kökünden):
  python scripts/extract_debug.py path/to/rapor.pdf
"""
import json
import sys
from pathlib import Path

# Proje kökünü path'e ekle
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app_refactored.structured_extractors import CreditIntelligencePDFExtractor


def main() -> None:
    if len(sys.argv) < 2:
        print("Kullanım: python scripts/extract_debug.py <pdf_dosyasi>", file=sys.stderr)
        sys.exit(1)
    file_path = Path(sys.argv[1])
    if not file_path.is_file():
        print(f"Dosya bulunamadı: {file_path}", file=sys.stderr)
        sys.exit(1)

    extractor = CreditIntelligencePDFExtractor()
    data = extractor.extract(str(file_path))

    debug = data.get("debug", {})
    print("=== DEBUG ===")
    print(json.dumps(debug, indent=2, ensure_ascii=False))

    memzuc = (data.get("sections") or {}).get("memzuc_doluluk_fallback") or []
    print("\n=== MEMZUC_DOLULUK satırları ===")
    for line in memzuc:
        if line.startswith("MEMZUC_DOLULUK"):
            print(line)
        else:
            print(f"(debug satırı) {line[:80]}...")


if __name__ == "__main__":
    main()
