#!/usr/bin/env python3
"""
İstihbarat raporu PDF'ini extract edip debug çıktısını (memzuc dahil) yazdırır.
Kullanım:
  python scripts/extract_debug.py path/to/rapor.pdf
  (backend/app_refactored veya proje kökünden çalıştırılabilir)
"""
import json
import sys
from pathlib import Path

# app_refactored paketinin bulunması için: script'in olduğu dizinin 2 üstü (backend)
_script_dir = Path(__file__).resolve().parent
_app_refactored_dir = _script_dir.parent
_backend_dir = _app_refactored_dir.parent
for _d in (_backend_dir, _app_refactored_dir):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

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
