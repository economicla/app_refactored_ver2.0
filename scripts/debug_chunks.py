"""
Veritabanındaki chunk'ları kontrol et.

Kullanım:
    python scripts/debug_chunks.py                         # Tüm dokümanları listele
    python scripts/debug_chunks.py --filename yeni-istihbarat-raporu.pdf  # Belirli dosyanın chunk'ları
    python scripts/debug_chunks.py --filename yeni-istihbarat-raporu.pdf --content  # İçeriklerle birlikte
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / "environment.env"
load_dotenv(dotenv_path=str(env_path))


async def main(filename: str = None, show_content: bool = False):
    import asyncpg

    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = int(os.getenv("POSTGRES_PORT", "35432"))
    user = os.getenv("POSTGRES_USER", "testusr1")
    password = os.getenv("POSTGRES_PASSWORD", "")
    db = os.getenv("POSTGRES_DB", "testdb1")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    conn = await asyncpg.connect(dsn)

    if filename:
        rows = await conn.fetch(
            "SELECT id, chunk_index, length(content) as content_len, "
            "left(content, 200) as preview, metadata "
            "FROM document_chunks WHERE filename = $1 ORDER BY chunk_index",
            filename,
        )
        print(f"\n📄 {filename}: {len(rows)} chunk(s)\n")
        for r in rows:
            meta = r["metadata"] or {}
            doc_type = meta.get("doc_type", "?") if isinstance(meta, dict) else "?"
            header = meta.get("header", "") if isinstance(meta, dict) else ""
            print(f"  Chunk {r['chunk_index']:3d} | {r['content_len']:6d} chars | header: {header[:60]}")
            if show_content:
                print(f"    preview: {r['preview'][:150]}...")
            print()
    else:
        rows = await conn.fetch(
            "SELECT filename, count(*) as chunks, sum(length(content)) as total_chars "
            "FROM document_chunks GROUP BY filename ORDER BY filename"
        )
        print(f"\n📚 {len(rows)} doküman:\n")
        for r in rows:
            print(f"  {r['filename']:50s} | {r['chunks']:4d} chunks | {r['total_chars']:8d} chars")

    await conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--content", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args.filename, args.content))
