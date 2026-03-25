"""
Veritabanındaki chunk'ları kontrol et.

Kullanım:
    python scripts/debug_chunks.py
    python scripts/debug_chunks.py --filename yeni-istihbarat-raporu.pdf
    python scripts/debug_chunks.py --filename yeni-istihbarat-raporu.pdf --content
"""

import asyncio
import json as _json
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
            "left(content, 300) as preview, doc_metadata "
            "FROM documents WHERE filename = $1 AND deleted_at IS NULL "
            "ORDER BY chunk_index",
            filename,
        )
        print(f"\n📄 {filename}: {len(rows)} chunk(s)\n")
        total_chars = 0
        for r in rows:
            meta = r["doc_metadata"]
            if isinstance(meta, str):
                meta = _json.loads(meta)
            meta = meta or {}
            doc_type = meta.get("doc_type", "?")
            header = meta.get("header", "")
            total_chars += r["content_len"]
            print(f"  Chunk {r['chunk_index']:3d} | {r['content_len']:6d} chars | type: {doc_type} | header: {str(header)[:60]}")
            if show_content:
                print(f"    preview: {r['preview'][:250]}")
                print()
        print(f"\n  TOPLAM: {total_chars} karakter, {len(rows)} chunk")
    else:
        rows = await conn.fetch(
            "SELECT filename, count(*) as chunks, sum(length(content)) as total_chars "
            "FROM documents WHERE deleted_at IS NULL "
            "GROUP BY filename ORDER BY filename"
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
