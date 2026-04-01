
"""

JinaEmbeddingAdapter - Jina API'yi IEmbeddingService interface'ine adapt et

Async-first implementation with connection pooling

"""

import httpx

import logging
import time

from typing import List, Optional

from app_refactored.core.interfaces import IEmbeddingService
 
logger = logging.getLogger(__name__)
 
 
class JinaEmbeddingAdapter(IEmbeddingService):

    """Jina Embedding API async adapter with connection pooling"""
 
    def __init__(

        self,

        host: str,

        port: int,

        model: str,

        timeout: int = 600,

        max_connections: int = 10,

        embed_batch_size: int = 128,

    ):

        """

        Initialize Jina adapter with async client

        Args:

            host: Jina API host (örn: "http://10.144.100.204")

            port: Jina API port (örn: 38001)

            model: Model adı (örn: "jinaai/jina-embeddings-v4")

            timeout: Request timeout (saniye)

            max_connections: Connection pool maksimum bağlantı sayısı

            embed_batch_size: Tek HTTP isteğinde en fazla kaç metin (ReadTimeout / payload önlemi)

        """

        self.host = host

        self.port = port

        self.model = model

        self.timeout = timeout

        self.embed_batch_size = max(1, int(embed_batch_size))

        self.endpoint = f"{host}:{self.port}/embed/text"

        self.health_endpoint = f"{host}:{self.port}/health"

        self.dimension = 2048  # Jina v4 default dimension

        # AsyncClient with connection pooling

        self.client = httpx.AsyncClient(

            timeout=httpx.Timeout(timeout),

            limits=httpx.Limits(

                max_connections=max_connections,

                max_keepalive_connections=max_connections

            )

        )
 
    async def embed_text(self, text: str) -> List[float]:

        """Tek metni embed et"""

        embeddings = await self.embed_batch([text])

        return embeddings[0] if embeddings else []
 
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:

        """Birden fazla metni embed et; büyük listeleri alt-batch'lere böler (timeout / bellek)."""

        if not texts:
            return []

        bs = self.embed_batch_size
        if len(texts) <= bs:
            return await self._embed_batch_chunk(texts)

        t_all = time.monotonic()
        out: List[List[float]] = []
        n = len(texts)
        for start in range(0, n, bs):
            chunk = texts[start : start + bs]
            part = await self._embed_batch_chunk(chunk)
            if len(part) != len(chunk):
                raise RuntimeError(
                    f"Embedding count mismatch in sub-batch: got {len(part)}, expected {len(chunk)}"
                )
            out.extend(part)
        total_ms = (time.monotonic() - t_all) * 1000
        logger.info(
            f"✅ Embeddings (batched): {len(out)} vectors in {total_ms:.0f}ms "
            f"({(n + bs - 1) // bs} HTTP calls, sub_batch_size={bs})"
        )
        return out

    async def _embed_batch_chunk(self, texts: List[str]) -> List[List[float]]:

        """Tek HTTP isteği — en fazla embed_batch_size metin."""

        try:
            t0 = time.monotonic()
            payload = {
                "model": self.model,
                "texts": texts
            }

            response = await self.client.post(
                self.endpoint,
                json=payload
            )

            response.raise_for_status()
            elapsed_ms = (time.monotonic() - t0) * 1000

            result = response.json()
            embeddings = result.get("embeddings", [])

            avg_input_len = sum(len(t) for t in texts) // max(len(texts), 1)
            logger.info(
                f"✅ Embeddings: {len(embeddings)} vectors, {elapsed_ms:.0f}ms "
                f"(batch={len(texts)}, avg_input={avg_input_len} chars)"
            )

            return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Embedding HTTP {e.response.status_code}: {e.response.text[:300]}")
            raise
        except Exception as e:
            logger.error(f"❌ Embedding failed: {type(e).__name__}: {e}")
            raise
 
    async def get_dimension(self) -> int:

        """Embedding boyutunu döndür"""

        return self.dimension
 
    async def is_available(self) -> bool:

        """Jina servisinin çalışıp çalışmadığını kontrol et - async"""

        try:

            response = await self.client.get(

                self.health_endpoint,

                timeout=httpx.Timeout(5)

            )

            return response.status_code == 200

        except Exception as e:

            logger.warning(f"Jina health check failed: {str(e)}")

            return False
 
    async def close(self):

        """Client bağlantısını kapat"""

        await self.client.aclose()
 
    async def __aenter__(self):

        """Context manager support"""

        return self
 
    async def __aexit__(self, exc_type, exc_val, exc_tb):

        """Context manager cleanup"""

        await self.close()
 
