
"""

IEmbeddingService Interface - Soyut Embedding Servisi

Herhangi bir embedding provider'ı implement edebilir (Jina, OpenAI, vb.)

"""

from abc import ABC, abstractmethod

from typing import List

from app_refactored.core.entities.domain_models import EmbeddingResult
 
 
class IEmbeddingService(ABC):

    """Embedding hizmetinin soyut arayüzü"""
 
    @abstractmethod

    async def embed_text(self, text: str) -> List[float]:

        """

        Tek bir metni embedding'e çevir

        Args:

            text: Embedding'e çevrilecek metin

        Returns:

            Embedding vektörü (List[float])

        Raises:

            EmbeddingServiceError: Embedding başarısız olursa

        """

        pass
 
    @abstractmethod

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:

        """

        Birden fazla metni batch olarak embedding'e çevir

        Args:

            texts: Embedding'e çevrilecek metinler listesi

        Returns:

            Embedding vektörleri listesi (List[List[float]])

        Raises:

            EmbeddingServiceError: Embedding başarısız olursa

        """

        pass
 
    @abstractmethod

    async def get_dimension(self) -> int:

        """

        Embedding vektörünün boyutunu döndür (örn: 2048)

        Returns:

            Boyut (int)

        """

        pass
 
    @abstractmethod

    async def is_available(self) -> bool:

        """

        Servisi kontrol et - sağlıklı mı?

        Returns:

            True: Servis çalışıyor, False: Çalışmıyor

        """

        pass
 
