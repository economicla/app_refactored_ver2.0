
"""

IDocumentRepository Interface - Soyut Doküman Deposu

Veritabanı operasyonları için (PostgreSQL, MongoDB, vb.)

"""

from abc import ABC, abstractmethod

from typing import List, Optional

from app_refactored.core.entities.domain_models import DocumentChunk, SearchResult
 
 
class IDocumentRepository(ABC):

    """Doküman deposu soyut arayüzü"""
 
    @abstractmethod

    async def save(self, document: DocumentChunk) -> DocumentChunk:

        """

        Bir doküman chunk'ını veritabanına kaydet

        Args:

            document: Kaydedilecek DocumentChunk nesnesi

        Returns:

            Kaydedilmiş DocumentChunk (ID ile)

        Raises:

            RepositoryError: Kayıt başarısız olursa

        """

        pass
 
    @abstractmethod

    async def save_batch(self, documents: List[DocumentChunk]) -> List[DocumentChunk]:

        """

        Birden fazla chunk'ı batch olarak kaydet

        Args:

            documents: Kaydedilecek DocumentChunk nesneleri listesi

        Returns:

            Kaydedilmiş DocumentChunk'lar listesi

        Raises:

            RepositoryError: Kayıt başarısız olursa

        """

        pass
 
    @abstractmethod

    async def search_similar(

        self, 

        embedding: List[float], 

        top_k: int = 5,

        threshold: float = 0.0

    ) -> SearchResult:

        """

        Embedding'e benzer dokümanları ara (vector similarity)

        Args:

            embedding: Arama embedding'i

            top_k: Döndürülecek en iyi k sonuç

            threshold: Minimum benzerlik eşiği (0.0 - 1.0)

        Returns:

            SearchResult: Benzer dokümanlar listesi

        Raises:

            RepositoryError: Arama başarısız olursa

        """

        pass
 
    @abstractmethod

    async def get_by_filename(self, filename: str) -> List[DocumentChunk]:

        """

        Dosya adına göre tüm chunk'ları getir

        Args:

            filename: Doküman dosya adı

        Returns:

            DocumentChunk'lar listesi

        Raises:

            RepositoryError: Sorgu başarısız olursa

        """

        pass
 
    @abstractmethod

    async def get_by_id(self, chunk_id: int) -> Optional[DocumentChunk]:

        """

        ID'ye göre tek bir chunk getir

        Args:

            chunk_id: Chunk ID'si

        Returns:

            DocumentChunk veya None

        Raises:

            RepositoryError: Sorgu başarısız olursa

        """

        pass
 
    @abstractmethod

    async def delete(self, chunk_id: int) -> bool:

        """

        ID'ye göre chunk'ı sil

        Args:

            chunk_id: Silinecek chunk ID'si

        Returns:

            True: Başarılı, False: Başarısız

        Raises:

            RepositoryError: Silme başarısız olursa

        """

        pass
 
    @abstractmethod

    async def delete_by_filename(self, filename: str) -> int:

        """

        Dosya adına göre tüm chunk'ları sil

        Args:

            filename: Doküman dosya adı

        Returns:

            Silinen chunk sayısı

        Raises:

            RepositoryError: Silme başarısız olursa

        """

        pass
 
    @abstractmethod

    async def count(self) -> int:

        """

        Toplam chunk sayısını döndür

        Returns:

            Chunk sayısı

        """

        pass
 
