
"""

ILLMService Interface - Soyut LLM Servisi

Herhangi bir LLM provider'ı implement edebilir (vLLM, OpenAI, Anthropic, vb.)

"""

from abc import ABC, abstractmethod

from typing import AsyncGenerator, Optional
 
 
class ILLMService(ABC):

    """LLM hizmetinin soyut arayüzü"""
 
    @abstractmethod

    async def generate_response(

        self,

        prompt: str,

        system_prompt: str = "",

        temperature: float = 0.7,

        max_tokens: int = 2000,

        top_p: float = 0.9

    ) -> str:

        """

        LLM'ye prompt göndererek yanıt al (sync)

        Args:

            prompt: LLM'ye gönderilecek prompt

            temperature: Yaratıcılık seviyesi (0.0 - 1.0)

            max_tokens: Maksimum token sayısı

            top_p: Nucleus sampling parametresi

        Returns:

            LLM'nin yanıtı (string)

        Raises:

            LLMServiceError: Yanıt oluşturulamadığında

        """

        pass
 
    @abstractmethod

    async def stream_response(

        self,

        prompt: str,

        system_prompt: str = "",

        temperature: float = 0.7,

        max_tokens: int = 2000,

        top_p: float = 0.9

    ) -> AsyncGenerator[str, None]:

        """

        LLM'ye prompt göndererek yanıtı stream et (async generator)

        Real-time cevap almak için kullan

        Args:

            prompt: LLM'ye gönderilecek prompt

            temperature: Yaratıcılık seviyesi (0.0 - 1.0)

            max_tokens: Maksimum token sayısı

            top_p: Nucleus sampling parametresi

        Yields:

            LLM'nin yanıtının parçaları (token'lar)

        Raises:

            LLMServiceError: Streaming başarısız olursa

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
 
    @abstractmethod

    async def get_model_name(self) -> str:

        """

        Kullanılan model adını döndür

        Returns:

            Model adı (örn: "openai/gpt-oss-120b")

        """

        pass
 
    @abstractmethod

    async def count_tokens(self, text: str) -> int:

        """

        Metindeki token sayısını hesapla

        Args:

            text: Token sayısı hesaplanacak metin

        Returns:

            Token sayısı

        Raises:

            LLMServiceError: Hesaplama başarısız olursa

        """

        pass
 
