
"""

VLLMAdapter - vLLM'yi ILLMService interface'ine adapt et

OpenAI-compatible API ile async communication

"""

from typing import AsyncGenerator

import httpx

import logging

from app_refactored.core.interfaces import ILLMService
 
logger = logging.getLogger(__name__)
 
 
class VLLMAdapter(ILLMService):

    """vLLM OpenAI-compatible async adapter with connection pooling"""
 
    def __init__(

        self,

        host: str,

        port: int,

        model: str,

        timeout: int = 300,

        max_connections: int = 20

    ):

        """

        Initialize vLLM adapter

        Args:

            host: vLLM host (örn: "http://10.144.100.204")

            port: vLLM port (örn: 8804)

            model: Model adı (örn: "openai/gpt-oss-120b")

            timeout: Request timeout (saniye)

            max_connections: Connection pool size

        """

        self.host = host

        self.port = port

        self.model = model

        self.timeout = timeout

        self.api_base = f"{host}:{port}/v1"

        self.completions_endpoint = f"{self.api_base}/chat/completions"

        self.models_endpoint = f"{self.api_base}/models"

        # AsyncClient with connection pooling

        self.client = httpx.AsyncClient(

            timeout=httpx.Timeout(timeout),

            limits=httpx.Limits(

                max_connections=max_connections,

                max_keepalive_connections=max_connections

            )

        )
 
    async def generate_response(

        self,

        prompt: str,

        system_prompt: str = "",

        temperature: float = 0,

        max_tokens: int = 2000,

        top_p: float = 0.9

    ) -> str:

        """LLM'ye prompt gönder ve yanıtı al (sync response)"""

        try:

            messages = []

            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {

                "model": self.model,

                "messages": messages,

                "temperature": temperature,

                "max_tokens": max_tokens,

                "top_p": top_p,

                "stream": False

            }

            response = await self.client.post(

                self.completions_endpoint,

                json=payload

            )

            response.raise_for_status()

            result = response.json()

            answer = result['choices'][0]['message']['content']

            logger.info(f"✅ Generated response ({len(answer)} chars)")

            return answer

        except Exception as e:

            logger.error(f"❌ Generate response failed: {str(e)}")

            raise
 
    async def stream_response(

        self,

        prompt: str,

        system_prompt: str = "",

        temperature: float = 0.7,

        max_tokens: int = 2000,

        top_p: float = 0.9

    ) -> AsyncGenerator[str, None]:

        """LLM'ye prompt gönder ve yanıtı stream et (real-time)"""

        try:
            # Messages dizisini oluştur
            messages = []

            #System prompt varsa ekle
            if system_prompt.strip():
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # User prompt ekle
            messages.append({
                "role": "user",
                "content": prompt
            })

            payload = {

                "model": self.model,

                "messages": messages,

                "temperature": temperature,

                "max_tokens": max_tokens,

                "top_p": top_p,

                "stream": True

            }

            logger.info(f"Streaming request: {self.model} (temp={temperature})")
            logger.debug(f"System Prompt: {system_prompt[:100]}..." if system_prompt else "No system prompt")

            async with self.client.stream(

                "POST",

                self.completions_endpoint,

                json=payload

            ) as response:

                response.raise_for_status()

                async for line in response.aiter_lines():

                    if line.startswith("data: "):

                        data_str = line[6:]  # "data: " kısmını çıkar

                        if data_str == "[DONE]":
                            logger.info("✔️ Stream completed successfully")

                            break

                        try:

                            import json

                            data = json.loads(data_str)

                            if 'choices' in data and len(data['choices']) > 0:

                                chunk = data['choices'][0].get('delta', {}).get('content', '')

                                if chunk:

                                    yield chunk

                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ JSON decode error: {str(e)}")

                            continue

            logger.info("✅ Stream completed")

        except Exception as e:

            logger.error(f"❌ Stream response failed: {str(e)}")

            raise
 
    async def is_available(self) -> bool:

        """vLLM servisinin çalışıp çalışmadığını kontrol et"""

        try:

            response = await self.client.get(

                self.models_endpoint,

                timeout=httpx.Timeout(5)

            )

            return response.status_code == 200

        except Exception as e:

            logger.warning(f"vLLM health check failed: {str(e)}")

            return False
 
    async def get_model_name(self) -> str:

        """Model adını döndür"""

        return self.model
 
    async def count_tokens(self, text: str) -> int:

        """

        Metindeki token sayısını tahmin et

        Rough calculation: 1 token ≈ 4 characters

        """

        estimated_tokens = len(text) // 4

        logger.info(f"📊 Estimated tokens: {estimated_tokens}")

        return estimated_tokens
 
    async def close(self):

        """Client bağlantısını kapat"""

        await self.client.aclose()

        logger.info("✅ vLLM client closed")
 
    async def __aenter__(self):

        """Context manager support"""

        return self
 
    async def __aexit__(self, exc_type, exc_val, exc_tb):

        """Context manager cleanup"""

        await self.close()
 
