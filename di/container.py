
"""

Dependency Injection Container - Tüm dependencies'leri yönet

"""

import logging

from app_refactored.adapters import (

    JinaEmbeddingAdapter,

    PostgresDocumentAdapter,

    VLLMAdapter

)

from app_refactored.use_cases import (

    RAGQueryUseCase,

    DocumentIngestionUseCase

)
 
logger = logging.getLogger(__name__)
 
 
class DIContainer:

    """Dependency Injection Container"""
 
    def __init__(

        self,

        # Required parameters (no defaults)

        jina_host: str,

        jina_port: int,

        jina_model: str,

        postgres_url: str,

        vllm_host: str,

        vllm_port: int,

        vllm_model: str,

        # Optional parameters (with defaults)

        jina_timeout: int = 600,

        postgres_pool_size: int = 20,

        postgres_max_overflow: int = 10,

        vllm_timeout: int = 300,

        chunk_size: int = 1000,

        chunk_overlap: int = 200

    ):

        """Initialize DI Container with all configuration"""

        self.config = {

            'jina': {

                'host': jina_host,

                'port': jina_port,

                'model': jina_model,

                'timeout': jina_timeout

            },

            'postgres': {

                'url': postgres_url,

                'pool_size': postgres_pool_size,

                'max_overflow': postgres_max_overflow

            },

            'vllm': {

                'host': vllm_host,

                'port': vllm_port,

                'model': vllm_model,

                'timeout': vllm_timeout

            },

            'rag': {

                'chunk_size': chunk_size,

                'chunk_overlap': chunk_overlap

            }

        }

        # Lazy initialization

        self._embedding_service = None

        self._document_repository = None

        self._llm_service = None

        self._rag_use_case = None

        self._ingestion_use_case = None
 
    def get_embedding_service(self):

        """Get IEmbeddingService implementation"""

        if self._embedding_service is None:

            logger.info("Initializing JinaEmbeddingAdapter...")

            self._embedding_service = JinaEmbeddingAdapter(

                host=self.config['jina']['host'],

                port=self.config['jina']['port'],

                model=self.config['jina']['model'],

                timeout=self.config['jina']['timeout']

            )

        return self._embedding_service
 
    def get_document_repository(self):

        """Get IDocumentRepository implementation"""

        if self._document_repository is None:

            logger.info("Initializing PostgresDocumentAdapter...")

            self._document_repository = PostgresDocumentAdapter(

                database_url=self.config['postgres']['url'],

                pool_size=self.config['postgres']['pool_size'],

                max_overflow=self.config['postgres']['max_overflow']

            )

        return self._document_repository
 
    def get_llm_service(self):

        """Get ILLMService implementation"""

        if self._llm_service is None:

            logger.info("Initializing VLLMAdapter...")

            self._llm_service = VLLMAdapter(

                host=self.config['vllm']['host'],

                port=self.config['vllm']['port'],

                model=self.config['vllm']['model'],

                timeout=self.config['vllm']['timeout']

            )

        return self._llm_service
 
    def get_rag_query_use_case(self) -> RAGQueryUseCase:

        """Get RAGQueryUseCase with all dependencies injected"""

        if self._rag_use_case is None:

            logger.info("Initializing RAGQueryUseCase...")

            self._rag_use_case = RAGQueryUseCase(

                embedding_service=self.get_embedding_service(),

                document_repository=self.get_document_repository(),

                llm_service=self.get_llm_service()

            )

        return self._rag_use_case
 
    def get_document_ingestion_use_case(self) -> DocumentIngestionUseCase:

        """Get DocumentIngestionUseCase with all dependencies injected"""

        if self._ingestion_use_case is None:

            logger.info("Initializing DocumentIngestionUseCase...")

            self._ingestion_use_case = DocumentIngestionUseCase(

                embedding_service=self.get_embedding_service(),

                document_repository=self.get_document_repository(),

                chunk_size=self.config['rag']['chunk_size'],

                chunk_overlap=self.config['rag']['chunk_overlap']

            )

        return self._ingestion_use_case
 
    async def close_all(self):

        """Tüm services'i kapat"""

        logger.info("Closing all services...")

        if self._embedding_service:

            await self._embedding_service.close()

        if self._document_repository:

            await self._document_repository.close()

        if self._llm_service:

            await self._llm_service.close()

        logger.info("✅ All services closed")
 
    async def __aenter__(self):

        """Async context manager support"""

        return self
 
    async def __aexit__(self, exc_type, exc_val, exc_tb):

        """Async context manager cleanup"""

        await self.close_all()
 
