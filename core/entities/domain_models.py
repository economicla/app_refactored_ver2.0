
"""

Pure Domain Models - Framework bağımsız saf Python sınıfları

Pydantic, SQLAlchemy vb. kütüphanelere bağımlı değildir

"""

from typing import List, Dict, Optional

from dataclasses import dataclass, field

from datetime import datetime
 
 
@dataclass

class DocumentChunk:

    """Bir dokümanın parçası (chunk)"""

    id: Optional[int] = None

    filename: str = ""

    chunk_index: int = 0

    content: str = ""

    embedding: Optional[List[float]] = None

    similarity_score: float = 0.0

    metadata: Dict[str, any] = field(default_factory=dict)

    created_at: Optional[datetime] = None
 
    def __post_init__(self):

        if self.metadata is None:

            self.metadata = {}
 
 
@dataclass

class RAGQuery:

    """RAG sorgusu"""

    query: str

    top_k: int = 5

    temperature: float = 0.3

    user_id: str = "anonymous"
 
 
@dataclass

class RAGResponse:

    """RAG yanıtı"""

    question: str

    answer: str

    sources: List[DocumentChunk] = field(default_factory=list)

    model: str = ""

    timestamp: Optional[datetime] = None

    user_id: str = "anonymous"
 
 
@dataclass

class DocumentIngestionResult:

    """Doküman yükleme sonucu"""

    status: str = "success"

    filename: str = ""

    chunks_ingested: int = 0

    total_tokens: int = 0

    timestamp: Optional[datetime] = None

    error: Optional[str] = None
 
 
@dataclass

class EmbeddingResult:

    """Embedding üretim sonucu"""

    text: str

    embedding: List[float]

    model: str = ""

    dimension: int = 0
 
 
@dataclass

class SearchResult:

    """Arama sonucu"""

    documents: List[DocumentChunk] = field(default_factory=list)

    query: str = ""

    search_time_ms: float = 0.0
 
