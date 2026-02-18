"""
FastAPI Main Application
Clean Architecture with Environment-Based Configuration
Production-Grade Banking System Implementation
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_refactored.di import DIContainer
from app_refactored.web_api import router, set_di_container
from app_refactored.infra.redis_client import redis_client

# Load environment variables from environment.env file


env_path = Path(__file__).parent / "environment.env"

load_dotenv(dotenv_path=str(env_path))
 
logger = logging.getLogger(__name__)

logger.info(f"Environment file loaded from: {env_path}")

# Global DIContainer
_container: Optional[DIContainer] = None


# ============================================================================
# Environment Configuration Functions
# ============================================================================

def get_env_int(key: str, default: int) -> int:
    """Safely convert environment variable to integer"""
    try:
        value = os.getenv(key, str(default))
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value for {key}, using default: {default}")
        return default


def get_env_float(key: str, default: float) -> float:
    """Safely convert environment variable to float"""
    try:
        value = os.getenv(key, str(default))
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float value for {key}, using default: {default}")
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Safely convert environment variable to boolean"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def load_configuration() -> dict:
    """Load and validate all configuration from environment variables"""
    config = {
        # API Configuration
        "api_host": os.getenv("RAG_API_HOST", "0.0.0.0"),
        "api_port": get_env_int("RAG_API_PORT", 8005),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "log_level": os.getenv("LOG_LEVEL", "info"),
        
        # PostgreSQL Configuration
        "postgres_host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "postgres_port": get_env_int("POSTGRES_PORT", 35432),
        "postgres_user": os.getenv("POSTGRES_USER", "testusr1"),
        "postgres_password": os.getenv("POSTGRES_PASSWORD", ""),
        "postgres_db": os.getenv("POSTGRES_DB", "testdb1"),
        "postgres_pool_size": get_env_int("POSTGRES_POOL_SIZE", 20),
        "postgres_max_overflow": get_env_int("POSTGRES_MAX_OVERFLOW", 10),
        
        # Jina Configuration
        "jina_host": os.getenv("JINA_HOST", "http://10.144.100.204"),
        "jina_port": get_env_int("JINA_PORT", 38001),
        "jina_model": os.getenv("JINA_MODEL", "jinaai/jina-embeddings-v3"),
        "jina_timeout": get_env_int("JINA_TIMEOUT", 600),
        
        # vLLM Configuration
        "vllm_host": os.getenv("VLLM_HOST", "http://10.144.100.204"),
        "vllm_port": get_env_int("VLLM_PORT", 8804),
        "vllm_model": os.getenv("VLLM_MODEL", "openai/gpt-oss-120b"),
        "vllm_timeout": get_env_int("VLLM_TIMEOUT", 300),
        
        # RAG Configuration
        "chunk_size": get_env_int("RAG_CHUNK_SIZE", 1000),
        "chunk_overlap": get_env_int("RAG_CHUNK_OVERLAP", 200),
        
        # Redis Configuration
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": get_env_int("REDIS_PORT", 6379),
        "redis_db": get_env_int("REDIS_DB", 0),
        "redis_password": os.getenv("REDIS_PASSWORD", ""),
        
        # Rate Limiting
        "rate_limit_enabled": get_env_bool("RATE_LIMIT_ENABLED", True),
        "rate_limit_requests": get_env_int("RATE_LIMIT_REQUESTS", 100),
        "rate_limit_window": get_env_int("RATE_LIMIT_WINDOW_SECONDS", 3600),
        
        # CORS
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        
        # Security Features
        "enable_audit_logging": get_env_bool("ENABLE_AUDIT_LOGGING", True),
        "enable_pii_detection": get_env_bool("ENABLE_PII_DETECTION", True),
        "data_classification_enabled": get_env_bool("DATA_CLASSIFICATION_ENABLED", True),
        "soft_delete_enabled": get_env_bool("SOFT_DELETE_ENABLED", True),
    }
    
    logger.info("✅ Configuration loaded successfully")
    return config


# Load configuration globally
CONFIG = load_configuration()


# ============================================================================
# DI Container Initialization & Shutdown
# ============================================================================

async def init_di_container():
    """Initialize DIContainer with environment variables"""
    global _container
    
    logger.info("🔧 Initializing DI Container with environment configuration...")
    
    # Construct PostgreSQL URL from environment variables
    postgres_url = (
        f"postgresql+asyncpg://"
        f"{CONFIG['postgres_user']}:{CONFIG['postgres_password']}"
        f"@{CONFIG['postgres_host']}:{CONFIG['postgres_port']}"
        f"/{CONFIG['postgres_db']}"
    )
    
    try:
        _container = DIContainer(
            # Jina Configuration
            jina_host=CONFIG['jina_host'],
            jina_port=CONFIG['jina_port'],
            jina_model=CONFIG['jina_model'],
            jina_timeout=CONFIG['jina_timeout'],
            
            # PostgreSQL Configuration
            postgres_url=postgres_url,
            postgres_pool_size=CONFIG['postgres_pool_size'],
            postgres_max_overflow=CONFIG['postgres_max_overflow'],
            
            # vLLM Configuration
            vllm_host=CONFIG['vllm_host'],
            vllm_port=CONFIG['vllm_port'],
            vllm_model=CONFIG['vllm_model'],
            vllm_timeout=CONFIG['vllm_timeout'],
            
            # RAG Configuration
            chunk_size=CONFIG['chunk_size'],
            chunk_overlap=CONFIG['chunk_overlap']
        )
        
        # Set DI Container in routes
        set_di_container(_container)
        
        logger.info("✅ DI Container initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize DI Container: {str(e)}")
        raise


async def shutdown_di_container():
    """Shutdown DIContainer and clean up resources"""
    global _container
    
    if _container:
        try:
            logger.info("🔌 Shutting down DI Container...")
            await _container.close_all()
            logger.info("✅ DI Container shutdown complete")
        except Exception as e:
            logger.error(f"❌ Error during DI Container shutdown: {str(e)}")


async def init_redis():

    """Initialize Redis connection"""

    try:

        logger.info("🔌 Connecting to Redis...")

        await redis_client.connect()

        logger.info(f"✅ Redis connected")

    except Exception as e:

        logger.error(f"❌ Failed to connect to Redis: {str(e)}")

        raise
 
async def shutdown_redis():
    """Shutdown Redis connection"""
    try:
        logger.info("🔌 Disconnecting Redis...")
        await redis_client.disconnect()
        logger.info("✅ Redis disconnected")
    except Exception as e:
        logger.error(f"❌ Error during Redis shutdown: {str(e)}")


# ============================================================================
# FastAPI Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Manager
    Handles startup and shutdown events with proper resource cleanup
    """
    # STARTUP
    logger.info("=" * 70)
    logger.info("🚀 RAG Application Starting Up...")
    logger.info("=" * 70)
    
    try:
        # Initialize DI Container
        await init_di_container()

        #Veritabanı tablolarını otomatik oluştur/kontrol et
        global _container
        if _container:
            repository = _container.get_document_repository()
            if hasattr(repository, 'create_tables'):
                logger.info("📊 Veritabanı tabloları kontrol ediliyor...")
                await repository.create_tables()
        
        # Initialize Redis
        #if CONFIG['rate_limit_enabled']:
        #    await init_redis()
        
        logger.info("✅ Application startup complete")
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {str(e)}")
        raise
    
    yield
    
    # SHUTDOWN
    logger.info("=" * 70)
    logger.info("🛑 RAG Application Shutting Down...")
    logger.info("=" * 70)
    
    try:
        # Shutdown Redis
        #if CONFIG['rate_limit_enabled']:
        #    await shutdown_redis()
        
        # Shutdown DI Container
        await shutdown_di_container()
        
        logger.info("✅ Application shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Application shutdown error: {str(e)}")


# ============================================================================
# FastAPI Application Factory
# ============================================================================

app = FastAPI(
    title="RAG Application - Clean Architecture v2.0",
    description="Production-grade RAG system with async adapters, DI, and environment-based configuration",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS Middleware
cors_origins = CONFIG['cors_origins']
if cors_origins == ['*']:
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in cors_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS configured for origins: {allow_origins}")

# Include routers
app.include_router(router)


# ============================================================================
# Health & System Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - system health check"""
    return {
        "status": "running",
        "version": "2.0.0",
        "architecture": "Clean Architecture with Dependency Injection",
        "environment": CONFIG['environment'],
        "port": CONFIG['api_port'],
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "services": {
            "api": "✅ running",
            #"redis": "✅ connected" if CONFIG['rate_limit_enabled'] else "⏸️ disabled",
            "database": "✅ configured"
        }
    }


@app.get("/api/v2/config/info")
async def config_info():
    """Configuration info endpoint (non-sensitive data)"""
    return {
        "environment": CONFIG['environment'],
        "api_host": CONFIG['api_host'],
        "api_port": CONFIG['api_port'],
        "redis_enabled": CONFIG['rate_limit_enabled'],
        "audit_logging_enabled": CONFIG['enable_audit_logging'],
        "features": {
            "analytics": CONFIG['environment'] == 'production',
            "compliance_reports": CONFIG['enable_audit_logging'],
            "user_isolation": CONFIG['enable_audit_logging'],
            "metadata_enrichment": CONFIG['data_classification_enabled']
        }
    }


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 70)
    logger.info(f"🚀 Starting RAG API v2.0")
    logger.info(f"📍 Host: {CONFIG['api_host']}")
    logger.info(f"📍 Port: {CONFIG['api_port']}")
    logger.info(f"🌍 Environment: {CONFIG['environment']}")
    logger.info(f"📚 Docs: http://{CONFIG['api_host']}:{CONFIG['api_port']}/api/docs")
    logger.info(f"🔄 ReDoc: http://{CONFIG['api_host']}:{CONFIG['api_port']}/api/redoc")
    logger.info("=" * 70)
    
    uvicorn.run(
        "app_refactored.main:app",
        host=CONFIG['api_host'],
        port=CONFIG['api_port'],
        reload=False,
        log_level=CONFIG['log_level'],
        access_log=True
    )
