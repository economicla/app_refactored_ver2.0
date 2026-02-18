"""

Redis Client - Production-Grade Implementation

Singleton pattern with connection pooling

"""
 
import logging

import os

from typing import Optional, Any

from redis.asyncio import Redis, from_url
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
 
logger = logging.getLogger(__name__)
 
 
class RedisClient:

    """Production-grade Redis async client with singleton pattern"""

    _instance: Optional['RedisClient'] = None

    _redis: Optional[Redis] = None

    def __new__(cls):

        """Singleton pattern"""

        if cls._instance is None:

            cls._instance = super(RedisClient, cls).__new__(cls)

        return cls._instance

    async def connect(self):

        """Initialize Redis connection from environment variables"""

        if self._redis is not None:

            logger.info("Redis already connected")

            return

        try:

            # Read from environment variables

            host = os.getenv("REDIS_HOST", "localhost")

            port = int(os.getenv("REDIS_PORT", 6379))

            db = int(os.getenv("REDIS_DB", 0))

            password = os.getenv("REDIS_PASSWORD", None)

            # Build URL

            if password:

                redis_url = f"redis://:{password}@{host}:{port}/{db}"

            else:

                redis_url = f"redis://{host}:{port}/{db}"

            # Connect

            self._redis = await from_url(redis_url, decode_responses=True)

            # Test connection

            await self._redis.ping()

            logger.info(f"✅ Redis connected: {host}:{port}/{db}")

        except RedisConnectionError as e:

            logger.error(f"❌ Redis connection failed: {str(e)}")

            raise

        except Exception as e:

            logger.error(f"❌ Redis error: {str(e)}")

            raise

    async def disconnect(self):

        """Close Redis connection"""

        if self._redis:

            await self._redis.close()

            self._redis = None

            logger.info("✅ Redis disconnected")

    async def check_rate_limit(

        self,

        identifier: str,

        limit: int = 100,

        window_seconds: int = 3600

    ) -> bool:

        """Check and increment rate limit atomically"""

        if not self._redis:

            logger.warning("Redis not connected, rate limiting disabled")

            return True

        try:

            key = f"rate_limit:{identifier}"

            current = await self._redis.incr(key)

            if current == 1:

                await self._redis.expire(key, window_seconds)

            return current <= limit

        except Exception as e:

            logger.error(f"Rate limit check failed: {str(e)}")

            return True  # Allow on error

    async def get_rate_limit_count(self, identifier: str) -> int:

        """Get current rate limit count"""

        if not self._redis:

            return 0

        try:

            key = f"rate_limit:{identifier}"

            count = await self._redis.get(key)

            return int(count) if count else 0

        except Exception as e:

            logger.error(f"Get rate limit count failed: {str(e)}")

            return 0

    async def cache_set(

        self,

        key: str,

        value: str,

        ttl: int = 3600

    ) -> bool:

        """Set cache value"""

        if not self._redis:

            return False

        try:

            await self._redis.setex(key, ttl, value)

            return True

        except Exception as e:

            logger.error(f"Cache set failed: {str(e)}")

            return False

    async def cache_get(self, key: str) -> Optional[str]:

        """Get cache value"""

        if not self._redis:

            return None

        try:

            return await self._redis.get(key)

        except Exception as e:

            logger.error(f"Cache get failed: {str(e)}")

            return None
 
 
# Singleton instance

redis_client = RedisClient()
 
