"""Redis service for managing stream data across multiple threads/processes."""
import json
from typing import Dict, Any, Optional, List
import redis.asyncio as redis
from loguru import logger

from app.core.config import settings


class RedisService:
    """Redis service for managing stream data."""
    
    def __init__(self):
        self.redis_client = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize Redis connection."""
        if self._initialized:
            return
            
        try:
            # Create Redis connection pool
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection established successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {str(e)}")
            self._initialized = False
            raise
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self._initialized = False
            logger.info("Redis connection closed")
    
    async def _ensure_connection(self):
        """Ensure Redis connection is available."""
        if not self._initialized:
            await self.initialize()
    
    # Stream Management Methods
    async def set_stream_status(self, camera_id: str, status_data: Dict[str, Any], ttl: int = 3600):
        """Set stream status for a camera."""
        try:
            await self._ensure_connection()
            key = f"stream:status:{camera_id}"
            await self.redis_client.setex(key, ttl, json.dumps(status_data))
            logger.debug(f"Set stream status for camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to set stream status for camera {camera_id}: {str(e)}")
    
    async def get_stream_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get stream status for a camera."""
        try:
            await self._ensure_connection()
            key = f"stream:status:{camera_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get stream status for camera {camera_id}: {str(e)}")
            return None
    
    async def set_stream_processor(self, camera_id: str, processor_data: Dict[str, Any], ttl: int = 3600):
        """Set stream processor data for a camera."""
        try:
            await self._ensure_connection()
            key = f"stream:processor:{camera_id}"
            await self.redis_client.setex(key, ttl, json.dumps(processor_data))
            logger.debug(f"Set stream processor for camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to set stream processor for camera {camera_id}: {str(e)}")
    
    async def get_stream_processor(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get stream processor data for a camera."""
        try:
            await self._ensure_connection()
            key = f"stream:processor:{camera_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get stream processor for camera {camera_id}: {str(e)}")
            return None
    
    async def add_active_stream(self, camera_id: str, stream_data: Dict[str, Any], ttl: int = 3600):
        """Add a camera to active streams list."""
        try:
            await self._ensure_connection()
            # Add to active streams set
            await self.redis_client.sadd("active_streams", camera_id)
            # Set stream data
            await self.set_stream_status(camera_id, stream_data, ttl)
            await self.set_stream_processor(camera_id, stream_data, ttl)
            logger.info(f"Added camera {camera_id} to active streams")
        except Exception as e:
            logger.error(f"Failed to add camera {camera_id} to active streams: {str(e)}")
    
    async def remove_active_stream(self, camera_id: str):
        """Remove a camera from active streams list."""
        try:
            await self._ensure_connection()
            # Remove from active streams set
            await self.redis_client.srem("active_streams", camera_id)
            # Remove stream data
            await self.redis_client.delete(f"stream:status:{camera_id}")
            await self.redis_client.delete(f"stream:processor:{camera_id}")
            logger.info(f"Removed camera {camera_id} from active streams")
        except Exception as e:
            logger.error(f"Failed to remove camera {camera_id} from active streams: {str(e)}")
    
    async def get_active_streams(self) -> List[str]:
        """Get list of active stream camera IDs."""
        try:
            await self._ensure_connection()
            active_streams = await self.redis_client.smembers("active_streams")
            return list(active_streams)
        except Exception as e:
            logger.error(f"Failed to get active streams: {str(e)}")
            return []
    
    async def get_all_streams_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active streams."""
        try:
            await self._ensure_connection()
            active_streams = await self.get_active_streams()
            all_statuses = {}
            
            for camera_id in active_streams:
                status = await self.get_stream_status(camera_id)
                if status:
                    all_statuses[camera_id] = status
            
            return all_statuses
        except Exception as e:
            logger.error(f"Failed to get all stream statuses: {str(e)}")
            return {}
    
    async def is_stream_active(self, camera_id: str) -> bool:
        """Check if a stream is active."""
        try:
            await self._ensure_connection()
            return await self.redis_client.sismember("active_streams", camera_id)
        except Exception as e:
            logger.error(f"Failed to check if stream {camera_id} is active: {str(e)}")
            return False
    
    async def update_stream_metrics(self, camera_id: str, metrics: Dict[str, Any], ttl: int = 3600):
        """Update stream metrics for a camera."""
        try:
            await self._ensure_connection()
            key = f"stream:metrics:{camera_id}"
            await self.redis_client.setex(key, ttl, json.dumps(metrics))
            logger.debug(f"Updated metrics for camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to update metrics for camera {camera_id}: {str(e)}")
    
    async def get_stream_metrics(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get stream metrics for a camera."""
        try:
            await self._ensure_connection()
            key = f"stream:metrics:{camera_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get metrics for camera {camera_id}: {str(e)}")
            return None
    
    async def cleanup_expired_streams(self):
        """Clean up expired streams (Redis TTL handles this automatically)."""
        try:
            await self._ensure_connection()
            # Redis TTL automatically removes expired keys
            # Just log the current state
            active_count = await self.redis_client.scard("active_streams")
            logger.debug(f"Active streams in Redis: {active_count}")
        except Exception as e:
            logger.error(f"Failed to cleanup expired streams: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            await self._ensure_connection()
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False


# Global Redis service instance
redis_service = RedisService()
