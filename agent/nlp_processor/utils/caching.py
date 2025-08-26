"""
Caching utilities for NLP Processor
Provides caching functionality for query results and component outputs
"""

import logging
import asyncio
import json
import hashlib
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class SimpleCacheManager:
    """
    Simple in-memory cache manager for NLP processing results
    """
    
    def __init__(self):
        """Initialize cache manager"""
        self.cache = {}
        self.cache_timestamps = {}
        self.default_ttl = 3600  # 1 hour
        self.max_cache_size = 1000
        
        logger.info("Simple cache manager initialized")
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            if key not in self.cache:
                return None
            
            # Check if expired
            if key in self.cache_timestamps:
                timestamp, ttl = self.cache_timestamps[key]
                if time.time() - timestamp > ttl:
                    # Remove expired entry
                    del self.cache[key]
                    del self.cache_timestamps[key]
                    return None
            
            return self.cache[key]
            
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            # Check cache size limit
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.cache_timestamps.keys(), 
                               key=lambda k: self.cache_timestamps[k][0])
                del self.cache[oldest_key]
                del self.cache_timestamps[oldest_key]
            
            # Store value and timestamp
            self.cache[key] = value
            self.cache_timestamps[key] = (time.time(), ttl)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
            return True
            
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_cache_size,
            'default_ttl': self.default_ttl
        }


class NullCacheManager:
    """
    Null cache manager that doesn't cache anything
    Useful for testing or when caching is disabled
    """
    
    def __init__(self):
        logger.info("Null cache manager initialized (caching disabled)")
    
    async def get(self, key: str) -> Optional[str]:
        return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        return True
    
    async def delete(self, key: str) -> bool:
        return True
    
    def clear(self):
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {'status': 'disabled'}


# Global cache manager instance
_cache_manager = None


def get_cache_manager():
    """
    Get cache manager instance
    
    Returns:
        Cache manager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        try:
            # Try to create simple cache manager
            _cache_manager = SimpleCacheManager()
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize cache manager: {e}")
            # Fallback to null cache manager
            _cache_manager = NullCacheManager()
    
    return _cache_manager


def reset_cache_manager():
    """Reset cache manager (useful for testing)"""
    global _cache_manager
    _cache_manager = None


# Convenience functions
async def cache_get(key: str) -> Optional[str]:
    """Get value from cache"""
    manager = get_cache_manager()
    return await manager.get(key)


async def cache_set(key: str, value: str, ttl: Optional[int] = None) -> bool:
    """Set value in cache"""
    manager = get_cache_manager()
    return await manager.set(key, value, ttl)


async def cache_delete(key: str) -> bool:
    """Delete value from cache"""
    manager = get_cache_manager()
    return await manager.delete(key)


def cache_clear():
    """Clear all cache entries"""
    manager = get_cache_manager()
    manager.clear()


def cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    manager = get_cache_manager()
    return manager.get_stats()
