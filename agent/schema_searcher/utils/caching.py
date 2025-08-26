# agent/schema_searcher/utils/caching.py
"""
Simple caching utilities for schema retrieval results.
"""

import hashlib
import time
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__) # type: ignore


class CacheManager:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def generate_key(self, query: str, config: Any) -> str:
        """Generate cache key from query and config"""
        config_str = str(getattr(config, 'cache_key_suffix', ''))
        combined = f"{query}:{config_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() > entry['expires']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with TTL"""
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl
        }
    
    def clear(self) -> None:
        """Clear all cached values"""
        self.cache.clear()
