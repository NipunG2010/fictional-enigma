"""
Cache Manager for HMM Microservice.

Provides in-memory caching for inference results and model parameters
with TTL and size-based eviction policies.
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np

from .config import ServiceConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value, timestamp, and access tracking."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access tracking."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """
    LRU cache with TTL support and size limits.
    
    Thread-safe implementation using asyncio locks.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
        """
        async with self._lock:
            current_time = time.time()
            
            # Use provided TTL or default
            entry_ttl = ttl if ttl is not None else self.default_ttl
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=entry_ttl
            )
            
            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict if over size limit
            await self._evict_if_needed()
    
    async def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all entries from cache."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0
    
    async def _evict_if_needed(self) -> None:
        """Evict entries if cache is over size limit."""
        while len(self._cache) > self.max_size:
            # Remove least recently used entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                self._expirations += 1
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "expirations": self._expirations,
            "total_requests": total_requests
        }


class CacheManager:
    """
    Cache manager for HMM microservice.
    
    Manages multiple cache instances for different data types with
    appropriate TTL and size configurations.
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize cache manager.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self._initialized = False
        
        # Cache instances for different data types
        self._inference_cache: Optional[LRUCache] = None
        self._model_cache: Optional[LRUCache] = None
        self._weight_cache: Optional[LRUCache] = None
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("Cache Manager initialized")
    
    async def initialize(self):
        """Initialize cache instances and start cleanup task."""
        if self._initialized:
            logger.debug("Cache manager already initialized")
            return
        
        logger.info("Initializing Cache Manager...")
        
        # Initialize cache instances
        self._inference_cache = LRUCache(
            max_size=self.config.cache_size,
            default_ttl=self.config.cache_ttl
        )
        
        # Model parameters cache (longer TTL, smaller size)
        self._model_cache = LRUCache(
            max_size=10,  # Only a few models
            default_ttl=3600  # 1 hour TTL
        )
        
        # Fusion weights cache (medium TTL)
        self._weight_cache = LRUCache(
            max_size=self.config.cache_size // 2,
            default_ttl=self.config.cache_ttl
        )
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        self._initialized = True
        logger.info("Cache Manager initialization complete")
    
    async def cleanup(self):
        """Cleanup cache manager resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._inference_cache:
            await self._inference_cache.clear()
        if self._model_cache:
            await self._model_cache.clear()
        if self._weight_cache:
            await self._weight_cache.clear()
        
        logger.info("Cache Manager cleanup complete")
    
    def _generate_observation_key(self, observations: np.ndarray) -> str:
        """
        Generate cache key for observation vector.
        
        Args:
            observations: Observation vector
            
        Returns:
            Cache key string
        """
        # Round to reduce key space while maintaining precision
        rounded_obs = np.round(observations, decimals=6)
        obs_bytes = rounded_obs.tobytes()
        return hashlib.md5(obs_bytes).hexdigest()
    
    def _generate_state_probs_key(self, state_probs: np.ndarray) -> str:
        """
        Generate cache key for state probabilities.
        
        Args:
            state_probs: State probability vector
            
        Returns:
            Cache key string
        """
        # Round to reduce key space
        rounded_probs = np.round(state_probs, decimals=6)
        probs_bytes = rounded_probs.tobytes()
        return hashlib.md5(probs_bytes).hexdigest()
    
    async def get_inference_result(self, observations: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        """
        Get cached inference result for observation vector.
        
        Args:
            observations: Observation vector
            
        Returns:
            Tuple of (state_probabilities, fusion_weights) or None if not cached
        """
        if not self._initialized or not self._inference_cache:
            return None
        
        key = self._generate_observation_key(observations)
        result = await self._inference_cache.get(key)
        
        if result is not None:
            logger.debug(f"Cache hit for inference result: {key}")
            return result
        
        logger.debug(f"Cache miss for inference result: {key}")
        return None
    
    async def cache_inference_result(
        self,
        observations: np.ndarray,
        state_probs: np.ndarray,
        fusion_weights: Dict[str, float],
        ttl: Optional[float] = None
    ) -> None:
        """
        Cache inference result.
        
        Args:
            observations: Observation vector
            state_probs: State probabilities
            fusion_weights: Fusion weights
            ttl: Optional TTL override
        """
        if not self._initialized or not self._inference_cache:
            return
        
        key = self._generate_observation_key(observations)
        result = (state_probs.copy(), fusion_weights.copy())
        
        await self._inference_cache.set(key, result, ttl)
        logger.debug(f"Cached inference result: {key}")
    
    async def get_fusion_weights(self, state_probs: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Get cached fusion weights for state probabilities.
        
        Args:
            state_probs: State probability vector
            
        Returns:
            Fusion weights or None if not cached
        """
        if not self._initialized or not self._weight_cache:
            return None
        
        key = self._generate_state_probs_key(state_probs)
        result = await self._weight_cache.get(key)
        
        if result is not None:
            logger.debug(f"Cache hit for fusion weights: {key}")
            return result
        
        logger.debug(f"Cache miss for fusion weights: {key}")
        return None
    
    async def cache_fusion_weights(
        self,
        state_probs: np.ndarray,
        fusion_weights: Dict[str, float],
        ttl: Optional[float] = None
    ) -> None:
        """
        Cache fusion weights for state probabilities.
        
        Args:
            state_probs: State probability vector
            fusion_weights: Fusion weights
            ttl: Optional TTL override
        """
        if not self._initialized or not self._weight_cache:
            return
        
        key = self._generate_state_probs_key(state_probs)
        await self._weight_cache.set(key, fusion_weights.copy(), ttl)
        logger.debug(f"Cached fusion weights: {key}")
    
    async def get_model_parameters(self, model_version: str) -> Optional[Dict[str, Any]]:
        """
        Get cached model parameters.
        
        Args:
            model_version: Model version string
            
        Returns:
            Model parameters or None if not cached
        """
        if not self._initialized or not self._model_cache:
            return None
        
        result = await self._model_cache.get(f"model:{model_version}")
        
        if result is not None:
            logger.debug(f"Cache hit for model parameters: {model_version}")
            return result
        
        logger.debug(f"Cache miss for model parameters: {model_version}")
        return None
    
    async def cache_model_parameters(
        self,
        model_version: str,
        parameters: Dict[str, Any],
        ttl: Optional[float] = None
    ) -> None:
        """
        Cache model parameters.
        
        Args:
            model_version: Model version string
            parameters: Model parameters
            ttl: Optional TTL override
        """
        if not self._initialized or not self._model_cache:
            return
        
        key = f"model:{model_version}"
        await self._model_cache.set(key, parameters, ttl)
        logger.debug(f"Cached model parameters: {model_version}")
    
    async def invalidate_model_cache(self, model_version: Optional[str] = None) -> None:
        """
        Invalidate model-related cache entries.
        
        Args:
            model_version: Specific version to invalidate (None for all)
        """
        if not self._initialized:
            return
        
        if model_version:
            # Invalidate specific model
            if self._model_cache:
                await self._model_cache.delete(f"model:{model_version}")
            logger.info(f"Invalidated cache for model: {model_version}")
        else:
            # Clear all caches when model changes
            if self._inference_cache:
                await self._inference_cache.clear()
            if self._model_cache:
                await self._model_cache.clear()
            if self._weight_cache:
                await self._weight_cache.clear()
            logger.info("Invalidated all caches")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task to remove expired entries."""
        cleanup_interval = 60  # 1 minute
        
        while True:
            try:
                await asyncio.sleep(cleanup_interval)
                
                # Cleanup expired entries
                total_cleaned = 0
                
                if self._inference_cache:
                    cleaned = await self._inference_cache.cleanup_expired()
                    total_cleaned += cleaned
                
                if self._model_cache:
                    cleaned = await self._model_cache.cleanup_expired()
                    total_cleaned += cleaned
                
                if self._weight_cache:
                    cleaned = await self._weight_cache.cleanup_expired()
                    total_cleaned += cleaned
                
                if total_cleaned > 0:
                    logger.debug(f"Cleaned up {total_cleaned} expired cache entries")
                
            except asyncio.CancelledError:
                logger.info("Cache cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "initialized": self._initialized,
            "inference_cache": None,
            "model_cache": None,
            "weight_cache": None
        }
        
        if self._inference_cache:
            stats["inference_cache"] = self._inference_cache.get_stats()
        
        if self._model_cache:
            stats["model_cache"] = self._model_cache.get_stats()
        
        if self._weight_cache:
            stats["weight_cache"] = self._weight_cache.get_stats()
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics in simplified format for health endpoints.
        
        Returns:
            Dictionary with aggregated cache statistics
        """
        cache_stats = self.get_cache_stats()
        
        # Aggregate statistics from all caches
        total_size = 0
        total_hits = 0
        total_misses = 0
        
        for cache_name in ["inference_cache", "model_cache", "weight_cache"]:
            cache_stat = cache_stats.get(cache_name)
            if cache_stat:
                total_size += cache_stat.get("size", 0)
                total_hits += cache_stat.get("hits", 0)
                total_misses += cache_stat.get("misses", 0)
        
        total_requests = total_hits + total_misses
        hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": total_size,
            "hits": total_hits,
            "misses": total_misses,
            "hit_rate": round(hit_rate, 3)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of cache manager.
        
        Returns:
            Dictionary with health information
        """
        stats = self.get_cache_stats()
        
        # Calculate overall health metrics
        total_hit_rate = 0.0
        active_caches = 0
        
        for cache_name in ["inference_cache", "model_cache", "weight_cache"]:
            cache_stats = stats.get(cache_name)
            if cache_stats:
                total_hit_rate += cache_stats.get("hit_rate", 0.0)
                active_caches += 1
        
        avg_hit_rate = total_hit_rate / active_caches if active_caches > 0 else 0.0
        
        return {
            "initialized": self._initialized,
            "active_caches": active_caches,
            "average_hit_rate": avg_hit_rate,
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done(),
            "cache_stats": stats
        }