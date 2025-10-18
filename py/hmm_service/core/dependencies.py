"""
Dependency injection for HMM Microservice.

Provides singleton instances of core services and manages their lifecycle.
"""

from functools import lru_cache
from typing import Optional

from .config import get_settings


# Global service instances (will be implemented in later tasks)
_inference_engine: Optional["HMMInferenceEngine"] = None
_cache_manager: Optional["CacheManager"] = None  
_model_loader: Optional["ModelLoader"] = None
_metrics_collector: Optional["MetricsCollector"] = None
_performance_manager: Optional["PerformanceManager"] = None


class ServiceDependencies:
    """Container for service dependencies."""
    
    def __init__(self):
        self.settings = get_settings()
        self._inference_engine = None
        self._cache_manager = None
        self._model_loader = None
        self._metrics_collector = None
        self._performance_manager = None
    
    @property
    def inference_engine(self):
        """Get inference engine instance."""
        if self._inference_engine is None:
            from core.inference_engine import HMMInferenceEngine
            self._inference_engine = HMMInferenceEngine(self.settings)
        return self._inference_engine
    
    @property
    def cache_manager(self):
        """Get cache manager instance."""
        if self._cache_manager is None:
            from core.cache import CacheManager
            self._cache_manager = CacheManager(self.settings)
        return self._cache_manager
    
    @property
    def model_loader(self):
        """Get model loader instance."""
        if self._model_loader is None:
            from core.model_loader import ModelLoader
            self._model_loader = ModelLoader(self.settings)
        return self._model_loader
    
    @property
    def metrics_collector(self):
        """Get metrics collector instance."""
        if self._metrics_collector is None:
            # Will be implemented in task 5
            from core.metrics import MetricsCollector
            self._metrics_collector = MetricsCollector(self.settings)
        return self._metrics_collector
    
    @property
    def performance_manager(self):
        """Get performance manager instance."""
        if self._performance_manager is None:
            from core.connection_pool import PerformanceManager
            self._performance_manager = PerformanceManager(self.settings)
        return self._performance_manager


@lru_cache()
def get_service_dependencies() -> ServiceDependencies:
    """Get cached service dependencies container."""
    return ServiceDependencies()


def get_inference_engine():
    """Dependency injection for inference engine."""
    return get_service_dependencies().inference_engine


def get_cache_manager():
    """Dependency injection for cache manager."""
    return get_service_dependencies().cache_manager


def get_model_loader():
    """Dependency injection for model loader."""
    return get_service_dependencies().model_loader


def get_metrics_collector():
    """Dependency injection for metrics collector."""
    return get_service_dependencies().metrics_collector


def get_performance_manager():
    """Dependency injection for performance manager."""
    return get_service_dependencies().performance_manager


# FastAPI dependency functions
async def get_inference_engine_dep():
    """FastAPI dependency for inference engine."""
    return get_inference_engine()


async def get_cache_manager_dep():
    """FastAPI dependency for cache manager."""
    return get_cache_manager()


async def get_model_loader_dep():
    """FastAPI dependency for model loader."""
    return get_model_loader()


async def get_metrics_collector_dep():
    """FastAPI dependency for metrics collector."""
    return get_metrics_collector()


async def get_performance_manager_dep():
    """FastAPI dependency for performance manager."""
    return get_performance_manager()