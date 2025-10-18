"""
Connection pooling and request handling for HMM Microservice.

Provides connection pooling for MinIO client, request queuing with timeout handling,
and concurrent request limiting for performance optimization.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import weakref
from concurrent.futures import ThreadPoolExecutor

from .config import ServiceConfig
from .circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker_manager,
    create_minio_circuit_breaker, create_inference_circuit_breaker
)
from .error_handling import NetworkError, SystemError, TimeoutError

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for connection pool."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    created_connections: int = 0
    closed_connections: int = 0
    connection_errors: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "created_connections": self.created_connections,
            "closed_connections": self.closed_connections,
            "connection_errors": self.connection_errors,
            "pool_hits": self.pool_hits,
            "pool_misses": self.pool_misses,
            "hit_rate": self.pool_hits / (self.pool_hits + self.pool_misses) if (self.pool_hits + self.pool_misses) > 0 else 0.0
        }


@dataclass
class PooledConnection:
    """Wrapper for pooled connection with metadata."""
    connection: Any
    created_at: float
    last_used: float
    use_count: int = 0
    is_active: bool = False
    
    def touch(self):
        """Update last used timestamp and increment use count."""
        self.last_used = time.time()
        self.use_count += 1
    
    def is_expired(self, max_age: float) -> bool:
        """Check if connection is expired based on max age."""
        return time.time() - self.created_at > max_age
    
    def is_idle_too_long(self, max_idle: float) -> bool:
        """Check if connection has been idle too long."""
        return time.time() - self.last_used > max_idle


class ConnectionPool:
    """
    Generic connection pool with lifecycle management.
    
    Provides connection pooling with configurable size limits, connection lifecycle
    management, and automatic cleanup of expired connections.
    """
    
    def __init__(
        self,
        connection_factory: Callable[[], Awaitable[Any]],
        connection_closer: Optional[Callable[[Any], Awaitable[None]]] = None,
        max_size: int = 10,
        min_size: int = 2,
        max_age: float = 3600.0,  # 1 hour
        max_idle: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0  # 1 minute
    ):
        """
        Initialize connection pool.
        
        Args:
            connection_factory: Async function to create new connections
            connection_closer: Optional async function to close connections
            max_size: Maximum number of connections in pool
            min_size: Minimum number of connections to maintain
            max_age: Maximum age of connections in seconds
            max_idle: Maximum idle time before connection is closed
            cleanup_interval: Interval for cleanup task in seconds
        """
        self.connection_factory = connection_factory
        self.connection_closer = connection_closer
        self.max_size = max_size
        self.min_size = min_size
        self.max_age = max_age
        self.max_idle = max_idle
        self.cleanup_interval = cleanup_interval
        
        # Pool state
        self._pool: List[PooledConnection] = []
        self._lock = asyncio.Lock()
        self._initialized = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = ConnectionStats()
        
        logger.info(f"Connection pool initialized: max_size={max_size}, min_size={min_size}")
    
    async def initialize(self):
        """Initialize connection pool and start cleanup task."""
        if self._initialized:
            return
        
        logger.info("Initializing connection pool...")
        
        # Create minimum number of connections
        for _ in range(self.min_size):
            try:
                await self._create_connection()
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        self._initialized = True
        logger.info(f"Connection pool initialized with {len(self._pool)} connections")
    
    async def cleanup(self):
        """Cleanup connection pool resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._lock:
            for pooled_conn in self._pool:
                await self._close_connection(pooled_conn)
            self._pool.clear()
        
        logger.info("Connection pool cleanup complete")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get connection from pool as async context manager.
        
        Yields:
            Connection object from pool
        """
        connection = None
        pooled_conn = None
        
        try:
            # Get connection from pool
            pooled_conn = await self._acquire_connection()
            connection = pooled_conn.connection
            
            # Mark as active
            pooled_conn.is_active = True
            pooled_conn.touch()
            
            yield connection
            
        finally:
            # Return connection to pool
            if pooled_conn:
                pooled_conn.is_active = False
                await self._release_connection(pooled_conn)
    
    async def _acquire_connection(self) -> PooledConnection:
        """Acquire connection from pool or create new one."""
        async with self._lock:
            # Try to find idle connection
            for pooled_conn in self._pool:
                if not pooled_conn.is_active and not pooled_conn.is_expired(self.max_age):
                    self._stats.pool_hits += 1
                    return pooled_conn
            
            # No idle connection available, create new one if under limit
            if len(self._pool) < self.max_size:
                self._stats.pool_misses += 1
                return await self._create_connection()
            
            # Pool is full, wait for connection to become available
            self._stats.pool_misses += 1
            
        # Wait for connection to become available (with timeout)
        timeout = 30.0  # 30 second timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
            
            async with self._lock:
                for pooled_conn in self._pool:
                    if not pooled_conn.is_active and not pooled_conn.is_expired(self.max_age):
                        return pooled_conn
        
        raise RuntimeError("Connection pool timeout: no connections available")
    
    async def _release_connection(self, pooled_conn: PooledConnection):
        """Release connection back to pool."""
        # Connection is already marked as inactive by caller
        # Just update statistics
        async with self._lock:
            self._stats.active_connections = sum(1 for conn in self._pool if conn.is_active)
            self._stats.idle_connections = len(self._pool) - self._stats.active_connections
    
    async def _create_connection(self) -> PooledConnection:
        """Create new connection and add to pool."""
        try:
            connection = await self.connection_factory()
            
            pooled_conn = PooledConnection(
                connection=connection,
                created_at=time.time(),
                last_used=time.time()
            )
            
            self._pool.append(pooled_conn)
            
            # Update statistics
            self._stats.created_connections += 1
            self._stats.total_connections = len(self._pool)
            self._stats.idle_connections += 1
            
            logger.debug(f"Created new connection: pool size={len(self._pool)}")
            return pooled_conn
            
        except Exception as e:
            self._stats.connection_errors += 1
            logger.error(f"Failed to create connection: {e}")
            raise
    
    async def _close_connection(self, pooled_conn: PooledConnection):
        """Close connection and remove from pool."""
        try:
            if self.connection_closer:
                await self.connection_closer(pooled_conn.connection)
            
            # Remove from pool
            if pooled_conn in self._pool:
                self._pool.remove(pooled_conn)
            
            # Update statistics
            self._stats.closed_connections += 1
            self._stats.total_connections = len(self._pool)
            
            logger.debug(f"Closed connection: pool size={len(self._pool)}")
            
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task to remove expired connections."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_connections()
                
            except asyncio.CancelledError:
                logger.info("Connection pool cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in connection pool cleanup: {e}")
    
    async def _cleanup_expired_connections(self):
        """Remove expired and idle connections from pool."""
        async with self._lock:
            expired_connections = []
            
            for pooled_conn in self._pool:
                # Skip active connections
                if pooled_conn.is_active:
                    continue
                
                # Check if connection should be removed
                if (pooled_conn.is_expired(self.max_age) or 
                    pooled_conn.is_idle_too_long(self.max_idle)):
                    expired_connections.append(pooled_conn)
            
            # Don't remove connections if we'd go below minimum
            connections_to_remove = expired_connections
            if len(self._pool) - len(expired_connections) < self.min_size:
                # Keep some connections to maintain minimum
                keep_count = self.min_size - (len(self._pool) - len(expired_connections))
                connections_to_remove = expired_connections[keep_count:]
            
            # Remove expired connections
            for pooled_conn in connections_to_remove:
                await self._close_connection(pooled_conn)
            
            if connections_to_remove:
                logger.debug(f"Cleaned up {len(connections_to_remove)} expired connections")
            
            # Update statistics
            self._stats.active_connections = sum(1 for conn in self._pool if conn.is_active)
            self._stats.idle_connections = len(self._pool) - self._stats.active_connections
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return self._stats.get_stats_dict()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of connection pool."""
        return {
            "initialized": self._initialized,
            "pool_size": len(self._pool),
            "max_size": self.max_size,
            "min_size": self.min_size,
            "active_connections": self._stats.active_connections,
            "idle_connections": self._stats.idle_connections,
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done(),
            "statistics": self.get_stats()
        }


@dataclass
class RequestQueueStats:
    """Statistics for request queue."""
    total_requests: int = 0
    queued_requests: int = 0
    processed_requests: int = 0
    timeout_requests: int = 0
    rejected_requests: int = 0
    average_queue_time: float = 0.0
    max_queue_time: float = 0.0
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        return {
            "total_requests": self.total_requests,
            "queued_requests": self.queued_requests,
            "processed_requests": self.processed_requests,
            "timeout_requests": self.timeout_requests,
            "rejected_requests": self.rejected_requests,
            "average_queue_time": self.average_queue_time,
            "max_queue_time": self.max_queue_time
        }


class RequestQueue:
    """
    Request queue with timeout handling and concurrent request limiting.
    
    Manages incoming requests with configurable queue size, timeout handling,
    and concurrent processing limits to prevent service overload.
    """
    
    def __init__(
        self,
        max_concurrent: int = 100,
        max_queue_size: int = 1000,
        default_timeout: float = 30.0
    ):
        """
        Initialize request queue.
        
        Args:
            max_concurrent: Maximum number of concurrent requests
            max_queue_size: Maximum number of queued requests
            default_timeout: Default timeout for requests in seconds
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        
        # Queue state
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._active_requests: int = 0
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = RequestQueueStats()
        self._queue_times: List[float] = []
        
        logger.info(f"Request queue initialized: max_concurrent={max_concurrent}, max_queue_size={max_queue_size}")
    
    async def process_request(
        self,
        request_handler: Callable[[], Awaitable[Any]],
        timeout: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> Any:
        """
        Process request with queue management and timeout handling.
        
        Args:
            request_handler: Async function to handle the request
            timeout: Request timeout in seconds (uses default if None)
            request_id: Optional request identifier for logging
            
        Returns:
            Result from request handler
            
        Raises:
            asyncio.TimeoutError: If request times out
            RuntimeError: If queue is full
        """
        request_timeout = timeout or self.default_timeout
        start_time = time.time()
        
        # Update statistics
        async with self._lock:
            self._stats.total_requests += 1
            
            # Check if queue is full
            if self._queue.qsize() >= self.max_queue_size:
                self._stats.rejected_requests += 1
                raise RuntimeError(f"Request queue is full (size: {self.max_queue_size})")
        
        try:
            # Wait for semaphore with timeout
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=request_timeout
            )
            
            queue_time = time.time() - start_time
            
            # Update queue statistics
            async with self._lock:
                self._active_requests += 1
                self._stats.queued_requests = self._queue.qsize()
                self._queue_times.append(queue_time)
                
                # Keep only recent queue times for average calculation
                if len(self._queue_times) > 1000:
                    self._queue_times = self._queue_times[-500:]
                
                # Update queue time statistics
                self._stats.average_queue_time = sum(self._queue_times) / len(self._queue_times)
                self._stats.max_queue_time = max(self._stats.max_queue_time, queue_time)
            
            try:
                # Process request with remaining timeout
                remaining_timeout = request_timeout - queue_time
                if remaining_timeout <= 0:
                    raise asyncio.TimeoutError("Request timed out in queue")
                
                result = await asyncio.wait_for(
                    request_handler(),
                    timeout=remaining_timeout
                )
                
                # Update success statistics
                async with self._lock:
                    self._stats.processed_requests += 1
                
                return result
                
            finally:
                # Release semaphore and update active count
                self._semaphore.release()
                async with self._lock:
                    self._active_requests -= 1
        
        except asyncio.TimeoutError:
            # Update timeout statistics
            async with self._lock:
                self._stats.timeout_requests += 1
            
            logger.warning(f"Request timeout: {request_id or 'unknown'}")
            raise
        
        except Exception as e:
            logger.error(f"Request processing error: {request_id or 'unknown'}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get request queue statistics."""
        stats = self._stats.get_stats_dict()
        stats.update({
            "active_requests": self._active_requests,
            "queue_size": self._queue.qsize(),
            "available_slots": self.max_concurrent - self._active_requests
        })
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of request queue."""
        queue_utilization = self._active_requests / self.max_concurrent
        queue_full_ratio = self._queue.qsize() / self.max_queue_size
        
        return {
            "active_requests": self._active_requests,
            "max_concurrent": self.max_concurrent,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "queue_utilization": queue_utilization,
            "queue_full_ratio": queue_full_ratio,
            "is_overloaded": queue_utilization > 0.8 or queue_full_ratio > 0.8,
            "statistics": self.get_stats()
        }


class MinIOConnectionPool:
    """
    Specialized connection pool for MinIO clients.
    
    Provides connection pooling specifically for MinIO operations with
    proper connection lifecycle management and error handling.
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize MinIO connection pool.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self._pool: Optional[ConnectionPool] = None
        self._initialized = False
        
        # Circuit breaker for MinIO operations
        self._circuit_breaker = create_minio_circuit_breaker()
        
        logger.info("MinIO connection pool initialized with circuit breaker")
    
    async def initialize(self):
        """Initialize MinIO connection pool."""
        if self._initialized:
            return
        
        logger.info("Initializing MinIO connection pool...")
        
        # Create connection pool with MinIO-specific factory
        self._pool = ConnectionPool(
            connection_factory=self._create_minio_client,
            connection_closer=self._close_minio_client,
            max_size=10,  # Reasonable limit for MinIO connections
            min_size=2,   # Keep minimum connections
            max_age=3600.0,  # 1 hour max age
            max_idle=300.0,  # 5 minutes max idle
            cleanup_interval=60.0  # 1 minute cleanup
        )
        
        await self._pool.initialize()
        self._initialized = True
        
        logger.info("MinIO connection pool initialization complete")
    
    async def cleanup(self):
        """Cleanup MinIO connection pool."""
        if self._pool:
            await self._pool.cleanup()
        logger.info("MinIO connection pool cleanup complete")
    
    async def _create_minio_client(self):
        """Create new MinIO client connection with circuit breaker protection."""
        async def _create_client():
            # Import MinIO client
            from minio import Minio
            
            # Create MinIO client with configuration
            client = Minio(
                endpoint=self.config.minio.endpoint,
                access_key=self.config.minio.access_key,
                secret_key=self.config.minio.secret_key,
                secure=self.config.minio.secure
            )
            
            # Test connection by listing buckets
            list(client.list_buckets())
            
            logger.debug(f"Created MinIO client: {self.config.minio.endpoint}")
            return client
        
        try:
            # Use circuit breaker for connection creation
            return await self._circuit_breaker.call(_create_client)
            
        except Exception as e:
            logger.error(f"Failed to create MinIO client: {e}")
            raise NetworkError(
                f"MinIO connection failed: {str(e)}",
                error_code="MINIO_CONNECTION_FAILED",
                service="minio",
                details={"endpoint": self.config.minio.endpoint}
            )
    
    async def _close_minio_client(self, client):
        """Close MinIO client connection."""
        try:
            # MinIO client doesn't have explicit close method
            # Just clear any internal state if needed
            pass
        except Exception as e:
            logger.error(f"Error closing MinIO client: {e}")
    
    @asynccontextmanager
    async def get_client(self):
        """
        Get MinIO client from pool with circuit breaker protection.
        
        Yields:
            MinIO client from pool
        """
        if not self._initialized or not self._pool:
            raise SystemError(
                "MinIO connection pool not initialized",
                error_code="MINIO_POOL_NOT_INITIALIZED",
                resource="minio_pool"
            )
        
        try:
            async with self._pool.get_connection() as client:
                yield client
        except Exception as e:
            # Let circuit breaker handle the error
            logger.error(f"MinIO client acquisition failed: {e}")
            raise NetworkError(
                f"Failed to get MinIO client: {str(e)}",
                error_code="MINIO_CLIENT_ACQUISITION_FAILED",
                service="minio",
                details={"pool_stats": self.get_stats()}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MinIO connection pool statistics."""
        if not self._pool:
            return {"initialized": False}
        
        return self._pool.get_stats()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of MinIO connection pool including circuit breaker."""
        if not self._pool:
            return {"initialized": False}
        
        health = self._pool.get_health_status()
        health["circuit_breaker"] = self._circuit_breaker.get_stats()
        health["circuit_breaker_healthy"] = self._circuit_breaker.get_state().value != "open"
        
        return health


class PerformanceManager:
    """
    Performance manager that coordinates connection pooling and request handling.
    
    Provides unified interface for managing connection pools, request queues,
    and performance optimization features.
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize performance manager.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self._initialized = False
        
        # Components
        self._minio_pool: Optional[MinIOConnectionPool] = None
        self._request_queue: Optional[RequestQueue] = None
        
        # Circuit breaker for inference operations
        self._inference_circuit_breaker = create_inference_circuit_breaker()
        
        logger.info("Performance manager initialized with circuit breakers")
    
    async def initialize(self):
        """Initialize performance manager components."""
        if self._initialized:
            return
        
        logger.info("Initializing performance manager...")
        
        # Initialize MinIO connection pool
        self._minio_pool = MinIOConnectionPool(self.config)
        await self._minio_pool.initialize()
        
        # Initialize request queue
        self._request_queue = RequestQueue(
            max_concurrent=self.config.max_concurrent_requests,
            max_queue_size=self.config.max_concurrent_requests * 10,  # 10x buffer
            default_timeout=self.config.request_timeout
        )
        
        self._initialized = True
        logger.info("Performance manager initialization complete")
    
    async def cleanup(self):
        """Cleanup performance manager resources."""
        if self._minio_pool:
            await self._minio_pool.cleanup()
        
        logger.info("Performance manager cleanup complete")
    
    @asynccontextmanager
    async def get_minio_client(self):
        """Get MinIO client from connection pool."""
        if not self._minio_pool:
            raise RuntimeError("MinIO connection pool not initialized")
        
        async with self._minio_pool.get_client() as client:
            yield client
    
    async def process_request(
        self,
        request_handler: Callable[[], Awaitable[Any]],
        timeout: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> Any:
        """Process request through request queue with circuit breaker protection."""
        if not self._request_queue:
            raise SystemError(
                "Request queue not initialized",
                error_code="REQUEST_QUEUE_NOT_INITIALIZED",
                resource="request_queue"
            )
        
        # Wrap request handler with circuit breaker
        async def protected_handler():
            return await self._inference_circuit_breaker.call(request_handler)
        
        try:
            return await self._request_queue.process_request(
                request_handler=protected_handler,
                timeout=timeout,
                request_id=request_id
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Request timeout after {timeout or self.config.request_timeout}s",
                operation="request_processing",
                timeout_seconds=timeout or self.config.request_timeout,
                details={"request_id": request_id}
            )
        except RuntimeError as e:
            if "queue is full" in str(e):
                raise SystemError(
                    "Service overloaded - request queue full",
                    error_code="SERVICE_OVERLOADED",
                    resource="request_queue",
                    details={
                        "queue_stats": self._request_queue.get_stats(),
                        "request_id": request_id
                    }
                )
            else:
                raise SystemError(
                    f"Request processing failed: {str(e)}",
                    error_code="REQUEST_PROCESSING_ERROR",
                    resource="request_queue",
                    details={"request_id": request_id}
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics including circuit breakers."""
        stats = {
            "initialized": self._initialized,
            "minio_pool": None,
            "request_queue": None,
            "circuit_breakers": {
                "inference": self._inference_circuit_breaker.get_stats()
            }
        }
        
        if self._minio_pool:
            stats["minio_pool"] = self._minio_pool.get_stats()
        
        if self._request_queue:
            stats["request_queue"] = self._request_queue.get_stats()
        
        # Add circuit breaker manager stats
        circuit_manager = get_circuit_breaker_manager()
        stats["all_circuit_breakers"] = circuit_manager.get_all_stats()
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of performance manager including circuit breakers."""
        health = {
            "initialized": self._initialized,
            "minio_pool_healthy": False,
            "request_queue_healthy": False,
            "circuit_breakers_healthy": False,
            "overall_healthy": False
        }
        
        if self._minio_pool:
            minio_health = self._minio_pool.get_health_status()
            health["minio_pool_healthy"] = (
                minio_health.get("initialized", False) and
                minio_health.get("circuit_breaker_healthy", False)
            )
            health["minio_pool"] = minio_health
        
        if self._request_queue:
            queue_health = self._request_queue.get_health_status()
            health["request_queue_healthy"] = not queue_health.get("is_overloaded", True)
            health["request_queue"] = queue_health
        
        # Check circuit breaker health
        inference_cb_healthy = self._inference_circuit_breaker.get_state().value != "open"
        health["circuit_breakers_healthy"] = inference_cb_healthy
        health["circuit_breakers"] = {
            "inference": self._inference_circuit_breaker.get_stats(),
            "inference_healthy": inference_cb_healthy
        }
        
        # Overall health check
        health["overall_healthy"] = (
            health["minio_pool_healthy"] and 
            health["request_queue_healthy"] and
            health["circuit_breakers_healthy"]
        )
        
        return health