"""
Circuit breaker implementation for HMM Microservice.

Provides circuit breaker patterns for MinIO operations, service overload protection,
and graceful degradation mechanisms.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from enum import Enum
from collections import deque
import threading

from .logging_config import get_logger
from .error_handling import NetworkError, SystemError, TimeoutError, RateLimitError


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    # Failure thresholds
    failure_threshold: int = 5  # Number of failures to open circuit
    success_threshold: int = 3  # Number of successes to close circuit from half-open
    
    # Time windows
    timeout_seconds: float = 60.0  # How long to keep circuit open
    rolling_window_seconds: float = 300.0  # Rolling window for failure counting
    
    # Rate limiting
    max_requests_per_second: float = 10.0
    max_concurrent_requests: int = 50
    
    # Recovery settings
    recovery_timeout_seconds: float = 30.0
    exponential_backoff: bool = True
    max_backoff_seconds: float = 300.0


class CircuitBreakerStats:
    """Statistics tracking for circuit breaker."""
    
    def __init__(self, window_seconds: float = 300.0):
        self.window_seconds = window_seconds
        self.requests = deque()
        self.failures = deque()
        self.successes = deque()
        self.lock = threading.Lock()
        
        # Counters
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        
        # State tracking
        self.state_changes = []
        self.last_failure_time = None
        self.last_success_time = None
    
    def record_request(self):
        """Record a request attempt."""
        now = time.time()
        with self.lock:
            self.requests.append(now)
            self.total_requests += 1
            self._cleanup_old_entries(now)
    
    def record_success(self):
        """Record a successful request."""
        now = time.time()
        with self.lock:
            self.successes.append(now)
            self.total_successes += 1
            self.last_success_time = now
            self._cleanup_old_entries(now)
    
    def record_failure(self):
        """Record a failed request."""
        now = time.time()
        with self.lock:
            self.failures.append(now)
            self.total_failures += 1
            self.last_failure_time = now
            self._cleanup_old_entries(now)
    
    def record_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Record a state change."""
        with self.lock:
            self.state_changes.append({
                "timestamp": time.time(),
                "from_state": old_state.value,
                "to_state": new_state.value
            })
            
            # Keep only recent state changes
            cutoff = time.time() - self.window_seconds
            self.state_changes = [
                change for change in self.state_changes
                if change["timestamp"] > cutoff
            ]
    
    def _cleanup_old_entries(self, now: float):
        """Remove entries outside the rolling window."""
        cutoff = now - self.window_seconds
        
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        
        while self.failures and self.failures[0] < cutoff:
            self.failures.popleft()
        
        while self.successes and self.successes[0] < cutoff:
            self.successes.popleft()
    
    def get_failure_rate(self) -> float:
        """Get current failure rate in the rolling window."""
        with self.lock:
            total_recent = len(self.requests)
            if total_recent == 0:
                return 0.0
            return len(self.failures) / total_recent
    
    def get_recent_failure_count(self) -> int:
        """Get number of failures in the rolling window."""
        with self.lock:
            return len(self.failures)
    
    def get_recent_success_count(self) -> int:
        """Get number of successes in the rolling window."""
        with self.lock:
            return len(self.successes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self.lock:
            now = time.time()
            self._cleanup_old_entries(now)
            
            return {
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "recent_requests": len(self.requests),
                "recent_failures": len(self.failures),
                "recent_successes": len(self.successes),
                "failure_rate": self.get_failure_rate(),
                "last_failure_time": self.last_failure_time,
                "last_success_time": self.last_success_time,
                "state_changes": list(self.state_changes)
            }


class CircuitBreaker:
    """Circuit breaker implementation with comprehensive failure handling."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"hmm_service.circuit_breaker.{name}")
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.next_attempt_time = None
        
        # Statistics
        self.stats = CircuitBreakerStats(config.rolling_window_seconds)
        
        # Rate limiting
        self.request_times = deque()
        self.concurrent_requests = 0
        self.lock = asyncio.Lock()
        
        # Backoff calculation
        self.backoff_multiplier = 1.0
        
        self.logger.info(f"Circuit breaker '{name}' initialized", extra={
            "config": {
                "failure_threshold": config.failure_threshold,
                "timeout_seconds": config.timeout_seconds,
                "max_requests_per_second": config.max_requests_per_second,
                "max_concurrent_requests": config.max_concurrent_requests
            }
        })
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            NetworkError: If circuit is open or rate limited
            SystemError: If function execution fails
        """
        async with self.lock:
            # Check if we can make the request
            await self._check_request_allowed()
            
            # Record request attempt
            self.stats.record_request()
            self.concurrent_requests += 1
        
        start_time = time.time()
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Circuit breaker '{self.name}' call succeeded", extra={
                "execution_time_ms": round(execution_time, 2),
                "state": self.state.value
            })
            
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure(e)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.warning(f"Circuit breaker '{self.name}' call failed", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": round(execution_time, 2),
                "state": self.state.value
            })
            
            # Re-raise the exception
            raise
            
        finally:
            async with self.lock:
                self.concurrent_requests -= 1
    
    async def _check_request_allowed(self):
        """Check if request is allowed based on circuit state and rate limits."""
        now = time.time()
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self.next_attempt_time and now < self.next_attempt_time:
                time_remaining = self.next_attempt_time - now
                raise NetworkError(
                    f"Circuit breaker '{self.name}' is open",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    service=self.name,
                    retry_after=int(time_remaining) + 1,
                    details={
                        "state": self.state.value,
                        "time_remaining_seconds": round(time_remaining, 2),
                        "consecutive_failures": self.consecutive_failures
                    }
                )
            else:
                # Transition to half-open for testing
                await self._transition_to_half_open()
        
        # Check rate limiting
        await self._check_rate_limits(now)
        
        # Check concurrent request limits
        if self.concurrent_requests >= self.config.max_concurrent_requests:
            raise RateLimitError(
                f"Too many concurrent requests for '{self.name}'",
                retry_after=5,
                limit_type="concurrent_requests",
                details={
                    "current_concurrent": self.concurrent_requests,
                    "max_concurrent": self.config.max_concurrent_requests
                }
            )
    
    async def _check_rate_limits(self, now: float):
        """Check and enforce rate limits."""
        # Clean up old request times
        cutoff = now - 1.0  # 1 second window
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
        
        # Check rate limit
        if len(self.request_times) >= self.config.max_requests_per_second:
            raise RateLimitError(
                f"Rate limit exceeded for '{self.name}'",
                retry_after=1,
                limit_type="requests_per_second",
                details={
                    "current_rate": len(self.request_times),
                    "max_rate": self.config.max_requests_per_second
                }
            )
        
        # Record this request time
        self.request_times.append(now)
    
    async def _record_success(self):
        """Record a successful request and update circuit state."""
        async with self.lock:
            self.stats.record_success()
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            
            # Reset backoff on success
            self.backoff_multiplier = 1.0
            
            # Transition from half-open to closed if enough successes
            if (self.state == CircuitState.HALF_OPEN and 
                self.consecutive_successes >= self.config.success_threshold):
                await self._transition_to_closed()
    
    async def _record_failure(self, error: Exception):
        """Record a failed request and update circuit state."""
        async with self.lock:
            self.stats.record_failure()
            self.last_failure_time = time.time()
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # Check if we should open the circuit
            if (self.state == CircuitState.CLOSED and 
                self.consecutive_failures >= self.config.failure_threshold):
                await self._transition_to_open()
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state goes back to open
                await self._transition_to_open()
    
    async def _transition_to_open(self):
        """Transition circuit to open state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        
        # Calculate next attempt time with exponential backoff
        if self.config.exponential_backoff:
            backoff_time = min(
                self.config.timeout_seconds * self.backoff_multiplier,
                self.config.max_backoff_seconds
            )
            self.backoff_multiplier *= 2
        else:
            backoff_time = self.config.timeout_seconds
        
        self.next_attempt_time = time.time() + backoff_time
        
        self.stats.record_state_change(old_state, self.state)
        
        self.logger.warning(f"Circuit breaker '{self.name}' opened", extra={
            "consecutive_failures": self.consecutive_failures,
            "backoff_time_seconds": round(backoff_time, 2),
            "next_attempt_time": self.next_attempt_time,
            "failure_rate": self.stats.get_failure_rate()
        })
    
    async def _transition_to_half_open(self):
        """Transition circuit to half-open state for testing."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.consecutive_successes = 0
        
        self.stats.record_state_change(old_state, self.state)
        
        self.logger.info(f"Circuit breaker '{self.name}' half-opened for testing", extra={
            "time_since_last_failure": time.time() - (self.last_failure_time or 0)
        })
    
    async def _transition_to_closed(self):
        """Transition circuit to closed state (normal operation)."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.next_attempt_time = None
        self.backoff_multiplier = 1.0
        
        self.stats.record_state_change(old_state, self.state)
        
        self.logger.info(f"Circuit breaker '{self.name}' closed (recovered)", extra={
            "consecutive_successes": self.consecutive_successes,
            "recovery_time_seconds": time.time() - (self.last_failure_time or 0)
        })
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "concurrent_requests": self.concurrent_requests,
            "next_attempt_time": self.next_attempt_time,
            "backoff_multiplier": self.backoff_multiplier,
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "max_requests_per_second": self.config.max_requests_per_second,
                "max_concurrent_requests": self.config.max_concurrent_requests
            },
            "statistics": self.stats.get_stats()
        }
    
    async def reset(self):
        """Reset circuit breaker to closed state (for testing/admin purposes)."""
        async with self.lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.next_attempt_time = None
            self.backoff_multiplier = 1.0
            
            if old_state != CircuitState.CLOSED:
                self.stats.record_state_change(old_state, self.state)
                
                self.logger.info(f"Circuit breaker '{self.name}' manually reset", extra={
                    "previous_state": old_state.value
                })


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = get_logger("hmm_service.circuit_breaker_manager")
    
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create or get existing circuit breaker."""
        if name in self.circuit_breakers:
            return self.circuit_breakers[name]
        
        config = config or CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        
        self.logger.info(f"Created circuit breaker: {name}")
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker."""
        return self.circuit_breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers."""
        return {
            name: cb.get_stats()
            for name, cb in self.circuit_breakers.items()
        }
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        for name, cb in self.circuit_breakers.items():
            await cb.reset()
            self.logger.info(f"Reset circuit breaker: {name}")


# Global circuit breaker manager
_circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager."""
    return _circuit_breaker_manager


def create_minio_circuit_breaker() -> CircuitBreaker:
    """Create circuit breaker specifically configured for MinIO operations."""
    config = CircuitBreakerConfig(
        failure_threshold=3,  # Open after 3 failures
        success_threshold=2,  # Close after 2 successes
        timeout_seconds=30.0,  # 30 second timeout
        rolling_window_seconds=300.0,  # 5 minute window
        max_requests_per_second=20.0,  # Allow higher rate for storage
        max_concurrent_requests=10,  # Limit concurrent MinIO operations
        recovery_timeout_seconds=60.0,
        exponential_backoff=True,
        max_backoff_seconds=300.0  # Max 5 minute backoff
    )
    
    return _circuit_breaker_manager.create_circuit_breaker("minio", config)


def create_inference_circuit_breaker() -> CircuitBreaker:
    """Create circuit breaker for inference operations."""
    config = CircuitBreakerConfig(
        failure_threshold=5,  # More tolerant for inference
        success_threshold=3,
        timeout_seconds=10.0,  # Shorter timeout for inference
        rolling_window_seconds=60.0,  # Shorter window
        max_requests_per_second=50.0,  # Higher rate for inference
        max_concurrent_requests=20,  # Allow more concurrent inference
        recovery_timeout_seconds=30.0,
        exponential_backoff=False,  # Fixed timeout for inference
        max_backoff_seconds=60.0
    )
    
    return _circuit_breaker_manager.create_circuit_breaker("inference", config)