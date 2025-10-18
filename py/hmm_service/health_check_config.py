"""
Health Check Configuration for Container Orchestration

This module provides configuration and utilities for optimizing health checks
for different container orchestration platforms.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class HealthCheckType(Enum):
    """Types of health checks for different orchestration needs."""
    LIVENESS = "liveness"      # Should the container be restarted?
    READINESS = "readiness"    # Should traffic be routed to this instance?
    STARTUP = "startup"        # Has the container finished initializing?


@dataclass
class HealthCheckConfig:
    """Configuration for health check behavior."""
    
    # Timing configuration
    initial_delay_seconds: int = 30
    period_seconds: int = 10
    timeout_seconds: int = 5
    failure_threshold: int = 3
    success_threshold: int = 1
    
    # Behavior configuration
    check_model_loaded: bool = True
    check_minio_connectivity: bool = True
    check_cache_initialized: bool = True
    check_performance_manager: bool = True
    
    # Response configuration
    include_detailed_info: bool = False
    max_response_time_ms: int = 5000  # Maximum allowed response time
    
    @classmethod
    def for_orchestration_type(cls, orchestration_type: str, check_type: HealthCheckType) -> 'HealthCheckConfig':
        """Create optimized configuration for specific orchestration platforms."""
        
        if orchestration_type.lower() == "kubernetes":
            return cls._kubernetes_config(check_type)
        elif orchestration_type.lower() == "docker-swarm":
            return cls._docker_swarm_config(check_type)
        elif orchestration_type.lower() == "docker-compose":
            return cls._docker_compose_config(check_type)
        else:
            return cls._default_config(check_type)
    
    @classmethod
    def _kubernetes_config(cls, check_type: HealthCheckType) -> 'HealthCheckConfig':
        """Optimized configuration for Kubernetes."""
        
        if check_type == HealthCheckType.LIVENESS:
            return cls(
                initial_delay_seconds=30,
                period_seconds=10,
                timeout_seconds=5,
                failure_threshold=3,
                success_threshold=1,
                check_model_loaded=False,  # Don't restart for model issues
                check_minio_connectivity=False,  # Don't restart for external deps
                check_cache_initialized=True,
                check_performance_manager=True,
                include_detailed_info=False,
                max_response_time_ms=3000
            )
        elif check_type == HealthCheckType.READINESS:
            return cls(
                initial_delay_seconds=15,
                period_seconds=5,
                timeout_seconds=3,
                failure_threshold=2,
                success_threshold=1,
                check_model_loaded=True,  # Must have model to serve traffic
                check_minio_connectivity=True,  # Need storage for model reloads
                check_cache_initialized=True,
                check_performance_manager=True,
                include_detailed_info=False,
                max_response_time_ms=2000
            )
        elif check_type == HealthCheckType.STARTUP:
            return cls(
                initial_delay_seconds=10,
                period_seconds=5,
                timeout_seconds=3,
                failure_threshold=12,  # Allow 60 seconds for startup
                success_threshold=1,
                check_model_loaded=True,
                check_minio_connectivity=True,
                check_cache_initialized=True,
                check_performance_manager=True,
                include_detailed_info=True,
                max_response_time_ms=5000
            )
    
    @classmethod
    def _docker_swarm_config(cls, check_type: HealthCheckType) -> 'HealthCheckConfig':
        """Optimized configuration for Docker Swarm."""
        
        # Docker Swarm only has one health check type
        return cls(
            initial_delay_seconds=40,
            period_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
            success_threshold=1,
            check_model_loaded=True,
            check_minio_connectivity=True,
            check_cache_initialized=True,
            check_performance_manager=True,
            include_detailed_info=False,
            max_response_time_ms=8000
        )
    
    @classmethod
    def _docker_compose_config(cls, check_type: HealthCheckType) -> 'HealthCheckConfig':
        """Optimized configuration for Docker Compose."""
        
        return cls(
            initial_delay_seconds=40,
            period_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
            success_threshold=1,
            check_model_loaded=True,
            check_minio_connectivity=True,
            check_cache_initialized=True,
            check_performance_manager=True,
            include_detailed_info=True,  # More info for development
            max_response_time_ms=10000
        )
    
    @classmethod
    def _default_config(cls, check_type: HealthCheckType) -> 'HealthCheckConfig':
        """Default configuration."""
        
        return cls(
            initial_delay_seconds=30,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=3,
            success_threshold=1,
            check_model_loaded=True,
            check_minio_connectivity=True,
            check_cache_initialized=True,
            check_performance_manager=True,
            include_detailed_info=False,
            max_response_time_ms=5000
        )


class HealthCheckManager:
    """Manager for health check configuration and behavior."""
    
    def __init__(self):
        self.orchestration_type = os.getenv("ORCHESTRATION_TYPE", "docker-compose")
        self.health_check_mode = os.getenv("HEALTH_CHECK_MODE", "default")
        
        # Load configurations for different check types
        self.liveness_config = HealthCheckConfig.for_orchestration_type(
            self.orchestration_type, HealthCheckType.LIVENESS
        )
        self.readiness_config = HealthCheckConfig.for_orchestration_type(
            self.orchestration_type, HealthCheckType.READINESS
        )
        self.startup_config = HealthCheckConfig.for_orchestration_type(
            self.orchestration_type, HealthCheckType.STARTUP
        )
    
    def get_config(self, check_type: HealthCheckType) -> HealthCheckConfig:
        """Get configuration for specific health check type."""
        
        if check_type == HealthCheckType.LIVENESS:
            return self.liveness_config
        elif check_type == HealthCheckType.READINESS:
            return self.readiness_config
        elif check_type == HealthCheckType.STARTUP:
            return self.startup_config
        else:
            return self.liveness_config
    
    def should_check_component(self, component: str, check_type: HealthCheckType) -> bool:
        """Determine if a component should be checked for given health check type."""
        
        config = self.get_config(check_type)
        
        component_checks = {
            "model_loaded": config.check_model_loaded,
            "minio_connectivity": config.check_minio_connectivity,
            "cache_initialized": config.check_cache_initialized,
            "performance_manager": config.check_performance_manager,
        }
        
        return component_checks.get(component, True)
    
    def get_kubernetes_probe_config(self, check_type: HealthCheckType) -> Dict[str, Any]:
        """Generate Kubernetes probe configuration."""
        
        config = self.get_config(check_type)
        
        probe_config = {
            "httpGet": {
                "path": "/health/ready" if check_type != HealthCheckType.LIVENESS else "/health",
                "port": 8000,
                "scheme": "HTTP"
            },
            "initialDelaySeconds": config.initial_delay_seconds,
            "periodSeconds": config.period_seconds,
            "timeoutSeconds": config.timeout_seconds,
            "successThreshold": config.success_threshold,
            "failureThreshold": config.failure_threshold
        }
        
        return probe_config
    
    def get_docker_healthcheck_config(self) -> Dict[str, Any]:
        """Generate Docker healthcheck configuration."""
        
        config = self.get_config(HealthCheckType.LIVENESS)  # Docker uses single health check
        
        healthcheck_config = {
            "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
            "interval": f"{config.period_seconds}s",
            "timeout": f"{config.timeout_seconds}s",
            "retries": config.failure_threshold,
            "start_period": f"{config.initial_delay_seconds}s"
        }
        
        return healthcheck_config
    
    def generate_orchestration_configs(self) -> Dict[str, Dict[str, Any]]:
        """Generate all orchestration configurations."""
        
        configs = {}
        
        # Kubernetes configuration
        configs["kubernetes"] = {
            "livenessProbe": self.get_kubernetes_probe_config(HealthCheckType.LIVENESS),
            "readinessProbe": self.get_kubernetes_probe_config(HealthCheckType.READINESS),
            "startupProbe": self.get_kubernetes_probe_config(HealthCheckType.STARTUP)
        }
        
        # Docker configuration
        configs["docker"] = {
            "healthcheck": self.get_docker_healthcheck_config()
        }
        
        return configs


def get_health_check_manager() -> HealthCheckManager:
    """Get singleton health check manager instance."""
    
    if not hasattr(get_health_check_manager, "_instance"):
        get_health_check_manager._instance = HealthCheckManager()
    
    return get_health_check_manager._instance


# Environment variable configuration
ORCHESTRATION_CONFIGS = {
    "kubernetes": {
        "HEALTH_CHECK_TIMEOUT": "3",
        "HEALTH_CHECK_PERIOD": "5",
        "HEALTH_CHECK_INITIAL_DELAY": "15",
        "HEALTH_CHECK_FAILURE_THRESHOLD": "2"
    },
    "docker-swarm": {
        "HEALTH_CHECK_TIMEOUT": "10",
        "HEALTH_CHECK_PERIOD": "30",
        "HEALTH_CHECK_INITIAL_DELAY": "40",
        "HEALTH_CHECK_FAILURE_THRESHOLD": "3"
    },
    "docker-compose": {
        "HEALTH_CHECK_TIMEOUT": "10",
        "HEALTH_CHECK_PERIOD": "30",
        "HEALTH_CHECK_INITIAL_DELAY": "40",
        "HEALTH_CHECK_FAILURE_THRESHOLD": "3"
    }
}


def configure_for_orchestration(orchestration_type: str) -> Dict[str, str]:
    """Get environment variable configuration for orchestration type."""
    
    return ORCHESTRATION_CONFIGS.get(orchestration_type.lower(), ORCHESTRATION_CONFIGS["docker-compose"])