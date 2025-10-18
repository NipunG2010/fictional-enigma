"""
Health check endpoints for HMM Microservice.

Provides health and readiness checks for monitoring and orchestration.
Optimized for different container orchestration platforms.
"""

import time
import psutil
import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, Response, Request
from pydantic import BaseModel, Field

from core.dependencies import (
    get_inference_engine_dep, 
    get_model_loader_dep,
    get_cache_manager_dep,
    get_performance_manager_dep
)
from core.config import get_settings
from core.logging_config import get_logger, PerformanceTimer
from health_check_config import get_health_check_manager, HealthCheckType


router = APIRouter()


class HealthResponse(BaseModel):
    """Basic health check response."""
    
    status: str = Field(..., description="Service health status")
    timestamp: int = Field(..., description="Response timestamp")
    uptime: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="Service version")


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    
    ready: bool = Field(..., description="Service readiness status")
    model_loaded: bool = Field(..., description="Whether HMM model is loaded")
    cache_initialized: bool = Field(..., description="Whether cache is initialized")
    last_inference: Optional[int] = Field(None, description="Last inference timestamp")
    checks: Dict[str, bool] = Field(..., description="Individual readiness checks")
    timestamp: int = Field(..., description="Response timestamp")


class DetailedHealthResponse(BaseModel):
    """Detailed health information."""
    
    status: str = Field(..., description="Service health status")
    timestamp: int = Field(..., description="Response timestamp")
    uptime: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="Service version")
    
    # System metrics
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    
    # Service metrics
    model_info: Dict[str, Any] = Field(..., description="Current model information")
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")
    performance_stats: Dict[str, Any] = Field(..., description="Performance manager statistics")
    
    # Configuration
    config: Dict[str, Any] = Field(..., description="Service configuration")


# Track service start time
_service_start_time = time.time()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Basic health check endpoint for load balancers"
)
async def health_check(request: Request) -> HealthResponse:
    """Basic health check endpoint."""
    logger = get_logger("hmm_service.health")
    
    with PerformanceTimer(logger, "health_check", "/health") as timer:
        current_time = time.time()
        uptime = current_time - _service_start_time
        
        # Log health check request (debug level to avoid noise)
        logger.debug("Health check requested", extra={
            "endpoint": "/health",
            "uptime": round(uptime, 2),
            "client_ip": request.client.host if request.client else None,
        })
        
        response = HealthResponse(
            status="healthy",
            timestamp=int(current_time),
            uptime=round(uptime, 2),
            version="1.0.0"
        )
        
        # Log audit trail for health check (minimal logging)
        logger.log_audit(
            "Health check completed",
            endpoint="/health",
            method="GET",
            status_code=200,
            client_ip=request.client.host if request.client else None
        )
        
        return response


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Readiness check for container orchestration"
)
async def readiness_check(
    request: Request,
    response: Response,
    model_loader=Depends(get_model_loader_dep),
    cache_manager=Depends(get_cache_manager_dep),
    inference_engine=Depends(get_inference_engine_dep),
    performance_manager=Depends(get_performance_manager_dep)
) -> ReadinessResponse:
    """Readiness check endpoint optimized for container orchestration."""
    logger = get_logger("hmm_service.health")
    health_manager = get_health_check_manager()
    
    with PerformanceTimer(logger, "readiness_check", "/health/ready") as timer:
        # Log readiness check start
        logger.info("Readiness check started", extra={
            "endpoint": "/health/ready",
            "orchestration_type": os.getenv("ORCHESTRATION_TYPE", "unknown"),
            "client_ip": request.client.host if request.client else None,
        })
        
        # Perform readiness checks based on orchestration configuration
        checks = {}
        
        try:
            # Check if inference engine is initialized
            engine_health = inference_engine.get_health_status()
            inference_initialized = engine_health.get("initialized", False)
            checks["inference_initialized"] = inference_initialized
            
            # Check if model is loaded (configurable based on orchestration)
            model_loaded = engine_health.get("model_loaded", False)
            if health_manager.should_check_component("model_loaded", HealthCheckType.READINESS):
                checks["model_loaded"] = model_loaded
            else:
                checks["model_loaded"] = True  # Skip check
            
            # Check if model is valid
            model_valid = engine_health.get("model_valid", False)
            checks["model_valid"] = model_valid if model_loaded else True
            
            # Check if cache is initialized (configurable)
            if health_manager.should_check_component("cache_initialized", HealthCheckType.READINESS):
                try:
                    cache_initialized = hasattr(cache_manager, '_initialized') and getattr(cache_manager, '_initialized', False)
                except:
                    cache_initialized = False
                checks["cache_initialized"] = cache_initialized
            else:
                checks["cache_initialized"] = True  # Skip check
            
            # Check MinIO connectivity (configurable)
            if health_manager.should_check_component("minio_connectivity", HealthCheckType.READINESS):
                try:
                    minio_connected = hasattr(model_loader, '_initialized') and getattr(model_loader, '_initialized', False)
                except:
                    minio_connected = False
                checks["minio_connected"] = minio_connected
            else:
                checks["minio_connected"] = True  # Skip check
            
            # Check performance manager health (configurable)
            if health_manager.should_check_component("performance_manager", HealthCheckType.READINESS):
                try:
                    perf_health = performance_manager.get_health_status()
                    performance_healthy = perf_health.get("overall_healthy", False)
                except:
                    performance_healthy = False
                checks["performance_manager_healthy"] = performance_healthy
            else:
                checks["performance_manager_healthy"] = True  # Skip check
            
            # Overall readiness based on all enabled checks
            ready = all(checks.values())
            
            # Get last inference time from engine
            last_inference = engine_health.get("last_inference_time")
            
            # Log readiness check results
            logger.info("Readiness check completed", extra={
                "endpoint": "/health/ready",
                "ready": ready,
                "orchestration_type": os.getenv("ORCHESTRATION_TYPE", "unknown"),
                "enabled_checks": [k for k, v in checks.items() if k != "error"],
                "failed_checks": [k for k, v in checks.items() if not v and k != "error"],
                "last_inference": last_inference,
            })
            
            if not ready:
                response.status_code = 503
                logger.warning("Service not ready", extra={
                    "endpoint": "/health/ready",
                    "failed_checks": {k: v for k, v in checks.items() if not v},
                })
            
            # Log audit trail for readiness check
            logger.log_audit(
                "Readiness check completed",
                endpoint="/health/ready",
                method="GET",
                status_code=200 if ready else 503,
                client_ip=request.client.host if request.client else None,
                ready=ready,
                orchestration_type=os.getenv("ORCHESTRATION_TYPE", "unknown")
            )
            
            return ReadinessResponse(
                ready=ready,
                model_loaded=model_loaded,
                cache_initialized=checks.get("cache_initialized", True),
                last_inference=last_inference,
                checks=checks,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            # Log readiness check error
            logger.error("Readiness check failed", extra={
                "endpoint": "/health/ready",
                "error": str(e),
                "error_type": type(e).__name__,
                "orchestration_type": os.getenv("ORCHESTRATION_TYPE", "unknown"),
            }, exc_info=True)
            
            response.status_code = 503
            
            # Log audit trail for failed readiness check
            logger.log_audit(
                "Readiness check failed",
                endpoint="/health/ready",
                method="GET",
                status_code=503,
                client_ip=request.client.host if request.client else None,
                error=str(e),
                orchestration_type=os.getenv("ORCHESTRATION_TYPE", "unknown")
            )
            
            return ReadinessResponse(
                ready=False,
                model_loaded=False,
                cache_initialized=False,
                last_inference=None,
                checks={"error": False, "error_message": str(e)},
                timestamp=int(time.time())
            )


@router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed health information",
    description="Detailed health and system information for monitoring"
)
async def detailed_health(
    request: Request,
    model_loader=Depends(get_model_loader_dep),
    cache_manager=Depends(get_cache_manager_dep),
    inference_engine=Depends(get_inference_engine_dep),
    performance_manager=Depends(get_performance_manager_dep)
) -> DetailedHealthResponse:
    """Detailed health information endpoint."""
    logger = get_logger("hmm_service.health")
    
    with PerformanceTimer(logger, "detailed_health_check", "/health/detailed") as timer:
        # Log detailed health check start
        logger.info("Detailed health check started", extra={
            "endpoint": "/health/detailed",
            "client_ip": request.client.host if request.client else None,
        })
        
        current_time = time.time()
        uptime = current_time - _service_start_time
        settings = get_settings()
        
        # Get system metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        cpu_usage = process.cpu_percent()
        
        # Get model info from inference engine
        model_info = inference_engine.get_model_info()
        
        # Get cache stats (basic implementation)
        cache_stats = {
            "size": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0
        }
        
        # Try to get cache stats if available
        try:
            if hasattr(cache_manager, 'get_stats'):
                cache_stats = cache_manager.get_stats()
        except Exception as e:
            logger.debug("Could not retrieve cache stats", extra={
                "error": str(e)
            })
        
        # Get performance manager stats
        performance_stats = {
            "initialized": False,
            "connection_pool": {},
            "request_queue": {}
        }
        
        try:
            performance_stats = performance_manager.get_stats()
        except Exception as e:
            logger.debug("Could not retrieve performance stats", extra={
                "error": str(e)
            })
        
        # Service configuration (sanitized)
        config = {
            "host": settings.host,
            "port": settings.port,
            "debug": settings.debug,
            "log_level": settings.log_level,
            "cache_size": settings.cache_size,
            "cache_ttl": settings.cache_ttl,
            "max_concurrent_requests": settings.max_concurrent_requests,
            "default_experiment_id": settings.default_experiment_id
        }
        
        # Determine overall status
        engine_health = inference_engine.get_health_status()
        status = "healthy" if engine_health.get("model_loaded", False) else "degraded"
        
        # Log detailed health metrics
        logger.info("Detailed health check completed", extra={
            "endpoint": "/health/detailed",
            "status": status,
            "uptime": round(uptime, 2),
            "memory_usage_mb": round(memory_usage_mb, 2),
            "cpu_usage_percent": round(cpu_usage, 2),
            "model_loaded": engine_health.get("model_loaded", False),
            "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
        })
        
        # Log audit trail for detailed health check
        logger.log_audit(
            "Detailed health check completed",
            endpoint="/health/detailed",
            method="GET",
            status_code=200,
            client_ip=request.client.host if request.client else None,
            status=status,
            memory_usage_mb=round(memory_usage_mb, 2),
            cpu_usage_percent=round(cpu_usage, 2)
        )
        
        return DetailedHealthResponse(
            status=status,
            timestamp=int(current_time),
            uptime=round(uptime, 2),
            version="1.0.0",
            memory_usage_mb=round(memory_usage_mb, 2),
            cpu_usage_percent=round(cpu_usage, 2),
            model_info=model_info,
            cache_stats=cache_stats,
            performance_stats=performance_stats,
            config=config
        )