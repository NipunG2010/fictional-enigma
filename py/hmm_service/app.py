"""
HMM Microservice - FastAPI application for Hidden Markov Model inference.

This service provides REST endpoints for HMM state probability calculation
and fusion weight computation for the IMP trading system.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from core.dependencies import get_inference_engine, get_cache_manager, get_model_loader, get_performance_manager
from core.config import get_settings
from routers import inference, health, models
from core.logging_config import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    settings = get_settings()
    
    # Startup
    logger = get_logger("hmm_service.startup")
    logger.info("Starting HMM Microservice initialization", extra={
        "service": "hmm-microservice",
        "version": "1.0.0",
        "environment": getattr(settings, 'environment', 'development')
    })
    
    startup_start_time = time.time()
    
    try:
        # Initialize performance manager first (includes connection pooling)
        logger.info("Initializing performance manager")
        performance_manager = get_performance_manager()
        await performance_manager.initialize()
        logger.info("Performance manager initialized successfully")
        
        # Initialize cache manager
        logger.info("Initializing cache manager")
        cache_manager = get_cache_manager()
        await cache_manager.initialize()
        logger.info("Cache manager initialized successfully")
        
        # Initialize model loader with performance manager
        logger.info("Initializing model loader")
        model_loader = get_model_loader()
        await model_loader.initialize(performance_manager=performance_manager)
        logger.info("Model loader initialized successfully")
        
        # Initialize inference engine with cache manager and load model
        logger.info("Initializing inference engine")
        inference_engine = get_inference_engine()
        await inference_engine.initialize(cache_manager=cache_manager)
        logger.info("Inference engine initialized successfully")
        
        # Load model into inference engine if available
        current_artifact, current_weights = model_loader.get_current_model()
        if current_artifact:
            logger.info("Loading default model into inference engine")
            await inference_engine.load_model(current_artifact, current_weights)
            logger.info("Default model loaded successfully", extra={
                "model_experiment_id": getattr(current_artifact, 'experiment_id', 'unknown'),
                "model_version": getattr(current_artifact, 'version', 'unknown')
            })
        else:
            logger.warning("No default model available to load")
        
        startup_time_ms = (time.time() - startup_start_time) * 1000
        logger.info("HMM Microservice startup complete", extra={
            "startup_time_ms": round(startup_time_ms, 2)
        })
        
        # Log performance metrics for startup
        logger.log_performance("service_startup", startup_time_ms)
        
    except Exception as e:
        startup_time_ms = (time.time() - startup_start_time) * 1000
        logger.error("Failed to initialize HMM Microservice", extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "startup_time_ms": round(startup_time_ms, 2)
        }, exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Initiating HMM Microservice shutdown")
    shutdown_start_time = time.time()
    
    try:
        # Cleanup resources
        logger.info("Cleaning up performance manager")
        performance_manager = get_performance_manager()
        await performance_manager.cleanup()
        
        logger.info("Cleaning up cache manager")
        cache_manager = get_cache_manager()
        await cache_manager.cleanup()
        
        logger.info("Cleaning up model loader")
        model_loader = get_model_loader()
        await model_loader.cleanup()
        
        shutdown_time_ms = (time.time() - shutdown_start_time) * 1000
        logger.info("HMM Microservice shutdown complete", extra={
            "shutdown_time_ms": round(shutdown_time_ms, 2)
        })
        
        # Log performance metrics for shutdown
        logger.log_performance("service_shutdown", shutdown_time_ms)
        
    except Exception as e:
        shutdown_time_ms = (time.time() - shutdown_start_time) * 1000
        logger.error("Error during shutdown", extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "shutdown_time_ms": round(shutdown_time_ms, 2)
        }, exc_info=True)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    # Setup logging first
    setup_logging(settings.log_level)
    
    app = FastAPI(
        title="HMM Microservice",
        description="Hidden Markov Model inference service for IMP trading system",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Add middleware
    setup_middleware(app, settings)
    
    # Add routers
    app.include_router(inference.router, prefix="/inference", tags=["inference"])
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(models.router, prefix="/models", tags=["models"])
    
    # Add global exception handler
    setup_exception_handlers(app)
    
    # Add request logging middleware
    setup_request_logging(app)
    
    return app


def setup_middleware(app: FastAPI, settings) -> None:
    """Configure application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware for production
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers with structured logging."""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with structured error response and logging."""
        logger = get_logger("hmm_service.exceptions")
        request_id = getattr(request.state, "request_id", None)
        
        # Log HTTP exception with context
        logger.warning("HTTP exception occurred", extra={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
        })
        
        # Log audit trail for HTTP errors
        logger.log_audit(
            "HTTP exception",
            endpoint=request.url.path,
            method=request.method,
            status_code=exc.status_code,
            client_ip=request.client.host if request.client else None,
            error_detail=exc.detail
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "error_code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "timestamp": int(time.time()),
                "request_id": request_id,
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions with comprehensive logging."""
        logger = get_logger("hmm_service.exceptions")
        request_id = getattr(request.state, "request_id", None)
        
        # Log unhandled exception with full context
        logger.error("Unhandled exception occurred", extra={
            "error": str(exc),
            "error_type": type(exc).__name__,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
        }, exc_info=True)
        
        # Log audit trail for system errors
        logger.log_audit(
            "System exception",
            endpoint=request.url.path,
            method=request.method,
            status_code=500,
            client_ip=request.client.host if request.client else None,
            error=str(exc),
            error_type=type(exc).__name__
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "error_code": "INTERNAL_500",
                "message": "An internal error occurred",
                "timestamp": int(time.time()),
                "request_id": request_id,
            }
        )


def setup_request_logging(app: FastAPI) -> None:
    """Setup request logging middleware with structured logging and audit trail."""
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests with timing, metadata, and audit trail."""
        import uuid
        from core.logging_config import set_request_context, get_logger
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Set request context for logging
        set_request_context(request_id)
        
        # Get structured logger
        logger = get_logger("hmm_service.requests")
        start_time = time.time()
        
        # Extract client information
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent", "unknown")
        content_length = request.headers.get("content-length", "0")
        
        # Log request start with audit information
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "content_length": content_length,
            }
        )
        
        # Log audit trail for API requests
        logger.log_audit(
            "API request initiated",
            endpoint=request.url.path,
            method=request.method,
            client_ip=client_ip,
            user_agent=user_agent,
            content_length=content_length
        )
        
        # Process request
        try:
            response = await call_next(request)
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Log successful response with performance metrics
            logger.info(
                "Request completed successfully",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "response_size": response.headers.get("content-length", "unknown"),
                }
            )
            
            # Log performance metrics
            logger.log_performance(
                "http_request",
                processing_time_ms,
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code
            )
            
            # Log audit trail for completed requests
            logger.log_audit(
                "API request completed",
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                client_ip=client_ip,
                processing_time_ms=round(processing_time_ms, 2)
            )
            
            # Add request ID and timing to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time-Ms"] = str(round(processing_time_ms, 2))
            return response
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Log error response with full context
            logger.error(
                "Request failed with exception",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True
            )
            
            # Log audit trail for failed requests
            logger.log_audit(
                "API request failed",
                endpoint=request.url.path,
                method=request.method,
                status_code=500,
                client_ip=client_ip,
                error=str(e),
                error_type=type(e).__name__,
                processing_time_ms=round(processing_time_ms, 2)
            )
            
            raise


# Create the app instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )