"""
Logging configuration for HMM Microservice.

Sets up structured JSON logging with proper formatters and handlers.
Provides specialized loggers for requests, performance, and audit trails.
"""

import logging
import logging.config
import sys
import time
from typing import Dict, Any, Optional
from contextvars import ContextVar

# Context variable for request ID
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class RequestContextFilter(logging.Filter):
    """Filter to add request context to log records."""
    
    def filter(self, record):
        """Add request ID to log record if available."""
        record.request_id = request_id_context.get()
        return True


class PerformanceFormatter(logging.Formatter):
    """Custom formatter for performance logs."""
    
    def format(self, record):
        """Format performance log records with timing information."""
        if hasattr(record, 'processing_time_ms'):
            record.msg = f"{record.msg} (took {record.processing_time_ms:.2f}ms)"
        return super().format(record)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup structured logging configuration with specialized loggers."""
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s %(pathname)s %(lineno)d %(funcName)s",
                "datefmt": "%Y-%m-%dT%H:%M:%SZ"
            },
            "json_performance": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s %(processing_time_ms)s %(operation)s %(endpoint)s",
                "datefmt": "%Y-%m-%dT%H:%M:%SZ"
            },
            "json_audit": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s %(user_id)s %(endpoint)s %(method)s %(status_code)s %(client_ip)s",
                "datefmt": "%Y-%m-%dT%H:%M:%SZ"
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(request_id)s]",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "filters": {
            "request_context": {
                "()": RequestContextFilter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "json",
                "stream": sys.stdout,
                "filters": ["request_context"]
            },
            "performance": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "json_performance",
                "stream": sys.stdout,
                "filters": ["request_context"]
            },
            "audit": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "json_audit",
                "stream": sys.stdout,
                "filters": ["request_context"]
            }
        },
        "loggers": {
            "hmm_service": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False
            },
            "hmm_service.performance": {
                "level": "INFO",
                "handlers": ["performance"],
                "propagate": False
            },
            "hmm_service.audit": {
                "level": "INFO",
                "handlers": ["audit"],
                "propagate": False
            },
            "hmm_service.requests": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "WARNING",  # Reduce noise, we have our own request logging
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"]
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Set up logger for the service
    logger = logging.getLogger("hmm_service")
    logger.info("Structured logging configured", extra={
        "log_level": log_level,
        "service": "hmm-microservice",
        "version": "1.0.0"
    })


class StructuredLogger:
    """Structured logger with context support and specialized methods."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.performance_logger = logging.getLogger("hmm_service.performance")
        self.audit_logger = logging.getLogger("hmm_service.audit")
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, extra=kwargs)
    
    def log_performance(self, operation: str, processing_time_ms: float, 
                       endpoint: str = None, **kwargs):
        """Log performance metrics for operations."""
        self.performance_logger.info(
            f"Performance: {operation}",
            extra={
                "operation": operation,
                "processing_time_ms": processing_time_ms,
                "endpoint": endpoint,
                **kwargs
            }
        )
    
    def log_audit(self, message: str, endpoint: str, method: str, 
                  status_code: int = None, client_ip: str = None, 
                  user_id: str = None, **kwargs):
        """Log audit trail for API requests."""
        self.audit_logger.info(
            message,
            extra={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "client_ip": client_ip,
                "user_id": user_id,
                **kwargs
            }
        )


class PerformanceTimer:
    """Context manager for timing operations and logging performance."""
    
    def __init__(self, logger: StructuredLogger, operation: str, 
                 endpoint: str = None, **kwargs):
        self.logger = logger
        self.operation = operation
        self.endpoint = endpoint
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log performance."""
        if self.start_time:
            processing_time_ms = (time.time() - self.start_time) * 1000
            self.logger.log_performance(
                self.operation,
                processing_time_ms,
                self.endpoint,
                **self.kwargs
            )


def set_request_context(request_id: str):
    """Set the request ID in the context for logging."""
    request_id_context.set(request_id)


def get_request_context() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_context.get()


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)