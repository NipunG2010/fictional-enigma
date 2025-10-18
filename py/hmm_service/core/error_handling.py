"""
Comprehensive error handling for HMM Microservice.

Provides structured error handling with detailed messages, fallback mechanisms,
and proper HTTP status codes for different error conditions.
"""

import time
import traceback
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from dataclasses import dataclass
from fastapi import HTTPException, Request
from pydantic import BaseModel

from .logging_config import get_logger


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    
    VALIDATION = "validation"
    MODEL = "model"
    SYSTEM = "system"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    TIMEOUT = "timeout"


class ErrorSeverity(Enum):
    """Error severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: Optional[float] = None
    processing_time_ms: Optional[float] = None
    additional_data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    
    error: str
    error_code: str
    message: str
    timestamp: int
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    retry_after: Optional[int] = None
    support_reference: Optional[str] = None


class HMMServiceError(Exception):
    """Base exception for HMM service errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.retry_after = retry_after
        self.timestamp = time.time()


class ValidationError(HMMServiceError):
    """Validation error with detailed field information."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["provided_value"] = str(value)
        if expected:
            error_details["expected"] = expected
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            details=error_details
        )


class ModelError(HMMServiceError):
    """Model-related errors with fallback information."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MODEL_ERROR",
        model_version: Optional[str] = None,
        fallback_available: bool = False,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if model_version:
            error_details["model_version"] = model_version
        error_details["fallback_available"] = fallback_available
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.MODEL,
            severity=ErrorSeverity.HIGH if not fallback_available else ErrorSeverity.MEDIUM,
            details=error_details
        )


class SystemError(HMMServiceError):
    """System-level errors with resource information."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "SYSTEM_ERROR",
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if resource:
            error_details["resource"] = resource
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            details=error_details
        )


class NetworkError(HMMServiceError):
    """Network-related errors with retry information."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "NETWORK_ERROR",
        service: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if service:
            error_details["service"] = service
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            retry_after=retry_after
        )


class RateLimitError(HMMServiceError):
    """Rate limiting errors with retry information."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        limit_type: str = "requests",
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["limit_type"] = limit_type
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.LOW,
            details=error_details,
            retry_after=retry_after
        )


class TimeoutError(HMMServiceError):
    """Timeout errors with operation information."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        if timeout_seconds:
            error_details["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            details=error_details
        )


class ErrorHandler:
    """Comprehensive error handler with logging and fallback mechanisms."""
    
    def __init__(self):
        self.logger = get_logger("hmm_service.error_handler")
        self._error_counts = {}
        self._fallback_responses = {}
    
    def handle_error(
        self,
        error: Union[Exception, HMMServiceError],
        context: Optional[ErrorContext] = None
    ) -> ErrorResponse:
        """
        Handle any error and return standardized error response.
        
        Args:
            error: Exception or HMMServiceError to handle
            context: Optional error context information
            
        Returns:
            Standardized error response
        """
        context = context or ErrorContext()
        
        # Convert to HMMServiceError if needed
        if not isinstance(error, HMMServiceError):
            hmm_error = self._convert_to_hmm_error(error)
        else:
            hmm_error = error
        
        # Log the error with appropriate level
        self._log_error(hmm_error, context)
        
        # Track error for monitoring
        self._track_error(hmm_error, context)
        
        # Generate error response
        return self._create_error_response(hmm_error, context)
    
    def _convert_to_hmm_error(self, error: Exception) -> HMMServiceError:
        """Convert generic exception to HMMServiceError."""
        error_message = str(error)
        error_type = type(error).__name__
        
        # Classify error based on type and message
        if isinstance(error, ValueError):
            if "validation" in error_message.lower() or "invalid" in error_message.lower():
                return ValidationError(
                    message=error_message,
                    details={"original_type": error_type}
                )
        
        elif isinstance(error, ConnectionError):
            return NetworkError(
                message=f"Connection error: {error_message}",
                details={"original_type": error_type}
            )
        
        elif isinstance(error, TimeoutError):
            return TimeoutError(
                message=f"Operation timeout: {error_message}",
                details={"original_type": error_type}
            )
        
        elif "model" in error_message.lower():
            return ModelError(
                message=error_message,
                details={"original_type": error_type}
            )
        
        elif "minio" in error_message.lower() or "s3" in error_message.lower():
            return NetworkError(
                message=f"Storage service error: {error_message}",
                service="minio",
                details={"original_type": error_type}
            )
        
        # Default to system error
        return SystemError(
            message=f"Unexpected error: {error_message}",
            details={
                "original_type": error_type,
                "traceback": traceback.format_exc()
            }
        )
    
    def _log_error(self, error: HMMServiceError, context: ErrorContext):
        """Log error with appropriate level and structured data."""
        log_data = {
            "error_code": error.error_code,
            "error_category": error.category.value,
            "error_severity": error.severity.value,
            "error_message": error.message,
            "error_details": error.details,
            "endpoint": context.endpoint,
            "method": context.method,
            "client_ip": context.client_ip,
            "processing_time_ms": context.processing_time_ms,
        }
        
        # Add cause information if available
        if error.cause:
            log_data["cause_type"] = type(error.cause).__name__
            log_data["cause_message"] = str(error.cause)
        
        # Log with appropriate level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.error(f"CRITICAL ERROR: {error.message}", extra=log_data, exc_info=True)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY: {error.message}", extra=log_data)
        else:
            self.logger.info(f"LOW SEVERITY: {error.message}", extra=log_data)
        
        # Log audit trail for error
        self.logger.log_audit(
            f"Error occurred: {error.error_code}",
            endpoint=context.endpoint or "unknown",
            method=context.method or "unknown",
            status_code=self._get_http_status_code(error),
            client_ip=context.client_ip,
            error=error.message,
            error_code=error.error_code,
            error_category=error.category.value,
            error_severity=error.severity.value
        )
    
    def _track_error(self, error: HMMServiceError, context: ErrorContext):
        """Track error for monitoring and alerting."""
        error_key = f"{error.category.value}:{error.error_code}"
        
        if error_key not in self._error_counts:
            self._error_counts[error_key] = {
                "count": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "severity": error.severity.value
            }
        
        self._error_counts[error_key]["count"] += 1
        self._error_counts[error_key]["last_seen"] = time.time()
        
        # Log metrics for monitoring
        self.logger.log_performance(
            "error_occurred",
            context.processing_time_ms or 0,
            endpoint=context.endpoint,
            error_code=error.error_code,
            error_category=error.category.value,
            error_severity=error.severity.value
        )
    
    def _create_error_response(self, error: HMMServiceError, context: ErrorContext) -> ErrorResponse:
        """Create standardized error response."""
        return ErrorResponse(
            error=error.category.value.upper(),
            error_code=error.error_code,
            message=error.message,
            timestamp=int(error.timestamp),
            request_id=context.request_id,
            details=error.details if error.details else None,
            retry_after=error.retry_after,
            support_reference=self._generate_support_reference(error, context)
        )
    
    def _generate_support_reference(self, error: HMMServiceError, context: ErrorContext) -> str:
        """Generate support reference for error tracking."""
        timestamp = int(error.timestamp)
        error_hash = hash(f"{error.error_code}:{error.message}") % 10000
        return f"HMM-{timestamp}-{error_hash:04d}"
    
    def _get_http_status_code(self, error: HMMServiceError) -> int:
        """Get appropriate HTTP status code for error."""
        status_map = {
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.AUTHENTICATION: 401,
            ErrorCategory.AUTHORIZATION: 403,
            ErrorCategory.RESOURCE: 404,
            ErrorCategory.RATE_LIMIT: 429,
            ErrorCategory.TIMEOUT: 408,
            ErrorCategory.MODEL: 503,
            ErrorCategory.NETWORK: 503,
            ErrorCategory.SYSTEM: 500,
        }
        
        return status_map.get(error.category, 500)
    
    def create_http_exception(self, error: HMMServiceError, context: Optional[ErrorContext] = None) -> HTTPException:
        """Create FastAPI HTTPException from HMMServiceError."""
        context = context or ErrorContext()
        error_response = self._create_error_response(error, context)
        status_code = self._get_http_status_code(error)
        
        return HTTPException(
            status_code=status_code,
            detail=error_response.dict(),
            headers={"X-Error-Code": error.error_code} if error.retry_after is None 
                   else {"X-Error-Code": error.error_code, "Retry-After": str(error.retry_after)}
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "error_counts": dict(self._error_counts),
            "total_errors": sum(stats["count"] for stats in self._error_counts.values()),
            "error_categories": list(set(key.split(":")[0] for key in self._error_counts.keys())),
            "recent_errors": [
                key for key, stats in self._error_counts.items()
                if time.time() - stats["last_seen"] < 300  # Last 5 minutes
            ]
        }


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _error_handler


def handle_validation_error(
    message: str,
    field: Optional[str] = None,
    value: Optional[Any] = None,
    expected: Optional[str] = None,
    context: Optional[ErrorContext] = None
) -> ErrorResponse:
    """Handle validation error with detailed information."""
    error = ValidationError(
        message=message,
        field=field,
        value=value,
        expected=expected
    )
    return _error_handler.handle_error(error, context)


def handle_model_error(
    message: str,
    error_code: str = "MODEL_ERROR",
    model_version: Optional[str] = None,
    fallback_available: bool = False,
    context: Optional[ErrorContext] = None
) -> ErrorResponse:
    """Handle model error with fallback information."""
    error = ModelError(
        message=message,
        error_code=error_code,
        model_version=model_version,
        fallback_available=fallback_available
    )
    return _error_handler.handle_error(error, context)


def handle_system_error(
    message: str,
    error_code: str = "SYSTEM_ERROR",
    resource: Optional[str] = None,
    context: Optional[ErrorContext] = None
) -> ErrorResponse:
    """Handle system error with resource information."""
    error = SystemError(
        message=message,
        error_code=error_code,
        resource=resource
    )
    return _error_handler.handle_error(error, context)


def create_error_context(request: Request, processing_time_ms: Optional[float] = None) -> ErrorContext:
    """Create error context from FastAPI request."""
    return ErrorContext(
        request_id=getattr(request.state, "request_id", None),
        endpoint=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        timestamp=time.time(),
        processing_time_ms=processing_time_ms
    )