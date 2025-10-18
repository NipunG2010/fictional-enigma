"""
Configuration management for HMM Microservice.

Handles environment variables, settings validation, and configuration loading.
"""

import os
import asyncio
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from minio import Minio
from minio.error import S3Error
import logging

logger = logging.getLogger(__name__)


class MinIOConfig(BaseSettings):
    """MinIO storage configuration with connection validation."""
    
    endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    secret_key: str = Field(default="minioadmin123", env="MINIO_SECRET_KEY")
    bucket: str = Field(default="hmm-artifacts", env="MINIO_BUCKET")
    secure: bool = Field(default=False, env="MINIO_SECURE")
    
    # Connection settings
    timeout: int = Field(default=30, env="MINIO_TIMEOUT")
    max_pool_size: int = Field(default=10, env="MINIO_MAX_POOL_SIZE")
    
    # Retry settings
    max_retries: int = Field(default=3, env="MINIO_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="MINIO_RETRY_DELAY")
    
    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v):
        """Validate MinIO endpoint format."""
        if not v or ":" not in v:
            raise ValueError("MinIO endpoint must be in format 'host:port'")
        return v
    
    @field_validator("bucket")
    @classmethod
    def validate_bucket_name(cls, v):
        """Validate MinIO bucket name."""
        if not v or len(v) < 3:
            raise ValueError("MinIO bucket name must be at least 3 characters")
        return v
    
    def validate_connection(self) -> bool:
        """Validate MinIO connection and bucket access."""
        try:
            client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            
            # Check if bucket exists and is accessible
            if not client.bucket_exists(self.bucket):
                logger.warning(f"MinIO bucket '{self.bucket}' does not exist")
                return False
                
            # Try to list objects to verify permissions
            list(client.list_objects(self.bucket, max_keys=1))
            logger.info(f"MinIO connection validated successfully for bucket '{self.bucket}'")
            return True
            
        except S3Error as e:
            logger.error(f"MinIO connection validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during MinIO validation: {e}")
            return False
    
    class Config:
        env_prefix = "MINIO_"


class ServiceConfig(BaseSettings):
    """Main service configuration with performance tuning."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HMM_SERVICE_HOST")
    port: int = Field(default=8000, env="HMM_SERVICE_PORT")
    workers: int = Field(default=1, env="HMM_SERVICE_WORKERS")
    debug: bool = Field(default=False, env="HMM_SERVICE_DEBUG")
    
    # Logging
    log_level: str = Field(default="INFO", env="HMM_LOG_LEVEL")
    log_format: str = Field(default="json", env="HMM_LOG_FORMAT")
    
    # Security
    allowed_origins: List[str] = Field(
        default=["*"], 
        env="HMM_ALLOWED_ORIGINS"
    )
    allowed_hosts: List[str] = Field(
        default=["*"], 
        env="HMM_ALLOWED_HOSTS"
    )
    api_key: Optional[str] = Field(default=None, env="HMM_API_KEY")
    
    # Model settings
    default_experiment_id: str = Field(
        default="production_hmm", 
        env="HMM_DEFAULT_EXPERIMENT_ID"
    )
    model_reload_interval: int = Field(
        default=300, 
        env="HMM_MODEL_RELOAD_INTERVAL"
    )
    model_validation_enabled: bool = Field(
        default=True, 
        env="HMM_MODEL_VALIDATION_ENABLED"
    )
    
    # Cache settings
    cache_size: int = Field(default=1000, env="HMM_CACHE_SIZE")
    cache_ttl: int = Field(default=300, env="HMM_CACHE_TTL")
    cache_enabled: bool = Field(default=True, env="HMM_CACHE_ENABLED")
    
    # Performance settings
    max_concurrent_requests: int = Field(
        default=100, 
        env="HMM_MAX_CONCURRENT_REQUESTS"
    )
    request_timeout: float = Field(
        default=30.0, 
        env="HMM_REQUEST_TIMEOUT"
    )
    inference_timeout: float = Field(
        default=5.0, 
        env="HMM_INFERENCE_TIMEOUT"
    )
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = Field(
        default=True, 
        env="HMM_CIRCUIT_BREAKER_ENABLED"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5, 
        env="HMM_CIRCUIT_BREAKER_FAILURE_THRESHOLD"
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=60, 
        env="HMM_CIRCUIT_BREAKER_RECOVERY_TIMEOUT"
    )
    
    # Health check settings
    health_check_interval: int = Field(
        default=30, 
        env="HMM_HEALTH_CHECK_INTERVAL"
    )
    startup_timeout: int = Field(
        default=60, 
        env="HMM_STARTUP_TIMEOUT"
    )
    
    # MinIO configuration
    minio: MinIOConfig = Field(default_factory=MinIOConfig)
    
    # Metrics and monitoring
    metrics_enabled: bool = Field(default=True, env="HMM_METRICS_ENABLED")
    metrics_port: int = Field(default=8001, env="HMM_METRICS_PORT")
    
    # Resource limits
    max_memory_mb: Optional[int] = Field(default=None, env="HMM_MAX_MEMORY_MB")
    max_cpu_percent: Optional[float] = Field(default=None, env="HMM_MAX_CPU_PERCENT")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v):
        """Validate log format."""
        valid_formats = ["json", "text"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of {valid_formats}")
        return v.lower()
    
    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse allowed origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @field_validator("port", "metrics_port")
    @classmethod
    def validate_port(cls, v):
        """Validate port numbers."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator("max_concurrent_requests")
    @classmethod
    def validate_max_concurrent_requests(cls, v):
        """Validate max concurrent requests."""
        if v <= 0:
            raise ValueError("Max concurrent requests must be positive")
        return v
    
    @field_validator("inference_timeout", "request_timeout")
    @classmethod
    def validate_timeouts(cls, v):
        """Validate timeout values."""
        if v <= 0:
            raise ValueError("Timeout values must be positive")
        return v
    
    @model_validator(mode="after")
    def validate_performance_settings(self):
        """Validate performance-related settings."""
        if self.inference_timeout >= self.request_timeout:
            raise ValueError("Inference timeout must be less than request timeout")
        
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("Max memory must be positive")
        
        if self.max_cpu_percent is not None and not 0 < self.max_cpu_percent <= 100:
            raise ValueError("Max CPU percent must be between 0 and 100")
        
        return self
    
    def validate_minio_connection(self) -> bool:
        """Validate MinIO connection."""
        return self.minio.validate_connection()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> ServiceConfig:
    """Get cached service configuration."""
    return ServiceConfig()