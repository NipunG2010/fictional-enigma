//! Signal emission module for publishing trading signals to message bus infrastructure
//! 
//! This module provides the core infrastructure for emitting validated trading signals
//! to Redis Streams and Kafka topics, with comprehensive error handling and audit logging.

use thiserror::Error;

pub mod publisher;
pub mod validation;
pub mod redis_publisher;
pub mod kafka_publisher;
pub mod circuit_breaker;
pub mod buffer;
pub mod audit;
pub mod audit_logger;
pub mod s3_uploader;
pub mod feature_audit;
pub mod signal_publisher;
pub mod config;
pub mod health_monitor;
pub mod metrics;
pub mod metrics_server;

// Re-export commonly used types
pub use publisher::PublisherTrait;
pub use validation::{SignalValidator, ValidationConfig, ValidationError};
pub use redis_publisher::{RedisPublisher, RedisConfig, RedisStreamConfig, RedisHealthConfig, OrderingStrategy};
pub use kafka_publisher::{KafkaPublisher, KafkaConfig, KafkaProducerConfig, PartitioningConfig, PartitioningStrategy, KafkaHealthConfig, KafkaSecurityConfig, KafkaSslConfig, KafkaMetrics};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerMetrics};
pub use buffer::{SignalBuffer, BufferConfig, BufferedSignal, BufferMetrics, OverflowStrategy, PersistenceConfig};
pub use audit::{
    AuditEvent, SignalEmissionEvent, FeatureComputationEvent, ValidationErrorEvent, 
    ValidationErrorDetail, PublisherEvent, PublisherOperationType, HmmWeightEvent,
    generate_correlation_id, generate_event_id, current_timestamp_ms
};
pub use audit_logger::{AuditLogger, AuditConfig, LogFileMetadata, AuditLoggerStats};
pub use s3_uploader::{S3Uploader, S3Config, UploadResult, S3UploaderStats};
pub use feature_audit::{
    FeatureAuditor, FeatureAuditConfig, DataQualityIssue, FeatureTimingInfo, 
    FeatureComputationTimer, create_feature_auditor, create_feature_auditor_with_config
};
pub use signal_publisher::{
    SignalPublisher, SignalPublisherConfig, PublisherBackend, SignalPublisherMetrics
};
pub use config::{
    SignalEmissionConfig, ConfigMetadata, ConfigSource, ConfigSummary, ConfigWatcher
};
pub use health_monitor::{
    HealthMonitor, HealthMonitorConfig, HealthHttpConfig, ComponentHealth, 
    HealthCheckResult, ServiceHealth, ServiceMetrics, HealthHttpServer
};
pub use metrics::{
    SignalEmissionMetrics, MetricsTimer, SignalEmissionMetricsSnapshot
};
pub use metrics_server::{
    MetricsServer, MetricsServerConfig, StandaloneMetricsServer
};

/// Comprehensive error types for signal emission operations
#[derive(Debug, Error)]
pub enum SignalEmissionError {
    /// Signal validation failed
    #[error("Signal validation failed: {message}")]
    ValidationError { message: String },
    
    /// Publisher-specific error
    #[error("Publisher error ({backend}): {message}")]
    PublisherError { backend: String, message: String },
    
    /// Redis connection or operation error
    #[error("Redis error: {0}")]
    RedisError(#[from] redis::RedisError),
    
    /// Kafka producer error (placeholder for future implementation)
    #[error("Kafka error: {0}")]
    KafkaError(String),
    
    /// Buffer overflow when local buffering is enabled
    #[error("Buffer overflow: maximum size {max_size} exceeded")]
    BufferOverflow { max_size: usize },
    
    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
    
    /// Audit logging error
    #[error("Audit logging error: {message}")]
    AuditError { message: String },
    
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// Network timeout error
    #[error("Network timeout: operation timed out after {timeout_ms}ms")]
    TimeoutError { timeout_ms: u64 },
    
    /// Circuit breaker is open, preventing operations
    #[error("Circuit breaker open: {reason}")]
    CircuitBreakerOpen { reason: String },
    
    /// Authentication/authorization error
    #[error("Authentication error: {message}")]
    AuthError { message: String },
    
    /// Resource exhaustion error
    #[error("Resource exhausted: {resource} - {message}")]
    ResourceExhausted { resource: String, message: String },
    
    /// Generic error from anyhow
    #[error("Internal error: {0}")]
    InternalError(#[from] anyhow::Error),
}

impl SignalEmissionError {
    /// Create a validation error with a custom message
    pub fn validation(message: impl Into<String>) -> Self {
        Self::ValidationError {
            message: message.into(),
        }
    }
    
    /// Create a publisher error with backend and message
    pub fn publisher(backend: impl Into<String>, message: impl Into<String>) -> Self {
        Self::PublisherError {
            backend: backend.into(),
            message: message.into(),
        }
    }
    
    /// Create a configuration error with a custom message
    pub fn config(message: impl Into<String>) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }
    
    /// Create an audit error with a custom message
    pub fn audit(message: impl Into<String>) -> Self {
        Self::AuditError {
            message: message.into(),
        }
    }
    
    /// Create a timeout error with duration
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::TimeoutError { timeout_ms }
    }
    
    /// Create a circuit breaker error with reason
    pub fn circuit_breaker(reason: impl Into<String>) -> Self {
        Self::CircuitBreakerOpen {
            reason: reason.into(),
        }
    }
    
    /// Create an authentication error with message
    pub fn auth(message: impl Into<String>) -> Self {
        Self::AuthError {
            message: message.into(),
        }
    }
    
    /// Create a resource exhausted error
    pub fn resource_exhausted(resource: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
            message: message.into(),
        }
    }
    
    /// Create a health check error
    pub fn health_check(message: impl Into<String>) -> Self {
        Self::InternalError(anyhow::anyhow!("Health check failed: {}", message.into()))
    }
    
    /// Create a shutdown error
    pub fn shutdown(message: impl Into<String>) -> Self {
        Self::InternalError(anyhow::anyhow!("Shutdown failed: {}", message.into()))
    }
    
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            // Validation and configuration errors are not retryable
            Self::ValidationError { .. } | Self::ConfigError { .. } => false,
            
            // Authentication errors are not retryable
            Self::AuthError { .. } => false,
            
            // Circuit breaker errors are not retryable (circuit needs to close first)
            Self::CircuitBreakerOpen { .. } => false,
            
            // Network and resource errors are typically retryable
            Self::RedisError(_) | Self::KafkaError(_) | Self::TimeoutError { .. } => true,
            
            // Publisher errors depend on the specific error
            Self::PublisherError { .. } => true,
            
            // Buffer overflow might be retryable after some time
            Self::BufferOverflow { .. } => true,
            
            // Audit errors are retryable
            Self::AuditError { .. } => true,
            
            // Serialization errors are not retryable
            Self::SerializationError(_) => false,
            
            // Resource exhaustion might be retryable
            Self::ResourceExhausted { .. } => true,
            
            // Internal errors might be retryable depending on the underlying error
            Self::InternalError(_) => true,
        }
    }
    
    /// Get the error category for metrics and logging
    pub fn category(&self) -> &'static str {
        match self {
            Self::ValidationError { .. } => "validation",
            Self::PublisherError { .. } => "publisher",
            Self::RedisError(_) => "redis",
            Self::KafkaError(_) => "kafka",
            Self::BufferOverflow { .. } => "buffer",
            Self::ConfigError { .. } => "config",
            Self::AuditError { .. } => "audit",
            Self::SerializationError(_) => "serialization",
            Self::TimeoutError { .. } => "timeout",
            Self::CircuitBreakerOpen { .. } => "circuit_breaker",
            Self::AuthError { .. } => "auth",
            Self::ResourceExhausted { .. } => "resource",
            Self::InternalError(_) => "internal",
        }
    }
}

/// Result type for signal emission operations
pub type Result<T> = std::result::Result<T, SignalEmissionError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let validation_err = SignalEmissionError::validation("Invalid signal");
        assert!(matches!(validation_err, SignalEmissionError::ValidationError { .. }));
        
        let publisher_err = SignalEmissionError::publisher("redis", "Connection failed");
        assert!(matches!(publisher_err, SignalEmissionError::PublisherError { .. }));
        
        let config_err = SignalEmissionError::config("Missing required field");
        assert!(matches!(config_err, SignalEmissionError::ConfigError { .. }));
    }
    
    #[test]
    fn test_error_retryability() {
        assert!(!SignalEmissionError::validation("test").is_retryable());
        assert!(!SignalEmissionError::config("test").is_retryable());
        assert!(!SignalEmissionError::auth("test").is_retryable());
        assert!(!SignalEmissionError::circuit_breaker("test").is_retryable());
        
        assert!(SignalEmissionError::publisher("redis", "test").is_retryable());
        assert!(SignalEmissionError::timeout(1000).is_retryable());
        assert!(SignalEmissionError::audit("test").is_retryable());
    }
    
    #[test]
    fn test_error_categories() {
        assert_eq!(SignalEmissionError::validation("test").category(), "validation");
        assert_eq!(SignalEmissionError::publisher("redis", "test").category(), "publisher");
        assert_eq!(SignalEmissionError::config("test").category(), "config");
        assert_eq!(SignalEmissionError::timeout(1000).category(), "timeout");
    }
}