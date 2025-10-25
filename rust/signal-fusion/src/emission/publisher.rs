//! Publisher trait and common types for signal emission backends
//! 
//! This module defines the core abstraction for publishing trading signals
//! to different message bus backends (Redis Streams, Kafka topics).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::TradingSignal;
use super::Result;

/// Result of a signal publication operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishResult {
    /// Unique identifier for the published message
    pub message_id: String,
    
    /// Backend that handled the publication
    pub backend: String,
    
    /// Timestamp when the message was published (Unix timestamp in milliseconds)
    pub published_at: i64,
    
    /// Latency of the publish operation in milliseconds
    pub latency_ms: u64,
    
    /// Number of retry attempts made
    pub retry_count: u32,
    
    /// Additional metadata from the backend
    pub metadata: HashMap<String, String>,
}

impl PublishResult {
    /// Create a new publish result
    pub fn new(
        message_id: impl Into<String>,
        backend: impl Into<String>,
        published_at: i64,
        latency_ms: u64,
    ) -> Self {
        Self {
            message_id: message_id.into(),
            backend: backend.into(),
            published_at,
            latency_ms,
            retry_count: 0,
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to the result
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Set the retry count
    pub fn with_retry_count(mut self, retry_count: u32) -> Self {
        self.retry_count = retry_count;
        self
    }
}

/// Health status of a publisher backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall health status
    pub status: HealthLevel,
    
    /// Timestamp of the health check (Unix timestamp in milliseconds)
    pub checked_at: i64,
    
    /// Response time of the health check in milliseconds
    pub response_time_ms: u64,
    
    /// Error message if unhealthy
    pub error_message: Option<String>,
    
    /// Additional health metrics
    pub metrics: HashMap<String, String>,
}

/// Health level enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthLevel {
    /// Service is fully operational
    Healthy,
    
    /// Service is operational but with degraded performance
    Degraded,
    
    /// Service is not operational
    Unhealthy,
}

impl HealthStatus {
    /// Create a healthy status
    pub fn healthy(response_time_ms: u64) -> Self {
        Self {
            status: HealthLevel::Healthy,
            checked_at: chrono::Utc::now().timestamp_millis(),
            response_time_ms,
            error_message: None,
            metrics: HashMap::new(),
        }
    }
    
    /// Create a degraded status
    pub fn degraded(response_time_ms: u64, message: impl Into<String>) -> Self {
        Self {
            status: HealthLevel::Degraded,
            checked_at: chrono::Utc::now().timestamp_millis(),
            response_time_ms,
            error_message: Some(message.into()),
            metrics: HashMap::new(),
        }
    }
    
    /// Create an unhealthy status
    pub fn unhealthy(error: impl Into<String>) -> Self {
        Self {
            status: HealthLevel::Unhealthy,
            checked_at: chrono::Utc::now().timestamp_millis(),
            response_time_ms: 0,
            error_message: Some(error.into()),
            metrics: HashMap::new(),
        }
    }
    
    /// Add a metric to the health status
    pub fn with_metric(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metrics.insert(key.into(), value.into());
        self
    }
    
    /// Check if the status indicates the service is operational
    pub fn is_operational(&self) -> bool {
        matches!(self.status, HealthLevel::Healthy | HealthLevel::Degraded)
    }
}

/// Configuration for publisher retry behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    
    /// Base delay between retries
    pub base_delay: Duration,
    
    /// Maximum delay between retries
    pub max_delay: Duration,
    
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
    
    /// Whether to add jitter to retry delays
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Calculate the delay for a given retry attempt
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        
        let delay_ms = self.base_delay.as_millis() as f64
            * self.backoff_multiplier.powi((attempt - 1) as i32);
        
        let delay = Duration::from_millis(delay_ms as u64).min(self.max_delay);
        
        if self.jitter {
            // Add up to 25% jitter
            let jitter_factor = 1.0 + (rand::random::<f64>() - 0.5) * 0.5;
            Duration::from_millis((delay.as_millis() as f64 * jitter_factor) as u64)
        } else {
            delay
        }
    }
    
    /// Check if more retries are allowed
    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_attempts
    }
}

/// Trait for signal publisher backends
/// 
/// This trait abstracts the different message bus backends (Redis, Kafka)
/// and provides a common interface for publishing trading signals.
#[async_trait]
pub trait PublisherTrait: Send + Sync {
    /// Publish a trading signal to the message bus
    /// 
    /// # Arguments
    /// * `signal` - The trading signal to publish
    /// 
    /// # Returns
    /// * `Ok(PublishResult)` - Success with publication details
    /// * `Err(SignalEmissionError)` - Publication failed
    async fn publish(&mut self, signal: &TradingSignal) -> Result<PublishResult>;
    
    /// Check the health of the publisher backend
    /// 
    /// # Returns
    /// * `Ok(HealthStatus)` - Health check completed
    /// * `Err(SignalEmissionError)` - Health check failed
    async fn health_check(&self) -> Result<HealthStatus>;
    
    /// Get the backend name for identification
    fn backend_name(&self) -> &str;
    
    /// Get publisher-specific metrics
    /// 
    /// # Returns
    /// A map of metric names to values for monitoring
    fn get_metrics(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    
    /// Gracefully shutdown the publisher
    /// 
    /// This method should clean up resources, flush pending messages,
    /// and close connections.
    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Check if the publisher supports batching
    fn supports_batching(&self) -> bool {
        false
    }
    
    /// Publish multiple signals in a batch (if supported)
    /// 
    /// Default implementation falls back to individual publishes.
    /// Backends that support batching should override this method.
    async fn publish_batch(&mut self, signals: &[TradingSignal]) -> Result<Vec<PublishResult>> {
        let mut results = Vec::with_capacity(signals.len());
        
        for signal in signals {
            match self.publish(signal).await {
                Ok(result) => results.push(result),
                Err(e) => return Err(e),
            }
        }
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_publish_result_creation() {
        let result = PublishResult::new("msg-123", "redis", 1640995200000, 50)
            .with_metadata("stream", "trading_signals")
            .with_retry_count(1);
        
        assert_eq!(result.message_id, "msg-123");
        assert_eq!(result.backend, "redis");
        assert_eq!(result.published_at, 1640995200000);
        assert_eq!(result.latency_ms, 50);
        assert_eq!(result.retry_count, 1);
        assert_eq!(result.metadata.get("stream"), Some(&"trading_signals".to_string()));
    }
    
    #[test]
    fn test_health_status_creation() {
        let healthy = HealthStatus::healthy(25)
            .with_metric("connections", "5");
        assert_eq!(healthy.status, HealthLevel::Healthy);
        assert_eq!(healthy.response_time_ms, 25);
        assert!(healthy.error_message.is_none());
        assert!(healthy.is_operational());
        
        let degraded = HealthStatus::degraded(100, "High latency");
        assert_eq!(degraded.status, HealthLevel::Degraded);
        assert_eq!(degraded.response_time_ms, 100);
        assert!(degraded.error_message.is_some());
        assert!(degraded.is_operational());
        
        let unhealthy = HealthStatus::unhealthy("Connection failed");
        assert_eq!(unhealthy.status, HealthLevel::Unhealthy);
        assert!(unhealthy.error_message.is_some());
        assert!(!unhealthy.is_operational());
    }
    
    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.base_delay, Duration::from_millis(100));
        assert_eq!(config.backoff_multiplier, 2.0);
        assert!(config.jitter);
    }
    
    #[test]
    fn test_retry_config_delay_calculation() {
        let config = RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter: false,
        };
        
        assert_eq!(config.delay_for_attempt(0), Duration::ZERO);
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(400));
        
        assert!(config.should_retry(0));
        assert!(config.should_retry(1));
        assert!(config.should_retry(2));
        assert!(!config.should_retry(3));
    }
    
    #[test]
    fn test_retry_config_max_delay() {
        let config = RetryConfig {
            max_attempts: 10,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(500),
            backoff_multiplier: 2.0,
            jitter: false,
        };
        
        // Should be capped at max_delay
        let delay = config.delay_for_attempt(5);
        assert_eq!(delay, Duration::from_millis(500));
    }
}