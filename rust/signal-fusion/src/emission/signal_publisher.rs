//! Unified signal publisher that coordinates Redis and Kafka publishers
//! 
//! This module provides the main SignalPublisher struct that manages multiple
//! backend publishers and provides a unified interface for signal emission.
//!
//! # Quick Start
//!
//! ```rust
//! use signal_fusion::emission::{SignalPublisher, SignalPublisherConfig, PublisherBackend};
//! use signal_fusion::{TradingSignal, SignalSide};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create configuration
//! let config = SignalPublisherConfig {
//!     backend: PublisherBackend::Redis,
//!     enabled: true,
//!     batch_size: 100,
//!     flush_interval_ms: 1000,
//!     ..Default::default()
//! };
//!
//! // Create publisher
//! let mut publisher = SignalPublisher::new(config).await?;
//!
//! // Create a trading signal
//! let signal = TradingSignal {
//!     timestamp: chrono::Utc::now().timestamp_millis(),
//!     symbol: "BTCUSDT".to_string(),
//!     side: SignalSide::Buy,
//!     strength: 0.75,
//!     confidence: 0.85,
//!     correlation_id: "req_123".to_string(),
//!     feature_checksum: "abc123".to_string(),
//!     // ... other required fields
//! };
//!
//! // Publish signal
//! publisher.publish_signal(signal).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Configuration
//!
//! The publisher can be configured to use different backends:
//!
//! ```rust
//! use signal_fusion::emission::{SignalPublisherConfig, PublisherBackend};
//!
//! // Redis only
//! let redis_config = SignalPublisherConfig {
//!     backend: PublisherBackend::Redis,
//!     ..Default::default()
//! };
//!
//! // Kafka only  
//! let kafka_config = SignalPublisherConfig {
//!     backend: PublisherBackend::Kafka,
//!     ..Default::default()
//! };
//!
//! // Both Redis and Kafka
//! let both_config = SignalPublisherConfig {
//!     backend: PublisherBackend::Both,
//!     ..Default::default()
//! };
//!
//! // Disabled (for testing)
//! let disabled_config = SignalPublisherConfig {
//!     backend: PublisherBackend::None,
//!     ..Default::default()
//! };
//! ```
//!
//! # Batch Publishing
//!
//! For better performance, publish multiple signals in batches:
//!
//! ```rust
//! # use signal_fusion::emission::SignalPublisher;
//! # use signal_fusion::TradingSignal;
//! # async fn example(mut publisher: SignalPublisher, signals: Vec<TradingSignal>) -> Result<(), Box<dyn std::error::Error>> {
//! // Publish multiple signals at once
//! publisher.publish_batch(signals).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Health Monitoring
//!
//! Check the health of all configured backends:
//!
//! ```rust
//! # use signal_fusion::emission::SignalPublisher;
//! # async fn example(publisher: SignalPublisher) -> Result<(), Box<dyn std::error::Error>> {
//! let health = publisher.health_check().await;
//! match health.level {
//!     signal_fusion::emission::HealthLevel::Healthy => {
//!         println!("All systems operational");
//!     },
//!     signal_fusion::emission::HealthLevel::Degraded => {
//!         println!("Some issues detected: {}", health.message);
//!     },
//!     signal_fusion::emission::HealthLevel::Unhealthy => {
//!         println!("System unhealthy: {}", health.message);
//!     },
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Metrics Collection
//!
//! Get performance metrics from the publisher:
//!
//! ```rust
//! # use signal_fusion::emission::SignalPublisher;
//! # async fn example(publisher: SignalPublisher) -> Result<(), Box<dyn std::error::Error>> {
//! let metrics = publisher.get_metrics();
//! println!("Signals published: {}", metrics.signals_published_total);
//! println!("Validation errors: {}", metrics.validation_errors_total);
//! println!("Publisher errors: {}", metrics.publisher_errors_total);
//! println!("Buffer utilization: {:.1}%", metrics.buffer_utilization_percent);
//! # Ok(())
//! # }
//! ```
//!
//! # Error Handling
//!
//! The publisher provides detailed error information:
//!
//! ```rust
//! # use signal_fusion::emission::{SignalPublisher, SignalEmissionError};
//! # use signal_fusion::TradingSignal;
//! # async fn example(mut publisher: SignalPublisher, signal: TradingSignal) -> Result<(), Box<dyn std::error::Error>> {
//! match publisher.publish_signal(signal).await {
//!     Ok(_) => println!("Signal published successfully"),
//!     Err(SignalEmissionError::ValidationError { message }) => {
//!         println!("Signal validation failed: {}", message);
//!     },
//!     Err(SignalEmissionError::PublisherError { backend, message }) => {
//!         println!("Publisher error on {}: {}", backend, message);
//!     },
//!     Err(SignalEmissionError::CircuitBreakerOpen { reason }) => {
//!         println!("Circuit breaker open: {}", reason);
//!     },
//!     Err(e) => {
//!         println!("Other error: {}", e);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};
use serde::{Deserialize, Serialize};

use crate::TradingSignal;
use super::{
    Result, SignalEmissionError,
    PublisherTrait,
    RedisPublisher, RedisConfig,
    KafkaPublisher, KafkaConfig,
    SignalBuffer, BufferConfig,
    AuditLogger, AuditConfig,
    SignalValidator, ValidationConfig,
    ValidationErrorEvent, SignalEmissionEvent,
};
use super::publisher::{PublishResult, HealthStatus, HealthLevel};

/// Backend selection for signal publishing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PublisherBackend {
    /// Publish only to Redis Streams
    Redis,
    /// Publish only to Kafka topics
    Kafka,
    /// Publish to both Redis and Kafka
    Both,
    /// No publishing (for testing)
    None,
}

impl std::fmt::Display for PublisherBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublisherBackend::Redis => write!(f, "redis"),
            PublisherBackend::Kafka => write!(f, "kafka"),
            PublisherBackend::Both => write!(f, "both"),
            PublisherBackend::None => write!(f, "none"),
        }
    }
}

impl std::str::FromStr for PublisherBackend {
    type Err = SignalEmissionError;
    
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "redis" => Ok(PublisherBackend::Redis),
            "kafka" => Ok(PublisherBackend::Kafka),
            "both" => Ok(PublisherBackend::Both),
            "none" => Ok(PublisherBackend::None),
            _ => Err(SignalEmissionError::config(format!("Invalid publisher backend: {}", s))),
        }
    }
}

/// Configuration for the unified signal publisher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalPublisherConfig {
    /// Backend selection for publishing
    pub backend: PublisherBackend,
    
    /// Whether signal emission is enabled
    pub enabled: bool,
    
    /// Redis configuration (required if backend includes Redis)
    pub redis: Option<RedisConfig>,
    
    /// Kafka configuration (required if backend includes Kafka)
    pub kafka: Option<KafkaConfig>,
    
    /// Buffer configuration for local buffering during outages
    pub buffer: BufferConfig,
    
    /// Audit logging configuration
    pub audit: AuditConfig,
    
    /// Signal validation configuration
    pub validation: ValidationConfig,
    
    /// Timeout for publish operations in milliseconds
    pub publish_timeout_ms: u64,
    
    /// Whether to fail fast on publisher errors or continue with available backends
    pub fail_fast: bool,
    
    /// Maximum number of concurrent publish operations
    pub max_concurrent_publishes: usize,
}

impl Default for SignalPublisherConfig {
    fn default() -> Self {
        Self {
            backend: PublisherBackend::None,
            enabled: true,
            redis: None,
            kafka: None,
            buffer: BufferConfig::default(),
            audit: AuditConfig::default(),
            validation: ValidationConfig::default(),
            publish_timeout_ms: 5000, // 5 seconds
            fail_fast: false,
            max_concurrent_publishes: 100,
        }
    }
}

impl SignalPublisherConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Check that required backend configurations are present
        match self.backend {
            PublisherBackend::Redis => {
                if self.redis.is_none() {
                    return Err(SignalEmissionError::config(
                        "Redis configuration required when backend is Redis"
                    ));
                }
            }
            PublisherBackend::Kafka => {
                if self.kafka.is_none() {
                    return Err(SignalEmissionError::config(
                        "Kafka configuration required when backend is Kafka"
                    ));
                }
            }
            PublisherBackend::Both => {
                if self.redis.is_none() {
                    return Err(SignalEmissionError::config(
                        "Redis configuration required when backend includes Redis"
                    ));
                }
                if self.kafka.is_none() {
                    return Err(SignalEmissionError::config(
                        "Kafka configuration required when backend includes Kafka"
                    ));
                }
            }
            PublisherBackend::None => {
                // No backend configuration required for testing mode
            }
        }
        
        // Validate timeout
        if self.publish_timeout_ms == 0 {
            return Err(SignalEmissionError::config(
                "Publish timeout must be greater than 0"
            ));
        }
        
        // Validate concurrency limit
        if self.max_concurrent_publishes == 0 {
            return Err(SignalEmissionError::config(
                "Max concurrent publishes must be greater than 0"
            ));
        }
        
        Ok(())
    }
}

/// Publisher instance information
struct PublisherInstance {
    name: String,
    publisher: Box<dyn PublisherTrait + Send + Sync>,
    last_health_check: Option<Instant>,
    last_health_status: Option<HealthStatus>,
}

/// Metrics for the signal publisher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalPublisherMetrics {
    /// Total number of signals published successfully
    pub signals_published_total: u64,
    
    /// Total number of publish errors
    pub publish_errors_total: u64,
    
    /// Total number of validation errors
    pub validation_errors_total: u64,
    
    /// Total number of signals buffered
    pub signals_buffered_total: u64,
    
    /// Current buffer size
    pub buffer_size: usize,
    
    /// Publisher-specific metrics
    pub publisher_metrics: HashMap<String, serde_json::Value>,
    
    /// Average publish latency in milliseconds
    pub avg_publish_latency_ms: f64,
    
    /// Health status of each publisher
    pub publisher_health: HashMap<String, HealthStatus>,
}

impl Default for SignalPublisherMetrics {
    fn default() -> Self {
        Self {
            signals_published_total: 0,
            publish_errors_total: 0,
            validation_errors_total: 0,
            signals_buffered_total: 0,
            buffer_size: 0,
            publisher_metrics: HashMap::new(),
            avg_publish_latency_ms: 0.0,
            publisher_health: HashMap::new(),
        }
    }
}

/// Unified signal publisher that coordinates multiple backends
pub struct SignalPublisher {
    config: SignalPublisherConfig,
    publishers: Arc<RwLock<HashMap<String, PublisherInstance>>>,
    buffer: Arc<RwLock<SignalBuffer>>,
    validator: SignalValidator,
    audit_logger: AuditLogger,
    metrics: Arc<RwLock<SignalPublisherMetrics>>,
    publish_semaphore: Arc<tokio::sync::Semaphore>,
}

impl SignalPublisher {
    /// Create a new SignalPublisher with the given configuration
    pub async fn new(config: SignalPublisherConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        info!("Initializing SignalPublisher with backend: {}", config.backend);
        
        // Initialize components
        let buffer = SignalBuffer::new(config.buffer.clone());
        let validator = SignalValidator::new();
        let audit_logger = AuditLogger::new(config.audit.clone()).await?;
        
        let mut publisher = Self {
            config: config.clone(),
            publishers: Arc::new(RwLock::new(HashMap::new())),
            buffer: Arc::new(RwLock::new(buffer)),
            validator,
            audit_logger,
            metrics: Arc::new(RwLock::new(SignalPublisherMetrics::default())),
            publish_semaphore: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_publishes)),
        };
        
        // Initialize publishers based on backend configuration
        publisher.initialize_publishers().await?;
        
        info!("SignalPublisher initialized successfully");
        Ok(publisher)
    }
    
    /// Initialize publishers based on configuration
    async fn initialize_publishers(&mut self) -> Result<()> {
        let mut publishers = self.publishers.write().await;
        
        match &self.config.backend {
            PublisherBackend::Redis => {
                if let Some(redis_config) = &self.config.redis {
                    let redis_publisher = RedisPublisher::new(redis_config.clone()).await?;
                    publishers.insert("redis".to_string(), PublisherInstance {
                        name: "redis".to_string(),
                        publisher: Box::new(redis_publisher),
                        last_health_check: None,
                        last_health_status: None,
                    });
                    info!("Redis publisher initialized");
                }
            }
            PublisherBackend::Kafka => {
                if let Some(kafka_config) = &self.config.kafka {
                    let kafka_publisher = KafkaPublisher::new(kafka_config.clone()).await?;
                    publishers.insert("kafka".to_string(), PublisherInstance {
                        name: "kafka".to_string(),
                        publisher: Box::new(kafka_publisher),
                        last_health_check: None,
                        last_health_status: None,
                    });
                    info!("Kafka publisher initialized");
                }
            }
            PublisherBackend::Both => {
                if let Some(redis_config) = &self.config.redis {
                    let redis_publisher = RedisPublisher::new(redis_config.clone()).await?;
                    publishers.insert("redis".to_string(), PublisherInstance {
                        name: "redis".to_string(),
                        publisher: Box::new(redis_publisher),
                        last_health_check: None,
                        last_health_status: None,
                    });
                    info!("Redis publisher initialized");
                }
                
                if let Some(kafka_config) = &self.config.kafka {
                    let kafka_publisher = KafkaPublisher::new(kafka_config.clone()).await?;
                    publishers.insert("kafka".to_string(), PublisherInstance {
                        name: "kafka".to_string(),
                        publisher: Box::new(kafka_publisher),
                        last_health_check: None,
                        last_health_status: None,
                    });
                    info!("Kafka publisher initialized");
                }
            }
            PublisherBackend::None => {
                info!("No publishers initialized (testing mode)");
            }
        }
        
        Ok(())
    }
    
    /// Publish a signal to all configured backends
    pub async fn publish_signal(&self, signal: TradingSignal) -> Result<Vec<PublishResult>> {
        if !self.config.enabled {
            debug!("Signal publishing disabled, skipping signal: {}", signal.to_compact_string());
            return Ok(vec![]);
        }
        
        let start_time = Instant::now();
        
        // Acquire semaphore permit for concurrency control
        let _permit = self.publish_semaphore.acquire().await
            .map_err(|_| SignalEmissionError::resource_exhausted("semaphore", "Failed to acquire publish permit"))?;
        
        // Validate signal
        if let Err(validation_error) = self.validator.validate(&signal) {
            let mut metrics = self.metrics.write().await;
            metrics.validation_errors_total += 1;
            
            // Log validation error - create a simple validation event
            let validation_event = ValidationErrorEvent::new(
                signal.correlation_id.clone(),
                serde_json::to_value(&signal).unwrap_or_default(),
                vec![], // We'll add a simple error detail
                "Signal validation failed during publish".to_string(),
            );
            self.audit_logger.log_validation_error(&validation_event).await.ok(); // Don't fail on audit errors
            
            return Err(validation_error);
        }
        
        debug!("Publishing signal: {}", signal.to_compact_string());
        
        // Get publishers
        let publishers = self.publishers.read().await;
        
        if publishers.is_empty() {
            debug!("No publishers configured, buffering signal");
            let mut buffer = self.buffer.write().await;
            buffer.push(signal.clone())?;
            
            let mut metrics = self.metrics.write().await;
            metrics.signals_buffered_total += 1;
            metrics.buffer_size = buffer.len();
            
            return Ok(vec![]);
        }
        
        // Drop the read lock and get write lock for publishing
        drop(publishers);
        let mut publishers = self.publishers.write().await;
        
        // Publish to all configured backends
        let mut results = Vec::new();
        let mut publish_errors = Vec::new();
        
        for (name, instance) in publishers.iter_mut() {
            match self.publish_to_backend_mut(name, &mut instance.publisher, &signal).await {
                Ok(result) => {
                    results.push(result);
                    debug!("Successfully published to {}: {}", name, signal.to_compact_string());
                }
                Err(error) => {
                    error!("Failed to publish to {}: {} - {}", name, error, signal.to_compact_string());
                    publish_errors.push((name.clone(), error));
                    
                    if self.config.fail_fast {
                        // Update metrics and return error immediately
                        let mut metrics = self.metrics.write().await;
                        metrics.publish_errors_total += 1;
                        
                        return Err(SignalEmissionError::publisher(
                            name,
                            format!("Publish failed in fail-fast mode: {}", publish_errors[0].1)
                        ));
                    }
                }
            }
        }
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        if !results.is_empty() {
            metrics.signals_published_total += 1;
            
            // Update average latency
            let latency_ms = start_time.elapsed().as_millis() as f64;
            metrics.avg_publish_latency_ms = 
                (metrics.avg_publish_latency_ms * (metrics.signals_published_total - 1) as f64 + latency_ms) 
                / metrics.signals_published_total as f64;
        }
        
        if !publish_errors.is_empty() {
            metrics.publish_errors_total += publish_errors.len() as u64;
        }
        
        // Log successful publication
        if !results.is_empty() {
            let backend_names = results.iter().map(|r| r.backend.clone()).collect::<Vec<_>>().join(",");
            let emission_event = SignalEmissionEvent::success(
                signal.correlation_id.clone(),
                signal.clone(),
                backend_names,
                start_time.elapsed().as_millis() as u64,
                0, // No retries in this implementation
            );
            self.audit_logger.log_signal_emission(&emission_event).await.ok(); // Don't fail on audit errors
        }
        
        // If we have some successes, return them; otherwise return the first error
        if !results.is_empty() {
            Ok(results)
        } else if !publish_errors.is_empty() {
            Err(SignalEmissionError::publisher(
                &publish_errors[0].0,
                format!("All publishers failed: {}", publish_errors[0].1)
            ))
        } else {
            // This shouldn't happen, but handle it gracefully
            Err(SignalEmissionError::publisher(
                "unknown",
                "No publishers available and no errors recorded"
            ))
        }
    }
    
    /// Publish to a specific backend with mutable access
    async fn publish_to_backend_mut(
        &self,
        _backend_name: &str,
        publisher: &mut Box<dyn PublisherTrait + Send + Sync>,
        signal: &TradingSignal,
    ) -> Result<PublishResult> {
        let timeout = Duration::from_millis(self.config.publish_timeout_ms);
        
        match tokio::time::timeout(timeout, publisher.publish(signal)).await {
            Ok(result) => result,
            Err(_) => Err(SignalEmissionError::timeout(self.config.publish_timeout_ms)),
        }
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> SignalPublisherMetrics {
        let mut metrics = self.metrics.read().await.clone();
        
        // Update buffer size
        let buffer = self.buffer.read().await;
        metrics.buffer_size = buffer.len();
        
        // Update publisher health status
        let publishers = self.publishers.read().await;
        for (name, instance) in publishers.iter() {
            if let Some(health) = &instance.last_health_status {
                metrics.publisher_health.insert(name.clone(), health.clone());
            }
        }
        
        metrics
    }
    
    /// Perform health check on all publishers
    pub async fn health_check(&self) -> HealthStatus {
        let mut overall_level = HealthLevel::Healthy;
        let mut component_healths = HashMap::new();
        let mut messages = Vec::new();
        
        // Check publishers
        let mut publishers = self.publishers.write().await;
        for (name, instance) in publishers.iter_mut() {
            let health = match instance.publisher.health_check().await {
                Ok(health) => {
                    instance.last_health_check = Some(Instant::now());
                    instance.last_health_status = Some(health.clone());
                    health
                }
                Err(error) => {
                    let health = HealthStatus::unhealthy(format!("Health check failed: {}", error));
                    instance.last_health_check = Some(Instant::now());
                    instance.last_health_status = Some(health.clone());
                    health
                }
            };
            
            component_healths.insert(format!("publisher_{}", name), health.clone());
            
            // Update overall health level
            match health.status {
                HealthLevel::Unhealthy => {
                    overall_level = HealthLevel::Unhealthy;
                    if let Some(msg) = &health.error_message {
                        messages.push(format!("{}: {}", name, msg));
                    }
                }
                HealthLevel::Degraded if overall_level == HealthLevel::Healthy => {
                    overall_level = HealthLevel::Degraded;
                    if let Some(msg) = &health.error_message {
                        messages.push(format!("{}: {}", name, msg));
                    }
                }
                _ => {}
            }
        }
        
        // Check buffer health
        let buffer = self.buffer.read().await;
        let buffer_metrics = buffer.metrics();
        let buffer_utilization = buffer_metrics.utilization;
        
        let buffer_utilization_pct = buffer_utilization * 100.0;
        let buffer_health = if buffer_utilization_pct > 90.0 {
            HealthLevel::Unhealthy
        } else if buffer_utilization_pct > 75.0 {
            HealthLevel::Degraded
        } else {
            HealthLevel::Healthy
        };
        
        if buffer_health != HealthLevel::Healthy {
            if overall_level == HealthLevel::Healthy {
                overall_level = buffer_health.clone();
            }
            messages.push(format!("Buffer utilization: {:.1}%", buffer_utilization_pct));
        }
        
        let buffer_health_status = match buffer_health {
            HealthLevel::Healthy => HealthStatus::healthy(0),
            HealthLevel::Degraded => HealthStatus::degraded(0, format!("Utilization: {:.1}%", buffer_utilization_pct)),
            HealthLevel::Unhealthy => HealthStatus::unhealthy(format!("Buffer overloaded: {:.1}%", buffer_utilization_pct)),
        }.with_metric("utilization_pct", format!("{:.1}", buffer_utilization_pct))
         .with_metric("current_size", buffer_metrics.current_size.to_string())
         .with_metric("max_size", buffer_metrics.max_size.to_string());
        
        component_healths.insert("buffer".to_string(), buffer_health_status);
        
        let overall_health = match overall_level {
            HealthLevel::Healthy => HealthStatus::healthy(0),
            HealthLevel::Degraded => HealthStatus::degraded(0, messages.join("; ")),
            HealthLevel::Unhealthy => HealthStatus::unhealthy(messages.join("; ")),
        };
        
        // Add component health details as metrics
        let mut final_health = overall_health;
        for (name, health) in component_healths {
            final_health = final_health.with_metric(
                format!("{}_status", name),
                format!("{:?}", health.status)
            );
        }
        
        final_health
    }
    
    /// Shutdown the publisher and clean up resources
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down SignalPublisher");
        
        // Shutdown publishers
        let mut publishers = self.publishers.write().await;
        for (name, instance) in publishers.iter_mut() {
            if let Err(error) = instance.publisher.shutdown().await {
                warn!("Error shutting down publisher {}: {}", name, error);
            }
        }
        publishers.clear();
        
        // Persist buffer if configured
        let mut buffer = self.buffer.write().await;
        if let Err(error) = buffer.persist() {
            warn!("Error persisting buffer during shutdown: {}", error);
        }
        
        info!("SignalPublisher shutdown complete");
        Ok(())
    }
    
    /// Get the current configuration
    pub fn get_config(&self) -> &SignalPublisherConfig {
        &self.config
    }
    
    /// Check if publishing is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
    
    /// Get the configured backend
    pub fn get_backend(&self) -> &PublisherBackend {
        &self.config.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SignalComponents, FusionWeights, SignalSide};
    
    fn create_test_signal() -> TradingSignal {
        let now = chrono::Utc::now().timestamp();
        TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            SignalComponents {
                s_ldc: 0.5,
                s_mr: 0.3,
                s_tsmom: 0.2,
            },
            FusionWeights {
                w_ldc: 0.5,
                w_mr: 0.3,
                w_tsmom: 0.2,
            },
            "v1.0".to_string(),
            "test-correlation-123".to_string(),
            "test-checksum-abc".to_string(),
            50,
        )
    }
    
    #[test]
    fn test_publisher_backend_parsing() {
        assert_eq!("redis".parse::<PublisherBackend>().unwrap(), PublisherBackend::Redis);
        assert_eq!("kafka".parse::<PublisherBackend>().unwrap(), PublisherBackend::Kafka);
        assert_eq!("both".parse::<PublisherBackend>().unwrap(), PublisherBackend::Both);
        assert_eq!("none".parse::<PublisherBackend>().unwrap(), PublisherBackend::None);
        
        assert!("invalid".parse::<PublisherBackend>().is_err());
    }
    
    #[test]
    fn test_publisher_backend_display() {
        assert_eq!(PublisherBackend::Redis.to_string(), "redis");
        assert_eq!(PublisherBackend::Kafka.to_string(), "kafka");
        assert_eq!(PublisherBackend::Both.to_string(), "both");
        assert_eq!(PublisherBackend::None.to_string(), "none");
    }
    
    #[test]
    fn test_config_validation() {
        // Valid config with no backend
        let config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
        
        // Invalid config - Redis backend without Redis config
        let config = SignalPublisherConfig {
            backend: PublisherBackend::Redis,
            redis: None,
            ..Default::default()
        };
        assert!(config.validate().is_err());
        
        // Invalid config - zero timeout
        let config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            publish_timeout_ms: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
        
        // Invalid config - zero concurrency
        let config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            max_concurrent_publishes: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
    
    #[tokio::test]
    async fn test_signal_publisher_creation_none_backend() {
        let config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            ..Default::default()
        };
        
        let publisher = SignalPublisher::new(config).await;
        assert!(publisher.is_ok());
        
        let publisher = publisher.unwrap();
        assert_eq!(publisher.get_backend(), &PublisherBackend::None);
        assert!(publisher.is_enabled());
    }
    
    #[tokio::test]
    async fn test_signal_publisher_disabled() {
        let config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            enabled: false,
            ..Default::default()
        };
        
        let publisher = SignalPublisher::new(config).await.unwrap();
        let signal = create_test_signal();
        
        let results = publisher.publish_signal(signal).await.unwrap();
        assert!(results.is_empty());
    }
    
    #[tokio::test]
    async fn test_signal_publisher_metrics() {
        let config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            ..Default::default()
        };
        
        let publisher = SignalPublisher::new(config).await.unwrap();
        let metrics = publisher.get_metrics().await;
        
        assert_eq!(metrics.signals_published_total, 0);
        assert_eq!(metrics.publish_errors_total, 0);
        assert_eq!(metrics.validation_errors_total, 0);
    }
    
    #[tokio::test]
    async fn test_signal_publisher_health_check() {
        let config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            ..Default::default()
        };
        
        let publisher = SignalPublisher::new(config).await.unwrap();
        let health = publisher.health_check().await;
        
        // Should be healthy with no publishers
        assert_eq!(health.status, HealthLevel::Healthy);
    }
    
    #[tokio::test]
    async fn test_signal_publisher_shutdown() {
        let config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            ..Default::default()
        };
        
        let publisher = SignalPublisher::new(config).await.unwrap();
        let result = publisher.shutdown().await;
        
        assert!(result.is_ok());
    }
}