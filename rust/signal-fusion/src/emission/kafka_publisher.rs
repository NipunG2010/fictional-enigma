//! Kafka publisher implementation for trading signal emission
//! 
//! This module provides a Kafka producer that publishes trading signals to Kafka topics
//! with configurable partitioning strategies, delivery confirmation, and resilience features.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord, Producer};
use rdkafka::util::Timeout;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::TradingSignal;
use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use super::publisher::{PublisherTrait, PublishResult, HealthStatus, HealthLevel};
use super::{SignalEmissionError, Result};

/// Kafka configuration for signal publishing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConfig {
    /// Kafka broker addresses (comma-separated)
    pub brokers: String,
    
    /// Topic name for trading signals
    pub topic: String,
    
    /// Producer configuration options
    pub producer: KafkaProducerConfig,
    
    /// Partitioning strategy configuration
    pub partitioning: PartitioningConfig,
    
    /// Health check configuration
    pub health: KafkaHealthConfig,
    
    /// Security configuration
    pub security: Option<KafkaSecurityConfig>,
}

impl Default for KafkaConfig {
    fn default() -> Self {
        Self {
            brokers: "localhost:9092".to_string(),
            topic: "trading_signals".to_string(),
            producer: KafkaProducerConfig::default(),
            partitioning: PartitioningConfig::default(),
            health: KafkaHealthConfig::default(),
            security: None,
        }
    }
}

/// Kafka producer-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaProducerConfig {
    /// Client ID for the producer
    pub client_id: String,
    
    /// Acknowledgment level (0, 1, or "all")
    pub acks: String,
    
    /// Request timeout in milliseconds
    pub request_timeout_ms: u32,
    
    /// Delivery timeout in milliseconds
    pub delivery_timeout_ms: u32,
    
    /// Batch size for batching messages
    pub batch_size: u32,
    
    /// Linger time in milliseconds for batching
    pub linger_ms: u32,
    
    /// Compression type (none, gzip, snappy, lz4, zstd)
    pub compression_type: String,
    
    /// Maximum in-flight requests per connection
    pub max_in_flight_requests_per_connection: u32,
    
    /// Enable idempotent producer
    pub enable_idempotence: bool,
    
    /// Retry configuration
    pub retries: u32,
    
    /// Retry backoff in milliseconds
    pub retry_backoff_ms: u32,
}

impl Default for KafkaProducerConfig {
    fn default() -> Self {
        Self {
            client_id: "signal-fusion-producer".to_string(),
            acks: "all".to_string(),
            request_timeout_ms: 30000,
            delivery_timeout_ms: 120000,
            batch_size: 16384,
            linger_ms: 5,
            compression_type: "snappy".to_string(),
            max_in_flight_requests_per_connection: 5,
            enable_idempotence: true,
            retries: 2147483647, // Max retries
            retry_backoff_ms: 100,
        }
    }
}

/// Partitioning strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitioningConfig {
    /// Partitioning strategy
    pub strategy: PartitioningStrategy,
    
    /// Custom partition key template (for custom strategy)
    pub custom_key_template: Option<String>,
    
    /// Number of partitions (for validation)
    pub partition_count: Option<u32>,
}

impl Default for PartitioningConfig {
    fn default() -> Self {
        Self {
            strategy: PartitioningStrategy::Symbol,
            custom_key_template: None,
            partition_count: None,
        }
    }
}

/// Available partitioning strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PartitioningStrategy {
    /// Partition by symbol (ensures all signals for a symbol go to same partition)
    Symbol,
    
    /// Round-robin partitioning
    RoundRobin,
    
    /// Custom key based on template
    Custom,
    
    /// Random partitioning
    Random,
}

/// Kafka health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaHealthConfig {
    /// Health check timeout in milliseconds
    pub timeout_ms: u64,
    
    /// Health check interval in seconds
    pub check_interval_sec: u64,
    
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
}

impl Default for KafkaHealthConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 5000,
            check_interval_sec: 30,
            failure_threshold: 3,
        }
    }
}

/// Kafka security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaSecurityConfig {
    /// Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL)
    pub protocol: String,
    
    /// SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512, GSSAPI)
    pub sasl_mechanism: Option<String>,
    
    /// SASL username
    pub sasl_username: Option<String>,
    
    /// SASL password
    pub sasl_password: Option<String>,
    
    /// SSL configuration
    pub ssl: Option<KafkaSslConfig>,
}

/// Kafka SSL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaSslConfig {
    /// Path to CA certificate file
    pub ca_location: Option<String>,
    
    /// Path to client certificate file
    pub certificate_location: Option<String>,
    
    /// Path to client private key file
    pub key_location: Option<String>,
    
    /// Private key password
    pub key_password: Option<String>,
}

/// Kafka publisher implementation
pub struct KafkaPublisher {
    /// Kafka producer instance
    producer: FutureProducer,
    
    /// Configuration
    config: KafkaConfig,
    
    /// Circuit breaker for resilience
    circuit_breaker: Arc<RwLock<CircuitBreaker>>,
    
    /// Partitioning state
    partitioning_state: Arc<RwLock<PartitioningState>>,
    
    /// Health check state
    health_state: Arc<RwLock<HealthState>>,
    
    /// Metrics
    metrics: Arc<RwLock<KafkaMetrics>>,
}

/// Internal partitioning state
#[derive(Debug)]
struct PartitioningState {
    round_robin_counter: u32,
    partition_count: Option<u32>,
}

/// Health check state
#[derive(Debug)]
struct HealthState {
    last_check: Option<Instant>,
    consecutive_failures: u32,
    last_error: Option<String>,
}

/// Kafka publisher metrics
#[derive(Debug, Default)]
pub struct KafkaMetrics {
    /// Total messages published
    pub messages_published: u64,
    
    /// Total publish errors
    pub publish_errors: u64,
    
    /// Total delivery confirmations received
    pub delivery_confirmations: u64,
    
    /// Total delivery failures
    pub delivery_failures: u64,
    
    /// Average publish latency in milliseconds
    pub avg_publish_latency_ms: f64,
    
    /// Circuit breaker state changes
    pub circuit_breaker_state_changes: u64,
    
    /// Health check failures
    pub health_check_failures: u64,
    
    /// Total batch operations
    pub batch_operations: u64,
    
    /// Total signals in batches
    pub batch_signals_total: u64,
    
    /// Average batch size
    pub avg_batch_size: f64,
}

impl KafkaPublisher {
    /// Create a new Kafka publisher with the given configuration
    pub async fn new(config: KafkaConfig) -> Result<Self> {
        // Validate configuration
        Self::validate_compression_type(&config.producer.compression_type)?;
        
        // Build Kafka client configuration
        let mut client_config = ClientConfig::new();
        
        // Basic configuration
        client_config
            .set("bootstrap.servers", &config.brokers)
            .set("client.id", &config.producer.client_id)
            .set("acks", &config.producer.acks)
            .set("request.timeout.ms", &config.producer.request_timeout_ms.to_string())
            .set("delivery.timeout.ms", &config.producer.delivery_timeout_ms.to_string())
            .set("batch.size", &config.producer.batch_size.to_string())
            .set("linger.ms", &config.producer.linger_ms.to_string())
            .set("compression.type", &config.producer.compression_type)
            .set("max.in.flight.requests.per.connection", &config.producer.max_in_flight_requests_per_connection.to_string())
            .set("enable.idempotence", &config.producer.enable_idempotence.to_string())
            .set("retries", &config.producer.retries.to_string())
            .set("retry.backoff.ms", &config.producer.retry_backoff_ms.to_string());
        
        // Security configuration
        if let Some(ref security) = config.security {
            client_config.set("security.protocol", &security.protocol);
            
            if let Some(ref mechanism) = security.sasl_mechanism {
                client_config.set("sasl.mechanism", mechanism);
            }
            
            if let Some(ref username) = security.sasl_username {
                client_config.set("sasl.username", username);
            }
            
            if let Some(ref password) = security.sasl_password {
                client_config.set("sasl.password", password);
            }
            
            if let Some(ref ssl) = security.ssl {
                if let Some(ref ca_location) = ssl.ca_location {
                    client_config.set("ssl.ca.location", ca_location);
                }
                
                if let Some(ref cert_location) = ssl.certificate_location {
                    client_config.set("ssl.certificate.location", cert_location);
                }
                
                if let Some(ref key_location) = ssl.key_location {
                    client_config.set("ssl.key.location", key_location);
                }
                
                if let Some(ref key_password) = ssl.key_password {
                    client_config.set("ssl.key.password", key_password);
                }
            }
        }
        
        // Create producer
        let producer: FutureProducer = client_config
            .create()
            .map_err(|e| SignalEmissionError::publisher("kafka", format!("Failed to create producer: {}", e)))?;
        
        // Create circuit breaker
        let circuit_breaker_config = CircuitBreakerConfig {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            success_threshold: 3,
            failure_window: Duration::from_secs(300),
            half_open_max_requests: 3,
        };
        let circuit_breaker = Arc::new(RwLock::new(CircuitBreaker::with_config(circuit_breaker_config)));
        
        // Initialize partitioning state
        let partitioning_state = Arc::new(RwLock::new(PartitioningState {
            round_robin_counter: 0,
            partition_count: config.partitioning.partition_count,
        }));
        
        // Initialize health state
        let health_state = Arc::new(RwLock::new(HealthState {
            last_check: None,
            consecutive_failures: 0,
            last_error: None,
        }));
        
        // Initialize metrics
        let metrics = Arc::new(RwLock::new(KafkaMetrics::default()));
        
        let publisher = Self {
            producer,
            config,
            circuit_breaker,
            partitioning_state,
            health_state,
            metrics,
        };
        
        info!(
            "Kafka publisher created: brokers={}, topic={}, compression={}",
            publisher.config.brokers, publisher.config.topic, publisher.config.producer.compression_type
        );
        
        Ok(publisher)
    }
    
    /// Get the partition key for a signal based on the configured strategy
    async fn get_partition_key(&self, signal: &TradingSignal) -> Result<Option<String>> {
        let partitioning_config = &self.config.partitioning;
        
        match partitioning_config.strategy {
            PartitioningStrategy::Symbol => {
                // Use symbol as partition key to ensure ordering per symbol
                Ok(Some(signal.symbol.clone()))
            }
            
            PartitioningStrategy::RoundRobin => {
                // No partition key for round-robin (let Kafka handle it)
                Ok(None)
            }
            
            PartitioningStrategy::Custom => {
                // Use custom key template
                if let Some(ref template) = partitioning_config.custom_key_template {
                    let key = template
                        .replace("{symbol}", &signal.symbol)
                        .replace("{side}", &signal.side.to_string())
                        .replace("{model_version}", &signal.model_version);
                    Ok(Some(key))
                } else {
                    Err(SignalEmissionError::config("Custom partitioning strategy requires custom_key_template"))
                }
            }
            
            PartitioningStrategy::Random => {
                // Generate random key
                let random_key = format!("random_{}", rand::random::<u32>());
                Ok(Some(random_key))
            }
        }
    }
    
    /// Get current metrics
    pub async fn get_kafka_metrics(&self) -> KafkaMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Reset metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = KafkaMetrics::default();
    }
    
    /// Get detailed metrics as a HashMap for monitoring systems
    pub async fn get_metrics_map(&self) -> HashMap<String, String> {
        let metrics = self.metrics.read().await;
        let circuit_breaker = self.circuit_breaker.read().await;
        let health_state = self.health_state.read().await;
        
        let mut map = HashMap::new();
        
        // Basic metrics
        map.insert("messages_published".to_string(), metrics.messages_published.to_string());
        map.insert("publish_errors".to_string(), metrics.publish_errors.to_string());
        map.insert("delivery_confirmations".to_string(), metrics.delivery_confirmations.to_string());
        map.insert("delivery_failures".to_string(), metrics.delivery_failures.to_string());
        map.insert("avg_publish_latency_ms".to_string(), format!("{:.2}", metrics.avg_publish_latency_ms));
        map.insert("health_check_failures".to_string(), metrics.health_check_failures.to_string());
        
        // Batch metrics
        map.insert("batch_operations".to_string(), metrics.batch_operations.to_string());
        map.insert("batch_signals_total".to_string(), metrics.batch_signals_total.to_string());
        map.insert("avg_batch_size".to_string(), format!("{:.2}", metrics.avg_batch_size));
        
        // Circuit breaker metrics
        let cb_metrics = circuit_breaker.metrics();
        map.insert("circuit_breaker_state".to_string(), format!("{:?}", cb_metrics.state));
        map.insert("circuit_breaker_failure_count".to_string(), cb_metrics.failure_count.to_string());
        map.insert("circuit_breaker_success_count".to_string(), cb_metrics.success_count.to_string());
        
        if let Some(seconds) = cb_metrics.last_failure_seconds_ago {
            map.insert("circuit_breaker_last_failure_seconds_ago".to_string(), seconds.to_string());
        }
        
        if let Some(seconds) = cb_metrics.time_until_recovery_seconds {
            map.insert("circuit_breaker_time_until_recovery_seconds".to_string(), seconds.to_string());
        }
        
        // Health state
        map.insert("consecutive_health_failures".to_string(), health_state.consecutive_failures.to_string());
        
        if let Some(ref error) = health_state.last_error {
            map.insert("last_health_error".to_string(), error.clone());
        }
        
        // Configuration info
        map.insert("kafka_brokers".to_string(), self.config.brokers.clone());
        map.insert("kafka_topic".to_string(), self.config.topic.clone());
        map.insert("compression_type".to_string(), self.config.producer.compression_type.clone());
        map.insert("partitioning_strategy".to_string(), format!("{:?}", self.config.partitioning.strategy));
        
        map
    }
    
    /// Get circuit breaker metrics
    pub async fn get_circuit_breaker_metrics(&self) -> super::circuit_breaker::CircuitBreakerMetrics {
        let circuit_breaker = self.circuit_breaker.read().await;
        circuit_breaker.metrics()
    }
    
    /// Manually reset the circuit breaker
    pub async fn reset_circuit_breaker(&self) {
        let mut circuit_breaker = self.circuit_breaker.write().await;
        circuit_breaker.reset();
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.circuit_breaker_state_changes += 1;
        
        info!("Kafka publisher circuit breaker manually reset");
    }
    
    /// Get producer statistics (if available)
    pub fn get_producer_statistics(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        
        // Add basic producer configuration as statistics
        stats.insert("client_id".to_string(), self.config.producer.client_id.clone());
        stats.insert("acks".to_string(), self.config.producer.acks.clone());
        stats.insert("request_timeout_ms".to_string(), self.config.producer.request_timeout_ms.to_string());
        stats.insert("delivery_timeout_ms".to_string(), self.config.producer.delivery_timeout_ms.to_string());
        stats.insert("batch_size".to_string(), self.config.producer.batch_size.to_string());
        stats.insert("linger_ms".to_string(), self.config.producer.linger_ms.to_string());
        stats.insert("max_in_flight_requests".to_string(), self.config.producer.max_in_flight_requests_per_connection.to_string());
        stats.insert("enable_idempotence".to_string(), self.config.producer.enable_idempotence.to_string());
        stats.insert("retries".to_string(), self.config.producer.retries.to_string());
        
        stats
    }
    
    /// Validate compression type configuration
    fn validate_compression_type(compression: &str) -> Result<()> {
        match compression.to_lowercase().as_str() {
            "none" | "gzip" | "snappy" | "lz4" | "zstd" => Ok(()),
            _ => Err(SignalEmissionError::config(
                format!("Unsupported compression type: {}. Supported types: none, gzip, snappy, lz4, zstd", compression)
            )),
        }
    }
    
    /// Publish multiple signals in a batch for improved throughput
    pub async fn publish_batch_signals(&mut self, signals: &[TradingSignal]) -> Result<Vec<PublishResult>> {
        if signals.is_empty() {
            return Ok(Vec::new());
        }
        
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(signals.len());
        
        // Check circuit breaker once for the entire batch
        {
            let mut circuit_breaker = self.circuit_breaker.write().await;
            if !circuit_breaker.is_request_allowed() {
                let mut metrics = self.metrics.write().await;
                metrics.publish_errors += signals.len() as u64;
                
                return Err(SignalEmissionError::circuit_breaker(
                    format!("Kafka circuit breaker is open: {:?}", circuit_breaker.state())
                ));
            }
        }
        
        debug!("Publishing batch of {} signals to Kafka", signals.len());
        
        // Process each signal individually but track as batch
        for signal in signals {
            match self.publish(signal).await {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    // If any signal fails, return the error
                    error!("Failed to publish signal in batch: symbol={}, error={}", signal.symbol, e);
                    return Err(e);
                }
            }
        }
        
        let batch_latency = start_time.elapsed();
        
        // Update batch metrics
        {
            let mut metrics = self.metrics.write().await;
            
            // Update batch-specific metrics
            metrics.batch_operations += 1;
            metrics.batch_signals_total += results.len() as u64;
            
            // Update average batch size
            metrics.avg_batch_size = metrics.batch_signals_total as f64 / metrics.batch_operations as f64;
        }
        
        info!("Published batch of {} signals to Kafka in {}ms", 
              results.len(), batch_latency.as_millis());
        
        Ok(results)
    }
}

impl KafkaPublisher {
    /// Internal method to publish with retry logic
    async fn publish_with_retry(&mut self, signal: &TradingSignal, attempt: u32) -> Result<PublishResult> {
        let start_time = Instant::now();
        
        // Check circuit breaker
        {
            let mut circuit_breaker = self.circuit_breaker.write().await;
            if !circuit_breaker.is_request_allowed() {
                let mut metrics = self.metrics.write().await;
                metrics.publish_errors += 1;
                
                return Err(SignalEmissionError::circuit_breaker(
                    format!("Kafka circuit breaker is open: {:?}", circuit_breaker.state())
                ));
            }
        }
        
        // Serialize signal to JSON
        let payload = signal.to_json()
            .map_err(|e| SignalEmissionError::publisher("kafka", format!("Failed to serialize signal: {}", e)))?;
        
        // Get partition key
        let partition_key = self.get_partition_key(signal).await?;
        
        // Create Kafka record - need to handle key lifetime properly
        let record_builder = FutureRecord::to(&self.config.topic)
            .payload(&payload)
            .timestamp(signal.timestamp * 1000); // Convert to milliseconds
        
        // Apply key if present
        let record = if let Some(ref key) = partition_key {
            record_builder.key(key)
        } else {
            record_builder
        };
        
        debug!(
            "Publishing signal to Kafka (attempt {}): topic={}, payload_size={}",
            attempt + 1, self.config.topic, payload.len()
        );
        
        // Publish message
        let delivery_result = self.producer
            .send(record, Timeout::After(Duration::from_millis(self.config.producer.request_timeout_ms as u64)))
            .await;
        
        let publish_latency = start_time.elapsed();
        
        // Handle result
        match delivery_result {
            Ok((partition, offset)) => {
                // Update metrics
                {
                    let mut metrics = self.metrics.write().await;
                    metrics.messages_published += 1;
                    metrics.delivery_confirmations += 1;
                    
                    // Update average latency
                    let latency_ms = publish_latency.as_millis() as f64;
                    if metrics.messages_published == 1 {
                        metrics.avg_publish_latency_ms = latency_ms;
                    } else {
                        metrics.avg_publish_latency_ms = 
                            (metrics.avg_publish_latency_ms * (metrics.messages_published - 1) as f64 + latency_ms) / 
                            metrics.messages_published as f64;
                    }
                }
                
                // Record success with circuit breaker
                {
                    let mut circuit_breaker = self.circuit_breaker.write().await;
                    circuit_breaker.record_success();
                }
                
                info!(
                    "Signal published to Kafka successfully: symbol={}, partition={}, offset={}, latency={}ms, attempt={}",
                    signal.symbol, partition, offset, publish_latency.as_millis(), attempt + 1
                );
                
                Ok(PublishResult::new(
                    format!("{}:{}", partition, offset),
                    "kafka",
                    chrono::Utc::now().timestamp_millis(),
                    publish_latency.as_millis() as u64,
                ).with_retry_count(attempt))
            }
            
            Err((kafka_error, _)) => {
                // Update metrics
                {
                    let mut metrics = self.metrics.write().await;
                    metrics.publish_errors += 1;
                    metrics.delivery_failures += 1;
                }
                
                let error_msg = format!("Kafka delivery failed: {}", kafka_error);
                
                // Check if we should retry
                let max_retries = 3; // Could be configurable
                let is_retryable = self.is_error_retryable(&kafka_error);
                
                if is_retryable && attempt < max_retries {
                    // Calculate exponential backoff delay
                    let base_delay = Duration::from_millis(self.config.producer.retry_backoff_ms as u64);
                    let delay = base_delay * 2_u32.pow(attempt);
                    let max_delay = Duration::from_secs(10);
                    let actual_delay = delay.min(max_delay);
                    
                    warn!(
                        "Kafka publish failed (attempt {}), retrying in {}ms: symbol={}, error={}",
                        attempt + 1, actual_delay.as_millis(), signal.symbol, error_msg
                    );
                    
                    // Wait before retry
                    tokio::time::sleep(actual_delay).await;
                    
                    // Retry the operation
                    return Box::pin(self.publish_with_retry(signal, attempt + 1)).await;
                } else {
                    // Record failure with circuit breaker
                    {
                        let mut circuit_breaker = self.circuit_breaker.write().await;
                        circuit_breaker.record_failure();
                    }
                    
                    error!(
                        "Failed to publish signal to Kafka after {} attempts: symbol={}, error={}, latency={}ms",
                        attempt + 1, signal.symbol, error_msg, publish_latency.as_millis()
                    );
                    
                    Err(SignalEmissionError::publisher("kafka", format!("{} (after {} attempts)", error_msg, attempt + 1)))
                }
            }
        }
    }
    
    /// Check if a Kafka error is retryable
    fn is_error_retryable(&self, error: &rdkafka::error::KafkaError) -> bool {
        use rdkafka::error::KafkaError;
        
        match error {
            // Network and timeout errors are retryable
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::NetworkException) |
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::RequestTimedOut) |
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::BrokerNotAvailable) |
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::LeaderNotAvailable) |
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::NotLeaderForPartition) => true,
            
            // Authentication and configuration errors are not retryable
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::SaslAuthenticationFailed) |
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::TopicAuthorizationFailed) |
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::UnknownTopicOrPartition) |
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::InvalidConfig) => false,
            
            // Message size errors are not retryable
            KafkaError::MessageProduction(rdkafka::error::RDKafkaErrorCode::MessageSizeTooLarge) => false,
            
            // Default to retryable for unknown errors
            _ => true,
        }
    }
    
    async fn health_check(&mut self) -> Result<HealthStatus> {
        let start_time = Instant::now();
        
        // Check if we need to perform health check
        let should_check = {
            let health_state = self.health_state.read().await;
            match health_state.last_check {
                Some(last_check) => {
                    start_time.duration_since(last_check).as_secs() >= self.config.health.check_interval_sec
                }
                None => true,
            }
        };
        
        if !should_check {
            let health_state = self.health_state.read().await;
            let status = if health_state.consecutive_failures >= self.config.health.failure_threshold {
                HealthLevel::Unhealthy
            } else {
                HealthLevel::Healthy
            };
            
            return Ok(HealthStatus {
                status,
                checked_at: health_state.last_check.map(|t| t.elapsed().as_millis() as i64).unwrap_or(0),
                response_time_ms: 0,
                error_message: health_state.last_error.clone(),
                metrics: HashMap::new(),
            });
        }
        
        // Perform actual health check by getting topic metadata
        let timeout = Duration::from_millis(self.config.health.timeout_ms);
        let metadata_result = tokio::task::spawn_blocking({
            let producer = self.producer.clone();
            let topic = self.config.topic.clone();
            move || producer.client().fetch_metadata(Some(&topic), rdkafka::util::Timeout::After(timeout))
        }).await;
        
        let check_duration = start_time.elapsed();
        
        let mut health_state = self.health_state.write().await;
        health_state.last_check = Some(start_time);
        
        match metadata_result {
            Ok(Ok(metadata)) => {
                // Check if topic exists
                let topic_exists = metadata.topics().iter()
                    .any(|topic| topic.name() == self.config.topic);
                
                if topic_exists {
                    health_state.consecutive_failures = 0;
                    health_state.last_error = None;
                    
                    debug!("Kafka health check passed: topic={}", self.config.topic);
                    
                    Ok(HealthStatus {
                        status: HealthLevel::Healthy,
                        checked_at: chrono::Utc::now().timestamp_millis(),
                        response_time_ms: check_duration.as_millis() as u64,
                        error_message: None,
                        metrics: HashMap::new(),
                    })
                } else {
                    health_state.consecutive_failures += 1;
                    let error_msg = format!("Topic '{}' not found", self.config.topic);
                    health_state.last_error = Some(error_msg.clone());
                    
                    let mut metrics = self.metrics.write().await;
                    metrics.health_check_failures += 1;
                    
                    warn!("Kafka health check failed: {}", error_msg);
                    
                    let status = if health_state.consecutive_failures >= self.config.health.failure_threshold {
                        HealthLevel::Unhealthy
                    } else {
                        HealthLevel::Degraded
                    };
                    
                    Ok(HealthStatus {
                        status,
                        checked_at: chrono::Utc::now().timestamp_millis(),
                        response_time_ms: check_duration.as_millis() as u64,
                        error_message: Some(error_msg),
                        metrics: HashMap::new(),
                    })
                }
            }
            
            Ok(Err(kafka_error)) => {
                health_state.consecutive_failures += 1;
                let error_msg = format!("Kafka metadata error: {}", kafka_error);
                health_state.last_error = Some(error_msg.clone());
                
                let mut metrics = self.metrics.write().await;
                metrics.health_check_failures += 1;
                
                warn!("Kafka health check failed: {}", error_msg);
                
                let status = if health_state.consecutive_failures >= self.config.health.failure_threshold {
                    HealthLevel::Unhealthy
                } else {
                    HealthLevel::Degraded
                };
                
                Ok(HealthStatus {
                    status,
                    checked_at: chrono::Utc::now().timestamp_millis(),
                    response_time_ms: check_duration.as_millis() as u64,
                    error_message: Some(error_msg),
                    metrics: HashMap::new(),
                })
            }
            
            Err(_) => {
                health_state.consecutive_failures += 1;
                let error_msg = format!("Kafka health check timeout after {}ms", self.config.health.timeout_ms);
                health_state.last_error = Some(error_msg.clone());
                
                let mut metrics = self.metrics.write().await;
                metrics.health_check_failures += 1;
                
                warn!("Kafka health check failed: {}", error_msg);
                
                let status = if health_state.consecutive_failures >= self.config.health.failure_threshold {
                    HealthLevel::Unhealthy
                } else {
                    HealthLevel::Degraded
                };
                
                Ok(HealthStatus {
                    status,
                    checked_at: chrono::Utc::now().timestamp_millis(),
                    response_time_ms: check_duration.as_millis() as u64,
                    error_message: Some(error_msg),
                    metrics: HashMap::new(),
                })
            }
        }
    }
    
    fn backend_name(&self) -> &str {
        "kafka"
    }
    
    fn supports_batching(&self) -> bool {
        true
    }
    
    async fn publish_batch(&mut self, signals: &[TradingSignal]) -> Result<Vec<PublishResult>> {
        self.publish_batch_signals(signals).await
    }
    
    fn get_metrics(&self) -> HashMap<String, String> {
        // Use blocking call since we can't make this async in the trait
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.get_metrics_map().await
            })
        })
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down Kafka publisher gracefully");
        
        // Set circuit breaker to open to prevent new requests
        {
            let mut circuit_breaker = self.circuit_breaker.write().await;
            // Force circuit breaker to open state by recording multiple failures
            for _ in 0..10 {
                circuit_breaker.record_failure();
            }
            info!("Circuit breaker set to open state for shutdown");
        }
        
        // Flush any pending messages with timeout
        let flush_timeout = Duration::from_secs(30);
        info!("Flushing pending messages with {}s timeout", flush_timeout.as_secs());
        
        match self.producer.flush(flush_timeout) {
            Ok(_) => {
                info!("Kafka producer flushed successfully during shutdown");
            }
            Err(e) => {
                warn!("Failed to flush Kafka producer during shutdown: {}", e);
                // Don't return error, continue with shutdown
            }
        }
        
        // Log final metrics
        let final_metrics = self.get_kafka_metrics().await;
        info!(
            "Kafka publisher shutdown complete - Final metrics: published={}, errors={}, avg_latency={:.2}ms, batches={}",
            final_metrics.messages_published,
            final_metrics.publish_errors,
            final_metrics.avg_publish_latency_ms,
            final_metrics.batch_operations
        );
        
        Ok(())
    }
}

#[async_trait]
impl PublisherTrait for KafkaPublisher {
    async fn publish(&mut self, signal: &TradingSignal) -> Result<PublishResult> {
        self.publish_with_retry(signal, 0).await
    }
    
    async fn health_check(&mut self) -> Result<HealthStatus> {
        let start_time = Instant::now();
        
        // Check if we need to perform health check
        let should_check = {
            let health_state = self.health_state.read().await;
            match health_state.last_check {
                Some(last_check) => {
                    start_time.duration_since(last_check).as_secs() >= self.config.health.check_interval_sec
                }
                None => true,
            }
        };
        
        if !should_check {
            let health_state = self.health_state.read().await;
            let status = if health_state.consecutive_failures >= self.config.health.failure_threshold {
                HealthLevel::Unhealthy
            } else {
                HealthLevel::Healthy
            };
            
            return Ok(HealthStatus {
                status,
                checked_at: health_state.last_check.map(|t| t.elapsed().as_millis() as i64).unwrap_or(0),
                response_time_ms: 0,
                error_message: health_state.last_error.clone(),
                metrics: HashMap::new(),
            });
        }
        
        // Perform actual health check by getting topic metadata
        let timeout = Duration::from_millis(self.config.health.timeout_ms);
        let metadata_result = tokio::task::spawn_blocking({
            let producer = self.producer.clone();
            let topic = self.config.topic.clone();
            move || producer.client().fetch_metadata(Some(&topic), rdkafka::util::Timeout::After(timeout))
        }).await;
        
        let check_duration = start_time.elapsed();
        
        let mut health_state = self.health_state.write().await;
        health_state.last_check = Some(start_time);
        
        match metadata_result {
            Ok(Ok(metadata)) => {
                // Check if topic exists
                let topic_exists = metadata.topics().iter()
                    .any(|topic| topic.name() == self.config.topic);
                
                if topic_exists {
                    health_state.consecutive_failures = 0;
                    health_state.last_error = None;
                    
                    debug!("Kafka health check passed: topic={}", self.config.topic);
                    
                    Ok(HealthStatus {
                        status: HealthLevel::Healthy,
                        checked_at: chrono::Utc::now().timestamp_millis(),
                        response_time_ms: check_duration.as_millis() as u64,
                        error_message: None,
                        metrics: HashMap::new(),
                    })
                } else {
                    health_state.consecutive_failures += 1;
                    let error_msg = format!("Topic '{}' not found", self.config.topic);
                    health_state.last_error = Some(error_msg.clone());
                    
                    let mut metrics = self.metrics.write().await;
                    metrics.health_check_failures += 1;
                    
                    warn!("Kafka health check failed: {}", error_msg);
                    
                    let status = if health_state.consecutive_failures >= self.config.health.failure_threshold {
                        HealthLevel::Unhealthy
                    } else {
                        HealthLevel::Degraded
                    };
                    
                    Ok(HealthStatus {
                        status,
                        checked_at: chrono::Utc::now().timestamp_millis(),
                        response_time_ms: check_duration.as_millis() as u64,
                        error_message: Some(error_msg),
                        metrics: HashMap::new(),
                    })
                }
            }
            
            Ok(Err(kafka_error)) => {
                health_state.consecutive_failures += 1;
                let error_msg = format!("Kafka metadata error: {}", kafka_error);
                health_state.last_error = Some(error_msg.clone());
                
                let mut metrics = self.metrics.write().await;
                metrics.health_check_failures += 1;
                
                warn!("Kafka health check failed: {}", error_msg);
                
                let status = if health_state.consecutive_failures >= self.config.health.failure_threshold {
                    HealthLevel::Unhealthy
                } else {
                    HealthLevel::Degraded
                };
                
                Ok(HealthStatus {
                    status,
                    checked_at: chrono::Utc::now().timestamp_millis(),
                    response_time_ms: check_duration.as_millis() as u64,
                    error_message: Some(error_msg),
                    metrics: HashMap::new(),
                })
            }
            
            Err(_) => {
                health_state.consecutive_failures += 1;
                let error_msg = format!("Kafka health check timeout after {}ms", self.config.health.timeout_ms);
                health_state.last_error = Some(error_msg.clone());
                
                let mut metrics = self.metrics.write().await;
                metrics.health_check_failures += 1;
                
                warn!("Kafka health check failed: {}", error_msg);
                
                let status = if health_state.consecutive_failures >= self.config.health.failure_threshold {
                    HealthLevel::Unhealthy
                } else {
                    HealthLevel::Degraded
                };
                
                Ok(HealthStatus {
                    status,
                    checked_at: chrono::Utc::now().timestamp_millis(),
                    response_time_ms: check_duration.as_millis() as u64,
                    error_message: Some(error_msg),
                    metrics: HashMap::new(),
                })
            }
        }
    }
    
    fn backend_name(&self) -> &str {
        "kafka"
    }
    
    fn supports_batching(&self) -> bool {
        true
    }
    
    async fn publish_batch(&mut self, signals: &[TradingSignal]) -> Result<Vec<PublishResult>> {
        self.publish_batch_signals(signals).await
    }
    
    fn get_metrics(&self) -> HashMap<String, String> {
        // Use blocking call since we can't make this async in the trait
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.get_metrics_map().await
            })
        })
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down Kafka publisher gracefully");
        
        // Set circuit breaker to open to prevent new requests
        {
            let mut circuit_breaker = self.circuit_breaker.write().await;
            // Force circuit breaker to open state by recording multiple failures
            for _ in 0..10 {
                circuit_breaker.record_failure();
            }
            info!("Circuit breaker set to open state for shutdown");
        }
        
        // Flush any pending messages with timeout
        let flush_timeout = Duration::from_secs(30);
        info!("Flushing pending messages with {}s timeout", flush_timeout.as_secs());
        
        match self.producer.flush(flush_timeout) {
            Ok(_) => {
                info!("Kafka producer flushed successfully during shutdown");
            }
            Err(e) => {
                warn!("Failed to flush Kafka producer during shutdown: {}", e);
                // Don't return error, continue with shutdown
            }
        }
        
        // Log final metrics
        let final_metrics = self.get_kafka_metrics().await;
        info!(
            "Kafka publisher shutdown complete - Final metrics: published={}, errors={}, avg_latency={:.2}ms, batches={}",
            final_metrics.messages_published,
            final_metrics.publish_errors,
            final_metrics.avg_publish_latency_ms,
            final_metrics.batch_operations
        );
        
        Ok(())
    }
}

impl Clone for KafkaMetrics {
    fn clone(&self) -> Self {
        Self {
            messages_published: self.messages_published,
            publish_errors: self.publish_errors,
            delivery_confirmations: self.delivery_confirmations,
            delivery_failures: self.delivery_failures,
            avg_publish_latency_ms: self.avg_publish_latency_ms,
            circuit_breaker_state_changes: self.circuit_breaker_state_changes,
            health_check_failures: self.health_check_failures,
            batch_operations: self.batch_operations,
            batch_signals_total: self.batch_signals_total,
            avg_batch_size: self.avg_batch_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SignalComponents, FusionWeights, SignalSide};
    
    fn create_test_signal() -> TradingSignal {
        let components = SignalComponents {
            s_ldc: 0.5,
            s_mr: 0.3,
            s_tsmom: 0.2,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        TradingSignal::new(
            chrono::Utc::now().timestamp(),
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components,
            weights,
            "v1.0".to_string(),
            "test-correlation-123".to_string(),
            "test-checksum-abc".to_string(),
            50,
        )
    }
    
    #[tokio::test]
    async fn test_kafka_config_default() {
        let config = KafkaConfig::default();
        assert_eq!(config.brokers, "localhost:9092");
        assert_eq!(config.topic, "trading_signals");
        assert_eq!(config.producer.compression_type, "snappy");
        assert_eq!(config.partitioning.strategy, PartitioningStrategy::Symbol);
    }
    
    #[tokio::test]
    async fn test_partition_key_generation() {
        let config = KafkaConfig::default();
        let publisher = KafkaPublisher::new(config).await;
        
        // Skip test if Kafka is not available
        if publisher.is_err() {
            return;
        }
        
        let publisher = publisher.unwrap();
        let signal = create_test_signal();
        
        // Test symbol partitioning
        let key = publisher.get_partition_key(&signal).await.unwrap();
        assert_eq!(key, Some("BTCUSDT".to_string()));
    }
    
    #[tokio::test]
    async fn test_custom_partition_key() {
        let mut config = KafkaConfig::default();
        config.partitioning.strategy = PartitioningStrategy::Custom;
        config.partitioning.custom_key_template = Some("{symbol}_{side}".to_string());
        
        let publisher = KafkaPublisher::new(config).await;
        
        // Skip test if Kafka is not available
        if publisher.is_err() {
            return;
        }
        
        let publisher = publisher.unwrap();
        let signal = create_test_signal();
        
        let key = publisher.get_partition_key(&signal).await.unwrap();
        assert_eq!(key, Some("BTCUSDT_BUY".to_string()));
    }
    
    #[tokio::test]
    async fn test_round_robin_partition_key() {
        let mut config = KafkaConfig::default();
        config.partitioning.strategy = PartitioningStrategy::RoundRobin;
        
        let publisher = KafkaPublisher::new(config).await;
        
        // Skip test if Kafka is not available
        if publisher.is_err() {
            return;
        }
        
        let publisher = publisher.unwrap();
        let signal = create_test_signal();
        
        let key = publisher.get_partition_key(&signal).await.unwrap();
        assert_eq!(key, None);
    }
    
    #[tokio::test]
    async fn test_metrics_initialization() {
        let config = KafkaConfig::default();
        let publisher = KafkaPublisher::new(config).await;
        
        // Skip test if Kafka is not available
        if publisher.is_err() {
            return;
        }
        
        let publisher = publisher.unwrap();
        let metrics = publisher.get_metrics();
        
        assert_eq!(metrics.get("messages_published").unwrap_or(&"0".to_string()), "0");
        assert_eq!(metrics.get("publish_errors").unwrap_or(&"0".to_string()), "0");
        assert_eq!(metrics.get("delivery_confirmations").unwrap_or(&"0".to_string()), "0");
        assert_eq!(metrics.get("delivery_failures").unwrap_or(&"0".to_string()), "0");
    }
}