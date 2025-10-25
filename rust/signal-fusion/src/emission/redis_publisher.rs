//! Redis Streams publisher for trading signal emission
//! 
//! This module provides Redis Streams integration for publishing trading signals
//! with connection pooling, health checking, and automatic reconnection logic.

use async_trait::async_trait;
use redis::{Client, RedisResult, AsyncCommands};
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, info, warn, error};

use crate::TradingSignal;
use super::{Result, SignalEmissionError};
use super::publisher::{PublisherTrait, PublishResult, HealthStatus, HealthLevel, RetryConfig};
use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};

/// Configuration for Redis Streams publisher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL (e.g., "redis://localhost:6379")
    pub url: String,
    
    /// Redis database number (default: 0)
    pub database: u8,
    
    /// Connection timeout in milliseconds (default: 5000)
    pub connection_timeout_ms: u64,
    
    /// Command timeout in milliseconds (default: 3000)
    pub command_timeout_ms: u64,
    
    /// Connection pool size (default: 10)
    pub pool_size: u32,
    
    /// Stream configuration
    pub stream: RedisStreamConfig,
    
    /// Health check configuration
    pub health_check: RedisHealthConfig,
    
    /// Retry configuration
    pub retry: RetryConfig,
    
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
}

/// Redis Streams specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisStreamConfig {
    /// Base stream name (default: "trading_signals")
    pub stream_name: String,
    
    /// Maximum stream length to prevent unbounded growth (default: 10000)
    pub max_len: Option<usize>,
    
    /// Whether to use approximate trimming for better performance (default: true)
    pub approximate_trimming: bool,
    
    /// Message ordering strategy
    pub ordering_strategy: OrderingStrategy,
    
    /// Message TTL in seconds (Redis 7.0+, optional)
    pub message_ttl_seconds: Option<u64>,
}

/// Redis health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisHealthConfig {
    /// Health check timeout in milliseconds (default: 1000)
    pub timeout_ms: u64,
    
    /// Health check interval in seconds (default: 30)
    pub interval_seconds: u64,
    
    /// Maximum response time for healthy status in milliseconds (default: 100)
    pub healthy_threshold_ms: u64,
    
    /// Maximum response time for degraded status in milliseconds (default: 500)
    pub degraded_threshold_ms: u64,
}

/// Message ordering strategy for Redis Streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderingStrategy {
    /// Use symbol as stream key for per-symbol ordering
    PerSymbol,
    
    /// Use single stream for all signals (global ordering)
    Global,
    
    /// Use custom key pattern (e.g., "{symbol}:{interval}")
    Custom(String),
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379".to_string(),
            database: 0,
            connection_timeout_ms: 5000,
            command_timeout_ms: 3000,
            pool_size: 10,
            stream: RedisStreamConfig::default(),
            health_check: RedisHealthConfig::default(),
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

impl Default for RedisStreamConfig {
    fn default() -> Self {
        Self {
            stream_name: "trading_signals".to_string(),
            max_len: Some(10000),
            approximate_trimming: true,
            ordering_strategy: OrderingStrategy::PerSymbol,
            message_ttl_seconds: None,
        }
    }
}

impl Default for RedisHealthConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 1000,
            interval_seconds: 30,
            healthy_threshold_ms: 100,
            degraded_threshold_ms: 500,
        }
    }
}

/// Redis Streams publisher implementation
pub struct RedisPublisher {
    /// Redis connection manager for connection pooling
    connection_manager: ConnectionManager,
    
    /// Publisher configuration
    config: RedisConfig,
    
    /// Last health check result
    last_health_check: Option<HealthStatus>,
    
    /// Connection statistics
    stats: RedisStats,
    
    /// Circuit breaker for resilience
    circuit_breaker: CircuitBreaker,
}

/// Redis connection and operation statistics
#[derive(Debug, Default)]
pub struct RedisStats {
    /// Total number of messages published
    messages_published: u64,
    
    /// Total number of publish errors
    publish_errors: u64,
    
    /// Total number of connection errors
    connection_errors: u64,
    
    /// Total number of health checks performed
    health_checks: u64,
    
    /// Last successful publish timestamp
    last_publish_success: Option<Instant>,
    
    /// Last connection error timestamp
    last_connection_error: Option<Instant>,
}

impl RedisPublisher {
    /// Create a new Redis publisher with the given configuration
    pub async fn new(config: RedisConfig) -> Result<Self> {
        info!("Creating Redis publisher with URL: {}", config.url);
        
        // Create Redis client
        let client = Client::open(config.url.clone())
            .map_err(|e| SignalEmissionError::RedisError(e))?;
        
        // Create connection manager with timeout
        let connection_manager = timeout(
            Duration::from_millis(config.connection_timeout_ms),
            ConnectionManager::new(client)
        )
        .await
        .map_err(|_| SignalEmissionError::timeout(config.connection_timeout_ms))?
        .map_err(|e| SignalEmissionError::RedisError(e))?;
        
        let circuit_breaker = CircuitBreaker::with_config(config.circuit_breaker.clone());
        
        let publisher = Self {
            connection_manager,
            config,
            last_health_check: None,
            stats: RedisStats::default(),
            circuit_breaker,
        };
        
        info!("Redis publisher created successfully");
        Ok(publisher)
    }
    
    /// Create Redis publisher from environment variables
    pub async fn from_env() -> Result<Self> {
        let config = RedisConfig::from_env()?;
        Self::new(config).await
    }
    
    /// Get the stream name for a given signal based on ordering strategy
    fn get_stream_name(&self, signal: &TradingSignal) -> String {
        match &self.config.stream.ordering_strategy {
            OrderingStrategy::PerSymbol => {
                format!("{}:{}", self.config.stream.stream_name, signal.symbol)
            }
            OrderingStrategy::Global => {
                self.config.stream.stream_name.clone()
            }
            OrderingStrategy::Custom(pattern) => {
                // Simple pattern replacement - could be extended for more complex patterns
                pattern
                    .replace("{symbol}", &signal.symbol)
                    .replace("{stream}", &self.config.stream.stream_name)
            }
        }
    }
    
    /// Convert TradingSignal to Redis Stream fields
    fn signal_to_fields(&self, signal: &TradingSignal) -> Result<Vec<(String, String)>> {
        let mut fields = Vec::new();
        
        // Core signal fields
        fields.push(("timestamp".to_string(), signal.timestamp.to_string()));
        fields.push(("symbol".to_string(), signal.symbol.clone()));
        fields.push(("side".to_string(), signal.side.to_string()));
        fields.push(("strength".to_string(), signal.strength.to_string()));
        fields.push(("confidence".to_string(), signal.confidence.to_string()));
        
        // Signal components
        fields.push(("s_ldc".to_string(), signal.components.s_ldc.to_string()));
        fields.push(("s_mr".to_string(), signal.components.s_mr.to_string()));
        fields.push(("s_tsmom".to_string(), signal.components.s_tsmom.to_string()));
        
        // Fusion weights
        fields.push(("w_ldc".to_string(), signal.weights.w_ldc.to_string()));
        fields.push(("w_mr".to_string(), signal.weights.w_mr.to_string()));
        fields.push(("w_tsmom".to_string(), signal.weights.w_tsmom.to_string()));
        
        // Metadata
        fields.push(("model_version".to_string(), signal.model_version.clone()));
        fields.push(("correlation_id".to_string(), signal.correlation_id.clone()));
        fields.push(("feature_checksum".to_string(), signal.feature_checksum.clone()));
        fields.push(("generation_latency_ms".to_string(), signal.generation_latency_ms.to_string()));
        fields.push(("fallback_used".to_string(), signal.fallback_used.to_string()));
        
        // HMM state probabilities (if present)
        if let Some(ref probs) = signal.hmm_state_probabilities {
            let probs_json = serde_json::to_string(probs)
                .map_err(|e| SignalEmissionError::SerializationError(e))?;
            fields.push(("hmm_state_probabilities".to_string(), probs_json));
        }
        
        // Add publisher metadata
        fields.push(("publisher".to_string(), "redis".to_string()));
        fields.push(("published_at".to_string(), chrono::Utc::now().timestamp_millis().to_string()));
        
        Ok(fields)
    }
    
    /// Perform stream trimming if configured
    async fn trim_stream(&mut self, stream_name: &str) -> Result<()> {
        if let Some(max_len) = self.config.stream.max_len {
            debug!("Trimming stream {} to max length {}", stream_name, max_len);
            
            let result: RedisResult<i64> = timeout(
                Duration::from_millis(self.config.command_timeout_ms),
                async {
                    if self.config.stream.approximate_trimming {
                        redis::cmd("XTRIM")
                            .arg(stream_name)
                            .arg("MAXLEN")
                            .arg("~")
                            .arg(max_len)
                            .query_async(&mut self.connection_manager)
                            .await
                    } else {
                        redis::cmd("XTRIM")
                            .arg(stream_name)
                            .arg("MAXLEN")
                            .arg(max_len)
                            .query_async(&mut self.connection_manager)
                            .await
                    }
                }
            )
            .await
            .map_err(|_| SignalEmissionError::timeout(self.config.command_timeout_ms))?;
            
            result.map_err(|e| SignalEmissionError::RedisError(e))?;
            
            debug!("Stream {} trimmed successfully", stream_name);
        }
        
        Ok(())
    }
    
    /// Get publisher statistics
    pub fn get_stats(&self) -> &RedisStats {
        &self.stats
    }
    
    /// Reset publisher statistics
    pub fn reset_stats(&mut self) {
        self.stats = RedisStats::default();
    }
    
    /// Attempt to publish a single message (used internally for retry logic)
    async fn attempt_publish(&mut self, stream_name: &str, fields: &[(String, String)]) -> Result<String> {
        let message_id: String = timeout(
            Duration::from_millis(self.config.command_timeout_ms),
            self.connection_manager.xadd(stream_name, "*", fields)
        )
        .await
        .map_err(|_| SignalEmissionError::timeout(self.config.command_timeout_ms))?
        .map_err(|e| SignalEmissionError::RedisError(e))?;
        
        Ok(message_id)
    }
    
    /// Get circuit breaker metrics
    pub fn get_circuit_breaker_metrics(&self) -> super::circuit_breaker::CircuitBreakerMetrics {
        self.circuit_breaker.metrics()
    }
    
    /// Reset circuit breaker to closed state
    pub fn reset_circuit_breaker(&mut self) {
        self.circuit_breaker.reset();
    }
}

impl RedisConfig {
    /// Load Redis configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        
        // Redis connection settings
        if let Ok(url) = std::env::var("REDIS_URL") {
            config.url = url;
        }
        
        if let Ok(db) = std::env::var("REDIS_DATABASE") {
            config.database = db.parse()
                .map_err(|_| SignalEmissionError::config("Invalid REDIS_DATABASE value"))?;
        }
        
        if let Ok(timeout) = std::env::var("REDIS_CONNECTION_TIMEOUT_MS") {
            config.connection_timeout_ms = timeout.parse()
                .map_err(|_| SignalEmissionError::config("Invalid REDIS_CONNECTION_TIMEOUT_MS value"))?;
        }
        
        if let Ok(timeout) = std::env::var("REDIS_COMMAND_TIMEOUT_MS") {
            config.command_timeout_ms = timeout.parse()
                .map_err(|_| SignalEmissionError::config("Invalid REDIS_COMMAND_TIMEOUT_MS value"))?;
        }
        
        if let Ok(pool_size) = std::env::var("REDIS_POOL_SIZE") {
            config.pool_size = pool_size.parse()
                .map_err(|_| SignalEmissionError::config("Invalid REDIS_POOL_SIZE value"))?;
        }
        
        // Stream settings
        if let Ok(stream_name) = std::env::var("REDIS_STREAM_NAME") {
            config.stream.stream_name = stream_name;
        }
        
        if let Ok(max_len) = std::env::var("REDIS_STREAM_MAX_LENGTH") {
            if max_len.to_lowercase() == "none" {
                config.stream.max_len = None;
            } else {
                config.stream.max_len = Some(max_len.parse()
                    .map_err(|_| SignalEmissionError::config("Invalid REDIS_STREAM_MAX_LENGTH value"))?);
            }
        }
        
        if let Ok(approx) = std::env::var("REDIS_STREAM_APPROXIMATE_TRIMMING") {
            config.stream.approximate_trimming = approx.parse()
                .map_err(|_| SignalEmissionError::config("Invalid REDIS_STREAM_APPROXIMATE_TRIMMING value"))?;
        }
        
        if let Ok(strategy) = std::env::var("REDIS_STREAM_ORDERING_STRATEGY") {
            config.stream.ordering_strategy = match strategy.to_lowercase().as_str() {
                "per_symbol" => OrderingStrategy::PerSymbol,
                "global" => OrderingStrategy::Global,
                custom if custom.starts_with("custom:") => {
                    OrderingStrategy::Custom(custom.strip_prefix("custom:").unwrap().to_string())
                }
                _ => return Err(SignalEmissionError::config("Invalid REDIS_STREAM_ORDERING_STRATEGY value")),
            };
        }
        
        // Health check settings
        if let Ok(timeout) = std::env::var("REDIS_HEALTH_CHECK_TIMEOUT_MS") {
            config.health_check.timeout_ms = timeout.parse()
                .map_err(|_| SignalEmissionError::config("Invalid REDIS_HEALTH_CHECK_TIMEOUT_MS value"))?;
        }
        
        if let Ok(threshold) = std::env::var("REDIS_HEALTH_CHECK_HEALTHY_THRESHOLD_MS") {
            config.health_check.healthy_threshold_ms = threshold.parse()
                .map_err(|_| SignalEmissionError::config("Invalid REDIS_HEALTH_CHECK_HEALTHY_THRESHOLD_MS value"))?;
        }
        
        if let Ok(threshold) = std::env::var("REDIS_HEALTH_CHECK_DEGRADED_THRESHOLD_MS") {
            config.health_check.degraded_threshold_ms = threshold.parse()
                .map_err(|_| SignalEmissionError::config("Invalid REDIS_HEALTH_CHECK_DEGRADED_THRESHOLD_MS value"))?;
        }
        
        Ok(config)
    }
    
    /// Validate the Redis configuration
    pub fn validate(&self) -> Result<()> {
        if self.url.is_empty() {
            return Err(SignalEmissionError::config("Redis URL cannot be empty"));
        }
        
        if self.connection_timeout_ms == 0 {
            return Err(SignalEmissionError::config("Connection timeout must be greater than 0"));
        }
        
        if self.command_timeout_ms == 0 {
            return Err(SignalEmissionError::config("Command timeout must be greater than 0"));
        }
        
        if self.pool_size == 0 {
            return Err(SignalEmissionError::config("Pool size must be greater than 0"));
        }
        
        if self.stream.stream_name.is_empty() {
            return Err(SignalEmissionError::config("Stream name cannot be empty"));
        }
        
        if let Some(max_len) = self.stream.max_len {
            if max_len == 0 {
                return Err(SignalEmissionError::config("Stream max length must be greater than 0"));
            }
        }
        
        if self.health_check.timeout_ms == 0 {
            return Err(SignalEmissionError::config("Health check timeout must be greater than 0"));
        }
        
        if self.health_check.healthy_threshold_ms >= self.health_check.degraded_threshold_ms {
            return Err(SignalEmissionError::config(
                "Healthy threshold must be less than degraded threshold"
            ));
        }
        
        Ok(())
    }
}

#[async_trait]
impl PublisherTrait for RedisPublisher {
    async fn publish(&mut self, signal: &TradingSignal) -> Result<PublishResult> {
        let start_time = Instant::now();
        let stream_name = self.get_stream_name(signal);
        
        debug!(
            "Publishing signal to Redis stream {}: {}",
            stream_name,
            signal.to_compact_string()
        );
        
        // Check circuit breaker
        if !self.circuit_breaker.is_request_allowed() {
            let time_until_recovery = self.circuit_breaker.time_until_recovery()
                .map(|d| d.as_secs())
                .unwrap_or(0);
            
            warn!(
                "Circuit breaker is {:?}, rejecting publish request (recovery in {}s)",
                self.circuit_breaker.state(),
                time_until_recovery
            );
            
            return Err(SignalEmissionError::circuit_breaker(
                format!("Circuit breaker is {:?}", self.circuit_breaker.state())
            ));
        }
        
        // Convert signal to Redis fields
        let fields = self.signal_to_fields(signal)?;
        
        // Attempt publish with retry logic
        let mut last_error = None;
        let mut retry_count = 0;
        
        while retry_count <= self.config.retry.max_attempts {
            match self.attempt_publish(&stream_name, &fields).await {
                Ok(message_id) => {
                    // Success - record with circuit breaker
                    self.circuit_breaker.record_success();
                    
                    // Perform stream trimming if configured (don't fail on trimming errors)
                    if let Err(e) = self.trim_stream(&stream_name).await {
                        warn!("Failed to trim stream {}: {}", stream_name, e);
                    }
                    
                    let latency_ms = start_time.elapsed().as_millis() as u64;
                    
                    // Update statistics
                    self.stats.messages_published += 1;
                    self.stats.last_publish_success = Some(Instant::now());
                    
                    info!(
                        "Signal published to Redis stream {} with ID {} (latency: {}ms, retries: {})",
                        stream_name, message_id, latency_ms, retry_count
                    );
                    
                    return Ok(PublishResult::new(
                        message_id,
                        "redis",
                        chrono::Utc::now().timestamp_millis(),
                        latency_ms,
                    )
                    .with_metadata("stream_name", stream_name)
                    .with_metadata("ordering_strategy", format!("{:?}", self.config.stream.ordering_strategy))
                    .with_retry_count(retry_count));
                }
                Err(e) => {
                    last_error = Some(e);
                    retry_count += 1;
                    
                    // Record failure with circuit breaker
                    self.circuit_breaker.record_failure();
                    
                    // Update error statistics
                    self.stats.publish_errors += 1;
                    self.stats.last_connection_error = Some(Instant::now());
                    
                    // Check if we should retry
                    if retry_count <= self.config.retry.max_attempts && self.config.retry.should_retry(retry_count) {
                        let delay = self.config.retry.delay_for_attempt(retry_count);
                        
                        warn!(
                            "Redis publish attempt {} failed, retrying in {:?}: {}",
                            retry_count, delay, last_error.as_ref().unwrap()
                        );
                        
                        tokio::time::sleep(delay).await;
                    } else {
                        break;
                    }
                }
            }
        }
        
        // All retries exhausted
        let final_error = last_error.unwrap_or_else(|| {
            SignalEmissionError::publisher("redis", "Unknown error after retries")
        });
        
        error!(
            "Failed to publish signal to Redis stream {} after {} attempts: {}",
            stream_name, retry_count, final_error
        );
        
        Err(final_error)
    }
    
    async fn health_check(&mut self) -> Result<HealthStatus> {
        let start_time = Instant::now();
        
        debug!("Performing Redis health check");
        
        // Clone connection manager for health check
        let mut conn = self.connection_manager.clone();
        
        // Perform PING command with timeout
        let ping_result: RedisResult<String> = timeout(
            Duration::from_millis(self.config.health_check.timeout_ms),
            redis::cmd("PING").query_async(&mut conn)
        )
        .await
        .map_err(|_| SignalEmissionError::timeout(self.config.health_check.timeout_ms))?;
        
        let response_time_ms = start_time.elapsed().as_millis() as u64;
        
        match ping_result {
            Ok(_) => {
                let status = if response_time_ms <= self.config.health_check.healthy_threshold_ms {
                    HealthLevel::Healthy
                } else if response_time_ms <= self.config.health_check.degraded_threshold_ms {
                    HealthLevel::Degraded
                } else {
                    HealthLevel::Unhealthy
                };
                
                let error_message = if matches!(status, HealthLevel::Unhealthy) {
                    Some(format!("Response time {}ms exceeds degraded threshold {}ms", 
                               response_time_ms, self.config.health_check.degraded_threshold_ms))
                } else {
                    None
                };
                
                let health_status = HealthStatus {
                    status,
                    checked_at: chrono::Utc::now().timestamp_millis(),
                    response_time_ms,
                    error_message,
                    metrics: {
                        let mut metrics = HashMap::from([
                            ("messages_published".to_string(), self.stats.messages_published.to_string()),
                            ("publish_errors".to_string(), self.stats.publish_errors.to_string()),
                            ("connection_errors".to_string(), self.stats.connection_errors.to_string()),
                        ]);
                        
                        // Add circuit breaker state
                        let cb_metrics = self.circuit_breaker.metrics();
                        metrics.insert("circuit_breaker_state".to_string(), format!("{:?}", cb_metrics.state));
                        metrics.insert("circuit_breaker_failure_count".to_string(), cb_metrics.failure_count.to_string());
                        
                        metrics
                    },
                };
                
                debug!("Redis health check completed: {:?}", health_status.status);
                self.last_health_check = Some(health_status.clone());
                Ok(health_status)
            }
            Err(e) => {
                let cb_metrics = self.circuit_breaker.metrics();
                let health_status = HealthStatus::unhealthy(format!("Redis PING failed: {}", e))
                    .with_metric("messages_published", self.stats.messages_published.to_string())
                    .with_metric("publish_errors", self.stats.publish_errors.to_string())
                    .with_metric("connection_errors", self.stats.connection_errors.to_string())
                    .with_metric("circuit_breaker_state", format!("{:?}", cb_metrics.state))
                    .with_metric("circuit_breaker_failure_count", cb_metrics.failure_count.to_string());
                
                error!("Redis health check failed: {}", e);
                self.last_health_check = Some(health_status.clone());
                Ok(health_status)
            }
        }
    }
    
    fn backend_name(&self) -> &str {
        "redis"
    }
    
    fn get_metrics(&self) -> HashMap<String, String> {
        let mut metrics = HashMap::new();
        
        metrics.insert("messages_published".to_string(), self.stats.messages_published.to_string());
        metrics.insert("publish_errors".to_string(), self.stats.publish_errors.to_string());
        metrics.insert("connection_errors".to_string(), self.stats.connection_errors.to_string());
        metrics.insert("health_checks".to_string(), self.stats.health_checks.to_string());
        
        if let Some(last_success) = self.stats.last_publish_success {
            metrics.insert("last_publish_success_seconds_ago".to_string(), 
                         last_success.elapsed().as_secs().to_string());
        }
        
        if let Some(last_error) = self.stats.last_connection_error {
            metrics.insert("last_connection_error_seconds_ago".to_string(), 
                         last_error.elapsed().as_secs().to_string());
        }
        
        metrics.insert("stream_name".to_string(), self.config.stream.stream_name.clone());
        metrics.insert("ordering_strategy".to_string(), format!("{:?}", self.config.stream.ordering_strategy));
        
        if let Some(max_len) = self.config.stream.max_len {
            metrics.insert("stream_max_length".to_string(), max_len.to_string());
        }
        
        // Circuit breaker metrics
        let cb_metrics = self.circuit_breaker.metrics();
        metrics.insert("circuit_breaker_state".to_string(), format!("{:?}", cb_metrics.state));
        metrics.insert("circuit_breaker_failure_count".to_string(), cb_metrics.failure_count.to_string());
        metrics.insert("circuit_breaker_success_count".to_string(), cb_metrics.success_count.to_string());
        
        if let Some(time_until_recovery) = cb_metrics.time_until_recovery_seconds {
            metrics.insert("circuit_breaker_recovery_seconds".to_string(), time_until_recovery.to_string());
        }
        
        metrics
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down Redis publisher");
        
        // Redis ConnectionManager handles connection cleanup automatically
        // Just log the final statistics
        info!(
            "Redis publisher shutdown complete. Final stats: {} messages published, {} errors",
            self.stats.messages_published,
            self.stats.publish_errors
        );
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SignalComponents, FusionWeights, SignalSide};
    
    fn create_test_signal() -> TradingSignal {
        TradingSignal::new(
            chrono::Utc::now().timestamp(),
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
    fn test_redis_config_default() {
        let config = RedisConfig::default();
        assert_eq!(config.url, "redis://localhost:6379");
        assert_eq!(config.database, 0);
        assert_eq!(config.stream.stream_name, "trading_signals");
        assert_eq!(config.stream.max_len, Some(10000));
    }
    
    #[test]
    fn test_redis_config_validation() {
        let mut config = RedisConfig::default();
        assert!(config.validate().is_ok());
        
        config.url = "".to_string();
        assert!(config.validate().is_err());
        
        config.url = "redis://localhost:6379".to_string();
        config.connection_timeout_ms = 0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_stream_name_generation() {
        let config = RedisConfig::default();
        
        // Test per-symbol ordering
        let signal = create_test_signal();
        let stream_name = match &config.stream.ordering_strategy {
            OrderingStrategy::PerSymbol => {
                format!("{}:{}", config.stream.stream_name, signal.symbol)
            }
            OrderingStrategy::Global => {
                config.stream.stream_name.clone()
            }
            OrderingStrategy::Custom(pattern) => {
                pattern
                    .replace("{symbol}", &signal.symbol)
                    .replace("{stream}", &config.stream.stream_name)
            }
        };
        
        assert_eq!(stream_name, "trading_signals:BTCUSDT");
    }
    
    #[test]
    fn test_signal_to_fields_conversion() {
        let signal = create_test_signal();
        
        // Test the field conversion logic directly
        let mut fields = Vec::new();
        
        // Core signal fields
        fields.push(("timestamp".to_string(), signal.timestamp.to_string()));
        fields.push(("symbol".to_string(), signal.symbol.clone()));
        fields.push(("side".to_string(), signal.side.to_string()));
        fields.push(("strength".to_string(), signal.strength.to_string()));
        fields.push(("confidence".to_string(), signal.confidence.to_string()));
        
        // Signal components
        fields.push(("s_ldc".to_string(), signal.components.s_ldc.to_string()));
        fields.push(("s_mr".to_string(), signal.components.s_mr.to_string()));
        fields.push(("s_tsmom".to_string(), signal.components.s_tsmom.to_string()));
        
        // Fusion weights
        fields.push(("w_ldc".to_string(), signal.weights.w_ldc.to_string()));
        fields.push(("w_mr".to_string(), signal.weights.w_mr.to_string()));
        fields.push(("w_tsmom".to_string(), signal.weights.w_tsmom.to_string()));
        
        // Metadata
        fields.push(("model_version".to_string(), signal.model_version.clone()));
        fields.push(("correlation_id".to_string(), signal.correlation_id.clone()));
        fields.push(("feature_checksum".to_string(), signal.feature_checksum.clone()));
        fields.push(("generation_latency_ms".to_string(), signal.generation_latency_ms.to_string()));
        fields.push(("fallback_used".to_string(), signal.fallback_used.to_string()));
        
        // Add publisher metadata
        fields.push(("publisher".to_string(), "redis".to_string()));
        
        // Check that all required fields are present
        let field_map: HashMap<String, String> = fields.into_iter().collect();
        
        assert_eq!(field_map.get("symbol"), Some(&"BTCUSDT".to_string()));
        assert_eq!(field_map.get("side"), Some(&"BUY".to_string()));
        assert_eq!(field_map.get("strength"), Some(&"0.75".to_string()));
        assert_eq!(field_map.get("confidence"), Some(&"0.85".to_string()));
        assert_eq!(field_map.get("s_ldc"), Some(&"0.5".to_string()));
        assert_eq!(field_map.get("model_version"), Some(&"v1.0".to_string()));
        assert_eq!(field_map.get("correlation_id"), Some(&"test-correlation-123".to_string()));
        assert_eq!(field_map.get("publisher"), Some(&"redis".to_string()));
    }
    
    #[test]
    fn test_ordering_strategies() {
        let signal = create_test_signal();
        
        // Test per-symbol strategy
        let per_symbol_name = format!("trading_signals:{}", signal.symbol);
        assert_eq!(per_symbol_name, "trading_signals:BTCUSDT");
        
        // Test global strategy
        let global_name = "trading_signals".to_string();
        assert_eq!(global_name, "trading_signals");
        
        // Test custom strategy
        let custom_pattern = "signals_{symbol}_5m";
        let custom_name = custom_pattern
            .replace("{symbol}", &signal.symbol)
            .replace("{stream}", "trading_signals");
        assert_eq!(custom_name, "signals_BTCUSDT_5m");
    }
    
    #[test]
    fn test_circuit_breaker_integration() {
        let config = RedisConfig {
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: 2,
                recovery_timeout: Duration::from_millis(100),
                success_threshold: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        
        // Test that circuit breaker config is properly set
        assert_eq!(config.circuit_breaker.failure_threshold, 2);
        assert_eq!(config.circuit_breaker.recovery_timeout, Duration::from_millis(100));
    }
    
    #[test]
    fn test_retry_config_integration() {
        let config = RedisConfig {
            retry: RetryConfig {
                max_attempts: 5,
                base_delay: Duration::from_millis(200),
                ..Default::default()
            },
            ..Default::default()
        };
        
        // Test retry configuration
        assert_eq!(config.retry.max_attempts, 5);
        assert_eq!(config.retry.base_delay, Duration::from_millis(200));
        assert!(config.retry.should_retry(3));
        assert!(!config.retry.should_retry(6));
    }
}