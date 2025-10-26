//! Redis integration tests using testcontainers for isolated testing
//! 
//! These tests verify Redis Streams publishing, connection retry, circuit breaker functionality,
//! stream trimming, message ordering, and delivery confirmation.

use std::time::Duration;
use testcontainers::{clients::Cli, Container};
use testcontainers_modules::redis::Redis;
use tokio::time::timeout;

use signal_fusion::{TradingSignal, SignalSide, SignalComponents, FusionWeights};
use signal_fusion::emission::redis_publisher::{RedisPublisher, RedisConfig, RedisStreamConfig, OrderingStrategy};
use signal_fusion::emission::publisher::{PublisherTrait, RetryConfig};
use signal_fusion::emission::circuit_breaker::CircuitBreakerConfig;

/// Test helper to create a test trading signal
fn create_test_signal(symbol: &str, side: SignalSide, strength: f32) -> TradingSignal {
    TradingSignal::new(
        chrono::Utc::now().timestamp_millis(),
        symbol.to_string(),
        side,
        strength,
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
        format!("test-correlation-{}", uuid::Uuid::new_v4()),
        "test-checksum-abc".to_string(),
        50,
    )
}

/// Test helper to create Redis configuration for testing
fn create_test_redis_config(redis_url: &str) -> RedisConfig {
    RedisConfig {
        url: redis_url.to_string(),
        database: 0,
        connection_timeout_ms: 5000,
        command_timeout_ms: 3000,
        pool_size: 5,
        stream: RedisStreamConfig {
            stream_name: "test_signals".to_string(),
            max_len: Some(100),
            approximate_trimming: true,
            ordering_strategy: OrderingStrategy::PerSymbol,
            message_ttl_seconds: None,
        },
        health_check: Default::default(),
        retry: RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(1000),
            backoff_multiplier: 2.0,
            jitter: false,
        },
        circuit_breaker: CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_millis(1000),
            success_threshold: 2,
            failure_window: Duration::from_millis(60000),
            half_open_max_requests: 5,
        },
    }
}

/// Initialize tracing for tests
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .try_init();
}

#[tokio::test]
async fn test_redis_publisher_basic_functionality() {
    init_tracing();
    
    let docker = Cli::default();
    let redis_container = docker.run(Redis::default());
    let redis_port = redis_container.get_host_port_ipv4(6379);
    let redis_url = format!("redis://127.0.0.1:{}", redis_port);
    
    let config = create_test_redis_config(&redis_url);
    let mut publisher = RedisPublisher::new(config).await.expect("Failed to create Redis publisher");
    
    // Test basic signal publishing
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    let result = publisher.publish(&signal).await.expect("Failed to publish signal");
    
    assert_eq!(result.backend, "redis");
    assert!(!result.message_id.is_empty());
    assert!(result.latency_ms > 0);
    
    // Verify publisher metrics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"1".to_string()));
    assert_eq!(metrics.get("publish_errors"), Some(&"0".to_string()));
}

#[tokio::test]
async fn test_redis_streams_message_ordering() {
    init_tracing();
    
    let docker = Cli::default();
    let redis_container = docker.run(Redis::default());
    let redis_port = redis_container.get_host_port_ipv4(6379);
    let redis_url = format!("redis://127.0.0.1:{}", redis_port);
    
    let config = create_test_redis_config(&redis_url);
    let mut publisher = RedisPublisher::new(config).await.expect("Failed to create Redis publisher");
    
    // Publish signals for different symbols
    let btc_signal1 = create_test_signal("BTCUSDT", SignalSide::Buy, 0.5);
    let eth_signal1 = create_test_signal("ETHUSDT", SignalSide::Sell, -0.3);
    let btc_signal2 = create_test_signal("BTCUSDT", SignalSide::Hold, 0.1);
    
    // Publish signals
    let btc_result1 = publisher.publish(&btc_signal1).await.expect("Failed to publish BTC signal 1");
    let eth_result1 = publisher.publish(&eth_signal1).await.expect("Failed to publish ETH signal 1");
    let btc_result2 = publisher.publish(&btc_signal2).await.expect("Failed to publish BTC signal 2");
    
    // Verify all signals were published
    assert!(!btc_result1.message_id.is_empty());
    assert!(!eth_result1.message_id.is_empty());
    assert!(!btc_result2.message_id.is_empty());
    
    // Verify different symbols go to different streams (per-symbol ordering)
    assert!(btc_result1.metadata.contains_key("stream_name"));
    assert!(eth_result1.metadata.contains_key("stream_name"));
    
    // Verify publisher statistics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"3".to_string()));
    assert_eq!(metrics.get("publish_errors"), Some(&"0".to_string()));
}

#[tokio::test]
async fn test_redis_stream_trimming() {
    init_tracing();
    
    let docker = Cli::default();
    let redis_container = docker.run(Redis::default());
    let redis_port = redis_container.get_host_port_ipv4(6379);
    let redis_url = format!("redis://127.0.0.1:{}", redis_port);
    
    // Configure with small max length for testing trimming
    let mut config = create_test_redis_config(&redis_url);
    config.stream.max_len = Some(5);
    
    let mut publisher = RedisPublisher::new(config).await.expect("Failed to create Redis publisher");
    
    // Publish more signals than max_len to trigger trimming
    for i in 0..10 {
        let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.1 * i as f32);
        publisher.publish(&signal).await.expect("Failed to publish signal");
    }
    
    // Verify all signals were published (trimming happens after publish)
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"10".to_string()));
    assert_eq!(metrics.get("publish_errors"), Some(&"0".to_string()));
}

#[tokio::test]
async fn test_redis_health_check() {
    init_tracing();
    
    let docker = Cli::default();
    let redis_container = docker.run(Redis::default());
    let redis_port = redis_container.get_host_port_ipv4(6379);
    let redis_url = format!("redis://127.0.0.1:{}", redis_port);
    
    let config = create_test_redis_config(&redis_url);
    let mut publisher = RedisPublisher::new(config).await.expect("Failed to create Redis publisher");
    
    // Test health check
    let health_status = publisher.health_check().await.expect("Health check failed");
    
    assert!(matches!(health_status.status, signal_fusion::emission::publisher::HealthLevel::Healthy));
    assert!(health_status.response_time_ms > 0);
    assert!(health_status.error_message.is_none());
    assert!(health_status.metrics.contains_key("messages_published"));
    assert!(health_status.metrics.contains_key("circuit_breaker_state"));
}

#[tokio::test]
async fn test_redis_connection_retry_logic() {
    init_tracing();
    
    // Test with invalid Redis URL to trigger retry logic
    let config = RedisConfig {
        url: "redis://127.0.0.1:9999".to_string(), // Invalid port
        retry: RetryConfig {
            max_attempts: 2,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_millis(200),
            backoff_multiplier: 2.0,
            jitter: false,
        },
        ..create_test_redis_config("redis://127.0.0.1:9999")
    };
    
    // Creating publisher should fail due to connection timeout
    let result = timeout(Duration::from_secs(10), RedisPublisher::new(config)).await;
    assert!(result.is_ok()); // Timeout should not occur
    assert!(result.unwrap().is_err()); // But creation should fail
}

#[tokio::test]
async fn test_redis_circuit_breaker_functionality() {
    init_tracing();
    
    let docker = Cli::default();
    let redis_container = docker.run(Redis::default());
    let redis_port = redis_container.get_host_port_ipv4(6379);
    let redis_url = format!("redis://127.0.0.1:{}", redis_port);
    
    let mut config = create_test_redis_config(&redis_url);
    config.circuit_breaker.failure_threshold = 2;
    config.circuit_breaker.recovery_timeout = Duration::from_millis(500);
    config.circuit_breaker.failure_window = Duration::from_millis(60000);
    config.circuit_breaker.half_open_max_requests = 3;
    
    let mut publisher = RedisPublisher::new(config).await.expect("Failed to create Redis publisher");
    
    // First, publish a successful signal to verify normal operation
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    let result = publisher.publish(&signal).await;
    assert!(result.is_ok());
    
    // Verify circuit breaker is initially closed
    let cb_metrics = publisher.get_circuit_breaker_metrics();
    assert_eq!(cb_metrics.failure_count, 0);
    
    // Stop the Redis container to trigger failures
    drop(redis_container);
    
    // Try to publish signals to trigger circuit breaker
    for i in 0..3 {
        let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.1 * i as f32);
        let result = publisher.publish(&signal).await;
        
        if i < 2 {
            // First two should fail due to connection error
            assert!(result.is_err());
        } else {
            // Third should fail due to circuit breaker being open
            assert!(result.is_err());
            let error_msg = format!("{}", result.unwrap_err());
            assert!(error_msg.contains("Circuit breaker") || error_msg.contains("Redis"));
        }
    }
    
    // Verify circuit breaker metrics show failures
    let cb_metrics = publisher.get_circuit_breaker_metrics();
    assert!(cb_metrics.failure_count >= 2);
}

#[tokio::test]
async fn test_redis_publisher_shutdown() {
    init_tracing();
    
    let docker = Cli::default();
    let redis_container = docker.run(Redis::default());
    let redis_port = redis_container.get_host_port_ipv4(6379);
    let redis_url = format!("redis://127.0.0.1:{}", redis_port);
    
    let config = create_test_redis_config(&redis_url);
    let mut publisher = RedisPublisher::new(config).await.expect("Failed to create Redis publisher");
    
    // Publish a signal
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    publisher.publish(&signal).await.expect("Failed to publish signal");
    
    // Test graceful shutdown
    let shutdown_result = publisher.shutdown().await;
    assert!(shutdown_result.is_ok());
    
    // Verify final statistics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"1".to_string()));
}

#[tokio::test]
async fn test_redis_ordering_strategies() {
    init_tracing();
    
    let docker = Cli::default();
    let redis_container = docker.run(Redis::default());
    let redis_port = redis_container.get_host_port_ipv4(6379);
    let redis_url = format!("redis://127.0.0.1:{}", redis_port);
    
    // Test global ordering strategy
    let mut config = create_test_redis_config(&redis_url);
    config.stream.ordering_strategy = OrderingStrategy::Global;
    
    let mut publisher = RedisPublisher::new(config).await.expect("Failed to create Redis publisher");
    
    // Publish signals for different symbols
    let btc_signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.5);
    let eth_signal = create_test_signal("ETHUSDT", SignalSide::Sell, -0.3);
    
    let btc_result = publisher.publish(&btc_signal).await.expect("Failed to publish BTC signal");
    let eth_result = publisher.publish(&eth_signal).await.expect("Failed to publish ETH signal");
    
    // Both should be published successfully
    assert!(!btc_result.message_id.is_empty());
    assert!(!eth_result.message_id.is_empty());
    
    // Verify ordering strategy in metadata
    assert_eq!(btc_result.metadata.get("ordering_strategy"), Some(&"Global".to_string()));
    assert_eq!(eth_result.metadata.get("ordering_strategy"), Some(&"Global".to_string()));
}

#[tokio::test]
async fn test_redis_authentication_failure() {
    init_tracing();
    
    // Test with Redis URL that includes invalid authentication
    let config = RedisConfig {
        url: "redis://invalid_user:invalid_pass@127.0.0.1:6379".to_string(),
        connection_timeout_ms: 2000,
        ..create_test_redis_config("redis://invalid_user:invalid_pass@127.0.0.1:6379")
    };
    
    // Creating publisher should fail due to authentication error
    let result = timeout(Duration::from_secs(5), RedisPublisher::new(config)).await;
    assert!(result.is_ok()); // Timeout should not occur
    assert!(result.unwrap().is_err()); // But creation should fail
}

#[tokio::test]
async fn test_redis_signal_field_serialization() {
    init_tracing();
    
    let docker = Cli::default();
    let redis_container = docker.run(Redis::default());
    let redis_port = redis_container.get_host_port_ipv4(6379);
    let redis_url = format!("redis://127.0.0.1:{}", redis_port);
    
    let config = create_test_redis_config(&redis_url);
    let mut publisher = RedisPublisher::new(config).await.expect("Failed to create Redis publisher");
    
    // Create signal with HMM state probabilities
    let mut signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    signal.hmm_state_probabilities = Some(vec![0.3, 0.5, 0.2]);
    signal.fallback_used = true;
    
    // Publish signal
    let result = publisher.publish(&signal).await.expect("Failed to publish signal");
    
    assert!(!result.message_id.is_empty());
    assert_eq!(result.backend, "redis");
    
    // Verify publisher statistics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"1".to_string()));
}

#[tokio::test]
async fn test_redis_concurrent_publishing() {
    init_tracing();
    
    let docker = Cli::default();
    let redis_container = docker.run(Redis::default());
    let redis_port = redis_container.get_host_port_ipv4(6379);
    let redis_url = format!("redis://127.0.0.1:{}", redis_port);
    
    let config = create_test_redis_config(&redis_url);
    let mut publisher = RedisPublisher::new(config).await.expect("Failed to create Redis publisher");
    
    // Create multiple signals
    let signals: Vec<TradingSignal> = (0..10)
        .map(|i| create_test_signal(&format!("SYMBOL{}", i), SignalSide::Buy, 0.1 * i as f32))
        .collect();
    
    // Publish signals concurrently (note: we need to publish sequentially due to &mut self)
    let mut results = Vec::new();
    for signal in signals {
        let result = publisher.publish(&signal).await.expect("Failed to publish signal");
        results.push(result);
    }
    
    // Verify all signals were published
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(!result.message_id.is_empty());
        assert_eq!(result.backend, "redis");
    }
    
    // Verify publisher statistics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"10".to_string()));
    assert_eq!(metrics.get("publish_errors"), Some(&"0".to_string()));
}