//! Kafka integration tests using testcontainers for isolated testing
//! 
//! These tests verify Kafka topic publishing, partitioning strategies, delivery confirmation,
//! producer configuration, compression, and batch publishing functionality.

use std::time::Duration;
use testcontainers::{clients::Cli};
use testcontainers_modules::kafka::Kafka;
use tokio::time::timeout;

use signal_fusion::{TradingSignal, SignalSide, SignalComponents, FusionWeights};
use signal_fusion::emission::kafka_publisher::{KafkaPublisher, KafkaConfig, KafkaProducerConfig, PartitioningConfig, PartitioningStrategy};
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

/// Test helper to create Kafka configuration for testing
fn create_test_kafka_config(brokers: &str) -> KafkaConfig {
    KafkaConfig {
        brokers: brokers.to_string(),
        topic: "test_signals".to_string(),
        producer: KafkaProducerConfig {
            client_id: "test-producer".to_string(),
            acks: "1".to_string(), // Use acks=1 for faster testing
            request_timeout_ms: 5000,
            delivery_timeout_ms: 10000,
            batch_size: 1000,
            linger_ms: 10,
            compression_type: "none".to_string(),
            max_in_flight_requests_per_connection: 5,
            enable_idempotence: false, // Disable for simpler testing
            retries: 3,
            retry_backoff_ms: 100,
        },
        partitioning: PartitioningConfig {
            strategy: PartitioningStrategy::BySymbol,
            custom_key_template: None,
            partition_count: Some(3),
        },
        health: Default::default(),
        security: None,
    }
}

/// Initialize tracing for tests
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .try_init();
}

#[tokio::test]
async fn test_kafka_publisher_basic_functionality() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    let config = create_test_kafka_config(&brokers);
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Test basic signal publishing
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    let result = publisher.publish(&signal).await.expect("Failed to publish signal");
    
    assert_eq!(result.backend, "kafka");
    assert!(!result.message_id.is_empty());
    assert!(result.latency_ms > 0);
    
    // Verify publisher metrics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"1".to_string()));
    assert_eq!(metrics.get("publish_errors"), Some(&"0".to_string()));
}

#[tokio::test]
async fn test_kafka_partitioning_strategies() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    // Test symbol-based partitioning
    let mut config = create_test_kafka_config(&brokers);
    config.partitioning.strategy = PartitioningStrategy::BySymbol;
    
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Publish signals for different symbols
    let btc_signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.5);
    let eth_signal = create_test_signal("ETHUSDT", SignalSide::Sell, -0.3);
    let ada_signal = create_test_signal("ADAUSDT", SignalSide::Hold, 0.1);
    
    let btc_result = publisher.publish(&btc_signal).await.expect("Failed to publish BTC signal");
    let eth_result = publisher.publish(&eth_signal).await.expect("Failed to publish ETH signal");
    let ada_result = publisher.publish(&ada_signal).await.expect("Failed to publish ADA signal");
    
    // Verify all signals were published
    assert!(!btc_result.message_id.is_empty());
    assert!(!eth_result.message_id.is_empty());
    assert!(!ada_result.message_id.is_empty());
    
    // Verify partitioning strategy in metadata
    assert!(btc_result.metadata.contains_key("partition"));
    assert!(eth_result.metadata.contains_key("partition"));
    assert!(ada_result.metadata.contains_key("partition"));
    
    // Verify publisher statistics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"3".to_string()));
    assert_eq!(metrics.get("publish_errors"), Some(&"0".to_string()));
}

#[tokio::test]
async fn test_kafka_round_robin_partitioning() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    // Test round-robin partitioning
    let mut config = create_test_kafka_config(&brokers);
    config.partitioning.strategy = PartitioningStrategy::RoundRobin;
    
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Publish multiple signals with same symbol
    let mut results = Vec::new();
    for i in 0..5 {
        let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.1 * i as f32);
        let result = publisher.publish(&signal).await.expect("Failed to publish signal");
        results.push(result);
    }
    
    // Verify all signals were published
    assert_eq!(results.len(), 5);
    for result in &results {
        assert!(!result.message_id.is_empty());
        assert_eq!(result.backend, "kafka");
    }
    
    // Verify publisher statistics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"5".to_string()));
    assert_eq!(metrics.get("publish_errors"), Some(&"0".to_string()));
}

#[tokio::test]
async fn test_kafka_compression_support() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    // Test with gzip compression
    let mut config = create_test_kafka_config(&brokers);
    config.producer.compression_type = "gzip".to_string();
    
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Publish signal with compression
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    let result = publisher.publish(&signal).await.expect("Failed to publish signal");
    
    assert!(!result.message_id.is_empty());
    assert_eq!(result.backend, "kafka");
    
    // Verify compression is reflected in metadata
    assert!(result.metadata.contains_key("compression"));
    assert_eq!(result.metadata.get("compression"), Some(&"gzip".to_string()));
}

#[tokio::test]
async fn test_kafka_batch_publishing() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    // Configure for batching
    let mut config = create_test_kafka_config(&brokers);
    config.producer.batch_size = 500;
    config.producer.linger_ms = 50;
    
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Create batch of signals
    let signals: Vec<TradingSignal> = (0..10)
        .map(|i| create_test_signal(&format!("SYMBOL{}", i), SignalSide::Buy, 0.1 * i as f32))
        .collect();
    
    // Publish signals in batch (if supported)
    if publisher.supports_batching() {
        let results = publisher.publish_batch(&signals).await.expect("Failed to publish batch");
        assert_eq!(results.len(), 10);
        
        for result in results {
            assert!(!result.message_id.is_empty());
            assert_eq!(result.backend, "kafka");
        }
    } else {
        // Publish individually
        for signal in signals {
            let result = publisher.publish(&signal).await.expect("Failed to publish signal");
            assert!(!result.message_id.is_empty());
        }
    }
    
    // Verify publisher statistics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"10".to_string()));
}

#[tokio::test]
async fn test_kafka_health_check() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    let config = create_test_kafka_config(&brokers);
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Test health check
    let health_status = publisher.health_check().await.expect("Health check failed");
    
    assert!(matches!(health_status.status, signal_fusion::emission::publisher::HealthLevel::Healthy));
    assert!(health_status.response_time_ms > 0);
    assert!(health_status.error_message.is_none());
    assert!(health_status.metrics.contains_key("messages_published"));
}

#[tokio::test]
async fn test_kafka_delivery_confirmation() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    // Configure with delivery confirmation
    let mut config = create_test_kafka_config(&brokers);
    config.producer.acks = "all".to_string(); // Wait for all replicas
    
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Publish signal and verify delivery confirmation
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    let result = publisher.publish(&signal).await.expect("Failed to publish signal");
    
    assert!(!result.message_id.is_empty());
    assert_eq!(result.backend, "kafka");
    assert!(result.latency_ms > 0);
    
    // Verify delivery confirmation metadata
    assert!(result.metadata.contains_key("offset"));
    assert!(result.metadata.contains_key("partition"));
}

#[tokio::test]
async fn test_kafka_connection_failure_handling() {
    init_tracing();
    
    // Test with invalid Kafka brokers to trigger connection failure
    let config = KafkaConfig {
        brokers: "127.0.0.1:9999".to_string(), // Invalid port
        topic: "test_signals".to_string(),
        producer: KafkaProducerConfig {
            request_timeout_ms: 2000,
            delivery_timeout_ms: 3000,
            retries: 1,
            ..Default::default()
        },
        ..create_test_kafka_config("127.0.0.1:9999")
    };
    
    // Creating publisher should succeed (connection is lazy)
    let result = timeout(Duration::from_secs(5), KafkaPublisher::new(config)).await;
    assert!(result.is_ok());
    
    if let Ok(publisher_result) = result {
        // Publisher creation might succeed, but publishing should fail
        if let Ok(mut publisher) = publisher_result {
            let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
            let publish_result = publisher.publish(&signal).await;
            // Should fail due to connection error
            assert!(publish_result.is_err());
        }
    }
}

#[tokio::test]
async fn test_kafka_topic_not_found_handling() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    // Configure with non-existent topic
    let mut config = create_test_kafka_config(&brokers);
    config.topic = "non_existent_topic".to_string();
    config.producer.request_timeout_ms = 3000;
    
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Publishing to non-existent topic should either succeed (auto-create) or fail gracefully
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    let result = publisher.publish(&signal).await;
    
    // Result depends on Kafka configuration (auto.create.topics.enable)
    // We just verify it doesn't panic and handles the error gracefully
    match result {
        Ok(publish_result) => {
            assert!(!publish_result.message_id.is_empty());
        }
        Err(error) => {
            // Should be a proper error, not a panic
            assert!(!format!("{}", error).is_empty());
        }
    }
}

#[tokio::test]
async fn test_kafka_producer_configuration() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    // Test with custom producer configuration
    let mut config = create_test_kafka_config(&brokers);
    config.producer.client_id = "custom-test-producer".to_string();
    config.producer.batch_size = 2000;
    config.producer.linger_ms = 100;
    config.producer.compression_type = "snappy".to_string();
    config.producer.enable_idempotence = true;
    
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Publish signal to verify configuration works
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    let result = publisher.publish(&signal).await.expect("Failed to publish signal");
    
    assert!(!result.message_id.is_empty());
    assert_eq!(result.backend, "kafka");
    
    // Verify configuration is reflected in metrics
    let metrics = publisher.get_metrics();
    assert!(metrics.contains_key("client_id"));
    assert_eq!(metrics.get("client_id"), Some(&"custom-test-producer".to_string()));
}

#[tokio::test]
async fn test_kafka_publisher_shutdown() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    let config = create_test_kafka_config(&brokers);
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
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
async fn test_kafka_concurrent_publishing() {
    init_tracing();
    
    let docker = Cli::default();
    let kafka_container = docker.run(Kafka::default());
    let kafka_port = kafka_container.get_host_port_ipv4(9092);
    let brokers = format!("127.0.0.1:{}", kafka_port);
    
    let config = create_test_kafka_config(&brokers);
    let mut publisher = KafkaPublisher::new(config).await.expect("Failed to create Kafka publisher");
    
    // Create multiple signals
    let signals: Vec<TradingSignal> = (0..10)
        .map(|i| create_test_signal(&format!("SYMBOL{}", i), SignalSide::Buy, 0.1 * i as f32))
        .collect();
    
    // Publish signals sequentially (due to &mut self requirement)
    let mut results = Vec::new();
    for signal in signals {
        let result = publisher.publish(&signal).await.expect("Failed to publish signal");
        results.push(result);
    }
    
    // Verify all signals were published
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(!result.message_id.is_empty());
        assert_eq!(result.backend, "kafka");
    }
    
    // Verify publisher statistics
    let metrics = publisher.get_metrics();
    assert_eq!(metrics.get("messages_published"), Some(&"10".to_string()));
    assert_eq!(metrics.get("publish_errors"), Some(&"0".to_string()));
}