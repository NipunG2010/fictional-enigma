//! Signal Emission Usage Examples
//! 
//! This example demonstrates common usage patterns for the signal emission system.

use signal_fusion::emission::{
    SignalPublisher, SignalEmissionConfig, SignalPublisherConfig, PublisherBackend,
    BufferConfig, AuditConfig, ValidationConfig,
};
use signal_fusion::{TradingSignal, SignalSide, SignalComponents, FusionWeights};
use chrono::Utc;
use uuid::Uuid;
use rand::Rng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Signal Emission Usage Examples ===\n");

    // Example 1: Basic configuration and publishing
    basic_usage_example().await?;

    // Example 2: Batch publishing
    batch_publishing_example().await?;

    // Example 3: Health monitoring
    health_monitoring_example().await?;

    // Example 4: Error handling
    error_handling_example().await?;

    // Example 5: Configuration from file
    configuration_from_file_example().await?;

    println!("\n=== All examples completed successfully ===");
    Ok(())
}

/// Example 1: Basic configuration and signal publishing
async fn basic_usage_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Example 1: Basic Usage ---");

    // Create a simple configuration
    let config = create_test_config();

    // Create signal publisher
    let mut publisher = SignalPublisher::new(config.publisher).await?;

    // Create a trading signal
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75, 0.85);

    // Publish the signal
    match publisher.publish_signal(signal).await {
        Ok(_) => println!("✓ Signal published successfully"),
        Err(e) => println!("✗ Failed to publish signal: {}", e),
    }

    // Get metrics
    let metrics = publisher.get_metrics().await;
    println!("📊 Signals published: {}", metrics.signals_published_total);
    println!("📊 Validation errors: {}", metrics.validation_errors_total);

    Ok(())
}

/// Example 2: Batch publishing for better performance
async fn batch_publishing_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 2: Batch Publishing ---");

    let config = create_test_config();
    let mut publisher = SignalPublisher::new(config.publisher).await?;

    // Create multiple signals
    let signals = vec![
        create_test_signal("BTCUSDT", SignalSide::Buy, 0.75, 0.85),
        create_test_signal("ETHUSDT", SignalSide::Sell, -0.60, 0.70),
        create_test_signal("ADAUSDT", SignalSide::Hold, 0.10, 0.45),
    ];

    println!("Publishing {} signals in batch...", signals.len());

    // Publish signals individually (batch publishing not available in current API)
    for signal in signals {
        match publisher.publish_signal(signal).await {
            Ok(_) => println!("✓ Signal published successfully"),
            Err(e) => println!("✗ Failed to publish signal: {}", e),
        }
    }

    let metrics = publisher.get_metrics().await;
    println!("📊 Total signals published: {}", metrics.signals_published_total);

    Ok(())
}

/// Example 3: Health monitoring and diagnostics
async fn health_monitoring_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 3: Health Monitoring ---");

    let config = create_test_config();
    let publisher = SignalPublisher::new(config.publisher).await?;

    // Check overall health
    let health = publisher.health_check().await;
    println!("🏥 Overall health: {:?}", health.status);
    if let Some(error) = &health.error_message {
        println!("🏥 Health error: {}", error);
    }
    println!("🏥 Response time: {}ms", health.response_time_ms);

    // Get detailed metrics
    let metrics = publisher.get_metrics().await;
    println!("\n📈 Performance Metrics:");
    println!("   Signals published: {}", metrics.signals_published_total);
    println!("   Validation errors: {}", metrics.validation_errors_total);
    println!("   Publisher errors: {}", metrics.publish_errors_total);

    Ok(())
}

/// Example 4: Error handling patterns
async fn error_handling_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 4: Error Handling ---");

    let config = create_test_config();
    let mut publisher = SignalPublisher::new(config.publisher).await?;

    // Create an invalid signal (strength out of range)
    let mut invalid_signal = create_test_signal("BTCUSDT", SignalSide::Buy, 2.0, 0.85); // Invalid strength
    invalid_signal.strength = 2.0; // This should cause validation error

    // Demonstrate error handling
    match publisher.publish_signal(invalid_signal).await {
        Ok(_) => println!("✓ Signal published (unexpected)"),
        Err(e) => {
            println!("✗ Expected error occurred: {}", e);
            
            // Handle different error types
            match &e {
                signal_fusion::emission::SignalEmissionError::ValidationError { message } => {
                    println!("   → Validation error: {}", message);
                },
                signal_fusion::emission::SignalEmissionError::PublisherError { backend, message } => {
                    println!("   → Publisher error on {}: {}", backend, message);
                },
                signal_fusion::emission::SignalEmissionError::CircuitBreakerOpen { reason } => {
                    println!("   → Circuit breaker open: {}", reason);
                },
                _ => {
                    println!("   → Other error: {}", e);
                }
            }
            
            // Check if error is retryable
            println!("   → Retryable: {}", e.is_retryable());
            println!("   → Category: {}", e.category());
        }
    }

    Ok(())
}

/// Example 5: Loading configuration from file
async fn configuration_from_file_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 5: Configuration from File ---");

    // Create a sample configuration file
    let config_content = r#"
[publisher]
enabled = true
backend = "none"  # Use "none" for testing
publish_timeout_ms = 5000
fail_fast = false
max_concurrent_publishes = 10

[buffer]
max_size = 500

[audit]
log_directory = "/tmp"
log_filename = "test_audit.jsonl"

[validation]
min_strength = -1.0
max_strength = 1.0

[metadata]
version = "1.0"
source = "file"
"#;

    // Write configuration to temporary file
    let config_path = "/tmp/signal_emission_example.toml";
    std::fs::write(config_path, config_content)?;

    // Load configuration from file
    match SignalEmissionConfig::from_file(config_path) {
        Ok(config) => {
            println!("✓ Configuration loaded from file");
            println!("   Backend: {}", config.publisher.backend);
            println!("   Enabled: {}", config.publisher.enabled);
            println!("   Buffer size: {}", config.buffer.max_size);
            
            // Configuration validation would be done here if available
            println!("✓ Configuration loaded successfully");
        },
        Err(e) => println!("✗ Failed to load configuration: {}", e),
    }

    // Clean up
    let _ = std::fs::remove_file(config_path);

    Ok(())
}

/// Helper function to create a test configuration
fn create_test_config() -> SignalEmissionConfig {
    SignalEmissionConfig {
        publisher: SignalPublisherConfig {
            enabled: true,
            backend: PublisherBackend::None, // Use "none" for testing
            redis: None,
            kafka: None,
            buffer: BufferConfig::default(),
            audit: AuditConfig::default(),
            validation: ValidationConfig::default(),
            publish_timeout_ms: 5000,
            fail_fast: false,
            max_concurrent_publishes: 10,
        },
        redis: None,
        kafka: None,
        buffer: BufferConfig::default(),
        audit: AuditConfig::default(),
        validation: ValidationConfig::default(),
        metadata: signal_fusion::emission::ConfigMetadata {
            version: "1.0".to_string(),
            source: signal_fusion::emission::ConfigSource::Default,
            loaded_at: chrono::Utc::now().timestamp(),
            file_path: Some("test.toml".into()),
            env_vars_used: vec![],
            warnings: vec![],
        },
    }
}

/// Helper function to create a test trading signal
fn create_test_signal(symbol: &str, side: SignalSide, strength: f32, confidence: f32) -> TradingSignal {
    TradingSignal {
        timestamp: Utc::now().timestamp_millis(),
        symbol: symbol.to_string(),
        side,
        strength,
        confidence,
        components: SignalComponents {
            s_ldc: strength * 0.4,
            s_mr: strength * 0.3,
            s_tsmom: strength * 0.3,
        },
        weights: FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        },
        model_version: "v1.0.0".to_string(),
        correlation_id: Uuid::new_v4().to_string(),
        feature_checksum: format!("checksum_{}", rand::thread_rng().gen::<u32>()),
        generation_latency_ms: 25,
        hmm_state_probabilities: Some(vec![0.7, 0.3]),
        fallback_used: false,
    }
}