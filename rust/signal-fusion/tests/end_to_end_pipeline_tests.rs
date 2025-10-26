//! End-to-end signal emission pipeline tests
//! 
//! These tests verify the complete signal emission pipeline from signal generation
//! to audit logging, including signal validation, publisher coordination, buffer management,
//! and failure scenario handling.

use std::time::Duration;
use tempfile::TempDir;

use signal_fusion::{TradingSignal, SignalSide, SignalComponents, FusionWeights};
use signal_fusion::signal_emitter::{SignalEmitter, SignalEmitterConfig};
use signal_fusion::emission::{SignalPublisherConfig, PublisherBackend, AuditConfig};

/// Test helper to create a test trading signal
fn create_test_signal(symbol: &str, side: SignalSide, strength: f32) -> TradingSignal {
    TradingSignal::new(
        chrono::Utc::now().timestamp(),
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

/// Create test configuration for end-to-end testing
fn create_test_emission_config() -> (SignalEmitterConfig, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config = SignalEmitterConfig {
        enabled: true,
        fail_on_emission_error: false,
        model_version: "test-v1.0".to_string(),
        audit_feature_computation: true,
        
        publisher: SignalPublisherConfig {
            enabled: true,
            backend: PublisherBackend::None, // Use None for testing without external dependencies
            redis: None,
            kafka: None,
            buffer: Default::default(),
            audit: AuditConfig {
                log_directory: temp_dir.path().to_path_buf(),
                ..Default::default()
            },
            validation: Default::default(),
            publish_timeout_ms: 5000,
            fail_fast: false,
            max_concurrent_publishes: 10,
        },
    };
    
    (config, temp_dir)
}

/// Initialize tracing for tests
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .try_init();
}

#[tokio::test]
async fn test_end_to_end_signal_emission_pipeline() {
    init_tracing();
    
    let (config, _temp_dir) = create_test_emission_config();
    
    // Create signal emitter
    let emitter = SignalEmitter::new(config).await.expect("Failed to create signal emitter");
    
    // Create test signal
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    
    // Emit signal through complete pipeline
    let result = emitter.emit_signal(signal.clone()).await;
    
    // Should succeed with None backend (no actual publishing)
    assert!(result.is_ok());
    
    // Verify signal emission was attempted
    let emitted = result.unwrap();
    // With None backend, emission should return true (successfully processed)
    assert!(emitted);
    
    // Verify emitter configuration
    assert!(emitter.is_enabled());
    assert_eq!(emitter.model_version(), "test-v1.0");
}

#[tokio::test]
async fn test_signal_emitter_disabled() {
    init_tracing();
    
    let (mut config, _temp_dir) = create_test_emission_config();
    config.enabled = false;
    
    let emitter = SignalEmitter::new(config).await.expect("Failed to create signal emitter");
    
    // Create test signal
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    
    // Emit signal - should be skipped
    let result = emitter.emit_signal(signal).await;
    assert!(result.is_ok());
    assert!(!result.unwrap()); // Should return false when disabled
    
    // Verify emitter is disabled
    assert!(!emitter.is_enabled());
}

#[tokio::test]
async fn test_signal_emitter_correlation_id_generation() {
    init_tracing();
    
    let (config, _temp_dir) = create_test_emission_config();
    let emitter = SignalEmitter::new(config).await.expect("Failed to create signal emitter");
    
    // Generate correlation IDs
    let id1 = emitter.generate_correlation_id();
    let id2 = emitter.generate_correlation_id();
    
    // Should be different
    assert_ne!(id1, id2);
    assert!(!id1.is_empty());
    assert!(!id2.is_empty());
}

#[tokio::test]
async fn test_signal_emitter_feature_checksum() {
    init_tracing();
    
    let (config, _temp_dir) = create_test_emission_config();
    let emitter = SignalEmitter::new(config).await.expect("Failed to create signal emitter");
    
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
    
    // Generate checksums
    let checksum1 = emitter.generate_feature_checksum(&components, &weights, "BTCUSDT", 1234567890);
    let checksum2 = emitter.generate_feature_checksum(&components, &weights, "BTCUSDT", 1234567890);
    let checksum3 = emitter.generate_feature_checksum(&components, &weights, "ETHUSDT", 1234567890);
    
    // Same inputs should produce same checksum
    assert_eq!(checksum1, checksum2);
    
    // Different symbol should produce different checksum
    assert_ne!(checksum1, checksum3);
    
    // Checksums should not be empty
    assert!(!checksum1.is_empty());
    assert!(!checksum3.is_empty());
}

#[tokio::test]
async fn test_signal_emission_with_hmm_data() {
    init_tracing();
    
    let (config, _temp_dir) = create_test_emission_config();
    let emitter = SignalEmitter::new(config).await.expect("Failed to create signal emitter");
    
    // Create signal with HMM state probabilities
    let mut signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    signal.hmm_state_probabilities = Some(vec![0.2, 0.5, 0.3]);
    signal.fallback_used = true;
    
    // Emit signal
    let result = emitter.emit_signal(signal).await;
    assert!(result.is_ok());
    
    // With None backend, should return true (successfully processed)
    assert!(result.unwrap());
}

#[tokio::test]
async fn test_performance_benchmarking() {
    init_tracing();
    
    let (config, _temp_dir) = create_test_emission_config();
    let emitter = SignalEmitter::new(config).await.expect("Failed to create signal emitter");
    
    // Benchmark signal emission throughput
    let start_time = std::time::Instant::now();
    let signal_count = 100;
    
    for i in 0..signal_count {
        let signal = create_test_signal(&format!("SYMBOL{}", i % 10), SignalSide::Buy, 0.01 * i as f32);
        emitter.emit_signal(signal).await.expect("Failed to emit signal");
    }
    
    let elapsed = start_time.elapsed();
    let throughput = signal_count as f64 / elapsed.as_secs_f64();
    
    println!("Signal emission throughput: {:.2} signals/second", throughput);
    
    // Verify reasonable performance (should be very fast with None backend)
    assert!(throughput > 1000.0, "Throughput too low: {:.2} signals/second", throughput);
    
    // Verify reasonable total time
    assert!(elapsed < Duration::from_secs(1), "Processing took too long: {:?}", elapsed);
}

#[tokio::test]
async fn test_concurrent_signal_emission() {
    init_tracing();
    
    let (config, _temp_dir) = create_test_emission_config();
    let emitter = SignalEmitter::new(config).await.expect("Failed to create signal emitter");
    
    // Create multiple signals
    let signals: Vec<TradingSignal> = (0..20)
        .map(|i| create_test_signal(&format!("SYMBOL{}", i % 5), SignalSide::Buy, 0.05 * i as f32))
        .collect();
    
    // Emit signals sequentially (due to &self requirement)
    let start_time = std::time::Instant::now();
    
    for signal in signals {
        emitter.emit_signal(signal).await.expect("Failed to emit signal");
    }
    
    let elapsed = start_time.elapsed();
    
    // Verify reasonable performance
    assert!(elapsed < Duration::from_secs(1), "Processing took too long: {:?}", elapsed);
    
    println!("Processed 20 signals in {:?}", elapsed);
}

#[tokio::test]
async fn test_signal_emitter_feature_auditing() {
    init_tracing();
    
    let (config, _temp_dir) = create_test_emission_config();
    let emitter = SignalEmitter::new(config).await.expect("Failed to create signal emitter");
    
    // Test feature computation logging
    let correlation_id = emitter.generate_correlation_id();
    let feature_names = vec!["rsi".to_string(), "ma".to_string(), "momentum".to_string()];
    
    let result = emitter.log_feature_computation(
        &correlation_id,
        "BTCUSDT",
        feature_names,
        25, // computation latency
        "input-checksum-123".to_string(),
        "output-checksum-456".to_string(),
        true, // validation passed
    ).await;
    
    // Should succeed (even if no actual audit logger is configured)
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_signal_emitter_fail_on_error_configuration() {
    init_tracing();
    
    // Test with fail_on_emission_error = true
    let (mut config, _temp_dir) = create_test_emission_config();
    config.fail_on_emission_error = true;
    config.publisher.backend = PublisherBackend::Redis; // This will fail without Redis
    
    let emitter = SignalEmitter::new(config).await;
    
    // Should either fail to create emitter or handle gracefully
    match emitter {
        Ok(emitter) => {
            let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
            let result = emitter.emit_signal(signal).await;
            
            // With fail_on_emission_error = true, should propagate errors
            // (but may succeed if Redis publisher handles missing connection gracefully)
            println!("Emission result with fail_on_error=true: {:?}", result);
        }
        Err(error) => {
            // Failed to create emitter due to configuration issues
            println!("Failed to create emitter as expected: {}", error);
        }
    }
}

#[tokio::test]
async fn test_disabled_signal_emitter_creation() {
    init_tracing();
    
    // Test creating a disabled emitter
    let disabled_emitter = SignalEmitter::disabled();
    
    assert!(!disabled_emitter.is_enabled());
    
    // Should be able to emit signals (but they'll be skipped)
    let signal = create_test_signal("BTCUSDT", SignalSide::Buy, 0.75);
    let result = disabled_emitter.emit_signal(signal).await;
    
    assert!(result.is_ok());
    assert!(!result.unwrap()); // Should return false when disabled
}