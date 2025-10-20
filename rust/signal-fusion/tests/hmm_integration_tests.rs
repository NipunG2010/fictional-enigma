//! Integration tests for HMM microservice client
//!
//! These tests verify the Rust client integration with the HMM microservice.
//! They require a running HMM service instance for full testing.

use signal_fusion::{
    hmm_client::{HmmClient, HmmClientConfig, HmmIntegration, HmmClientError},
    SignalComponents, FusionWeights,
};
use std::time::Duration;
use tokio;

/// Test basic HMM client creation and configuration
#[tokio::test]
async fn test_hmm_client_creation() {
    let client = HmmClient::new();
    assert!(client.is_ok());
    
    let config = HmmClientConfig::default();
    let client_with_config = HmmClient::with_config(config);
    assert!(client_with_config.is_ok());
}

/// Test HMM integration creation
#[tokio::test]
async fn test_hmm_integration_creation() {
    let integration = HmmIntegration::new();
    assert!(integration.is_ok());
    
    let config = HmmClientConfig::default();
    let integration_with_config = HmmIntegration::with_config(config);
    assert!(integration_with_config.is_ok());
}

/// Test fallback mechanism with invalid service URL
#[tokio::test]
async fn test_fallback_mechanism() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(1000),
        retry_attempts: 1,
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        },
        ..Default::default()
    };
    
    let client = HmmClient::with_config(config).unwrap();
    
    // This should trigger fallback
    let observations = [0.1, 0.2, 0.3];
    let result = client.get_fusion_weights(observations, Some("fallback_test".to_string())).await;
    
    // Should succeed with fallback weights
    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.weights.w_ldc, 0.5);
    assert_eq!(response.weights.w_mr, 0.3);
    assert_eq!(response.weights.w_tsmom, 0.2);
}

/// Test HMM integration fallback
#[tokio::test]
async fn test_integration_fallback() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.4,
            w_tsmom: 0.2,
        },
        timeout: Duration::from_millis(500),
        retry_attempts: 1,
        ..Default::default()
    };
    
    let mut integration = HmmIntegration::with_config(config).unwrap();
    
    let signals = SignalComponents {
        s_ldc: 0.05,
        s_mr: -0.02,
        s_tsmom: 0.08,
    };
    
    // Should gracefully fall back
    let weights = integration.get_fusion_weights_for_signals(&signals).await;
    assert!(weights.is_ok());
    
    let weights = weights.unwrap();
    assert_eq!(weights.w_ldc, 0.4);
    assert_eq!(weights.w_mr, 0.4);
    assert_eq!(weights.w_tsmom, 0.2);
}

/// Test circuit breaker functionality
#[tokio::test]
async fn test_circuit_breaker() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 1,
        enable_fallback: false, // Disable fallback to test circuit breaker
        circuit_breaker_threshold: 2,
        circuit_breaker_timeout: Duration::from_millis(500),
        ..Default::default()
    };
    
    let client = HmmClient::with_config(config).unwrap();
    
    // Make requests to trigger circuit breaker
    let observations = [0.1, 0.2, 0.3];
    
    // First request should fail
    let result1 = client.get_state_probabilities(observations, None).await;
    assert!(result1.is_err());
    
    // Second request should fail and open circuit breaker
    let result2 = client.get_state_probabilities(observations, None).await;
    assert!(result2.is_err());
    
    // Check circuit breaker status
    let (state, failure_count) = client.get_circuit_breaker_status();
    assert_eq!(state, "open");
    assert_eq!(failure_count, 2);
    
    // Third request should fail immediately due to open circuit breaker
    let result3 = client.get_state_probabilities(observations, None).await;
    assert!(result3.is_err());
    
    match result3.unwrap_err() {
        HmmClientError::CircuitBreakerOpen { .. } => {
            // Expected - circuit breaker is open
        }
        HmmClientError::ServiceUnavailable { .. } => {
            // Also acceptable - may happen if circuit breaker just opened
        }
        other => panic!("Expected CircuitBreakerOpen or ServiceUnavailable error, got: {:?}", other),
    }
    
    // Verify metrics
    let metrics = client.get_circuit_breaker_metrics();
    assert_eq!(metrics.circuit_breaker_opens, 1);
    assert!(metrics.rejected_requests > 0);
}

/// Test circuit breaker timeout and half-open state
#[tokio::test]
async fn test_circuit_breaker_timeout_recovery() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 1,
        enable_fallback: false,
        circuit_breaker_threshold: 2,
        circuit_breaker_timeout: Duration::from_millis(300),
        ..Default::default()
    };
    
    let client = HmmClient::with_config(config).unwrap();
    let observations = [0.1, 0.2, 0.3];
    
    // Open the circuit breaker
    let _ = client.get_state_probabilities(observations, None).await;
    let _ = client.get_state_probabilities(observations, None).await;
    
    let (state, _) = client.get_circuit_breaker_status();
    assert_eq!(state, "open");
    
    // Wait for timeout to allow transition to half-open
    tokio::time::sleep(Duration::from_millis(350)).await;
    
    // Next request should attempt in half-open state
    let result = client.get_state_probabilities(observations, None).await;
    assert!(result.is_err()); // Will still fail since service is invalid
    
    // Should have transitioned to half-open and back to open
    let (state, _) = client.get_circuit_breaker_status();
    assert_eq!(state, "open");
    
    // Verify metrics show half-open attempt
    let metrics = client.get_circuit_breaker_metrics();
    assert_eq!(metrics.half_open_attempts, 1);
    assert_eq!(metrics.circuit_breaker_opens, 2); // Initial + after half-open failure
}

/// Test circuit breaker state transitions with detailed logging
#[tokio::test]
async fn test_circuit_breaker_state_transitions() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 1,
        enable_fallback: false,
        circuit_breaker_threshold: 3,
        circuit_breaker_timeout: Duration::from_millis(200),
        ..Default::default()
    };
    
    let client = HmmClient::with_config(config).unwrap();
    let observations = [0.1, 0.2, 0.3];
    
    // Initial state should be closed
    let (state, count) = client.get_circuit_breaker_status();
    assert_eq!(state, "closed");
    assert_eq!(count, 0);
    
    // First failure - should remain closed
    let _ = client.get_state_probabilities(observations, None).await;
    let (state, count) = client.get_circuit_breaker_status();
    assert_eq!(state, "closed");
    assert_eq!(count, 1);
    
    // Second failure - should remain closed
    let _ = client.get_state_probabilities(observations, None).await;
    let (state, count) = client.get_circuit_breaker_status();
    assert_eq!(state, "closed");
    assert_eq!(count, 2);
    
    // Third failure - should open
    let _ = client.get_state_probabilities(observations, None).await;
    let (state, count) = client.get_circuit_breaker_status();
    assert_eq!(state, "open");
    assert_eq!(count, 3);
    
    // Verify metrics
    let metrics = client.get_circuit_breaker_metrics();
    assert_eq!(metrics.failed_requests, 3);
    assert_eq!(metrics.circuit_breaker_opens, 1);
}

/// Test circuit breaker metrics tracking
#[tokio::test]
async fn test_circuit_breaker_metrics_tracking() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 1,
        enable_fallback: false,
        circuit_breaker_threshold: 2,
        circuit_breaker_timeout: Duration::from_millis(200),
        ..Default::default()
    };
    
    let client = HmmClient::with_config(config).unwrap();
    let observations = [0.1, 0.2, 0.3];
    
    // Make several requests to trigger circuit breaker
    let _ = client.get_state_probabilities(observations, None).await;
    let _ = client.get_state_probabilities(observations, None).await;
    
    // Circuit should be open now
    let (state, _) = client.get_circuit_breaker_status();
    assert_eq!(state, "open");
    
    // Try more requests while open (should be rejected)
    let _ = client.get_state_probabilities(observations, None).await;
    let _ = client.get_state_probabilities(observations, None).await;
    
    // Check metrics
    let metrics = client.get_circuit_breaker_metrics();
    assert_eq!(metrics.total_requests, 2); // Only first 2 counted
    assert_eq!(metrics.failed_requests, 2);
    assert_eq!(metrics.successful_requests, 0);
    assert_eq!(metrics.circuit_breaker_opens, 1);
    assert!(metrics.rejected_requests >= 2); // At least 2 rejected
}

/// Test request serialization and validation
#[tokio::test]
async fn test_request_validation() {
    use signal_fusion::hmm_client::InferenceRequest;
    use serde_json;
    
    let request = InferenceRequest {
        observations: [0.1, -0.05, 0.08],
        timestamp: Some(1234567890),
        request_id: Some("test_123".to_string()),
    };
    
    // Test serialization
    let json = serde_json::to_string(&request);
    assert!(json.is_ok());
    
    let json_str = json.unwrap();
    assert!(json_str.contains("observations"));
    assert!(json_str.contains("timestamp"));
    assert!(json_str.contains("request_id"));
    
    // Test that observations are properly formatted
    assert!(json_str.contains("[0.1,-0.05,0.08]"));
}

/// Test configuration validation
#[test]
fn test_config_validation() {
    let config = HmmClientConfig::default();
    
    // Verify default values
    assert_eq!(config.base_url.as_str(), "http://localhost:8000/");
    assert_eq!(config.timeout, Duration::from_millis(5000));
    assert_eq!(config.retry_attempts, 3);
    assert_eq!(config.retry_delay, Duration::from_millis(100));
    assert!(config.enable_fallback);
    assert_eq!(config.circuit_breaker_threshold, 5);
    
    // Verify fallback weights sum to 1
    let total = config.fallback_weights.w_ldc + 
                config.fallback_weights.w_mr + 
                config.fallback_weights.w_tsmom;
    assert!((total - 1.0).abs() < 1e-6);
}

/// Test error handling for different scenarios
#[tokio::test]
async fn test_error_handling() {
    // Test with very short timeout to trigger timeout error
    let config = HmmClientConfig {
        base_url: "http://httpbin.org/delay/2".parse().unwrap(), // 2 second delay
        timeout: Duration::from_millis(100), // 100ms timeout
        retry_attempts: 1,
        enable_fallback: false,
        ..Default::default()
    };
    
    let client = HmmClient::with_config(config).unwrap();
    
    let observations = [0.1, 0.2, 0.3];
    let result = client.get_state_probabilities(observations, None).await;
    
    // Should fail due to timeout or network error
    assert!(result.is_err());
    
    // Verify error type
    match result.unwrap_err() {
        HmmClientError::Network(_) => {
            // Expected - network timeout or connection error
        }
        HmmClientError::ServiceUnavailable { .. } => {
            // Also acceptable - service might be unavailable
        }
        other => {
            // Log unexpected error type but don't fail test
            println!("Unexpected error type: {:?}", other);
        }
    }
}

/// Integration test with actual HMM service (requires running service)
#[tokio::test]
#[ignore] // Ignored by default - run with `cargo test -- --ignored`
async fn test_real_hmm_service_integration() {
    let mut integration = HmmIntegration::new().unwrap();
    
    // Check if service is available
    let service_ready = integration.is_service_ready().await;
    if !service_ready {
        println!("HMM service not ready, skipping integration test");
        return;
    }
    
    // Test with realistic market data
    let signals = SignalComponents {
        s_ldc: 0.015,   // Positive LDC signal
        s_mr: -0.008,   // Negative mean reversion
        s_tsmom: 0.022, // Positive momentum
    };
    
    // Get fusion weights
    let weights = integration.get_fusion_weights_for_signals(&signals).await;
    assert!(weights.is_ok());
    
    let weights = weights.unwrap();
    
    // Verify weights are valid probabilities
    let total = weights.w_ldc + weights.w_mr + weights.w_tsmom;
    assert!((total - 1.0).abs() < 1e-6, "Weights don't sum to 1: {}", total);
    
    assert!(weights.w_ldc >= 0.0 && weights.w_ldc <= 1.0);
    assert!(weights.w_mr >= 0.0 && weights.w_mr <= 1.0);
    assert!(weights.w_tsmom >= 0.0 && weights.w_tsmom <= 1.0);
    
    println!("Integration test successful - weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
             weights.w_ldc, weights.w_mr, weights.w_tsmom);
}

/// Test service health checks (requires running service)
#[tokio::test]
#[ignore] // Ignored by default
async fn test_service_health_checks() {
    let client = HmmClient::new().unwrap();
    
    // Test health check
    match client.health_check().await {
        Ok(health) => {
            assert_eq!(health.status, "healthy");
            println!("Health check passed: {}", health.status);
        }
        Err(e) => {
            println!("Health check failed (service may not be running): {}", e);
            return; // Skip rest of test if service not available
        }
    }
    
    // Test readiness check
    match client.readiness_check().await {
        Ok(readiness) => {
            println!("Readiness check: ready={}, model_loaded={}", 
                     readiness.ready, readiness.model_loaded);
        }
        Err(e) => {
            println!("Readiness check failed: {}", e);
        }
    }
    
    // Test model info
    match client.get_model_info().await {
        Ok(model_info) => {
            println!("Model info: loaded={}, version={:?}, states={:?}", 
                     model_info.loaded, model_info.version, model_info.n_states);
        }
        Err(e) => {
            println!("Model info request failed: {}", e);
        }
    }
}

/// Benchmark test for performance validation (requires running service)
#[tokio::test]
#[ignore] // Ignored by default
async fn test_performance_benchmark() {
    let mut integration = HmmIntegration::new().unwrap();
    
    if !integration.is_service_ready().await {
        println!("HMM service not ready, skipping performance test");
        return;
    }
    
    let signals = SignalComponents {
        s_ldc: 0.01,
        s_mr: 0.02,
        s_tsmom: 0.03,
    };
    
    let num_requests = 10;
    let start_time = std::time::Instant::now();
    
    for i in 0..num_requests {
        let test_signals = SignalComponents {
            s_ldc: signals.s_ldc + (i as f32 * 0.001),
            s_mr: signals.s_mr + (i as f32 * 0.001),
            s_tsmom: signals.s_tsmom + (i as f32 * 0.001),
        };
        
        let result = integration.get_fusion_weights_for_signals(&test_signals).await;
        assert!(result.is_ok(), "Request {} failed: {:?}", i, result.err());
    }
    
    let total_duration = start_time.elapsed();
    let avg_duration = total_duration / num_requests;
    
    println!("Performance test: {} requests in {:?} (avg: {:?})", 
             num_requests, total_duration, avg_duration);
    
    // Verify average latency is reasonable (should be < 50ms for local service)
    assert!(avg_duration < Duration::from_millis(100), 
            "Average latency too high: {:?}", avg_duration);
}

/// Test cache integration with fallback
#[tokio::test]
async fn test_cache_integration_with_fallback() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 1,
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        },
        ..Default::default()
    };
    
    let mut integration = HmmIntegration::with_config(config).unwrap();
    
    let signals = SignalComponents {
        s_ldc: 0.05,
        s_mr: -0.02,
        s_tsmom: 0.08,
    };
    
    // First request - should use fallback and cache it
    let weights1 = integration.get_fusion_weights_for_signals(&signals).await;
    assert!(weights1.is_ok());
    
    // Check cache stats - should have 1 miss (no cache hit on first request)
    let stats = integration.get_cache_stats();
    assert_eq!(stats.misses, 1);
    
    // Second request with same signals - should hit cache
    let weights2 = integration.get_fusion_weights_for_signals(&signals).await;
    assert!(weights2.is_ok());
    
    // Check cache stats - should have 1 hit now
    let stats = integration.get_cache_stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.size, 1);
    
    // Verify weights are the same
    let w1 = weights1.unwrap();
    let w2 = weights2.unwrap();
    assert_eq!(w1.w_ldc, w2.w_ldc);
    assert_eq!(w1.w_mr, w2.w_mr);
    assert_eq!(w1.w_tsmom, w2.w_tsmom);
}

/// Test cache with different observations
#[tokio::test]
async fn test_cache_with_different_observations() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 1,
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        },
        ..Default::default()
    };
    
    let mut integration = HmmIntegration::with_config(config).unwrap();
    
    // Request with first set of signals
    let signals1 = SignalComponents {
        s_ldc: 0.05,
        s_mr: -0.02,
        s_tsmom: 0.08,
    };
    let _ = integration.get_fusion_weights_for_signals(&signals1).await;
    
    // Request with second set of signals
    let signals2 = SignalComponents {
        s_ldc: 0.10,
        s_mr: 0.03,
        s_tsmom: -0.05,
    };
    let _ = integration.get_fusion_weights_for_signals(&signals2).await;
    
    // Check cache stats - should have 2 misses, 0 hits, 2 entries
    let stats = integration.get_cache_stats();
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.size, 2);
    
    // Request first signals again - should hit cache
    let _ = integration.get_fusion_weights_for_signals(&signals1).await;
    
    let stats = integration.get_cache_stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 2);
}

/// Test cache cleanup functionality
#[tokio::test]
async fn test_cache_cleanup() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 1,
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        },
        ..Default::default()
    };
    
    // Create integration with short TTL for testing
    let mut integration = HmmIntegration::with_config_and_cache(
        config,
        Duration::from_millis(100), // 100ms TTL
        1000,
    ).unwrap();
    
    let signals = SignalComponents {
        s_ldc: 0.05,
        s_mr: -0.02,
        s_tsmom: 0.08,
    };
    
    // Add entry to cache
    let _ = integration.get_fusion_weights_for_signals(&signals).await;
    assert_eq!(integration.get_cache_stats().size, 1);
    
    // Wait for TTL to expire
    tokio::time::sleep(Duration::from_millis(150)).await;
    
    // Manually trigger cleanup
    integration.cleanup_cache();
    
    // Cache should be empty now
    let stats = integration.get_cache_stats();
    assert_eq!(stats.size, 0);
    assert_eq!(stats.evictions, 1);
}

/// Test cache clear functionality
#[tokio::test]
async fn test_cache_clear() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 1,
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        },
        ..Default::default()
    };
    
    let mut integration = HmmIntegration::with_config(config).unwrap();
    
    // Add multiple entries
    for i in 0..5 {
        let signals = SignalComponents {
            s_ldc: 0.01 * i as f32,
            s_mr: 0.02 * i as f32,
            s_tsmom: 0.03 * i as f32,
        };
        let _ = integration.get_fusion_weights_for_signals(&signals).await;
    }
    
    assert_eq!(integration.get_cache_stats().size, 5);
    
    // Clear cache
    integration.clear_cache();
    
    // Cache should be empty
    assert_eq!(integration.get_cache_stats().size, 0);
}

/// Test cache hit rate calculation
#[tokio::test]
async fn test_cache_hit_rate() {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 1,
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        },
        ..Default::default()
    };
    
    let mut integration = HmmIntegration::with_config(config).unwrap();
    
    let signals = SignalComponents {
        s_ldc: 0.05,
        s_mr: -0.02,
        s_tsmom: 0.08,
    };
    
    // First request - miss
    let _ = integration.get_fusion_weights_for_signals(&signals).await;
    
    // Next 3 requests - hits
    for _ in 0..3 {
        let _ = integration.get_fusion_weights_for_signals(&signals).await;
    }
    
    let stats = integration.get_cache_stats();
    assert_eq!(stats.hits, 3);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hit_rate, 0.75); // 3 hits out of 4 total requests
}

/// Test cache with real HMM service (requires running service)
#[tokio::test]
#[ignore] // Ignored by default
async fn test_cache_with_real_service() {
    let mut integration = HmmIntegration::new().unwrap();
    
    if !integration.is_service_ready().await {
        println!("HMM service not ready, skipping cache integration test");
        return;
    }
    
    let signals = SignalComponents {
        s_ldc: 0.015,
        s_mr: -0.008,
        s_tsmom: 0.022,
    };
    
    // First request - should miss cache and fetch from service
    let start = std::time::Instant::now();
    let weights1 = integration.get_fusion_weights_for_signals(&signals).await;
    let first_duration = start.elapsed();
    assert!(weights1.is_ok());
    
    let stats = integration.get_cache_stats();
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 0);
    
    // Second request - should hit cache (much faster)
    let start = std::time::Instant::now();
    let weights2 = integration.get_fusion_weights_for_signals(&signals).await;
    let second_duration = start.elapsed();
    assert!(weights2.is_ok());
    
    let stats = integration.get_cache_stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    
    // Cache hit should be significantly faster
    println!("First request (cache miss): {:?}", first_duration);
    println!("Second request (cache hit): {:?}", second_duration);
    println!("Cache hit rate: {:.2}%", stats.hit_rate * 100.0);
    
    // Verify cache hit is faster (should be < 1ms vs potentially 10-50ms for service call)
    assert!(second_duration < first_duration / 2, 
            "Cache hit should be significantly faster than service call");
}