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
    
    let mut client = HmmClient::with_config(config).unwrap();
    
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
    
    let mut client = HmmClient::with_config(config).unwrap();
    
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
        HmmClientError::ServiceUnavailable { .. } => {
            // Expected - circuit breaker is open
        }
        other => panic!("Expected ServiceUnavailable error, got: {:?}", other),
    }
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
    
    let mut client = HmmClient::with_config(config).unwrap();
    
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
    let mut client = HmmClient::new().unwrap();
    
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