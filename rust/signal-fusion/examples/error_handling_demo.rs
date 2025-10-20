//! Error Handling Demonstration
//!
//! This example demonstrates the comprehensive error handling capabilities
//! of the HMM client, including:
//! - Exponential backoff retry logic
//! - Error classification (transient vs permanent)
//! - Structured error context for debugging
//! - Fallback activation logging
//! - Circuit breaker integration

use signal_fusion::{
    hmm_client::{HmmClient, HmmClientConfig, HmmClientError},
    FusionWeights,
};
use std::time::Duration;
use tracing::Level;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging to see detailed error messages
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();

    println!("=== HMM Client Error Handling Demo ===\n");

    // Example 1: Transient vs Permanent Error Classification
    println!("1. Error Classification Demo");
    println!("----------------------------");
    demonstrate_error_classification().await;
    println!();

    // Example 2: Exponential Backoff Retry Logic
    println!("2. Exponential Backoff Demo");
    println!("---------------------------");
    demonstrate_exponential_backoff().await;
    println!();

    // Example 3: Structured Error Context
    println!("3. Structured Error Context Demo");
    println!("--------------------------------");
    demonstrate_error_context().await;
    println!();

    // Example 4: Fallback Activation
    println!("4. Fallback Activation Demo");
    println!("---------------------------");
    demonstrate_fallback_activation().await;
    println!();

    // Example 5: Circuit Breaker Integration
    println!("5. Circuit Breaker Integration Demo");
    println!("-----------------------------------");
    demonstrate_circuit_breaker().await;
    println!();

    Ok(())
}

/// Demonstrate error classification (transient vs permanent)
async fn demonstrate_error_classification() {
    println!("Testing error classification...\n");

    // Create errors of different types
    let errors = vec![
        HmmClientError::InvalidRequest {
            message: "Bad request format".to_string(),
        },
        HmmClientError::ValidationError {
            field: "observations".to_string(),
            message: "Invalid observation values".to_string(),
        },
        HmmClientError::ServiceUnavailable {
            status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
        },
        HmmClientError::ServiceUnavailable {
            status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
        },
        HmmClientError::Timeout {
            duration: Duration::from_secs(5),
        },
        HmmClientError::CircuitBreakerOpen {
            reason: "Too many failures".to_string(),
        },
    ];

    for error in errors {
        let context = error.error_context();
        println!("Error: {}", error);
        println!("  Type: {}", context.error_type);
        println!("  Is Transient: {}", context.is_transient);
        println!("  Retry Recommended: {}", context.retry_recommended);
        println!("  Details: {}", context.details);
        println!();
    }
}

/// Demonstrate exponential backoff retry logic
async fn demonstrate_exponential_backoff() {
    println!("Testing exponential backoff with invalid service...\n");

    let config = HmmClientConfig {
        base_url: "http://invalid-host-12345.example.com:9999"
            .parse()
            .unwrap(),
        timeout: Duration::from_millis(500),
        retry_attempts: 3,
        retry_delay: Duration::from_millis(100), // Base delay
        enable_fallback: false,
        circuit_breaker_threshold: 10, // High threshold to avoid circuit breaker
        ..Default::default()
    };

    let client = HmmClient::with_config(config).unwrap();
    let observations = [0.1, 0.2, 0.3];

    println!("Making request with 3 retry attempts...");
    println!("Expected backoff delays: 100ms, 200ms, 400ms\n");

    let start = std::time::Instant::now();
    let result = client
        .get_fusion_weights(observations, Some("backoff_test".to_string()))
        .await;
    let duration = start.elapsed();

    match result {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => {
            let context = e.error_context();
            println!("Request failed after {:?}", duration);
            println!("Error type: {}", context.error_type);
            println!("Error message: {}", context.message);
            println!("Total time includes exponential backoff delays");
        }
    }
}

/// Demonstrate structured error context
async fn demonstrate_error_context() {
    println!("Testing structured error context...\n");

    // Create a client with very short timeout to trigger timeout error
    let config = HmmClientConfig {
        base_url: "http://httpbin.org/delay/2".parse().unwrap(), // 2 second delay
        timeout: Duration::from_millis(100), // 100ms timeout
        retry_attempts: 1,
        enable_fallback: false,
        ..Default::default()
    };

    let client = HmmClient::with_config(config).unwrap();
    let observations = [0.1, 0.2, 0.3];

    let result = client
        .get_fusion_weights(observations, Some("context_test".to_string()))
        .await;

    match result {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => {
            let context = e.error_context();
            println!("Structured Error Context:");
            println!("  error_type: {}", context.error_type);
            println!("  is_transient: {}", context.is_transient);
            println!("  message: {}", context.message);
            println!("  details: {}", context.details);
            println!("  retry_recommended: {}", context.retry_recommended);
            println!("\nThis structured context can be:");
            println!("  - Logged to monitoring systems");
            println!("  - Used for alerting and diagnostics");
            println!("  - Serialized to JSON for analysis");
        }
    }
}

/// Demonstrate fallback activation with enhanced logging
async fn demonstrate_fallback_activation() {
    println!("Testing fallback activation...\n");

    let fallback_weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.35,
        w_tsmom: 0.25,
    };

    let config = HmmClientConfig {
        base_url: "http://invalid-host-12345.example.com:9999"
            .parse()
            .unwrap(),
        timeout: Duration::from_millis(500),
        retry_attempts: 2,
        enable_fallback: true,
        fallback_weights: fallback_weights.clone(),
        circuit_breaker_threshold: 10,
        ..Default::default()
    };

    let client = HmmClient::with_config(config).unwrap();
    let observations = [0.05, -0.02, 0.08];

    println!("Making request to invalid service with fallback enabled...");
    println!("Expected: Fallback weights will be used\n");

    let result = client
        .get_fusion_weights(observations, Some("fallback_test".to_string()))
        .await;

    match result {
        Ok(response) => {
            println!("✓ Fallback activated successfully!");
            println!("  Fallback weights:");
            println!("    w_ldc: {:.3}", response.weights.w_ldc);
            println!("    w_mr: {:.3}", response.weights.w_mr);
            println!("    w_tsmom: {:.3}", response.weights.w_tsmom);
            println!("\nCheck the logs above for detailed fallback activation messages");
        }
        Err(e) => {
            println!("Unexpected error: {}", e);
        }
    }
}

/// Demonstrate circuit breaker integration with error handling
async fn demonstrate_circuit_breaker() {
    println!("Testing circuit breaker integration...\n");

    let config = HmmClientConfig {
        base_url: "http://invalid-host-12345.example.com:9999"
            .parse()
            .unwrap(),
        timeout: Duration::from_millis(500),
        retry_attempts: 1,
        enable_fallback: false,
        circuit_breaker_threshold: 3, // Open after 3 failures
        circuit_breaker_timeout: Duration::from_secs(2),
        ..Default::default()
    };

    let client = HmmClient::with_config(config).unwrap();
    let observations = [0.1, 0.2, 0.3];

    println!("Making requests to trigger circuit breaker...");
    println!("Circuit breaker threshold: 3 failures\n");

    // Make requests to trigger circuit breaker
    for i in 1..=5 {
        println!("Request {}:", i);
        let result = client
            .get_fusion_weights(observations, Some(format!("cb_test_{}", i)))
            .await;

        match result {
            Ok(_) => println!("  ✓ Success (unexpected)"),
            Err(e) => {
                let context = e.error_context();
                println!("  ✗ Failed: {}", context.error_type);
                
                if matches!(e, HmmClientError::CircuitBreakerOpen { .. }) {
                    println!("  → Circuit breaker is now OPEN");
                    println!("  → Subsequent requests will be rejected immediately");
                }
            }
        }

        let (state, failures) = client.get_circuit_breaker_status();
        println!("  Circuit breaker state: {} (failures: {})", state, failures);
        
        let metrics = client.get_circuit_breaker_metrics();
        println!("  Metrics: total={}, failed={}, rejected={}", 
                 metrics.total_requests, metrics.failed_requests, metrics.rejected_requests);
        println!();

        // Small delay between requests
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    println!("Circuit breaker demo complete!");
    println!("Check the logs for detailed state transition messages");
}
