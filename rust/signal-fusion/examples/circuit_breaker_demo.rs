//! Circuit Breaker Demonstration
//!
//! This example demonstrates the enhanced circuit breaker functionality
//! including state transitions, timeout-based recovery, and metrics tracking.

use signal_fusion::hmm_client::{HmmClient, HmmClientConfig};
use std::time::Duration;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Circuit Breaker Demonstration ===\n");

    // Configure client with low threshold for demonstration
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(), // Invalid URL to trigger failures
        timeout: Duration::from_millis(500),
        retry_attempts: 1,
        enable_fallback: true,
        circuit_breaker_threshold: 3,
        circuit_breaker_timeout: Duration::from_secs(2),
        ..Default::default()
    };

    let client = HmmClient::with_config(config)?;
    let observations = [0.1, 0.2, 0.3];

    // Phase 1: Demonstrate circuit breaker opening
    println!("Phase 1: Triggering circuit breaker to open");
    println!("-------------------------------------------");

    for i in 1..=3 {
        println!("\nAttempt {}: Making request...", i);
        let result = client.get_fusion_weights(observations, Some(format!("req_{}", i))).await;
        
        let (state, failure_count) = client.get_circuit_breaker_status();
        
        if result.is_ok() {
            let response = result.unwrap();
            println!("  User-facing result: Success (using fallback weights)");
            println!("  Weights: LDC={:.2}, MR={:.2}, TSMOM={:.2}", 
                     response.weights.w_ldc, response.weights.w_mr, response.weights.w_tsmom);
        } else {
            println!("  User-facing result: Failed");
        }
        
        println!("  Circuit breaker state: {}", state);
        println!("  Failure count: {}", failure_count);
        println!("  Note: Service call failed, but fallback provided valid response");
    }

    // Phase 2: Demonstrate circuit breaker rejecting requests
    println!("\n\nPhase 2: Circuit breaker is open - requests rejected");
    println!("-----------------------------------------------------");

    for i in 1..=2 {
        println!("\nAttempt {}: Making request while circuit is open...", i);
        let result = client.get_fusion_weights(observations, Some(format!("req_open_{}", i))).await;
        
        let (state, _) = client.get_circuit_breaker_status();
        println!("  Result: {:?}", if result.is_ok() { "Success (fallback)" } else { "Rejected" });
        println!("  Circuit breaker state: {}", state);
    }

    // Phase 3: Wait for timeout and demonstrate half-open state
    println!("\n\nPhase 3: Waiting for circuit breaker timeout...");
    println!("------------------------------------------------");
    println!("Waiting 2.5 seconds for timeout-based recovery...");
    tokio::time::sleep(Duration::from_millis(2500)).await;

    println!("\nMaking test request in half-open state...");
    let result = client.get_fusion_weights(observations, Some("req_halfopen".to_string())).await;
    
    let (state, _) = client.get_circuit_breaker_status();
    println!("  Result: {:?}", if result.is_ok() { "Success (fallback)" } else { "Failed" });
    println!("  Circuit breaker state: {}", state);

    // Phase 4: Display metrics
    println!("\n\nPhase 4: Circuit Breaker Metrics");
    println!("----------------------------------");
    
    let metrics = client.get_circuit_breaker_metrics();
    println!("Total service requests attempted: {}", metrics.total_requests);
    println!("Successful service requests: {}", metrics.successful_requests);
    println!("Failed service requests: {}", metrics.failed_requests);
    println!("Circuit breaker opens: {}", metrics.circuit_breaker_opens);
    println!("Circuit breaker closes: {}", metrics.circuit_breaker_closes);
    println!("Half-open attempts: {}", metrics.half_open_attempts);
    println!("Rejected requests (circuit open): {}", metrics.rejected_requests);

    // Calculate service success rate
    if metrics.total_requests > 0 {
        let service_success_rate = (metrics.successful_requests as f64 / metrics.total_requests as f64) * 100.0;
        println!("\nService success rate: {:.2}%", service_success_rate);
        println!("(Note: 0% is expected since we used an invalid service URL)");
        println!("User-facing success rate: 100% (thanks to fallback mechanism)");
    }

    println!("\n=== Key Takeaways ===");
    println!("1. Circuit breaker tracks actual service health (0% success)");
    println!("2. Fallback mechanism ensures users always get valid responses");
    println!("3. Circuit breaker prevents wasting resources on failing service");
    println!("4. After timeout, circuit breaker attempts recovery (half-open state)");

    println!("\n=== Demonstration Complete ===");

    Ok(())
}
