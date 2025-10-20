//! Monitoring and Metrics Demo
//!
//! This example demonstrates the comprehensive monitoring and metrics capabilities
//! of the HMM integration, including:
//! - Request metrics (count, duration, errors)
//! - Cache metrics (hits, misses, size, evictions)
//! - Circuit breaker state metrics
//! - Fallback activation metrics
//! - Metrics export in JSON and Prometheus formats

use signal_fusion::{
    SignalComponents, MetricsFormat,
    hmm_client::{HmmClientConfig, HmmIntegration},
};
use std::time::Duration;
use url::Url;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== HMM Integration Monitoring Demo ===\n");

    // Configure HMM client with custom settings
    let config = HmmClientConfig {
        base_url: Url::parse("http://localhost:8000")?,
        timeout: Duration::from_millis(5000),
        retry_attempts: 3,
        retry_delay: Duration::from_millis(100),
        enable_fallback: true,
        fallback_weights: signal_fusion::FusionWeights {
            w_ldc: 0.33,
            w_mr: 0.33,
            w_tsmom: 0.34,
        },
        circuit_breaker_threshold: 5,
        circuit_breaker_timeout: Duration::from_secs(30),
    };

    // Create HMM integration with custom cache settings
    let mut integration = HmmIntegration::with_config_and_cache(
        config,
        Duration::from_secs(60),  // 60 second TTL
        1000,                      // Max 1000 entries
    )?;

    println!("✓ HMM Integration initialized\n");

    // Simulate some signal processing requests
    println!("--- Simulating Signal Processing ---\n");

    let test_signals = vec![
        SignalComponents { s_ldc: 0.5, s_mr: 0.3, s_tsmom: 0.2 },
        SignalComponents { s_ldc: -0.4, s_mr: 0.1, s_tsmom: 0.3 },
        SignalComponents { s_ldc: 0.2, s_mr: -0.3, s_tsmom: 0.5 },
        SignalComponents { s_ldc: 0.5, s_mr: 0.3, s_tsmom: 0.2 }, // Duplicate for cache hit
        SignalComponents { s_ldc: 0.1, s_mr: 0.1, s_tsmom: 0.1 },
    ];

    for (i, signals) in test_signals.iter().enumerate() {
        println!("Request {}: Processing signals [LDC:{:.2}, MR:{:.2}, TSMOM:{:.2}]",
                 i + 1, signals.s_ldc, signals.s_mr, signals.s_tsmom);

        match integration.get_fusion_weights_for_signals(signals).await {
            Ok(weights) => {
                println!("  ✓ Weights: [LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}]",
                         weights.w_ldc, weights.w_mr, weights.w_tsmom);
            }
            Err(e) => {
                println!("  ✗ Error: {}", e);
            }
        }
    }

    println!("\n--- Current Metrics ---\n");

    // Get comprehensive metrics
    let metrics = integration.get_metrics();

    // Display request metrics
    println!("Request Metrics:");
    println!("  Total Requests:      {}", metrics.requests.total_requests);
    println!("  Successful:          {}", metrics.requests.successful_requests);
    println!("  Failed:              {}", metrics.requests.failed_requests);
    println!("  Avg Duration:        {:.2}ms", metrics.requests.avg_duration_ms);
    println!("  Min Duration:        {}ms", metrics.requests.min_duration_ms);
    println!("  Max Duration:        {}ms", metrics.requests.max_duration_ms);
    println!("  Timeout Errors:      {}", metrics.requests.timeout_errors);
    println!("  Network Errors:      {}", metrics.requests.network_errors);
    println!("  Validation Errors:   {}", metrics.requests.validation_errors);

    // Display cache metrics
    println!("\nCache Metrics:");
    println!("  Cache Hits:          {}", metrics.cache.hits);
    println!("  Cache Misses:        {}", metrics.cache.misses);
    println!("  Cache Size:          {}", metrics.cache.size);
    println!("  Cache Evictions:     {}", metrics.cache.evictions);
    println!("  Hit Rate:            {:.1}%", metrics.cache.hit_rate * 100.0);

    // Display circuit breaker metrics
    println!("\nCircuit Breaker Metrics:");
    println!("  Total Requests:      {}", metrics.circuit_breaker.total_requests);
    println!("  Successful:          {}", metrics.circuit_breaker.successful_requests);
    println!("  Failed:              {}", metrics.circuit_breaker.failed_requests);
    println!("  Opens:               {}", metrics.circuit_breaker.circuit_breaker_opens);
    println!("  Closes:              {}", metrics.circuit_breaker.circuit_breaker_closes);
    println!("  Half-Open Attempts:  {}", metrics.circuit_breaker.half_open_attempts);
    println!("  Rejected Requests:   {}", metrics.circuit_breaker.rejected_requests);

    // Display fallback metrics
    println!("\nFallback Metrics:");
    println!("  Total Activations:   {}", metrics.fallback.total_activations);
    println!("  Circuit Breaker:     {}", metrics.fallback.circuit_breaker_activations);
    println!("  Network Errors:      {}", metrics.fallback.network_error_activations);
    println!("  Timeouts:            {}", metrics.fallback.timeout_activations);
    println!("  Service Errors:      {}", metrics.fallback.service_error_activations);
    println!("  Currently Active:    {}", metrics.fallback.currently_active);

    // Display system metrics
    println!("\nSystem Metrics:");
    println!("  Uptime:              {}s", metrics.uptime_seconds);
    println!("  Timestamp:           {}", metrics.timestamp);

    // Export metrics in JSON format
    println!("\n--- Metrics Export (JSON) ---\n");
    match integration.export_metrics(MetricsFormat::Json) {
        Ok(json) => {
            println!("{}", json);
        }
        Err(e) => {
            eprintln!("Failed to export JSON metrics: {}", e);
        }
    }

    // Export metrics in Prometheus format
    println!("\n--- Metrics Export (Prometheus) ---\n");
    match integration.export_metrics(MetricsFormat::Prometheus) {
        Ok(prometheus) => {
            // Show first 20 lines
            for line in prometheus.lines().take(20) {
                println!("{}", line);
            }
            println!("... (truncated)");
        }
        Err(e) => {
            eprintln!("Failed to export Prometheus metrics: {}", e);
        }
    }

    // Get circuit breaker status
    let (cb_state, cb_failures) = integration.get_circuit_breaker_status();
    println!("\n--- Circuit Breaker Status ---");
    println!("  State:               {}", cb_state);
    println!("  Failure Count:       {}", cb_failures);

    // Get cache statistics
    let cache_stats = integration.get_cache_stats();
    println!("\n--- Cache Statistics ---");
    println!("  Current Size:        {}", cache_stats.size);
    println!("  Hit Rate:            {:.1}%", cache_stats.hit_rate * 100.0);
    println!("  Total Hits:          {}", cache_stats.hits);
    println!("  Total Misses:        {}", cache_stats.misses);
    println!("  Total Evictions:     {}", cache_stats.evictions);

    println!("\n=== Demo Complete ===");

    Ok(())
}
