# Monitoring and Metrics Guide

This guide covers the comprehensive monitoring and metrics capabilities of the HMM integration component.

## Overview

The HMM integration provides detailed metrics tracking across four key areas:

1. **Request Metrics** - Track HTTP requests to the HMM service
2. **Cache Metrics** - Monitor weight cache performance
3. **Circuit Breaker Metrics** - Track circuit breaker state and transitions
4. **Fallback Metrics** - Monitor fallback activation patterns

## Metrics Categories

### Request Metrics

Tracks all HTTP requests made to the HMM service:

- **total_requests**: Total number of requests made
- **successful_requests**: Number of successful requests (2xx responses)
- **failed_requests**: Number of failed requests
- **total_duration_ms**: Cumulative request duration
- **avg_duration_ms**: Average request duration
- **min_duration_ms**: Minimum request duration
- **max_duration_ms**: Maximum request duration
- **timeout_errors**: Number of timeout errors
- **network_errors**: Number of network errors
- **validation_errors**: Number of validation errors

### Cache Metrics

Monitors the performance of the weight cache:

- **hits**: Number of cache hits
- **misses**: Number of cache misses
- **size**: Current number of entries in cache
- **evictions**: Total number of evicted entries
- **hit_rate**: Cache hit rate (0.0 to 1.0)

### Circuit Breaker Metrics

Tracks circuit breaker behavior and state transitions:

- **total_requests**: Total requests through circuit breaker
- **successful_requests**: Successful requests
- **failed_requests**: Failed requests
- **circuit_breaker_opens**: Number of times circuit opened
- **circuit_breaker_closes**: Number of times circuit closed
- **half_open_attempts**: Number of half-open test attempts
- **rejected_requests**: Requests rejected while circuit open

### Fallback Metrics

Monitors fallback weight activation:

- **total_activations**: Total fallback activations
- **circuit_breaker_activations**: Fallbacks due to circuit breaker
- **network_error_activations**: Fallbacks due to network errors
- **timeout_activations**: Fallbacks due to timeouts
- **service_error_activations**: Fallbacks due to service errors
- **currently_active**: Whether fallback is currently active

## Usage

### Basic Metrics Collection

```rust
use signal_fusion::hmm_client::HmmIntegration;

// Create integration
let mut integration = HmmIntegration::new()?;

// Process some signals
let signals = SignalComponents {
    s_ldc: 0.5,
    s_mr: 0.3,
    s_tsmom: 0.2,
};
let weights = integration.get_fusion_weights_for_signals(&signals).await?;

// Get comprehensive metrics
let metrics = integration.get_metrics();

println!("Total Requests: {}", metrics.requests.total_requests);
println!("Cache Hit Rate: {:.1}%", metrics.cache.hit_rate * 100.0);
println!("Circuit Breaker Opens: {}", metrics.circuit_breaker.circuit_breaker_opens);
println!("Fallback Activations: {}", metrics.fallback.total_activations);
```

### Exporting Metrics

#### JSON Format

```rust
use signal_fusion::MetricsFormat;

let json = integration.export_metrics(MetricsFormat::Json)?;
println!("{}", json);
```

Example JSON output:

```json
{
  "requests": {
    "total_requests": 100,
    "successful_requests": 95,
    "failed_requests": 5,
    "total_duration_ms": 5000,
    "avg_duration_ms": 50.0,
    "min_duration_ms": 10,
    "max_duration_ms": 200,
    "timeout_errors": 2,
    "network_errors": 2,
    "validation_errors": 1
  },
  "cache": {
    "hits": 80,
    "misses": 20,
    "size": 50,
    "evictions": 5,
    "hit_rate": 0.8
  },
  "circuit_breaker": {
    "total_requests": 100,
    "successful_requests": 95,
    "failed_requests": 5,
    "circuit_breaker_opens": 1,
    "circuit_breaker_closes": 1,
    "half_open_attempts": 1,
    "rejected_requests": 0
  },
  "fallback": {
    "total_activations": 5,
    "circuit_breaker_activations": 2,
    "network_error_activations": 2,
    "timeout_activations": 1,
    "service_error_activations": 0,
    "currently_active": false
  },
  "timestamp": 1234567890,
  "uptime_seconds": 3600
}
```

#### Prometheus Format

```rust
let prometheus = integration.export_metrics(MetricsFormat::Prometheus)?;
println!("{}", prometheus);
```

Example Prometheus output:

```
# HELP hmm_requests_total Total number of HMM service requests
# TYPE hmm_requests_total counter
hmm_requests_total 100

# HELP hmm_requests_successful Number of successful HMM service requests
# TYPE hmm_requests_successful counter
hmm_requests_successful 95

# HELP hmm_cache_hits Total number of cache hits
# TYPE hmm_cache_hits counter
hmm_cache_hits 80

# HELP hmm_cache_hit_rate Cache hit rate (0.0 to 1.0)
# TYPE hmm_cache_hit_rate gauge
hmm_cache_hit_rate 0.8

# HELP hmm_circuit_breaker_opens Number of times circuit breaker opened
# TYPE hmm_circuit_breaker_opens counter
hmm_circuit_breaker_opens 1

# HELP hmm_fallback_activations_total Total number of fallback activations
# TYPE hmm_fallback_activations_total counter
hmm_fallback_activations_total 5
```

### Monitoring Specific Components

#### Cache Statistics

```rust
let cache_stats = integration.get_cache_stats();
println!("Cache Size: {}", cache_stats.size);
println!("Hit Rate: {:.1}%", cache_stats.hit_rate * 100.0);
```

#### Circuit Breaker Status

```rust
let (state, failures) = integration.get_circuit_breaker_status();
println!("Circuit Breaker State: {}", state);
println!("Failure Count: {}", failures);
```

#### Request Metrics

```rust
let metrics_collector = integration.metrics_collector();
let request_metrics = metrics_collector.get_request_metrics();
println!("Average Duration: {:.2}ms", request_metrics.avg_duration_ms);
```

## Integration with Monitoring Systems

### Prometheus Integration

To expose metrics to Prometheus, create an HTTP endpoint that serves the Prometheus format:

```rust
use axum::{routing::get, Router};

async fn metrics_handler(
    State(integration): State<Arc<Mutex<HmmIntegration>>>,
) -> String {
    let integration = integration.lock().await;
    integration.export_metrics(MetricsFormat::Prometheus)
        .unwrap_or_else(|e| format!("Error: {}", e))
}

let app = Router::new()
    .route("/metrics", get(metrics_handler))
    .with_state(Arc::new(Mutex::new(integration)));
```

### JSON API Integration

For JSON-based monitoring systems:

```rust
use axum::{routing::get, Router, Json};

async fn metrics_json_handler(
    State(integration): State<Arc<Mutex<HmmIntegration>>>,
) -> Json<HmmIntegrationMetrics> {
    let integration = integration.lock().await;
    Json(integration.get_metrics())
}

let app = Router::new()
    .route("/api/metrics", get(metrics_json_handler))
    .with_state(Arc::new(Mutex::new(integration)));
```

## Alerting Recommendations

### Critical Alerts

1. **Circuit Breaker Open**
   - Alert when: `circuit_breaker_opens > 0` in last 5 minutes
   - Action: Check HMM service health

2. **High Failure Rate**
   - Alert when: `failed_requests / total_requests > 0.1` (10%)
   - Action: Investigate service issues

3. **Fallback Active**
   - Alert when: `currently_active == true` for > 5 minutes
   - Action: Check service connectivity

### Warning Alerts

1. **Low Cache Hit Rate**
   - Alert when: `hit_rate < 0.5` (50%)
   - Action: Review cache TTL and size settings

2. **High Request Latency**
   - Alert when: `avg_duration_ms > 100ms`
   - Action: Check service performance

3. **Frequent Evictions**
   - Alert when: `evictions` increasing rapidly
   - Action: Consider increasing cache size

## Performance Tuning

### Cache Optimization

Monitor cache metrics to optimize settings:

```rust
// Adjust cache size based on hit rate
let stats = integration.get_cache_stats();
if stats.hit_rate < 0.7 {
    // Consider increasing cache size or TTL
    let integration = HmmIntegration::with_config_and_cache(
        config,
        Duration::from_secs(120),  // Longer TTL
        2000,                       // Larger cache
    )?;
}
```

### Request Optimization

Monitor request metrics to identify issues:

```rust
let metrics = integration.get_metrics();
if metrics.requests.avg_duration_ms > 100.0 {
    // Consider adjusting timeout or retry settings
    println!("High latency detected: {:.2}ms", metrics.requests.avg_duration_ms);
}
```

## Example: Complete Monitoring Setup

See `examples/monitoring_demo.rs` for a complete example demonstrating:

- Metrics collection during signal processing
- Real-time metrics display
- JSON and Prometheus export
- Cache and circuit breaker monitoring

Run the example:

```bash
cargo run --example monitoring_demo
```

## Best Practices

1. **Regular Monitoring**: Check metrics at regular intervals (e.g., every 30 seconds)
2. **Baseline Establishment**: Establish normal operating ranges for each metric
3. **Trend Analysis**: Monitor trends over time, not just absolute values
4. **Correlation**: Correlate metrics across categories (e.g., cache misses vs. request latency)
5. **Alerting**: Set up alerts for critical thresholds
6. **Logging**: Combine metrics with structured logging for complete observability

## Troubleshooting

### High Cache Miss Rate

**Symptoms**: `hit_rate < 0.5`

**Possible Causes**:
- Cache TTL too short
- Cache size too small
- High variability in signal observations

**Solutions**:
- Increase cache TTL
- Increase cache max_size
- Review observation rounding precision

### Circuit Breaker Frequently Opening

**Symptoms**: `circuit_breaker_opens` increasing rapidly

**Possible Causes**:
- HMM service instability
- Network issues
- Threshold too low

**Solutions**:
- Check HMM service health
- Verify network connectivity
- Adjust circuit breaker threshold

### High Fallback Activation Rate

**Symptoms**: `total_activations` increasing, `currently_active == true`

**Possible Causes**:
- Service unavailable
- Network connectivity issues
- Configuration problems

**Solutions**:
- Verify HMM service is running
- Check network connectivity
- Review fallback configuration
- Check service logs for errors

## Requirements Mapping

This implementation satisfies the following requirements:

- **Requirement 2.5**: Cache metrics (hits, misses, size, evictions)
- **Requirement 4.5**: Circuit breaker state metrics and transitions
- **Requirement 6.5**: Error metrics and fallback activation tracking

All metrics are collected automatically during normal operation and can be exported in multiple formats for integration with monitoring systems.
