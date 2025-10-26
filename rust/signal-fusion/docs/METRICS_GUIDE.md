# Signal Emission Metrics Guide

This guide explains how to use the signal emission metrics system for monitoring and observability.

## Overview

The signal emission metrics system provides comprehensive monitoring for all signal emission operations using Prometheus metrics. It tracks:

- Signal publishing metrics (count, latency, errors)
- Validation metrics (success/failure rates, error types)
- Publisher backend metrics (Redis, Kafka performance)
- Buffer utilization metrics
- Audit logging metrics
- Health check metrics

## Quick Start

### Basic Usage

```rust
use signal_fusion::emission::{SignalEmissionMetrics, MetricsTimer};
use std::sync::Arc;

// Create metrics collector
let metrics = Arc::new(SignalEmissionMetrics::new()?);

// Record signal publication
metrics.record_signal_published("BTCUSDT", "redis", "BUY");

// Record latency with timer
let timer = MetricsTimer::start();
// ... do some work ...
metrics.record_emission_latency("BTCUSDT", "redis", timer.elapsed_seconds());

// Record validation error
metrics.record_validation_error("BTCUSDT", "invalid_strength", "strength");

// Update buffer metrics
metrics.update_buffer_size(50, 100); // 50 signals in buffer, max 100
```

### HTTP Metrics Server

```rust
use signal_fusion::emission::{MetricsServer, MetricsServerConfig, SignalEmissionMetrics};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create metrics
    let metrics = Arc::new(SignalEmissionMetrics::new()?);
    
    // Configure server
    let config = MetricsServerConfig {
        bind_address: "0.0.0.0".to_string(),
        port: 9090,
        enable_health_endpoint: true,
        enable_json_endpoint: true,
        ..Default::default()
    };
    
    // Start server
    let mut server = MetricsServer::new(config, metrics.clone());
    server.start().await?;
    
    // Server is now running on http://localhost:9090
    // Endpoints:
    // - /metrics (Prometheus format)
    // - /health (Health check)
    // - /metrics/json (JSON format for debugging)
    
    // Your application logic here...
    
    Ok(())
}
```

## Available Metrics

### Signal Publishing Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `signal_emission_signals_published_total` | Counter | `symbol`, `backend`, `side` | Total signals published |
| `signal_emission_validation_errors_total` | Counter | `symbol`, `error_type`, `field` | Validation errors |
| `signal_emission_publisher_errors_total` | Counter | `backend`, `error_type` | Publisher errors |
| `signal_emission_buffer_overflows_total` | Counter | - | Buffer overflow events |

### Latency Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `signal_emission_duration_seconds` | Histogram | `symbol`, `backend` | End-to-end emission latency |
| `signal_emission_validation_duration_seconds` | Histogram | - | Signal validation latency |
| `signal_emission_publishing_duration_seconds` | Histogram | `backend` | Publishing latency |

### Buffer Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `signal_emission_buffer_size_current` | Gauge | - | Current buffer size |
| `signal_emission_buffer_utilization_ratio` | Gauge | - | Buffer utilization (0.0-1.0) |
| `signal_emission_buffer_operations_total` | Counter | `operation` | Buffer operations |

### Publisher Backend Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `signal_emission_publisher_connections_active` | Gauge | `backend` | Active connections |
| `signal_emission_publisher_connection_errors_total` | Counter | `backend`, `error_type` | Connection errors |
| `signal_emission_publisher_delivery_confirmations_total` | Counter | `backend`, `status` | Delivery confirmations |

### Audit Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `signal_emission_audit_events_logged_total` | Counter | `event_type` | Audit events logged |
| `signal_emission_audit_logging_errors_total` | Counter | `error_type` | Audit logging errors |
| `signal_emission_audit_file_size_bytes` | Gauge | - | Audit file size |
| `signal_emission_audit_s3_uploads_total` | Counter | `status` | S3 upload results |

### Health Check Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `signal_emission_health_check_duration_seconds` | Histogram | `component` | Health check latency |
| `signal_emission_health_check_status` | Gauge | `component` | Health status (1=healthy, 0=unhealthy) |

## Prometheus Configuration

Add this to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'signal-emission'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
    metrics_path: /metrics
```

## Grafana Dashboard

Example Grafana queries:

### Signal Publishing Rate
```promql
rate(signal_emission_signals_published_total[5m])
```

### Error Rate
```promql
rate(signal_emission_validation_errors_total[5m]) + 
rate(signal_emission_publisher_errors_total[5m])
```

### Average Latency
```promql
histogram_quantile(0.95, rate(signal_emission_duration_seconds_bucket[5m]))
```

### Buffer Utilization
```promql
signal_emission_buffer_utilization_ratio
```

### Health Status
```promql
signal_emission_health_check_status
```

## Alerting Rules

Example Prometheus alerting rules:

```yaml
groups:
  - name: signal_emission
    rules:
      - alert: HighErrorRate
        expr: rate(signal_emission_validation_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High signal validation error rate"
          
      - alert: BufferNearFull
        expr: signal_emission_buffer_utilization_ratio > 0.8
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Signal buffer utilization high"
          
      - alert: ComponentUnhealthy
        expr: signal_emission_health_check_status == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Signal emission component unhealthy"
```

## Best Practices

1. **Use appropriate labels**: Include relevant labels for filtering and aggregation
2. **Monitor latency percentiles**: Use histogram metrics for latency analysis
3. **Set up alerting**: Configure alerts for error rates and health status
4. **Regular cleanup**: Monitor buffer utilization and audit file sizes
5. **Performance impact**: Metrics collection has minimal overhead but monitor in high-throughput scenarios

## Troubleshooting

### High Memory Usage
- Check buffer utilization metrics
- Verify audit log rotation is working
- Monitor connection pool sizes

### Missing Metrics
- Verify metrics server is running
- Check Prometheus scrape configuration
- Ensure metrics are being recorded in application code

### High Latency
- Check individual component latencies
- Monitor publisher backend performance
- Verify network connectivity to Redis/Kafka

## Integration with Existing Systems

The metrics system integrates seamlessly with:
- Existing signal-fusion workflow
- Redis and Kafka publishers
- Audit logging system
- Health monitoring
- Circuit breaker patterns

For more examples, see the `examples/` directory in the signal-fusion crate.