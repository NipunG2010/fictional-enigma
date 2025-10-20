# HMM Integration Troubleshooting Guide

Comprehensive troubleshooting guide for the HMM integration component.

## Table of Contents

- [Connection Issues](#connection-issues)
- [Performance Problems](#performance-problems)
- [Circuit Breaker Issues](#circuit-breaker-issues)
- [Cache Problems](#cache-problems)
- [Error Handling](#error-handling)
- [Configuration Issues](#configuration-issues)
- [Service Health](#service-health)
- [Debugging Tools](#debugging-tools)

## Connection Issues

### Problem: Connection Refused

**Symptoms:**
```
Error: Network(reqwest::Error { kind: Request, source: hyper::Error(Connect, ConnectError) })
```

**Possible Causes:**
1. HMM service is not running
2. Incorrect service URL or port
3. Network connectivity issues
4. Firewall blocking connection

**Solutions:**

1. **Verify service is running:**
```bash
# Check if service is running
curl http://localhost:8000/health

# Check Docker container
docker ps | grep hmm-service

# Check service logs
docker logs hmm-service
```

2. **Verify configuration:**
```rust
let config = HmmClientConfig {
    base_url: "http://localhost:8000".parse()?,  // Check URL
    ..Default::default()
};
```

3. **Test connectivity:**
```bash
# Test network connectivity
ping localhost
telnet localhost 8000

# Check firewall rules
sudo iptables -L
```

4. **Enable detailed logging:**
```rust
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

### Problem: DNS Resolution Failure

**Symptoms:**
```
Error: Network(reqwest::Error { kind: Request, source: dns error })
```

**Solutions:**

1. **Use IP address instead of hostname:**
```rust
let config = HmmClientConfig {
    base_url: "http://127.0.0.1:8000".parse()?,
    ..Default::default()
};
```

2. **Check DNS configuration:**
```bash
# Test DNS resolution
nslookup hmm-service
dig hmm-service

# Check /etc/hosts
cat /etc/hosts
```

3. **Verify Docker network:**
```bash
# Check Docker network
docker network ls
docker network inspect bridge
```

### Problem: SSL/TLS Errors

**Symptoms:**
```
Error: Network(reqwest::Error { kind: Request, source: invalid certificate })
```

**Solutions:**

1. **Disable certificate verification (development only):**
```rust
let client = reqwest::Client::builder()
    .danger_accept_invalid_certs(true)
    .build()?;
```

2. **Add custom CA certificate:**
```rust
let cert = reqwest::Certificate::from_pem(&cert_bytes)?;
let client = reqwest::Client::builder()
    .add_root_certificate(cert)
    .build()?;
```

## Performance Problems

### Problem: High Latency

**Symptoms:**
- Request duration > 100ms
- Slow signal generation
- Timeouts under load

**Diagnosis:**
```rust
let metrics = integration.get_metrics();
println!("Average latency: {:.2}ms", metrics.requests.avg_duration_ms);
println!("Max latency: {}ms", metrics.requests.max_duration_ms);
println!("Cache hit rate: {:.1}%", metrics.cache.hit_rate * 100.0);
```

**Solutions:**

1. **Increase cache TTL and size:**
```rust
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(120),  // Longer TTL
    5000,                       // Larger cache
)?;
```

2. **Optimize timeout settings:**
```rust
let config = HmmClientConfig {
    timeout: Duration::from_millis(3000),  // Reduce timeout
    retry_attempts: 2,                      // Fewer retries
    ..Default::default()
};
```

3. **Check service performance:**
```bash
# Monitor service metrics
curl http://localhost:8000/metrics

# Check service logs for slow queries
docker logs hmm-service | grep "processing_time"
```

4. **Use connection pooling:**
```rust
// Already enabled by default in reqwest
// Verify pool settings if needed
let client = reqwest::Client::builder()
    .pool_max_idle_per_host(10)
    .pool_idle_timeout(Duration::from_secs(30))
    .build()?;
```

### Problem: Memory Usage Growing

**Symptoms:**
- Increasing memory consumption
- Out of memory errors
- Slow performance over time

**Diagnosis:**
```rust
let cache_stats = integration.get_cache_stats();
println!("Cache size: {}", cache_stats.size);
println!("Evictions: {}", cache_stats.evictions);
```

**Solutions:**

1. **Limit cache size:**
```rust
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(60),
    1000,  // Smaller max size
)?;
```

2. **Reduce cache TTL:**
```rust
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(30),  // Shorter TTL
    cache_max_size,
)?;
```

3. **Monitor memory usage:**
```bash
# Check process memory
ps aux | grep your_process

# Use valgrind for detailed analysis
valgrind --leak-check=full ./your_binary
```

### Problem: Low Throughput

**Symptoms:**
- Cannot process many requests per second
- Requests queuing up
- High CPU usage

**Solutions:**

1. **Use async/await properly:**
```rust
// Process multiple symbols concurrently
let futures: Vec<_> = symbols.iter()
    .map(|symbol| {
        let mut integration = integration.clone();
        async move {
            integration.get_fusion_weights_for_signals(&signals).await
        }
    })
    .collect();

let results = futures::future::join_all(futures).await;
```

2. **Batch requests if possible:**
```rust
// Process in batches
for chunk in signals.chunks(10) {
    let futures: Vec<_> = chunk.iter()
        .map(|s| integration.get_fusion_weights_for_signals(s))
        .collect();
    let results = futures::future::join_all(futures).await;
}
```

3. **Optimize service:**
```bash
# Scale HMM service horizontally
docker-compose up --scale hmm-service=3

# Use load balancer
# Configure base_url to point to load balancer
```

## Circuit Breaker Issues

### Problem: Circuit Breaker Stuck Open

**Symptoms:**
- Circuit breaker remains open indefinitely
- All requests use fallback weights
- Service appears unavailable

**Diagnosis:**
```rust
let (state, failures) = integration.get_circuit_breaker_status();
println!("State: {}, Failures: {}", state, failures);

let metrics = integration.get_circuit_breaker_metrics();
println!("Opens: {}", metrics.circuit_breaker_opens);
println!("Closes: {}", metrics.circuit_breaker_closes);
```

**Solutions:**

1. **Verify service is actually running:**
```bash
curl http://localhost:8000/health
```

2. **Increase circuit breaker timeout:**
```rust
let config = HmmClientConfig {
    circuit_breaker_timeout: Duration::from_secs(120),  // Longer timeout
    ..Default::default()
};
```

3. **Check for persistent service issues:**
```bash
# Check service logs
docker logs hmm-service --tail 100

# Check service health
curl http://localhost:8000/readiness
```

4. **Manually test recovery:**
```rust
// Force a health check
let health = integration.is_service_ready().await;
println!("Service ready: {}", health);
```

### Problem: Circuit Breaker Opens Too Frequently

**Symptoms:**
- Circuit breaker opens on transient errors
- Frequent state transitions
- Unnecessary fallback activations

**Diagnosis:**
```rust
let metrics = integration.get_circuit_breaker_metrics();
println!("Opens: {}", metrics.circuit_breaker_opens);
println!("Half-open attempts: {}", metrics.half_open_attempts);
```

**Solutions:**

1. **Increase failure threshold:**
```rust
let config = HmmClientConfig {
    circuit_breaker_threshold: 10,  // Higher threshold
    ..Default::default()
};
```

2. **Increase retry attempts:**
```rust
let config = HmmClientConfig {
    retry_attempts: 5,  // More retries before failure
    retry_delay: Duration::from_millis(200),
    ..Default::default()
};
```

3. **Adjust timeout:**
```rust
let config = HmmClientConfig {
    timeout: Duration::from_millis(10000),  // More generous timeout
    ..Default::default()
};
```

### Problem: Circuit Breaker Never Opens

**Symptoms:**
- Service failures don't trigger circuit breaker
- No fallback activation despite errors
- Continuous failed requests

**Diagnosis:**
```rust
let metrics = integration.get_circuit_breaker_metrics();
println!("Failed requests: {}", metrics.failed_requests);
println!("Opens: {}", metrics.circuit_breaker_opens);
```

**Solutions:**

1. **Verify circuit breaker is enabled:**
```rust
let config = HmmClientConfig {
    circuit_breaker_threshold: 5,  // Must be > 0
    enable_fallback: true,          // Enable fallback
    ..Default::default()
};
```

2. **Check error types:**
```rust
// Some errors might not count as failures
// Enable debug logging to see error classification
```

3. **Lower threshold for testing:**
```rust
let config = HmmClientConfig {
    circuit_breaker_threshold: 2,  // Lower for testing
    ..Default::default()
};
```

## Cache Problems

### Problem: Low Cache Hit Rate

**Symptoms:**
- Cache hit rate < 50%
- High service load
- Increased latency

**Diagnosis:**
```rust
let stats = integration.get_cache_stats();
println!("Hit rate: {:.1}%", stats.hit_rate * 100.0);
println!("Hits: {}, Misses: {}", stats.hits, stats.misses);
println!("Size: {}, Evictions: {}", stats.size, stats.evictions);
```

**Solutions:**

1. **Increase cache TTL:**
```rust
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(120),  // Longer TTL
    cache_max_size,
)?;
```

2. **Increase cache size:**
```rust
let integration = HmmIntegration::with_config_and_cache(
    config,
    cache_ttl,
    5000,  // Larger cache
)?;
```

3. **Check observation variability:**
```rust
// High variability in observations reduces cache effectiveness
// Consider rounding observations more aggressively
let rounded_obs = [
    (obs[0] * 100.0).round() / 100.0,  // Round to 2 decimals
    (obs[1] * 100.0).round() / 100.0,
    (obs[2] * 100.0).round() / 100.0,
];
```

### Problem: Cache Not Working

**Symptoms:**
- Cache hit rate is 0%
- All requests go to service
- Cache size remains 0

**Diagnosis:**
```rust
let stats = integration.get_cache_stats();
println!("Cache stats: {:?}", stats);
```

**Solutions:**

1. **Verify cache is initialized:**
```rust
// Check cache configuration
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(60),
    1000,
)?;
```

2. **Check for cache key issues:**
```rust
// Enable debug logging to see cache operations
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

3. **Verify observations are valid:**
```rust
// NaN or infinite values won't cache properly
let signals = SignalComponents {
    s_ldc: 0.015,
    s_mr: -0.008,
    s_tsmom: 0.022,
};
assert!(!signals.s_ldc.is_nan());
assert!(signals.s_ldc.is_finite());
```

### Problem: Excessive Cache Evictions

**Symptoms:**
- High eviction count
- Cache size at maximum
- Reduced cache effectiveness

**Diagnosis:**
```rust
let stats = integration.get_cache_stats();
println!("Evictions: {}", stats.evictions);
println!("Size: {} / max", stats.size);
```

**Solutions:**

1. **Increase cache size:**
```rust
let integration = HmmIntegration::with_config_and_cache(
    config,
    cache_ttl,
    10000,  // Much larger cache
)?;
```

2. **Reduce TTL to allow natural expiration:**
```rust
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(30),  // Shorter TTL
    cache_max_size,
)?;
```

## Error Handling

### Problem: Unhandled Errors

**Symptoms:**
- Application crashes on errors
- No fallback behavior
- Lost trading signals

**Solutions:**

1. **Always handle errors:**
```rust
match integration.get_fusion_weights_for_signals(&signals).await {
    Ok(weights) => {
        // Use weights
    }
    Err(e) => {
        error!("Failed to get weights: {}", e);
        // Use default weights
        let weights = FusionWeights {
            w_ldc: 0.33,
            w_mr: 0.33,
            w_tsmom: 0.34,
        };
    }
}
```

2. **Enable fallback:**
```rust
let config = HmmClientConfig {
    enable_fallback: true,
    fallback_weights: FusionWeights {
        w_ldc: 0.35,
        w_mr: 0.30,
        w_tsmom: 0.35,
    },
    ..Default::default()
};
```

3. **Log errors with context:**
```rust
if let Err(e) = integration.get_fusion_weights_for_signals(&signals).await {
    error!("Failed to get weights for {:?}: {}", signals, e);
}
```

### Problem: Timeout Errors

**Symptoms:**
```
Error: Timeout { duration: 5s }
```

**Solutions:**

1. **Increase timeout:**
```rust
let config = HmmClientConfig {
    timeout: Duration::from_millis(10000),  // 10 seconds
    ..Default::default()
};
```

2. **Check service performance:**
```bash
# Monitor service response times
curl -w "@curl-format.txt" http://localhost:8000/predict
```

3. **Reduce retry attempts:**
```rust
let config = HmmClientConfig {
    retry_attempts: 1,  // Fail faster
    ..Default::default()
};
```

### Problem: Validation Errors

**Symptoms:**
```
Error: ValidationError { field: "observations", message: "value out of range" }
```

**Solutions:**

1. **Validate inputs before sending:**
```rust
fn validate_signals(signals: &SignalComponents) -> Result<(), String> {
    if signals.s_ldc < -1.0 || signals.s_ldc > 1.0 {
        return Err(format!("s_ldc out of range: {}", signals.s_ldc));
    }
    if signals.s_mr < -1.0 || signals.s_mr > 1.0 {
        return Err(format!("s_mr out of range: {}", signals.s_mr));
    }
    if signals.s_tsmom < -1.0 || signals.s_tsmom > 1.0 {
        return Err(format!("s_tsmom out of range: {}", signals.s_tsmom));
    }
    Ok(())
}
```

2. **Clamp values to valid range:**
```rust
fn clamp_signals(signals: &mut SignalComponents) {
    signals.s_ldc = signals.s_ldc.clamp(-1.0, 1.0);
    signals.s_mr = signals.s_mr.clamp(-1.0, 1.0);
    signals.s_tsmom = signals.s_tsmom.clamp(-1.0, 1.0);
}
```

3. **Handle NaN and infinite values:**
```rust
fn sanitize_signals(signals: &mut SignalComponents) {
    if !signals.s_ldc.is_finite() {
        signals.s_ldc = 0.0;
    }
    if !signals.s_mr.is_finite() {
        signals.s_mr = 0.0;
    }
    if !signals.s_tsmom.is_finite() {
        signals.s_tsmom = 0.0;
    }
}
```

## Configuration Issues

### Problem: Configuration Not Loading

**Symptoms:**
- Default values used instead of config file
- Environment variables ignored
- Configuration errors

**Solutions:**

1. **Verify file path:**
```rust
use std::path::Path;

let path = Path::new("hmm_integration.toml");
if !path.exists() {
    eprintln!("Config file not found: {:?}", path);
}

let config = HmmIntegrationConfig::from_file(path)?;
```

2. **Check TOML syntax:**
```bash
# Validate TOML syntax
cat hmm_integration.toml | toml-cli validate
```

3. **Verify environment variables:**
```rust
// Print environment variables
for (key, value) in std::env::vars() {
    if key.starts_with("HMM_") {
        println!("{}: {}", key, value);
    }
}
```

4. **Use explicit configuration:**
```rust
let config = HmmClientConfig {
    base_url: "http://localhost:8000".parse()?,
    timeout: Duration::from_millis(5000),
    retry_attempts: 3,
    retry_delay: Duration::from_millis(100),
    enable_fallback: true,
    fallback_weights: FusionWeights {
        w_ldc: 0.33,
        w_mr: 0.33,
        w_tsmom: 0.34,
    },
    circuit_breaker_threshold: 5,
    circuit_breaker_timeout: Duration::from_secs(30),
};
```

### Problem: Invalid Configuration Values

**Symptoms:**
```
Error: ValidationError { field: "timeout", message: "must be > 0" }
```

**Solutions:**

1. **Check value ranges:**
```toml
[service]
timeout_ms = 5000  # Must be > 0
retry_attempts = 3  # Must be <= 10

[circuit_breaker]
threshold = 5  # Must be > 0
timeout_sec = 30  # Must be > 0
```

2. **Validate before use:**
```rust
if config.timeout.as_millis() == 0 {
    return Err("Timeout must be greater than 0".into());
}
```

## Service Health

### Problem: Model Not Loaded

**Symptoms:**
```
Error: ModelError { message: "No model loaded" }
```

**Solutions:**

1. **Check service readiness:**
```rust
let readiness = client.readiness_check().await?;
if !readiness.model_loaded {
    error!("Model not loaded");
    // Trigger reload
    client.reload_model().await?;
}
```

2. **Verify MinIO connectivity:**
```bash
# Check MinIO is running
curl http://localhost:9000/minio/health/live

# Check model artifacts exist
mc ls minio/hmm-models/
```

3. **Check service logs:**
```bash
docker logs hmm-service | grep -i "model"
```

4. **Manually reload model:**
```rust
let response = client.reload_model().await?;
if response.success {
    println!("Model reloaded: {:?}", response.model_version);
} else {
    error!("Reload failed: {}", response.message);
}
```

### Problem: Service Degraded Performance

**Symptoms:**
- Slow responses
- Increasing error rate
- High resource usage

**Solutions:**

1. **Check service metrics:**
```bash
curl http://localhost:8000/metrics
```

2. **Monitor resource usage:**
```bash
docker stats hmm-service
```

3. **Check for memory leaks:**
```bash
# Monitor memory over time
watch -n 5 'docker stats hmm-service --no-stream'
```

4. **Restart service if needed:**
```bash
docker restart hmm-service
```

## Debugging Tools

### Enable Debug Logging

```rust
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .with_target(true)
    .with_thread_ids(true)
    .with_line_number(true)
    .init();
```

### Metrics Dashboard

```rust
// Print comprehensive metrics
let metrics = integration.get_metrics();
println!("{:#?}", metrics);

// Export to JSON for analysis
let json = integration.export_metrics(MetricsFormat::Json)?;
std::fs::write("metrics.json", json)?;
```

### Request Tracing

```rust
// Use request IDs for tracing
let request_id = format!("req-{}", uuid::Uuid::new_v4());
let response = client.get_fusion_weights(
    observations,
    Some(request_id.clone())
).await?;
println!("Request {} completed", request_id);
```

### Network Debugging

```bash
# Capture network traffic
sudo tcpdump -i any -w hmm-traffic.pcap port 8000

# Analyze with Wireshark
wireshark hmm-traffic.pcap
```

### Performance Profiling

```bash
# Profile with perf
perf record -g ./your_binary
perf report

# Profile with flamegraph
cargo flamegraph --bin your_binary
```

## Common Error Messages

### "Connection refused"
- Service not running
- Wrong port or host
- Firewall blocking

### "Timeout"
- Service overloaded
- Network latency
- Timeout too short

### "Circuit breaker open"
- Too many failures
- Service unavailable
- Need to wait for recovery

### "Model not loaded"
- Service just started
- MinIO connectivity issue
- Model artifacts missing

### "Validation error"
- Invalid input values
- Out of range signals
- NaN or infinite values

## Getting Help

If you're still experiencing issues:

1. **Check logs:**
```bash
# Application logs
tail -f /var/log/your-app.log

# Service logs
docker logs hmm-service --tail 100 -f
```

2. **Collect diagnostics:**
```rust
// Export metrics
let metrics = integration.export_metrics(MetricsFormat::Json)?;
std::fs::write("diagnostics.json", metrics)?;

// Get service status
let (loaded, version) = integration.get_service_status().await?;
println!("Service: loaded={}, version={:?}", loaded, version);

// Get circuit breaker status
let (state, failures) = integration.get_circuit_breaker_status();
println!("Circuit breaker: state={}, failures={}", state, failures);
```

3. **Review documentation:**
- [API Reference](./API_REFERENCE.md)
- [Configuration Guide](../CONFIG.md)
- [Performance Tuning](./PERFORMANCE_TUNING.md)

## Requirements Coverage

This troubleshooting guide addresses:

- **Requirement 1.1**: HTTP client communication issues
- **Requirement 3.3**: Fallback configuration problems
- **Requirement 6.4**: Error diagnosis and resolution

