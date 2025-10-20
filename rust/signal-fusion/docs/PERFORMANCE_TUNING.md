# HMM Integration Performance Tuning Guide

Comprehensive guide for optimizing the performance of the HMM integration component.

## Table of Contents

- [Performance Targets](#performance-targets)
- [Latency Optimization](#latency-optimization)
- [Throughput Optimization](#throughput-optimization)
- [Memory Optimization](#memory-optimization)
- [Cache Tuning](#cache-tuning)
- [Network Optimization](#network-optimization)
- [Configuration Profiles](#configuration-profiles)
- [Benchmarking](#benchmarking)
- [Monitoring](#monitoring)

## Performance Targets

### Target Metrics

| Metric | Target | Acceptable | Critical |
|--------|--------|------------|----------|
| Cache Hit Latency | < 1ms | < 5ms | > 10ms |
| Service Call Latency | < 20ms | < 50ms | > 100ms |
| End-to-End Latency | < 30ms | < 100ms | > 200ms |
| Cache Hit Rate | > 80% | > 60% | < 40% |
| Throughput | > 1000 req/s | > 500 req/s | < 100 req/s |
| Memory Usage | < 100MB | < 500MB | > 1GB |
| Error Rate | < 0.1% | < 1% | > 5% |

### Measuring Performance

```rust
use std::time::Instant;

// Measure single request
let start = Instant::now();
let weights = integration.get_fusion_weights_for_signals(&signals).await?;
let duration = start.elapsed();
println!("Request took: {:?}", duration);

// Get aggregate metrics
let metrics = integration.get_metrics();
println!("Average latency: {:.2}ms", metrics.requests.avg_duration_ms);
println!("Cache hit rate: {:.1}%", metrics.cache.hit_rate * 100.0);
```

## Latency Optimization

### 1. Cache Configuration

The cache is the primary latency optimization. Tune it for maximum effectiveness:

```rust
// High-performance cache configuration
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(120),  // Long TTL for stable weights
    10000,                      // Large cache for high hit rate
)?;
```

**Tuning Guidelines:**
- **TTL**: Longer TTL (60-120s) for stable market conditions
- **Size**: Larger cache (5000-10000) for diverse signal patterns
- **Trade-off**: Memory usage vs. hit rate

**Impact:**
- Cache hit: ~1ms latency
- Cache miss: ~20-50ms latency (includes service call)

### 2. Timeout Configuration

Balance responsiveness with reliability:

```rust
// Low-latency configuration
let config = HmmClientConfig {
    timeout: Duration::from_millis(2000),  // Fail fast
    retry_attempts: 2,                      // Minimal retries
    retry_delay: Duration::from_millis(50), // Short delays
    ..Default::default()
};
```

**Tuning Guidelines:**
- **Development**: 5000-10000ms timeout for debugging
- **Production**: 2000-3000ms timeout for responsiveness
- **High-frequency**: 1000-2000ms timeout for speed

**Impact:**
- Shorter timeout: Faster failure detection, more timeouts
- Longer timeout: Fewer timeouts, slower failure detection

### 3. Circuit Breaker Tuning

Optimize for fast failover:

```rust
// Fast failover configuration
let config = HmmClientConfig {
    circuit_breaker_threshold: 3,           // Quick detection
    circuit_breaker_timeout: Duration::from_secs(30), // Fast recovery
    enable_fallback: true,                  // Immediate fallback
    ..Default::default()
};
```

**Tuning Guidelines:**
- **Threshold**: Lower (3-5) for fast failover
- **Timeout**: Shorter (30-60s) for quick recovery attempts
- **Fallback**: Always enable for zero-latency fallback

**Impact:**
- Lower threshold: Faster failover, more false positives
- Shorter timeout: More recovery attempts, more load

### 4. Connection Pooling

Optimize HTTP connection reuse:

```rust
// Connection pool configuration
let client = reqwest::Client::builder()
    .pool_max_idle_per_host(20)             // More idle connections
    .pool_idle_timeout(Duration::from_secs(90)) // Keep connections alive
    .tcp_keepalive(Duration::from_secs(60)) // TCP keepalive
    .http2_keep_alive_interval(Duration::from_secs(30))
    .build()?;
```

**Impact:**
- Saves 10-50ms per request by reusing connections
- Reduces load on HMM service

## Throughput Optimization

### 1. Concurrent Processing

Process multiple signals concurrently:

```rust
use futures::future::join_all;

// Process multiple symbols concurrently
async fn process_symbols(
    symbols: Vec<String>,
    integration: &mut HmmIntegration,
) -> Vec<Result<FusionWeights>> {
    let futures: Vec<_> = symbols.iter()
        .map(|symbol| {
            let signals = get_signals_for_symbol(symbol);
            integration.get_fusion_weights_for_signals(&signals)
        })
        .collect();
    
    join_all(futures).await
}
```

**Tuning Guidelines:**
- Process 10-50 symbols concurrently
- Use tokio runtime with multiple threads
- Monitor memory usage under load

**Impact:**
- 10x-50x throughput improvement
- Linear scaling up to service capacity

### 2. Batch Processing

Process signals in batches:

```rust
// Batch processing with rate limiting
async fn process_batch(
    signals: Vec<SignalComponents>,
    integration: &mut HmmIntegration,
    batch_size: usize,
) -> Vec<Result<FusionWeights>> {
    let mut results = Vec::new();
    
    for chunk in signals.chunks(batch_size) {
        let futures: Vec<_> = chunk.iter()
            .map(|s| integration.get_fusion_weights_for_signals(s))
            .collect();
        
        let batch_results = join_all(futures).await;
        results.extend(batch_results);
        
        // Optional: rate limiting
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    results
}
```

**Tuning Guidelines:**
- Batch size: 10-50 requests per batch
- Add delays between batches if needed
- Monitor service load

### 3. Async Runtime Configuration

Optimize tokio runtime:

```rust
// Multi-threaded runtime
let runtime = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(8)              // Match CPU cores
    .thread_name("hmm-worker")
    .thread_stack_size(3 * 1024 * 1024)
    .enable_all()
    .build()?;

runtime.block_on(async {
    // Your async code
});
```

**Tuning Guidelines:**
- Worker threads: Match CPU core count
- Stack size: 2-4 MB per thread
- Enable I/O and time drivers

## Memory Optimization

### 1. Cache Size Management

Balance memory usage with performance:

```rust
// Memory-efficient configuration
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(60),   // Shorter TTL
    1000,                       // Smaller cache
)?;

// Monitor memory usage
let stats = integration.get_cache_stats();
println!("Cache size: {} entries", stats.size);
println!("Estimated memory: ~{}KB", stats.size * 100 / 1024);
```

**Memory Estimates:**
- Per cache entry: ~100 bytes
- 1000 entries: ~100KB
- 10000 entries: ~1MB

**Tuning Guidelines:**
- Start with 1000 entries
- Increase if hit rate < 70%
- Monitor total memory usage

### 2. Connection Pool Limits

Limit connection pool size:

```rust
let client = reqwest::Client::builder()
    .pool_max_idle_per_host(10)  // Limit idle connections
    .build()?;
```

**Memory Impact:**
- Each connection: ~10-50KB
- 10 connections: ~100-500KB

### 3. Metrics Collection

Optimize metrics storage:

```rust
// Metrics use atomic counters (minimal memory)
// No tuning needed, but can disable if necessary
```

## Cache Tuning

### Cache Hit Rate Optimization

Maximize cache effectiveness:

```rust
// High hit rate configuration
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(120),  // Long TTL
    5000,                       // Large size
)?;

// Monitor hit rate
let stats = integration.get_cache_stats();
if stats.hit_rate < 0.7 {
    println!("Warning: Low cache hit rate: {:.1}%", stats.hit_rate * 100.0);
}
```

**Factors Affecting Hit Rate:**
1. **Signal Variability**: High variability = lower hit rate
2. **TTL**: Shorter TTL = lower hit rate
3. **Cache Size**: Smaller size = more evictions = lower hit rate
4. **Observation Rounding**: More rounding = higher hit rate

### Observation Rounding

Increase cache hits by rounding observations:

```rust
// Round observations to increase cache hits
fn round_observations(obs: [f32; 3], decimals: u32) -> [f32; 3] {
    let factor = 10_f32.powi(decimals as i32);
    [
        (obs[0] * factor).round() / factor,
        (obs[1] * factor).round() / factor,
        (obs[2] * factor).round() / factor,
    ]
}

// Use rounded observations
let rounded = round_observations([s_ldc, s_mr, s_tsmom], 2);
let signals = SignalComponents {
    s_ldc: rounded[0],
    s_mr: rounded[1],
    s_tsmom: rounded[2],
};
```

**Impact:**
- 2 decimals: Higher hit rate, less precision
- 3 decimals: Balanced (default)
- 4 decimals: Lower hit rate, more precision

### Cache Eviction Strategy

Monitor and tune eviction:

```rust
// Check eviction rate
let stats = integration.get_cache_stats();
let eviction_rate = stats.evictions as f64 / stats.misses as f64;
if eviction_rate > 0.5 {
    println!("Warning: High eviction rate: {:.1}%", eviction_rate * 100.0);
    println!("Consider increasing cache size");
}
```

## Network Optimization

### 1. HTTP/2 Configuration

Enable HTTP/2 for better performance:

```rust
let client = reqwest::Client::builder()
    .http2_prior_knowledge()    // Force HTTP/2
    .http2_keep_alive_interval(Duration::from_secs(30))
    .http2_keep_alive_timeout(Duration::from_secs(10))
    .build()?;
```

**Benefits:**
- Multiplexing: Multiple requests on one connection
- Header compression: Reduced bandwidth
- Server push: Potential for proactive caching

### 2. TCP Tuning

Optimize TCP settings:

```rust
let client = reqwest::Client::builder()
    .tcp_nodelay(true)          // Disable Nagle's algorithm
    .tcp_keepalive(Duration::from_secs(60))
    .build()?;
```

**Impact:**
- `tcp_nodelay`: Reduces latency for small requests
- `tcp_keepalive`: Keeps connections alive

### 3. DNS Caching

Use IP addresses or configure DNS caching:

```rust
// Use IP address to avoid DNS lookups
let config = HmmClientConfig {
    base_url: "http://10.0.1.100:8000".parse()?,
    ..Default::default()
};

// Or configure DNS caching at OS level
// /etc/systemd/resolved.conf
// Cache=yes
// CacheFromLocalhost=yes
```

### 4. Compression

Enable compression for large responses:

```rust
let client = reqwest::Client::builder()
    .gzip(true)
    .brotli(true)
    .build()?;
```

**Impact:**
- Reduces bandwidth usage
- May increase CPU usage
- Most effective for large responses

## Configuration Profiles

### Development Profile

Optimized for debugging and development:

```rust
let dev_config = HmmClientConfig {
    base_url: "http://localhost:8000".parse()?,
    timeout: Duration::from_millis(10000),      // Long timeout
    retry_attempts: 1,                          // Minimal retries
    retry_delay: Duration::from_millis(100),
    enable_fallback: true,
    fallback_weights: FusionWeights {
        w_ldc: 0.33,
        w_mr: 0.33,
        w_tsmom: 0.34,
    },
    circuit_breaker_threshold: 10,              // Lenient
    circuit_breaker_timeout: Duration::from_secs(60),
};

let integration = HmmIntegration::with_config_and_cache(
    dev_config,
    Duration::from_secs(30),    // Short TTL for testing
    100,                         // Small cache
)?;
```

### Production Profile

Optimized for reliability and performance:

```rust
let prod_config = HmmClientConfig {
    base_url: "http://hmm-service:8000".parse()?,
    timeout: Duration::from_millis(2000),       // Fast timeout
    retry_attempts: 3,                          // Reasonable retries
    retry_delay: Duration::from_millis(200),
    enable_fallback: true,
    fallback_weights: FusionWeights {
        w_ldc: 0.35,
        w_mr: 0.30,
        w_tsmom: 0.35,
    },
    circuit_breaker_threshold: 3,               // Aggressive
    circuit_breaker_timeout: Duration::from_secs(60),
};

let integration = HmmIntegration::with_config_and_cache(
    prod_config,
    Duration::from_secs(120),   // Long TTL
    5000,                        // Large cache
)?;
```

### High-Frequency Trading Profile

Optimized for ultra-low latency:

```rust
let hft_config = HmmClientConfig {
    base_url: "http://10.0.1.100:8000".parse()?, // IP address
    timeout: Duration::from_millis(1000),         // Very fast timeout
    retry_attempts: 1,                            // No retries
    retry_delay: Duration::from_millis(0),
    enable_fallback: true,                        // Instant fallback
    fallback_weights: FusionWeights {
        w_ldc: 0.35,
        w_mr: 0.30,
        w_tsmom: 0.35,
    },
    circuit_breaker_threshold: 2,                 // Very aggressive
    circuit_breaker_timeout: Duration::from_secs(30),
};

let integration = HmmIntegration::with_config_and_cache(
    hft_config,
    Duration::from_secs(300),   // Very long TTL
    10000,                       // Very large cache
)?;
```

### High-Throughput Profile

Optimized for processing many symbols:

```rust
let throughput_config = HmmClientConfig {
    base_url: "http://hmm-service:8000".parse()?,
    timeout: Duration::from_millis(3000),
    retry_attempts: 2,
    retry_delay: Duration::from_millis(100),
    enable_fallback: true,
    fallback_weights: FusionWeights {
        w_ldc: 0.33,
        w_mr: 0.33,
        w_tsmom: 0.34,
    },
    circuit_breaker_threshold: 5,
    circuit_breaker_timeout: Duration::from_secs(45),
};

let integration = HmmIntegration::with_config_and_cache(
    throughput_config,
    Duration::from_secs(180),   // Very long TTL
    20000,                       // Very large cache
)?;

// Use multi-threaded runtime
let runtime = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(16)
    .enable_all()
    .build()?;
```

## Benchmarking

### Latency Benchmarks

```rust
use std::time::Instant;

async fn benchmark_latency(integration: &mut HmmIntegration, iterations: usize) {
    let signals = SignalComponents {
        s_ldc: 0.015,
        s_mr: -0.008,
        s_tsmom: 0.022,
    };
    
    let mut durations = Vec::new();
    
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = integration.get_fusion_weights_for_signals(&signals).await;
        durations.push(start.elapsed().as_micros());
    }
    
    durations.sort();
    let p50 = durations[iterations / 2];
    let p95 = durations[iterations * 95 / 100];
    let p99 = durations[iterations * 99 / 100];
    
    println!("Latency (µs): p50={}, p95={}, p99={}", p50, p95, p99);
}
```

### Throughput Benchmarks

```rust
async fn benchmark_throughput(integration: &mut HmmIntegration, duration_secs: u64) {
    let start = Instant::now();
    let mut count = 0;
    
    while start.elapsed().as_secs() < duration_secs {
        let signals = SignalComponents {
            s_ldc: rand::random::<f32>() * 0.1,
            s_mr: rand::random::<f32>() * 0.1,
            s_tsmom: rand::random::<f32>() * 0.1,
        };
        
        let _ = integration.get_fusion_weights_for_signals(&signals).await;
        count += 1;
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    let throughput = count as f64 / elapsed;
    
    println!("Throughput: {:.0} req/s", throughput);
}
```

### Cache Effectiveness

```rust
async fn benchmark_cache(integration: &mut HmmIntegration, iterations: usize) {
    // Same signal for cache hits
    let signals = SignalComponents {
        s_ldc: 0.015,
        s_mr: -0.008,
        s_tsmom: 0.022,
    };
    
    for _ in 0..iterations {
        let _ = integration.get_fusion_weights_for_signals(&signals).await;
    }
    
    let stats = integration.get_cache_stats();
    println!("Cache hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("Hits: {}, Misses: {}", stats.hits, stats.misses);
}
```

## Monitoring

### Real-Time Performance Monitoring

```rust
use tokio::time::{interval, Duration};

async fn monitor_performance(integration: Arc<Mutex<HmmIntegration>>) {
    let mut interval = interval(Duration::from_secs(30));
    
    loop {
        interval.tick().await;
        
        let integration = integration.lock().await;
        let metrics = integration.get_metrics();
        
        println!("=== Performance Metrics ===");
        println!("Requests: {} total, {} successful, {} failed",
                 metrics.requests.total_requests,
                 metrics.requests.successful_requests,
                 metrics.requests.failed_requests);
        println!("Latency: avg={:.2}ms, max={}ms",
                 metrics.requests.avg_duration_ms,
                 metrics.requests.max_duration_ms);
        println!("Cache: {:.1}% hit rate, {} entries",
                 metrics.cache.hit_rate * 100.0,
                 metrics.cache.size);
        println!("Circuit Breaker: {} opens, {} closes",
                 metrics.circuit_breaker.circuit_breaker_opens,
                 metrics.circuit_breaker.circuit_breaker_closes);
    }
}
```

### Performance Alerts

```rust
fn check_performance_alerts(metrics: &HmmIntegrationMetrics) {
    // High latency alert
    if metrics.requests.avg_duration_ms > 100.0 {
        warn!("High average latency: {:.2}ms", metrics.requests.avg_duration_ms);
    }
    
    // Low cache hit rate alert
    if metrics.cache.hit_rate < 0.6 {
        warn!("Low cache hit rate: {:.1}%", metrics.cache.hit_rate * 100.0);
    }
    
    // High error rate alert
    let error_rate = metrics.requests.failed_requests as f64 
        / metrics.requests.total_requests as f64;
    if error_rate > 0.05 {
        error!("High error rate: {:.1}%", error_rate * 100.0);
    }
    
    // Circuit breaker alert
    if metrics.circuit_breaker.circuit_breaker_opens > 0 {
        warn!("Circuit breaker opened {} times",
              metrics.circuit_breaker.circuit_breaker_opens);
    }
}
```

## Performance Checklist

### Before Deployment

- [ ] Run latency benchmarks (target: p99 < 100ms)
- [ ] Run throughput benchmarks (target: > 500 req/s)
- [ ] Test cache effectiveness (target: > 70% hit rate)
- [ ] Test circuit breaker behavior
- [ ] Test fallback mechanism
- [ ] Profile memory usage (target: < 500MB)
- [ ] Test concurrent load
- [ ] Verify configuration values
- [ ] Enable monitoring and alerting
- [ ] Document performance characteristics

### After Deployment

- [ ] Monitor latency metrics
- [ ] Monitor cache hit rate
- [ ] Monitor error rates
- [ ] Monitor circuit breaker state
- [ ] Monitor memory usage
- [ ] Review logs for warnings
- [ ] Tune configuration based on metrics
- [ ] Scale horizontally if needed

## Requirements Coverage

This performance tuning guide addresses:

- **Requirement 1.2**: Request parsing within 5ms (cache optimization)
- **Requirement 2.1**: Cache performance optimization
- **Requirement 5.4**: Fusion computation within 5ms

## See Also

- [API Reference](./API_REFERENCE.md)
- [Configuration Guide](../CONFIG.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Monitoring and Metrics](./MONITORING_METRICS.md)
