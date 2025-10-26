# Signal Fusion & Emission

A high-performance Rust crate for trading signal fusion and emission to message bus infrastructure (Redis Streams/Kafka) with comprehensive audit logging and monitoring.

## Quick Start

### Basic Usage

```rust
use signal_fusion::{SignalFusion, TradingSignal, SignalSide};
use signal_fusion::emission::{SignalPublisher, SignalEmissionConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = SignalEmissionConfig::from_file("signal_emission.toml")?;
    
    // Create signal publisher
    let mut publisher = SignalPublisher::new(config.publisher).await?;
    
    // Create a trading signal
    let signal = TradingSignal {
        timestamp: chrono::Utc::now().timestamp_millis(),
        symbol: "BTCUSDT".to_string(),
        side: SignalSide::Buy,
        strength: 0.75,
        confidence: 0.85,
        // ... other fields
    };
    
    // Publish signal
    publisher.publish_signal(signal).await?;
    
    Ok(())
}
```

### Configuration

Create a `signal_emission.toml` file:

```toml
[publisher]
enabled = true
backend = "redis"  # "redis", "kafka", "both", or "none"
batch_size = 100
flush_interval_ms = 1000

[redis]
url = "redis://localhost:6379"
stream_name = "trading_signals"
max_stream_length = 10000
connection_pool_size = 10

[kafka]
brokers = ["localhost:9092"]
topic = "trading_signals"
partition_strategy = "symbol"  # "symbol", "round_robin", or "custom"
compression = "gzip"

[buffer]
max_size = 1000
persist_to_disk = true
persist_path = "/tmp/signal_buffer"

[audit]
enabled = true
file_path = "/var/log/imp/signals.jsonl"
s3_bucket = "imp-audit-logs"
s3_prefix = "signals/"
upload_interval_sec = 300
```

## Features

### Signal Publishing
- **Redis Streams**: High-performance stream publishing with ordering guarantees
- **Kafka Topics**: Scalable topic-based publishing with configurable partitioning
- **Dual Backend**: Support both Redis and Kafka simultaneously
- **Message Ordering**: Per-symbol ordering for consistent signal delivery

### Reliability & Resilience
- **Circuit Breaker**: Automatic failure detection and recovery
- **Retry Logic**: Exponential backoff for transient failures
- **Local Buffering**: Continue operation during message bus outages
- **Health Monitoring**: Comprehensive health checks for all components

### Audit & Compliance
- **Comprehensive Logging**: Full audit trail of all signal emissions
- **Correlation Tracking**: Link signals to source features and computations
- **S3 Archival**: Automatic upload of audit logs to object storage
- **Performance Metrics**: Detailed latency and throughput measurements

### Monitoring & Observability
- **Prometheus Metrics**: Export metrics for monitoring systems
- **Health Endpoints**: HTTP endpoints for external monitoring
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Performance Tracking**: Real-time performance metrics collection

## Configuration Options

### Publisher Configuration

```toml
[publisher]
enabled = true                    # Enable/disable signal emission
backend = "redis"                # Backend selection: redis, kafka, both, none
batch_size = 100                 # Batch size for bulk operations
flush_interval_ms = 1000         # Maximum time to wait before flushing batch

[publisher.retry_policy]
max_attempts = 3                 # Maximum retry attempts
base_delay_ms = 100             # Base delay between retries
max_delay_ms = 5000             # Maximum delay between retries
backoff_multiplier = 2.0        # Exponential backoff multiplier
jitter = true                   # Add random jitter to delays

[publisher.circuit_breaker]
failure_threshold = 5           # Failures before opening circuit
timeout_sec = 30               # Timeout before attempting recovery
```

### Redis Configuration

```toml
[redis]
url = "redis://localhost:6379"           # Redis connection URL
stream_name = "trading_signals"          # Stream name for signals
max_stream_length = 10000               # Maximum stream length (MAXLEN)
connection_pool_size = 10               # Connection pool size

[redis.health]
check_interval_sec = 30                 # Health check interval
timeout_ms = 1000                      # Health check timeout
```

### Kafka Configuration

```toml
[kafka]
brokers = ["localhost:9092"]            # Kafka broker addresses
topic = "trading_signals"               # Topic name for signals
partition_strategy = "symbol"           # Partitioning strategy
compression = "gzip"                    # Compression algorithm

[kafka.producer]
acks = "all"                           # Acknowledgment level
retries = 3                            # Producer retries
batch_size = 16384                     # Batch size in bytes
linger_ms = 5                          # Linger time for batching
```

### Buffer Configuration

```toml
[buffer]
max_size = 1000                        # Maximum buffer size
overflow_strategy = "drop_oldest"       # Overflow handling strategy
persist_to_disk = true                 # Enable disk persistence
persist_path = "/tmp/signal_buffer"    # Persistence file path
```

### Audit Configuration

```toml
[audit]
enabled = true                         # Enable audit logging
file_path = "/var/log/imp/signals.jsonl"  # Local audit log file
max_file_size_mb = 100                # Maximum file size before rotation
max_files = 10                        # Maximum number of rotated files

[audit.s3]
bucket = "imp-audit-logs"             # S3 bucket for archival
prefix = "signals/"                   # S3 key prefix
upload_interval_sec = 300             # Upload interval
region = "us-east-1"                  # AWS region
```

## Environment Variables

All configuration options can be overridden with environment variables:

```bash
# Publisher configuration
SIGNAL_PUBLISHER_ENABLED=true
SIGNAL_PUBLISHER_BACKEND=redis
SIGNAL_PUBLISHER_BATCH_SIZE=100

# Redis configuration
REDIS_URL=redis://localhost:6379
REDIS_STREAM_NAME=trading_signals
REDIS_MAX_STREAM_LENGTH=10000

# Kafka configuration
KAFKA_BROKERS=localhost:9092
KAFKA_TOPIC=trading_signals
KAFKA_PARTITION_STRATEGY=symbol

# Buffer configuration
SIGNAL_BUFFER_MAX_SIZE=1000
SIGNAL_BUFFER_PERSIST_TO_DISK=true

# Audit configuration
AUDIT_ENABLED=true
AUDIT_FILE_PATH=/var/log/imp/signals.jsonl
AUDIT_S3_BUCKET=imp-audit-logs
```

## API Reference

### Core Types

```rust
/// Trading signal with emission metadata
pub struct TradingSignal {
    pub timestamp: i64,
    pub symbol: String,
    pub side: SignalSide,
    pub strength: f32,        // -1.0 to 1.0
    pub confidence: f32,      // 0.0 to 1.0
    pub correlation_id: String,
    pub feature_checksum: String,
    // ... additional fields
}

/// Signal side enumeration
pub enum SignalSide {
    Buy,
    Sell,
    Hold,
}
```

### Signal Publisher

```rust
/// Main signal publisher interface
impl SignalPublisher {
    /// Create a new signal publisher with configuration
    pub async fn new(config: SignalPublisherConfig) -> Result<Self>;
    
    /// Publish a single signal
    pub async fn publish_signal(&mut self, signal: TradingSignal) -> Result<()>;
    
    /// Publish multiple signals in batch
    pub async fn publish_batch(&mut self, signals: Vec<TradingSignal>) -> Result<()>;
    
    /// Check health of all configured backends
    pub async fn health_check(&self) -> HealthStatus;
    
    /// Get current metrics
    pub fn get_metrics(&self) -> SignalPublisherMetrics;
    
    /// Graceful shutdown
    pub async fn shutdown(self) -> Result<()>;
}
```

### Configuration Loading

```rust
/// Load configuration from file
let config = SignalEmissionConfig::from_file("config.toml")?;

/// Load configuration with environment variable overrides
let config = SignalEmissionConfig::from_file_with_env("config.toml")?;

/// Create configuration programmatically
let config = SignalEmissionConfig::builder()
    .publisher_backend(PublisherBackend::Redis)
    .redis_url("redis://localhost:6379")
    .audit_enabled(true)
    .build()?;
```

## Examples

### Basic Signal Emission

```rust
use signal_fusion::emission::{SignalPublisher, SignalEmissionConfig};

let config = SignalEmissionConfig::from_file("config.toml")?;
let mut publisher = SignalPublisher::new(config.publisher).await?;

let signal = TradingSignal {
    timestamp: chrono::Utc::now().timestamp_millis(),
    symbol: "BTCUSDT".to_string(),
    side: SignalSide::Buy,
    strength: 0.75,
    confidence: 0.85,
    correlation_id: "req_123".to_string(),
    feature_checksum: "abc123".to_string(),
    // ... other required fields
};

publisher.publish_signal(signal).await?;
```

### Batch Publishing

```rust
let signals = vec![signal1, signal2, signal3];
publisher.publish_batch(signals).await?;
```

### Health Monitoring

```rust
let health = publisher.health_check().await;
match health.status {
    ServiceStatus::Healthy => println!("All systems operational"),
    ServiceStatus::Degraded => println!("Some issues detected: {:?}", health.components),
    ServiceStatus::Unhealthy => println!("System unhealthy: {:?}", health.components),
}
```

### Metrics Collection

```rust
let metrics = publisher.get_metrics();
println!("Signals published: {}", metrics.signals_published_total);
println!("Average latency: {:.2}ms", metrics.avg_emission_latency_ms);
println!("Buffer utilization: {:.1}%", metrics.buffer_utilization_percent);
```

## Troubleshooting

### Common Issues

#### Connection Refused
```
Error: Redis error: Connection refused (os error 111)
```
**Solution**: Ensure Redis server is running and accessible at the configured URL.

#### Authentication Failed
```
Error: Redis error: NOAUTH Authentication required
```
**Solution**: Add authentication to Redis URL: `redis://:password@localhost:6379`

#### Buffer Overflow
```
Error: Buffer overflow: maximum size 1000 exceeded
```
**Solution**: Increase buffer size or check message bus connectivity.

#### Circuit Breaker Open
```
Error: Circuit breaker open: too many failures
```
**Solution**: Check backend health and wait for circuit breaker to recover.

### Debug Logging

Enable debug logging for detailed diagnostics:

```rust
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

### Health Check Endpoint

Start the health monitoring server:

```rust
use signal_fusion::emission::{HealthMonitor, HealthMonitorConfig};

let health_config = HealthMonitorConfig {
    http: Some(HealthHttpConfig {
        bind_address: "0.0.0.0:8080".to_string(),
        health_path: "/health".to_string(),
        metrics_path: "/metrics".to_string(),
    }),
    ..Default::default()
};

let health_monitor = HealthMonitor::new(health_config);
health_monitor.start_http_server().await?;
```

Access health status at `http://localhost:8080/health`

## Performance

### Benchmarks

Run performance benchmarks:

```bash
cargo bench --bench signal_emission_benchmarks
```

### Optimization Tips

1. **Use Batch Publishing**: Publish multiple signals together for better throughput
2. **Tune Buffer Size**: Larger buffers reduce I/O overhead but use more memory
3. **Configure Connection Pools**: More connections can improve concurrent performance
4. **Enable Compression**: Use gzip compression for Kafka to reduce network usage
5. **Adjust Timeouts**: Balance between responsiveness and reliability

### Performance Targets

- **Signal Validation**: <1ms per signal
- **Redis Publishing**: <5ms per signal
- **Kafka Publishing**: <10ms per signal
- **Audit Logging**: <2ms per event
- **End-to-End Latency**: <20ms for complete emission pipeline

## License

Licensed under the same terms as the main IMP project.

## Contributing

1. Run tests: `cargo test`
2. Run benchmarks: `cargo bench`
3. Check formatting: `cargo fmt --check`
4. Run linting: `cargo clippy`
5. Update documentation: `cargo doc --no-deps --open`

For more detailed documentation, see the `docs/` directory.