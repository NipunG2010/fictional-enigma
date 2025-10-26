# Signal Emission Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the signal emission system.

## Quick Diagnostics

### Check System Health

```rust
use signal_fusion::emission::SignalPublisher;

let health = publisher.health_check().await;
println!("Health status: {:?}", health);
```

### Enable Debug Logging

```rust
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

### Check Metrics

```rust
let metrics = publisher.get_metrics();
println!("Metrics: {:#?}", metrics);
```

## Common Issues

### 1. Connection Issues

#### Redis Connection Refused

**Error:**
```
Error: Redis error: Connection refused (os error 111)
```

**Causes:**
- Redis server is not running
- Wrong host/port configuration
- Network connectivity issues
- Firewall blocking connection

**Solutions:**
1. Check if Redis is running:
   ```bash
   redis-cli ping
   ```

2. Verify configuration:
   ```toml
   [redis]
   url = "redis://localhost:6379"  # Check host and port
   ```

3. Test network connectivity:
   ```bash
   telnet localhost 6379
   ```

4. Check firewall rules:
   ```bash
   sudo ufw status
   ```

#### Redis Authentication Failed

**Error:**
```
Error: Redis error: NOAUTH Authentication required
```

**Solutions:**
1. Add password to Redis URL:
   ```toml
   [redis]
   url = "redis://:your_password@localhost:6379"
   ```

2. Or use environment variable:
   ```bash
   export REDIS_URL="redis://:your_password@localhost:6379"
   ```

#### Kafka Connection Issues

**Error:**
```
Error: Kafka error: Broker transport failure
```

**Solutions:**
1. Check Kafka broker status:
   ```bash
   kafka-topics.sh --bootstrap-server localhost:9092 --list
   ```

2. Verify broker configuration:
   ```toml
   [kafka]
   brokers = ["localhost:9092"]  # Check broker addresses
   ```

3. Check if topic exists:
   ```bash
   kafka-topics.sh --bootstrap-server localhost:9092 --describe --topic trading_signals
   ```

### 2. Performance Issues

#### High Latency

**Symptoms:**
- Slow signal publishing
- High emission latency in metrics
- Timeouts

**Diagnostics:**
```rust
let metrics = publisher.get_metrics();
if metrics.avg_emission_latency_ms > 100.0 {
    println!("High latency detected: {:.2}ms", metrics.avg_emission_latency_ms);
}
```

**Solutions:**
1. Increase connection pool size:
   ```toml
   [redis]
   connection_pool_size = 20  # Increase from default 10
   ```

2. Use batch publishing:
   ```rust
   publisher.publish_batch(signals).await?;
   ```

3. Tune buffer settings:
   ```toml
   [buffer]
   max_size = 2000  # Increase buffer size
   ```

4. Optimize network settings:
   ```toml
   [redis]
   connection_timeout_ms = 3000
   command_timeout_ms = 2000
   ```

#### Memory Usage Issues

**Symptoms:**
- High memory consumption
- Out of memory errors
- Slow garbage collection

**Solutions:**
1. Reduce buffer size:
   ```toml
   [buffer]
   max_size = 500  # Reduce from default 1000
   ```

2. Enable buffer persistence:
   ```toml
   [buffer]
   persist_to_disk = true
   ```

3. Tune batch sizes:
   ```toml
   [publisher]
   batch_size = 50  # Reduce from default 100
   ```

### 3. Circuit Breaker Issues

#### Circuit Breaker Stuck Open

**Error:**
```
Error: Circuit breaker open: too many failures
```

**Diagnostics:**
```rust
let health = publisher.health_check().await;
for (component, status) in health.components {
    if status.error_message.is_some() {
        println!("Component {} unhealthy: {:?}", component, status.error_message);
    }
}
```

**Solutions:**
1. Check backend health:
   ```bash
   # Redis
   redis-cli ping
   
   # Kafka
   kafka-broker-api-versions.sh --bootstrap-server localhost:9092
   ```

2. Adjust circuit breaker settings:
   ```toml
   [publisher.circuit_breaker]
   failure_threshold = 10  # Increase threshold
   timeout_sec = 60       # Increase timeout
   ```

3. Wait for automatic recovery or restart the service

#### Frequent Circuit Breaker Opens

**Symptoms:**
- Circuit breaker opens frequently
- Intermittent publishing failures

**Solutions:**
1. Increase failure threshold:
   ```toml
   [publisher.circuit_breaker]
   failure_threshold = 10  # Increase from default 5
   ```

2. Improve retry policy:
   ```toml
   [publisher.retry_policy]
   max_attempts = 5
   base_delay_ms = 200
   max_delay_ms = 10000
   ```

3. Check network stability and backend performance

### 4. Buffer Issues

#### Buffer Overflow

**Error:**
```
Error: Buffer overflow: maximum size 1000 exceeded
```

**Solutions:**
1. Increase buffer size:
   ```toml
   [buffer]
   max_size = 2000
   ```

2. Change overflow strategy:
   ```toml
   [buffer]
   overflow_strategy = "drop_oldest"  # or "drop_newest"
   ```

3. Check backend connectivity and performance

4. Enable buffer persistence:
   ```toml
   [buffer]
   persist_to_disk = true
   persist_path = "/tmp/signal_buffer.json"
   ```

#### Buffer Not Draining

**Symptoms:**
- Buffer utilization stays high
- Signals not being published

**Diagnostics:**
```rust
let metrics = publisher.get_metrics();
println!("Buffer utilization: {:.1}%", metrics.buffer_utilization_percent);
```

**Solutions:**
1. Check backend health
2. Verify network connectivity
3. Check circuit breaker status
4. Increase flush frequency:
   ```toml
   [publisher]
   flush_interval_ms = 500  # Reduce from default 1000
   ```

### 5. Validation Issues

#### Signal Validation Failures

**Error:**
```
Error: Signal validation failed: Invalid signal strength: 1.5 (must be between -1.0 and 1.0)
```

**Solutions:**
1. Check signal data before publishing:
   ```rust
   if signal.strength < -1.0 || signal.strength > 1.0 {
       println!("Invalid strength: {}", signal.strength);
   }
   ```

2. Adjust validation rules:
   ```toml
   [validation.rules]
   strength_min = -1.0
   strength_max = 1.0
   strict_mode = false  # Allow some validation errors
   ```

3. Fix signal generation logic

#### Schema Validation Errors

**Error:**
```
Error: Signal validation failed: Missing required field: correlation_id
```

**Solutions:**
1. Ensure all required fields are set:
   ```rust
   let signal = TradingSignal {
       timestamp: chrono::Utc::now().timestamp_millis(),
       symbol: "BTCUSDT".to_string(),
       side: SignalSide::Buy,
       strength: 0.75,
       confidence: 0.85,
       correlation_id: uuid::Uuid::new_v4().to_string(), // Required
       feature_checksum: "abc123".to_string(),           // Required
       // ... other fields
   };
   ```

2. Use validation helpers:
   ```rust
   use signal_fusion::emission::SignalValidator;
   
   let validator = SignalValidator::new(validation_config);
   validator.validate(&signal)?;
   ```

### 6. Audit Logging Issues

#### Audit Logs Not Written

**Symptoms:**
- No audit log files created
- Missing audit entries

**Solutions:**
1. Check audit configuration:
   ```toml
   [audit]
   enabled = true
   file_path = "/var/log/imp/signals.jsonl"
   ```

2. Verify file permissions:
   ```bash
   mkdir -p /var/log/imp
   chmod 755 /var/log/imp
   ```

3. Check disk space:
   ```bash
   df -h /var/log
   ```

#### S3 Upload Failures

**Error:**
```
Error: Audit logging error: S3 upload failed: Access Denied
```

**Solutions:**
1. Check S3 credentials:
   ```toml
   [audit.s3.credentials]
   access_key_id = "your-access-key"
   secret_access_key = "your-secret-key"
   ```

2. Verify S3 bucket permissions:
   ```bash
   aws s3 ls s3://imp-audit-logs/
   ```

3. Check IAM policy:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:PutObject",
           "s3:GetObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::imp-audit-logs",
           "arn:aws:s3:::imp-audit-logs/*"
         ]
       }
     ]
   }
   ```

### 7. Configuration Issues

#### Configuration Not Loading

**Error:**
```
Error: Configuration error: No such file or directory (os error 2)
```

**Solutions:**
1. Check file path:
   ```rust
   let config = SignalEmissionConfig::from_file("signal_emission.toml")?;
   ```

2. Use absolute path:
   ```rust
   let config = SignalEmissionConfig::from_file("/etc/imp/signal_emission.toml")?;
   ```

3. Check file permissions:
   ```bash
   ls -la signal_emission.toml
   ```

#### Environment Variable Override Not Working

**Solutions:**
1. Check environment variable format:
   ```bash
   export PUBLISHER_ENABLED=true
   export REDIS_URL=redis://localhost:6379
   export KAFKA_BROKERS=localhost:9092
   ```

2. Use configuration with environment overrides:
   ```rust
   let config = SignalEmissionConfig::from_file_with_env("config.toml")?;
   ```

### 8. Monitoring Issues

#### Metrics Not Available

**Solutions:**
1. Enable metrics collection:
   ```toml
   [monitoring]
   prometheus_enabled = true
   ```

2. Start metrics server:
   ```toml
   [monitoring.health]
   http_enabled = true
   bind_address = "0.0.0.0:8080"
   metrics_path = "/metrics"
   ```

3. Check metrics endpoint:
   ```bash
   curl http://localhost:8080/metrics
   ```

#### Health Check Endpoint Not Responding

**Solutions:**
1. Check server configuration:
   ```toml
   [monitoring.health]
   http_enabled = true
   bind_address = "0.0.0.0:8080"
   health_path = "/health"
   ```

2. Verify port is not in use:
   ```bash
   netstat -tlnp | grep 8080
   ```

3. Check firewall rules:
   ```bash
   sudo ufw allow 8080
   ```

## Debugging Tools

### 1. Enable Detailed Logging

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

tracing_subscriber::registry()
    .with(tracing_subscriber::fmt::layer().with_target(true))
    .with(tracing_subscriber::EnvFilter::from_default_env())
    .init();
```

Set log level:
```bash
export RUST_LOG=signal_fusion=debug
```

### 2. Health Check Script

```bash
#!/bin/bash
# health_check.sh

echo "Checking Redis..."
redis-cli ping

echo "Checking Kafka..."
kafka-topics.sh --bootstrap-server localhost:9092 --list

echo "Checking signal emission health..."
curl -s http://localhost:8080/health | jq .

echo "Checking metrics..."
curl -s http://localhost:8080/metrics | grep signal_emission
```

### 3. Configuration Validation

```rust
use signal_fusion::emission::SignalEmissionConfig;

fn validate_config() -> Result<(), Box<dyn std::error::Error>> {
    let config = SignalEmissionConfig::from_file("signal_emission.toml")?;
    
    // Validate configuration
    if config.publisher.enabled && config.publisher.backend == PublisherBackend::None {
        println!("Warning: Publisher enabled but backend is None");
    }
    
    if config.redis.is_some() && config.publisher.backend == PublisherBackend::Kafka {
        println!("Warning: Redis configured but not used");
    }
    
    println!("Configuration is valid");
    Ok(())
}
```

### 4. Performance Monitoring

```rust
use std::time::Instant;

async fn monitor_performance(publisher: &SignalPublisher) {
    let start = Instant::now();
    
    // Publish test signal
    let result = publisher.publish_signal(test_signal).await;
    
    let duration = start.elapsed();
    println!("Publish latency: {:?}", duration);
    
    let metrics = publisher.get_metrics();
    println!("Buffer utilization: {:.1}%", metrics.buffer_utilization_percent);
    println!("Error rate: {:.2}%", 
        (metrics.publisher_errors_total as f64 / metrics.signals_published_total as f64) * 100.0
    );
}
```

## Getting Help

### 1. Collect Diagnostic Information

Before reporting issues, collect:

- Configuration file
- Log output with debug level
- Health check results
- Metrics snapshot
- System information (OS, Rust version, dependencies)

### 2. Enable Debug Mode

```toml
[development]
enabled = true
```

### 3. Test with Minimal Configuration

```toml
[publisher]
enabled = true
backend = "none"  # Disable actual publishing for testing

[validation]
enabled = false   # Disable validation for testing

[audit]
enabled = false   # Disable audit logging for testing
```

### 4. Check Dependencies

```bash
cargo tree | grep -E "(redis|rdkafka|tokio)"
```

### 5. Run Integration Tests

```bash
cargo test --test signal_emission_integration_tests
```

This troubleshooting guide covers the most common issues. For additional help, check the logs with debug level enabled and collect the diagnostic information listed above.