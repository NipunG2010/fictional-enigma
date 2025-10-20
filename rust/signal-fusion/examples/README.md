# HMM Integration Examples

This directory contains comprehensive examples demonstrating how to integrate the Rust inference engine with the HMM microservice for regime-aware signal fusion.

## Overview

The examples cover all aspects of the HMM integration:
- Basic client usage and API calls
- Custom configuration management
- Error handling and fallback mechanisms
- Circuit breaker behavior
- Monitoring and metrics collection
- Production-ready integration patterns

## Prerequisites

Before running the examples, ensure:
1. The HMM microservice is running on `http://localhost:8000`
2. Rust toolchain is installed (1.70+)
3. All dependencies are available

To start the HMM service:
```bash
cd py/hmm_service
docker-compose up -d
# or
python app.py
```

## Examples

### 1. Basic HMM Integration (`hmm_integration_example.rs`)

**Purpose**: Demonstrates the complete HMM integration workflow from basic API calls to production-ready patterns.

**What it covers**:
- Basic HMM client usage with health checks
- Signal fusion with regime-aware weights
- Error handling and fallback scenarios
- Production integration patterns

**Run**:
```bash
cargo run --example hmm_integration_example
```

**Key sections**:
- `basic_hmm_client_example()`: Shows how to create a client, check service health, and make API calls
- `signal_fusion_with_hmm_example()`: Demonstrates end-to-end signal fusion with HMM weights
- `error_handling_example()`: Shows fallback behavior when service is unavailable
- `production_integration_example()`: Production-ready pattern with monitoring and health checks

**Requirements covered**: 1.1, 3.1, 3.2

### 2. Configuration Management (`configuration_demo.rs`)

**Purpose**: Shows all the ways to configure the HMM integration.

**What it covers**:
- Default configuration
- Environment variable configuration
- TOML file configuration
- Configuration validation
- Converting to client configuration
- Saving configuration to file

**Run**:
```bash
cargo run --example configuration_demo
```

**Key features**:
- Demonstrates loading from multiple sources
- Shows configuration validation
- Provides examples for different environments (dev, staging, production)

**Requirements covered**: 3.3, 3.4

### 3. Circuit Breaker Demo (`circuit_breaker_demo.rs`)

**Purpose**: Demonstrates the circuit breaker pattern for handling service failures gracefully.

**What it covers**:
- Circuit breaker state transitions (Closed → Open → Half-Open)
- Failure counting and threshold triggering
- Timeout-based recovery
- Metrics tracking
- Fallback activation

**Run**:
```bash
cargo run --example circuit_breaker_demo
```

**Key phases**:
1. **Phase 1**: Triggering circuit breaker to open (3 consecutive failures)
2. **Phase 2**: Circuit breaker rejecting requests while open
3. **Phase 3**: Timeout-based recovery to half-open state
4. **Phase 4**: Displaying comprehensive metrics

**Requirements covered**: 4.1, 4.2, 4.3, 4.4, 4.5

### 4. Monitoring and Metrics (`monitoring_demo.rs`)

**Purpose**: Demonstrates comprehensive monitoring and metrics collection capabilities.

**What it covers**:
- Request metrics (count, duration, errors)
- Cache metrics (hits, misses, size, evictions)
- Circuit breaker state metrics
- Fallback activation metrics
- Metrics export (JSON and Prometheus formats)

**Run**:
```bash
cargo run --example monitoring_demo
```

**Metrics categories**:
- **Request Metrics**: Total requests, success/failure rates, latency statistics
- **Cache Metrics**: Hit rate, size, evictions
- **Circuit Breaker Metrics**: State transitions, rejected requests
- **Fallback Metrics**: Activation reasons, current status
- **System Metrics**: Uptime, timestamps

**Export formats**:
- JSON: Structured format for logging and analysis
- Prometheus: Time-series format for monitoring systems

**Requirements covered**: 2.5, 4.5, 6.5

### 5. Signal Fusion Demo (`signal_fusion_demo.rs`)

**Purpose**: Demonstrates the signal fusion engine with regime-aware weighting.

**What it covers**:
- Signal component validation
- Weight normalization
- Threshold application
- Cooldown period management
- Trading signal generation

**Run**:
```bash
cargo run --example signal_fusion_demo
```

**Requirements covered**: 5.1, 5.2, 5.3, 5.4, 5.5

### 6. Error Handling Demo (`error_handling_demo.rs`)

**Purpose**: Comprehensive error handling scenarios and recovery strategies.

**What it covers**:
- Network errors and retries
- Timeout handling
- Service unavailability
- Invalid responses
- Fallback activation
- Error classification (transient vs permanent)

**Run**:
```bash
cargo run --example error_handling_demo
```

**Requirements covered**: 6.1, 6.2, 6.3, 6.4, 6.5

## Common Usage Patterns

### Pattern 1: Basic Integration

```rust
use signal_fusion::hmm_client::HmmIntegration;

#[tokio::main]
async fn main() -> Result<()> {
    // Create integration with defaults
    let mut integration = HmmIntegration::new()?;
    
    // Get weights for signals
    let signals = SignalComponents { 
        s_ldc: 0.05, 
        s_mr: -0.02, 
        s_tsmom: 0.08 
    };
    
    let weights = integration
        .get_fusion_weights_for_signals(&signals)
        .await?;
    
    println!("Weights: {:?}", weights);
    Ok(())
}
```

### Pattern 2: Custom Configuration

```rust
use signal_fusion::hmm_client::{HmmClientConfig, HmmIntegration};
use std::time::Duration;

let config = HmmClientConfig {
    base_url: "http://production:8000".parse()?,
    timeout: Duration::from_millis(2000),
    retry_attempts: 3,
    enable_fallback: true,
    circuit_breaker_threshold: 5,
    ..Default::default()
};

let mut integration = HmmIntegration::with_config(config)?;
```

### Pattern 3: Production Monitoring

```rust
// Get comprehensive metrics
let metrics = integration.get_metrics();

// Export for monitoring system
let prometheus_metrics = integration.export_metrics(MetricsFormat::Prometheus)?;

// Check circuit breaker status
let (state, failures) = integration.get_circuit_breaker_status();
if failures > 0 {
    warn!("Circuit breaker: {} (failures: {})", state, failures);
}
```

## Configuration Files

### Example TOML Configuration

See `hmm_integration.example.toml` for a complete configuration example:

```toml
[hmm_service]
url = "http://localhost:8000"
timeout_ms = 5000
retry_attempts = 3
retry_delay_ms = 100

[circuit_breaker]
threshold = 5
timeout_sec = 30

[cache]
ttl_sec = 60
max_size = 1000

[fallback]
enabled = true
w_ldc = 0.33
w_mr = 0.33
w_tsmom = 0.34

[signal_fusion]
threshold = 0.3
cooldown_sec = 60
```

### Environment Variables

All configuration can be overridden via environment variables:

```bash
# Service configuration
export HMM_SERVICE_URL=http://localhost:8000
export HMM_SERVICE_TIMEOUT_MS=5000
export HMM_SERVICE_RETRY_ATTEMPTS=3

# Circuit breaker
export HMM_CIRCUIT_BREAKER_THRESHOLD=5
export HMM_CIRCUIT_BREAKER_TIMEOUT_SEC=30

# Cache
export HMM_CACHE_TTL_SEC=60
export HMM_CACHE_MAX_SIZE=1000

# Fallback weights
export HMM_FALLBACK_W_LDC=0.33
export HMM_FALLBACK_W_MR=0.33
export HMM_FALLBACK_W_TSMOM=0.34

# Signal fusion
export SIGNAL_FUSION_THRESHOLD=0.3
export SIGNAL_FUSION_COOLDOWN_SEC=60
```

## Troubleshooting

### Service Connection Issues

If you see connection errors:
1. Verify HMM service is running: `curl http://localhost:8000/health`
2. Check service logs for errors
3. Verify network connectivity
4. Check firewall settings

### Circuit Breaker Opening

If the circuit breaker opens frequently:
1. Check HMM service health and performance
2. Review service logs for errors
3. Consider increasing `circuit_breaker_threshold`
4. Verify network stability

### Low Cache Hit Rate

If cache hit rate is low:
1. Increase `cache_ttl_sec` for longer retention
2. Increase `cache_max_size` for more entries
3. Review signal patterns for high variability
4. Check if observations are being rounded consistently

### High Latency

If requests are slow:
1. Check HMM service performance
2. Review network latency
3. Verify cache is working (check hit rate)
4. Consider reducing `timeout_ms` to fail faster
5. Enable fallback for faster degraded operation

## Performance Targets

The integration is designed to meet these performance targets:

- **Cache hit latency**: <1ms
- **Cache miss + service call**: <25ms
- **Fallback activation**: <1ms
- **Full fusion pipeline**: <30ms
- **Concurrent requests**: 100+ simultaneous

## Testing

To run integration tests:

```bash
# Run all tests
cargo test --manifest-path rust/signal-fusion/Cargo.toml

# Run specific test
cargo test --manifest-path rust/signal-fusion/Cargo.toml hmm_integration

# Run with logging
RUST_LOG=debug cargo test --manifest-path rust/signal-fusion/Cargo.toml
```

## Benchmarks

To run performance benchmarks:

```bash
cargo bench --manifest-path rust/signal-fusion/Cargo.toml
```

This will benchmark:
- Cache hit/miss latency
- Service call latency
- Fallback activation latency
- Full fusion pipeline performance

## Additional Resources

- **Design Document**: `../../.kiro/specs/rust-hmm-integration/design.md`
- **Requirements**: `../../.kiro/specs/rust-hmm-integration/requirements.md`
- **API Documentation**: Run `cargo doc --open` in the signal-fusion directory
- **HMM Service API**: `http://localhost:8000/docs` (when service is running)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the design and requirements documents
3. Check HMM service logs and documentation
4. Review existing integration tests for patterns
