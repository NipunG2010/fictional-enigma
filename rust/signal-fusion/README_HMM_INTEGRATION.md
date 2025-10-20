# HMM Microservice Integration for Rust

This document provides comprehensive guidance on integrating the Rust inference engine with the HMM microservice for regime-aware signal fusion.

## Overview

The HMM microservice provides Hidden Markov Model inference capabilities that enable regime-aware signal fusion. The Rust client library provides a robust HTTP client with comprehensive error handling, retry logic, circuit breaker patterns, and fallback mechanisms.

## Documentation

Complete documentation is available in the following guides:

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation for all types and methods
- **[Configuration Guide](CONFIG.md)** - Detailed configuration options and examples
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Performance Tuning](docs/PERFORMANCE_TUNING.md)** - Optimization strategies and benchmarking
- **[Circuit Breaker](docs/CIRCUIT_BREAKER.md)** - Circuit breaker pattern implementation
- **[Monitoring & Metrics](docs/MONITORING_METRICS.md)** - Metrics collection and monitoring
- **[Error Handling Guide](ERROR_HANDLING_GUIDE.md)** - Error types and handling strategies
- **[Signal Fusion Guide](SIGNAL_FUSION_GUIDE.md)** - Signal fusion implementation details

## Quick Start

### Basic Usage

```rust
use signal_fusion::hmm_client::{HmmClient, HmmIntegration};
use signal_fusion::SignalComponents;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create HMM integration
    let mut hmm_integration = HmmIntegration::new()?;
    
    // Market signal components
    let signals = SignalComponents {
        s_ldc: 0.015,   // Lorentzian Distance Classifier signal
        s_mr: -0.008,   // Mean Reversion signal  
        s_tsmom: 0.022, // Time Series Momentum signal
    };
    
    // Get regime-aware fusion weights
    let fusion_weights = hmm_integration
        .get_fusion_weights_for_signals(&signals)
        .await?;
    
    println!("Fusion weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
             fusion_weights.w_ldc, fusion_weights.w_mr, fusion_weights.w_tsmom);
    
    Ok(())
}
```

### Production Configuration

```rust
use signal_fusion::hmm_client::{HmmClientConfig, HmmIntegration};
use signal_fusion::FusionWeights;
use std::time::Duration;

let production_config = HmmClientConfig {
    base_url: "http://hmm-service:8000".parse()?,
    timeout: Duration::from_millis(2000),
    retry_attempts: 3,
    retry_delay: Duration::from_millis(200),
    enable_fallback: true,
    fallback_weights: FusionWeights {
        w_ldc: 0.35,
        w_mr: 0.35, 
        w_tsmom: 0.30,
    },
    circuit_breaker_threshold: 3,
    circuit_breaker_timeout: Duration::from_secs(120),
};

let mut hmm_integration = HmmIntegration::with_config(production_config)?;
```

## API Reference

### HmmClient

The core HTTP client for communicating with the HMM microservice.

#### Methods

- `new()` - Create client with default configuration
- `with_config(config)` - Create client with custom configuration
- `get_state_probabilities(observations, request_id)` - Get HMM state probabilities
- `get_fusion_weights(observations, request_id)` - Get fusion weights for signals
- `predict(observations, request_id)` - Get complete prediction (states + weights)
- `health_check()` - Check service health
- `readiness_check()` - Check service readiness
- `get_model_info()` - Get current model information
- `reload_model()` - Trigger model reload

#### Example

```rust
let mut client = HmmClient::new()?;

// Check service health
let health = client.health_check().await?;
println!("Service status: {}", health.status);

// Get state probabilities
let observations = [0.015, -0.008, 0.022];
let response = client.get_state_probabilities(observations, None).await?;
println!("State probabilities: {:?}", response.state_probabilities);
```

### HmmIntegration

High-level integration helper that simplifies common workflows.

#### Methods

- `new()` - Create integration with default configuration
- `with_config(config)` - Create integration with custom configuration  
- `get_fusion_weights_for_signals(signals)` - Get weights for signal components
- `is_service_ready()` - Check if service is ready for requests
- `get_service_status()` - Get service and model status
- `get_circuit_breaker_status()` - Get circuit breaker state

#### Example

```rust
let mut integration = HmmIntegration::new()?;

// Check service readiness
if integration.is_service_ready().await {
    let signals = SignalComponents { s_ldc: 0.01, s_mr: 0.02, s_tsmom: 0.03 };
    let weights = integration.get_fusion_weights_for_signals(&signals).await?;
    // Use weights for signal fusion...
}
```

## Error Handling

The client provides comprehensive error handling with automatic fallback mechanisms.

### Error Types

```rust
pub enum HmmClientError {
    Network(reqwest::Error),                    // Network/HTTP errors
    ServiceUnavailable { status: StatusCode },  // Service unavailable
    InvalidRequest { message: String },         // Bad request (400)
    ModelError { message: String },             // Model/inference errors
    Timeout { duration: Duration },             // Request timeout
    ValidationError { field: String, message: String }, // Validation errors
    FallbackActivated { reason: String },       // Fallback mechanism used
}
```

### Automatic Fallback

The client automatically falls back to default weights when the HMM service is unavailable:

```rust
// Configure fallback weights
let config = HmmClientConfig {
    enable_fallback: true,
    fallback_weights: FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    },
    ..Default::default()
};

let mut client = HmmClient::with_config(config)?;

// This will use fallback weights if service is unavailable
let weights = client.get_fusion_weights([0.1, 0.2, 0.3], None).await?;
```

### Circuit Breaker Pattern

The client implements a circuit breaker to prevent cascading failures:

```rust
let config = HmmClientConfig {
    circuit_breaker_threshold: 5,              // Open after 5 failures
    circuit_breaker_timeout: Duration::from_secs(30), // Stay open for 30s
    ..Default::default()
};

let mut client = HmmClient::with_config(config)?;

// Check circuit breaker status
let (state, failure_count) = client.get_circuit_breaker_status();
println!("Circuit breaker: {} (failures: {})", state, failure_count);
```

### Retry Logic

Automatic retry with exponential backoff:

```rust
let config = HmmClientConfig {
    retry_attempts: 3,                          // Retry up to 3 times
    retry_delay: Duration::from_millis(100),    // Base delay of 100ms
    ..Default::default()
};
```

## Integration Patterns

### Signal Fusion Workflow

Complete workflow for regime-aware signal fusion:

```rust
use signal_fusion::{SignalFusion, SignalComponents, hmm_client::HmmIntegration};

async fn trading_signal_generation() -> Result<()> {
    let mut hmm_integration = HmmIntegration::new()?;
    let mut signal_fusion = SignalFusion::new(0.1, 60); // 0.1 threshold, 60s cooldown
    
    // Market data
    let signals = SignalComponents {
        s_ldc: 0.08,    // Strong LDC signal
        s_mr: -0.02,    // Weak mean reversion
        s_tsmom: 0.15,  // Strong momentum
    };
    
    // Get regime-aware weights
    let fusion_weights = hmm_integration
        .get_fusion_weights_for_signals(&signals)
        .await?;
    
    // Generate trading signal
    let timestamp = chrono::Utc::now().timestamp();
    if let Some(trading_signal) = signal_fusion.fuse_signals(
        signals,
        fusion_weights,
        timestamp,
        "BTCUSDT",
        "hmm_v2.1.0",
    )? {
        println!("Generated signal: {} {} (strength: {:.3})", 
                 trading_signal.side, trading_signal.symbol, trading_signal.strength);
    }
    
    Ok(())
}
```

### Health Monitoring

Monitor HMM service health in production:

```rust
async fn monitor_hmm_service(integration: &mut HmmIntegration) -> Result<()> {
    // Check service readiness
    let ready = integration.is_service_ready().await;
    
    // Get detailed status
    let (model_loaded, model_version) = integration.get_service_status().await?;
    
    // Check circuit breaker
    let (cb_state, failures) = integration.get_circuit_breaker_status();
    
    println!("HMM Service Status:");
    println!("  Ready: {}", ready);
    println!("  Model Loaded: {}", model_loaded);
    println!("  Model Version: {:?}", model_version);
    println!("  Circuit Breaker: {} ({} failures)", cb_state, failures);
    
    Ok(())
}
```

### Graceful Degradation

Handle service failures gracefully:

```rust
async fn robust_signal_processing(
    integration: &mut HmmIntegration,
    signals: &SignalComponents
) -> Result<FusionWeights> {
    match integration.get_fusion_weights_for_signals(signals).await {
        Ok(weights) => {
            // HMM service provided regime-aware weights
            Ok(weights)
        }
        Err(e) => {
            // Fall back to static weights
            warn!("HMM service failed, using static weights: {}", e);
            Ok(FusionWeights {
                w_ldc: 0.33,
                w_mr: 0.33,
                w_tsmom: 0.34,
            })
        }
    }
}
```

## Configuration Examples

### Development Environment

```rust
let dev_config = HmmClientConfig {
    base_url: "http://localhost:8000".parse()?,
    timeout: Duration::from_millis(10000),      // Generous timeout
    retry_attempts: 1,                          // Minimal retries
    enable_fallback: true,
    circuit_breaker_threshold: 10,              // Lenient threshold
    ..Default::default()
};
```

### Staging Environment

```rust
let staging_config = HmmClientConfig {
    base_url: "http://hmm-service-staging:8000".parse()?,
    timeout: Duration::from_millis(5000),
    retry_attempts: 2,
    enable_fallback: true,
    circuit_breaker_threshold: 5,
    circuit_breaker_timeout: Duration::from_secs(60),
    ..Default::default()
};
```

### Production Environment

```rust
let production_config = HmmClientConfig {
    base_url: "http://hmm-service:8000".parse()?,
    timeout: Duration::from_millis(2000),       // Strict timeout
    retry_attempts: 3,                          // Reasonable retries
    retry_delay: Duration::from_millis(200),    // Conservative backoff
    enable_fallback: true,
    fallback_weights: FusionWeights {           // Production fallback weights
        w_ldc: 0.35,
        w_mr: 0.35,
        w_tsmom: 0.30,
    },
    circuit_breaker_threshold: 3,               // Aggressive circuit breaker
    circuit_breaker_timeout: Duration::from_secs(120),
};
```

## Performance Considerations

### Request Latency

- Target latency: <20ms for inference requests
- Configure appropriate timeouts based on environment
- Use circuit breaker to prevent slow requests from cascading

### Caching

The HMM service implements server-side caching. For additional client-side caching:

```rust
use std::collections::HashMap;
use std::time::{Duration, Instant};

struct CachedWeights {
    weights: FusionWeights,
    timestamp: Instant,
}

struct ClientCache {
    cache: HashMap<[u32; 3], CachedWeights>, // Quantized observations as key
    ttl: Duration,
}

impl ClientCache {
    fn get_cached_weights(&self, observations: [f32; 3]) -> Option<FusionWeights> {
        let key = [
            (observations[0] * 1000.0) as u32,
            (observations[1] * 1000.0) as u32,
            (observations[2] * 1000.0) as u32,
        ];
        
        if let Some(cached) = self.cache.get(&key) {
            if cached.timestamp.elapsed() < self.ttl {
                return Some(cached.weights.clone());
            }
        }
        None
    }
}
```

### Connection Pooling

The underlying `reqwest` client automatically handles connection pooling. For high-throughput scenarios:

```rust
let client = reqwest::Client::builder()
    .pool_max_idle_per_host(10)
    .pool_idle_timeout(Duration::from_secs(30))
    .build()?;
```

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hmm_integration_fallback() {
        let config = HmmClientConfig {
            base_url: "http://invalid-host:9999".parse().unwrap(),
            enable_fallback: true,
            fallback_weights: FusionWeights {
                w_ldc: 0.5,
                w_mr: 0.3,
                w_tsmom: 0.2,
            },
            ..Default::default()
        };
        
        let mut integration = HmmIntegration::with_config(config).unwrap();
        let signals = SignalComponents { s_ldc: 0.1, s_mr: 0.2, s_tsmom: 0.3 };
        
        // Should fall back to configured weights
        let weights = integration.get_fusion_weights_for_signals(&signals).await.unwrap();
        assert_eq!(weights.w_ldc, 0.5);
    }
}
```

### Integration Tests

```rust
#[tokio::test]
#[ignore] // Requires running HMM service
async fn test_end_to_end_integration() {
    let mut integration = HmmIntegration::new().unwrap();
    
    // Verify service is ready
    assert!(integration.is_service_ready().await);
    
    // Test inference
    let signals = SignalComponents { s_ldc: 0.015, s_mr: -0.008, s_tsmom: 0.022 };
    let weights = integration.get_fusion_weights_for_signals(&signals).await.unwrap();
    
    // Verify weights are valid
    assert!((weights.w_ldc + weights.w_mr + weights.w_tsmom - 1.0).abs() < 1e-6);
    assert!(weights.w_ldc >= 0.0 && weights.w_ldc <= 1.0);
}
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error: Network(reqwest::Error { kind: Request, ... })
   ```
   - Verify HMM service is running
   - Check service URL and port
   - Verify network connectivity

2. **Timeout Errors**
   ```
   Error: Timeout { duration: 2s }
   ```
   - Increase timeout in configuration
   - Check service performance
   - Verify service is not overloaded

3. **Circuit Breaker Open**
   ```
   Error: ServiceUnavailable { status: 503 }
   ```
   - Wait for circuit breaker timeout
   - Check service health
   - Verify service configuration

4. **Model Not Loaded**
   ```
   Error: ModelError { message: "No model loaded" }
   ```
   - Check HMM service logs
   - Verify MinIO connectivity
   - Trigger model reload

### Debugging

Enable detailed logging:

```rust
use tracing::{info, warn, error, debug};
use tracing_subscriber;

// Initialize logging
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();

// Client will now log detailed information
let mut client = HmmClient::new()?;
```

### Monitoring

Key metrics to monitor:

- Request latency (p50, p95, p99)
- Error rate by error type
- Circuit breaker state changes
- Fallback activation frequency
- Service availability

## Best Practices

1. **Always Enable Fallback**: Configure meaningful fallback weights for production
2. **Set Appropriate Timeouts**: Balance responsiveness with reliability
3. **Monitor Circuit Breaker**: Track circuit breaker state changes
4. **Use Request IDs**: Include request IDs for tracing and debugging
5. **Handle Errors Gracefully**: Don't let HMM service failures break trading
6. **Test Failure Scenarios**: Regularly test with service unavailable
7. **Monitor Performance**: Track request latency and success rates
8. **Use Environment-Specific Config**: Different settings for dev/staging/prod

## Examples

See the complete examples in:
- `examples/hmm_integration_example.rs` - Comprehensive integration examples
- `src/hmm_client.rs` - Unit tests and usage patterns

Run examples with:
```bash
cargo run --example hmm_integration_example
```