# Circuit Breaker Implementation

## Overview

The HMM client includes a robust circuit breaker pattern to handle service failures gracefully. The circuit breaker prevents cascading failures by stopping requests to a failing service and allowing it time to recover.

## State Machine

The circuit breaker operates in three states:

### 1. Closed (Normal Operation)
- All requests are allowed through
- Failures are counted
- When failure count reaches threshold, transitions to **Open**

### 2. Open (Service Failing)
- Requests are immediately rejected without attempting service call
- Saves resources by not making doomed requests
- After timeout period, transitions to **Half-Open**
- Rejected requests increment the `rejected_requests` metric

### 3. Half-Open (Testing Recovery)
- Allows a single test request through
- If successful: transitions to **Closed** (service recovered)
- If failed: transitions back to **Open** (service still failing)

## State Transitions

```
┌─────────┐
│ Closed  │ ◄──────────────────────┐
└────┬────┘                        │
     │                             │
     │ failures >= threshold       │ success
     │                             │
     ▼                        ┌────┴─────┐
┌─────────┐   timeout        │Half-Open │
│  Open   │ ────────────────►└──────────┘
└─────────┘                        │
     ▲                             │
     │                             │
     └─────────────────────────────┘
              failure
```

## Configuration

```rust
use signal_fusion::hmm_client::HmmClientConfig;
use std::time::Duration;

let config = HmmClientConfig {
    circuit_breaker_threshold: 5,              // Open after 5 failures
    circuit_breaker_timeout: Duration::from_secs(30), // Try recovery after 30s
    enable_fallback: true,                     // Use fallback weights
    ..Default::default()
};
```

### Configuration Parameters

- **circuit_breaker_threshold**: Number of consecutive failures before opening (default: 5)
- **circuit_breaker_timeout**: Time to wait before attempting recovery (default: 30s)
- **enable_fallback**: Whether to use fallback weights when service fails (default: true)

## Metrics

The circuit breaker tracks comprehensive metrics:

```rust
pub struct CircuitBreakerMetrics {
    pub total_requests: u64,           // Total service requests attempted
    pub successful_requests: u64,      // Successful service responses
    pub failed_requests: u64,          // Failed service requests
    pub circuit_breaker_opens: u64,    // Times circuit opened
    pub circuit_breaker_closes: u64,   // Times circuit closed
    pub half_open_attempts: u64,       // Recovery attempts made
    pub rejected_requests: u64,        // Requests rejected while open
}
```

### Accessing Metrics

```rust
let client = HmmClient::new()?;

// Get current state and failure count
let (state, failure_count) = client.get_circuit_breaker_status();
println!("State: {}, Failures: {}", state, failure_count);

// Get detailed metrics
let metrics = client.get_circuit_breaker_metrics();
println!("Total requests: {}", metrics.total_requests);
println!("Success rate: {:.2}%", 
    (metrics.successful_requests as f64 / metrics.total_requests as f64) * 100.0);
```

## Logging

The circuit breaker provides detailed logging at different levels:

### INFO Level
- Circuit breaker initialization with configuration
- State transitions (Open → Half-Open → Closed)
- Successful recovery events

### WARN Level
- Circuit breaker opening due to failures
- Failure in half-open state (reopening circuit)

### DEBUG Level
- Individual request successes/failures
- Failure count increments
- Rejected requests while open

## Fallback Mechanism

When `enable_fallback` is true, the client returns fallback weights instead of errors:

```rust
let config = HmmClientConfig {
    enable_fallback: true,
    fallback_weights: FusionWeights {
        w_ldc: 0.33,
        w_mr: 0.33,
        w_tsmom: 0.34,
    },
    ..Default::default()
};
```

### Behavior with Fallback

- **User-facing**: Always receives valid response (100% success rate)
- **Internal**: Circuit breaker tracks actual service health
- **Benefit**: System remains operational even when HMM service is down

## Best Practices

### 1. Tune Threshold Based on Traffic

```rust
// High-traffic system: Lower threshold
let config = HmmClientConfig {
    circuit_breaker_threshold: 3,
    circuit_breaker_timeout: Duration::from_secs(10),
    ..Default::default()
};

// Low-traffic system: Higher threshold
let config = HmmClientConfig {
    circuit_breaker_threshold: 10,
    circuit_breaker_timeout: Duration::from_secs(60),
    ..Default::default()
};
```

### 2. Monitor Metrics

```rust
// Periodically check circuit breaker health
tokio::spawn(async move {
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        
        let metrics = client.get_circuit_breaker_metrics();
        let (state, _) = client.get_circuit_breaker_status();
        
        if state == "open" {
            warn!("Circuit breaker is open - service degraded");
        }
        
        if metrics.total_requests > 0 {
            let success_rate = (metrics.successful_requests as f64 
                / metrics.total_requests as f64) * 100.0;
            info!("HMM service success rate: {:.2}%", success_rate);
        }
    }
});
```

### 3. Alert on Prolonged Failures

```rust
// Alert if circuit breaker opens frequently
if metrics.circuit_breaker_opens > 10 {
    error!("Circuit breaker opened {} times - investigate HMM service", 
           metrics.circuit_breaker_opens);
}

// Alert if many requests rejected
if metrics.rejected_requests > 100 {
    warn!("Circuit breaker rejected {} requests - service may be down", 
          metrics.rejected_requests);
}
```

### 4. Use Appropriate Timeouts

```rust
// Fast recovery for transient issues
let config = HmmClientConfig {
    circuit_breaker_timeout: Duration::from_secs(10),
    ..Default::default()
};

// Longer timeout for persistent issues
let config = HmmClientConfig {
    circuit_breaker_timeout: Duration::from_secs(60),
    ..Default::default()
};
```

## Testing

### Unit Tests

The circuit breaker includes comprehensive unit tests:

```bash
cargo test circuit_breaker --lib
```

### Integration Tests

Test with actual service failures:

```bash
cargo test circuit_breaker --test hmm_integration_tests
```

### Demo Example

Run the interactive demonstration:

```bash
cargo run --example circuit_breaker_demo
```

## Performance Impact

The circuit breaker has minimal performance overhead:

- **Closed state**: Single atomic check per request (~1ns)
- **Open state**: Immediate rejection, no network call (saves ~100ms)
- **Half-open state**: Single test request

## Troubleshooting

### Circuit Breaker Stuck Open

**Symptom**: Circuit breaker remains open indefinitely

**Causes**:
1. Service is actually down
2. Timeout too short for service recovery
3. Network issues preventing recovery

**Solutions**:
```rust
// Increase timeout
let config = HmmClientConfig {
    circuit_breaker_timeout: Duration::from_secs(120),
    ..Default::default()
};

// Check service health manually
let health = client.health_check().await?;
println!("Service status: {}", health.status);
```

### Too Many False Opens

**Symptom**: Circuit breaker opens frequently on transient errors

**Causes**:
1. Threshold too low
2. Network instability
3. Service under heavy load

**Solutions**:
```rust
// Increase threshold
let config = HmmClientConfig {
    circuit_breaker_threshold: 10,
    ..Default::default()
};

// Increase retry attempts
let config = HmmClientConfig {
    retry_attempts: 5,
    retry_delay: Duration::from_millis(200),
    ..Default::default()
};
```

### Metrics Not Updating

**Symptom**: Metrics show zero values

**Causes**:
1. No requests made yet
2. Using different client instance

**Solutions**:
```rust
// Ensure using same client instance
let client = Arc::new(HmmClient::new()?);

// Share across threads
let client_clone = Arc::clone(&client);
tokio::spawn(async move {
    let metrics = client_clone.get_circuit_breaker_metrics();
    println!("Metrics: {:?}", metrics);
});
```

## Related Documentation

- [HMM Client API](./HMM_CLIENT.md)
- [Fallback Mechanism](./FALLBACK.md)
- [Error Handling](./ERROR_HANDLING.md)
- [Monitoring and Metrics](./MONITORING.md)
