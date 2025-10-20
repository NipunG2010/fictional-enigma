# Signal Fusion Engine Guide

## Overview

The Signal Fusion Engine combines multiple trading signals (LDC, MR, TSMOM) using regime-aware weights to generate final trading signals. This guide covers the enhanced features implemented to meet production requirements.

## Features

### 1. Input Signal Validation (Requirement 5.3)

All signal components are validated to ensure they are within the expected range `[-1.0, 1.0]` and are finite values.

```rust
let components = SignalComponents {
    s_ldc: 0.8,
    s_mr: 0.6,
    s_tsmom: 0.4,
};

// Validation happens automatically in fuse_signals()
// Invalid signals will return an error
components.validate()?; // Manual validation if needed
```

**Validation Rules:**
- All signals must be finite (not NaN or Infinity)
- All signals must be in range `[-1.0, 1.0]`
- Validation errors are logged with detailed information

### 2. Weight Normalization (Requirement 5.2)

Weights can be automatically normalized to ensure proper signal scaling. Normalization preserves the sign of weights while ensuring the sum of absolute values equals 1.0.

```rust
let weights = FusionWeights {
    w_ldc: 0.6,
    w_mr: 0.3,
    w_tsmom: 0.1,
};

// Automatic normalization (default behavior)
let mut fusion = SignalFusion::new(0.3, 60);

// Disable normalization if needed
let mut fusion = SignalFusion::with_normalization(0.3, 60, false);

// Manual normalization
let normalized = weights.normalize();
```

**Normalization Behavior:**
- Preserves sign of weights (negative weights remain negative)
- Handles zero-sum weights by returning equal weights (0.33, 0.33, 0.34)
- Logs warning when weight sum is near zero

### 3. Detailed Logging (Requirement 5.5)

The fusion engine provides comprehensive logging at multiple levels:

```rust
// Initialize tracing for logging
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::INFO)
    .init();
```

**Log Levels:**
- `DEBUG`: Input validation, weight normalization, threshold checks, cooldown checks
- `INFO`: Signal generation with full details (side, strength, confidence, components, weights)
- `WARN`: Validation failures, zero-sum weights

**Example Log Output:**
```
INFO signal_fusion: Generated BUY signal for BTCUSDT: strength=0.6600, confidence=0.6600, 
  components=[LDC:0.800, MR:0.600, TSMOM:0.400], 
  weights=[LDC:0.500, MR:0.300, TSMOM:0.200]
```

### 4. Threshold and Cooldown Logic (Requirement 5.4)

The engine implements robust threshold and cooldown mechanisms:

**Threshold:**
- Signals below the threshold are suppressed
- Threshold is applied to the absolute value of the fused signal
- Configurable per SignalFusion instance

**Cooldown:**
- Prevents signal generation within a specified time period
- Measured in seconds
- Tracked per SignalFusion instance
- Can be checked without generating a signal

```rust
let mut fusion = SignalFusion::new(0.3, 60); // threshold=0.3, cooldown=60s

// Check if in cooldown period
if fusion.is_in_cooldown(current_timestamp) {
    println!("Still in cooldown period");
}

// Get cooldown configuration
let cooldown = fusion.cooldown_period(); // Returns 60
let threshold = fusion.threshold(); // Returns 0.3
```

## Usage Examples

### Basic Signal Fusion

```rust
use signal_fusion::{SignalComponents, FusionWeights, SignalFusion};

let mut fusion = SignalFusion::new(0.3, 60);

let components = SignalComponents {
    s_ldc: 0.8,
    s_mr: 0.6,
    s_tsmom: 0.4,
};

let weights = FusionWeights {
    w_ldc: 0.5,
    w_mr: 0.3,
    w_tsmom: 0.2,
};

match fusion.fuse_signals(components, weights, 1000, "BTCUSDT", "v1.0") {
    Ok(Some(signal)) => {
        println!("Generated {} signal with strength {:.4}", 
            signal.side, signal.strength);
    }
    Ok(None) => println!("No signal generated"),
    Err(e) => eprintln!("Error: {}", e),
}
```

### Handling Validation Errors

```rust
let components = SignalComponents {
    s_ldc: 2.0, // Invalid: out of range
    s_mr: 0.6,
    s_tsmom: 0.4,
};

// Pre-validate if needed
if let Err(e) = components.validate() {
    eprintln!("Invalid components: {}", e);
    return;
}

// Or handle during fusion
match fusion.fuse_signals(components, weights, timestamp, "BTCUSDT", "v1.0") {
    Err(e) => {
        // Log and handle validation error
        eprintln!("Fusion failed: {}", e);
    }
    _ => {}
}
```

### Weight Normalization Control

```rust
// With normalization (default)
let mut fusion_normalized = SignalFusion::new(0.3, 60);

// Without normalization
let mut fusion_raw = SignalFusion::with_normalization(0.3, 60, false);

// Manual normalization
let weights = FusionWeights { w_ldc: 0.6, w_mr: 0.3, w_tsmom: 0.1 };
let normalized = weights.normalize();
println!("Normalized: LDC={:.3}, MR={:.3}, TSMOM={:.3}",
    normalized.w_ldc, normalized.w_mr, normalized.w_tsmom);
```

### Cooldown Management

```rust
let mut fusion = SignalFusion::new(0.3, 60);

// Generate first signal
let signal1 = fusion.fuse_signals(components, weights, 1000, "BTCUSDT", "v1.0")?;

// Check cooldown status
if fusion.is_in_cooldown(1030) {
    println!("In cooldown, signal will be suppressed");
}

// Attempt signal during cooldown (returns None)
let signal2 = fusion.fuse_signals(components, weights, 1030, "BTCUSDT", "v1.0")?;
assert!(signal2.is_none());

// Signal after cooldown period
let signal3 = fusion.fuse_signals(components, weights, 1070, "BTCUSDT", "v1.0")?;
assert!(signal3.is_some());
```

## Performance Characteristics

The signal fusion engine is designed to meet strict performance requirements:

- **Fusion computation**: < 5ms (Requirement 5.4)
- **Validation overhead**: Minimal (< 1ms)
- **Normalization**: O(1) constant time
- **Memory usage**: Minimal per-instance state

## Error Handling

The engine provides detailed error messages for debugging:

```rust
match fusion.fuse_signals(components, weights, timestamp, symbol, version) {
    Ok(Some(signal)) => {
        // Process signal
    }
    Ok(None) => {
        // No signal generated (below threshold or in cooldown)
    }
    Err(e) => {
        // Validation or computation error
        eprintln!("Fusion error: {}", e);
    }
}
```

**Common Error Scenarios:**
- Signal component out of range: `s_ldc out of range [-1, 1]: 2.0`
- Weight out of range: `w_ldc out of range [-1, 1]: 1.5`
- Non-finite values: `s_mr out of range [-1, 1]: NaN`

## Testing

The implementation includes comprehensive tests covering:

- ✅ Signal validation (valid, out of range, NaN)
- ✅ Weight validation (valid, out of range)
- ✅ Weight normalization (positive, negative, zero-sum)
- ✅ Threshold logic (above, below)
- ✅ Cooldown logic (active, expired)
- ✅ Signal generation (BUY, SELL)
- ✅ Error handling (invalid inputs)

Run tests with:
```bash
cargo test --lib --manifest-path rust/signal-fusion/Cargo.toml
```

## Integration with HMM Service

The signal fusion engine integrates with the HMM service to obtain regime-aware weights:

```rust
use signal_fusion::{HmmIntegration, SignalComponents, SignalFusion};

// Create HMM integration
let mut hmm = HmmIntegration::new()?;

// Get regime-aware weights
let components = SignalComponents { s_ldc: 0.8, s_mr: 0.6, s_tsmom: 0.4 };
let weights = hmm.get_fusion_weights_for_signals(&components).await?;

// Fuse signals with regime-aware weights
let mut fusion = SignalFusion::new(0.3, 60);
let signal = fusion.fuse_signals(components, weights, timestamp, "BTCUSDT", "v1.0")?;
```

## Best Practices

1. **Always enable logging** in production for audit trails
2. **Use weight normalization** unless you have specific requirements
3. **Set appropriate thresholds** based on your risk tolerance
4. **Configure cooldown periods** to prevent over-trading
5. **Handle validation errors** gracefully with fallback logic
6. **Monitor signal generation rates** to detect anomalies
7. **Test with edge cases** (zero weights, extreme signals, etc.)

## Configuration Recommendations

### Conservative Trading
```rust
let fusion = SignalFusion::new(0.5, 300); // High threshold, long cooldown
```

### Aggressive Trading
```rust
let fusion = SignalFusion::new(0.2, 30); // Low threshold, short cooldown
```

### Balanced Trading
```rust
let fusion = SignalFusion::new(0.3, 60); // Medium threshold and cooldown
```

## Troubleshooting

### No Signals Generated

**Possible causes:**
1. Signals below threshold → Lower threshold or check signal strength
2. Cooldown active → Wait for cooldown period or reduce cooldown
3. Validation failures → Check logs for validation errors

### Unexpected Signal Strength

**Possible causes:**
1. Weight normalization enabled → Disable if you want raw weights
2. Invalid weight configuration → Verify weight values
3. Signal component issues → Validate input signals

### Performance Issues

**Possible causes:**
1. Excessive logging → Reduce log level in production
2. Frequent validation errors → Fix upstream signal generation
3. High signal generation rate → Increase cooldown period

## See Also

- [HMM Integration Guide](README_HMM_INTEGRATION.md)
- [Circuit Breaker Guide](CIRCUIT_BREAKER_GUIDE.md)
- [Configuration Guide](CONFIG.md)
- [Examples](examples/)
