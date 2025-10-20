# HMM Integration Configuration Guide

This guide explains how to configure the HMM integration component for the signal fusion system.

## Configuration Methods

The HMM integration supports three configuration methods with the following priority:

1. **TOML Configuration File** (highest priority)
2. **Environment Variables** (medium priority)
3. **Default Values** (lowest priority)

When using a configuration file, environment variables can still override specific settings.

## Configuration Structure

The configuration is organized into five main sections:

### 1. Service Configuration

Controls connection to the HMM microservice:

```toml
[service]
url = "http://localhost:8000"      # HMM service base URL
timeout_ms = 5000                   # Request timeout in milliseconds
retry_attempts = 3                  # Number of retry attempts
retry_delay_ms = 100                # Delay between retries in milliseconds
```

**Environment Variables:**
- `HMM_SERVICE_URL`
- `HMM_SERVICE_TIMEOUT_MS`
- `HMM_SERVICE_RETRY_ATTEMPTS`
- `HMM_SERVICE_RETRY_DELAY_MS`

### 2. Circuit Breaker Configuration

Controls failure detection and recovery:

```toml
[circuit_breaker]
threshold = 5                       # Failures before opening circuit
timeout_sec = 30                    # Seconds before recovery attempt
```

**Environment Variables:**
- `HMM_CIRCUIT_BREAKER_THRESHOLD`
- `HMM_CIRCUIT_BREAKER_TIMEOUT_SEC`

### 3. Cache Configuration

Controls weight caching behavior:

```toml
[cache]
ttl_sec = 60                        # Cache entry time-to-live in seconds
max_size = 1000                     # Maximum number of cache entries
```

**Environment Variables:**
- `HMM_CACHE_TTL_SEC`
- `HMM_CACHE_MAX_SIZE`

### 4. Fallback Configuration

Controls fallback behavior when service is unavailable:

```toml
[fallback]
enabled = true                      # Enable fallback to static weights
w_ldc = 0.33                        # Fallback weight for LDC signal
w_mr = 0.33                         # Fallback weight for MR signal
w_tsmom = 0.34                      # Fallback weight for TSMOM signal
```

**Environment Variables:**
- `HMM_ENABLE_FALLBACK`
- `HMM_FALLBACK_W_LDC`
- `HMM_FALLBACK_W_MR`
- `HMM_FALLBACK_W_TSMOM`

### 5. Signal Fusion Configuration

Controls signal generation behavior:

```toml
[signal_fusion]
threshold = 0.3                     # Minimum signal strength threshold
cooldown_sec = 60                   # Cooldown period between signals
```

**Environment Variables:**
- `SIGNAL_FUSION_THRESHOLD`
- `SIGNAL_FUSION_COOLDOWN_SEC`

## Usage Examples

### Using Default Configuration

```rust
use signal_fusion::config::HmmIntegrationConfig;

let config = HmmIntegrationConfig::default();
let client_config = config.to_client_config()?;
let client = HmmClient::with_config(client_config)?;
```

### Loading from TOML File

```rust
use signal_fusion::config::HmmIntegrationConfig;

let config = HmmIntegrationConfig::from_file("hmm_integration.toml")?;
let client_config = config.to_client_config()?;
let client = HmmClient::with_config(client_config)?;
```

### Loading from Environment Variables

```rust
use signal_fusion::config::HmmIntegrationConfig;

// Set environment variables first
std::env::set_var("HMM_SERVICE_URL", "http://production:8000");
std::env::set_var("HMM_CIRCUIT_BREAKER_THRESHOLD", "3");

let config = HmmIntegrationConfig::from_env()?;
let client_config = config.to_client_config()?;
let client = HmmClient::with_config(client_config)?;
```

### Combining File and Environment Variables

```rust
use signal_fusion::config::HmmIntegrationConfig;
use std::path::Path;

// Load from file, then override with environment variables
let config = HmmIntegrationConfig::load(Some(Path::new("hmm_integration.toml")))?;
let client_config = config.to_client_config()?;
let client = HmmClient::with_config(client_config)?;
```

## Configuration Validation

All configurations are automatically validated when loaded. The following rules apply:

### Service Configuration
- URL must be a valid HTTP/HTTPS URL
- Timeout must be between 1ms and 60,000ms (60 seconds)
- Retry attempts must not exceed 10

### Circuit Breaker
- Threshold must be greater than 0
- Timeout must be greater than 0

### Cache
- TTL must be greater than 0
- Max size must be greater than 0

### Fallback Weights
- All weights must be in range [-1.0, 1.0]
- Weights should typically sum to approximately 1.0

### Signal Fusion
- Threshold must be in range [0.0, 1.0]

## Example Configuration Files

### Development Configuration

```toml
[service]
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

### Production Configuration

```toml
[service]
url = "http://hmm-service:8000"
timeout_ms = 3000
retry_attempts = 5
retry_delay_ms = 200

[circuit_breaker]
threshold = 3
timeout_sec = 60

[cache]
ttl_sec = 120
max_size = 5000

[fallback]
enabled = true
w_ldc = 0.35
w_mr = 0.30
w_tsmom = 0.35

[signal_fusion]
threshold = 0.4
cooldown_sec = 30
```

## Best Practices

### Development
- Use longer timeouts (5000ms) for easier debugging
- Higher circuit breaker threshold (5) to avoid premature failures
- Enable fallback for graceful degradation
- Lower signal threshold (0.3) for more signals during testing

### Production
- Use shorter timeouts (3000ms) for faster failure detection
- Lower circuit breaker threshold (3) for quicker failover
- Always enable fallback for reliability
- Higher signal threshold (0.4) to reduce false signals
- Larger cache size (5000) for better performance
- Longer cache TTL (120s) to reduce service load

### Performance Tuning
- **High Latency**: Increase timeout_ms and retry_delay_ms
- **High Load**: Increase cache max_size and ttl_sec
- **Frequent Failures**: Decrease circuit_breaker threshold
- **Too Many Signals**: Increase signal_fusion threshold
- **Missed Signals**: Decrease signal_fusion threshold and cooldown_sec

## Troubleshooting

### Configuration Not Loading
- Check file path is correct
- Verify TOML syntax is valid
- Check file permissions
- Review validation error messages

### Environment Variables Not Working
- Verify variable names match exactly (case-sensitive)
- Check variable values are valid (e.g., numbers for numeric fields)
- Ensure variables are set before loading configuration

### Validation Errors
- Review error message for specific field
- Check value ranges in validation rules
- Verify URL format is correct
- Ensure weights are in valid range

## Running the Configuration Demo

To see all configuration methods in action:

```bash
cd rust/signal-fusion
cargo run --example configuration_demo
```

This will demonstrate:
- Default configuration
- Environment variable configuration
- TOML file configuration
- Configuration validation
- Saving configuration to file
