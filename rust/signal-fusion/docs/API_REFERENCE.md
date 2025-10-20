# HMM Integration API Reference

Complete API reference for the Rust HMM Integration component.

## Table of Contents

- [Core Types](#core-types)
- [HmmClient](#hmmclient)
- [HmmIntegration](#hmmintegration)
- [Configuration](#configuration)
- [Data Models](#data-models)
- [Error Types](#error-types)
- [Metrics](#metrics)

## Core Types

### SignalComponents

Represents the three trading signal components used as HMM observations.

```rust
pub struct SignalComponents {
    pub s_ldc: f32,    // Lorentzian Distance Classifier signal [-1.0, 1.0]
    pub s_mr: f32,     // Mean Reversion signal [-1.0, 1.0]
    pub s_tsmom: f32,  // Time Series Momentum signal [-1.0, 1.0]
}
```

**Validation Rules:**
- All signals must be in range `[-1.0, 1.0]`
- NaN and infinite values are rejected

**Example:**
```rust
let signals = SignalComponents {
    s_ldc: 0.015,
    s_mr: -0.008,
    s_tsmom: 0.022,
};
```

### FusionWeights

Represents the regime-aware weights for signal fusion.

```rust
pub struct FusionWeights {
    pub w_ldc: f32,    // Weight for LDC signal [-1.0, 1.0]
    pub w_mr: f32,     // Weight for MR signal [-1.0, 1.0]
    pub w_tsmom: f32,  // Weight for TSMOM signal [-1.0, 1.0]
}
```

**Validation Rules:**
- All weights must be in range `[-1.0, 1.0]`
- Weights typically sum to approximately 1.0 (not enforced)

**Example:**
```rust
let weights = FusionWeights {
    w_ldc: 0.45,
    w_mr: 0.30,
    w_tsmom: 0.25,
};
```

### TradingSignal

Complete trading signal with metadata.

```rust
pub struct TradingSignal {
    pub timestamp: i64,              // Unix timestamp
    pub symbol: String,              // Trading symbol (e.g., "BTCUSDT")
    pub side: String,                // "BUY" or "SELL"
    pub strength: f32,               // Signal strength [0.0, 3.0]
    pub confidence: f32,             // Confidence level [0.0, 1.0]
    pub components: SignalComponents, // Original signal components
    pub weights: FusionWeights,      // Applied fusion weights
    pub model_version: String,       // HMM model version
}
```

## HmmClient

Low-level HTTP client for direct communication with the HMM microservice.

### Constructor Methods

#### `new() -> Result<Self, HmmClientError>`

Creates a client with default configuration.

```rust
let client = HmmClient::new()?;
```

**Default Configuration:**
- URL: `http://localhost:8000`
- Timeout: 5000ms
- Retry attempts: 3
- Circuit breaker threshold: 5
- Fallback enabled: true

#### `with_config(config: HmmClientConfig) -> Result<Self, HmmClientError>`

Creates a client with custom configuration.

```rust
let config = HmmClientConfig {
    base_url: "http://production:8000".parse()?,
    timeout: Duration::from_millis(3000),
    retry_attempts: 5,
    ..Default::default()
};
let client = HmmClient::with_config(config)?;
```

### Inference Methods

#### `get_state_probabilities(observations: [f32; 3], request_id: Option<String>) -> Result<StateProbabilitiesResponse>`

Gets HMM state probabilities for given observations.

**Parameters:**
- `observations`: Array of three signal values
- `request_id`: Optional request ID for tracing

**Returns:** `StateProbabilitiesResponse` containing:
- `state_probabilities`: Vec of probabilities for each state
- `most_likely_state`: Index of most likely state
- `timestamp`: Server timestamp
- `processing_time_ms`: Server processing time

**Example:**
```rust
let observations = [0.015, -0.008, 0.022];
let response = client.get_state_probabilities(observations, None).await?;
println!("Most likely state: {}", response.most_likely_state);
println!("Probabilities: {:?}", response.state_probabilities);
```

**Errors:**
- `Network`: Connection or HTTP errors
- `Timeout`: Request exceeded timeout
- `ServiceUnavailable`: Service returned 503
- `ValidationError`: Invalid observations

#### `get_fusion_weights(observations: [f32; 3], request_id: Option<String>) -> Result<FusionWeightsResponse>`

Gets fusion weights for signal combination.

**Parameters:**
- `observations`: Array of three signal values
- `request_id`: Optional request ID for tracing

**Returns:** `FusionWeightsResponse` containing:
- `weights`: FusionWeights struct
- `state_probabilities`: Vec of state probabilities
- `most_likely_state`: Index of most likely state
- `timestamp`: Server timestamp
- `processing_time_ms`: Server processing time

**Example:**
```rust
let observations = [0.015, -0.008, 0.022];
let response = client.get_fusion_weights(observations, None).await?;
println!("Weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}",
         response.weights.w_ldc,
         response.weights.w_mr,
         response.weights.w_tsmom);
```

#### `predict(observations: [f32; 3], request_id: Option<String>) -> Result<PredictionResponse>`

Gets complete prediction with all information.

**Parameters:**
- `observations`: Array of three signal values
- `request_id`: Optional request ID for tracing

**Returns:** `PredictionResponse` containing:
- `state_probabilities`: Vec of state probabilities
- `most_likely_state`: Index of most likely state
- `confidence`: Prediction confidence [0.0, 1.0]
- `fusion_weights`: FusionWeights struct
- `timestamp`: Server timestamp
- `processing_time_ms`: Server processing time
- `model_version`: HMM model version string
- `request_id`: Echo of request ID

**Example:**
```rust
let observations = [0.015, -0.008, 0.022];
let response = client.predict(observations, Some("req-123".to_string())).await?;
println!("Model version: {}", response.model_version);
println!("Confidence: {:.2}", response.confidence);
```

### Health Check Methods

#### `health_check() -> Result<HealthResponse>`

Checks if the service is alive.

**Returns:** `HealthResponse` containing:
- `status`: "healthy" or error message
- `timestamp`: Server timestamp

**Example:**
```rust
let health = client.health_check().await?;
println!("Service status: {}", health.status);
```

#### `readiness_check() -> Result<ReadinessResponse>`

Checks if the service is ready to handle requests.

**Returns:** `ReadinessResponse` containing:
- `ready`: Boolean indicating readiness
- `model_loaded`: Boolean indicating if model is loaded
- `model_version`: Optional model version string
- `timestamp`: Server timestamp

**Example:**
```rust
let readiness = client.readiness_check().await?;
if readiness.ready && readiness.model_loaded {
    println!("Service ready with model: {:?}", readiness.model_version);
} else {
    println!("Service not ready");
}
```

### Model Management Methods

#### `get_model_info() -> Result<ModelInfoResponse>`

Gets information about the currently loaded model.

**Returns:** `ModelInfoResponse` containing:
- `model_loaded`: Boolean indicating if model is loaded
- `model_version`: Optional model version string
- `n_states`: Optional number of HMM states
- `timestamp`: Server timestamp

**Example:**
```rust
let info = client.get_model_info().await?;
if let Some(version) = info.model_version {
    println!("Model version: {}", version);
    println!("States: {:?}", info.n_states);
}
```

#### `reload_model() -> Result<ReloadResponse>`

Triggers a model reload from storage.

**Returns:** `ReloadResponse` containing:
- `success`: Boolean indicating reload success
- `message`: Status message
- `model_version`: Optional new model version
- `timestamp`: Server timestamp

**Example:**
```rust
let response = client.reload_model().await?;
if response.success {
    println!("Model reloaded: {:?}", response.model_version);
} else {
    println!("Reload failed: {}", response.message);
}
```

### Status Methods

#### `get_circuit_breaker_status() -> (String, usize)`

Gets current circuit breaker state and failure count.

**Returns:** Tuple of:
- State string: "closed", "open", or "half_open"
- Failure count: Number of consecutive failures

**Example:**
```rust
let (state, failures) = client.get_circuit_breaker_status();
println!("Circuit breaker: {} ({} failures)", state, failures);
```

#### `get_circuit_breaker_metrics() -> CircuitBreakerMetrics`

Gets detailed circuit breaker metrics.

**Returns:** `CircuitBreakerMetrics` struct with:
- `total_requests`: Total requests attempted
- `successful_requests`: Successful requests
- `failed_requests`: Failed requests
- `circuit_breaker_opens`: Times circuit opened
- `circuit_breaker_closes`: Times circuit closed
- `half_open_attempts`: Recovery attempts
- `rejected_requests`: Requests rejected while open

**Example:**
```rust
let metrics = client.get_circuit_breaker_metrics();
let success_rate = (metrics.successful_requests as f64 
    / metrics.total_requests as f64) * 100.0;
println!("Success rate: {:.2}%", success_rate);
```

## HmmIntegration

High-level integration helper that simplifies common workflows.

### Constructor Methods

#### `new() -> Result<Self, HmmClientError>`

Creates integration with default configuration.

```rust
let integration = HmmIntegration::new()?;
```

#### `with_config(config: HmmClientConfig) -> Result<Self, HmmClientError>`

Creates integration with custom configuration.

```rust
let config = HmmClientConfig {
    base_url: "http://production:8000".parse()?,
    ..Default::default()
};
let integration = HmmIntegration::with_config(config)?;
```

#### `with_config_and_cache(config: HmmClientConfig, cache_ttl: Duration, cache_max_size: usize) -> Result<Self>`

Creates integration with custom configuration and cache settings.

```rust
let integration = HmmIntegration::with_config_and_cache(
    config,
    Duration::from_secs(120),  // 2 minute TTL
    5000,                       // 5000 entry cache
)?;
```

### Core Methods

#### `get_fusion_weights_for_signals(signals: &SignalComponents) -> Result<FusionWeights>`

Gets fusion weights for signal components with automatic caching and fallback.

**Parameters:**
- `signals`: Reference to SignalComponents

**Returns:** `FusionWeights` struct

**Behavior:**
1. Checks cache for existing weights
2. If cache miss, calls HMM service
3. On service failure, uses fallback weights
4. Caches successful responses

**Example:**
```rust
let signals = SignalComponents {
    s_ldc: 0.015,
    s_mr: -0.008,
    s_tsmom: 0.022,
};
let weights = integration.get_fusion_weights_for_signals(&signals).await?;
```

#### `is_service_ready() -> bool`

Checks if HMM service is ready for requests.

**Returns:** Boolean indicating readiness

**Example:**
```rust
if integration.is_service_ready().await {
    // Process signals
} else {
    // Use fallback or wait
}
```

#### `get_service_status() -> Result<(bool, Option<String>)>`

Gets service status and model version.

**Returns:** Tuple of:
- Boolean indicating if model is loaded
- Optional model version string

**Example:**
```rust
let (loaded, version) = integration.get_service_status().await?;
println!("Model loaded: {}, version: {:?}", loaded, version);
```

### Metrics Methods

#### `get_metrics() -> HmmIntegrationMetrics`

Gets comprehensive metrics for all components.

**Returns:** `HmmIntegrationMetrics` struct containing:
- `requests`: Request metrics
- `cache`: Cache metrics
- `circuit_breaker`: Circuit breaker metrics
- `fallback`: Fallback metrics
- `timestamp`: Current timestamp
- `uptime_seconds`: Integration uptime

**Example:**
```rust
let metrics = integration.get_metrics();
println!("Total requests: {}", metrics.requests.total_requests);
println!("Cache hit rate: {:.1}%", metrics.cache.hit_rate * 100.0);
```

#### `export_metrics(format: MetricsFormat) -> Result<String>`

Exports metrics in specified format.

**Parameters:**
- `format`: `MetricsFormat::Json` or `MetricsFormat::Prometheus`

**Returns:** Formatted metrics string

**Example:**
```rust
// JSON format
let json = integration.export_metrics(MetricsFormat::Json)?;
println!("{}", json);

// Prometheus format
let prometheus = integration.export_metrics(MetricsFormat::Prometheus)?;
println!("{}", prometheus);
```

#### `get_cache_stats() -> CacheStats`

Gets cache statistics.

**Returns:** `CacheStats` struct with:
- `hits`: Cache hits
- `misses`: Cache misses
- `size`: Current cache size
- `evictions`: Total evictions
- `hit_rate`: Hit rate [0.0, 1.0]

**Example:**
```rust
let stats = integration.get_cache_stats();
println!("Cache: {} hits, {} misses ({:.1}% hit rate)",
         stats.hits, stats.misses, stats.hit_rate * 100.0);
```

## Configuration

### HmmClientConfig

Complete configuration for HMM client.

```rust
pub struct HmmClientConfig {
    pub base_url: Url,                      // HMM service URL
    pub timeout: Duration,                  // Request timeout
    pub retry_attempts: usize,              // Number of retries
    pub retry_delay: Duration,              // Delay between retries
    pub enable_fallback: bool,              // Enable fallback weights
    pub fallback_weights: FusionWeights,    // Fallback weight values
    pub circuit_breaker_threshold: usize,   // Failures before opening
    pub circuit_breaker_timeout: Duration,  // Recovery timeout
}
```

**Default Values:**
```rust
HmmClientConfig {
    base_url: "http://localhost:8000".parse().unwrap(),
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
}
```

### HmmIntegrationConfig

Configuration loaded from files or environment.

```rust
pub struct HmmIntegrationConfig {
    pub service: ServiceConfig,
    pub circuit_breaker: CircuitBreakerConfig,
    pub cache: CacheConfig,
    pub fallback: FallbackConfig,
    pub signal_fusion: SignalFusionConfig,
}
```

**Loading Methods:**
```rust
// From file
let config = HmmIntegrationConfig::from_file("config.toml")?;

// From environment
let config = HmmIntegrationConfig::from_env()?;

// Auto-detect (file then env)
let config = HmmIntegrationConfig::load(Some(Path::new("config.toml")))?;
```

## Data Models

### Request Models

#### InferenceRequest

```rust
pub struct InferenceRequest {
    pub observations: [f32; 3],
    pub timestamp: Option<i64>,
    pub request_id: Option<String>,
}
```

### Response Models

#### StateProbabilitiesResponse

```rust
pub struct StateProbabilitiesResponse {
    pub state_probabilities: Vec<f32>,
    pub most_likely_state: usize,
    pub timestamp: i64,
    pub processing_time_ms: f32,
}
```

#### FusionWeightsResponse

```rust
pub struct FusionWeightsResponse {
    pub weights: FusionWeights,
    pub state_probabilities: Vec<f32>,
    pub most_likely_state: usize,
    pub timestamp: i64,
    pub processing_time_ms: f32,
}
```

#### PredictionResponse

```rust
pub struct PredictionResponse {
    pub state_probabilities: Vec<f32>,
    pub most_likely_state: usize,
    pub confidence: f32,
    pub fusion_weights: FusionWeights,
    pub timestamp: i64,
    pub processing_time_ms: f32,
    pub model_version: String,
    pub request_id: Option<String>,
}
```

#### HealthResponse

```rust
pub struct HealthResponse {
    pub status: String,
    pub timestamp: i64,
}
```

#### ReadinessResponse

```rust
pub struct ReadinessResponse {
    pub ready: bool,
    pub model_loaded: bool,
    pub model_version: Option<String>,
    pub timestamp: i64,
}
```

## Error Types

### HmmClientError

```rust
pub enum HmmClientError {
    Network(reqwest::Error),
    ServiceUnavailable { status: StatusCode },
    InvalidRequest { message: String },
    ModelError { message: String },
    Timeout { duration: Duration },
    ValidationError { field: String, message: String },
    FallbackActivated { reason: String },
}
```

**Error Handling:**
```rust
match client.get_fusion_weights(observations, None).await {
    Ok(response) => {
        // Use weights
    }
    Err(HmmClientError::Network(e)) => {
        // Handle network error
    }
    Err(HmmClientError::Timeout { duration }) => {
        // Handle timeout
    }
    Err(HmmClientError::FallbackActivated { reason }) => {
        // Using fallback weights
    }
    Err(e) => {
        // Handle other errors
    }
}
```

## Metrics

### HmmIntegrationMetrics

Complete metrics structure.

```rust
pub struct HmmIntegrationMetrics {
    pub requests: RequestMetrics,
    pub cache: CacheMetrics,
    pub circuit_breaker: CircuitBreakerMetrics,
    pub fallback: FallbackMetrics,
    pub timestamp: i64,
    pub uptime_seconds: u64,
}
```

### RequestMetrics

```rust
pub struct RequestMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_duration_ms: u64,
    pub avg_duration_ms: f64,
    pub min_duration_ms: u64,
    pub max_duration_ms: u64,
    pub timeout_errors: u64,
    pub network_errors: u64,
    pub validation_errors: u64,
}
```

### CacheMetrics

```rust
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub evictions: u64,
    pub hit_rate: f64,
}
```

### CircuitBreakerMetrics

```rust
pub struct CircuitBreakerMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub circuit_breaker_opens: u64,
    pub circuit_breaker_closes: u64,
    pub half_open_attempts: u64,
    pub rejected_requests: u64,
}
```

### FallbackMetrics

```rust
pub struct FallbackMetrics {
    pub total_activations: u64,
    pub circuit_breaker_activations: u64,
    pub network_error_activations: u64,
    pub timeout_activations: u64,
    pub service_error_activations: u64,
    pub currently_active: bool,
}
```

## Requirements Coverage

This API implementation satisfies:

- **Requirement 1.1**: HTTP client for HMM service communication
- **Requirement 1.2**: Request parsing within 5ms
- **Requirement 3.3**: Configurable fallback weights
- **Requirement 3.4**: Service recovery attempts
- **Requirement 6.4**: Structured error logs with context

## See Also

- [Configuration Guide](../CONFIG.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Performance Tuning](./PERFORMANCE_TUNING.md)
- [Circuit Breaker](./CIRCUIT_BREAKER.md)
- [Monitoring and Metrics](./MONITORING_METRICS.md)
