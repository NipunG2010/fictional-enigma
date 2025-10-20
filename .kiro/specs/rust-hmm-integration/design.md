# Design Document

## Overview

The Rust HMM Integration provides HTTP client functionality for communicating with the HMM microservice, enabling regime-aware signal fusion in the trading system. The design focuses on low latency (<100ms end-to-end), reliability through fallback mechanisms, and graceful degradation during service failures.

The integration consists of three main components: an HTTP client with circuit breaker protection, a weight caching layer for performance optimization, and a signal fusion engine that combines trading signals using regime-aware weights.

## Architecture

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Rust Inference Engine                          │
├─────────────────────────────────────────────────────────────┤
│  Signal Generation Layer                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
│  │   LDC   │  │   MR    │  │  TSMOM  │                    │
│  │ Engine  │  │ Engine  │  │ Engine  │                    │
│  └─────────┘  └─────────┘  └─────────┘                    │
│       │            │            │                           │
│       └────────────┴────────────┘                           │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         HMM Integration Layer                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │   │
│  │  │ Weight Cache │  │ HMM Client   │  │ Circuit  │ │   │
│  │  │              │  │              │  │ Breaker  │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Signal Fusion Engine                        │   │
│  │  - Weighted signal combination                      │   │
│  │  - Threshold application                            │   │
│  │  - Signal validation                                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ HTTP/JSON
                         ▼
              ┌──────────────────┐
              │ HMM Microservice │
              │    (FastAPI)     │
              └──────────────────┘
```

### Data Flow

```
Signal Components [s_LDC, s_MR, s_TSMOM]
           │
           ▼
    ┌─────────────┐
    │ Check Cache │
    └─────────────┘
           │
      ┌────┴────┐
      │         │
   Cache      Cache
    Hit       Miss
      │         │
      │         ▼
      │  ┌──────────────┐
      │  │ Circuit      │
      │  │ Breaker      │
      │  │ Check        │
      │  └──────────────┘
      │         │
      │    ┌────┴────┐
      │    │         │
      │  Open     Closed
      │    │         │
      │    │         ▼
      │    │  ┌──────────────┐
      │    │  │ HTTP Request │
      │    │  │ to HMM       │
      │    │  │ Service      │
      │    │  └──────────────┘
      │    │         │
      │    │    ┌────┴────┐
      │    │    │         │
      │    │ Success   Failure
      │    │    │         │
      │    │    │         ▼
      │    │    │  ┌──────────────┐
      │    │    │  │ Update       │
      │    │    │  │ Circuit      │
      │    │    │  │ Breaker      │
      │    │    │  └──────────────┘
      │    │    │         │
      │    ▼    ▼         ▼
      │  ┌──────────────────┐
      │  │ Fallback Weights │
      │  └──────────────────┘
      │         │
      └─────────┴─────────┐
                          │
                          ▼
                   ┌─────────────┐
                   │ Fusion      │
                   │ Weights     │
                   └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Signal      │
                   │ Fusion      │
                   └─────────────┘
                          │
                          ▼
                   Trading Signal
```

## Components and Interfaces

### 1. HMM Client (`hmm_client.rs`)

**Responsibilities:**
- HTTP communication with HMM microservice
- Request serialization and response parsing
- Retry logic with exponential backoff
- Error handling and classification

**Key Structures:**

```rust
pub struct HmmClient {
    client: Client,
    config: HmmClientConfig,
    circuit_breaker_state: CircuitBreakerState,
    failure_count: usize,
}

pub struct HmmClientConfig {
    pub base_url: Url,
    pub timeout: Duration,
    pub retry_attempts: usize,
    pub retry_delay: Duration,
    pub enable_fallback: bool,
    pub fallback_weights: FusionWeights,
    pub circuit_breaker_threshold: usize,
    pub circuit_breaker_timeout: Duration,
}
```

**Key Methods:**

```rust
impl HmmClient {
    // Create client with configuration
    pub fn with_config(config: HmmClientConfig) -> Result<Self>;
    
    // Get fusion weights from service
    pub async fn get_fusion_weights(
        &mut self,
        observations: [f32; 3],
        request_id: Option<String>,
    ) -> Result<FusionWeightsResponse, HmmClientError>;
    
    // Get state probabilities
    pub async fn get_state_probabilities(
        &mut self,
        observations: [f32; 3],
        request_id: Option<String>,
    ) -> Result<StateProbabilitiesResponse, HmmClientError>;
    
    // Complete prediction with all data
    pub async fn predict(
        &mut self,
        observations: [f32; 3],
        request_id: Option<String>,
    ) -> Result<PredictionResponse, HmmClientError>;
    
    // Health and readiness checks
    pub async fn health_check(&mut self) -> Result<HealthResponse, HmmClientError>;
    pub async fn readiness_check(&mut self) -> Result<ReadinessResponse, HmmClientError>;
}
```

### 2. Circuit Breaker

**Responsibilities:**
- Track service failure rates
- Prevent cascading failures
- Automatic recovery attempts
- State management (Closed, Open, Half-Open)

**State Machine:**

```
     Closed ──────────────────────────────────┐
        │                                     │
        │ Failures >= Threshold               │
        ▼                                     │
      Open ────────────────────────────────┐  │
        │                                  │  │
        │ Timeout Elapsed                  │  │
        ▼                                  │  │
    Half-Open                              │  │
        │                                  │  │
        ├─ Success ─────────────────────────┘  │
        │                                      │
        └─ Failure ───────────────────────────┘
```

**Implementation:**

```rust
enum CircuitBreakerState {
    Closed,
    Open { opened_at: Instant },
    HalfOpen,
}

impl HmmClient {
    fn is_circuit_breaker_open(&mut self) -> bool;
    fn record_success(&mut self);
    fn record_failure(&mut self);
}
```

**Configuration:**
- Failure threshold: 5 consecutive failures
- Open timeout: 30 seconds
- Half-open test: Single request

### 3. Weight Cache

**Responsibilities:**
- Cache fusion weights by observation vector
- TTL-based expiration (60 seconds)
- Thread-safe concurrent access
- Cache hit/miss metrics

**Design:**

```rust
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

pub struct WeightCache {
    cache: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    ttl: Duration,
    max_size: usize,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

struct CacheKey {
    observations: [OrderedFloat<f32>; 3],
}

struct CacheEntry {
    weights: FusionWeights,
    timestamp: Instant,
}

impl WeightCache {
    pub fn new(ttl: Duration, max_size: usize) -> Self;
    pub fn get(&self, observations: &[f32; 3]) -> Option<FusionWeights>;
    pub fn insert(&self, observations: [f32; 3], weights: FusionWeights);
    pub fn evict_expired(&self);
    pub fn get_stats(&self) -> CacheStats;
}
```

**Cache Key Strategy:**
- Round observations to 3 decimal places for key generation
- Use ordered floats for HashMap compatibility
- Hash based on rounded values

**Eviction Policy:**
- TTL-based: Remove entries older than 60 seconds
- Size-based: LRU eviction when max_size exceeded
- Periodic cleanup: Every 10 seconds

### 4. Signal Fusion Engine

**Responsibilities:**
- Combine signals using regime-aware weights
- Apply signal thresholds
- Validate input and output ranges
- Generate trading signals with metadata

**Implementation:**

```rust
pub struct SignalFusion {
    threshold: f32,
    cooldown_period: u64,
    last_signal_time: Option<i64>,
}

impl SignalFusion {
    pub fn fuse_signals(
        &mut self,
        components: SignalComponents,
        weights: FusionWeights,
        timestamp: i64,
        symbol: &str,
        model_version: &str,
    ) -> Result<Option<TradingSignal>>;
}
```

**Fusion Algorithm:**

```rust
// Weighted combination
fused_signal = s_ldc * w_ldc + s_mr * w_mr + s_tsmom * w_tsmom

// Threshold application
if |fused_signal| < threshold {
    return None  // No signal
}

// Side determination
side = if fused_signal > 0 { "BUY" } else { "SELL" }

// Confidence calculation
confidence = min(|fused_signal|, 1.0)
```

**Signal Validation:**
- Input signals: [-1.0, 1.0]
- Weights: [-1.0, 1.0]
- Fused signal: [-3.0, 3.0] (before normalization)
- Confidence: [0.0, 1.0]

### 5. HMM Integration Helper

**Responsibilities:**
- High-level API for signal fusion workflow
- Automatic error handling and fallback
- Request ID generation
- Service status monitoring

**Implementation:**

```rust
pub struct HmmIntegration {
    client: HmmClient,
    cache: WeightCache,
    request_counter: u64,
}

impl HmmIntegration {
    pub fn new() -> Result<Self>;
    pub fn with_config(config: HmmClientConfig) -> Result<Self>;
    
    pub async fn get_fusion_weights_for_signals(
        &mut self,
        signal_components: &SignalComponents,
    ) -> Result<FusionWeights>;
    
    pub async fn is_service_ready(&mut self) -> bool;
    pub async fn get_service_status(&mut self) -> Result<(bool, Option<String>)>;
    pub fn get_circuit_breaker_status(&self) -> (String, usize);
}
```

## Data Models

### Request Models

```rust
#[derive(Serialize)]
pub struct InferenceRequest {
    pub observations: [f32; 3],
    pub timestamp: Option<i64>,
    pub request_id: Option<String>,
}
```

### Response Models

```rust
#[derive(Deserialize)]
pub struct FusionWeightsResponse {
    pub weights: FusionWeights,
    pub state_probabilities: Vec<f32>,
    pub most_likely_state: usize,
    pub timestamp: i64,
    pub processing_time_ms: f32,
}

#[derive(Deserialize)]
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

### Signal Models

```rust
#[derive(Serialize, Deserialize)]
pub struct SignalComponents {
    pub s_ldc: f32,
    pub s_mr: f32,
    pub s_tsmom: f32,
}

#[derive(Serialize, Deserialize)]
pub struct FusionWeights {
    pub w_ldc: f32,
    pub w_mr: f32,
    pub w_tsmom: f32,
}

#[derive(Serialize, Deserialize)]
pub struct TradingSignal {
    pub timestamp: i64,
    pub symbol: String,
    pub side: String,
    pub strength: f32,
    pub confidence: f32,
    pub components: SignalComponents,
    pub weights: FusionWeights,
    pub model_version: String,
}
```

## Error Handling

### Error Types

```rust
#[derive(Error, Debug)]
pub enum HmmClientError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Service unavailable: {status}")]
    ServiceUnavailable { status: StatusCode },
    
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },
    
    #[error("Model error: {message}")]
    ModelError { message: String },
    
    #[error("Timeout after {duration:?}")]
    Timeout { duration: Duration },
    
    #[error("Validation error: {field} - {message}")]
    ValidationError { field: String, message: String },
}
```

### Error Handling Strategy

**Transient Errors (Retry):**
- Network timeouts
- 500 Internal Server Error
- Connection refused

**Permanent Errors (No Retry):**
- 400 Bad Request
- 422 Validation Error
- Invalid JSON response

**Fallback Activation:**
- Circuit breaker open
- All retry attempts exhausted
- Service unavailable

### Retry Logic

```rust
async fn request_with_retry<T>(&mut self, request_fn: impl Fn() -> RequestBuilder) -> Result<T> {
    for attempt in 0..=retry_attempts {
        match request_fn().send().await {
            Ok(response) if response.status().is_success() => {
                return Ok(response.json().await?)
            }
            Ok(response) if should_retry(response.status()) => {
                sleep(retry_delay * (attempt + 1)).await;
                continue;
            }
            Ok(response) => return Err(permanent_error(response)),
            Err(e) if attempt < retry_attempts => {
                sleep(retry_delay * (attempt + 1)).await;
                continue;
            }
            Err(e) => return Err(e.into()),
        }
    }
}
```

**Retry Configuration:**
- Max attempts: 3
- Base delay: 100ms
- Backoff: Linear (100ms, 200ms, 300ms)
- Total max time: 600ms

## Testing Strategy

### Unit Tests

**HMM Client:**
- Configuration validation
- Request serialization
- Response deserialization
- Error handling

**Circuit Breaker:**
- State transitions
- Failure counting
- Timeout handling
- Recovery logic

**Weight Cache:**
- Cache hit/miss
- TTL expiration
- Size limits
- Thread safety

**Signal Fusion:**
- Weight application
- Threshold logic
- Signal validation
- Cooldown periods

### Integration Tests

**End-to-End Flow:**
- Signal generation → Weight fetching → Fusion → Output
- Cache hit scenarios
- Fallback activation
- Circuit breaker behavior

**Service Communication:**
- Successful requests
- Error responses
- Timeout handling
- Retry logic

**Mock Service Tests:**
- Use `mockito` or `wiremock` for HTTP mocking
- Test all response scenarios
- Verify request formats
- Test error conditions

### Performance Tests

**Latency Benchmarks:**
- Cache hit: <1ms
- Cache miss + service call: <25ms
- Fallback activation: <1ms
- Full fusion pipeline: <30ms

**Load Tests:**
- Concurrent requests: 100+
- Sustained throughput: 1000 req/s
- Memory usage under load
- Cache effectiveness

## Configuration

### Environment Variables

```bash
# HMM Service Configuration
HMM_SERVICE_URL=http://localhost:8000
HMM_SERVICE_TIMEOUT_MS=5000
HMM_SERVICE_RETRY_ATTEMPTS=3
HMM_SERVICE_RETRY_DELAY_MS=100

# Circuit Breaker Configuration
HMM_CIRCUIT_BREAKER_THRESHOLD=5
HMM_CIRCUIT_BREAKER_TIMEOUT_SEC=30

# Cache Configuration
HMM_CACHE_TTL_SEC=60
HMM_CACHE_MAX_SIZE=1000

# Fallback Configuration
HMM_ENABLE_FALLBACK=true
HMM_FALLBACK_W_LDC=0.33
HMM_FALLBACK_W_MR=0.33
HMM_FALLBACK_W_TSMOM=0.34

# Signal Fusion Configuration
SIGNAL_FUSION_THRESHOLD=0.3
SIGNAL_FUSION_COOLDOWN_SEC=60
```

### Configuration File (TOML)

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

## Performance Optimization

### Caching Strategy
- Cache fusion weights for 60 seconds
- Round observations to 3 decimals for cache keys
- Periodic cleanup of expired entries
- Target 80%+ cache hit rate

### Connection Management
- HTTP connection pooling via `reqwest`
- Keep-alive connections
- Connection timeout: 5 seconds
- Idle timeout: 90 seconds

### Async Processing
- Non-blocking HTTP requests
- Concurrent weight fetching for multiple symbols
- Async cache operations
- Tokio runtime for async execution

### Memory Management
- Bounded cache size (1000 entries)
- Efficient serialization with `serde`
- Zero-copy where possible
- Minimal allocations in hot paths

## Monitoring and Observability

### Metrics

```rust
// Request metrics
hmm_requests_total{endpoint, status}
hmm_request_duration_seconds{endpoint}
hmm_request_errors_total{error_type}

// Cache metrics
hmm_cache_hits_total
hmm_cache_misses_total
hmm_cache_size
hmm_cache_evictions_total

// Circuit breaker metrics
hmm_circuit_breaker_state{state}
hmm_circuit_breaker_failures_total
hmm_circuit_breaker_state_changes_total

// Fallback metrics
hmm_fallback_activations_total{reason}
hmm_fallback_active{boolean}
```

### Logging

```rust
// Request logging
debug!("Requesting fusion weights for observations: {:?}", observations);
info!("Fusion weights computed: LDC={:.3}, MR={:.3}, TSMOM={:.3}", w_ldc, w_mr, w_tsmom);

// Error logging
warn!("HMM service request failed, attempt {}/{}: {}", attempt, max_attempts, error);
error!("Circuit breaker opened after {} failures", failure_count);

// Fallback logging
warn!("Using fallback weights due to: {}", reason);
info!("Circuit breaker recovered, resuming normal operation");
```

## Security Considerations

### Network Security
- HTTPS support for production
- TLS certificate validation
- Request timeout limits
- Rate limiting on client side

### Input Validation
- Validate observation ranges
- Sanitize request IDs
- Validate response formats
- Check weight ranges

### Error Information
- Avoid leaking sensitive data in errors
- Sanitize error messages
- Log security events
- Monitor for anomalies
