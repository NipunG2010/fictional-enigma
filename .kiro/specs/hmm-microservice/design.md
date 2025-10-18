# Design Document

## Overview

The HMM Microservice is a FastAPI-based Python service that provides real-time Hidden Markov Model inference for the IMP trading system. The service loads trained HMM models from MinIO storage and exposes REST endpoints for state probability calculation and fusion weight computation. This enables the Rust inference engine to apply regime-aware signal fusion based on current market conditions.

The service is designed for high availability, low latency (<20ms response time), and seamless integration with the existing system architecture. It supports hot-reloading of models, comprehensive monitoring, and graceful degradation patterns.

## Architecture

### Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HMM Microservice                         │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Application Layer                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Inference     │  │     Health      │  │   Model     │ │
│  │   Endpoints     │  │    Endpoints    │  │ Management  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Business Logic Layer                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ HMM Inference   │  │  Weight Fusion  │  │   Cache     │ │
│  │    Engine       │  │    Calculator   │  │  Manager    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Data Access Layer                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Model Loader    │  │  MinIO Client   │  │  Metrics    │ │
│  │   & Manager     │  │                 │  │ Collector   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   MinIO Storage │
                    │  ┌─────────────┐ │
                    │  │ HMM Models  │ │
                    │  │   & Weights │ │
                    │  └─────────────┘ │
                    └─────────────────┘
```

### Integration Architecture

```
┌─────────────────┐    HTTP/JSON     ┌─────────────────┐
│  Rust Inference │ ──────────────► │ HMM Microservice│
│     Engine      │                 │   (FastAPI)     │
│                 │ ◄────────────── │                 │
└─────────────────┘                 └─────────────────┘
        │                                     │
        │                                     │
        ▼                                     ▼
┌─────────────────┐                 ┌─────────────────┐
│ Signal Emission │                 │  MinIO Storage  │
│   (Redis/Kafka) │                 │  (HMM Models)   │
└─────────────────┘                 └─────────────────┘
```

## Components and Interfaces

### 1. FastAPI Application (`app.py`)

**Responsibilities:**
- Application initialization and configuration
- Middleware setup (CORS, logging, metrics)
- Route registration and error handling
- Graceful shutdown handling

**Key Components:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models, initialize cache
    # Shutdown: Cleanup resources

app = FastAPI(
    title="HMM Microservice",
    version="1.0.0",
    lifespan=lifespan
)
```

### 2. Inference Endpoints (`routers/inference.py`)

**Primary Endpoints:**

#### `POST /inference/state-probabilities`
- **Input:** `{"observations": [s_ldc, s_mr, s_tsmom], "timestamp": 1234567890}`
- **Output:** `{"state_probabilities": [0.3, 0.5, 0.2], "most_likely_state": 1, "confidence": 0.5}`
- **Latency Target:** <20ms

#### `POST /inference/fusion-weights`
- **Input:** `{"observations": [s_ldc, s_mr, s_tsmom], "timestamp": 1234567890}`
- **Output:** `{"weights": {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3}, "state_probabilities": [0.3, 0.5, 0.2]}`
- **Latency Target:** <20ms

#### `POST /inference/predict`
- **Input:** `{"observations": [s_ldc, s_mr, s_tsmom], "timestamp": 1234567890}`
- **Output:** Complete prediction with state probabilities, weights, and metadata
- **Latency Target:** <20ms

### 3. Health Check Endpoints (`routers/health.py`)

#### `GET /health`
- **Output:** `{"status": "healthy", "timestamp": 1234567890, "uptime": 3600}`
- **Latency Target:** <5ms

#### `GET /health/ready`
- **Output:** `{"ready": true, "model_loaded": true, "last_inference": 1234567890}`
- **Checks:** Model loaded, cache initialized, MinIO accessible

#### `GET /metrics`
- **Output:** Prometheus-formatted metrics
- **Metrics:** Request counts, latency percentiles, error rates, cache hit rates

### 4. Model Management Endpoints (`routers/models.py`)

#### `POST /models/reload`
- **Input:** `{"experiment_id": "hmm_v2", "version": "latest", "validate": true}`
- **Output:** `{"success": true, "model_info": {...}, "reload_time": 1.5}`

#### `GET /models/current`
- **Output:** Current model metadata and status

#### `GET /models/available`
- **Output:** List of available models in MinIO

### 5. HMM Inference Engine (`core/inference_engine.py`)

**Responsibilities:**
- Load and manage HMM models from artifacts
- Perform forward filtering for state probability calculation
- Handle model validation and error recovery

**Key Methods:**
```python
class HMMInferenceEngine:
    async def load_model(self, artifact: HMMArtifact, weights: FusionWeights)
    async def predict_state_probabilities(self, observations: np.ndarray) -> np.ndarray
    async def compute_fusion_weights(self, state_probs: np.ndarray) -> Dict[str, float]
    def validate_model(self) -> bool
```

### 6. Cache Manager (`core/cache.py`)

**Responsibilities:**
- Cache frequently accessed model parameters
- Cache recent inference results for identical inputs
- Implement TTL and size-based eviction policies

**Cache Strategy:**
- **Model Parameters:** Cache loaded HMM matrices (A, μ, Σ) in memory
- **Inference Results:** Cache results for identical observation vectors (5-minute TTL)
- **Fusion Weights:** Cache computed weights for state probability patterns

### 7. Model Loader (`core/model_loader.py`)

**Responsibilities:**
- Load HMM artifacts from MinIO storage
- Validate model integrity and compatibility
- Handle model versioning and fallback logic

**Key Features:**
- Automatic fallback to previous model version on load failure
- Model validation using existing `ArtifactValidator`
- Support for hot-reloading without service restart

## Data Models

### Request/Response Models

```python
class InferenceRequest(BaseModel):
    observations: List[float] = Field(..., min_items=3, max_items=3)
    timestamp: Optional[int] = None
    request_id: Optional[str] = None

class StateProabilitiesResponse(BaseModel):
    state_probabilities: List[float]
    most_likely_state: int
    confidence: float
    timestamp: int
    processing_time_ms: float

class FusionWeightsResponse(BaseModel):
    weights: Dict[str, float]  # {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3}
    state_probabilities: List[float]
    most_likely_state: int
    timestamp: int
    processing_time_ms: float

class PredictionResponse(BaseModel):
    state_probabilities: List[float]
    most_likely_state: int
    confidence: float
    fusion_weights: Dict[str, float]
    timestamp: int
    processing_time_ms: float
    model_version: str
    request_id: Optional[str] = None
```

### Configuration Models

```python
class ServiceConfig(BaseModel):
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Model settings
    default_experiment_id: str = "production_hmm"
    model_reload_interval: int = 300  # seconds
    
    # Cache settings
    cache_size: int = 1000
    cache_ttl: int = 300  # seconds
    
    # MinIO settings
    minio_config: MinIOConfig
    
    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
```

## Error Handling

### Error Response Format

```python
class ErrorResponse(BaseModel):
    error: str
    error_code: str
    message: str
    timestamp: int
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
```

### Error Categories

1. **Validation Errors (400)**
   - Invalid observation vector format
   - Missing required fields
   - Out-of-range values

2. **Model Errors (503)**
   - Model not loaded
   - Model validation failure
   - Inference computation failure

3. **System Errors (500)**
   - MinIO connection failure
   - Cache system failure
   - Unexpected computation errors

4. **Rate Limiting (429)**
   - Too many concurrent requests
   - Request rate exceeded

### Fallback Strategies

1. **Model Fallback:** If current model fails, automatically fall back to previous validated version
2. **Static Weights:** If HMM inference fails, return pre-configured static weights
3. **Circuit Breaker:** Temporarily disable failing components and return cached results

## Testing Strategy

### Unit Tests
- **Model Loading:** Test artifact loading and validation
- **Inference Logic:** Test state probability calculations
- **Weight Computation:** Test fusion weight calculations
- **Cache Operations:** Test cache hit/miss scenarios
- **Error Handling:** Test all error conditions

### Integration Tests
- **End-to-End API:** Test complete request/response cycles
- **MinIO Integration:** Test model loading from storage
- **Performance:** Test latency and throughput requirements
- **Concurrent Requests:** Test handling multiple simultaneous requests

### Load Tests
- **Throughput:** Target 100+ requests/second
- **Latency:** Maintain <20ms p95 latency under load
- **Memory Usage:** Monitor memory consumption under sustained load
- **Error Rates:** Ensure <1% error rate under normal load

### Compatibility Tests
- **Rust Integration:** Test with actual Rust inference engine
- **Model Formats:** Test with different HMM artifact versions
- **Network Conditions:** Test with various network latencies

## Deployment Configuration

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# Service Configuration
HMM_SERVICE_HOST=0.0.0.0
HMM_SERVICE_PORT=8000
HMM_SERVICE_WORKERS=1

# Model Configuration
HMM_DEFAULT_EXPERIMENT_ID=production_hmm
HMM_MODEL_RELOAD_INTERVAL=300

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET=hmm-artifacts
MINIO_SECURE=false

# Cache Configuration
HMM_CACHE_SIZE=1000
HMM_CACHE_TTL=300

# Monitoring
HMM_LOG_LEVEL=INFO
HMM_METRICS_ENABLED=true
```

### Health Check Configuration

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Monitoring and Observability

### Metrics Collection

**Request Metrics:**
- `hmm_requests_total{method, endpoint, status}` - Total request count
- `hmm_request_duration_seconds{method, endpoint}` - Request latency histogram
- `hmm_concurrent_requests` - Current concurrent requests

**Model Metrics:**
- `hmm_model_load_duration_seconds` - Model loading time
- `hmm_model_reload_total{status}` - Model reload attempts
- `hmm_inference_duration_seconds` - Inference computation time

**Cache Metrics:**
- `hmm_cache_hits_total` - Cache hit count
- `hmm_cache_misses_total` - Cache miss count
- `hmm_cache_size` - Current cache size

**System Metrics:**
- `hmm_memory_usage_bytes` - Memory consumption
- `hmm_cpu_usage_percent` - CPU utilization
- `hmm_minio_connection_status` - MinIO connectivity

### Logging Strategy

**Structured Logging Format:**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "service": "hmm-microservice",
  "request_id": "req-123",
  "endpoint": "/inference/predict",
  "processing_time_ms": 15.2,
  "model_version": "v1.2.0",
  "state_probabilities": [0.3, 0.5, 0.2]
}
```

**Log Levels:**
- **DEBUG:** Detailed inference calculations, cache operations
- **INFO:** Request/response logging, model operations
- **WARNING:** Fallback activations, performance degradation
- **ERROR:** Model failures, system errors
- **CRITICAL:** Service unavailability, data corruption

### Alerting Rules

1. **High Latency:** p95 latency > 50ms for 5 minutes
2. **High Error Rate:** Error rate > 5% for 2 minutes
3. **Model Load Failure:** Model reload fails 3 consecutive times
4. **Memory Usage:** Memory usage > 80% for 10 minutes
5. **MinIO Connectivity:** MinIO connection failures for 1 minute

## Security Considerations

### Authentication & Authorization
- **API Keys:** Support for API key-based authentication
- **Rate Limiting:** Per-client request rate limiting
- **IP Whitelisting:** Restrict access to known Rust inference engines

### Data Security
- **Input Validation:** Strict validation of all input parameters
- **Output Sanitization:** Ensure no sensitive data in responses
- **Audit Logging:** Log all requests for compliance

### Network Security
- **TLS Support:** HTTPS endpoints for production deployment
- **CORS Configuration:** Restrict cross-origin requests
- **Request Size Limits:** Prevent large payload attacks

## Performance Optimization

### Caching Strategy
- **Model Caching:** Keep loaded models in memory
- **Result Caching:** Cache identical inference results
- **Connection Pooling:** Reuse MinIO connections

### Computational Optimization
- **NumPy Vectorization:** Use vectorized operations for matrix computations
- **Async Processing:** Non-blocking I/O for MinIO operations
- **Memory Management:** Efficient memory usage for large models

### Scaling Considerations
- **Horizontal Scaling:** Support multiple service instances
- **Load Balancing:** Distribute requests across instances
- **Resource Limits:** Configure appropriate CPU/memory limits