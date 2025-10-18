# HMM Microservice Documentation

Welcome to the HMM Microservice documentation. This service provides Hidden Markov Model inference capabilities for the IMP trading system, enabling regime-aware signal fusion through real-time state probability calculation and fusion weight computation.

## Quick Start

### Service Overview

The HMM Microservice is a FastAPI-based Python service that:
- Calculates HMM state probabilities from observation vectors
- Computes fusion weights for regime-aware signal combination
- Supports hot-reloading of models without downtime
- Provides comprehensive health monitoring and metrics
- Integrates with MinIO for model artifact storage

### Basic Usage

```bash
# Start the service
docker-compose up -d hmm-service

# Check service health
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [0.75, -0.32, 0.18],
    "timestamp": 1640995200
  }'
```

## Documentation Structure

### 📋 [API Specification](api_specification.yaml)
Complete OpenAPI 3.0 specification with:
- All endpoint definitions and schemas
- Request/response examples
- Error codes and status codes
- Authentication and security information

### 💡 [API Examples](api_examples.md)
Practical usage examples including:
- Endpoint usage with curl commands
- Client integration examples (Python, Rust, JavaScript)
- Error handling patterns
- Performance optimization tips

### 🔧 [Error Codes & Troubleshooting](error_codes_troubleshooting.md)
Comprehensive troubleshooting guide with:
- Complete error code reference
- Common issues and solutions
- Monitoring and diagnostics
- Recovery procedures

### 🚀 [Deployment Guide](../DEPLOYMENT.md)
Production deployment information including:
- Docker configuration
- Environment variables
- Scaling considerations
- Security best practices

## API Endpoints Overview

### Inference Endpoints
- `POST /inference/state-probabilities` - Calculate HMM state probabilities
- `POST /inference/fusion-weights` - Calculate fusion weights for signal combination
- `POST /inference/predict` - Complete prediction with both probabilities and weights

### Health Monitoring
- `GET /health` - Basic health check (< 5ms response time)
- `GET /health/ready` - Readiness check for orchestration
- `GET /health/detailed` - Comprehensive system information

### Model Management
- `POST /models/reload` - Hot-reload models without downtime
- `GET /models/current` - Current model information
- `GET /models/available` - List all available models in storage

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Inference Latency | < 20ms (p95) | State probability and weight calculation |
| Health Check Latency | < 5ms | Basic health endpoint response time |
| Throughput | 100+ req/sec | Concurrent request handling capacity |
| Availability | 99.9% | Service uptime target |
| Memory Usage | < 512MB | Typical memory footprint |

## Integration Examples

### Python Client
```python
import requests

# Simple prediction
response = requests.post('http://localhost:8000/inference/predict', json={
    'observations': [0.75, -0.32, 0.18]
})
result = response.json()
print(f"Most likely state: {result['most_likely_state']}")
print(f"Fusion weights: {result['fusion_weights']}")
```

### Rust Integration
```rust
// See rust/signal-fusion/src/hmm_client.rs for complete implementation
let client = HMMClient::new("http://localhost:8000");
let prediction = client.predict(vec![0.75, -0.32, 0.18]).await?;
println!("State: {}, Weights: {:?}", prediction.most_likely_state, prediction.fusion_weights);
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HMM_SERVICE_HOST` | `0.0.0.0` | Service bind address |
| `HMM_SERVICE_PORT` | `8000` | Service port |
| `HMM_LOG_LEVEL` | `INFO` | Logging level |
| `HMM_CACHE_SIZE` | `1000` | Cache size limit |
| `HMM_CACHE_TTL` | `300` | Cache TTL in seconds |
| `HMM_MAX_CONCURRENT_REQUESTS` | `100` | Concurrent request limit |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO server endpoint |
| `MINIO_BUCKET` | `hmm-artifacts` | MinIO bucket for models |

### Model Configuration

Models are loaded from MinIO storage with the following structure:
```
hmm-artifacts/
├── experiments/
│   └── {experiment_id}/
│       └── {version}/
│           ├── hmm_artifact.json      # HMM parameters
│           └── fusion_weights.json    # Fusion weight matrices
```

## Monitoring and Observability

### Health Checks

The service provides multiple health check endpoints for different use cases:

1. **Load Balancer Health Check** (`/health`):
   - Ultra-fast response (< 5ms)
   - Basic service status
   - Use for frequent health checks

2. **Readiness Check** (`/health/ready`):
   - Comprehensive readiness validation
   - Model loading status
   - Use for container orchestration

3. **Detailed Health** (`/health/detailed`):
   - System metrics and performance data
   - Use for monitoring dashboards

### Logging

The service uses structured JSON logging with the following levels:
- **DEBUG**: Detailed computation and cache operations
- **INFO**: Request/response logging and model operations
- **WARNING**: Fallback activations and performance issues
- **ERROR**: Model failures and system errors
- **CRITICAL**: Service unavailability and data corruption

### Metrics

Key metrics exposed through the detailed health endpoint:
- Request latency percentiles (p50, p95, p99)
- Request rate and error rate
- Cache hit/miss rates
- Memory and CPU usage
- Model inference statistics

## Security Considerations

### Current Implementation
- No authentication required (development/internal use)
- Input validation on all endpoints
- Structured error responses (no sensitive data exposure)
- Request rate limiting and queue management

### Production Recommendations
- Implement API key authentication
- Enable TLS/HTTPS endpoints
- Configure IP whitelisting
- Set up comprehensive audit logging
- Implement request signing for critical operations

## Development and Testing

### Local Development
```bash
# Start development environment
cd py/hmm_service
docker-compose -f docker-compose.development.yml up -d

# Run tests
python -m pytest tests/

# View API documentation
open http://localhost:8000/docs
```

### Testing Endpoints
```bash
# Test all endpoints
./test_basic.py

# Test MinIO integration
./test_minio_integration.py

# Test inference engine
./test_inference_engine.py
```

## Troubleshooting Quick Reference

### Common Issues

1. **Service Not Ready**
   ```bash
   curl http://localhost:8000/health/ready
   # Check model_loaded and minio_connected status
   ```

2. **High Latency**
   ```bash
   curl http://localhost:8000/health/detailed | jq '.performance_stats'
   # Check queue size and processing times
   ```

3. **Model Loading Failures**
   ```bash
   curl http://localhost:8000/models/available
   curl -X POST http://localhost:8000/models/reload -d '{"validate_model": true}'
   ```

4. **Memory Issues**
   ```bash
   curl http://localhost:8000/health/detailed | jq '.memory_usage_mb'
   # Monitor memory growth over time
   ```

### Emergency Procedures

1. **Service Recovery**: Restart with `docker-compose restart hmm-service`
2. **Model Recovery**: Reload model via `/models/reload` endpoint
3. **Cache Issues**: Restart service to reinitialize cache
4. **Storage Issues**: Check MinIO connectivity and model artifacts

## Support and Contributing

### Getting Help
1. Check this documentation first
2. Review error codes and troubleshooting guide
3. Examine service logs for detailed error information
4. Test with provided examples and client code

### Reporting Issues
When reporting issues, please include:
- Service version and configuration
- Complete error messages and logs
- Steps to reproduce the issue
- Expected vs actual behavior
- System environment details

### Contributing
- Follow existing code style and patterns
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure backward compatibility when possible

## Related Documentation

- [IMP System Architecture](../../../docs/architecture.md)
- [HMM Training Guide](../../imp/hmm/README.md)
- [Rust Integration Examples](../../../rust/signal-fusion/README_HMM_INTEGRATION.md)
- [Performance Testing](../../../docs/ldc-engine/PERFORMANCE_TESTING.md)

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-01  
**Maintainer**: IMP Development Team