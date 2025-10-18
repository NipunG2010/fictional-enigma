# Error Codes and Troubleshooting Guide

This document provides comprehensive information about error codes, common issues, and troubleshooting steps for the HMM Microservice.

## Table of Contents

- [Error Code Reference](#error-code-reference)
- [Common Issues](#common-issues)
- [Troubleshooting Steps](#troubleshooting-steps)
- [Monitoring and Diagnostics](#monitoring-and-diagnostics)
- [Performance Issues](#performance-issues)
- [Recovery Procedures](#recovery-procedures)

## Error Code Reference

### HTTP Status Codes

| Status Code | Description | When It Occurs |
|-------------|-------------|----------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid request format or parameters |
| 404 | Not Found | Resource not found (e.g., no model loaded) |
| 429 | Too Many Requests | Rate limiting or service overload |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Validation Errors (HTTP 400)

#### INVALID_OBSERVATIONS
**Error Code:** `VALIDATION_ERROR.INVALID_OBSERVATIONS`

**Description:** The observation vector is invalid.

**Common Causes:**
- Wrong number of observations (not exactly 3)
- Non-numeric values in observations
- Values outside reasonable range (|value| > 1000)
- Missing observations field

**Example Response:**
```json
{
  "error": "VALIDATION_ERROR",
  "error_code": "INVALID_OBSERVATIONS",
  "message": "Observations must contain exactly 3 numeric values",
  "timestamp": 1640995200,
  "details": {
    "field": "observations",
    "provided_length": 2,
    "required_length": 3
  }
}
```

**Resolution:**
- Ensure observations array has exactly 3 numeric values
- Check that values are within reasonable bounds
- Verify data types are numeric (int or float)

#### INVALID_TIMESTAMP
**Error Code:** `VALIDATION_ERROR.INVALID_TIMESTAMP`

**Description:** The provided timestamp is invalid.

**Common Causes:**
- Negative timestamp values
- Timestamp too far in the future
- Non-integer timestamp

**Resolution:**
- Use Unix timestamp (seconds since epoch)
- Ensure timestamp is positive integer
- Use current time if timestamp is optional

#### MISSING_REQUIRED_FIELD
**Error Code:** `VALIDATION_ERROR.MISSING_REQUIRED_FIELD`

**Description:** A required field is missing from the request.

**Resolution:**
- Check API documentation for required fields
- Ensure all required fields are included in request body

### Model Errors (HTTP 404/503)

#### NO_MODEL_LOADED
**Error Code:** `MODEL_ERROR.NO_MODEL_LOADED`

**Description:** No HMM model is currently loaded in the service.

**Common Causes:**
- Service started without loading a model
- Model loading failed during startup
- Model was unloaded due to error

**Example Response:**
```json
{
  "error": "MODEL_ERROR",
  "error_code": "NO_MODEL_LOADED",
  "message": "No model is currently loaded",
  "timestamp": 1640995200
}
```

**Resolution:**
1. Check service logs for model loading errors
2. Reload model using `/models/reload` endpoint
3. Verify model artifacts exist in MinIO storage
4. Check MinIO connectivity

#### MODEL_VALIDATION_FAILED
**Error Code:** `MODEL_ERROR.MODEL_VALIDATION_FAILED`

**Description:** Model failed validation checks during loading.

**Common Causes:**
- Corrupted model artifacts
- Incompatible model format
- Missing required model components
- Invalid model parameters

**Resolution:**
1. Check model artifact integrity
2. Verify model was trained with compatible library version
3. Re-train model if corruption is suspected
4. Check model validation logs for specific issues

#### MODEL_NOT_FOUND
**Error Code:** `MODEL_ERROR.MODEL_NOT_FOUND`

**Description:** Requested model experiment/version not found in storage.

**Common Causes:**
- Incorrect experiment ID or version
- Model not uploaded to MinIO
- Storage connectivity issues

**Resolution:**
1. Verify experiment ID and version exist using `/models/available`
2. Check MinIO storage for model artifacts
3. Upload missing model artifacts
4. Use correct experiment ID and version

### System Errors (HTTP 500)

#### INFERENCE_COMPUTATION_FAILED
**Error Code:** `SYSTEM_ERROR.INFERENCE_COMPUTATION_FAILED`

**Description:** HMM inference computation failed.

**Common Causes:**
- Numerical instability in HMM calculations
- Memory allocation errors
- Invalid model state
- Corrupted model parameters

**Resolution:**
1. Check service memory usage
2. Reload model to reset state
3. Verify model parameters are valid
4. Check for numerical overflow/underflow in logs

#### CACHE_ERROR
**Error Code:** `SYSTEM_ERROR.CACHE_ERROR`

**Description:** Cache system error occurred.

**Common Causes:**
- Cache memory exhaustion
- Cache corruption
- Cache initialization failure

**Resolution:**
1. Restart service to reinitialize cache
2. Check available memory
3. Reduce cache size configuration if needed
4. Clear cache manually if possible

#### MINIO_CONNECTION_FAILED
**Error Code:** `SYSTEM_ERROR.MINIO_CONNECTION_FAILED`

**Description:** Failed to connect to MinIO storage.

**Common Causes:**
- MinIO service unavailable
- Network connectivity issues
- Invalid credentials
- Firewall blocking connection

**Resolution:**
1. Check MinIO service status
2. Verify network connectivity to MinIO
3. Check MinIO credentials in configuration
4. Test MinIO connection manually

### Service Availability Errors (HTTP 503)

#### SERVICE_OVERLOADED
**Error Code:** `SERVICE_UNAVAILABLE.SERVICE_OVERLOADED`

**Description:** Service is temporarily overloaded.

**Common Causes:**
- Too many concurrent requests
- Request queue full
- Resource exhaustion (CPU/memory)
- Slow downstream dependencies

**Example Response:**
```json
{
  "error": "SERVICE_UNAVAILABLE",
  "error_code": "SERVICE_OVERLOADED",
  "message": "Service is temporarily overloaded, please retry",
  "timestamp": 1640995200,
  "details": {
    "retry_after_seconds": 5,
    "current_queue_size": 100,
    "max_queue_size": 100
  }
}
```

**Resolution:**
1. Implement exponential backoff retry logic
2. Reduce request rate temporarily
3. Scale service horizontally if possible
4. Check system resources (CPU, memory)

#### STORAGE_UNAVAILABLE
**Error Code:** `SERVICE_UNAVAILABLE.STORAGE_UNAVAILABLE`

**Description:** MinIO storage is temporarily unavailable.

**Resolution:**
1. Check MinIO service health
2. Verify network connectivity
3. Wait for storage to recover
4. Use cached models if available

## Common Issues

### 1. High Latency

**Symptoms:**
- Response times > 50ms consistently
- Timeouts on client side
- Queue buildup in service

**Diagnosis:**
```bash
# Check detailed health for performance metrics
curl http://localhost:8000/health/detailed

# Monitor processing times in logs
docker logs hmm-service | grep "processing_time_ms"
```

**Solutions:**
- Check system resources (CPU, memory)
- Verify model size and complexity
- Enable caching if not already enabled
- Scale service horizontally
- Optimize model parameters

### 2. Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- Out of memory errors
- Service crashes after extended operation

**Diagnosis:**
```bash
# Monitor memory usage
curl http://localhost:8000/health/detailed | jq '.memory_usage_mb'

# Check for memory growth over time
docker stats hmm-service
```

**Solutions:**
- Restart service periodically
- Reduce cache size
- Check for memory leaks in model loading
- Monitor garbage collection

### 3. Model Loading Failures

**Symptoms:**
- Service starts but no model loaded
- Model reload requests fail
- Inference requests return "no model loaded" error

**Diagnosis:**
```bash
# Check current model status
curl http://localhost:8000/models/current

# Check available models
curl http://localhost:8000/models/available

# Check service logs
docker logs hmm-service | grep -i "model"
```

**Solutions:**
- Verify MinIO connectivity
- Check model artifact integrity
- Ensure correct experiment ID and version
- Validate model format compatibility

### 4. Cache Issues

**Symptoms:**
- Low cache hit rates
- Inconsistent response times
- Cache-related errors in logs

**Diagnosis:**
```bash
# Check cache statistics
curl http://localhost:8000/health/detailed | jq '.cache_stats'
```

**Solutions:**
- Adjust cache size and TTL settings
- Clear cache and restart service
- Monitor cache performance metrics
- Optimize cache key generation

## Troubleshooting Steps

### Step 1: Check Service Health

```bash
# Basic health check
curl http://localhost:8000/health

# Comprehensive readiness check
curl http://localhost:8000/health/ready

# Detailed system information
curl http://localhost:8000/health/detailed
```

### Step 2: Verify Model Status

```bash
# Check current model
curl http://localhost:8000/models/current

# List available models
curl http://localhost:8000/models/available

# Reload model if needed
curl -X POST http://localhost:8000/models/reload \
  -H "Content-Type: application/json" \
  -d '{"validate_model": true}'
```

### Step 3: Test Inference Endpoints

```bash
# Test basic inference
curl -X POST http://localhost:8000/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [0.5, 0.0, -0.3],
    "request_id": "test-request"
  }'
```

### Step 4: Check Logs

```bash
# View recent logs
docker logs --tail 100 hmm-service

# Follow logs in real-time
docker logs -f hmm-service

# Filter for errors
docker logs hmm-service | grep -i error

# Filter for specific request ID
docker logs hmm-service | grep "test-request"
```

### Step 5: Verify Dependencies

```bash
# Check MinIO connectivity
curl http://minio:9000/minio/health/live

# Test MinIO from service container
docker exec hmm-service curl http://minio:9000/minio/health/live

# Check network connectivity
docker exec hmm-service ping minio
```

## Monitoring and Diagnostics

### Key Metrics to Monitor

1. **Response Time Metrics:**
   - p50, p95, p99 latency
   - Request rate (requests/second)
   - Error rate percentage

2. **System Metrics:**
   - CPU usage percentage
   - Memory usage (MB)
   - Disk I/O
   - Network I/O

3. **Application Metrics:**
   - Cache hit rate
   - Model inference count
   - Queue size and wait times
   - Model reload frequency

### Log Analysis

**Important Log Patterns:**

```bash
# High latency requests
grep "processing_time_ms.*[5-9][0-9]\|processing_time_ms.*[0-9]{3}" logs.txt

# Error patterns
grep -E "(ERROR|CRITICAL|Exception|Failed)" logs.txt

# Model operations
grep -E "(model.*load|reload|validation)" logs.txt

# Cache performance
grep -E "(cache.*hit|cache.*miss)" logs.txt
```

### Health Check Automation

```bash
#!/bin/bash
# health_check.sh - Automated health monitoring

SERVICE_URL="http://localhost:8000"
ALERT_THRESHOLD=3

check_health() {
    local response=$(curl -s -w "%{http_code}" -o /dev/null "$SERVICE_URL/health")
    echo $response
}

check_readiness() {
    local response=$(curl -s "$SERVICE_URL/health/ready")
    local ready=$(echo $response | jq -r '.ready // false')
    echo $ready
}

main() {
    local failures=0
    
    while true; do
        health_code=$(check_health)
        ready_status=$(check_readiness)
        
        if [[ $health_code != "200" ]] || [[ $ready_status != "true" ]]; then
            failures=$((failures + 1))
            echo "Health check failed: HTTP $health_code, Ready: $ready_status"
            
            if [[ $failures -ge $ALERT_THRESHOLD ]]; then
                echo "ALERT: Service unhealthy for $failures consecutive checks"
                # Send alert notification here
            fi
        else
            failures=0
            echo "Service healthy"
        fi
        
        sleep 30
    done
}

main
```

## Performance Issues

### Latency Optimization

1. **Enable Caching:**
   ```bash
   # Verify cache is enabled and properly sized
   curl http://localhost:8000/health/detailed | jq '.cache_stats'
   ```

2. **Connection Pooling:**
   - Ensure clients use connection pooling
   - Monitor connection pool utilization
   - Adjust pool sizes based on load

3. **Model Optimization:**
   - Use smaller models if accuracy permits
   - Optimize model parameters
   - Consider model quantization

### Memory Optimization

1. **Cache Management:**
   ```bash
   # Adjust cache settings in environment
   export HMM_CACHE_SIZE=500
   export HMM_CACHE_TTL=180
   ```

2. **Model Loading:**
   - Load models on-demand
   - Implement model unloading for unused models
   - Monitor model memory usage

### Throughput Optimization

1. **Concurrent Processing:**
   ```bash
   # Increase concurrent request limit
   export HMM_MAX_CONCURRENT_REQUESTS=200
   ```

2. **Load Balancing:**
   - Deploy multiple service instances
   - Use load balancer with health checks
   - Implement circuit breaker patterns

## Recovery Procedures

### Service Recovery

1. **Graceful Restart:**
   ```bash
   # Send SIGTERM for graceful shutdown
   docker kill -s TERM hmm-service
   
   # Wait for graceful shutdown
   sleep 10
   
   # Start new instance
   docker-compose up -d hmm-service
   ```

2. **Force Recovery:**
   ```bash
   # Force kill if graceful shutdown fails
   docker kill -s KILL hmm-service
   
   # Remove container
   docker rm hmm-service
   
   # Start fresh instance
   docker-compose up -d hmm-service
   ```

### Model Recovery

1. **Reload Current Model:**
   ```bash
   curl -X POST http://localhost:8000/models/reload \
     -H "Content-Type: application/json" \
     -d '{"validate_model": true}'
   ```

2. **Fallback to Previous Version:**
   ```bash
   # List available models to find previous version
   curl http://localhost:8000/models/available
   
   # Load specific previous version
   curl -X POST http://localhost:8000/models/reload \
     -H "Content-Type: application/json" \
     -d '{
       "experiment_id": "production_hmm",
       "version": "1.1.0",
       "validate_model": true
     }'
   ```

### Data Recovery

1. **MinIO Recovery:**
   - Check MinIO service status
   - Verify data integrity
   - Restore from backup if needed

2. **Cache Recovery:**
   - Clear corrupted cache
   - Restart service to reinitialize cache
   - Monitor cache performance after recovery

### Emergency Procedures

1. **Service Unavailable:**
   ```bash
   # Check all dependencies
   docker-compose ps
   
   # Restart all services
   docker-compose down
   docker-compose up -d
   
   # Verify recovery
   curl http://localhost:8000/health/ready
   ```

2. **Data Corruption:**
   - Stop service immediately
   - Backup current state
   - Restore from known good backup
   - Validate data integrity
   - Restart service

3. **Security Incident:**
   - Isolate affected service
   - Review access logs
   - Change credentials
   - Update security configurations
   - Monitor for suspicious activity

## Contact and Support

For additional support:

1. **Check Documentation:**
   - API specification: `/docs/api_specification.yaml`
   - Usage examples: `/docs/api_examples.md`

2. **Review Logs:**
   - Service logs: `docker logs hmm-service`
   - Application logs: Check structured JSON logs

3. **Monitor Metrics:**
   - Health endpoints: `/health/*`
   - System metrics: Monitor CPU, memory, network

4. **Escalation:**
   - Create detailed issue report with logs
   - Include reproduction steps
   - Provide system configuration details