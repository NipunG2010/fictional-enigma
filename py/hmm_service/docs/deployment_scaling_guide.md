# Deployment and Scaling Guide

This guide provides comprehensive information for deploying and scaling the HMM Microservice in production environments.

## Table of Contents

- [Container Orchestration](#container-orchestration)
- [Health Check Configuration](#health-check-configuration)
- [Deployment Strategies](#deployment-strategies)
- [Scaling Guidelines](#scaling-guidelines)
- [Resource Requirements](#resource-requirements)
- [Performance Tuning](#performance-tuning)
- [Monitoring and Observability](#monitoring-and-observability)

## Container Orchestration

### Health Check Endpoints

The HMM Microservice provides three health check endpoints optimized for different orchestration needs:

#### 1. Liveness Probe: `/health`
- **Purpose**: Determine if the service is alive and should be restarted
- **Response Time**: < 5ms
- **Use Case**: Container orchestration liveness checks
- **Failure Action**: Restart container

```yaml
# Kubernetes example
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

#### 2. Readiness Probe: `/health/ready`
- **Purpose**: Determine if the service is ready to receive traffic
- **Response Time**: < 20ms
- **Use Case**: Load balancer traffic routing decisions
- **Failure Action**: Remove from load balancer pool

```yaml
# Kubernetes example
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

#### 3. Startup Probe: `/health/ready`
- **Purpose**: Determine if the service has completed initialization
- **Response Time**: Variable (depends on model loading)
- **Use Case**: Initial container startup validation
- **Failure Action**: Restart container if startup fails

```yaml
# Kubernetes example
startupProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 12  # Allow up to 60 seconds for startup
```

### Docker Configuration

#### Basic Docker Run
```bash
docker run -d \
  --name hmm-service \
  --health-cmd="curl -f http://localhost:8000/health || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  --health-start-period=40s \
  -p 8000:8000 \
  -e HMM_SERVICE_HOST=0.0.0.0 \
  -e HMM_SERVICE_PORT=8000 \
  -e MINIO_ENDPOINT=minio:9000 \
  imp/hmm-service:v1.0.0
```

#### Docker Compose
```yaml
version: '3.8'
services:
  hmm-service:
    image: imp/hmm-service:v1.0.0
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    environment:
      - HMM_SERVICE_HOST=0.0.0.0
      - HMM_SERVICE_PORT=8000
      - HMM_LOG_LEVEL=INFO
    restart: unless-stopped
```

## Deployment Strategies

### 1. Blue-Green Deployment

**Advantages:**
- Zero downtime
- Easy rollback
- Full environment testing

**Implementation:**
```bash
# Deploy to green environment
docker service create --name hmm-service-green \
  --replicas 3 \
  --health-cmd "curl -f http://localhost:8000/health" \
  --health-interval 30s \
  imp/hmm-service:v1.1.0

# Wait for health checks to pass
docker service ps hmm-service-green

# Switch traffic (update load balancer)
# Remove old service
docker service rm hmm-service-blue
```

### 2. Rolling Update

**Advantages:**
- Gradual deployment
- Resource efficient
- Automatic rollback on failure

**Kubernetes Configuration:**
```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Ensure zero downtime
  template:
    spec:
      containers:
      - name: hmm-service
        image: imp/hmm-service:v1.1.0
        # Health checks ensure traffic only goes to ready pods
```

### 3. Canary Deployment

**Advantages:**
- Risk mitigation
- Performance validation
- Gradual traffic shift

**Implementation:**
```bash
# Deploy canary with 10% traffic
kubectl apply -f canary-deployment.yaml
kubectl patch service hmm-service -p '{"spec":{"selector":{"version":"canary"}}}'

# Monitor metrics and gradually increase traffic
# Full deployment when validated
```

## Scaling Guidelines

### Horizontal Scaling

#### Kubernetes Horizontal Pod Autoscaler (HPA)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hmm-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hmm-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

#### Docker Swarm Scaling
```bash
# Scale up
docker service scale hmm-service=5

# Auto-scaling with external tools
# Use Prometheus metrics + custom scaling logic
```

### Vertical Scaling

#### Resource Adjustment
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

#### Performance-Based Scaling Triggers

| Metric | Scale Up Threshold | Scale Down Threshold |
|--------|-------------------|---------------------|
| CPU Usage | > 70% for 5 minutes | < 30% for 10 minutes |
| Memory Usage | > 80% for 5 minutes | < 40% for 10 minutes |
| Request Latency (p95) | > 50ms for 3 minutes | < 20ms for 10 minutes |
| Queue Size | > 80 requests | < 20 requests |
| Error Rate | > 5% for 2 minutes | N/A (investigate) |

## Resource Requirements

### Minimum Requirements (Single Instance)
- **CPU**: 250m (0.25 cores)
- **Memory**: 256Mi
- **Storage**: 1Gi (for logs and cache)
- **Network**: 100Mbps

### Recommended Production (Per Instance)
- **CPU**: 500m (0.5 cores)
- **Memory**: 512Mi
- **Storage**: 2Gi
- **Network**: 1Gbps

### High-Load Production (Per Instance)
- **CPU**: 1000m (1 core)
- **Memory**: 1Gi
- **Storage**: 4Gi
- **Network**: 1Gbps

### Capacity Planning

#### Requests per Second Capacity
| Instance Size | RPS Capacity | Concurrent Requests |
|---------------|--------------|-------------------|
| Minimum | 50 RPS | 25 |
| Recommended | 100 RPS | 50 |
| High-Load | 200 RPS | 100 |

#### Memory Usage Patterns
- **Base Memory**: ~128Mi (service overhead)
- **Model Memory**: ~50-100Mi (depends on model size)
- **Cache Memory**: ~50-200Mi (configurable)
- **Request Buffer**: ~10Mi per 100 concurrent requests

## Performance Tuning

### Environment Variables

#### Core Performance Settings
```bash
# Concurrent request handling
HMM_MAX_CONCURRENT_REQUESTS=100
HMM_REQUEST_TIMEOUT=30.0
HMM_INFERENCE_TIMEOUT=5.0

# Cache optimization
HMM_CACHE_SIZE=1000
HMM_CACHE_TTL=300
HMM_CACHE_ENABLED=true

# Connection pooling
HMM_MINIO_MAX_CONNECTIONS=10
HMM_MINIO_TIMEOUT=30
```

#### Memory Optimization
```bash
# Python memory settings
PYTHONMALLOC=malloc
MALLOC_ARENA_MAX=2
MALLOC_MMAP_THRESHOLD_=131072
MALLOC_TRIM_THRESHOLD_=131072
MALLOC_TOP_PAD_=131072
MALLOC_MMAP_MAX_=65536
```

### JVM-like Tuning for Python
```bash
# Garbage collection optimization
PYTHONGC=1
PYTHONHASHSEED=0

# Disable debug features in production
PYTHONOPTIMIZE=2
PYTHONDONTWRITEBYTECODE=1
```

### Model Loading Optimization
```bash
# Model caching
HMM_MODEL_CACHE_ENABLED=true
HMM_MODEL_PRELOAD=true
HMM_MODEL_VALIDATION_TIMEOUT=10.0

# Inference optimization
HMM_INFERENCE_BATCH_SIZE=1
HMM_INFERENCE_PARALLEL=false
```

## Monitoring and Observability

### Key Performance Indicators (KPIs)

#### Service Level Indicators (SLIs)
1. **Availability**: Uptime percentage
2. **Latency**: Request response time (p50, p95, p99)
3. **Throughput**: Requests per second
4. **Error Rate**: Percentage of failed requests

#### Service Level Objectives (SLOs)
- **Availability**: 99.9% uptime
- **Latency**: p95 < 50ms, p99 < 100ms
- **Error Rate**: < 1% of requests
- **Throughput**: Handle 100+ RPS per instance

### Health Check Monitoring

#### Automated Health Monitoring Script
```bash
#!/bin/bash
# health_monitor.sh

SERVICE_URL="http://localhost:8000"
ALERT_THRESHOLD=3
CONSECUTIVE_FAILURES=0

monitor_health() {
    while true; do
        # Check basic health
        if ! curl -f -s "$SERVICE_URL/health" > /dev/null; then
            CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
            echo "$(date): Health check failed ($CONSECUTIVE_FAILURES/$ALERT_THRESHOLD)"
            
            if [ $CONSECUTIVE_FAILURES -ge $ALERT_THRESHOLD ]; then
                echo "$(date): ALERT - Service unhealthy for $CONSECUTIVE_FAILURES checks"
                # Send alert (email, Slack, PagerDuty, etc.)
                send_alert "HMM Service Health Check Failed"
            fi
        else
            if [ $CONSECUTIVE_FAILURES -gt 0 ]; then
                echo "$(date): Service recovered after $CONSECUTIVE_FAILURES failures"
            fi
            CONSECUTIVE_FAILURES=0
        fi
        
        sleep 30
    done
}

send_alert() {
    local message="$1"
    # Implement your alerting mechanism here
    echo "ALERT: $message" | mail -s "HMM Service Alert" ops@company.com
}

monitor_health
```

### Load Testing

#### Basic Load Test
```bash
# Using Apache Bench
ab -n 1000 -c 10 -H "Content-Type: application/json" \
   -p test_payload.json \
   http://localhost:8000/inference/predict

# Using curl for sustained load
for i in {1..100}; do
    curl -X POST http://localhost:8000/inference/predict \
         -H "Content-Type: application/json" \
         -d '{"observations": [0.5, 0.0, -0.3]}' &
done
wait
```

#### Performance Benchmarking
```python
import asyncio
import aiohttp
import time
import statistics

async def benchmark_service():
    payload = {"observations": [0.5, 0.0, -0.3]}
    latencies = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(100):
            task = make_request(session, payload, latencies)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    print(f"Mean latency: {statistics.mean(latencies):.3f}s")
    print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.3f}s")
    print(f"P99 latency: {statistics.quantiles(latencies, n=100)[98]:.3f}s")

async def make_request(session, payload, latencies):
    start = time.time()
    async with session.post('http://localhost:8000/inference/predict', 
                           json=payload) as response:
        await response.json()
        latencies.append(time.time() - start)

# Run benchmark
asyncio.run(benchmark_service())
```

## Troubleshooting Deployment Issues

### Common Deployment Problems

#### 1. Service Not Starting
**Symptoms**: Container exits immediately or fails health checks
**Diagnosis**:
```bash
# Check container logs
docker logs hmm-service

# Check resource constraints
docker stats hmm-service

# Verify environment variables
docker exec hmm-service env | grep HMM_
```

#### 2. Model Loading Failures
**Symptoms**: Service starts but readiness checks fail
**Diagnosis**:
```bash
# Check model availability
curl http://localhost:8000/models/available

# Check MinIO connectivity
curl http://localhost:8000/health/detailed | jq '.checks.minio_connected'

# Force model reload
curl -X POST http://localhost:8000/models/reload
```

#### 3. High Memory Usage
**Symptoms**: OOMKilled containers or memory alerts
**Solutions**:
- Reduce cache size: `HMM_CACHE_SIZE=500`
- Increase memory limits
- Enable memory profiling
- Check for memory leaks in logs

#### 4. Performance Degradation
**Symptoms**: High latency or timeouts
**Solutions**:
- Scale horizontally
- Optimize cache settings
- Check downstream dependencies (MinIO)
- Review resource allocation

### Emergency Procedures

#### Service Recovery
```bash
# 1. Quick restart
kubectl rollout restart deployment/hmm-service

# 2. Scale down and up
kubectl scale deployment hmm-service --replicas=0
kubectl scale deployment hmm-service --replicas=3

# 3. Rollback to previous version
kubectl rollout undo deployment/hmm-service
```

#### Data Recovery
```bash
# 1. Check MinIO connectivity
kubectl exec -it hmm-service-pod -- curl http://minio:9000/minio/health/live

# 2. Reload models
kubectl exec -it hmm-service-pod -- \
  curl -X POST http://localhost:8000/models/reload

# 3. Clear cache
kubectl exec -it hmm-service-pod -- \
  curl -X POST http://localhost:8000/admin/cache/clear
```

## Security Considerations

### Container Security
- Run as non-root user
- Use read-only root filesystem
- Drop unnecessary capabilities
- Implement resource limits
- Regular security updates

### Network Security
- Use TLS for external communication
- Implement network policies
- Restrict egress traffic
- Monitor network connections

### Secrets Management
- Use Kubernetes secrets or Docker secrets
- Rotate credentials regularly
- Avoid hardcoded credentials
- Implement least privilege access

## Best Practices

### Deployment Best Practices
1. **Always use health checks** for proper orchestration
2. **Implement graceful shutdown** handling
3. **Use resource limits** to prevent resource starvation
4. **Monitor deployment metrics** during rollouts
5. **Test deployments** in staging environment first
6. **Implement proper logging** for troubleshooting
7. **Use immutable infrastructure** principles
8. **Automate deployment processes** with CI/CD

### Scaling Best Practices
1. **Monitor key metrics** before scaling decisions
2. **Use gradual scaling** to avoid thundering herd
3. **Implement circuit breakers** for downstream dependencies
4. **Test scaling scenarios** in non-production environments
5. **Document scaling procedures** and thresholds
6. **Consider cost implications** of scaling decisions
7. **Plan for peak load scenarios** in advance
8. **Implement proper load balancing** strategies