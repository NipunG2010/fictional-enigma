# HMM Microservice Deployment Guide

This guide covers deployment options for the HMM Microservice in different environments.

## Quick Start

### Development Environment
```bash
# Setup development environment
make setup-env-dev
make deploy-dev

# Or manually:
cp .env.development .env
docker-compose -f docker-compose.development.yml up -d
```

### Production Environment
```bash
# Setup production environment (update secrets first!)
make setup-env-prod
make deploy-prod

# Or manually:
cp .env.production .env
# Edit .env with production secrets
docker-compose -f docker-compose.production.yml up -d
```

## Environment Configurations

### Development (.env.development)
- Debug mode enabled
- Verbose logging
- Hot-reload enabled
- Relaxed security settings
- Smaller cache sizes
- Local MinIO instance

### Production (.env.production)
- Optimized for performance
- JSON structured logging
- Security headers enabled
- Larger cache sizes
- External MinIO/S3 storage
- Resource limits configured

## Docker Images

### Standard Image (Dockerfile)
- Multi-stage build for optimization
- Non-root user for security
- Health checks included
- Production-ready

### Development Image (Dockerfile.development)
- Development tools included
- Debugger support (port 5678)
- Hot-reload capabilities
- Source code mounted as volume

## Service Dependencies

### Required Services
- **MinIO**: Object storage for HMM model artifacts
- **HMM Service**: Main application service

### Optional Services
- **Redis**: Distributed caching (use `--profile redis`)
- **Nginx**: Reverse proxy and load balancer (use `--profile nginx`)

## Configuration Management

### Environment Variables
All configuration is managed through environment variables. See:
- `.env.example` - Template with all options
- `.env.development` - Development defaults
- `.env.production` - Production template

### Key Configuration Areas

#### Service Settings
```bash
HMM_SERVICE_HOST=0.0.0.0
HMM_SERVICE_PORT=8000
HMM_SERVICE_WORKERS=4
HMM_LOG_LEVEL=INFO
```

#### MinIO Configuration
```bash
MINIO_ENDPOINT=minio.example.com:9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_BUCKET=hmm-artifacts-prod
MINIO_SECURE=true
```

#### Performance Tuning
```bash
HMM_MAX_CONCURRENT_REQUESTS=500
HMM_CACHE_SIZE=5000
HMM_INFERENCE_TIMEOUT=5.0
HMM_CIRCUIT_BREAKER_ENABLED=true
```

## Deployment Commands

### Docker Compose Commands
```bash
# Development
docker-compose -f docker-compose.development.yml up -d
docker-compose -f docker-compose.development.yml logs -f

# Production
docker-compose -f docker-compose.production.yml up -d
docker-compose -f docker-compose.production.yml logs -f

# With profiles
docker-compose -f docker-compose.production.yml --profile nginx up -d
```

### Makefile Commands
```bash
# Development
make deploy-dev          # Full development deployment
make docker-run-dev      # Start development containers
make docker-logs-dev     # View development logs

# Production
make deploy-prod         # Full production deployment
make docker-run-prod     # Start production containers
make docker-logs         # View production logs

# Management
make health             # Check service health
make status             # Check if service is running
make docker-clean       # Clean up containers and volumes
```

## Health Checks and Monitoring

### Health Endpoints
- `GET /health` - Basic health check
- `GET /health/ready` - Readiness check (includes model status)
- `GET /metrics` - Prometheus metrics (restricted in production)

### Monitoring Setup
```bash
# Check service status
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/health/ready

# View metrics (development only by default)
curl http://localhost:8001/metrics
```

## Security Considerations

### Production Security
1. **Update default credentials** in `.env.production`
2. **Configure HTTPS** in nginx.conf
3. **Restrict metrics access** to internal networks
4. **Use strong API keys** for authentication
5. **Enable CORS restrictions** for allowed origins

### Network Security
- Services communicate through internal Docker network
- Only necessary ports exposed to host
- Nginx provides additional security layer

## Scaling and Performance

### Horizontal Scaling
```bash
# Scale service instances
docker-compose -f docker-compose.production.yml up -d --scale hmm-service=3
```

### Resource Limits
Configure in docker-compose files:
```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '1.0'
    reservations:
      memory: 512M
      cpus: '0.5'
```

### Performance Tuning
Key environment variables for performance:
- `HMM_MAX_CONCURRENT_REQUESTS` - Concurrent request limit
- `HMM_CACHE_SIZE` - In-memory cache size
- `HMM_INFERENCE_TIMEOUT` - Inference operation timeout
- `HMM_SERVICE_WORKERS` - Number of worker processes

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
make docker-logs

# Check configuration
docker-compose config

# Validate environment
python -c "from core.config import get_settings; print(get_settings())"
```

#### MinIO Connection Issues
```bash
# Test MinIO connectivity
docker-compose exec hmm-service python -c "
from core.config import get_settings
settings = get_settings()
print('MinIO validation:', settings.validate_minio_connection())
"
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Monitor request metrics
curl http://localhost:8001/metrics | grep hmm_

# Check cache hit rates
curl http://localhost:8000/health/ready
```

### Log Analysis
```bash
# Follow logs with filtering
docker-compose logs -f hmm-service | grep ERROR

# JSON log parsing (production)
docker-compose logs hmm-service | jq '.level, .message'
```

## Backup and Recovery

### Model Artifacts
- HMM models stored in MinIO/S3
- Backup MinIO data volume: `minio_prod_data`
- Consider cross-region replication for production

### Configuration Backup
- Store environment files in secure configuration management
- Version control Docker configurations
- Document any manual configuration changes

## Updates and Maintenance

### Rolling Updates
```bash
# Build new image
docker build -t hmm-service:v1.1.0 .

# Update with zero downtime
docker-compose -f docker-compose.production.yml up -d --no-deps hmm-service
```

### Maintenance Mode
```bash
# Graceful shutdown
docker-compose -f docker-compose.production.yml stop hmm-service

# Maintenance tasks
docker-compose -f docker-compose.production.yml run --rm hmm-service python maintenance_script.py

# Restart
docker-compose -f docker-compose.production.yml start hmm-service
```