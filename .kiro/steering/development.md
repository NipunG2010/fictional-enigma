---
inclusion: always
---

# Development Best Practices & Patterns

## Code Quality Standards

### Rust Development
- **Error Handling**: Use `anyhow` for applications, `thiserror` for libraries
- **Async Code**: Prefer `tokio` for I/O, avoid blocking in async contexts
- **Performance**: Benchmark critical paths with `criterion`
- **Memory**: Use `Arc<T>` for shared data, avoid unnecessary clones
- **Configuration**: TOML files with environment variable overrides
- **Logging**: Structured logging with `tracing` crate and correlation IDs

### Python Development
- **Type Hints**: Use comprehensive type annotations (mypy strict mode)
- **Error Handling**: Custom exception hierarchies with detailed messages
- **Data Classes**: Use `pydantic` for validation, `dataclasses` for simple structs
- **Async Code**: Use `asyncio` for I/O-bound operations
- **Testing**: pytest with fixtures, parametrized tests, and >90% coverage
- **Documentation**: Google-style docstrings with examples

## Architecture Patterns

### Service Communication
- **REST APIs**: Use FastAPI with OpenAPI documentation
- **Circuit Breakers**: Implement for all external service calls
- **Retry Logic**: Exponential backoff with jitter
- **Caching**: TTL-based caching for expensive operations
- **Health Checks**: Comprehensive health endpoints for all services

### Data Management
- **Parquet**: Primary format for time series data
- **Partitioning**: By symbol/interval/date for efficient queries
- **Compression**: Snappy for balance of speed and size
- **Schemas**: Enforce consistent schemas across components
- **Versioning**: Semantic versioning for all artifacts

### Configuration Management
- **Environment-Specific**: Separate configs for dev/staging/prod
- **Secrets**: Environment variables, never in code
- **Validation**: Use pydantic for Python, serde for Rust
- **Defaults**: Sensible defaults with override capability
- **Documentation**: Document all configuration options

## Testing Strategy

### Unit Testing
- **Coverage**: Aim for >90% code coverage
- **Isolation**: Mock external dependencies
- **Fast**: Unit tests should run in <1 second each
- **Deterministic**: No flaky tests, use fixed seeds for randomness

### Integration Testing
- **End-to-End**: Test complete workflows
- **Service Integration**: Test service-to-service communication
- **Data Pipeline**: Test data flow from ingestion to output
- **Performance**: Validate latency and throughput targets

### Notebook Testing
- **Execution**: All notebooks must execute without errors
- **Reproducibility**: Use fixed seeds and deterministic data
- **Output Validation**: Check key outputs and visualizations
- **Documentation**: Keep notebooks well-documented and current

## Performance Guidelines

### Rust Optimization
- **Profile First**: Use `perf` and `flamegraph` to identify bottlenecks
- **Parallel Processing**: Use `rayon` for CPU-bound parallelism
- **Memory Allocation**: Minimize allocations in hot paths
- **SIMD**: Use when appropriate for numerical computations
- **Benchmarking**: Continuous benchmarking with criterion

### Python Optimization
- **Vectorization**: Use NumPy/Polars for numerical operations
- **Caching**: Cache expensive computations with `functools.lru_cache`
- **Profiling**: Use `cProfile` and `line_profiler` for optimization
- **Memory**: Monitor memory usage with `memory_profiler`
- **Async**: Use async/await for I/O-bound operations

### Data Processing
- **Lazy Evaluation**: Use Polars lazy frames when possible
- **Chunking**: Process large datasets in chunks
- **Compression**: Use appropriate compression for storage vs speed tradeoffs
- **Indexing**: Create appropriate indexes for query patterns

## Monitoring & Observability

### Logging
- **Structured**: Use JSON format with consistent fields
- **Correlation IDs**: Track requests across services
- **Log Levels**: Use appropriate levels (DEBUG, INFO, WARN, ERROR)
- **Sensitive Data**: Never log secrets or PII
- **Performance**: Log timing for critical operations

### Metrics
- **Business Metrics**: Signal quality, regime detection accuracy
- **System Metrics**: Latency, throughput, error rates
- **Resource Metrics**: CPU, memory, disk usage
- **Custom Metrics**: Domain-specific measurements

### Alerting
- **SLA Violations**: Alert on performance target breaches
- **Error Rates**: Alert on elevated error rates
- **Service Health**: Alert on service unavailability
- **Data Quality**: Alert on data anomalies

## Security Best Practices

### Secrets Management
- **Environment Variables**: Use for all secrets
- **Rotation**: Regular rotation of API keys and passwords
- **Least Privilege**: Minimal required permissions
- **Audit**: Log all access to sensitive resources

### API Security
- **Authentication**: Use proper authentication mechanisms
- **Rate Limiting**: Implement rate limiting on all endpoints
- **Input Validation**: Validate all inputs thoroughly
- **HTTPS**: Use HTTPS for all external communication

### Data Security
- **Encryption**: Encrypt sensitive data at rest and in transit
- **Access Control**: Implement proper access controls
- **Audit Trail**: Log all data access and modifications
- **Compliance**: Follow relevant regulatory requirements

## Development Workflow

### Git Workflow
- **Feature Branches**: Use feature branches for all changes
- **Pull Requests**: Require PR reviews for all changes
- **Commit Messages**: Use conventional commit format
- **Linear History**: Prefer rebase over merge commits

### Code Review
- **Automated Checks**: Run linting, testing, and security scans
- **Human Review**: Require human review for all changes
- **Documentation**: Update documentation with code changes
- **Performance**: Review performance implications of changes

### Deployment
- **Staging**: Test all changes in staging environment
- **Blue-Green**: Use blue-green deployments for zero downtime
- **Rollback**: Have rollback procedures for all deployments
- **Monitoring**: Monitor deployments closely

## Troubleshooting Guidelines

### Common Issues
- **Performance**: Check for memory leaks, inefficient queries
- **Connectivity**: Verify network connectivity and DNS resolution
- **Configuration**: Check environment variables and config files
- **Dependencies**: Verify all dependencies are available and correct versions

### Debugging Tools
- **Rust**: Use `gdb`, `lldb`, or `rust-gdb` for debugging
- **Python**: Use `pdb` or IDE debuggers
- **Profiling**: Use appropriate profiling tools for performance issues
- **Logging**: Increase log levels for detailed debugging

### Documentation
- **Runbooks**: Maintain runbooks for common operational tasks
- **Troubleshooting**: Document common issues and solutions
- **Architecture**: Keep architecture documentation current
- **APIs**: Maintain up-to-date API documentation