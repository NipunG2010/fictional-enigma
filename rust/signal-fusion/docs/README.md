# HMM Integration Documentation

Complete documentation for the Rust HMM Integration component.

## Getting Started

New to the HMM integration? Start here:

1. **[Quick Start](../README_HMM_INTEGRATION.md#quick-start)** - Get up and running in 5 minutes
2. **[Configuration Guide](../CONFIG.md)** - Configure the integration for your environment
3. **[API Reference](API_REFERENCE.md)** - Learn the API basics

## Core Documentation

### API and Usage

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
  - Core types (SignalComponents, FusionWeights, TradingSignal)
  - HmmClient methods (inference, health checks, model management)
  - HmmIntegration helper methods
  - Configuration structures
  - Data models and response types
  - Error types and handling

### Configuration

- **[Configuration Guide](../CONFIG.md)** - Configuration management
  - Configuration methods (TOML, environment variables, defaults)
  - Service configuration (URL, timeout, retries)
  - Circuit breaker settings
  - Cache configuration
  - Fallback weights
  - Signal fusion parameters
  - Environment-specific examples

### Reliability and Resilience

- **[Circuit Breaker](CIRCUIT_BREAKER.md)** - Circuit breaker pattern
  - State machine (Closed, Open, Half-Open)
  - Configuration and tuning
  - Metrics and monitoring
  - Best practices
  - Troubleshooting

- **[Error Handling Guide](../ERROR_HANDLING_GUIDE.md)** - Error handling
  - Error types and classification
  - Retry logic and backoff
  - Fallback mechanisms
  - Error logging and context

### Performance

- **[Performance Tuning](PERFORMANCE_TUNING.md)** - Optimization guide
  - Performance targets and metrics
  - Latency optimization (cache, timeouts, circuit breaker)
  - Throughput optimization (concurrency, batching)
  - Memory optimization
  - Cache tuning strategies
  - Network optimization
  - Configuration profiles (dev, prod, HFT, high-throughput)
  - Benchmarking tools
  - Performance monitoring

### Monitoring

- **[Monitoring & Metrics](MONITORING_METRICS.md)** - Observability
  - Metrics categories (requests, cache, circuit breaker, fallback)
  - Metrics collection and export
  - JSON and Prometheus formats
  - Integration with monitoring systems
  - Alerting recommendations
  - Performance tuning based on metrics

### Troubleshooting

- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Problem solving
  - Connection issues (refused, DNS, SSL/TLS)
  - Performance problems (latency, memory, throughput)
  - Circuit breaker issues (stuck open, frequent opens)
  - Cache problems (low hit rate, not working, evictions)
  - Error handling (timeouts, validation)
  - Configuration issues
  - Service health problems
  - Debugging tools and techniques

## Additional Resources

### Implementation Details

- **[Signal Fusion Guide](../SIGNAL_FUSION_GUIDE.md)** - Signal fusion implementation
- **[Weight Cache Implementation](../WEIGHT_CACHE_IMPLEMENTATION.md)** - Cache internals
- **[Circuit Breaker Implementation](../CIRCUIT_BREAKER_IMPLEMENTATION.md)** - Circuit breaker internals

### Examples

All examples are located in the `examples/` directory:

- **[hmm_integration_example.rs](../examples/hmm_integration_example.rs)** - Comprehensive integration examples
- **[configuration_demo.rs](../examples/configuration_demo.rs)** - Configuration methods
- **[circuit_breaker_demo.rs](../examples/circuit_breaker_demo.rs)** - Circuit breaker behavior
- **[error_handling_demo.rs](../examples/error_handling_demo.rs)** - Error handling patterns
- **[monitoring_demo.rs](../examples/monitoring_demo.rs)** - Metrics and monitoring
- **[signal_fusion_demo.rs](../examples/signal_fusion_demo.rs)** - Signal fusion workflow

Run examples with:
```bash
cargo run --example hmm_integration_example
```

### Tests

Comprehensive test suite demonstrating usage:

- **Unit Tests**: `cargo test --lib`
- **Integration Tests**: `cargo test --test hmm_integration_tests`
- **Benchmarks**: `cargo bench`

## Documentation by Use Case

### I want to...

#### Get Started
- [Quick Start Guide](../README_HMM_INTEGRATION.md#quick-start)
- [Basic Usage Example](../examples/hmm_integration_example.rs)
- [Configuration Basics](../CONFIG.md#usage-examples)

#### Configure the Integration
- [Configuration Guide](../CONFIG.md)
- [Environment Variables](../CONFIG.md#environment-variables)
- [TOML Configuration](../CONFIG.md#configuration-file-toml)
- [Configuration Examples](../CONFIG.md#example-configuration-files)

#### Optimize Performance
- [Performance Tuning Guide](PERFORMANCE_TUNING.md)
- [Cache Optimization](PERFORMANCE_TUNING.md#cache-tuning)
- [Latency Optimization](PERFORMANCE_TUNING.md#latency-optimization)
- [Throughput Optimization](PERFORMANCE_TUNING.md#throughput-optimization)

#### Handle Errors
- [Error Handling Guide](../ERROR_HANDLING_GUIDE.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Error Types](API_REFERENCE.md#error-types)
- [Fallback Mechanisms](API_REFERENCE.md#automatic-fallback)

#### Monitor the System
- [Monitoring & Metrics Guide](MONITORING_METRICS.md)
- [Metrics Collection](MONITORING_METRICS.md#usage)
- [Prometheus Integration](MONITORING_METRICS.md#prometheus-integration)
- [Alerting Recommendations](MONITORING_METRICS.md#alerting-recommendations)

#### Troubleshoot Issues
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Connection Issues](TROUBLESHOOTING.md#connection-issues)
- [Performance Problems](TROUBLESHOOTING.md#performance-problems)
- [Circuit Breaker Issues](TROUBLESHOOTING.md#circuit-breaker-issues)
- [Cache Problems](TROUBLESHOOTING.md#cache-problems)

#### Understand the API
- [API Reference](API_REFERENCE.md)
- [HmmClient API](API_REFERENCE.md#hmmclient)
- [HmmIntegration API](API_REFERENCE.md#hmmintegration)
- [Data Models](API_REFERENCE.md#data-models)

#### Deploy to Production
- [Production Configuration](../CONFIG.md#production-configuration)
- [Performance Tuning](PERFORMANCE_TUNING.md#production-profile)
- [Monitoring Setup](MONITORING_METRICS.md#integration-with-monitoring-systems)
- [Best Practices](../README_HMM_INTEGRATION.md#best-practices)

## Architecture Overview

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

## Key Features

### Reliability
- **Circuit Breaker**: Automatic failure detection and recovery
- **Fallback Weights**: Continue operation when service unavailable
- **Retry Logic**: Exponential backoff for transient failures
- **Health Checks**: Monitor service availability

### Performance
- **Weight Caching**: Sub-millisecond cache hits
- **Connection Pooling**: Efficient HTTP connection reuse
- **Async/Await**: Non-blocking concurrent processing
- **Configurable Timeouts**: Balance speed and reliability

### Observability
- **Comprehensive Metrics**: Request, cache, circuit breaker, fallback
- **Multiple Export Formats**: JSON and Prometheus
- **Structured Logging**: Detailed operation logs
- **Request Tracing**: Track requests end-to-end

### Developer Experience
- **Simple API**: High-level helper for common workflows
- **Flexible Configuration**: TOML files, environment variables, or code
- **Rich Examples**: Comprehensive example code
- **Complete Documentation**: API reference, guides, and troubleshooting

## Requirements Coverage

This documentation satisfies the following requirements from the specification:

- **Requirement 1.1**: HTTP client for HMM service communication
- **Requirement 3.3**: Configurable fallback weights
- **Requirement 6.4**: Structured error logs with context

All requirements are documented with:
- API reference and usage examples
- Configuration options and best practices
- Troubleshooting guides and solutions
- Performance tuning recommendations

## Contributing

When updating documentation:

1. Keep examples up-to-date with code changes
2. Add troubleshooting entries for new issues
3. Update performance recommendations based on benchmarks
4. Include requirement references where applicable
5. Test all code examples before committing

## Version History

- **v1.0.0** - Initial release with complete documentation
  - API Reference
  - Configuration Guide
  - Troubleshooting Guide
  - Performance Tuning Guide
  - Circuit Breaker documentation
  - Monitoring & Metrics guide

## Support

For issues and questions:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Review [Common Issues](TROUBLESHOOTING.md#common-error-messages)
3. Enable debug logging for detailed diagnostics
4. Collect metrics and logs for analysis

## License

See the main project LICENSE file.
