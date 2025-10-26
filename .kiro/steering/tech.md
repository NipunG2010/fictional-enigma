---
inclusion: always
---

# Technology Stack & Build System

## Core Technologies

### Rust Components (Production-Ready)
- **polars**: High-performance dataframe operations (lazy evaluation, Arrow backend)
- **ndarray**: Numerical computing for feature vectors and ML operations
- **rayon**: Data parallelism for k-NN queries and concurrent processing
- **hnsw_rs**: Approximate nearest neighbor indexing for large datasets
- **tokio**: Async runtime for service communication
- **reqwest**: HTTP client with circuit breaker and retry logic
- **serde**: Serialization/deserialization for data interchange
- **anyhow/thiserror**: Comprehensive error handling
- **criterion**: Performance benchmarking for critical paths
- **config**: TOML/JSON configuration management
- **tracing**: Structured logging with correlation IDs

### Python Components (Research & Services)
- **hmmlearn**: Primary HMM training library (production-tested)
- **pomegranate**: Alternative HMM library for comparison
- **polars**: Primary data manipulation (faster than pandas)
- **pandas**: Legacy support for existing notebooks
- **scikit-learn**: Model evaluation and validation
- **jupyter/jupyterlab**: Interactive research environment
- **matplotlib/seaborn/plotly**: Comprehensive visualization suite
- **fastapi**: Production HMM inference microservice
- **uvicorn**: ASGI server for FastAPI
- **pydantic**: Data validation and settings management
- **pytest**: Testing framework with coverage
- **black/isort**: Code formatting and import sorting
- **mypy**: Static type checking

### Infrastructure (Containerized)
- **MinIO**: S3-compatible object storage for artifacts
- **Redis**: Signal bus and caching layer
- **Parquet**: Columnar storage with Snappy compression
- **Docker/Docker Compose**: Multi-service orchestration
- **Arrow IPC**: Zero-copy data transfer between services
- **Nginx**: Reverse proxy for production deployment

## Build System

### Rust Workspace (Multi-Crate)
```bash
# Build all components (feature-pipeline, ldc-engine, signal-fusion, etc.)
cargo build --release

# Run comprehensive test suite
cargo test
cargo test --release  # Release mode tests for performance validation

# Run specific components
cargo run --bin feature-pipeline
cargo run --bin ldc-engine
cargo run --bin signal-fusion
cargo run --bin inference-engine

# Performance benchmarks (criterion-based)
cargo bench
cargo bench --bench ldc_performance  # LDC-specific benchmarks
cargo bench --bench signal_fusion    # Signal fusion benchmarks

# End-to-end integration tests
cd rust/end-to-end-tests
cargo test --release

# Code quality
cargo clippy -- -D warnings
cargo fmt --check
```

### Python Environment (Research & Services)
```bash
# Setup development environment
cd py
./setup_research_env.sh  # Linux/macOS (recommended)
# or setup_research_env.bat  # Windows

# Install with all optional dependencies
pip install -e ".[dev,optimization,research,minio]"

# Development workflow (Makefile-based)
make install-dev    # Install with pre-commit hooks
make test          # Run all tests (pytest)
make test-fast     # Skip slow integration tests
make test-minio    # Test MinIO integration specifically
make format        # Format code (black, isort)
make lint          # Run flake8 + mypy
make type-check    # Run mypy type checking
make jupyter       # Start Jupyter Lab with custom kernel
make clean         # Clean artifacts and cache

# HMM Service (FastAPI microservice)
cd py/hmm_service
make start-dev     # Development server with hot reload
make start-prod    # Production server with gunicorn
make test          # Service-specific tests
make docker-build  # Build production Docker image
```

### Infrastructure Services
```bash
# Start local development stack
docker-compose up -d

# Available services:
# - MinIO: http://localhost:9001 (admin/admin123)
# - MinIO API: http://localhost:9000
# - Redis: localhost:6379
# - HMM Service: http://localhost:8000 (when running)

# Production deployment
cd py/hmm_service
docker-compose -f docker-compose.production.yml up -d
```

## Development Workflow

### Rust Development (Performance-Critical)
- **Linting**: `cargo clippy -- -D warnings` (treat warnings as errors)
- **Formatting**: `cargo fmt --check` in CI, `cargo fmt` locally
- **Testing**: Comprehensive test suite in `tests/` directories
  - Unit tests: `#[cfg(test)]` modules
  - Integration tests: `tests/` directory
  - Benchmarks: `benches/` with criterion
  - End-to-end: `rust/end-to-end-tests/` crate
- **Performance**: Benchmark all critical paths with criterion
- **Configuration**: TOML files with environment variable overrides
- **Logging**: Structured logging with tracing crate

### Python Development (Research & Services)
- **Code Style**: Black (88 char line length) + isort for imports
- **Type Checking**: mypy with strict mode for new code
- **Testing**: pytest with coverage reports (>90% target)
- **Documentation**: Comprehensive docstrings (Google style)
- **Pre-commit**: Automated formatting and linting
- **Notebooks**: Jupyter with custom kernel configuration
- **Services**: FastAPI with OpenAPI documentation

### Data Formats & Standards
- **Time Series**: Parquet with Snappy compression, partitioned by `symbol/interval/date`
- **Features**: Arrow IPC for zero-copy transfers between Rust/Python
- **Models**: JSON for HMM parameters with versioning metadata
- **Training Data**: Binary format with memory-mapped access for LDC
- **Signals**: JSONL with full audit trail and correlation IDs
- **Artifacts**: MinIO/S3 storage with semantic versioning

### Configuration Management
- **Rust**: TOML files with `config` crate, environment variable overrides
- **Python**: Pydantic settings with `.env` file support
- **Services**: Docker Compose with environment-specific overrides
- **Secrets**: Environment variables, never committed to git

## Performance Targets (Validated)
- **Feature computation**: <50ms per bar ✅
- **LDC k-NN query**: <10ms for 50k samples ✅
- **HMM inference**: <20ms ✅
- **Signal fusion**: <10ms ✅
- **End-to-end signal generation**: <100ms ✅
- **Concurrent processing**: 10+ symbols simultaneously ✅
- **Service availability**: 99.9% uptime with circuit breakers

## Environment Requirements
- **Python**: >= 3.9 (tested with 3.9, 3.10, 3.11)
- **Rust**: >= 1.70 (MSRV, tested with latest stable)
- **Docker**: >= 20.10 for multi-stage builds
- **Memory**: 8GB+ recommended for full dataset processing
- **Storage**: SSD recommended for Parquet I/O performance

## Quality Gates
- **Rust**: All tests pass, clippy clean, benchmarks within targets
- **Python**: >90% test coverage, mypy clean, all notebooks executable
- **Integration**: End-to-end tests pass, service health checks green
- **Performance**: Benchmarks within 10% of targets