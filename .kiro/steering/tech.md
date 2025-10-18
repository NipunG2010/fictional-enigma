---
inclusion: always
---

# Technology Stack & Build System

## Core Technologies

### Rust Components
- **polars**: High-performance dataframe operations (lazy evaluation, Arrow backend)
- **ndarray**: Numerical computing for feature vectors and ML operations
- **rayon**: Data parallelism for k-NN queries and concurrent processing
- **hnsw_rs**: Approximate nearest neighbor indexing for large datasets
- **tokio**: Async runtime for service communication
- **reqwest**: HTTP client for microservice communication
- **serde**: Serialization/deserialization for data interchange
- **anyhow/thiserror**: Error handling

### Python Components
- **hmmlearn/pomegranate**: HMM training and inference
- **polars/pandas**: Data manipulation and analysis
- **scikit-learn**: Model evaluation and validation
- **jupyter/jupyterlab**: Interactive research environment
- **matplotlib/seaborn/plotly**: Visualization
- **fastapi**: REST API framework for microservices
- **vectorbt**: Backtesting framework
- **pydantic**: Data validation and settings management

### Infrastructure
- **MinIO/S3**: Object storage for data and model artifacts
- **Redis/Kafka**: Signal bus for real-time distribution
- **Parquet**: Columnar storage format for time series data
- **Docker/Docker Compose**: Containerization
- **Arrow IPC**: Zero-copy data transfer between services

## Build System

### Rust Workspace
```bash
# Build all Rust components
cargo build --release

# Run tests
cargo test

# Run specific component
cargo run --bin ldc-engine

# Performance benchmarks
cargo bench

# Lint and format
cargo clippy
cargo fmt
```

### Python Environment
```bash
# Setup development environment
cd py
./setup_research_env.sh  # Linux/macOS
# or setup_research_env.bat  # Windows

# Install dependencies
pip install -e ".[dev,optimization,research]"

# Common development commands
make install-dev    # Install with pre-commit hooks
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format code (black, isort)
make lint          # Run flake8
make type-check    # Run mypy
make jupyter       # Start Jupyter Lab
make clean         # Clean artifacts
```

### Infrastructure Services
```bash
# Start local development services
docker-compose up -d

# Services available:
# - MinIO: http://localhost:9001 (admin/admin123)
# - Redis: localhost:6379
```

## Development Workflow

### Rust Development
- Use `cargo clippy` for linting
- Format with `cargo fmt`
- Write comprehensive tests in `tests/` directories
- Use `#[cfg(test)]` for unit tests
- Benchmark performance-critical code with `criterion`

### Python Development
- Use Black for code formatting (88 char line length)
- Use isort for import sorting
- Type hints with mypy checking
- pytest for testing with coverage reports
- Pre-commit hooks for code quality

### Data Formats
- **Time Series**: Parquet partitioned by `symbol/interval/date`
- **Features**: Arrow IPC for zero-copy transfers
- **Models**: JSON for HMM parameters, binary for LDC snapshots
- **Signals**: JSONL with full metadata and audit trail

## Performance Targets
- **Feature computation**: <50ms per bar
- **LDC k-NN query**: <10ms for 50k samples
- **HMM inference**: <20ms
- **End-to-end signal generation**: <100ms
- **Concurrent processing**: 10+ symbols simultaneously

## Environment Setup
- Python >= 3.9 required
- Rust >= 1.70 required
- Docker for infrastructure services
- Jupyter Lab for research notebooks