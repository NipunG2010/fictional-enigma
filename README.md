# IMP - Intelligent Market Prediction

A hybrid Rust+Python trading system for high-frequency market prediction using LDC (k-NN), MR (Mean Reversion), TSMOM (Time Series Momentum), and HMM-based regime detection.

## Architecture

- **Rust Core**: High-performance feature pipeline, LDC/MR/TSMOM inference, signal fusion
- **Python Research**: HMM training, backtesting, experimentation
- **Storage**: S3/MinIO + Parquet for data, JSON for model artifacts
- **Communication**: REST/gRPC between services, Redis/Kafka for signals

## Quick Start

### Prerequisites

- Rust 1.75+
- Python 3.9+
- Docker & Docker Compose
- Git

### Setup

1. **Clone and initialize**:
   ```bash
   git clone <repository-url>
   cd IMP
   git submodule update --init --recursive
   ```

2. **Start infrastructure**:
   ```bash
   docker-compose up -d minio redis
   ```

3. **Build Rust components**:
   ```bash
   cd rust
   cargo build --release
   ```

4. **Setup Python environment**:
   ```bash
   cd py
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

### Running the System

1. **Start MinIO and Redis**:
   ```bash
   docker-compose up -d
   ```

2. **Run Rust inference engine**:
   ```bash
   cd rust
   cargo run --bin inference-engine
   ```

3. **Run Python HMM service**:
   ```bash
   cd py
   uvicorn imp.hmm.inference:app --host 0.0.0.0 --port 8000
   ```

## Project Structure

```
IMP/
├── rust/                    # Rust workspace
│   ├── inference-engine/    # Main inference service
│   ├── feature-pipeline/    # Feature computation
│   ├── ldc-engine/         # LDC k-NN implementation
│   └── signal-fusion/      # Signal combination logic
├── py/                     # Python workspace
│   ├── imp/               # Python package
│   │   ├── hmm/          # HMM training & inference
│   │   ├── research/     # Research notebooks
│   │   ├── training/     # Model training services
│   │   └── utils/        # Utility functions
│   ├── tests/            # Python tests
│   └── notebooks/        # Jupyter notebooks
├── docs/                 # Documentation
├── docker-compose.yml    # Infrastructure services
└── .github/workflows/    # CI/CD pipeline
```

## Development

### Rust Development

```bash
cd rust
cargo test                    # Run tests
cargo clippy                  # Lint code
cargo fmt                     # Format code
cargo run --bin inference-engine  # Run inference engine
```

### Python Development

```bash
cd py
pytest                       # Run tests
black .                      # Format code
isort .                      # Sort imports
flake8 .                     # Lint code
mypy .                       # Type check
```

### Infrastructure

- **MinIO**: S3-compatible storage at http://localhost:9000
  - Username: `minioadmin`
  - Password: `minioadmin123`
- **Redis**: Signal bus at `localhost:6379`

## Testing

### Unit Tests
```bash
# Rust
cd rust && cargo test

# Python
cd py && pytest tests/
```

### Integration Tests
```bash
# Start services
docker-compose up -d

# Run integration tests
cd py && pytest tests/integration/ -v
```

## Configuration

### Environment Variables

- `MINIO_ENDPOINT`: MinIO server endpoint (default: http://localhost:9000)
- `MINIO_ACCESS_KEY`: MinIO access key (default: minioadmin)
- `MINIO_SECRET_KEY`: MinIO secret key (default: minioadmin123)
- `REDIS_URL`: Redis connection URL (default: redis://localhost:6379)
- `LOG_LEVEL`: Logging level (default: info)

### Configuration Files

- `rust/config.toml`: Rust service configuration
- `py/config.yaml`: Python service configuration

## API Endpoints

### Rust Inference Engine (Port 8080)
- `GET /health`: Health check
- `POST /signal`: Generate trading signal
- `GET /metrics`: Prometheus metrics

### Python HMM Service (Port 8000)
- `GET /health`: Health check
- `POST /predict`: HMM state prediction
- `POST /train`: Train new HMM model

## Monitoring

- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualization
- **Health Checks**: Built-in health endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub.
