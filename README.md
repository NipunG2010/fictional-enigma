# IMP - Intelligent Market Prediction

A production-grade hybrid Rust+Python trading system that combines machine learning with classical technical analysis to generate adaptive trading signals. The system leverages Lorentzian Distance Classification (LDC) for pattern recognition, Mean Reversion (MR) and Time Series Momentum (TSMOM) for directional signals, and Hidden Markov Models (HMM) for regime-aware signal fusion.

## Overview

IMP solves a fundamental challenge in algorithmic trading: different strategies perform better in different market conditions. Rather than using static signal weights, IMP dynamically adapts its strategy mix based on detected market regimes, allowing it to be aggressive during trending markets and conservative during ranging or volatile periods.

The system is built as a hybrid architecture that leverages Rust for high-performance real-time inference and Python for research, experimentation, and model training. This separation allows data scientists to iterate quickly in Jupyter notebooks while maintaining production-grade performance for live trading.

## Architecture

### System Components

The architecture consists of five major components working together:

#### 1. Feature Pipeline (Rust)
The feature pipeline ingests raw OHLCV (Open, High, Low, Close, Volume) data and computes technical indicators in real-time using Polars for efficient dataframe operations.

**Key Features**:
- RSI (Relative Strength Index) for momentum measurement
- Multiple moving averages (SMA, EMA) for trend identification
- Standard deviation and z-scores for volatility assessment
- Multi-timeframe momentum indicators
- Lazy evaluation for memory efficiency
- Partitioned Parquet storage by symbol/interval/date

**Performance**: Processes 1+ months of 5-minute data with sub-second latency.

#### 2. Signal Generators (Rust)
Three independent signal generators analyze the computed features and produce directional signals:

**LDC Engine (Lorentzian Distance Classification)**:
- k-NN classifier using Lorentzian distance metric
- Maintains ring-buffer of labeled historical feature vectors
- Parallel k-NN queries using rayon for multithreading
- Optional HNSW indexing for approximate nearest neighbor search
- Generates `s_LDC` signal by identifying similar historical patterns
- Handles 50k+ training samples with <10ms query latency

**MR Scorer (Mean Reversion)**:
- Detects price deviations from moving averages
- Z-score based thresholding for entry signals
- Generates `s_MR` signal for mean reversion opportunities
- Optimized for ranging market conditions

**TSMOM Scorer (Time Series Momentum)**:
- Multi-timeframe momentum analysis
- Trend strength measurement
- Generates `s_TSMOM` signal for momentum trades
- Optimized for trending market conditions

#### 3. HMM Regime Detection (Python)
Hidden Markov Models analyze the three signals to identify distinct market regimes and their transition probabilities.

**Capabilities**:
- Trains HMMs with 2-4 states using hmmlearn or pomegranate
- Observes `[s_LDC, s_MR, s_TSMOM]` as multivariate time series
- Identifies market states (trending, ranging, volatile, calm)
- Computes state transition probabilities
- Evaluates model quality using AIC/BIC and log-likelihood
- Exports artifacts as JSON (transition matrix A, means μ, covariances Σ)

**Research Environment**:
- Comprehensive Jupyter notebooks for experimentation
- Interactive parameter tuning with ipywidgets
- Regime visualization and economic interpretation
- Cross-validation and walk-forward analysis
- Experiment tracking and versioning

#### 4. Signal Fusion (Rust)
Combines the three signals using regime-specific weights to produce the final trading signal.

**Fusion Logic**:
- Fetches current regime probabilities from HMM service
- Applies per-state optimized weights: `w_LDC[state]`, `w_MR[state]`, `w_TSMOM[state]`
- Computes weighted signal: `signal_final = Σ(P(state) × (w_LDC × s_LDC + w_MR × s_MR + w_TSMOM × s_TSMOM))`
- Falls back to static weights if HMM service unavailable
- Emits final signal to Redis/Kafka for downstream consumption

**Adaptive Behavior**: In trending regimes, TSMOM receives higher weight; in ranging regimes, MR dominates; in uncertain regimes, LDC pattern recognition takes precedence.

#### 5. HMM Inference Service (Python - Planned Phase 4)
FastAPI microservice that provides real-time regime detection for the Rust fusion engine.

**Endpoints**:
- `POST /predict`: Returns state probabilities for current signal observations
- `POST /train`: Triggers model retraining with new data
- `GET /health`: Service health check
- `GET /weights`: Returns current regime-specific fusion weights

**Features**:
- Efficient forward filtering algorithm for state probability calculation
- Model artifact caching for low-latency inference
- Graceful degradation on model loading failures
- Comprehensive monitoring and logging

### Data Flow

```
Raw OHLCV Data
    ↓
[Feature Pipeline] → Parquet Storage
    ↓
Technical Indicators (RSI, MA, momentum, volatility)
    ↓
    ├─→ [LDC Engine] → s_LDC
    ├─→ [MR Scorer] → s_MR
    └─→ [TSMOM Scorer] → s_TSMOM
         ↓
    [s_LDC, s_MR, s_TSMOM]
         ↓
    [HMM Service] → State Probabilities + Weights
         ↓
    [Signal Fusion] → Final Trading Signal
         ↓
    Redis/Kafka → Downstream Trading Systems
```

### Technology Stack

**Rust Components**:
- `polars`: High-performance dataframe operations
- `ndarray`: Numerical computing for feature vectors
- `rayon`: Data parallelism for k-NN queries
- `hnsw_rs`: Approximate nearest neighbor indexing
- `tokio`: Async runtime for service communication
- `reqwest`: HTTP client for HMM service calls
- `serde`: Serialization for data interchange

**Python Components**:
- `hmmlearn` / `pomegranate`: HMM training and inference
- `polars` / `pandas`: Data manipulation
- `scikit-learn`: Model evaluation and validation
- `jupyter` / `jupyterlab`: Interactive research environment
- `matplotlib` / `seaborn` / `plotly`: Visualization
- `fastapi`: REST API framework (Phase 4)
- `vectorbt`: Backtesting framework (Phase 5)

**Infrastructure**:
- `MinIO` / `S3`: Object storage for data and artifacts
- `Redis` / `Kafka`: Signal bus for real-time distribution
- `Parquet`: Columnar storage format for time series
- `Docker` / `Docker Compose`: Containerization
- `Prometheus` / `Grafana`: Monitoring and alerting (Phase 6)

### Storage Architecture

**Time Series Data**:
- Format: Parquet with Snappy compression
- Partitioning: `symbol/interval/date` for efficient queries
- Schema: `timestamp, open, high, low, close, volume, features...`

**Model Artifacts**:
- HMM Models: JSON format with `{A, mu, sigma, weights, metadata}`
- LDC Training Data: Binary format with memory-mapped access
- Fusion Weights: JSON with per-state weight configurations
- Versioning: Semantic versioning with experiment tracking

**Signal Logs**:
- Format: JSONL (JSON Lines) for append-only logging
- Schema: `{timestamp, symbol, s_LDC, s_MR, s_TSMOM, state_probs, signal_final, metadata}`
- Retention: Configurable with automated archival to S3

### Performance Characteristics

**Latency Targets**:
- Feature computation: <50ms per bar
- LDC k-NN query: <10ms for 50k samples
- MR/TSMOM scoring: <5ms
- HMM inference: <20ms
- Signal fusion: <10ms
- **End-to-end**: <100ms for complete signal generation

**Throughput**:
- Concurrent processing: 10+ symbols simultaneously
- Feature pipeline: 1000+ bars/second
- LDC queries: 100+ queries/second
- Signal emission: Real-time with <1s delay

**Scalability**:
- Horizontal scaling via multiple Rust inference instances
- Stateless design allows load balancing
- Shared storage via MinIO/S3
- Redis pub/sub for signal distribution

### Design Principles

**Separation of Concerns**:
- Rust handles performance-critical real-time inference
- Python handles research, training, and experimentation
- Clear interfaces between components via REST/gRPC

**Fault Tolerance**:
- Graceful degradation: Falls back to static weights if HMM service fails
- Circuit breakers for external service calls
- Comprehensive error handling and logging
- Health checks for all services

**Observability**:
- Structured logging with correlation IDs
- Metrics collection for all critical paths
- Signal audit trail for debugging and compliance
- Performance profiling and benchmarking tools

**Research Velocity**:
- Jupyter notebooks for rapid experimentation
- Interactive parameter tuning with immediate feedback
- Experiment tracking and reproducibility
- Easy artifact export to production

## Project Structure

```
IMP/
├── rust/                           # Rust workspace for high-performance inference
│   ├── feature-pipeline/           # Technical indicator computation
│   ├── ldc-engine/                 # Lorentzian k-NN classifier
│   ├── signal-fusion/              # Regime-aware signal combination
│   ├── training-data-cli/          # LDC training data management
│   └── inference-engine/           # Main inference orchestration (Phase 4)
│
├── py/                             # Python workspace for research and training
│   ├── imp/                        # Main Python package
│   │   ├── hmm/                    # HMM training and inference
│   │   │   ├── trainer.py          # Multi-library HMM training
│   │   │   ├── inference.py        # FastAPI service (Phase 4)
│   │   │   ├── artifact_management.py  # Experiment tracking
│   │   │   └── regime_analysis.py  # Regime analysis tools
│   │   ├── data/                   # Data loading and preprocessing
│   │   ├── visualization/          # Regime visualization
│   │   ├── evaluation/             # Model evaluation and cross-validation
│   │   └── tuning/                 # Interactive parameter optimization
│   ├── notebooks/                  # Jupyter research environment
│   │   ├── 00_getting_started_tutorial.ipynb
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_hmm_training_comparison.ipynb
│   │   ├── 03_regime_analysis.ipynb
│   │   ├── 04_parameter_optimization.ipynb
│   │   └── 06_production_deployment_tutorial.ipynb
│   └── tests/                      # Python test suite
│
├── docs/                           # Documentation
│   ├── plan.md                     # Implementation plan (6 phases)
│   ├── architecture.md             # Architecture deep-dive
│   └── ldc-engine/                 # LDC-specific documentation
│
├── docker-compose.yml              # Infrastructure services (MinIO, Redis)
└── .github/workflows/              # CI/CD pipeline
```

## Why This Architecture?

**Rust for Inference**: Trading systems need consistent low-latency performance. Rust provides memory safety without garbage collection pauses, making it ideal for real-time signal generation where every millisecond counts.

**Python for Research**: Data scientists need rapid iteration. Python's rich ecosystem (Jupyter, scikit-learn, visualization libraries) allows quick experimentation with HMM configurations, parameter tuning, and backtesting without recompiling.

**Hybrid Approach**: By separating concerns, we get the best of both worlds. Researchers can iterate in notebooks and export artifacts that the Rust engine consumes for production inference. The HMM service only needs to run inference (forward filtering), not training, keeping latency low.

**Parquet Storage**: Columnar format provides excellent compression and query performance for time series data. Partitioning by symbol/interval/date allows efficient range queries without scanning entire datasets.

**Regime-Aware Fusion**: Static signal weights perform poorly across different market conditions. By detecting regimes and using state-specific weights, the system adapts its strategy mix automatically, improving risk-adjusted returns.

## Documentation

- **[Implementation Plan](docs/plan.md)**: Complete 6-phase development roadmap
- **[Architecture Deep-Dive](docs/architecture.md)**: Detailed technical specifications
- **[LDC Engine Docs](docs/ldc-engine/)**: LDC-specific testing and performance guides
- **[Setup Guide](docs/setup.md)**: Installation and configuration (to be created)

## License

MIT License - see LICENSE file for details.

## Implementation Status

**Currently at: Phase 3, Step 2 - HMM Model Development** 🔄

Progress through the 6-phase implementation plan:

- ✅ **Phase 0**: Foundation & Infrastructure
- ✅ **Phase 1**: MVP Feature Pipeline & Basic Signals  
- ✅ **Phase 2**: LDC Engine Implementation
- ✅ **Phase 3**: Python Research & HMM Prototyping
  - ✅ **Phase 3.1**: HMM Research Environment Setup
  - ✅ **Phase 3.2**: HMM Model Development
  - ✅ **Phase 3.3**: Fusion Weight Optimization
  - ✅ **Phase 3.4**: Artifact Management
- ⏳ **Phase 4**: HMM Microservice & Integration (next)
- ⏳ **Phase 5**: Backtesting & Validation
- ⏳ **Phase 6**: Production Hardening (optional)

See [docs/plan.md](docs/plan.md) for detailed phase descriptions and success criteria.
