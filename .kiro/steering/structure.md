---
inclusion: always
---

# Project Structure & Organization

## Repository Layout (Current Implementation)

```
IMP/
├── rust/                           # Rust workspace (production inference)
│   ├── Cargo.toml                  # Workspace configuration
│   ├── feature-pipeline/           # ✅ Technical indicator computation
│   ├── ldc-engine/                 # ✅ Lorentzian k-NN classifier
│   │   ├── benches/                # Performance benchmarks
│   │   ├── tests/                  # Comprehensive test suite
│   │   └── examples/               # Usage examples
│   ├── signal-fusion/              # ✅ Regime-aware signal combination
│   │   ├── docs/                   # Integration guides
│   │   ├── examples/               # Configuration examples
│   │   └── tests/                  # Integration tests
│   ├── training-data-cli/          # ✅ LDC training data management
│   ├── inference-engine/           # ✅ Main inference orchestration
│   └── end-to-end-tests/           # ✅ Full system integration tests
│
├── py/                             # Python workspace (research & services)
│   ├── imp/                        # ✅ Main Python package
│   │   ├── hmm/                    # ✅ HMM training, inference, optimization
│   │   │   ├── trainer.py          # Multi-library HMM training
│   │   │   ├── weight_optimizer.py # Fusion weight optimization
│   │   │   ├── artifact_management.py # Model versioning & storage
│   │   │   └── regime_analysis.py  # Regime interpretation tools
│   │   ├── data/                   # ✅ Data loading and preprocessing
│   │   ├── visualization/          # ✅ Regime visualization
│   │   ├── evaluation/             # ✅ Model evaluation and cross-validation
│   │   ├── tuning/                 # ✅ Interactive parameter optimization
│   │   └── utils/                  # ✅ Shared utilities
│   ├── hmm_service/                # ✅ FastAPI microservice
│   │   ├── core/                   # Service core (inference, caching, etc.)
│   │   ├── routers/                # API endpoints
│   │   └── docs/                   # API documentation
│   ├── notebooks/                  # ✅ Jupyter research environment
│   ├── tests/                      # ✅ Comprehensive Python test suite
│   ├── examples/                   # ✅ Example scripts and demos
│   ├── docs/                       # ✅ Python-specific documentation
│   ├── pyproject.toml              # Python package configuration
│   └── Makefile                    # Development commands
│
├── docs/                           # ✅ Project documentation
│   ├── plan.md                     # Implementation plan (6 phases)
│   ├── architecture.md             # Architecture deep-dive
│   └── ldc-engine/                 # LDC-specific documentation
│
├── notebooks/                      # ✅ Shared research notebooks (10 notebooks)
│   ├── 00_getting_started_tutorial.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_hmm_training_comparison.ipynb
│   ├── 03_regime_analysis.ipynb
│   ├── 04_parameter_optimization.ipynb
│   ├── 05_parameter_tuning_demo.ipynb
│   ├── 06_production_deployment_tutorial.ipynb
│   ├── 07_systematic_hmm_training.ipynb
│   ├── 08_fusion_weight_optimization.ipynb
│   └── 09_minio_deployment_workflow.ipynb
│
├── validation/                     # ✅ Reference implementations
├── docker-compose.yml              # ✅ Infrastructure services
└── .github/workflows/              # CI/CD pipeline (planned)
```

## Naming Conventions

### Rust Components
- **Crates**: kebab-case (`ldc-engine`, `feature-pipeline`)
- **Modules**: snake_case (`regime_analysis`, `weight_optimizer`)
- **Structs**: PascalCase (`HMMInference`, `LDCEngine`)
- **Functions**: snake_case (`compute_features`, `train_hmm`)
- **Constants**: SCREAMING_SNAKE_CASE (`DEFAULT_K_NEIGHBORS`)

### Python Components
- **Packages**: snake_case (`imp`, `hmm`, `data`)
- **Modules**: snake_case (`artifact_management`, `regime_analysis`)
- **Classes**: PascalCase (`HMMTrainer`, `RegimeAnalyzer`)
- **Functions**: snake_case (`train_with_validation`, `plot_state_probabilities`)
- **Constants**: SCREAMING_SNAKE_CASE (`DEFAULT_N_STATES`)

### File Organization
- **Tests**: Mirror source structure (`tests/test_hmm.py` for `imp/hmm/`)
- **Examples**: Descriptive names (`regime_analysis_example.py`)
- **Notebooks**: Numbered sequence (`01_data_exploration.ipynb`)
- **Documentation**: Uppercase (`README.md`, `IMPLEMENTATION_SUMMARY.md`)

## Data Organization

### Storage Hierarchy (MinIO/S3 + Local)
```
# MinIO/S3 Storage (Production)
artifacts/
├── hmm/                            # HMM model artifacts
│   ├── systematic_training_results/
│   │   ├── hmm_2_states.json       # 2-state HMM model
│   │   ├── hmm_3_states.json       # 3-state HMM model
│   │   └── hmm_4_states.json       # 4-state HMM model
│   └── fusion_weights/
│       ├── weights_scipy_v1.json   # Scipy-optimized weights
│       └── weights_grid_v1.json    # Grid-search weights
├── ldc/                            # LDC training snapshots
│   ├── training_snapshot_v1.bin
│   └── hnsw_index_v1.bin
└── experiments/                    # Experiment tracking
    └── exp_20241026_123456/
        ├── config.json
        ├── results.json
        └── artifacts/

# Local Storage (Development)
data/
├── raw/                            # Raw OHLCV data (Parquet)
│   └── symbol=BTCUSDT/
│       └── interval=5m/
│           └── date=2024-01-01/
├── features/                       # Computed technical indicators
│   └── symbol=BTCUSDT/
│       └── interval=5m/
│           └── date=2024-01-01/
├── processed_data/                 # Notebook outputs
│   ├── hmm_observations.parquet
│   ├── quality_report.json
│   └── regime_analysis_report.json
└── signals/                        # Generated trading signals
    └── symbol=BTCUSDT/
        └── date=2024-01-01/
```

### Artifact Formats (Standardized)
- **HMM Models**: JSON with `{A, mu, sigma, weights, metadata, version, timestamp}`
- **Fusion Weights**: JSON with per-state weights and optimization metadata
- **LDC Training**: Binary format with memory-mapped access for performance
- **Features**: Parquet with Snappy compression, partitioned by symbol/interval/date
- **Signals**: JSONL with full audit trail, correlation IDs, and metadata
- **Experiments**: JSON with configuration, results, and artifact references

## Module Responsibilities

### Rust Components

#### `feature-pipeline`
- OHLCV data ingestion and validation
- Technical indicator computation (RSI, MA, momentum, volatility)
- Lazy evaluation with Polars
- Partitioned Parquet output

#### `ldc-engine`
- Ring-buffer for labeled feature vectors
- Lorentzian distance kernel implementation
- Parallel k-NN queries with rayon
- Optional HNSW indexing for large datasets
- Signal generation and validation

#### `signal-fusion`
- HMM service integration
- Regime-aware weight application
- Final signal computation and emission
- Fallback mechanisms for service failures

#### `training-data-cli`
- Training snapshot creation and management
- Label generation from future returns
- Data quality validation
- CLI tools for data preparation

### Python Components

#### `imp.hmm`
- Multi-library HMM training (hmmlearn, pomegranate)
- Model validation and selection
- Artifact management and versioning
- Regime analysis and interpretation

#### `imp.data`
- Data loading and preprocessing
- Feature engineering utilities
- Data quality reporting
- Integration with Rust outputs

#### `imp.visualization`
- Regime visualization and dashboards
- Interactive plotting with Plotly
- State probability analysis
- Economic interpretation tools

#### `imp.evaluation`
- Cross-validation frameworks
- Model comparison and selection
- Performance metrics calculation
- Walk-forward analysis

#### `imp.tuning`
- Interactive parameter optimization
- Grid search and Bayesian optimization
- Real-time model training
- Results visualization

## Development Patterns

### Error Handling
- **Rust**: Use `anyhow` for application errors, `thiserror` for library errors
- **Python**: Custom exception hierarchies with detailed error messages
- **Both**: Comprehensive logging with structured data

### Configuration Management
- **Rust**: Use `config` crate with TOML/JSON files
- **Python**: Use `pydantic` for settings validation
- **Environment**: Support both file-based and environment variable configuration

### Testing Strategy
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark critical paths
- **Notebook Tests**: Validate research notebook execution

### Documentation Standards
- **Code**: Comprehensive docstrings/doc comments
- **APIs**: OpenAPI specs for REST endpoints
- **Architecture**: High-level design documents
- **Examples**: Working code samples for all major features
- **Troubleshooting**: Common issues and solutions

## Workspace Management

### Rust Workspace
- Shared dependencies in workspace `Cargo.toml`
- Consistent versioning across all crates
- Centralized feature flags and build configuration
- Cross-crate integration testing

### Python Package
- Single package with multiple submodules
- Optional dependencies for different use cases
- Development tools integrated via Makefile
- Jupyter kernel configuration for research

### Cross-Language Integration (Production-Ready)
- **Data Contracts**: Parquet schemas with Arrow compatibility
- **API Specifications**: OpenAPI 3.0 specs for HMM service
- **Service Communication**: REST with circuit breakers and retry logic
- **Artifact Exchange**: JSON schemas with semantic versioning
- **Error Handling**: Consistent error codes and structured logging
- **Configuration**: TOML/JSON with environment variable overrides
- **Monitoring**: Structured logs with correlation IDs across services

### Notebook Integration Patterns
- **Data Loading**: Standardized loaders in `imp.data` package
- **Model Training**: Consistent API across `imp.hmm.trainer`
- **Visualization**: Reusable components in `imp.visualization`
- **Artifact Management**: Unified storage via `imp.hmm.artifact_management`
- **Testing**: Notebook validation via pytest integration