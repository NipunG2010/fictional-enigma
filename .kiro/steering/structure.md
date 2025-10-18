---
inclusion: always
---

# Project Structure & Organization

## Repository Layout

```
IMP/
├── rust/                           # Rust workspace for high-performance inference
│   ├── Cargo.toml                  # Workspace configuration
│   ├── feature-pipeline/           # Technical indicator computation
│   ├── ldc-engine/                 # Lorentzian k-NN classifier
│   ├── signal-fusion/              # Regime-aware signal combination
│   ├── training-data-cli/          # LDC training data management
│   └── inference-engine/           # Main inference orchestration
│
├── py/                             # Python workspace for research and training
│   ├── imp/                        # Main Python package
│   │   ├── hmm/                    # HMM training and inference
│   │   ├── data/                   # Data loading and preprocessing
│   │   ├── visualization/          # Regime visualization
│   │   ├── evaluation/             # Model evaluation and cross-validation
│   │   ├── tuning/                 # Interactive parameter optimization
│   │   └── utils/                  # Shared utilities
│   ├── notebooks/                  # Jupyter research environment
│   ├── tests/                      # Python test suite
│   ├── examples/                   # Example scripts and demos
│   ├── docs/                       # Python-specific documentation
│   ├── pyproject.toml              # Python package configuration
│   └── Makefile                    # Development commands
│
├── docs/                           # Project documentation
│   ├── plan.md                     # Implementation plan (6 phases)
│   ├── architecture.md             # Architecture deep-dive
│   └── ldc-engine/                 # LDC-specific documentation
│
├── notebooks/                      # Shared research notebooks
├── validation/                     # Reference implementations and validation
├── docker-compose.yml              # Infrastructure services
└── .github/workflows/              # CI/CD pipeline
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

### Storage Hierarchy
```
data/
├── raw/                            # Raw OHLCV data
│   └── symbol=BTCUSDT/
│       └── interval=5m/
│           └── date=2024-01-01/
├── features/                       # Computed technical indicators
│   └── symbol=BTCUSDT/
│       └── interval=5m/
│           └── date=2024-01-01/
├── models/                         # Trained model artifacts
│   ├── hmm/
│   │   ├── hmm_v1.json
│   │   └── weights_v1.json
│   └── ldc/
│       ├── training_snapshot_v1.bin
│       └── hnsw_index_v1.bin
└── signals/                        # Generated trading signals
    └── symbol=BTCUSDT/
        └── date=2024-01-01/
```

### Artifact Formats
- **HMM Models**: JSON with `{A, mu, sigma, weights, metadata}`
- **LDC Training**: Binary format with memory-mapped access
- **Features**: Parquet with Snappy compression
- **Signals**: JSONL with full audit trail

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

### Cross-Language Integration
- Well-defined data contracts (Parquet, JSON schemas)
- REST API specifications for service communication
- Shared validation logic where possible
- Consistent error handling and logging formats