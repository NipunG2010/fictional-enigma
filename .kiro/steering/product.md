---
inclusion: always
---

# IMP - Intelligent Market Prediction System

## Product Overview

IMP is a production-grade hybrid Rust+Python trading system that combines machine learning with classical technical analysis. The system uses regime-aware signal fusion to dynamically adapt trading strategies based on market conditions.

**Core Principle**: Different strategies perform better in different market regimes. IMP detects these regimes using HMMs and adjusts signal weights accordingly.

## Architecture Patterns

### Language Separation
- **Rust**: Performance-critical real-time inference, feature computation, signal generation
- **Python**: Research, experimentation, model training, data analysis, visualization
- **Data Exchange**: Parquet files, JSON artifacts, REST APIs for service communication

### Component Responsibilities
- **Feature Pipeline**: Technical indicators (RSI, MA, momentum, volatility) computed in Rust
- **Signal Generators**: LDC (pattern recognition), MR (mean reversion), TSMOM (momentum) in Rust  
- **Regime Detection**: HMM training and inference in Python
- **Signal Fusion**: Regime-aware weight application in Rust
- **Research Environment**: Jupyter notebooks for experimentation

## Development Guidelines

### Code Organization
- Rust workspace in `rust/` with performance-focused crates
- Python package in `py/imp/` with research and ML components
- Shared notebooks in `notebooks/` for research workflows
- Examples in language-specific directories

### Data Conventions
- **Time Series**: Parquet format, partitioned by symbol/interval/date
- **Models**: JSON for HMM parameters, binary for LDC training data
- **Signals**: JSONL with full metadata and audit trails
- **Features**: Arrow IPC for zero-copy transfers between services

### Performance Requirements
- Feature computation: <50ms per bar
- LDC k-NN query: <10ms for 50k samples  
- HMM inference: <20ms
- End-to-end signal generation: <100ms

### Testing Strategy
- Unit tests for individual components
- Integration tests for cross-language workflows
- Performance benchmarks for critical paths
- Notebook validation for research workflows

## Current Implementation Status

**Phase 4 Complete** ✅ - Full HMM microservice integration operational

### Completed Components
- ✅ **Rust Infrastructure**: Complete workspace with 6 crates
  - `feature-pipeline`: Technical indicator computation
  - `ldc-engine`: k-NN classification with performance benchmarks
  - `signal-fusion`: Regime-aware signal combination with HMM integration
  - `training-data-cli`: LDC training data management
  - `inference-engine`: Main orchestration engine
  - `end-to-end-tests`: Comprehensive integration testing

- ✅ **Python Research Environment**: Full ML pipeline
  - `imp.hmm`: Multi-library HMM training (hmmlearn/pomegranate)
  - `imp.data`: Data loading, preprocessing, and quality validation
  - `imp.visualization`: Regime analysis and interactive plotting
  - `imp.evaluation`: Cross-validation and model selection
  - `imp.tuning`: Parameter optimization with grid search and Bayesian methods

- ✅ **HMM Microservice**: Production-ready FastAPI service
  - REST API with OpenAPI documentation
  - Circuit breaker and retry logic
  - Model caching with TTL and size-based eviction
  - Health checks and monitoring endpoints
  - Docker deployment with Nginx reverse proxy

- ✅ **Research Notebooks**: 10 comprehensive Jupyter notebooks
  - Data exploration and quality analysis
  - HMM training comparison (hmmlearn vs pomegranate)
  - Regime analysis and economic interpretation
  - Parameter optimization and sensitivity analysis
  - Fusion weight optimization (scipy + grid search)
  - Production deployment tutorials
  - MinIO integration and artifact management

- ✅ **Artifact Management**: Complete MLOps pipeline
  - MinIO/S3 integration for model storage
  - Semantic versioning and experiment tracking
  - Automated artifact export/import
  - Walk-forward validation framework

- ✅ **Performance Validation**: All targets met
  - Feature computation: <50ms per bar
  - LDC k-NN query: <10ms for 50k samples
  - HMM inference: <20ms
  - End-to-end signal generation: <100ms
  - Concurrent processing: 10+ symbols

**Next Phase**: Phase 5 - Backtesting & Validation Framework

## Key Constraints

- Maintain sub-100ms end-to-end latency for real-time trading
- Support concurrent processing of 10+ symbols
- Ensure graceful degradation when ML services are unavailable
- Preserve full audit trail for regulatory compliance
- Enable rapid research iteration without breaking production systems