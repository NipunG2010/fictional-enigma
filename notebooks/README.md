# HMM Research Environment - Jupyter Notebooks

This directory contains a comprehensive set of Jupyter notebooks for Hidden Markov Model (HMM) research and experimentation. The notebooks provide an interactive environment for regime detection, model comparison, and parameter optimization.

## Quick Start

**New to the HMM Research Environment?** Start with:
- **00_getting_started_tutorial.ipynb** - Complete introduction and end-to-end workflow

**Ready for production?** See:
- **06_production_deployment_tutorial.ipynb** - Production deployment guide

## Notebook Overview

### 00_getting_started_tutorial.ipynb ⭐ NEW
**Purpose**: Complete introduction to the HMM Research Environment

**Key Features**:
- End-to-end workflow from data loading to production deployment
- Step-by-step guidance for beginners
- Best practices and common patterns
- Comprehensive examples with explanations
- Links to additional resources

**Perfect for**: First-time users, onboarding, quick reference

### 01_data_exploration.ipynb
**Purpose**: Data loading, exploration, and preprocessing for HMM training

**Key Features**:
- Load LDC signal data from Rust engine output
- Comprehensive data quality assessment
- Interactive visualizations and statistical analysis
- Data preprocessing and normalization
- Export processed data for subsequent notebooks

**Outputs**:
- `processed_data/signals_processed.parquet` - Preprocessed signal data
- `processed_data/observations.npy` - HMM training observations
- Data quality reports and visualizations

### 02_hmm_training_comparison.ipynb
**Purpose**: Compare HMM implementations (hmmlearn vs pomegranate)

**Key Features**:
- Multi-library HMM training comparison
- Performance benchmarking and evaluation
- Model quality metrics (AIC, BIC, log-likelihood)
- Training time analysis
- Automated model selection recommendations

**Outputs**:
- `model_comparison_results/hmm_comparison_results.csv` - Comparison results
- `model_comparison_results/best_model_info.json` - Best model configuration
- Performance visualizations and recommendations

### 03_regime_analysis.ipynb
**Purpose**: Comprehensive regime detection and analysis

**Key Features**:
- Regime detection using optimal HMM configuration
- State probability visualization and analysis
- Regime persistence and transition analysis
- Economic interpretation of detected market states
- Regime quality assessment and validation

**Outputs**:
- `regime_analysis_results/regime_analysis_results.json` - Complete analysis
- `regime_analysis_results/signals_with_regimes.parquet` - Data with regime labels
- `regime_analysis_results/state_probabilities.npy` - State probability arrays
- Comprehensive regime visualizations and interpretations

### 04_parameter_optimization.ipynb
**Purpose**: Interactive hyperparameter tuning and optimization

**Key Features**:
- Interactive parameter tuning interface with ipywidgets
- Real-time model training and evaluation
- Automated grid search optimization
- Bayesian optimization (if scikit-optimize available)
- Model comparison and selection tools

**Outputs**:
- `parameter_optimization_results/optimization_results.json` - All optimization results
- `parameter_optimization_results/grid_search_results.csv` - Grid search details
- Interactive tuning interface and optimization recommendations

### 05_parameter_tuning_demo.ipynb
**Purpose**: Interactive demonstration of parameter tuning capabilities

**Key Features**:
- Live parameter tuning with immediate visual feedback
- Interactive widgets for all HMM parameters
- Real-time performance metrics
- Model comparison tools
- Configuration saving and loading

**Perfect for**: Interactive experimentation, parameter exploration, demonstrations

### 06_production_deployment_tutorial.ipynb ⭐ NEW
**Purpose**: Complete guide for deploying models to production

**Key Features**:
- Production readiness testing procedures
- Artifact validation and packaging
- Deployment documentation generation
- Performance monitoring setup
- Rollback procedures and strategies
- Real-time inference simulation

**Perfect for**: Production deployment, MLOps, model lifecycle management

### 08_fusion_weight_optimization.ipynb ⭐ NEW
**Purpose**: Optimize fusion weights for combining signals based on market regimes

**Key Features**:
- Load trained HMM models and signal data
- Optimize weights using scipy SLSQP and grid search methods
- Visualize optimized weights per state with bar charts
- Compare performance metrics (Sharpe ratio, drawdown, win rate)
- Statistical significance testing of improvements
- Walk-forward validation for robustness assessment
- Export optimized weights for production use

**Outputs**:
- `fusion_weight_results/fusion_weights_scipy.json` - Scipy optimized weights
- `fusion_weight_results/fusion_weights_grid.json` - Grid search optimized weights
- `fusion_weight_results/optimization_summary.json` - Complete optimization results
- Performance visualizations and validation reports

**Perfect for**: Signal fusion optimization, regime-based weight allocation, performance improvement

### 09_minio_deployment_workflow.ipynb ⭐ NEW
**Purpose**: Complete workflow for storing, versioning, and deploying HMM artifacts using MinIO

**Key Features**:
- Train models and upload to MinIO with semantic versioning
- List and download artifacts by version
- Tagging workflow for staging and production deployment
- Production artifact retrieval patterns
- Experiment tracking integration
- Comprehensive troubleshooting guide for common MinIO issues

**Outputs**:
- Artifacts stored in MinIO with version control
- Tagged artifacts for staging/production environments
- Experiment tracking metadata
- Performance benchmarks

**Perfect for**: MLOps workflows, artifact management, production deployment, version control

## Utilities

### utils/notebook_utils.py
Common utilities for notebook environment setup, data validation, and progress tracking.

### utils/data_loaders.py
Data loading functions for LDC signals, sample data generation, and preprocessing utilities.

### utils/plotting_helpers.py
Visualization utilities for regime analysis, model comparison, and interactive plotting.

## Getting Started

### Prerequisites
```bash
# Required packages
pip install jupyter numpy pandas matplotlib seaborn scikit-learn hmmlearn ipywidgets

# Optional packages for enhanced functionality
pip install plotly pomegranate scikit-optimize
```

### Running the Notebooks

#### For Beginners

1. **Start with the Tutorial**:
   ```bash
   jupyter notebook 00_getting_started_tutorial.ipynb
   ```
   - Complete introduction to the environment
   - Learn all key concepts and workflows
   - Get hands-on experience with examples

#### For Research Workflow

1. **Data Exploration**:
   ```bash
   jupyter notebook 01_data_exploration.ipynb
   ```
   - Load and preprocess your LDC signal data
   - Generate sample data if real data is not available
   - Understand data characteristics and quality

2. **Compare HMM Implementations**:
   ```bash
   jupyter notebook 02_hmm_training_comparison.ipynb
   ```
   - Compare different HMM libraries and configurations
   - Identify the best performing model setup
   - Understand trade-offs between different approaches

3. **Analyze Regimes**:
   ```bash
   jupyter notebook 03_regime_analysis.ipynb
   ```
   - Perform detailed regime detection analysis
   - Understand market state characteristics
   - Generate economic interpretations

4. **Optimize Parameters**:
   ```bash
   jupyter notebook 04_parameter_optimization.ipynb
   ```
   - Fine-tune model parameters interactively
   - Run automated optimization procedures
   - Select optimal configuration for production

5. **Interactive Tuning** (Optional):
   ```bash
   jupyter notebook 05_parameter_tuning_demo.ipynb
   ```
   - Experiment with parameters interactively
   - Get immediate visual feedback
   - Compare configurations side-by-side

#### For Production Deployment

1. **Deployment Tutorial**:
   ```bash
   jupyter notebook 06_production_deployment_tutorial.ipynb
   ```
   - Validate models for production readiness
   - Create deployment packages
   - Set up monitoring and rollback procedures

### Workflow Integration

The notebooks are designed to work together in sequence:

```
00_getting_started_tutorial.ipynb (Introduction)
    ↓
01_data_exploration.ipynb
    ↓ (processed_data/)
02_hmm_training_comparison.ipynb
    ↓ (model_comparison_results/)
03_regime_analysis.ipynb
    ↓ (regime_analysis_results/)
04_parameter_optimization.ipynb
    ↓ (parameter_optimization_results/)
05_parameter_tuning_demo.ipynb (Optional)
    ↓
07_systematic_hmm_training.ipynb
    ↓ (systematic_training_results/)
08_fusion_weight_optimization.ipynb ⭐ NEW
    ↓ (fusion_weight_results/)
09_minio_deployment_workflow.ipynb ⭐ NEW
    ↓ (MinIO artifact storage)
06_production_deployment_tutorial.ipynb (Deployment)
```

Each notebook can also be run independently with sample data if previous outputs are not available.

## Output Directory Structure

```
notebooks/
├── processed_data/                    # From 01_data_exploration
│   ├── signals_processed.parquet
│   ├── signals_processed.json
│   └── observations.npy
├── model_comparison_results/          # From 02_hmm_training_comparison
│   ├── hmm_comparison_results.csv
│   ├── hmm_comparison_results.json
│   └── best_model_info.json
├── regime_analysis_results/           # From 03_regime_analysis
│   ├── regime_analysis_results.json
│   ├── signals_with_regimes.parquet
│   ├── state_probabilities.npy
│   └── state_sequence.npy
├── parameter_optimization_results/    # From 04_parameter_optimization
│   ├── optimization_results.json
│   ├── grid_search_results.csv
│   └── grid_search_results.json
├── systematic_training_results/       # From 07_systematic_hmm_training
│   ├── hmm_2_states.json
│   ├── hmm_3_states.json
│   ├── hmm_4_states.json
│   ├── hmm_best.json
│   └── training_report.json
└── fusion_weight_results/             # From 08_fusion_weight_optimization ⭐ NEW
    ├── fusion_weights_scipy.json
    ├── fusion_weights_grid.json
    ├── optimization_summary.json
    ├── fusion_weight_comparison.png
    ├── fusion_metrics_comparison.png
    ├── fusion_cumulative_returns.png
    └── fusion_walk_forward.png
```

## Key Features

### Interactive Widgets
- Real-time parameter adjustment with immediate feedback
- Progress bars and status indicators
- Interactive model comparison tools

### Comprehensive Analysis
- Statistical validation of regime detection
- Economic interpretation of market states
- Model quality assessment and diagnostics

### Production Integration
- Export formats compatible with existing HMM system
- Artifact validation and compatibility checking
- Reproducible experiment tracking

### Visualization
- Interactive plots with Plotly (when available)
- Static plots with Matplotlib/Seaborn as fallback
- Comprehensive diagnostic visualizations

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Run the dependency check in each notebook
   - Install missing packages as indicated

2. **Data Loading Errors**:
   - Notebooks will fall back to sample data generation
   - Check data paths and file formats

3. **Memory Issues with Large Datasets**:
   - Reduce sample sizes in configuration sections
   - Use data subsampling for initial exploration

4. **Interactive Widgets Not Working**:
   - Ensure ipywidgets is installed and enabled
   - Try restarting the Jupyter kernel

### Performance Tips

1. **For Large Datasets**:
   - Use data subsampling for initial exploration
   - Increase training iterations gradually
   - Monitor memory usage

2. **For Faster Experimentation**:
   - Start with fewer states and simpler covariance types
   - Use shorter time series for initial testing
   - Leverage cached results between notebook runs

## Contributing

When adding new notebooks or modifying existing ones:

1. Follow the established naming convention
2. Update this README with new notebook descriptions
3. Ensure compatibility with the utilities modules
4. Add appropriate error handling and fallbacks
5. Include comprehensive documentation and examples

## Integration with Production System

The notebooks are designed to integrate seamlessly with the existing HMM implementation in `py/imp/hmm/`. Results can be exported in formats compatible with:

- `HMMArtifact` models for production deployment
- `FusionWeights` for regime-specific signal fusion
- Rust inference engine for real-time regime detection

See the individual notebooks for specific integration examples and export procedures.