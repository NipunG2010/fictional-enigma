# IMP Python Research Environment

Python research environment for HMM-based market regime detection and analysis.

## Overview

The IMP Python Research Environment provides a comprehensive Jupyter-based platform for:

- **HMM Training**: Train Hidden Markov Models using hmmlearn and pomegranate
- **Regime Analysis**: Detect and analyze market regimes with advanced visualization
- **Parameter Optimization**: Interactive hyperparameter tuning and model selection
- **Data Integration**: Seamless integration with Rust LDC engine outputs
- **Production Deployment**: Export research artifacts for production use

## Quick Start

### Automated Setup (Recommended)

**Linux/macOS:**
```bash
cd py
./setup_research_env.sh
```

**Windows:**
```cmd
cd py
setup_research_env.bat
```

### Manual Setup

```bash
cd py

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -e ".[dev,optimization,research]"

# Setup Jupyter kernel
python -m ipykernel install --user --name=imp-research --display-name="IMP Research Environment"

# Validate installation
python -m imp.utils.env_validator
```

### Start Jupyter Lab

```bash
jupyter lab
```

Then select the "IMP Research Environment" kernel when opening notebooks.

## Project Structure

```
py/
├── imp/                          # Main package
│   ├── hmm/                      # HMM components
│   │   ├── trainer.py           # Multi-library HMM training
│   │   ├── inference.py         # HMM inference
│   │   ├── artifact_management.py  # Experiment tracking
│   │   └── regime_analysis.py   # Regime analysis tools
│   ├── data/                     # Data processing
│   │   ├── ldc_loader.py        # LDC signal loading
│   │   ├── preprocessor.py      # Data preprocessing
│   │   └── validator.py         # Data validation
│   ├── visualization/            # Visualization tools
│   │   └── regime_visualizer.py # Regime visualization
│   ├── evaluation/               # Model evaluation
│   │   └── evaluator.py         # Cross-validation and metrics
│   ├── tuning/                   # Parameter optimization
│   │   └── parameter_tuner.py   # Interactive tuning
│   └── utils/                    # Utilities
│       └── env_validator.py     # Environment validation
├── notebooks/                    # Research notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_hmm_training_comparison.ipynb
│   ├── 03_regime_analysis.ipynb
│   ├── 04_parameter_optimization.ipynb
│   ├── 05_parameter_tuning_demo.ipynb
│   └── utils/                   # Notebook utilities
├── tests/                        # Test suite
│   ├── test_hmm.py
│   ├── test_research_framework.py
│   ├── test_notebook_integration.py
│   └── RESEARCH_TESTING_GUIDE.md
├── examples/                     # Example scripts
│   ├── regime_analysis_example.py
│   ├── parameter_tuning_example.py
│   └── artifact_management_example.py
├── docs/                         # Documentation
│   ├── DEVELOPMENT_SETUP.md     # Setup guide
│   ├── DEVELOPMENT_WORKFLOW.md  # Workflow guide
│   └── QUICK_REFERENCE.md       # Quick reference
├── pyproject.toml               # Package configuration
├── Makefile                     # Development commands
└── setup_research_env.sh        # Setup script
```

## Features

### Multi-Library HMM Support

Train HMM models using both hmmlearn and pomegranate:

```python
from imp.hmm.trainer import EnhancedHMMTrainer

# Create trainer
trainer = EnhancedHMMTrainer(
    n_states=3,
    library='hmmlearn',  # or 'pomegranate'
    covariance_type='full'
)

# Train with validation
artifact, metrics = trainer.train_with_validation(
    observations,
    validation_split=0.2
)
```

### Interactive Visualization

Create interactive regime analysis dashboards:

```python
from imp.visualization.regime_visualizer import RegimeVisualizer

viz = RegimeVisualizer(artifact)

# Interactive state probability plot
fig = viz.plot_state_probabilities(state_probs, interactive=True)

# Transition matrix heatmap
fig = viz.plot_transition_matrix(annotate=True)

# Full dashboard
dashboard = viz.create_regime_dashboard(observations, state_probs)
```

### Parameter Optimization

Interactive parameter tuning in Jupyter:

```python
from imp.tuning.parameter_tuner import HMMParameterTuner

tuner = HMMParameterTuner(observations)
interface = tuner.create_tuning_interface()
display(interface)
```

### Comprehensive Evaluation

Cross-validation and model comparison:

```python
from imp.evaluation.evaluator import HMMEvaluator

evaluator = HMMEvaluator()

# Cross-validation
cv_results = evaluator.cross_validate(
    observations,
    trainer_configs=[
        {'n_states': 2, 'library': 'hmmlearn'},
        {'n_states': 3, 'library': 'hmmlearn'},
    ],
    cv_folds=5
)

# Regime stability analysis
stability = evaluator.regime_stability_analysis(state_probs)
```

### Experiment Tracking

Track and version research experiments:

```python
from imp.hmm.artifact_management import ExperimentTracker

tracker = ExperimentTracker(Path("experiments"))
exp_id = tracker.log_experiment(
    research_artifact,
    notes="Initial regime detection experiment"
)
```

## Development Commands

Using the Makefile:

```bash
# Setup
make install          # Install all dependencies
make install-dev      # Install with pre-commit hooks
make validate         # Validate environment

# Testing
make test             # Run all tests
make test-fast        # Run fast tests only
make test-research    # Run research framework tests
make coverage         # Generate coverage report

# Code Quality
make format           # Format code
make lint             # Lint code
make type-check       # Type checking
make pre-commit       # Run all pre-commit hooks

# Development
make jupyter          # Start Jupyter Lab
make clean            # Clean build artifacts
```

## Documentation

- **[Development Setup Guide](docs/DEVELOPMENT_SETUP.md)**: Complete setup instructions
- **[Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)**: Best practices and workflows
- **[Quick Reference](docs/QUICK_REFERENCE.md)**: Command and API reference
- **[Testing Guide](tests/RESEARCH_TESTING_GUIDE.md)**: Testing framework documentation

## Example Notebooks

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`)
   - Load and explore LDC signal data
   - Data quality analysis
   - Preprocessing pipeline

2. **HMM Training Comparison** (`notebooks/02_hmm_training_comparison.ipynb`)
   - Compare hmmlearn vs pomegranate
   - Performance benchmarking
   - Model selection

3. **Regime Analysis** (`notebooks/03_regime_analysis.ipynb`)
   - Detect market regimes
   - Visualize state transitions
   - Economic interpretation

4. **Parameter Optimization** (`notebooks/04_parameter_optimization.ipynb`)
   - Interactive parameter tuning
   - Grid search and Bayesian optimization
   - Model comparison

5. **Parameter Tuning Demo** (`notebooks/05_parameter_tuning_demo.ipynb`)
   - Interactive widget demonstration
   - Real-time model training
   - Results visualization

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m research        # Research framework
pytest tests/ -m notebook        # Notebook execution
pytest tests/ -m integration     # Integration tests

# Run with coverage
pytest tests/ --cov=imp --cov-report=html
```

## API Reference

### Core Classes

- **`EnhancedHMMTrainer`**: Multi-library HMM training
- **`RegimeVisualizer`**: Regime analysis visualization
- **`HMMEvaluator`**: Model evaluation and comparison
- **`HMMParameterTuner`**: Interactive parameter tuning
- **`ExperimentTracker`**: Experiment tracking and versioning
- **`RegimeAnalyzer`**: Advanced regime analysis
- **`LDCDataLoader`**: LDC signal data integration

### Key Functions

- **`train_with_validation()`**: Train HMM with cross-validation
- **`plot_state_probabilities()`**: Visualize state probabilities
- **`cross_validate()`**: Perform time series cross-validation
- **`regime_stability_analysis()`**: Analyze regime persistence
- **`create_tuning_interface()`**: Create interactive tuning UI

## Environment Validation

Validate your environment setup:

```bash
python -m imp.utils.env_validator
```

This checks:
- ✅ Python version (>= 3.9)
- ✅ Core dependencies (numpy, pandas, scikit-learn)
- ✅ HMM libraries (hmmlearn, pomegranate)
- ✅ Jupyter environment
- ✅ Visualization libraries
- ✅ Jupyter kernel configuration
- ✅ Directory structure

## Troubleshooting

### Import Errors

```bash
# Reinstall package
pip install -e .

# Check installation
python -c "import imp; print(imp.__file__)"
```

### Jupyter Kernel Issues

```bash
# List kernels
jupyter kernelspec list

# Reinstall kernel
python -m ipykernel install --user --name=imp-research
```

### Widget Display Issues

```bash
# Enable widgets
jupyter nbextension enable --py widgetsnbextension

# For JupyterLab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab build
```

See [Development Setup Guide](docs/DEVELOPMENT_SETUP.md) for more troubleshooting tips.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Commit: `git commit -m "feat: your feature"`
7. Push: `git push origin feature/your-feature`
8. Create a Pull Request

See [Development Workflow](docs/DEVELOPMENT_WORKFLOW.md) for detailed guidelines.

## Dependencies

### Core Dependencies
- numpy >= 1.24.0
- pandas >= 2.0.0
- polars >= 0.20.0
- scikit-learn >= 1.3.0
- pydantic >= 2.0.0

### HMM Libraries
- hmmlearn >= 0.3.0
- pomegranate >= 0.14.0

### Jupyter Environment
- jupyter >= 1.0.0
- jupyterlab >= 4.0.0
- ipykernel >= 6.25.0
- ipywidgets >= 8.0.0

### Visualization
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.15.0

### Testing
- pytest >= 7.4.0
- pytest-cov >= 4.1.0

See [pyproject.toml](pyproject.toml) for complete dependency list.

## License

MIT License - see LICENSE file for details.

## Support

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` and `notebooks/` directories
- **Issues**: Open an issue on GitHub
- **Testing Guide**: See `tests/RESEARCH_TESTING_GUIDE.md`

## Next Steps

After setup:

1. ✅ Run environment validation: `python -m imp.utils.env_validator`
2. ✅ Start Jupyter Lab: `jupyter lab`
3. ✅ Open example notebooks in `notebooks/`
4. ✅ Run test suite: `pytest tests/`
5. ✅ Read documentation in `docs/`

Happy researching! 🚀
