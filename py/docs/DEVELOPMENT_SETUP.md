# Development Environment Setup Guide

This guide provides comprehensive instructions for setting up the IMP HMM Research Environment for development and experimentation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Manual Setup](#manual-setup)
4. [Environment Validation](#environment-validation)
5. [Jupyter Configuration](#jupyter-configuration)
6. [Development Workflow](#development-workflow)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 8GB RAM minimum (16GB recommended for large datasets)
- **Disk Space**: 2GB for dependencies and data

### Required Tools

- Python 3.9+
- pip (Python package manager)
- git (version control)
- Virtual environment tool (venv, conda, or virtualenv)

## Quick Start

### Automated Setup (Recommended)

The fastest way to get started is using the automated setup script:

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

This script will:
1. Check Python version compatibility
2. Create a virtual environment
3. Install all required dependencies
4. Configure Jupyter kernel
5. Set up directory structure
6. Validate the installation

### Verify Installation

After setup completes, activate the environment and validate:

```bash
# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate.bat  # Windows

# Run validation
python -m imp.utils.env_validator
```

## Manual Setup

If you prefer manual setup or need to customize the installation:

### 1. Create Virtual Environment

```bash
cd py
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate.bat  # Windows
```

### 2. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

Install the package with all optional dependencies:

```bash
# Core + development dependencies
pip install -e ".[dev]"

# Add optimization tools
pip install -e ".[optimization]"

# Add research environment (Jupyter, visualization, etc.)
pip install -e ".[research]"

# Or install everything at once
pip install -e ".[dev,optimization,research]"
```

### 4. Configure Jupyter Kernel

```bash
python -m ipykernel install --user --name=imp-research --display-name="IMP Research Environment"
```

### 5. Install Jupyter Extensions (Optional)

```bash
# Install widget extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Build JupyterLab
jupyter lab build
```

### 6. Create Directory Structure

```bash
mkdir -p notebooks/utils
mkdir -p notebooks/processed_data
mkdir -p notebooks/model_comparison_results
mkdir -p notebooks/regime_analysis_results
mkdir -p processed_data
mkdir -p temp_configs
```

## Environment Validation

### Running Validation

The environment validator checks all dependencies and configurations:

```bash
python -m imp.utils.env_validator
```

### Validation Checks

The validator verifies:

- ✅ Python version (>= 3.9)
- ✅ Core dependencies (numpy, pandas, polars, scikit-learn)
- ✅ HMM libraries (hmmlearn, pomegranate)
- ✅ Jupyter environment (jupyter, ipykernel, ipywidgets)
- ✅ Visualization libraries (matplotlib, seaborn, plotly)
- ✅ Testing framework (pytest)
- ✅ Jupyter kernel configuration
- ✅ Directory structure

### Interpreting Results

- **✓ Green checkmarks**: Component is properly installed
- **✗ Red X marks**: Required component is missing
- **! Yellow warnings**: Optional component or configuration issue

## Jupyter Configuration

### Starting Jupyter Lab

```bash
# From the py/ directory
jupyter lab
```

This will open JupyterLab in your default browser at `http://localhost:8888`

### Selecting the Kernel

When opening a notebook:

1. Click on the kernel name in the top-right corner
2. Select "IMP Research Environment" from the dropdown
3. The kernel should now be active

### Jupyter Lab Extensions

Recommended extensions for enhanced functionality:

```bash
# Widget manager (for ipywidgets)
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Table of contents
jupyter labextension install @jupyterlab/toc

# Git integration
jupyter labextension install @jupyterlab/git
```

### Jupyter Configuration File

Create `~/.jupyter/jupyter_lab_config.py` for custom settings:

```python
# Allow connections from any IP (for remote access)
c.ServerApp.ip = '0.0.0.0'

# Disable token authentication (use with caution)
# c.ServerApp.token = ''

# Set default directory
c.ServerApp.root_dir = '/path/to/your/project/notebooks'

# Enable autosave
c.ContentsManager.autosave_interval = 60  # seconds
```

## Development Workflow

### Typical Research Workflow

1. **Data Exploration**
   ```bash
   jupyter lab notebooks/01_data_exploration.ipynb
   ```

2. **Model Training**
   ```bash
   jupyter lab notebooks/02_hmm_training_comparison.ipynb
   ```

3. **Regime Analysis**
   ```bash
   jupyter lab notebooks/03_regime_analysis.ipynb
   ```

4. **Parameter Optimization**
   ```bash
   jupyter lab notebooks/04_parameter_optimization.ipynb
   ```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m research
pytest tests/ -m notebook
pytest tests/ -m integration

# Run with coverage
pytest tests/ --cov=imp --cov-report=html

# Run specific test file
pytest tests/test_research_framework.py -v
```

### Code Quality Checks

```bash
# Format code with black
black imp/ tests/

# Sort imports
isort imp/ tests/

# Lint with flake8
flake8 imp/ tests/

# Type checking with mypy
mypy imp/
```

### Pre-commit Hooks

Set up pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'imp'`

**Solution**:
```bash
# Ensure you're in the py/ directory
cd py

# Reinstall in editable mode
pip install -e .
```

#### 2. Jupyter Kernel Not Found

**Problem**: "IMP Research Environment" kernel not available

**Solution**:
```bash
# Reinstall kernel
python -m ipykernel install --user --name=imp-research --display-name="IMP Research Environment"

# List available kernels
jupyter kernelspec list

# Remove old kernel if needed
jupyter kernelspec remove imp-research
```

#### 3. Widget Display Issues

**Problem**: ipywidgets not displaying in Jupyter

**Solution**:
```bash
# Reinstall ipywidgets
pip install --upgrade ipywidgets

# Enable extension
jupyter nbextension enable --py widgetsnbextension

# For JupyterLab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab build
```

#### 4. Pomegranate Installation Issues

**Problem**: pomegranate fails to install

**Solution**:
```bash
# Install build dependencies first
pip install numpy cython

# Then install pomegranate
pip install pomegranate

# Or use conda
conda install -c conda-forge pomegranate
```

#### 5. Memory Issues with Large Datasets

**Problem**: Kernel crashes with large datasets

**Solution**:
- Increase Jupyter memory limit in config
- Use data sampling for initial exploration
- Process data in chunks with polars
- Use memory profiling to identify issues:
  ```python
  from memory_profiler import profile
  
  @profile
  def your_function():
      # Your code here
      pass
  ```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/your-org/IMP/issues)
2. Review the [API Documentation](../README.md)
3. Run environment validation: `python -m imp.utils.env_validator`
4. Check Jupyter logs: `jupyter lab --debug`

## Best Practices

### Environment Management

1. **Always use virtual environments** - Never install packages globally
2. **Keep dependencies updated** - Regularly update packages for security and features
3. **Document custom configurations** - Track any manual changes to the environment
4. **Use requirements.txt for reproducibility** - Generate with `pip freeze > requirements.txt`

### Notebook Development

1. **Restart kernel regularly** - Avoid stale state issues
2. **Clear outputs before committing** - Keep git diffs clean
3. **Use cell magic for timing** - `%%time` or `%%timeit` for performance analysis
4. **Document assumptions** - Add markdown cells explaining your approach
5. **Version control notebooks** - Use git for tracking changes

### Code Organization

1. **Keep notebooks focused** - One notebook per analysis task
2. **Extract reusable code** - Move common functions to `notebooks/utils/`
3. **Use relative imports** - Import from `imp` package for production code
4. **Test notebook code** - Extract and test critical functions

### Performance Optimization

1. **Profile before optimizing** - Use `line_profiler` and `memory_profiler`
2. **Use vectorized operations** - Leverage numpy/pandas for speed
3. **Consider polars for large data** - Faster than pandas for many operations
4. **Cache expensive computations** - Use `joblib.Memory` or manual caching

### Data Management

1. **Use version control for data** - Track data versions with DVC or similar
2. **Document data sources** - Keep README with data provenance
3. **Validate data quality** - Use `imp.data.validator` before training
4. **Use appropriate formats** - Parquet for large datasets, JSON for metadata

### Collaboration

1. **Clear notebook outputs** - Before committing to git
2. **Document environment** - Keep `requirements.txt` updated
3. **Use consistent naming** - Follow project conventions
4. **Review before merging** - Have peers review notebooks and code

### Security

1. **Never commit credentials** - Use environment variables or config files
2. **Sanitize outputs** - Remove sensitive data from notebook outputs
3. **Use .gitignore** - Exclude data files and credentials
4. **Review dependencies** - Check for security vulnerabilities regularly

## Additional Resources

### Documentation

- [HMM Training Guide](../imp/hmm/README.md)
- [Data Integration Guide](../imp/data/README.md)
- [Visualization Examples](../notebooks/README.md)
- [Testing Guide](../tests/RESEARCH_TESTING_GUIDE.md)

### Example Notebooks

- `notebooks/01_data_exploration.ipynb` - Data loading and exploration
- `notebooks/02_hmm_training_comparison.ipynb` - Compare HMM libraries
- `notebooks/03_regime_analysis.ipynb` - Regime detection and analysis
- `notebooks/04_parameter_optimization.ipynb` - Hyperparameter tuning

### External Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)
- [pomegranate Documentation](https://pomegranate.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

## Next Steps

After completing the setup:

1. ✅ Run environment validation
2. ✅ Start Jupyter Lab
3. ✅ Open example notebooks
4. ✅ Run test suite
5. ✅ Begin your research!

For questions or issues, please refer to the troubleshooting section or open an issue on GitHub.
