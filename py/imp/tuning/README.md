# HMM Parameter Tuning Framework

Interactive and automated parameter tuning framework for Hidden Markov Models.

## Features

### 🎛️ Interactive Parameter Tuning
- Widget-based interface for Jupyter notebooks
- Real-time model training and evaluation
- Interactive sliders and dropdowns for all parameters
- Progress indicators and status updates
- Comprehensive metrics display
- Automatic visualization generation

### 🔍 Grid Search Optimization
- Exhaustive search over parameter combinations
- Configurable parameter grids
- Parallel evaluation support
- Detailed results tracking
- Quick search utilities

### 🎯 Bayesian Optimization
- Efficient parameter search using Gaussian Processes
- Automatic exploration-exploitation balance
- Fewer evaluations needed than grid search
- Continuous and categorical parameter support
- Requires `scikit-optimize` package

### 💾 Configuration Management
- Save and load parameter configurations
- Export results to JSON
- Reproducible experiments
- Version tracking
- Metadata preservation

### 📊 Results Comparison
- Compare multiple experiments
- Visual comparison plots
- Statistical summaries
- Best model selection
- Performance ranking

## Installation

### Basic Installation
```bash
pip install -e .
```

### With Bayesian Optimization Support
```bash
pip install -e ".[optimization]"
```

This installs `scikit-optimize` for Bayesian optimization capabilities.

## Quick Start

### Interactive Tuning

```python
import numpy as np
from imp.tuning import HMMParameterTuner

# Load your data
observations = np.load('your_data.npy')

# Create tuner
tuner = HMMParameterTuner(observations)

# Display interactive interface
tuner.create_tuning_interface()
```

### Grid Search

```python
from imp.tuning.optimization import quick_grid_search

# Quick grid search with default parameters
result = quick_grid_search(
    observations,
    n_states_range=[2, 3, 4, 5],
    covariance_types=['full', 'diag'],
    verbose=True
)

print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score}")
```

### Bayesian Optimization

```python
from imp.tuning.optimization import BayesianOptimizer

# Define parameter space
param_space = {
    'n_states': {'type': 'integer', 'low': 2, 'high': 10},
    'library': {'type': 'categorical', 'categories': ['hmmlearn']},
    'covariance_type': {'type': 'categorical', 'categories': ['full', 'diag', 'spherical']}
}

# Run optimization
optimizer = BayesianOptimizer(
    observations=observations,
    param_space=param_space,
    n_calls=20,
    verbose=True
)

result = optimizer.fit()
```

## Usage Examples

### 1. Interactive Widget Interface

The interactive interface provides real-time parameter tuning:

```python
from imp.tuning import HMMParameterTuner
from pathlib import Path

# Create tuner with custom config directory
tuner = HMMParameterTuner(
    observations=your_data,
    config_dir=Path('./my_configs')
)

# Display interface
interface = tuner.create_tuning_interface()
display(interface)

# After training models, get the best one
best_result = tuner.get_best_result(
    metric='log_likelihood',
    higher_is_better=True
)

# Export all results
tuner.export_results(Path('./results.json'))
```

### 2. Programmatic Configuration

Train models programmatically without widgets:

```python
from imp.tuning import TuningConfig
from imp.hmm.trainer import EnhancedHMMTrainer

# Define configuration
config = TuningConfig(
    n_states=3,
    library='hmmlearn',
    covariance_type='full',
    n_iterations=100,
    validation_split=0.2,
    random_state=42
)

# Train model
trainer = EnhancedHMMTrainer(
    n_states=config.n_states,
    library=config.library,
    covariance_type=config.covariance_type,
    random_state=config.random_state
)

artifact, metrics = trainer.train_with_validation(
    observations,
    validation_split=config.validation_split,
    n_iterations=config.n_iterations
)
```

### 3. Custom Grid Search

Define custom parameter grids:

```python
from imp.tuning.optimization import GridSearchOptimizer

# Define parameter grid
param_grid = {
    'n_states': [2, 3, 4, 5, 6],
    'library': ['hmmlearn'],
    'covariance_type': ['full', 'diag', 'spherical'],
    'random_state': [42]
}

# Create optimizer
optimizer = GridSearchOptimizer(
    observations=observations,
    param_grid=param_grid,
    scoring_metric='log_likelihood',
    higher_is_better=True,
    validation_split=0.2,
    n_iterations=100,
    verbose=True
)

# Run optimization
result = optimizer.fit()

# Access results
print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
print(f"Total time: {result.optimization_time:.2f}s")

# Save results
result.save(Path('./grid_search_results.json'))
```

### 4. Advanced Bayesian Optimization

Fine-tune Bayesian optimization:

```python
from imp.tuning.optimization import BayesianOptimizer

# Define search space
param_space = {
    'n_states': {
        'type': 'integer',
        'low': 2,
        'high': 10
    },
    'library': {
        'type': 'categorical',
        'categories': ['hmmlearn', 'pomegranate']
    },
    'covariance_type': {
        'type': 'categorical',
        'categories': ['full', 'diag', 'spherical']
    }
}

# Create optimizer
optimizer = BayesianOptimizer(
    observations=observations,
    param_space=param_space,
    scoring_metric='bic',  # Use BIC instead of log-likelihood
    higher_is_better=False,  # Lower BIC is better
    validation_split=0.2,
    n_iterations=100,
    n_calls=30,  # Number of optimization iterations
    random_state=42,
    verbose=True
)

# Run optimization
result = optimizer.fit()

# Analyze results
for i, res in enumerate(result.all_results):
    if res['score'] is not None:
        print(f"Iteration {i+1}: {res['params']} -> {res['score']:.4f}")
```

## Configuration Options

### TuningConfig Parameters

- `n_states` (int): Number of hidden states (default: 3)
- `library` (str): HMM library to use ('hmmlearn' or 'pomegranate', default: 'hmmlearn')
- `covariance_type` (str): Covariance matrix type ('full', 'diag', 'spherical', default: 'full')
- `n_iterations` (int): Number of EM iterations (default: 100)
- `validation_split` (float): Fraction of data for validation (default: 0.2)
- `random_state` (int): Random seed for reproducibility (default: 42)

### Scoring Metrics

Available metrics for optimization:
- `log_likelihood`: Model log-likelihood (higher is better)
- `aic`: Akaike Information Criterion (lower is better)
- `bic`: Bayesian Information Criterion (lower is better)
- `perplexity`: Model perplexity (lower is better)

## Best Practices

### 1. Data Preparation
```python
# Ensure data is properly scaled
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
observations_scaled = scaler.fit_transform(observations)
```

### 2. Start Simple
```python
# Begin with a small parameter grid
param_grid = {
    'n_states': [2, 3],
    'covariance_type': ['full']
}
```

### 3. Use Validation Split
```python
# Always use validation for model selection
config = TuningConfig(
    validation_split=0.2,  # 20% for validation
    n_iterations=100
)
```

### 4. Save Configurations
```python
# Save successful configurations
tuner.save_button.click()  # In interactive mode

# Or programmatically
result.save(Path('./best_config.json'))
```

### 5. Compare Multiple Models
```python
# Train multiple configurations
configs = [
    TuningConfig(n_states=2),
    TuningConfig(n_states=3),
    TuningConfig(n_states=4)
]

results = []
for config in configs:
    # Train and evaluate
    # Store results
    pass

# Compare
tuner.compare_button.click()  # In interactive mode
```

## Troubleshooting

### Issue: Training Fails
**Solution**: Check data quality, reduce model complexity, or increase iterations.

```python
# Validate data
assert not np.any(np.isnan(observations))
assert not np.any(np.isinf(observations))

# Start with fewer states
config = TuningConfig(n_states=2)

# Increase iterations
config = TuningConfig(n_iterations=200)
```

### Issue: Slow Optimization
**Solution**: Use Bayesian optimization or reduce parameter grid.

```python
# Use Bayesian optimization instead of grid search
from imp.tuning.optimization import quick_bayesian_search

result = quick_bayesian_search(observations, n_calls=20)
```

### Issue: Models Don't Converge
**Solution**: Adjust initialization or increase iterations.

```python
config = TuningConfig(
    n_iterations=500,  # More iterations
    random_state=123   # Try different seed
)
```

## API Reference

### HMMParameterTuner

Main class for interactive parameter tuning.

**Methods:**
- `create_tuning_interface()`: Create widget interface
- `get_best_result(metric, higher_is_better)`: Get best result
- `export_results(filepath)`: Export all results

### GridSearchOptimizer

Grid search optimization.

**Methods:**
- `fit()`: Run grid search
- `_evaluate_params(params)`: Evaluate single configuration

### BayesianOptimizer

Bayesian optimization using Gaussian Processes.

**Methods:**
- `fit()`: Run Bayesian optimization
- `_create_search_space()`: Create search space
- `_evaluate_params(params)`: Evaluate single configuration

## Examples

See the `notebooks/05_parameter_tuning_demo.ipynb` notebook for comprehensive examples.

## Testing

Run tests:
```bash
pytest tests/test_parameter_tuning.py -v
```

## Dependencies

### Required
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- hmmlearn
- ipywidgets
- jupyter

### Optional
- scikit-optimize (for Bayesian optimization)
- pomegranate (for alternative HMM implementation)

## License

MIT License
