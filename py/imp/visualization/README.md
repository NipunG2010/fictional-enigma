# HMM Visualization Module

This module provides advanced visualization tools for Hidden Markov Model (HMM) regime analysis, specifically designed for market regime detection and analysis.

## Features

- **State Probability Visualization**: Both static (matplotlib) and interactive (plotly) plots
- **Transition Matrix Heatmaps**: Customizable visualization of state transition probabilities
- **Regime Statistics**: Comprehensive calculation and formatting of regime persistence metrics
- **Interactive Dashboards**: Jupyter notebook widgets for interactive regime analysis
- **Model Comparison**: Side-by-side comparison of different HMM models
- **Market Regime Analysis**: Specialized tools for financial market regime detection

## Classes

### RegimeVisualizer

The main class for HMM regime visualization and analysis.

#### Initialization

```python
from imp.hmm.models import HMMArtifact
from imp.visualization import RegimeVisualizer

# Initialize with an HMM artifact
visualizer = RegimeVisualizer(artifact)
```

#### Key Methods

##### plot_state_probabilities()

Plot state probabilities over time with both static and interactive options.

```python
# Static plot
fig = visualizer.plot_state_probabilities(
    state_probs, timestamps, interactive=False
)

# Interactive plot (requires plotly)
fig = visualizer.plot_state_probabilities(
    state_probs, timestamps, interactive=True
)
```

##### plot_transition_matrix()

Visualize the HMM transition matrix as a heatmap.

```python
fig = visualizer.plot_transition_matrix(
    annotate=True, cmap="Blues"
)
```

##### calculate_regime_statistics()

Calculate comprehensive regime statistics including persistence metrics.

```python
stats = visualizer.calculate_regime_statistics(
    observations, state_probs, timestamps
)
```

##### create_regime_dashboard()

Create an interactive dashboard for Jupyter notebooks (requires ipywidgets).

```python
dashboard = visualizer.create_regime_dashboard(
    observations, state_probs, timestamps
)
```

##### plot_regime_comparison()

Compare regime detection across multiple models.

```python
fig = visualizer.plot_regime_comparison(
    observations,
    [state_probs_1, state_probs_2],
    ['Model 1', 'Model 2'],
    timestamps
)
```

## Dependencies

### Required
- numpy
- matplotlib
- seaborn
- pandas

### Optional
- plotly (for interactive plots)
- ipywidgets (for Jupyter dashboards)

## Usage Examples

### Basic Visualization

```python
import numpy as np
from imp.hmm.models import HMMArtifact
from imp.visualization import RegimeVisualizer

# Create or load HMM artifact
artifact = HMMArtifact(...)

# Initialize visualizer
visualizer = RegimeVisualizer(artifact)

# Generate or load state probabilities
state_probs = np.random.dirichlet([1, 1, 1], size=100)
observations = np.random.randn(100, 2)

# Create visualizations
fig1 = visualizer.plot_state_probabilities(state_probs)
fig2 = visualizer.plot_transition_matrix()

# Calculate statistics
stats = visualizer.calculate_regime_statistics(observations, state_probs)
print(f"State frequencies: {[stats['state_statistics'][f'state_{i}']['frequency'] for i in range(3)]}")
```

### Jupyter Notebook Usage

```python
# In a Jupyter notebook
from imp.visualization import RegimeVisualizer

visualizer = RegimeVisualizer(artifact)

# Create interactive dashboard
dashboard = visualizer.create_regime_dashboard(
    observations, state_probs, timestamps,
    title="Market Regime Analysis"
)

# Display dashboard
display(dashboard)
```

### Model Comparison

```python
# Compare different HMM models
models_probs = [state_probs_hmmlearn, state_probs_pomegranate]
model_names = ['HMMLearn', 'Pomegranate']

fig = visualizer.plot_regime_comparison(
    market_signals,
    models_probs,
    model_names,
    timestamps
)
```

## Output Statistics

The `calculate_regime_statistics()` method returns a comprehensive dictionary containing:

- **n_states**: Number of HMM states
- **total_observations**: Total number of observations
- **state_statistics**: Per-state frequency and probability statistics
- **transition_statistics**: Empirical vs theoretical transition matrices
- **regime_persistence**: Duration and episode statistics for each state

### Example Statistics Output

```python
{
    'n_states': 3,
    'total_observations': 1000,
    'state_statistics': {
        'state_0': {
            'frequency': 0.35,
            'mean_probability': 0.82,
            'observation_count': 350
        },
        # ... other states
    },
    'regime_persistence': {
        'state_0': {
            'mean_duration': 5.2,
            'median_duration': 4.0,
            'total_episodes': 67
        },
        # ... other states
    }
}
```

## Error Handling

The module includes comprehensive error handling for:

- Invalid state probability dimensions
- Mismatched observation and probability lengths
- Missing optional dependencies (plotly, ipywidgets)
- Timestamp conversion errors
- Empty or invalid data

## Integration with HMM Research Environment

This visualization module is designed to work seamlessly with:

- **HMM Training Framework**: Visualize results from enhanced HMM trainers
- **Jupyter Research Environment**: Interactive analysis in notebooks
- **LDC Signal Integration**: Visualize regime detection on trading signals
- **Production Artifacts**: Validate and analyze production HMM models

## Performance Considerations

- Large datasets (>10,000 points) may benefit from data sampling for interactive plots
- Static plots are generally faster for large time series
- Dashboard widgets may consume significant memory for very long time series
- Consider using data aggregation for very high-frequency data

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/test_visualization.py -v
```

## Examples

See `examples/visualization_demo.py` for a comprehensive demonstration of all features.