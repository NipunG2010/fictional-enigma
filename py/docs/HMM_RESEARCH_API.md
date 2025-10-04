# HMM Research Environment API Documentation

## Overview

The HMM Research Environment provides a comprehensive suite of tools for Hidden Markov Model experimentation, regime detection, and production deployment. This document provides detailed API documentation for all research environment components.

## Table of Contents

1. [HMM Training](#hmm-training)
2. [Regime Analysis](#regime-analysis)
3. [Visualization](#visualization)
4. [Model Evaluation](#model-evaluation)
5. [Data Integration](#data-integration)
6. [Artifact Management](#artifact-management)
7. [Parameter Tuning](#parameter-tuning)

---

## HMM Training

### EnhancedHMMTrainer

Enhanced HMM trainer supporting multiple libraries (hmmlearn and pomegranate).

**Location:** `py/imp/hmm/trainer.py`

#### Constructor

```python
EnhancedHMMTrainer(
    n_states: int = 3,
    library: str = "hmmlearn",
    covariance_type: str = "full",
    random_state: Optional[int] = None
)
```

**Parameters:**
- `n_states` (int): Number of hidden states (default: 3)
- `library` (str): HMM library to use - "hmmlearn" or "pomegranate" (default: "hmmlearn")
- `covariance_type` (str): Type of covariance - "full", "diag", or "spherical" (default: "full")
- `random_state` (int, optional): Random seed for reproducibility

**Example:**

```python
from imp.hmm.trainer import EnhancedHMMTrainer

# Create trainer with 3 states using hmmlearn
trainer = EnhancedHMMTrainer(
    n_states=3,
    library="hmmlearn",
    covariance_type="full",
    random_state=42
)
```

#### Methods

##### train()

Train HMM model on observations.

```python
train(
    observations: np.ndarray,
    n_iterations: int = 100,
    convergence_threshold: float = 1e-4
) -> HMMArtifact
```

**Parameters:**
- `observations` (np.ndarray): Training data of shape (n_samples, n_features)
- `n_iterations` (int): Maximum training iterations (default: 100)
- `convergence_threshold` (float): Convergence threshold (default: 1e-4)

**Returns:**
- `HMMArtifact`: Trained model artifact

**Example:**

```python
import numpy as np

# Generate sample data
observations = np.random.randn(1000, 3)

# Train model
artifact = trainer.train(
    observations,
    n_iterations=100,
    convergence_threshold=1e-4
)
```

##### train_with_validation()

Train HMM with validation split.

```python
train_with_validation(
    observations: np.ndarray,
    validation_split: float = 0.2,
    **kwargs
) -> Tuple[HMMArtifact, Dict[str, float]]
```

**Parameters:**
- `observations` (np.ndarray): Training data
- `validation_split` (float): Fraction of data for validation (default: 0.2)
- `**kwargs`: Additional arguments passed to train()

**Returns:**
- Tuple of (HMMArtifact, validation metrics dict)

**Example:**

```python
# Train with validation
artifact, metrics = trainer.train_with_validation(
    observations,
    validation_split=0.2,
    n_iterations=100
)

print(f"Validation log-likelihood: {metrics['log_likelihood']:.4f}")
print(f"AIC: {metrics['aic']:.4f}")
print(f"BIC: {metrics['bic']:.4f}")
```

---

## Regime Analysis

### RegimeAnalyzer

Comprehensive regime analysis and characterization tools.

**Location:** `py/imp/hmm/regime_analysis.py`

#### Constructor

```python
RegimeAnalyzer(artifact: HMMArtifact)
```

**Parameters:**
- `artifact` (HMMArtifact): Trained HMM artifact

#### Methods

##### analyze_regimes()

Perform comprehensive regime analysis.

```python
analyze_regimes(
    observations: np.ndarray,
    state_probs: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `observations` (np.ndarray): Original observations
- `state_probs` (np.ndarray): State probabilities from inference
- `feature_names` (List[str], optional): Names of features

**Returns:**
- Dictionary containing regime statistics, transitions, and characterization

**Example:**

```python
from imp.hmm.regime_analysis import RegimeAnalyzer
from imp.hmm.inference import HMMInference

# Perform inference
inference = HMMInference(artifact)
state_probs = inference.predict_proba(observations)

# Analyze regimes
analyzer = RegimeAnalyzer(artifact)
analysis = analyzer.analyze_regimes(
    observations,
    state_probs,
    feature_names=['s_LDC', 's_MR', 's_TSMOM']
)

# Access results
print(f"State durations: {analysis['state_durations']}")
print(f"Transition frequencies: {analysis['transition_frequencies']}")
```

##### calculate_state_statistics()

Calculate statistical characteristics for each state.

```python
calculate_state_statistics(
    observations: np.ndarray,
    state_sequence: np.ndarray
) -> Dict[int, Dict[str, float]]
```

**Returns:**
- Dictionary mapping state index to statistics (mean, std, volatility, etc.)

##### get_regime_interpretation()

Get economic interpretation of detected regimes.

```python
get_regime_interpretation(
    state_stats: Dict[int, Dict[str, float]]
) -> Dict[int, str]
```

**Returns:**
- Dictionary mapping state index to interpretation string

---

## Visualization

### RegimeVisualizer

Advanced visualization tools for regime analysis.

**Location:** `py/imp/visualization/regime_visualizer.py`

#### Constructor

```python
RegimeVisualizer(artifact: HMMArtifact)
```

#### Methods

##### plot_state_probabilities()

Plot state probabilities over time.

```python
plot_state_probabilities(
    state_probs: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    interactive: bool = True,
    title: str = "State Probabilities Over Time"
) -> Union[plt.Figure, go.Figure]
```

**Parameters:**
- `state_probs` (np.ndarray): State probabilities of shape (n_samples, n_states)
- `timestamps` (np.ndarray, optional): Time indices or timestamps
- `interactive` (bool): Use plotly for interactive plot (default: True)
- `title` (str): Plot title

**Returns:**
- matplotlib Figure or plotly Figure

**Example:**

```python
from imp.visualization.regime_visualizer import RegimeVisualizer

visualizer = RegimeVisualizer(artifact)

# Create interactive plot
fig = visualizer.plot_state_probabilities(
    state_probs,
    timestamps=timestamps,
    interactive=True
)
fig.show()
```

##### plot_transition_matrix()

Visualize transition matrix as heatmap.

```python
plot_transition_matrix(
    annotate: bool = True,
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure
```

**Example:**

```python
fig = visualizer.plot_transition_matrix(
    annotate=True,
    cmap="Blues"
)
plt.show()
```

##### create_regime_dashboard()

Create comprehensive interactive dashboard.

```python
create_regime_dashboard(
    observations: np.ndarray,
    state_probs: np.ndarray,
    timestamps: Optional[np.ndarray] = None
) -> widgets.VBox
```

**Returns:**
- IPython widget container with interactive dashboard

---

## Model Evaluation

### HMMEvaluator

Comprehensive model evaluation and comparison framework.

**Location:** `py/imp/evaluation/evaluator.py`

#### Methods

##### cross_validate()

Perform time series cross-validation.

```python
cross_validate(
    observations: np.ndarray,
    trainer_config: Dict[str, Any],
    cv_folds: int = 5,
    gap: int = 0
) -> Dict[str, Any]
```

**Parameters:**
- `observations` (np.ndarray): Time series data
- `trainer_config` (dict): Configuration for EnhancedHMMTrainer
- `cv_folds` (int): Number of cross-validation folds (default: 5)
- `gap` (int): Gap between train and validation sets (default: 0)

**Returns:**
- Dictionary with cross-validation results and metrics

**Example:**

```python
from imp.evaluation.evaluator import HMMEvaluator

evaluator = HMMEvaluator()

config = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'full'
}

results = evaluator.cross_validate(
    observations,
    trainer_config=config,
    cv_folds=5
)

print(f"Mean CV score: {results['mean_score']:.4f}")
print(f"Std CV score: {results['std_score']:.4f}")
```

##### compare_models()

Compare multiple model configurations.

```python
compare_models(
    observations: np.ndarray,
    configs: List[Dict[str, Any]],
    metrics: List[str] = ['log_likelihood', 'aic', 'bic']
) -> pd.DataFrame
```

**Returns:**
- DataFrame with comparison results

---

## Data Integration

### LDCDataLoader

Load and preprocess LDC signal data.

**Location:** `py/imp/data/ldc_loader.py`

#### Methods

##### load_signals()

Load LDC signals from parquet file.

```python
load_signals(
    file_path: Union[str, Path],
    signals: Optional[List[str]] = None
) -> pd.DataFrame
```

**Parameters:**
- `file_path` (str or Path): Path to parquet file
- `signals` (List[str], optional): Specific signals to load (default: all)

**Returns:**
- DataFrame with signal data

**Example:**

```python
from imp.data.ldc_loader import LDCDataLoader

loader = LDCDataLoader()

# Load all signals
df = loader.load_signals('processed_data/signals_processed.parquet')

# Load specific signals
df = loader.load_signals(
    'processed_data/signals_processed.parquet',
    signals=['s_LDC', 's_MR', 's_TSMOM']
)
```

### SignalPreprocessor

Preprocess signals for HMM training.

**Location:** `py/imp/data/preprocessor.py`

#### Methods

##### preprocess()

Comprehensive signal preprocessing.

```python
preprocess(
    data: pd.DataFrame,
    handle_missing: str = 'forward_fill',
    handle_outliers: bool = True,
    normalize: bool = True,
    outlier_threshold: float = 3.0
) -> Tuple[np.ndarray, Dict[str, Any]]
```

**Parameters:**
- `data` (pd.DataFrame): Raw signal data
- `handle_missing` (str): Method for missing values - 'forward_fill', 'interpolate', 'drop'
- `handle_outliers` (bool): Whether to handle outliers (default: True)
- `normalize` (bool): Whether to normalize data (default: True)
- `outlier_threshold` (float): Z-score threshold for outliers (default: 3.0)

**Returns:**
- Tuple of (preprocessed array, preprocessing metadata)

---

## Artifact Management

### ArtifactManager

Manage HMM artifacts with versioning and validation.

**Location:** `py/imp/hmm/artifact_management.py`

#### Methods

##### save_artifact()

Save artifact with versioning.

```python
save_artifact(
    artifact: HMMArtifact,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None
) -> Path
```

**Parameters:**
- `artifact` (HMMArtifact): Artifact to save
- `name` (str): Artifact name
- `metadata` (dict, optional): Additional metadata
- `version` (str, optional): Version string (auto-generated if None)

**Returns:**
- Path to saved artifact

**Example:**

```python
from imp.hmm.artifact_management import ArtifactManager

manager = ArtifactManager(artifacts_dir='artifacts/')

# Save artifact
path = manager.save_artifact(
    artifact,
    name='market_regime_detector',
    metadata={
        'training_date': '2025-01-15',
        'data_source': 'BTCUSDT_5m',
        'performance': {'aic': 1234.5, 'bic': 1250.3}
    }
)

print(f"Artifact saved to: {path}")
```

##### load_artifact()

Load artifact by name and version.

```python
load_artifact(
    name: str,
    version: Optional[str] = None
) -> Tuple[HMMArtifact, Dict[str, Any]]
```

**Returns:**
- Tuple of (artifact, metadata)

##### validate_artifact()

Validate artifact for production deployment.

```python
validate_artifact(
    artifact: HMMArtifact,
    validation_data: Optional[np.ndarray] = None
) -> Dict[str, Any]
```

**Returns:**
- Validation report dictionary

---

## Parameter Tuning

### HMMParameterTuner

Interactive parameter tuning for Jupyter notebooks.

**Location:** `py/imp/tuning/parameter_tuner.py`

#### Methods

##### create_tuning_interface()

Create interactive tuning interface.

```python
create_tuning_interface() -> widgets.VBox
```

**Returns:**
- IPython widget container

**Example:**

```python
from imp.tuning.parameter_tuner import HMMParameterTuner

tuner = HMMParameterTuner(observations)
interface = tuner.create_tuning_interface()
display(interface)
```

##### optimize_parameters()

Automated parameter optimization.

```python
optimize_parameters(
    param_grid: Dict[str, List[Any]],
    optimization_method: str = 'grid_search',
    cv_folds: int = 5
) -> Dict[str, Any]
```

**Parameters:**
- `param_grid` (dict): Parameter grid to search
- `optimization_method` (str): 'grid_search' or 'bayesian'
- `cv_folds` (int): Cross-validation folds

**Returns:**
- Dictionary with best parameters and results

---

## Error Handling

All components raise specific exceptions for better error handling:

- `HMMResearchError`: Base exception for research environment
- `ModelTrainingError`: Errors during model training
- `VisualizationError`: Errors during visualization
- `DataIntegrationError`: Errors loading/preprocessing data
- `ArtifactValidationError`: Errors validating artifacts

**Example:**

```python
from imp.hmm.trainer import EnhancedHMMTrainer, ModelTrainingError

try:
    trainer = EnhancedHMMTrainer(n_states=3)
    artifact = trainer.train(observations)
except ModelTrainingError as e:
    print(f"Training failed: {e}")
    # Handle error appropriately
```

---

## Best Practices

### 1. Data Preprocessing

Always preprocess data before training:

```python
from imp.data.preprocessor import SignalPreprocessor

preprocessor = SignalPreprocessor()
processed_data, metadata = preprocessor.preprocess(
    raw_data,
    handle_missing='forward_fill',
    normalize=True
)
```

### 2. Model Validation

Use cross-validation for robust evaluation:

```python
from imp.evaluation.evaluator import HMMEvaluator

evaluator = HMMEvaluator()
results = evaluator.cross_validate(observations, config, cv_folds=5)
```

### 3. Artifact Management

Always save artifacts with metadata:

```python
manager.save_artifact(
    artifact,
    name='production_model',
    metadata={
        'training_samples': len(observations),
        'validation_score': metrics['log_likelihood'],
        'created_by': 'researcher_name'
    }
)
```

### 4. Visualization

Use interactive visualizations for exploration:

```python
visualizer = RegimeVisualizer(artifact)
fig = visualizer.plot_state_probabilities(state_probs, interactive=True)
```

---

## See Also

- [Tutorial Notebooks](../notebooks/README.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Best Practices](BEST_PRACTICES.md)
- [Integration Examples](INTEGRATION_EXAMPLES.md)
