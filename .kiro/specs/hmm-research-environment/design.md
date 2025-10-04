# HMM Research Environment Design Document

## Overview

The HMM Research Environment design builds upon the existing HMM implementation in py/imp/hmm/ to create a comprehensive Jupyter-based research platform. The design integrates hmmlearn and pomegranate libraries, provides advanced visualization tools for regime analysis, and ensures seamless integration with the LDC trading system for production deployment.

## Architecture

### Current Foundation Analysis

The existing implementation provides a solid base:

```python
# Existing Structure (py/imp/hmm/)
├── models.py          # HMMArtifact, FusionWeights, HMMPrediction
├── trainer.py         # HMMTrainer with hmmlearn integration  
├── inference.py       # HMM inference capabilities
└── __init__.py        # Module exports
```

**Current Capabilities:**
- HMMArtifact with comprehensive validation
- HMMTrainer using hmmlearn with Gaussian HMM
- Pydantic models for type safety and validation
- Basic training and artifact management

### Enhanced Research Environment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HMM Research Environment                     │
├─────────────────────────────────────────────────────────────────┤
│  Jupyter Notebooks              │  Enhanced HMM Components      │
├─────────────────────────────────┼─────────────────────────────────┤
│  • Interactive Experimentation  │  • Multi-Library Support       │
│  • Parameter Tuning Widgets     │  • Advanced Visualization      │
│  • Model Comparison Tools       │  • Regime Analysis Tools       │
│  • Visualization Dashboards     │  • Performance Evaluation      │
│  • Research Documentation       │  • Production Integration      │
└─────────────────────────────────┴─────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Integration Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  LDC Signal Data ←→ HMM Training ←→ Artifact Generation         │
│  Rust Engine ←→ Python Research ←→ Production Deployment       │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Enhanced HMM Training Framework

**Multi-Library HMM Trainer Interface:**
```python
from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import numpy as np
from hmmlearn import hmm as hmmlearn_hmm
try:
    import pomegranate as pom
    POMEGRANATE_AVAILABLE = True
except ImportError:
    POMEGRANATE_AVAILABLE = False

class BaseHMMTrainer(ABC):
    """Abstract base class for HMM trainers."""
    
    @abstractmethod
    def train(self, observations: np.ndarray, **kwargs) -> HMMArtifact:
        pass
    
    @abstractmethod
    def evaluate(self, observations: np.ndarray) -> Dict[str, float]:
        pass

class EnhancedHMMTrainer:
    """Enhanced HMM trainer supporting multiple libraries."""
    
    def __init__(self, 
                 n_states: int = 3,
                 library: str = "hmmlearn",
                 covariance_type: str = "full"):
        self.n_states = n_states
        self.library = library
        self.covariance_type = covariance_type
        self.trainer = self._create_trainer()
    
    def _create_trainer(self) -> BaseHMMTrainer:
        if self.library == "hmmlearn":
            return HMMLearnTrainer(self.n_states, self.covariance_type)
        elif self.library == "pomegranate" and POMEGRANATE_AVAILABLE:
            return PomegranateTrainer(self.n_states)
        else:
            raise ValueError(f"Unsupported library: {self.library}")
    
    def train_with_validation(self, 
                            observations: np.ndarray,
                            validation_split: float = 0.2,
                            **kwargs) -> Tuple[HMMArtifact, Dict[str, float]]:
        """Train HMM with cross-validation."""
        # Split data
        split_idx = int(len(observations) * (1 - validation_split))
        train_data = observations[:split_idx]
        val_data = observations[split_idx:]
        
        # Train model
        artifact = self.trainer.train(train_data, **kwargs)
        
        # Evaluate on validation set
        metrics = self.trainer.evaluate(val_data)
        
        return artifact, metrics
```

### 2. Advanced Visualization Framework

**Regime Analysis Visualization Interface:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

class RegimeVisualizer:
    """Advanced visualization tools for HMM regime analysis."""
    
    def __init__(self, artifact: HMMArtifact):
        self.artifact = artifact
        self.n_states = artifact.n_states
        
    def plot_state_probabilities(self, 
                               state_probs: np.ndarray,
                               timestamps: np.ndarray = None,
                               interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """Plot state probabilities over time."""
        if interactive:
            return self._plot_interactive_states(state_probs, timestamps)
        else:
            return self._plot_static_states(state_probs, timestamps)
    
    def plot_transition_matrix(self, 
                             annotate: bool = True,
                             cmap: str = "Blues") -> plt.Figure:
        """Visualize transition matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        transition_matrix = np.array(self.artifact.transition_matrix)
        
        sns.heatmap(transition_matrix, 
                   annot=annotate,
                   fmt='.3f',
                   cmap=cmap,
                   square=True,
                   ax=ax,
                   cbar_kws={'label': 'Transition Probability'})
        
        ax.set_title('HMM State Transition Matrix')
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        
        return fig
    
    def create_regime_dashboard(self, 
                              observations: np.ndarray,
                              state_probs: np.ndarray,
                              timestamps: np.ndarray = None) -> widgets.VBox:
        """Create interactive dashboard for regime analysis."""
        
        # State probability plot
        state_plot = self.plot_state_probabilities(state_probs, timestamps, interactive=True)
        
        # Transition matrix plot
        transition_plot = self.plot_transition_matrix()
        
        # Regime statistics
        regime_stats = self._calculate_regime_statistics(observations, state_probs)
        
        # Create widgets
        state_selector = widgets.Dropdown(
            options=[(f'State {i}', i) for i in range(self.n_states)],
            value=0,
            description='State:'
        )
        
        def update_display(state_idx):
            # Update visualizations based on selected state
            pass
        
        state_selector.observe(lambda change: update_display(change['new']), names='value')
        
        return widgets.VBox([
            widgets.HTML("<h3>HMM Regime Analysis Dashboard</h3>"),
            state_selector,
            widgets.HTML(f"<h4>Regime Statistics</h4>"),
            widgets.HTML(self._format_regime_stats(regime_stats))
        ])
```

### 3. Jupyter Notebook Framework

**Research Notebook Structure:**
```
notebooks/
├── 01_data_exploration.ipynb          # Data loading and exploration
├── 02_hmm_training_comparison.ipynb   # Compare hmmlearn vs pomegranate
├── 03_regime_analysis.ipynb           # Regime detection and analysis
├── 04_parameter_optimization.ipynb    # Hyperparameter tuning
├── 05_model_evaluation.ipynb          # Cross-validation and metrics
├── 06_visualization_gallery.ipynb     # Visualization examples
├── 07_production_integration.ipynb    # Artifact generation and testing
└── utils/
    ├── notebook_utils.py              # Common utilities
    ├── data_loaders.py                # Data loading functions
    └── plotting_helpers.py            # Plotting utilities
```

**Interactive Parameter Tuning Interface:**
```python
import ipywidgets as widgets
from IPython.display import display, clear_output

class HMMParameterTuner:
    """Interactive parameter tuning for Jupyter notebooks."""
    
    def __init__(self, observations: np.ndarray):
        self.observations = observations
        self.results = {}
        
    def create_tuning_interface(self) -> widgets.VBox:
        """Create interactive parameter tuning interface."""
        
        # Parameter widgets
        n_states_slider = widgets.IntSlider(
            value=3, min=2, max=10, step=1,
            description='States:', style={'description_width': 'initial'}
        )
        
        covariance_dropdown = widgets.Dropdown(
            options=['full', 'diag', 'spherical'],
            value='full',
            description='Covariance:'
        )
        
        library_dropdown = widgets.Dropdown(
            options=['hmmlearn', 'pomegranate'],
            value='hmmlearn',
            description='Library:'
        )
        
        iterations_slider = widgets.IntSlider(
            value=100, min=10, max=1000, step=10,
            description='Iterations:'
        )
        
        train_button = widgets.Button(
            description='Train Model',
            button_style='success'
        )
        
        output_area = widgets.Output()
        
        def on_train_clicked(b):
            with output_area:
                clear_output(wait=True)
                print("Training HMM model...")
                
                # Train model with current parameters
                trainer = EnhancedHMMTrainer(
                    n_states=n_states_slider.value,
                    library=library_dropdown.value,
                    covariance_type=covariance_dropdown.value
                )
                
                artifact, metrics = trainer.train_with_validation(
                    self.observations,
                    n_iterations=iterations_slider.value
                )
                
                # Store results
                config_key = f"{library_dropdown.value}_{n_states_slider.value}_{covariance_dropdown.value}"
                self.results[config_key] = {
                    'artifact': artifact,
                    'metrics': metrics
                }
                
                # Display results
                print(f"Training completed!")
                print(f"Log-likelihood: {metrics.get('log_likelihood', 'N/A'):.4f}")
                print(f"AIC: {metrics.get('aic', 'N/A'):.4f}")
                print(f"BIC: {metrics.get('bic', 'N/A'):.4f}")
                
                # Plot results
                visualizer = RegimeVisualizer(artifact)
                fig = visualizer.plot_transition_matrix()
                plt.show()
        
        train_button.on_click(on_train_clicked)
        
        return widgets.VBox([
            widgets.HTML("<h3>HMM Parameter Tuning</h3>"),
            widgets.HBox([n_states_slider, covariance_dropdown]),
            widgets.HBox([library_dropdown, iterations_slider]),
            train_button,
            output_area
        ])
```

### 4. Model Evaluation Framework

**Comprehensive Evaluation Interface:**
```python
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import pandas as pd

class HMMEvaluator:
    """Comprehensive HMM model evaluation."""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def cross_validate(self, 
                      observations: np.ndarray,
                      trainer_configs: List[Dict],
                      cv_folds: int = 5) -> pd.DataFrame:
        """Perform time series cross-validation."""
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        results = []
        
        for config in trainer_configs:
            config_name = f"{config['library']}_{config['n_states']}_{config['covariance_type']}"
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(observations)):
                train_data = observations[train_idx]
                val_data = observations[val_idx]
                
                # Train model
                trainer = EnhancedHMMTrainer(**config)
                artifact = trainer.train(train_data)
                
                # Evaluate
                metrics = trainer.evaluate(val_data)
                fold_scores.append(metrics['log_likelihood'])
                
                results.append({
                    'config': config_name,
                    'fold': fold,
                    'log_likelihood': metrics['log_likelihood'],
                    'aic': metrics.get('aic', np.nan),
                    'bic': metrics.get('bic', np.nan)
                })
        
        return pd.DataFrame(results)
    
    def regime_stability_analysis(self, 
                                state_probs: np.ndarray,
                                min_duration: int = 5) -> Dict[str, Any]:
        """Analyze regime stability and persistence."""
        
        # Decode most likely state sequence
        most_likely_states = np.argmax(state_probs, axis=1)
        
        # Calculate state durations
        state_durations = {}
        current_state = most_likely_states[0]
        current_duration = 1
        
        for i in range(1, len(most_likely_states)):
            if most_likely_states[i] == current_state:
                current_duration += 1
            else:
                if current_state not in state_durations:
                    state_durations[current_state] = []
                state_durations[current_state].append(current_duration)
                current_state = most_likely_states[i]
                current_duration = 1
        
        # Add final duration
        if current_state not in state_durations:
            state_durations[current_state] = []
        state_durations[current_state].append(current_duration)
        
        # Calculate statistics
        stability_metrics = {}
        for state, durations in state_durations.items():
            stability_metrics[f'state_{state}'] = {
                'mean_duration': np.mean(durations),
                'median_duration': np.median(durations),
                'max_duration': np.max(durations),
                'stable_periods': sum(1 for d in durations if d >= min_duration),
                'total_periods': len(durations)
            }
        
        return stability_metrics
```

## Data Models

### Enhanced Artifact Management

**Research Artifact Schema:**
```python
from datetime import datetime
from typing import Optional, List, Dict, Any

class ResearchArtifact(BaseModel):
    """Extended artifact for research environment."""
    
    # Inherit from existing HMMArtifact
    base_artifact: HMMArtifact
    
    # Research-specific metadata
    research_metadata: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any]
    evaluation_metrics: Dict[str, float]
    cross_validation_scores: Optional[List[float]] = None
    
    # Experiment tracking
    experiment_id: str
    researcher: str
    created_at: datetime = Field(default_factory=datetime.now)
    notebook_path: Optional[str] = None
    
    # Model comparison
    comparison_baseline: Optional[str] = None
    performance_ranking: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True

class ExperimentTracker:
    """Track and manage research experiments."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.experiments = {}
    
    def log_experiment(self, 
                      artifact: ResearchArtifact,
                      notes: str = "") -> str:
        """Log research experiment with versioning."""
        
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_data = {
            'artifact': artifact,
            'notes': notes,
            'timestamp': datetime.now()
        }
        
        # Save to disk
        experiment_path = self.experiment_dir / f"{experiment_id}.json"
        with open(experiment_path, 'w') as f:
            json.dump(artifact.dict(), f, indent=2, default=str)
        
        self.experiments[experiment_id] = experiment_data
        return experiment_id
```

## Error Handling

### Research-Specific Error Management

```python
class HMMResearchError(Exception):
    """Base exception for HMM research environment."""
    pass

class ModelTrainingError(HMMResearchError):
    """Error during model training."""
    pass

class VisualizationError(HMMResearchError):
    """Error during visualization generation."""
    pass

class DataIntegrationError(HMMResearchError):
    """Error integrating with LDC signal data."""
    pass

# Error handling with user-friendly messages for notebooks
def safe_train_model(trainer, observations, **kwargs):
    """Safe model training with error handling for notebooks."""
    try:
        return trainer.train(observations, **kwargs)
    except Exception as e:
        error_msg = f"""
        Model training failed: {str(e)}
        
        Suggestions:
        1. Check data quality and preprocessing
        2. Try different initialization parameters
        3. Reduce model complexity (fewer states)
        4. Increase number of iterations
        """
        raise ModelTrainingError(error_msg) from e
```

## Testing Strategy

### Research Environment Testing

```python
# Notebook testing framework
class NotebookTester:
    """Test notebook execution and outputs."""
    
    def test_notebook_execution(self, notebook_path: Path) -> bool:
        """Test that notebook executes without errors."""
        # Implementation for automated notebook testing
        pass
    
    def validate_outputs(self, notebook_path: Path) -> Dict[str, bool]:
        """Validate notebook outputs and visualizations."""
        # Implementation for output validation
        pass

# Integration testing with LDC engine
def test_ldc_integration():
    """Test integration with Rust LDC engine outputs."""
    # Load sample LDC signals
    # Train HMM model
    # Validate artifact compatibility
    pass
```

## Implementation Considerations

### Jupyter Environment Setup

1. **Environment Management**: Use conda/pip for dependency management
2. **Kernel Configuration**: Ensure proper Python kernel with all dependencies
3. **Extension Support**: Install Jupyter extensions for enhanced functionality
4. **Resource Management**: Configure memory and CPU limits for large datasets

### Performance Optimization

1. **Data Loading**: Efficient loading of large signal datasets
2. **Model Training**: Parallel training for hyperparameter optimization
3. **Visualization**: Efficient plotting for large time series
4. **Memory Management**: Proper cleanup of large numpy arrays

### Production Integration

1. **Artifact Compatibility**: Ensure research artifacts work with production system
2. **Version Control**: Track model versions and experiment history
3. **Deployment Pipeline**: Automated testing before production deployment
4. **Monitoring**: Track model performance in research vs production