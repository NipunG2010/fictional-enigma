# HMM Research Environment Integration Examples

## Table of Contents

1. [End-to-End Research Workflow](#end-to-end-research-workflow)
2. [LDC Signal Integration](#ldc-signal-integration)
3. [Production Deployment Pipeline](#production-deployment-pipeline)
4. [Multi-Library Comparison](#multi-library-comparison)
5. [Automated Hyperparameter Tuning](#automated-hyperparameter-tuning)
6. [Real-Time Regime Detection](#real-time-regime-detection)

---

## End-to-End Research Workflow

Complete workflow from data loading to production deployment.

### Step 1: Load and Validate Data

```python
from pathlib import Path
import numpy as np
import pandas as pd
from imp.data.ldc_loader import LDCDataLoader
from imp.data.validator import DataValidator
from imp.data.preprocessor import SignalPreprocessor

# Load LDC signals
loader = LDCDataLoader()
df = loader.load_signals(
    'processed_data/signals_processed.parquet',
    signals=['s_LDC', 's_MR', 's_TSMOM']
)

print(f"Loaded {len(df)} samples with {len(df.columns)} signals")

# Validate data quality
validator = DataValidator()
validation_report = validator.validate(df)

if not validation_report['is_valid']:
    print("Data quality issues:")
    for issue in validation_report['issues']:
        print(f"  - {issue}")
else:
    print("Data validation passed ✓")

# Preprocess data
preprocessor = SignalPreprocessor()
observations, preprocessing_metadata = preprocessor.preprocess(
    df,
    handle_missing='forward_fill',
    handle_outliers=True,
    normalize=True,
    outlier_threshold=3.0
)

print(f"Preprocessed data shape: {observations.shape}")
print(f"Normalization params: {preprocessing_metadata['normalization']}")
```

### Step 2: Train and Compare Models

```python
from imp.hmm.trainer import EnhancedHMMTrainer
from imp.evaluation.evaluator import HMMEvaluator

# Define configurations to test
configs = [
    {'n_states': 2, 'covariance_type': 'diag', 'library': 'hmmlearn'},
    {'n_states': 3, 'covariance_type': 'diag', 'library': 'hmmlearn'},
    {'n_states': 3, 'covariance_type': 'full', 'library': 'hmmlearn'},
    {'n_states': 4, 'covariance_type': 'full', 'library': 'hmmlearn'},
]

# Split data
split_idx = int(len(observations) * 0.8)
train_data = observations[:split_idx]
test_data = observations[split_idx:]

# Compare models
evaluator = HMMEvaluator()
comparison_results = evaluator.compare_models(
    train_data,
    configs,
    metrics=['log_likelihood', 'aic', 'bic']
)

print("\nModel Comparison Results:")
print(comparison_results.sort_values('aic'))

# Select best configuration
best_config = comparison_results.loc[comparison_results['aic'].idxmin()]
print(f"\nBest configuration: {best_config['config']}")
```

### Step 3: Train Final Model with Cross-Validation

```python
# Train with best configuration
best_params = {
    'n_states': int(best_config['n_states']),
    'covariance_type': best_config['covariance_type'],
    'library': best_config['library']
}

# Cross-validation
cv_results = evaluator.cross_validate(
    train_data,
    trainer_config=best_params,
    cv_folds=5
)

print(f"\nCross-validation results:")
print(f"  Mean score: {cv_results['mean_score']:.4f}")
print(f"  Std score: {cv_results['std_score']:.4f}")
print(f"  Fold scores: {cv_results['fold_scores']}")

# Train final model on all training data
trainer = EnhancedHMMTrainer(**best_params, random_state=42)
artifact = trainer.train(train_data, n_iterations=200)

print(f"\nFinal model trained with {artifact.n_states} states")
```

### Step 4: Analyze Regimes

```python
from imp.hmm.inference import HMMInference
from imp.hmm.regime_analysis import RegimeAnalyzer

# Perform inference
inference = HMMInference(artifact)
state_probs = inference.predict_proba(test_data)
state_sequence = inference.predict(test_data)

# Analyze regimes
analyzer = RegimeAnalyzer(artifact)
regime_analysis = analyzer.analyze_regimes(
    test_data,
    state_probs,
    feature_names=['s_LDC', 's_MR', 's_TSMOM']
)

print("\nRegime Analysis:")
print(f"State durations: {regime_analysis['state_durations']}")
print(f"Transition frequencies: {regime_analysis['transition_frequencies']}")

# Get regime interpretation
state_stats = analyzer.calculate_state_statistics(test_data, state_sequence)
interpretations = analyzer.get_regime_interpretation(state_stats)

for state, interpretation in interpretations.items():
    print(f"\nState {state}: {interpretation}")
```

### Step 5: Visualize Results

```python
from imp.visualization.regime_visualizer import RegimeVisualizer
import matplotlib.pyplot as plt

visualizer = RegimeVisualizer(artifact)

# Plot state probabilities
fig1 = visualizer.plot_state_probabilities(
    state_probs,
    interactive=False,
    title='Market Regime Detection - Test Set'
)
fig1.savefig('regime_probabilities.png', dpi=300, bbox_inches='tight')

# Plot transition matrix
fig2 = visualizer.plot_transition_matrix(annotate=True)
fig2.savefig('transition_matrix.png', dpi=300, bbox_inches='tight')

plt.show()
```

### Step 6: Save Artifact for Production

```python
from imp.hmm.artifact_management import ArtifactManager
from datetime import datetime

manager = ArtifactManager(artifacts_dir='artifacts/')

# Prepare comprehensive metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'data_source': 'BTCUSDT_5m',
    'data_period': '2024-01-01 to 2024-12-31',
    'training_samples': len(train_data),
    'test_samples': len(test_data),
    'training_config': best_params,
    'cv_results': {
        'mean_score': float(cv_results['mean_score']),
        'std_score': float(cv_results['std_score'])
    },
    'test_performance': {
        'log_likelihood': float(inference.score(test_data))
    },
    'preprocessing': preprocessing_metadata,
    'regime_interpretation': interpretations,
    'created_by': 'research_team'
}

# Validate before saving
validation_report = manager.validate_artifact(artifact, validation_data=test_data)

if validation_report['is_valid']:
    artifact_path = manager.save_artifact(
        artifact,
        name='market_regime_detector',
        version='1.0.0',
        metadata=metadata
    )
    print(f"\nArtifact saved to: {artifact_path}")
else:
    print("Artifact validation failed:")
    for error in validation_report['errors']:
        print(f"  - {error}")
```

---

## LDC Signal Integration

Integrate with Rust LDC engine output.

### Loading LDC Signals from Parquet

```python
from imp.data.ldc_loader import LDCDataLoader
import pandas as pd

loader = LDCDataLoader()

# Load from Rust engine output
df = loader.load_signals(
    'rust/ldc-engine/ldc_pipeline_results.parquet',
    signals=['s_LDC', 's_MR', 's_TSMOM']
)

# Check signal availability
print(f"Available signals: {df.columns.tolist()}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Total samples: {len(df)}")

# Basic statistics
print("\nSignal Statistics:")
print(df.describe())
```

### Feature Engineering for HMM

```python
from imp.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Create additional features
features_df = engineer.create_features(
    df,
    include_lags=True,
    lag_periods=[1, 5, 10],
    include_rolling_stats=True,
    rolling_windows=[20, 50]
)

print(f"Original features: {df.shape[1]}")
print(f"Engineered features: {features_df.shape[1]}")

# Select features for HMM
selected_features = [
    's_LDC', 's_MR', 's_TSMOM',
    's_LDC_lag_1', 's_MR_lag_1',
    's_LDC_rolling_mean_20', 's_LDC_rolling_std_20'
]

hmm_features = features_df[selected_features].values
```

### Quality Reporting

```python
from imp.data.quality_report import QualityReporter

reporter = QualityReporter()

# Generate quality report
report = reporter.generate_report(
    df,
    output_path='data_quality_report.html'
)

print("\nData Quality Summary:")
print(f"  Missing values: {report['missing_percentage']:.2f}%")
print(f"  Outliers detected: {report['outlier_count']}")
print(f"  Quality score: {report['quality_score']:.2f}/100")

# Visualize quality
reporter.plot_quality_dashboard(df)
```

---

## Production Deployment Pipeline

Complete pipeline for deploying models to production.

### Automated Testing Pipeline

```python
from imp.hmm.artifact_management import ArtifactManager
from imp.hmm.inference import HMMInference
import numpy as np

def test_artifact_production_readiness(artifact_name: str, version: str):
    """Test artifact before production deployment."""
    
    manager = ArtifactManager()
    
    # Load artifact
    artifact, metadata = manager.load_artifact(artifact_name, version)
    
    # Test 1: Artifact structure validation
    print("Test 1: Artifact structure validation...")
    validation_report = manager.validate_artifact(artifact)
    assert validation_report['is_valid'], f"Validation failed: {validation_report['errors']}"
    print("  ✓ Passed")
    
    # Test 2: Inference functionality
    print("Test 2: Inference functionality...")
    inference = HMMInference(artifact)
    test_data = np.random.randn(100, artifact.n_features)
    
    try:
        state_probs = inference.predict_proba(test_data)
        assert state_probs.shape == (100, artifact.n_states)
        print("  ✓ Passed")
    except Exception as e:
        raise AssertionError(f"Inference failed: {e}")
    
    # Test 3: Performance threshold
    print("Test 3: Performance threshold...")
    min_log_likelihood = metadata.get('performance_threshold', -2000)
    test_score = inference.score(test_data)
    assert test_score > min_log_likelihood, f"Performance below threshold: {test_score}"
    print(f"  ✓ Passed (score: {test_score:.2f})")
    
    # Test 4: Metadata completeness
    print("Test 4: Metadata completeness...")
    required_fields = ['training_date', 'training_config', 'regime_interpretation']
    for field in required_fields:
        assert field in metadata, f"Missing metadata field: {field}"
    print("  ✓ Passed")
    
    print("\n✓ All tests passed. Artifact ready for production.")
    return True

# Run tests
test_artifact_production_readiness('market_regime_detector', '1.0.0')
```

### Deployment Script

```python
from pathlib import Path
import shutil
import json
from datetime import datetime

def deploy_artifact_to_production(
    artifact_name: str,
    version: str,
    production_dir: Path,
    dry_run: bool = False
):
    """Deploy artifact to production environment."""
    
    manager = ArtifactManager()
    
    # Load artifact
    artifact, metadata = manager.load_artifact(artifact_name, version)
    
    # Run production readiness tests
    print("Running production readiness tests...")
    test_artifact_production_readiness(artifact_name, version)
    
    # Prepare deployment
    deployment_info = {
        'artifact_name': artifact_name,
        'version': version,
        'deployment_date': datetime.now().isoformat(),
        'deployed_by': 'deployment_script',
        'metadata': metadata
    }
    
    if dry_run:
        print("\n[DRY RUN] Would deploy:")
        print(json.dumps(deployment_info, indent=2))
        return
    
    # Create production directory
    production_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy artifact
    source_path = manager.get_artifact_path(artifact_name, version)
    dest_path = production_dir / f"{artifact_name}_v{version}.pkl"
    shutil.copy(source_path, dest_path)
    
    # Save deployment info
    info_path = production_dir / f"{artifact_name}_v{version}_deployment.json"
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    # Update production symlink
    symlink_path = production_dir / f"{artifact_name}_production.pkl"
    if symlink_path.exists():
        symlink_path.unlink()
    symlink_path.symlink_to(dest_path.name)
    
    print(f"\n✓ Deployed {artifact_name} v{version} to production")
    print(f"  Artifact: {dest_path}")
    print(f"  Symlink: {symlink_path}")
    
    return deployment_info

# Deploy
deploy_artifact_to_production(
    'market_regime_detector',
    '1.0.0',
    Path('production/artifacts'),
    dry_run=True  # Set to False for actual deployment
)
```

### Rollback Procedure

```python
def rollback_to_previous_version(
    artifact_name: str,
    production_dir: Path
):
    """Rollback to previous production version."""
    
    # Find deployment history
    deployment_files = sorted(
        production_dir.glob(f"{artifact_name}_*_deployment.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if len(deployment_files) < 2:
        raise ValueError("No previous version available for rollback")
    
    # Get previous version
    previous_deployment = deployment_files[1]
    with open(previous_deployment) as f:
        previous_info = json.load(f)
    
    previous_version = previous_info['version']
    
    print(f"Rolling back to version {previous_version}...")
    
    # Update symlink
    symlink_path = production_dir / f"{artifact_name}_production.pkl"
    artifact_path = production_dir / f"{artifact_name}_v{previous_version}.pkl"
    
    if symlink_path.exists():
        symlink_path.unlink()
    symlink_path.symlink_to(artifact_path.name)
    
    print(f"✓ Rolled back to {artifact_name} v{previous_version}")
    
    return previous_version

# Rollback if needed
# rollback_to_previous_version('market_regime_detector', Path('production/artifacts'))
```

---

## Multi-Library Comparison

Compare hmmlearn and pomegranate implementations.

```python
from imp.hmm.trainer import EnhancedHMMTrainer
from imp.evaluation.evaluator import HMMEvaluator
import time

def compare_hmm_libraries(observations, n_states=3):
    """Compare hmmlearn and pomegranate performance."""
    
    results = {}
    
    for library in ['hmmlearn', 'pomegranate']:
        print(f"\nTesting {library}...")
        
        try:
            # Create trainer
            trainer = EnhancedHMMTrainer(
                n_states=n_states,
                library=library,
                covariance_type='full',
                random_state=42
            )
            
            # Time training
            start_time = time.time()
            artifact = trainer.train(observations, n_iterations=100)
            training_time = time.time() - start_time
            
            # Evaluate
            inference = HMMInference(artifact)
            
            start_time = time.time()
            state_probs = inference.predict_proba(observations)
            inference_time = time.time() - start_time
            
            log_likelihood = inference.score(observations)
            
            results[library] = {
                'training_time': training_time,
                'inference_time': inference_time,
                'log_likelihood': log_likelihood,
                'converged': artifact.metadata.get('converged', False),
                'n_iterations': artifact.metadata.get('n_iterations', 0)
            }
            
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Inference time: {inference_time:.4f}s")
            print(f"  Log-likelihood: {log_likelihood:.2f}")
            print(f"  Converged: {results[library]['converged']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[library] = {'error': str(e)}
    
    return results

# Run comparison
comparison = compare_hmm_libraries(train_data, n_states=3)

# Print summary
print("\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)

for library, metrics in comparison.items():
    if 'error' not in metrics:
        print(f"\n{library.upper()}:")
        print(f"  Training: {metrics['training_time']:.2f}s")
        print(f"  Inference: {metrics['inference_time']:.4f}s")
        print(f"  Log-likelihood: {metrics['log_likelihood']:.2f}")
```

---

## Automated Hyperparameter Tuning

Automated parameter optimization with Bayesian optimization.

```python
from imp.tuning.optimization import BayesianOptimizer
from imp.tuning.parameter_tuner import HMMParameterTuner

# Define parameter space
param_space = {
    'n_states': [2, 3, 4, 5],
    'covariance_type': ['diag', 'full', 'spherical'],
    'n_iterations': [50, 100, 200]
}

# Create optimizer
optimizer = BayesianOptimizer(
    observations=train_data,
    param_space=param_space,
    cv_folds=3
)

# Run optimization
print("Starting Bayesian optimization...")
best_params, optimization_history = optimizer.optimize(
    n_iterations=20,
    random_state=42
)

print(f"\nBest parameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Train final model with best parameters
trainer = EnhancedHMMTrainer(**best_params, random_state=42)
final_artifact = trainer.train(train_data)

# Evaluate
inference = HMMInference(final_artifact)
test_score = inference.score(test_data)

print(f"\nFinal model test score: {test_score:.2f}")

# Plot optimization history
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(optimization_history['iteration'], optimization_history['score'])
ax.set_xlabel('Iteration')
ax.set_ylabel('Cross-Validation Score')
ax.set_title('Bayesian Optimization Progress')
ax.grid(True, alpha=0.3)
plt.show()
```

---

## Real-Time Regime Detection

Implement real-time regime detection for live trading.

```python
from imp.hmm.inference import HMMInference
from imp.hmm.artifact_management import ArtifactManager
from collections import deque
import numpy as np

class RealtimeRegimeDetector:
    """Real-time regime detection for live trading."""
    
    def __init__(self, artifact_name: str, version: str, window_size: int = 100):
        """
        Initialize detector.
        
        Args:
            artifact_name: Name of HMM artifact
            version: Artifact version
            window_size: Size of rolling window for inference
        """
        # Load artifact
        manager = ArtifactManager()
        self.artifact, self.metadata = manager.load_artifact(artifact_name, version)
        
        # Create inference engine
        self.inference = HMMInference(self.artifact)
        
        # Initialize rolling window
        self.window_size = window_size
        self.observation_buffer = deque(maxlen=window_size)
        
        # Get preprocessing parameters
        self.preprocessing = self.metadata.get('preprocessing', {})
        self.normalization = self.preprocessing.get('normalization', {})
        
        print(f"Initialized detector with {self.artifact.n_states} states")
    
    def preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        """Preprocess new observation using training parameters."""
        
        if self.normalization:
            mean = np.array(self.normalization['mean'])
            std = np.array(self.normalization['std'])
            observation = (observation - mean) / std
        
        return observation
    
    def update(self, observation: np.ndarray) -> dict:
        """
        Update detector with new observation.
        
        Args:
            observation: New observation vector
            
        Returns:
            Dictionary with current regime probabilities and prediction
        """
        # Preprocess
        processed_obs = self.preprocess_observation(observation)
        
        # Add to buffer
        self.observation_buffer.append(processed_obs)
        
        # Need minimum observations for inference
        if len(self.observation_buffer) < 10:
            return {
                'ready': False,
                'message': f'Collecting data... ({len(self.observation_buffer)}/10)'
            }
        
        # Convert buffer to array
        observations = np.array(self.observation_buffer)
        
        # Perform inference
        state_probs = self.inference.predict_proba(observations)
        current_probs = state_probs[-1]  # Latest probabilities
        
        # Get most likely state
        current_state = np.argmax(current_probs)
        
        # Get regime interpretation
        regime_interpretation = self.metadata.get('regime_interpretation', {})
        regime_name = regime_interpretation.get(str(current_state), {}).get('name', f'State {current_state}')
        
        return {
            'ready': True,
            'current_state': int(current_state),
            'state_probabilities': current_probs.tolist(),
            'regime_name': regime_name,
            'confidence': float(current_probs[current_state]),
            'buffer_size': len(self.observation_buffer)
        }
    
    def get_regime_recommendation(self, result: dict) -> str:
        """Get trading recommendation based on current regime."""
        
        if not result['ready']:
            return "Insufficient data"
        
        state = result['current_state']
        confidence = result['confidence']
        
        # Get regime interpretation
        regime_interpretation = self.metadata.get('regime_interpretation', {})
        regime_info = regime_interpretation.get(str(state), {})
        
        recommendation = regime_info.get('trading_implication', 'No recommendation')
        
        if confidence < 0.6:
            recommendation += " (Low confidence - use caution)"
        
        return recommendation

# Example usage
detector = RealtimeRegimeDetector('market_regime_detector', '1.0.0')

# Simulate real-time updates
print("\nSimulating real-time regime detection...")

for i in range(150):
    # Simulate new observation (in practice, this comes from live data)
    new_observation = np.random.randn(3)  # 3 features: s_LDC, s_MR, s_TSMOM
    
    # Update detector
    result = detector.update(new_observation)
    
    # Print every 10 observations
    if i % 10 == 0:
        if result['ready']:
            print(f"\nObservation {i}:")
            print(f"  Regime: {result['regime_name']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  State probabilities: {[f'{p:.2%}' for p in result['state_probabilities']]}")
            print(f"  Recommendation: {detector.get_regime_recommendation(result)}")
        else:
            print(f"\nObservation {i}: {result['message']}")
```

---

## Batch Processing Pipeline

Process multiple datasets efficiently.

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def process_single_dataset(file_path: Path, artifact_name: str, version: str):
    """Process a single dataset file."""
    
    # Load data
    loader = LDCDataLoader()
    df = loader.load_signals(file_path)
    
    # Preprocess
    preprocessor = SignalPreprocessor()
    observations, _ = preprocessor.preprocess(df, normalize=True)
    
    # Load artifact
    manager = ArtifactManager()
    artifact, _ = manager.load_artifact(artifact_name, version)
    
    # Perform inference
    inference = HMMInference(artifact)
    state_probs = inference.predict_proba(observations)
    state_sequence = inference.predict(observations)
    
    # Create results dataframe
    results_df = df.copy()
    for i in range(artifact.n_states):
        results_df[f'state_{i}_prob'] = state_probs[:, i]
    results_df['predicted_state'] = state_sequence
    
    # Save results
    output_path = file_path.parent / f"{file_path.stem}_regimes.parquet"
    results_df.to_parquet(output_path)
    
    return {
        'file': file_path.name,
        'samples': len(df),
        'output': output_path
    }

def batch_process_datasets(
    data_dir: Path,
    artifact_name: str,
    version: str,
    max_workers: int = 4
):
    """Process multiple datasets in parallel."""
    
    # Find all parquet files
    data_files = list(data_dir.glob('*.parquet'))
    
    print(f"Found {len(data_files)} files to process")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_dataset, file_path, artifact_name, version)
            for file_path in data_files
        ]
        
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
                print(f"✓ Processed {result['file']}: {result['samples']} samples")
            except Exception as e:
                print(f"✗ Error processing file: {e}")
    
    print(f"\n✓ Batch processing complete: {len(results)}/{len(data_files)} files processed")
    
    return results

# Run batch processing
# results = batch_process_datasets(
#     Path('data/signals'),
#     'market_regime_detector',
#     '1.0.0',
#     max_workers=4
# )
```

---

## Additional Resources

- [API Documentation](HMM_RESEARCH_API.md)
- [Best Practices](BEST_PRACTICES.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Tutorial Notebooks](../notebooks/README.md)
