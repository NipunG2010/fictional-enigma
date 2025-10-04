# Example Configurations for HMM Research Environment

This document provides example configurations for different market scenarios and use cases.

## Table of Contents

1. [Basic Configurations](#basic-configurations)
2. [Market Scenario Configurations](#market-scenario-configurations)
3. [Performance-Optimized Configurations](#performance-optimized-configurations)
4. [Production Configurations](#production-configurations)

---

## Basic Configurations

### Simple 2-State Model

Best for: Initial exploration, binary regime detection (trending vs mean-reverting)

```python
from imp.hmm.trainer import EnhancedHMMTrainer

config = {
    'n_states': 2,
    'library': 'hmmlearn',
    'covariance_type': 'diag',
    'random_state': 42
}

trainer = EnhancedHMMTrainer(**config)
artifact = trainer.train(observations, n_iterations=100)
```

**Characteristics:**
- Fast training
- Easy interpretation
- Good for clear regime separation
- Lower computational requirements

### Standard 3-State Model

Best for: Most market scenarios, balanced complexity

```python
config = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'full',
    'random_state': 42
}

trainer = EnhancedHMMTrainer(**config)
artifact = trainer.train(
    observations,
    n_iterations=200,
    convergence_threshold=1e-4
)
```

**Characteristics:**
- Captures low/medium/high volatility regimes
- Good balance of complexity and interpretability
- Recommended starting point

### Complex 4-State Model

Best for: Detailed regime analysis, complex market dynamics

```python
config = {
    'n_states': 4,
    'library': 'hmmlearn',
    'covariance_type': 'full',
    'random_state': 42
}

trainer = EnhancedHMMTrainer(**config)
artifact = trainer.train(
    observations,
    n_iterations=300,
    convergence_threshold=1e-5
)
```

**Characteristics:**
- Captures nuanced market states
- Requires more data for reliable training
- Higher computational cost
- May be prone to overfitting

---

## Market Scenario Configurations

### High-Frequency Trading (HFT)

Optimized for: Fast inference, real-time regime detection

```python
# Configuration
config = {
    'n_states': 2,  # Simple for speed
    'library': 'hmmlearn',
    'covariance_type': 'diag',  # Faster than full
    'random_state': 42
}

# Data preprocessing
from imp.data.preprocessor import SignalPreprocessor

preprocessor = SignalPreprocessor()
observations, metadata = preprocessor.preprocess(
    data,
    handle_missing='forward_fill',  # Fast method
    handle_outliers=False,  # Skip for speed
    normalize=True
)

# Training
trainer = EnhancedHMMTrainer(**config)
artifact = trainer.train(
    observations,
    n_iterations=50,  # Fewer iterations for speed
    convergence_threshold=1e-3  # Less strict
)
```

**Key Features:**
- Minimal latency
- Simple model structure
- Fast preprocessing
- Quick convergence

### Swing Trading

Optimized for: Medium-term regime detection, balanced accuracy

```python
config = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'full',
    'random_state': 42
}

# Use more historical data
observations = load_data(lookback_days=180)

# Standard preprocessing
preprocessor = SignalPreprocessor()
observations, metadata = preprocessor.preprocess(
    observations,
    handle_missing='interpolate',
    handle_outliers=True,
    normalize=True,
    outlier_threshold=3.0
)

# Training with validation
trainer = EnhancedHMMTrainer(**config)
artifact, metrics = trainer.train_with_validation(
    observations,
    validation_split=0.2,
    n_iterations=200
)
```

**Key Features:**
- Balanced complexity
- Comprehensive preprocessing
- Validation for robustness
- Medium-term regime persistence

### Long-Term Portfolio Management

Optimized for: Stable regime detection, economic interpretation

```python
config = {
    'n_states': 4,
    'library': 'hmmlearn',
    'covariance_type': 'full',
    'random_state': 42
}

# Use extensive historical data
observations = load_data(lookback_days=730)  # 2 years

# Thorough preprocessing
preprocessor = SignalPreprocessor()
observations, metadata = preprocessor.preprocess(
    observations,
    handle_missing='interpolate',
    handle_outliers=True,
    normalize=True,
    outlier_threshold=2.5  # More conservative
)

# Training with cross-validation
from imp.evaluation.evaluator import HMMEvaluator

evaluator = HMMEvaluator()
cv_results = evaluator.cross_validate(
    observations,
    trainer_config=config,
    cv_folds=10  # More folds for robustness
)

# Train final model
trainer = EnhancedHMMTrainer(**config)
artifact = trainer.train(
    observations,
    n_iterations=500,  # More iterations
    convergence_threshold=1e-6  # Stricter convergence
)
```

**Key Features:**
- Detailed regime characterization
- Extensive validation
- Long training for stability
- Economic interpretability

### Cryptocurrency Markets

Optimized for: High volatility, rapid regime changes

```python
config = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'full',
    'random_state': 42
}

# Aggressive outlier handling for crypto
preprocessor = SignalPreprocessor()
observations, metadata = preprocessor.preprocess(
    data,
    handle_missing='forward_fill',
    handle_outliers=True,
    normalize=True,
    outlier_threshold=4.0  # More lenient for crypto volatility
)

# Multiple random seeds for stability
best_artifact = None
best_score = -np.inf

for seed in [42, 123, 456, 789, 1011]:
    trainer = EnhancedHMMTrainer(**{**config, 'random_state': seed})
    artifact = trainer.train(observations, n_iterations=200)
    
    inference = HMMInference(artifact)
    score = inference.score(test_data)
    
    if score > best_score:
        best_score = score
        best_artifact = artifact

print(f"Best model score: {best_score:.2f}")
```

**Key Features:**
- Handles extreme volatility
- Multiple initialization attempts
- Robust to outliers
- Captures rapid regime changes

---

## Performance-Optimized Configurations

### Memory-Constrained Environment

```python
# Use diagonal covariance to reduce memory
config = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'diag',  # Less memory than full
    'random_state': 42
}

# Process data in batches
batch_size = 1000
results = []

for i in range(0, len(observations), batch_size):
    batch = observations[i:i+batch_size]
    
    # Use float32 instead of float64
    batch = batch.astype(np.float32)
    
    inference = HMMInference(artifact)
    batch_result = inference.predict_proba(batch)
    results.append(batch_result)

state_probs = np.vstack(results)
```

**Memory Savings:**
- Diagonal covariance: ~50% less memory
- Float32: 50% less memory
- Batch processing: Constant memory usage

### CPU-Constrained Environment

```python
# Simplify model for faster computation
config = {
    'n_states': 2,  # Fewer states
    'library': 'hmmlearn',
    'covariance_type': 'spherical',  # Simplest covariance
    'random_state': 42
}

# Reduce data size
observations_subset = observations[::2]  # Use every other sample

# Quick training
trainer = EnhancedHMMTrainer(**config)
artifact = trainer.train(
    observations_subset,
    n_iterations=50,  # Fewer iterations
    convergence_threshold=1e-2  # Less strict
)
```

**Speed Improvements:**
- Fewer states: 2-3x faster
- Spherical covariance: 2x faster
- Data subsampling: Proportional speedup
- Fewer iterations: Proportional speedup

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from imp.hmm.trainer import EnhancedHMMTrainer

# Define configurations to test
configs = [
    {'n_states': 2, 'covariance_type': 'diag'},
    {'n_states': 3, 'covariance_type': 'diag'},
    {'n_states': 3, 'covariance_type': 'full'},
    {'n_states': 4, 'covariance_type': 'full'},
]

def train_model(config):
    """Train a single model configuration."""
    trainer = EnhancedHMMTrainer(**config, library='hmmlearn', random_state=42)
    artifact = trainer.train(observations, n_iterations=100)
    
    inference = HMMInference(artifact)
    score = inference.score(test_data)
    
    return {
        'config': config,
        'artifact': artifact,
        'score': score
    }

# Train models in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(train_model, configs))

# Select best model
best_result = max(results, key=lambda x: x['score'])
best_artifact = best_result['artifact']

print(f"Best configuration: {best_result['config']}")
print(f"Best score: {best_result['score']:.2f}")
```

**Speedup:**
- 4 workers: ~4x faster for multiple configurations
- Scales with number of CPU cores

---

## Production Configurations

### Production-Ready Configuration

```python
from imp.hmm.trainer import EnhancedHMMTrainer
from imp.hmm.artifact_management import ArtifactManager
from imp.evaluation.evaluator import HMMEvaluator
from datetime import datetime

# Configuration
config = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'full',
    'random_state': 42
}

# Comprehensive preprocessing
preprocessor = SignalPreprocessor()
observations, preprocessing_metadata = preprocessor.preprocess(
    data,
    handle_missing='forward_fill',
    handle_outliers=True,
    normalize=True,
    outlier_threshold=3.0
)

# Split data
train_size = int(len(observations) * 0.7)
val_size = int(len(observations) * 0.15)

train_data = observations[:train_size]
val_data = observations[train_size:train_size+val_size]
test_data = observations[train_size+val_size:]

# Cross-validation on training data
evaluator = HMMEvaluator()
cv_results = evaluator.cross_validate(
    train_data,
    trainer_config=config,
    cv_folds=5
)

print(f"CV Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")

# Train final model
trainer = EnhancedHMMTrainer(**config)
artifact = trainer.train(train_data, n_iterations=200)

# Validate on validation set
inference = HMMInference(artifact)
val_score = inference.score(val_data)
print(f"Validation Score: {val_score:.4f}")

# Final test on test set
test_score = inference.score(test_data)
print(f"Test Score: {test_score:.4f}")

# Regime analysis
from imp.hmm.regime_analysis import RegimeAnalyzer

state_probs = inference.predict_proba(test_data)
state_sequence = inference.predict(test_data)

analyzer = RegimeAnalyzer(artifact)
regime_analysis = analyzer.analyze_regimes(
    test_data,
    state_probs,
    feature_names=['s_LDC', 's_MR', 's_TSMOM']
)

state_stats = analyzer.calculate_state_statistics(test_data, state_sequence)
interpretations = analyzer.get_regime_interpretation(state_stats)

# Save with comprehensive metadata
manager = ArtifactManager(artifacts_dir='production/artifacts')

metadata = {
    'training_date': datetime.now().isoformat(),
    'data_source': 'BTCUSDT_5m',
    'training_samples': len(train_data),
    'validation_samples': len(val_data),
    'test_samples': len(test_data),
    'training_config': config,
    'cv_results': {
        'mean_score': float(cv_results['mean_score']),
        'std_score': float(cv_results['std_score'])
    },
    'validation_performance': {
        'log_likelihood': float(val_score)
    },
    'test_performance': {
        'log_likelihood': float(test_score)
    },
    'preprocessing': preprocessing_metadata,
    'regime_interpretation': interpretations,
    'regime_analysis': regime_analysis,
    'created_by': 'production_pipeline',
    'quality_checks': {
        'cv_passed': cv_results['std_score'] < 100,
        'validation_passed': val_score > -2000,
        'test_passed': test_score > -2000
    }
}

# Validate before saving
validation_report = manager.validate_artifact(artifact, validation_data=test_data)

if validation_report['is_valid']:
    artifact_path = manager.save_artifact(
        artifact,
        name='production_regime_detector',
        version='1.0.0',
        metadata=metadata
    )
    print(f"\n✓ Production artifact saved: {artifact_path}")
else:
    print("\n✗ Artifact validation failed:")
    for error in validation_report['errors']:
        print(f"  - {error}")
```

### A/B Testing Configuration

```python
# Configuration A: Current production model
config_a = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'full',
    'random_state': 42
}

# Configuration B: Candidate model
config_b = {
    'n_states': 4,
    'library': 'hmmlearn',
    'covariance_type': 'full',
    'random_state': 42
}

# Train both models
trainer_a = EnhancedHMMTrainer(**config_a)
artifact_a = trainer_a.train(train_data, n_iterations=200)

trainer_b = EnhancedHMMTrainer(**config_b)
artifact_b = trainer_b.train(train_data, n_iterations=200)

# Compare on test set
inference_a = HMMInference(artifact_a)
inference_b = HMMInference(artifact_b)

score_a = inference_a.score(test_data)
score_b = inference_b.score(test_data)

print(f"Model A (current): {score_a:.4f}")
print(f"Model B (candidate): {score_b:.4f}")
print(f"Improvement: {((score_b - score_a) / abs(score_a) * 100):.2f}%")

# Statistical significance test
from scipy import stats

# Get log-likelihoods for each sample
ll_a = [inference_a.score(test_data[i:i+100]) for i in range(0, len(test_data)-100, 100)]
ll_b = [inference_b.score(test_data[i:i+100]) for i in range(0, len(test_data)-100, 100)]

t_stat, p_value = stats.ttest_rel(ll_a, ll_b)

print(f"\nStatistical test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05 and score_b > score_a:
    print("  ✓ Model B is significantly better. Recommend deployment.")
else:
    print("  ✗ No significant improvement. Keep Model A.")
```

---

## Configuration Selection Guide

### Decision Tree

```
Start
  │
  ├─ Need fast inference? (HFT)
  │   └─ Use: 2 states, diagonal covariance, minimal preprocessing
  │
  ├─ Limited memory?
  │   └─ Use: Diagonal/spherical covariance, batch processing, float32
  │
  ├─ Limited CPU?
  │   └─ Use: Fewer states, spherical covariance, data subsampling
  │
  ├─ High volatility market? (Crypto)
  │   └─ Use: 3 states, full covariance, lenient outlier threshold
  │
  ├─ Long-term analysis?
  │   └─ Use: 4 states, full covariance, extensive validation
  │
  └─ General purpose?
      └─ Use: 3 states, full covariance, standard preprocessing
```

### Performance vs Accuracy Trade-offs

| Configuration | Speed | Memory | Accuracy | Use Case |
|--------------|-------|--------|----------|----------|
| 2 states, diag | ⚡⚡⚡ | 💾 | ⭐⭐ | HFT, real-time |
| 3 states, diag | ⚡⚡ | 💾💾 | ⭐⭐⭐ | Balanced |
| 3 states, full | ⚡ | 💾💾💾 | ⭐⭐⭐⭐ | Standard |
| 4 states, full | ⚡ | 💾💾💾💾 | ⭐⭐⭐⭐⭐ | Research |

---

## Additional Resources

- [API Documentation](HMM_RESEARCH_API.md)
- [Best Practices](BEST_PRACTICES.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Integration Examples](INTEGRATION_EXAMPLES.md)
