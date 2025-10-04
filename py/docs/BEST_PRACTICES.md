# HMM Research Environment Best Practices

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Model Training](#model-training)
3. [Model Evaluation](#model-evaluation)
4. [Regime Analysis](#regime-analysis)
5. [Visualization](#visualization)
6. [Artifact Management](#artifact-management)
7. [Production Deployment](#production-deployment)
8. [Research Workflow](#research-workflow)

---

## Data Preparation

### 1. Always Validate Data Quality

Before training, validate your data:

```python
from imp.data.validator import DataValidator

validator = DataValidator()
report = validator.validate(data)

if not report['is_valid']:
    print("Data quality issues found:")
    for issue in report['issues']:
        print(f"  - {issue}")
```

**Why:** Poor data quality leads to unreliable models and wasted training time.

### 2. Handle Missing Values Appropriately

Choose the right strategy for your use case:

```python
from imp.data.preprocessor import SignalPreprocessor

preprocessor = SignalPreprocessor()

# For time series with temporal continuity
processed_data, _ = preprocessor.preprocess(
    data,
    handle_missing='forward_fill'  # Preserves temporal structure
)

# For data with clear trends
processed_data, _ = preprocessor.preprocess(
    data,
    handle_missing='interpolate'  # Smooth interpolation
)

# When missing data is minimal
processed_data, _ = preprocessor.preprocess(
    data,
    handle_missing='drop'  # Remove rows with missing values
)
```

**Why:** Different missing value strategies affect model behavior differently.

### 3. Normalize Features

Always normalize features before training:

```python
processed_data, metadata = preprocessor.preprocess(
    data,
    normalize=True,
    handle_outliers=True,
    outlier_threshold=3.0
)

# Save normalization parameters for production
print(f"Normalization params: {metadata['normalization']}")
```

**Why:** HMMs are sensitive to feature scales. Normalization ensures all features contribute equally.

### 4. Split Data Chronologically

For time series, always split chronologically:

```python
# CORRECT: Chronological split
split_idx = int(len(observations) * 0.8)
train_data = observations[:split_idx]
test_data = observations[split_idx:]

# INCORRECT: Random split (breaks temporal structure)
# train_data, test_data = train_test_split(observations)  # DON'T DO THIS
```

**Why:** Random splits leak future information into training, causing overfitting.

### 5. Check for Data Leakage

Ensure no future information leaks into training:

```python
# CORRECT: Only use past data for normalization
train_mean = train_data.mean(axis=0)
train_std = train_data.std(axis=0)

train_normalized = (train_data - train_mean) / train_std
test_normalized = (test_data - train_mean) / train_std  # Use train stats

# INCORRECT: Using test data statistics
# all_mean = observations.mean(axis=0)  # DON'T DO THIS
```

**Why:** Using future data for preprocessing inflates performance metrics.

---

## Model Training

### 1. Start Simple, Then Increase Complexity

Begin with simple models:

```python
# Start with 2 states and diagonal covariance
trainer = EnhancedHMMTrainer(
    n_states=2,
    covariance_type='diag',
    random_state=42
)

artifact = trainer.train(observations)

# If performance is insufficient, increase complexity
trainer = EnhancedHMMTrainer(
    n_states=3,
    covariance_type='full',
    random_state=42
)
```

**Why:** Simple models train faster and are less prone to overfitting.

### 2. Use Multiple Random Seeds

Train with different random seeds:

```python
results = []

for seed in [42, 123, 456, 789, 1011]:
    trainer = EnhancedHMMTrainer(
        n_states=3,
        random_state=seed
    )
    artifact = trainer.train(observations)
    
    # Evaluate
    inference = HMMInference(artifact)
    log_likelihood = inference.score(test_data)
    
    results.append({
        'seed': seed,
        'log_likelihood': log_likelihood,
        'artifact': artifact
    })

# Select best model
best_result = max(results, key=lambda x: x['log_likelihood'])
best_artifact = best_result['artifact']
```

**Why:** HMM training is sensitive to initialization. Multiple seeds help find better local optima.

### 3. Monitor Convergence

Track training progress:

```python
# Enable verbose output
trainer = EnhancedHMMTrainer(n_states=3, verbose=True)

artifact = trainer.train(
    observations,
    n_iterations=200,
    convergence_threshold=1e-4
)

# Check if converged
if artifact.metadata.get('converged', False):
    print(f"Converged in {artifact.metadata['n_iterations']} iterations")
else:
    print("Warning: Model did not converge. Consider increasing iterations.")
```

**Why:** Non-converged models may have unstable parameters.

### 4. Use Validation Split

Always validate during training:

```python
artifact, metrics = trainer.train_with_validation(
    observations,
    validation_split=0.2,
    n_iterations=100
)

print(f"Training log-likelihood: {metrics['train_log_likelihood']:.4f}")
print(f"Validation log-likelihood: {metrics['log_likelihood']:.4f}")

# Check for overfitting
if metrics['train_log_likelihood'] - metrics['log_likelihood'] > 100:
    print("Warning: Possible overfitting detected")
```

**Why:** Validation helps detect overfitting early.

### 5. Save Training Configuration

Always save configuration with artifacts:

```python
from imp.hmm.artifact_management import ArtifactManager

manager = ArtifactManager()

training_config = {
    'n_states': 3,
    'covariance_type': 'full',
    'n_iterations': 100,
    'convergence_threshold': 1e-4,
    'random_seed': 42,
    'library': 'hmmlearn'
}

manager.save_artifact(
    artifact,
    name='market_regime_detector',
    metadata={
        'training_config': training_config,
        'training_date': '2025-01-15',
        'data_samples': len(observations),
        'validation_metrics': metrics
    }
)
```

**Why:** Reproducibility is critical for research and debugging.

---

## Model Evaluation

### 1. Use Cross-Validation

Always use time series cross-validation:

```python
from imp.evaluation.evaluator import HMMEvaluator

evaluator = HMMEvaluator()

config = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'full'
}

cv_results = evaluator.cross_validate(
    observations,
    trainer_config=config,
    cv_folds=5,
    gap=10  # Gap between train and validation
)

print(f"Mean CV score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
```

**Why:** Single train/test split can be misleading. CV provides robust estimates.

### 2. Compare Multiple Metrics

Don't rely on a single metric:

```python
metrics_to_track = {
    'log_likelihood': metrics['log_likelihood'],
    'aic': metrics['aic'],
    'bic': metrics['bic'],
    'regime_stability': analyzer.calculate_stability_score(state_probs)
}

print("Model Performance:")
for metric, value in metrics_to_track.items():
    print(f"  {metric}: {value:.4f}")
```

**Why:** Different metrics capture different aspects of model quality.

### 3. Analyze Regime Stability

Check if regimes are meaningful:

```python
from imp.hmm.regime_analysis import RegimeAnalyzer

analyzer = RegimeAnalyzer(artifact)
analysis = analyzer.analyze_regimes(observations, state_probs)

# Check average state duration
for state, stats in analysis['state_durations'].items():
    print(f"State {state}: avg duration = {stats['mean']:.1f} periods")

# Regimes should persist for meaningful periods
if any(stats['mean'] < 5 for stats in analysis['state_durations'].values()):
    print("Warning: Some regimes are too short-lived")
```

**Why:** Very short-lived regimes may indicate overfitting or noise.

### 4. Validate Economic Interpretation

Ensure regimes make economic sense:

```python
# Get regime characteristics
state_stats = analyzer.calculate_state_statistics(observations, state_sequence)

for state, stats in state_stats.items():
    print(f"\nState {state}:")
    print(f"  Mean return: {stats['mean_return']:.4f}")
    print(f"  Volatility: {stats['volatility']:.4f}")
    print(f"  Trend strength: {stats['trend']:.4f}")

# Check if regimes are distinct
# High volatility state should have higher volatility than low volatility state
```

**Why:** Models should capture real market dynamics, not just fit noise.

### 5. Test on Out-of-Sample Data

Always test on truly unseen data:

```python
# Split data: train (60%), validation (20%), test (20%)
n = len(observations)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train_data = observations[:train_end]
val_data = observations[train_end:val_end]
test_data = observations[val_end:]

# Train on train, tune on validation, final test on test
artifact = trainer.train(train_data)

# Final evaluation on test set
inference = HMMInference(artifact)
test_score = inference.score(test_data)

print(f"Test set log-likelihood: {test_score:.4f}")
```

**Why:** True generalization performance is only measured on completely unseen data.

---

## Regime Analysis

### 1. Visualize Before Interpreting

Always visualize regimes first:

```python
from imp.visualization.regime_visualizer import RegimeVisualizer

visualizer = RegimeVisualizer(artifact)

# Plot state probabilities
fig = visualizer.plot_state_probabilities(
    state_probs,
    timestamps=timestamps,
    interactive=True
)
fig.show()

# Plot transition matrix
fig = visualizer.plot_transition_matrix()
plt.show()
```

**Why:** Visual inspection reveals patterns that statistics might miss.

### 2. Analyze Transition Patterns

Understand how regimes transition:

```python
transition_matrix = np.array(artifact.transition_matrix)

# Identify persistent states (high diagonal values)
persistence = np.diag(transition_matrix)
print("State persistence probabilities:", persistence)

# Identify common transitions
for i in range(artifact.n_states):
    for j in range(artifact.n_states):
        if i != j and transition_matrix[i, j] > 0.1:
            print(f"Common transition: State {i} -> State {j} ({transition_matrix[i, j]:.3f})")
```

**Why:** Transition patterns reveal regime dynamics.

### 3. Calculate Regime Statistics

Characterize each regime:

```python
analyzer = RegimeAnalyzer(artifact)
state_stats = analyzer.calculate_state_statistics(observations, state_sequence)

# Create regime profiles
for state, stats in state_stats.items():
    profile = {
        'volatility': 'high' if stats['volatility'] > 0.02 else 'low',
        'trend': 'strong' if abs(stats['trend']) > 0.01 else 'weak',
        'mean_reversion': 'high' if stats['mean_reversion'] > 0.5 else 'low'
    }
    print(f"State {state} profile: {profile}")
```

**Why:** Statistical characterization helps interpret regimes.

### 4. Validate Against Known Events

Compare regimes with known market events:

```python
# Mark known high volatility periods
high_vol_periods = [
    ('2024-03-10', '2024-03-15'),  # Market crash
    ('2024-08-01', '2024-08-05'),  # Fed announcement
]

# Check if high volatility state aligns with these periods
# This requires timestamp alignment
```

**Why:** Validation against known events builds confidence in the model.

### 5. Document Regime Interpretation

Always document your interpretation:

```python
regime_interpretation = {
    'state_0': {
        'name': 'Low Volatility Trending',
        'characteristics': 'Low volatility, positive trend, low mean reversion',
        'trading_implication': 'Favor trend-following strategies'
    },
    'state_1': {
        'name': 'High Volatility Mean Reverting',
        'characteristics': 'High volatility, weak trend, high mean reversion',
        'trading_implication': 'Favor mean reversion strategies'
    },
    'state_2': {
        'name': 'Neutral',
        'characteristics': 'Medium volatility, no clear trend',
        'trading_implication': 'Reduce position sizes'
    }
}

# Save with artifact
manager.save_artifact(
    artifact,
    name='market_regime_detector',
    metadata={'regime_interpretation': regime_interpretation}
)
```

**Why:** Documentation ensures consistent interpretation across team members.

---

## Visualization

### 1. Use Interactive Plots for Exploration

During exploration, use interactive plots:

```python
# Interactive plotly plot
fig = visualizer.plot_state_probabilities(
    state_probs,
    interactive=True
)
fig.show()
```

**Why:** Interactive plots allow zooming and detailed inspection.

### 2. Use Static Plots for Reports

For reports and papers, use static plots:

```python
# High-quality matplotlib plot
fig = visualizer.plot_state_probabilities(
    state_probs,
    interactive=False
)

# Save with high DPI
fig.savefig('regime_analysis.png', dpi=300, bbox_inches='tight')
```

**Why:** Static plots are more reliable for publication.

### 3. Add Context to Plots

Always add meaningful labels and titles:

```python
fig = visualizer.plot_state_probabilities(
    state_probs,
    timestamps=timestamps,
    title='BTC/USDT Market Regime Detection (5m intervals)'
)

# Add annotations for important events
ax = fig.gca()
ax.axvline(x=crash_date, color='red', linestyle='--', label='Market Crash')
ax.legend()
```

**Why:** Context makes plots more informative.

### 4. Use Consistent Color Schemes

Maintain consistent colors across plots:

```python
state_colors = {
    0: '#1f77b4',  # Blue for low volatility
    1: '#ff7f0e',  # Orange for high volatility
    2: '#2ca02c',  # Green for neutral
}

# Use in all visualizations
visualizer = RegimeVisualizer(artifact, state_colors=state_colors)
```

**Why:** Consistency aids interpretation across multiple plots.

### 5. Create Comprehensive Dashboards

For presentations, create dashboards:

```python
dashboard = visualizer.create_regime_dashboard(
    observations,
    state_probs,
    timestamps
)

display(dashboard)
```

**Why:** Dashboards provide holistic view of regime analysis.

---

## Artifact Management

### 1. Use Semantic Versioning

Version artifacts semantically:

```python
# Major.Minor.Patch
# Major: Breaking changes (different n_states, features)
# Minor: Improvements (better training, more data)
# Patch: Bug fixes, minor tweaks

manager.save_artifact(
    artifact,
    name='market_regime_detector',
    version='1.2.0',  # Semantic version
    metadata={'changelog': 'Improved training convergence'}
)
```

**Why:** Clear versioning prevents confusion.

### 2. Always Validate Before Saving

Validate artifacts before saving:

```python
validation_report = manager.validate_artifact(artifact, validation_data=test_data)

if validation_report['is_valid']:
    manager.save_artifact(artifact, name='production_model')
else:
    print("Validation failed:", validation_report['errors'])
    # Fix issues before saving
```

**Why:** Invalid artifacts cause production failures.

### 3. Include Comprehensive Metadata

Save rich metadata:

```python
metadata = {
    'training_date': datetime.now().isoformat(),
    'data_source': 'BTCUSDT_5m',
    'data_period': '2024-01-01 to 2024-12-31',
    'training_samples': len(train_data),
    'validation_samples': len(val_data),
    'training_config': training_config,
    'performance_metrics': metrics,
    'regime_interpretation': regime_interpretation,
    'created_by': 'researcher_name',
    'notes': 'Trained on full year of data with improved preprocessing'
}

manager.save_artifact(artifact, name='model', metadata=metadata)
```

**Why:** Metadata enables reproducibility and debugging.

### 4. Maintain Artifact Registry

Keep track of all artifacts:

```python
# Create artifact registry
registry = {
    'production': {
        'name': 'market_regime_detector',
        'version': '1.2.0',
        'deployed_date': '2025-01-15',
        'performance': {'aic': 1234.5}
    },
    'candidate': {
        'name': 'market_regime_detector',
        'version': '1.3.0',
        'status': 'testing',
        'performance': {'aic': 1220.1}
    }
}

# Save registry
with open('artifacts/registry.json', 'w') as f:
    json.dump(registry, f, indent=2)
```

**Why:** Registry provides overview of model lifecycle.

### 5. Clean Up Old Artifacts

Regularly clean up old artifacts:

```python
# Keep only last 5 versions
manager.delete_old_versions(
    'market_regime_detector',
    keep_latest=5
)

# Archive old artifacts instead of deleting
manager.archive_old_versions(
    'market_regime_detector',
    archive_dir='artifacts/archive',
    keep_latest=3
)
```

**Why:** Prevents disk space issues and confusion.

---

## Production Deployment

### 1. Test Thoroughly Before Deployment

Run comprehensive tests:

```python
# Test artifact loading
artifact, metadata = manager.load_artifact('production_model')

# Test inference
inference = HMMInference(artifact)
test_probs = inference.predict_proba(test_data)

# Test performance
test_score = inference.score(test_data)
assert test_score > threshold, "Performance below threshold"

# Test integration with production system
# (Run integration tests here)
```

**Why:** Production failures are costly.

### 2. Use Canary Deployment

Deploy gradually:

```python
# Deploy to 10% of traffic first
deployment_config = {
    'model_version': '1.3.0',
    'traffic_percentage': 10,
    'rollback_threshold': 0.95  # Rollback if performance drops below 95%
}

# Monitor performance
# If successful, increase to 50%, then 100%
```

**Why:** Gradual deployment limits risk.

### 3. Monitor Production Performance

Track performance in production:

```python
from imp.tuning.performance_tracking import PerformanceTracker

tracker = PerformanceTracker()

# Log predictions
tracker.log_prediction(
    model_version='1.3.0',
    timestamp=datetime.now(),
    prediction=state_probs,
    actual_outcome=actual_regime  # If available
)

# Check for performance degradation
report = tracker.generate_report(days=7)
if report['performance_drop'] > 0.1:
    print("Warning: Performance degradation detected")
```

**Why:** Models can degrade over time.

### 4. Maintain Model Documentation

Document production models:

```python
production_docs = {
    'model_name': 'market_regime_detector',
    'version': '1.3.0',
    'deployment_date': '2025-01-15',
    'training_data': 'BTCUSDT 5m, 2024-01-01 to 2024-12-31',
    'performance': {
        'validation_log_likelihood': -1234.5,
        'test_log_likelihood': -1245.2
    },
    'regime_interpretation': regime_interpretation,
    'known_limitations': [
        'Performance degrades during extreme volatility',
        'Requires at least 1000 samples for reliable inference'
    ],
    'monitoring_metrics': ['log_likelihood', 'regime_stability'],
    'rollback_procedure': 'Revert to v1.2.0 if log_likelihood drops below -1300'
}

# Save documentation
with open('docs/production_model_v1.3.0.json', 'w') as f:
    json.dump(production_docs, f, indent=2)
```

**Why:** Documentation enables effective operations.

### 5. Plan for Model Updates

Establish update cadence:

```python
update_schedule = {
    'retraining_frequency': 'monthly',
    'evaluation_frequency': 'weekly',
    'data_refresh': 'daily',
    'next_retrain_date': '2025-02-15',
    'retrain_triggers': [
        'Performance drops below threshold',
        'New market regime detected',
        'Significant data distribution shift'
    ]
}
```

**Why:** Regular updates maintain model relevance.

---

## Research Workflow

### 1. Follow Structured Workflow

Use consistent research process:

```
1. Data Exploration → 2. Preprocessing → 3. Model Training →
4. Evaluation → 5. Regime Analysis → 6. Validation →
7. Documentation → 8. Production Deployment
```

**Why:** Structured workflow ensures nothing is missed.

### 2. Use Notebooks for Exploration

Leverage Jupyter notebooks:

```python
# Start with exploration notebooks
# notebooks/01_data_exploration.ipynb
# notebooks/02_hmm_training_comparison.ipynb

# Move to production code once approach is validated
# py/imp/hmm/
```

**Why:** Notebooks are great for exploration, code is better for production.

### 3. Version Control Everything

Use git for version control:

```bash
# Track notebooks
git add notebooks/*.ipynb

# Track code
git add py/imp/

# Track artifacts (use git-lfs for large files)
git lfs track "artifacts/*.pkl"
git add artifacts/*.pkl

# Commit with meaningful messages
git commit -m "Add regime detection model v1.3.0 with improved convergence"
```

**Why:** Version control enables collaboration and reproducibility.

### 4. Document Experiments

Keep experiment log:

```python
experiment_log = {
    'experiment_id': 'exp_20250115_001',
    'date': '2025-01-15',
    'hypothesis': 'Using 4 states will better capture market regimes',
    'configuration': {
        'n_states': 4,
        'covariance_type': 'full'
    },
    'results': {
        'log_likelihood': -1234.5,
        'aic': 2500.1,
        'regime_stability': 0.85
    },
    'conclusion': 'No improvement over 3-state model. Stick with 3 states.',
    'next_steps': 'Try different covariance types'
}

# Save to experiment log
with open('experiments/log.jsonl', 'a') as f:
    f.write(json.dumps(experiment_log) + '\n')
```

**Why:** Experiment logs prevent repeating failed approaches.

### 5. Collaborate Effectively

Share work with team:

```python
# Create shareable reports
from imp.tuning.report_generator import ReportGenerator

generator = ReportGenerator()
report = generator.generate_comprehensive_report(
    artifact,
    observations,
    state_probs,
    output_format='html'
)

# Share report
report.save('reports/regime_analysis_2025_01_15.html')
```

**Why:** Clear communication accelerates research.

---

## Summary Checklist

Before deploying a model to production:

- [ ] Data quality validated
- [ ] Features normalized
- [ ] Multiple random seeds tested
- [ ] Cross-validation performed
- [ ] Multiple metrics evaluated
- [ ] Regime stability analyzed
- [ ] Economic interpretation validated
- [ ] Visualizations created
- [ ] Artifact validated
- [ ] Comprehensive metadata saved
- [ ] Documentation completed
- [ ] Integration tests passed
- [ ] Performance monitoring setup
- [ ] Rollback plan established
- [ ] Team review completed

---

## Additional Resources

- [API Documentation](HMM_RESEARCH_API.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Integration Examples](INTEGRATION_EXAMPLES.md)
- [Tutorial Notebooks](../notebooks/README.md)
