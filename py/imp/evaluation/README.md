# HMM Evaluation Framework

Comprehensive model evaluation and comparison framework for Hidden Markov Models.

## Overview

The evaluation module provides tools for:
- Evaluating individual HMM models with multiple metrics
- Performing time series cross-validation
- Analyzing regime stability and persistence
- Comparing multiple model configurations
- Statistical significance testing
- Multi-criteria model selection

## Components

### HMMEvaluator

Main class for model evaluation and comparison.

```python
from imp.evaluation import HMMEvaluator

evaluator = HMMEvaluator(random_state=42)
```

### EvaluationMetrics

Dataclass containing evaluation metrics:
- `log_likelihood`: Model log-likelihood
- `aic`: Akaike Information Criterion
- `bic`: Bayesian Information Criterion
- `perplexity`: Model perplexity
- `n_parameters`: Number of model parameters
- `n_samples`: Number of samples evaluated

### RegimeStabilityMetrics

Dataclass containing regime stability analysis:
- `state_durations`: Duration of each regime occurrence
- `mean_durations`: Average duration per state
- `median_durations`: Median duration per state
- `max_durations`: Maximum duration per state
- `stable_periods`: Number of stable periods (≥ min_duration)
- `total_periods`: Total number of periods per state
- `transition_frequencies`: Frequency of state transitions
- `state_persistence`: Probability of remaining in each state

### ModelComparison

Dataclass for model comparison results:
- `config_name`: Configuration identifier
- `metrics`: Evaluation metrics
- `cv_scores`: Cross-validation scores (optional)
- `stability_metrics`: Regime stability metrics (optional)
- `rank`: Model ranking

## Usage Examples

### 1. Evaluate a Single Model

```python
from imp.evaluation import HMMEvaluator
from imp.hmm.trainer import EnhancedHMMTrainer
import numpy as np

# Generate or load data
observations = np.random.randn(300, 3)

# Train model
trainer = EnhancedHMMTrainer(n_states=3, library="hmmlearn")
trainer.train(observations, n_iterations=100)

# Evaluate
evaluator = HMMEvaluator()
metrics = evaluator.evaluate_model(trainer.trainer, observations)

print(f"Log-likelihood: {metrics.log_likelihood:.4f}")
print(f"BIC: {metrics.bic:.4f}")
```

### 2. Cross-Validation

```python
trainer_config = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'diag',
    'random_state': 42
}

cv_results = evaluator.cross_validate(
    observations,
    trainer_config,
    cv_folds=5,
    n_iterations=100
)

print(f"CV Log-likelihood: {cv_results['log_likelihood_mean']:.4f} ± {cv_results['log_likelihood_std']:.4f}")
```

### 3. Regime Stability Analysis

```python
# Get state probabilities
state_probs = trainer.trainer.predict_state_probabilities(observations)

# Analyze stability
stability = evaluator.regime_stability_analysis(state_probs, min_duration=10)

for state in range(3):
    print(f"State {state}:")
    print(f"  Mean duration: {stability.mean_durations[state]:.2f}")
    print(f"  Persistence: {stability.state_persistence[state]:.2%}")
```

### 4. Compare Multiple Models

```python
trainer_configs = [
    {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag'},
    {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag'},
    {'n_states': 4, 'library': 'hmmlearn', 'covariance_type': 'diag'},
]

comparison_df = evaluator.compare_models(
    observations,
    trainer_configs,
    n_iterations=100,
    perform_cv=True,
    cv_folds=5,
    analyze_stability=True
)

print(comparison_df)
```

### 5. Statistical Significance Testing

```python
# After comparing models
config_names = list(evaluator.evaluation_results.keys())

sig_result = evaluator.statistical_significance_test(
    config_names[0],
    config_names[1],
    metric='log_likelihood'
)

print(f"p-value: {sig_result['p_value']:.4f}")
print(f"Significant: {sig_result['significant']}")
```

### 6. Model Selection

```python
# Select best model by single criterion
best_bic = evaluator.select_best_model(criteria=['bic'])

# Select best model by multiple criteria with weights
best_weighted = evaluator.select_best_model(
    criteria=['bic', 'log_likelihood'],
    weights=[0.6, 0.4]
)

print(f"Best by BIC: {best_bic}")
print(f"Best by weighted criteria: {best_weighted}")
```

### 7. Get Evaluation Summary

```python
summary = evaluator.get_evaluation_summary()

print(f"Configurations evaluated: {summary['n_configurations']}")
print(f"Best by BIC: {summary['best_by_bic']}")
print(f"Best by AIC: {summary['best_by_aic']}")
print(f"Best by likelihood: {summary['best_by_likelihood']}")
```

## Evaluation Metrics

### Information Criteria

- **AIC (Akaike Information Criterion)**: `AIC = -2 * log_likelihood + 2 * n_parameters`
  - Lower is better
  - Penalizes model complexity
  - Good for prediction

- **BIC (Bayesian Information Criterion)**: `BIC = -2 * log_likelihood + n_parameters * log(n_samples)`
  - Lower is better
  - Stronger penalty for complexity than AIC
  - Good for model selection

### Log-Likelihood

- Measures how well the model fits the data
- Higher is better
- Can lead to overfitting without regularization

### Perplexity

- Measures model uncertainty
- Lower is better
- Exponential of negative log-likelihood per sample

## Cross-Validation

The framework uses `TimeSeriesSplit` from scikit-learn for proper time series cross-validation:
- Respects temporal ordering
- No data leakage from future to past
- Expanding window approach

## Regime Stability Analysis

Analyzes the stability and persistence of detected market regimes:

- **Duration Statistics**: How long each regime lasts
- **Transition Frequencies**: How often states transition
- **Persistence**: Probability of remaining in same state
- **Stable Periods**: Number of long-lasting regimes

## Statistical Significance Testing

Uses paired t-test to compare models:
- Tests if performance difference is statistically significant
- Calculates effect size (Cohen's d)
- Requires cross-validation scores

## Model Selection

Supports multi-criteria decision making:
- Combine multiple metrics with custom weights
- Automatic normalization of metrics
- Handles both "higher is better" and "lower is better" metrics

## Best Practices

1. **Always use cross-validation** for reliable performance estimates
2. **Consider multiple metrics** - no single metric tells the whole story
3. **Analyze regime stability** - unstable regimes may not be useful
4. **Test statistical significance** - ensure differences are meaningful
5. **Use appropriate sample sizes** - small datasets may give unreliable results

## Integration with Research Workflow

The evaluation framework integrates seamlessly with:
- `EnhancedHMMTrainer` for model training
- Jupyter notebooks for interactive analysis
- Parameter tuning framework for optimization
- Visualization tools for result presentation

## See Also

- [Model Evaluation Example](../../examples/model_evaluation_example.py)
- [HMM Trainer Documentation](../hmm/README.md)
- [Parameter Tuning Documentation](../tuning/README.md)
