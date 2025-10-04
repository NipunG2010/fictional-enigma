# Regime Analysis Module

## Overview

The regime analysis module provides comprehensive tools for analyzing market regimes detected by Hidden Markov Models (HMMs). It translates statistical patterns into economic interpretations and actionable trading insights.

## Features

### 1. Regime Characterization
- Statistical analysis of each market state
- Volatility, trend strength, and mean reversion metrics
- Detailed feature statistics (mean, std, skewness, kurtosis)
- Sample counts and distribution analysis

### 2. State Persistence Analysis
- Duration statistics (mean, median, min, max)
- Transition frequency and probability calculation
- Stable period identification
- Regime stability assessment

### 3. Economic Interpretation
- Regime type classification (High Volatility, Trending, Mean Reverting, Neutral)
- Market condition assessment (Bull, Bear, Sideways)
- Risk level evaluation (Low, Medium, High)
- Trading recommendations based on regime characteristics
- Key characteristic identification

### 4. Feature Importance Analysis
- Feature contribution scoring
- Ranked feature importance
- Dominant feature identification
- Understanding regime drivers

### 5. Regime Validation
- Regime stability metrics
- State separability assessment
- Temporal consistency evaluation
- Event correlation analysis

### 6. Comprehensive Reporting
- Executive summary generation
- Dominant regime identification
- Risk assessment
- Actionable insights and recommendations

## Usage

### Basic Usage

```python
from imp.hmm.models import HMMArtifact
from imp.hmm.regime_analysis import RegimeAnalyzer
import numpy as np

# Load trained HMM artifact
artifact = HMMArtifact(...)

# Initialize analyzer
analyzer = RegimeAnalyzer(artifact)

# Decode state sequence from observations
state_sequence = model.predict(observations)

# Characterize regimes
characteristics = analyzer.characterize_regimes(observations, state_sequence)

# Analyze persistence
persistence = analyzer.analyze_state_persistence(state_sequence)

# Generate interpretations
interpretations = analyzer.interpret_regimes(characteristics, persistence)

# Analyze feature importance
feature_importance = analyzer.analyze_feature_importance(observations, state_sequence)

# Validate regimes
validation = analyzer.validate_regimes(observations, state_sequence)

# Generate comprehensive report
report = analyzer.generate_regime_report(observations, state_sequence)
```

### Advanced Usage with Event Correlation

```python
# Correlate regimes with known market events
timestamps = np.arange(len(observations))
known_events = [
    {'timestamp': 150, 'event': 'Market crash', 'type': 'volatility'},
    {'timestamp': 300, 'event': 'Policy change', 'type': 'regime_shift'}
]

validation = analyzer.validate_regimes(
    observations,
    state_sequence,
    timestamps,
    known_events
)

# Check event correlations
if validation.get('event_correlations'):
    for corr in validation['event_correlations']:
        print(f"Event: {corr['event']['event']}")
        print(f"Correlation: {corr['correlation_strength']}")
```

## Data Classes

### RegimeCharacteristics
Statistical characteristics of a market regime.

**Attributes:**
- `state_id`: State identifier
- `mean_values`: Mean values for each feature
- `std_values`: Standard deviations for each feature
- `volatility`: Overall volatility measure
- `trend_strength`: Trend strength score (0-1)
- `mean_reversion_score`: Mean reversion score (0-1)
- `sample_count`: Number of observations in this regime
- `feature_statistics`: Detailed statistics per feature

### StatePersistence
State persistence and transition statistics.

**Attributes:**
- `state_id`: State identifier
- `mean_duration`: Average duration in this state
- `median_duration`: Median duration
- `max_duration`: Maximum observed duration
- `min_duration`: Minimum observed duration
- `total_occurrences`: Number of times state was entered
- `stable_periods`: Count of periods exceeding stability threshold
- `transition_frequencies`: Count of transitions to other states
- `transition_probabilities`: Probability of transitioning to each state

### EconomicInterpretation
Economic interpretation of a market regime.

**Attributes:**
- `state_id`: State identifier
- `regime_type`: Classification (High Volatility, Trending, Mean Reverting, Neutral)
- `market_condition`: Market phase (Bull, Bear, Sideways)
- `risk_level`: Risk assessment (Low, Medium, High)
- `trading_recommendation`: Actionable trading advice
- `key_characteristics`: List of notable characteristics
- `correlated_events`: Events correlated with this regime

### FeatureImportance
Feature importance for regime characterization.

**Attributes:**
- `state_id`: State identifier
- `feature_scores`: Normalized importance scores per feature
- `ranked_features`: Features sorted by importance
- `dominant_features`: Features with >25% contribution

## Regime Classification Logic

### Regime Type Classification
- **High Volatility**: volatility > 0.5
- **Trending**: trend_strength > 0.6
- **Mean Reverting**: mean_reversion_score > 0.6
- **Neutral**: None of the above conditions met

### Market Condition Classification
- **Bull**: Average signal value > 0.3
- **Bear**: Average signal value < -0.3
- **Sideways**: Average signal value between -0.3 and 0.3

### Risk Level Assessment
- **High**: volatility > 0.7
- **Medium**: volatility between 0.3 and 0.7
- **Low**: volatility < 0.3

## Trading Recommendations

The module generates context-aware trading recommendations based on:

1. **Regime Type**
   - Trending: Favor momentum strategies
   - Mean Reverting: Favor mean reversion strategies
   - High Volatility: Reduce position sizes, increase stop losses

2. **Market Condition**
   - Bull: Consider long bias
   - Bear: Consider short bias or defensive positioning
   - Sideways: Range-bound strategies

3. **Risk Level**
   - High: Implement strict risk controls
   - Medium: Standard risk management
   - Low: May increase position sizes

4. **Persistence**
   - High persistence (>10 periods): Suitable for longer holding periods
   - Low persistence (<5 periods): Favor shorter timeframes

## Validation Metrics

### Regime Stability
Measures how frequently regimes change. Higher values indicate more stable regimes.

**Formula**: `1.0 - min(1.0, transition_rate * 2)`

### State Separability
Measures how well-separated the states are in feature space.

**Formula**: `between_variance / (between_variance + within_variance)`

### Temporal Consistency
Measures consistency of state distribution over time using entropy.

**Formula**: `1.0 - (entropy / max_entropy)`

## Report Structure

The comprehensive report includes:

```json
{
  "metadata": {
    "n_states": 3,
    "n_observations": 500,
    "feature_names": ["s_ldc", "s_mr", "s_tsmom"],
    "generated_at": "2025-01-04T12:00:00"
  },
  "regime_characteristics": {
    "0": { ... },
    "1": { ... },
    "2": { ... }
  },
  "state_persistence": {
    "0": { ... },
    "1": { ... },
    "2": { ... }
  },
  "economic_interpretations": {
    "0": { ... },
    "1": { ... },
    "2": { ... }
  },
  "feature_importance": {
    "0": { ... },
    "1": { ... },
    "2": { ... }
  },
  "validation": {
    "regime_stability": 0.85,
    "state_separability": 0.72,
    "temporal_consistency": 0.68
  },
  "summary": {
    "total_states": 3,
    "dominant_regime": { ... },
    "most_persistent_regime": { ... },
    "highest_risk_regime": { ... },
    "key_insights": [ ... ]
  }
}
```

## Examples

See `py/examples/regime_analysis_example.py` for a complete working example demonstrating all features.

## Testing

Comprehensive tests are available in `py/tests/test_regime_analysis.py`:

```bash
pytest py/tests/test_regime_analysis.py -v
```

## Integration with Research Workflow

The regime analysis module integrates seamlessly with the HMM research environment:

1. **Training**: Use `HMMTrainer` to train models
2. **Inference**: Use `HMMInference` for state prediction
3. **Analysis**: Use `RegimeAnalyzer` for comprehensive regime analysis
4. **Visualization**: Use `RegimeVisualizer` for plotting (see visualization module)
5. **Production**: Export insights for trading system integration

## Best Practices

1. **Data Quality**: Ensure observations are properly preprocessed and normalized
2. **State Sequence**: Use Viterbi algorithm for most likely state sequence
3. **Validation**: Always validate regime detection quality before deployment
4. **Event Correlation**: Correlate with known market events to validate interpretations
5. **Persistence Threshold**: Adjust `min_stable_duration` based on your trading timeframe
6. **Feature Engineering**: Include relevant features for your market and strategy

## Performance Considerations

- **Memory**: Regime analysis is memory-efficient, processing observations in batches
- **Computation**: Most operations are O(n) or O(n*k) where n=samples, k=states
- **Scalability**: Handles sequences of 10,000+ observations efficiently

## Future Enhancements

Potential future additions:
- Real-time regime monitoring
- Regime change detection algorithms
- Multi-timeframe regime analysis
- Regime-based portfolio optimization
- Machine learning-based regime classification
- Integration with external market indicators

## References

- Requirements 7.1-7.5 in `.kiro/specs/hmm-research-environment/requirements.md`
- Design document: `.kiro/specs/hmm-research-environment/design.md`
- Task 8 in `.kiro/specs/hmm-research-environment/tasks.md`
