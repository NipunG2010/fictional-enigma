# HMM Systematic Training Scripts

This directory contains scripts for systematic training and evaluation of Hidden Markov Models (HMM) for market regime detection.

## Overview

The systematic training pipeline trains HMM models with 2-4 states on multivariate observations `[s_LDC, s_MR, s_TSMOM]`, evaluates them comprehensively, and selects the best configuration for production deployment.

## Main Script: `train_hmm_systematic.py`

### Purpose

Orchestrates the complete HMM training pipeline:
1. **Data Loading & Validation**: Loads and validates signal observations from Parquet files
2. **Systematic Training**: Trains models with 2, 3, and 4 states using cross-validation
3. **Comprehensive Evaluation**: Evaluates models using AIC, BIC, CV scores, and interpretability metrics
4. **Model Selection**: Ranks models and selects the best configuration
5. **Reporting**: Generates detailed reports and saves production-ready artifacts

### Requirements

The script expects input data in Parquet format with the following columns:
- `s_LDC`: LDC (Laguerre Directional Change) signal
- `s_MR`: Mean Reversion signal
- `s_TSMOM`: Time Series Momentum signal

### Command-Line Arguments

#### Required Arguments

- `--data-path PATH`: Path to Parquet file containing `[s_LDC, s_MR, s_TSMOM]` observations
  - Must be a valid Parquet file
  - Must contain all three required signal columns
  - Example: `notebooks/processed_data/signals_processed.parquet`

#### Optional Arguments

- `--output-dir PATH`: Directory for saving artifacts and reports
  - Default: `output/hmm_training`
  - Creates directory if it doesn't exist
  - Example: `results/hmm_models`

- `--n-states N [N ...]`: List of state counts to train
  - Default: `2 3 4`
  - Can specify any positive integers
  - Example: `--n-states 2 3 4 5`

- `--cv-folds N`: Number of cross-validation folds
  - Default: `5`
  - Must be >= 2
  - Example: `--cv-folds 10`

### Usage Examples

#### Basic Usage

Train models with default settings (2-4 states, 5-fold CV):

```bash
cd py
python scripts/train_hmm_systematic.py \
    --data-path ../notebooks/processed_data/signals_processed.parquet
```

#### Custom Output Directory

Specify a custom output directory:

```bash
python scripts/train_hmm_systematic.py \
    --data-path ../notebooks/processed_data/signals_processed.parquet \
    --output-dir results/hmm_experiment_001
```

#### Custom State Range

Train models with 2, 3, 4, and 5 states:

```bash
python scripts/train_hmm_systematic.py \
    --data-path ../notebooks/processed_data/signals_processed.parquet \
    --n-states 2 3 4 5
```

#### Custom Cross-Validation

Use 10-fold cross-validation for more robust evaluation:

```bash
python scripts/train_hmm_systematic.py \
    --data-path ../notebooks/processed_data/signals_processed.parquet \
    --cv-folds 10
```

#### Full Custom Configuration

Combine all options:

```bash
python scripts/train_hmm_systematic.py \
    --data-path ../notebooks/processed_data/signals_processed.parquet \
    --output-dir results/hmm_full_experiment \
    --n-states 2 3 4 5 6 \
    --cv-folds 10
```

### Output Structure

The script creates the following output structure:

```
output/hmm_training/
├── hmm_2_states.json          # Model artifact for 2-state HMM
├── hmm_3_states.json          # Model artifact for 3-state HMM
├── hmm_4_states.json          # Model artifact for 4-state HMM
├── hmm_best.json              # Best model (copy with selection metadata)
└── training_report.json       # Comprehensive training report
```

#### Artifact Files (`hmm_*_states.json`)

Each artifact file contains:
- **Model Parameters**: Transition matrix, initial probabilities, means, covariances
- **Metadata**: AIC, BIC, log-likelihood, convergence information
- **Training Configuration**: Number of states, covariance type, random seed

#### Best Model (`hmm_best.json`)

The best model file includes everything from the individual artifact plus:
- **Selection Metadata**: Combined score, confidence score, ranking position
- **Component Scores**: Normalized AIC, BIC, CV, and interpretability scores
- **Justification**: Explanation of why this model was selected

#### Training Report (`training_report.json`)

Comprehensive report containing:
- **Timestamp**: When training was performed
- **Configuration**: Data path, state range, CV folds
- **Model Evaluations**: Detailed metrics for all trained models
  - Basic metrics (AIC, BIC, log-likelihood)
  - Cross-validation results
  - Regime characteristics
  - State persistence analysis
  - Economic interpretations
  - Interpretability scores
- **Rankings**: Models ranked by combined score with justifications

### Model Selection Criteria

Models are ranked using a weighted scoring system:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **AIC** | 30% | Statistical fit (lower is better) |
| **BIC** | 30% | Model complexity penalty (lower is better) |
| **CV Score** | 20% | Cross-validation log-likelihood (higher is better) |
| **Interpretability** | 20% | Regime distinctiveness and persistence (higher is better) |

#### Interpretability Score Components

The interpretability score combines:
- **Volatility Distinctiveness** (40%): How distinct regimes are from each other
- **State Persistence** (40%): How stable regimes are over time
- **Sample Adequacy** (20%): Whether sufficient data exists for each regime

#### Confidence Score

The confidence score indicates reliability of the selection:
- **High (>0.7)**: All metrics agree, model is production-ready
- **Moderate (0.5-0.7)**: Some metric disagreement, validate on additional data
- **Low (<0.5)**: Significant metric disagreement, consider more data or different approach

## Troubleshooting

### Common Issues

#### 1. Missing Required Columns

**Error**: `ValueError: Missing required columns: ['s_LDC', 's_MR', 's_TSMOM']`

**Solution**: Ensure your Parquet file contains all three required signal columns with exact names:
```python
import pandas as pd
df = pd.read_parquet('your_data.parquet')
print(df.columns)  # Should include s_LDC, s_MR, s_TSMOM
```

#### 2. NaN Values in Data

**Warning**: `Data contains X% NaN values`

**Solution**: The script automatically removes rows with NaN values. If too many rows are removed:
- Check data quality in source
- Ensure signals are properly calculated
- Consider imputation strategies before training

#### 3. Training Failures

**Error**: `✗ X_states training failed: ...`

**Solution**: Training can fail for several reasons:
- **Insufficient data**: Need at least 100+ observations per state
- **Poor initialization**: Try running again (random initialization)
- **Numerical instability**: Check for extreme values in data
- **Convergence issues**: Data may not fit HMM assumptions well

The script continues with remaining configurations even if one fails.

#### 4. All Training Failed

**Error**: `All training configurations failed!`

**Solution**:
1. Check data quality and shape:
   ```python
   df = pd.read_parquet('your_data.parquet')
   print(df[['s_LDC', 's_MR', 's_TSMOM']].describe())
   print(df[['s_LDC', 's_MR', 's_TSMOM']].isna().sum())
   ```
2. Ensure sufficient data (recommended: 500+ observations)
3. Check for extreme outliers or constant values
4. Verify signals are properly normalized/scaled

#### 5. Low Confidence Score

**Warning**: `⚠️ Low confidence score (X)`

**Solution**: Low confidence indicates metric disagreement:
- **Collect more data**: Increase sample size for more reliable estimates
- **Review data quality**: Check for outliers, missing values, or errors
- **Try different configurations**: Experiment with covariance types
- **Validate assumptions**: Ensure data fits HMM assumptions (Markov property)

#### 6. Memory Issues

**Error**: `MemoryError` or system slowdown

**Solution**:
- Reduce number of states: Use `--n-states 2 3` instead of `2 3 4 5 6`
- Reduce CV folds: Use `--cv-folds 3` instead of 10
- Subsample data if very large (>100k observations)
- Use a machine with more RAM

#### 7. Import Errors

**Error**: `ModuleNotFoundError: No module named 'imp'`

**Solution**: Ensure you're running from the `py/` directory and the package is installed:
```bash
cd py
pip install -e .
python scripts/train_hmm_systematic.py --data-path ...
```

### Data Quality Checks

Before running training, validate your data:

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet('your_data.parquet')

# Check required columns
required = ['s_LDC', 's_MR', 's_TSMOM']
print("Columns present:", all(col in df.columns for col in required))

# Check for NaN
print("\nNaN counts:")
print(df[required].isna().sum())

# Check data ranges
print("\nData ranges:")
print(df[required].describe())

# Check for constant values
print("\nStandard deviations:")
print(df[required].std())

# Check sample size
print(f"\nTotal observations: {len(df)}")
print(f"Valid observations: {len(df.dropna(subset=required))}")
```

### Performance Considerations

#### Training Time

Typical training times (on modern CPU):
- **2 states**: 1-2 minutes
- **3 states**: 2-4 minutes
- **4 states**: 3-6 minutes
- **5+ states**: 5-15 minutes

Total pipeline time: 10-30 minutes for 2-4 states with 5-fold CV

#### Scaling Recommendations

| Data Size | Recommended Settings |
|-----------|---------------------|
| < 1,000 obs | `--cv-folds 3`, states 2-3 |
| 1,000-10,000 obs | `--cv-folds 5`, states 2-4 (default) |
| 10,000-50,000 obs | `--cv-folds 5-10`, states 2-5 |
| > 50,000 obs | `--cv-folds 10`, states 2-6, consider subsampling |

## Integration with Notebooks

The systematic training script is designed to work seamlessly with the notebook workflow:

### From Notebook to Script

After exploring data in notebooks, run systematic training:

```python
# In notebook: Save processed data
df[['s_LDC', 's_MR', 's_TSMOM']].to_parquet('processed_data/signals.parquet')

# Then run script from terminal
!python scripts/train_hmm_systematic.py --data-path processed_data/signals.parquet
```

### From Script to Notebook

Load trained models in notebooks for analysis:

```python
import json
from imp.hmm.models import HMMArtifact

# Load best model
with open('output/hmm_training/hmm_best.json', 'r') as f:
    artifact_dict = json.load(f)
    
best_model = HMMArtifact(**artifact_dict)

# Load training report
with open('output/hmm_training/training_report.json', 'r') as f:
    report = json.load(f)

# Analyze results
print(f"Best model: {report['evaluation_summary']['rankings'][0]['config_name']}")
print(f"Combined score: {report['evaluation_summary']['rankings'][0]['combined_score']:.3f}")
```

## Related Files

- **`notebooks/07_systematic_hmm_training.ipynb`**: Interactive notebook version with visualizations
- **`tests/test_systematic_training.py`**: Unit and integration tests
- **`py/imp/hmm/trainer.py`**: Core HMM training functionality
- **`py/imp/hmm/regime_analysis.py`**: Regime characterization and interpretation
- **`py/imp/hmm/models.py`**: HMM artifact data models

## Advanced Usage

### Programmatic Usage

Use the `SystematicHMMTrainer` class directly in Python:

```python
from pathlib import Path
from scripts.train_hmm_systematic import SystematicHMMTrainer

# Create trainer
trainer = SystematicHMMTrainer(
    data_path=Path('data/signals.parquet'),
    output_dir=Path('results/experiment_001'),
    n_states_range=[2, 3, 4],
    cv_folds=5
)

# Run pipeline
results = trainer.run()

# Access results
best_model = results['best_model']
print(f"Best configuration: {best_model['config_name']}")
print(f"Combined score: {best_model['scores']['combined_score']:.3f}")
print(f"Confidence: {best_model['scores']['confidence_score']:.3f}")

# Access individual components
observations = trainer.load_and_validate_data()
training_results = trainer.train_all_configurations(observations)
evaluation_summary = trainer.evaluate_all_models(observations)
```

### Custom Evaluation Metrics

Extend the evaluation framework:

```python
class CustomHMMTrainer(SystematicHMMTrainer):
    def _calculate_interpretability_score(self, characteristics, persistence):
        # Custom interpretability calculation
        # Add your own logic here
        return custom_score
    
    def _rank_models(self, models):
        # Custom ranking logic
        # Adjust weights or add new criteria
        rankings = super()._rank_models(models)
        # Modify rankings...
        return rankings
```

### Batch Processing

Process multiple datasets:

```bash
#!/bin/bash
# batch_train.sh

for dataset in data/*.parquet; do
    name=$(basename "$dataset" .parquet)
    python scripts/train_hmm_systematic.py \
        --data-path "$dataset" \
        --output-dir "results/$name" \
        --n-states 2 3 4
done
```

## Best Practices

### 1. Data Preparation

- **Normalize signals**: Ensure signals are on comparable scales
- **Remove outliers**: Extreme values can affect training
- **Check stationarity**: HMM assumes stationary distributions
- **Sufficient data**: Aim for 500+ observations minimum

### 2. Model Selection

- **Start simple**: Begin with 2-3 states before trying more
- **Use cross-validation**: Always use CV for reliable estimates
- **Check interpretability**: High statistical fit doesn't guarantee usability
- **Validate out-of-sample**: Test selected model on held-out data

### 3. Production Deployment

- **Use `hmm_best.json`**: This file is production-ready
- **Monitor confidence**: Low confidence requires additional validation
- **Version artifacts**: Keep track of training dates and data versions
- **Document assumptions**: Record any data preprocessing or filtering

### 4. Iterative Improvement

- **Compare experiments**: Use different data periods or preprocessing
- **Track metrics**: Monitor AIC, BIC, and interpretability over time
- **A/B testing**: Compare new models against production baseline
- **Regular retraining**: Update models as new data becomes available

## Support

For issues or questions:
1. Check this README's troubleshooting section
2. Review test files in `tests/test_systematic_training.py`
3. Consult the design document at `.kiro/specs/hmm-systematic-training/design.md`
4. Check the requirements document at `.kiro/specs/hmm-systematic-training/requirements.md`

## Version History

- **v1.0**: Initial implementation with 2-4 state training, comprehensive evaluation, and model selection
