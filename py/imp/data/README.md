# LDC Signal Data Integration and Preprocessing

This module provides comprehensive utilities for loading, preprocessing, validating, and engineering features from LDC signal data for HMM training.

## Components

### 1. LDCDataLoader
Load LDC signal data from various sources:
- Rust LDC engine output (parquet files)
- Partitioned data directories (symbol/date/interval structure)
- CSV files (legacy format)

```python
from imp.data import LDCDataLoader, LDCSignalConfig

# Configure loader
config = LDCSignalConfig(
    signals=['s_LDC', 's_MR', 's_TSMOM'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Load data
loader = LDCDataLoader(config)
df = loader.load_from_directory('rust/sample')

# Or load from specific file
df = loader.load_from_file('rust/sample/features.parquet')

# Get statistics
stats = loader.get_signal_statistics()
print(stats)
```

### 2. SignalPreprocessor
Preprocess signals for HMM training:
- Missing value imputation
- Outlier detection and treatment
- Scaling and normalization
- Trend removal
- Feature transformations

```python
from imp.data import SignalPreprocessor, PreprocessingConfig

# Configure preprocessing
config = PreprocessingConfig(
    scaling_method='standardize',
    handle_missing='forward_fill',
    outlier_method='zscore',
    outlier_threshold=3.0,
    outlier_action='clip',
    remove_trend=False
)

# Preprocess data
preprocessor = SignalPreprocessor(config)
df_processed = preprocessor.fit_transform(df)

# Get preprocessing statistics
stats = preprocessor.get_preprocessing_stats()

# Get recommendations
recommendations = preprocessor.get_recommendations(df)
for rec in recommendations:
    print(f"• {rec}")
```

### 3. DataValidator
Validate data quality for HMM training:
- Data completeness checks
- Numerical stability checks
- Temporal consistency checks
- Statistical property checks
- HMM-specific requirements

```python
from imp.data import DataValidator

# Create validator
validator = DataValidator(
    min_samples=100,
    max_missing_pct=10.0,
    check_stationarity=True,
    check_multicollinearity=True
)

# Validate data
report = validator.validate(df)

# Print report
report.print_summary()

# Quick check
is_valid = validator.quick_check(df)
```

### 4. FeatureEngineer
Engineer features for multivariate HMM observations:
- Returns and log returns
- Rolling statistics (mean, std, z-score)
- Momentum indicators
- Volatility measures
- Lagged features
- PCA components

```python
from imp.data import FeatureEngineer, FeatureConfig

# Configure feature engineering
config = FeatureConfig(
    add_returns=True,
    add_rolling_stats=True,
    rolling_windows=[5, 10, 20],
    add_momentum=True,
    add_volatility=True,
    add_lags=False,
    add_pca=False
)

# Engineer features
engineer = FeatureEngineer(config)
df_features = engineer.fit_transform(df_processed)

# Get feature importance
importance = engineer.get_feature_importance(df_features)
print(importance)

# Select top features
df_top = engineer.select_top_features(df_features, n_features=10, method='variance')
```

### 5. DataQualityReporter
Generate comprehensive data quality reports:
- Quality metrics
- Statistical summaries
- Temporal analysis
- Correlation analysis
- Actionable recommendations

```python
from imp.data import DataQualityReporter

# Create reporter
reporter = DataQualityReporter()

# Generate report
report = reporter.generate_report(
    df=df_processed,
    validation_report=validation_report,
    preprocessing_stats=preprocessor.get_preprocessing_stats(),
    feature_importance=engineer.get_feature_importance(df_features)
)

# Print report
reporter.print_report(detailed=True)

# Save report
reporter.save_report('reports/data_quality.json', format='json')
reporter.save_report('reports/data_quality.html', format='html')

# Create visualization dashboard
fig = reporter.plot_quality_dashboard(df_processed, save_path='reports/dashboard.png')
```

## Complete Pipeline Example

```python
from imp.data import (
    LDCDataLoader, LDCSignalConfig,
    SignalPreprocessor, PreprocessingConfig,
    DataValidator,
    FeatureEngineer, FeatureConfig,
    DataQualityReporter
)
import numpy as np

# 1. Load data
print("Step 1: Loading data...")
loader_config = LDCSignalConfig(
    signals=['s_LDC', 's_MR', 's_TSMOM'],
    features=['rsi', 'cci', 'adx']
)
loader = LDCDataLoader(loader_config)
df = loader.load_from_directory('rust/sample')

# 2. Validate raw data
print("\nStep 2: Validating data...")
validator = DataValidator()
validation_report = validator.validate(df)
validation_report.print_summary()

if not validation_report.is_valid:
    print("⚠️  Data validation failed. Proceeding with preprocessing...")

# 3. Preprocess data
print("\nStep 3: Preprocessing data...")
preproc_config = PreprocessingConfig(
    scaling_method='standardize',
    handle_missing='forward_fill',
    outlier_method='zscore',
    outlier_threshold=3.0,
    outlier_action='clip'
)
preprocessor = SignalPreprocessor(preproc_config)
df_processed = preprocessor.fit_transform(df)

# 4. Engineer features
print("\nStep 4: Engineering features...")
feature_config = FeatureConfig(
    add_returns=True,
    add_rolling_stats=True,
    rolling_windows=[5, 10],
    add_momentum=True,
    add_volatility=True
)
engineer = FeatureEngineer(feature_config)
df_features = engineer.fit_transform(df_processed)

# 5. Generate quality report
print("\nStep 5: Generating quality report...")
reporter = DataQualityReporter()
report = reporter.generate_report(
    df=df_features,
    validation_report=validation_report,
    preprocessing_stats=preprocessor.get_preprocessing_stats(),
    feature_importance=engineer.get_feature_importance(df_features)
)
reporter.print_report(detailed=True)

# 6. Prepare for HMM training
print("\nStep 6: Preparing for HMM training...")
observations = df_features.values
print(f"✓ Observations shape: {observations.shape}")
print(f"✓ Ready for HMM training!")

# Save processed data
df_features.to_parquet('processed_data/hmm_observations.parquet')
reporter.save_report('processed_data/quality_report.json')
print("\n✓ Pipeline complete!")
```

## Integration with HMM Training

After preprocessing, use the data with the HMM trainer:

```python
from imp.hmm import EnhancedHMMTrainer
from imp.data import LDCDataLoader, SignalPreprocessor

# Load and preprocess
loader = LDCDataLoader()
df = loader.load_from_directory('rust/sample')

preprocessor = SignalPreprocessor()
df_processed = preprocessor.fit_transform(df)

# Convert to observations
observations = df_processed.values

# Train HMM
trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn')
artifact, metrics = trainer.train_with_validation(observations)

print(f"Model trained: {metrics}")
```

## Configuration Best Practices

### For High-Frequency Trading Data
```python
PreprocessingConfig(
    scaling_method='robust',  # Robust to outliers
    handle_missing='forward_fill',
    outlier_method='iqr',
    outlier_threshold=1.5,
    outlier_action='clip',
    remove_trend=False
)
```

### For Daily/Weekly Data
```python
PreprocessingConfig(
    scaling_method='standardize',
    handle_missing='interpolate',
    outlier_method='zscore',
    outlier_threshold=3.0,
    outlier_action='remove',
    remove_trend=True  # Remove long-term trends
)
```

### For Noisy Signals
```python
FeatureConfig(
    add_returns=True,
    add_rolling_stats=True,
    rolling_windows=[10, 20, 50],  # Longer windows
    smooth_signals=True,
    smoothing_window=5
)
```

## Troubleshooting

### Issue: High missing data percentage
**Solution**: Use `handle_missing='interpolate'` or increase `max_missing_pct` in validator

### Issue: Non-stationary time series
**Solution**: Set `remove_trend=True` in PreprocessingConfig or use differencing

### Issue: Extreme outliers affecting training
**Solution**: Use `outlier_action='clip'` or `outlier_method='iqr'` with lower threshold

### Issue: Too many features after engineering
**Solution**: Use `engineer.select_top_features()` or enable PCA with `add_pca=True`

### Issue: Multicollinearity warnings
**Solution**: Enable PCA or manually remove highly correlated features

## Performance Considerations

- **Large datasets**: Use `load_from_file()` with specific columns instead of loading all data
- **Memory usage**: Process data in chunks for very large datasets
- **Feature engineering**: Disable unused feature types to speed up processing
- **Validation**: Set `check_stationarity=False` for faster validation on large datasets

## Requirements

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- pydantic >= 1.8.0
- statsmodels >= 0.12.0 (optional, for stationarity tests)
