# HMM Research Environment Troubleshooting Guide

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Training Problems](#training-problems)
3. [Data Loading Issues](#data-loading-issues)
4. [Visualization Problems](#visualization-problems)
5. [Performance Issues](#performance-issues)
6. [Artifact Management Issues](#artifact-management-issues)
7. [Jupyter Notebook Issues](#jupyter-notebook-issues)

---

## Installation Issues

### Problem: Missing Dependencies

**Symptom:**
```
ImportError: No module named 'hmmlearn'
```

**Solution:**
```bash
# Install all research dependencies
cd py
pip install -e ".[research]"

# Or install specific packages
pip install hmmlearn pomegranate jupyter ipywidgets plotly
```

### Problem: Pomegranate Not Available

**Symptom:**
```
WARNING: Pomegranate library not available. Only hmmlearn will be supported.
```

**Solution:**
Pomegranate is optional. If you need it:

```bash
pip install pomegranate

# For Python 3.9+, you may need:
pip install pomegranate==0.14.8
```

**Note:** If pomegranate installation fails, you can still use hmmlearn for all functionality.

### Problem: Jupyter Widgets Not Displaying

**Symptom:**
Interactive widgets show as plain text or don't render.

**Solution:**
```bash
# Enable jupyter widgets extension
jupyter nbextension enable --py widgetsnbextension

# For JupyterLab
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Restart Jupyter server
```

---

## Training Problems

### Problem: Model Fails to Converge

**Symptom:**
```
ModelTrainingError: Model failed to converge after 100 iterations
```

**Solutions:**

1. **Increase iterations:**
```python
artifact = trainer.train(
    observations,
    n_iterations=500  # Increase from default 100
)
```

2. **Adjust convergence threshold:**
```python
artifact = trainer.train(
    observations,
    convergence_threshold=1e-3  # Less strict than default 1e-4
)
```

3. **Try different initialization:**
```python
trainer = EnhancedHMMTrainer(
    n_states=3,
    random_state=42  # Try different seeds
)
```

4. **Reduce model complexity:**
```python
trainer = EnhancedHMMTrainer(
    n_states=2,  # Reduce from 3
    covariance_type='diag'  # Simpler than 'full'
)
```

### Problem: Numerical Instability

**Symptom:**
```
RuntimeWarning: invalid value encountered in log
```

**Solutions:**

1. **Check data quality:**
```python
from imp.data.validator import DataValidator

validator = DataValidator()
report = validator.validate(data)

if not report['is_valid']:
    print("Data issues:", report['issues'])
```

2. **Normalize data:**
```python
from imp.data.preprocessor import SignalPreprocessor

preprocessor = SignalPreprocessor()
processed_data, _ = preprocessor.preprocess(
    data,
    normalize=True,
    handle_outliers=True
)
```

3. **Remove extreme values:**
```python
# Clip extreme values
data_clipped = np.clip(data, -5, 5)
```

### Problem: Poor Model Performance

**Symptom:**
Low log-likelihood, high AIC/BIC, or unstable regimes.

**Solutions:**

1. **Validate data preprocessing:**
```python
# Check for missing values
print(f"Missing values: {np.isnan(data).sum()}")

# Check data distribution
import matplotlib.pyplot as plt
plt.hist(data.flatten(), bins=50)
plt.show()
```

2. **Try different number of states:**
```python
from imp.evaluation.evaluator import HMMEvaluator

evaluator = HMMEvaluator()

# Test multiple configurations
configs = [
    {'n_states': 2, 'covariance_type': 'full'},
    {'n_states': 3, 'covariance_type': 'full'},
    {'n_states': 4, 'covariance_type': 'full'},
]

results = evaluator.compare_models(observations, configs)
print(results)
```

3. **Use cross-validation:**
```python
results = evaluator.cross_validate(
    observations,
    trainer_config={'n_states': 3},
    cv_folds=5
)
```

---

## Data Loading Issues

### Problem: Cannot Load Parquet Files

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'signals.parquet'
```

**Solutions:**

1. **Check file path:**
```python
from pathlib import Path

file_path = Path('processed_data/signals_processed.parquet')
if not file_path.exists():
    print(f"File not found: {file_path}")
    print(f"Current directory: {Path.cwd()}")
```

2. **Use absolute paths:**
```python
from imp.data.ldc_loader import LDCDataLoader

loader = LDCDataLoader()
df = loader.load_signals('/absolute/path/to/signals.parquet')
```

### Problem: Missing Signal Columns

**Symptom:**
```
KeyError: 's_LDC'
```

**Solutions:**

1. **Check available columns:**
```python
df = loader.load_signals('signals.parquet')
print("Available columns:", df.columns.tolist())
```

2. **Load only available signals:**
```python
available_signals = ['s_LDC', 's_MR', 's_TSMOM']
existing_signals = [s for s in available_signals if s in df.columns]

df_subset = df[existing_signals]
```

### Problem: Data Type Issues

**Symptom:**
```
TypeError: Cannot convert data to numpy array
```

**Solutions:**

1. **Convert to numeric:**
```python
df = df.apply(pd.to_numeric, errors='coerce')
```

2. **Handle non-numeric columns:**
```python
# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols]
```

---

## Visualization Problems

### Problem: Plots Not Displaying in Jupyter

**Symptom:**
Plots don't show or show blank.

**Solutions:**

1. **Enable matplotlib inline:**
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

2. **For interactive plots:**
```python
%matplotlib widget
# or
%matplotlib notebook
```

3. **For plotly:**
```python
import plotly.io as pio
pio.renderers.default = 'notebook'
```

### Problem: Interactive Widgets Not Working

**Symptom:**
Widgets show but don't respond to interaction.

**Solutions:**

1. **Restart kernel:**
```
Kernel -> Restart & Clear Output
```

2. **Check widget installation:**
```python
import ipywidgets
print(ipywidgets.__version__)
```

3. **Re-enable extensions:**
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

### Problem: Memory Error with Large Plots

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Downsample data:**
```python
# Plot every 10th point
downsampled = state_probs[::10]
visualizer.plot_state_probabilities(downsampled)
```

2. **Use static plots:**
```python
fig = visualizer.plot_state_probabilities(
    state_probs,
    interactive=False  # Use matplotlib instead of plotly
)
```

---

## Performance Issues

### Problem: Training Takes Too Long

**Symptom:**
Training doesn't complete in reasonable time.

**Solutions:**

1. **Reduce data size:**
```python
# Use subset for initial experiments
observations_subset = observations[:1000]
artifact = trainer.train(observations_subset)
```

2. **Simplify model:**
```python
trainer = EnhancedHMMTrainer(
    n_states=2,  # Fewer states
    covariance_type='diag'  # Simpler covariance
)
```

3. **Reduce iterations:**
```python
artifact = trainer.train(
    observations,
    n_iterations=50  # Reduce from default
)
```

### Problem: High Memory Usage

**Symptom:**
System runs out of memory during training or inference.

**Solutions:**

1. **Process in batches:**
```python
batch_size = 1000
results = []

for i in range(0, len(observations), batch_size):
    batch = observations[i:i+batch_size]
    batch_result = inference.predict_proba(batch)
    results.append(batch_result)

state_probs = np.vstack(results)
```

2. **Use float32 instead of float64:**
```python
observations = observations.astype(np.float32)
```

3. **Clear unused variables:**
```python
import gc

del large_variable
gc.collect()
```

---

## Artifact Management Issues

### Problem: Cannot Save Artifact

**Symptom:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Check directory permissions:**
```python
from pathlib import Path

artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(parents=True, exist_ok=True)
```

2. **Use different directory:**
```python
from imp.hmm.artifact_management import ArtifactManager

manager = ArtifactManager(artifacts_dir='/tmp/artifacts')
```

### Problem: Artifact Validation Fails

**Symptom:**
```
ArtifactValidationError: Invalid transition matrix
```

**Solutions:**

1. **Check artifact structure:**
```python
print(f"States: {artifact.n_states}")
print(f"Features: {artifact.n_features}")
print(f"Transition matrix shape: {np.array(artifact.transition_matrix).shape}")
```

2. **Validate before saving:**
```python
from imp.hmm.artifact_management import ArtifactManager

manager = ArtifactManager()
validation_report = manager.validate_artifact(artifact)

if not validation_report['is_valid']:
    print("Validation errors:", validation_report['errors'])
```

### Problem: Version Conflicts

**Symptom:**
Multiple versions of same artifact causing confusion.

**Solutions:**

1. **List available versions:**
```python
versions = manager.list_versions('market_regime_detector')
print("Available versions:", versions)
```

2. **Load specific version:**
```python
artifact, metadata = manager.load_artifact(
    'market_regime_detector',
    version='v1.2.0'
)
```

3. **Clean old versions:**
```python
manager.delete_old_versions(
    'market_regime_detector',
    keep_latest=3
)
```

---

## Jupyter Notebook Issues

### Problem: Kernel Dies During Execution

**Symptom:**
```
The kernel appears to have died. It will restart automatically.
```

**Solutions:**

1. **Reduce memory usage:**
- Process data in smaller batches
- Clear unused variables
- Restart kernel periodically

2. **Increase kernel memory limit:**
```bash
# In jupyter_notebook_config.py
c.NotebookApp.max_buffer_size = 1073741824  # 1GB
```

3. **Check for infinite loops:**
Review code for potential infinite loops or recursive calls.

### Problem: Notebook Won't Save

**Symptom:**
```
Autosave failed! Saving to .ipynb_checkpoints
```

**Solutions:**

1. **Check disk space:**
```bash
df -h
```

2. **Save to different location:**
```
File -> Save As -> Choose different directory
```

3. **Export and re-import:**
```
File -> Download as -> Notebook (.ipynb)
```

### Problem: Cannot Import Custom Modules

**Symptom:**
```
ModuleNotFoundError: No module named 'imp'
```

**Solutions:**

1. **Install package in development mode:**
```bash
cd py
pip install -e .
```

2. **Add to Python path:**
```python
import sys
from pathlib import Path

# Add py directory to path
py_dir = Path.cwd().parent / 'py'
sys.path.insert(0, str(py_dir))
```

3. **Restart kernel:**
```
Kernel -> Restart
```

---

## Common Error Messages

### "Model did not converge"

**Cause:** Training stopped before reaching convergence threshold.

**Fix:** Increase iterations or relax convergence threshold.

### "Singular matrix"

**Cause:** Covariance matrix is not invertible, often due to insufficient data or collinearity.

**Fix:** Use diagonal covariance, add more data, or remove correlated features.

### "Invalid probability distribution"

**Cause:** Probabilities don't sum to 1 or contain invalid values.

**Fix:** Check data preprocessing and normalization.

### "Dimension mismatch"

**Cause:** Input data dimensions don't match model expectations.

**Fix:** Verify data shape matches (n_samples, n_features).

---

## Getting Help

If you encounter issues not covered here:

1. **Check logs:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Enable verbose output:**
```python
trainer = EnhancedHMMTrainer(n_states=3, verbose=True)
```

3. **Review examples:**
See `py/examples/` for working code examples.

4. **Check documentation:**
See [API Documentation](HMM_RESEARCH_API.md) for detailed usage.

5. **Validate environment:**
```python
from imp.utils.env_validator import validate_environment

report = validate_environment()
print(report)
```

---

## Debug Checklist

When troubleshooting, work through this checklist:

- [ ] All dependencies installed correctly
- [ ] Data loaded successfully and has correct shape
- [ ] Data preprocessed (normalized, no NaN/inf values)
- [ ] Model parameters are reasonable
- [ ] Sufficient training iterations
- [ ] Jupyter extensions enabled (for widgets)
- [ ] Enough memory available
- [ ] File paths are correct
- [ ] Artifact validation passes
- [ ] Error messages reviewed carefully

---

## Performance Optimization Tips

1. **Use appropriate data types:** float32 instead of float64
2. **Batch processing:** Process large datasets in chunks
3. **Parallel training:** Use multiple cores for hyperparameter search
4. **Cache results:** Save intermediate results to avoid recomputation
5. **Profile code:** Use `%timeit` and `%prun` to identify bottlenecks

```python
# Profile training
%timeit artifact = trainer.train(observations)

# Detailed profiling
%prun artifact = trainer.train(observations)
```
