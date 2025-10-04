# Quick Reference Guide

Quick reference for common tasks in the IMP HMM Research Environment.

## Setup Commands

```bash
# Initial setup
cd py
./setup_research_env.sh  # Linux/macOS
# or
setup_research_env.bat   # Windows

# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate.bat # Windows

# Validate environment
python -m imp.utils.env_validator

# Start Jupyter
jupyter lab
```

## Common Development Tasks

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific categories
pytest tests/ -m research
pytest tests/ -m notebook
pytest tests/ -m integration
pytest tests/ -m "not slow"

# With coverage
pytest tests/ --cov=imp --cov-report=html

# Specific file
pytest tests/test_hmm.py -v

# Specific test
pytest tests/test_hmm.py::test_hmm_training -v
```

### Code Quality

```bash
# Format code
black imp/ tests/
isort imp/ tests/

# Check formatting
black --check imp/ tests/
isort --check-only imp/ tests/

# Lint
flake8 imp/ tests/

# Type check
mypy imp/
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Stage changes
git add .

# Commit
git commit -m "feat: your feature description"

# Push
git push origin feature/your-feature

# Update from main
git checkout main
git pull origin main
git checkout feature/your-feature
git rebase main
```

## Python API Quick Reference

### HMM Training

```python
from imp.hmm.trainer import EnhancedHMMTrainer

# Create trainer
trainer = EnhancedHMMTrainer(
    n_states=3,
    library='hmmlearn',  # or 'pomegranate'
    covariance_type='full'
)

# Train with validation
artifact, metrics = trainer.train_with_validation(
    observations,
    validation_split=0.2
)

# Access results
print(f"Log-likelihood: {metrics['log_likelihood']}")
print(f"AIC: {metrics['aic']}")
print(f"BIC: {metrics['bic']}")
```

### Data Loading

```python
from imp.data.ldc_loader import LDCDataLoader

# Load LDC signals
loader = LDCDataLoader()
signals_df = loader.load_signals("path/to/signals.parquet")

# Preprocess for HMM
observations = loader.prepare_hmm_observations(
    signals_df,
    features=['s_LDC', 's_MR', 's_TSMOM']
)
```

### Visualization

```python
from imp.visualization.regime_visualizer import RegimeVisualizer

# Create visualizer
viz = RegimeVisualizer(artifact)

# Plot state probabilities
fig = viz.plot_state_probabilities(
    state_probs,
    timestamps=timestamps,
    interactive=True
)

# Plot transition matrix
fig = viz.plot_transition_matrix(annotate=True)

# Create dashboard
dashboard = viz.create_regime_dashboard(
    observations,
    state_probs,
    timestamps
)
```

### Regime Analysis

```python
from imp.hmm.regime_analysis import RegimeAnalyzer

# Create analyzer
analyzer = RegimeAnalyzer(artifact)

# Analyze regimes
analysis = analyzer.analyze_regimes(
    observations,
    state_sequence
)

# Get regime statistics
stats = analyzer.get_regime_statistics(state_sequence)

# Economic interpretation
interpretation = analyzer.interpret_regimes(
    observations,
    state_sequence,
    market_data
)
```

### Evaluation

```python
from imp.evaluation.evaluator import HMMEvaluator

# Create evaluator
evaluator = HMMEvaluator()

# Cross-validation
cv_results = evaluator.cross_validate(
    observations,
    trainer_configs=[
        {'n_states': 2, 'library': 'hmmlearn'},
        {'n_states': 3, 'library': 'hmmlearn'},
        {'n_states': 3, 'library': 'pomegranate'},
    ],
    cv_folds=5
)

# Regime stability analysis
stability = evaluator.regime_stability_analysis(
    state_probs,
    min_duration=5
)
```

### Artifact Management

```python
from imp.hmm.artifact_management import (
    ResearchArtifact,
    ExperimentTracker
)

# Create research artifact
research_artifact = ResearchArtifact(
    base_artifact=artifact,
    training_config=config,
    evaluation_metrics=metrics,
    experiment_id="exp_001",
    researcher="Your Name"
)

# Track experiment
tracker = ExperimentTracker(Path("experiments"))
exp_id = tracker.log_experiment(
    research_artifact,
    notes="Initial experiment"
)
```

## Jupyter Notebook Snippets

### Setup Cell

```python
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# IMP imports
from imp.hmm.trainer import EnhancedHMMTrainer
from imp.visualization.regime_visualizer import RegimeVisualizer
from imp.data.ldc_loader import LDCDataLoader

# Configuration
%matplotlib inline
%load_ext autoreload
%autoreload 2

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
```

### Interactive Parameter Tuning

```python
from imp.tuning.parameter_tuner import HMMParameterTuner

# Create tuner
tuner = HMMParameterTuner(observations)

# Display tuning interface
interface = tuner.create_tuning_interface()
display(interface)

# Access results
results = tuner.results
```

### Progress Bar

```python
from tqdm.notebook import tqdm

for i in tqdm(range(100), desc="Processing"):
    # Your code here
    pass
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code
    pass

# Run and check output
memory_intensive_function()
```

## Configuration Files

### pyproject.toml

```toml
[project]
name = "imp-python"
version = "0.1.0"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    # ... other dependencies
]

[project.optional-dependencies]
dev = ["pytest>=7.4.0", "black>=23.0.0"]
research = ["jupyter>=1.0.0", "plotly>=5.15.0"]
```

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: slow tests
    integration: integration tests
    notebook: notebook tests
    research: research framework tests
```

### .gitignore

```
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data
*.parquet
*.csv
*.npy
processed_data/
notebooks/processed_data/

# IDE
.vscode/
.idea/

# Environment
.env
.env.local
```

## Troubleshooting

### Import Errors

```bash
# Reinstall package
pip install -e .

# Check PYTHONPATH
echo $PYTHONPATH

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Jupyter Kernel Issues

```bash
# List kernels
jupyter kernelspec list

# Remove kernel
jupyter kernelspec remove imp-research

# Reinstall kernel
python -m ipykernel install --user --name=imp-research
```

### Widget Display Issues

```bash
# Enable widgets
jupyter nbextension enable --py widgetsnbextension

# For JupyterLab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab build
```

### Memory Issues

```python
# Clear variables
%reset

# Garbage collection
import gc
gc.collect()

# Check memory usage
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024 ** 2:.2f} MB")
```

## Useful Links

- [Full Setup Guide](DEVELOPMENT_SETUP.md)
- [Workflow Guide](DEVELOPMENT_WORKFLOW.md)
- [Testing Guide](../tests/RESEARCH_TESTING_GUIDE.md)
- [HMM Documentation](../imp/hmm/README.md)
- [Data Integration](../imp/data/README.md)

## Keyboard Shortcuts

### Jupyter Lab

- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell
- `Alt + Enter`: Run cell and insert below
- `A`: Insert cell above
- `B`: Insert cell below
- `D + D`: Delete cell
- `M`: Change to markdown
- `Y`: Change to code
- `Ctrl + S`: Save notebook

### Terminal

- `Ctrl + C`: Interrupt process
- `Ctrl + D`: Exit shell
- `Ctrl + R`: Search command history
- `Ctrl + L`: Clear screen
- `Tab`: Auto-complete

## Environment Variables

```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/py"

# Set Jupyter config
export JUPYTER_CONFIG_DIR="~/.jupyter"

# Disable warnings
export PYTHONWARNINGS="ignore"

# Set random seed
export PYTHONHASHSEED=42
```

## Performance Tips

1. **Use vectorized operations** instead of loops
2. **Profile before optimizing** with `line_profiler`
3. **Use polars** for large datasets instead of pandas
4. **Cache expensive computations** with `@lru_cache`
5. **Process data in chunks** for memory efficiency
6. **Use multiprocessing** for CPU-bound tasks
7. **Monitor memory** with `memory_profiler`

## Common Patterns

### Safe File Operations

```python
from pathlib import Path

# Check if file exists
if Path("data.parquet").exists():
    df = pd.read_parquet("data.parquet")

# Create directory if needed
Path("output").mkdir(parents=True, exist_ok=True)

# Iterate over files
for file in Path("data").glob("*.parquet"):
    print(file)
```

### Error Handling

```python
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    cleanup()
```

### Context Managers

```python
# File operations
with open("file.txt", "r") as f:
    content = f.read()

# Timer
from contextlib import contextmanager
import time

@contextmanager
def timer(name):
    start = time.time()
    yield
    print(f"{name}: {time.time() - start:.2f}s")

with timer("Training"):
    model.train(data)
```

## Getting Help

```python
# In Python/Jupyter
help(function_name)
?function_name
??function_name  # Show source

# In terminal
python -m pydoc module_name
man command_name
command_name --help
```
