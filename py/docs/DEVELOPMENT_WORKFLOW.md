# Development Workflow Guide

This guide describes best practices and workflows for developing with the IMP HMM Research Environment.

## Table of Contents

1. [Daily Development Workflow](#daily-development-workflow)
2. [Research Workflow](#research-workflow)
3. [Testing Workflow](#testing-workflow)
4. [Code Review Process](#code-review-process)
5. [Release Process](#release-process)
6. [Best Practices](#best-practices)

## Daily Development Workflow

### Starting Your Day

```bash
# 1. Activate virtual environment
cd py
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate.bat  # Windows

# 2. Update dependencies (if needed)
git pull origin main
pip install -e ".[dev,optimization,research]"

# 3. Validate environment
python -m imp.utils.env_validator

# 4. Run tests to ensure everything works
pytest tests/ -m "not slow" --tb=short
```

### During Development

```bash
# 1. Create a feature branch
git checkout -b feature/your-feature-name

# 2. Make your changes
# Edit code in imp/ or notebooks/

# 3. Run tests frequently
pytest tests/test_your_module.py -v

# 4. Format code before committing
black imp/ tests/
isort imp/ tests/

# 5. Commit changes
git add .
git commit -m "feat: add your feature description"

# 6. Push to remote
git push origin feature/your-feature-name
```

### End of Day

```bash
# 1. Ensure all tests pass
pytest tests/ -v

# 2. Check code quality
flake8 imp/ tests/
mypy imp/

# 3. Update documentation if needed
# Edit relevant .md files

# 4. Commit and push
git add .
git commit -m "docs: update documentation"
git push
```

## Research Workflow

### Starting a New Research Project

```bash
# 1. Create a new notebook
jupyter lab

# In Jupyter:
# - File > New > Notebook
# - Select "IMP Research Environment" kernel
# - Save as notebooks/XX_your_research_topic.ipynb

# 2. Set up notebook structure
```

**Recommended Notebook Structure:**

```python
# Cell 1: Title and Description
"""
# Research Title

**Author:** Your Name
**Date:** YYYY-MM-DD
**Objective:** Brief description of research goals

## Summary
- Key finding 1
- Key finding 2
"""

# Cell 2: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imp.hmm.trainer import EnhancedHMMTrainer
from imp.hmm.artifact_management import ResearchArtifact
from imp.visualization.regime_visualizer import RegimeVisualizer
from imp.data.ldc_loader import LDCDataLoader

# Cell 3: Configuration
CONFIG = {
    'n_states': 3,
    'library': 'hmmlearn',
    'covariance_type': 'full',
    'random_seed': 42
}

# Cell 4: Load Data
# Your data loading code

# Cell 5+: Analysis
# Your analysis code
```

### Iterative Research Process

1. **Explore Data**
   - Load and visualize data
   - Check for quality issues
   - Understand distributions

2. **Develop Hypothesis**
   - Document assumptions
   - Define success criteria
   - Plan experiments

3. **Run Experiments**
   - Train models with different configurations
   - Compare results systematically
   - Document findings

4. **Analyze Results**
   - Visualize outcomes
   - Statistical validation
   - Interpret findings

5. **Document Conclusions**
   - Summarize key findings
   - Note limitations
   - Suggest next steps

### Saving Research Artifacts

```python
# In your notebook
from imp.hmm.artifact_management import ExperimentTracker
from pathlib import Path

# Initialize tracker
tracker = ExperimentTracker(Path("../processed_data/experiments"))

# Train model
trainer = EnhancedHMMTrainer(**CONFIG)
artifact, metrics = trainer.train_with_validation(observations)

# Create research artifact
research_artifact = ResearchArtifact(
    base_artifact=artifact,
    training_config=CONFIG,
    evaluation_metrics=metrics,
    experiment_id="exp_001",
    researcher="Your Name",
    notebook_path="notebooks/03_regime_analysis.ipynb"
)

# Log experiment
experiment_id = tracker.log_experiment(
    research_artifact,
    notes="Initial regime detection experiment with 3 states"
)

print(f"Experiment saved: {experiment_id}")
```

## Testing Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m research        # Research framework tests
pytest tests/ -m notebook        # Notebook execution tests
pytest tests/ -m integration     # Integration tests
pytest tests/ -m "not slow"      # Skip slow tests

# Run specific test file
pytest tests/test_hmm.py -v

# Run specific test function
pytest tests/test_hmm.py::test_hmm_training -v

# Run with coverage
pytest tests/ --cov=imp --cov-report=html

# Run with detailed output
pytest tests/ -vv --tb=long
```

### Writing Tests

**Test Structure:**

```python
# tests/test_your_module.py
import pytest
import numpy as np
from imp.your_module import YourClass

class TestYourClass:
    """Tests for YourClass."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data."""
        return np.random.randn(100, 3)
    
    def test_initialization(self):
        """Test class initialization."""
        obj = YourClass(param=value)
        assert obj.param == value
    
    def test_method_with_fixture(self, sample_data):
        """Test method using fixture."""
        obj = YourClass()
        result = obj.process(sample_data)
        assert result.shape == sample_data.shape
    
    @pytest.mark.slow
    def test_expensive_operation(self):
        """Test that takes a long time."""
        # Expensive test code
        pass
    
    @pytest.mark.integration
    def test_integration_with_other_module(self):
        """Test integration between modules."""
        # Integration test code
        pass
```

### Test-Driven Development (TDD)

1. **Write Test First**
   ```python
   def test_new_feature():
       """Test for feature that doesn't exist yet."""
       result = new_feature(input_data)
       assert result == expected_output
   ```

2. **Run Test (Should Fail)**
   ```bash
   pytest tests/test_new_feature.py -v
   # Expected: FAILED
   ```

3. **Implement Feature**
   ```python
   def new_feature(input_data):
       # Implementation
       return output
   ```

4. **Run Test Again (Should Pass)**
   ```bash
   pytest tests/test_new_feature.py -v
   # Expected: PASSED
   ```

5. **Refactor**
   - Improve code quality
   - Ensure tests still pass

## Code Review Process

### Before Submitting PR

```bash
# 1. Ensure all tests pass
pytest tests/ -v

# 2. Check code formatting
black --check imp/ tests/
isort --check-only imp/ tests/

# 3. Run linter
flake8 imp/ tests/

# 4. Type checking
mypy imp/

# 5. Update documentation
# Edit relevant .md files

# 6. Clear notebook outputs
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

### Creating Pull Request

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**
   - Go to repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in PR template

3. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] All tests passing
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)
   - [ ] Notebooks tested and outputs cleared
   ```

### Reviewing PRs

**As a Reviewer:**

1. **Check Code Quality**
   - Follows style guidelines
   - Well-documented
   - No obvious bugs

2. **Verify Tests**
   - Tests are comprehensive
   - Tests pass locally
   - Coverage is adequate

3. **Review Documentation**
   - README updated if needed
   - Docstrings are clear
   - Examples are provided

4. **Test Functionality**
   ```bash
   git checkout feature-branch
   pip install -e ".[dev,optimization,research]"
   pytest tests/ -v
   ```

5. **Provide Feedback**
   - Be constructive
   - Suggest improvements
   - Ask questions

## Release Process

### Preparing a Release

1. **Update Version**
   ```toml
   # py/pyproject.toml
   [project]
   version = "0.2.0"  # Update version
   ```

2. **Update CHANGELOG**
   ```markdown
   # CHANGELOG.md
   
   ## [0.2.0] - 2025-01-15
   
   ### Added
   - New feature X
   - New feature Y
   
   ### Changed
   - Improved performance of Z
   
   ### Fixed
   - Bug in component A
   ```

3. **Run Full Test Suite**
   ```bash
   pytest tests/ -v --cov=imp
   ```

4. **Build Documentation**
   ```bash
   # Generate API docs
   sphinx-build -b html docs/ docs/_build/
   ```

5. **Create Release Tag**
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

### Post-Release

1. **Monitor CI/CD**
   - Check GitHub Actions
   - Verify all tests pass

2. **Update Documentation**
   - Deploy docs to GitHub Pages
   - Update README if needed

3. **Announce Release**
   - Create GitHub Release
   - Update project board

## Best Practices

### Code Organization

```
py/
├── imp/                    # Main package
│   ├── hmm/               # HMM components
│   ├── data/              # Data processing
│   ├── visualization/     # Visualization tools
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── notebooks/             # Research notebooks
├── examples/              # Example scripts
└── docs/                  # Documentation
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_CASE`
- **Private**: `_leading_underscore`

### Documentation Standards

```python
def function_name(param1: type, param2: type) -> return_type:
    """Brief description of function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is invalid
        TypeError: When param2 has wrong type
    
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output
    """
    # Implementation
    pass
```

### Git Commit Messages

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(hmm): add pomegranate trainer support

Implement PomegranateTrainer class with consistent interface
to existing HMMLearnTrainer. Includes cross-validation support
and comprehensive error handling.

Closes #123
```

### Performance Optimization

1. **Profile First**
   ```python
   from line_profiler import profile
   
   @profile
   def slow_function():
       # Code to profile
       pass
   ```

2. **Use Appropriate Data Structures**
   - numpy for numerical operations
   - pandas for tabular data
   - polars for large datasets

3. **Vectorize Operations**
   ```python
   # Bad
   result = [x**2 for x in data]
   
   # Good
   result = np.square(data)
   ```

4. **Cache Expensive Computations**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def expensive_function(param):
       # Expensive computation
       return result
   ```

### Security Practices

1. **Never Commit Secrets**
   ```bash
   # Use .env files
   echo "API_KEY=your_key" > .env
   echo ".env" >> .gitignore
   ```

2. **Validate Inputs**
   ```python
   from pydantic import BaseModel, validator
   
   class Config(BaseModel):
       n_states: int
       
       @validator('n_states')
       def validate_states(cls, v):
           if v < 2:
               raise ValueError('n_states must be >= 2')
           return v
   ```

3. **Sanitize Outputs**
   - Remove sensitive data from logs
   - Clear notebook outputs before committing

## Additional Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Git Best Practices](https://git-scm.com/book/en/v2)
- [Pytest Documentation](https://docs.pytest.org/)
- [Jupyter Best Practices](https://jupyter.org/documentation)

## Getting Help

- Check documentation in `py/docs/`
- Review example notebooks in `notebooks/`
- Ask questions in team chat
- Open GitHub issues for bugs
