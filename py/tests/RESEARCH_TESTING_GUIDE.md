# HMM Research Environment Testing Guide

## Overview

This guide describes the comprehensive testing framework for the HMM research environment. The framework ensures reliability, performance, and compatibility between research and production components.

## Test Structure

### Test Files

1. **test_research_framework.py** - Core research framework tests
   - NotebookTester class for automated notebook execution
   - PerformanceBenchmark for library comparisons
   - IntegrationTester for research/production compatibility
   - Visualization generation tests

2. **test_notebook_integration.py** - Notebook-specific tests
   - Widget functionality
   - Interactive plotting
   - Notebook utilities
   - Data persistence between cells

3. **test_performance_compatibility.py** - Performance and compatibility tests
   - Cross-library compatibility (hmmlearn vs pomegranate)
   - Performance benchmarks
   - Memory usage tests
   - Artifact format compatibility
   - Production system integration

4. **test_hmm.py** - Core HMM functionality tests
5. **test_evaluation.py** - Model evaluation framework tests
6. **test_visualization.py** - Visualization component tests
7. **test_data_integration.py** - LDC data integration tests
8. **test_artifact_management.py** - Artifact management tests
9. **test_regime_analysis.py** - Regime analysis tests

## Running Tests

### Quick Start

Run all tests (excluding slow tests):
```bash
cd py
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_research_framework.py -v
```

### Using the Test Runner

The comprehensive test runner provides detailed reporting:

```bash
# Run all tests with default settings
python tests/run_research_tests.py

# Include slow-running tests
python tests/run_research_tests.py --include-slow

# Include notebook execution tests
python tests/run_research_tests.py --include-notebooks

# Verbose output
python tests/run_research_tests.py --verbose

# Custom output files
python tests/run_research_tests.py --output my_report.txt --json-output my_results.json
```

### Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.slow` - Slow-running tests (>5 seconds)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.skipif` - Conditional test skipping

Run tests by marker:
```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m "integration"

# Run slow tests only
pytest tests/ -m "slow"
```

## Test Categories

### 1. Notebook Testing

**NotebookTester Class**

Automated notebook execution and validation:

```python
from tests.test_research_framework import NotebookTester

# Initialize tester
tester = NotebookTester(notebook_dir=Path("notebooks"))

# Execute single notebook
result = tester.execute_notebook(
    Path("notebooks/01_data_exploration.ipynb"),
    timeout=300
)

# Test all notebooks
results_df = tester.test_all_notebooks(pattern="*.ipynb")

# Get summary
summary = tester.get_summary()
```

**Features:**
- Automated notebook execution
- Output validation
- Error detection
- Execution time tracking
- Cell-by-cell validation

### 2. Performance Benchmarking

**PerformanceBenchmark Class**

Compare HMM implementations:

```python
from tests.test_research_framework import PerformanceBenchmark

# Initialize benchmark
benchmark = PerformanceBenchmark(random_state=42)

# Benchmark training
results = benchmark.benchmark_training(
    observations,
    n_states=3,
    libraries=['hmmlearn', 'pomegranate'],
    n_iterations=100,
    n_runs=3
)

# Test scalability
scalability = benchmark.benchmark_scalability(
    observations,
    sample_sizes=[100, 500, 1000, 5000]
)

# Compare libraries
comparison = benchmark.compare_libraries(
    observations,
    n_states_range=[2, 3, 4, 5]
)

# Generate report
report = benchmark.get_performance_report()
```

**Metrics Tracked:**
- Training time
- Inference speed
- Memory usage
- Model quality (log-likelihood, AIC, BIC)
- Scalability with data size and state count

### 3. Integration Testing

**IntegrationTester Class**

Test research/production compatibility:

```python
from tests.test_research_framework import IntegrationTester

# Initialize tester
tester = IntegrationTester()

# Test artifact compatibility
artifact_results = tester.test_artifact_compatibility(
    artifact,
    weights=fusion_weights
)

# Test data pipeline
pipeline_results = tester.test_data_pipeline_integration(
    sample_data_path=Path("data/sample.parquet")
)

# Test complete workflow
workflow_results = tester.test_research_to_production_workflow(
    observations
)

# Generate report
report = tester.get_integration_report()
```

**Tests:**
- Artifact format validation
- Serialization/deserialization
- Inference compatibility
- Data pipeline integration
- End-to-end workflow

### 4. Visualization Testing

Tests for visualization components:

```python
# Test plot generation
from imp.visualization.regime_visualizer import RegimeVisualizer

visualizer = RegimeVisualizer(artifact)

# Test transition matrix plot
fig = visualizer.plot_transition_matrix()
assert fig is not None

# Test state probability plot
fig = visualizer.plot_state_probabilities(state_probs, interactive=False)
assert fig is not None

# Test regime statistics
stats = visualizer.calculate_regime_statistics(observations, state_probs)
assert 'n_states' in stats
```

### 5. Cross-Library Compatibility

Tests ensuring compatibility between hmmlearn and pomegranate:

```python
# Test both libraries produce compatible artifacts
for library in ['hmmlearn', 'pomegranate']:
    trainer = EnhancedHMMTrainer(n_states=3, library=library)
    artifact = trainer.train(observations)
    
    # Verify artifact structure
    assert hasattr(artifact, 'transition_matrix')
    assert hasattr(artifact, 'means')
    
    # Test inference compatibility
    inference = HMMInference()
    inference.load_artifact(artifact)
    prediction = inference.predict(test_data)
    assert prediction is not None
```

## Test Data

### Fixtures

Common test fixtures are provided:

```python
@pytest.fixture
def sample_observations():
    """Generate sample observation data."""
    np.random.seed(42)
    return np.random.randn(200, 3)

@pytest.fixture
def trained_artifact(sample_observations):
    """Create trained HMM artifact."""
    trainer = EnhancedHMMTrainer(n_states=3, random_state=42)
    return trainer.train(sample_observations, n_iterations=50)
```

### Synthetic Data Generation

Generate test data with known properties:

```python
def generate_regime_data(n_samples=200, n_features=3, n_regimes=3):
    """Generate data with distinct regimes."""
    observations = []
    for i in range(n_samples):
        regime = i // (n_samples // n_regimes)
        mean = regime * 2.0
        obs = np.random.randn(n_features) + mean
        observations.append(obs)
    return np.array(observations)
```

## Continuous Integration

### GitHub Actions

Example workflow for CI:

```yaml
name: Research Environment Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        cd py
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        cd py
        python tests/run_research_tests.py --output ci_report.txt
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: py/ci_report.txt
```

## Best Practices

### 1. Test Organization

- Keep tests focused and independent
- Use descriptive test names
- Group related tests in classes
- Use fixtures for common setup

### 2. Performance Testing

- Use `@pytest.mark.slow` for long-running tests
- Set reasonable timeouts
- Test with multiple data sizes
- Compare against baselines

### 3. Integration Testing

- Test complete workflows
- Verify artifact compatibility
- Test error handling
- Validate production integration

### 4. Notebook Testing

- Test notebooks can execute without errors
- Validate expected outputs
- Check for error cells
- Test with fresh kernel

### 5. Mocking and Isolation

- Mock external dependencies when appropriate
- Use temporary directories for file operations
- Clean up resources after tests
- Isolate tests from each other

## Troubleshooting

### Common Issues

**1. Notebook execution fails**
```bash
# Check Jupyter is installed
jupyter --version

# Install nbconvert if missing
pip install nbconvert

# Check kernel availability
jupyter kernelspec list
```

**2. Pomegranate not available**
```bash
# Install pomegranate (optional)
pip install pomegranate

# Tests will skip if not available
pytest tests/ -v  # Will show skipped tests
```

**3. Tests timeout**
```bash
# Increase timeout for slow tests
pytest tests/ --timeout=600

# Or skip slow tests
pytest tests/ -m "not slow"
```

**4. Memory issues with large datasets**
```bash
# Run tests sequentially
pytest tests/ -n 1

# Or reduce dataset sizes in fixtures
```

## Test Coverage

Generate coverage report:

```bash
# Run tests with coverage
pytest tests/ --cov=imp --cov-report=html

# View coverage report
open htmlcov/index.html
```

Target coverage goals:
- Core modules: >90%
- Utilities: >80%
- Examples: >70%

## Adding New Tests

### Template for New Test File

```python
"""
Tests for [component name].
"""

import pytest
import numpy as np
from imp.[module] import [Component]


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return np.random.randn(100, 3)


class Test[ComponentName]:
    """Test cases for [ComponentName]."""
    
    def test_initialization(self):
        """Test component initialization."""
        component = [Component]()
        assert component is not None
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        component = [Component]()
        result = component.process(sample_data)
        assert result is not None
    
    @pytest.mark.slow
    def test_performance(self, sample_data):
        """Test performance with large dataset."""
        # Performance test implementation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Jupyter Testing](https://nbconvert.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [HMM Research Environment Design](../../.kiro/specs/hmm-research-environment/design.md)

## Support

For issues or questions:
1. Check this guide
2. Review test output and error messages
3. Check existing test examples
4. Consult the design document
