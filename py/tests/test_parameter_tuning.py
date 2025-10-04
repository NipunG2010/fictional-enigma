"""
Tests for the interactive parameter tuning framework.
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

from imp.tuning.parameter_tuner import (
    HMMParameterTuner,
    TuningConfig,
    TuningResult
)
from imp.tuning.optimization import (
    GridSearchOptimizer,
    BayesianOptimizer,
    create_default_param_grid,
    create_default_param_space,
    quick_grid_search,
    SKOPT_AVAILABLE
)
from imp.hmm.models import HMMArtifact


@pytest.fixture
def sample_observations():
    """Create sample observation data for testing."""
    np.random.seed(42)
    
    # Generate 3-state synthetic data
    n_samples = 200
    n_features = 2
    
    # State 0: low values
    state0 = np.random.randn(60, n_features) * 0.5 - 1
    # State 1: medium values
    state1 = np.random.randn(80, n_features) * 0.7
    # State 2: high values
    state2 = np.random.randn(60, n_features) * 0.5 + 1
    
    observations = np.vstack([state0, state1, state2])
    
    # Shuffle to simulate state transitions
    np.random.shuffle(observations)
    
    return observations


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for configs."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestTuningConfig:
    """Tests for TuningConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TuningConfig()
        
        assert config.n_states == 3
        assert config.library == "hmmlearn"
        assert config.covariance_type == "full"
        assert config.n_iterations == 100
        assert config.validation_split == 0.2
        assert config.random_state == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TuningConfig(
            n_states=5,
            library="pomegranate",
            covariance_type="diag",
            n_iterations=200
        )
        
        assert config.n_states == 5
        assert config.library == "pomegranate"
        assert config.covariance_type == "diag"
        assert config.n_iterations == 200
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TuningConfig(n_states=4)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['n_states'] == 4
        assert 'library' in config_dict
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'n_states': 4,
            'library': 'hmmlearn',
            'covariance_type': 'diag',
            'n_iterations': 150,
            'validation_split': 0.3,
            'random_state': 123
        }
        
        config = TuningConfig.from_dict(config_dict)
        
        assert config.n_states == 4
        assert config.library == 'hmmlearn'
        assert config.covariance_type == 'diag'
        assert config.n_iterations == 150


class TestTuningResult:
    """Tests for TuningResult dataclass."""
    
    def test_result_creation(self, sample_observations):
        """Test creating a tuning result."""
        config = TuningConfig(n_states=3)
        
        # Create a minimal artifact
        artifact = HMMArtifact(
            version="v1.0",
            n_states=3,
            transition_matrix=[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
            initial_probabilities=[0.33, 0.33, 0.34],
            means=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            covariances=[
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]]
            ],
            training_window_start=0,
            training_window_end=100,
            metadata={'test': True}
        )
        
        metrics = {'log_likelihood': -100.5, 'aic': 250.0}
        
        result = TuningResult(
            config=config,
            artifact=artifact,
            metrics=metrics,
            timestamp="2024-01-01T00:00:00",
            experiment_id="test_exp"
        )
        
        assert result.config.n_states == 3
        assert result.metrics['log_likelihood'] == -100.5
        assert result.experiment_id == "test_exp"
    
    def test_result_serialization(self, sample_observations):
        """Test result serialization and deserialization."""
        config = TuningConfig(n_states=2)
        
        artifact = HMMArtifact(
            version="v1.0",
            n_states=2,
            transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
            initial_probabilities=[0.5, 0.5],
            means=[[0.0, 0.0], [1.0, 1.0]],
            covariances=[
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]]
            ],
            training_window_start=0,
            training_window_end=100,
            metadata={}
        )
        
        result = TuningResult(
            config=config,
            artifact=artifact,
            metrics={'log_likelihood': -50.0},
            timestamp="2024-01-01T00:00:00",
            experiment_id="test"
        )
        
        # Serialize
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        
        # Deserialize
        restored_result = TuningResult.from_dict(result_dict)
        assert restored_result.config.n_states == 2
        assert restored_result.metrics['log_likelihood'] == -50.0


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""
    
    def test_grid_search_initialization(self, sample_observations):
        """Test grid search optimizer initialization."""
        param_grid = {
            'n_states': [2, 3],
            'library': ['hmmlearn'],
            'covariance_type': ['full', 'diag']
        }
        
        optimizer = GridSearchOptimizer(
            observations=sample_observations,
            param_grid=param_grid,
            verbose=False
        )
        
        assert optimizer.observations.shape == sample_observations.shape
        assert optimizer.param_grid == param_grid
    
    def test_grid_search_fit(self, sample_observations):
        """Test grid search optimization."""
        param_grid = {
            'n_states': [2, 3],
            'library': ['hmmlearn'],
            'covariance_type': ['full']
        }
        
        optimizer = GridSearchOptimizer(
            observations=sample_observations,
            param_grid=param_grid,
            n_iterations=50,  # Fewer iterations for faster testing
            verbose=False
        )
        
        result = optimizer.fit()
        
        assert result.best_params is not None
        assert result.best_score is not None
        assert len(result.all_results) == 2  # 2 states x 1 library x 1 cov_type
        assert result.optimization_time > 0
    
    def test_grid_search_with_multiple_params(self, sample_observations):
        """Test grid search with multiple parameter values."""
        param_grid = {
            'n_states': [2, 3, 4],
            'library': ['hmmlearn'],
            'covariance_type': ['full', 'diag']
        }
        
        optimizer = GridSearchOptimizer(
            observations=sample_observations,
            param_grid=param_grid,
            n_iterations=30,
            verbose=False
        )
        
        result = optimizer.fit()
        
        # Should have 3 * 1 * 2 = 6 combinations
        assert len(result.all_results) == 6
        assert result.best_params['n_states'] in [2, 3, 4]
        assert result.best_params['covariance_type'] in ['full', 'diag']


@pytest.mark.skipif(not SKOPT_AVAILABLE, reason="scikit-optimize not available")
class TestBayesianOptimizer:
    """Tests for BayesianOptimizer."""
    
    def test_bayesian_optimizer_initialization(self, sample_observations):
        """Test Bayesian optimizer initialization."""
        param_space = {
            'n_states': {'type': 'integer', 'low': 2, 'high': 5},
            'library': {'type': 'categorical', 'categories': ['hmmlearn']},
            'covariance_type': {'type': 'categorical', 'categories': ['full', 'diag']}
        }
        
        optimizer = BayesianOptimizer(
            observations=sample_observations,
            param_space=param_space,
            n_calls=5,
            verbose=False
        )
        
        assert optimizer.observations.shape == sample_observations.shape
        assert optimizer.param_space == param_space
    
    def test_bayesian_optimizer_fit(self, sample_observations):
        """Test Bayesian optimization."""
        param_space = {
            'n_states': {'type': 'integer', 'low': 2, 'high': 4},
            'library': {'type': 'categorical', 'categories': ['hmmlearn']},
            'covariance_type': {'type': 'categorical', 'categories': ['full']}
        }
        
        optimizer = BayesianOptimizer(
            observations=sample_observations,
            param_space=param_space,
            n_calls=10,  # Minimum required by scikit-optimize
            n_iterations=30,
            verbose=False
        )
        
        result = optimizer.fit()
        
        assert result.best_params is not None
        assert result.best_score is not None
        assert len(result.all_results) == 10
        assert result.optimization_time > 0


class TestOptimizationUtilities:
    """Tests for optimization utility functions."""
    
    def test_create_default_param_grid(self):
        """Test default parameter grid creation."""
        param_grid = create_default_param_grid()
        
        assert 'n_states' in param_grid
        assert 'library' in param_grid
        assert 'covariance_type' in param_grid
        assert isinstance(param_grid['n_states'], list)
    
    def test_create_default_param_space(self):
        """Test default parameter space creation."""
        param_space = create_default_param_space()
        
        assert 'n_states' in param_space
        assert 'library' in param_space
        assert 'covariance_type' in param_space
        assert param_space['n_states']['type'] == 'integer'
    
    def test_quick_grid_search(self, sample_observations):
        """Test quick grid search utility."""
        result = quick_grid_search(
            sample_observations,
            n_states_range=[2, 3],
            covariance_types=['full'],
            verbose=False
        )
        
        assert result.best_params is not None
        assert result.best_params['n_states'] in [2, 3]


class TestParameterTunerCore:
    """Tests for core HMMParameterTuner functionality (non-widget)."""
    
    def test_tuner_initialization(self, sample_observations, temp_config_dir):
        """Test parameter tuner initialization."""
        # Skip if widgets not available
        try:
            tuner = HMMParameterTuner(
                observations=sample_observations,
                config_dir=temp_config_dir
            )
            
            assert tuner.observations.shape == sample_observations.shape
            assert tuner.config_dir == temp_config_dir
            assert len(tuner.results) == 0
        except ImportError:
            pytest.skip("IPython widgets not available")
    
    def test_get_best_result_empty(self, sample_observations, temp_config_dir):
        """Test getting best result when no results exist."""
        try:
            tuner = HMMParameterTuner(
                observations=sample_observations,
                config_dir=temp_config_dir
            )
            
            best = tuner.get_best_result()
            assert best is None
        except ImportError:
            pytest.skip("IPython widgets not available")
    
    def test_export_results(self, sample_observations, temp_config_dir):
        """Test exporting results to file."""
        try:
            tuner = HMMParameterTuner(
                observations=sample_observations,
                config_dir=temp_config_dir
            )
            
            # Add a mock result
            config = TuningConfig(n_states=2)
            artifact = HMMArtifact(
                version="v1.0",
                n_states=2,
                transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
                initial_probabilities=[0.5, 0.5],
                means=[[0.0, 0.0], [1.0, 1.0]],
                covariances=[
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]]
                ],
                training_window_start=0,
                training_window_end=100,
                metadata={}
            )
            
            result = TuningResult(
                config=config,
                artifact=artifact,
                metrics={'log_likelihood': -50.0},
                timestamp="2024-01-01T00:00:00",
                experiment_id="test"
            )
            
            tuner.results['test'] = result
            
            # Export
            export_path = temp_config_dir / "results.json"
            tuner.export_results(export_path)
            
            assert export_path.exists()
            
            # Verify content
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert 'results' in data
            assert 'test' in data['results']
        except ImportError:
            pytest.skip("IPython widgets not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
