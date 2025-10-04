"""
Tests for the visualization module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from imp.hmm.models import HMMArtifact
from imp.visualization.regime_visualizer import RegimeVisualizer


@pytest.fixture
def sample_hmm_artifact():
    """Create a sample HMM artifact for testing."""
    return HMMArtifact(
        version="1.0.0",
        n_states=3,
        transition_matrix=[
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.1, 0.3, 0.6]
        ],
        initial_probabilities=[0.33, 0.33, 0.34],
        means=[
            [0.1, 0.2],
            [0.5, 0.6],
            [-0.2, -0.1]
        ],
        covariances=[
            [[0.1, 0.05], [0.05, 0.1]],
            [[0.2, 0.1], [0.1, 0.2]],
            [[0.15, 0.08], [0.08, 0.15]]
        ],
        training_window_start=0,
        training_window_end=1000,
        metadata={"test": True}
    )


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_timesteps = 100
    n_features = 2
    n_states = 3
    
    # Generate sample observations
    observations = np.random.randn(n_timesteps, n_features)
    
    # Generate sample state probabilities
    state_probs = np.random.dirichlet([1, 1, 1], size=n_timesteps)
    
    # Generate timestamps
    timestamps = np.arange(n_timesteps)
    
    return observations, state_probs, timestamps


class TestRegimeVisualizer:
    """Test cases for RegimeVisualizer."""
    
    def test_initialization(self, sample_hmm_artifact):
        """Test RegimeVisualizer initialization."""
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        assert visualizer.n_states == 3
        assert visualizer.transition_matrix.shape == (3, 3)
        assert len(visualizer.state_colors) == 3
        assert all(color.startswith('#') for color in visualizer.state_colors)
    
    def test_color_generation(self, sample_hmm_artifact):
        """Test state color generation."""
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        colors = visualizer._generate_state_colors()
        
        assert len(colors) == 3
        assert all(isinstance(color, str) for color in colors)
        assert all(color.startswith('#') and len(color) == 7 for color in colors)
    
    def test_plot_state_probabilities_static(self, sample_hmm_artifact, sample_data):
        """Test static state probability plotting."""
        observations, state_probs, timestamps = sample_data
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        fig = visualizer.plot_state_probabilities(
            state_probs, timestamps, interactive=False
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)
    
    def test_plot_state_probabilities_invalid_shape(self, sample_hmm_artifact):
        """Test error handling for invalid state probability shape."""
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        # Wrong number of states
        invalid_probs = np.random.rand(100, 2)  # Should be 3 states
        
        with pytest.raises(ValueError, match="State probabilities must have 3 columns"):
            visualizer.plot_state_probabilities(invalid_probs, interactive=False)
    
    def test_plot_transition_matrix(self, sample_hmm_artifact):
        """Test transition matrix plotting."""
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        fig = visualizer.plot_transition_matrix()
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Main plot + colorbar
        plt.close(fig)
    
    def test_calculate_regime_statistics(self, sample_hmm_artifact, sample_data):
        """Test regime statistics calculation."""
        observations, state_probs, timestamps = sample_data
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        stats = visualizer.calculate_regime_statistics(observations, state_probs, timestamps)
        
        assert 'n_states' in stats
        assert 'total_observations' in stats
        assert 'state_statistics' in stats
        assert 'transition_statistics' in stats
        assert 'regime_persistence' in stats
        
        assert stats['n_states'] == 3
        assert stats['total_observations'] == 100
        
        # Check state statistics
        for i in range(3):
            state_key = f'state_{i}'
            if state_key in stats['state_statistics']:
                state_stats = stats['state_statistics'][state_key]
                assert 'frequency' in state_stats
                assert 'mean_probability' in state_stats
                assert 'observation_count' in state_stats
                assert 0 <= state_stats['frequency'] <= 1
    
    def test_calculate_regime_statistics_mismatched_lengths(self, sample_hmm_artifact):
        """Test error handling for mismatched observation and probability lengths."""
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        observations = np.random.randn(100, 2)
        state_probs = np.random.dirichlet([1, 1, 1], size=50)  # Different length
        
        with pytest.raises(ValueError, match="Observations and state probabilities must have same length"):
            visualizer.calculate_regime_statistics(observations, state_probs)
    
    def test_persistence_metrics(self, sample_hmm_artifact):
        """Test persistence metrics calculation."""
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        # Create deterministic state sequence for testing
        states = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0])
        timestamps = np.arange(len(states))
        
        persistence = visualizer._calculate_persistence_metrics(states, timestamps)
        
        # State 0: appears twice with durations [3, 1]
        assert 'state_0' in persistence
        assert persistence['state_0']['mean_duration'] == 2.0
        assert persistence['state_0']['total_episodes'] == 2
        
        # State 1: appears once with duration [2]
        assert 'state_1' in persistence
        assert persistence['state_1']['mean_duration'] == 2.0
        assert persistence['state_1']['total_episodes'] == 1
        
        # State 2: appears once with duration [4]
        assert 'state_2' in persistence
        assert persistence['state_2']['mean_duration'] == 4.0
        assert persistence['state_2']['total_episodes'] == 1
    
    def test_format_regime_statistics(self, sample_hmm_artifact, sample_data):
        """Test regime statistics formatting."""
        observations, state_probs, timestamps = sample_data
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        stats = visualizer.calculate_regime_statistics(observations, state_probs, timestamps)
        formatted = visualizer.format_regime_statistics(stats)
        
        assert isinstance(formatted, str)
        assert '<h4>Regime Analysis Summary</h4>' in formatted
        assert 'Total Observations' in formatted
        assert 'Number of States' in formatted
        assert '<table' in formatted
    
    def test_get_state_durations(self, sample_hmm_artifact):
        """Test state duration calculation."""
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        # Test with known sequence
        states = np.array([0, 0, 1, 1, 1, 2, 0])
        durations = visualizer._get_state_durations(states)
        
        assert durations[0] == [2, 1]  # State 0 appears with durations 2 and 1
        assert durations[1] == [3]     # State 1 appears with duration 3
        assert durations[2] == [1]     # State 2 appears with duration 1
    
    def test_get_state_durations_empty(self, sample_hmm_artifact):
        """Test state duration calculation with empty array."""
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        states = np.array([])
        durations = visualizer._get_state_durations(states)
        
        for i in range(3):
            assert durations[i] == []
    
    def test_plot_regime_comparison(self, sample_hmm_artifact, sample_data):
        """Test regime comparison plotting."""
        observations, state_probs, timestamps = sample_data
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        # Create second set of state probabilities for comparison
        state_probs_2 = np.random.dirichlet([1, 1, 1], size=len(state_probs))
        
        fig = visualizer.plot_regime_comparison(
            observations[:, 0],  # Use univariate for simplicity
            [state_probs, state_probs_2],
            ['Model 1', 'Model 2'],
            timestamps
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # Observations + 2 models
        plt.close(fig)
    
    def test_create_regime_dashboard_no_widgets(self, sample_hmm_artifact, sample_data):
        """Test dashboard creation when widgets are not available."""
        observations, state_probs, timestamps = sample_data
        visualizer = RegimeVisualizer(sample_hmm_artifact)
        
        # Mock widgets not being available
        import imp.visualization.regime_visualizer as rv
        original_widgets_available = rv.WIDGETS_AVAILABLE
        rv.WIDGETS_AVAILABLE = False
        
        try:
            result = visualizer.create_regime_dashboard(observations, state_probs, timestamps)
            assert isinstance(result, str)
            assert "IPython widgets not available" in result
        finally:
            rv.WIDGETS_AVAILABLE = original_widgets_available


if __name__ == "__main__":
    pytest.main([__file__])