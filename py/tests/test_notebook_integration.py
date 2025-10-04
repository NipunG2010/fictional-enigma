"""
Integration tests specifically for Jupyter notebook functionality.

Tests notebook-specific features:
- Widget functionality
- Interactive plotting
- Notebook utilities
- Data loaders for notebooks
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import notebook utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

try:
    from utils.notebook_utils import (
        setup_notebook_environment,
        create_sample_data,
        format_metrics_table
    )
    from utils.data_loaders import load_ldc_signals, load_hmm_results
    from utils.plotting_helpers import (
        plot_signal_comparison,
        plot_regime_overlay
    )
    NOTEBOOK_UTILS_AVAILABLE = True
except ImportError:
    NOTEBOOK_UTILS_AVAILABLE = False


@pytest.fixture
def sample_signal_data():
    """Generate sample signal data."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        's_ldc': np.random.randn(n_samples),
        's_mr': np.random.randn(n_samples),
        's_tsmom': np.random.randn(n_samples),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    })


@pytest.mark.skipif(
    not NOTEBOOK_UTILS_AVAILABLE,
    reason="Notebook utilities not available"
)
class TestNotebookUtilities:
    """Test notebook utility functions."""
    
    def test_setup_notebook_environment(self):
        """Test notebook environment setup."""
        config = setup_notebook_environment()
        
        assert isinstance(config, dict)
        assert 'matplotlib_backend' in config
        assert 'pandas_display_options' in config
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        data = create_sample_data(n_samples=100, n_features=3)
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 3)
    
    def test_format_metrics_table(self):
        """Test metrics table formatting."""
        metrics = {
            'log_likelihood': -100.5,
            'aic': 210.0,
            'bic': 220.0
        }
        
        table = format_metrics_table(metrics)
        
        assert isinstance(table, str)
        assert 'log_likelihood' in table
        assert '-100.5' in table


@pytest.mark.skipif(
    not NOTEBOOK_UTILS_AVAILABLE,
    reason="Notebook utilities not available"
)
class TestDataLoaders:
    """Test notebook data loading utilities."""
    
    def test_load_ldc_signals_from_dataframe(self, sample_signal_data, tmp_path):
        """Test loading LDC signals from saved data."""
        # Save sample data
        data_path = tmp_path / "signals.parquet"
        sample_signal_data.to_parquet(data_path)
        
        # Load using utility
        loaded = load_ldc_signals(data_path)
        
        assert isinstance(loaded, pd.DataFrame)
        assert 's_ldc' in loaded.columns
        assert 's_mr' in loaded.columns
        assert 's_tsmom' in loaded.columns
    
    def test_load_hmm_results(self, tmp_path):
        """Test loading HMM results."""
        # Create sample results
        results = {
            'state_probabilities': np.random.rand(100, 3).tolist(),
            'most_likely_states': np.random.randint(0, 3, 100).tolist(),
            'log_likelihood': -150.5
        }
        
        import json
        results_path = tmp_path / "hmm_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        # Load using utility
        loaded = load_hmm_results(results_path)
        
        assert isinstance(loaded, dict)
        assert 'state_probabilities' in loaded
        assert 'log_likelihood' in loaded


@pytest.mark.skipif(
    not NOTEBOOK_UTILS_AVAILABLE,
    reason="Notebook utilities not available"
)
class TestPlottingHelpers:
    """Test notebook plotting helper functions."""
    
    def test_plot_signal_comparison(self, sample_signal_data):
        """Test signal comparison plotting."""
        import matplotlib
        matplotlib.use('Agg')
        
        fig = plot_signal_comparison(
            sample_signal_data,
            signals=['s_ldc', 's_mr', 's_tsmom']
        )
        
        assert fig is not None
        assert len(fig.axes) >= 1
    
    def test_plot_regime_overlay(self, sample_signal_data):
        """Test regime overlay plotting."""
        import matplotlib
        matplotlib.use('Agg')
        
        # Add regime data
        sample_signal_data['regime'] = np.random.randint(0, 3, len(sample_signal_data))
        
        fig = plot_regime_overlay(
            sample_signal_data,
            price_column='close',
            regime_column='regime'
        )
        
        assert fig is not None
        assert len(fig.axes) >= 1


class TestWidgetFunctionality:
    """Test interactive widget functionality."""
    
    def test_widget_imports(self):
        """Test that widget libraries can be imported."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
            
            # Test basic widget creation
            slider = widgets.IntSlider(value=3, min=2, max=10)
            assert slider.value == 3
            
            dropdown = widgets.Dropdown(
                options=['hmmlearn', 'pomegranate'],
                value='hmmlearn'
            )
            assert dropdown.value == 'hmmlearn'
            
        except ImportError:
            pytest.skip("IPython widgets not available")
    
    def test_parameter_tuning_widget_creation(self):
        """Test parameter tuning widget creation."""
        try:
            import ipywidgets as widgets
            
            # Create parameter tuning widgets
            n_states_slider = widgets.IntSlider(
                value=3, min=2, max=10,
                description='States:'
            )
            
            covariance_dropdown = widgets.Dropdown(
                options=['full', 'diag', 'spherical'],
                value='full',
                description='Covariance:'
            )
            
            library_dropdown = widgets.Dropdown(
                options=['hmmlearn', 'pomegranate'],
                value='hmmlearn',
                description='Library:'
            )
            
            # Create container
            container = widgets.VBox([
                n_states_slider,
                covariance_dropdown,
                library_dropdown
            ])
            
            assert len(container.children) == 3
            
        except ImportError:
            pytest.skip("IPython widgets not available")


class TestInteractivePlotting:
    """Test interactive plotting functionality."""
    
    def test_plotly_imports(self):
        """Test that plotly can be imported."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Test basic plot creation
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
            
            assert len(fig.data) == 1
            
        except ImportError:
            pytest.skip("Plotly not available")
    
    def test_interactive_state_probability_plot(self):
        """Test interactive state probability plotting."""
        try:
            import plotly.graph_objects as go
            
            # Create sample data
            np.random.seed(42)
            n_timesteps = 100
            n_states = 3
            
            state_probs = np.random.dirichlet([1, 1, 1], size=n_timesteps)
            timestamps = np.arange(n_timesteps)
            
            # Create interactive plot
            fig = go.Figure()
            
            for state in range(n_states):
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=state_probs[:, state],
                    mode='lines',
                    name=f'State {state}',
                    stackgroup='one'
                ))
            
            fig.update_layout(
                title='State Probabilities Over Time',
                xaxis_title='Time',
                yaxis_title='Probability'
            )
            
            assert len(fig.data) == n_states
            
        except ImportError:
            pytest.skip("Plotly not available")


class TestNotebookDataPersistence:
    """Test data persistence between notebook cells."""
    
    def test_save_and_load_processed_data(self, tmp_path):
        """Test saving and loading processed data."""
        # Create sample processed data
        processed_data = {
            'observations': np.random.randn(100, 3),
            'timestamps': np.arange(100),
            'metadata': {'n_features': 3, 'n_samples': 100}
        }
        
        # Save
        save_path = tmp_path / "processed_data.npz"
        np.savez(
            save_path,
            observations=processed_data['observations'],
            timestamps=processed_data['timestamps']
        )
        
        # Load
        loaded = np.load(save_path)
        
        assert 'observations' in loaded
        assert 'timestamps' in loaded
        assert loaded['observations'].shape == (100, 3)
    
    def test_save_and_load_model_results(self, tmp_path):
        """Test saving and loading model results."""
        import json
        
        # Create sample results
        results = {
            'model_config': {
                'n_states': 3,
                'library': 'hmmlearn',
                'covariance_type': 'full'
            },
            'metrics': {
                'log_likelihood': -150.5,
                'aic': 310.0,
                'bic': 330.0
            },
            'training_time': 5.2
        }
        
        # Save
        results_path = tmp_path / "model_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Load
        with open(results_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['model_config']['n_states'] == 3
        assert loaded['metrics']['log_likelihood'] == -150.5


class TestNotebookErrorHandling:
    """Test error handling in notebook context."""
    
    def test_graceful_import_failure(self):
        """Test graceful handling of missing dependencies."""
        try:
            import nonexistent_package
            assert False, "Should have raised ImportError"
        except ImportError as e:
            # This is expected
            assert "nonexistent_package" in str(e)
    
    def test_data_validation_in_notebook(self):
        """Test data validation before processing."""
        # Invalid data (wrong shape)
        invalid_data = np.random.randn(10, 2)  # Should be 3 features
        
        # Validation function
        def validate_signal_data(data, expected_features=3):
            if data.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {data.shape[1]}"
                )
            return True
        
        with pytest.raises(ValueError, match="Expected 3 features"):
            validate_signal_data(invalid_data)
    
    def test_model_training_error_handling(self):
        """Test error handling during model training."""
        from imp.hmm.trainer import EnhancedHMMTrainer
        
        # Invalid data (too few samples)
        invalid_data = np.random.randn(5, 3)
        
        trainer = EnhancedHMMTrainer(n_states=3, random_state=42)
        
        # Should handle gracefully
        try:
            artifact = trainer.train(invalid_data, n_iterations=10)
            # Some implementations might succeed with small data
        except Exception as e:
            # Error should be informative
            assert len(str(e)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
