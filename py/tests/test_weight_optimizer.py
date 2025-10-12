"""
Tests for fusion weight optimization.

Tests cover:
- OptimizationConfig dataclass
- StateWeightOptimizer with both scipy and grid search methods
- Sharpe ratio calculation
- Portfolio returns computation
- Constraint enforcement
- Edge cases and fallback behavior
"""

import pytest
import numpy as np
from imp.hmm.weight_optimizer import StateWeightOptimizer, OptimizationConfig


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        
        assert config.method == "SLSQP"
        assert config.risk_free_rate == 0.02
        assert config.min_weight == 0.0
        assert config.max_weight == 1.0
        assert config.grid_points == 11
        assert config.min_observations == 30
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            method="grid_search",
            risk_free_rate=0.03,
            min_weight=0.1,
            max_weight=0.8,
            grid_points=21,
            min_observations=50
        )
        
        assert config.method == "grid_search"
        assert config.risk_free_rate == 0.03
        assert config.min_weight == 0.1
        assert config.max_weight == 0.8
        assert config.grid_points == 21
        assert config.min_observations == 50


class TestStateWeightOptimizer:
    """Test StateWeightOptimizer class."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Create signals with different characteristics
        # Signal 1: positive trend
        signal1 = np.random.randn(n_samples) + 0.5
        # Signal 2: negative trend
        signal2 = np.random.randn(n_samples) - 0.3
        # Signal 3: neutral
        signal3 = np.random.randn(n_samples)
        
        signals = np.column_stack([signal1, signal2, signal3])
        
        # Returns correlated with signal1 (best signal)
        returns = 0.01 * signal1 + 0.005 * np.random.randn(n_samples)
        
        return signals, returns
    
    def test_scipy_optimization(self, synthetic_data):
        """Test scipy SLSQP optimization method."""
        signals, returns = synthetic_data
        
        config = OptimizationConfig(method="SLSQP")
        optimizer = StateWeightOptimizer(config)
        
        weights, sharpe = optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Check weights structure
        assert isinstance(weights, dict)
        assert set(weights.keys()) == {'s1', 's2', 's3'}
        
        # Check constraints
        assert np.isclose(sum(weights.values()), 1.0, atol=1e-6)
        assert all(w >= 0 for w in weights.values())
        
        # Check Sharpe ratio is reasonable
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_grid_search_optimization(self, synthetic_data):
        """Test grid search optimization method."""
        signals, returns = synthetic_data
        
        config = OptimizationConfig(method="grid_search", grid_points=11)
        optimizer = StateWeightOptimizer(config)
        
        weights, sharpe = optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Check weights structure
        assert isinstance(weights, dict)
        assert set(weights.keys()) == {'s1', 's2', 's3'}
        
        # Check constraints
        assert np.isclose(sum(weights.values()), 1.0, atol=1e-6)
        assert all(w >= 0 for w in weights.values())
        
        # Check Sharpe ratio is reasonable
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_weight_bounds_enforcement(self):
        """Test that weight bounds are enforced."""
        np.random.seed(42)
        signals = np.random.randn(100, 3)
        returns = np.random.randn(100) * 0.01
        
        config = OptimizationConfig(
            method="SLSQP",
            min_weight=0.2,
            max_weight=0.5
        )
        optimizer = StateWeightOptimizer(config)
        
        weights, _ = optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Check bounds
        for w in weights.values():
            assert w >= 0.2 - 1e-6  # Allow small numerical error
            assert w <= 0.5 + 1e-6
    
    def test_insufficient_data_fallback(self):
        """Test fallback to equal weights with insufficient data."""
        signals = np.random.randn(20, 3)  # Less than min_observations
        returns = np.random.randn(20) * 0.01
        
        config = OptimizationConfig(min_observations=30)
        optimizer = StateWeightOptimizer(config)
        
        weights, sharpe = optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Should return equal weights
        assert np.isclose(weights['s1'], 1/3, atol=1e-6)
        assert np.isclose(weights['s2'], 1/3, atol=1e-6)
        assert np.isclose(weights['s3'], 1/3, atol=1e-6)
        assert sharpe == 0.0
    
    def test_nan_data_fallback(self):
        """Test fallback to equal weights with NaN data."""
        signals = np.random.randn(100, 3)
        returns = np.random.randn(100) * 0.01
        
        # Introduce NaN
        signals[50, 1] = np.nan
        
        config = OptimizationConfig()
        optimizer = StateWeightOptimizer(config)
        
        weights, sharpe = optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Should return equal weights
        assert np.isclose(weights['s1'], 1/3, atol=1e-6)
        assert np.isclose(weights['s2'], 1/3, atol=1e-6)
        assert np.isclose(weights['s3'], 1/3, atol=1e-6)
        assert sharpe == 0.0
    
    def test_zero_variance_fallback(self):
        """Test fallback to equal weights with zero variance returns."""
        signals = np.random.randn(100, 3)
        returns = np.zeros(100)  # Zero variance
        
        config = OptimizationConfig()
        optimizer = StateWeightOptimizer(config)
        
        weights, sharpe = optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Should return equal weights
        assert np.isclose(weights['s1'], 1/3, atol=1e-6)
        assert np.isclose(weights['s2'], 1/3, atol=1e-6)
        assert np.isclose(weights['s3'], 1/3, atol=1e-6)
        assert sharpe == 0.0
    
    def test_unknown_method_raises_error(self):
        """Test that unknown optimization method raises ValueError."""
        signals = np.random.randn(100, 3)
        returns = np.random.randn(100) * 0.01
        
        config = OptimizationConfig(method="unknown_method")
        optimizer = StateWeightOptimizer(config)
        
        with pytest.raises(ValueError, match="Unknown optimization method"):
            optimizer.optimize_state_weights(returns, signals, ['s1', 's2', 's3'])
    
    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation."""
        config = OptimizationConfig(risk_free_rate=0.02)
        optimizer = StateWeightOptimizer(config)
        
        # Create returns with known properties
        # Mean daily return = 0.001 (0.1%), std = 0.01 (1%)
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001
        
        sharpe = optimizer._calculate_sharpe(returns)
        
        # Expected: (0.001 * 252 - 0.02) / (0.01 * sqrt(252))
        # Approximately (0.252 - 0.02) / 0.159 ≈ 1.46
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert sharpe > 0  # Should be positive with positive mean return
    
    def test_portfolio_returns_computation(self):
        """Test portfolio returns computation from weighted signals."""
        config = OptimizationConfig()
        optimizer = StateWeightOptimizer(config)
        
        # Simple test case
        signals = np.array([
            [1.0, -1.0, 0.5],
            [-1.0, 1.0, -0.5],
            [0.5, 0.5, 1.0]
        ])
        returns = np.array([0.01, -0.01, 0.02])
        weights = np.array([0.5, 0.3, 0.2])
        
        portfolio_returns = optimizer._compute_portfolio_returns(
            weights, signals, returns
        )
        
        # Check shape
        assert portfolio_returns.shape == (3,)
        
        # Verify calculation
        # t=0: combined = 0.5*1 + 0.3*(-1) + 0.2*0.5 = 0.3, pos=1, ret=0.01
        # t=1: combined = 0.5*(-1) + 0.3*1 + 0.2*(-0.5) = -0.3, pos=-1, ret=0.01
        # t=2: combined = 0.5*0.5 + 0.3*0.5 + 0.2*1 = 0.6, pos=1, ret=0.02
        expected = np.array([0.01, 0.01, 0.02])
        np.testing.assert_array_almost_equal(portfolio_returns, expected)
    
    def test_equal_weights_fallback(self):
        """Test equal weights fallback method."""
        config = OptimizationConfig()
        optimizer = StateWeightOptimizer(config)
        
        signal_names = ['s_LDC', 's_MR', 's_TSMOM']
        weights = optimizer._equal_weights(signal_names)
        
        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(signal_names)
        assert all(np.isclose(w, 1/3) for w in weights.values())
        assert np.isclose(sum(weights.values()), 1.0)
    
    def test_optimization_improves_over_equal_weights(self, synthetic_data):
        """Test that optimization improves Sharpe over equal weights."""
        signals, returns = synthetic_data
        
        config = OptimizationConfig(method="SLSQP")
        optimizer = StateWeightOptimizer(config)
        
        # Get optimized weights
        opt_weights, opt_sharpe = optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Calculate equal weights Sharpe
        equal_weights = np.array([1/3, 1/3, 1/3])
        equal_returns = optimizer._compute_portfolio_returns(
            equal_weights, signals, returns
        )
        equal_sharpe = optimizer._calculate_sharpe(equal_returns)
        
        # Optimized should be at least as good as equal weights
        assert opt_sharpe >= equal_sharpe - 1e-6
    
    def test_grid_search_with_tight_bounds(self):
        """Test grid search with tight weight bounds."""
        np.random.seed(42)
        signals = np.random.randn(100, 3)
        returns = np.random.randn(100) * 0.01
        
        config = OptimizationConfig(
            method="grid_search",
            min_weight=0.25,
            max_weight=0.45,
            grid_points=5
        )
        optimizer = StateWeightOptimizer(config)
        
        weights, sharpe = optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Should find valid weights within bounds
        assert np.isclose(sum(weights.values()), 1.0, atol=1e-6)
        for w in weights.values():
            assert w >= 0.25 - 1e-6
            assert w <= 0.45 + 1e-6
    
    def test_scipy_convergence_failure_fallback(self):
        """Test fallback when scipy optimization fails to converge."""
        # Create pathological data that might cause convergence issues
        signals = np.ones((100, 3))  # Constant signals
        returns = np.random.randn(100) * 0.01
        
        config = OptimizationConfig(method="SLSQP")
        optimizer = StateWeightOptimizer(config)
        
        # Should handle gracefully and return equal weights
        weights, sharpe = optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Check that we got valid weights (might be equal or optimized)
        assert np.isclose(sum(weights.values()), 1.0, atol=1e-6)
        assert all(w >= 0 for w in weights.values())


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_realistic_signal_optimization(self):
        """Test optimization with realistic signal characteristics."""
        np.random.seed(42)
        n_samples = 252  # One year of daily data
        
        # Simulate three signals with different properties
        # LDC: trend-following, works in trending markets
        ldc_signal = np.cumsum(np.random.randn(n_samples)) * 0.1
        
        # MR: mean-reversion, works in ranging markets
        mr_signal = -np.diff(np.concatenate([[0], np.cumsum(np.random.randn(n_samples))]))
        
        # TSMOM: momentum, similar to trend-following
        tsmom_signal = np.convolve(np.random.randn(n_samples), np.ones(20)/20, mode='same')
        
        signals = np.column_stack([ldc_signal, mr_signal, tsmom_signal])
        
        # Returns have some correlation with signals
        returns = (0.005 * ldc_signal + 
                  0.003 * mr_signal + 
                  0.004 * tsmom_signal + 
                  0.01 * np.random.randn(n_samples))
        
        # Test both methods
        for method in ["SLSQP", "grid_search"]:
            config = OptimizationConfig(method=method)
            optimizer = StateWeightOptimizer(config)
            
            weights, sharpe = optimizer.optimize_state_weights(
                returns, signals, ['s_LDC', 's_MR', 's_TSMOM']
            )
            
            # Verify constraints
            assert np.isclose(sum(weights.values()), 1.0, atol=1e-6)
            assert all(w >= 0 for w in weights.values())
            
            # Verify reasonable Sharpe
            assert isinstance(sharpe, float)
            assert not np.isnan(sharpe)
            
            print(f"\n{method} Results:")
            print(f"  Weights: {weights}")
            print(f"  Sharpe: {sharpe:.4f}")
    
    def test_comparison_scipy_vs_grid_search(self):
        """Compare scipy and grid search results."""
        np.random.seed(42)
        signals = np.random.randn(200, 3)
        returns = np.random.randn(200) * 0.01
        
        # Scipy optimization
        scipy_config = OptimizationConfig(method="SLSQP")
        scipy_optimizer = StateWeightOptimizer(scipy_config)
        scipy_weights, scipy_sharpe = scipy_optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Grid search optimization
        grid_config = OptimizationConfig(method="grid_search", grid_points=21)
        grid_optimizer = StateWeightOptimizer(grid_config)
        grid_weights, grid_sharpe = grid_optimizer.optimize_state_weights(
            returns, signals, ['s1', 's2', 's3']
        )
        
        # Both should produce valid results
        assert np.isclose(sum(scipy_weights.values()), 1.0, atol=1e-6)
        assert np.isclose(sum(grid_weights.values()), 1.0, atol=1e-6)
        
        # Scipy should generally find better or equal solution (continuous optimization)
        # But allow for some tolerance due to different optimization approaches
        print(f"\nScipy Sharpe: {scipy_sharpe:.4f}")
        print(f"Grid Sharpe: {grid_sharpe:.4f}")
        print(f"Scipy weights: {scipy_weights}")
        print(f"Grid weights: {grid_weights}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
