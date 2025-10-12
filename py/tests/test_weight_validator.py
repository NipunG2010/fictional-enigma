"""
Tests for WeightValidator class.

Tests constraint validation, performance comparison, and statistical
significance testing for fusion weight optimization.
"""

import pytest
import numpy as np
from imp.hmm.weight_optimizer import WeightValidator


class TestWeightValidator:
    """Test suite for WeightValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return WeightValidator(risk_free_rate=0.02)
    
    @pytest.fixture
    def valid_weights(self):
        """Valid state weights for 2 states, 3 signals."""
        return [
            {'s_LDC': 0.5, 's_MR': 0.3, 's_TSMOM': 0.2},
            {'s_LDC': 0.3, 's_MR': 0.4, 's_TSMOM': 0.3}
        ]
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        T = 252  # One year of daily data
        
        # Generate signals
        observations = np.random.randn(T, 3)
        
        # Generate returns with some correlation to signals
        returns = 0.001 + 0.01 * np.random.randn(T)
        
        # Generate state sequence (2 states)
        state_sequence = np.random.choice([0, 1], size=T)
        
        return observations, returns, state_sequence
    
    # ========================================================================
    # Constraint Validation Tests
    # ========================================================================
    
    def test_validate_constraints_valid_weights(self, validator, valid_weights):
        """Test constraint validation with valid weights."""
        result = validator._validate_constraints(
            valid_weights,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_constraints_weights_not_sum_to_one(self, validator):
        """Test constraint validation when weights don't sum to 1."""
        invalid_weights = [
            {'s_LDC': 0.5, 's_MR': 0.3, 's_TSMOM': 0.3}  # Sum = 1.1
        ]
        
        result = validator._validate_constraints(
            invalid_weights,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert 'sum' in result['errors'][0].lower()
    
    def test_validate_constraints_negative_weight(self, validator):
        """Test constraint validation with negative weight."""
        invalid_weights = [
            {'s_LDC': 0.6, 's_MR': 0.5, 's_TSMOM': -0.1}  # Negative weight
        ]
        
        result = validator._validate_constraints(
            invalid_weights,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['valid'] is False
        assert any('negative' in err.lower() for err in result['errors'])
    
    def test_validate_constraints_weight_exceeds_one(self, validator):
        """Test constraint validation when weight exceeds 1."""
        invalid_weights = [
            {'s_LDC': 1.5, 's_MR': 0.0, 's_TSMOM': -0.5}
        ]
        
        result = validator._validate_constraints(
            invalid_weights,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['valid'] is False
        assert any('exceeds' in err.lower() for err in result['errors'])
    
    def test_validate_constraints_missing_signal(self, validator):
        """Test constraint validation with missing signal."""
        invalid_weights = [
            {'s_LDC': 0.5, 's_MR': 0.5}  # Missing s_TSMOM
        ]
        
        result = validator._validate_constraints(
            invalid_weights,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['valid'] is False
        assert any('missing' in err.lower() for err in result['errors'])
    
    def test_validate_constraints_extra_signal(self, validator):
        """Test constraint validation with extra signal."""
        invalid_weights = [
            {'s_LDC': 0.25, 's_MR': 0.25, 's_TSMOM': 0.25, 's_EXTRA': 0.25}
        ]
        
        result = validator._validate_constraints(
            invalid_weights,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['valid'] is False
        assert any('unexpected' in err.lower() for err in result['errors'])
    
    def test_validate_constraints_nan_weight(self, validator):
        """Test constraint validation with NaN weight."""
        invalid_weights = [
            {'s_LDC': 0.5, 's_MR': 0.5, 's_TSMOM': np.nan}
        ]
        
        result = validator._validate_constraints(
            invalid_weights,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['valid'] is False
        assert any('invalid' in err.lower() for err in result['errors'])
    
    def test_validate_constraints_empty_weights(self, validator):
        """Test constraint validation with empty weights."""
        result = validator._validate_constraints(
            [],
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    # ========================================================================
    # Performance Comparison Tests
    # ========================================================================
    
    def test_compare_performance_basic(self, validator, valid_weights, sample_data):
        """Test basic performance comparison."""
        observations, returns, state_sequence = sample_data
        
        result = validator._compare_performance(
            valid_weights,
            observations,
            returns,
            state_sequence,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        # Check all required metrics present
        assert 'optimized_sharpe' in result
        assert 'baseline_sharpe' in result
        assert 'optimized_total_return' in result
        assert 'baseline_total_return' in result
        assert 'optimized_volatility' in result
        assert 'baseline_volatility' in result
        assert 'optimized_max_drawdown' in result
        assert 'baseline_max_drawdown' in result
        assert 'optimized_win_rate' in result
        assert 'baseline_win_rate' in result
        assert 'improvement' in result
        assert 'improvement_pct' in result
        
        # Check metrics are numeric
        assert isinstance(result['optimized_sharpe'], float)
        assert isinstance(result['baseline_sharpe'], float)
        assert isinstance(result['improvement'], float)
        assert isinstance(result['improvement_pct'], float)
    
    def test_compare_performance_returns_arrays(self, validator, valid_weights, sample_data):
        """Test that performance comparison returns strategy returns."""
        observations, returns, state_sequence = sample_data
        
        result = validator._compare_performance(
            valid_weights,
            observations,
            returns,
            state_sequence,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert 'optimized_returns' in result
        assert 'baseline_returns' in result
        assert len(result['optimized_returns']) == len(returns)
        assert len(result['baseline_returns']) == len(returns)
    
    def test_compute_strategy_returns(self, validator, valid_weights, sample_data):
        """Test strategy returns computation."""
        observations, returns, state_sequence = sample_data
        
        portfolio_returns = validator._compute_strategy_returns(
            valid_weights,
            observations,
            returns,
            state_sequence,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert len(portfolio_returns) == len(returns)
        assert not np.any(np.isnan(portfolio_returns))
        assert not np.any(np.isinf(portfolio_returns))
    
    def test_calculate_metrics(self, validator):
        """Test metrics calculation."""
        # Generate sample returns
        np.random.seed(42)
        returns = 0.001 + 0.01 * np.random.randn(252)
        
        metrics = validator._calculate_metrics(returns)
        
        assert 'sharpe' in metrics
        assert 'total_return' in metrics
        assert 'volatility' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # Check reasonable values
        assert -10 < metrics['sharpe'] < 10  # Reasonable Sharpe range
        assert 0 <= metrics['win_rate'] <= 1  # Win rate is probability
        assert metrics['volatility'] >= 0  # Volatility is non-negative
        assert metrics['max_drawdown'] <= 0  # Drawdown is negative
    
    def test_calculate_metrics_empty_returns(self, validator):
        """Test metrics calculation with empty returns."""
        metrics = validator._calculate_metrics(np.array([]))
        
        assert metrics['sharpe'] == 0.0
        assert metrics['total_return'] == 0.0
        assert metrics['volatility'] == 0.0
        assert metrics['max_drawdown'] == 0.0
        assert metrics['win_rate'] == 0.0
    
    def test_calculate_metrics_zero_volatility(self, validator):
        """Test metrics calculation with zero volatility."""
        returns = np.zeros(100)
        metrics = validator._calculate_metrics(returns)
        
        assert metrics['sharpe'] == 0.0
        assert metrics['volatility'] == 0.0
    
    # ========================================================================
    # Statistical Significance Tests
    # ========================================================================
    
    def test_test_significance_basic(self, validator):
        """Test statistical significance testing."""
        np.random.seed(42)
        
        # Generate returns where optimized is better
        baseline_returns = 0.0005 + 0.01 * np.random.randn(252)
        optimized_returns = baseline_returns + 0.0002  # Slightly better
        
        result = validator._test_significance(optimized_returns, baseline_returns)
        
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'significant_at_5pct' in result
        assert 'significant_at_1pct' in result
        assert 'degrees_of_freedom' in result
        assert 'mean_difference' in result
        assert 'interpretation' in result
        
        # Check types
        assert isinstance(result['t_statistic'], float)
        assert isinstance(result['p_value'], float)
        assert isinstance(result['significant_at_5pct'], bool)
        assert isinstance(result['significant_at_1pct'], bool)
        assert isinstance(result['degrees_of_freedom'], int)
    
    def test_test_significance_clearly_better(self, validator):
        """Test significance when optimized is clearly better."""
        np.random.seed(42)
        
        # Generate returns where optimized is significantly better
        baseline_returns = 0.0 + 0.01 * np.random.randn(252)
        optimized_returns = baseline_returns + 0.001  # Much better
        
        result = validator._test_significance(optimized_returns, baseline_returns)
        
        # Should be significant
        assert result['p_value'] < 0.05
        assert result['significant_at_5pct'] is True
        assert result['mean_difference'] > 0
    
    def test_test_significance_no_difference(self, validator):
        """Test significance when there's no difference."""
        np.random.seed(42)
        
        # Same returns
        returns = 0.001 + 0.01 * np.random.randn(252)
        
        result = validator._test_significance(returns, returns)
        
        # Should not be significant (identical returns)
        # Note: t_statistic will be NaN due to zero variance in differences
        assert np.isnan(result['t_statistic']) or result['t_statistic'] == 0.0
        assert result['p_value'] == 1.0 or np.isnan(result['p_value'])
        assert result['significant_at_5pct'] is False
        assert result['mean_difference'] == 0.0
    
    # ========================================================================
    # Full Validation Tests
    # ========================================================================
    
    def test_validate_weights_full_valid(self, validator, valid_weights, sample_data):
        """Test full validation with valid weights."""
        observations, returns, state_sequence = sample_data
        
        result = validator.validate_weights(
            valid_weights,
            observations,
            returns,
            state_sequence,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert 'constraints_valid' in result
        assert 'constraint_errors' in result
        assert 'performance_comparison' in result
        assert 'statistical_tests' in result
        assert 'recommendation' in result
        
        assert result['constraints_valid'] is True
        assert len(result['constraint_errors']) == 0
    
    def test_validate_weights_constraint_failure(self, validator, sample_data):
        """Test full validation with constraint violations."""
        observations, returns, state_sequence = sample_data
        
        # Invalid weights (don't sum to 1)
        invalid_weights = [
            {'s_LDC': 0.5, 's_MR': 0.3, 's_TSMOM': 0.3},
            {'s_LDC': 0.3, 's_MR': 0.4, 's_TSMOM': 0.3}
        ]
        
        result = validator.validate_weights(
            invalid_weights,
            observations,
            returns,
            state_sequence,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['constraints_valid'] is False
        assert len(result['constraint_errors']) > 0
        assert 'REJECT' in result['recommendation']
    
    def test_validate_weights_performance_improvement(self, validator, sample_data):
        """Test validation when optimized weights improve performance."""
        observations, returns, state_sequence = sample_data
        
        # Create weights that should perform better (higher weight on first signal)
        # This is synthetic, but tests the logic
        optimized_weights = [
            {'s_LDC': 0.6, 's_MR': 0.3, 's_TSMOM': 0.1},
            {'s_LDC': 0.5, 's_MR': 0.3, 's_TSMOM': 0.2}
        ]
        
        result = validator.validate_weights(
            optimized_weights,
            observations,
            returns,
            state_sequence,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['constraints_valid'] is True
        assert 'performance_comparison' in result
        assert 'statistical_tests' in result
        
        # Should have recommendation
        assert len(result['recommendation']) > 0
    
    def test_validate_weights_recommendation_logic(self, validator):
        """Test recommendation logic based on performance."""
        np.random.seed(42)
        T = 252
        
        observations = np.random.randn(T, 3)
        returns = 0.001 + 0.01 * np.random.randn(T)
        state_sequence = np.random.choice([0, 1], size=T)
        
        # Valid weights
        weights = [
            {'s_LDC': 0.4, 's_MR': 0.3, 's_TSMOM': 0.3},
            {'s_LDC': 0.3, 's_MR': 0.4, 's_TSMOM': 0.3}
        ]
        
        result = validator.validate_weights(
            weights,
            observations,
            returns,
            state_sequence,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        # Should have one of: ACCEPT, CAUTION, or REJECT
        recommendation = result['recommendation']
        assert any(keyword in recommendation for keyword in ['ACCEPT', 'CAUTION', 'REJECT'])
    
    # ========================================================================
    # Edge Cases
    # ========================================================================
    
    def test_validate_weights_single_state(self, validator):
        """Test validation with single state."""
        np.random.seed(42)
        T = 100
        
        observations = np.random.randn(T, 3)
        returns = 0.001 + 0.01 * np.random.randn(T)
        state_sequence = np.zeros(T, dtype=int)  # All same state
        
        weights = [
            {'s_LDC': 0.4, 's_MR': 0.3, 's_TSMOM': 0.3}
        ]
        
        result = validator.validate_weights(
            weights,
            observations,
            returns,
            state_sequence,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['constraints_valid'] is True
    
    def test_validate_weights_many_states(self, validator):
        """Test validation with many states."""
        np.random.seed(42)
        T = 500
        n_states = 5
        
        observations = np.random.randn(T, 3)
        returns = 0.001 + 0.01 * np.random.randn(T)
        state_sequence = np.random.choice(n_states, size=T)
        
        # Create valid weights for all states
        weights = [
            {'s_LDC': 0.4, 's_MR': 0.3, 's_TSMOM': 0.3}
            for _ in range(n_states)
        ]
        
        result = validator.validate_weights(
            weights,
            observations,
            returns,
            state_sequence,
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        assert result['constraints_valid'] is True
        assert len(result['constraint_errors']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
