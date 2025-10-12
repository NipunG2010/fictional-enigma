"""
Tests for walk-forward validation functionality.

This module tests the WalkForwardValidator class to ensure proper
time-series cross-validation and overfitting detection.
"""

import pytest
import numpy as np
from imp.hmm.weight_optimizer import (
    WalkForwardValidator,
    WalkForwardConfig,
    OptimizationConfig
)


class TestWalkForwardValidator:
    """Test suite for WalkForwardValidator."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        T = 500
        n_signals = 3
        n_states = 2
        
        # Generate synthetic signals
        observations = np.random.randn(T, n_signals)
        
        # Generate synthetic returns with some correlation to signals
        returns = np.random.randn(T) * 0.01
        for i in range(n_signals):
            returns += observations[:, i] * 0.002
        
        # Generate state sequence
        state_sequence = np.random.choice([0, 1], size=T, p=[0.6, 0.4])
        
        return {
            'observations': observations,
            'returns': returns,
            'state_sequence': state_sequence,
            'n_states': n_states
        }
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        wf_config = WalkForwardConfig(
            n_folds=5,
            train_ratio=0.7,
            min_train_size=100,
            min_test_size=30,
            overfitting_threshold=50.0
        )
        opt_config = OptimizationConfig(
            method="SLSQP",
            min_observations=30
        )
        return WalkForwardValidator(wf_config, opt_config)
    
    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator.config.n_folds == 5
        assert validator.config.train_ratio == 0.7
        assert validator.config.overfitting_threshold == 50.0
        assert validator.optimizer is not None
    
    def test_generate_fold_splits(self, validator):
        """Test fold split generation."""
        T = 500
        fold_splits = validator._generate_fold_splits(T)
        
        # Should generate folds
        assert len(fold_splits) > 0
        
        # Check each fold
        for train_indices, test_indices in fold_splits:
            # Training set should come before test set
            assert np.max(train_indices) <= np.min(test_indices)
            
            # No overlap
            assert len(np.intersect1d(train_indices, test_indices)) == 0
            
            # Minimum sizes
            assert len(train_indices) >= validator.config.min_train_size
            assert len(test_indices) >= validator.config.min_test_size
    
    def test_generate_fold_splits_expanding_window(self, validator):
        """Test that fold splits use expanding window."""
        T = 500
        fold_splits = validator._generate_fold_splits(T)
        
        # Training set should grow with each fold
        train_sizes = [len(train) for train, _ in fold_splits]
        assert all(train_sizes[i] <= train_sizes[i+1] for i in range(len(train_sizes)-1))
    
    def test_generate_fold_splits_insufficient_data(self, validator):
        """Test fold generation with insufficient data."""
        T = 50  # Too small
        fold_splits = validator._generate_fold_splits(T)
        
        # Should return empty or very few folds
        assert len(fold_splits) <= 1
    
    def test_validate_fold(self, validator, sample_data):
        """Test single fold validation."""
        T = len(sample_data['returns'])
        train_indices = np.arange(0, 300)
        test_indices = np.arange(300, 400)
        
        result = validator._validate_fold(
            fold_idx=0,
            train_indices=train_indices,
            test_indices=test_indices,
            observations=sample_data['observations'],
            returns=sample_data['returns'],
            state_sequence=sample_data['state_sequence'],
            n_states=sample_data['n_states'],
            signal_names=['s_LDC', 's_MR', 's_TSMOM']
        )
        
        # Check result structure
        assert 'fold_idx' in result
        assert 'train_size' in result
        assert 'test_size' in result
        assert 'optimized_weights' in result
        assert 'in_sample_sharpe' in result
        assert 'out_of_sample_sharpe' in result
        assert 'degradation_pct' in result
        
        # Check values
        assert result['fold_idx'] == 0
        assert result['train_size'] == 300
        assert result['test_size'] == 100
        assert len(result['optimized_weights']) == sample_data['n_states']
        
        # Sharpe ratios should be finite
        assert np.isfinite(result['in_sample_sharpe'])
        assert np.isfinite(result['out_of_sample_sharpe'])
    
    def test_validate_robustness(self, validator, sample_data):
        """Test full walk-forward validation."""
        result = validator.validate_robustness(
            observations=sample_data['observations'],
            returns=sample_data['returns'],
            state_sequence=sample_data['state_sequence'],
            n_states=sample_data['n_states'],
            signal_names=['s_LDC', 's_MR', 's_TSMOM']
        )
        
        # Check result structure
        assert 'fold_results' in result
        assert 'aggregate_metrics' in result
        assert 'overfitting_detected' in result
        assert 'overfitting_details' in result
        assert 'recommendation' in result
        assert 'n_folds' in result
        
        # Check fold results
        assert len(result['fold_results']) > 0
        for fold_result in result['fold_results']:
            assert 'in_sample_sharpe' in fold_result
            assert 'out_of_sample_sharpe' in fold_result
            assert 'degradation_pct' in fold_result
        
        # Check aggregate metrics
        agg = result['aggregate_metrics']
        assert 'mean_in_sample_sharpe' in agg
        assert 'mean_out_of_sample_sharpe' in agg
        assert 'mean_degradation_pct' in agg
        assert 'consistency_ratio' in agg
        
        # Check overfitting detection
        assert isinstance(result['overfitting_detected'], bool)
        assert isinstance(result['recommendation'], str)
    
    def test_validate_robustness_insufficient_data(self, validator):
        """Test validation with insufficient data."""
        T = 50
        observations = np.random.randn(T, 3)
        returns = np.random.randn(T) * 0.01
        state_sequence = np.random.choice([0, 1], size=T)
        
        result = validator.validate_robustness(
            observations=observations,
            returns=returns,
            state_sequence=state_sequence,
            n_states=2
        )
        
        # Should return error
        assert 'error' in result
        assert result['overfitting_detected'] == False
    
    def test_aggregate_fold_results(self, validator):
        """Test aggregation of fold results."""
        fold_results = [
            {
                'in_sample_sharpe': 1.5,
                'out_of_sample_sharpe': 1.2,
                'degradation_pct': 20.0
            },
            {
                'in_sample_sharpe': 1.8,
                'out_of_sample_sharpe': 1.0,
                'degradation_pct': 44.4
            },
            {
                'in_sample_sharpe': 1.6,
                'out_of_sample_sharpe': 1.3,
                'degradation_pct': 18.75
            }
        ]
        
        agg = validator._aggregate_fold_results(fold_results)
        
        # Check calculations
        assert np.isclose(agg['mean_in_sample_sharpe'], 1.633, atol=0.01)
        assert np.isclose(agg['mean_out_of_sample_sharpe'], 1.167, atol=0.01)
        assert np.isclose(agg['mean_degradation_pct'], 27.72, atol=0.01)
        assert agg['max_degradation_pct'] == 44.4
        assert agg['min_degradation_pct'] == 18.75
        
        # Consistency ratio
        expected_ratio = 1.167 / 1.633
        assert np.isclose(agg['consistency_ratio'], expected_ratio, atol=0.01)
    
    def test_detect_overfitting_no_overfitting(self, validator):
        """Test overfitting detection when no overfitting present."""
        fold_results = [
            {
                'in_sample_sharpe': 1.5,
                'out_of_sample_sharpe': 1.4,
                'degradation_pct': 6.7
            },
            {
                'in_sample_sharpe': 1.6,
                'out_of_sample_sharpe': 1.5,
                'degradation_pct': 6.25
            }
        ]
        
        agg = validator._aggregate_fold_results(fold_results)
        overfitting, details = validator._detect_overfitting(fold_results, agg)
        
        # Should not detect overfitting
        assert overfitting == False
        assert details['folds_with_degradation'] == 0
    
    def test_detect_overfitting_with_overfitting(self, validator):
        """Test overfitting detection when overfitting present."""
        fold_results = [
            {
                'in_sample_sharpe': 2.0,
                'out_of_sample_sharpe': 0.5,
                'degradation_pct': 75.0
            },
            {
                'in_sample_sharpe': 1.8,
                'out_of_sample_sharpe': 0.6,
                'degradation_pct': 66.7
            }
        ]
        
        agg = validator._aggregate_fold_results(fold_results)
        overfitting, details = validator._detect_overfitting(fold_results, agg)
        
        # Should detect overfitting
        assert overfitting == True
        assert details['folds_with_degradation'] == 2
        assert details['mean_degradation_pct'] > validator.config.overfitting_threshold
    
    def test_generate_recommendation_robust(self, validator):
        """Test recommendation generation for robust weights."""
        aggregate_metrics = {
            'mean_out_of_sample_sharpe': 1.5,
            'consistency_ratio': 0.85
        }
        
        recommendation = validator._generate_recommendation(
            aggregate_metrics,
            overfitting_detected=False,
            overfitting_details={}
        )
        
        assert 'ROBUST' in recommendation or 'robust' in recommendation.lower()
        assert 'production' in recommendation.lower()
    
    def test_generate_recommendation_overfitting(self, validator):
        """Test recommendation generation when overfitting detected."""
        aggregate_metrics = {
            'mean_out_of_sample_sharpe': 0.5,
            'consistency_ratio': 0.3
        }
        overfitting_details = {
            'mean_degradation_pct': 70.0,
            'max_degradation_pct': 80.0,
            'folds_with_degradation': 4,
            'total_folds': 5
        }
        
        recommendation = validator._generate_recommendation(
            aggregate_metrics,
            overfitting_detected=True,
            overfitting_details=overfitting_details
        )
        
        assert 'OVERFITTING' in recommendation
        assert 'equal weights' in recommendation.lower()
    
    def test_generate_recommendation_poor_generalization(self, validator):
        """Test recommendation for poor generalization."""
        aggregate_metrics = {
            'mean_out_of_sample_sharpe': -0.5,
            'consistency_ratio': -0.3
        }
        
        recommendation = validator._generate_recommendation(
            aggregate_metrics,
            overfitting_detected=False,
            overfitting_details={}
        )
        
        assert 'POOR GENERALIZATION' in recommendation or 'negative' in recommendation.lower()
    
    def test_compute_portfolio_returns(self, validator, sample_data):
        """Test portfolio returns computation."""
        state_weights = [
            {'s_LDC': 0.5, 's_MR': 0.3, 's_TSMOM': 0.2},
            {'s_LDC': 0.3, 's_MR': 0.4, 's_TSMOM': 0.3}
        ]
        
        portfolio_returns = validator._compute_portfolio_returns(
            state_weights,
            sample_data['observations'],
            sample_data['returns'],
            sample_data['state_sequence'],
            ['s_LDC', 's_MR', 's_TSMOM']
        )
        
        # Check shape
        assert portfolio_returns.shape == sample_data['returns'].shape
        
        # Check finite values
        assert np.all(np.isfinite(portfolio_returns))
    
    def test_calculate_sharpe(self, validator):
        """Test Sharpe ratio calculation."""
        # Positive returns
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        sharpe = validator._calculate_sharpe(returns)
        assert np.isfinite(sharpe)
        
        # Zero variance
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = validator._calculate_sharpe(returns)
        assert sharpe == 0.0
        
        # Empty returns
        returns = np.array([])
        sharpe = validator._calculate_sharpe(returns)
        assert sharpe == 0.0
    
    def test_walk_forward_with_different_configs(self, sample_data):
        """Test walk-forward validation with different configurations."""
        # Test with different number of folds
        for n_folds in [3, 5, 7]:
            wf_config = WalkForwardConfig(
                n_folds=n_folds,
                min_train_size=100,
                min_test_size=30
            )
            opt_config = OptimizationConfig(method="SLSQP")
            validator = WalkForwardValidator(wf_config, opt_config)
            
            result = validator.validate_robustness(
                observations=sample_data['observations'],
                returns=sample_data['returns'],
                state_sequence=sample_data['state_sequence'],
                n_states=sample_data['n_states']
            )
            
            # Should complete successfully
            assert 'fold_results' in result
            assert len(result['fold_results']) > 0
    
    def test_walk_forward_with_grid_search(self, sample_data):
        """Test walk-forward validation with grid search optimization."""
        wf_config = WalkForwardConfig(n_folds=3)
        opt_config = OptimizationConfig(
            method="grid_search",
            grid_points=5,
            min_observations=30
        )
        validator = WalkForwardValidator(wf_config, opt_config)
        
        result = validator.validate_robustness(
            observations=sample_data['observations'],
            returns=sample_data['returns'],
            state_sequence=sample_data['state_sequence'],
            n_states=sample_data['n_states']
        )
        
        # Should complete successfully
        assert 'fold_results' in result
        assert len(result['fold_results']) > 0
        assert 'aggregate_metrics' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
