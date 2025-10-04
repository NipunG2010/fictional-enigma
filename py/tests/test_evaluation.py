"""
Tests for HMM evaluation and comparison framework.
"""

import pytest
import numpy as np
import pandas as pd
from imp.evaluation import HMMEvaluator, ModelComparison, EvaluationMetrics
from imp.hmm.trainer import EnhancedHMMTrainer


@pytest.fixture
def sample_observations():
    """Generate sample observation data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 3
    
    # Generate synthetic data with regime changes
    observations = []
    for i in range(n_samples):
        if i < 70:
            # Regime 1: low volatility
            obs = np.random.randn(n_features) * 0.5
        elif i < 140:
            # Regime 2: high volatility
            obs = np.random.randn(n_features) * 2.0
        else:
            # Regime 3: medium volatility with trend
            obs = np.random.randn(n_features) * 1.0 + 0.5
        observations.append(obs)
    
    return np.array(observations)


@pytest.fixture
def trained_trainer(sample_observations):
    """Create a trained HMM trainer."""
    trainer = EnhancedHMMTrainer(n_states=3, library="hmmlearn", random_state=42)
    trainer.train(sample_observations, n_iterations=50)
    return trainer


@pytest.fixture
def evaluator():
    """Create HMM evaluator instance."""
    return HMMEvaluator(random_state=42)


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating evaluation metrics."""
        metrics = EvaluationMetrics(
            log_likelihood=-100.5,
            aic=210.0,
            bic=220.0,
            perplexity=1.5,
            n_parameters=15,
            n_samples=100
        )
        
        assert metrics.log_likelihood == -100.5
        assert metrics.aic == 210.0
        assert metrics.bic == 220.0
        assert metrics.perplexity == 1.5
        assert metrics.n_parameters == 15
        assert metrics.n_samples == 100
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = EvaluationMetrics(
            log_likelihood=-100.5,
            aic=210.0,
            bic=220.0,
            perplexity=1.5,
            n_parameters=15,
            n_samples=100
        )
        
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['log_likelihood'] == -100.5
        assert metrics_dict['aic'] == 210.0


class TestHMMEvaluator:
    """Test HMMEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = HMMEvaluator(random_state=42)
        assert evaluator.random_state == 42
        assert len(evaluator.evaluation_results) == 0
    
    def test_evaluate_model(self, evaluator, trained_trainer, sample_observations):
        """Test single model evaluation."""
        metrics = evaluator.evaluate_model(trained_trainer.trainer, sample_observations)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.log_likelihood < 0  # Log-likelihood should be negative
        assert metrics.aic > 0
        assert metrics.bic > 0
        assert metrics.perplexity > 0
        assert metrics.n_parameters > 0
        assert metrics.n_samples == len(sample_observations)
    
    def test_cross_validate(self, evaluator, sample_observations):
        """Test cross-validation."""
        trainer_config = {
            'n_states': 3,
            'library': 'hmmlearn',
            'covariance_type': 'diag',
            'random_state': 42
        }
        
        cv_results = evaluator.cross_validate(
            sample_observations,
            trainer_config,
            cv_folds=3,
            n_iterations=50
        )
        
        assert isinstance(cv_results, dict)
        assert 'log_likelihood_mean' in cv_results
        assert 'log_likelihood_std' in cv_results
        assert 'log_likelihood_values' in cv_results
        assert len(cv_results['log_likelihood_values']) == 3
    
    def test_cross_validate_invalid_folds(self, evaluator, sample_observations):
        """Test cross-validation with invalid number of folds."""
        trainer_config = {'n_states': 3, 'library': 'hmmlearn'}
        
        with pytest.raises(ValueError, match="Number of CV folds must be at least 2"):
            evaluator.cross_validate(sample_observations, trainer_config, cv_folds=1)
    
    def test_regime_stability_analysis(self, evaluator, trained_trainer, sample_observations):
        """Test regime stability analysis."""
        state_probs = trained_trainer.trainer.predict_state_probabilities(sample_observations)
        
        stability_metrics = evaluator.regime_stability_analysis(state_probs, min_duration=5)
        
        assert stability_metrics.mean_durations is not None
        assert stability_metrics.median_durations is not None
        assert stability_metrics.max_durations is not None
        assert stability_metrics.stable_periods is not None
        assert stability_metrics.total_periods is not None
        assert stability_metrics.transition_frequencies is not None
        assert stability_metrics.state_persistence is not None
        
        # Check that all states are represented
        n_states = state_probs.shape[1]
        assert len(stability_metrics.mean_durations) == n_states
        assert len(stability_metrics.state_persistence) == n_states
        
        # Check that persistence values are probabilities
        for persistence in stability_metrics.state_persistence.values():
            assert 0 <= persistence <= 1
    
    def test_regime_stability_to_dict(self, evaluator, trained_trainer, sample_observations):
        """Test converting stability metrics to dictionary."""
        state_probs = trained_trainer.trainer.predict_state_probabilities(sample_observations)
        stability_metrics = evaluator.regime_stability_analysis(state_probs)
        
        metrics_dict = stability_metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert 'mean_durations' in metrics_dict
        assert 'state_persistence' in metrics_dict
    
    def test_compare_models(self, evaluator, sample_observations):
        """Test comparing multiple models."""
        trainer_configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        ]
        
        comparison_df = evaluator.compare_models(
            sample_observations,
            trainer_configs,
            n_iterations=50,
            perform_cv=False,  # Skip CV for speed
            analyze_stability=True
        )
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'config' in comparison_df.columns
        assert 'rank' in comparison_df.columns
        assert 'log_likelihood' in comparison_df.columns
        assert 'aic' in comparison_df.columns
        assert 'bic' in comparison_df.columns
        
        # Check that ranks are assigned
        assert comparison_df['rank'].notna().all()
        assert set(comparison_df['rank']) == {1, 2}
    
    def test_compare_models_with_cv(self, evaluator, sample_observations):
        """Test comparing models with cross-validation."""
        trainer_configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        ]
        
        comparison_df = evaluator.compare_models(
            sample_observations,
            trainer_configs,
            n_iterations=50,
            perform_cv=True,
            cv_folds=3,
            analyze_stability=False
        )
        
        assert isinstance(comparison_df, pd.DataFrame)
        # Check for CV columns
        cv_columns = [col for col in comparison_df.columns if col.startswith('cv_')]
        assert len(cv_columns) > 0
    
    def test_statistical_significance_test(self, evaluator, sample_observations):
        """Test statistical significance testing between models."""
        trainer_configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        ]
        
        # First compare models to populate evaluation_results
        evaluator.compare_models(
            sample_observations,
            trainer_configs,
            n_iterations=50,
            perform_cv=True,
            cv_folds=3
        )
        
        # Get config names
        config_names = list(evaluator.evaluation_results.keys())
        
        # Test significance
        result = evaluator.statistical_significance_test(
            config_names[0],
            config_names[1],
            metric='log_likelihood'
        )
        
        assert isinstance(result, dict)
        assert 'config1' in result
        assert 'config2' in result
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'cohens_d' in result
        assert isinstance(result['significant'], bool)
    
    def test_statistical_significance_missing_config(self, evaluator):
        """Test significance test with missing configuration."""
        with pytest.raises(ValueError, match="not found in evaluation results"):
            evaluator.statistical_significance_test('config1', 'config2')
    
    def test_select_best_model(self, evaluator, sample_observations):
        """Test selecting best model."""
        trainer_configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        ]
        
        evaluator.compare_models(
            sample_observations,
            trainer_configs,
            n_iterations=50,
            perform_cv=False
        )
        
        best_config = evaluator.select_best_model(criteria=['bic'])
        
        assert isinstance(best_config, str)
        assert best_config in evaluator.evaluation_results
    
    def test_select_best_model_multiple_criteria(self, evaluator, sample_observations):
        """Test selecting best model with multiple criteria."""
        trainer_configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        ]
        
        evaluator.compare_models(
            sample_observations,
            trainer_configs,
            n_iterations=50,
            perform_cv=False
        )
        
        best_config = evaluator.select_best_model(
            criteria=['bic', 'log_likelihood'],
            weights=[0.6, 0.4]
        )
        
        assert isinstance(best_config, str)
        assert best_config in evaluator.evaluation_results
    
    def test_select_best_model_invalid_weights(self, evaluator, sample_observations):
        """Test selecting best model with invalid weights."""
        trainer_configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        ]
        
        evaluator.compare_models(sample_observations, trainer_configs, n_iterations=50)
        
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            evaluator.select_best_model(criteria=['bic'], weights=[0.5])
    
    def test_select_best_model_no_results(self, evaluator):
        """Test selecting best model with no evaluation results."""
        with pytest.raises(ValueError, match="No evaluation results available"):
            evaluator.select_best_model()
    
    def test_get_evaluation_summary(self, evaluator, sample_observations):
        """Test getting evaluation summary."""
        trainer_configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        ]
        
        evaluator.compare_models(
            sample_observations,
            trainer_configs,
            n_iterations=50,
            perform_cv=False
        )
        
        summary = evaluator.get_evaluation_summary()
        
        assert isinstance(summary, dict)
        assert 'n_configurations' in summary
        assert summary['n_configurations'] == 2
        assert 'configurations' in summary
        assert 'best_by_bic' in summary
        assert 'best_by_aic' in summary
        assert 'best_by_likelihood' in summary
    
    def test_get_evaluation_summary_empty(self, evaluator):
        """Test getting summary with no results."""
        summary = evaluator.get_evaluation_summary()
        assert 'message' in summary


class TestModelComparison:
    """Test ModelComparison dataclass."""
    
    def test_comparison_creation(self):
        """Test creating model comparison."""
        metrics = EvaluationMetrics(
            log_likelihood=-100.5,
            aic=210.0,
            bic=220.0,
            perplexity=1.5,
            n_parameters=15,
            n_samples=100
        )
        
        comparison = ModelComparison(
            config_name="test_config",
            metrics=metrics,
            rank=1
        )
        
        assert comparison.config_name == "test_config"
        assert comparison.metrics == metrics
        assert comparison.rank == 1
    
    def test_comparison_to_dict(self):
        """Test converting comparison to dictionary."""
        metrics = EvaluationMetrics(
            log_likelihood=-100.5,
            aic=210.0,
            bic=220.0,
            perplexity=1.5,
            n_parameters=15,
            n_samples=100
        )
        
        comparison = ModelComparison(
            config_name="test_config",
            metrics=metrics,
            rank=1
        )
        
        comp_dict = comparison.to_dict()
        assert isinstance(comp_dict, dict)
        assert comp_dict['config_name'] == "test_config"
        assert comp_dict['rank'] == 1
        assert 'metrics' in comp_dict


class TestIntegration:
    """Integration tests for evaluation framework."""
    
    def test_full_evaluation_workflow(self, sample_observations):
        """Test complete evaluation workflow."""
        evaluator = HMMEvaluator(random_state=42)
        
        # Define configurations to compare
        configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'full', 'random_state': 42},
        ]
        
        # Compare models
        comparison_df = evaluator.compare_models(
            sample_observations,
            configs,
            n_iterations=50,
            perform_cv=True,
            cv_folds=3,
            analyze_stability=True
        )
        
        # Verify results
        assert len(comparison_df) == 3
        assert comparison_df['rank'].notna().all()
        
        # Select best model
        best_config = evaluator.select_best_model(criteria=['bic', 'log_likelihood'])
        assert best_config in evaluator.evaluation_results
        
        # Get summary
        summary = evaluator.get_evaluation_summary()
        assert summary['n_configurations'] == 3
        
        # Test significance between top 2 models
        sorted_configs = comparison_df.sort_values('rank')['config'].tolist()
        if len(sorted_configs) >= 2:
            sig_result = evaluator.statistical_significance_test(
                sorted_configs[0],
                sorted_configs[1],
                metric='log_likelihood'
            )
            assert 'p_value' in sig_result
