"""
Tests for advanced hyperparameter optimization and model selection.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from imp.tuning import (
    AutomatedModelSelector,
    SelectionCriteria,
    EnsembleEvaluator,
    SensitivityAnalyzer,
    ReportGenerator,
    ReportConfig,
    PerformanceTracker
)
from imp.hmm.models import HMMArtifact


@pytest.fixture
def sample_observations():
    """Create sample observation data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 3
    
    # Generate synthetic data with regime changes
    observations = []
    for i in range(n_samples):
        if i < 100:
            # Regime 1: low volatility
            obs = np.random.randn(n_features) * 0.5
        else:
            # Regime 2: high volatility
            obs = np.random.randn(n_features) * 2.0
        observations.append(obs)
    
    return np.array(observations)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestAutomatedModelSelector:
    """Test automated model selection pipeline."""
    
    def test_initialization(self, sample_observations):
        """Test selector initialization."""
        selector = AutomatedModelSelector(sample_observations)
        
        assert selector.observations.shape == sample_observations.shape
        assert len(selector.selection_criteria) > 0
        assert selector.evaluator is not None
    
    def test_custom_criteria(self, sample_observations):
        """Test custom selection criteria."""
        criteria = [
            SelectionCriteria(metric_name='bic', weight=0.5, higher_is_better=False),
            SelectionCriteria(metric_name='log_likelihood', weight=0.5, higher_is_better=True)
        ]
        
        selector = AutomatedModelSelector(sample_observations, selection_criteria=criteria)
        
        assert len(selector.selection_criteria) == 2
        assert selector.selection_criteria[0].weight == 0.5
    
    def test_invalid_criteria_weights(self, sample_observations):
        """Test that invalid criteria weights raise error."""
        criteria = [
            SelectionCriteria(metric_name='bic', weight=0.3, higher_is_better=False),
            SelectionCriteria(metric_name='log_likelihood', weight=0.5, higher_is_better=True)
        ]
        
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            AutomatedModelSelector(sample_observations, selection_criteria=criteria)
    
    def test_grid_search_selection(self, sample_observations):
        """Test model selection with grid search."""
        selector = AutomatedModelSelector(sample_observations)
        
        param_grid = {
            'n_states': [2, 3],
            'library': ['hmmlearn'],
            'covariance_type': ['diag'],
            'random_state': [42]
        }
        
        result = selector.select_best_model(
            optimization_method='grid_search',
            param_grid=param_grid,
            cv_folds=2,
            n_iterations=50,
            verbose=False
        )
        
        assert result.best_config is not None
        assert result.best_artifact is not None
        assert result.best_score > 0
        assert isinstance(result.all_comparisons, pd.DataFrame)
        assert len(result.all_comparisons) > 0
    
    def test_selection_report(self, sample_observations):
        """Test selection report generation."""
        selector = AutomatedModelSelector(sample_observations)
        
        param_grid = {
            'n_states': [2, 3],
            'library': ['hmmlearn'],
            'covariance_type': ['diag'],
            'random_state': [42]
        }
        
        result = selector.select_best_model(
            optimization_method='grid_search',
            param_grid=param_grid,
            cv_folds=2,
            n_iterations=50,
            verbose=False
        )
        
        report = selector.get_selection_report()
        
        assert isinstance(report, str)
        assert 'AUTOMATED MODEL SELECTION REPORT' in report
        assert 'Selection Criteria' in report


class TestEnsembleEvaluator:
    """Test ensemble model evaluation."""
    
    def test_initialization(self):
        """Test ensemble evaluator initialization."""
        evaluator = EnsembleEvaluator(random_state=42)
        
        assert evaluator.random_state == 42
        assert len(evaluator.ensemble_members) == 0
    
    def test_create_ensemble(self, sample_observations):
        """Test ensemble creation."""
        evaluator = EnsembleEvaluator(random_state=42)
        
        configs = [
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'spherical', 'random_state': 43}
        ]
        
        result = evaluator.create_ensemble(
            observations=sample_observations,
            configs=configs,
            n_iterations=50,
            weighting_strategy='uniform',
            validation_split=0.2
        )
        
        assert len(result.members) >= 1  # At least one model should train successfully
        assert result.ensemble_predictions.shape[0] == sample_observations.shape[0]
        assert 'avg_member_log_likelihood' in result.ensemble_performance
        assert 'pairwise_disagreement' in result.diversity_metrics
    
    def test_performance_weighting(self, sample_observations):
        """Test performance-based weighting."""
        evaluator = EnsembleEvaluator(random_state=42)
        
        configs = [
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'spherical', 'random_state': 43}
        ]
        
        result = evaluator.create_ensemble(
            observations=sample_observations,
            configs=configs,
            n_iterations=50,
            weighting_strategy='performance',
            validation_split=0.2
        )
        
        # Check that weights sum to 1
        total_weight = sum(m.weight for m in result.members)
        assert np.isclose(total_weight, 1.0)
    
    def test_ensemble_comparison(self, sample_observations):
        """Test ensemble vs individual comparison."""
        evaluator = EnsembleEvaluator(random_state=42)
        
        configs = [
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'spherical', 'random_state': 43}
        ]
        
        result = evaluator.create_ensemble(
            observations=sample_observations,
            configs=configs,
            n_iterations=50,
            weighting_strategy='uniform',
            validation_split=0.2
        )
        
        comparison_df = evaluator.compare_ensemble_vs_individual(result, sample_observations)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) >= 2  # At least 1 member + 1 ensemble
        assert 'model' in comparison_df.columns
        assert 'type' in comparison_df.columns
    
    def test_ensemble_report(self, sample_observations):
        """Test ensemble report generation."""
        evaluator = EnsembleEvaluator(random_state=42)
        
        configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42}
        ]
        
        result = evaluator.create_ensemble(
            observations=sample_observations,
            configs=configs,
            n_iterations=50,
            weighting_strategy='uniform',
            validation_split=0.2
        )
        
        report = evaluator.get_ensemble_report(result)
        
        assert isinstance(report, str)
        assert 'ENSEMBLE EVALUATION REPORT' in report
        assert 'Ensemble Members' in report


class TestSensitivityAnalyzer:
    """Test hyperparameter sensitivity analysis."""
    
    def test_initialization(self, sample_observations):
        """Test sensitivity analyzer initialization."""
        baseline_config = {
            'n_states': 3,
            'library': 'hmmlearn',
            'covariance_type': 'diag',
            'random_state': 42
        }
        
        analyzer = SensitivityAnalyzer(
            observations=sample_observations,
            baseline_config=baseline_config,
            n_iterations=50
        )
        
        assert analyzer.baseline_config == baseline_config
        assert analyzer.baseline_metric is not None
    
    def test_analyze_parameter(self, sample_observations):
        """Test single parameter sensitivity analysis."""
        baseline_config = {
            'n_states': 3,
            'library': 'hmmlearn',
            'covariance_type': 'diag',
            'random_state': 42
        }
        
        analyzer = SensitivityAnalyzer(
            observations=sample_observations,
            baseline_config=baseline_config,
            n_iterations=50
        )
        
        result = analyzer.analyze_parameter(
            parameter_name='n_states',
            parameter_values=[2, 3, 4],
            verbose=False
        )
        
        assert result.parameter_name == 'n_states'
        assert len(result.parameter_values) == 3
        assert len(result.metric_values) == 3
        assert result.sensitivity_score >= 0
    
    def test_analyze_all_parameters(self, sample_observations):
        """Test multiple parameter sensitivity analysis."""
        baseline_config = {
            'n_states': 3,
            'library': 'hmmlearn',
            'covariance_type': 'diag',
            'random_state': 42
        }
        
        analyzer = SensitivityAnalyzer(
            observations=sample_observations,
            baseline_config=baseline_config,
            n_iterations=50
        )
        
        param_ranges = {
            'n_states': [2, 3, 4],
            'covariance_type': ['diag', 'spherical']
        }
        
        results = analyzer.analyze_all_parameters(param_ranges, verbose=False)
        
        assert len(results) == 2
        assert 'n_states' in results
        assert 'covariance_type' in results
    
    def test_sensitivity_ranking(self, sample_observations):
        """Test parameter sensitivity ranking."""
        baseline_config = {
            'n_states': 3,
            'library': 'hmmlearn',
            'covariance_type': 'diag',
            'random_state': 42
        }
        
        analyzer = SensitivityAnalyzer(
            observations=sample_observations,
            baseline_config=baseline_config,
            n_iterations=50
        )
        
        param_ranges = {
            'n_states': [2, 3, 4],
            'covariance_type': ['diag', 'spherical']
        }
        
        analyzer.analyze_all_parameters(param_ranges, verbose=False)
        ranking_df = analyzer.get_sensitivity_ranking()
        
        assert isinstance(ranking_df, pd.DataFrame)
        assert 'parameter' in ranking_df.columns
        assert 'sensitivity_score' in ranking_df.columns
        assert 'rank' in ranking_df.columns
    
    def test_sensitivity_report(self, sample_observations):
        """Test sensitivity report generation."""
        baseline_config = {
            'n_states': 3,
            'library': 'hmmlearn',
            'covariance_type': 'diag',
            'random_state': 42
        }
        
        analyzer = SensitivityAnalyzer(
            observations=sample_observations,
            baseline_config=baseline_config,
            n_iterations=50
        )
        
        analyzer.analyze_parameter('n_states', [2, 3, 4], verbose=False)
        report = analyzer.get_sensitivity_report()
        
        assert isinstance(report, str)
        assert 'HYPERPARAMETER SENSITIVITY ANALYSIS REPORT' in report
        assert 'Parameter Sensitivity Ranking' in report


class TestReportGenerator:
    """Test automated report generation."""
    
    def test_initialization(self, temp_dir):
        """Test report generator initialization."""
        generator = ReportGenerator(output_dir=temp_dir)
        
        assert generator.output_dir == temp_dir
        assert generator.figures_dir.exists()
    
    def test_report_config(self):
        """Test report configuration."""
        config = ReportConfig(
            include_sensitivity=True,
            include_ensemble=False,
            output_format='markdown'
        )
        
        assert config.include_sensitivity is True
        assert config.include_ensemble is False
        assert config.output_format == 'markdown'
    
    def test_invalid_output_format(self):
        """Test invalid output format raises error."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            ReportConfig(output_format='invalid')


class TestPerformanceTracker:
    """Test model performance tracking."""
    
    def test_initialization(self, temp_dir):
        """Test performance tracker initialization."""
        tracker = PerformanceTracker(tracking_dir=temp_dir)
        
        assert tracker.tracking_dir == temp_dir
        assert tracker.baseline_window == 10
        assert tracker.warning_threshold == 0.05
    
    def test_record_performance(self, temp_dir, sample_observations):
        """Test recording performance snapshot."""
        from imp.hmm.trainer import EnhancedHMMTrainer
        
        tracker = PerformanceTracker(tracking_dir=temp_dir)
        
        # Train a model
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', covariance_type='diag')
        artifact = trainer.train(sample_observations, n_iterations=50)
        
        # Record performance
        snapshot = tracker.record_performance(
            model_id='test_model',
            model_version='v1.0',
            artifact=artifact,
            observations=sample_observations
        )
        
        assert snapshot.model_id == 'test_model'
        assert snapshot.model_version == 'v1.0'
        assert 'log_likelihood' in snapshot.metrics
        assert 'n_samples' in snapshot.data_stats
    
    def test_regression_detection(self, temp_dir, sample_observations):
        """Test performance regression detection."""
        from imp.hmm.trainer import EnhancedHMMTrainer
        
        tracker = PerformanceTracker(
            tracking_dir=temp_dir,
            baseline_window=3,
            warning_threshold=0.05,
            critical_threshold=0.10
        )
        
        # Record multiple snapshots with degrading performance
        for i in range(5):
            trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', covariance_type='diag')
            artifact = trainer.train(sample_observations[:150-i*10], n_iterations=50)
            
            tracker.record_performance(
                model_id='test_model',
                model_version=f'v1.{i}',
                artifact=artifact,
                observations=sample_observations[:150-i*10]
            )
        
        # Check if regressions were detected
        alerts = tracker.get_alert_history(model_id='test_model')
        
        # We expect some alerts due to training on less data
        assert isinstance(alerts, pd.DataFrame)
    
    def test_performance_history(self, temp_dir, sample_observations):
        """Test retrieving performance history."""
        from imp.hmm.trainer import EnhancedHMMTrainer
        
        tracker = PerformanceTracker(tracking_dir=temp_dir)
        
        # Record multiple snapshots
        for i in range(3):
            trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', covariance_type='diag')
            artifact = trainer.train(sample_observations, n_iterations=50)
            
            tracker.record_performance(
                model_id='test_model',
                model_version=f'v1.{i}',
                artifact=artifact,
                observations=sample_observations
            )
        
        # Get history
        history = tracker.get_performance_history('test_model')
        
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 3
        assert 'timestamp' in history.columns
        assert 'model_version' in history.columns
    
    def test_monitoring_report(self, temp_dir, sample_observations):
        """Test monitoring report generation."""
        from imp.hmm.trainer import EnhancedHMMTrainer
        
        tracker = PerformanceTracker(tracking_dir=temp_dir)
        
        # Record a snapshot
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', covariance_type='diag')
        artifact = trainer.train(sample_observations, n_iterations=50)
        
        tracker.record_performance(
            model_id='test_model',
            model_version='v1.0',
            artifact=artifact,
            observations=sample_observations
        )
        
        # Generate report
        report = tracker.generate_monitoring_report('test_model')
        
        assert isinstance(report, str)
        assert 'PERFORMANCE MONITORING REPORT' in report
        assert 'test_model' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
