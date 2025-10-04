"""
Comprehensive HMM model evaluation and comparison framework.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import logging

from ..hmm.trainer import EnhancedHMMTrainer, BaseHMMTrainer
from ..hmm.models import HMMArtifact

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    log_likelihood: float
    aic: float
    bic: float
    perplexity: float
    n_parameters: int
    n_samples: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'log_likelihood': self.log_likelihood,
            'aic': self.aic,
            'bic': self.bic,
            'perplexity': self.perplexity,
            'n_parameters': self.n_parameters,
            'n_samples': self.n_samples
        }


@dataclass
class RegimeStabilityMetrics:
    """Container for regime stability analysis results."""
    state_durations: Dict[int, List[int]]
    mean_durations: Dict[int, float]
    median_durations: Dict[int, float]
    max_durations: Dict[int, float]
    stable_periods: Dict[int, int]
    total_periods: Dict[int, int]
    transition_frequencies: Dict[Tuple[int, int], int]
    state_persistence: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean_durations': self.mean_durations,
            'median_durations': self.median_durations,
            'max_durations': self.max_durations,
            'stable_periods': self.stable_periods,
            'total_periods': self.total_periods,
            'transition_frequencies': {f"{k[0]}->{k[1]}": v for k, v in self.transition_frequencies.items()},
            'state_persistence': self.state_persistence
        }


@dataclass
class ModelComparison:
    """Container for model comparison results."""
    config_name: str
    metrics: EvaluationMetrics
    cv_scores: Optional[Dict[str, List[float]]] = None
    stability_metrics: Optional[RegimeStabilityMetrics] = None
    rank: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'config_name': self.config_name,
            'metrics': self.metrics.to_dict(),
            'rank': self.rank
        }
        if self.cv_scores:
            result['cv_scores'] = self.cv_scores
        if self.stability_metrics:
            result['stability_metrics'] = self.stability_metrics.to_dict()
        return result


class HMMEvaluator:
    """Comprehensive HMM model evaluation and comparison framework."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize HMM evaluator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.evaluation_results: Dict[str, ModelComparison] = {}
        
    def evaluate_model(self, 
                      trainer: BaseHMMTrainer,
                      observations: np.ndarray) -> EvaluationMetrics:
        """
        Evaluate a single HMM model.
        
        Args:
            trainer: Trained HMM trainer instance
            observations: Observation data for evaluation
            
        Returns:
            EvaluationMetrics containing evaluation results
        """
        try:
            metrics_dict = trainer.evaluate(observations)
            
            # Extract n_parameters from trainer if available
            n_params = trainer._calculate_n_params(observations.shape[1])
            
            return EvaluationMetrics(
                log_likelihood=metrics_dict['log_likelihood'],
                aic=metrics_dict['aic'],
                bic=metrics_dict['bic'],
                perplexity=metrics_dict['perplexity'],
                n_parameters=n_params,
                n_samples=observations.shape[0]
            )
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def cross_validate(self,
                      observations: np.ndarray,
                      trainer_config: Dict[str, Any],
                      cv_folds: int = 5,
                      n_iterations: int = 100) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            observations: Observation data
            trainer_config: Configuration for EnhancedHMMTrainer
            cv_folds: Number of cross-validation folds
            n_iterations: Number of training iterations
            
        Returns:
            Dictionary of cross-validation scores for each metric
        """
        if cv_folds < 2:
            raise ValueError("Number of CV folds must be at least 2")
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_results = {
            "log_likelihood": [],
            "aic": [],
            "bic": [],
            "perplexity": []
        }
        
        logger.info(f"Performing {cv_folds}-fold time series cross-validation")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(observations)):
            logger.info(f"Processing fold {fold + 1}/{cv_folds}")
            
            train_data = observations[train_idx]
            val_data = observations[val_idx]
            
            try:
                # Create fresh trainer for each fold
                trainer = EnhancedHMMTrainer(**trainer_config)
                
                # Train model
                trainer.train(train_data, n_iterations)
                
                # Evaluate
                metrics = self.evaluate_model(trainer.trainer, val_data)
                
                # Store results
                cv_results["log_likelihood"].append(metrics.log_likelihood)
                cv_results["aic"].append(metrics.aic)
                cv_results["bic"].append(metrics.bic)
                cv_results["perplexity"].append(metrics.perplexity)
                        
            except Exception as e:
                logger.warning(f"Fold {fold + 1} failed: {str(e)}")
                # Add NaN for failed folds
                for metric_name in cv_results:
                    cv_results[metric_name].append(np.nan)
        
        # Calculate summary statistics
        summary = {}
        for metric_name, values in cv_results.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                summary[f"{metric_name}_mean"] = np.mean(valid_values)
                summary[f"{metric_name}_std"] = np.std(valid_values)
                summary[f"{metric_name}_values"] = valid_values
        
        logger.info(f"Cross-validation completed")
        return summary
    
    def regime_stability_analysis(self,
                                 state_probs: np.ndarray,
                                 min_duration: int = 5) -> RegimeStabilityMetrics:
        """
        Analyze regime stability and persistence.
        
        Args:
            state_probs: State probability matrix (n_samples, n_states)
            min_duration: Minimum duration to consider a regime stable
            
        Returns:
            RegimeStabilityMetrics containing stability analysis results
        """
        # Decode most likely state sequence
        most_likely_states = np.argmax(state_probs, axis=1)
        n_states = state_probs.shape[1]
        
        # Calculate state durations
        state_durations: Dict[int, List[int]] = {i: [] for i in range(n_states)}
        current_state = most_likely_states[0]
        current_duration = 1
        
        # Track transitions
        transition_frequencies: Dict[Tuple[int, int], int] = {}
        
        for i in range(1, len(most_likely_states)):
            if most_likely_states[i] == current_state:
                current_duration += 1
            else:
                # Record duration
                state_durations[current_state].append(current_duration)
                
                # Record transition
                transition_key = (current_state, most_likely_states[i])
                transition_frequencies[transition_key] = transition_frequencies.get(transition_key, 0) + 1
                
                # Move to next state
                current_state = most_likely_states[i]
                current_duration = 1
        
        # Add final duration
        state_durations[current_state].append(current_duration)
        
        # Calculate statistics
        mean_durations = {}
        median_durations = {}
        max_durations = {}
        stable_periods = {}
        total_periods = {}
        state_persistence = {}
        
        for state in range(n_states):
            durations = state_durations[state]
            if durations:
                mean_durations[state] = float(np.mean(durations))
                median_durations[state] = float(np.median(durations))
                max_durations[state] = int(np.max(durations))
                stable_periods[state] = sum(1 for d in durations if d >= min_duration)
                total_periods[state] = len(durations)
                
                # Calculate persistence (probability of staying in same state)
                total_time_in_state = sum(durations)
                state_persistence[state] = total_time_in_state / len(most_likely_states)
            else:
                mean_durations[state] = 0.0
                median_durations[state] = 0.0
                max_durations[state] = 0
                stable_periods[state] = 0
                total_periods[state] = 0
                state_persistence[state] = 0.0
        
        return RegimeStabilityMetrics(
            state_durations=state_durations,
            mean_durations=mean_durations,
            median_durations=median_durations,
            max_durations=max_durations,
            stable_periods=stable_periods,
            total_periods=total_periods,
            transition_frequencies=transition_frequencies,
            state_persistence=state_persistence
        )
    
    def compare_models(self,
                      observations: np.ndarray,
                      trainer_configs: List[Dict[str, Any]],
                      n_iterations: int = 100,
                      perform_cv: bool = True,
                      cv_folds: int = 5,
                      analyze_stability: bool = True) -> pd.DataFrame:
        """
        Compare multiple HMM configurations.
        
        Args:
            observations: Observation data
            trainer_configs: List of trainer configurations to compare
            n_iterations: Number of training iterations
            perform_cv: Whether to perform cross-validation
            cv_folds: Number of CV folds if perform_cv is True
            analyze_stability: Whether to analyze regime stability
            
        Returns:
            DataFrame with comparison results
        """
        comparisons = []
        
        for config in trainer_configs:
            config_name = self._get_config_name(config)
            logger.info(f"Evaluating configuration: {config_name}")
            
            try:
                # Train model
                trainer = EnhancedHMMTrainer(**config)
                artifact = trainer.train(observations, n_iterations)
                
                # Evaluate on full dataset
                metrics = self.evaluate_model(trainer.trainer, observations)
                
                # Cross-validation
                cv_scores = None
                if perform_cv:
                    try:
                        cv_scores = self.cross_validate(
                            observations, config, cv_folds, n_iterations
                        )
                    except Exception as e:
                        logger.warning(f"Cross-validation failed for {config_name}: {str(e)}")
                
                # Stability analysis
                stability_metrics = None
                if analyze_stability:
                    try:
                        state_probs = trainer.trainer.predict_state_probabilities(observations)
                        stability_metrics = self.regime_stability_analysis(state_probs)
                    except Exception as e:
                        logger.warning(f"Stability analysis failed for {config_name}: {str(e)}")
                
                # Create comparison object
                comparison = ModelComparison(
                    config_name=config_name,
                    metrics=metrics,
                    cv_scores=cv_scores,
                    stability_metrics=stability_metrics
                )
                
                comparisons.append(comparison)
                self.evaluation_results[config_name] = comparison
                
            except Exception as e:
                logger.error(f"Configuration {config_name} failed: {str(e)}")
                continue
        
        # Rank models
        comparisons = self._rank_models(comparisons)
        
        # Convert to DataFrame
        return self._comparisons_to_dataframe(comparisons)
    
    def statistical_significance_test(self,
                                     config1: str,
                                     config2: str,
                                     metric: str = 'log_likelihood') -> Dict[str, Any]:
        """
        Test statistical significance between two model configurations.
        
        Args:
            config1: Name of first configuration
            config2: Name of second configuration
            metric: Metric to compare (must have CV scores)
            
        Returns:
            Dictionary with test results
        """
        if config1 not in self.evaluation_results:
            raise ValueError(f"Configuration {config1} not found in evaluation results")
        if config2 not in self.evaluation_results:
            raise ValueError(f"Configuration {config2} not found in evaluation results")
        
        comp1 = self.evaluation_results[config1]
        comp2 = self.evaluation_results[config2]
        
        if not comp1.cv_scores or not comp2.cv_scores:
            raise ValueError("Both configurations must have cross-validation scores")
        
        metric_key = f"{metric}_values"
        if metric_key not in comp1.cv_scores or metric_key not in comp2.cv_scores:
            raise ValueError(f"Metric {metric} not found in CV scores")
        
        scores1 = comp1.cv_scores[metric_key]
        scores2 = comp2.cv_scores[metric_key]
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(scores1) - np.mean(scores2)
        pooled_std = np.sqrt((np.std(scores1)**2 + np.std(scores2)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'config1': config1,
            'config2': config2,
            'metric': metric,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'mean_diff': float(mean_diff),
            'cohens_d': float(cohens_d),
            'config1_mean': float(np.mean(scores1)),
            'config2_mean': float(np.mean(scores2)),
            'config1_std': float(np.std(scores1)),
            'config2_std': float(np.std(scores2))
        }
    
    def select_best_model(self,
                         criteria: List[str] = ['bic', 'log_likelihood'],
                         weights: Optional[List[float]] = None) -> str:
        """
        Select best model based on multiple criteria.
        
        Args:
            criteria: List of metrics to consider (lower is better for AIC/BIC)
            weights: Optional weights for each criterion (must sum to 1)
            
        Returns:
            Name of best configuration
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        if weights is None:
            weights = [1.0 / len(criteria)] * len(criteria)
        
        if len(weights) != len(criteria):
            raise ValueError("Number of weights must match number of criteria")
        
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")
        
        # Calculate weighted scores
        scores = {}
        
        for config_name, comparison in self.evaluation_results.items():
            weighted_score = 0.0
            
            for criterion, weight in zip(criteria, weights):
                # Get metric value
                if hasattr(comparison.metrics, criterion):
                    value = getattr(comparison.metrics, criterion)
                else:
                    logger.warning(f"Criterion {criterion} not found for {config_name}")
                    continue
                
                # Normalize (lower is better for AIC/BIC, higher is better for log_likelihood)
                if criterion in ['aic', 'bic', 'perplexity']:
                    # Lower is better - invert
                    normalized = -value
                else:
                    # Higher is better
                    normalized = value
                
                weighted_score += weight * normalized
            
            scores[config_name] = weighted_score
        
        # Select best
        best_config = max(scores, key=scores.get)
        logger.info(f"Best model: {best_config} (score: {scores[best_config]:.4f})")
        
        return best_config
    
    def _get_config_name(self, config: Dict[str, Any]) -> str:
        """Generate configuration name from config dict."""
        library = config.get('library', 'hmmlearn')
        n_states = config.get('n_states', 3)
        cov_type = config.get('covariance_type', 'full')
        return f"{library}_{n_states}states_{cov_type}"
    
    def _rank_models(self, comparisons: List[ModelComparison]) -> List[ModelComparison]:
        """Rank models by BIC (lower is better)."""
        # Sort by BIC
        sorted_comparisons = sorted(comparisons, key=lambda x: x.metrics.bic)
        
        # Assign ranks
        for rank, comparison in enumerate(sorted_comparisons, 1):
            comparison.rank = rank
        
        return sorted_comparisons
    
    def _comparisons_to_dataframe(self, comparisons: List[ModelComparison]) -> pd.DataFrame:
        """Convert comparison results to DataFrame."""
        rows = []
        
        for comp in comparisons:
            row = {
                'config': comp.config_name,
                'rank': comp.rank,
                'log_likelihood': comp.metrics.log_likelihood,
                'aic': comp.metrics.aic,
                'bic': comp.metrics.bic,
                'perplexity': comp.metrics.perplexity,
                'n_parameters': comp.metrics.n_parameters
            }
            
            # Add CV scores if available
            if comp.cv_scores:
                for metric, value in comp.cv_scores.items():
                    if not metric.endswith('_values'):
                        row[f'cv_{metric}'] = value
            
            # Add stability metrics if available
            if comp.stability_metrics:
                for state, duration in comp.stability_metrics.mean_durations.items():
                    row[f'mean_duration_state_{state}'] = duration
                for state, persistence in comp.stability_metrics.state_persistence.items():
                    row[f'persistence_state_{state}'] = persistence
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.evaluation_results:
            return {"message": "No evaluation results available"}
        
        summary = {
            'n_configurations': len(self.evaluation_results),
            'configurations': list(self.evaluation_results.keys()),
            'best_by_bic': min(self.evaluation_results.items(), 
                              key=lambda x: x[1].metrics.bic)[0],
            'best_by_aic': min(self.evaluation_results.items(), 
                              key=lambda x: x[1].metrics.aic)[0],
            'best_by_likelihood': max(self.evaluation_results.items(), 
                                     key=lambda x: x[1].metrics.log_likelihood)[0]
        }
        
        return summary
