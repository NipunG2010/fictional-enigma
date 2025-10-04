"""
Automated model selection pipeline with multiple evaluation criteria.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import logging
from dataclasses import dataclass, field

from ..hmm.trainer import EnhancedHMMTrainer
from ..hmm.models import HMMArtifact
from ..evaluation.evaluator import HMMEvaluator, ModelComparison
from .optimization import GridSearchOptimizer, BayesianOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class SelectionCriteria:
    """Criteria for model selection."""
    metric_name: str
    weight: float
    higher_is_better: bool = True
    threshold: Optional[float] = None
    
    def __post_init__(self):
        if self.weight < 0 or self.weight > 1:
            raise ValueError("Weight must be between 0 and 1")


@dataclass
class ModelSelectionResult:
    """Result from automated model selection."""
    best_config: Dict[str, Any]
    best_artifact: HMMArtifact
    best_score: float
    all_comparisons: pd.DataFrame
    selection_criteria: List[SelectionCriteria]
    optimization_method: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'best_config': self.best_config,
            'best_artifact': self.best_artifact.model_dump(),
            'best_score': self.best_score,
            'all_comparisons': self.all_comparisons.to_dict('records'),
            'selection_criteria': [
                {
                    'metric_name': c.metric_name,
                    'weight': c.weight,
                    'higher_is_better': c.higher_is_better,
                    'threshold': c.threshold
                }
                for c in self.selection_criteria
            ],
            'optimization_method': self.optimization_method,
            'timestamp': self.timestamp
        }
    
    def save(self, filepath: Path):
        """Save results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Model selection results saved to {filepath}")


class AutomatedModelSelector:
    """
    Automated model selection pipeline with multiple evaluation criteria.
    
    This class orchestrates the entire model selection process including:
    - Hyperparameter optimization (grid search or Bayesian)
    - Multi-criteria evaluation
    - Statistical significance testing
    - Model ranking and selection
    """
    
    def __init__(self,
                 observations: np.ndarray,
                 selection_criteria: Optional[List[SelectionCriteria]] = None,
                 random_state: int = 42):
        """
        Initialize automated model selector.
        
        Args:
            observations: Training data
            selection_criteria: List of criteria for model selection
            random_state: Random seed for reproducibility
        """
        self.observations = observations
        self.random_state = random_state
        
        # Default selection criteria if not provided
        if selection_criteria is None:
            self.selection_criteria = self._create_default_criteria()
        else:
            self.selection_criteria = selection_criteria
        
        # Validate criteria weights sum to 1
        total_weight = sum(c.weight for c in self.selection_criteria)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Selection criteria weights must sum to 1.0, got {total_weight}")
        
        self.evaluator = HMMEvaluator(random_state=random_state)
        self.optimization_results: Optional[OptimizationResult] = None
        self.evaluation_results: Optional[pd.DataFrame] = None
    
    def _create_default_criteria(self) -> List[SelectionCriteria]:
        """Create default selection criteria."""
        return [
            SelectionCriteria(metric_name='bic', weight=0.4, higher_is_better=False),
            SelectionCriteria(metric_name='log_likelihood', weight=0.3, higher_is_better=True),
            SelectionCriteria(metric_name='stability_score', weight=0.3, higher_is_better=True)
        ]
    
    def select_best_model(self,
                         optimization_method: str = 'grid_search',
                         param_grid: Optional[Dict[str, List[Any]]] = None,
                         param_space: Optional[Dict[str, Dict[str, Any]]] = None,
                         n_calls: int = 20,
                         cv_folds: int = 5,
                         n_iterations: int = 100,
                         verbose: bool = True) -> ModelSelectionResult:
        """
        Run automated model selection pipeline.
        
        Args:
            optimization_method: 'grid_search' or 'bayesian'
            param_grid: Parameter grid for grid search
            param_space: Parameter space for Bayesian optimization
            n_calls: Number of calls for Bayesian optimization
            cv_folds: Number of cross-validation folds
            n_iterations: Number of EM iterations per model
            verbose: Whether to print progress
            
        Returns:
            ModelSelectionResult with best model and evaluation details
        """
        logger.info(f"Starting automated model selection with {optimization_method}")
        
        # Step 1: Hyperparameter optimization
        if optimization_method == 'grid_search':
            if param_grid is None:
                param_grid = self._create_default_param_grid()
            
            optimizer = GridSearchOptimizer(
                observations=self.observations,
                param_grid=param_grid,
                scoring_metric='log_likelihood',
                higher_is_better=True,
                n_iterations=n_iterations,
                verbose=verbose
            )
            self.optimization_results = optimizer.fit()
        
        elif optimization_method == 'bayesian':
            if param_space is None:
                param_space = self._create_default_param_space()
            
            optimizer = BayesianOptimizer(
                observations=self.observations,
                param_space=param_space,
                scoring_metric='log_likelihood',
                higher_is_better=True,
                n_calls=n_calls,
                n_iterations=n_iterations,
                verbose=verbose
            )
            self.optimization_results = optimizer.fit()
        
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Step 2: Extract top configurations for detailed evaluation
        top_configs = self._extract_top_configs(n_top=min(10, len(self.optimization_results.all_results)))
        
        logger.info(f"Evaluating top {len(top_configs)} configurations with cross-validation")
        
        # Step 3: Detailed evaluation with cross-validation
        self.evaluation_results = self.evaluator.compare_models(
            observations=self.observations,
            trainer_configs=top_configs,
            n_iterations=n_iterations,
            perform_cv=True,
            cv_folds=cv_folds,
            analyze_stability=True
        )
        
        # Step 4: Multi-criteria selection
        best_config_name = self._select_by_criteria()
        
        # Step 5: Train final model with best configuration
        best_config = self._get_config_from_name(best_config_name, top_configs)
        
        logger.info(f"Training final model with best configuration: {best_config}")
        
        trainer = EnhancedHMMTrainer(**best_config)
        best_artifact = trainer.train(self.observations, n_iterations)
        
        # Calculate final score
        best_score = self._calculate_composite_score(best_config_name)
        
        # Create result
        result = ModelSelectionResult(
            best_config=best_config,
            best_artifact=best_artifact,
            best_score=best_score,
            all_comparisons=self.evaluation_results,
            selection_criteria=self.selection_criteria,
            optimization_method=optimization_method,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Model selection completed. Best score: {best_score:.4f}")
        
        return result
    
    def _extract_top_configs(self, n_top: int) -> List[Dict[str, Any]]:
        """Extract top N configurations from optimization results."""
        # Filter successful results
        successful_results = [
            r for r in self.optimization_results.all_results
            if r.get('score') is not None
        ]
        
        # Sort by score
        sorted_results = sorted(
            successful_results,
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Extract top configs
        top_configs = [r['params'] for r in sorted_results[:n_top]]
        
        return top_configs
    
    def _select_by_criteria(self) -> str:
        """Select best model based on multiple criteria."""
        scores = {}
        
        for _, row in self.evaluation_results.iterrows():
            config_name = row['config']
            composite_score = 0.0
            
            for criterion in self.selection_criteria:
                # Get metric value
                metric_value = self._get_metric_value(row, criterion.metric_name)
                
                if metric_value is None:
                    logger.warning(f"Metric {criterion.metric_name} not found for {config_name}")
                    continue
                
                # Check threshold if specified
                if criterion.threshold is not None:
                    if criterion.higher_is_better:
                        if metric_value < criterion.threshold:
                            logger.info(f"{config_name} failed threshold for {criterion.metric_name}")
                            composite_score = float('-inf')
                            break
                    else:
                        if metric_value > criterion.threshold:
                            logger.info(f"{config_name} failed threshold for {criterion.metric_name}")
                            composite_score = float('-inf')
                            break
                
                # Normalize and weight
                normalized_value = self._normalize_metric(
                    metric_value,
                    criterion.metric_name,
                    criterion.higher_is_better
                )
                
                composite_score += criterion.weight * normalized_value
            
            scores[config_name] = composite_score
        
        # Select best
        best_config = max(scores, key=scores.get)
        logger.info(f"Best configuration by multi-criteria: {best_config} (score: {scores[best_config]:.4f})")
        
        return best_config
    
    def _get_metric_value(self, row: pd.Series, metric_name: str) -> Optional[float]:
        """Extract metric value from evaluation row."""
        # Handle special metrics
        if metric_name == 'stability_score':
            # Calculate average stability across states
            stability_cols = [col for col in row.index if col.startswith('persistence_state_')]
            if stability_cols:
                return row[stability_cols].mean()
            return None
        
        # Standard metrics
        if metric_name in row.index:
            return row[metric_name]
        
        # CV metrics
        cv_metric = f'cv_{metric_name}_mean'
        if cv_metric in row.index:
            return row[cv_metric]
        
        return None
    
    def _normalize_metric(self, value: float, metric_name: str, higher_is_better: bool) -> float:
        """Normalize metric value for comparison."""
        # Get all values for this metric
        all_values = []
        for _, row in self.evaluation_results.iterrows():
            metric_value = self._get_metric_value(row, metric_name)
            if metric_value is not None:
                all_values.append(metric_value)
        
        if not all_values:
            return 0.0
        
        # Min-max normalization
        min_val = min(all_values)
        max_val = max(all_values)
        
        if max_val == min_val:
            return 1.0
        
        normalized = (value - min_val) / (max_val - min_val)
        
        # Invert if lower is better
        if not higher_is_better:
            normalized = 1.0 - normalized
        
        return normalized
    
    def _calculate_composite_score(self, config_name: str) -> float:
        """Calculate composite score for a configuration."""
        row = self.evaluation_results[self.evaluation_results['config'] == config_name].iloc[0]
        
        composite_score = 0.0
        for criterion in self.selection_criteria:
            metric_value = self._get_metric_value(row, criterion.metric_name)
            if metric_value is not None:
                normalized_value = self._normalize_metric(
                    metric_value,
                    criterion.metric_name,
                    criterion.higher_is_better
                )
                composite_score += criterion.weight * normalized_value
        
        return composite_score
    
    def _get_config_from_name(self, config_name: str, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get configuration dictionary from config name."""
        # Parse config name (format: library_Nstates_covtype)
        parts = config_name.split('_')
        library = parts[0]
        n_states = int(parts[1].replace('states', ''))
        cov_type = parts[2]
        
        # Find matching config
        for config in configs:
            if (config.get('library') == library and
                config.get('n_states') == n_states and
                config.get('covariance_type') == cov_type):
                return config
        
        # If not found, create new config
        return {
            'library': library,
            'n_states': n_states,
            'covariance_type': cov_type,
            'random_state': self.random_state
        }
    
    def _create_default_param_grid(self) -> Dict[str, List[Any]]:
        """Create default parameter grid."""
        return {
            'n_states': [2, 3, 4, 5],
            'library': ['hmmlearn'],
            'covariance_type': ['full', 'diag'],
            'random_state': [self.random_state]
        }
    
    def _create_default_param_space(self) -> Dict[str, Dict[str, Any]]:
        """Create default parameter space for Bayesian optimization."""
        return {
            'n_states': {'type': 'integer', 'low': 2, 'high': 8},
            'library': {'type': 'categorical', 'categories': ['hmmlearn']},
            'covariance_type': {'type': 'categorical', 'categories': ['full', 'diag', 'spherical']}
        }
    
    def get_selection_report(self) -> str:
        """Generate human-readable selection report."""
        if self.evaluation_results is None:
            return "No evaluation results available"
        
        report = []
        report.append("="*70)
        report.append("AUTOMATED MODEL SELECTION REPORT")
        report.append("="*70)
        report.append("")
        
        # Selection criteria
        report.append("Selection Criteria:")
        for criterion in self.selection_criteria:
            direction = "higher is better" if criterion.higher_is_better else "lower is better"
            report.append(f"  - {criterion.metric_name}: weight={criterion.weight:.2f} ({direction})")
            if criterion.threshold is not None:
                report.append(f"    threshold: {criterion.threshold}")
        report.append("")
        
        # Top 5 models
        report.append("Top 5 Models:")
        top_5 = self.evaluation_results.nsmallest(5, 'rank')
        for _, row in top_5.iterrows():
            report.append(f"\n  Rank {row['rank']}: {row['config']}")
            report.append(f"    Log-Likelihood: {row['log_likelihood']:.4f}")
            report.append(f"    BIC: {row['bic']:.4f}")
            report.append(f"    AIC: {row['aic']:.4f}")
            
            # CV scores if available
            if 'cv_log_likelihood_mean' in row.index:
                report.append(f"    CV Log-Likelihood: {row['cv_log_likelihood_mean']:.4f} ± {row['cv_log_likelihood_std']:.4f}")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
