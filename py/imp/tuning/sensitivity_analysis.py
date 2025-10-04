"""
Hyperparameter sensitivity analysis tools for understanding parameter impact.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from scipy import stats

from ..hmm.trainer import EnhancedHMMTrainer

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis."""
    parameter_name: str
    parameter_values: List[Any]
    metric_values: List[float]
    metric_name: str
    baseline_value: Any
    baseline_metric: float
    sensitivity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'parameter_name': self.parameter_name,
            'parameter_values': self.parameter_values,
            'metric_values': self.metric_values,
            'metric_name': self.metric_name,
            'baseline_value': self.baseline_value,
            'baseline_metric': self.baseline_metric,
            'sensitivity_score': self.sensitivity_score
        }


@dataclass
class InteractionResult:
    """Result from parameter interaction analysis."""
    param1_name: str
    param2_name: str
    param1_values: List[Any]
    param2_values: List[Any]
    metric_matrix: np.ndarray
    metric_name: str
    interaction_strength: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'param1_name': self.param1_name,
            'param2_name': self.param2_name,
            'param1_values': self.param1_values,
            'param2_values': self.param2_values,
            'metric_matrix': self.metric_matrix.tolist(),
            'metric_name': self.metric_name,
            'interaction_strength': self.interaction_strength
        }


class SensitivityAnalyzer:
    """
    Hyperparameter sensitivity analysis for HMM models.
    
    Analyzes how changes in hyperparameters affect model performance
    and identifies critical parameters.
    """
    
    def __init__(self,
                 observations: np.ndarray,
                 baseline_config: Dict[str, Any],
                 metric_name: str = 'log_likelihood',
                 higher_is_better: bool = True,
                 n_iterations: int = 100,
                 random_state: int = 42):
        """
        Initialize sensitivity analyzer.
        
        Args:
            observations: Training data
            baseline_config: Baseline configuration for comparison
            metric_name: Metric to analyze
            higher_is_better: Whether higher metric values are better
            n_iterations: Number of training iterations
            random_state: Random seed
        """
        self.observations = observations
        self.baseline_config = baseline_config
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        self.sensitivity_results: Dict[str, SensitivityResult] = {}
        self.interaction_results: Dict[Tuple[str, str], InteractionResult] = {}
        
        # Calculate baseline metric
        self.baseline_metric = self._evaluate_config(baseline_config)
    
    def analyze_parameter(self,
                         parameter_name: str,
                         parameter_values: List[Any],
                         verbose: bool = True) -> SensitivityResult:
        """
        Analyze sensitivity to a single parameter.
        
        Args:
            parameter_name: Name of parameter to analyze
            parameter_values: List of values to test
            verbose: Whether to print progress
            
        Returns:
            SensitivityResult with analysis results
        """
        if verbose:
            logger.info(f"Analyzing sensitivity to {parameter_name}")
        
        metric_values = []
        
        for i, value in enumerate(parameter_values):
            if verbose:
                logger.info(f"  Testing {parameter_name}={value} ({i+1}/{len(parameter_values)})")
            
            # Create config with modified parameter
            config = self.baseline_config.copy()
            config[parameter_name] = value
            
            try:
                metric = self._evaluate_config(config)
                metric_values.append(metric)
                
                if verbose:
                    logger.info(f"    {self.metric_name}: {metric:.4f}")
            
            except Exception as e:
                logger.warning(f"    Failed: {str(e)}")
                metric_values.append(np.nan)
        
        # Calculate sensitivity score (coefficient of variation)
        valid_metrics = [m for m in metric_values if not np.isnan(m)]
        if valid_metrics:
            sensitivity_score = np.std(valid_metrics) / (np.abs(np.mean(valid_metrics)) + 1e-10)
        else:
            sensitivity_score = 0.0
        
        result = SensitivityResult(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            metric_values=metric_values,
            metric_name=self.metric_name,
            baseline_value=self.baseline_config.get(parameter_name),
            baseline_metric=self.baseline_metric,
            sensitivity_score=sensitivity_score
        )
        
        self.sensitivity_results[parameter_name] = result
        
        if verbose:
            logger.info(f"  Sensitivity score: {sensitivity_score:.4f}")
        
        return result
    
    def analyze_all_parameters(self,
                              param_ranges: Dict[str, List[Any]],
                              verbose: bool = True) -> Dict[str, SensitivityResult]:
        """
        Analyze sensitivity to all specified parameters.
        
        Args:
            param_ranges: Dictionary mapping parameter names to value lists
            verbose: Whether to print progress
            
        Returns:
            Dictionary of SensitivityResults
        """
        results = {}
        
        for param_name, param_values in param_ranges.items():
            result = self.analyze_parameter(param_name, param_values, verbose)
            results[param_name] = result
        
        return results
    
    def analyze_interaction(self,
                           param1_name: str,
                           param1_values: List[Any],
                           param2_name: str,
                           param2_values: List[Any],
                           verbose: bool = True) -> InteractionResult:
        """
        Analyze interaction between two parameters.
        
        Args:
            param1_name: Name of first parameter
            param1_values: Values for first parameter
            param2_name: Name of second parameter
            param2_values: Values for second parameter
            verbose: Whether to print progress
            
        Returns:
            InteractionResult with analysis results
        """
        if verbose:
            logger.info(f"Analyzing interaction between {param1_name} and {param2_name}")
        
        metric_matrix = np.zeros((len(param1_values), len(param2_values)))
        
        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
                if verbose:
                    logger.info(f"  Testing {param1_name}={val1}, {param2_name}={val2}")
                
                # Create config with both parameters modified
                config = self.baseline_config.copy()
                config[param1_name] = val1
                config[param2_name] = val2
                
                try:
                    metric = self._evaluate_config(config)
                    metric_matrix[i, j] = metric
                    
                    if verbose:
                        logger.info(f"    {self.metric_name}: {metric:.4f}")
                
                except Exception as e:
                    logger.warning(f"    Failed: {str(e)}")
                    metric_matrix[i, j] = np.nan
        
        # Calculate interaction strength using ANOVA-like approach
        # Measure how much the effect of param1 depends on param2
        interaction_strength = self._calculate_interaction_strength(metric_matrix)
        
        result = InteractionResult(
            param1_name=param1_name,
            param2_name=param2_name,
            param1_values=param1_values,
            param2_values=param2_values,
            metric_matrix=metric_matrix,
            metric_name=self.metric_name,
            interaction_strength=interaction_strength
        )
        
        self.interaction_results[(param1_name, param2_name)] = result
        
        if verbose:
            logger.info(f"  Interaction strength: {interaction_strength:.4f}")
        
        return result
    
    def _evaluate_config(self, config: Dict[str, Any]) -> float:
        """Evaluate a configuration and return metric value."""
        trainer = EnhancedHMMTrainer(**config)
        artifact = trainer.train(self.observations, self.n_iterations)
        
        # Get metric from artifact or evaluate
        if self.metric_name in artifact.metadata:
            return artifact.metadata[self.metric_name]
        else:
            metrics = trainer.trainer.evaluate(self.observations)
            return metrics[self.metric_name]
    
    def _calculate_interaction_strength(self, metric_matrix: np.ndarray) -> float:
        """Calculate interaction strength from metric matrix."""
        # Remove NaN values
        valid_mask = ~np.isnan(metric_matrix)
        if not valid_mask.any():
            return 0.0
        
        # Calculate row and column effects
        row_means = np.nanmean(metric_matrix, axis=1)
        col_means = np.nanmean(metric_matrix, axis=0)
        grand_mean = np.nanmean(metric_matrix)
        
        # Calculate interaction as deviation from additive model
        interaction_sum = 0.0
        count = 0
        
        for i in range(metric_matrix.shape[0]):
            for j in range(metric_matrix.shape[1]):
                if valid_mask[i, j]:
                    expected = row_means[i] + col_means[j] - grand_mean
                    actual = metric_matrix[i, j]
                    interaction_sum += (actual - expected) ** 2
                    count += 1
        
        if count > 0:
            interaction_variance = interaction_sum / count
            total_variance = np.nanvar(metric_matrix)
            
            if total_variance > 0:
                return interaction_variance / total_variance
        
        return 0.0
    
    def plot_sensitivity(self,
                        parameter_name: str,
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot sensitivity analysis results for a parameter.
        
        Args:
            parameter_name: Name of parameter to plot
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if parameter_name not in self.sensitivity_results:
            raise ValueError(f"No sensitivity results for parameter: {parameter_name}")
        
        result = self.sensitivity_results[parameter_name]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot metric values
        ax.plot(result.parameter_values, result.metric_values, 'o-', linewidth=2, markersize=8)
        
        # Mark baseline
        if result.baseline_value in result.parameter_values:
            baseline_idx = result.parameter_values.index(result.baseline_value)
            ax.plot(result.baseline_value, result.metric_values[baseline_idx],
                   'r*', markersize=15, label='Baseline')
        
        ax.set_xlabel(parameter_name, fontsize=12)
        ax.set_ylabel(result.metric_name, fontsize=12)
        ax.set_title(f'Sensitivity Analysis: {parameter_name}\n'
                    f'Sensitivity Score: {result.sensitivity_score:.4f}',
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sensitivity plot saved to {save_path}")
        
        return fig
    
    def plot_interaction(self,
                        param1_name: str,
                        param2_name: str,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot parameter interaction heatmap.
        
        Args:
            param1_name: Name of first parameter
            param2_name: Name of second parameter
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        key = (param1_name, param2_name)
        if key not in self.interaction_results:
            raise ValueError(f"No interaction results for parameters: {param1_name}, {param2_name}")
        
        result = self.interaction_results[key]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(result.metric_matrix,
                   xticklabels=result.param2_values,
                   yticklabels=result.param1_values,
                   annot=True,
                   fmt='.3f',
                   cmap='viridis',
                   ax=ax,
                   cbar_kws={'label': result.metric_name})
        
        ax.set_xlabel(param2_name, fontsize=12)
        ax.set_ylabel(param1_name, fontsize=12)
        ax.set_title(f'Parameter Interaction: {param1_name} vs {param2_name}\n'
                    f'Interaction Strength: {result.interaction_strength:.4f}',
                    fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Interaction plot saved to {save_path}")
        
        return fig
    
    def plot_all_sensitivities(self,
                              figsize: Tuple[int, int] = (14, 10),
                              save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot sensitivity analysis for all analyzed parameters.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not self.sensitivity_results:
            raise ValueError("No sensitivity results available")
        
        n_params = len(self.sensitivity_results)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (param_name, result) in enumerate(self.sensitivity_results.items()):
            ax = axes[idx]
            
            # Plot metric values
            ax.plot(result.parameter_values, result.metric_values, 'o-', linewidth=2, markersize=6)
            
            # Mark baseline
            if result.baseline_value in result.parameter_values:
                baseline_idx = result.parameter_values.index(result.baseline_value)
                ax.plot(result.baseline_value, result.metric_values[baseline_idx],
                       'r*', markersize=12)
            
            ax.set_xlabel(param_name, fontsize=10)
            ax.set_ylabel(result.metric_name, fontsize=10)
            ax.set_title(f'{param_name}\nSensitivity: {result.sensitivity_score:.3f}', fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, y=1.00)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sensitivity plots saved to {save_path}")
        
        return fig
    
    def get_sensitivity_ranking(self) -> pd.DataFrame:
        """
        Get ranking of parameters by sensitivity.
        
        Returns:
            DataFrame with parameters ranked by sensitivity score
        """
        if not self.sensitivity_results:
            return pd.DataFrame()
        
        rankings = []
        for param_name, result in self.sensitivity_results.items():
            rankings.append({
                'parameter': param_name,
                'sensitivity_score': result.sensitivity_score,
                'baseline_value': result.baseline_value,
                'baseline_metric': result.baseline_metric,
                'metric_range': max(result.metric_values) - min(result.metric_values)
            })
        
        df = pd.DataFrame(rankings)
        df = df.sort_values('sensitivity_score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_sensitivity_report(self) -> str:
        """Generate human-readable sensitivity report."""
        if not self.sensitivity_results:
            return "No sensitivity analysis results available"
        
        report = []
        report.append("="*70)
        report.append("HYPERPARAMETER SENSITIVITY ANALYSIS REPORT")
        report.append("="*70)
        report.append("")
        
        # Baseline configuration
        report.append("Baseline Configuration:")
        for param, value in self.baseline_config.items():
            report.append(f"  {param}: {value}")
        report.append(f"\nBaseline {self.metric_name}: {self.baseline_metric:.4f}")
        report.append("")
        
        # Parameter rankings
        ranking_df = self.get_sensitivity_ranking()
        report.append("Parameter Sensitivity Ranking:")
        report.append("")
        
        for _, row in ranking_df.iterrows():
            report.append(f"{row['rank']}. {row['parameter']}")
            report.append(f"   Sensitivity Score: {row['sensitivity_score']:.4f}")
            report.append(f"   Metric Range: {row['metric_range']:.4f}")
            report.append(f"   Baseline Value: {row['baseline_value']}")
            report.append("")
        
        # Interaction results
        if self.interaction_results:
            report.append("Parameter Interactions:")
            report.append("")
            
            for (param1, param2), result in self.interaction_results.items():
                report.append(f"  {param1} × {param2}")
                report.append(f"    Interaction Strength: {result.interaction_strength:.4f}")
                report.append("")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_results(self, filepath: Path):
        """Save all sensitivity analysis results to file."""
        results = {
            'baseline_config': self.baseline_config,
            'baseline_metric': self.baseline_metric,
            'metric_name': self.metric_name,
            'sensitivity_results': {
                name: result.to_dict()
                for name, result in self.sensitivity_results.items()
            },
            'interaction_results': {
                f"{k[0]}_{k[1]}": result.to_dict()
                for k, result in self.interaction_results.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Sensitivity analysis results saved to {filepath}")
