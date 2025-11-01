"""
Validation reporting and analysis for walk-forward validation.

This module provides comprehensive reporting capabilities including:
- In-sample vs out-of-sample performance comparison
- Model stability tracking and retraining recommendations
- Statistical significance tests
- Visualization and export functionality
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .walk_forward_validator import ValidationWindow, ValidationReport
from .performance_analyzer import PerformanceMetrics


logger = logging.getLogger(__name__)


class ValidationReportGenerator:
    """
    Generate comprehensive validation reports with statistical analysis.
    
    This class provides:
    - In-sample vs out-of-sample performance comparison
    - Model stability tracking
    - Retraining recommendations
    - Statistical significance tests
    - Visualization and export
    
    Requirements: 5.4, 5.5
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize ValidationReportGenerator.
        
        Args:
            output_dir: Directory for saving reports and visualizations
        """
        self.output_dir = output_dir or Path("./validation_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ValidationReportGenerator with output_dir={self.output_dir}")
    
    def generate_comparison_report(
        self,
        validation_report: ValidationReport,
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate in-sample vs out-of-sample performance comparison.
        
        Args:
            validation_report: Validation report from walk-forward analysis
            include_visualizations: Whether to generate visualization plots
            
        Returns:
            Comprehensive comparison report
            
        Requirements: 5.4
        """
        logger.info("Generating in-sample vs out-of-sample comparison report")
        
        # Extract valid windows
        valid_windows = [w for w in validation_report.windows 
                        if w.train_metrics and w.test_metrics]
        
        if not valid_windows:
            logger.warning("No valid windows for comparison report")
            return self._empty_comparison_report()
        
        # Collect metrics for comparison
        comparison_data = self._collect_comparison_metrics(valid_windows)
        
        # Statistical analysis
        statistical_tests = self._perform_statistical_tests(comparison_data)
        
        # Performance consistency analysis
        consistency_analysis = self._analyze_performance_consistency(comparison_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            validation_report,
            comparison_data,
            statistical_tests,
            consistency_analysis
        )
        
        # Create comprehensive report
        report = {
            'summary': {
                'num_windows': len(valid_windows),
                'avg_in_sample_return': validation_report.avg_in_sample_return,
                'avg_out_of_sample_return': validation_report.avg_out_of_sample_return,
                'performance_degradation': validation_report.performance_degradation,
                'stability_score': validation_report.stability_score
            },
            'comparison_metrics': comparison_data,
            'statistical_tests': statistical_tests,
            'consistency_analysis': consistency_analysis,
            'recommendations': recommendations,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'num_retrains': validation_report.num_retrains,
                'retrain_windows': validation_report.retrain_windows
            }
        }
        
        # Save report
        report_path = self.output_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved comparison report to {report_path}")
        
        # Generate visualizations
        if include_visualizations:
            self._generate_comparison_visualizations(validation_report, comparison_data)
        
        return report
    
    def _collect_comparison_metrics(
        self,
        windows: List[ValidationWindow]
    ) -> Dict[str, Any]:
        """Collect metrics for in-sample vs out-of-sample comparison."""
        metrics = {
            'returns': {
                'in_sample': [],
                'out_of_sample': []
            },
            'sharpe_ratios': {
                'in_sample': [],
                'out_of_sample': []
            },
            'max_drawdowns': {
                'in_sample': [],
                'out_of_sample': []
            },
            'win_rates': {
                'in_sample': [],
                'out_of_sample': []
            },
            'volatility': {
                'in_sample': [],
                'out_of_sample': []
            }
        }
        
        for window in windows:
            # Returns
            metrics['returns']['in_sample'].append(window.train_metrics.total_return)
            metrics['returns']['out_of_sample'].append(window.test_metrics.total_return)
            
            # Sharpe ratios
            metrics['sharpe_ratios']['in_sample'].append(window.train_metrics.sharpe_ratio)
            metrics['sharpe_ratios']['out_of_sample'].append(window.test_metrics.sharpe_ratio)
            
            # Max drawdowns
            metrics['max_drawdowns']['in_sample'].append(window.train_metrics.max_drawdown)
            metrics['max_drawdowns']['out_of_sample'].append(window.test_metrics.max_drawdown)
            
            # Win rates
            metrics['win_rates']['in_sample'].append(window.train_metrics.win_rate)
            metrics['win_rates']['out_of_sample'].append(window.test_metrics.win_rate)
            
            # Volatility
            metrics['volatility']['in_sample'].append(window.train_metrics.annualized_volatility)
            metrics['volatility']['out_of_sample'].append(window.test_metrics.annualized_volatility)
        
        return metrics
    
    def _perform_statistical_tests(
        self,
        comparison_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform statistical significance tests.
        
        Requirements: 5.5
        """
        tests = {}
        
        for metric_name, metric_data in comparison_data.items():
            in_sample = np.array(metric_data['in_sample'])
            out_of_sample = np.array(metric_data['out_of_sample'])
            
            if len(in_sample) < 2:
                continue
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(in_sample, out_of_sample)
            
            # Effect size (Cohen's d)
            diff = in_sample - out_of_sample
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                w_stat, w_pvalue = stats.wilcoxon(in_sample, out_of_sample)
            except ValueError:
                w_stat, w_pvalue = 0.0, 1.0
            
            tests[metric_name] = {
                'paired_ttest': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_at_0.05': p_value < 0.05,
                    'significant_at_0.01': p_value < 0.01
                },
                'effect_size': {
                    'cohens_d': float(cohens_d),
                    'interpretation': self._interpret_cohens_d(cohens_d)
                },
                'wilcoxon_test': {
                    'statistic': float(w_stat),
                    'p_value': float(w_pvalue),
                    'significant_at_0.05': w_pvalue < 0.05
                },
                'descriptive_stats': {
                    'in_sample_mean': float(np.mean(in_sample)),
                    'out_of_sample_mean': float(np.mean(out_of_sample)),
                    'difference': float(np.mean(in_sample) - np.mean(out_of_sample)),
                    'in_sample_std': float(np.std(in_sample)),
                    'out_of_sample_std': float(np.std(out_of_sample))
                }
            }
        
        return tests
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _analyze_performance_consistency(
        self,
        comparison_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze performance consistency across windows.
        
        Requirements: 5.4
        """
        consistency = {}
        
        for metric_name, metric_data in comparison_data.items():
            out_of_sample = np.array(metric_data['out_of_sample'])
            
            if len(out_of_sample) < 2:
                continue
            
            # Coefficient of variation
            mean_val = np.mean(out_of_sample)
            std_val = np.std(out_of_sample)
            cv = abs(std_val / mean_val) if mean_val != 0 else float('inf')
            
            # Percentage of positive windows
            positive_pct = np.sum(out_of_sample > 0) / len(out_of_sample) * 100
            
            # Trend analysis (linear regression)
            x = np.arange(len(out_of_sample))
            if len(x) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, out_of_sample)
                trend = {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'direction': 'improving' if slope > 0 else 'degrading',
                    'significant': p_value < 0.05
                }
            else:
                trend = None
            
            consistency[metric_name] = {
                'coefficient_of_variation': float(cv),
                'consistency_score': float(1.0 / (1.0 + cv)),  # Higher is more consistent
                'positive_windows_pct': float(positive_pct),
                'min': float(np.min(out_of_sample)),
                'max': float(np.max(out_of_sample)),
                'median': float(np.median(out_of_sample)),
                'trend': trend
            }
        
        return consistency
    
    def _generate_recommendations(
        self,
        validation_report: ValidationReport,
        comparison_data: Dict[str, Any],
        statistical_tests: Dict[str, Any],
        consistency_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate actionable recommendations based on validation results.
        
        Requirements: 5.4, 5.5
        """
        recommendations = []
        
        # Check performance degradation
        if validation_report.performance_degradation > 0.2:
            recommendations.append({
                'priority': 'high',
                'category': 'performance',
                'issue': 'Significant performance degradation detected',
                'recommendation': f'Out-of-sample performance is {validation_report.performance_degradation:.1%} '
                                f'worse than in-sample. Consider more frequent retraining or model adjustments.',
                'action': 'Reduce retraining threshold or review model assumptions'
            })
        
        # Check stability
        if validation_report.stability_score < 0.5:
            recommendations.append({
                'priority': 'high',
                'category': 'stability',
                'issue': 'Low performance stability across windows',
                'recommendation': f'Stability score is {validation_report.stability_score:.2f}. '
                                f'Model performance is inconsistent across time periods.',
                'action': 'Review model robustness and consider ensemble methods'
            })
        
        # Check statistical significance
        returns_test = statistical_tests.get('returns', {})
        if returns_test.get('paired_ttest', {}).get('significant_at_0.05', False):
            recommendations.append({
                'priority': 'medium',
                'category': 'overfitting',
                'issue': 'Statistically significant difference between in-sample and out-of-sample returns',
                'recommendation': 'Model may be overfitting to training data. '
                                f'P-value: {returns_test["paired_ttest"]["p_value"]:.4f}',
                'action': 'Implement regularization or reduce model complexity'
            })
        
        # Check retraining frequency
        if validation_report.num_retrains > len(validation_report.windows) * 0.5:
            recommendations.append({
                'priority': 'medium',
                'category': 'retraining',
                'issue': 'Frequent retraining detected',
                'recommendation': f'Model was retrained {validation_report.num_retrains} times '
                                f'across {len(validation_report.windows)} windows. '
                                f'This may indicate model instability.',
                'action': 'Review retraining threshold or improve model robustness'
            })
        elif validation_report.num_retrains == 0 and len(validation_report.windows) > 5:
            recommendations.append({
                'priority': 'low',
                'category': 'retraining',
                'issue': 'No retraining occurred',
                'recommendation': 'Model was never retrained. Consider if retraining threshold is too high.',
                'action': 'Review retraining threshold setting'
            })
        
        # Check consistency
        returns_consistency = consistency_analysis.get('returns', {})
        if returns_consistency.get('positive_windows_pct', 0) < 60:
            recommendations.append({
                'priority': 'high',
                'category': 'consistency',
                'issue': 'Low percentage of profitable windows',
                'recommendation': f'Only {returns_consistency["positive_windows_pct"]:.1f}% of windows '
                                f'were profitable. Strategy may not be robust.',
                'action': 'Review strategy logic and risk management'
            })
        
        # Check trend
        returns_trend = returns_consistency.get('trend')
        if returns_trend and returns_trend.get('significant', False):
            if returns_trend['direction'] == 'degrading':
                recommendations.append({
                    'priority': 'high',
                    'category': 'trend',
                    'issue': 'Significant degrading performance trend',
                    'recommendation': f'Performance shows statistically significant decline over time '
                                    f'(slope={returns_trend["slope"]:.4f}, p={returns_trend["p_value"]:.4f})',
                    'action': 'Investigate market regime changes or model decay'
                })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def _generate_comparison_visualizations(
        self,
        validation_report: ValidationReport,
        comparison_data: Dict[str, Any]
    ) -> None:
        """Generate visualization plots for validation results."""
        logger.info("Generating validation visualizations")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Returns comparison plot
        self._plot_returns_comparison(validation_report, comparison_data)
        
        # 2. Performance metrics comparison
        self._plot_metrics_comparison(comparison_data)
        
        # 3. Stability over time
        self._plot_stability_over_time(validation_report)
        
        # 4. Retraining timeline
        self._plot_retraining_timeline(validation_report)
        
        logger.info(f"Saved visualizations to {self.output_dir}")
    
    def _plot_returns_comparison(
        self,
        validation_report: ValidationReport,
        comparison_data: Dict[str, Any]
    ) -> None:
        """Plot in-sample vs out-of-sample returns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Line plot
        windows = range(len(comparison_data['returns']['in_sample']))
        ax1.plot(windows, comparison_data['returns']['in_sample'], 
                marker='o', label='In-Sample', linewidth=2)
        ax1.plot(windows, comparison_data['returns']['out_of_sample'], 
                marker='s', label='Out-of-Sample', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Return')
        ax1.set_title('Returns: In-Sample vs Out-of-Sample')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [
            comparison_data['returns']['in_sample'],
            comparison_data['returns']['out_of_sample']
        ]
        ax2.boxplot(data_to_plot, labels=['In-Sample', 'Out-of-Sample'])
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Return')
        ax2.set_title('Return Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'returns_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, comparison_data: Dict[str, Any]) -> None:
        """Plot comparison of multiple metrics."""
        metrics_to_plot = ['sharpe_ratios', 'max_drawdowns', 'win_rates']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric_name in enumerate(metrics_to_plot):
            if metric_name not in comparison_data:
                continue
            
            ax = axes[idx]
            data_to_plot = [
                comparison_data[metric_name]['in_sample'],
                comparison_data[metric_name]['out_of_sample']
            ]
            
            ax.boxplot(data_to_plot, labels=['In-Sample', 'Out-of-Sample'])
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stability_over_time(self, validation_report: ValidationReport) -> None:
        """Plot performance stability over time."""
        valid_windows = [w for w in validation_report.windows 
                        if w.train_metrics and w.test_metrics]
        
        if not valid_windows:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate rolling statistics
        window_size = min(3, len(valid_windows))
        returns = [w.test_metrics.total_return for w in valid_windows]
        
        if len(returns) >= window_size:
            rolling_mean = pd.Series(returns).rolling(window=window_size).mean()
            rolling_std = pd.Series(returns).rolling(window=window_size).std()
            
            windows = range(len(returns))
            ax.plot(windows, returns, marker='o', label='Out-of-Sample Return', alpha=0.6)
            ax.plot(windows, rolling_mean, linewidth=2, label=f'{window_size}-Window Moving Average')
            ax.fill_between(windows, 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std,
                           alpha=0.2, label='±1 Std Dev')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Window')
        ax.set_ylabel('Return')
        ax.set_title('Performance Stability Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stability_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_retraining_timeline(self, validation_report: ValidationReport) -> None:
        """Plot retraining timeline."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        valid_windows = [w for w in validation_report.windows 
                        if w.train_metrics and w.test_metrics]
        
        if not valid_windows:
            return
        
        windows = range(len(valid_windows))
        returns = [w.test_metrics.total_return for w in valid_windows]
        retrained = [w.retrained for w in valid_windows]
        
        # Plot returns
        ax.plot(windows, returns, marker='o', linewidth=2, label='Out-of-Sample Return')
        
        # Mark retraining points
        retrain_windows = [i for i, r in enumerate(retrained) if r]
        if retrain_windows:
            retrain_returns = [returns[i] for i in retrain_windows]
            ax.scatter(retrain_windows, retrain_returns, 
                      color='red', s=200, marker='*', 
                      label='Retraining Point', zorder=5)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Window')
        ax.set_ylabel('Return')
        ax.set_title('Retraining Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'retraining_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _empty_comparison_report(self) -> Dict[str, Any]:
        """Return empty comparison report."""
        return {
            'summary': {
                'num_windows': 0,
                'avg_in_sample_return': 0.0,
                'avg_out_of_sample_return': 0.0,
                'performance_degradation': 0.0,
                'stability_score': 0.0
            },
            'comparison_metrics': {},
            'statistical_tests': {},
            'consistency_analysis': {},
            'recommendations': [],
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'num_retrains': 0,
                'retrain_windows': []
            }
        }
    
    def export_to_csv(self, validation_report: ValidationReport) -> Path:
        """
        Export validation results to CSV for further analysis.
        
        Args:
            validation_report: Validation report to export
            
        Returns:
            Path to exported CSV file
        """
        valid_windows = [w for w in validation_report.windows 
                        if w.train_metrics and w.test_metrics]
        
        if not valid_windows:
            logger.warning("No valid windows to export")
            return None
        
        # Create DataFrame
        data = []
        for window in valid_windows:
            row = {
                'window_id': window.window_id,
                'train_start': window.train_start,
                'train_end': window.train_end,
                'test_start': window.test_start,
                'test_end': window.test_end,
                'retrained': window.retrained,
                'model_version': window.model_version,
                
                # Training metrics
                'train_return': window.train_metrics.total_return,
                'train_sharpe': window.train_metrics.sharpe_ratio,
                'train_max_dd': window.train_metrics.max_drawdown,
                'train_win_rate': window.train_metrics.win_rate,
                
                # Testing metrics
                'test_return': window.test_metrics.total_return,
                'test_sharpe': window.test_metrics.sharpe_ratio,
                'test_max_dd': window.test_metrics.max_drawdown,
                'test_win_rate': window.test_metrics.win_rate,
                
                # Degradation
                'return_degradation': (window.train_metrics.total_return - 
                                      window.test_metrics.total_return)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Export to CSV
        csv_path = self.output_dir / 'validation_results.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Exported validation results to {csv_path}")
        
        return csv_path
