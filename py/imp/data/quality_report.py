"""
Data quality reporting and recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class DataQualityReporter:
    """
    Generate comprehensive data quality reports with visualizations and recommendations.
    """
    
    def __init__(self):
        """Initialize data quality reporter."""
        self.report_data: Dict = {}
    
    def generate_report(self, 
                       df: pd.DataFrame,
                       validation_report=None,
                       preprocessing_stats: Optional[Dict] = None,
                       feature_importance: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            validation_report: Optional ValidationReport object
            preprocessing_stats: Optional preprocessing statistics
            feature_importance: Optional feature importance DataFrame
        
        Returns:
            Dictionary containing report data
        """
        print("📊 Generating data quality report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overview': self._generate_overview(df),
            'quality_metrics': self._compute_quality_metrics(df),
            'statistical_summary': self._generate_statistical_summary(df),
            'temporal_analysis': self._analyze_temporal_properties(df),
            'correlation_analysis': self._analyze_correlations(df),
            'recommendations': []
        }
        
        # Add validation results if available
        if validation_report:
            report['validation'] = {
                'is_valid': validation_report.is_valid,
                'checks_passed': validation_report.checks_passed,
                'checks_failed': validation_report.checks_failed,
                'warnings': validation_report.warnings
            }
        
        # Add preprocessing stats if available
        if preprocessing_stats:
            report['preprocessing'] = preprocessing_stats
        
        # Add feature importance if available
        if feature_importance is not None:
            report['feature_importance'] = feature_importance.to_dict('records')
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(df, validation_report)
        
        self.report_data = report
        
        print("✓ Report generated")
        
        return report
    
    def _generate_overview(self, df: pd.DataFrame) -> Dict:
        """Generate overview section."""
        return {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'features': df.columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'date_range': {
                'start': str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None,
                'end': str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None
            }
        }
    
    def _compute_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute data quality metrics."""
        metrics = {}
        
        # Missing data metrics
        missing_counts = df.isna().sum()
        metrics['missing_data'] = {
            'total_missing': int(missing_counts.sum()),
            'missing_pct': float((missing_counts.sum() / df.size) * 100),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'complete_rows': int((~df.isna().any(axis=1)).sum()),
            'complete_rows_pct': float(((~df.isna().any(axis=1)).sum() / len(df)) * 100)
        }
        
        # Numerical stability metrics
        metrics['numerical_stability'] = {}
        for col in df.columns:
            metrics['numerical_stability'][col] = {
                'has_inf': bool(np.isinf(df[col]).any()),
                'has_nan': bool(df[col].isna().any()),
                'is_constant': bool(df[col].nunique() <= 1),
                'range': float(df[col].max() - df[col].min()) if not df[col].isna().all() else None
            }
        
        # Variability metrics
        metrics['variability'] = {}
        for col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                metrics['variability'][col] = {
                    'std': float(data.std()),
                    'cv': float(data.std() / data.mean()) if data.mean() != 0 else None,
                    'unique_values': int(data.nunique()),
                    'unique_ratio': float(data.nunique() / len(data))
                }
        
        return metrics
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict:
        """Generate statistical summary."""
        summary = {}
        
        for col in df.columns:
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            summary[col] = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'q25': float(data.quantile(0.25)),
                'median': float(data.median()),
                'q75': float(data.quantile(0.75)),
                'max': float(data.max()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis())
            }
        
        return summary
    
    def _analyze_temporal_properties(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal properties."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return {'available': False, 'reason': 'Index is not DatetimeIndex'}
        
        analysis = {'available': True}
        
        # Time interval analysis
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            analysis['intervals'] = {
                'min': str(time_diffs.min()),
                'max': str(time_diffs.max()),
                'mean': str(time_diffs.mean()),
                'is_regular': bool(time_diffs.nunique() == 1),
                'inferred_freq': str(pd.infer_freq(df.index)) if len(df) > 2 else None
            }
        
        # Duplicate timestamps
        analysis['duplicates'] = {
            'has_duplicates': bool(df.index.duplicated().any()),
            'n_duplicates': int(df.index.duplicated().sum())
        }
        
        # Monotonicity
        analysis['monotonic'] = {
            'is_increasing': bool(df.index.is_monotonic_increasing),
            'is_decreasing': bool(df.index.is_monotonic_decreasing)
        }
        
        return analysis
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze feature correlations."""
        if df.shape[1] < 2:
            return {'available': False, 'reason': 'Need at least 2 features'}
        
        corr_matrix = df.corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        analysis = {
            'available': True,
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr,
            'mean_abs_correlation': float(corr_matrix.abs().mean().mean())
        }
        
        return analysis
    
    def _generate_recommendations(self, 
                                 df: pd.DataFrame,
                                 validation_report=None) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on data quality metrics
        missing_pct = (df.isna().sum().sum() / df.size) * 100
        if missing_pct > 5:
            recommendations.append(
                f"High missing data ({missing_pct:.1f}%). "
                f"Use SignalPreprocessor with handle_missing='interpolate' or 'forward_fill'"
            )
        
        # Based on feature characteristics
        for col in df.columns:
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            # Check skewness
            skewness = data.skew()
            if abs(skewness) > 2:
                recommendations.append(
                    f"{col}: High skewness ({skewness:.2f}). "
                    f"Consider log transformation or robust scaling"
                )
            
            # Check for low variability
            if data.std() < 0.01:
                recommendations.append(
                    f"{col}: Very low variability (std={data.std():.4f}). "
                    f"May not contribute to regime detection"
                )
        
        # Based on correlations
        if df.shape[1] > 1:
            corr_matrix = df.corr().abs()
            high_corr_count = ((corr_matrix > 0.95).sum().sum() - len(corr_matrix)) // 2
            
            if high_corr_count > 0:
                recommendations.append(
                    f"Found {high_corr_count} highly correlated feature pairs (>0.95). "
                    f"Consider feature selection or PCA"
                )
        
        # Based on sample size
        if len(df) < 100:
            recommendations.append(
                f"Small sample size ({len(df)}). "
                f"Consider collecting more data for robust HMM training (recommended: >200)"
            )
        
        # Add validation-based recommendations
        if validation_report and validation_report.recommendations:
            recommendations.extend(validation_report.recommendations)
        
        # Remove duplicates
        recommendations = list(dict.fromkeys(recommendations))
        
        return recommendations
    
    def print_report(self, detailed: bool = True):
        """Print formatted report to console."""
        if not self.report_data:
            print("No report data available. Call generate_report first.")
            return
        
        report = self.report_data
        
        print("\n" + "="*70)
        print("📊 DATA QUALITY REPORT")
        print("="*70)
        print(f"Generated: {report['timestamp']}")
        print()
        
        # Overview
        print("📋 OVERVIEW")
        print("-" * 70)
        overview = report['overview']
        print(f"Samples: {overview['n_samples']:,}")
        print(f"Features: {overview['n_features']}")
        print(f"Memory: {overview['memory_usage_mb']:.2f} MB")
        if overview['date_range']['start']:
            print(f"Date Range: {overview['date_range']['start']} to {overview['date_range']['end']}")
        print()
        
        # Quality Metrics
        print("✓ QUALITY METRICS")
        print("-" * 70)
        missing = report['quality_metrics']['missing_data']
        print(f"Missing Data: {missing['missing_pct']:.2f}% ({missing['total_missing']:,} values)")
        print(f"Complete Rows: {missing['complete_rows_pct']:.1f}% ({missing['complete_rows']:,} rows)")
        print()
        
        if detailed:
            # Statistical Summary
            print("📈 STATISTICAL SUMMARY")
            print("-" * 70)
            stats_df = pd.DataFrame(report['statistical_summary']).T
            print(stats_df[['mean', 'std', 'min', 'max', 'skewness']].to_string())
            print()
            
            # Correlation Analysis
            if report['correlation_analysis']['available']:
                corr_analysis = report['correlation_analysis']
                print("🔗 CORRELATION ANALYSIS")
                print("-" * 70)
                print(f"Mean Absolute Correlation: {corr_analysis['mean_abs_correlation']:.3f}")
                
                if corr_analysis['high_correlations']:
                    print(f"\nHigh Correlations (>0.7): {len(corr_analysis['high_correlations'])}")
                    for hc in corr_analysis['high_correlations'][:5]:  # Show top 5
                        print(f"  • {hc['feature1']} - {hc['feature2']}: {hc['correlation']:.3f}")
                print()
        
        # Recommendations
        if report['recommendations']:
            print("💡 RECOMMENDATIONS")
            print("-" * 70)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
            print()
        
        print("="*70)
    
    def save_report(self, output_path: Union[str, Path], format: str = 'json'):
        """
        Save report to file.
        
        Args:
            output_path: Path to save report
            format: Output format ('json', 'html', 'txt')
        """
        if not self.report_data:
            raise ValueError("No report data available. Call generate_report first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.report_data, f, indent=2, default=str)
        elif format == 'txt':
            # Redirect print to file
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            self.print_report(detailed=True)
            report_text = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            with open(output_path, 'w') as f:
                f.write(report_text)
        elif format == 'html':
            self._save_html_report(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✓ Report saved to {output_path}")
    
    def _save_html_report(self, output_path: Path):
        """Save report as HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
            </style>
        </head>
        <body>
            <h1>📊 Data Quality Report</h1>
            <p>Generated: {self.report_data['timestamp']}</p>
            
            <h2>Overview</h2>
            <div class="metric">
                <p><strong>Samples:</strong> {self.report_data['overview']['n_samples']:,}</p>
                <p><strong>Features:</strong> {self.report_data['overview']['n_features']}</p>
                <p><strong>Memory:</strong> {self.report_data['overview']['memory_usage_mb']:.2f} MB</p>
            </div>
            
            <h2>Quality Metrics</h2>
            <div class="metric">
                <p><strong>Missing Data:</strong> {self.report_data['quality_metrics']['missing_data']['missing_pct']:.2f}%</p>
                <p><strong>Complete Rows:</strong> {self.report_data['quality_metrics']['missing_data']['complete_rows_pct']:.1f}%</p>
            </div>
            
            <h2>Recommendations</h2>
        """
        
        for rec in self.report_data['recommendations']:
            html += f'<div class="recommendation">{rec}</div>\n'
        
        html += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def plot_quality_dashboard(self, df: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Create visualization dashboard for data quality.
        
        Args:
            df: DataFrame to visualize
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Quality Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Missing data heatmap
        ax = axes[0, 0]
        if df.isna().any().any():
            sns.heatmap(df.isna().T, cmap='RdYlGn_r', cbar=True, ax=ax)
            ax.set_title('Missing Data Pattern')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Features')
        else:
            ax.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', fontsize=14)
            ax.set_title('Missing Data Pattern')
            ax.axis('off')
        
        # 2. Feature distributions
        ax = axes[0, 1]
        df.boxplot(ax=ax, rot=45)
        ax.set_title('Feature Distributions')
        ax.set_ylabel('Value')
        
        # 3. Correlation heatmap
        ax = axes[1, 0]
        if df.shape[1] > 1:
            sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlations')
        else:
            ax.text(0.5, 0.5, 'Need >1 feature', ha='center', va='center', fontsize=14)
            ax.set_title('Feature Correlations')
            ax.axis('off')
        
        # 4. Time series plot (if temporal)
        ax = axes[1, 1]
        if isinstance(df.index, pd.DatetimeIndex):
            for col in df.columns[:5]:  # Plot first 5 features
                ax.plot(df.index, df[col], label=col, alpha=0.7)
            ax.set_title('Time Series Overview')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not Time Series', ha='center', va='center', fontsize=14)
            ax.set_title('Time Series Overview')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dashboard saved to {save_path}")
        
        return fig
