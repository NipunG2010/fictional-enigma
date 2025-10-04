"""
Automated report generation for model comparison and selection recommendations.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import logging
from dataclasses import dataclass

from .model_selection import ModelSelectionResult, SelectionCriteria
from .ensemble_evaluation import EnsembleResult
from .sensitivity_analysis import SensitivityAnalyzer
from ..evaluation.evaluator import HMMEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_sensitivity: bool = True
    include_ensemble: bool = True
    include_cv_details: bool = True
    include_visualizations: bool = True
    output_format: str = 'html'  # 'html', 'markdown', or 'pdf'
    
    def __post_init__(self):
        if self.output_format not in ['html', 'markdown', 'pdf']:
            raise ValueError(f"Unsupported output format: {self.output_format}")


class ReportGenerator:
    """
    Automated report generation for model comparison and selection.
    
    Generates comprehensive reports including:
    - Model selection results
    - Performance comparisons
    - Sensitivity analysis
    - Ensemble evaluation
    - Recommendations
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports and figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
    
    def generate_full_report(self,
                            selection_result: ModelSelectionResult,
                            sensitivity_analyzer: Optional[SensitivityAnalyzer] = None,
                            ensemble_result: Optional[EnsembleResult] = None,
                            config: Optional[ReportConfig] = None) -> Path:
        """
        Generate comprehensive model selection report.
        
        Args:
            selection_result: Results from automated model selection
            sensitivity_analyzer: Optional sensitivity analysis results
            ensemble_result: Optional ensemble evaluation results
            config: Report configuration
            
        Returns:
            Path to generated report file
        """
        if config is None:
            config = ReportConfig()
        
        logger.info(f"Generating {config.output_format} report")
        
        # Generate report sections
        sections = []
        
        # Title and summary
        sections.append(self._generate_title_section(selection_result))
        sections.append(self._generate_executive_summary(selection_result))
        
        # Model selection results
        sections.append(self._generate_selection_section(selection_result))
        
        # Performance comparison
        sections.append(self._generate_comparison_section(selection_result))
        
        # Cross-validation details
        if config.include_cv_details:
            sections.append(self._generate_cv_section(selection_result))
        
        # Sensitivity analysis
        if config.include_sensitivity and sensitivity_analyzer:
            sections.append(self._generate_sensitivity_section(sensitivity_analyzer))
        
        # Ensemble evaluation
        if config.include_ensemble and ensemble_result:
            sections.append(self._generate_ensemble_section(ensemble_result))
        
        # Recommendations
        sections.append(self._generate_recommendations_section(
            selection_result, sensitivity_analyzer, ensemble_result
        ))
        
        # Generate visualizations
        if config.include_visualizations:
            self._generate_all_visualizations(
                selection_result, sensitivity_analyzer, ensemble_result
            )
        
        # Combine sections and save
        if config.output_format == 'html':
            report_path = self._save_html_report(sections)
        elif config.output_format == 'markdown':
            report_path = self._save_markdown_report(sections)
        else:  # pdf
            report_path = self._save_pdf_report(sections)
        
        logger.info(f"Report generated: {report_path}")
        
        return report_path
    
    def _generate_title_section(self, selection_result: ModelSelectionResult) -> Dict[str, Any]:
        """Generate title section."""
        return {
            'type': 'title',
            'content': {
                'title': 'HMM Model Selection Report',
                'subtitle': 'Automated Hyperparameter Optimization and Model Comparison',
                'timestamp': selection_result.timestamp,
                'optimization_method': selection_result.optimization_method
            }
        }
    
    def _generate_executive_summary(self, selection_result: ModelSelectionResult) -> Dict[str, Any]:
        """Generate executive summary."""
        best_config = selection_result.best_config
        
        summary_text = f"""
This report presents the results of automated model selection for Hidden Markov Models (HMM).
The optimization was performed using {selection_result.optimization_method} and evaluated
{len(selection_result.all_comparisons)} different configurations.

**Best Model Configuration:**
- Number of States: {best_config.get('n_states', 'N/A')}
- Library: {best_config.get('library', 'N/A')}
- Covariance Type: {best_config.get('covariance_type', 'N/A')}
- Selection Score: {selection_result.best_score:.4f}

The selected model was chosen based on multiple criteria including model fit,
complexity, and stability metrics.
"""
        
        return {
            'type': 'section',
            'title': 'Executive Summary',
            'content': summary_text
        }
    
    def _generate_selection_section(self, selection_result: ModelSelectionResult) -> Dict[str, Any]:
        """Generate model selection section."""
        # Selection criteria table
        criteria_data = []
        for criterion in selection_result.selection_criteria:
            criteria_data.append({
                'Metric': criterion.metric_name,
                'Weight': f"{criterion.weight:.2f}",
                'Direction': 'Higher is better' if criterion.higher_is_better else 'Lower is better',
                'Threshold': criterion.threshold if criterion.threshold else 'None'
            })
        
        criteria_df = pd.DataFrame(criteria_data)
        
        content = f"""
### Selection Criteria

The following criteria were used for model selection:

{criteria_df.to_markdown(index=False)}

### Best Model Details

**Configuration:**
```json
{json.dumps(selection_result.best_config, indent=2)}
```

**Performance Metrics:**
- Log-Likelihood: {selection_result.best_artifact.metadata.get('convergence_log_likelihood', 'N/A')}
- AIC: {selection_result.best_artifact.metadata.get('aic', 'N/A')}
- BIC: {selection_result.best_artifact.metadata.get('bic', 'N/A')}
- Converged: {selection_result.best_artifact.metadata.get('converged', 'N/A')}
"""
        
        return {
            'type': 'section',
            'title': 'Model Selection Results',
            'content': content
        }
    
    def _generate_comparison_section(self, selection_result: ModelSelectionResult) -> Dict[str, Any]:
        """Generate performance comparison section."""
        # Get top 10 models
        top_models = selection_result.all_comparisons.nsmallest(10, 'rank')
        
        # Create comparison table
        comparison_cols = ['rank', 'config', 'log_likelihood', 'aic', 'bic']
        available_cols = [col for col in comparison_cols if col in top_models.columns]
        comparison_table = top_models[available_cols]
        
        content = f"""
### Top Performing Models

The following table shows the top 10 models ranked by the selection criteria:

{comparison_table.to_markdown(index=False)}

### Performance Distribution

See the accompanying visualizations for detailed performance distributions across all evaluated models.
"""
        
        return {
            'type': 'section',
            'title': 'Performance Comparison',
            'content': content
        }
    
    def _generate_cv_section(self, selection_result: ModelSelectionResult) -> Dict[str, Any]:
        """Generate cross-validation section."""
        # Extract CV columns
        cv_cols = [col for col in selection_result.all_comparisons.columns if col.startswith('cv_')]
        
        if not cv_cols:
            return {
                'type': 'section',
                'title': 'Cross-Validation Results',
                'content': 'Cross-validation was not performed for this analysis.'
            }
        
        # Get CV results for top models
        top_models = selection_result.all_comparisons.nsmallest(5, 'rank')
        cv_data = top_models[['config'] + cv_cols]
        
        content = f"""
### Cross-Validation Performance

Cross-validation was performed to assess model generalization:

{cv_data.to_markdown(index=False)}

Lower standard deviations indicate more stable model performance across different data splits.
"""
        
        return {
            'type': 'section',
            'title': 'Cross-Validation Results',
            'content': content
        }
    
    def _generate_sensitivity_section(self, sensitivity_analyzer: SensitivityAnalyzer) -> Dict[str, Any]:
        """Generate sensitivity analysis section."""
        ranking_df = sensitivity_analyzer.get_sensitivity_ranking()
        
        content = f"""
### Parameter Sensitivity Analysis

The following parameters were analyzed for their impact on model performance:

{ranking_df.to_markdown(index=False)}

**Interpretation:**
- Higher sensitivity scores indicate parameters that have a larger impact on model performance
- Parameters with high sensitivity should be carefully tuned
- Parameters with low sensitivity can use default values

See the sensitivity plots in the figures directory for detailed visualizations.
"""
        
        return {
            'type': 'section',
            'title': 'Sensitivity Analysis',
            'content': content
        }
    
    def _generate_ensemble_section(self, ensemble_result: EnsembleResult) -> Dict[str, Any]:
        """Generate ensemble evaluation section."""
        # Create member summary table
        member_data = []
        for i, member in enumerate(ensemble_result.members):
            member_data.append({
                'Member': f"Model {i+1}",
                'States': member.config.get('n_states', 'N/A'),
                'Library': member.config.get('library', 'N/A'),
                'Covariance': member.config.get('covariance_type', 'N/A'),
                'Weight': f"{member.weight:.4f}",
                'Log-Likelihood': f"{member.performance_metrics.get('log_likelihood', 'N/A'):.4f}"
            })
        
        member_df = pd.DataFrame(member_data)
        
        content = f"""
### Ensemble Composition

The ensemble consists of {len(ensemble_result.members)} models:

{member_df.to_markdown(index=False)}

### Ensemble Performance

{json.dumps(ensemble_result.ensemble_performance, indent=2)}

### Diversity Metrics

{json.dumps(ensemble_result.diversity_metrics, indent=2)}

**Interpretation:**
- Higher diversity scores indicate models that make different predictions
- Ensemble methods can improve robustness by combining diverse models
- Weights are assigned based on individual model performance
"""
        
        return {
            'type': 'section',
            'title': 'Ensemble Evaluation',
            'content': content
        }
    
    def _generate_recommendations_section(self,
                                         selection_result: ModelSelectionResult,
                                         sensitivity_analyzer: Optional[SensitivityAnalyzer],
                                         ensemble_result: Optional[EnsembleResult]) -> Dict[str, Any]:
        """Generate recommendations section."""
        recommendations = []
        
        # Model selection recommendation
        best_config = selection_result.best_config
        recommendations.append(f"**Primary Recommendation:** Use the selected model with "
                             f"{best_config.get('n_states')} states and "
                             f"{best_config.get('covariance_type')} covariance.")
        
        # Sensitivity-based recommendations
        if sensitivity_analyzer:
            ranking_df = sensitivity_analyzer.get_sensitivity_ranking()
            if not ranking_df.empty:
                most_sensitive = ranking_df.iloc[0]['parameter']
                recommendations.append(f"**Parameter Tuning:** Focus on tuning '{most_sensitive}' "
                                     f"as it has the highest impact on performance.")
        
        # Ensemble recommendation
        if ensemble_result:
            avg_ll = ensemble_result.ensemble_performance.get('avg_member_log_likelihood', 0)
            best_ll = selection_result.best_artifact.metadata.get('convergence_log_likelihood', 0)
            
            if avg_ll > best_ll:
                recommendations.append("**Ensemble Approach:** Consider using an ensemble of models "
                                     "as it shows better average performance than individual models.")
        
        # Stability recommendations
        if 'converged' in selection_result.best_artifact.metadata:
            if not selection_result.best_artifact.metadata['converged']:
                recommendations.append("**Training:** The selected model did not fully converge. "
                                     "Consider increasing the number of iterations.")
        
        # General recommendations
        recommendations.append("**Validation:** Always validate the selected model on held-out test data "
                             "before deployment to production.")
        recommendations.append("**Monitoring:** Implement performance monitoring to detect model degradation "
                             "over time.")
        
        content = "### Recommendations\n\n" + "\n\n".join(recommendations)
        
        return {
            'type': 'section',
            'title': 'Recommendations',
            'content': content
        }
    
    def _generate_all_visualizations(self,
                                    selection_result: ModelSelectionResult,
                                    sensitivity_analyzer: Optional[SensitivityAnalyzer],
                                    ensemble_result: Optional[EnsembleResult]):
        """Generate all visualization figures."""
        # Performance comparison plot
        self._plot_performance_comparison(selection_result)
        
        # Sensitivity plots
        if sensitivity_analyzer and sensitivity_analyzer.sensitivity_results:
            sensitivity_analyzer.plot_all_sensitivities(
                save_path=self.figures_dir / 'sensitivity_analysis.png'
            )
        
        # Ensemble plots
        if ensemble_result:
            self._plot_ensemble_weights(ensemble_result)
    
    def _plot_performance_comparison(self, selection_result: ModelSelectionResult):
        """Plot performance comparison across models."""
        df = selection_result.all_comparisons
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Log-likelihood distribution
        ax = axes[0, 0]
        ax.hist(df['log_likelihood'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(selection_result.best_artifact.metadata.get('convergence_log_likelihood', 0),
                  color='red', linestyle='--', linewidth=2, label='Best Model')
        ax.set_xlabel('Log-Likelihood')
        ax.set_ylabel('Frequency')
        ax.set_title('Log-Likelihood Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: AIC vs BIC
        ax = axes[0, 1]
        ax.scatter(df['aic'], df['bic'], alpha=0.6)
        best_aic = selection_result.best_artifact.metadata.get('aic', 0)
        best_bic = selection_result.best_artifact.metadata.get('bic', 0)
        ax.scatter([best_aic], [best_bic], color='red', s=200, marker='*',
                  label='Best Model', zorder=5)
        ax.set_xlabel('AIC')
        ax.set_ylabel('BIC')
        ax.set_title('AIC vs BIC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Rank vs Log-Likelihood
        ax = axes[1, 0]
        top_20 = df.nsmallest(20, 'rank')
        ax.plot(top_20['rank'], top_20['log_likelihood'], 'o-')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Log-Likelihood')
        ax.set_title('Top 20 Models: Rank vs Performance')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Configuration distribution
        ax = axes[1, 1]
        if 'n_states' in df.columns:
            state_counts = df['n_states'].value_counts().sort_index()
            ax.bar(state_counts.index, state_counts.values, edgecolor='black', alpha=0.7)
            ax.axvline(selection_result.best_config.get('n_states', 0),
                      color='red', linestyle='--', linewidth=2, label='Best Model')
            ax.set_xlabel('Number of States')
            ax.set_ylabel('Frequency')
            ax.set_title('State Count Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, y=1.00)
        plt.tight_layout()
        
        save_path = self.figures_dir / 'performance_comparison.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Performance comparison plot saved to {save_path}")
    
    def _plot_ensemble_weights(self, ensemble_result: EnsembleResult):
        """Plot ensemble member weights."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        members = [f"Model {i+1}" for i in range(len(ensemble_result.members))]
        weights = [m.weight for m in ensemble_result.members]
        
        ax.bar(members, weights, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Ensemble Member')
        ax.set_ylabel('Weight')
        ax.set_title('Ensemble Member Weights')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        save_path = self.figures_dir / 'ensemble_weights.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Ensemble weights plot saved to {save_path}")
    
    def _save_html_report(self, sections: List[Dict[str, Any]]) -> Path:
        """Save report as HTML."""
        html_content = self._sections_to_html(sections)
        
        report_path = self.output_dir / f"model_selection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _save_markdown_report(self, sections: List[Dict[str, Any]]) -> Path:
        """Save report as Markdown."""
        md_content = self._sections_to_markdown(sections)
        
        report_path = self.output_dir / f"model_selection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write(md_content)
        
        return report_path
    
    def _save_pdf_report(self, sections: List[Dict[str, Any]]) -> Path:
        """Save report as PDF (requires additional dependencies)."""
        # For now, save as markdown and suggest conversion
        md_path = self._save_markdown_report(sections)
        logger.warning("PDF generation requires additional dependencies. "
                      "Markdown report generated instead. "
                      "Use pandoc to convert: pandoc report.md -o report.pdf")
        return md_path
    
    def _sections_to_html(self, sections: List[Dict[str, Any]]) -> str:
        """Convert sections to HTML."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>HMM Model Selection Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }",
            "h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }",
            "h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 8px; margin-top: 30px; }",
            "h3 { color: #666; margin-top: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            "code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }",
            "pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }",
            ".timestamp { color: #888; font-size: 0.9em; }",
            "</style>",
            "</head>",
            "<body>"
        ]
        
        for section in sections:
            if section['type'] == 'title':
                content = section['content']
                html_parts.append(f"<h1>{content['title']}</h1>")
                html_parts.append(f"<p class='timestamp'>{content['subtitle']}<br>")
                html_parts.append(f"Generated: {content['timestamp']}</p>")
            elif section['type'] == 'section':
                html_parts.append(f"<h2>{section['title']}</h2>")
                # Convert markdown to basic HTML
                content_html = section['content'].replace('\n\n', '</p><p>')
                content_html = content_html.replace('**', '<strong>').replace('**', '</strong>')
                html_parts.append(f"<div>{content_html}</div>")
        
        html_parts.extend(["</body>", "</html>"])
        
        return "\n".join(html_parts)
    
    def _sections_to_markdown(self, sections: List[Dict[str, Any]]) -> str:
        """Convert sections to Markdown."""
        md_parts = []
        
        for section in sections:
            if section['type'] == 'title':
                content = section['content']
                md_parts.append(f"# {content['title']}\n")
                md_parts.append(f"*{content['subtitle']}*\n")
                md_parts.append(f"*Generated: {content['timestamp']}*\n")
                md_parts.append(f"*Optimization Method: {content['optimization_method']}*\n")
            elif section['type'] == 'section':
                md_parts.append(f"\n## {section['title']}\n")
                md_parts.append(section['content'])
            
            md_parts.append("\n---\n")
        
        return "\n".join(md_parts)
