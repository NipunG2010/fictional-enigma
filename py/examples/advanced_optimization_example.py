"""
Example demonstrating advanced hyperparameter optimization and model selection.

This example shows how to:
1. Use automated model selection with multiple criteria
2. Perform sensitivity analysis on hyperparameters
3. Create and evaluate model ensembles
4. Generate comprehensive reports
5. Track model performance over time
"""

import numpy as np
from pathlib import Path
import logging

from imp.tuning import (
    AutomatedModelSelector,
    SelectionCriteria,
    EnsembleEvaluator,
    SensitivityAnalyzer,
    ReportGenerator,
    ReportConfig,
    PerformanceTracker
)
from imp.hmm.trainer import EnhancedHMMTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=500, n_features=3, n_regimes=3):
    """Generate synthetic market data with regime changes."""
    np.random.seed(42)
    
    observations = []
    regime_lengths = [n_samples // n_regimes] * n_regimes
    
    for regime_idx, length in enumerate(regime_lengths):
        # Different volatility and mean for each regime
        volatility = 0.5 + regime_idx * 0.5
        mean_shift = regime_idx * 0.3
        
        for _ in range(length):
            obs = np.random.randn(n_features) * volatility + mean_shift
            observations.append(obs)
    
    return np.array(observations)


def example_automated_model_selection():
    """Example 1: Automated model selection with multiple criteria."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Automated Model Selection")
    print("="*70 + "\n")
    
    # Generate data
    observations = generate_synthetic_data(n_samples=300)
    logger.info(f"Generated {len(observations)} observations with {observations.shape[1]} features")
    
    # Define selection criteria
    criteria = [
        SelectionCriteria(metric_name='bic', weight=0.4, higher_is_better=False),
        SelectionCriteria(metric_name='log_likelihood', weight=0.3, higher_is_better=True),
        SelectionCriteria(metric_name='stability_score', weight=0.3, higher_is_better=True)
    ]
    
    # Create selector
    selector = AutomatedModelSelector(
        observations=observations,
        selection_criteria=criteria,
        random_state=42
    )
    
    # Define parameter grid
    param_grid = {
        'n_states': [2, 3, 4],
        'library': ['hmmlearn'],
        'covariance_type': ['diag', 'full'],
        'random_state': [42]
    }
    
    # Run model selection
    logger.info("Running grid search optimization...")
    result = selector.select_best_model(
        optimization_method='grid_search',
        param_grid=param_grid,
        cv_folds=3,
        n_iterations=100,
        verbose=True
    )
    
    # Print results
    print("\n" + "-"*70)
    print("BEST MODEL CONFIGURATION:")
    print("-"*70)
    print(f"States: {result.best_config['n_states']}")
    print(f"Library: {result.best_config['library']}")
    print(f"Covariance: {result.best_config['covariance_type']}")
    print(f"Selection Score: {result.best_score:.4f}")
    
    print("\n" + "-"*70)
    print("TOP 5 MODELS:")
    print("-"*70)
    top_5 = result.all_comparisons.nsmallest(5, 'rank')
    print(top_5[['rank', 'config', 'log_likelihood', 'bic']].to_string(index=False))
    
    # Print selection report
    print("\n" + selector.get_selection_report())
    
    return result, observations


def example_sensitivity_analysis(observations):
    """Example 2: Hyperparameter sensitivity analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Hyperparameter Sensitivity Analysis")
    print("="*70 + "\n")
    
    # Define baseline configuration
    baseline_config = {
        'n_states': 3,
        'library': 'hmmlearn',
        'covariance_type': 'diag',
        'random_state': 42
    }
    
    # Create analyzer
    analyzer = SensitivityAnalyzer(
        observations=observations,
        baseline_config=baseline_config,
        metric_name='log_likelihood',
        n_iterations=100
    )
    
    # Analyze multiple parameters
    param_ranges = {
        'n_states': [2, 3, 4, 5],
        'covariance_type': ['diag', 'spherical', 'full']
    }
    
    logger.info("Analyzing parameter sensitivity...")
    results = analyzer.analyze_all_parameters(param_ranges, verbose=True)
    
    # Get sensitivity ranking
    ranking = analyzer.get_sensitivity_ranking()
    
    print("\n" + "-"*70)
    print("PARAMETER SENSITIVITY RANKING:")
    print("-"*70)
    print(ranking.to_string(index=False))
    
    # Print sensitivity report
    print("\n" + analyzer.get_sensitivity_report())
    
    # Save results
    output_dir = Path("./optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    analyzer.save_results(output_dir / "sensitivity_analysis.json")
    logger.info(f"Sensitivity results saved to {output_dir}")
    
    # Generate plots
    try:
        analyzer.plot_all_sensitivities(save_path=output_dir / "sensitivity_plots.png")
        logger.info("Sensitivity plots saved")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return analyzer


def example_ensemble_evaluation(observations):
    """Example 3: Ensemble model evaluation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Ensemble Model Evaluation")
    print("="*70 + "\n")
    
    # Create ensemble evaluator
    evaluator = EnsembleEvaluator(random_state=42)
    
    # Define diverse model configurations
    configs = [
        {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        {'n_states': 4, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'full', 'random_state': 42}
    ]
    
    # Create ensemble with performance-based weighting
    logger.info("Creating ensemble with performance-based weighting...")
    result = evaluator.create_ensemble(
        observations=observations,
        configs=configs,
        n_iterations=100,
        weighting_strategy='performance',
        validation_split=0.2
    )
    
    print("\n" + "-"*70)
    print("ENSEMBLE COMPOSITION:")
    print("-"*70)
    for i, member in enumerate(result.members):
        print(f"\nMember {i+1}:")
        print(f"  States: {member.config['n_states']}")
        print(f"  Covariance: {member.config['covariance_type']}")
        print(f"  Weight: {member.weight:.4f}")
        print(f"  Log-Likelihood: {member.performance_metrics.get('log_likelihood', 'N/A'):.4f}")
    
    print("\n" + "-"*70)
    print("ENSEMBLE PERFORMANCE:")
    print("-"*70)
    for metric, value in result.ensemble_performance.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print("\n" + "-"*70)
    print("DIVERSITY METRICS:")
    print("-"*70)
    for metric, value in result.diversity_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Compare ensemble vs individual models
    comparison = evaluator.compare_ensemble_vs_individual(result, observations)
    
    print("\n" + "-"*70)
    print("ENSEMBLE VS INDIVIDUAL COMPARISON:")
    print("-"*70)
    print(comparison.to_string(index=False))
    
    # Print ensemble report
    print("\n" + evaluator.get_ensemble_report(result))
    
    return result


def example_report_generation(selection_result, sensitivity_analyzer, ensemble_result):
    """Example 4: Automated report generation."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Automated Report Generation")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path("./optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create report generator
    generator = ReportGenerator(output_dir=output_dir)
    
    # Configure report
    config = ReportConfig(
        include_sensitivity=True,
        include_ensemble=True,
        include_cv_details=True,
        include_visualizations=True,
        output_format='markdown'
    )
    
    # Generate comprehensive report
    logger.info("Generating comprehensive report...")
    report_path = generator.generate_full_report(
        selection_result=selection_result,
        sensitivity_analyzer=sensitivity_analyzer,
        ensemble_result=ensemble_result,
        config=config
    )
    
    print(f"\n✅ Report generated: {report_path}")
    print(f"📊 Figures saved to: {generator.figures_dir}")
    
    return report_path


def example_performance_tracking(observations):
    """Example 5: Model performance tracking and regression detection."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Performance Tracking and Regression Detection")
    print("="*70 + "\n")
    
    # Create tracking directory
    tracking_dir = Path("./performance_tracking")
    tracking_dir.mkdir(exist_ok=True)
    
    # Create performance tracker
    tracker = PerformanceTracker(
        tracking_dir=tracking_dir,
        baseline_window=5,
        warning_threshold=0.05,
        critical_threshold=0.10
    )
    
    # Simulate model evolution over time
    logger.info("Simulating model performance over time...")
    
    for version in range(1, 8):
        # Train model (with slight degradation over time)
        data_size = max(200, len(observations) - version * 20)
        train_data = observations[:data_size]
        
        trainer = EnhancedHMMTrainer(
            n_states=3,
            library='hmmlearn',
            covariance_type='diag',
            random_state=42
        )
        artifact = trainer.train(train_data, n_iterations=100)
        
        # Record performance
        snapshot = tracker.record_performance(
            model_id='production_model',
            model_version=f'v1.{version}',
            artifact=artifact,
            observations=train_data,
            metadata={'data_size': data_size}
        )
        
        print(f"\nVersion v1.{version}:")
        print(f"  Data Size: {data_size}")
        print(f"  Log-Likelihood: {snapshot.metrics.get('log_likelihood', 'N/A'):.4f}")
        print(f"  BIC: {snapshot.metrics.get('bic', 'N/A'):.4f}")
    
    # Get performance history
    history = tracker.get_performance_history('production_model')
    
    print("\n" + "-"*70)
    print("PERFORMANCE HISTORY:")
    print("-"*70)
    print(history[['timestamp', 'model_version', 'log_likelihood', 'bic']].to_string(index=False))
    
    # Get alert history
    alerts = tracker.get_alert_history(model_id='production_model')
    
    if not alerts.empty:
        print("\n" + "-"*70)
        print("REGRESSION ALERTS:")
        print("-"*70)
        print(alerts[['timestamp', 'severity', 'metric_name', 'message']].to_string(index=False))
    else:
        print("\n✅ No performance regressions detected")
    
    # Generate monitoring report
    print("\n" + tracker.generate_monitoring_report('production_model'))
    
    # Plot performance trend
    try:
        tracker.plot_performance_trend(
            model_id='production_model',
            metric_name='log_likelihood',
            save_path=tracking_dir / 'performance_trend.png'
        )
        logger.info(f"Performance trend plot saved to {tracking_dir}")
    except Exception as e:
        logger.warning(f"Could not generate trend plot: {e}")
    
    return tracker


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ADVANCED HYPERPARAMETER OPTIMIZATION EXAMPLES")
    print("="*70)
    
    # Example 1: Automated model selection
    selection_result, observations = example_automated_model_selection()
    
    # Example 2: Sensitivity analysis
    sensitivity_analyzer = example_sensitivity_analysis(observations)
    
    # Example 3: Ensemble evaluation
    ensemble_result = example_ensemble_evaluation(observations)
    
    # Example 4: Report generation
    report_path = example_report_generation(
        selection_result,
        sensitivity_analyzer,
        ensemble_result
    )
    
    # Example 5: Performance tracking
    tracker = example_performance_tracking(observations)
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 Results saved to: ./optimization_results/")
    print(f"📁 Tracking data saved to: ./performance_tracking/")
    print(f"\n📄 Comprehensive report: {report_path}")


if __name__ == '__main__':
    main()
