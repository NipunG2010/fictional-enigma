"""
Example demonstrating comprehensive HMM model evaluation and comparison.

This example shows how to:
1. Evaluate individual HMM models
2. Compare multiple model configurations
3. Perform cross-validation
4. Analyze regime stability
5. Test statistical significance
6. Select the best model based on multiple criteria
"""

import numpy as np
import pandas as pd
from imp.evaluation import HMMEvaluator
from imp.hmm.trainer import EnhancedHMMTrainer


def generate_synthetic_data(n_samples: int = 500, n_features: int = 3, random_state: int = 42):
    """Generate synthetic data with regime changes."""
    np.random.seed(random_state)
    
    observations = []
    true_states = []
    
    for i in range(n_samples):
        if i < n_samples // 3:
            # Regime 1: low volatility, negative trend
            obs = np.random.randn(n_features) * 0.5 - 0.2
            state = 0
        elif i < 2 * n_samples // 3:
            # Regime 2: high volatility, no trend
            obs = np.random.randn(n_features) * 2.0
            state = 1
        else:
            # Regime 3: medium volatility, positive trend
            obs = np.random.randn(n_features) * 1.0 + 0.3
            state = 2
        
        observations.append(obs)
        true_states.append(state)
    
    return np.array(observations), np.array(true_states)


def example_single_model_evaluation():
    """Example: Evaluate a single HMM model."""
    print("=" * 60)
    print("Example 1: Single Model Evaluation")
    print("=" * 60)
    
    # Generate data
    observations, _ = generate_synthetic_data(n_samples=300)
    
    # Train model
    print("\nTraining HMM model...")
    trainer = EnhancedHMMTrainer(
        n_states=3,
        library="hmmlearn",
        covariance_type="diag",
        random_state=42
    )
    artifact = trainer.train(observations, n_iterations=100)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluator = HMMEvaluator(random_state=42)
    metrics = evaluator.evaluate_model(trainer.trainer, observations)
    
    print(f"\nEvaluation Metrics:")
    print(f"  Log-likelihood: {metrics.log_likelihood:.4f}")
    print(f"  AIC: {metrics.aic:.4f}")
    print(f"  BIC: {metrics.bic:.4f}")
    print(f"  Perplexity: {metrics.perplexity:.4f}")
    print(f"  Number of parameters: {metrics.n_parameters}")
    print(f"  Number of samples: {metrics.n_samples}")


def example_cross_validation():
    """Example: Perform cross-validation."""
    print("\n" + "=" * 60)
    print("Example 2: Cross-Validation")
    print("=" * 60)
    
    # Generate data
    observations, _ = generate_synthetic_data(n_samples=300)
    
    # Configure model
    trainer_config = {
        'n_states': 3,
        'library': 'hmmlearn',
        'covariance_type': 'diag',
        'random_state': 42
    }
    
    # Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    evaluator = HMMEvaluator(random_state=42)
    cv_results = evaluator.cross_validate(
        observations,
        trainer_config,
        cv_folds=5,
        n_iterations=100
    )
    
    print(f"\nCross-Validation Results:")
    print(f"  Log-likelihood: {cv_results['log_likelihood_mean']:.4f} ± {cv_results['log_likelihood_std']:.4f}")
    print(f"  AIC: {cv_results['aic_mean']:.4f} ± {cv_results['aic_std']:.4f}")
    print(f"  BIC: {cv_results['bic_mean']:.4f} ± {cv_results['bic_std']:.4f}")
    print(f"  Perplexity: {cv_results['perplexity_mean']:.4f} ± {cv_results['perplexity_std']:.4f}")


def example_regime_stability():
    """Example: Analyze regime stability."""
    print("\n" + "=" * 60)
    print("Example 3: Regime Stability Analysis")
    print("=" * 60)
    
    # Generate data
    observations, _ = generate_synthetic_data(n_samples=300)
    
    # Train model
    print("\nTraining HMM model...")
    trainer = EnhancedHMMTrainer(
        n_states=3,
        library="hmmlearn",
        covariance_type="diag",
        random_state=42
    )
    trainer.train(observations, n_iterations=100)
    
    # Get state probabilities
    state_probs = trainer.trainer.predict_state_probabilities(observations)
    
    # Analyze stability
    print("\nAnalyzing regime stability...")
    evaluator = HMMEvaluator(random_state=42)
    stability = evaluator.regime_stability_analysis(state_probs, min_duration=10)
    
    print(f"\nRegime Stability Metrics:")
    for state in range(3):
        print(f"\n  State {state}:")
        print(f"    Mean duration: {stability.mean_durations[state]:.2f} periods")
        print(f"    Median duration: {stability.median_durations[state]:.2f} periods")
        print(f"    Max duration: {stability.max_durations[state]} periods")
        print(f"    Stable periods (≥10): {stability.stable_periods[state]}")
        print(f"    Total periods: {stability.total_periods[state]}")
        print(f"    Persistence: {stability.state_persistence[state]:.2%}")
    
    print(f"\n  Transition Frequencies:")
    for (from_state, to_state), count in stability.transition_frequencies.items():
        print(f"    State {from_state} → State {to_state}: {count} transitions")


def example_model_comparison():
    """Example: Compare multiple model configurations."""
    print("\n" + "=" * 60)
    print("Example 4: Model Comparison")
    print("=" * 60)
    
    # Generate data
    observations, _ = generate_synthetic_data(n_samples=300)
    
    # Define configurations to compare
    trainer_configs = [
        {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        {'n_states': 4, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'full', 'random_state': 42},
        {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'spherical', 'random_state': 42},
    ]
    
    # Compare models
    print("\nComparing model configurations...")
    evaluator = HMMEvaluator(random_state=42)
    comparison_df = evaluator.compare_models(
        observations,
        trainer_configs,
        n_iterations=100,
        perform_cv=True,
        cv_folds=3,
        analyze_stability=True
    )
    
    print("\nModel Comparison Results:")
    print(comparison_df.to_string())
    
    # Get summary
    summary = evaluator.get_evaluation_summary()
    print(f"\n\nEvaluation Summary:")
    print(f"  Number of configurations: {summary['n_configurations']}")
    print(f"  Best by BIC: {summary['best_by_bic']}")
    print(f"  Best by AIC: {summary['best_by_aic']}")
    print(f"  Best by log-likelihood: {summary['best_by_likelihood']}")


def example_statistical_significance():
    """Example: Test statistical significance between models."""
    print("\n" + "=" * 60)
    print("Example 5: Statistical Significance Testing")
    print("=" * 60)
    
    # Generate data
    observations, _ = generate_synthetic_data(n_samples=300)
    
    # Define configurations
    trainer_configs = [
        {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
    ]
    
    # Compare models with CV
    print("\nComparing models with cross-validation...")
    evaluator = HMMEvaluator(random_state=42)
    comparison_df = evaluator.compare_models(
        observations,
        trainer_configs,
        n_iterations=100,
        perform_cv=True,
        cv_folds=5,
        analyze_stability=False
    )
    
    # Get config names
    config_names = comparison_df['config'].tolist()
    
    # Test significance
    print(f"\nTesting significance between {config_names[0]} and {config_names[1]}...")
    sig_result = evaluator.statistical_significance_test(
        config_names[0],
        config_names[1],
        metric='log_likelihood'
    )
    
    print(f"\nStatistical Significance Test Results:")
    print(f"  Metric: {sig_result['metric']}")
    print(f"  t-statistic: {sig_result['t_statistic']:.4f}")
    print(f"  p-value: {sig_result['p_value']:.4f}")
    print(f"  Significant (α=0.05): {sig_result['significant']}")
    print(f"  Cohen's d: {sig_result['cohens_d']:.4f}")
    print(f"\n  {config_names[0]}:")
    print(f"    Mean: {sig_result['config1_mean']:.4f}")
    print(f"    Std: {sig_result['config1_std']:.4f}")
    print(f"\n  {config_names[1]}:")
    print(f"    Mean: {sig_result['config2_mean']:.4f}")
    print(f"    Std: {sig_result['config2_std']:.4f}")


def example_model_selection():
    """Example: Select best model based on multiple criteria."""
    print("\n" + "=" * 60)
    print("Example 6: Model Selection")
    print("=" * 60)
    
    # Generate data
    observations, _ = generate_synthetic_data(n_samples=300)
    
    # Define configurations
    trainer_configs = [
        {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
        {'n_states': 4, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
    ]
    
    # Compare models
    print("\nComparing models...")
    evaluator = HMMEvaluator(random_state=42)
    evaluator.compare_models(
        observations,
        trainer_configs,
        n_iterations=100,
        perform_cv=False,
        analyze_stability=False
    )
    
    # Select best by single criterion
    print("\nSelecting best model by BIC...")
    best_bic = evaluator.select_best_model(criteria=['bic'])
    print(f"  Best model: {best_bic}")
    
    # Select best by multiple criteria
    print("\nSelecting best model by weighted criteria (60% BIC, 40% log-likelihood)...")
    best_weighted = evaluator.select_best_model(
        criteria=['bic', 'log_likelihood'],
        weights=[0.6, 0.4]
    )
    print(f"  Best model: {best_weighted}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HMM Model Evaluation Framework Examples")
    print("=" * 60)
    
    # Run examples
    example_single_model_evaluation()
    example_cross_validation()
    example_regime_stability()
    example_model_comparison()
    example_statistical_significance()
    example_model_selection()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
