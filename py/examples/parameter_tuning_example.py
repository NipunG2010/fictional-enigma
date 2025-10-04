"""
Example script demonstrating HMM parameter tuning.

This script shows how to use the parameter tuning framework
both interactively and programmatically.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imp.tuning import HMMParameterTuner, TuningConfig
from imp.tuning.optimization import (
    GridSearchOptimizer,
    quick_grid_search,
    create_default_param_grid
)
from imp.hmm.trainer import EnhancedHMMTrainer


def generate_synthetic_data(n_samples=500, n_features=3, n_states=3):
    """Generate synthetic HMM data with known regimes."""
    np.random.seed(42)
    
    # Generate data for each state
    samples_per_state = n_samples // n_states
    states_data = []
    
    for state in range(n_states):
        # Each state has different mean and variance
        mean = np.random.randn(n_features) * 2
        cov = np.eye(n_features) * (0.5 + state * 0.3)
        
        data = np.random.multivariate_normal(mean, cov, samples_per_state)
        states_data.append(data)
    
    # Combine and shuffle
    observations = np.vstack(states_data)
    np.random.shuffle(observations)
    
    return observations


def example_programmatic_tuning():
    """Example: Programmatic parameter tuning."""
    print("="*60)
    print("Example 1: Programmatic Parameter Tuning")
    print("="*60)
    
    # Generate data
    observations = generate_synthetic_data(n_samples=300, n_features=2, n_states=3)
    print(f"\nGenerated {len(observations)} observations with {observations.shape[1]} features")
    
    # Define configuration
    config = TuningConfig(
        n_states=3,
        library='hmmlearn',
        covariance_type='full',
        n_iterations=100,
        validation_split=0.2,
        random_state=42
    )
    
    print(f"\nTraining with configuration:")
    print(f"  States: {config.n_states}")
    print(f"  Library: {config.library}")
    print(f"  Covariance: {config.covariance_type}")
    print(f"  Iterations: {config.n_iterations}")
    
    # Train model
    trainer = EnhancedHMMTrainer(
        n_states=config.n_states,
        library=config.library,
        covariance_type=config.covariance_type,
        random_state=config.random_state
    )
    
    artifact, metrics = trainer.train_with_validation(
        observations,
        validation_split=config.validation_split,
        n_iterations=config.n_iterations
    )
    
    print("\n✅ Training completed!")
    print("\nValidation Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    print("\nTraining Metrics:")
    if 'convergence_log_likelihood' in artifact.metadata:
        print(f"  Log-Likelihood: {artifact.metadata['convergence_log_likelihood']:.4f}")
    if 'aic' in artifact.metadata:
        print(f"  AIC: {artifact.metadata['aic']:.4f}")
    if 'bic' in artifact.metadata:
        print(f"  BIC: {artifact.metadata['bic']:.4f}")


def example_grid_search():
    """Example: Grid search optimization."""
    print("\n" + "="*60)
    print("Example 2: Grid Search Optimization")
    print("="*60)
    
    # Generate data
    observations = generate_synthetic_data(n_samples=300, n_features=2, n_states=3)
    print(f"\nGenerated {len(observations)} observations")
    
    # Define parameter grid
    param_grid = {
        'n_states': [2, 3, 4],
        'library': ['hmmlearn'],
        'covariance_type': ['full', 'diag'],
        'random_state': [42]
    }
    
    print(f"\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"\nTotal combinations: {total_combinations}")
    
    # Run grid search
    print("\nRunning grid search...")
    optimizer = GridSearchOptimizer(
        observations=observations,
        param_grid=param_grid,
        scoring_metric='log_likelihood',
        higher_is_better=True,
        n_iterations=50,  # Fewer iterations for demo
        verbose=True
    )
    
    result = optimizer.fit()
    
    print(f"\n{'='*60}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Best parameters: {result.best_params}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")
    
    # Show all results
    print("\nAll Results:")
    for i, res in enumerate(result.all_results):
        if res['score'] is not None:
            params = res['params']
            score = res['score']
            print(f"  {i+1}. States={params['n_states']}, "
                  f"Cov={params['covariance_type']}: {score:.4f}")


def example_quick_grid_search():
    """Example: Quick grid search utility."""
    print("\n" + "="*60)
    print("Example 3: Quick Grid Search")
    print("="*60)
    
    # Generate data
    observations = generate_synthetic_data(n_samples=300, n_features=2, n_states=3)
    print(f"\nGenerated {len(observations)} observations")
    
    # Quick grid search
    print("\nRunning quick grid search...")
    result = quick_grid_search(
        observations,
        n_states_range=[2, 3, 4],
        covariance_types=['full', 'diag'],
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print("QUICK GRID SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Best configuration: {result.best_params}")
    print(f"Best score: {result.best_score:.4f}")


def example_configuration_management():
    """Example: Save and load configurations."""
    print("\n" + "="*60)
    print("Example 4: Configuration Management")
    print("="*60)
    
    # Create temporary directory
    config_dir = Path('./temp_configs')
    config_dir.mkdir(exist_ok=True)
    
    # Generate data
    observations = generate_synthetic_data(n_samples=200, n_features=2, n_states=3)
    
    # Create tuner
    try:
        tuner = HMMParameterTuner(
            observations=observations,
            config_dir=config_dir
        )
        
        print(f"\n✅ Tuner created with config directory: {config_dir}")
        print("\nNote: Use tuner.create_tuning_interface() in a Jupyter notebook")
        print("      to access the interactive widget interface.")
        
        # Example of exporting results (if any exist)
        if len(tuner.results) > 0:
            export_path = config_dir / 'results.json'
            tuner.export_results(export_path)
            print(f"\n✅ Results exported to: {export_path}")
        else:
            print("\nNo results to export yet. Train models first!")
    
    except ImportError as e:
        print(f"\n⚠️ Interactive tuning requires ipywidgets: {e}")
        print("Install with: pip install ipywidgets jupyter")


def example_comparison():
    """Example: Compare multiple configurations."""
    print("\n" + "="*60)
    print("Example 5: Model Comparison")
    print("="*60)
    
    # Generate data
    observations = generate_synthetic_data(n_samples=300, n_features=2, n_states=3)
    print(f"\nGenerated {len(observations)} observations")
    
    # Test multiple configurations
    configs = [
        TuningConfig(n_states=2, covariance_type='full'),
        TuningConfig(n_states=3, covariance_type='full'),
        TuningConfig(n_states=4, covariance_type='full'),
        TuningConfig(n_states=3, covariance_type='diag'),
    ]
    
    print(f"\nTesting {len(configs)} configurations...")
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Training: {config.n_states} states, {config.covariance_type}")
        
        trainer = EnhancedHMMTrainer(
            n_states=config.n_states,
            library=config.library,
            covariance_type=config.covariance_type,
            random_state=config.random_state
        )
        
        artifact, metrics = trainer.train_with_validation(
            observations,
            validation_split=config.validation_split,
            n_iterations=50  # Fewer iterations for demo
        )
        
        results.append({
            'config': config,
            'artifact': artifact,
            'metrics': metrics
        })
        
        val_ll = metrics.get('log_likelihood', 'N/A')
        if isinstance(val_ll, float):
            print(f"  Validation LL: {val_ll:.4f}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    for i, result in enumerate(results):
        config = result['config']
        metrics = result['metrics']
        artifact = result['artifact']
        
        val_ll = metrics.get('log_likelihood', 'N/A')
        train_ll = artifact.metadata.get('convergence_log_likelihood', 'N/A')
        aic = artifact.metadata.get('aic', 'N/A')
        
        print(f"\nConfiguration {i+1}:")
        print(f"  States: {config.n_states}")
        print(f"  Covariance: {config.covariance_type}")
        print(f"  Train LL: {train_ll if train_ll == 'N/A' else f'{train_ll:.4f}'}")
        print(f"  Val LL: {val_ll if val_ll == 'N/A' else f'{val_ll:.4f}'}")
        print(f"  AIC: {aic if aic == 'N/A' else f'{aic:.4f}'}")
    
    # Find best
    best_idx = max(range(len(results)), 
                   key=lambda i: results[i]['metrics'].get('log_likelihood', float('-inf')))
    
    print(f"\n🏆 Best configuration: #{best_idx + 1}")
    print(f"   {results[best_idx]['config'].n_states} states, "
          f"{results[best_idx]['config'].covariance_type} covariance")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("HMM PARAMETER TUNING EXAMPLES")
    print("="*60)
    
    # Run examples
    example_programmatic_tuning()
    example_grid_search()
    example_quick_grid_search()
    example_configuration_management()
    example_comparison()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    print("\nFor interactive tuning, see notebooks/05_parameter_tuning_demo.ipynb")


if __name__ == '__main__':
    main()
