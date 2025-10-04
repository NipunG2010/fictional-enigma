"""
Example demonstrating HMM artifact management and production integration.

This example shows:
1. Creating research artifacts from trained HMM models
2. Tracking experiments with versioning
3. Validating artifacts for production deployment
4. Exporting artifacts for the Rust inference engine
5. Comparing multiple experiments
"""

import numpy as np
from pathlib import Path
import tempfile

from imp.hmm import (
    HMMTrainer,
    HMMArtifact,
    ResearchArtifact,
    ExperimentTracker,
    ArtifactValidator,
    ArtifactExporter
)


def create_sample_data(n_samples: int = 1000, n_features: int = 3) -> np.ndarray:
    """Create sample observation data for HMM training."""
    # Generate synthetic data with 3 regimes
    np.random.seed(42)
    
    # Regime 1: Low volatility, positive trend
    regime1 = np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([0.5, 0.3, 0.2])
    
    # Regime 2: High volatility, negative trend
    regime2 = np.random.randn(n_samples // 3, n_features) * 1.5 + np.array([-0.3, 0.8, -0.2])
    
    # Regime 3: Medium volatility, neutral
    regime3 = np.random.randn(n_samples // 3, n_features) * 1.0 + np.array([0.1, -0.2, 0.4])
    
    # Concatenate regimes
    observations = np.vstack([regime1, regime2, regime3])
    
    # Shuffle to simulate regime transitions
    np.random.shuffle(observations)
    
    return observations


def example_1_basic_artifact_creation():
    """Example 1: Create research artifact from trained HMM."""
    print("=" * 70)
    print("Example 1: Basic Artifact Creation")
    print("=" * 70)
    
    # Create sample data
    observations = create_sample_data(n_samples=1000, n_features=3)
    print(f"Created sample data: {observations.shape}")
    
    # Train HMM model
    trainer = HMMTrainer(n_states=3)
    hmm_artifact = trainer.train(observations, n_iterations=50)
    print(f"Trained HMM with {hmm_artifact.n_states} states")
    
    # Create research artifact
    research_artifact = ResearchArtifact.from_hmm_artifact(
        artifact=hmm_artifact,
        experiment_id="exp_example_001",
        researcher="example_user",
        training_config={
            "n_states": 3,
            "library": "hmmlearn",
            "n_iterations": 50,
            "covariance_type": "full"
        },
        evaluation_metrics={
            "aic": hmm_artifact.metadata.get("aic", 0),
            "bic": hmm_artifact.metadata.get("bic", 0),
            "log_likelihood": hmm_artifact.metadata.get("convergence_log_likelihood", 0)
        },
        notes="Example HMM training for demonstration",
        tags=["example", "tutorial"]
    )
    
    print(f"Created research artifact: {research_artifact.research_metadata.experiment_id}")
    print(f"Artifact version: {research_artifact.artifact_version}")
    print(f"Artifact hash: {research_artifact.artifact_hash[:16]}...")
    print(f"Integrity check: {research_artifact.verify_integrity()}")
    print()


def example_2_experiment_tracking():
    """Example 2: Track multiple experiments."""
    print("=" * 70)
    print("Example 2: Experiment Tracking")
    print("=" * 70)
    
    # Create temporary directory for experiments
    with tempfile.TemporaryDirectory() as temp_dir:
        experiment_dir = Path(temp_dir) / "experiments"
        
        # Initialize experiment tracker
        tracker = ExperimentTracker(experiment_dir)
        print(f"Initialized experiment tracker at: {experiment_dir}")
        
        # Train and log multiple experiments with different configurations
        configs = [
            {"n_states": 2, "n_iterations": 50},
            {"n_states": 3, "n_iterations": 50},
            {"n_states": 4, "n_iterations": 50}
        ]
        
        for i, config in enumerate(configs):
            # Create sample data
            observations = create_sample_data(n_samples=1000, n_features=3)
            
            # Train model
            trainer = HMMTrainer(n_states=config["n_states"])
            hmm_artifact = trainer.train(observations, n_iterations=config["n_iterations"])
            
            # Create research artifact
            research_artifact = ResearchArtifact.from_hmm_artifact(
                artifact=hmm_artifact,
                experiment_id=f"exp_states_{config['n_states']}",
                researcher="example_user",
                training_config=config,
                evaluation_metrics={
                    "aic": hmm_artifact.metadata.get("aic", 0),
                    "bic": hmm_artifact.metadata.get("bic", 0)
                },
                tags=["comparison", "n_states_experiment"]
            )
            
            # Log experiment
            exp_id = tracker.log_experiment(
                research_artifact,
                notes=f"Experiment with {config['n_states']} states"
            )
            print(f"Logged experiment: {exp_id}")
        
        # List all experiments
        print("\nAll experiments:")
        all_experiments = tracker.list_experiments()
        for exp in all_experiments:
            print(f"  - {exp['experiment_id']}: {exp['n_states']} states, "
                  f"AIC={exp.get('notes', 'N/A')}")
        
        # Compare experiments
        print("\nComparing experiments:")
        exp_ids = [f"exp_states_{n}" for n in [2, 3, 4]]
        comparison = tracker.compare_experiments(exp_ids)
        
        print(f"Number of experiments compared: {len(comparison['experiments'])}")
        if "aic" in comparison["best_by_metric"]:
            print(f"Best model by AIC: {comparison['best_by_metric']['aic']}")
        if "bic" in comparison["best_by_metric"]:
            print(f"Best model by BIC: {comparison['best_by_metric']['bic']}")
        
        print()


def example_3_artifact_validation():
    """Example 3: Validate artifacts for production."""
    print("=" * 70)
    print("Example 3: Artifact Validation")
    print("=" * 70)
    
    # Create and train model
    observations = create_sample_data(n_samples=1000, n_features=3)
    trainer = HMMTrainer(n_states=3)
    hmm_artifact = trainer.train(observations, n_iterations=50)
    
    # Create research artifact
    research_artifact = ResearchArtifact.from_hmm_artifact(
        artifact=hmm_artifact,
        experiment_id="exp_validation_001",
        researcher="example_user",
        training_config={"n_states": 3},
        evaluation_metrics={
            "aic": hmm_artifact.metadata.get("aic", 0),
            "bic": hmm_artifact.metadata.get("bic", 0)
        }
    )
    
    # Run validation checks
    print("Running production validation checks...")
    production_checks = ArtifactValidator.validate_for_production(research_artifact)
    
    print("\nProduction Validation Results:")
    for check_name, passed in production_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    # Run Rust compatibility checks
    print("\nRunning Rust compatibility checks...")
    rust_checks = ArtifactValidator.validate_rust_compatibility(research_artifact)
    
    print("\nRust Compatibility Results:")
    for check_name, passed in rust_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    # Run all validations
    print("\nRunning all validations...")
    all_results = ArtifactValidator.run_all_validations(research_artifact)
    
    print(f"\nOverall validation: {'✓ PASSED' if all_results['all_passed'] else '✗ FAILED'}")
    if all_results['failed_checks']:
        print(f"Failed checks: {', '.join(all_results['failed_checks'])}")
    else:
        print("All validation checks passed!")
    
    print()


def example_4_export_for_production():
    """Example 4: Export artifacts for production deployment."""
    print("=" * 70)
    print("Example 4: Export for Production")
    print("=" * 70)
    
    # Create and train model
    observations = create_sample_data(n_samples=1000, n_features=3)
    trainer = HMMTrainer(n_states=3)
    hmm_artifact = trainer.train(observations, n_iterations=50)
    
    # Create research artifact
    research_artifact = ResearchArtifact.from_hmm_artifact(
        artifact=hmm_artifact,
        experiment_id="exp_production_001",
        researcher="example_user",
        training_config={"n_states": 3, "library": "hmmlearn"},
        evaluation_metrics={
            "aic": hmm_artifact.metadata.get("aic", 0),
            "bic": hmm_artifact.metadata.get("bic", 0)
        },
        tags=["production-ready"]
    )
    
    # Mark as production ready
    validation_results = ArtifactValidator.run_all_validations(research_artifact)
    if validation_results["all_passed"]:
        research_artifact.production_ready = True
        research_artifact.validation_status = validation_results["production_checks"]
        print("Artifact validated and marked as production ready")
    
    # Export for production
    with tempfile.TemporaryDirectory() as temp_dir:
        export_dir = Path(temp_dir) / "production_export"
        
        print(f"\nExporting to: {export_dir}")
        exported_files = ArtifactExporter.export_for_production(
            research_artifact,
            export_dir,
            validate=True
        )
        
        print("\nExported files:")
        for file_type, file_path in exported_files.items():
            print(f"  - {file_type}: {file_path.name}")
        
        # Show manifest content
        manifest_path = exported_files.get("manifest")
        if manifest_path and manifest_path.exists():
            import json
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            print("\nDeployment Manifest:")
            print(f"  Experiment ID: {manifest['experiment_id']}")
            print(f"  Artifact Version: {manifest['artifact_version']}")
            print(f"  Number of States: {manifest['n_states']}")
            print(f"  Library: {manifest['library']}")
            print(f"  Production Ready: {manifest['production_ready']}")
    
    print()


def example_5_complete_workflow():
    """Example 5: Complete research-to-production workflow."""
    print("=" * 70)
    print("Example 5: Complete Research-to-Production Workflow")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        experiment_dir = Path(temp_dir) / "experiments"
        export_dir = Path(temp_dir) / "production"
        
        # Step 1: Initialize tracker
        tracker = ExperimentTracker(experiment_dir)
        print("Step 1: Initialized experiment tracker")
        
        # Step 2: Train multiple models
        print("\nStep 2: Training multiple models...")
        best_artifact = None
        best_aic = float('inf')
        
        for n_states in [2, 3, 4]:
            observations = create_sample_data(n_samples=1000, n_features=3)
            trainer = HMMTrainer(n_states=n_states)
            hmm_artifact = trainer.train(observations, n_iterations=50)
            
            aic = hmm_artifact.metadata.get("aic", float('inf'))
            print(f"  Trained model with {n_states} states: AIC={aic:.2f}")
            
            # Create and log research artifact
            research_artifact = ResearchArtifact.from_hmm_artifact(
                artifact=hmm_artifact,
                experiment_id=f"exp_workflow_{n_states}states",
                researcher="workflow_user",
                training_config={"n_states": n_states},
                evaluation_metrics={
                    "aic": aic,
                    "bic": hmm_artifact.metadata.get("bic", 0)
                }
            )
            
            tracker.log_experiment(research_artifact)
            
            # Track best model
            if aic < best_aic:
                best_aic = aic
                best_artifact = research_artifact
        
        # Step 3: Select best model
        print(f"\nStep 3: Selected best model: {best_artifact.research_metadata.experiment_id}")
        print(f"  Best AIC: {best_aic:.2f}")
        
        # Step 4: Validate best model
        print("\nStep 4: Validating best model...")
        validation_results = ArtifactValidator.run_all_validations(best_artifact)
        
        if validation_results["all_passed"]:
            print("  ✓ All validation checks passed")
            best_artifact.production_ready = True
        else:
            print(f"  ✗ Validation failed: {validation_results['failed_checks']}")
        
        # Step 5: Export for production
        if best_artifact.production_ready:
            print("\nStep 5: Exporting for production...")
            exported_files = ArtifactExporter.export_for_production(
                best_artifact,
                export_dir,
                validate=True
            )
            print(f"  Exported {len(exported_files)} files to {export_dir}")
            print("  ✓ Ready for deployment to Rust inference engine")
        
        # Step 6: Compare all experiments
        print("\nStep 6: Experiment comparison summary:")
        all_experiments = tracker.list_experiments()
        print(f"  Total experiments: {len(all_experiments)}")
        
        exp_ids = [exp['experiment_id'] for exp in all_experiments]
        comparison = tracker.compare_experiments(exp_ids)
        
        if "aic" in comparison["best_by_metric"]:
            print(f"  Best by AIC: {comparison['best_by_metric']['aic']}")
    
    print("\n✓ Complete workflow finished successfully!")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("HMM Artifact Management Examples")
    print("=" * 70 + "\n")
    
    # Run examples
    example_1_basic_artifact_creation()
    example_2_experiment_tracking()
    example_3_artifact_validation()
    example_4_export_for_production()
    example_5_complete_workflow()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
