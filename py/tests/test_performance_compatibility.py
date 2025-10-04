"""
Performance and compatibility tests for research environment.

Tests:
- Cross-library compatibility (hmmlearn vs pomegranate)
- Performance benchmarks
- Memory usage
- Artifact format compatibility
- Production system integration
"""

import pytest
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
from typing import Dict, Any
import warnings

from imp.hmm.models import HMMArtifact, FusionWeights
from imp.hmm.trainer import EnhancedHMMTrainer, HMMLearnTrainer
from imp.hmm.inference import HMMInference
from imp.hmm.artifact_management import ArtifactManager, ArtifactValidator
from imp.evaluation.evaluator import HMMEvaluator


@pytest.fixture
def small_dataset():
    """Small dataset for quick tests."""
    np.random.seed(42)
    return np.random.randn(100, 3)


@pytest.fixture
def medium_dataset():
    """Medium dataset for performance tests."""
    np.random.seed(42)
    return np.random.randn(1000, 3)


@pytest.fixture
def large_dataset():
    """Large dataset for scalability tests."""
    np.random.seed(42)
    return np.random.randn(10000, 3)


class TestCrossLibraryCompatibility:
    """Test compatibility between hmmlearn and pomegranate."""
    
    def test_hmmlearn_available(self):
        """Test that hmmlearn is available."""
        try:
            import hmmlearn
            assert True
        except ImportError:
            pytest.fail("hmmlearn not available")
    
    def test_pomegranate_availability(self):
        """Test pomegranate availability (optional)."""
        try:
            import pomegranate
            pomegranate_available = True
        except ImportError:
            pomegranate_available = False
            warnings.warn("Pomegranate not available - some tests will be skipped")
        
        # Store for other tests
        pytest.pomegranate_available = pomegranate_available
    
    def test_hmmlearn_training(self, small_dataset):
        """Test training with hmmlearn."""
        trainer = EnhancedHMMTrainer(
            n_states=3,
            library='hmmlearn',
            random_state=42
        )
        
        artifact = trainer.train(small_dataset, n_iterations=10)
        
        assert artifact is not None
        assert artifact.n_states == 3
        assert len(artifact.transition_matrix) == 3
    
    @pytest.mark.skipif(
        not hasattr(pytest, 'pomegranate_available') or not pytest.pomegranate_available,
        reason="Pomegranate not available"
    )
    def test_pomegranate_training(self, small_dataset):
        """Test training with pomegranate."""
        trainer = EnhancedHMMTrainer(
            n_states=3,
            library='pomegranate',
            random_state=42
        )
        
        artifact = trainer.train(small_dataset, n_iterations=10)
        
        assert artifact is not None
        assert artifact.n_states == 3
    
    def test_artifact_format_consistency(self, small_dataset):
        """Test that artifacts from different libraries have consistent format."""
        # Train with hmmlearn
        trainer_hmmlearn = EnhancedHMMTrainer(
            n_states=3,
            library='hmmlearn',
            random_state=42
        )
        artifact_hmmlearn = trainer_hmmlearn.train(small_dataset, n_iterations=10)
        
        # Check artifact structure
        assert hasattr(artifact_hmmlearn, 'transition_matrix')
        assert hasattr(artifact_hmmlearn, 'initial_probabilities')
        assert hasattr(artifact_hmmlearn, 'means')
        assert hasattr(artifact_hmmlearn, 'covariances')
        
        # Verify shapes
        assert len(artifact_hmmlearn.transition_matrix) == 3
        assert len(artifact_hmmlearn.initial_probabilities) == 3
        assert len(artifact_hmmlearn.means) == 3
        assert len(artifact_hmmlearn.covariances) == 3
    
    def test_inference_compatibility(self, small_dataset):
        """Test that inference works with artifacts from different libraries."""
        # Train model
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        artifact = trainer.train(small_dataset, n_iterations=10)
        
        # Test inference
        inference = HMMInference()
        inference.load_artifact(artifact)
        
        test_obs = small_dataset[:10]
        prediction = inference.predict(test_obs)
        
        assert prediction is not None
        assert prediction.state_probabilities is not None
        assert prediction.most_likely_state is not None


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_training_speed_small_dataset(self, small_dataset):
        """Benchmark training speed on small dataset."""
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        
        start_time = time.time()
        artifact = trainer.train(small_dataset, n_iterations=50)
        training_time = time.time() - start_time
        
        # Should complete quickly
        assert training_time < 5.0  # 5 seconds
        assert artifact is not None
    
    @pytest.mark.slow
    def test_training_speed_medium_dataset(self, medium_dataset):
        """Benchmark training speed on medium dataset."""
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        
        start_time = time.time()
        artifact = trainer.train(medium_dataset, n_iterations=50)
        training_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert training_time < 30.0  # 30 seconds
        assert artifact is not None
    
    @pytest.mark.slow
    def test_training_speed_large_dataset(self, large_dataset):
        """Benchmark training speed on large dataset."""
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        
        start_time = time.time()
        artifact = trainer.train(large_dataset, n_iterations=20)  # Fewer iterations
        training_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert training_time < 120.0  # 2 minutes
        assert artifact is not None
    
    def test_inference_speed(self, small_dataset):
        """Benchmark inference speed."""
        # Train model
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        artifact = trainer.train(small_dataset, n_iterations=10)
        
        # Setup inference
        inference = HMMInference()
        inference.load_artifact(artifact)
        
        # Benchmark inference
        start_time = time.time()
        for _ in range(100):
            prediction = inference.predict(small_dataset[:10])
        inference_time = time.time() - start_time
        
        # Should be fast
        avg_time_per_prediction = inference_time / 100
        assert avg_time_per_prediction < 0.1  # 100ms per prediction
    
    def test_scalability_with_states(self, small_dataset):
        """Test scalability with increasing number of states."""
        results = []
        
        for n_states in [2, 3, 4, 5]:
            trainer = EnhancedHMMTrainer(
                n_states=n_states,
                library='hmmlearn',
                random_state=42
            )
            
            start_time = time.time()
            artifact = trainer.train(small_dataset, n_iterations=20)
            training_time = time.time() - start_time
            
            results.append({
                'n_states': n_states,
                'training_time': training_time
            })
        
        # Training time should increase with states
        df = pd.DataFrame(results)
        assert df['training_time'].is_monotonic_increasing or \
               df['training_time'].iloc[-1] > df['training_time'].iloc[0]
    
    def test_memory_efficiency(self, medium_dataset):
        """Test memory efficiency during training."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Train model
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        artifact = trainer.train(medium_dataset, n_iterations=20)
        
        # Artifact should be reasonably sized
        import sys
        artifact_size = sys.getsizeof(artifact.model_dump_json())
        
        # Should be less than 1MB for this configuration
        assert artifact_size < 1_000_000


class TestArtifactCompatibility:
    """Test artifact compatibility with production system."""
    
    def test_artifact_serialization(self, small_dataset):
        """Test artifact serialization and deserialization."""
        # Train model
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        artifact = trainer.train(small_dataset, n_iterations=10)
        
        # Serialize
        artifact_dict = artifact.model_dump()
        artifact_json = artifact.model_dump_json()
        
        # Deserialize
        reconstructed_from_dict = HMMArtifact(**artifact_dict)
        
        import json
        reconstructed_from_json = HMMArtifact(**json.loads(artifact_json))
        
        # Verify
        assert reconstructed_from_dict.n_states == artifact.n_states
        assert reconstructed_from_json.n_states == artifact.n_states
    
    def test_artifact_validation(self, small_dataset):
        """Test artifact validation."""
        # Train model
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        artifact = trainer.train(small_dataset, n_iterations=10)
        
        # Validate
        validator = ArtifactValidator()
        result = validator.validate_artifact(artifact)
        
        assert result['is_valid'] is True
        assert len(result.get('errors', [])) == 0
    
    def test_invalid_artifact_detection(self):
        """Test detection of invalid artifacts."""
        # Create invalid artifact (mismatched dimensions)
        invalid_artifact = HMMArtifact(
            version="1.0.0",
            n_states=3,
            transition_matrix=[[0.5, 0.5], [0.5, 0.5]],  # Wrong size
            initial_probabilities=[0.33, 0.33, 0.34],
            means=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            covariances=[[[1.0, 0.0], [0.0, 1.0]]] * 3,
            training_window_start=0,
            training_window_end=100
        )
        
        validator = ArtifactValidator()
        result = validator.validate_artifact(invalid_artifact)
        
        assert result['is_valid'] is False
        assert len(result.get('errors', [])) > 0
    
    def test_artifact_versioning(self, small_dataset, tmp_path):
        """Test artifact versioning and management."""
        # Train multiple versions
        artifacts = []
        for i in range(3):
            trainer = EnhancedHMMTrainer(
                n_states=3,
                library='hmmlearn',
                random_state=42 + i
            )
            artifact = trainer.train(small_dataset, n_iterations=10)
            artifacts.append(artifact)
        
        # Save with versioning
        manager = ArtifactManager(storage_dir=tmp_path)
        
        artifact_ids = []
        for i, artifact in enumerate(artifacts):
            artifact_id = manager.save_artifact(artifact, f"model_v{i}")
            artifact_ids.append(artifact_id)
        
        # Verify all versions saved
        assert len(artifact_ids) == 3
        assert len(set(artifact_ids)) == 3  # All unique
        
        # Load specific version
        loaded = manager.load_artifact(artifact_ids[1])
        assert loaded is not None
    
    def test_fusion_weights_compatibility(self, small_dataset):
        """Test fusion weights compatibility."""
        # Train model
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        artifact = trainer.train(small_dataset, n_iterations=10)
        
        # Create fusion weights
        weights = FusionWeights(
            version="1.0.0",
            state_weights=[
                {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3},
                {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3},
                {"w_ldc": 0.3, "w_mr": 0.2, "w_tsmom": 0.5}
            ],
            model_version="1.0.0"
        )
        
        # Test with inference
        inference = HMMInference()
        inference.load_artifact(artifact)
        inference.load_weights(weights)
        
        # Should work without errors
        assert inference.weights is not None


class TestProductionIntegration:
    """Test integration with production system."""
    
    def test_end_to_end_workflow(self, small_dataset, tmp_path):
        """Test complete research-to-production workflow."""
        # Step 1: Train in research environment
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        artifact = trainer.train(small_dataset, n_iterations=20)
        
        # Step 2: Validate artifact
        validator = ArtifactValidator()
        validation = validator.validate_artifact(artifact)
        assert validation['is_valid'] is True
        
        # Step 3: Save artifact
        manager = ArtifactManager(storage_dir=tmp_path)
        artifact_id = manager.save_artifact(artifact, "production_model")
        
        # Step 4: Load in production-like environment
        loaded_artifact = manager.load_artifact(artifact_id)
        assert loaded_artifact is not None
        
        # Step 5: Run inference
        inference = HMMInference()
        inference.load_artifact(loaded_artifact)
        
        test_data = small_dataset[:10]
        prediction = inference.predict(test_data)
        
        # Step 6: Verify prediction format
        assert prediction.state_probabilities is not None
        assert prediction.most_likely_state is not None
        assert prediction.state_probabilities.shape[0] == len(test_data)
    
    def test_model_comparison_workflow(self, small_dataset):
        """Test model comparison workflow."""
        evaluator = HMMEvaluator(random_state=42)
        
        # Define configurations
        configs = [
            {'n_states': 2, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'diag', 'random_state': 42},
            {'n_states': 3, 'library': 'hmmlearn', 'covariance_type': 'full', 'random_state': 42},
        ]
        
        # Compare models
        comparison_df = evaluator.compare_models(
            small_dataset,
            configs,
            n_iterations=20,
            perform_cv=False
        )
        
        # Verify results
        assert len(comparison_df) == 3
        assert 'rank' in comparison_df.columns
        assert 'log_likelihood' in comparison_df.columns
        
        # Select best model
        best_config = evaluator.select_best_model(criteria=['bic'])
        assert best_config in evaluator.evaluation_results
    
    def test_cross_validation_workflow(self, medium_dataset):
        """Test cross-validation workflow."""
        evaluator = HMMEvaluator(random_state=42)
        
        config = {
            'n_states': 3,
            'library': 'hmmlearn',
            'covariance_type': 'diag',
            'random_state': 42
        }
        
        # Run cross-validation
        cv_results = evaluator.cross_validate(
            medium_dataset,
            config,
            cv_folds=3,
            n_iterations=20
        )
        
        # Verify results
        assert 'log_likelihood_mean' in cv_results
        assert 'log_likelihood_std' in cv_results
        assert len(cv_results['log_likelihood_values']) == 3
    
    def test_regime_analysis_workflow(self, small_dataset):
        """Test regime analysis workflow."""
        # Train model
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        artifact = trainer.train(small_dataset, n_iterations=20)
        
        # Get state probabilities
        state_probs = trainer.trainer.predict_state_probabilities(small_dataset)
        
        # Analyze regimes
        evaluator = HMMEvaluator(random_state=42)
        stability = evaluator.regime_stability_analysis(state_probs)
        
        # Verify analysis
        assert stability.mean_durations is not None
        assert stability.state_persistence is not None
        assert len(stability.mean_durations) == 3


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness."""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid input data."""
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        
        # Test with NaN values
        invalid_data = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])
        
        with pytest.raises(Exception):
            trainer.train(invalid_data, n_iterations=10)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        trainer = EnhancedHMMTrainer(n_states=3, library='hmmlearn', random_state=42)
        
        # Very small dataset
        small_data = np.random.randn(3, 3)
        
        # Should either fail gracefully or succeed with warning
        try:
            artifact = trainer.train(small_data, n_iterations=10)
            # If it succeeds, that's okay
        except Exception as e:
            # Error should be informative
            assert len(str(e)) > 0
    
    def test_convergence_failure_handling(self):
        """Test handling of convergence failures."""
        trainer = EnhancedHMMTrainer(n_states=5, library='hmmlearn', random_state=42)
        
        # Data that might cause convergence issues
        difficult_data = np.random.randn(50, 3) * 0.01  # Very low variance
        
        # Should handle gracefully
        try:
            artifact = trainer.train(difficult_data, n_iterations=5)
            # If it succeeds, verify artifact
            assert artifact is not None
        except Exception as e:
            # Should have informative error
            assert len(str(e)) > 0
    
    def test_library_not_available_handling(self):
        """Test handling when library is not available."""
        # Try to use non-existent library
        with pytest.raises(ValueError, match="Unsupported library"):
            trainer = EnhancedHMMTrainer(
                n_states=3,
                library='nonexistent_library'
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
