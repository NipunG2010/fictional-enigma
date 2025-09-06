"""
Advanced and comprehensive tests for HMM components.
Tests edge cases, performance, error handling, and complex scenarios.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any
import time
from unittest.mock import patch, MagicMock

from imp.hmm.models import HMMArtifact, FusionWeights, HMMPrediction, SignalComponents
from imp.hmm.trainer import HMMTrainer
from imp.hmm.inference import HMMInference


class TestHMMArtifactAdvanced:
    """Advanced tests for HMMArtifact model validation and edge cases."""
    
    def test_artifact_validation_invalid_transition_matrix(self):
        """Test validation of invalid transition matrix."""
        with pytest.raises(ValueError):
            HMMArtifact(
                version="v1.0",
                n_states=3,
                transition_matrix=[[0.5, 0.3], [0.2, 0.8]],  # Wrong dimensions
                initial_probabilities=[0.5, 0.3, 0.2],
                means=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
                covariances=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 3,
                training_window_start=0,
                training_window_end=1000,
                metadata={"test": True}
            )
    
    def test_artifact_validation_non_probability_transitions(self):
        """Test validation of transition matrix that doesn't sum to 1."""
        with pytest.raises(ValueError):
            HMMArtifact(
                version="v1.0",
                n_states=2,
                transition_matrix=[[0.5, 0.3], [0.2, 0.8]],  # Rows don't sum to 1
                initial_probabilities=[0.5, 0.5],
                means=[[0.0, 0.0], [1.0, 1.0]],
                covariances=[[[1.0, 0.0], [0.0, 1.0]]] * 2,
                training_window_start=0,
                training_window_end=1000,
                metadata={"test": True}
            )
    
    def test_artifact_validation_negative_probabilities(self):
        """Test validation of negative probabilities."""
        with pytest.raises(ValueError):
            HMMArtifact(
                version="v1.0",
                n_states=2,
                transition_matrix=[[0.7, 0.3], [0.2, 0.8]],
                initial_probabilities=[-0.5, 1.5],  # Invalid probabilities
                means=[[0.0, 0.0], [1.0, 1.0]],
                covariances=[[[1.0, 0.0], [0.0, 1.0]]] * 2,
                training_window_start=0,
                training_window_end=1000,
                metadata={"test": True}
            )
    
    def test_artifact_validation_invalid_covariance_matrix(self):
        """Test validation of non-positive definite covariance matrix."""
        with pytest.raises(ValueError):
            HMMArtifact(
                version="v1.0",
                n_states=2,
                transition_matrix=[[0.7, 0.3], [0.2, 0.8]],
                initial_probabilities=[0.5, 0.5],
                means=[[0.0, 0.0], [1.0, 1.0]],
                covariances=[
                    [[1.0, 0.0], [0.0, 1.0]],  # Valid
                    [[1.0, 2.0], [2.0, 1.0]]   # Not positive definite
                ],
                training_window_start=0,
                training_window_end=1000,
                metadata={"test": True}
            )
    
    def test_artifact_serialization_roundtrip(self):
        """Test serialization and deserialization of complex artifacts."""
        artifact = HMMArtifact(
            version="v2.1",
            n_states=4,
            transition_matrix=[
                [0.6, 0.2, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.6, 0.2],
                [0.1, 0.1, 0.1, 0.7]
            ],
            initial_probabilities=[0.4, 0.3, 0.2, 0.1],
            means=[
                [0.0, 0.0, 0.0],
                [2.0, 1.5, -0.5],
                [-1.0, 0.5, 1.0],
                [0.5, -1.0, 0.0]
            ],
            covariances=[
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[2.0, 0.5, 0.0], [0.5, 1.5, 0.0], [0.0, 0.0, 0.8]],
                [[0.8, 0.0, 0.2], [0.0, 1.2, 0.0], [0.2, 0.0, 1.0]],
                [[1.1, 0.1, 0.1], [0.1, 1.0, 0.0], [0.1, 0.0, 0.9]]
            ],
            training_window_start=1000,
            training_window_end=5000,
            metadata={
                "algorithm": "baum-welch",
                "convergence_threshold": 1e-6,
                "random_seed": 42,
                "n_iterations": 150,
                "log_likelihood": -1234.56
            }
        )
        
        # Test JSON serialization
        json_str = artifact.model_dump_json()
        artifact_restored = HMMArtifact.model_validate_json(json_str)
        
        assert artifact_restored.version == artifact.version
        assert artifact_restored.n_states == artifact.n_states
        assert artifact_restored.metadata == artifact.metadata
        np.testing.assert_array_almost_equal(
            np.array(artifact_restored.transition_matrix),
            np.array(artifact.transition_matrix)
        )


class TestFusionWeightsAdvanced:
    """Advanced tests for FusionWeights validation and optimization."""
    
    def test_weights_validation_inconsistent_states(self):
        """Test validation when number of state weights doesn't match n_states."""
        with pytest.raises(ValueError):
            FusionWeights(
                version="v1.0",
                state_weights=[
                    {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3},
                    {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3}
                ],  # Only 2 states but model has 3
                model_version="v1.0",
                training_metrics={"sharpe_ratio": 1.5},
                metadata={"n_states": 3}  # Mismatch
            )
    
    def test_weights_validation_non_normalized_weights(self):
        """Test validation when weights don't sum to 1."""
        with pytest.raises(ValueError):
            FusionWeights(
                version="v1.0",
                state_weights=[
                    {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.4},  # Sums to 1.1
                    {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3}
                ],
                model_version="v1.0",
                training_metrics={"sharpe_ratio": 1.5},
                metadata={"n_states": 2}
            )
    
    def test_weights_validation_negative_weights(self):
        """Test validation of negative weights."""
        with pytest.raises(ValueError):
            FusionWeights(
                version="v1.0",
                state_weights=[
                    {"w_ldc": -0.1, "w_mr": 0.6, "w_tsmom": 0.5},  # Negative weight
                    {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3}
                ],
                model_version="v1.0",
                training_metrics={"sharpe_ratio": 1.5},
                metadata={"n_states": 2}
            )
    
    def test_weights_optimization_scenarios(self):
        """Test various weight optimization scenarios."""
        # Test extreme weight distributions
        extreme_weights = FusionWeights(
            version="v1.0",
            state_weights=[
                {"w_ldc": 0.95, "w_mr": 0.03, "w_tsmom": 0.02},  # LDC dominant
                {"w_ldc": 0.01, "w_mr": 0.01, "w_tsmom": 0.98},  # TSMOM dominant
                {"w_ldc": 0.02, "w_mr": 0.96, "w_tsmom": 0.02}   # MR dominant
            ],
            model_version="v1.0",
            training_metrics={
                "sharpe_ratio": 2.1,
                "max_drawdown": 0.05,
                "win_rate": 0.65,
                "profit_factor": 1.8
            },
            metadata={
                "optimization_method": "genetic_algorithm",
                "n_generations": 100,
                "population_size": 50
            }
        )
        
        assert extreme_weights.state_weights[0]["w_ldc"] > 0.9
        assert extreme_weights.state_weights[1]["w_tsmom"] > 0.9
        assert extreme_weights.state_weights[2]["w_mr"] > 0.9


class TestHMMTrainerAdvanced:
    """Advanced tests for HMMTrainer with complex scenarios."""
    
    def test_trainer_with_different_state_counts(self):
        """Test trainer with various numbers of states."""
        np.random.seed(42)
        n_samples = 200
        n_features = 3
        
        for n_states in [2, 3, 5, 7]:
            # Generate data with clear state separation
            observations = []
            for state in range(n_states):
                mean = np.random.normal(0, 2, n_features)
                cov = np.eye(n_features) * (0.5 + np.random.random())
                state_data = np.random.multivariate_normal(mean, cov, n_samples // n_states)
                observations.append(state_data)
            
            observations = np.vstack(observations)
            np.random.shuffle(observations)
            
            trainer = HMMTrainer(n_states=n_states)
            artifact = trainer.train(observations, n_iterations=20)
            
            assert artifact.n_states == n_states
            assert len(artifact.transition_matrix) == n_states
            assert len(artifact.initial_probabilities) == n_states
            assert len(artifact.means) == n_states
            assert len(artifact.covariances) == n_states
    
    def test_trainer_with_high_dimensional_data(self):
        """Test trainer with high-dimensional feature vectors."""
        np.random.seed(42)
        n_samples = 300
        n_features = 10  # High dimensional
        
        # Generate high-dimensional data with 3 states
        observations = []
        for state in range(3):
            mean = np.random.normal(0, 1, n_features)
            # Create correlated covariance matrix
            cov = np.random.randn(n_features, n_features)
            cov = cov @ cov.T + np.eye(n_features) * 0.1
            state_data = np.random.multivariate_normal(mean, cov, n_samples // 3)
            observations.append(state_data)
        
        observations = np.vstack(observations)
        np.random.shuffle(observations)
        
        trainer = HMMTrainer(n_states=3)
        artifact = trainer.train(observations, n_iterations=30)
        
        assert artifact.n_states == 3
        assert len(artifact.means[0]) == n_features
        assert len(artifact.covariances[0]) == n_features
        assert len(artifact.covariances[0][0]) == n_features
    
    def test_trainer_with_noisy_data(self):
        """Test trainer robustness with noisy data."""
        np.random.seed(42)
        n_samples = 500
        n_features = 3
        
        # Generate clean data
        clean_observations = []
        for state in range(3):
            mean = np.array([state * 2, state * 1.5, state * -1])
            cov = np.eye(n_features) * 0.5
            state_data = np.random.multivariate_normal(mean, cov, n_samples // 3)
            clean_observations.append(state_data)
        
        clean_observations = np.vstack(clean_observations)
        
        # Add various levels of noise
        noise_levels = [0.1, 0.5, 1.0, 2.0]
        for noise_level in noise_levels:
            noisy_observations = clean_observations + np.random.normal(0, noise_level, clean_observations.shape)
            np.random.shuffle(noisy_observations)
            
            trainer = HMMTrainer(n_states=3)
            artifact = trainer.train(noisy_observations, n_iterations=25)
            
            # Should still converge despite noise
            assert artifact.n_states == 3
            assert artifact.metadata["convergence_log_likelihood"] is not None
    
    def test_trainer_with_insufficient_data(self):
        """Test trainer behavior with insufficient data."""
        np.random.seed(42)
        n_samples = 10  # Very small dataset
        n_features = 3
        
        observations = np.random.normal(0, 1, (n_samples, n_features))
        
        trainer = HMMTrainer(n_states=3)
        
        # Should handle insufficient data gracefully
        with pytest.raises((ValueError, RuntimeError)):
            artifact = trainer.train(observations, n_iterations=10)
    
    def test_trainer_file_operations(self):
        """Test file save/load operations with complex artifacts."""
        np.random.seed(42)
        n_samples = 100
        n_features = 3
        
        # Generate test data
        observations = []
        for state in range(3):
            mean = np.random.normal(0, 1, n_features)
            cov = np.eye(n_features) * 0.5
            state_data = np.random.multivariate_normal(mean, cov, n_samples // 3)
            observations.append(state_data)
        
        observations = np.vstack(observations)
        np.random.shuffle(observations)
        
        trainer = HMMTrainer(n_states=3)
        original_artifact = trainer.train(observations, n_iterations=20)
        
        # Test save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            trainer.save_artifact(original_artifact, temp_path)
            loaded_artifact = trainer.load_artifact(temp_path)
            
            # Verify all fields match
            assert loaded_artifact.version == original_artifact.version
            assert loaded_artifact.n_states == original_artifact.n_states
            np.testing.assert_array_almost_equal(
                np.array(loaded_artifact.transition_matrix),
                np.array(original_artifact.transition_matrix)
            )
            np.testing.assert_array_almost_equal(
                np.array(loaded_artifact.initial_probabilities),
                np.array(original_artifact.initial_probabilities)
            )
            assert loaded_artifact.metadata == original_artifact.metadata
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_trainer_convergence_behavior(self):
        """Test trainer convergence with different iteration counts."""
        np.random.seed(42)
        n_samples = 200
        n_features = 3
        
        # Generate well-separated data
        observations = []
        for state in range(3):
            mean = np.array([state * 3, state * 2, state * -1])
            cov = np.eye(n_features) * 0.3
            state_data = np.random.multivariate_normal(mean, cov, n_samples // 3)
            observations.append(state_data)
        
        observations = np.vstack(observations)
        np.random.shuffle(observations)
        
        iteration_counts = [5, 10, 25, 50, 100]
        log_likelihoods = []
        
        for n_iter in iteration_counts:
            trainer = HMMTrainer(n_states=3)
            artifact = trainer.train(observations, n_iterations=n_iter)
            log_likelihoods.append(artifact.metadata["convergence_log_likelihood"])
        
        # Log likelihood should generally improve with more iterations
        # (though not strictly monotonic due to randomness)
        assert all(ll is not None for ll in log_likelihoods)
        assert log_likelihoods[-1] >= log_likelihoods[0] - 10  # Allow some tolerance


class TestHMMInferenceAdvanced:
    """Advanced tests for HMMInference with complex scenarios."""
    
    def test_inference_without_loaded_model(self):
        """Test inference behavior when model is not loaded."""
        inference = HMMInference()
        observation = np.array([0.5, 0.3, 0.2])
        
        with pytest.raises(ValueError, match="Model not loaded"):
            inference.predict(observation)
    
    def test_inference_with_malformed_observation(self):
        """Test inference with malformed observation vectors."""
        # Create a valid model first
        artifact = HMMArtifact(
            version="v1.0",
            n_states=2,
            transition_matrix=[[0.7, 0.3], [0.2, 0.8]],
            initial_probabilities=[0.5, 0.5],
            means=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            covariances=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 2,
            training_window_start=0,
            training_window_end=1000,
            metadata={"test": True}
        )
        
        weights = FusionWeights(
            version="v1.0",
            state_weights=[
                {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3},
                {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3}
            ],
            model_version="v1.0",
            training_metrics={"sharpe_ratio": 1.5},
            metadata={"n_states": 2}
        )
        
        inference = HMMInference()
        inference.load_model(artifact, weights)
        
        # Test with wrong dimensions
        with pytest.raises((ValueError, IndexError)):
            inference.predict(np.array([0.5, 0.3]))  # Wrong number of features
        
        # Test with NaN values
        with pytest.raises((ValueError, RuntimeError)):
            inference.predict(np.array([0.5, np.nan, 0.2]))
        
        # Test with infinite values
        with pytest.raises((ValueError, RuntimeError)):
            inference.predict(np.array([0.5, np.inf, 0.2]))
    
    def test_inference_batch_processing(self):
        """Test batch inference with various batch sizes."""
        # Create a valid model
        artifact = HMMArtifact(
            version="v1.0",
            n_states=3,
            transition_matrix=[[0.6, 0.2, 0.2], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]],
            initial_probabilities=[0.4, 0.3, 0.3],
            means=[[0.0, 0.0, 0.0], [2.0, 1.0, -1.0], [-1.0, 0.5, 1.0]],
            covariances=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 3,
            training_window_start=0,
            training_window_end=1000,
            metadata={"test": True}
        )
        
        weights = FusionWeights(
            version="v1.0",
            state_weights=[
                {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3},
                {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3},
                {"w_ldc": 0.3, "w_mr": 0.2, "w_tsmom": 0.5}
            ],
            model_version="v1.0",
            training_metrics={"sharpe_ratio": 1.5},
            metadata={"n_states": 3}
        )
        
        inference = HMMInference()
        inference.load_model(artifact, weights)
        
        # Test various batch sizes
        batch_sizes = [1, 5, 10, 50, 100]
        
        for batch_size in batch_sizes:
            observations = np.random.normal(0, 1, (batch_size, 3))
            predictions = inference.predict_batch(observations)
            
            assert len(predictions) == batch_size
            for pred in predictions:
                assert isinstance(pred, HMMPrediction)
                assert len(pred.state_probabilities) == 3
                assert 0 <= pred.most_likely_state < 3
                assert 0 <= pred.confidence <= 1
                assert "w_ldc" in pred.fused_weights
                assert "w_mr" in pred.fused_weights
                assert "w_tsmom" in pred.fused_weights
    
    def test_inference_consistency(self):
        """Test inference consistency with identical inputs."""
        # Create a valid model
        artifact = HMMArtifact(
            version="v1.0",
            n_states=2,
            transition_matrix=[[0.7, 0.3], [0.2, 0.8]],
            initial_probabilities=[0.5, 0.5],
            means=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            covariances=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 2,
            training_window_start=0,
            training_window_end=1000,
            metadata={"test": True}
        )
        
        weights = FusionWeights(
            version="v1.0",
            state_weights=[
                {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3},
                {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3}
            ],
            model_version="v1.0",
            training_metrics={"sharpe_ratio": 1.5},
            metadata={"n_states": 2}
        )
        
        inference = HMMInference()
        inference.load_model(artifact, weights)
        
        observation = np.array([0.5, 0.3, 0.2])
        
        # Make multiple predictions with same input
        predictions = []
        for _ in range(10):
            pred = inference.predict(observation)
            predictions.append(pred)
        
        # All predictions should be identical (except timestamp)
        first_pred = predictions[0]
        for pred in predictions[1:]:
            assert pred.state_probabilities == first_pred.state_probabilities
            assert pred.most_likely_state == first_pred.most_likely_state
            assert pred.confidence == first_pred.confidence
            assert pred.fused_weights == first_pred.fused_weights
    
    def test_inference_edge_cases(self):
        """Test inference with edge case observations."""
        # Create a valid model
        artifact = HMMArtifact(
            version="v1.0",
            n_states=2,
            transition_matrix=[[0.7, 0.3], [0.2, 0.8]],
            initial_probabilities=[0.5, 0.5],
            means=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            covariances=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 2,
            training_window_start=0,
            training_window_end=1000,
            metadata={"test": True}
        )
        
        weights = FusionWeights(
            version="v1.0",
            state_weights=[
                {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3},
                {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3}
            ],
            model_version="v1.0",
            training_metrics={"sharpe_ratio": 1.5},
            metadata={"n_states": 2}
        )
        
        inference = HMMInference()
        inference.load_model(artifact, weights)
        
        # Test extreme values
        extreme_observations = [
            np.array([0.0, 0.0, 0.0]),      # All zeros
            np.array([1.0, 1.0, 1.0]),      # All ones
            np.array([-1.0, -1.0, -1.0]),   # All negative ones
            np.array([10.0, -10.0, 5.0]),   # Large values
            np.array([0.001, 0.001, 0.001]) # Very small values
        ]
        
        for obs in extreme_observations:
            pred = inference.predict(obs)
            assert isinstance(pred, HMMPrediction)
            assert len(pred.state_probabilities) == 2
            assert abs(sum(pred.state_probabilities) - 1.0) < 1e-10  # Should sum to 1
            assert 0 <= pred.most_likely_state < 2
            assert 0 <= pred.confidence <= 1


class TestHMMPerformance:
    """Performance tests for HMM components."""
    
    def test_trainer_performance_large_dataset(self):
        """Test trainer performance with large datasets."""
        np.random.seed(42)
        n_samples = 10000  # Large dataset
        n_features = 5
        
        # Generate large dataset
        observations = []
        for state in range(3):
            mean = np.random.normal(0, 1, n_features)
            cov = np.eye(n_features) * 0.5
            state_data = np.random.multivariate_normal(mean, cov, n_samples // 3)
            observations.append(state_data)
        
        observations = np.vstack(observations)
        np.random.shuffle(observations)
        
        trainer = HMMTrainer(n_states=3)
        
        start_time = time.time()
        artifact = trainer.train(observations, n_iterations=20)
        training_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert training_time < 60  # 60 seconds
        assert artifact.n_states == 3
        assert artifact.metadata["convergence_log_likelihood"] is not None
    
    def test_inference_performance_high_throughput(self):
        """Test inference performance with high throughput."""
        # Create a valid model
        artifact = HMMArtifact(
            version="v1.0",
            n_states=3,
            transition_matrix=[[0.6, 0.2, 0.2], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]],
            initial_probabilities=[0.4, 0.3, 0.3],
            means=[[0.0, 0.0, 0.0], [2.0, 1.0, -1.0], [-1.0, 0.5, 1.0]],
            covariances=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 3,
            training_window_start=0,
            training_window_end=1000,
            metadata={"test": True}
        )
        
        weights = FusionWeights(
            version="v1.0",
            state_weights=[
                {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3},
                {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3},
                {"w_ldc": 0.3, "w_mr": 0.2, "w_tsmom": 0.5}
            ],
            model_version="v1.0",
            training_metrics={"sharpe_ratio": 1.5},
            metadata={"n_states": 3}
        )
        
        inference = HMMInference()
        inference.load_model(artifact, weights)
        
        # Test high throughput inference
        n_predictions = 1000
        observations = np.random.normal(0, 1, (n_predictions, 3))
        
        start_time = time.time()
        predictions = inference.predict_batch(observations)
        inference_time = time.time() - start_time
        
        # Should handle 1000 predictions quickly
        assert inference_time < 10  # 10 seconds
        assert len(predictions) == n_predictions
        
        # Calculate throughput
        throughput = n_predictions / inference_time
        assert throughput > 100  # At least 100 predictions per second


class TestHMMIntegration:
    """Integration tests combining multiple HMM components."""
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline from training to inference."""
        np.random.seed(42)
        n_samples = 500
        n_features = 3
        
        # Generate training data
        observations = []
        for state in range(3):
            mean = np.array([state * 2, state * 1.5, state * -1])
            cov = np.eye(n_features) * 0.5
            state_data = np.random.multivariate_normal(mean, cov, n_samples // 3)
            observations.append(state_data)
        
        observations = np.vstack(observations)
        np.random.shuffle(observations)
        
        # Step 1: Train HMM
        trainer = HMMTrainer(n_states=3)
        artifact = trainer.train(observations, n_iterations=30)
        
        # Step 2: Compute fusion weights
        returns = np.random.normal(0.01, 0.05, len(observations))  # Mock returns
        weights = trainer.compute_state_weights(observations, artifact, returns)
        
        # Step 3: Load model for inference
        inference = HMMInference()
        inference.load_model(artifact, weights)
        
        # Step 4: Make predictions on new data
        test_observations = np.random.normal(0, 1, (10, 3))
        predictions = inference.predict_batch(test_observations)
        
        # Verify pipeline works end-to-end
        assert len(predictions) == 10
        for pred in predictions:
            assert isinstance(pred, HMMPrediction)
            assert len(pred.state_probabilities) == 3
            assert 0 <= pred.most_likely_state < 3
            assert 0 <= pred.confidence <= 1
    
    def test_model_persistence_integration(self):
        """Test model persistence and loading integration."""
        np.random.seed(42)
        n_samples = 200
        n_features = 3
        
        # Generate data
        observations = []
        for state in range(2):
            mean = np.array([state * 3, state * 2, state * -1])
            cov = np.eye(n_features) * 0.3
            state_data = np.random.multivariate_normal(mean, cov, n_samples // 2)
            observations.append(state_data)
        
        observations = np.vstack(observations)
        np.random.shuffle(observations)
        
        # Train and save model
        trainer = HMMTrainer(n_states=2)
        artifact = trainer.train(observations, n_iterations=20)
        returns = np.random.normal(0.01, 0.05, len(observations))
        weights = trainer.compute_state_weights(observations, artifact, returns)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            trainer.save_artifact(artifact, temp_path)
            
            # Load model in new trainer instance
            new_trainer = HMMTrainer(n_states=2)
            loaded_artifact = new_trainer.load_artifact(temp_path)
            
            # Load in inference engine
            inference = HMMInference()
            inference.load_model(loaded_artifact, weights)
            
            # Test inference with loaded model
            test_obs = np.array([1.0, 0.5, -0.5])
            prediction = inference.predict(test_obs)
            
            assert isinstance(prediction, HMMPrediction)
            assert len(prediction.state_probabilities) == 2
            
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
