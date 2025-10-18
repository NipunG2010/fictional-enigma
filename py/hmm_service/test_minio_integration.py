"""
Test MinIO integration with existing HMM artifacts for the HMM microservice.

This test verifies compatibility with current HMMArtifact format,
tests loading production models from MinIO, and validates artifact integrity checking.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
import hashlib
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

# Import HMM service components
from core.model_loader import ModelLoader
from core.inference_engine import HMMInferenceEngine
from core.config import ServiceConfig, MinIOConfig

# Import HMM models and artifact management
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from imp.hmm.models import HMMArtifact, FusionWeights
from imp.hmm.artifact_management import (
    MinIOArtifactStore,
    ResearchArtifact,
    ResearchMetadata,
    ArtifactValidator,
    ExperimentTracker,
    MinIOConnectionError,
    MinIOUploadError,
    MinIODownloadError,
    ValidationError
)


# Skip all tests if MinIO is not available
pytestmark = pytest.mark.integration


@pytest.fixture
def service_config():
    """Create service configuration for testing."""
    return ServiceConfig(
        host="0.0.0.0",
        port=8000,
        default_experiment_id="test_production_hmm",
        model_reload_interval=0,  # Disable auto-reload for tests
        minio=MinIOConfig(
            endpoint="localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin123",
            secure=False,
            bucket="hmm-artifacts"
        ),
        cache_size=100,
        cache_ttl=300,
        max_concurrent_requests=10,
        request_timeout=30.0
    )


@pytest.fixture
def minio_store():
    """Create MinIO store instance and verify connection."""
    try:
        config = MinIOConfig.from_env()
        store = MinIOArtifactStore(config)
        return store
    except MinIOConnectionError:
        pytest.skip("MinIO is not available. Run 'docker-compose up -d' to start MinIO.")


@pytest.fixture
def sample_production_hmm_artifact():
    """Create a production-ready HMM artifact matching current format."""
    return HMMArtifact(
        version="v2.1.0",
        n_states=3,
        transition_matrix=[
            [0.85, 0.10, 0.05],
            [0.15, 0.70, 0.15],
            [0.05, 0.25, 0.70]
        ],
        initial_probabilities=[0.4, 0.4, 0.2],
        means=[
            [0.02, -0.01, 0.03],  # Bull market state
            [-0.01, 0.00, -0.02], # Neutral state
            [-0.05, 0.02, -0.08]  # Bear market state
        ],
        covariances=[
            # Bull market - lower volatility
            [[0.0004, 0.0001, 0.0002], [0.0001, 0.0003, 0.0001], [0.0002, 0.0001, 0.0005]],
            # Neutral - medium volatility
            [[0.0008, 0.0002, 0.0003], [0.0002, 0.0006, 0.0002], [0.0003, 0.0002, 0.0009]],
            # Bear market - higher volatility
            [[0.0015, 0.0005, 0.0008], [0.0005, 0.0012, 0.0004], [0.0008, 0.0004, 0.0018]]
        ],
        training_window_start=int(datetime(2023, 1, 1).timestamp()),
        training_window_end=int(datetime(2024, 1, 1).timestamp()),
        metadata={
            "library": "hmmlearn",
            "algorithm": "baum-welch",
            "covariance_type": "full",
            "n_iter": 1000,
            "convergence_threshold": 1e-6,
            "training_samples": 50000,
            "validation_score": 0.892,
            "cross_validation_scores": [0.885, 0.898, 0.887, 0.901, 0.889],
            "created_by": "production_training_pipeline",
            "data_source": "market_data_2023",
            "feature_engineering_version": "v1.2.3"
        }
    )


@pytest.fixture
def sample_production_fusion_weights():
    """Create production fusion weights matching current format."""
    return FusionWeights(
        version="v2.1.0",
        state_weights=[
            # Bull market state - favor momentum signals
            {"w_ldc": 0.3, "w_mr": 0.2, "w_tsmom": 0.5},
            # Neutral state - balanced approach
            {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3},
            # Bear market state - favor mean reversion
            {"w_ldc": 0.2, "w_mr": 0.6, "w_tsmom": 0.2}
        ],
        model_version="v2.1.0",
        training_metrics={
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.12,
            "win_rate": 0.58,
            "profit_factor": 1.42,
            "calmar_ratio": 2.1
        },
        metadata={
            "optimization_method": "bayesian_optimization",
            "n_states": 3,
            "optimization_samples": 10000,
            "validation_period": "2023-Q4",
            "backtest_period": "2022-2023",
            "created_by": "weight_optimization_pipeline"
        }
    )


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for experiments."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestMinIOArtifactCompatibility:
    """Test compatibility with current HMMArtifact format."""
    
    def test_hmm_artifact_format_validation(self, sample_production_hmm_artifact):
        """Test that production HMM artifact matches expected format."""
        artifact = sample_production_hmm_artifact
        
        # Validate basic structure
        assert artifact.version == "v2.1.0"
        assert artifact.n_states == 3
        assert len(artifact.transition_matrix) == 3
        assert len(artifact.initial_probabilities) == 3
        assert len(artifact.means) == 3
        assert len(artifact.covariances) == 3
        
        # Validate transition matrix properties
        for i, row in enumerate(artifact.transition_matrix):
            assert len(row) == 3
            assert abs(sum(row) - 1.0) < 1e-6, f"Row {i} doesn't sum to 1"
            assert all(p >= 0 for p in row), f"Row {i} has negative probabilities"
        
        # Validate initial probabilities
        assert abs(sum(artifact.initial_probabilities) - 1.0) < 1e-6
        assert all(p >= 0 for p in artifact.initial_probabilities)
        
        # Validate means structure
        for i, mean in enumerate(artifact.means):
            assert len(mean) == 3, f"Mean {i} should have 3 features"
        
        # Validate covariance matrices
        for i, cov in enumerate(artifact.covariances):
            assert len(cov) == 3, f"Covariance {i} should be 3x3"
            for j, row in enumerate(cov):
                assert len(row) == 3, f"Covariance {i} row {j} should have 3 elements"
            
            # Check positive definiteness
            cov_matrix = np.array(cov)
            eigenvals = np.linalg.eigvals(cov_matrix)
            assert all(eigenvals > 0), f"Covariance {i} is not positive definite"
        
        # Validate metadata
        assert "library" in artifact.metadata
        assert "algorithm" in artifact.metadata
        assert "covariance_type" in artifact.metadata
    
    def test_fusion_weights_format_validation(self, sample_production_fusion_weights):
        """Test that production fusion weights match expected format."""
        weights = sample_production_fusion_weights
        
        # Validate basic structure
        assert weights.version == "v2.1.0"
        assert weights.model_version == "v2.1.0"
        assert len(weights.state_weights) == 3
        
        # Validate state weights
        expected_signals = {"w_ldc", "w_mr", "w_tsmom"}
        for i, state_weight in enumerate(weights.state_weights):
            assert set(state_weight.keys()) == expected_signals, f"State {i} missing signals"
            assert abs(sum(state_weight.values()) - 1.0) < 1e-6, f"State {i} weights don't sum to 1"
            assert all(w >= 0 for w in state_weight.values()), f"State {i} has negative weights"
        
        # Validate training metrics
        assert "sharpe_ratio" in weights.training_metrics
        assert "max_drawdown" in weights.training_metrics
        
        # Validate metadata
        assert "optimization_method" in weights.metadata
        assert "n_states" in weights.metadata
        assert weights.metadata["n_states"] == 3
    
    def test_artifact_serialization_compatibility(self, sample_production_hmm_artifact, sample_production_fusion_weights):
        """Test that artifacts can be serialized/deserialized correctly."""
        # Test HMM artifact serialization
        hmm_dict = sample_production_hmm_artifact.model_dump()
        hmm_json = json.dumps(hmm_dict, default=str)
        hmm_restored_dict = json.loads(hmm_json)
        hmm_restored = HMMArtifact(**hmm_restored_dict)
        
        assert hmm_restored.version == sample_production_hmm_artifact.version
        assert hmm_restored.n_states == sample_production_hmm_artifact.n_states
        assert np.allclose(hmm_restored.transition_matrix, sample_production_hmm_artifact.transition_matrix)
        
        # Test fusion weights serialization
        weights_dict = sample_production_fusion_weights.model_dump()
        weights_json = json.dumps(weights_dict, default=str)
        weights_restored_dict = json.loads(weights_json)
        weights_restored = FusionWeights(**weights_restored_dict)
        
        assert weights_restored.version == sample_production_fusion_weights.version
        assert weights_restored.state_weights == sample_production_fusion_weights.state_weights


class TestMinIOProductionModelLoading:
    """Test loading production models from MinIO."""
    
    @pytest.mark.asyncio
    async def test_upload_and_download_production_model(
        self, minio_store, sample_production_hmm_artifact, sample_production_fusion_weights, temp_experiment_dir
    ):
        """Test uploading and downloading a complete production model."""
        # Create research artifact for upload
        research_metadata = ResearchMetadata(
            experiment_id="test_production_hmm_v2",
            researcher="production_pipeline",
            training_config={
                "n_iter": 1000,
                "convergence_threshold": 1e-6,
                "algorithm": "baum-welch",
                "covariance_type": "full"
            },
            library_used="hmmlearn",
            evaluation_metrics={
                "log_likelihood": -15234.67,
                "aic": 30589.34,
                "bic": 30712.89,
                "validation_score": 0.892
            },
            tags=["production", "validated", "v2.1.0"]
        )
        
        # Calculate artifact hash
        artifact_str = json.dumps(sample_production_hmm_artifact.model_dump(), sort_keys=True)
        artifact_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
        
        research_artifact = ResearchArtifact(
            base_artifact=sample_production_hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v2.1.0",
            artifact_hash=artifact_hash,
            production_ready=True
        )
        
        # Upload to MinIO
        uploaded_paths = minio_store.upload_artifact(
            research_artifact,
            fusion_weights=sample_production_fusion_weights,
            tags=["production", "integration_test"]
        )
        
        assert "hmm_artifact" in uploaded_paths
        assert "fusion_weights" in uploaded_paths
        assert "metadata" in uploaded_paths
        
        # Download from MinIO
        downloaded = minio_store.download_artifact(
            "test_production_hmm_v2",
            "v2.1.0",
            validate_integrity=True
        )
        
        assert "hmm_artifact" in downloaded
        assert "fusion_weights" in downloaded
        assert "metadata" in downloaded
        
        # Validate downloaded HMM artifact
        downloaded_hmm = HMMArtifact(**downloaded["hmm_artifact"])
        assert downloaded_hmm.version == sample_production_hmm_artifact.version
        assert downloaded_hmm.n_states == sample_production_hmm_artifact.n_states
        assert np.allclose(downloaded_hmm.transition_matrix, sample_production_hmm_artifact.transition_matrix)
        assert np.allclose(downloaded_hmm.initial_probabilities, sample_production_hmm_artifact.initial_probabilities)
        
        # Validate downloaded fusion weights
        downloaded_weights = FusionWeights(**downloaded["fusion_weights"])
        assert downloaded_weights.version == sample_production_fusion_weights.version
        assert downloaded_weights.state_weights == sample_production_fusion_weights.state_weights
        
        # Validate metadata integrity
        assert downloaded["metadata"]["artifact_hash"] == artifact_hash
        assert downloaded["metadata"]["production_ready"] is True
        assert "production" in downloaded["metadata"]["tags"]
    
    @pytest.mark.asyncio
    async def test_model_loader_integration(
        self, service_config, minio_store, sample_production_hmm_artifact, sample_production_fusion_weights
    ):
        """Test ModelLoader integration with MinIO artifacts."""
        # First upload a production model
        research_metadata = ResearchMetadata(
            experiment_id=service_config.default_experiment_id,
            researcher="integration_test",
            training_config={"test": True},
            library_used="hmmlearn",
            tags=["production"]
        )
        
        artifact_str = json.dumps(sample_production_hmm_artifact.model_dump(), sort_keys=True)
        artifact_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
        
        research_artifact = ResearchArtifact(
            base_artifact=sample_production_hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v2.1.0",
            artifact_hash=artifact_hash,
            production_ready=True
        )
        
        minio_store.upload_artifact(research_artifact, fusion_weights=sample_production_fusion_weights)
        
        # Test ModelLoader
        model_loader = ModelLoader(service_config)
        await model_loader.initialize()
        
        # Load the model
        success = await model_loader.load_model(
            service_config.default_experiment_id,
            "v2.1.0",
            validate=True
        )
        
        assert success is True
        
        # Verify model info
        model_info = model_loader.get_model_info()
        assert model_info["loaded"] is True
        assert model_info["version"] == "v2.1.0"
        assert model_info["n_states"] == 3
        assert model_info["has_fusion_weights"] is True
        
        # Get loaded model
        hmm_artifact, fusion_weights = model_loader.get_current_model()
        assert hmm_artifact is not None
        assert fusion_weights is not None
        assert hmm_artifact.version == "v2.1.0"
        assert len(fusion_weights.state_weights) == 3
        
        await model_loader.cleanup()
    
    @pytest.mark.asyncio
    async def test_inference_engine_with_loaded_model(
        self, service_config, sample_production_hmm_artifact, sample_production_fusion_weights
    ):
        """Test HMMInferenceEngine with production model."""
        # Initialize inference engine
        inference_engine = HMMInferenceEngine(service_config)
        await inference_engine.initialize()
        
        # Load model
        success = await inference_engine.load_model(
            sample_production_hmm_artifact,
            sample_production_fusion_weights
        )
        
        assert success is True
        assert inference_engine.validate_model() is True
        
        # Test inference with realistic market data
        test_observations = np.array([0.015, -0.008, 0.022])  # [s_ldc, s_mr, s_tsmom]
        
        # Test state probability prediction
        state_probs = await inference_engine.predict_state_probabilities(test_observations)
        
        assert isinstance(state_probs, np.ndarray)
        assert len(state_probs) == 3
        assert abs(state_probs.sum() - 1.0) < 1e-6
        assert all(p >= 0 for p in state_probs)
        
        # Test fusion weight computation
        fusion_weights = await inference_engine.compute_fusion_weights(state_probs)
        
        assert isinstance(fusion_weights, dict)
        assert set(fusion_weights.keys()) == {"w_ldc", "w_mr", "w_tsmom"}
        assert abs(sum(fusion_weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in fusion_weights.values())
        
        # Test complete prediction
        prediction = await inference_engine.predict_complete(test_observations)
        
        assert len(prediction.state_probabilities) == 3
        assert 0 <= prediction.most_likely_state < 3
        assert 0 <= prediction.confidence <= 1
        assert set(prediction.fused_weights.keys()) == {"w_ldc", "w_mr", "w_tsmom"}
        assert prediction.timestamp > 0


class TestArtifactIntegrityValidation:
    """Test artifact integrity checking and validation."""
    
    def test_artifact_validator_production_checks(
        self, sample_production_hmm_artifact, sample_production_fusion_weights
    ):
        """Test ArtifactValidator with production artifacts."""
        # Create research artifact
        research_metadata = ResearchMetadata(
            experiment_id="validation_test",
            researcher="test",
            training_config={},
            library_used="hmmlearn"
        )
        
        artifact_str = json.dumps(sample_production_hmm_artifact.model_dump(), sort_keys=True)
        artifact_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
        
        research_artifact = ResearchArtifact(
            base_artifact=sample_production_hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v2.1.0",
            artifact_hash=artifact_hash,
            production_ready=True
        )
        
        # Run validation
        validation_results = ArtifactValidator.run_all_validations(research_artifact)
        
        # Check that all validations pass
        assert validation_results["all_passed"] is True
        assert len(validation_results["failed_checks"]) == 0
        
        # Check specific validation categories
        production_checks = validation_results["production_checks"]
        assert production_checks["integrity_check"] is True
        assert production_checks["hmm_artifact_valid"] is True
        assert production_checks["transition_matrix_valid"] is True
        assert production_checks["covariances_positive_definite"] is True
        
        rust_checks = validation_results["rust_compatibility_checks"]
        assert rust_checks["json_serializable"] is True
        assert rust_checks["dimensions_consistent"] is True
        assert rust_checks["no_invalid_values"] is True
    
    def test_integrity_hash_validation(self, sample_production_hmm_artifact, minio_store):
        """Test that artifact hash validation works correctly."""
        # Create research artifact with correct hash
        research_metadata = ResearchMetadata(
            experiment_id="hash_test",
            researcher="test",
            training_config={},
            library_used="hmmlearn"
        )
        
        artifact_str = json.dumps(sample_production_hmm_artifact.model_dump(), sort_keys=True)
        correct_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
        
        research_artifact = ResearchArtifact(
            base_artifact=sample_production_hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v1.0.0",
            artifact_hash=correct_hash,
            production_ready=True
        )
        
        # Upload artifact
        minio_store.upload_artifact(research_artifact)
        
        # Download with integrity validation
        downloaded = minio_store.download_artifact(
            "hash_test",
            "v1.0.0",
            validate_integrity=True
        )
        
        # Should succeed without errors
        assert downloaded["metadata"]["artifact_hash"] == correct_hash
        
        # Test with corrupted hash (simulate corruption)
        research_artifact.artifact_hash = "corrupted_hash"
        minio_store.upload_artifact(research_artifact)
        
        # Download should fail integrity check
        with pytest.raises(ValidationError, match="integrity check failed"):
            minio_store.download_artifact(
                "hash_test",
                "v1.0.0",
                validate_integrity=True
            )
    
    def test_model_validation_edge_cases(self, sample_production_hmm_artifact):
        """Test model validation with edge cases and invalid data."""
        # Test with invalid transition matrix (rows don't sum to 1)
        invalid_artifact = sample_production_hmm_artifact.model_copy()
        invalid_artifact.transition_matrix[0] = [0.5, 0.3, 0.1]  # Sums to 0.9
        
        with pytest.raises(ValueError, match="probabilities must sum to 1"):
            HMMArtifact(**invalid_artifact.model_dump())
        
        # Test with negative probabilities
        invalid_artifact = sample_production_hmm_artifact.model_copy()
        invalid_artifact.initial_probabilities = [0.5, -0.1, 0.6]
        
        with pytest.raises(ValueError, match="must be non-negative"):
            HMMArtifact(**invalid_artifact.model_dump())
        
        # Test with non-positive definite covariance
        invalid_artifact = sample_production_hmm_artifact.model_copy()
        invalid_artifact.covariances[0] = [
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],  # This makes the matrix non-positive definite
            [0.0, 0.0, 1.0]
        ]
        
        with pytest.raises(ValueError, match="not positive definite"):
            HMMArtifact(**invalid_artifact.model_dump())


class TestProductionDeploymentWorkflow:
    """Test complete production deployment workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_production_deployment(
        self, minio_store, sample_production_hmm_artifact, sample_production_fusion_weights, service_config
    ):
        """Test complete end-to-end production deployment workflow."""
        experiment_id = "e2e_production_test"
        
        # Step 1: Create and validate production artifact
        research_metadata = ResearchMetadata(
            experiment_id=experiment_id,
            researcher="production_pipeline",
            training_config={
                "algorithm": "baum-welch",
                "n_iter": 1000,
                "convergence_threshold": 1e-6
            },
            library_used="hmmlearn",
            evaluation_metrics={
                "log_likelihood": -15234.67,
                "validation_score": 0.892,
                "cross_validation_mean": 0.889,
                "cross_validation_std": 0.006
            },
            tags=["staging"]
        )
        
        artifact_str = json.dumps(sample_production_hmm_artifact.model_dump(), sort_keys=True)
        artifact_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
        
        research_artifact = ResearchArtifact(
            base_artifact=sample_production_hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v2.1.0",
            artifact_hash=artifact_hash,
            production_ready=True
        )
        
        # Step 2: Upload to MinIO
        uploaded_paths = minio_store.upload_artifact(
            research_artifact,
            fusion_weights=sample_production_fusion_weights
        )
        
        assert len(uploaded_paths) == 3  # hmm_artifact, fusion_weights, metadata
        
        # Step 3: Validate artifact
        validation_results = ArtifactValidator.run_all_validations(research_artifact)
        assert validation_results["all_passed"] is True
        
        # Step 4: Tag as production
        minio_store.tag_artifact(experiment_id, "v2.1.0", "production")
        
        # Step 5: Deploy to production
        deployment_record = minio_store.deploy_artifact(
            experiment_id,
            "v2.1.0",
            deployment_environment="production",
            deployment_notes="End-to-end integration test deployment"
        )
        
        assert deployment_record["experiment_id"] == experiment_id
        assert deployment_record["deployment_environment"] == "production"
        
        # Step 6: Load in ModelLoader
        model_loader = ModelLoader(service_config)
        await model_loader.initialize()
        
        success = await model_loader.load_model(experiment_id, "v2.1.0", validate=True)
        assert success is True
        
        # Step 7: Test inference
        hmm_artifact, fusion_weights = model_loader.get_current_model()
        
        inference_engine = HMMInferenceEngine(service_config)
        await inference_engine.initialize()
        await inference_engine.load_model(hmm_artifact, fusion_weights)
        
        # Test with realistic market observations
        test_observations = np.array([0.012, -0.005, 0.018])
        prediction = await inference_engine.predict_complete(test_observations)
        
        assert len(prediction.state_probabilities) == 3
        assert 0 <= prediction.most_likely_state < 3
        assert 0 <= prediction.confidence <= 1
        
        # Step 8: Verify production artifact retrieval
        prod_artifact = minio_store.get_production_artifact()
        assert prod_artifact is not None
        assert prod_artifact["metadata"]["experiment_id"] == experiment_id
        assert "production" in prod_artifact["metadata"]["tags"]
        
        # Cleanup
        await model_loader.cleanup()
    
    def test_fallback_to_previous_production_model(self, minio_store, sample_production_hmm_artifact):
        """Test fallback to previous production model when current fails."""
        # Upload first production model
        research_metadata_v1 = ResearchMetadata(
            experiment_id="fallback_test_v1",
            researcher="test",
            training_config={},
            library_used="hmmlearn",
            created_at=datetime(2024, 1, 1),
            tags=["production"]
        )
        
        artifact_v1 = sample_production_hmm_artifact.model_copy()
        artifact_v1.version = "v1.0.0"
        
        artifact_str = json.dumps(artifact_v1.model_dump(), sort_keys=True)
        artifact_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
        
        research_artifact_v1 = ResearchArtifact(
            base_artifact=artifact_v1,
            research_metadata=research_metadata_v1,
            artifact_version="v1.0.0",
            artifact_hash=artifact_hash,
            production_ready=True
        )
        
        minio_store.upload_artifact(research_artifact_v1)
        minio_store.tag_artifact("fallback_test_v1", "v1.0.0", "production")
        
        # Upload second production model (newer)
        research_metadata_v2 = ResearchMetadata(
            experiment_id="fallback_test_v2",
            researcher="test",
            training_config={},
            library_used="hmmlearn",
            created_at=datetime(2024, 6, 1),
            tags=["production"]
        )
        
        artifact_v2 = sample_production_hmm_artifact.model_copy()
        artifact_v2.version = "v2.0.0"
        
        artifact_str = json.dumps(artifact_v2.model_dump(), sort_keys=True)
        artifact_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
        
        research_artifact_v2 = ResearchArtifact(
            base_artifact=artifact_v2,
            research_metadata=research_metadata_v2,
            artifact_version="v2.0.0",
            artifact_hash=artifact_hash,
            production_ready=True
        )
        
        minio_store.upload_artifact(research_artifact_v2)
        minio_store.tag_artifact("fallback_test_v2", "v2.0.0", "production")
        
        # Get production artifact with fallback
        prod_artifact = minio_store.get_production_artifact(fallback_to_last_known=True)
        
        # Should get the newer one (v2)
        assert prod_artifact is not None
        assert prod_artifact["metadata"]["experiment_id"] == "fallback_test_v2"
        
        # Test deployment history
        history = minio_store.get_deployment_history("fallback_test_v2")
        assert isinstance(history, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])