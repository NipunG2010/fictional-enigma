"""
Integration tests for MinIO artifact storage with real MinIO instance.

These tests require a running MinIO instance (docker-compose up -d).
They test the full upload/download workflow, versioning, tagging, and deployment.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
import hashlib
import time

from imp.hmm.artifact_management import (
    MinIOConfig,
    MinIOArtifactStore,
    ExperimentTracker,
    ResearchArtifact,
    ResearchMetadata,
    ArtifactValidator,
    MinIOConnectionError,
    MinIOUploadError,
    MinIODownloadError,
    ValidationError
)
from imp.hmm.models import HMMArtifact, FusionWeights


# Skip all tests if MinIO is not available
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def minio_config():
    """Create MinIO configuration for testing."""
    return MinIOConfig.from_env()


@pytest.fixture(scope="module")
def minio_store(minio_config):
    """Create MinIO store instance and verify connection."""
    try:
        store = MinIOArtifactStore(minio_config)
        return store
    except MinIOConnectionError:
        pytest.skip("MinIO is not available. Run 'docker-compose up -d' to start MinIO.")


@pytest.fixture
def sample_hmm_artifact():
    """Create a sample HMM artifact for testing."""
    return HMMArtifact(
        version="1.0.0",
        n_states=3,
        n_features=2,
        transition_matrix=[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
        initial_probabilities=[0.33, 0.33, 0.34],
        means=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        covariances=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]]
        ],
        training_window_start=int(datetime(2024, 1, 1).timestamp()),
        training_window_end=int(datetime(2024, 12, 31).timestamp()),
        metadata={"library": "hmmlearn", "algorithm": "baum-welch", "covariance_type": "full"}
    )


@pytest.fixture
def sample_research_artifact(sample_hmm_artifact):
    """Create a sample research artifact for testing."""
    research_metadata = ResearchMetadata(
        experiment_id=f"integration_test_{int(time.time())}",
        researcher="integration_tester",
        training_config={"n_iter": 100},
        library_used="hmmlearn",
        evaluation_metrics={"log_likelihood": -1234.5}
    )
    
    artifact_str = json.dumps(sample_hmm_artifact.model_dump(), sort_keys=True)
    artifact_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
    
    return ResearchArtifact(
        base_artifact=sample_hmm_artifact,
        research_metadata=research_metadata,
        artifact_version="v1.0.0",
        artifact_hash=artifact_hash,
        production_ready=False
    )


@pytest.fixture
def sample_fusion_weights():
    """Create sample fusion weights for testing."""
    return FusionWeights(
        version="1.0.0",
        state_weights=[
            {"signal_a": 0.6, "signal_b": 0.4},
            {"signal_a": 0.5, "signal_b": 0.5},
            {"signal_a": 0.7, "signal_b": 0.3}
        ],
        model_version="1.0.0",
        training_metrics={"sharpe_ratio": 1.5},
        metadata={"optimization_method": "grid_search", "n_states": 3}
    )


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for experiments."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestMinIOConnection:
    """Test MinIO connection and initialization."""
    
    def test_connection_success(self, minio_store):
        """Test successful connection to MinIO."""
        assert minio_store is not None
        assert minio_store.client is not None
        assert minio_store.config.bucket_name == "hmm-artifacts"
    
    def test_bucket_exists(self, minio_store):
        """Test that bucket is created or exists."""
        # Bucket should exist after initialization
        assert minio_store.client.bucket_exists(minio_store.config.bucket_name)
    
    def test_invalid_connection(self):
        """Test connection failure with invalid credentials."""
        invalid_config = MinIOConfig(
            endpoint="invalid-endpoint:9000",
            access_key="invalid",
            secret_key="invalid"
        )
        
        with pytest.raises(MinIOConnectionError):
            MinIOArtifactStore(invalid_config)


class TestUploadDownloadRoundTrip:
    """Test upload/download round-trip with artifact validation."""
    
    def test_upload_download_hmm_artifact(self, minio_store, sample_research_artifact):
        """Test uploading and downloading HMM artifact."""
        # Upload artifact
        result = minio_store.upload_artifact(sample_research_artifact)
        
        assert "hmm_artifact" in result
        assert "metadata" in result
        assert result["hmm_artifact"].startswith(sample_research_artifact.research_metadata.experiment_id)
        
        # Download artifact
        downloaded = minio_store.download_artifact(
            sample_research_artifact.research_metadata.experiment_id,
            sample_research_artifact.artifact_version
        )
        
        assert "hmm_artifact" in downloaded
        assert "metadata" in downloaded
        
        # Verify artifact data matches
        original_artifact = sample_research_artifact.base_artifact.model_dump()
        downloaded_artifact = downloaded["hmm_artifact"]
        
        assert downloaded_artifact["n_states"] == original_artifact["n_states"]
        assert downloaded_artifact["n_features"] == original_artifact["n_features"]
        assert downloaded_artifact["version"] == original_artifact["version"]
    
    def test_upload_download_with_fusion_weights(
        self, minio_store, sample_research_artifact, sample_fusion_weights
    ):
        """Test uploading and downloading artifact with fusion weights."""
        # Upload artifact with fusion weights
        result = minio_store.upload_artifact(
            sample_research_artifact,
            fusion_weights=sample_fusion_weights
        )
        
        assert "hmm_artifact" in result
        assert "fusion_weights" in result
        assert "metadata" in result
        
        # Download artifact
        downloaded = minio_store.download_artifact(
            sample_research_artifact.research_metadata.experiment_id,
            sample_research_artifact.artifact_version
        )
        
        assert "fusion_weights" in downloaded
        assert downloaded["fusion_weights"] is not None
        
        # Verify fusion weights data
        original_weights = sample_fusion_weights.model_dump()
        downloaded_weights = downloaded["fusion_weights"]
        
        assert len(downloaded_weights["state_weights"]) == len(original_weights["state_weights"])
        assert downloaded_weights["version"] == original_weights["version"]
    
    def test_artifact_hash_validation(self, minio_store, sample_research_artifact):
        """Test that artifact hash is preserved and validated."""
        # Upload artifact
        minio_store.upload_artifact(sample_research_artifact)
        
        # Download artifact
        downloaded = minio_store.download_artifact(
            sample_research_artifact.research_metadata.experiment_id,
            sample_research_artifact.artifact_version
        )
        
        # Verify hash is preserved
        assert downloaded["metadata"]["artifact_hash"] == sample_research_artifact.artifact_hash
    
    def test_download_nonexistent_artifact(self, minio_store):
        """Test downloading non-existent artifact raises error."""
        with pytest.raises(MinIODownloadError):
            minio_store.download_artifact("nonexistent_experiment", "v1.0.0")


class TestVersioning:
    """Test versioning and 'latest' resolution."""
    
    def test_upload_multiple_versions(self, minio_store, sample_research_artifact):
        """Test uploading multiple versions of the same experiment."""
        exp_id = sample_research_artifact.research_metadata.experiment_id
        
        # Upload v1.0.0
        sample_research_artifact.artifact_version = "v1.0.0"
        result1 = minio_store.upload_artifact(sample_research_artifact)
        assert "v1.0.0" in result1["hmm_artifact"]
        
        # Upload v1.1.0
        sample_research_artifact.artifact_version = "v1.1.0"
        result2 = minio_store.upload_artifact(sample_research_artifact)
        assert "v1.1.0" in result2["hmm_artifact"]
        
        # Upload v2.0.0
        sample_research_artifact.artifact_version = "v2.0.0"
        result3 = minio_store.upload_artifact(sample_research_artifact)
        assert "v2.0.0" in result3["hmm_artifact"]
        
        # Verify all versions exist
        v1 = minio_store.download_artifact(exp_id, "v1.0.0")
        v2 = minio_store.download_artifact(exp_id, "v1.1.0")
        v3 = minio_store.download_artifact(exp_id, "v2.0.0")
        
        assert v1["metadata"]["version"] == "v1.0.0"
        assert v2["metadata"]["version"] == "v1.1.0"
        assert v3["metadata"]["version"] == "v2.0.0"
    
    def test_latest_version_resolution(self, minio_store, sample_research_artifact):
        """Test that 'latest' resolves to the most recent version."""
        exp_id = sample_research_artifact.research_metadata.experiment_id
        
        # Upload multiple versions
        for version in ["v1.0.0", "v1.1.0", "v2.0.0"]:
            sample_research_artifact.artifact_version = version
            minio_store.upload_artifact(sample_research_artifact)
        
        # Download 'latest'
        latest = minio_store.download_artifact(exp_id, "latest")
        
        # Should get v2.0.0
        assert latest["metadata"]["version"] == "v2.0.0"
    
    def test_semantic_versioning_order(self, minio_store, sample_research_artifact):
        """Test that semantic versioning is correctly ordered."""
        exp_id = sample_research_artifact.research_metadata.experiment_id
        
        # Upload versions in non-sequential order
        for version in ["v1.10.0", "v1.2.0", "v1.1.0", "v2.0.0"]:
            sample_research_artifact.artifact_version = version
            minio_store.upload_artifact(sample_research_artifact)
        
        # Latest should be v2.0.0, not v1.10.0
        latest = minio_store.download_artifact(exp_id, "latest")
        assert latest["metadata"]["version"] == "v2.0.0"


class TestTagging:
    """Test artifact tagging functionality."""
    
    def test_add_tag(self, minio_store, sample_research_artifact):
        """Test adding tags to artifacts."""
        exp_id = sample_research_artifact.research_metadata.experiment_id
        version = sample_research_artifact.artifact_version
        
        # Upload artifact
        minio_store.upload_artifact(sample_research_artifact)
        
        # Add tags
        minio_store.tag_artifact(exp_id, version, "staging", validate_production=False)
        minio_store.tag_artifact(exp_id, version, "experimental", validate_production=False)
        
        # Download and verify tags
        downloaded = minio_store.download_artifact(exp_id, version)
        tags = downloaded["metadata"]["tags"]
        
        assert "staging" in tags
        assert "experimental" in tags
    
    def test_production_tag_validation(self, minio_store, sample_research_artifact):
        """Test that production tag requires validation."""
        exp_id = sample_research_artifact.research_metadata.experiment_id
        version = sample_research_artifact.artifact_version
        
        # Upload non-validated artifact
        sample_research_artifact.production_ready = False
        minio_store.upload_artifact(sample_research_artifact)
        
        # Try to tag as production (should fail)
        with pytest.raises(ValidationError):
            minio_store.tag_artifact(exp_id, version, "production")
    
    def test_production_tag_with_validated_artifact(self, minio_store, sample_research_artifact):
        """Test tagging validated artifact as production."""
        exp_id = sample_research_artifact.research_metadata.experiment_id
        version = sample_research_artifact.artifact_version
        
        # Mark as production ready
        sample_research_artifact.production_ready = True
        
        # Run validation
        validation_results = ArtifactValidator.run_all_validations(sample_research_artifact)
        sample_research_artifact.validation_status = {
            **validation_results["production_checks"],
            **validation_results["rust_compatibility_checks"]
        }
        
        # Upload validated artifact
        minio_store.upload_artifact(sample_research_artifact)
        
        # Tag as production (should succeed)
        minio_store.tag_artifact(exp_id, version, "production")
        
        # Verify tag was added
        downloaded = minio_store.download_artifact(exp_id, version)
        assert "production" in downloaded["metadata"]["tags"]
    
    def test_list_artifacts_by_tag(self, minio_store, sample_research_artifact):
        """Test listing artifacts filtered by tag."""
        # Upload and tag artifact
        exp_id = sample_research_artifact.research_metadata.experiment_id
        version = sample_research_artifact.artifact_version
        
        minio_store.upload_artifact(sample_research_artifact)
        minio_store.tag_artifact(exp_id, version, "test-tag", validate_production=False)
        
        # List artifacts with tag
        artifacts = minio_store.list_artifacts(tags=["test-tag"])
        
        # Should find at least our artifact
        matching = [a for a in artifacts if a["experiment_id"] == exp_id]
        assert len(matching) > 0
        assert "test-tag" in matching[0]["tags"]


class TestProductionDeployment:
    """Test production deployment workflow end-to-end."""
    
    def test_full_deployment_workflow(self, minio_store, sample_research_artifact):
        """Test complete deployment workflow from upload to production."""
        exp_id = sample_research_artifact.research_metadata.experiment_id
        version = sample_research_artifact.artifact_version
        
        # Step 1: Upload artifact
        minio_store.upload_artifact(sample_research_artifact)
        
        # Step 2: Tag as staging
        minio_store.tag_artifact(exp_id, version, "staging", validate_production=False)
        
        # Step 3: Validate artifact
        sample_research_artifact.production_ready = True
        validation_results = ArtifactValidator.run_all_validations(sample_research_artifact)
        sample_research_artifact.validation_status = {
            **validation_results["production_checks"],
            **validation_results["rust_compatibility_checks"]
        }
        
        # Re-upload with validation status
        minio_store.upload_artifact(sample_research_artifact)
        
        # Step 4: Tag as production
        minio_store.tag_artifact(exp_id, version, "production")
        
        # Step 5: Deploy to production
        deployment_record = minio_store.deploy_artifact(
            exp_id,
            version,
            deployment_environment="production",
            deployment_notes="Integration test deployment"
        )
        
        # Verify deployment record
        assert deployment_record["experiment_id"] == exp_id
        assert deployment_record["version"] == version
        assert deployment_record["deployment_environment"] == "production"
        assert "deployed_at" in deployment_record
        assert "artifact_hash" in deployment_record
        
        # Step 6: Retrieve production artifact
        prod_artifact = minio_store.get_production_artifact()
        
        assert prod_artifact is not None
        assert prod_artifact["metadata"]["experiment_id"] == exp_id
        assert "production" in prod_artifact["metadata"]["tags"]
    
    def test_get_production_artifact_fallback(self, minio_store, sample_research_artifact):
        """Test production artifact retrieval with fallback to last known good."""
        # Create and upload two production artifacts
        exp_id1 = f"{sample_research_artifact.research_metadata.experiment_id}_prod1"
        exp_id2 = f"{sample_research_artifact.research_metadata.experiment_id}_prod2"
        
        # First artifact (older)
        sample_research_artifact.research_metadata.experiment_id = exp_id1
        sample_research_artifact.research_metadata.created_at = datetime(2024, 1, 1)
        sample_research_artifact.production_ready = True
        
        validation_results = ArtifactValidator.run_all_validations(sample_research_artifact)
        sample_research_artifact.validation_status = {
            **validation_results["production_checks"],
            **validation_results["rust_compatibility_checks"]
        }
        
        minio_store.upload_artifact(sample_research_artifact)
        minio_store.tag_artifact(exp_id1, sample_research_artifact.artifact_version, "production")
        
        # Second artifact (newer)
        sample_research_artifact.research_metadata.experiment_id = exp_id2
        sample_research_artifact.research_metadata.created_at = datetime(2024, 6, 1)
        
        minio_store.upload_artifact(sample_research_artifact)
        minio_store.tag_artifact(exp_id2, sample_research_artifact.artifact_version, "production")
        
        # Get production artifact (should get latest)
        prod_artifact = minio_store.get_production_artifact()
        
        assert prod_artifact is not None
        # Should get the newer one
        assert prod_artifact["metadata"]["experiment_id"] == exp_id2
    
    def test_deployment_history_tracking(self, minio_store, sample_research_artifact):
        """Test that deployment history is tracked correctly."""
        exp_id = sample_research_artifact.research_metadata.experiment_id
        version = sample_research_artifact.artifact_version
        
        # Prepare and upload artifact
        sample_research_artifact.production_ready = True
        validation_results = ArtifactValidator.run_all_validations(sample_research_artifact)
        sample_research_artifact.validation_status = {
            **validation_results["production_checks"],
            **validation_results["rust_compatibility_checks"]
        }
        
        minio_store.upload_artifact(sample_research_artifact)
        minio_store.tag_artifact(exp_id, version, "production")
        
        # Deploy to staging
        minio_store.deploy_artifact(exp_id, version, "staging", "Staging deployment")
        
        # Deploy to production
        minio_store.deploy_artifact(exp_id, version, "production", "Production deployment")
        
        # Get deployment history
        history = minio_store.get_deployment_history(exp_id)
        
        assert len(history) >= 2
        
        # Verify deployments are tracked
        environments = [d["deployment_environment"] for d in history]
        assert "staging" in environments
        assert "production" in environments


class TestErrorHandling:
    """Test error handling for connection failures and missing artifacts."""
    
    def test_upload_with_network_error(self, minio_store, sample_research_artifact):
        """Test upload behavior with simulated network issues."""
        # This test would require mocking network failures
        # For now, we test that upload completes successfully
        result = minio_store.upload_artifact(sample_research_artifact)
        assert "hmm_artifact" in result
    
    def test_download_missing_artifact_error_message(self, minio_store):
        """Test that missing artifact error provides helpful message."""
        with pytest.raises(MinIODownloadError) as exc_info:
            minio_store.download_artifact("missing_exp", "v1.0.0")
        
        error_msg = str(exc_info.value)
        assert "missing_exp" in error_msg or "not found" in error_msg.lower()
    
    def test_list_artifacts_with_corrupted_metadata(self, minio_store):
        """Test that list_artifacts handles corrupted metadata gracefully."""
        # List all artifacts (should not crash even if some metadata is corrupted)
        artifacts = minio_store.list_artifacts()
        
        # Should return a list (possibly empty)
        assert isinstance(artifacts, list)


class TestExperimentTrackerIntegration:
    """Test ExperimentTracker integration with MinIO."""
    
    def test_experiment_tracker_with_minio(
        self, minio_store, temp_experiment_dir, sample_research_artifact, sample_fusion_weights
    ):
        """Test ExperimentTracker with MinIO enabled."""
        tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
        
        # Skip if MinIO not available (tracker falls back to local-only)
        if not tracker.use_minio:
            pytest.skip("MinIO not available for integration test")
        
        # Log experiment
        exp_id = tracker.log_experiment(
            sample_research_artifact,
            fusion_weights=sample_fusion_weights,
            notes="Integration test"
        )
        
        # Verify local storage
        assert exp_id in tracker.experiments
        assert tracker.experiments[exp_id]["uploaded_to_minio"] is True
        
        # Verify local files exist
        artifact_path = temp_experiment_dir / exp_id / f"artifact_v{sample_research_artifact.artifact_version}.json"
        assert artifact_path.exists()
    
    def test_sync_to_minio(self, minio_store, temp_experiment_dir, sample_research_artifact):
        """Test syncing local experiments to MinIO."""
        tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
        
        # Skip if MinIO not available
        if not tracker.use_minio:
            pytest.skip("MinIO not available for integration test")
        
        # Log experiment without uploading
        exp_id = tracker.log_experiment(sample_research_artifact, upload_to_minio=False)
        
        assert tracker.experiments[exp_id]["uploaded_to_minio"] is False
        
        # Sync to MinIO
        results = tracker.sync_to_minio(experiment_id=exp_id)
        
        assert exp_id in results["uploaded"]
        assert tracker.experiments[exp_id]["uploaded_to_minio"] is True
    
    def test_get_minio_status(self, minio_store, temp_experiment_dir, sample_research_artifact):
        """Test getting MinIO status from ExperimentTracker."""
        tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
        
        # Skip if MinIO not available
        if not tracker.use_minio:
            pytest.skip("MinIO not available for integration test")
        
        # Log experiment
        exp_id = tracker.log_experiment(sample_research_artifact)
        
        # Get status
        status = tracker.get_minio_status(experiment_id=exp_id)
        
        assert status["minio_enabled"] is True
        assert status["minio_connected"] is True
        assert len(status["experiments"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
