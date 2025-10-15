"""
Tests for ExperimentTracker MinIO integration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import json
import hashlib

from imp.hmm.artifact_management import (
    ExperimentTracker,
    ResearchArtifact,
    ResearchMetadata,
    MinIOConfig,
    MinIOArtifactStore,
    MinIOConnectionError,
    MinIOUploadError,
    ArtifactManagementError
)
from imp.hmm.models import HMMArtifact, FusionWeights


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for experiments."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_artifact():
    """Create sample research artifact."""
    hmm_artifact = HMMArtifact(
        version="v1.0",
        n_states=3,
        transition_matrix=[[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]],
        initial_probabilities=[0.6, 0.3, 0.1],
        means=[[0.5, 0.3], [-0.2, 0.8], [0.1, -0.4]],
        covariances=[
            [[1.0, 0.1], [0.1, 1.0]],
            [[0.8, 0.0], [0.0, 0.9]],
            [[1.2, -0.1], [-0.1, 1.1]]
        ],
        training_window_start=0,
        training_window_end=1000,
        metadata={
            "library": "hmmlearn",
            "algorithm": "baum-welch",
            "covariance_type": "full"
        }
    )
    
    research_metadata = ResearchMetadata(
        experiment_id="test_exp_001",
        researcher="test_researcher",
        created_at=datetime.now(),
        training_config={"epochs": 100, "learning_rate": 0.01},
        evaluation_metrics={"accuracy": 0.95, "log_likelihood": -123.45},
        tags=["test", "development"],
        library_used="hmmlearn",
        notebook_path="notebooks/test.ipynb",
        notes="Test artifact"
    )
    
    # Calculate artifact hash
    artifact_str = json.dumps(hmm_artifact.model_dump(), sort_keys=True)
    artifact_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
    
    return ResearchArtifact(
        base_artifact=hmm_artifact,
        research_metadata=research_metadata,
        artifact_version="v1.0.0",
        artifact_hash=artifact_hash,
        production_ready=False
    )


@pytest.fixture
def sample_fusion_weights():
    """Create sample fusion weights."""
    return FusionWeights(
        version="v1.0",
        state_weights=[
            {"hmm": 0.6, "transformer": 0.4},
            {"hmm": 0.5, "transformer": 0.5},
            {"hmm": 0.7, "transformer": 0.3}
        ],
        model_version="v1.0",
        training_metrics={"accuracy": 0.92, "f1_score": 0.89},
        metadata={"fusion_strategy": "weighted_average", "samples": 1000}
    )


class TestExperimentTrackerBackwardCompatibility:
    """Test backward compatibility with local-only storage."""
    
    def test_initialize_without_minio(self, temp_experiment_dir):
        """Test initialization without MinIO (backward compatible)."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        assert tracker.experiment_dir == temp_experiment_dir
        assert tracker.use_minio is False
        assert tracker.minio_store is None
        assert tracker.experiments == {}
    
    def test_log_experiment_local_only(self, temp_experiment_dir, sample_artifact):
        """Test logging experiment without MinIO (backward compatible)."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        exp_id = tracker.log_experiment(sample_artifact, notes="Test experiment")
        
        assert exp_id == "test_exp_001"
        assert exp_id in tracker.experiments
        assert tracker.experiments[exp_id]["uploaded_to_minio"] is False
        
        # Verify local file exists
        artifact_path = temp_experiment_dir / exp_id / "artifact_vv1.0.0.json"
        assert artifact_path.exists()
    
    def test_log_experiment_with_fusion_weights_local(
        self, temp_experiment_dir, sample_artifact, sample_fusion_weights
    ):
        """Test logging experiment with fusion weights locally."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        exp_id = tracker.log_experiment(
            sample_artifact,
            fusion_weights=sample_fusion_weights,
            notes="Test with weights"
        )
        
        assert exp_id in tracker.experiments
        assert tracker.experiments[exp_id]["has_fusion_weights"] is True
        
        # Verify fusion weights file exists
        weights_path = temp_experiment_dir / exp_id / "fusion_weights_vv1.0.0.json"
        assert weights_path.exists()


class TestExperimentTrackerMinIOIntegration:
    """Test MinIO integration features."""
    
    def test_initialize_with_minio_unavailable(self, temp_experiment_dir):
        """Test initialization when MinIO is unavailable falls back gracefully."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_store.side_effect = MinIOConnectionError("Connection failed")
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            
            # Should fall back to local-only mode
            assert tracker.use_minio is False
            assert tracker.minio_store is None
    
    def test_initialize_with_minio_enabled(self, temp_experiment_dir):
        """Test initialization with MinIO enabled."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            
            assert tracker.use_minio is True
            assert tracker.minio_store == mock_minio
    
    def test_log_experiment_with_minio_upload(
        self, temp_experiment_dir, sample_artifact, sample_fusion_weights
    ):
        """Test logging experiment with MinIO upload."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_minio.upload_artifact.return_value = {
                "hmm_artifact": "test_exp_001/v1.0.0/hmm_artifact.json",
                "fusion_weights": "test_exp_001/v1.0.0/fusion_weights.json",
                "metadata": "test_exp_001/v1.0.0/metadata.json"
            }
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            exp_id = tracker.log_experiment(
                sample_artifact,
                fusion_weights=sample_fusion_weights,
                notes="Test with MinIO"
            )
            
            # Verify local storage
            assert exp_id in tracker.experiments
            artifact_path = temp_experiment_dir / exp_id / "artifact_vv1.0.0.json"
            assert artifact_path.exists()
            
            # Verify MinIO upload was called
            mock_minio.upload_artifact.assert_called_once()
            
            # Verify index updated with MinIO status
            assert tracker.experiments[exp_id]["uploaded_to_minio"] is True
            assert "minio_paths" in tracker.experiments[exp_id]
    
    def test_log_experiment_minio_upload_fails(
        self, temp_experiment_dir, sample_artifact
    ):
        """Test that local storage succeeds even if MinIO upload fails."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_minio.upload_artifact.side_effect = MinIOUploadError("Upload failed")
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            exp_id = tracker.log_experiment(sample_artifact)
            
            # Local storage should succeed
            assert exp_id in tracker.experiments
            artifact_path = temp_experiment_dir / exp_id / "artifact_vv1.0.0.json"
            assert artifact_path.exists()
            
            # MinIO upload should have failed
            assert tracker.experiments[exp_id]["uploaded_to_minio"] is False
    
    def test_log_experiment_skip_minio_upload(
        self, temp_experiment_dir, sample_artifact
    ):
        """Test logging experiment without uploading to MinIO."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            exp_id = tracker.log_experiment(
                sample_artifact,
                upload_to_minio=False
            )
            
            # Verify MinIO upload was not called
            mock_minio.upload_artifact.assert_not_called()
            assert tracker.experiments[exp_id]["uploaded_to_minio"] is False


class TestExperimentTrackerSyncMethods:
    """Test sync methods between local and MinIO storage."""
    
    def test_sync_to_minio_without_minio_enabled(self, temp_experiment_dir):
        """Test sync_to_minio raises error when MinIO not enabled."""
        tracker = ExperimentTracker(temp_experiment_dir, use_minio=False)
        
        with pytest.raises(ArtifactManagementError, match="MinIO is not enabled"):
            tracker.sync_to_minio()
    
    def test_sync_to_minio_single_experiment(
        self, temp_experiment_dir, sample_artifact
    ):
        """Test syncing single experiment to MinIO."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_minio.upload_artifact.return_value = {
                "hmm_artifact": "test_exp_001/v1.0.0/hmm_artifact.json"
            }
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            
            # Log experiment locally without uploading
            exp_id = tracker.log_experiment(sample_artifact, upload_to_minio=False)
            
            # Sync to MinIO
            results = tracker.sync_to_minio(experiment_id=exp_id)
            
            assert len(results["uploaded"]) == 1
            assert exp_id in results["uploaded"]
            assert len(results["skipped"]) == 0
            assert len(results["failed"]) == 0
            
            # Verify index updated
            assert tracker.experiments[exp_id]["uploaded_to_minio"] is True
    
    def test_sync_to_minio_all_experiments(
        self, temp_experiment_dir, sample_artifact
    ):
        """Test syncing all experiments to MinIO."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_minio.upload_artifact.return_value = {
                "hmm_artifact": "test/v1.0.0/hmm_artifact.json"
            }
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            
            # Log multiple experiments locally
            exp_id1 = tracker.log_experiment(sample_artifact, upload_to_minio=False)
            
            # Create second artifact
            sample_artifact.research_metadata.experiment_id = "test_exp_002"
            exp_id2 = tracker.log_experiment(sample_artifact, upload_to_minio=False)
            
            # Sync all to MinIO
            results = tracker.sync_to_minio()
            
            assert len(results["uploaded"]) == 2
            assert exp_id1 in results["uploaded"]
            assert exp_id2 in results["uploaded"]
    
    def test_sync_to_minio_skip_already_uploaded(
        self, temp_experiment_dir, sample_artifact
    ):
        """Test that already uploaded experiments are skipped."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_minio.upload_artifact.return_value = {}
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            
            # Log experiment with upload
            exp_id = tracker.log_experiment(sample_artifact, upload_to_minio=True)
            
            # Try to sync again
            results = tracker.sync_to_minio(experiment_id=exp_id)
            
            assert len(results["skipped"]) == 1
            assert exp_id in results["skipped"]
            assert len(results["uploaded"]) == 0
    
    def test_sync_to_minio_force_reupload(
        self, temp_experiment_dir, sample_artifact
    ):
        """Test force re-upload of already uploaded experiments."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_minio.upload_artifact.return_value = {}
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            
            # Log experiment with upload
            exp_id = tracker.log_experiment(sample_artifact, upload_to_minio=True)
            
            # Force sync again
            results = tracker.sync_to_minio(experiment_id=exp_id, force=True)
            
            assert len(results["uploaded"]) == 1
            assert exp_id in results["uploaded"]
            assert len(results["skipped"]) == 0
    
    def test_sync_from_minio_without_minio_enabled(self, temp_experiment_dir):
        """Test sync_from_minio raises error when MinIO not enabled."""
        tracker = ExperimentTracker(temp_experiment_dir, use_minio=False)
        
        with pytest.raises(ArtifactManagementError, match="MinIO is not enabled"):
            tracker.sync_from_minio()
    
    def test_sync_from_minio_download_artifact(
        self, temp_experiment_dir, sample_artifact
    ):
        """Test downloading artifact from MinIO to local storage."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            
            # Mock list_artifacts
            mock_minio.list_artifacts.return_value = [{
                "experiment_id": "test_exp_001",
                "version": "v1.0.0",
                "artifact_hash": "abc123",
                "production_ready": False
            }]
            
            # Mock download_artifact
            mock_minio.download_artifact.return_value = {
                "hmm_artifact": sample_artifact.base_artifact.model_dump(),
                "metadata": {
                    "experiment_id": "test_exp_001",
                    "researcher": "test_researcher",
                    "created_at": datetime.now().isoformat(),
                    "training_config": {},
                    "evaluation_metrics": {},
                    "tags": ["test"],
                    "library_used": "hmmlearn",
                    "notebook_path": "test.ipynb",
                    "notes": "Test"
                },
                "fusion_weights": None
            }
            
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            
            # Sync from MinIO
            results = tracker.sync_from_minio(experiment_id="test_exp_001")
            
            assert len(results["downloaded"]) == 1
            assert "test_exp_001" in results["downloaded"]
            
            # Verify local file created
            assert "test_exp_001" in tracker.experiments
            artifact_path = temp_experiment_dir / "test_exp_001" / "artifact_vv1.0.0.json"
            assert artifact_path.exists()
    
    def test_sync_from_minio_skip_existing(
        self, temp_experiment_dir, sample_artifact
    ):
        """Test that existing local artifacts are skipped."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_minio.upload_artifact.return_value = {}
            mock_minio.list_artifacts.return_value = [{
                "experiment_id": "test_exp_001",
                "version": "v1.0.0",
                "artifact_hash": "abc123"
            }]
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            
            # Log experiment locally first
            tracker.log_experiment(sample_artifact, upload_to_minio=False)
            
            # Try to sync from MinIO
            results = tracker.sync_from_minio(experiment_id="test_exp_001")
            
            assert len(results["skipped"]) == 1
            assert "test_exp_001" in results["skipped"]
    
    def test_get_minio_status_without_minio(self, temp_experiment_dir):
        """Test getting MinIO status when MinIO is not enabled."""
        tracker = ExperimentTracker(temp_experiment_dir, use_minio=False)
        
        status = tracker.get_minio_status()
        
        assert status["minio_enabled"] is False
        assert "message" in status
    
    def test_get_minio_status_with_minio(
        self, temp_experiment_dir, sample_artifact
    ):
        """Test getting MinIO status for experiments."""
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            mock_minio = Mock()
            mock_minio.upload_artifact.return_value = {}
            mock_minio.list_artifacts.return_value = [{
                "experiment_id": "test_exp_001",
                "version": "v1.0.0"
            }]
            mock_store.return_value = mock_minio
            
            tracker = ExperimentTracker(temp_experiment_dir, use_minio=True)
            
            # Log experiment
            exp_id = tracker.log_experiment(sample_artifact, upload_to_minio=True)
            
            # Get status
            status = tracker.get_minio_status(experiment_id=exp_id)
            
            assert status["minio_enabled"] is True
            assert status["minio_connected"] is True
            assert len(status["experiments"]) == 1
            
            exp_status = status["experiments"][0]
            assert exp_status["experiment_id"] == exp_id
            assert exp_status["uploaded_to_minio"] is True
            assert exp_status["in_minio"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
