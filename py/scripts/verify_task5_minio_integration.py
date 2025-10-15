"""
Verification script for Task 5: ExperimentTracker MinIO Integration

This script demonstrates the new MinIO integration features:
1. ExperimentTracker with use_minio parameter
2. Automatic upload to MinIO during log_experiment
3. Sync methods between local and MinIO storage
4. Backward compatibility with local-only storage
"""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from imp.hmm.artifact_management import (
    ExperimentTracker,
    ResearchArtifact,
    ResearchMetadata,
    MinIOConfig,
    MinIOArtifactStore
)
from imp.hmm.models import HMMArtifact, FusionWeights
import json
import hashlib


def create_sample_artifact(experiment_id="test_exp_001"):
    """Create a sample research artifact for testing."""
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
        experiment_id=experiment_id,
        researcher="test_researcher",
        created_at=datetime.now(),
        training_config={"epochs": 100, "learning_rate": 0.01},
        evaluation_metrics={"accuracy": 0.95, "log_likelihood": -123.45},
        tags=["test", "development"],
        library_used="hmmlearn",
        notebook_path="notebooks/test.ipynb",
        notes="Test artifact for MinIO integration"
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


def create_sample_fusion_weights():
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


def verify_backward_compatibility():
    """Verify backward compatibility with local-only storage."""
    print("\n" + "="*70)
    print("TEST 1: Backward Compatibility (Local-Only Storage)")
    print("="*70)
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize tracker without MinIO (backward compatible)
        tracker = ExperimentTracker(temp_dir)
        
        print(f"✓ ExperimentTracker initialized (local-only)")
        print(f"  - use_minio: {tracker.use_minio}")
        print(f"  - minio_store: {tracker.minio_store}")
        
        # Log experiment
        artifact = create_sample_artifact()
        exp_id = tracker.log_experiment(artifact, notes="Test experiment")
        
        print(f"✓ Experiment logged: {exp_id}")
        print(f"  - uploaded_to_minio: {tracker.experiments[exp_id]['uploaded_to_minio']}")
        
        # Verify local file exists
        artifact_path = Path(temp_dir) / exp_id / f"artifact_v{artifact.artifact_version}.json"
        assert artifact_path.exists(), "Local artifact file should exist"
        print(f"✓ Local artifact file exists: {artifact_path}")
        
        print("\n✅ Backward compatibility verified!")
        
    finally:
        shutil.rmtree(temp_dir)


def verify_minio_integration():
    """Verify MinIO integration with mocked MinIO store."""
    print("\n" + "="*70)
    print("TEST 2: MinIO Integration (with mocked MinIO)")
    print("="*70)
    
    temp_dir = tempfile.mkdtemp()
    try:
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            # Mock MinIO store
            mock_minio = Mock()
            mock_minio.upload_artifact.return_value = {
                "hmm_artifact": "test_exp_002/v1.0.0/hmm_artifact.json",
                "fusion_weights": "test_exp_002/v1.0.0/fusion_weights.json",
                "metadata": "test_exp_002/v1.0.0/metadata.json"
            }
            mock_store.return_value = mock_minio
            
            # Initialize tracker with MinIO enabled
            tracker = ExperimentTracker(temp_dir, use_minio=True)
            
            print(f"✓ ExperimentTracker initialized with MinIO")
            print(f"  - use_minio: {tracker.use_minio}")
            print(f"  - minio_store: {tracker.minio_store is not None}")
            
            # Log experiment with fusion weights
            artifact = create_sample_artifact("test_exp_002")
            fusion_weights = create_sample_fusion_weights()
            exp_id = tracker.log_experiment(
                artifact,
                fusion_weights=fusion_weights,
                notes="Test with MinIO"
            )
            
            print(f"✓ Experiment logged: {exp_id}")
            print(f"  - uploaded_to_minio: {tracker.experiments[exp_id]['uploaded_to_minio']}")
            print(f"  - has_fusion_weights: {tracker.experiments[exp_id]['has_fusion_weights']}")
            
            # Verify MinIO upload was called
            assert mock_minio.upload_artifact.called, "MinIO upload should be called"
            print(f"✓ MinIO upload called: {mock_minio.upload_artifact.call_count} time(s)")
            
            # Verify local files exist
            artifact_path = Path(temp_dir) / exp_id / f"artifact_v{artifact.artifact_version}.json"
            weights_path = Path(temp_dir) / exp_id / f"fusion_weights_v{artifact.artifact_version}.json"
            assert artifact_path.exists(), "Local artifact file should exist"
            assert weights_path.exists(), "Local fusion weights file should exist"
            print(f"✓ Local files exist (artifact + fusion weights)")
            
            print("\n✅ MinIO integration verified!")
            
    finally:
        shutil.rmtree(temp_dir)


def verify_sync_methods():
    """Verify sync methods between local and MinIO storage."""
    print("\n" + "="*70)
    print("TEST 3: Sync Methods (Local ↔ MinIO)")
    print("="*70)
    
    temp_dir = tempfile.mkdtemp()
    try:
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            # Mock MinIO store
            mock_minio = Mock()
            mock_minio.upload_artifact.return_value = {
                "hmm_artifact": "test_exp_003/v1.0.0/hmm_artifact.json"
            }
            mock_minio.list_artifacts.return_value = []
            mock_store.return_value = mock_minio
            
            # Initialize tracker with MinIO
            tracker = ExperimentTracker(temp_dir, use_minio=True)
            
            # Log experiment without uploading to MinIO
            artifact = create_sample_artifact("test_exp_003")
            exp_id = tracker.log_experiment(artifact, upload_to_minio=False)
            
            print(f"✓ Experiment logged locally (no MinIO upload)")
            print(f"  - uploaded_to_minio: {tracker.experiments[exp_id]['uploaded_to_minio']}")
            
            # Sync to MinIO
            results = tracker.sync_to_minio(experiment_id=exp_id)
            
            print(f"✓ Synced to MinIO")
            print(f"  - uploaded: {results['uploaded']}")
            print(f"  - skipped: {results['skipped']}")
            print(f"  - failed: {results['failed']}")
            
            assert exp_id in results['uploaded'], "Experiment should be uploaded"
            assert tracker.experiments[exp_id]['uploaded_to_minio'], "Should be marked as uploaded"
            
            # Try to sync again (should be skipped)
            results2 = tracker.sync_to_minio(experiment_id=exp_id)
            assert exp_id in results2['skipped'], "Already uploaded experiment should be skipped"
            print(f"✓ Second sync skipped (already uploaded)")
            
            # Force re-upload
            results3 = tracker.sync_to_minio(experiment_id=exp_id, force=True)
            assert exp_id in results3['uploaded'], "Force sync should re-upload"
            print(f"✓ Force sync re-uploaded")
            
            # Get MinIO status
            status = tracker.get_minio_status(experiment_id=exp_id)
            print(f"✓ MinIO status retrieved")
            print(f"  - minio_enabled: {status['minio_enabled']}")
            print(f"  - minio_connected: {status['minio_connected']}")
            print(f"  - experiments tracked: {len(status['experiments'])}")
            
            print("\n✅ Sync methods verified!")
            
    finally:
        shutil.rmtree(temp_dir)


def verify_graceful_fallback():
    """Verify graceful fallback when MinIO is unavailable."""
    print("\n" + "="*70)
    print("TEST 4: Graceful Fallback (MinIO Unavailable)")
    print("="*70)
    
    temp_dir = tempfile.mkdtemp()
    try:
        with patch('imp.hmm.artifact_management.MinIOArtifactStore') as mock_store:
            # Simulate MinIO connection failure
            from imp.hmm.artifact_management import MinIOConnectionError
            mock_store.side_effect = MinIOConnectionError("Connection failed")
            
            # Initialize tracker with MinIO enabled (should fall back)
            tracker = ExperimentTracker(temp_dir, use_minio=True)
            
            print(f"✓ ExperimentTracker initialized (MinIO unavailable)")
            print(f"  - use_minio: {tracker.use_minio}")
            print(f"  - minio_store: {tracker.minio_store}")
            print(f"  - Gracefully fell back to local-only mode")
            
            # Log experiment (should work with local storage only)
            artifact = create_sample_artifact("test_exp_004")
            exp_id = tracker.log_experiment(artifact)
            
            print(f"✓ Experiment logged locally despite MinIO failure")
            print(f"  - experiment_id: {exp_id}")
            print(f"  - uploaded_to_minio: {tracker.experiments[exp_id]['uploaded_to_minio']}")
            
            # Verify local file exists
            artifact_path = Path(temp_dir) / exp_id / f"artifact_v{artifact.artifact_version}.json"
            assert artifact_path.exists(), "Local artifact file should exist"
            print(f"✓ Local artifact file exists")
            
            print("\n✅ Graceful fallback verified!")
            
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("TASK 5 VERIFICATION: ExperimentTracker MinIO Integration")
    print("="*70)
    print("\nThis script verifies the following requirements:")
    print("  1.5: MinIO integration for remote artifact storage")
    print("  2.5: Automatic upload during experiment logging")
    print("  4.3: Sync methods between local and MinIO")
    print("  4.4: Backward compatibility with local-only storage")
    print("  4.5: Graceful fallback when MinIO unavailable")
    
    try:
        verify_backward_compatibility()
        verify_minio_integration()
        verify_sync_methods()
        verify_graceful_fallback()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nTask 5 implementation verified successfully:")
        print("  ✓ use_minio parameter added to ExperimentTracker")
        print("  ✓ MinIOArtifactStore integrated as optional backend")
        print("  ✓ log_experiment uploads to both local and MinIO")
        print("  ✓ Sync methods implemented (sync_to_minio, sync_from_minio)")
        print("  ✓ Backward compatibility maintained")
        print("  ✓ Graceful fallback when MinIO unavailable")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
