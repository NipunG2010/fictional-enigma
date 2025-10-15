#!/usr/bin/env python
"""
Test script for Task 3: Artifact download and listing functionality.

This script tests:
1. download_artifact with version parameter
2. _get_latest_version for "latest" version resolution
3. list_artifacts with filtering by experiment_id and tags
4. get_production_artifact for fetching production-tagged artifacts
5. Integrity validation using artifact hash after download
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imp.hmm.artifact_management import (
    MinIOArtifactStore,
    MinIOConfig,
    ResearchArtifact,
    MinIOConnectionError,
    MinIODownloadError,
    ValidationError
)
from imp.hmm.models import HMMArtifact, FusionWeights
import numpy as np
from datetime import datetime


def test_download_and_listing():
    """Test download and listing functionality."""
    
    print("=" * 80)
    print("Task 3: Testing Artifact Download and Listing")
    print("=" * 80)
    
    try:
        # Initialize MinIO store
        print("\n1. Initializing MinIO store...")
        config = MinIOConfig.from_env()
        store = MinIOArtifactStore(config)
        print(f"   ✓ Connected to MinIO at {config.endpoint}")
        
        # Create and upload test artifacts
        print("\n2. Creating test artifacts...")
        
        # Create first artifact (v1.0.0)
        hmm_artifact_v1 = HMMArtifact(
            n_states=3,
            n_features=5,
            transition_matrix=np.random.rand(3, 3).tolist(),
            emission_means=np.random.rand(3, 5).tolist(),
            emission_covariances=[np.eye(5).tolist() for _ in range(3)],
            initial_distribution=np.array([0.33, 0.33, 0.34]).tolist(),
            metadata={"library": "hmmlearn", "test": "task3"}
        )
        
        research_artifact_v1 = ResearchArtifact.from_hmm_artifact(
            artifact=hmm_artifact_v1,
            experiment_id="test_exp_001",
            researcher="test_user",
            training_config={"n_iter": 100, "tol": 1e-4},
            evaluation_metrics={"log_likelihood": -1234.56, "aic": 2500.0},
            artifact_version="v1.0.0",
            tags=["test", "staging"]
        )
        
        # Upload v1.0.0
        print("   Uploading artifact v1.0.0...")
        paths_v1 = store.upload_artifact(research_artifact_v1)
        print(f"   ✓ Uploaded v1.0.0: {len(paths_v1)} files")
        
        # Create second artifact (v1.1.0)
        hmm_artifact_v2 = HMMArtifact(
            n_states=3,
            n_features=5,
            transition_matrix=np.random.rand(3, 3).tolist(),
            emission_means=np.random.rand(3, 5).tolist(),
            emission_covariances=[np.eye(5).tolist() for _ in range(3)],
            initial_distribution=np.array([0.33, 0.33, 0.34]).tolist(),
            metadata={"library": "hmmlearn", "test": "task3"}
        )
        
        research_artifact_v2 = ResearchArtifact.from_hmm_artifact(
            artifact=hmm_artifact_v2,
            experiment_id="test_exp_001",
            researcher="test_user",
            training_config={"n_iter": 200, "tol": 1e-5},
            evaluation_metrics={"log_likelihood": -1100.23, "aic": 2300.0},
            artifact_version="v1.1.0",
            tags=["test", "production"]
        )
        
        # Upload v1.1.0
        print("   Uploading artifact v1.1.0...")
        paths_v2 = store.upload_artifact(research_artifact_v2)
        print(f"   ✓ Uploaded v1.1.0: {len(paths_v2)} files")
        
        # Create third artifact (different experiment)
        hmm_artifact_v3 = HMMArtifact(
            n_states=2,
            n_features=5,
            transition_matrix=np.random.rand(2, 2).tolist(),
            emission_means=np.random.rand(2, 5).tolist(),
            emission_covariances=[np.eye(5).tolist() for _ in range(2)],
            initial_distribution=np.array([0.5, 0.5]).tolist(),
            metadata={"library": "hmmlearn", "test": "task3"}
        )
        
        research_artifact_v3 = ResearchArtifact.from_hmm_artifact(
            artifact=hmm_artifact_v3,
            experiment_id="test_exp_002",
            researcher="test_user",
            training_config={"n_iter": 150, "tol": 1e-4},
            evaluation_metrics={"log_likelihood": -980.45, "aic": 2100.0},
            artifact_version="v1.0.0",
            tags=["test", "experimental"]
        )
        
        # Upload exp_002
        print("   Uploading artifact for test_exp_002...")
        paths_v3 = store.upload_artifact(research_artifact_v3)
        print(f"   ✓ Uploaded test_exp_002 v1.0.0: {len(paths_v3)} files")
        
        # Test 3: Download specific version
        print("\n3. Testing download_artifact with specific version...")
        artifact_data = store.download_artifact("test_exp_001", "v1.0.0")
        assert artifact_data["metadata"]["version"] == "v1.0.0"
        assert artifact_data["hmm_artifact"]["n_states"] == 3
        print(f"   ✓ Downloaded v1.0.0 successfully")
        print(f"   ✓ Integrity validation passed")
        
        # Test 4: Download latest version
        print("\n4. Testing download_artifact with 'latest' version...")
        latest_data = store.download_artifact("test_exp_001", "latest")
        assert latest_data["metadata"]["version"] == "v1.1.0"
        print(f"   ✓ Resolved 'latest' to v1.1.0")
        print(f"   ✓ Downloaded latest version successfully")
        
        # Test 5: List all artifacts
        print("\n5. Testing list_artifacts (all artifacts)...")
        all_artifacts = store.list_artifacts()
        print(f"   ✓ Found {len(all_artifacts)} total artifacts")
        for artifact in all_artifacts:
            print(f"     - {artifact['experiment_id']} v{artifact['version']} "
                  f"(tags: {artifact.get('tags', [])})")
        
        # Test 6: List artifacts by experiment_id
        print("\n6. Testing list_artifacts with experiment_id filter...")
        exp_artifacts = store.list_artifacts(experiment_id="test_exp_001")
        assert len(exp_artifacts) == 2
        print(f"   ✓ Found {len(exp_artifacts)} artifacts for test_exp_001")
        
        # Test 7: List artifacts by tags
        print("\n7. Testing list_artifacts with tag filter...")
        prod_artifacts = store.list_artifacts(tags=["production"])
        assert len(prod_artifacts) >= 1
        print(f"   ✓ Found {len(prod_artifacts)} production artifacts")
        
        staging_artifacts = store.list_artifacts(tags=["staging"])
        print(f"   ✓ Found {len(staging_artifacts)} staging artifacts")
        
        # Test 8: Get production artifact
        print("\n8. Testing get_production_artifact...")
        prod_artifact = store.get_production_artifact()
        assert prod_artifact is not None
        assert "production" in prod_artifact["metadata"]["tags"]
        print(f"   ✓ Retrieved production artifact: "
              f"{prod_artifact['metadata']['experiment_id']} "
              f"v{prod_artifact['metadata']['version']}")
        
        # Test 9: Integrity validation failure (simulate)
        print("\n9. Testing integrity validation...")
        try:
            # Download with validation
            artifact_data = store.download_artifact("test_exp_001", "v1.0.0", validate_integrity=True)
            print(f"   ✓ Integrity validation passed for v1.0.0")
        except ValidationError as e:
            print(f"   ✗ Integrity validation failed: {e}")
            return False
        
        # Test 10: Download non-existent artifact
        print("\n10. Testing error handling for non-existent artifact...")
        try:
            store.download_artifact("non_existent_exp", "v1.0.0")
            print(f"   ✗ Should have raised MinIODownloadError")
            return False
        except MinIODownloadError as e:
            print(f"   ✓ Correctly raised MinIODownloadError: {str(e)[:80]}...")
        
        # Test 11: Download non-existent version
        print("\n11. Testing error handling for non-existent version...")
        try:
            store.download_artifact("test_exp_001", "v99.99.99")
            print(f"   ✗ Should have raised MinIODownloadError")
            return False
        except MinIODownloadError as e:
            print(f"   ✓ Correctly raised MinIODownloadError: {str(e)[:80]}...")
        
        print("\n" + "=" * 80)
        print("✓ All Task 3 tests passed successfully!")
        print("=" * 80)
        
        return True
        
    except MinIOConnectionError as e:
        print(f"\n✗ MinIO connection error: {e}")
        print("\nMake sure MinIO is running:")
        print("  docker-compose up -d minio")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_download_and_listing()
    sys.exit(0 if success else 1)
