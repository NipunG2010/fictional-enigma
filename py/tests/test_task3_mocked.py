#!/usr/bin/env python
"""
Unit tests for Task 3: Artifact download and listing functionality (with mocked MinIO).

This test verifies:
1. download_artifact with version parameter
2. _get_latest_version for "latest" version resolution
3. list_artifacts with filtering by experiment_id and tags
4. get_production_artifact for fetching production-tagged artifacts
5. Integrity validation using artifact hash after download
"""

import pytest
import sys
import json
import hashlib
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imp.hmm.artifact_management import (
    MinIOArtifactStore,
    MinIOConfig,
    MinIODownloadError,
    ValidationError
)
from imp.hmm.models import HMMArtifact
import numpy as np


class MockMinioObject:
    """Mock MinIO object."""
    def __init__(self, object_name):
        self.object_name = object_name


class MockMinioResponse:
    """Mock MinIO response."""
    def __init__(self, data):
        self._data = data
    
    def read(self):
        return self._data
    
    def close(self):
        pass
    
    def release_conn(self):
        pass


@pytest.fixture
def mock_minio_client():
    """Create a mock MinIO client."""
    with patch('imp.hmm.artifact_management.Minio') as mock_minio:
        client = MagicMock()
        mock_minio.return_value = client
        
        # Mock bucket operations
        client.bucket_exists.return_value = True
        client.list_buckets.return_value = []
        
        yield client


@pytest.fixture
def store(mock_minio_client):
    """Create MinIOArtifactStore with mocked client."""
    config = MinIOConfig(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin123",
        secure=False,
        bucket_name="test-bucket"
    )
    return MinIOArtifactStore(config)


@pytest.fixture
def sample_artifact_data():
    """Create sample artifact data."""
    # Create valid transition matrix (rows sum to 1)
    transition_matrix = [
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ]
    
    hmm_artifact = HMMArtifact(
        version="1.0.0",
        n_states=3,
        transition_matrix=transition_matrix,
        initial_probabilities=[0.33, 0.33, 0.34],
        means=[[1.0, 2.0, 3.0, 4.0, 5.0] for _ in range(3)],
        covariances=[np.eye(5).tolist() for _ in range(3)],
        training_window_start=0,
        training_window_end=1000,
        metadata={"library": "hmmlearn"}
    )
    
    # Calculate hash
    artifact_str = json.dumps(hmm_artifact.model_dump(), sort_keys=True)
    artifact_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
    
    metadata = {
        "experiment_id": "test_exp_001",
        "version": "v1.0.0",
        "researcher": "test_user",
        "created_at": datetime.now().isoformat(),
        "training_config": {"n_iter": 100},
        "evaluation_metrics": {"log_likelihood": -1234.56},
        "tags": ["test", "staging"],
        "production_ready": False,
        "artifact_hash": artifact_hash,
        "library_used": "hmmlearn",
        "n_states": 3,
        "notebook_path": None,
        "notes": ""
    }
    
    return {
        "hmm_artifact": hmm_artifact.model_dump(),
        "metadata": metadata,
        "fusion_weights": None
    }


def test_download_artifact_specific_version(store, mock_minio_client, sample_artifact_data):
    """Test downloading artifact with specific version."""
    
    # Setup mock responses
    def mock_get_object(bucket, object_name):
        if "metadata.json" in object_name:
            data = json.dumps(sample_artifact_data["metadata"]).encode()
        elif "hmm_artifact.json" in object_name:
            data = json.dumps(sample_artifact_data["hmm_artifact"]).encode()
        else:
            raise Exception("NoSuchKey")
        return MockMinioResponse(data)
    
    mock_minio_client.get_object.side_effect = mock_get_object
    
    # Test download
    result = store.download_artifact("test_exp_001", "v1.0.0")
    
    assert result["metadata"]["experiment_id"] == "test_exp_001"
    assert result["metadata"]["version"] == "v1.0.0"
    assert result["hmm_artifact"]["n_states"] == 3
    print("✓ Test passed: download_artifact with specific version")


def test_download_artifact_latest_version(store, mock_minio_client, sample_artifact_data):
    """Test downloading artifact with 'latest' version resolution."""
    
    # Mock list_objects to return multiple versions
    mock_objects = [
        MockMinioObject("test_exp_001/v1.0.0/"),
        MockMinioObject("test_exp_001/v1.1.0/"),
        MockMinioObject("test_exp_001/v1.2.0/"),
    ]
    mock_minio_client.list_objects.return_value = mock_objects
    
    # Update metadata for latest version
    latest_metadata = sample_artifact_data["metadata"].copy()
    latest_metadata["version"] = "v1.2.0"
    
    # Setup mock responses
    def mock_get_object(bucket, object_name):
        if "metadata.json" in object_name:
            data = json.dumps(latest_metadata).encode()
        elif "hmm_artifact.json" in object_name:
            data = json.dumps(sample_artifact_data["hmm_artifact"]).encode()
        else:
            raise Exception("NoSuchKey")
        return MockMinioResponse(data)
    
    mock_minio_client.get_object.side_effect = mock_get_object
    
    # Test download with "latest"
    result = store.download_artifact("test_exp_001", "latest")
    
    assert result["metadata"]["version"] == "v1.2.0"
    print("✓ Test passed: download_artifact with 'latest' version resolution")


def test_get_latest_version(store, mock_minio_client):
    """Test _get_latest_version method."""
    
    # Mock list_objects to return multiple versions
    mock_objects = [
        MockMinioObject("test_exp_001/v1.0.0/"),
        MockMinioObject("test_exp_001/v1.1.0/"),
        MockMinioObject("test_exp_001/v2.0.0/"),
        MockMinioObject("test_exp_001/v1.10.0/"),
    ]
    mock_minio_client.list_objects.return_value = mock_objects
    
    # Test
    latest = store._get_latest_version("test_exp_001")
    
    assert latest == "v2.0.0"
    print("✓ Test passed: _get_latest_version correctly identifies latest version")


def test_list_artifacts_all(store, mock_minio_client):
    """Test list_artifacts without filters."""
    
    # Mock list_objects
    mock_objects = [
        MockMinioObject("exp1/v1.0.0/metadata.json"),
        MockMinioObject("exp2/v1.0.0/metadata.json"),
    ]
    mock_minio_client.list_objects.return_value = mock_objects
    
    # Mock metadata responses
    metadata1 = {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["test"]}
    metadata2 = {"experiment_id": "exp2", "version": "v1.0.0", "tags": ["production"]}
    
    def mock_get_object(bucket, object_name):
        if "exp1" in object_name:
            data = json.dumps(metadata1).encode()
        else:
            data = json.dumps(metadata2).encode()
        return MockMinioResponse(data)
    
    mock_minio_client.get_object.side_effect = mock_get_object
    
    # Test
    artifacts = store.list_artifacts()
    
    assert len(artifacts) == 2
    assert any(a["experiment_id"] == "exp1" for a in artifacts)
    assert any(a["experiment_id"] == "exp2" for a in artifacts)
    print("✓ Test passed: list_artifacts returns all artifacts")


def test_list_artifacts_by_experiment_id(store, mock_minio_client):
    """Test list_artifacts with experiment_id filter."""
    
    # Mock list_objects with prefix
    mock_objects = [
        MockMinioObject("exp1/v1.0.0/metadata.json"),
        MockMinioObject("exp1/v1.1.0/metadata.json"),
    ]
    mock_minio_client.list_objects.return_value = mock_objects
    
    # Mock metadata responses
    metadata1 = {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["test"]}
    metadata2 = {"experiment_id": "exp1", "version": "v1.1.0", "tags": ["test"]}
    
    responses = [metadata1, metadata2]
    response_idx = [0]
    
    def mock_get_object(bucket, object_name):
        data = json.dumps(responses[response_idx[0]]).encode()
        response_idx[0] += 1
        return MockMinioResponse(data)
    
    mock_minio_client.get_object.side_effect = mock_get_object
    
    # Test
    artifacts = store.list_artifacts(experiment_id="exp1")
    
    assert len(artifacts) == 2
    assert all(a["experiment_id"] == "exp1" for a in artifacts)
    print("✓ Test passed: list_artifacts filters by experiment_id")


def test_list_artifacts_by_tags(store, mock_minio_client):
    """Test list_artifacts with tag filter."""
    
    # Mock list_objects
    mock_objects = [
        MockMinioObject("exp1/v1.0.0/metadata.json"),
        MockMinioObject("exp2/v1.0.0/metadata.json"),
        MockMinioObject("exp3/v1.0.0/metadata.json"),
    ]
    mock_minio_client.list_objects.return_value = mock_objects
    
    # Mock metadata responses
    metadata1 = {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["production", "validated"]}
    metadata2 = {"experiment_id": "exp2", "version": "v1.0.0", "tags": ["staging"]}
    metadata3 = {"experiment_id": "exp3", "version": "v1.0.0", "tags": ["production"]}
    
    responses = [metadata1, metadata2, metadata3]
    response_idx = [0]
    
    def mock_get_object(bucket, object_name):
        data = json.dumps(responses[response_idx[0]]).encode()
        response_idx[0] += 1
        return MockMinioResponse(data)
    
    mock_minio_client.get_object.side_effect = mock_get_object
    
    # Test
    artifacts = store.list_artifacts(tags=["production"])
    
    assert len(artifacts) == 2
    assert all("production" in a["tags"] for a in artifacts)
    print("✓ Test passed: list_artifacts filters by tags")


def test_get_production_artifact(store, mock_minio_client, sample_artifact_data):
    """Test get_production_artifact method."""
    
    # Mock list_objects for list_artifacts
    mock_objects = [
        MockMinioObject("exp1/v1.0.0/metadata.json"),
        MockMinioObject("exp2/v1.1.0/metadata.json"),
    ]
    mock_minio_client.list_objects.return_value = mock_objects
    
    # Mock metadata responses
    metadata1 = {
        "experiment_id": "exp1",
        "version": "v1.0.0",
        "tags": ["production"],
        "created_at": "2024-01-01T10:00:00"
    }
    metadata2 = {
        "experiment_id": "exp2",
        "version": "v1.1.0",
        "tags": ["production"],
        "created_at": "2024-01-02T10:00:00"  # More recent
    }
    
    # Prepare full artifact data for exp2
    prod_metadata = sample_artifact_data["metadata"].copy()
    prod_metadata.update(metadata2)
    prod_metadata["artifact_hash"] = sample_artifact_data["metadata"]["artifact_hash"]
    
    call_count = [0]
    
    def mock_get_object(bucket, object_name):
        call_count[0] += 1
        # First two calls are for list_artifacts
        if call_count[0] <= 2:
            if call_count[0] == 1:
                data = json.dumps(metadata1).encode()
            else:
                data = json.dumps(metadata2).encode()
        # Subsequent calls are for download_artifact
        elif "metadata.json" in object_name:
            data = json.dumps(prod_metadata).encode()
        elif "hmm_artifact.json" in object_name:
            data = json.dumps(sample_artifact_data["hmm_artifact"]).encode()
        else:
            raise Exception("NoSuchKey")
        return MockMinioResponse(data)
    
    mock_minio_client.get_object.side_effect = mock_get_object
    
    # Test
    prod_artifact = store.get_production_artifact()
    
    assert prod_artifact is not None
    assert prod_artifact["metadata"]["experiment_id"] == "exp2"
    assert "production" in prod_artifact["metadata"]["tags"]
    print("✓ Test passed: get_production_artifact returns latest production artifact")


def test_integrity_validation_success(store, mock_minio_client, sample_artifact_data):
    """Test integrity validation passes with correct hash."""
    
    # Setup mock responses
    def mock_get_object(bucket, object_name):
        if "metadata.json" in object_name:
            data = json.dumps(sample_artifact_data["metadata"]).encode()
        elif "hmm_artifact.json" in object_name:
            data = json.dumps(sample_artifact_data["hmm_artifact"]).encode()
        else:
            raise Exception("NoSuchKey")
        return MockMinioResponse(data)
    
    mock_minio_client.get_object.side_effect = mock_get_object
    
    # Test download with validation
    result = store.download_artifact("test_exp_001", "v1.0.0", validate_integrity=True)
    
    assert result["metadata"]["artifact_hash"] == sample_artifact_data["metadata"]["artifact_hash"]
    print("✓ Test passed: integrity validation succeeds with correct hash")


def test_integrity_validation_failure(store, mock_minio_client, sample_artifact_data):
    """Test integrity validation fails with incorrect hash."""
    
    # Modify metadata to have wrong hash
    bad_metadata = sample_artifact_data["metadata"].copy()
    bad_metadata["artifact_hash"] = "wrong_hash_value"
    
    # Setup mock responses
    def mock_get_object(bucket, object_name):
        if "metadata.json" in object_name:
            data = json.dumps(bad_metadata).encode()
        elif "hmm_artifact.json" in object_name:
            data = json.dumps(sample_artifact_data["hmm_artifact"]).encode()
        else:
            raise Exception("NoSuchKey")
        return MockMinioResponse(data)
    
    mock_minio_client.get_object.side_effect = mock_get_object
    
    # Test should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        store.download_artifact("test_exp_001", "v1.0.0", validate_integrity=True)
    
    assert "integrity check failed" in str(exc_info.value).lower()
    print("✓ Test passed: integrity validation fails with incorrect hash")


def test_download_nonexistent_artifact(store, mock_minio_client):
    """Test error handling for non-existent artifact."""
    
    # Mock to raise S3Error
    from minio.error import S3Error
    
    def mock_get_object(bucket, object_name):
        error = S3Error(
            "NoSuchKey",
            "The specified key does not exist",
            "resource",
            "request_id",
            "host_id",
            Mock(status=404)
        )
        raise error
    
    mock_minio_client.get_object.side_effect = mock_get_object
    mock_minio_client.list_objects.return_value = []
    
    # Test should raise MinIODownloadError
    with pytest.raises(MinIODownloadError) as exc_info:
        store.download_artifact("nonexistent_exp", "v1.0.0")
    
    assert "not found" in str(exc_info.value).lower()
    print("✓ Test passed: raises MinIODownloadError for non-existent artifact")


if __name__ == "__main__":
    print("=" * 80)
    print("Running Task 3 Unit Tests (Mocked)")
    print("=" * 80)
    
    # Run tests manually
    pytest.main([__file__, "-v", "-s"])
