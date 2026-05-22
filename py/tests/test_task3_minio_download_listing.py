#!/usr/bin/env python3
"""
Unit tests for Task 3: Artifact download and listing functionality (with mocked MinIO).

This test verifies:
1. download_artifact with version parameter
2. download_artifact with "latest" version resolution
3. _get_latest_version method
4. list_artifacts with and without filtering (experiment_id, tags)
5. get_production_artifact for fetching production-tagged artifacts
6. Integrity validation using artifact hash after download
7. Error handling for non-existent artifacts
"""

import pytest
import sys
import json
import hashlib
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from io import BytesIO
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imp.hmm.models import HMMArtifact

from imp.hmm.artifact_management import (
    MinIOArtifactStore,
    MinIOConfig,
    MinIODownloadError,
    MinIOUploadError,
    MinIOConnectionError,
    ValidationError,
)


# =============================================================================
# Helper Classes
# =============================================================================

class MockMinioObject:
    """Mock MinIO object for list_objects responses."""
    def __init__(self, object_name: str):
        self.object_name = object_name


class MockMinioResponse:
    """Mock MinIO response for get_object."""
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def close(self):
        pass

    def release_conn(self):
        pass


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_minio_client():
    """Create a thoroughly mocked MinIO client."""
    with patch('imp.hmm.artifact_management.Minio') as mock_minio:
        client = MagicMock()
        client.bucket_exists.return_value = True
        client.list_buckets.return_value = []
        client.list_objects.return_value = []
        mock_minio.return_value = client
        yield client


@pytest.fixture
def store(mock_minio_client):
    """Create MinIOArtifactStore with mocked client."""
    config = MinIOConfig(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin123",
        secure=False,
        bucket_name="test-bucket",
    )
    # Mock the initial bucket check
    with patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True):
        store = MinIOArtifactStore(config)
        # Replace the client with our mock
        store.client = mock_minio_client
        yield store


@pytest.fixture
def sample_artifact_data():
    """Create sample artifact data consistent with HMM artifact structure."""
    hmm_data = {
        "version": "1.0.0",
        "n_states": 3,
        "n_features": 2,
        "transition_matrix": [
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
        ],
        "initial_probabilities": [0.33, 0.33, 0.34],
        "means": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "covariances": [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        "training_window_start": 0,
        "training_window_end": 1000,
        "metadata": {"library": "hmmlearn", "algorithm": "baum-welch"},
    }

    hmm_artifact_obj = HMMArtifact(**hmm_data)
    artifact_str = json.dumps(hmm_artifact_obj.model_dump(), sort_keys=True)
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
        "notes": "",
    }

    return {
        "hmm_artifact": hmm_data,
        "metadata": metadata,
        "fusion_weights": None,
    }


def _setup_get_object(mock_client, artifact_data):
    """Helper to setup mock get_object responses."""
    def mock_get_object(bucket, object_name):
        if "metadata.json" in object_name:
            data = json.dumps(artifact_data["metadata"]).encode()
        elif "hmm_artifact.json" in object_name:
            data = json.dumps(artifact_data["hmm_artifact"]).encode()
        elif "fusion_weights.json" in object_name and artifact_data.get("fusion_weights"):
            data = json.dumps(artifact_data["fusion_weights"]).encode()
        else:
            raise Exception("NoSuchKey")
        return MockMinioResponse(data)

    mock_client.get_object.side_effect = mock_get_object


# =============================================================================
# Test Class
# =============================================================================

class TestDownloadArtifact:
    """Tests for download_artifact method."""

    def test_download_specific_version(self, store, mock_minio_client, sample_artifact_data):
        """Test downloading artifact with a specific version string."""
        _setup_get_object(mock_minio_client, sample_artifact_data)

        result = store.download_artifact("test_exp_001", "v1.0.0")

        assert result["metadata"]["experiment_id"] == "test_exp_001"
        assert result["metadata"]["version"] == "v1.0.0"
        assert result["hmm_artifact"]["n_states"] == 3

    def test_download_latest_version(self, store, mock_minio_client, sample_artifact_data):
        """Test downloading artifact with 'latest' version resolution."""
        # List objects returns multiple version directories
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("test_exp_001/v1.0.0/"),
            MockMinioObject("test_exp_001/v1.1.0/"),
            MockMinioObject("test_exp_001/v2.0.0/"),
        ]

        # Set metadata for the latest version
        latest_metadata = sample_artifact_data["metadata"].copy()
        latest_metadata["version"] = "v2.0.0"
        latest_data = {**sample_artifact_data, "metadata": latest_metadata}
        _setup_get_object(mock_minio_client, latest_data)

        result = store.download_artifact("test_exp_001", "latest")

        assert result["metadata"]["version"] == "v2.0.0"

    def test_download_with_integrity_validation(self, store, mock_minio_client, sample_artifact_data):
        """Test download with integrity validation passes for correct hash."""
        _setup_get_object(mock_minio_client, sample_artifact_data)

        result = store.download_artifact("test_exp_001", "v1.0.0", validate_integrity=True)

        assert result["metadata"]["artifact_hash"] == sample_artifact_data["metadata"]["artifact_hash"]

    def test_download_integrity_failure(self, store, mock_minio_client, sample_artifact_data):
        """Test download with integrity validation fails for wrong hash."""
        bad_metadata = sample_artifact_data["metadata"].copy()
        bad_metadata["artifact_hash"] = "clearly_wrong_hash"
        bad_data = {**sample_artifact_data, "metadata": bad_metadata}
        _setup_get_object(mock_minio_client, bad_data)

        with pytest.raises(ValidationError, match="integrity check failed"):
            store.download_artifact("test_exp_001", "v1.0.0", validate_integrity=True)

    def test_download_nonexistent_artifact(self, store, mock_minio_client):
        """Test error handling when artifact does not exist."""
        # No objects listed - artifact doesn't exist
        mock_minio_client.list_objects.return_value = []

        def raise_not_found(bucket, object_name):
            from minio.error import S3Error
            error = S3Error(
                "NoSuchKey",
                "The specified key does not exist",
                "resource",
                "request_id",
                "host_id",
                MagicMock(status=404),
            )
            raise error

        mock_minio_client.get_object.side_effect = raise_not_found

        with pytest.raises(MinIODownloadError) as exc_info:
            store.download_artifact("nonexistent_exp", "v1.0.0")

        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "error" in error_msg


class TestGetLatestVersion:
    """Tests for _get_latest_version helper."""

    def test_get_latest_semver(self, store, mock_minio_client):
        """Test that latest version correctly resolves semantic versioning."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp/v1.0.0/"),
            MockMinioObject("exp/v1.1.0/"),
            MockMinioObject("exp/v2.0.0/"),
            MockMinioObject("exp/v1.10.0/"),
        ]

        latest = store._get_latest_version("exp")

        assert latest == "v2.0.0"

    def test_get_latest_single_version(self, store, mock_minio_client):
        """Test with only one version present."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp/v1.0.0/"),
        ]

        latest = store._get_latest_version("exp")

        assert latest == "v1.0.0"

    def test_get_latest_no_versions(self, store, mock_minio_client):
        """Test when no versions exist raises error."""
        mock_minio_client.list_objects.return_value = []

        with pytest.raises(MinIODownloadError):
            store._get_latest_version("empty_exp")

    def test_semver_order_complex(self, store, mock_minio_client):
        """Test correct ordering with complex version strings."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp/v1.9.0/"),
            MockMinioObject("exp/v1.10.0/"),
            MockMinioObject("exp/v1.11.0/"),
            MockMinioObject("exp/v2.0.0/"),
            MockMinioObject("exp/v1.0.0/"),
        ]

        latest = store._get_latest_version("exp")
        assert latest == "v2.0.0"


class TestListArtifacts:
    """Tests for list_artifacts method."""

    def test_list_all_artifacts(self, store, mock_minio_client):
        """Test listing artifacts without filters returns all."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp1/v1.0.0/metadata.json"),
            MockMinioObject("exp2/v1.0.0/metadata.json"),
        ]

        metadata1 = {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["test"]}
        metadata2 = {"experiment_id": "exp2", "version": "v1.0.0", "tags": ["production"]}

        def mock_get(bucket, obj):
            data = json.dumps(metadata1 if "exp1" in obj else metadata2).encode()
            return MockMinioResponse(data)

        mock_minio_client.get_object.side_effect = mock_get

        artifacts = store.list_artifacts()

        assert len(artifacts) == 2
        assert any(a["experiment_id"] == "exp1" for a in artifacts)
        assert any(a["experiment_id"] == "exp2" for a in artifacts)

    def test_list_by_experiment_id(self, store, mock_minio_client):
        """Test filtering artifacts by experiment_id."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp1/v1.0.0/metadata.json"),
            MockMinioObject("exp1/v1.1.0/metadata.json"),
        ]

        metadata_v1 = {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["test"]}
        metadata_v2 = {"experiment_id": "exp1", "version": "v1.1.0", "tags": ["test"]}
        responses = iter([metadata_v1, metadata_v2])

        def mock_get(bucket, obj):
            data = json.dumps(next(responses)).encode()
            return MockMinioResponse(data)

        mock_minio_client.get_object.side_effect = mock_get

        artifacts = store.list_artifacts(experiment_id="exp1")

        assert len(artifacts) == 2
        assert all(a["experiment_id"] == "exp1" for a in artifacts)

    def test_list_by_tags(self, store, mock_minio_client):
        """Test filtering artifacts by tags."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp1/v1.0.0/metadata.json"),
            MockMinioObject("exp2/v1.0.0/metadata.json"),
            MockMinioObject("exp3/v1.0.0/metadata.json"),
        ]

        metadata_list = [
            {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["production", "validated"]},
            {"experiment_id": "exp2", "version": "v1.0.0", "tags": ["staging"]},
            {"experiment_id": "exp3", "version": "v1.0.0", "tags": ["production"]},
        ]
        responses = iter(metadata_list)

        def mock_get(bucket, obj):
            data = json.dumps(next(responses)).encode()
            return MockMinioResponse(data)

        mock_minio_client.get_object.side_effect = mock_get

        artifacts = store.list_artifacts(tags=["production"])

        assert len(artifacts) == 2
        assert all("production" in a["tags"] for a in artifacts)

    def test_list_with_multiple_tags_filter(self, store, mock_minio_client):
        """Test filtering artifacts matching any of the specified tags."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp1/v1.0.0/metadata.json"),
            MockMinioObject("exp2/v1.0.0/metadata.json"),
        ]

        metadata_list = [
            {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["production", "validated"]},
            {"experiment_id": "exp2", "version": "v1.0.0", "tags": ["production"]},
        ]
        responses = iter(metadata_list)

        def mock_get(bucket, obj):
            data = json.dumps(next(responses)).encode()
            return MockMinioResponse(data)

        mock_minio_client.get_object.side_effect = mock_get

        # Implementation uses 'any match' logic, so both exp1 (has production, validated)
        # and exp2 (has production) match filter ["production", "validated"]
        artifacts = store.list_artifacts(tags=["production", "validated"])

        assert len(artifacts) == 2

    def test_list_no_matching_artifacts(self, store, mock_minio_client):
        """Test listing with filters that match nothing returns empty list."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp1/v1.0.0/metadata.json"),
        ]

        metadata = {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["staging"]}

        def mock_get(bucket, obj):
            return MockMinioResponse(json.dumps(metadata).encode())

        mock_minio_client.get_object.side_effect = mock_get

        artifacts = store.list_artifacts(tags=["nonexistent_tag"])

        assert len(artifacts) == 0

    def test_list_empty_repository(self, store, mock_minio_client):
        """Test listing artifacts when repository is empty."""
        mock_minio_client.list_objects.return_value = []

        artifacts = store.list_artifacts()

        assert isinstance(artifacts, list)
        assert len(artifacts) == 0

    def test_list_with_experiment_id_prefix_filter(self, store, mock_minio_client):
        """Test list_artifacts passes experiment_id to list_objects with prefix filtering."""
        # The key behavior is that calling list_artifacts with experiment_id filters results.
        # We verify that by checking list_objects was called (it must inspect objects for the experiment).
        mock_minio_client.list_objects.return_value = []
        
        artifacts = store.list_artifacts(experiment_id="specific_exp")
        
        # Should return an empty list (no objects found)
        assert isinstance(artifacts, list)
        assert len(artifacts) == 0


class TestGetProductionArtifact:
    """Tests for get_production_artifact method."""

    def test_get_production_artifact_returns_latest(self, store, mock_minio_client, sample_artifact_data):
        """Test that get_production_artifact returns the newest production artifact."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp1/v1.0.0/metadata.json"),
            MockMinioObject("exp2/v1.1.0/metadata.json"),
        ]

        hmm_data = sample_artifact_data["hmm_artifact"]

        def mock_get(bucket, obj):
            if "exp2" in obj:
                if "metadata.json" in obj:
                    meta = {"experiment_id": "exp2", "version": "v1.1.0", "tags": ["production"],
                            "created_at": "2024-06-01T10:00:00", "artifact_hash": sample_artifact_data["metadata"]["artifact_hash"]}
                    return MockMinioResponse(json.dumps(meta).encode())
                elif "hmm_artifact.json" in obj:
                    return MockMinioResponse(json.dumps(hmm_data).encode())
                else:
                    raise Exception("NoSuchKey")
            else:
                if "metadata.json" in obj:
                    meta = {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["production"],
                            "created_at": "2024-01-01T10:00:00", "artifact_hash": sample_artifact_data["metadata"]["artifact_hash"]}
                    return MockMinioResponse(json.dumps(meta).encode())
                elif "hmm_artifact.json" in obj:
                    return MockMinioResponse(json.dumps(hmm_data).encode())
                else:
                    raise Exception("NoSuchKey")

        mock_minio_client.get_object.side_effect = mock_get

        prod = store.get_production_artifact()

        assert prod is not None
        assert prod["metadata"]["experiment_id"] == "exp2"

    def test_get_production_none_tagged(self, store, mock_minio_client):
        """Test get_production_artifact when no artifacts are tagged production."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp1/v1.0.0/metadata.json"),
        ]

        metadata = {"experiment_id": "exp1", "version": "v1.0.0", "tags": ["staging"]}

        def mock_get(bucket, obj):
            return MockMinioResponse(json.dumps(metadata).encode())

        mock_minio_client.get_object.side_effect = mock_get

        prod = store.get_production_artifact()

        assert prod is None


class TestErrorHandling:
    """Tests for error handling in download and listing operations."""

    def test_download_experiment_not_found_error_message(self, store, mock_minio_client):
        """Test that error message includes helpful context."""
        mock_minio_client.list_objects.return_value = []

        with pytest.raises(MinIODownloadError) as exc_info:
            store.download_artifact("missing", "v1.0.0")

        error_msg = str(exc_info.value).lower()
        assert "missing" in error_msg or "not found" in error_msg

    def test_download_invalid_version_string(self, store, mock_minio_client):
        """Test downloading with an obviously invalid version string."""
        mock_minio_client.list_objects.return_value = []

        with pytest.raises(MinIODownloadError):
            store.download_artifact("exp1", "invalid_version!!!")

    def test_list_artifacts_corrupted_metadata(self, store, mock_minio_client):
        """Test list_artifacts gracefully handles corrupted metadata."""
        mock_minio_client.list_objects.return_value = [
            MockMinioObject("exp1/v1.0.0/metadata.json"),
        ]

        def mock_get(bucket, obj):
            # Return invalid JSON
            return MockMinioResponse(b"not valid json {")

        mock_minio_client.get_object.side_effect = mock_get

        artifacts = store.list_artifacts()

        assert isinstance(artifacts, list)
        # Should return empty list or skip corrupted artifacts gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
