"""
Tests for MinIO artifact storage functionality.
"""

import pytest
import json
import os
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from imp.hmm.artifact_management import (
    MinIOConfig,
    MinIOArtifactStore,
    MinIOConnectionError,
    MinIOUploadError,
    MinIODownloadError,
)


class TestMinIOConfig:
    """Test MinIOConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MinIOConfig()
        
        assert config.endpoint == "localhost:9000"
        assert config.access_key == "minioadmin"
        assert config.secret_key == "minioadmin123"
        assert config.secure is False
        assert config.bucket_name == "hmm-artifacts"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MinIOConfig(
            endpoint="minio.example.com:9000",
            access_key="custom_key",
            secret_key="custom_secret",
            secure=True,
            bucket_name="custom-bucket"
        )
        
        assert config.endpoint == "minio.example.com:9000"
        assert config.access_key == "custom_key"
        assert config.secret_key == "custom_secret"
        assert config.secure is True
        assert config.bucket_name == "custom-bucket"
    
    def test_from_env(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "MINIO_ENDPOINT": "env.minio.com:9000",
            "MINIO_ACCESS_KEY": "env_key",
            "MINIO_SECRET_KEY": "env_secret",
            "MINIO_SECURE": "true",
            "MINIO_BUCKET": "env-bucket"
        }
        
        with patch.dict(os.environ, env_vars):
            config = MinIOConfig.from_env()
            
            assert config.endpoint == "env.minio.com:9000"
            assert config.access_key == "env_key"
            assert config.secret_key == "env_secret"
            assert config.secure is True
            assert config.bucket_name == "env-bucket"
    
    def test_from_env_defaults(self):
        """Test from_env uses defaults when env vars not set."""
        # Clear any existing env vars
        env_vars = {
            "MINIO_ENDPOINT": "",
            "MINIO_ACCESS_KEY": "",
            "MINIO_SECRET_KEY": "",
            "MINIO_SECURE": "",
            "MINIO_BUCKET": ""
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = MinIOConfig.from_env()
            
            assert config.endpoint == "localhost:9000"
            assert config.access_key == "minioadmin"
            assert config.secret_key == "minioadmin123"
            assert config.secure is False
            assert config.bucket_name == "hmm-artifacts"


class TestMinIOArtifactStore:
    """Test MinIOArtifactStore class."""
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_initialization_success(self, mock_minio_class):
        """Test successful initialization."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        mock_minio_class.return_value = mock_client
        
        config = MinIOConfig()
        store = MinIOArtifactStore(config)
        
        assert store.config == config
        assert store.client == mock_client
        mock_minio_class.assert_called_once_with(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin123",
            secure=False
        )
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', False)
    def test_initialization_minio_not_available(self):
        """Test initialization fails when MinIO not available."""
        with pytest.raises(MinIOConnectionError, match="MinIO library not available"):
            MinIOArtifactStore()
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_ensure_bucket_exists_creates_bucket(self, mock_minio_class):
        """Test bucket creation when it doesn't exist."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = False
        mock_client.list_buckets.return_value = []
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        mock_client.bucket_exists.assert_called_with("hmm-artifacts")
        mock_client.make_bucket.assert_called_once_with("hmm-artifacts")
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_ensure_bucket_exists_already_exists(self, mock_minio_class):
        """Test bucket not created when it already exists."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        mock_client.bucket_exists.assert_called_with("hmm-artifacts")
        mock_client.make_bucket.assert_not_called()
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_validate_connection_success(self, mock_minio_class):
        """Test successful connection validation."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = [Mock(), Mock()]
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        mock_client.list_buckets.assert_called_once()
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    @patch('imp.hmm.artifact_management.S3Error', Exception)
    def test_validate_connection_failure(self, mock_minio_class):
        """Test connection validation failure."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.side_effect = Exception("Connection failed")
        mock_minio_class.return_value = mock_client
        
        with pytest.raises(MinIOConnectionError, match="connection validation failed"):
            MinIOArtifactStore()
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_upload_json_success(self, mock_minio_class):
        """Test successful JSON upload."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        test_data = {"key": "value", "number": 42}
        store._upload_json("test/object.json", test_data)
        
        # Verify put_object was called
        assert mock_client.put_object.called
        call_args = mock_client.put_object.call_args
        assert call_args[0][0] == "hmm-artifacts"
        assert call_args[0][1] == "test/object.json"
        assert call_args[1]["content_type"] == "application/json"
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    @patch('imp.hmm.artifact_management.S3Error', Exception)
    def test_upload_json_retry_logic(self, mock_minio_class):
        """Test upload retry logic with exponential backoff."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        
        # Fail twice, succeed on third attempt
        mock_client.put_object.side_effect = [
            Exception("Temporary error"),
            Exception("Temporary error"),
            None
        ]
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        test_data = {"key": "value"}
        store._upload_json("test/retry.json", test_data, max_retries=3, retry_delay=0.01)
        
        # Should have been called 3 times
        assert mock_client.put_object.call_count == 3
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    @patch('imp.hmm.artifact_management.S3Error', Exception)
    def test_upload_json_max_retries_exceeded(self, mock_minio_class):
        """Test upload fails after max retries."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        mock_client.put_object.side_effect = Exception("Persistent error")
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        test_data = {"key": "value"}
        with pytest.raises(MinIOUploadError, match="after 3 attempts"):
            store._upload_json("test/fail.json", test_data, max_retries=3, retry_delay=0.01)
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_download_json_success(self, mock_minio_class):
        """Test successful JSON download."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        
        # Mock response object
        test_data = {"key": "value", "number": 42}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(test_data).encode('utf-8')
        mock_client.get_object.return_value = mock_response
        
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        result = store._download_json("test/object.json")
        
        assert result == test_data
        mock_client.get_object.assert_called_once_with("hmm-artifacts", "test/object.json")
        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_called_once()
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_download_json_not_found(self, mock_minio_class):
        """Test download fails when object not found."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        
        # Create a real exception with code attribute
        error = Exception("NoSuchKey")
        error.code = "NoSuchKey"
        mock_client.get_object.side_effect = error
        
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        # The error will be caught as a generic exception since we can't properly mock S3Error
        with pytest.raises(MinIODownloadError):
            store._download_json("test/missing.json")
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_download_json_invalid_json(self, mock_minio_class):
        """Test download fails with invalid JSON."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        
        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.read.return_value = b"not valid json {"
        mock_client.get_object.return_value = mock_response
        
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        with pytest.raises(MinIODownloadError, match="Invalid JSON"):
            store._download_json("test/invalid.json")


class TestUploadArtifact:
    """Test upload_artifact method."""
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_upload_artifact_success(self, mock_minio_class):
        """Test successful artifact upload."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        # Create mock research artifact
        from imp.hmm.models import HMMArtifact, FusionWeights
        from imp.hmm.artifact_management import ResearchArtifact, ResearchMetadata
        
        hmm_artifact = HMMArtifact(
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
        
        research_metadata = ResearchMetadata(
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={"n_iter": 100},
            library_used="hmmlearn",
            evaluation_metrics={"log_likelihood": -1234.5}
        )
        
        research_artifact = ResearchArtifact(
            base_artifact=hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v1.0.0",
            artifact_hash="abc123",
            production_ready=False
        )
        
        fusion_weights = FusionWeights(
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
        
        # Upload artifact
        result = store.upload_artifact(
            research_artifact,
            fusion_weights=fusion_weights,
            tags=["test", "validated"]
        )
        
        # Verify result structure
        assert "hmm_artifact" in result
        assert "fusion_weights" in result
        assert "metadata" in result
        
        assert result["hmm_artifact"] == "exp_001/v1.0.0/hmm_artifact.json"
        assert result["fusion_weights"] == "exp_001/v1.0.0/fusion_weights.json"
        assert result["metadata"] == "exp_001/v1.0.0/metadata.json"
        
        # Verify put_object was called 3 times (hmm, weights, metadata)
        assert mock_client.put_object.call_count == 3
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_upload_artifact_without_fusion_weights(self, mock_minio_class):
        """Test artifact upload without fusion weights."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        # Create mock research artifact
        from imp.hmm.models import HMMArtifact
        from imp.hmm.artifact_management import ResearchArtifact, ResearchMetadata
        
        hmm_artifact = HMMArtifact(
            version="1.0.0",
            n_states=2,
            n_features=1,
            transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
            initial_probabilities=[0.5, 0.5],
            means=[[1.0], [2.0]],
            covariances=[[[1.0]], [[1.0]]],
            training_window_start=int(datetime(2024, 1, 1).timestamp()),
            training_window_end=int(datetime(2024, 12, 31).timestamp()),
            metadata={"library": "hmmlearn"}
        )
        
        research_metadata = ResearchMetadata(
            experiment_id="exp_002",
            researcher="test_researcher",
            training_config={},
            library_used="hmmlearn"
        )
        
        research_artifact = ResearchArtifact(
            base_artifact=hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v1.0.0",
            artifact_hash="def456"
        )
        
        # Upload artifact without fusion weights
        result = store.upload_artifact(research_artifact)
        
        # Verify result structure
        assert "hmm_artifact" in result
        assert "fusion_weights" not in result
        assert "metadata" in result
        
        # Verify put_object was called 2 times (hmm, metadata)
        assert mock_client.put_object.call_count == 2
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    @patch('imp.hmm.artifact_management.S3Error', Exception)
    def test_upload_artifact_with_retry(self, mock_minio_class):
        """Test artifact upload with retry on failure."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        
        # Fail first upload, succeed on retry
        mock_client.put_object.side_effect = [
            Exception("Temporary error"),
            None,  # HMM artifact succeeds on retry
            None   # Metadata succeeds
        ]
        
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        # Create mock research artifact
        from imp.hmm.models import HMMArtifact
        from imp.hmm.artifact_management import ResearchArtifact, ResearchMetadata
        
        hmm_artifact = HMMArtifact(
            version="1.0.0",
            n_states=2,
            n_features=1,
            transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
            initial_probabilities=[0.5, 0.5],
            means=[[1.0], [2.0]],
            covariances=[[[1.0]], [[1.0]]],
            training_window_start=int(datetime(2024, 1, 1).timestamp()),
            training_window_end=int(datetime(2024, 12, 31).timestamp()),
            metadata={"library": "hmmlearn"}
        )
        
        research_metadata = ResearchMetadata(
            experiment_id="exp_003",
            researcher="test_researcher",
            training_config={},
            library_used="hmmlearn"
        )
        
        research_artifact = ResearchArtifact(
            base_artifact=hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v1.0.0",
            artifact_hash="ghi789"
        )
        
        # Upload with retry
        result = store.upload_artifact(
            research_artifact,
            max_retries=3,
            retry_delay=0.01
        )
        
        # Should succeed after retry
        assert "hmm_artifact" in result
        assert "metadata" in result
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    @patch('imp.hmm.artifact_management.S3Error', Exception)
    def test_upload_artifact_failure_after_retries(self, mock_minio_class):
        """Test artifact upload fails after max retries."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        mock_client.put_object.side_effect = Exception("Persistent error")
        
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        # Create mock research artifact
        from imp.hmm.models import HMMArtifact
        from imp.hmm.artifact_management import ResearchArtifact, ResearchMetadata
        
        hmm_artifact = HMMArtifact(
            version="1.0.0",
            n_states=2,
            n_features=1,
            transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
            initial_probabilities=[0.5, 0.5],
            means=[[1.0], [2.0]],
            covariances=[[[1.0]], [[1.0]]],
            training_window_start=int(datetime(2024, 1, 1).timestamp()),
            training_window_end=int(datetime(2024, 12, 31).timestamp()),
            metadata={"library": "hmmlearn"}
        )
        
        research_metadata = ResearchMetadata(
            experiment_id="exp_004",
            researcher="test_researcher",
            training_config={},
            library_used="hmmlearn"
        )
        
        research_artifact = ResearchArtifact(
            base_artifact=hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v1.0.0",
            artifact_hash="jkl012"
        )
        
        # Upload should fail
        with pytest.raises(MinIOUploadError):
            store.upload_artifact(
                research_artifact,
                max_retries=2,
                retry_delay=0.01
            )
    
    @patch('imp.hmm.artifact_management.MINIO_AVAILABLE', True)
    @patch('imp.hmm.artifact_management.Minio')
    def test_upload_artifact_structured_paths(self, mock_minio_class):
        """Test that upload creates correct structured paths."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.list_buckets.return_value = []
        mock_minio_class.return_value = mock_client
        
        store = MinIOArtifactStore()
        
        # Create mock research artifact
        from imp.hmm.models import HMMArtifact
        from imp.hmm.artifact_management import ResearchArtifact, ResearchMetadata
        
        hmm_artifact = HMMArtifact(
            version="1.0.0",
            n_states=2,
            n_features=1,
            transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
            initial_probabilities=[0.5, 0.5],
            means=[[1.0], [2.0]],
            covariances=[[[1.0]], [[1.0]]],
            training_window_start=int(datetime(2024, 1, 1).timestamp()),
            training_window_end=int(datetime(2024, 12, 31).timestamp()),
            metadata={"library": "hmmlearn"}
        )
        
        research_metadata = ResearchMetadata(
            experiment_id="my_experiment",
            researcher="test_researcher",
            training_config={},
            library_used="hmmlearn"
        )
        
        research_artifact = ResearchArtifact(
            base_artifact=hmm_artifact,
            research_metadata=research_metadata,
            artifact_version="v2.1.3",
            artifact_hash="xyz999"
        )
        
        # Upload artifact
        result = store.upload_artifact(research_artifact)
        
        # Verify structured paths
        assert result["hmm_artifact"] == "my_experiment/v2.1.3/hmm_artifact.json"
        assert result["metadata"] == "my_experiment/v2.1.3/metadata.json"
        
        # Verify the actual calls to put_object have correct paths
        calls = mock_client.put_object.call_args_list
        paths = [call[0][1] for call in calls]
        
        assert "my_experiment/v2.1.3/hmm_artifact.json" in paths
        assert "my_experiment/v2.1.3/metadata.json" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
