# Artifact Storage with MinIO Design Document

## Overview

This design extends the existing artifact management system with MinIO integration for persistent, versioned storage of HMM models and fusion weights. The implementation focuses on simplicity, reliability, and seamless integration with existing components.

## Architecture

### Current Foundation

```python
# Existing Components (py/imp/hmm/)
├── models.py                    # ✅ HMMArtifact, FusionWeights
├── artifact_management.py       # ✅ ResearchArtifact, ExperimentTracker (local)
│                                # ✅ ArtifactValidator, ArtifactExporter
└── trainer.py                   # ✅ HMM training with artifact generation

# Infrastructure
└── docker-compose.yml           # ✅ MinIO service configured
```

### Enhanced Architecture with MinIO

```
┌─────────────────────────────────────────────────────────────┐
│              Research Environment (Python)                  │
├─────────────────────────────────────────────────────────────┤
│  Train HMM → Generate Artifact → Validate → Store          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MinIOArtifactStore                             │
├─────────────────────────────────────────────────────────────┤
│  • Upload artifacts with versioning                         │
│  • Download by version or tag                               │
│  • List and search artifacts                                │
│  • Tag management (production, staging, etc.)               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MinIO Object Storage                           │
├─────────────────────────────────────────────────────────────┤
│  Bucket: hmm-artifacts/                                     │
│    ├── exp_001/                                             │
│    │   ├── v1.0.0/                                          │
│    │   │   ├── hmm_artifact.json                            │
│    │   │   ├── fusion_weights.json                          │
│    │   │   └── metadata.json                                │
│    │   └── v1.1.0/                                          │
│    └── exp_002/                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. MinIO Configuration

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MinIOConfig:
    """Configuration for MinIO connection."""
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin123"
    secure: bool = False
    bucket_name: str = "hmm-artifacts"
    
    @classmethod
    def from_env(cls) -> "MinIOConfig":
        """Load configuration from environment variables."""
        import os
        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            bucket_name=os.getenv("MINIO_BUCKET", "hmm-artifacts")
        )
```

### 2. MinIOArtifactStore

```python
from minio import Minio
from minio.error import S3Error
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MinIOArtifactStore:
    """Store and retrieve HMM artifacts from MinIO."""
    
    def __init__(self, config: Optional[MinIOConfig] = None):
        """
        Initialize MinIO artifact store.
        
        Args:
            config: MinIO configuration (defaults to env vars)
        """
        self.config = config or MinIOConfig.from_env()
        self.client = Minio(
            self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure
        )
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.config.bucket_name):
                self.client.make_bucket(self.config.bucket_name)
                logger.info(f"Created bucket: {self.config.bucket_name}")
        except S3Error as e:
            logger.error(f"Failed to create bucket: {e}")
            raise
    
    def upload_artifact(
        self,
        research_artifact: ResearchArtifact,
        fusion_weights: Optional[FusionWeights] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Upload research artifact to MinIO.
        
        Args:
            research_artifact: Research artifact to upload
            fusion_weights: Optional fusion weights
            tags: Optional tags (e.g., ["production", "validated"])
            
        Returns:
            Dictionary with uploaded object paths
        """
        experiment_id = research_artifact.research_metadata.experiment_id
        version = research_artifact.artifact_version
        
        # Create paths
        base_path = f"{experiment_id}/{version}"
        
        uploaded_paths = {}
        
        # Upload HMM artifact
        hmm_path = f"{base_path}/hmm_artifact.json"
        self._upload_json(
            hmm_path,
            research_artifact.base_artifact.model_dump()
        )
        uploaded_paths["hmm_artifact"] = hmm_path
        
        # Upload fusion weights if provided
        if fusion_weights:
            weights_path = f"{base_path}/fusion_weights.json"
            self._upload_json(
                weights_path,
                fusion_weights.model_dump()
            )
            uploaded_paths["fusion_weights"] = weights_path
        
        # Upload metadata
        metadata = {
            "experiment_id": experiment_id,
            "version": version,
            "researcher": research_artifact.research_metadata.researcher,
            "created_at": research_artifact.research_metadata.created_at.isoformat(),
            "training_config": research_artifact.research_metadata.training_config,
            "evaluation_metrics": research_artifact.research_metadata.evaluation_metrics,
            "tags": tags or research_artifact.research_metadata.tags,
            "production_ready": research_artifact.production_ready,
            "artifact_hash": research_artifact.artifact_hash
        }
        metadata_path = f"{base_path}/metadata.json"
        self._upload_json(metadata_path, metadata)
        uploaded_paths["metadata"] = metadata_path
        
        logger.info(f"Uploaded artifact {experiment_id} v{version} to MinIO")
        return uploaded_paths
    
    def download_artifact(
        self,
        experiment_id: str,
        version: str = "latest"
    ) -> Dict[str, Any]:
        """
        Download artifact from MinIO.
        
        Args:
            experiment_id: Experiment identifier
            version: Version to download ("latest" for most recent)
            
        Returns:
            Dictionary with artifact components
        """
        # Resolve "latest" version
        if version == "latest":
            version = self._get_latest_version(experiment_id)
        
        base_path = f"{experiment_id}/{version}"
        
        # Download components
        artifact_data = {}
        
        # Download HMM artifact
        hmm_path = f"{base_path}/hmm_artifact.json"
        artifact_data["hmm_artifact"] = self._download_json(hmm_path)
        
        # Download fusion weights (optional)
        weights_path = f"{base_path}/fusion_weights.json"
        try:
            artifact_data["fusion_weights"] = self._download_json(weights_path)
        except S3Error:
            artifact_data["fusion_weights"] = None
        
        # Download metadata
        metadata_path = f"{base_path}/metadata.json"
        artifact_data["metadata"] = self._download_json(metadata_path)
        
        logger.info(f"Downloaded artifact {experiment_id} v{version} from MinIO")
        return artifact_data
    
    def list_artifacts(
        self,
        experiment_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available artifacts.
        
        Args:
            experiment_id: Filter by experiment ID
            tags: Filter by tags
            
        Returns:
            List of artifact metadata
        """
        artifacts = []
        
        # List all objects
        prefix = f"{experiment_id}/" if experiment_id else ""
        objects = self.client.list_objects(
            self.config.bucket_name,
            prefix=prefix,
            recursive=True
        )
        
        # Find metadata files
        metadata_paths = [
            obj.object_name for obj in objects
            if obj.object_name.endswith("metadata.json")
        ]
        
        # Download and filter metadata
        for metadata_path in metadata_paths:
            try:
                metadata = self._download_json(metadata_path)
                
                # Apply tag filter
                if tags:
                    artifact_tags = metadata.get("tags", [])
                    if not any(tag in artifact_tags for tag in tags):
                        continue
                
                artifacts.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        
        return artifacts
    
    def tag_artifact(
        self,
        experiment_id: str,
        version: str,
        tag: str
    ) -> None:
        """
        Add tag to artifact (e.g., "production").
        
        Args:
            experiment_id: Experiment identifier
            version: Version to tag
            tag: Tag to add
        """
        metadata_path = f"{experiment_id}/{version}/metadata.json"
        
        # Download current metadata
        metadata = self._download_json(metadata_path)
        
        # Add tag
        tags = metadata.get("tags", [])
        if tag not in tags:
            tags.append(tag)
            metadata["tags"] = tags
            
            # Upload updated metadata
            self._upload_json(metadata_path, metadata)
            logger.info(f"Tagged {experiment_id} v{version} with '{tag}'")
    
    def get_production_artifact(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest production-tagged artifact.
        
        Returns:
            Artifact data or None if no production artifact exists
        """
        artifacts = self.list_artifacts(tags=["production"])
        
        if not artifacts:
            logger.warning("No production artifacts found")
            return None
        
        # Sort by created_at and get latest
        artifacts.sort(key=lambda x: x["created_at"], reverse=True)
        latest = artifacts[0]
        
        return self.download_artifact(
            latest["experiment_id"],
            latest["version"]
        )
    
    def _upload_json(self, object_name: str, data: Dict[str, Any]) -> None:
        """Upload JSON data to MinIO."""
        import io
        
        json_bytes = json.dumps(data, indent=2, default=str).encode('utf-8')
        json_stream = io.BytesIO(json_bytes)
        
        self.client.put_object(
            self.config.bucket_name,
            object_name,
            json_stream,
            length=len(json_bytes),
            content_type="application/json"
        )
    
    def _download_json(self, object_name: str) -> Dict[str, Any]:
        """Download JSON data from MinIO."""
        response = self.client.get_object(
            self.config.bucket_name,
            object_name
        )
        data = json.loads(response.read().decode('utf-8'))
        response.close()
        response.release_conn()
        return data
    
    def _get_latest_version(self, experiment_id: str) -> str:
        """Get latest version for experiment."""
        objects = self.client.list_objects(
            self.config.bucket_name,
            prefix=f"{experiment_id}/",
            recursive=False
        )
        
        versions = [
            obj.object_name.split('/')[1]
            for obj in objects
            if '/' in obj.object_name
        ]
        
        if not versions:
            raise ValueError(f"No versions found for experiment {experiment_id}")
        
        # Sort versions (semantic versioning)
        versions.sort(key=lambda v: [int(x) for x in v.lstrip('v').split('.')])
        return versions[-1]
```

### 3. Integration with ExperimentTracker

```python
# Extend existing ExperimentTracker to support MinIO

class ExperimentTracker:
    """Track and manage research experiments with local and MinIO storage."""
    
    def __init__(
        self,
        experiment_dir: Union[str, Path],
        use_minio: bool = True,
        minio_config: Optional[MinIOConfig] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory for local storage
            use_minio: Whether to use MinIO for storage
            minio_config: MinIO configuration
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_minio = use_minio
        if use_minio:
            self.minio_store = MinIOArtifactStore(minio_config)
        else:
            self.minio_store = None
        
        # Existing local storage code...
    
    def log_experiment(
        self,
        artifact: ResearchArtifact,
        fusion_weights: Optional[FusionWeights] = None,
        notes: str = "",
        upload_to_minio: bool = True
    ) -> str:
        """
        Log experiment with optional MinIO upload.
        
        Args:
            artifact: Research artifact
            fusion_weights: Optional fusion weights
            notes: Additional notes
            upload_to_minio: Whether to upload to MinIO
            
        Returns:
            Experiment ID
        """
        # Existing local storage code...
        
        # Upload to MinIO if enabled
        if upload_to_minio and self.minio_store:
            try:
                self.minio_store.upload_artifact(
                    artifact,
                    fusion_weights,
                    tags=artifact.research_metadata.tags
                )
            except Exception as e:
                logger.error(f"Failed to upload to MinIO: {e}")
                # Continue - local storage succeeded
        
        return experiment_id
```

## Implementation Considerations

### MinIO Connection

- Use environment variables for configuration
- Support both local (docker-compose) and remote MinIO
- Handle connection failures gracefully

### Versioning Strategy

- Use semantic versioning (v1.0.0, v1.1.0, v2.0.0)
- Support "latest" alias for convenience
- Store version in metadata for tracking

### Storage Organization

```
hmm-artifacts/
├── exp_001/
│   ├── v1.0.0/
│   │   ├── hmm_artifact.json
│   │   ├── fusion_weights.json
│   │   └── metadata.json
│   └── v1.1.0/
│       ├── hmm_artifact.json
│       ├── fusion_weights.json
│       └── metadata.json
└── exp_002/
    └── v1.0.0/
        ├── hmm_artifact.json
        └── metadata.json
```

### Error Handling

- Retry uploads with exponential backoff
- Validate artifacts before upload
- Provide clear error messages
- Fall back to local storage if MinIO unavailable

### Production Deployment

1. Train and validate model
2. Upload to MinIO with "staging" tag
3. Test in staging environment
4. Tag as "production" when validated
5. Rust inference engine fetches "production" artifact
