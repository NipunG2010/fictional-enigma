"""
Production integration and artifact management for HMM research environment.

This module provides enhanced artifact management with research-specific metadata,
experiment tracking, versioning, and validation utilities for production deployment.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import json
import hashlib
import logging
import os
import io
import time
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    S3Error = Exception

from .models import HMMArtifact, FusionWeights

logger = logging.getLogger(__name__)


class ArtifactManagementError(Exception):
    """Base exception for artifact management errors."""
    pass


class ValidationError(ArtifactManagementError):
    """Exception raised during artifact validation."""
    pass


class ExportError(ArtifactManagementError):
    """Exception raised during artifact export."""
    pass


class MinIOConnectionError(ArtifactManagementError):
    """Exception raised for MinIO connection errors."""
    pass


class MinIOUploadError(ArtifactManagementError):
    """Exception raised for MinIO upload errors."""
    pass


class MinIODownloadError(ArtifactManagementError):
    """Exception raised for MinIO download errors."""
    pass


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
        """
        Load configuration from environment variables.
        
        Returns:
            MinIOConfig instance with values from environment
        """
        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT") or "localhost:9000",
            access_key=os.getenv("MINIO_ACCESS_KEY") or "minioadmin",
            secret_key=os.getenv("MINIO_SECRET_KEY") or "minioadmin123",
            secure=(os.getenv("MINIO_SECURE") or "false").lower() == "true",
            bucket_name=os.getenv("MINIO_BUCKET") or "hmm-artifacts"
        )


class MinIOArtifactStore:
    """Store and retrieve HMM artifacts from MinIO with versioning support."""
    
    def __init__(self, config: Optional[MinIOConfig] = None):
        """
        Initialize MinIO artifact store.
        
        Args:
            config: MinIO configuration (defaults to env vars)
            
        Raises:
            MinIOConnectionError: If MinIO is not available or connection fails
        """
        if not MINIO_AVAILABLE:
            raise MinIOConnectionError(
                "MinIO library not available. Install with: pip install minio"
            )
        
        self.config = config or MinIOConfig.from_env()
        
        try:
            self.client = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure
            )
            logger.info(f"MinIO client initialized for endpoint: {self.config.endpoint}")
        except Exception as e:
            raise MinIOConnectionError(f"Failed to initialize MinIO client: {e}")
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        # Validate connection
        self._validate_connection()
    
    def _ensure_bucket_exists(self) -> None:
        """
        Create bucket if it doesn't exist.
        
        Raises:
            MinIOConnectionError: If bucket creation fails
        """
        try:
            if not self.client.bucket_exists(self.config.bucket_name):
                self.client.make_bucket(self.config.bucket_name)
                logger.info(f"Created MinIO bucket: {self.config.bucket_name}")
            else:
                logger.debug(f"MinIO bucket already exists: {self.config.bucket_name}")
        except S3Error as e:
            raise MinIOConnectionError(f"Failed to create/check bucket: {e}")
        except Exception as e:
            raise MinIOConnectionError(f"Unexpected error with bucket: {e}")
    
    def _validate_connection(self) -> None:
        """
        Validate MinIO connection by listing buckets.
        
        Raises:
            MinIOConnectionError: If connection validation fails
        """
        try:
            # Try to list buckets as a connection test
            buckets = self.client.list_buckets()
            logger.info(f"MinIO connection validated. Found {len(buckets)} bucket(s)")
        except S3Error as e:
            raise MinIOConnectionError(f"MinIO connection validation failed: {e}")
        except Exception as e:
            raise MinIOConnectionError(f"Unexpected error validating connection: {e}")
    
    def _upload_json(
        self,
        object_name: str,
        data: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> None:
        """
        Upload JSON data to MinIO with retry logic.
        
        Args:
            object_name: Object path in bucket
            data: Dictionary to upload as JSON
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            
        Raises:
            MinIOUploadError: If upload fails after all retries
        """
        json_bytes = json.dumps(data, indent=2, default=str).encode('utf-8')
        json_stream = io.BytesIO(json_bytes)
        
        for attempt in range(max_retries):
            try:
                self.client.put_object(
                    self.config.bucket_name,
                    object_name,
                    json_stream,
                    length=len(json_bytes),
                    content_type="application/json"
                )
                logger.debug(f"Uploaded JSON to MinIO: {object_name}")
                return
            except S3Error as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Upload attempt {attempt + 1} failed for {object_name}. "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    time.sleep(wait_time)
                    json_stream.seek(0)  # Reset stream position
                else:
                    raise MinIOUploadError(
                        f"Failed to upload {object_name} after {max_retries} attempts: {e}"
                    )
            except Exception as e:
                raise MinIOUploadError(f"Unexpected error uploading {object_name}: {e}")
    
    def _download_json(self, object_name: str) -> Dict[str, Any]:
        """
        Download JSON data from MinIO.
        
        Args:
            object_name: Object path in bucket
            
        Returns:
            Dictionary loaded from JSON
            
        Raises:
            MinIODownloadError: If download fails
        """
        try:
            response = self.client.get_object(
                self.config.bucket_name,
                object_name
            )
            data = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            logger.debug(f"Downloaded JSON from MinIO: {object_name}")
            return data
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise MinIODownloadError(f"Object not found: {object_name}")
            raise MinIODownloadError(f"Failed to download {object_name}: {e}")
        except json.JSONDecodeError as e:
            raise MinIODownloadError(f"Invalid JSON in {object_name}: {e}")
        except Exception as e:
            raise MinIODownloadError(f"Unexpected error downloading {object_name}: {e}")
    
    def upload_artifact(
        self,
        research_artifact: "ResearchArtifact",
        fusion_weights: Optional["FusionWeights"] = None,
        tags: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, str]:
        """
        Upload research artifact to MinIO with versioning.
        
        Args:
            research_artifact: Research artifact to upload
            fusion_weights: Optional fusion weights
            tags: Optional tags (e.g., ["production", "validated"])
            max_retries: Maximum number of retry attempts for uploads
            retry_delay: Initial delay between retries (exponential backoff)
            
        Returns:
            Dictionary with uploaded object paths
            
        Raises:
            MinIOUploadError: If upload fails after all retries
        """
        experiment_id = research_artifact.research_metadata.experiment_id
        version = research_artifact.artifact_version
        
        # Create structured paths
        base_path = f"{experiment_id}/{version}"
        
        uploaded_paths = {}
        
        try:
            # Upload HMM artifact
            hmm_path = f"{base_path}/hmm_artifact.json"
            self._upload_json(
                hmm_path,
                research_artifact.base_artifact.model_dump(),
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            uploaded_paths["hmm_artifact"] = hmm_path
            logger.debug(f"Uploaded HMM artifact to {hmm_path}")
            
            # Upload fusion weights if provided
            if fusion_weights:
                weights_path = f"{base_path}/fusion_weights.json"
                self._upload_json(
                    weights_path,
                    fusion_weights.model_dump(),
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
                uploaded_paths["fusion_weights"] = weights_path
                logger.debug(f"Uploaded fusion weights to {weights_path}")
            
            # Prepare metadata
            metadata_tags = tags if tags is not None else research_artifact.research_metadata.tags
            metadata = {
                "experiment_id": experiment_id,
                "version": version,
                "researcher": research_artifact.research_metadata.researcher,
                "created_at": research_artifact.research_metadata.created_at.isoformat(),
                "training_config": research_artifact.research_metadata.training_config,
                "evaluation_metrics": research_artifact.research_metadata.evaluation_metrics,
                "tags": metadata_tags,
                "production_ready": research_artifact.production_ready,
                "artifact_hash": research_artifact.artifact_hash,
                "library_used": research_artifact.research_metadata.library_used,
                "n_states": research_artifact.base_artifact.n_states,
                "notebook_path": research_artifact.research_metadata.notebook_path,
                "notes": research_artifact.research_metadata.notes
            }
            
            # Upload metadata
            metadata_path = f"{base_path}/metadata.json"
            self._upload_json(
                metadata_path,
                metadata,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            uploaded_paths["metadata"] = metadata_path
            logger.debug(f"Uploaded metadata to {metadata_path}")
            
            logger.info(
                f"Successfully uploaded artifact {experiment_id} v{version} to MinIO "
                f"({len(uploaded_paths)} files)"
            )
            
            return uploaded_paths
            
        except MinIOUploadError:
            # Re-raise MinIOUploadError as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise MinIOUploadError(
                f"Unexpected error uploading artifact {experiment_id} v{version}: {e}"
            )
    
    def download_artifact(
        self,
        experiment_id: str,
        version: str = "latest",
        validate_integrity: bool = True
    ) -> Dict[str, Any]:
        """
        Download artifact from MinIO with integrity validation.
        
        Args:
            experiment_id: Experiment identifier
            version: Version to download ("latest" for most recent)
            validate_integrity: Whether to validate artifact hash after download
            
        Returns:
            Dictionary with artifact components and metadata
            
        Raises:
            MinIODownloadError: If download fails or artifact not found
            ValidationError: If integrity validation fails
        """
        try:
            # Resolve "latest" version
            if version == "latest":
                version = self._get_latest_version(experiment_id)
                logger.debug(f"Resolved 'latest' to version {version} for {experiment_id}")
            
            base_path = f"{experiment_id}/{version}"
            
            # Download components
            artifact_data = {}
            
            # Download metadata first to get artifact hash
            metadata_path = f"{base_path}/metadata.json"
            try:
                artifact_data["metadata"] = self._download_json(metadata_path)
            except MinIODownloadError as e:
                raise MinIODownloadError(
                    f"Artifact {experiment_id} v{version} not found. "
                    f"Available versions: {self._list_versions(experiment_id)}"
                )
            
            # Download HMM artifact
            hmm_path = f"{base_path}/hmm_artifact.json"
            artifact_data["hmm_artifact"] = self._download_json(hmm_path)
            
            # Download fusion weights (optional)
            weights_path = f"{base_path}/fusion_weights.json"
            try:
                artifact_data["fusion_weights"] = self._download_json(weights_path)
            except MinIODownloadError:
                artifact_data["fusion_weights"] = None
                logger.debug(f"No fusion weights found for {experiment_id} v{version}")
            
            # Validate integrity if requested
            if validate_integrity:
                expected_hash = artifact_data["metadata"].get("artifact_hash")
                if expected_hash:
                    # Reconstruct artifact to calculate hash
                    hmm_artifact = HMMArtifact(**artifact_data["hmm_artifact"])
                    artifact_str = json.dumps(hmm_artifact.model_dump(), sort_keys=True)
                    actual_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
                    
                    if actual_hash != expected_hash:
                        raise ValidationError(
                            f"Artifact integrity check failed for {experiment_id} v{version}. "
                            f"Expected hash: {expected_hash}, Actual hash: {actual_hash}"
                        )
                    logger.debug(f"Integrity validation passed for {experiment_id} v{version}")
                else:
                    logger.warning(f"No artifact hash in metadata for {experiment_id} v{version}")
            
            logger.info(f"Downloaded artifact {experiment_id} v{version} from MinIO")
            return artifact_data
            
        except (MinIODownloadError, ValidationError):
            # Re-raise these as-is
            raise
        except Exception as e:
            raise MinIODownloadError(
                f"Unexpected error downloading artifact {experiment_id} v{version}: {e}"
            )
    
    def _get_latest_version(self, experiment_id: str) -> str:
        """
        Get latest version for experiment using semantic versioning.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Latest version string (e.g., "v1.2.3")
            
        Raises:
            MinIODownloadError: If no versions found for experiment
        """
        try:
            versions = self._list_versions(experiment_id)
            
            if not versions:
                raise MinIODownloadError(
                    f"No versions found for experiment {experiment_id}"
                )
            
            # Sort versions using semantic versioning
            # Remove 'v' prefix if present and split into parts
            def version_key(v: str) -> tuple:
                v_clean = v.lstrip('v')
                try:
                    return tuple(int(x) for x in v_clean.split('.'))
                except ValueError:
                    # If not a valid semantic version, use string comparison
                    return (0, 0, 0)
            
            versions.sort(key=version_key)
            latest = versions[-1]
            
            logger.debug(f"Latest version for {experiment_id}: {latest}")
            return latest
            
        except MinIODownloadError:
            raise
        except Exception as e:
            raise MinIODownloadError(
                f"Failed to determine latest version for {experiment_id}: {e}"
            )
    
    def _list_versions(self, experiment_id: str) -> List[str]:
        """
        List all versions for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            List of version strings
        """
        try:
            objects = self.client.list_objects(
                self.config.bucket_name,
                prefix=f"{experiment_id}/",
                recursive=False
            )
            
            # Extract version directories
            versions = set()
            for obj in objects:
                # Object name format: experiment_id/version/
                parts = obj.object_name.rstrip('/').split('/')
                if len(parts) >= 2:
                    versions.add(parts[1])
            
            return sorted(list(versions))
            
        except S3Error as e:
            logger.warning(f"Failed to list versions for {experiment_id}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Unexpected error listing versions for {experiment_id}: {e}")
            return []
    
    def list_artifacts(
        self,
        experiment_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available artifacts with optional filtering.
        
        Args:
            experiment_id: Filter by experiment ID
            tags: Filter by tags (any match)
            
        Returns:
            List of artifact metadata dictionaries
            
        Raises:
            MinIODownloadError: If listing fails
        """
        try:
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
                except MinIODownloadError as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error loading metadata from {metadata_path}: {e}")
            
            logger.info(f"Listed {len(artifacts)} artifact(s) from MinIO")
            return artifacts
            
        except S3Error as e:
            raise MinIODownloadError(f"Failed to list artifacts: {e}")
        except Exception as e:
            raise MinIODownloadError(f"Unexpected error listing artifacts: {e}")
    
    def get_production_artifact(self, fallback_to_last_known: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the latest production-tagged artifact with fallback support.
        
        Args:
            fallback_to_last_known: If True, fall back to last known good version if current fails
            
        Returns:
            Artifact data dictionary or None if no production artifact exists
            
        Raises:
            MinIODownloadError: If download fails and no fallback available
        """
        try:
            # List all production artifacts
            artifacts = self.list_artifacts(tags=["production"])
            
            if not artifacts:
                logger.warning("No production artifacts found in MinIO")
                return None
            
            # Sort by created_at timestamp and get latest
            artifacts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Try to download artifacts in order (latest first)
            for idx, artifact_meta in enumerate(artifacts):
                try:
                    logger.info(
                        f"Attempting to download production artifact: {artifact_meta['experiment_id']} "
                        f"v{artifact_meta['version']}"
                    )
                    
                    artifact_data = self.download_artifact(
                        artifact_meta["experiment_id"],
                        artifact_meta["version"]
                    )
                    
                    if idx > 0:
                        logger.warning(
                            f"Fell back to previous production artifact: "
                            f"{artifact_meta['experiment_id']} v{artifact_meta['version']}"
                        )
                    
                    return artifact_data
                    
                except (MinIODownloadError, ValidationError) as e:
                    if idx < len(artifacts) - 1 and fallback_to_last_known:
                        logger.warning(
                            f"Failed to download {artifact_meta['experiment_id']} "
                            f"v{artifact_meta['version']}: {e}. Trying fallback..."
                        )
                        continue
                    else:
                        raise
            
            # If we get here, all downloads failed
            logger.error("All production artifacts failed to download")
            return None
            
        except MinIODownloadError:
            raise
        except Exception as e:
            raise MinIODownloadError(f"Failed to get production artifact: {e}")
    
    def tag_artifact(
        self,
        experiment_id: str,
        version: str,
        tag: str,
        validate_production: bool = True
    ) -> None:
        """
        Add tag to artifact with validation for production tags.
        
        Args:
            experiment_id: Experiment identifier
            version: Version to tag
            tag: Tag to add (e.g., "production", "staging", "experimental")
            validate_production: Whether to validate artifact before tagging as production
            
        Raises:
            ValidationError: If production tag validation fails
            MinIODownloadError: If artifact not found
        """
        metadata_path = f"{experiment_id}/{version}/metadata.json"
        
        # Download current metadata
        try:
            metadata = self._download_json(metadata_path)
        except MinIODownloadError:
            raise MinIODownloadError(
                f"Cannot tag artifact {experiment_id} v{version}: artifact not found"
            )
        
        # Validate for production tag
        if tag.lower() == "production" and validate_production:
            # Check if artifact is marked as production_ready
            if not metadata.get("production_ready", False):
                raise ValidationError(
                    f"Cannot tag {experiment_id} v{version} as 'production': "
                    f"artifact is not marked as production_ready. "
                    f"Run validation checks first."
                )
            
            # Download and validate the full artifact
            try:
                artifact_data = self.download_artifact(experiment_id, version, validate_integrity=True)
                
                # Reconstruct ResearchArtifact for validation
                hmm_artifact = HMMArtifact(**artifact_data["hmm_artifact"])
                research_metadata = ResearchMetadata(**artifact_data["metadata"])
                
                research_artifact = ResearchArtifact(
                    base_artifact=hmm_artifact,
                    research_metadata=research_metadata,
                    artifact_version=version,
                    artifact_hash=metadata.get("artifact_hash", ""),
                    production_ready=metadata.get("production_ready", False),
                    validation_status=metadata.get("validation_status", {})
                )
                
                # Run validation checks
                validation_results = ArtifactValidator.run_all_validations(research_artifact)
                
                if not validation_results["all_passed"]:
                    failed_checks = validation_results.get("failed_checks", [])
                    raise ValidationError(
                        f"Cannot tag {experiment_id} v{version} as 'production': "
                        f"validation failed. Failed checks: {', '.join(failed_checks)}"
                    )
                
                logger.info(f"Production validation passed for {experiment_id} v{version}")
                
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(
                    f"Failed to validate artifact {experiment_id} v{version} for production: {e}"
                )
        
        # Add tag
        tags = metadata.get("tags", [])
        if tag not in tags:
            tags.append(tag)
            metadata["tags"] = tags
            
            # Add tagging metadata
            if "tagging_history" not in metadata:
                metadata["tagging_history"] = []
            
            metadata["tagging_history"].append({
                "tag": tag,
                "tagged_at": datetime.now().isoformat(),
                "validated": tag.lower() == "production" and validate_production
            })
            
            # Upload updated metadata
            self._upload_json(metadata_path, metadata)
            logger.info(f"Tagged {experiment_id} v{version} with '{tag}'")
    
    def remove_tag(
        self,
        experiment_id: str,
        version: str,
        tag: str
    ) -> None:
        """
        Remove tag from artifact.
        
        Args:
            experiment_id: Experiment identifier
            version: Version to untag
            tag: Tag to remove
            
        Raises:
            MinIODownloadError: If artifact not found
        """
        metadata_path = f"{experiment_id}/{version}/metadata.json"
        
        # Download current metadata
        try:
            metadata = self._download_json(metadata_path)
        except MinIODownloadError:
            raise MinIODownloadError(
                f"Cannot remove tag from artifact {experiment_id} v{version}: artifact not found"
            )
        
        # Remove tag
        tags = metadata.get("tags", [])
        if tag in tags:
            tags.remove(tag)
            metadata["tags"] = tags
            
            # Add to tagging history
            if "tagging_history" not in metadata:
                metadata["tagging_history"] = []
            
            metadata["tagging_history"].append({
                "tag": tag,
                "action": "removed",
                "removed_at": datetime.now().isoformat()
            })
            
            # Upload updated metadata
            self._upload_json(metadata_path, metadata)
            logger.info(f"Removed tag '{tag}' from {experiment_id} v{version}")
        else:
            logger.warning(f"Tag '{tag}' not found on {experiment_id} v{version}")
    
    def deploy_artifact(
        self,
        experiment_id: str,
        version: str,
        deployment_environment: str = "production",
        deployment_notes: str = ""
    ) -> Dict[str, Any]:
        """
        Deploy artifact and track deployment metadata.
        
        Args:
            experiment_id: Experiment identifier
            version: Version to deploy
            deployment_environment: Target environment (e.g., "production", "staging")
            deployment_notes: Additional deployment notes
            
        Returns:
            Deployment record with timestamp and artifact details
            
        Raises:
            MinIODownloadError: If artifact not found
            ValidationError: If artifact validation fails
        """
        # Download artifact to validate it exists and is valid
        artifact_data = self.download_artifact(experiment_id, version, validate_integrity=True)
        
        # Create deployment record
        deployment_record = {
            "experiment_id": experiment_id,
            "version": version,
            "deployment_environment": deployment_environment,
            "deployed_at": datetime.now().isoformat(),
            "deployment_notes": deployment_notes,
            "artifact_hash": artifact_data["metadata"].get("artifact_hash"),
            "n_states": artifact_data["metadata"].get("n_states"),
            "library": artifact_data["metadata"].get("library_used"),
            "researcher": artifact_data["metadata"].get("researcher")
        }
        
        # Update artifact metadata with deployment info
        metadata_path = f"{experiment_id}/{version}/metadata.json"
        metadata = self._download_json(metadata_path)
        
        if "deployment_history" not in metadata:
            metadata["deployment_history"] = []
        
        metadata["deployment_history"].append(deployment_record)
        metadata["last_deployed_at"] = deployment_record["deployed_at"]
        metadata["last_deployment_environment"] = deployment_environment
        
        # Upload updated metadata
        self._upload_json(metadata_path, metadata)
        
        logger.info(
            f"Deployed artifact {experiment_id} v{version} to {deployment_environment} "
            f"at {deployment_record['deployed_at']}"
        )
        
        return deployment_record
    
    def get_deployment_history(
        self,
        experiment_id: str,
        version: Optional[str] = None,
        environment: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get deployment history for artifacts.
        
        Args:
            experiment_id: Experiment identifier
            version: Optional version filter
            environment: Optional environment filter
            
        Returns:
            List of deployment records
            
        Raises:
            MinIODownloadError: If artifact not found
        """
        deployment_history = []
        
        if version:
            # Get history for specific version
            metadata_path = f"{experiment_id}/{version}/metadata.json"
            try:
                metadata = self._download_json(metadata_path)
                history = metadata.get("deployment_history", [])
                
                if environment:
                    history = [
                        d for d in history 
                        if d.get("deployment_environment") == environment
                    ]
                
                deployment_history.extend(history)
            except MinIODownloadError as e:
                logger.warning(f"Failed to get deployment history for {experiment_id} v{version}: {e}")
        else:
            # Get history for all versions
            versions = self._list_versions(experiment_id)
            
            for ver in versions:
                metadata_path = f"{experiment_id}/{ver}/metadata.json"
                try:
                    metadata = self._download_json(metadata_path)
                    history = metadata.get("deployment_history", [])
                    
                    if environment:
                        history = [
                            d for d in history 
                            if d.get("deployment_environment") == environment
                        ]
                    
                    deployment_history.extend(history)
                except MinIODownloadError as e:
                    logger.warning(f"Failed to get deployment history for {experiment_id} v{ver}: {e}")
        
        # Sort by deployment timestamp
        deployment_history.sort(key=lambda x: x.get("deployed_at", ""), reverse=True)
        
        logger.info(f"Retrieved {len(deployment_history)} deployment record(s)")
        return deployment_history
    
    def get_artifact_lineage(
        self,
        experiment_id: str,
        version: str
    ) -> Dict[str, Any]:
        """
        Get complete lineage information for an artifact.
        
        Args:
            experiment_id: Experiment identifier
            version: Version to get lineage for
            
        Returns:
            Dictionary with lineage information including training config, data sources, etc.
            
        Raises:
            MinIODownloadError: If artifact not found
        """
        # Download artifact metadata
        metadata_path = f"{experiment_id}/{version}/metadata.json"
        metadata = self._download_json(metadata_path)
        
        lineage = {
            "experiment_id": experiment_id,
            "version": version,
            "created_at": metadata.get("created_at"),
            "researcher": metadata.get("researcher"),
            "training_config": metadata.get("training_config", {}),
            "library_used": metadata.get("library_used"),
            "notebook_path": metadata.get("notebook_path"),
            "evaluation_metrics": metadata.get("evaluation_metrics", {}),
            "tags": metadata.get("tags", []),
            "tagging_history": metadata.get("tagging_history", []),
            "deployment_history": metadata.get("deployment_history", []),
            "artifact_hash": metadata.get("artifact_hash"),
            "production_ready": metadata.get("production_ready", False)
        }
        
        logger.info(f"Retrieved lineage for {experiment_id} v{version}")
        return lineage


class ResearchMetadata(BaseModel):
    """Research-specific metadata for HMM artifacts."""
    
    experiment_id: str = Field(description="Unique experiment identifier")
    researcher: str = Field(description="Name or ID of researcher")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    notebook_path: Optional[str] = Field(None, description="Path to source notebook")
    
    # Training configuration
    training_config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")
    library_used: str = Field(description="HMM library used (hmmlearn/pomegranate)")
    
    # Evaluation metrics
    evaluation_metrics: Dict[str, float] = Field(default_factory=dict, description="Evaluation metrics")
    cross_validation_scores: Optional[List[float]] = Field(None, description="CV scores")
    
    # Model comparison
    comparison_baseline: Optional[str] = Field(None, description="Baseline model for comparison")
    performance_ranking: Optional[int] = Field(None, description="Performance rank among compared models")
    
    # Additional notes
    notes: str = Field(default="", description="Additional research notes")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResearchArtifact(BaseModel):
    """Enhanced artifact for research environment with experiment tracking."""
    
    # Core HMM artifact
    base_artifact: HMMArtifact = Field(description="Base HMM artifact")
    
    # Research metadata
    research_metadata: ResearchMetadata = Field(description="Research-specific metadata")
    
    # Versioning
    artifact_version: str = Field(description="Artifact version (semantic versioning)")
    artifact_hash: str = Field(description="Hash of artifact for integrity checking")
    
    # Production compatibility
    production_ready: bool = Field(default=False, description="Whether artifact is production-ready")
    validation_status: Dict[str, bool] = Field(default_factory=dict, description="Validation checks status")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def from_hmm_artifact(cls,
                         artifact: HMMArtifact,
                         experiment_id: str,
                         researcher: str,
                         training_config: Dict[str, Any],
                         evaluation_metrics: Optional[Dict[str, float]] = None,
                         **kwargs) -> "ResearchArtifact":
        """
        Create ResearchArtifact from HMMArtifact.
        
        Args:
            artifact: Base HMM artifact
            experiment_id: Unique experiment identifier
            researcher: Researcher name/ID
            training_config: Training configuration used
            evaluation_metrics: Evaluation metrics
            **kwargs: Additional metadata fields
            
        Returns:
            ResearchArtifact instance
        """
        # Create research metadata
        research_metadata = ResearchMetadata(
            experiment_id=experiment_id,
            researcher=researcher,
            training_config=training_config,
            library_used=artifact.metadata.get("library", "unknown"),
            evaluation_metrics=evaluation_metrics or {},
            **kwargs
        )
        
        # Calculate artifact hash
        artifact_hash = cls._calculate_artifact_hash(artifact)
        
        # Generate version
        artifact_version = kwargs.get("artifact_version", "1.0.0")
        
        return cls(
            base_artifact=artifact,
            research_metadata=research_metadata,
            artifact_version=artifact_version,
            artifact_hash=artifact_hash,
            production_ready=False,
            validation_status={}
        )
    
    @staticmethod
    def _calculate_artifact_hash(artifact: HMMArtifact) -> str:
        """Calculate hash of artifact for integrity checking."""
        # Create deterministic string representation
        artifact_str = json.dumps(artifact.model_dump(), sort_keys=True)
        return hashlib.sha256(artifact_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify artifact integrity using hash."""
        current_hash = self._calculate_artifact_hash(self.base_artifact)
        return current_hash == self.artifact_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchArtifact":
        """Create from dictionary."""
        return cls(**data)


class ExperimentTracker:
    """Track and manage research experiments with versioning and optional MinIO storage."""
    
    def __init__(
        self,
        experiment_dir: Union[str, Path],
        use_minio: bool = False,
        minio_config: Optional[MinIOConfig] = None
    ):
        """
        Initialize experiment tracker with optional MinIO integration.
        
        Args:
            experiment_dir: Directory for storing experiment artifacts locally
            use_minio: Whether to use MinIO for remote storage (default: False)
            minio_config: MinIO configuration (defaults to env vars if use_minio=True)
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file for tracking experiments
        self.index_file = self.experiment_dir / "experiment_index.json"
        self.experiments = self._load_index()
        
        # MinIO integration
        self.use_minio = use_minio
        self.minio_store = None
        
        if use_minio:
            try:
                self.minio_store = MinIOArtifactStore(minio_config)
                logger.info(f"Experiment tracker initialized with MinIO support at {self.experiment_dir}")
            except MinIOConnectionError as e:
                logger.warning(f"Failed to initialize MinIO: {e}. Falling back to local-only storage.")
                self.use_minio = False
        else:
            logger.info(f"Experiment tracker initialized (local-only) at {self.experiment_dir}")
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load experiment index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load experiment index: {e}")
                return {}
        else:
            # Create empty index file
            self._save_index_internal({})
        return {}
    
    def _save_index(self) -> None:
        """Save experiment index to disk."""
        self._save_index_internal(self.experiments)
    
    def _save_index_internal(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Internal method to save index data to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save experiment index: {e}")
            raise ArtifactManagementError(f"Failed to save index: {e}")
    
    def log_experiment(
        self,
        artifact: ResearchArtifact,
        fusion_weights: Optional[FusionWeights] = None,
        notes: str = "",
        upload_to_minio: bool = True
    ) -> str:
        """
        Log research experiment with versioning and optional MinIO upload.
        
        Args:
            artifact: Research artifact to log
            fusion_weights: Optional fusion weights to store with artifact
            notes: Additional notes about the experiment
            upload_to_minio: Whether to upload to MinIO (if enabled)
            
        Returns:
            Experiment ID
        """
        experiment_id = artifact.research_metadata.experiment_id
        
        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save artifact locally
        artifact_path = exp_dir / f"artifact_v{artifact.artifact_version}.json"
        try:
            with open(artifact_path, 'w') as f:
                json.dump(artifact.to_dict(), f, indent=2, default=str)
        except Exception as e:
            raise ExportError(f"Failed to save artifact: {e}")
        
        # Save fusion weights locally if provided
        if fusion_weights:
            weights_path = exp_dir / f"fusion_weights_v{artifact.artifact_version}.json"
            try:
                with open(weights_path, 'w') as f:
                    json.dump(fusion_weights.model_dump(), f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save fusion weights locally: {e}")
        
        # Update index
        self.experiments[experiment_id] = {
            "experiment_id": experiment_id,
            "researcher": artifact.research_metadata.researcher,
            "created_at": artifact.research_metadata.created_at.isoformat(),
            "artifact_version": artifact.artifact_version,
            "artifact_path": str(artifact_path),
            "artifact_hash": artifact.artifact_hash,
            "production_ready": artifact.production_ready,
            "notes": notes,
            "tags": artifact.research_metadata.tags,
            "library": artifact.research_metadata.library_used,
            "n_states": artifact.base_artifact.n_states,
            "has_fusion_weights": fusion_weights is not None,
            "uploaded_to_minio": False
        }
        
        self._save_index()
        logger.info(f"Experiment {experiment_id} logged locally")
        
        # Upload to MinIO if enabled
        if upload_to_minio and self.use_minio and self.minio_store:
            try:
                uploaded_paths = self.minio_store.upload_artifact(
                    artifact,
                    fusion_weights,
                    tags=artifact.research_metadata.tags
                )
                
                # Update index with MinIO upload status
                self.experiments[experiment_id]["uploaded_to_minio"] = True
                self.experiments[experiment_id]["minio_paths"] = uploaded_paths
                self._save_index()
                
                logger.info(f"Experiment {experiment_id} uploaded to MinIO")
            except (MinIOUploadError, MinIOConnectionError) as e:
                logger.error(f"Failed to upload to MinIO: {e}. Artifact saved locally only.")
                # Continue - local storage succeeded
        
        return experiment_id
    
    def load_experiment(self, experiment_id: str, version: Optional[str] = None) -> ResearchArtifact:
        """
        Load experiment artifact.
        
        Args:
            experiment_id: Experiment identifier
            version: Specific version to load (None for latest)
            
        Returns:
            ResearchArtifact
        """
        if experiment_id not in self.experiments:
            raise ArtifactManagementError(f"Experiment {experiment_id} not found")
        
        exp_info = self.experiments[experiment_id]
        exp_dir = self.experiment_dir / experiment_id
        
        # Find artifact file
        if version:
            artifact_path = exp_dir / f"artifact_v{version}.json"
        else:
            artifact_path = Path(exp_info["artifact_path"])
        
        if not artifact_path.exists():
            raise ArtifactManagementError(f"Artifact file not found: {artifact_path}")
        
        try:
            with open(artifact_path, 'r') as f:
                data = json.load(f)
            artifact = ResearchArtifact.from_dict(data)
            
            # Verify integrity
            if not artifact.verify_integrity():
                logger.warning(f"Artifact integrity check failed for {experiment_id}")
            
            return artifact
        except Exception as e:
            raise ArtifactManagementError(f"Failed to load artifact: {e}")
    
    def list_experiments(self,
                        researcher: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        production_ready: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.
        
        Args:
            researcher: Filter by researcher
            tags: Filter by tags (any match)
            production_ready: Filter by production readiness
            
        Returns:
            List of experiment info dictionaries
        """
        experiments = list(self.experiments.values())
        
        # Apply filters
        if researcher:
            experiments = [e for e in experiments if e.get("researcher") == researcher]
        
        if tags:
            experiments = [e for e in experiments 
                          if any(tag in e.get("tags", []) for tag in tags)]
        
        if production_ready is not None:
            experiments = [e for e in experiments 
                          if e.get("production_ready") == production_ready]
        
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            "experiments": [],
            "metrics_comparison": {},
            "best_by_metric": {}
        }
        
        for exp_id in experiment_ids:
            try:
                artifact = self.load_experiment(exp_id)
                exp_info = {
                    "experiment_id": exp_id,
                    "n_states": artifact.base_artifact.n_states,
                    "library": artifact.research_metadata.library_used,
                    "metrics": artifact.research_metadata.evaluation_metrics,
                    "production_ready": artifact.production_ready
                }
                comparison["experiments"].append(exp_info)
            except Exception as e:
                logger.warning(f"Failed to load experiment {exp_id}: {e}")
        
        # Compare metrics
        if comparison["experiments"]:
            # Collect all metric names
            all_metrics = set()
            for exp in comparison["experiments"]:
                all_metrics.update(exp["metrics"].keys())
            
            # Compare each metric
            for metric in all_metrics:
                values = []
                for exp in comparison["experiments"]:
                    if metric in exp["metrics"]:
                        values.append({
                            "experiment_id": exp["experiment_id"],
                            "value": exp["metrics"][metric]
                        })
                
                comparison["metrics_comparison"][metric] = values
                
                # Find best (assuming higher is better for most metrics except AIC/BIC)
                if values:
                    if metric.lower() in ["aic", "bic", "perplexity"]:
                        # Lower is better
                        best = min(values, key=lambda x: x["value"])
                    else:
                        # Higher is better
                        best = max(values, key=lambda x: x["value"])
                    
                    comparison["best_by_metric"][metric] = best["experiment_id"]
        
        return comparison
    
    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete experiment and its artifacts.
        
        Args:
            experiment_id: Experiment to delete
        """
        if experiment_id not in self.experiments:
            raise ArtifactManagementError(f"Experiment {experiment_id} not found")
        
        # Delete experiment directory
        exp_dir = self.experiment_dir / experiment_id
        if exp_dir.exists():
            import shutil
            shutil.rmtree(exp_dir)
        
        # Remove from index
        del self.experiments[experiment_id]
        self._save_index()
        
        logger.info(f"Experiment {experiment_id} deleted")
    
    def sync_to_minio(
        self,
        experiment_id: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Sync local experiments to MinIO storage.
        
        Args:
            experiment_id: Specific experiment to sync (None for all)
            force: Force re-upload even if already uploaded
            
        Returns:
            Dictionary with sync results (uploaded, skipped, failed)
        """
        if not self.use_minio or not self.minio_store:
            raise ArtifactManagementError(
                "MinIO is not enabled. Initialize ExperimentTracker with use_minio=True"
            )
        
        results = {
            "uploaded": [],
            "skipped": [],
            "failed": []
        }
        
        # Determine which experiments to sync
        if experiment_id:
            if experiment_id not in self.experiments:
                raise ArtifactManagementError(f"Experiment {experiment_id} not found")
            experiments_to_sync = [experiment_id]
        else:
            experiments_to_sync = list(self.experiments.keys())
        
        logger.info(f"Syncing {len(experiments_to_sync)} experiment(s) to MinIO")
        
        for exp_id in experiments_to_sync:
            exp_info = self.experiments[exp_id]
            
            # Skip if already uploaded and not forcing
            if exp_info.get("uploaded_to_minio", False) and not force:
                results["skipped"].append(exp_id)
                logger.debug(f"Skipping {exp_id} (already uploaded)")
                continue
            
            try:
                # Load artifact from local storage
                artifact = self.load_experiment(exp_id)
                
                # Load fusion weights if available
                fusion_weights = None
                if exp_info.get("has_fusion_weights", False):
                    exp_dir = self.experiment_dir / exp_id
                    weights_path = exp_dir / f"fusion_weights_v{artifact.artifact_version}.json"
                    if weights_path.exists():
                        try:
                            with open(weights_path, 'r') as f:
                                weights_data = json.load(f)
                            fusion_weights = FusionWeights(**weights_data)
                        except Exception as e:
                            logger.warning(f"Failed to load fusion weights for {exp_id}: {e}")
                
                # Upload to MinIO
                uploaded_paths = self.minio_store.upload_artifact(
                    artifact,
                    fusion_weights,
                    tags=artifact.research_metadata.tags
                )
                
                # Update index
                self.experiments[exp_id]["uploaded_to_minio"] = True
                self.experiments[exp_id]["minio_paths"] = uploaded_paths
                self._save_index()
                
                results["uploaded"].append(exp_id)
                logger.info(f"Synced {exp_id} to MinIO")
                
            except Exception as e:
                results["failed"].append({"experiment_id": exp_id, "error": str(e)})
                logger.error(f"Failed to sync {exp_id} to MinIO: {e}")
        
        logger.info(
            f"Sync complete: {len(results['uploaded'])} uploaded, "
            f"{len(results['skipped'])} skipped, {len(results['failed'])} failed"
        )
        
        return results
    
    def sync_from_minio(
        self,
        experiment_id: Optional[str] = None,
        version: str = "latest",
        overwrite_local: bool = False
    ) -> Dict[str, Any]:
        """
        Sync experiments from MinIO to local storage.
        
        Args:
            experiment_id: Specific experiment to sync (None for all available)
            version: Version to download ("latest" for most recent)
            overwrite_local: Whether to overwrite existing local artifacts
            
        Returns:
            Dictionary with sync results (downloaded, skipped, failed)
        """
        if not self.use_minio or not self.minio_store:
            raise ArtifactManagementError(
                "MinIO is not enabled. Initialize ExperimentTracker with use_minio=True"
            )
        
        results = {
            "downloaded": [],
            "skipped": [],
            "failed": []
        }
        
        try:
            # List available artifacts in MinIO
            if experiment_id:
                minio_artifacts = self.minio_store.list_artifacts(experiment_id=experiment_id)
            else:
                minio_artifacts = self.minio_store.list_artifacts()
            
            logger.info(f"Found {len(minio_artifacts)} artifact(s) in MinIO")
            
            for artifact_meta in minio_artifacts:
                exp_id = artifact_meta["experiment_id"]
                exp_version = artifact_meta["version"]
                
                # Skip if local version exists and not overwriting
                if exp_id in self.experiments and not overwrite_local:
                    local_version = self.experiments[exp_id].get("artifact_version")
                    if local_version == exp_version:
                        results["skipped"].append(exp_id)
                        logger.debug(f"Skipping {exp_id} v{exp_version} (already exists locally)")
                        continue
                
                try:
                    # Download from MinIO
                    artifact_data = self.minio_store.download_artifact(exp_id, exp_version)
                    
                    # Reconstruct ResearchArtifact
                    hmm_artifact = HMMArtifact(**artifact_data["hmm_artifact"])
                    research_metadata = ResearchMetadata(**artifact_data["metadata"])
                    
                    research_artifact = ResearchArtifact(
                        base_artifact=hmm_artifact,
                        research_metadata=research_metadata,
                        artifact_version=exp_version,
                        artifact_hash=artifact_meta.get("artifact_hash", ""),
                        production_ready=artifact_meta.get("production_ready", False),
                        validation_status={}
                    )
                    
                    # Reconstruct FusionWeights if available
                    fusion_weights = None
                    if artifact_data.get("fusion_weights"):
                        fusion_weights = FusionWeights(**artifact_data["fusion_weights"])
                    
                    # Save locally (without re-uploading to MinIO)
                    exp_dir = self.experiment_dir / exp_id
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    
                    artifact_path = exp_dir / f"artifact_v{exp_version}.json"
                    with open(artifact_path, 'w') as f:
                        json.dump(research_artifact.to_dict(), f, indent=2, default=str)
                    
                    if fusion_weights:
                        weights_path = exp_dir / f"fusion_weights_v{exp_version}.json"
                        with open(weights_path, 'w') as f:
                            json.dump(fusion_weights.model_dump(), f, indent=2, default=str)
                    
                    # Update index
                    self.experiments[exp_id] = {
                        "experiment_id": exp_id,
                        "researcher": research_metadata.researcher,
                        "created_at": research_metadata.created_at.isoformat(),
                        "artifact_version": exp_version,
                        "artifact_path": str(artifact_path),
                        "artifact_hash": research_artifact.artifact_hash,
                        "production_ready": research_artifact.production_ready,
                        "notes": research_metadata.notes,
                        "tags": research_metadata.tags,
                        "library": research_metadata.library_used,
                        "n_states": hmm_artifact.n_states,
                        "has_fusion_weights": fusion_weights is not None,
                        "uploaded_to_minio": True,
                        "synced_from_minio": True
                    }
                    self._save_index()
                    
                    results["downloaded"].append(exp_id)
                    logger.info(f"Downloaded {exp_id} v{exp_version} from MinIO")
                    
                except Exception as e:
                    results["failed"].append({"experiment_id": exp_id, "error": str(e)})
                    logger.error(f"Failed to download {exp_id} from MinIO: {e}")
            
            logger.info(
                f"Sync from MinIO complete: {len(results['downloaded'])} downloaded, "
                f"{len(results['skipped'])} skipped, {len(results['failed'])} failed"
            )
            
            return results
            
        except Exception as e:
            raise ArtifactManagementError(f"Failed to sync from MinIO: {e}")
    
    def get_minio_status(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get MinIO sync status for experiments.
        
        Args:
            experiment_id: Specific experiment to check (None for all)
            
        Returns:
            Dictionary with sync status information
        """
        if not self.use_minio:
            return {
                "minio_enabled": False,
                "message": "MinIO is not enabled"
            }
        
        status = {
            "minio_enabled": True,
            "minio_connected": self.minio_store is not None,
            "experiments": []
        }
        
        # Determine which experiments to check
        if experiment_id:
            if experiment_id not in self.experiments:
                raise ArtifactManagementError(f"Experiment {experiment_id} not found")
            experiments_to_check = [experiment_id]
        else:
            experiments_to_check = list(self.experiments.keys())
        
        for exp_id in experiments_to_check:
            exp_info = self.experiments[exp_id]
            exp_status = {
                "experiment_id": exp_id,
                "local_version": exp_info.get("artifact_version"),
                "uploaded_to_minio": exp_info.get("uploaded_to_minio", False),
                "synced_from_minio": exp_info.get("synced_from_minio", False)
            }
            
            # Check if artifact exists in MinIO
            if self.minio_store:
                try:
                    minio_artifacts = self.minio_store.list_artifacts(experiment_id=exp_id)
                    exp_status["in_minio"] = len(minio_artifacts) > 0
                    exp_status["minio_versions"] = [a["version"] for a in minio_artifacts]
                except Exception as e:
                    exp_status["in_minio"] = None
                    exp_status["minio_check_error"] = str(e)
            
            status["experiments"].append(exp_status)
        
        return status


class ArtifactValidator:
    """Validate artifacts for production deployment compatibility."""
    
    @staticmethod
    def validate_for_production(artifact: ResearchArtifact) -> Dict[str, bool]:
        """
        Validate artifact for production deployment.
        
        Args:
            artifact: Research artifact to validate
            
        Returns:
            Dictionary of validation checks and their status
        """
        validation_results = {}
        
        # Check 1: Artifact integrity
        validation_results["integrity_check"] = artifact.verify_integrity()
        
        # Check 2: HMM artifact validation
        try:
            base_artifact = artifact.base_artifact
            validation_results["hmm_artifact_valid"] = True
        except Exception as e:
            logger.error(f"HMM artifact validation failed: {e}")
            validation_results["hmm_artifact_valid"] = False
        
        # Check 3: Transition matrix properties
        try:
            transition_matrix = np.array(base_artifact.transition_matrix)
            # Check row sums
            row_sums = transition_matrix.sum(axis=1)
            validation_results["transition_matrix_valid"] = np.allclose(row_sums, 1.0, atol=1e-6)
        except Exception as e:
            logger.error(f"Transition matrix validation failed: {e}")
            validation_results["transition_matrix_valid"] = False
        
        # Check 4: Covariance matrices are positive definite
        try:
            all_positive_definite = True
            for cov in base_artifact.covariances:
                cov_matrix = np.array(cov)
                try:
                    np.linalg.cholesky(cov_matrix)
                except np.linalg.LinAlgError:
                    all_positive_definite = False
                    break
            validation_results["covariances_positive_definite"] = all_positive_definite
        except Exception as e:
            logger.error(f"Covariance validation failed: {e}")
            validation_results["covariances_positive_definite"] = False
        
        # Check 5: Evaluation metrics present
        validation_results["has_evaluation_metrics"] = bool(
            artifact.research_metadata.evaluation_metrics
        )
        
        # Check 6: Reasonable number of states
        validation_results["reasonable_n_states"] = 2 <= base_artifact.n_states <= 10
        
        # Check 7: Training window validity
        validation_results["valid_training_window"] = (
            base_artifact.training_window_end > base_artifact.training_window_start
        )
        
        # Check 8: Metadata completeness
        required_metadata = ["library", "algorithm", "covariance_type"]
        validation_results["metadata_complete"] = all(
            key in base_artifact.metadata for key in required_metadata
        )
        
        return validation_results
    
    @staticmethod
    def validate_rust_compatibility(artifact: ResearchArtifact) -> Dict[str, bool]:
        """
        Validate compatibility with Rust inference engine.
        
        Args:
            artifact: Research artifact to validate
            
        Returns:
            Dictionary of compatibility checks
        """
        compatibility_results = {}
        
        base_artifact = artifact.base_artifact
        
        # Check 1: Data types are JSON-serializable
        try:
            json.dumps(base_artifact.model_dump())
            compatibility_results["json_serializable"] = True
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            compatibility_results["json_serializable"] = False
        
        # Check 2: Array dimensions are consistent
        try:
            n_states = base_artifact.n_states
            n_features = len(base_artifact.means[0]) if base_artifact.means else 0
            
            # Check transition matrix
            trans_shape_ok = (
                len(base_artifact.transition_matrix) == n_states and
                all(len(row) == n_states for row in base_artifact.transition_matrix)
            )
            
            # Check means
            means_shape_ok = (
                len(base_artifact.means) == n_states and
                all(len(mean) == n_features for mean in base_artifact.means)
            )
            
            # Check covariances
            covs_shape_ok = (
                len(base_artifact.covariances) == n_states and
                all(len(cov) == n_features and all(len(row) == n_features for row in cov)
                    for cov in base_artifact.covariances)
            )
            
            compatibility_results["dimensions_consistent"] = (
                trans_shape_ok and means_shape_ok and covs_shape_ok
            )
        except Exception as e:
            logger.error(f"Dimension check failed: {e}")
            compatibility_results["dimensions_consistent"] = False
        
        # Check 3: No NaN or Inf values
        try:
            def has_invalid_values(data):
                if isinstance(data, (list, tuple)):
                    return any(has_invalid_values(item) for item in data)
                elif isinstance(data, (int, float)):
                    return np.isnan(data) or np.isinf(data)
                return False
            
            no_invalid = not any([
                has_invalid_values(base_artifact.transition_matrix),
                has_invalid_values(base_artifact.initial_probabilities),
                has_invalid_values(base_artifact.means),
                has_invalid_values(base_artifact.covariances)
            ])
            
            compatibility_results["no_invalid_values"] = no_invalid
        except Exception as e:
            logger.error(f"Invalid value check failed: {e}")
            compatibility_results["no_invalid_values"] = False
        
        # Check 4: Version format is valid
        compatibility_results["valid_version_format"] = bool(
            base_artifact.version and isinstance(base_artifact.version, str)
        )
        
        return compatibility_results
    
    @staticmethod
    def run_all_validations(artifact: ResearchArtifact) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Args:
            artifact: Research artifact to validate
            
        Returns:
            Complete validation report
        """
        production_checks = ArtifactValidator.validate_for_production(artifact)
        rust_checks = ArtifactValidator.validate_rust_compatibility(artifact)
        
        all_checks = {**production_checks, **rust_checks}
        all_passed = all(all_checks.values())
        
        return {
            "all_passed": all_passed,
            "production_checks": production_checks,
            "rust_compatibility_checks": rust_checks,
            "failed_checks": [k for k, v in all_checks.items() if not v]
        }


class ArtifactExporter:
    """Export artifacts for production deployment."""
    
    @staticmethod
    def export_for_production(artifact: ResearchArtifact,
                             output_dir: Union[str, Path],
                             validate: bool = True) -> Dict[str, Path]:
        """
        Export artifact for production deployment.
        
        Args:
            artifact: Research artifact to export
            output_dir: Output directory for exported files
            validate: Whether to validate before export
            
        Returns:
            Dictionary of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate if requested
        if validate:
            validation_results = ArtifactValidator.run_all_validations(artifact)
            if not validation_results["all_passed"]:
                raise ValidationError(
                    f"Artifact validation failed: {validation_results['failed_checks']}"
                )
        
        exported_files = {}
        
        # Export base HMM artifact
        hmm_artifact_path = output_dir / "hmm_artifact.json"
        try:
            with open(hmm_artifact_path, 'w') as f:
                json.dump(artifact.base_artifact.model_dump(), f, indent=2)
            exported_files["hmm_artifact"] = hmm_artifact_path
        except Exception as e:
            raise ExportError(f"Failed to export HMM artifact: {e}")
        
        # Export research metadata
        metadata_path = output_dir / "research_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(artifact.research_metadata.model_dump(), f, indent=2, default=str)
            exported_files["metadata"] = metadata_path
        except Exception as e:
            raise ExportError(f"Failed to export metadata: {e}")
        
        # Export validation report
        validation_path = output_dir / "validation_report.json"
        try:
            validation_results = ArtifactValidator.run_all_validations(artifact)
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            exported_files["validation_report"] = validation_path
        except Exception as e:
            logger.warning(f"Failed to export validation report: {e}")
        
        # Create deployment manifest
        manifest_path = output_dir / "deployment_manifest.json"
        try:
            manifest = {
                "artifact_version": artifact.artifact_version,
                "artifact_hash": artifact.artifact_hash,
                "experiment_id": artifact.research_metadata.experiment_id,
                "created_at": artifact.research_metadata.created_at.isoformat(),
                "researcher": artifact.research_metadata.researcher,
                "n_states": artifact.base_artifact.n_states,
                "library": artifact.research_metadata.library_used,
                "production_ready": artifact.production_ready,
                "exported_files": {k: str(v) for k, v in exported_files.items()}
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            exported_files["manifest"] = manifest_path
        except Exception as e:
            logger.warning(f"Failed to create deployment manifest: {e}")
        
        logger.info(f"Artifact exported to {output_dir}")
        return exported_files
    
    @staticmethod
    def export_fusion_weights(fusion_weights: FusionWeights,
                             output_path: Union[str, Path]) -> Path:
        """
        Export fusion weights for production.
        
        Args:
            fusion_weights: Fusion weights to export
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(fusion_weights.model_dump(), f, indent=2)
            logger.info(f"Fusion weights exported to {output_path}")
            return output_path
        except Exception as e:
            raise ExportError(f"Failed to export fusion weights: {e}")
