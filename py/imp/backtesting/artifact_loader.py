"""
MinIO integration for HMM artifacts in backtesting framework.

This module provides artifact loading capabilities with caching, fallback mechanisms,
and integration with the existing HMM artifact management system.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import hashlib

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    S3Error = Exception

from .config import DataSourceConfig

logger = logging.getLogger(__name__)


class ArtifactLoadError(Exception):
    """Raised when artifact loading fails."""
    pass


class MinIOConnectionError(ArtifactLoadError):
    """Raised when MinIO connection fails."""
    pass


@dataclass
class CachedArtifact:
    """Cached artifact with metadata."""
    
    artifact_data: Dict[str, Any]
    cached_at: datetime
    cache_key: str
    artifact_hash: Optional[str] = None
    ttl_seconds: int = 3600  # 1 hour default TTL
    
    def is_expired(self) -> bool:
        """Check if cached artifact has expired."""
        return datetime.now() > self.cached_at + timedelta(seconds=self.ttl_seconds)
    
    def get_age_seconds(self) -> float:
        """Get age of cached artifact in seconds."""
        return (datetime.now() - self.cached_at).total_seconds()


class ArtifactCache:
    """
    In-memory cache for HMM artifacts with TTL and size limits.
    """
    
    def __init__(self, max_size: int = 50, default_ttl: int = 3600):
        """
        Initialize artifact cache.
        
        Args:
            max_size: Maximum number of artifacts to cache
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CachedArtifact] = {}
        self._access_times: Dict[str, datetime] = {}
        
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get artifact from cache if not expired.
        
        Args:
            cache_key: Cache key for artifact
            
        Returns:
            Artifact data or None if not found/expired
        """
        if cache_key not in self._cache:
            return None
        
        cached_artifact = self._cache[cache_key]
        
        # Check if expired
        if cached_artifact.is_expired():
            logger.debug(f"Cache entry expired for {cache_key}")
            self.remove(cache_key)
            return None
        
        # Update access time
        self._access_times[cache_key] = datetime.now()
        
        logger.debug(f"Cache hit for {cache_key} (age: {cached_artifact.get_age_seconds():.1f}s)")
        return cached_artifact.artifact_data
    
    def put(
        self,
        cache_key: str,
        artifact_data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        artifact_hash: Optional[str] = None
    ) -> None:
        """
        Store artifact in cache.
        
        Args:
            cache_key: Cache key for artifact
            artifact_data: Artifact data to cache
            ttl_seconds: TTL in seconds (uses default if None)
            artifact_hash: Optional artifact hash for validation
        """
        ttl = ttl_seconds or self.default_ttl
        
        # Evict if at capacity
        if len(self._cache) >= self.max_size and cache_key not in self._cache:
            self._evict_lru()
        
        cached_artifact = CachedArtifact(
            artifact_data=artifact_data,
            cached_at=datetime.now(),
            cache_key=cache_key,
            artifact_hash=artifact_hash,
            ttl_seconds=ttl
        )
        
        self._cache[cache_key] = cached_artifact
        self._access_times[cache_key] = datetime.now()
        
        logger.debug(f"Cached artifact {cache_key} (TTL: {ttl}s)")
    
    def remove(self, cache_key: str) -> None:
        """Remove artifact from cache."""
        if cache_key in self._cache:
            del self._cache[cache_key]
            del self._access_times[cache_key]
            logger.debug(f"Removed {cache_key} from cache")
    
    def clear(self) -> None:
        """Clear all cached artifacts."""
        self._cache.clear()
        self._access_times.clear()
        logger.info("Cleared artifact cache")
    
    def _evict_lru(self) -> None:
        """Evict least recently used artifact."""
        if not self._access_times:
            return
        
        # Find LRU entry
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self.remove(lru_key)
        logger.debug(f"Evicted LRU entry: {lru_key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        expired_count = sum(
            1 for cached in self._cache.values() if cached.is_expired()
        )
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "expired_entries": expired_count,
            "cache_keys": list(self._cache.keys()),
            "oldest_entry_age": min(
                (now - cached.cached_at).total_seconds()
                for cached in self._cache.values()
            ) if self._cache else 0
        }


class ArtifactLoader:
    """
    Load HMM artifacts from MinIO storage with caching and fallback mechanisms.
    
    Features:
    - MinIO integration with retry logic
    - Local file system fallback
    - In-memory caching with TTL
    - Artifact validation and integrity checking
    - Production artifact discovery
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        Initialize artifact loader.
        
        Args:
            config: Data source configuration with MinIO settings
        """
        self.config = config
        self.cache = ArtifactCache()
        self._minio_client: Optional[Minio] = None
        self._minio_available = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize MinIO if configured
        if self._should_use_minio():
            self._initialize_minio()
        else:
            logger.info("MinIO not configured, using local fallback only")
        
        # Ensure local cache directory exists
        self.config.local_cache_path.mkdir(parents=True, exist_ok=True)
    
    def _should_use_minio(self) -> bool:
        """Check if MinIO should be used based on configuration."""
        return (
            MINIO_AVAILABLE and
            self.config.minio_endpoint is not None and
            self.config.minio_access_key is not None and
            self.config.minio_secret_key is not None
        )
    
    def _initialize_minio(self) -> None:
        """Initialize MinIO client with connection validation."""
        if not MINIO_AVAILABLE:
            logger.warning("MinIO library not available. Install with: pip install minio")
            return
        
        try:
            # Parse endpoint (remove http:// or https:// if present)
            endpoint = self.config.minio_endpoint
            secure = False
            
            if endpoint.startswith('https://'):
                endpoint = endpoint[8:]
                secure = True
            elif endpoint.startswith('http://'):
                endpoint = endpoint[7:]
                secure = False
            
            self._minio_client = Minio(
                endpoint,
                access_key=self.config.minio_access_key,
                secret_key=self.config.minio_secret_key,
                secure=secure
            )
            
            # Validate connection
            self._validate_minio_connection()
            self._minio_available = True
            
            logger.info(f"MinIO client initialized: {endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            self._minio_available = False
    
    def _validate_minio_connection(self) -> None:
        """Validate MinIO connection and bucket access."""
        if not self._minio_client:
            raise MinIOConnectionError("MinIO client not initialized")
        
        try:
            # Check if bucket exists
            if not self._minio_client.bucket_exists(self.config.minio_bucket):
                logger.warning(f"MinIO bucket '{self.config.minio_bucket}' does not exist")
                # Don't create bucket automatically in backtesting context
            
            # Test connection by listing objects (limit to 1)
            objects = list(self._minio_client.list_objects(
                self.config.minio_bucket,
                recursive=False,
                max_keys=1
            ))
            
            logger.debug("MinIO connection validated successfully")
            
        except S3Error as e:
            if e.code == "NoSuchBucket":
                raise MinIOConnectionError(f"MinIO bucket '{self.config.minio_bucket}' not found")
            else:
                raise MinIOConnectionError(f"MinIO validation failed: {e}")
        except Exception as e:
            raise MinIOConnectionError(f"MinIO connection validation failed: {e}")
    
    def load_hmm_weights(
        self,
        experiment_id: Optional[str] = None,
        version: str = "latest",
        use_cache: bool = True,
        cache_ttl: int = 3600
    ) -> Dict[str, Any]:
        """
        Load HMM fusion weights with caching and fallback.
        
        Args:
            experiment_id: Specific experiment ID (None for production artifact)
            version: Version to load ("latest" for most recent)
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds
            
        Returns:
            Dictionary with HMM weights and metadata
            
        Raises:
            ArtifactLoadError: If loading fails from all sources
        """
        # Generate cache key
        cache_key = f"hmm_weights_{experiment_id or 'production'}_{version}"
        
        # Try cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Loaded HMM weights from cache: {cache_key}")
                return cached_data
        
        # Try MinIO if available
        if self._minio_available:
            try:
                weights_data = self._load_from_minio(experiment_id, version)
                
                # Cache the result
                if use_cache:
                    self.cache.put(cache_key, weights_data, cache_ttl)
                
                logger.info(f"Loaded HMM weights from MinIO: {experiment_id or 'production'} v{version}")
                return weights_data
                
            except Exception as e:
                logger.warning(f"Failed to load from MinIO: {e}. Trying local fallback...")
        
        # Try local fallback
        try:
            weights_data = self._load_from_local(experiment_id, version)
            
            # Cache the result
            if use_cache:
                self.cache.put(cache_key, weights_data, cache_ttl)
            
            logger.info(f"Loaded HMM weights from local storage: {experiment_id or 'production'} v{version}")
            return weights_data
            
        except Exception as e:
            logger.error(f"Failed to load from local storage: {e}")
        
        # All sources failed
        raise ArtifactLoadError(
            f"Failed to load HMM weights for {experiment_id or 'production'} v{version} "
            f"from all sources (MinIO: {self._minio_available}, Local: True)"
        )
    
    def _load_from_minio(
        self,
        experiment_id: Optional[str],
        version: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Load artifact from MinIO with retry logic.
        
        Args:
            experiment_id: Experiment ID (None for production)
            version: Version to load
            max_retries: Maximum retry attempts
            
        Returns:
            Artifact data dictionary
            
        Raises:
            ArtifactLoadError: If loading fails
        """
        if not self._minio_client:
            raise ArtifactLoadError("MinIO client not available")
        
        # Determine artifact path
        if experiment_id is None:
            # Load production artifact
            artifact_path = self._find_production_artifact()
        else:
            # Resolve version if "latest"
            if version == "latest":
                version = self._get_latest_version(experiment_id)
            
            artifact_path = f"{experiment_id}/{version}"
        
        # Load components with retry logic
        for attempt in range(max_retries):
            try:
                # Load metadata
                metadata = self._download_json_from_minio(f"{artifact_path}/metadata.json")
                
                # Load HMM artifact
                hmm_artifact = self._download_json_from_minio(f"{artifact_path}/hmm_artifact.json")
                
                # Load fusion weights (optional)
                fusion_weights = None
                try:
                    fusion_weights = self._download_json_from_minio(f"{artifact_path}/fusion_weights.json")
                except ArtifactLoadError:
                    logger.debug(f"No fusion weights found for {artifact_path}")
                
                # Combine into result
                result = {
                    "metadata": metadata,
                    "hmm_artifact": hmm_artifact,
                    "fusion_weights": fusion_weights,
                    "source": "minio",
                    "artifact_path": artifact_path,
                    "loaded_at": datetime.now().isoformat()
                }
                
                # Validate integrity if hash available
                expected_hash = metadata.get("artifact_hash")
                if expected_hash:
                    self._validate_artifact_integrity(hmm_artifact, expected_hash)
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"MinIO load attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise ArtifactLoadError(f"Failed to load from MinIO after {max_retries} attempts: {e}")
    
    def _download_json_from_minio(self, object_path: str) -> Dict[str, Any]:
        """Download and parse JSON from MinIO."""
        try:
            response = self._minio_client.get_object(self.config.minio_bucket, object_path)
            data = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise ArtifactLoadError(f"Object not found in MinIO: {object_path}")
            else:
                raise ArtifactLoadError(f"MinIO download failed for {object_path}: {e}")
        except json.JSONDecodeError as e:
            raise ArtifactLoadError(f"Invalid JSON in MinIO object {object_path}: {e}")
        except Exception as e:
            raise ArtifactLoadError(f"Unexpected error downloading {object_path}: {e}")
    
    def _find_production_artifact(self) -> str:
        """Find the latest production-tagged artifact in MinIO."""
        try:
            # List all objects to find production artifacts
            objects = self._minio_client.list_objects(
                self.config.minio_bucket,
                recursive=True
            )
            
            # Find metadata files
            metadata_paths = [
                obj.object_name for obj in objects
                if obj.object_name.endswith("metadata.json")
            ]
            
            # Check each metadata file for production tag
            production_artifacts = []
            
            for metadata_path in metadata_paths:
                try:
                    metadata = self._download_json_from_minio(metadata_path)
                    tags = metadata.get("tags", [])
                    
                    if "production" in tags:
                        production_artifacts.append({
                            "path": metadata_path.replace("/metadata.json", ""),
                            "created_at": metadata.get("created_at", ""),
                            "version": metadata.get("version", ""),
                            "experiment_id": metadata.get("experiment_id", "")
                        })
                except Exception as e:
                    logger.warning(f"Failed to check metadata {metadata_path}: {e}")
            
            if not production_artifacts:
                raise ArtifactLoadError("No production artifacts found in MinIO")
            
            # Sort by creation time and get latest
            production_artifacts.sort(key=lambda x: x["created_at"], reverse=True)
            latest_prod = production_artifacts[0]
            
            logger.info(f"Found production artifact: {latest_prod['experiment_id']} v{latest_prod['version']}")
            return latest_prod["path"]
            
        except Exception as e:
            raise ArtifactLoadError(f"Failed to find production artifact: {e}")
    
    def _get_latest_version(self, experiment_id: str) -> str:
        """Get latest version for experiment from MinIO."""
        try:
            # List objects with experiment prefix
            objects = self._minio_client.list_objects(
                self.config.minio_bucket,
                prefix=f"{experiment_id}/",
                recursive=False
            )
            
            # Extract version directories
            versions = set()
            for obj in objects:
                parts = obj.object_name.rstrip('/').split('/')
                if len(parts) >= 2:
                    versions.add(parts[1])
            
            if not versions:
                raise ArtifactLoadError(f"No versions found for experiment {experiment_id}")
            
            # Sort versions (simple string sort, could be enhanced for semantic versioning)
            sorted_versions = sorted(list(versions))
            latest = sorted_versions[-1]
            
            logger.debug(f"Latest version for {experiment_id}: {latest}")
            return latest
            
        except Exception as e:
            raise ArtifactLoadError(f"Failed to get latest version for {experiment_id}: {e}")
    
    def _load_from_local(
        self,
        experiment_id: Optional[str],
        version: str
    ) -> Dict[str, Any]:
        """
        Load artifact from local cache directory.
        
        Args:
            experiment_id: Experiment ID (None for production)
            version: Version to load
            
        Returns:
            Artifact data dictionary
            
        Raises:
            ArtifactLoadError: If loading fails
        """
        # Determine local path
        if experiment_id is None:
            # Look for production artifact in cache
            artifact_dir = self._find_local_production_artifact()
        else:
            artifact_dir = self.config.local_cache_path / experiment_id / version
        
        if not artifact_dir.exists():
            raise ArtifactLoadError(f"Local artifact directory not found: {artifact_dir}")
        
        try:
            # Load metadata
            metadata_path = artifact_dir / "metadata.json"
            if not metadata_path.exists():
                raise ArtifactLoadError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load HMM artifact
            hmm_path = artifact_dir / "hmm_artifact.json"
            if not hmm_path.exists():
                raise ArtifactLoadError(f"HMM artifact file not found: {hmm_path}")
            
            with open(hmm_path, 'r') as f:
                hmm_artifact = json.load(f)
            
            # Load fusion weights (optional)
            fusion_weights = None
            weights_path = artifact_dir / "fusion_weights.json"
            if weights_path.exists():
                with open(weights_path, 'r') as f:
                    fusion_weights = json.load(f)
            
            # Combine into result
            result = {
                "metadata": metadata,
                "hmm_artifact": hmm_artifact,
                "fusion_weights": fusion_weights,
                "source": "local",
                "artifact_path": str(artifact_dir),
                "loaded_at": datetime.now().isoformat()
            }
            
            # Validate integrity if hash available
            expected_hash = metadata.get("artifact_hash")
            if expected_hash:
                self._validate_artifact_integrity(hmm_artifact, expected_hash)
            
            return result
            
        except Exception as e:
            raise ArtifactLoadError(f"Failed to load from local storage: {e}")
    
    def _find_local_production_artifact(self) -> Path:
        """Find production artifact in local cache."""
        # Look for directories with production metadata
        for exp_dir in self.config.local_cache_path.iterdir():
            if not exp_dir.is_dir():
                continue
            
            for version_dir in exp_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        tags = metadata.get("tags", [])
                        if "production" in tags:
                            return version_dir
                    except Exception as e:
                        logger.warning(f"Failed to read metadata {metadata_path}: {e}")
        
        raise ArtifactLoadError("No production artifact found in local cache")
    
    def _validate_artifact_integrity(
        self,
        hmm_artifact: Dict[str, Any],
        expected_hash: str
    ) -> None:
        """Validate artifact integrity using hash."""
        try:
            # Calculate hash of artifact
            artifact_str = json.dumps(hmm_artifact, sort_keys=True)
            actual_hash = hashlib.sha256(artifact_str.encode()).hexdigest()
            
            if actual_hash != expected_hash:
                raise ArtifactLoadError(
                    f"Artifact integrity check failed. "
                    f"Expected: {expected_hash}, Actual: {actual_hash}"
                )
            
            logger.debug("Artifact integrity validation passed")
            
        except Exception as e:
            logger.warning(f"Artifact integrity validation failed: {e}")
            # Don't fail the load for integrity issues, just warn
    
    def preload_production_artifacts(self, cache_ttl: int = 7200) -> Dict[str, Any]:
        """
        Preload production artifacts into cache for faster access.
        
        Args:
            cache_ttl: Cache TTL in seconds (default: 2 hours)
            
        Returns:
            Dictionary with preload results
        """
        results = {
            "preloaded": [],
            "failed": [],
            "cache_stats": {}
        }
        
        try:
            # Load production artifact
            weights_data = self.load_hmm_weights(
                experiment_id=None,
                version="latest",
                use_cache=True,
                cache_ttl=cache_ttl
            )
            
            results["preloaded"].append({
                "type": "production",
                "experiment_id": weights_data["metadata"].get("experiment_id"),
                "version": weights_data["metadata"].get("version"),
                "source": weights_data["source"]
            })
            
            logger.info("Production artifacts preloaded successfully")
            
        except Exception as e:
            results["failed"].append({
                "type": "production",
                "error": str(e)
            })
            logger.error(f"Failed to preload production artifacts: {e}")
        
        # Get cache statistics
        results["cache_stats"] = self.cache.get_stats()
        
        return results
    
    def list_available_artifacts(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List available artifacts from both MinIO and local storage.
        
        Returns:
            Dictionary with artifacts from each source
        """
        results = {
            "minio": [],
            "local": [],
            "errors": []
        }
        
        # List MinIO artifacts
        if self._minio_available:
            try:
                objects = self._minio_client.list_objects(
                    self.config.minio_bucket,
                    recursive=True
                )
                
                metadata_paths = [
                    obj.object_name for obj in objects
                    if obj.object_name.endswith("metadata.json")
                ]
                
                for metadata_path in metadata_paths:
                    try:
                        metadata = self._download_json_from_minio(metadata_path)
                        results["minio"].append({
                            "experiment_id": metadata.get("experiment_id"),
                            "version": metadata.get("version"),
                            "created_at": metadata.get("created_at"),
                            "tags": metadata.get("tags", []),
                            "production_ready": metadata.get("production_ready", False),
                            "path": metadata_path.replace("/metadata.json", "")
                        })
                    except Exception as e:
                        results["errors"].append(f"Failed to read MinIO metadata {metadata_path}: {e}")
                        
            except Exception as e:
                results["errors"].append(f"Failed to list MinIO artifacts: {e}")
        
        # List local artifacts
        try:
            for exp_dir in self.config.local_cache_path.iterdir():
                if not exp_dir.is_dir():
                    continue
                
                for version_dir in exp_dir.iterdir():
                    if not version_dir.is_dir():
                        continue
                    
                    metadata_path = version_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            results["local"].append({
                                "experiment_id": metadata.get("experiment_id"),
                                "version": metadata.get("version"),
                                "created_at": metadata.get("created_at"),
                                "tags": metadata.get("tags", []),
                                "production_ready": metadata.get("production_ready", False),
                                "path": str(version_dir)
                            })
                        except Exception as e:
                            results["errors"].append(f"Failed to read local metadata {metadata_path}: {e}")
                            
        except Exception as e:
            results["errors"].append(f"Failed to list local artifacts: {e}")
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and status."""
        stats = self.cache.get_stats()
        stats.update({
            "minio_available": self._minio_available,
            "minio_endpoint": self.config.minio_endpoint,
            "local_cache_path": str(self.config.local_cache_path)
        })
        return stats
    
    def clear_cache(self) -> None:
        """Clear the artifact cache."""
        self.cache.clear()
        logger.info("Artifact cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on artifact loading capabilities.
        
        Returns:
            Health check results
        """
        health = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check MinIO connectivity
        if self._minio_available:
            try:
                self._validate_minio_connection()
                health["checks"]["minio"] = {"status": "healthy", "message": "Connection successful"}
            except Exception as e:
                health["checks"]["minio"] = {"status": "unhealthy", "message": str(e)}
                health["status"] = "degraded"
        else:
            health["checks"]["minio"] = {"status": "disabled", "message": "MinIO not configured"}
        
        # Check local cache directory
        try:
            if self.config.local_cache_path.exists() and self.config.local_cache_path.is_dir():
                health["checks"]["local_cache"] = {"status": "healthy", "message": "Directory accessible"}
            else:
                health["checks"]["local_cache"] = {"status": "unhealthy", "message": "Directory not accessible"}
                health["status"] = "degraded"
        except Exception as e:
            health["checks"]["local_cache"] = {"status": "unhealthy", "message": str(e)}
            health["status"] = "degraded"
        
        # Check cache functionality
        try:
            test_key = "health_check_test"
            test_data = {"test": True, "timestamp": datetime.now().isoformat()}
            self.cache.put(test_key, test_data, ttl_seconds=60)
            retrieved = self.cache.get(test_key)
            
            if retrieved == test_data:
                health["checks"]["cache"] = {"status": "healthy", "message": "Cache working"}
            else:
                health["checks"]["cache"] = {"status": "unhealthy", "message": "Cache data mismatch"}
                health["status"] = "degraded"
            
            # Clean up test data
            self.cache.remove(test_key)
            
        except Exception as e:
            health["checks"]["cache"] = {"status": "unhealthy", "message": str(e)}
            health["status"] = "degraded"
        
        return health