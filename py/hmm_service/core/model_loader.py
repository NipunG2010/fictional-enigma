"""
Model Loader for HMM Microservice.

Handles loading HMM models from MinIO storage with versioning,
validation, and hot-reloading capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import json

from .config import ServiceConfig
from .error_handling import ModelError, NetworkError, SystemError, get_error_handler
from .circuit_breaker import get_circuit_breaker_manager

# Import HMM models and artifact management
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from imp.hmm.models import HMMArtifact, FusionWeights
from imp.hmm.artifact_management import MinIOArtifactStore, MinIOConfig, ArtifactValidator

logger = logging.getLogger(__name__)


# Legacy exception classes for backward compatibility
class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""
    pass


class ModelValidationError(ModelError):
    """Exception raised when model validation fails."""
    pass


class ModelLoader:
    """
    Model loader with MinIO integration and hot-reloading support.
    
    Manages loading, validation, and versioning of HMM models and fusion weights
    from MinIO storage with fallback mechanisms and automatic reloading.
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize model loader.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self._initialized = False
        
        # Error handling
        self._error_handler = get_error_handler()
        
        # Circuit breaker for MinIO operations
        circuit_manager = get_circuit_breaker_manager()
        self._minio_circuit_breaker = circuit_manager.get_circuit_breaker("minio")
        if not self._minio_circuit_breaker:
            # Create if it doesn't exist
            from .circuit_breaker import create_minio_circuit_breaker
            self._minio_circuit_breaker = create_minio_circuit_breaker()
        
        # MinIO integration
        self._minio_store: Optional[MinIOArtifactStore] = None
        self._minio_available = False
        self._performance_manager: Optional["PerformanceManager"] = None
        
        # Current model state
        self._current_artifact: Optional[HMMArtifact] = None
        self._current_weights: Optional[FusionWeights] = None
        self._current_version: Optional[str] = None
        self._load_time: Optional[float] = None
        
        # Fallback model state
        self._fallback_artifact: Optional[HMMArtifact] = None
        self._fallback_weights: Optional[FusionWeights] = None
        self._fallback_version: Optional[str] = None
        
        # Model management
        self._available_models: List[Dict[str, Any]] = []
        self._last_model_check: Optional[float] = None
        
        # Hot-reloading
        self._reload_task: Optional[asyncio.Task] = None
        self._reload_enabled = True
        
        # Statistics
        self._load_count = 0
        self._reload_count = 0
        self._fallback_count = 0
        
        logger.info("Model Loader initialized")
    
    async def initialize(self, performance_manager: Optional["PerformanceManager"] = None):
        """Initialize model loader and MinIO connection."""
        if self._initialized:
            logger.debug("Model loader already initialized")
            return
        
        logger.info("Initializing Model Loader...")
        
        # Store performance manager reference
        self._performance_manager = performance_manager
        
        # Initialize MinIO connection
        await self._initialize_minio()
        
        # Load default model
        await self._load_default_model()
        
        # Start hot-reloading task if enabled
        if self._reload_enabled and self.config.model_reload_interval > 0:
            self._reload_task = asyncio.create_task(self._periodic_reload_check())
            logger.info(f"Hot-reloading enabled with {self.config.model_reload_interval}s interval")
        
        self._initialized = True
        logger.info("Model Loader initialization complete")
    
    async def cleanup(self):
        """Cleanup model loader resources."""
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Model Loader cleanup complete")
    
    async def _initialize_minio(self):
        """Initialize MinIO connection with circuit breaker protection."""
        async def _create_minio_store():
            # Create MinIO configuration from service config
            minio_config = MinIOConfig(
                endpoint=self.config.minio.endpoint,
                access_key=self.config.minio.access_key,
                secret_key=self.config.minio.secret_key,
                secure=self.config.minio.secure,
                bucket_name=self.config.minio.bucket
            )
            
            store = MinIOArtifactStore(minio_config)
            
            # Test connection by listing experiments (if available)
            try:
                # This will test the connection
                list(store.list_experiments())
            except Exception as test_error:
                logger.warning(f"MinIO connection test failed: {test_error}")
                # Still return the store - it might work for other operations
            
            return store
        
        try:
            # Use circuit breaker for MinIO initialization
            self._minio_store = await self._minio_circuit_breaker.call(_create_minio_store)
            self._minio_available = True
            logger.info(f"MinIO connection established: {self.config.minio.endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO: {e}")
            self._minio_available = False
            
            # Handle circuit breaker specific errors
            if "circuit breaker" in str(e).lower():
                logger.warning("MinIO circuit breaker is open - will retry later")
            
            # Continue without MinIO - will use fallback mechanisms
    
    async def _load_default_model(self):
        """Load the default production model."""
        try:
            if self._minio_available and self._minio_store:
                # Try to load production model from MinIO
                logger.info(f"Loading default model: {self.config.default_experiment_id}")
                await self.load_model_from_minio(
                    experiment_id=self.config.default_experiment_id,
                    version="latest",
                    set_as_fallback=True
                )
            else:
                logger.warning("MinIO not available, no default model loaded")
                
        except Exception as e:
            logger.error(f"Failed to load default model: {e}")
            # Continue without default model - service will handle gracefully
    
    async def load_model_from_minio(
        self,
        experiment_id: str,
        version: str = "latest",
        validate: bool = True,
        set_as_fallback: bool = False
    ) -> Tuple[HMMArtifact, Optional[FusionWeights]]:
        """
        Load model from MinIO storage with circuit breaker protection.
        
        Args:
            experiment_id: Experiment identifier
            version: Model version ("latest" for most recent)
            validate: Whether to validate model before loading
            set_as_fallback: Whether to set as fallback model
            
        Returns:
            Tuple of (HMMArtifact, FusionWeights)
            
        Raises:
            ModelError: If loading fails
            NetworkError: If MinIO connection fails
        """
        if not self._minio_available or not self._minio_store:
            raise NetworkError(
                "MinIO not available for model loading",
                error_code="MINIO_NOT_AVAILABLE",
                service="minio",
                details={
                    "experiment_id": experiment_id,
                    "version": version,
                    "minio_available": self._minio_available
                }
            )
        
        async def _load_from_minio():
            start_time = time.time()
            logger.info(f"Loading model {experiment_id} v{version} from MinIO...")
            
            # Download artifact from MinIO with circuit breaker protection
            artifact_data = self._minio_store.download_artifact(
                experiment_id=experiment_id,
                version=version,
                validate_integrity=True
            )
            
            # Reconstruct HMM artifact
            hmm_artifact = HMMArtifact(**artifact_data["hmm_artifact"])
            
            # Reconstruct fusion weights if available
            fusion_weights = None
            if artifact_data.get("fusion_weights"):
                fusion_weights = FusionWeights(**artifact_data["fusion_weights"])
            
            # Validate model if requested
            if validate:
                await self._validate_model(hmm_artifact, fusion_weights)
            
            load_duration = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_duration:.3f}s: {experiment_id} v{hmm_artifact.version}")
            
            return hmm_artifact, fusion_weights
        
        try:
            # Use circuit breaker for MinIO operations
            hmm_artifact, fusion_weights = await self._minio_circuit_breaker.call(_load_from_minio)
            
            # Update current or fallback model
            if set_as_fallback:
                self._fallback_artifact = hmm_artifact
                self._fallback_weights = fusion_weights
                self._fallback_version = hmm_artifact.version
                logger.info(f"Fallback model set: {experiment_id} v{hmm_artifact.version}")
            else:
                self._current_artifact = hmm_artifact
                self._current_weights = fusion_weights
                self._current_version = hmm_artifact.version
                self._load_time = time.time()
                self._load_count += 1
            
            return hmm_artifact, fusion_weights
            
        except NetworkError:
            # Re-raise network errors (including circuit breaker errors)
            raise
            
        except Exception as e:
            logger.error(f"Failed to load model {experiment_id} v{version}: {e}")
            
            # Convert to appropriate error type
            if "validation" in str(e).lower():
                raise ModelError(
                    f"Model validation failed: {str(e)}",
                    error_code="MODEL_VALIDATION_FAILED",
                    model_version=version,
                    fallback_available=self._fallback_artifact is not None,
                    details={
                        "experiment_id": experiment_id,
                        "version": version,
                        "validation_error": str(e)
                    }
                )
            else:
                raise ModelError(
                    f"Model loading failed: {str(e)}",
                    error_code="MODEL_LOAD_FAILED",
                    model_version=version,
                    fallback_available=self._fallback_artifact is not None,
                    details={
                        "experiment_id": experiment_id,
                        "version": version,
                        "original_error": str(e),
                        "error_type": type(e).__name__
                    }
                )
    
    async def _validate_model(self, hmm_artifact: HMMArtifact, fusion_weights: Optional[FusionWeights]):
        """
        Validate model for production use.
        
        Args:
            hmm_artifact: HMM artifact to validate
            fusion_weights: Optional fusion weights to validate
            
        Raises:
            ModelValidationError: If validation fails
        """
        try:
            # Create temporary research artifact for validation
            from imp.hmm.artifact_management import ResearchArtifact, ResearchMetadata
            
            research_metadata = ResearchMetadata(
                experiment_id="validation",
                researcher="model_loader",
                library_used=hmm_artifact.metadata.get("library", "unknown")
            )
            
            research_artifact = ResearchArtifact(
                base_artifact=hmm_artifact,
                research_metadata=research_metadata,
                artifact_version=hmm_artifact.version,
                artifact_hash="",  # Skip hash validation for now
                production_ready=True
            )
            
            # Run validation checks
            validation_results = ArtifactValidator.run_all_validations(research_artifact)
            
            if not validation_results["all_passed"]:
                failed_checks = validation_results.get("failed_checks", [])
                raise ModelValidationError(f"Model validation failed: {', '.join(failed_checks)}")
            
            logger.debug("Model validation passed")
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            raise ModelValidationError(f"Validation failed: {e}")
    
    async def reload_current_model(self, validate: bool = True) -> bool:
        """
        Reload the current model from MinIO.
        
        Args:
            validate: Whether to validate reloaded model
            
        Returns:
            True if reload successful
        """
        if not self._current_version:
            logger.warning("No current model to reload")
            return False
        
        try:
            logger.info(f"Reloading current model: {self.config.default_experiment_id} v{self._current_version}")
            
            # Try to reload current model
            await self.load_model_from_minio(
                experiment_id=self.config.default_experiment_id,
                version=self._current_version,
                validate=validate
            )
            
            self._reload_count += 1
            logger.info("Model reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            
            # Try fallback to previous version
            if await self._try_fallback():
                return True
            
            return False
    
    async def load_latest_model(self, validate: bool = True) -> bool:
        """
        Load the latest version of the default model.
        
        Args:
            validate: Whether to validate loaded model
            
        Returns:
            True if load successful
        """
        try:
            logger.info(f"Loading latest model: {self.config.default_experiment_id}")
            
            # Load latest version
            await self.load_model_from_minio(
                experiment_id=self.config.default_experiment_id,
                version="latest",
                validate=validate
            )
            
            logger.info("Latest model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load latest model: {e}")
            
            # Try fallback
            if await self._try_fallback():
                return True
            
            return False
    
    async def _try_fallback(self) -> bool:
        """
        Try to fallback to the fallback model.
        
        Returns:
            True if fallback successful
        """
        if not self._fallback_artifact:
            logger.error("No fallback model available")
            return False
        
        try:
            logger.warning(f"Falling back to model version: {self._fallback_version}")
            
            # Use fallback model as current
            self._current_artifact = self._fallback_artifact
            self._current_weights = self._fallback_weights
            self._current_version = self._fallback_version
            self._load_time = time.time()
            self._fallback_count += 1
            
            logger.info("Fallback model activated")
            return True
            
        except Exception as e:
            logger.error(f"Fallback failed: {e}")
            return False
    
    async def _periodic_reload_check(self):
        """Periodic task to check for model updates."""
        while True:
            try:
                await asyncio.sleep(self.config.model_reload_interval)
                
                if not self._minio_available or not self._minio_store:
                    continue
                
                # Check for newer model versions
                await self._check_for_updates()
                
            except asyncio.CancelledError:
                logger.info("Model reload check task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in model reload check: {e}")
    
    async def _check_for_updates(self):
        """Check for model updates in MinIO."""
        try:
            # List available models
            artifacts = self._minio_store.list_artifacts(
                experiment_id=self.config.default_experiment_id,
                tags=["production"]
            )
            
            if not artifacts:
                logger.debug("No production models found")
                return
            
            # Find latest version
            latest_artifact = max(artifacts, key=lambda x: x.get("created_at", ""))
            latest_version = latest_artifact["version"]
            
            # Check if we have a newer version
            if latest_version != self._current_version:
                logger.info(f"New model version available: {latest_version} (current: {self._current_version})")
                
                # Try to load new version
                try:
                    await self.load_model_from_minio(
                        experiment_id=self.config.default_experiment_id,
                        version=latest_version,
                        validate=True
                    )
                    logger.info(f"Auto-updated to model version: {latest_version}")
                    
                except Exception as e:
                    logger.error(f"Failed to auto-update model: {e}")
            
            # Update available models list
            self._available_models = artifacts
            self._last_model_check = time.time()
            
        except Exception as e:
            logger.error(f"Failed to check for model updates: {e}")
    
    def get_current_model(self) -> Tuple[Optional[HMMArtifact], Optional[FusionWeights]]:
        """
        Get currently loaded model.
        
        Returns:
            Tuple of (HMMArtifact, FusionWeights) or (None, None) if no model loaded
        """
        return self._current_artifact, self._current_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self._current_artifact:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "version": self._current_version,
            "experiment_id": self.config.default_experiment_id,
            "n_states": self._current_artifact.n_states,
            "library": self._current_artifact.metadata.get("library", "unknown"),
            "load_time": self._load_time,
            "has_fusion_weights": self._current_weights is not None,
            "has_fallback": self._fallback_artifact is not None,
            "fallback_version": self._fallback_version,
            "training_window": {
                "start": self._current_artifact.training_window_start,
                "end": self._current_artifact.training_window_end
            }
        }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        
        Returns:
            List of model metadata dictionaries
        """
        return self._available_models.copy()
    
    async def refresh_available_models(self) -> List[Dict[str, Any]]:
        """
        Refresh list of available models from MinIO.
        
        Returns:
            Updated list of model metadata dictionaries
        """
        if not self._minio_available or not self._minio_store:
            return []
        
        try:
            artifacts = self._minio_store.list_artifacts()
            self._available_models = artifacts
            self._last_model_check = time.time()
            
            logger.info(f"Refreshed available models: {len(artifacts)} found")
            return artifacts
            
        except Exception as e:
            logger.error(f"Failed to refresh available models: {e}")
            return self._available_models
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get model loader statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "initialized": self._initialized,
            "minio_available": self._minio_available,
            "load_count": self._load_count,
            "reload_count": self._reload_count,
            "fallback_count": self._fallback_count,
            "current_model_loaded": self._current_artifact is not None,
            "fallback_model_available": self._fallback_artifact is not None,
            "last_model_check": self._last_model_check,
            "available_models_count": len(self._available_models),
            "reload_enabled": self._reload_enabled,
            "reload_interval": self.config.model_reload_interval
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of model loader including circuit breaker information.
        
        Returns:
            Dictionary with health information
        """
        # Get circuit breaker status
        circuit_breaker_stats = self._minio_circuit_breaker.get_stats()
        circuit_breaker_healthy = self._minio_circuit_breaker.get_state().value != "open"
        
        return {
            "initialized": self._initialized,
            "minio_connected": self._minio_available,
            "minio_circuit_breaker_healthy": circuit_breaker_healthy,
            "model_loaded": self._current_artifact is not None,
            "fallback_available": self._fallback_artifact is not None,
            "reload_task_running": self._reload_task is not None and not self._reload_task.done(),
            "last_load_time": self._load_time,
            "current_version": self._current_version,
            "circuit_breaker": circuit_breaker_stats,
            "statistics": self.get_statistics()
        }
    
    async def set_reload_enabled(self, enabled: bool):
        """
        Enable or disable automatic reloading.
        
        Args:
            enabled: Whether to enable automatic reloading
        """
        if enabled == self._reload_enabled:
            return
        
        self._reload_enabled = enabled
        
        if enabled and self.config.model_reload_interval > 0:
            # Start reload task
            if not self._reload_task or self._reload_task.done():
                self._reload_task = asyncio.create_task(self._periodic_reload_check())
                logger.info("Automatic model reloading enabled")
        else:
            # Stop reload task
            if self._reload_task:
                self._reload_task.cancel()
                try:
                    await self._reload_task
                except asyncio.CancelledError:
                    pass
                logger.info("Automatic model reloading disabled")
    
    async def load_model(self, experiment_id: str, version: str = "latest", validate: bool = True) -> bool:
        """
        Load model by experiment ID and version.
        
        Args:
            experiment_id: Experiment identifier
            version: Model version to load
            validate: Whether to validate model
            
        Returns:
            True if successful
        """
        try:
            await self.load_model_from_minio(
                experiment_id=experiment_id,
                version=version,
                validate=validate
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load model {experiment_id} v{version}: {e}")
            return False
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in storage.
        
        Returns:
            List of model information dictionaries
        """
        if not self._minio_available or not self._minio_store:
            # Return cached list if MinIO not available
            return self._available_models
        
        try:
            # Refresh from MinIO
            artifacts = await self.refresh_available_models()
            
            # Convert to expected format
            models = []
            for artifact in artifacts:
                model_info = {
                    "experiment_id": artifact.get("experiment_id", "unknown"),
                    "version": artifact.get("version", "unknown"),
                    "created_at": artifact.get("created_at", 0),
                    "size_mb": artifact.get("size_mb", 0.0),
                    "n_states": artifact.get("n_states", 0),
                    "n_features": artifact.get("n_features", 0),
                    "validation_score": artifact.get("validation_score", 0.0),
                    "has_fusion_weights": artifact.get("has_fusion_weights", False),
                    "library": artifact.get("library", "unknown"),
                    "tags": artifact.get("tags", [])
                }
                models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            # Return cached list as fallback
            return self._available_models