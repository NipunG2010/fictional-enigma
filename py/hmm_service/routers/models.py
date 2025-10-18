"""
Model management endpoints for HMM Microservice.

Provides endpoints for model reloading, status, and management.
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from core.dependencies import get_model_loader_dep, get_inference_engine_dep
from core.logging_config import get_logger, PerformanceTimer


router = APIRouter()


class ModelReloadRequest(BaseModel):
    """Request model for model reloading."""
    
    experiment_id: Optional[str] = Field(
        None,
        description="Experiment ID to load (uses default if not specified)"
    )
    version: Optional[str] = Field(
        "latest",
        description="Model version to load"
    )
    validate_model: bool = Field(
        True,
        description="Whether to validate model before activation"
    )


class ModelReloadResponse(BaseModel):
    """Response model for model reloading."""
    
    success: bool = Field(..., description="Whether reload was successful")
    model_info: Dict[str, Any] = Field(..., description="Information about loaded model")
    reload_time: float = Field(..., description="Time taken to reload in seconds")
    previous_model: Optional[Dict[str, Any]] = Field(None, description="Previous model info")
    timestamp: int = Field(..., description="Reload timestamp")


class CurrentModelResponse(BaseModel):
    """Response model for current model information."""
    
    experiment_id: str = Field(..., description="Current experiment ID")
    version: str = Field(..., description="Current model version")
    load_time: float = Field(..., description="Model load timestamp")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    performance_stats: Dict[str, Any] = Field(..., description="Model performance statistics")
    timestamp: int = Field(..., description="Response timestamp")


class AvailableModelsResponse(BaseModel):
    """Response model for available models."""
    
    models: List[Dict[str, Any]] = Field(..., description="List of available models")
    total_count: int = Field(..., description="Total number of available models")
    timestamp: int = Field(..., description="Response timestamp")


@router.post(
    "/reload",
    response_model=ModelReloadResponse,
    summary="Reload HMM model",
    description="Hot-reload HMM model from MinIO storage"
)
async def reload_model(
    request: ModelReloadRequest,
    http_request: Request,
    model_loader=Depends(get_model_loader_dep),
    inference_engine=Depends(get_inference_engine_dep)
) -> ModelReloadResponse:
    """Reload HMM model from storage."""
    logger = get_logger("hmm_service.models")
    
    # Log model reload request start
    logger.info("Model reload requested", extra={
        "endpoint": "/models/reload",
        "experiment_id": request.experiment_id,
        "version": request.version,
        "validate_model": request.validate_model,
        "client_ip": http_request.client.host if http_request.client else None,
    })
    
    # Log audit trail for model reload request
    logger.log_audit(
        "Model reload requested",
        endpoint="/models/reload",
        method="POST",
        client_ip=http_request.client.host if http_request.client else None,
        experiment_id=request.experiment_id,
        version=request.version
    )
    
    with PerformanceTimer(logger, "model_reload", "/models/reload") as timer:
        try:
            # Get current model info before reload
            previous_model_info = inference_engine.get_model_info()
            previous_model = None
            if previous_model_info.get("loaded", False):
                previous_model = {
                    "experiment_id": "current_model",  # We don't track experiment_id in model info yet
                    "version": previous_model_info.get("version", "unknown"),
                    "load_time": previous_model_info.get("load_time")
                }
                
                logger.info("Current model identified for replacement", extra={
                    "current_version": previous_model_info.get("version", "unknown"),
                    "current_n_states": previous_model_info.get("n_states", 0),
                })
            
            # Determine experiment ID to load
            experiment_id = request.experiment_id
            if not experiment_id:
                from core.config import get_settings
                settings = get_settings()
                experiment_id = settings.default_experiment_id
                logger.info("Using default experiment ID", extra={
                    "experiment_id": experiment_id
                })
            
            # Load model through model loader with performance logging
            logger.info("Loading model from storage", extra={
                "experiment_id": experiment_id,
                "version": request.version,
                "validate": request.validate_model
            })
            
            with PerformanceTimer(logger, "model_storage_load", "/models/reload"):
                success = await model_loader.load_model(
                    experiment_id=experiment_id,
                    version=request.version,
                    validate=request.validate_model
                )
            
            if not success:
                raise Exception("Model loading failed")
            
            # Get the loaded model artifacts
            hmm_artifact, fusion_weights = model_loader.get_current_model()
            
            if hmm_artifact is None:
                raise Exception("No model artifact available after loading")
            
            logger.info("Model artifacts loaded successfully", extra={
                "experiment_id": experiment_id,
                "n_states": getattr(hmm_artifact, 'n_states', 0),
                "n_features": getattr(hmm_artifact, 'n_features', 0),
                "has_fusion_weights": fusion_weights is not None,
            })
            
            # Load model into inference engine
            with PerformanceTimer(logger, "inference_engine_load", "/models/reload"):
                await inference_engine.load_model(hmm_artifact, fusion_weights)
            
            # Get new model info
            new_model_info_dict = inference_engine.get_model_info()
            new_model_info = {
                "experiment_id": experiment_id,
                "version": new_model_info_dict.get("version", request.version),
                "load_time": new_model_info_dict.get("load_time"),
                "n_states": new_model_info_dict.get("n_states"),
                "n_features": new_model_info_dict.get("n_features"),
                "has_fusion_weights": new_model_info_dict.get("has_fusion_weights", False),
                "validation_passed": request.validate_model
            }
            
            reload_time = timer.start_time and (time.time() - timer.start_time) or 0
            
            # Log successful model reload
            logger.info("Model reload completed successfully", extra={
                "endpoint": "/models/reload",
                "experiment_id": experiment_id,
                "version": new_model_info_dict.get("version", request.version),
                "n_states": new_model_info_dict.get("n_states"),
                "n_features": new_model_info_dict.get("n_features"),
                "reload_time": round(reload_time, 3),
                "has_fusion_weights": new_model_info_dict.get("has_fusion_weights", False),
            })
            
            # Log audit trail for successful model reload
            logger.log_audit(
                "Model reload completed successfully",
                endpoint="/models/reload",
                method="POST",
                status_code=200,
                client_ip=http_request.client.host if http_request.client else None,
                experiment_id=experiment_id,
                version=new_model_info_dict.get("version", request.version),
                reload_time=round(reload_time, 3)
            )
            
            return ModelReloadResponse(
                success=True,
                model_info=new_model_info,
                reload_time=round(reload_time, 3),
                previous_model=previous_model,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            reload_time = timer.start_time and (time.time() - timer.start_time) or 0
            
            # Log model reload error
            logger.error("Model reload failed", extra={
                "endpoint": "/models/reload",
                "experiment_id": request.experiment_id,
                "version": request.version,
                "error": str(e),
                "error_type": type(e).__name__,
                "reload_time": round(reload_time, 3),
            }, exc_info=True)
            
            # Determine appropriate error code
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                status_code = 404
                error_code = "MODEL_NOT_FOUND"
            elif "validation" in str(e).lower():
                status_code = 400
                error_code = "MODEL_VALIDATION_FAILED"
            elif "connection" in str(e).lower() or "minio" in str(e).lower():
                status_code = 503
                error_code = "STORAGE_UNAVAILABLE"
            else:
                status_code = 500
                error_code = "MODEL_RELOAD_FAILED"
            
            # Log audit trail for failed model reload
            logger.log_audit(
                "Model reload failed",
                endpoint="/models/reload",
                method="POST",
                status_code=status_code,
                client_ip=http_request.client.host if http_request.client else None,
                experiment_id=request.experiment_id,
                version=request.version,
                error=str(e),
                error_code=error_code,
                reload_time=round(reload_time, 3)
            )
            
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": "MODEL_RELOAD_FAILED",
                    "error_code": error_code,
                    "message": f"Failed to reload model: {str(e)}",
                    "timestamp": int(time.time()),
                    "reload_time": round(reload_time, 3)
                }
            )


@router.get(
    "/current",
    response_model=CurrentModelResponse,
    summary="Get current model info",
    description="Get information about currently loaded model"
)
async def get_current_model(
    request: Request,
    model_loader=Depends(get_model_loader_dep),
    inference_engine=Depends(get_inference_engine_dep)
) -> CurrentModelResponse:
    """Get current model information."""
    logger = get_logger("hmm_service.models")
    
    # Log current model info request
    logger.debug("Current model info requested", extra={
        "endpoint": "/models/current",
        "client_ip": request.client.host if request.client else None,
    })
    
    with PerformanceTimer(logger, "get_current_model", "/models/current") as timer:
        try:
            # Get model info from inference engine
            model_info_dict = inference_engine.get_model_info()
            
            if not model_info_dict.get("loaded", False):
                logger.warning("No model currently loaded", extra={
                    "endpoint": "/models/current",
                })
                
                # Log audit trail for no model loaded
                logger.log_audit(
                    "Current model info request - no model loaded",
                    endpoint="/models/current",
                    method="GET",
                    status_code=404,
                    client_ip=request.client.host if request.client else None
                )
                
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "NO_MODEL_LOADED",
                        "error_code": "MODEL_NOT_FOUND",
                        "message": "No model is currently loaded",
                        "timestamp": int(time.time())
                    }
                )
            
            # Get additional model metadata from model loader if available
            try:
                hmm_artifact, _ = model_loader.get_current_model()
                training_samples = hmm_artifact.metadata.get("training_samples", 0) if hmm_artifact else 0
                validation_score = hmm_artifact.metadata.get("validation_score", 0.0) if hmm_artifact else 0.0
                artifact_size_mb = hmm_artifact.metadata.get("artifact_size_mb", 0.0) if hmm_artifact else 0.0
            except Exception as e:
                logger.debug("Could not retrieve additional model metadata", extra={
                    "error": str(e)
                })
                training_samples = 0
                validation_score = 0.0
                artifact_size_mb = 0.0
            
            model_info = {
                "n_states": model_info_dict.get("n_states", 0),
                "n_features": model_info_dict.get("n_features", 0),
                "has_fusion_weights": model_info_dict.get("has_fusion_weights", False),
                "library": model_info_dict.get("library", "unknown"),
                "training_window": model_info_dict.get("training_window", {}),
                "training_samples": training_samples,
                "validation_score": validation_score,
                "artifact_size_mb": artifact_size_mb
            }
            
            # Basic performance stats (will be enhanced with metrics in task 5)
            performance_stats = {
                "total_inferences": model_info_dict.get("inference_count", 0),
                "avg_inference_time_ms": 0.0,  # Will be calculated from metrics
                "error_rate": 0.0,  # Will be calculated from metrics
                "cache_hit_rate": 0.0  # Will be calculated from cache stats
            }
            
            # Get default experiment ID from settings
            from core.config import get_settings
            settings = get_settings()
            experiment_id = settings.default_experiment_id  # We don't track this in model info yet
            
            # Log successful current model info retrieval
            logger.info("Current model info retrieved successfully", extra={
                "endpoint": "/models/current",
                "experiment_id": experiment_id,
                "version": model_info_dict.get("version", "unknown"),
                "n_states": model_info_dict.get("n_states", 0),
                "n_features": model_info_dict.get("n_features", 0),
                "has_fusion_weights": model_info_dict.get("has_fusion_weights", False),
                "total_inferences": model_info_dict.get("inference_count", 0),
            })
            
            # Log audit trail for successful current model info
            logger.log_audit(
                "Current model info retrieved",
                endpoint="/models/current",
                method="GET",
                status_code=200,
                client_ip=request.client.host if request.client else None,
                experiment_id=experiment_id,
                version=model_info_dict.get("version", "unknown")
            )
            
            return CurrentModelResponse(
                experiment_id=experiment_id,
                version=model_info_dict.get("version", "unknown"),
                load_time=model_info_dict.get("load_time", 0),
                model_info=model_info,
                performance_stats=performance_stats,
                timestamp=int(time.time())
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log current model info error
            logger.error("Failed to get current model info", extra={
                "endpoint": "/models/current",
                "error": str(e),
                "error_type": type(e).__name__,
            }, exc_info=True)
            
            # Log audit trail for failed current model info
            logger.log_audit(
                "Current model info request failed",
                endpoint="/models/current",
                method="GET",
                status_code=500,
                client_ip=request.client.host if request.client else None,
                error=str(e)
            )
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "MODEL_INFO_FAILED",
                    "error_code": "INTERNAL_ERROR",
                    "message": f"Failed to get current model info: {str(e)}",
                    "timestamp": int(time.time())
                }
            )


@router.get(
    "/available",
    response_model=AvailableModelsResponse,
    summary="List available models",
    description="List all available models in MinIO storage"
)
async def list_available_models(
    request: Request,
    model_loader=Depends(get_model_loader_dep)
) -> AvailableModelsResponse:
    """List available models in storage."""
    logger = get_logger("hmm_service.models")
    
    # Log available models request
    logger.info("Available models list requested", extra={
        "endpoint": "/models/available",
        "client_ip": request.client.host if request.client else None,
    })
    
    # Log audit trail for available models request
    logger.log_audit(
        "Available models list requested",
        endpoint="/models/available",
        method="GET",
        client_ip=request.client.host if request.client else None
    )
    
    with PerformanceTimer(logger, "list_available_models", "/models/available") as timer:
        try:
            # Get available models from model loader
            with PerformanceTimer(logger, "storage_list_models", "/models/available"):
                available_models = await model_loader.list_available_models()
            
            # Convert to response format
            models = []
            for model_info in available_models:
                model_dict = {
                    "experiment_id": model_info.get("experiment_id", "unknown"),
                    "version": model_info.get("version", "unknown"),
                    "created_at": model_info.get("created_at", 0),
                    "size_mb": model_info.get("size_mb", 0.0),
                    "n_states": model_info.get("n_states", 0),
                    "n_features": model_info.get("n_features", 0),
                    "validation_score": model_info.get("validation_score", 0.0),
                    "has_fusion_weights": model_info.get("has_fusion_weights", False),
                    "library": model_info.get("library", "unknown")
                }
                models.append(model_dict)
            
            # Log successful available models retrieval
            logger.info("Available models retrieved successfully", extra={
                "endpoint": "/models/available",
                "total_models": len(models),
                "experiment_ids": list(set(m["experiment_id"] for m in models)),
                "model_count_by_experiment": {
                    exp_id: len([m for m in models if m["experiment_id"] == exp_id])
                    for exp_id in set(m["experiment_id"] for m in models)
                },
            })
            
            # Log audit trail for successful available models
            logger.log_audit(
                "Available models retrieved successfully",
                endpoint="/models/available",
                method="GET",
                status_code=200,
                client_ip=request.client.host if request.client else None,
                total_models=len(models)
            )
            
            return AvailableModelsResponse(
                models=models,
                total_count=len(models),
                timestamp=int(time.time())
            )
            
        except Exception as e:
            # Log available models error
            logger.error("Failed to list available models", extra={
                "endpoint": "/models/available",
                "error": str(e),
                "error_type": type(e).__name__,
            }, exc_info=True)
            
            # Determine appropriate error code
            if "connection" in str(e).lower() or "minio" in str(e).lower():
                status_code = 503
                error_code = "STORAGE_UNAVAILABLE"
            elif "not found" in str(e).lower():
                status_code = 404
                error_code = "NO_MODELS_FOUND"
            else:
                status_code = 500
                error_code = "LIST_MODELS_FAILED"
            
            # Log audit trail for failed available models
            logger.log_audit(
                "Available models request failed",
                endpoint="/models/available",
                method="GET",
                status_code=status_code,
                client_ip=request.client.host if request.client else None,
                error=str(e),
                error_code=error_code
            )
            
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": "LIST_MODELS_FAILED",
                    "error_code": error_code,
                    "message": f"Failed to list available models: {str(e)}",
                    "timestamp": int(time.time())
                }
            )