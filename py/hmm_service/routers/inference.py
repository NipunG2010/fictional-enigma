"""
Inference endpoints for HMM Microservice.

Provides REST endpoints for state probability calculation and fusion weight computation.
"""

import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, validator

from core.dependencies import get_inference_engine_dep, get_metrics_collector_dep, get_performance_manager_dep
from core.logging_config import get_logger, PerformanceTimer
from core.error_handling import (
    ValidationError, ModelError, SystemError, NetworkError, TimeoutError,
    ErrorContext, get_error_handler, create_error_context
)


router = APIRouter()


# Request/Response Models
class InferenceRequest(BaseModel):
    """Request model for HMM inference."""
    
    observations: List[float] = Field(
        ..., 
        min_items=3, 
        max_items=3,
        description="Observation vector [s_LDC, s_MR, s_TSMOM]"
    )
    timestamp: Optional[int] = Field(
        None,
        description="Unix timestamp for the observation"
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional request identifier"
    )
    
    @validator("observations")
    def validate_observations(cls, v):
        """Validate observation values."""
        if len(v) != 3:
            raise ValueError("Observations must contain exactly 3 values")
        
        # Check for valid numeric values
        for i, obs in enumerate(v):
            if not isinstance(obs, (int, float)):
                raise ValueError(f"Observation {i} must be numeric")
            if abs(obs) > 1000:  # Reasonable bounds check
                raise ValueError(f"Observation {i} value {obs} is out of reasonable range")
        
        return v


class StateProbabilitiesResponse(BaseModel):
    """Response model for state probabilities."""
    
    state_probabilities: List[float] = Field(
        ...,
        description="Probability distribution over HMM states"
    )
    most_likely_state: int = Field(
        ...,
        description="Index of the most likely state"
    )
    confidence: float = Field(
        ...,
        description="Confidence score (max probability)"
    )
    timestamp: int = Field(
        ...,
        description="Response timestamp"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request identifier if provided"
    )


class FusionWeightsResponse(BaseModel):
    """Response model for fusion weights."""
    
    weights: Dict[str, float] = Field(
        ...,
        description="Fusion weights for each signal type"
    )
    state_probabilities: List[float] = Field(
        ...,
        description="State probabilities used for weight computation"
    )
    most_likely_state: int = Field(
        ...,
        description="Index of the most likely state"
    )
    timestamp: int = Field(
        ...,
        description="Response timestamp"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request identifier if provided"
    )


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    
    state_probabilities: List[float] = Field(
        ...,
        description="Probability distribution over HMM states"
    )
    most_likely_state: int = Field(
        ...,
        description="Index of the most likely state"
    )
    confidence: float = Field(
        ...,
        description="Confidence score (max probability)"
    )
    fusion_weights: Dict[str, float] = Field(
        ...,
        description="Fusion weights for each signal type"
    )
    timestamp: int = Field(
        ...,
        description="Response timestamp"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    model_version: str = Field(
        ...,
        description="Version of the HMM model used"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request identifier if provided"
    )


# Endpoints
@router.post(
    "/state-probabilities",
    response_model=StateProbabilitiesResponse,
    summary="Calculate HMM state probabilities",
    description="Calculate state probabilities for given observation vector"
)
async def calculate_state_probabilities(
    request: InferenceRequest,
    http_request: Request,
    inference_engine=Depends(get_inference_engine_dep),
    metrics=Depends(get_metrics_collector_dep),
    performance_manager=Depends(get_performance_manager_dep)
) -> StateProbabilitiesResponse:
    """Calculate HMM state probabilities for observation vector."""
    import numpy as np
    
    logger = get_logger("hmm_service.inference")
    request_id = request.request_id or getattr(http_request.state, "request_id", None)
    
    # Log inference request start with input validation
    logger.info("State probabilities calculation started", extra={
        "endpoint": "/inference/state-probabilities",
        "observations": request.observations,
        "input_timestamp": request.timestamp,
        "observations_count": len(request.observations),
        "client_ip": http_request.client.host if http_request.client else None,
    })
    
    # Log audit trail for inference request
    logger.log_audit(
        "State probabilities inference requested",
        endpoint="/inference/state-probabilities",
        method="POST",
        client_ip=http_request.client.host if http_request.client else None,
        observations_count=len(request.observations)
    )
    
    async def process_inference():
        # Convert observations to numpy array
        observations = np.array(request.observations, dtype=np.float64)
        
        # Log performance details for inference computation
        with PerformanceTimer(logger, "hmm_inference_computation", "/inference/state-probabilities") as timer:
            return await inference_engine.predict_state_probabilities(observations)
    
    try:
        # Process request through performance manager (with queue and timeout handling)
        with PerformanceTimer(logger, "state_probabilities_request", "/inference/state-probabilities") as timer:
            state_probs = await performance_manager.process_request(
                request_handler=process_inference,
                request_id=request_id
            )
        
        # Find most likely state and confidence
        most_likely_state = int(np.argmax(state_probs))
        confidence = float(state_probs[most_likely_state])
        
        processing_time = timer.start_time and (time.time() - timer.start_time) * 1000 or 0
        
        # Log successful inference with results
        logger.info("State probabilities calculated successfully", extra={
            "endpoint": "/inference/state-probabilities",
            "most_likely_state": most_likely_state,
            "confidence": confidence,
            "state_count": len(state_probs),
            "processing_time_ms": round(processing_time, 2),
        })
        
        response = StateProbabilitiesResponse(
            state_probabilities=state_probs.tolist(),
            most_likely_state=most_likely_state,
            confidence=confidence,
            timestamp=request.timestamp or int(time.time()),
            processing_time_ms=round(processing_time, 2),
            request_id=request_id
        )
        
        # Log audit trail for successful response
        logger.log_audit(
            "State probabilities inference completed",
            endpoint="/inference/state-probabilities",
            method="POST",
            status_code=200,
            client_ip=http_request.client.host if http_request.client else None,
            most_likely_state=most_likely_state,
            confidence=confidence,
            processing_time_ms=round(processing_time, 2)
        )
        
        # Record metrics (will be implemented in task 5)
        # metrics.record_inference_request("state_probabilities", processing_time)
        
        return response
        
    except Exception as e:
        processing_time = time.time() * 1000  # Fallback timing
        
        # Log inference error with full context
        logger.error("State probabilities calculation failed", extra={
            "endpoint": "/inference/state-probabilities",
            "error": str(e),
            "error_type": type(e).__name__,
            "observations": request.observations,
            "processing_time_ms": round(processing_time, 2),
        }, exc_info=True)
        
        # Use comprehensive error handler
        error_handler = get_error_handler()
        context = create_error_context(http_request, processing_time)
        
        # Handle the error and create HTTP exception
        if isinstance(e, (ValidationError, ModelError, SystemError, NetworkError, TimeoutError)):
            # Already a structured error
            http_exception = error_handler.create_http_exception(e, context)
        else:
            # Convert generic exception to structured error
            error_response = error_handler.handle_error(e, context)
            http_exception = HTTPException(
                status_code=400 if "validation" in str(e).lower() else 500,
                detail=error_response.dict()
            )
        
        # Record error metrics (will be implemented in task 5)
        # metrics.record_inference_error("state_probabilities", str(e))
        
        raise http_exception


@router.post(
    "/fusion-weights",
    response_model=FusionWeightsResponse,
    summary="Calculate fusion weights",
    description="Calculate fusion weights based on current market regime"
)
async def calculate_fusion_weights(
    request: InferenceRequest,
    http_request: Request,
    inference_engine=Depends(get_inference_engine_dep),
    metrics=Depends(get_metrics_collector_dep),
    performance_manager=Depends(get_performance_manager_dep)
) -> FusionWeightsResponse:
    """Calculate fusion weights for current market regime."""
    import numpy as np
    
    logger = get_logger("hmm_service.inference")
    request_id = request.request_id or getattr(http_request.state, "request_id", None)
    
    # Log fusion weights request start
    logger.info("Fusion weights calculation started", extra={
        "endpoint": "/inference/fusion-weights",
        "observations": request.observations,
        "input_timestamp": request.timestamp,
        "observations_count": len(request.observations),
        "client_ip": http_request.client.host if http_request.client else None,
    })
    
    # Log audit trail for fusion weights request
    logger.log_audit(
        "Fusion weights inference requested",
        endpoint="/inference/fusion-weights",
        method="POST",
        client_ip=http_request.client.host if http_request.client else None,
        observations_count=len(request.observations)
    )
    
    async def process_fusion_weights():
        # Convert observations to numpy array
        observations = np.array(request.observations, dtype=np.float64)
        
        # Calculate state probabilities first with performance logging
        with PerformanceTimer(logger, "state_probabilities_computation", "/inference/fusion-weights"):
            state_probs = await inference_engine.predict_state_probabilities(observations)
        
        # Calculate fusion weights based on state probabilities
        with PerformanceTimer(logger, "fusion_weights_computation", "/inference/fusion-weights"):
            weights = await inference_engine.compute_fusion_weights(state_probs)
        
        return state_probs, weights
    
    try:
        # Process request through performance manager
        with PerformanceTimer(logger, "fusion_weights_request", "/inference/fusion-weights") as timer:
            state_probs, weights = await performance_manager.process_request(
                request_handler=process_fusion_weights,
                request_id=request_id
            )
        
        # Find most likely state
        most_likely_state = int(np.argmax(state_probs))
        
        processing_time = timer.start_time and (time.time() - timer.start_time) * 1000 or 0
        
        # Log successful fusion weights calculation with results
        logger.info("Fusion weights calculated successfully", extra={
            "endpoint": "/inference/fusion-weights",
            "most_likely_state": most_likely_state,
            "weights": weights,
            "state_count": len(state_probs),
            "processing_time_ms": round(processing_time, 2),
        })
        
        response = FusionWeightsResponse(
            weights=weights,
            state_probabilities=state_probs.tolist(),
            most_likely_state=most_likely_state,
            timestamp=request.timestamp or int(time.time()),
            processing_time_ms=round(processing_time, 2),
            request_id=request_id
        )
        
        # Log audit trail for successful response
        logger.log_audit(
            "Fusion weights inference completed",
            endpoint="/inference/fusion-weights",
            method="POST",
            status_code=200,
            client_ip=http_request.client.host if http_request.client else None,
            most_likely_state=most_likely_state,
            weights=weights,
            processing_time_ms=round(processing_time, 2)
        )
        
        # Record metrics (will be implemented in task 5)
        # metrics.record_inference_request("fusion_weights", processing_time)
        
        return response
        
    except Exception as e:
        processing_time = time.time() * 1000  # Fallback timing
        
        # Log fusion weights error with full context
        logger.error("Fusion weights calculation failed", extra={
            "endpoint": "/inference/fusion-weights",
            "error": str(e),
            "error_type": type(e).__name__,
            "observations": request.observations,
            "processing_time_ms": round(processing_time, 2),
        }, exc_info=True)
        
        # Use comprehensive error handler
        error_handler = get_error_handler()
        context = create_error_context(http_request, processing_time)
        
        # Handle the error and create HTTP exception
        if isinstance(e, (ValidationError, ModelError, SystemError, NetworkError, TimeoutError)):
            # Already a structured error
            http_exception = error_handler.create_http_exception(e, context)
        else:
            # Convert generic exception to structured error
            error_response = error_handler.handle_error(e, context)
            http_exception = HTTPException(
                status_code=400 if "validation" in str(e).lower() else 500,
                detail=error_response.dict()
            )
        
        # Record error metrics (will be implemented in task 5)
        # metrics.record_inference_error("fusion_weights", str(e))
        
        raise http_exception


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Complete HMM prediction",
    description="Get complete prediction with state probabilities and fusion weights"
)
async def predict(
    request: InferenceRequest,
    http_request: Request,
    inference_engine=Depends(get_inference_engine_dep),
    metrics=Depends(get_metrics_collector_dep),
    performance_manager=Depends(get_performance_manager_dep)
) -> PredictionResponse:
    """Get complete HMM prediction with state probabilities and fusion weights."""
    import numpy as np
    
    logger = get_logger("hmm_service.inference")
    request_id = request.request_id or getattr(http_request.state, "request_id", None)
    
    # Log complete prediction request start
    logger.info("Complete prediction started", extra={
        "endpoint": "/inference/predict",
        "observations": request.observations,
        "input_timestamp": request.timestamp,
        "observations_count": len(request.observations),
        "client_ip": http_request.client.host if http_request.client else None,
    })
    
    # Log audit trail for complete prediction request
    logger.log_audit(
        "Complete prediction inference requested",
        endpoint="/inference/predict",
        method="POST",
        client_ip=http_request.client.host if http_request.client else None,
        observations_count=len(request.observations)
    )
    
    async def process_complete_prediction():
        # Convert observations to numpy array
        observations = np.array(request.observations, dtype=np.float64)
        
        # Get complete prediction from inference engine with performance logging
        with PerformanceTimer(logger, "complete_prediction_computation", "/inference/predict"):
            return await inference_engine.predict_complete(
                observations, 
                request_id=request_id
            )
    
    try:
        # Process request through performance manager
        with PerformanceTimer(logger, "complete_prediction_request", "/inference/predict") as timer:
            prediction = await performance_manager.process_request(
                request_handler=process_complete_prediction,
                request_id=request_id
            )
        
        # Get model info for version
        model_info = inference_engine.get_model_info()
        model_version = model_info.get("version", "unknown")
        
        processing_time = timer.start_time and (time.time() - timer.start_time) * 1000 or 0
        
        # Log successful complete prediction with comprehensive results
        logger.info("Complete prediction generated successfully", extra={
            "endpoint": "/inference/predict",
            "most_likely_state": prediction.most_likely_state,
            "confidence": prediction.confidence,
            "fusion_weights": prediction.fused_weights,
            "model_version": model_version,
            "state_count": len(prediction.state_probabilities),
            "processing_time_ms": round(processing_time, 2),
        })
        
        response = PredictionResponse(
            state_probabilities=prediction.state_probabilities,
            most_likely_state=prediction.most_likely_state,
            confidence=prediction.confidence,
            fusion_weights=prediction.fused_weights,
            timestamp=request.timestamp or int(time.time()),
            processing_time_ms=round(processing_time, 2),
            model_version=model_version,
            request_id=request_id
        )
        
        # Log audit trail for successful complete prediction
        logger.log_audit(
            "Complete prediction inference completed",
            endpoint="/inference/predict",
            method="POST",
            status_code=200,
            client_ip=http_request.client.host if http_request.client else None,
            most_likely_state=prediction.most_likely_state,
            confidence=prediction.confidence,
            model_version=model_version,
            processing_time_ms=round(processing_time, 2)
        )
        
        # Record metrics (will be implemented in task 5)
        # metrics.record_inference_request("predict", processing_time)
        
        return response
        
    except Exception as e:
        processing_time = time.time() * 1000  # Fallback timing
        
        # Log complete prediction error with full context
        logger.error("Complete prediction failed", extra={
            "endpoint": "/inference/predict",
            "error": str(e),
            "error_type": type(e).__name__,
            "observations": request.observations,
            "processing_time_ms": round(processing_time, 2),
        }, exc_info=True)
        
        # Use comprehensive error handler
        error_handler = get_error_handler()
        context = create_error_context(http_request, processing_time)
        
        # Handle the error and create HTTP exception
        if isinstance(e, (ValidationError, ModelError, SystemError, NetworkError, TimeoutError)):
            # Already a structured error
            http_exception = error_handler.create_http_exception(e, context)
        else:
            # Convert generic exception to structured error
            error_response = error_handler.handle_error(e, context)
            http_exception = HTTPException(
                status_code=400 if "validation" in str(e).lower() else 500,
                detail=error_response.dict()
            )
        
        # Record error metrics (will be implemented in task 5)
        # metrics.record_inference_error("predict", str(e))
        
        raise http_exception