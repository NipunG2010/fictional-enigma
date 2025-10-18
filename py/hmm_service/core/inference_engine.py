"""
HMM Inference Engine - Core inference logic for state probability calculation.

This module implements the HMM forward filtering algorithm and model management
for real-time inference in the microservice.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from .config import ServiceConfig
from .error_handling import (
    ModelError, ValidationError, SystemError, TimeoutError,
    ErrorContext, get_error_handler
)
from .fallback import FallbackManager, FallbackConfig, create_fallback_manager

# Import HMM models from the main package
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from imp.hmm.models import HMMArtifact, FusionWeights, HMMPrediction

logger = logging.getLogger(__name__)


# Legacy exception classes for backward compatibility
class ModelValidationError(ModelError):
    """Exception raised when model validation fails."""
    pass


class InferenceError(SystemError):
    """Exception raised during inference computation."""
    pass


class HMMInferenceEngine:
    """
    HMM inference engine for state probability calculation.
    
    Implements forward filtering algorithm for real-time state probability
    computation and manages loaded HMM models with validation.
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize the inference engine.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self._initialized = False
        
        # Error handling
        self._error_handler = get_error_handler()
        
        # Fallback management
        fallback_config = FallbackConfig(
            enable_static_weights=True,
            enable_previous_model=True,
            enable_cached_results=True,
            static_weights={"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3}
        )
        self._fallback_manager = create_fallback_manager(fallback_config)
        
        # Model state
        self._hmm_artifact: Optional[HMMArtifact] = None
        self._fusion_weights: Optional[FusionWeights] = None
        self._model_loaded = False
        self._model_version = None
        self._model_load_time = None
        
        # Cached model parameters for fast inference
        self._transition_matrix: Optional[np.ndarray] = None
        self._initial_probs: Optional[np.ndarray] = None
        self._means: Optional[np.ndarray] = None
        self._covariances: Optional[List[np.ndarray]] = None
        self._inv_covariances: Optional[List[np.ndarray]] = None
        self._log_det_covariances: Optional[np.ndarray] = None
        
        # State tracking for sequential inference
        self._last_state_probs: Optional[np.ndarray] = None
        self._inference_count = 0
        
        # Cache manager (will be injected)
        self._cache_manager = None
        
        logger.info("HMM Inference Engine initialized with error handling and fallback support")
    
    async def initialize(self, cache_manager=None):
        """
        Initialize the inference engine.
        
        Args:
            cache_manager: Optional cache manager instance
        """
        if self._initialized:
            logger.debug("Inference engine already initialized")
            return
        
        logger.info("Initializing HMM Inference Engine...")
        
        # Set cache manager if provided
        if cache_manager:
            self._cache_manager = cache_manager
            self._fallback_manager.set_cache_manager(cache_manager)
            logger.debug("Cache manager integrated with inference engine and fallback manager")
        
        # Pre-compute constants and validate configuration
        self._validate_config()
        
        self._initialized = True
        logger.info("HMM Inference Engine initialization complete")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        
        if self.config.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
    
    async def load_model(self, hmm_artifact: HMMArtifact, fusion_weights: Optional[FusionWeights] = None) -> bool:
        """
        Load HMM model from artifact with validation and preprocessing.
        
        Args:
            hmm_artifact: HMM model artifact
            fusion_weights: Optional fusion weights for signal combination
            
        Returns:
            True if model loaded successfully
            
        Raises:
            ModelValidationError: If model validation fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Loading HMM model: {hmm_artifact.n_states} states, version {hmm_artifact.version}")
            
            # Validate model integrity
            if not self._validate_hmm_artifact(hmm_artifact):
                raise ModelError(
                    "HMM artifact validation failed",
                    error_code="MODEL_VALIDATION_FAILED",
                    model_version=hmm_artifact.version,
                    fallback_available=len(self._fallback_manager._previous_models) > 0
                )
            
            # Validate fusion weights if provided
            if fusion_weights and not self._validate_fusion_weights(fusion_weights, hmm_artifact):
                raise ModelError(
                    "Fusion weights validation failed",
                    error_code="FUSION_WEIGHTS_VALIDATION_FAILED",
                    model_version=hmm_artifact.version,
                    fallback_available=len(self._fallback_manager._previous_models) > 0
                )
            
            # Store previous model info for fallback before loading new one
            if self._model_loaded and self._hmm_artifact:
                previous_model_info = self.get_model_info()
                self._fallback_manager.store_previous_model(previous_model_info)
            
            # Store artifacts
            self._hmm_artifact = hmm_artifact
            self._fusion_weights = fusion_weights
            
            # Preprocess model parameters for fast inference
            await self._preprocess_model_parameters()
            
            # Update model state
            self._model_loaded = True
            self._model_version = hmm_artifact.version
            self._model_load_time = time.time()
            self._inference_count = 0
            self._last_state_probs = None
            
            # Deactivate fallback mode if it was active
            if self._fallback_manager.is_fallback_active():
                self._fallback_manager.deactivate_fallback()
            
            load_duration = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_duration:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model_loaded = False
            
            # Convert to appropriate error type
            if isinstance(e, ModelError):
                raise e
            else:
                raise ModelError(
                    f"Model loading failed: {str(e)}",
                    error_code="MODEL_LOAD_FAILED",
                    model_version=hmm_artifact.version if hmm_artifact else None,
                    fallback_available=len(self._fallback_manager._previous_models) > 0,
                    details={"original_error": str(e), "error_type": type(e).__name__}
                )
    
    def _validate_hmm_artifact(self, artifact: HMMArtifact) -> bool:
        """
        Validate HMM artifact for inference compatibility.
        
        Args:
            artifact: HMM artifact to validate
            
        Returns:
            True if valid
        """
        try:
            # Check basic structure
            if artifact.n_states <= 0:
                logger.error("Invalid number of states")
                return False
            
            # Validate transition matrix
            trans_matrix = np.array(artifact.transition_matrix)
            if trans_matrix.shape != (artifact.n_states, artifact.n_states):
                logger.error(f"Invalid transition matrix shape: {trans_matrix.shape}")
                return False
            
            # Check row sums (should be 1.0)
            row_sums = trans_matrix.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                logger.error(f"Transition matrix rows don't sum to 1: {row_sums}")
                return False
            
            # Validate initial probabilities
            initial_probs = np.array(artifact.initial_probabilities)
            if len(initial_probs) != artifact.n_states:
                logger.error(f"Invalid initial probabilities length: {len(initial_probs)}")
                return False
            
            if not np.allclose(initial_probs.sum(), 1.0, atol=1e-6):
                logger.error(f"Initial probabilities don't sum to 1: {initial_probs.sum()}")
                return False
            
            # Validate means
            means = np.array(artifact.means)
            if means.shape[0] != artifact.n_states:
                logger.error(f"Invalid means shape: {means.shape}")
                return False
            
            n_features = means.shape[1]
            
            # Validate covariances
            if len(artifact.covariances) != artifact.n_states:
                logger.error(f"Invalid number of covariance matrices: {len(artifact.covariances)}")
                return False
            
            for i, cov in enumerate(artifact.covariances):
                cov_matrix = np.array(cov)
                if cov_matrix.shape != (n_features, n_features):
                    logger.error(f"Invalid covariance matrix {i} shape: {cov_matrix.shape}")
                    return False
                
                # Check positive definiteness
                try:
                    np.linalg.cholesky(cov_matrix)
                except np.linalg.LinAlgError:
                    logger.error(f"Covariance matrix {i} is not positive definite")
                    return False
            
            logger.debug("HMM artifact validation passed")
            return True
            
        except Exception as e:
            logger.error(f"HMM artifact validation error: {e}")
            return False
    
    def _validate_fusion_weights(self, fusion_weights: FusionWeights, hmm_artifact: HMMArtifact) -> bool:
        """
        Validate fusion weights compatibility with HMM model.
        
        Args:
            fusion_weights: Fusion weights to validate
            hmm_artifact: Associated HMM artifact
            
        Returns:
            True if valid
        """
        try:
            # Check number of states matches
            if len(fusion_weights.state_weights) != hmm_artifact.n_states:
                logger.error(
                    f"Fusion weights states ({len(fusion_weights.state_weights)}) "
                    f"don't match HMM states ({hmm_artifact.n_states})"
                )
                return False
            
            # Check weight structure
            expected_signals = {"w_ldc", "w_mr", "w_tsmom"}
            for i, state_weight in enumerate(fusion_weights.state_weights):
                if set(state_weight.keys()) != expected_signals:
                    logger.error(f"State {i} weights missing expected signals: {expected_signals}")
                    return False
                
                # Check weights sum to 1
                weight_sum = sum(state_weight.values())
                if not np.isclose(weight_sum, 1.0, atol=1e-6):
                    logger.error(f"State {i} weights don't sum to 1: {weight_sum}")
                    return False
                
                # Check non-negative weights
                for signal, weight in state_weight.items():
                    if weight < 0:
                        logger.error(f"State {i} weight {signal} is negative: {weight}")
                        return False
            
            logger.debug("Fusion weights validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Fusion weights validation error: {e}")
            return False
    
    async def _preprocess_model_parameters(self):
        """
        Preprocess model parameters for efficient inference.
        
        Converts lists to numpy arrays and precomputes inverse covariances
        and log determinants for faster likelihood computation.
        """
        if not self._hmm_artifact:
            raise ValueError("No HMM artifact loaded")
        
        logger.debug("Preprocessing model parameters...")
        
        # Convert to numpy arrays
        self._transition_matrix = np.array(self._hmm_artifact.transition_matrix)
        self._initial_probs = np.array(self._hmm_artifact.initial_probabilities)
        self._means = np.array(self._hmm_artifact.means)
        
        # Preprocess covariances for fast likelihood computation
        self._covariances = []
        self._inv_covariances = []
        self._log_det_covariances = []
        
        for cov_list in self._hmm_artifact.covariances:
            cov_matrix = np.array(cov_list)
            self._covariances.append(cov_matrix)
            
            # Precompute inverse and log determinant
            inv_cov = np.linalg.inv(cov_matrix)
            self._inv_covariances.append(inv_cov)
            
            # Use Cholesky decomposition for stable log determinant
            chol = np.linalg.cholesky(cov_matrix)
            log_det = 2 * np.sum(np.log(np.diag(chol)))
            self._log_det_covariances.append(log_det)
        
        self._log_det_covariances = np.array(self._log_det_covariances)
        
        logger.debug("Model parameters preprocessed successfully")
    
    async def predict_state_probabilities(self, observations: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """
        Calculate state probabilities using forward filtering algorithm with comprehensive error handling.
        
        Args:
            observations: Observation vector [s_ldc, s_mr, s_tsmom]
            use_cache: Whether to use cache for computation
            
        Returns:
            State probability distribution
            
        Raises:
            ValidationError: If input validation fails
            ModelError: If model is not loaded or invalid
            SystemError: If inference computation fails
        """
        start_time = time.time()
        
        try:
            # Validate initialization state
            if not self._initialized:
                raise SystemError(
                    "Inference engine not initialized",
                    error_code="ENGINE_NOT_INITIALIZED",
                    resource="inference_engine"
                )
            
            # Validate model state
            if not self._model_loaded:
                raise ModelError(
                    "No model loaded for inference",
                    error_code="MODEL_NOT_LOADED",
                    fallback_available=len(self._fallback_manager._previous_models) > 0
                )
            
            # Validate input observations
            self._validate_observations(observations)
            
            # Try cache first if enabled and this is not sequential inference
            if use_cache and self._cache_manager and self._last_state_probs is None:
                try:
                    cached_result = await self._cache_manager.get_inference_result(observations)
                    if cached_result is not None:
                        state_probs, _ = cached_result  # We only need state probs here
                        logger.debug("Using cached state probabilities")
                        return state_probs
                except Exception as cache_error:
                    logger.warning(f"Cache lookup failed: {cache_error}")
                    # Continue with computation
            
            # Perform inference computation
            state_probs = await self._compute_state_probabilities_safe(observations)
            
            # Update state for next inference
            self._last_state_probs = state_probs
            self._inference_count += 1
            
            inference_time = (time.time() - start_time) * 1000
            logger.debug(f"State probabilities computed in {inference_time:.2f}ms: {state_probs}")
            
            return state_probs
            
        except (ValidationError, ModelError, SystemError):
            # Re-raise known errors
            raise
            
        except Exception as e:
            # Handle unexpected errors with fallback
            logger.error(f"Unexpected error in state probability computation: {e}")
            
            try:
                # Try fallback mechanism
                state_probs, _ = await self._fallback_manager.handle_model_failure(
                    e, observations, request_id=None
                )
                
                processing_time = (time.time() - start_time) * 1000
                logger.warning(f"Used fallback for state probabilities in {processing_time:.2f}ms")
                
                return state_probs
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise SystemError(
                    f"Inference failed and fallback unavailable: {str(e)}",
                    error_code="INFERENCE_TOTAL_FAILURE",
                    resource="inference_engine",
                    details={
                        "original_error": str(e),
                        "fallback_error": str(fallback_error),
                        "processing_time_ms": (time.time() - start_time) * 1000
                    }
                )
    
    def _validate_observations(self, observations: np.ndarray):
        """Validate input observations with detailed error messages."""
        if not isinstance(observations, np.ndarray):
            raise ValidationError(
                "Observations must be a numpy array",
                field="observations",
                value=type(observations).__name__,
                expected="numpy.ndarray"
            )
        
        if observations.shape != (3,):
            raise ValidationError(
                f"Expected 3D observation vector, got shape {observations.shape}",
                field="observations.shape",
                value=str(observations.shape),
                expected="(3,)"
            )
        
        # Check for invalid values
        if np.any(np.isnan(observations)):
            nan_indices = np.where(np.isnan(observations))[0].tolist()
            raise ValidationError(
                f"Observation vector contains NaN values at indices: {nan_indices}",
                field="observations",
                value=observations.tolist(),
                expected="finite numeric values"
            )
        
        if np.any(np.isinf(observations)):
            inf_indices = np.where(np.isinf(observations))[0].tolist()
            raise ValidationError(
                f"Observation vector contains infinite values at indices: {inf_indices}",
                field="observations",
                value=observations.tolist(),
                expected="finite numeric values"
            )
        
        # Check reasonable bounds (configurable)
        max_abs_value = 1000.0  # Could be made configurable
        if np.any(np.abs(observations) > max_abs_value):
            large_indices = np.where(np.abs(observations) > max_abs_value)[0].tolist()
            raise ValidationError(
                f"Observation values exceed reasonable bounds at indices: {large_indices}",
                field="observations",
                value=observations.tolist(),
                expected=f"values with absolute value <= {max_abs_value}"
            )
    
    async def _compute_state_probabilities_safe(self, observations: np.ndarray) -> np.ndarray:
        """Safely compute state probabilities with error handling."""
        try:
            # Compute emission probabilities for all states
            log_likelihoods = self._compute_log_likelihoods(observations)
            
            # Check for numerical issues in log likelihoods
            if np.any(np.isnan(log_likelihoods)) or np.any(np.isinf(log_likelihoods)):
                raise SystemError(
                    "Numerical instability in likelihood computation",
                    error_code="LIKELIHOOD_NUMERICAL_ERROR",
                    resource="hmm_computation",
                    details={
                        "log_likelihoods": log_likelihoods.tolist(),
                        "observations": observations.tolist()
                    }
                )
            
            # Forward filtering step
            if self._last_state_probs is None:
                # First observation - use initial probabilities
                log_probs = np.log(self._initial_probs) + log_likelihoods
            else:
                # Subsequent observations - use transition probabilities
                log_trans_probs = np.log(self._transition_matrix)
                log_prev_probs = np.log(self._last_state_probs)
                
                # Forward step: log P(s_t | o_1:t) ∝ log P(o_t | s_t) + log Σ P(s_t | s_{t-1}) P(s_{t-1} | o_1:{t-1})
                log_probs = log_likelihoods + logsumexp(
                    log_trans_probs + log_prev_probs[:, np.newaxis], axis=0
                )
            
            # Normalize to get probabilities
            log_probs_normalized = log_probs - logsumexp(log_probs)
            state_probs = np.exp(log_probs_normalized)
            
            # Validate output
            if np.any(np.isnan(state_probs)) or np.any(np.isinf(state_probs)):
                raise SystemError(
                    "Numerical instability in probability normalization",
                    error_code="PROBABILITY_NUMERICAL_ERROR",
                    resource="hmm_computation",
                    details={
                        "state_probabilities": state_probs.tolist(),
                        "log_probs": log_probs.tolist()
                    }
                )
            
            # Check probability constraints
            prob_sum = state_probs.sum()
            if not np.isclose(prob_sum, 1.0, atol=1e-6):
                logger.warning(f"State probabilities don't sum to 1: {prob_sum}, normalizing")
                state_probs = state_probs / prob_sum
            
            return state_probs
            
        except SystemError:
            # Re-raise system errors
            raise
            
        except Exception as e:
            # Convert unexpected errors to system errors
            raise SystemError(
                f"HMM computation failed: {str(e)}",
                error_code="HMM_COMPUTATION_ERROR",
                resource="hmm_computation",
                details={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                    "observations": observations.tolist()
                }
            )
    
    def _compute_log_likelihoods(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihoods for all states efficiently.
        
        Args:
            observations: Observation vector
            
        Returns:
            Log-likelihood for each state
        """
        n_states = self._hmm_artifact.n_states
        log_likelihoods = np.zeros(n_states)
        
        for i in range(n_states):
            # Compute log-likelihood using precomputed inverse covariance
            diff = observations - self._means[i]
            
            # Efficient computation: -0.5 * (x - μ)^T Σ^{-1} (x - μ) - 0.5 * log|2πΣ|
            mahalanobis_sq = np.dot(diff, np.dot(self._inv_covariances[i], diff))
            
            log_likelihood = (
                -0.5 * mahalanobis_sq
                - 0.5 * self._log_det_covariances[i]
                - 0.5 * len(observations) * np.log(2 * np.pi)
            )
            
            log_likelihoods[i] = log_likelihood
        
        return log_likelihoods
    
    async def compute_fusion_weights(self, state_probs: np.ndarray, use_cache: bool = True) -> Dict[str, float]:
        """
        Compute fusion weights from state probabilities with comprehensive error handling.
        
        Args:
            state_probs: State probability distribution
            use_cache: Whether to use cache for computation
            
        Returns:
            Dictionary with fusion weights for each signal
            
        Raises:
            ValidationError: If input validation fails
            ModelError: If model is not loaded or fusion weights unavailable
            SystemError: If computation fails
        """
        try:
            # Validate model state
            if not self._model_loaded:
                raise ModelError(
                    "No model loaded for fusion weight computation",
                    error_code="MODEL_NOT_LOADED",
                    fallback_available=len(self._fallback_manager._previous_models) > 0
                )
            
            # Handle missing fusion weights with fallback
            if self._fusion_weights is None:
                logger.warning("No fusion weights loaded, using equal weights fallback")
                return {"w_ldc": 1/3, "w_mr": 1/3, "w_tsmom": 1/3}
            
            # Validate input state probabilities
            self._validate_state_probabilities(state_probs)
            
            # Try cache first if enabled
            if use_cache and self._cache_manager:
                try:
                    cached_weights = await self._cache_manager.get_fusion_weights(state_probs)
                    if cached_weights is not None:
                        logger.debug("Using cached fusion weights")
                        return cached_weights
                except Exception as cache_error:
                    logger.warning(f"Cache lookup for fusion weights failed: {cache_error}")
                    # Continue with computation
            
            # Compute fusion weights safely
            fusion_weights = await self._compute_fusion_weights_safe(state_probs)
            
            # Cache the result if caching is enabled
            if use_cache and self._cache_manager:
                try:
                    await self._cache_manager.cache_fusion_weights(state_probs, fusion_weights)
                except Exception as cache_error:
                    logger.warning(f"Failed to cache fusion weights: {cache_error}")
                    # Don't fail the request for cache errors
            
            logger.debug(f"Fusion weights computed: {fusion_weights}")
            return fusion_weights
            
        except (ValidationError, ModelError, SystemError):
            # Re-raise known errors
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in fusion weight computation: {e}")
            raise SystemError(
                f"Fusion weight computation failed: {str(e)}",
                error_code="FUSION_WEIGHTS_COMPUTATION_ERROR",
                resource="fusion_weights",
                details={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                    "state_probabilities": state_probs.tolist() if isinstance(state_probs, np.ndarray) else str(state_probs)
                }
            )
    
    def _validate_state_probabilities(self, state_probs: np.ndarray):
        """Validate state probabilities with detailed error messages."""
        if not isinstance(state_probs, np.ndarray):
            raise ValidationError(
                "State probabilities must be a numpy array",
                field="state_probabilities",
                value=type(state_probs).__name__,
                expected="numpy.ndarray"
            )
        
        if len(state_probs) != self._hmm_artifact.n_states:
            raise ValidationError(
                f"State probabilities length doesn't match model states",
                field="state_probabilities.length",
                value=str(len(state_probs)),
                expected=str(self._hmm_artifact.n_states)
            )
        
        # Check for invalid values
        if np.any(np.isnan(state_probs)):
            nan_indices = np.where(np.isnan(state_probs))[0].tolist()
            raise ValidationError(
                f"State probabilities contain NaN values at indices: {nan_indices}",
                field="state_probabilities",
                value=state_probs.tolist(),
                expected="finite numeric values"
            )
        
        if np.any(np.isinf(state_probs)):
            inf_indices = np.where(np.isinf(state_probs))[0].tolist()
            raise ValidationError(
                f"State probabilities contain infinite values at indices: {inf_indices}",
                field="state_probabilities",
                value=state_probs.tolist(),
                expected="finite numeric values"
            )
        
        # Check probability constraints
        if np.any(state_probs < 0):
            negative_indices = np.where(state_probs < 0)[0].tolist()
            raise ValidationError(
                f"State probabilities contain negative values at indices: {negative_indices}",
                field="state_probabilities",
                value=state_probs.tolist(),
                expected="non-negative values"
            )
        
        if np.any(state_probs > 1):
            large_indices = np.where(state_probs > 1)[0].tolist()
            raise ValidationError(
                f"State probabilities contain values > 1 at indices: {large_indices}",
                field="state_probabilities",
                value=state_probs.tolist(),
                expected="values <= 1"
            )
        
        prob_sum = state_probs.sum()
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            raise ValidationError(
                f"State probabilities don't sum to 1: {prob_sum}",
                field="state_probabilities.sum",
                value=str(prob_sum),
                expected="1.0 (±1e-6)"
            )
    
    async def _compute_fusion_weights_safe(self, state_probs: np.ndarray) -> Dict[str, float]:
        """Safely compute fusion weights with error handling."""
        try:
            # Initialize fusion weights
            fusion_weights = {"w_ldc": 0.0, "w_mr": 0.0, "w_tsmom": 0.0}
            
            # Compute weighted average of per-state weights
            for i, prob in enumerate(state_probs):
                if i >= len(self._fusion_weights.state_weights):
                    raise SystemError(
                        f"State index {i} exceeds available fusion weights",
                        error_code="FUSION_WEIGHTS_INDEX_ERROR",
                        resource="fusion_weights",
                        details={
                            "state_index": i,
                            "available_states": len(self._fusion_weights.state_weights)
                        }
                    )
                
                state_weights = self._fusion_weights.state_weights[i]
                
                # Validate state weights structure
                expected_signals = {"w_ldc", "w_mr", "w_tsmom"}
                if set(state_weights.keys()) != expected_signals:
                    raise SystemError(
                        f"Invalid fusion weight structure for state {i}",
                        error_code="FUSION_WEIGHTS_STRUCTURE_ERROR",
                        resource="fusion_weights",
                        details={
                            "state_index": i,
                            "available_signals": list(state_weights.keys()),
                            "expected_signals": list(expected_signals)
                        }
                    )
                
                # Accumulate weighted contributions
                for signal in fusion_weights:
                    weight_contribution = prob * state_weights[signal]
                    
                    # Check for numerical issues
                    if np.isnan(weight_contribution) or np.isinf(weight_contribution):
                        raise SystemError(
                            f"Numerical error in weight computation for state {i}, signal {signal}",
                            error_code="FUSION_WEIGHTS_NUMERICAL_ERROR",
                            resource="fusion_weights",
                            details={
                                "state_index": i,
                                "signal": signal,
                                "probability": float(prob),
                                "state_weight": state_weights[signal],
                                "contribution": float(weight_contribution)
                            }
                        )
                    
                    fusion_weights[signal] += weight_contribution
            
            # Validate and normalize final weights
            total_weight = sum(fusion_weights.values())
            
            if np.isnan(total_weight) or np.isinf(total_weight):
                raise SystemError(
                    "Numerical error in total weight computation",
                    error_code="FUSION_WEIGHTS_TOTAL_ERROR",
                    resource="fusion_weights",
                    details={
                        "fusion_weights": fusion_weights,
                        "total_weight": float(total_weight)
                    }
                )
            
            if total_weight <= 0:
                raise SystemError(
                    f"Invalid total weight: {total_weight}",
                    error_code="FUSION_WEIGHTS_INVALID_TOTAL",
                    resource="fusion_weights",
                    details={
                        "fusion_weights": fusion_weights,
                        "total_weight": float(total_weight)
                    }
                )
            
            # Normalize weights to ensure they sum to 1
            if not np.isclose(total_weight, 1.0, atol=1e-6):
                logger.warning(f"Fusion weights don't sum to 1: {total_weight}, normalizing")
                for signal in fusion_weights:
                    fusion_weights[signal] /= total_weight
            
            # Final validation
            final_total = sum(fusion_weights.values())
            if not np.isclose(final_total, 1.0, atol=1e-6):
                raise SystemError(
                    f"Normalization failed, final total: {final_total}",
                    error_code="FUSION_WEIGHTS_NORMALIZATION_ERROR",
                    resource="fusion_weights",
                    details={
                        "fusion_weights": fusion_weights,
                        "final_total": float(final_total)
                    }
                )
            
            return fusion_weights
            
        except SystemError:
            # Re-raise system errors
            raise
            
        except Exception as e:
            # Convert unexpected errors to system errors
            raise SystemError(
                f"Unexpected error in fusion weight computation: {str(e)}",
                error_code="FUSION_WEIGHTS_UNEXPECTED_ERROR",
                resource="fusion_weights",
                details={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                    "state_probabilities": state_probs.tolist()
                }
            )
    
    def validate_model(self) -> bool:
        """
        Validate currently loaded model.
        
        Returns:
            True if model is valid and ready for inference
        """
        if not self._model_loaded or not self._hmm_artifact:
            return False
        
        try:
            # Re-validate artifacts
            hmm_valid = self._validate_hmm_artifact(self._hmm_artifact)
            
            fusion_valid = True
            if self._fusion_weights:
                fusion_valid = self._validate_fusion_weights(self._fusion_weights, self._hmm_artifact)
            
            # Check preprocessed parameters
            params_valid = all([
                self._transition_matrix is not None,
                self._initial_probs is not None,
                self._means is not None,
                self._covariances is not None,
                self._inv_covariances is not None,
                self._log_det_covariances is not None
            ])
            
            return hmm_valid and fusion_valid and params_valid
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self._model_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "version": self._model_version,
            "n_states": self._hmm_artifact.n_states,
            "n_features": self._means.shape[1] if self._means is not None else 0,
            "has_fusion_weights": self._fusion_weights is not None,
            "load_time": self._model_load_time,
            "inference_count": self._inference_count,
            "library": self._hmm_artifact.metadata.get("library", "unknown"),
            "training_window": {
                "start": self._hmm_artifact.training_window_start,
                "end": self._hmm_artifact.training_window_end
            }
        }
    
    def reset_state(self):
        """Reset inference state (clear previous state probabilities)."""
        self._last_state_probs = None
        self._inference_count = 0
        logger.debug("Inference state reset")
    
    async def predict_complete(self, observations: np.ndarray, request_id: Optional[str] = None, use_cache: bool = True) -> HMMPrediction:
        """
        Perform complete prediction with comprehensive error handling and fallback support.
        
        Args:
            observations: Observation vector [s_ldc, s_mr, s_tsmom]
            request_id: Optional request ID for tracking
            use_cache: Whether to use cache for computation
            
        Returns:
            Complete HMM prediction result
            
        Raises:
            ValidationError: If input validation fails
            ModelError: If model is not available
            SystemError: If prediction computation fails
        """
        start_time = time.time()
        
        try:
            # Try to get complete result from cache first (only for non-sequential inference)
            if use_cache and self._cache_manager and self._last_state_probs is None:
                try:
                    cached_result = await self._cache_manager.get_inference_result(observations)
                    if cached_result is not None:
                        state_probs, fusion_weights = cached_result
                        
                        # Validate cached result
                        if (isinstance(state_probs, np.ndarray) and 
                            isinstance(fusion_weights, dict) and
                            len(state_probs) > 0 and
                            all(k in fusion_weights for k in ["w_ldc", "w_mr", "w_tsmom"])):
                            
                            # Find most likely state and confidence
                            most_likely_state = int(np.argmax(state_probs))
                            confidence = float(state_probs[most_likely_state])
                            
                            prediction = HMMPrediction(
                                state_probabilities=state_probs.tolist(),
                                most_likely_state=most_likely_state,
                                confidence=confidence,
                                fused_weights=fusion_weights,
                                timestamp=int(time.time())
                            )
                            
                            processing_time = (time.time() - start_time) * 1000
                            logger.debug(f"Complete prediction from cache in {processing_time:.2f}ms")
                            return prediction
                        else:
                            logger.warning("Invalid cached result format, proceeding with computation")
                            
                except Exception as cache_error:
                    logger.warning(f"Cache lookup failed: {cache_error}")
                    # Continue with computation
            
            # Compute state probabilities with error handling
            try:
                state_probs = await self.predict_state_probabilities(observations, use_cache=use_cache)
            except (ValidationError, ModelError, SystemError) as e:
                # Try fallback for state probabilities
                if isinstance(e, ModelError) or isinstance(e, SystemError):
                    try:
                        logger.warning(f"State probability computation failed, trying fallback: {e}")
                        state_probs, fallback_weights = await self._fallback_manager.handle_model_failure(
                            e, observations, request_id
                        )
                        
                        # Use fallback weights and return early
                        most_likely_state = int(np.argmax(state_probs))
                        confidence = float(state_probs[most_likely_state])
                        
                        prediction = HMMPrediction(
                            state_probabilities=state_probs.tolist(),
                            most_likely_state=most_likely_state,
                            confidence=confidence,
                            fused_weights=fallback_weights,
                            timestamp=int(time.time())
                        )
                        
                        processing_time = (time.time() - start_time) * 1000
                        logger.info(f"Complete prediction using fallback in {processing_time:.2f}ms")
                        return prediction
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        raise SystemError(
                            f"Complete prediction failed and fallback unavailable: {str(e)}",
                            error_code="COMPLETE_PREDICTION_TOTAL_FAILURE",
                            resource="inference_engine",
                            details={
                                "original_error": str(e),
                                "fallback_error": str(fallback_error),
                                "request_id": request_id
                            }
                        )
                else:
                    # Re-raise validation errors
                    raise
            
            # Compute fusion weights with error handling
            try:
                fusion_weights = await self.compute_fusion_weights(state_probs, use_cache=use_cache)
            except (ValidationError, ModelError, SystemError) as e:
                # For fusion weight errors, use fallback weights
                logger.warning(f"Fusion weight computation failed, using fallback: {e}")
                fusion_weights = {"w_ldc": 1/3, "w_mr": 1/3, "w_tsmom": 1/3}
            
            # Cache the complete result if caching is enabled
            if use_cache and self._cache_manager:
                try:
                    await self._cache_manager.cache_inference_result(observations, state_probs, fusion_weights)
                except Exception as cache_error:
                    logger.warning(f"Failed to cache complete result: {cache_error}")
                    # Don't fail the request for cache errors
            
            # Create prediction result with validation
            try:
                most_likely_state = int(np.argmax(state_probs))
                confidence = float(state_probs[most_likely_state])
                
                # Validate prediction components
                if not (0 <= most_likely_state < len(state_probs)):
                    raise SystemError(
                        f"Invalid most likely state: {most_likely_state}",
                        error_code="PREDICTION_VALIDATION_ERROR",
                        resource="prediction_result"
                    )
                
                if not (0 <= confidence <= 1):
                    raise SystemError(
                        f"Invalid confidence value: {confidence}",
                        error_code="PREDICTION_VALIDATION_ERROR",
                        resource="prediction_result"
                    )
                
                prediction = HMMPrediction(
                    state_probabilities=state_probs.tolist(),
                    most_likely_state=most_likely_state,
                    confidence=confidence,
                    fused_weights=fusion_weights,
                    timestamp=int(time.time())
                )
                
                processing_time = (time.time() - start_time) * 1000
                logger.debug(f"Complete prediction computed in {processing_time:.2f}ms")
                
                return prediction
                
            except Exception as result_error:
                raise SystemError(
                    f"Failed to create prediction result: {str(result_error)}",
                    error_code="PREDICTION_RESULT_ERROR",
                    resource="prediction_result",
                    details={
                        "result_error": str(result_error),
                        "state_probabilities": state_probs.tolist() if isinstance(state_probs, np.ndarray) else str(state_probs),
                        "fusion_weights": fusion_weights
                    }
                )
            
        except (ValidationError, ModelError, SystemError):
            # Re-raise known errors
            raise
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in complete prediction: {e}")
            
            # Try emergency fallback
            try:
                state_probs, fusion_weights = await self._fallback_manager.handle_model_failure(
                    e, observations, request_id
                )
                
                most_likely_state = int(np.argmax(state_probs))
                confidence = float(state_probs[most_likely_state])
                
                prediction = HMMPrediction(
                    state_probabilities=state_probs.tolist(),
                    most_likely_state=most_likely_state,
                    confidence=confidence,
                    fused_weights=fusion_weights,
                    timestamp=int(time.time())
                )
                
                processing_time = (time.time() - start_time) * 1000
                logger.warning(f"Complete prediction using emergency fallback in {processing_time:.2f}ms")
                return prediction
                
            except Exception as emergency_error:
                logger.error(f"Emergency fallback failed: {emergency_error}")
                raise SystemError(
                    f"Complete prediction failed catastrophically: {str(e)}",
                    error_code="COMPLETE_PREDICTION_CATASTROPHIC_FAILURE",
                    resource="inference_engine",
                    details={
                        "original_error": str(e),
                        "emergency_error": str(emergency_error),
                        "request_id": request_id,
                        "processing_time_ms": (time.time() - start_time) * 1000
                    }
                )
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of the inference engine.
        
        Returns:
            Dictionary with health information including fallback status
        """
        return {
            "initialized": self._initialized,
            "model_loaded": self._model_loaded,
            "model_valid": self.validate_model(),
            "inference_count": self._inference_count,
            "last_inference_time": self._model_load_time,
            "model_version": self._model_version,
            "fallback_status": self._fallback_manager.get_fallback_status(),
            "error_handling": {
                "error_statistics": self._error_handler.get_error_statistics(),
                "fallback_active": self._fallback_manager.is_fallback_active()
            }
        }