"""
Fallback mechanisms for HMM Microservice.

Provides fallback strategies for model failures, service degradation,
and static weight responses when HMM inference is unavailable.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .logging_config import get_logger
from .error_handling import ModelError, SystemError


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    
    STATIC_WEIGHTS = "static_weights"
    PREVIOUS_MODEL = "previous_model"
    EQUAL_WEIGHTS = "equal_weights"
    CACHED_RESULTS = "cached_results"
    DEGRADED_SERVICE = "degraded_service"


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""
    
    enable_static_weights: bool = True
    enable_previous_model: bool = True
    enable_cached_results: bool = True
    
    # Static weight configuration
    static_weights: Dict[str, float] = None
    
    # Cache configuration for fallback
    max_cache_age_seconds: int = 300  # 5 minutes
    min_cache_confidence: float = 0.6
    
    # Model fallback configuration
    max_model_age_days: int = 7
    
    def __post_init__(self):
        """Set default static weights if not provided."""
        if self.static_weights is None:
            self.static_weights = {
                "w_ldc": 0.4,
                "w_mr": 0.3,
                "w_tsmom": 0.3
            }


class FallbackManager:
    """Manages fallback strategies for service resilience."""
    
    def __init__(self, config: FallbackConfig):
        self.config = config
        self.logger = get_logger("hmm_service.fallback")
        
        # Fallback state tracking
        self._fallback_active = False
        self._current_strategy = None
        self._fallback_start_time = None
        self._fallback_count = 0
        
        # Previous model storage
        self._previous_models = []
        self._max_previous_models = 3
        
        # Static responses cache
        self._static_responses_cache = {}
        
        self.logger.info("Fallback manager initialized", extra={
            "static_weights_enabled": config.enable_static_weights,
            "previous_model_enabled": config.enable_previous_model,
            "cached_results_enabled": config.enable_cached_results,
            "static_weights": config.static_weights
        })
    
    async def handle_model_failure(
        self,
        error: Exception,
        observations: np.ndarray,
        request_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Handle model failure with appropriate fallback strategy.
        
        Args:
            error: The error that caused model failure
            observations: Input observations for inference
            request_id: Optional request ID for tracking
            
        Returns:
            Tuple of (state_probabilities, fusion_weights)
        """
        self.logger.warning("Model failure detected, activating fallback", extra={
            "error": str(error),
            "error_type": type(error).__name__,
            "observations": observations.tolist(),
            "request_id": request_id
        })
        
        # Activate fallback mode
        if not self._fallback_active:
            self._activate_fallback()
        
        # Try fallback strategies in order of preference
        strategies = self._get_fallback_strategies()
        
        for strategy in strategies:
            try:
                result = await self._execute_fallback_strategy(
                    strategy, observations, request_id
                )
                if result is not None:
                    state_probs, fusion_weights = result
                    
                    self.logger.info("Fallback strategy successful", extra={
                        "strategy": strategy.value,
                        "state_probabilities": state_probs.tolist(),
                        "fusion_weights": fusion_weights,
                        "request_id": request_id
                    })
                    
                    return state_probs, fusion_weights
                    
            except Exception as fallback_error:
                self.logger.warning("Fallback strategy failed", extra={
                    "strategy": strategy.value,
                    "error": str(fallback_error),
                    "request_id": request_id
                })
                continue
        
        # If all fallback strategies fail, use emergency static response
        self.logger.error("All fallback strategies failed, using emergency response", extra={
            "request_id": request_id
        })
        
        return self._get_emergency_response()
    
    def _activate_fallback(self):
        """Activate fallback mode."""
        self._fallback_active = True
        self._fallback_start_time = time.time()
        self._fallback_count += 1
        
        self.logger.warning("Fallback mode activated", extra={
            "fallback_count": self._fallback_count,
            "start_time": self._fallback_start_time
        })
    
    def deactivate_fallback(self):
        """Deactivate fallback mode when service recovers."""
        if self._fallback_active:
            fallback_duration = time.time() - self._fallback_start_time
            
            self.logger.info("Fallback mode deactivated", extra={
                "fallback_duration_seconds": round(fallback_duration, 2),
                "strategy_used": self._current_strategy.value if self._current_strategy else None
            })
            
            self._fallback_active = False
            self._current_strategy = None
            self._fallback_start_time = None
    
    def _get_fallback_strategies(self) -> List[FallbackStrategy]:
        """Get ordered list of fallback strategies to try."""
        strategies = []
        
        # Prefer cached results if available and recent
        if self.config.enable_cached_results:
            strategies.append(FallbackStrategy.CACHED_RESULTS)
        
        # Try previous model if available
        if self.config.enable_previous_model and self._previous_models:
            strategies.append(FallbackStrategy.PREVIOUS_MODEL)
        
        # Use static weights as reliable fallback
        if self.config.enable_static_weights:
            strategies.append(FallbackStrategy.STATIC_WEIGHTS)
        
        # Equal weights as last resort
        strategies.append(FallbackStrategy.EQUAL_WEIGHTS)
        
        return strategies
    
    async def _execute_fallback_strategy(
        self,
        strategy: FallbackStrategy,
        observations: np.ndarray,
        request_id: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        """Execute specific fallback strategy."""
        self._current_strategy = strategy
        
        if strategy == FallbackStrategy.CACHED_RESULTS:
            return await self._try_cached_results(observations)
        
        elif strategy == FallbackStrategy.PREVIOUS_MODEL:
            return await self._try_previous_model(observations)
        
        elif strategy == FallbackStrategy.STATIC_WEIGHTS:
            return self._get_static_weights_response(observations)
        
        elif strategy == FallbackStrategy.EQUAL_WEIGHTS:
            return self._get_equal_weights_response(observations)
        
        else:
            self.logger.error(f"Unknown fallback strategy: {strategy}")
            return None
    
    async def _try_cached_results(self, observations: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        """Try to use cached results for similar observations."""
        if not hasattr(self, '_cache_manager') or self._cache_manager is None:
            return None
        
        try:
            # Look for cached results within tolerance
            cached_result = await self._cache_manager.get_inference_result(observations)
            
            if cached_result is not None:
                state_probs, fusion_weights = cached_result
                
                # Check cache age and confidence
                cache_age = time.time() - getattr(cached_result, 'timestamp', 0)
                confidence = np.max(state_probs) if len(state_probs) > 0 else 0
                
                if (cache_age <= self.config.max_cache_age_seconds and 
                    confidence >= self.config.min_cache_confidence):
                    
                    self.logger.debug("Using cached results for fallback", extra={
                        "cache_age_seconds": round(cache_age, 2),
                        "confidence": confidence
                    })
                    
                    return state_probs, fusion_weights
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to retrieve cached results: {e}")
            return None
    
    async def _try_previous_model(self, observations: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        """Try to use previous model version for inference."""
        if not self._previous_models:
            return None
        
        # Try most recent previous model first
        for model_info in reversed(self._previous_models):
            try:
                model_age_days = (time.time() - model_info['load_time']) / (24 * 3600)
                
                if model_age_days > self.config.max_model_age_days:
                    self.logger.debug(f"Previous model too old: {model_age_days:.1f} days")
                    continue
                
                # Try to load and use previous model
                # This would require implementing model switching logic
                # For now, we'll simulate with static response based on model characteristics
                
                self.logger.info("Using previous model for fallback", extra={
                    "model_version": model_info.get('version', 'unknown'),
                    "model_age_days": round(model_age_days, 2)
                })
                
                # Generate response based on previous model characteristics
                return self._generate_previous_model_response(observations, model_info)
                
            except Exception as e:
                self.logger.warning(f"Failed to use previous model: {e}")
                continue
        
        return None
    
    def _generate_previous_model_response(
        self, 
        observations: np.ndarray, 
        model_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate response based on previous model characteristics."""
        n_states = model_info.get('n_states', 3)
        
        # Generate plausible state probabilities based on observations
        # This is a simplified heuristic - in practice, you'd use the actual previous model
        obs_sum = np.sum(np.abs(observations))
        
        if obs_sum > 1.0:  # High signal strength
            state_probs = np.array([0.1, 0.2, 0.7])  # Favor trending state
        elif obs_sum > 0.5:  # Medium signal strength
            state_probs = np.array([0.3, 0.5, 0.2])  # Favor mean-reverting state
        else:  # Low signal strength
            state_probs = np.array([0.6, 0.3, 0.1])  # Favor consolidation state
        
        # Normalize to ensure proper probabilities
        state_probs = state_probs / state_probs.sum()
        
        # Generate fusion weights based on most likely state
        most_likely_state = np.argmax(state_probs)
        
        if most_likely_state == 0:  # Consolidation
            fusion_weights = {"w_ldc": 0.5, "w_mr": 0.4, "w_tsmom": 0.1}
        elif most_likely_state == 1:  # Mean reversion
            fusion_weights = {"w_ldc": 0.2, "w_mr": 0.6, "w_tsmom": 0.2}
        else:  # Trending
            fusion_weights = {"w_ldc": 0.3, "w_mr": 0.2, "w_tsmom": 0.5}
        
        return state_probs, fusion_weights
    
    def _get_static_weights_response(self, observations: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Get response using configured static weights."""
        # Generate reasonable state probabilities
        # Use a simple heuristic based on observation magnitude
        obs_magnitude = np.linalg.norm(observations)
        
        if obs_magnitude > 1.5:
            state_probs = np.array([0.2, 0.3, 0.5])  # High volatility
        elif obs_magnitude > 0.8:
            state_probs = np.array([0.3, 0.5, 0.2])  # Medium volatility
        else:
            state_probs = np.array([0.5, 0.3, 0.2])  # Low volatility
        
        return state_probs, self.config.static_weights.copy()
    
    def _get_equal_weights_response(self, observations: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Get response using equal weights (last resort)."""
        # Equal state probabilities
        state_probs = np.array([1/3, 1/3, 1/3])
        
        # Equal fusion weights
        fusion_weights = {"w_ldc": 1/3, "w_mr": 1/3, "w_tsmom": 1/3}
        
        return state_probs, fusion_weights
    
    def _get_emergency_response(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """Get emergency response when all fallbacks fail."""
        self.logger.critical("Using emergency response - all fallback strategies failed")
        
        # Conservative equal weights
        state_probs = np.array([1/3, 1/3, 1/3])
        fusion_weights = {"w_ldc": 1/3, "w_mr": 1/3, "w_tsmom": 1/3}
        
        return state_probs, fusion_weights
    
    def store_previous_model(self, model_info: Dict[str, Any]):
        """Store information about previous model for fallback."""
        model_record = {
            **model_info,
            'load_time': time.time()
        }
        
        self._previous_models.append(model_record)
        
        # Keep only the most recent models
        if len(self._previous_models) > self._max_previous_models:
            removed = self._previous_models.pop(0)
            self.logger.debug("Removed old model from fallback storage", extra={
                "removed_version": removed.get('version', 'unknown')
            })
        
        self.logger.info("Stored model for fallback", extra={
            "model_version": model_info.get('version', 'unknown'),
            "total_stored": len(self._previous_models)
        })
    
    def is_fallback_active(self) -> bool:
        """Check if fallback mode is currently active."""
        return self._fallback_active
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get current fallback status and statistics."""
        status = {
            "active": self._fallback_active,
            "current_strategy": self._current_strategy.value if self._current_strategy else None,
            "fallback_count": self._fallback_count,
            "previous_models_available": len(self._previous_models),
            "config": {
                "static_weights_enabled": self.config.enable_static_weights,
                "previous_model_enabled": self.config.enable_previous_model,
                "cached_results_enabled": self.config.enable_cached_results,
                "static_weights": self.config.static_weights
            }
        }
        
        if self._fallback_active and self._fallback_start_time:
            status["fallback_duration_seconds"] = time.time() - self._fallback_start_time
        
        return status
    
    def set_cache_manager(self, cache_manager):
        """Set cache manager for cached results fallback."""
        self._cache_manager = cache_manager
        self.logger.debug("Cache manager set for fallback operations")


# Default fallback configuration
DEFAULT_FALLBACK_CONFIG = FallbackConfig()


def create_fallback_manager(config: Optional[FallbackConfig] = None) -> FallbackManager:
    """Create fallback manager with configuration."""
    return FallbackManager(config or DEFAULT_FALLBACK_CONFIG)