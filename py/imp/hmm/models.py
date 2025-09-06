"""
HMM model definitions and data structures.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, field_validator, Field, ConfigDict
import numpy as np


class HMMArtifact(BaseModel):
    """HMM model artifact containing all necessary parameters."""
    
    version: str
    n_states: int = Field(gt=0, description="Number of states must be positive")
    transition_matrix: List[List[float]]
    initial_probabilities: List[float]
    means: List[List[float]]
    covariances: List[List[List[float]]]
    training_window_start: int
    training_window_end: int
    metadata: Dict[str, Any]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('transition_matrix')
    @classmethod
    def validate_transition_matrix(cls, v, info):
        """Validate transition matrix dimensions and probabilities."""
        if not info.data or 'n_states' not in info.data:
            return v
        
        n_states = info.data['n_states']
        
        # Check dimensions
        if len(v) != n_states:
            raise ValueError(f"Transition matrix must have {n_states} rows, got {len(v)}")
        
        for i, row in enumerate(v):
            if len(row) != n_states:
                raise ValueError(f"Row {i} must have {n_states} columns, got {len(row)}")
            
            # Check probabilities sum to 1
            row_sum = sum(row)
            if not np.isclose(row_sum, 1.0, atol=1e-6):
                raise ValueError(f"Row {i} probabilities must sum to 1, got {row_sum}")
            
            # Check non-negative probabilities
            for j, prob in enumerate(row):
                if prob < 0:
                    raise ValueError(f"Transition probability [{i}][{j}] must be non-negative, got {prob}")
        
        return v
    
    @field_validator('initial_probabilities')
    @classmethod
    def validate_initial_probabilities(cls, v, info):
        """Validate initial probabilities."""
        if not info.data or 'n_states' not in info.data:
            return v
        
        n_states = info.data['n_states']
        
        if len(v) != n_states:
            raise ValueError(f"Initial probabilities must have {n_states} elements, got {len(v)}")
        
        # Check probabilities sum to 1
        prob_sum = sum(v)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            raise ValueError(f"Initial probabilities must sum to 1, got {prob_sum}")
        
        # Check non-negative probabilities
        for i, prob in enumerate(v):
            if prob < 0:
                raise ValueError(f"Initial probability {i} must be non-negative, got {prob}")
        
        return v
    
    @field_validator('means')
    @classmethod
    def validate_means(cls, v, info):
        """Validate means dimensions."""
        if not info.data or 'n_states' not in info.data:
            return v
        
        n_states = info.data['n_states']
        
        if len(v) != n_states:
            raise ValueError(f"Means must have {n_states} elements, got {len(v)}")
        
        # Check all means have same dimension
        if v:
            n_features = len(v[0])
            for i, mean in enumerate(v):
                if len(mean) != n_features:
                    raise ValueError(f"Mean {i} must have {n_features} features, got {len(mean)}")
        
        return v
    
    @field_validator('covariances')
    @classmethod
    def validate_covariances(cls, v, info):
        """Validate covariance matrices."""
        if not info.data or 'n_states' not in info.data or 'means' not in info.data:
            return v
        
        n_states = info.data['n_states']
        means = info.data['means']
        
        if len(v) != n_states:
            raise ValueError(f"Covariances must have {n_states} elements, got {len(v)}")
        
        if means:
            n_features = len(means[0])
            
            for i, cov in enumerate(v):
                if len(cov) != n_features:
                    raise ValueError(f"Covariance {i} must have {n_features} rows, got {len(cov)}")
                
                for j, row in enumerate(cov):
                    if len(row) != n_features:
                        raise ValueError(f"Covariance {i} row {j} must have {n_features} columns, got {len(row)}")
                
                # Check if matrix is positive definite
                cov_matrix = np.array(cov)
                try:
                    np.linalg.cholesky(cov_matrix)
                except np.linalg.LinAlgError:
                    raise ValueError(f"Covariance matrix {i} is not positive definite")
        
        return v


class FusionWeights(BaseModel):
    """Per-state fusion weights for signal combination."""
    
    version: str
    state_weights: List[Dict[str, float]]  # List of weight dicts per state
    model_version: str  # References HMMArtifact version
    training_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    @field_validator('state_weights')
    @classmethod
    def validate_state_weights(cls, v):
        """Validate state weights."""
        # Check that all state weights have the same keys
        if not v:
            return v
        
        expected_keys = set(v[0].keys())
        for i, state_weight in enumerate(v):
            if set(state_weight.keys()) != expected_keys:
                raise ValueError(f"State {i} weights must have keys {expected_keys}, got {set(state_weight.keys())}")
            
            # Check that weights sum to 1
            weight_sum = sum(state_weight.values())
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                raise ValueError(f"State {i} weights must sum to 1, got {weight_sum}")
            
            # Check non-negative weights
            for key, weight in state_weight.items():
                if weight < 0:
                    raise ValueError(f"State {i} weight '{key}' must be non-negative, got {weight}")
        
        return v
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata_consistency(cls, v, info):
        """Validate metadata consistency with state weights."""
        if not info.data or 'state_weights' not in info.data:
            return v
        
        state_weights = info.data['state_weights']
        
        # Check n_states consistency if present in metadata
        if 'n_states' in v:
            if v['n_states'] != len(state_weights):
                raise ValueError(f"Metadata n_states ({v['n_states']}) must match number of state weights ({len(state_weights)})")
        
        return v


class HMMPrediction(BaseModel):
    """HMM prediction result."""
    
    state_probabilities: List[float]
    most_likely_state: int
    confidence: float
    fused_weights: Dict[str, float]
    timestamp: int
    
    @field_validator('state_probabilities')
    @classmethod
    def validate_state_probabilities(cls, v):
        """Validate state probabilities."""
        if not v:
            raise ValueError("State probabilities cannot be empty")
        
        # Check probabilities sum to 1
        prob_sum = sum(v)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            raise ValueError(f"State probabilities must sum to 1, got {prob_sum}")
        
        # Check non-negative probabilities
        for i, prob in enumerate(v):
            if prob < 0:
                raise ValueError(f"State probability {i} must be non-negative, got {prob}")
        
        return v
    
    @field_validator('most_likely_state')
    @classmethod
    def validate_most_likely_state(cls, v, info):
        """Validate most likely state index."""
        if not info.data or 'state_probabilities' not in info.data:
            return v
        
        n_states = len(info.data['state_probabilities'])
        if not 0 <= v < n_states:
            raise ValueError(f"Most likely state must be between 0 and {n_states-1}, got {v}")
        
        return v
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence value."""
        if not 0 <= v <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        
        return v
    
    @field_validator('fused_weights')
    @classmethod
    def validate_fused_weights(cls, v):
        """Validate fused weights."""
        if not v:
            raise ValueError("Fused weights cannot be empty")
        
        # Check that weights sum to 1
        weight_sum = sum(v.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(f"Fused weights must sum to 1, got {weight_sum}")
        
        # Check non-negative weights
        for key, weight in v.items():
            if weight < 0:
                raise ValueError(f"Fused weight '{key}' must be non-negative, got {weight}")
        
        return v


class SignalComponents(BaseModel):
    """Individual signal components."""
    
    s_ldc: float
    s_mr: float
    s_tsmom: float
    timestamp: int
