"""
HMM inference module for online state prediction and weight computation.
"""

from typing import List, Dict, Optional
import numpy as np
from hmmlearn import hmm

from .models import HMMArtifact, FusionWeights, HMMPrediction, SignalComponents


class HMMInference:
    """Online HMM inference for state prediction and weight computation."""
    
    def __init__(self):
        self.model: Optional[hmm.GaussianHMM] = None
        self.artifact: Optional[HMMArtifact] = None
        self.weights: Optional[FusionWeights] = None
        
    def load_model(self, artifact: HMMArtifact, weights: FusionWeights) -> None:
        """Load trained HMM model and weights."""
        self.artifact = artifact
        self.weights = weights
        
        # Reconstruct HMM model from artifact
        self.model = hmm.GaussianHMM(
            n_components=artifact.n_states,
            covariance_type='full'
        )
        
        # Set the model parameters
        self.model.startprob_ = np.array(artifact.initial_probabilities)
        self.model.transmat_ = np.array(artifact.transition_matrix)
        self.model.means_ = np.array(artifact.means)
        self.model.covars_ = np.array(artifact.covariances)
    
    def predict(self, observation: np.ndarray) -> HMMPrediction:
        """
        Predict state probabilities and compute fused weights.
        
        Args:
            observation: Current observation vector [s_ldc, s_mr, s_tsmom]
            
        Returns:
            HMMPrediction with state probabilities and fused weights
        """
        if self.model is None or self.weights is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure observation is 2D for hmmlearn
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        # Get state probabilities
        state_probs = self.model.predict_proba(observation)[0]
        most_likely_state = np.argmax(state_probs)
        confidence = np.max(state_probs)
        
        # Compute fused weights
        fused_weights = {}
        for key in ["w_ldc", "w_mr", "w_tsmom"]:
            fused_weights[key] = sum(
                state_probs[i] * self.weights.state_weights[i][key]
                for i in range(len(state_probs))
            )
        
        return HMMPrediction(
            state_probabilities=state_probs.tolist(),
            most_likely_state=int(most_likely_state),
            confidence=float(confidence),
            fused_weights=fused_weights,
            timestamp=int(np.datetime64('now').astype('datetime64[s]').astype(int))
        )
    
    def predict_batch(self, observations: np.ndarray) -> List[HMMPrediction]:
        """Predict for a batch of observations."""
        predictions = []
        for obs in observations:
            pred = self.predict(obs)
            predictions.append(pred)
        return predictions
