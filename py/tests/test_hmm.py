"""
Test HMM components
"""

import pytest
import numpy as np
from imp.hmm.models import HMMArtifact, FusionWeights, HMMPrediction, SignalComponents
from imp.hmm.trainer import HMMTrainer
from imp.hmm.inference import HMMInference


def test_hmm_artifact_creation():
    """Test HMM artifact creation"""
    artifact = HMMArtifact(
        version="v1.0",
        n_states=3,
        transition_matrix=[[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]],
        initial_probabilities=[0.5, 0.3, 0.2],
        means=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
        covariances=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 3,
        training_window_start=0,
        training_window_end=1000,
        metadata={"test": True}
    )
    
    assert artifact.version == "v1.0"
    assert artifact.n_states == 3
    assert len(artifact.transition_matrix) == 3
    assert len(artifact.initial_probabilities) == 3


def test_fusion_weights_creation():
    """Test fusion weights creation"""
    weights = FusionWeights(
        version="v1.0",
        state_weights=[
            {"w_ldc": 0.4, "w_mr": 0.3, "w_tsmom": 0.3},
            {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3},
            {"w_ldc": 0.3, "w_mr": 0.2, "w_tsmom": 0.5}
        ],
        model_version="v1.0",
        training_metrics={"sharpe_ratio": 1.5},
        metadata={"test": True}
    )
    
    assert weights.version == "v1.0"
    assert len(weights.state_weights) == 3
    assert weights.state_weights[0]["w_ldc"] == 0.4


def test_signal_components_creation():
    """Test signal components creation"""
    components = SignalComponents(
        s_ldc=0.8,
        s_mr=0.2,
        s_tsmom=0.1,
        timestamp=1000
    )
    
    assert components.s_ldc == 0.8
    assert components.s_mr == 0.2
    assert components.s_tsmom == 0.1
    assert components.timestamp == 1000


def test_hmm_trainer_creation():
    """Test HMM trainer creation"""
    trainer = HMMTrainer(n_states=3)
    assert trainer.n_states == 3
    assert trainer.model is None


def test_hmm_inference_creation():
    """Test HMM inference creation"""
    inference = HMMInference()
    assert inference.model is None
    assert inference.artifact is None
    assert inference.weights is None


@pytest.mark.slow
def test_hmm_training():
    """Test HMM training with synthetic data"""
    # Generate synthetic observation data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # Create 3 different regimes
    regime1 = np.random.multivariate_normal([0, 0, 0], np.eye(3), n_samples // 3)
    regime2 = np.random.multivariate_normal([2, 2, 2], np.eye(3), n_samples // 3)
    regime3 = np.random.multivariate_normal([-2, -2, -2], np.eye(3), n_samples - 2 * (n_samples // 3))
    
    observations = np.vstack([regime1, regime2, regime3])
    np.random.shuffle(observations)
    
    # Train HMM
    trainer = HMMTrainer(n_states=3)
    artifact = trainer.train(observations, n_iterations=10)
    
    assert artifact.n_states == 3
    assert len(artifact.transition_matrix) == 3
    assert len(artifact.means) == 3
    assert len(artifact.covariances) == 3


if __name__ == "__main__":
    pytest.main([__file__])
