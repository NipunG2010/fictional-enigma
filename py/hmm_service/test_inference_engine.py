#!/usr/bin/env python3
"""
Test script for HMM Inference Engine functionality.

This script tests the core inference engine with a mock HMM model
to verify that the implementation works correctly.
"""

import asyncio
import numpy as np
import sys
import os

# Add paths for imports
sys.path.append('.')
sys.path.append('..')

from core.inference_engine import HMMInferenceEngine
from core.cache import CacheManager
from core.config import get_settings

# Import HMM models
from imp.hmm.models import HMMArtifact, FusionWeights


def create_mock_hmm_artifact():
    """Create a mock HMM artifact for testing."""
    return HMMArtifact(
        version="test_v1.0.0",
        n_states=3,
        transition_matrix=[
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5]
        ],
        initial_probabilities=[0.33, 0.33, 0.34],
        means=[
            [0.1, 0.2, 0.3],  # State 0 means
            [0.4, 0.5, 0.6],  # State 1 means
            [0.7, 0.8, 0.9]   # State 2 means
        ],
        covariances=[
            # State 0 covariance (3x3)
            [[0.1, 0.0, 0.0],
             [0.0, 0.1, 0.0],
             [0.0, 0.0, 0.1]],
            # State 1 covariance (3x3)
            [[0.2, 0.0, 0.0],
             [0.0, 0.2, 0.0],
             [0.0, 0.0, 0.2]],
            # State 2 covariance (3x3)
            [[0.15, 0.0, 0.0],
             [0.0, 0.15, 0.0],
             [0.0, 0.0, 0.15]]
        ],
        training_window_start=1000000,
        training_window_end=2000000,
        metadata={
            "library": "test",
            "algorithm": "baum_welch",
            "covariance_type": "full"
        }
    )


def create_mock_fusion_weights():
    """Create mock fusion weights for testing."""
    return FusionWeights(
        version="test_weights_v1.0.0",
        state_weights=[
            {"w_ldc": 0.5, "w_mr": 0.3, "w_tsmom": 0.2},  # State 0
            {"w_ldc": 0.2, "w_mr": 0.5, "w_tsmom": 0.3},  # State 1
            {"w_ldc": 0.3, "w_mr": 0.2, "w_tsmom": 0.5}   # State 2
        ],
        model_version="test_v1.0.0",
        training_metrics={"accuracy": 0.85, "log_likelihood": -1234.5},
        metadata={"n_states": 3, "signals": ["ldc", "mr", "tsmom"]}
    )


async def test_inference_engine():
    """Test the HMM inference engine functionality."""
    print("Testing HMM Inference Engine...")
    
    # Create configuration
    config = get_settings()
    
    # Create cache manager
    cache_manager = CacheManager(config)
    await cache_manager.initialize()
    
    # Create inference engine
    inference_engine = HMMInferenceEngine(config)
    await inference_engine.initialize(cache_manager=cache_manager)
    
    # Create mock models
    hmm_artifact = create_mock_hmm_artifact()
    fusion_weights = create_mock_fusion_weights()
    
    print(f"Created mock HMM model with {hmm_artifact.n_states} states")
    
    # Load model into inference engine
    success = await inference_engine.load_model(hmm_artifact, fusion_weights)
    print(f"Model loaded: {success}")
    
    # Validate model
    is_valid = inference_engine.validate_model()
    print(f"Model validation: {is_valid}")
    
    # Test inference with sample observations
    test_observations = [
        np.array([0.1, 0.2, 0.3]),  # Close to state 0 mean
        np.array([0.4, 0.5, 0.6]),  # Close to state 1 mean
        np.array([0.7, 0.8, 0.9]),  # Close to state 2 mean
        np.array([0.0, 0.0, 0.0]),  # Different observation
    ]
    
    print("\nTesting inference...")
    for i, obs in enumerate(test_observations):
        print(f"\nObservation {i+1}: {obs}")
        
        # Test state probabilities
        state_probs = await inference_engine.predict_state_probabilities(obs)
        print(f"State probabilities: {state_probs}")
        print(f"Most likely state: {np.argmax(state_probs)}")
        
        # Test fusion weights
        fusion_weights_result = await inference_engine.compute_fusion_weights(state_probs)
        print(f"Fusion weights: {fusion_weights_result}")
        
        # Test complete prediction
        prediction = await inference_engine.predict_complete(obs)
        print(f"Complete prediction - State: {prediction.most_likely_state}, "
              f"Confidence: {prediction.confidence:.3f}")
    
    # Test caching
    print("\nTesting caching...")
    obs = test_observations[0]
    
    # First call (should compute)
    start_time = asyncio.get_event_loop().time()
    state_probs1 = await inference_engine.predict_state_probabilities(obs, use_cache=True)
    time1 = asyncio.get_event_loop().time() - start_time
    
    # Reset state for cache test
    inference_engine.reset_state()
    
    # Second call (should use cache)
    start_time = asyncio.get_event_loop().time()
    state_probs2 = await inference_engine.predict_state_probabilities(obs, use_cache=True)
    time2 = asyncio.get_event_loop().time() - start_time
    
    print(f"First call time: {time1*1000:.2f}ms")
    print(f"Second call time: {time2*1000:.2f}ms")
    print(f"Results match: {np.allclose(state_probs1, state_probs2)}")
    
    # Get cache statistics
    cache_stats = cache_manager.get_cache_stats()
    print(f"\nCache statistics:")
    for cache_name, stats in cache_stats.items():
        if stats and isinstance(stats, dict):
            print(f"  {cache_name}: {stats.get('hits', 0)} hits, "
                  f"{stats.get('misses', 0)} misses, "
                  f"hit rate: {stats.get('hit_rate', 0):.2%}")
    
    # Get model info
    model_info = inference_engine.get_model_info()
    print(f"\nModel info: {model_info}")
    
    # Get health status
    health_status = inference_engine.get_health_status()
    print(f"Health status: {health_status}")
    
    # Cleanup
    await cache_manager.cleanup()
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_inference_engine())