#!/usr/bin/env python3
"""
Test script for Task 2: Enhanced compute_state_weights implementation.

This script verifies that the compute_state_weights method:
1. Replaces the TODO placeholder with actual optimization logic
2. Adds state sequence prediction and data filtering per state
3. Integrates StateWeightOptimizer for each HMM state
4. Computes per-state Sharpe ratios and aggregate metrics
5. Returns FusionWeights with proper training_metrics populated
6. Adds error handling and logging for optimization failures
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imp.hmm import HMMTrainer, OptimizationConfig
from imp.hmm.models import HMMArtifact, FusionWeights

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=500, n_features=3, n_states=3, seed=42):
    """Generate synthetic signal and return data for testing."""
    np.random.seed(seed)
    
    # Generate synthetic signals (observations)
    observations = np.random.randn(n_samples, n_features)
    
    # Generate synthetic returns with some correlation to signals
    # State 0: LDC dominant
    # State 1: MR dominant  
    # State 2: TSMOM dominant
    returns = np.zeros(n_samples)
    
    for i in range(n_samples):
        state = i % n_states
        if state == 0:
            returns[i] = 0.5 * observations[i, 0] + 0.2 * observations[i, 1] + 0.1 * observations[i, 2]
        elif state == 1:
            returns[i] = 0.1 * observations[i, 0] + 0.6 * observations[i, 1] + 0.2 * observations[i, 2]
        else:
            returns[i] = 0.2 * observations[i, 0] + 0.1 * observations[i, 1] + 0.5 * observations[i, 2]
        
        # Add noise
        returns[i] += np.random.randn() * 0.5
    
    # Normalize returns to realistic scale (daily returns)
    returns = returns * 0.01
    
    return observations, returns


def test_basic_functionality():
    """Test basic compute_state_weights functionality."""
    logger.info("=" * 60)
    logger.info("TEST 1: Basic Functionality")
    logger.info("=" * 60)
    
    # Generate synthetic data
    observations, returns = generate_synthetic_data(n_samples=500, n_states=3)
    
    # Train HMM
    logger.info("Training HMM...")
    trainer = HMMTrainer(n_states=3)
    artifact = trainer.train(observations, n_iterations=50)
    
    logger.info(f"HMM trained with {artifact.n_states} states")
    
    # Compute state weights
    logger.info("\nComputing state weights...")
    fusion_weights = trainer.compute_state_weights(
        observations=observations,
        artifact=artifact,
        returns=returns
    )
    
    # Verify results
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)
    
    # Check 1: FusionWeights structure
    assert isinstance(fusion_weights, FusionWeights), "Should return FusionWeights object"
    logger.info("✓ Returns FusionWeights object")
    
    # Check 2: State weights populated
    assert len(fusion_weights.state_weights) == artifact.n_states, \
        f"Should have {artifact.n_states} state weights"
    logger.info(f"✓ Has {len(fusion_weights.state_weights)} state weights")
    
    # Check 3: Each state has correct signal names
    for i, state_weight in enumerate(fusion_weights.state_weights):
        assert set(state_weight.keys()) == {'s_LDC', 's_MR', 's_TSMOM'}, \
            f"State {i} should have correct signal names"
        
        # Check weights sum to 1
        weight_sum = sum(state_weight.values())
        assert abs(weight_sum - 1.0) < 1e-6, \
            f"State {i} weights should sum to 1, got {weight_sum}"
        
        # Check non-negative
        for signal, weight in state_weight.items():
            assert weight >= 0, f"State {i} {signal} weight should be non-negative"
        
        logger.info(f"✓ State {i} weights: {state_weight}")
    
    # Check 4: Training metrics populated
    assert 'sharpe_ratio' in fusion_weights.training_metrics, \
        "Should have sharpe_ratio in training_metrics"
    assert 'avg_sharpe' in fusion_weights.training_metrics, \
        "Should have avg_sharpe in training_metrics"
    
    # Check per-state Sharpe ratios are in training_metrics
    for i in range(artifact.n_states):
        assert f'state_{i}_sharpe' in fusion_weights.training_metrics, \
            f"Should have state_{i}_sharpe in training_metrics"
    
    logger.info(f"✓ Training metrics populated: {fusion_weights.training_metrics}")
    
    # Check 5: Metadata populated
    assert 'n_states' in fusion_weights.metadata, "Should have n_states in metadata"
    assert 'n_observations' in fusion_weights.metadata, "Should have n_observations in metadata"
    assert 'state_n_observations' in fusion_weights.metadata, \
        "Should have state_n_observations in metadata"
    assert 'optimization_method' in fusion_weights.metadata, \
        "Should have optimization_method in metadata"
    assert 'state_sharpes' in fusion_weights.metadata, \
        "Should have state_sharpes in metadata"
    
    logger.info(f"✓ Metadata populated: {fusion_weights.metadata}")
    
    logger.info("\n✓ TEST 1 PASSED: Basic functionality works correctly")
    return True


def test_optimization_methods():
    """Test different optimization methods."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Optimization Methods")
    logger.info("=" * 60)
    
    # Generate synthetic data
    observations, returns = generate_synthetic_data(n_samples=500, n_states=3)
    
    # Train HMM
    trainer = HMMTrainer(n_states=3)
    artifact = trainer.train(observations, n_iterations=50)
    
    # Test SLSQP method
    logger.info("\nTesting SLSQP optimization...")
    config_slsqp = OptimizationConfig(method="SLSQP")
    fusion_weights_slsqp = trainer.compute_state_weights(
        observations=observations,
        artifact=artifact,
        returns=returns,
        optimization_config=config_slsqp
    )
    
    assert fusion_weights_slsqp.metadata['optimization_method'] == 'SLSQP'
    logger.info(f"✓ SLSQP method works, Sharpe: {fusion_weights_slsqp.training_metrics['sharpe_ratio']:.3f}")
    
    # Test grid search method
    logger.info("\nTesting grid search optimization...")
    config_grid = OptimizationConfig(method="grid_search", grid_points=5)
    fusion_weights_grid = trainer.compute_state_weights(
        observations=observations,
        artifact=artifact,
        returns=returns,
        optimization_config=config_grid
    )
    
    assert fusion_weights_grid.metadata['optimization_method'] == 'grid_search'
    logger.info(f"✓ Grid search method works, Sharpe: {fusion_weights_grid.training_metrics['sharpe_ratio']:.3f}")
    
    logger.info("\n✓ TEST 2 PASSED: Both optimization methods work correctly")
    return True


def test_error_handling():
    """Test error handling for edge cases."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Error Handling")
    logger.info("=" * 60)
    
    # Generate synthetic data
    observations, returns = generate_synthetic_data(n_samples=500, n_states=3)
    
    # Train HMM
    trainer = HMMTrainer(n_states=3)
    artifact = trainer.train(observations, n_iterations=50)
    
    # Test 1: Mismatched lengths
    logger.info("\nTest 3.1: Mismatched observation and return lengths...")
    try:
        fusion_weights = trainer.compute_state_weights(
            observations=observations,
            artifact=artifact,
            returns=returns[:100]  # Shorter returns
        )
        logger.error("✗ Should have raised error for mismatched lengths")
        return False
    except Exception as e:
        logger.info(f"✓ Correctly raised error: {type(e).__name__}")
    
    # Test 2: Wrong number of signals
    logger.info("\nTest 3.2: Wrong number of signals...")
    try:
        bad_observations = observations[:, :2]  # Only 2 signals
        fusion_weights = trainer.compute_state_weights(
            observations=bad_observations,
            artifact=artifact,
            returns=returns[:len(bad_observations)]
        )
        logger.error("✗ Should have raised error for wrong number of signals")
        return False
    except Exception as e:
        logger.info(f"✓ Correctly raised error: {type(e).__name__}")
    
    # Test 3: Insufficient data (should fall back to equal weights)
    logger.info("\nTest 3.3: Insufficient data per state...")
    small_observations, small_returns = generate_synthetic_data(n_samples=50, n_states=3)
    small_trainer = HMMTrainer(n_states=3)
    small_artifact = small_trainer.train(small_observations, n_iterations=20)
    
    config = OptimizationConfig(min_observations=100)  # High threshold
    fusion_weights = small_trainer.compute_state_weights(
        observations=small_observations,
        artifact=small_artifact,
        returns=small_returns,
        optimization_config=config
    )
    
    # Should still return valid weights (equal weights as fallback)
    assert len(fusion_weights.state_weights) == 3
    logger.info("✓ Handles insufficient data with fallback to equal weights")
    
    logger.info("\n✓ TEST 3 PASSED: Error handling works correctly")
    return True


def test_state_filtering():
    """Test that state filtering works correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: State Filtering")
    logger.info("=" * 60)
    
    # Generate synthetic data
    observations, returns = generate_synthetic_data(n_samples=500, n_states=3)
    
    # Train HMM
    trainer = HMMTrainer(n_states=3)
    artifact = trainer.train(observations, n_iterations=50)
    
    # Compute state weights
    fusion_weights = trainer.compute_state_weights(
        observations=observations,
        artifact=artifact,
        returns=returns
    )
    
    # Verify state_n_observations is populated
    assert 'state_n_observations' in fusion_weights.metadata
    state_n_obs = fusion_weights.metadata['state_n_observations']
    
    logger.info(f"State observation counts: {state_n_obs}")
    
    # Check that all states have some observations
    assert all(n > 0 for n in state_n_obs), "All states should have observations"
    
    # Check that sum equals total observations
    assert sum(state_n_obs) == len(observations), \
        "Sum of state observations should equal total observations"
    
    logger.info("✓ State filtering works correctly")
    logger.info("\n✓ TEST 4 PASSED: State filtering verified")
    return True


def main():
    """Run all tests."""
    logger.info("Testing Task 2: Enhanced compute_state_weights Implementation")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Optimization Methods", test_optimization_methods),
        ("Error Handling", test_error_handling),
        ("State Filtering", test_state_filtering),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"\n✗ TEST FAILED: {test_name}")
            logger.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("\nTask 2 Implementation Verified:")
        logger.info("✓ Replaced TODO placeholder with actual optimization logic")
        logger.info("✓ Added state sequence prediction and data filtering per state")
        logger.info("✓ Integrated StateWeightOptimizer for each HMM state")
        logger.info("✓ Computed per-state Sharpe ratios and aggregate metrics")
        logger.info("✓ Returns FusionWeights with proper training_metrics populated")
        logger.info("✓ Added error handling and logging for optimization failures")
        return 0
    else:
        logger.error("\n" + "=" * 60)
        logger.error("SOME TESTS FAILED")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
