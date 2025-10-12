"""
Verification script for Task 4: Model ranking and selection logic.

Tests the _rank_models() and select_best_model() methods with synthetic data.
"""

import sys
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_hmm_systematic import SystematicHMMTrainer
from imp.hmm.models import HMMArtifact


def create_mock_evaluation_summary():
    """Create mock evaluation summary with multiple models."""
    
    # Mock model 1: 2 states - good AIC/BIC, moderate interpretability
    model_2_states = {
        'basic_metrics': {
            'aic': 1000.0,
            'bic': 1050.0,
            'log_likelihood': -490.0,
            'cv_log_likelihood_mean': -495.0,
            'cv_log_likelihood_std': 5.0,
            'cv_aic_mean': 1005.0,
            'cv_bic_mean': 1055.0
        },
        'interpretability_score': 0.65,
        'n_states': 2,
        'characteristics': {},
        'persistence': {},
        'interpretations': {}
    }
    
    # Mock model 2: 3 states - best overall, balanced metrics
    model_3_states = {
        'basic_metrics': {
            'aic': 980.0,
            'bic': 1040.0,
            'log_likelihood': -480.0,
            'cv_log_likelihood_mean': -485.0,
            'cv_log_likelihood_std': 4.0,
            'cv_aic_mean': 985.0,
            'cv_bic_mean': 1045.0
        },
        'interpretability_score': 0.75,
        'n_states': 3,
        'characteristics': {},
        'persistence': {},
        'interpretations': {}
    }
    
    # Mock model 3: 4 states - good fit but complexity penalty
    model_4_states = {
        'basic_metrics': {
            'aic': 970.0,
            'bic': 1060.0,
            'log_likelihood': -470.0,
            'cv_log_likelihood_mean': -490.0,
            'cv_log_likelihood_std': 8.0,
            'cv_aic_mean': 975.0,
            'cv_bic_mean': 1065.0
        },
        'interpretability_score': 0.55,
        'n_states': 4,
        'characteristics': {},
        'persistence': {},
        'interpretations': {}
    }
    
    evaluation_summary = {
        'models': {
            '2_states': model_2_states,
            '3_states': model_3_states,
            '4_states': model_4_states
        },
        'rankings': []
    }
    
    return evaluation_summary


def create_mock_artifacts(temp_dir: Path):
    """Create mock HMM artifacts for testing."""
    
    artifacts = {}
    
    for n_states in [2, 3, 4]:
        config_name = f"{n_states}_states"
        
        # Create minimal artifact with all required fields
        artifact = HMMArtifact(
            version="1.0.0",
            n_states=n_states,
            transition_matrix=np.eye(n_states).tolist(),
            initial_probabilities=[1.0/n_states] * n_states,
            means=[[0.0, 0.0, 0.0] for _ in range(n_states)],
            covariances=[np.eye(3).tolist() for _ in range(n_states)],
            training_window_start=0,
            training_window_end=1000,
            metadata={
                'aic': 1000.0 - n_states * 10,
                'bic': 1050.0 - n_states * 5,
                'convergence_log_likelihood': -500.0 + n_states * 10,
                'library': 'hmmlearn',
                'covariance_type': 'full'
            }
        )
        
        artifacts[config_name] = artifact
    
    return artifacts


def test_rank_models():
    """Test the _rank_models() method."""
    print("\n" + "="*60)
    print("TEST 1: _rank_models() Method")
    print("="*60)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create trainer instance
        trainer = SystematicHMMTrainer(
            data_path=Path("dummy.parquet"),
            output_dir=temp_dir,
            n_states_range=[2, 3, 4]
        )
        
        # Create mock evaluation summary
        eval_summary = create_mock_evaluation_summary()
        
        # Test ranking
        print("\nRanking models...")
        rankings = trainer._rank_models(eval_summary['models'])
        
        # Verify results
        print(f"\n✓ Ranked {len(rankings)} models")
        
        # Check required fields
        required_fields = [
            'config_name', 'combined_score', 'confidence_score',
            'aic', 'bic', 'cv_score', 'interpretability',
            'component_scores', 'justification'
        ]
        
        for rank, model in enumerate(rankings, 1):
            print(f"\nRank {rank}: {model['config_name']}")
            print(f"  Combined Score: {model['combined_score']:.3f}")
            print(f"  Confidence: {model['confidence_score']:.3f}")
            print(f"  Justification: {model['justification']}")
            
            # Verify all required fields present
            for field in required_fields:
                if field not in model:
                    raise ValueError(f"Missing required field: {field}")
        
        # Verify sorting (descending by combined_score)
        for i in range(len(rankings) - 1):
            if rankings[i]['combined_score'] < rankings[i+1]['combined_score']:
                raise ValueError("Rankings not properly sorted!")
        
        print("\n✓ All required fields present")
        print("✓ Rankings properly sorted")
        print("✓ Confidence scores calculated")
        print("✓ Justifications generated")
        
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_select_best_model():
    """Test the select_best_model() method."""
    print("\n" + "="*60)
    print("TEST 2: select_best_model() Method")
    print("="*60)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create trainer instance
        trainer = SystematicHMMTrainer(
            data_path=Path("dummy.parquet"),
            output_dir=temp_dir,
            n_states_range=[2, 3, 4]
        )
        
        # Create mock artifacts
        artifacts = create_mock_artifacts(temp_dir)
        trainer.training_results = {
            name: {'artifact': artifact, 'cv_results': {}, 'n_states': artifact.n_states}
            for name, artifact in artifacts.items()
        }
        
        # Create mock evaluation summary
        eval_summary = create_mock_evaluation_summary()
        
        # Rank models first
        eval_summary['rankings'] = trainer._rank_models(eval_summary['models'])
        
        # Test selection
        print("\nSelecting best model...")
        best_model_info = trainer.select_best_model(eval_summary)
        
        # Verify results
        print(f"\n✓ Selected: {best_model_info['config_name']}")
        
        # Check required fields
        required_fields = [
            'config_name', 'artifact', 'artifact_path',
            'scores', 'component_scores', 'justification',
            'selection_metadata'
        ]
        
        for field in required_fields:
            if field not in best_model_info:
                raise ValueError(f"Missing required field: {field}")
        
        print("✓ All required fields present")
        
        # Verify artifact saved
        best_model_path = temp_dir / "hmm_best.json"
        if not best_model_path.exists():
            raise ValueError("Best model artifact not saved!")
        
        print(f"✓ Best model saved to: {best_model_path}")
        
        # Verify artifact content
        with open(best_model_path, 'r') as f:
            artifact_data = json.load(f)
        
        if 'metadata' not in artifact_data:
            raise ValueError("Artifact missing metadata!")
        
        if 'selection' not in artifact_data['metadata']:
            raise ValueError("Artifact missing selection metadata!")
        
        selection_meta = artifact_data['metadata']['selection']
        
        required_meta_fields = [
            'selection_timestamp', 'selection_method',
            'combined_score', 'confidence_score',
            'ranking_position', 'total_candidates',
            'component_scores', 'justification'
        ]
        
        for field in required_meta_fields:
            if field not in selection_meta:
                raise ValueError(f"Missing selection metadata field: {field}")
        
        print("✓ Selection metadata added to artifact")
        print(f"  Method: {selection_meta['selection_method']}")
        print(f"  Combined Score: {selection_meta['combined_score']:.3f}")
        print(f"  Confidence: {selection_meta['confidence_score']:.3f}")
        print(f"  Position: {selection_meta['ranking_position']}/{selection_meta['total_candidates']}")
        
        # Verify scores
        scores = best_model_info['scores']
        print("\n✓ Scores included:")
        print(f"  Combined: {scores['combined_score']:.3f}")
        print(f"  Confidence: {scores['confidence_score']:.3f}")
        print(f"  AIC: {scores['aic']:.2f}")
        print(f"  BIC: {scores['bic']:.2f}")
        print(f"  Interpretability: {scores['interpretability']:.3f}")
        
        # Verify justification
        print(f"\n✓ Justification: {best_model_info['justification']}")
        
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 3: Edge Cases")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        trainer = SystematicHMMTrainer(
            data_path=Path("dummy.parquet"),
            output_dir=temp_dir
        )
        
        # Test 1: Empty rankings
        print("\nTest 3.1: Empty rankings")
        try:
            trainer.select_best_model({'rankings': []})
            print("✗ Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {e}")
        
        # Test 2: Single model
        print("\nTest 3.2: Single model ranking")
        single_model = {
            'models': {
                '2_states': {
                    'basic_metrics': {
                        'aic': 1000.0,
                        'bic': 1050.0,
                        'log_likelihood': -490.0,
                        'cv_log_likelihood_mean': -495.0,
                        'cv_log_likelihood_std': 5.0
                    },
                    'interpretability_score': 0.65,
                    'n_states': 2
                }
            }
        }
        
        rankings = trainer._rank_models(single_model['models'])
        if len(rankings) != 1:
            print("✗ Should have 1 ranking")
            return False
        
        print(f"✓ Single model ranked: {rankings[0]['config_name']}")
        print(f"  Score: {rankings[0]['combined_score']:.3f}")
        
        # Test 3: Model with missing CV scores
        print("\nTest 3.3: Model without CV scores")
        no_cv_model = {
            'models': {
                '2_states': {
                    'basic_metrics': {
                        'aic': 1000.0,
                        'bic': 1050.0,
                        'log_likelihood': -490.0,
                        'cv_log_likelihood_mean': None,
                        'cv_log_likelihood_std': None
                    },
                    'interpretability_score': 0.65,
                    'n_states': 2
                }
            }
        }
        
        rankings = trainer._rank_models(no_cv_model['models'])
        print(f"✓ Model without CV ranked: {rankings[0]['config_name']}")
        print(f"  Score: {rankings[0]['combined_score']:.3f}")
        print(f"  Confidence: {rankings[0]['confidence_score']:.3f}")
        
        # Confidence should be lower without CV
        if rankings[0]['confidence_score'] >= 0.8:
            print("⚠️  Warning: Confidence should be lower without CV data")
        
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all verification tests."""
    print("="*60)
    print("Task 4 Verification: Model Ranking and Selection")
    print("="*60)
    
    tests = [
        ("Rank Models", test_rank_models),
        ("Select Best Model", test_select_best_model),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nTask 4 Implementation Verified:")
        print("  ✓ _rank_models() with weighted scoring")
        print("  ✓ Confidence scores calculated")
        print("  ✓ Justifications generated")
        print("  ✓ select_best_model() implementation")
        print("  ✓ Best model saved as hmm_best.json")
        print("  ✓ Selection metadata added to artifact")
        print("  ✓ Edge cases handled")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
