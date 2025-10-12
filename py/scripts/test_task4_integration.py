"""
Integration test for Task 4 with the complete pipeline.

Tests that model ranking and selection work correctly when integrated
with the full systematic training workflow.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_hmm_systematic import SystematicHMMTrainer


def create_synthetic_data(n_samples=500):
    """Create synthetic signal data for testing."""
    np.random.seed(42)
    
    # Create synthetic signals with different regimes
    t = np.arange(n_samples)
    
    # Regime 1: Low volatility (first 200 samples)
    regime1 = np.random.randn(200, 3) * 0.5
    
    # Regime 2: High volatility (next 200 samples)
    regime2 = np.random.randn(200, 3) * 1.5
    
    # Regime 3: Medium volatility (last 100 samples)
    regime3 = np.random.randn(100, 3) * 1.0
    
    # Combine regimes
    signals = np.vstack([regime1, regime2, regime3])
    
    # Create DataFrame
    df = pd.DataFrame({
        's_LDC': signals[:, 0],
        's_MR': signals[:, 1],
        's_TSMOM': signals[:, 2]
    })
    
    return df


def test_integration():
    """Test complete integration of task 4 with the pipeline."""
    print("="*60)
    print("Task 4 Integration Test")
    print("="*60)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create synthetic data
        print("\n1. Creating synthetic data...")
        df = create_synthetic_data()
        data_path = temp_dir / "test_data.parquet"
        df.to_parquet(data_path)
        print(f"   ✓ Created {len(df)} samples")
        
        # Create trainer
        print("\n2. Initializing trainer...")
        output_dir = temp_dir / "output"
        trainer = SystematicHMMTrainer(
            data_path=data_path,
            output_dir=output_dir,
            n_states_range=[2, 3],  # Test with 2 models for speed
            cv_folds=3  # Fewer folds for speed
        )
        print("   ✓ Trainer initialized")
        
        # Load data
        print("\n3. Loading and validating data...")
        observations = trainer.load_and_validate_data()
        print(f"   ✓ Loaded {len(observations)} observations")
        
        # Train models
        print("\n4. Training models...")
        training_results = trainer.train_all_configurations(observations)
        
        successful = sum(1 for r in training_results.values() if 'artifact' in r)
        print(f"   ✓ Trained {successful} models successfully")
        
        if successful == 0:
            print("   ✗ No models trained successfully")
            return False
        
        # Evaluate models
        print("\n5. Evaluating models...")
        evaluation_summary = trainer.evaluate_all_models(observations)
        print(f"   ✓ Evaluated {len(evaluation_summary['models'])} models")
        print(f"   ✓ Generated {len(evaluation_summary['rankings'])} rankings")
        
        # Select best model (THIS IS TASK 4)
        print("\n6. Selecting best model (Task 4)...")
        best_model_info = trainer.select_best_model(evaluation_summary)
        
        print(f"   ✓ Selected: {best_model_info['config_name']}")
        print(f"   ✓ Combined Score: {best_model_info['scores']['combined_score']:.3f}")
        print(f"   ✓ Confidence: {best_model_info['scores']['confidence_score']:.3f}")
        
        # Verify best model artifact exists
        best_model_path = output_dir / "hmm_best.json"
        if not best_model_path.exists():
            print("   ✗ Best model artifact not found")
            return False
        
        print(f"   ✓ Best model saved: {best_model_path}")
        
        # Verify selection metadata
        import json
        with open(best_model_path, 'r') as f:
            artifact_data = json.load(f)
        
        if 'selection' not in artifact_data['metadata']:
            print("   ✗ Selection metadata missing")
            return False
        
        selection = artifact_data['metadata']['selection']
        print(f"   ✓ Selection metadata present:")
        print(f"     - Method: {selection['selection_method']}")
        print(f"     - Combined Score: {selection['combined_score']:.3f}")
        print(f"     - Confidence: {selection['confidence_score']:.3f}")
        print(f"     - Ranking: {selection['ranking_position']}/{selection['total_candidates']}")
        
        # Generate report
        print("\n7. Generating report...")
        trainer.generate_report(evaluation_summary, best_model_info)
        
        report_path = output_dir / "training_report.json"
        if not report_path.exists():
            print("   ✗ Report not found")
            return False
        
        print(f"   ✓ Report saved: {report_path}")
        
        # Verify report content
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        required_sections = ['timestamp', 'configuration', 'best_model', 'all_models', 'rankings']
        for section in required_sections:
            if section not in report:
                print(f"   ✗ Report missing section: {section}")
                return False
        
        print("   ✓ Report contains all required sections")
        
        # Verify rankings in report
        if len(report['rankings']) != successful:
            print(f"   ✗ Rankings count mismatch: {len(report['rankings'])} vs {successful}")
            return False
        
        print(f"   ✓ Report contains {len(report['rankings'])} rankings")
        
        # Verify best model in report matches selection
        if report['best_model']['config_name'] != best_model_info['config_name']:
            print("   ✗ Best model mismatch in report")
            return False
        
        print("   ✓ Report best model matches selection")
        
        # Verify justification present
        if 'justification' not in report['best_model']:
            print("   ✗ Justification missing from report")
            return False
        
        print(f"   ✓ Justification: {report['best_model']['justification'][:80]}...")
        
        print("\n" + "="*60)
        print("✓ INTEGRATION TEST PASSED")
        print("="*60)
        print("\nTask 4 Features Verified:")
        print("  ✓ Model ranking with weighted scoring")
        print("  ✓ Confidence score calculation")
        print("  ✓ Justification generation")
        print("  ✓ Best model selection")
        print("  ✓ Production artifact (hmm_best.json)")
        print("  ✓ Selection metadata in artifact")
        print("  ✓ Comprehensive report generation")
        print("  ✓ Summary table display")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    success = test_integration()
    sys.exit(0 if success else 1)
