#!/usr/bin/env python3
"""
Quick test script to verify the evaluation framework implementation.
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_hmm_systematic import SystematicHMMTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test the evaluation framework."""
    
    # Use existing test data
    data_path = Path("py/processed_data/hmm_observations.parquet")
    output_dir = Path("py/temp_test_output")
    
    if not data_path.exists():
        logger.error(f"Test data not found: {data_path}")
        logger.info("Please run notebook 01_data_exploration.ipynb first to generate test data")
        return 1
    
    logger.info("="*60)
    logger.info("Testing Evaluation Framework")
    logger.info("="*60)
    
    # Create trainer with small state range for quick test
    trainer = SystematicHMMTrainer(
        data_path=data_path,
        output_dir=output_dir,
        n_states_range=[2, 3],  # Just test 2 and 3 states
        cv_folds=3  # Fewer folds for speed
    )
    
    try:
        # Load data
        logger.info("\n1. Loading data...")
        observations = trainer.load_and_validate_data()
        logger.info(f"   ✓ Loaded {len(observations)} observations")
        
        # Train models
        logger.info("\n2. Training models...")
        training_results = trainer.train_all_configurations(observations)
        
        successful = sum(1 for r in training_results.values() if 'artifact' in r)
        if successful == 0:
            logger.error("   ✗ No models trained successfully")
            return 1
        
        logger.info(f"   ✓ Trained {successful} models")
        
        # Evaluate models (THIS IS WHAT WE'RE TESTING)
        logger.info("\n3. Evaluating models...")
        evaluation_summary = trainer.evaluate_all_models(observations)
        
        # Check results
        logger.info("\n4. Checking evaluation results...")
        
        if not evaluation_summary['models']:
            logger.error("   ✗ No models evaluated")
            return 1
        
        logger.info(f"   ✓ Evaluated {len(evaluation_summary['models'])} models")
        
        # Verify each model has required fields
        for config_name, model_data in evaluation_summary['models'].items():
            logger.info(f"\n   Checking {config_name}:")
            
            required_fields = ['basic_metrics', 'interpretability_score', 'n_states']
            for field in required_fields:
                if field in model_data:
                    logger.info(f"     ✓ {field}: present")
                else:
                    logger.error(f"     ✗ {field}: missing")
                    return 1
            
            # Check if regime analysis succeeded
            if 'characteristics' in model_data:
                logger.info(f"     ✓ characteristics: {len(model_data['characteristics'])} states")
            if 'persistence' in model_data:
                logger.info(f"     ✓ persistence: {len(model_data['persistence'])} states")
            if 'interpretations' in model_data:
                logger.info(f"     ✓ interpretations: {len(model_data['interpretations'])} states")
            
            # Check interpretability score
            interp_score = model_data['interpretability_score']
            logger.info(f"     ✓ interpretability_score: {interp_score:.3f}")
            
            if interp_score < 0 or interp_score > 1:
                logger.error(f"     ✗ interpretability_score out of range [0, 1]")
                return 1
        
        # Check rankings
        logger.info("\n5. Checking rankings...")
        
        if not evaluation_summary['rankings']:
            logger.error("   ✗ No rankings generated")
            return 1
        
        logger.info(f"   ✓ Generated {len(evaluation_summary['rankings'])} rankings")
        
        # Verify rankings are sorted
        scores = [r['combined_score'] for r in evaluation_summary['rankings']]
        if scores != sorted(scores, reverse=True):
            logger.error("   ✗ Rankings not properly sorted")
            return 1
        
        logger.info("   ✓ Rankings properly sorted")
        
        # Display top model
        best_model = evaluation_summary['rankings'][0]
        logger.info(f"\n   Best Model: {best_model['config_name']}")
        logger.info(f"     Combined Score: {best_model['combined_score']:.3f}")
        logger.info(f"     AIC: {best_model['aic']:.2f}")
        logger.info(f"     BIC: {best_model['bic']:.2f}")
        logger.info(f"     Interpretability: {best_model['interpretability']:.3f}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
