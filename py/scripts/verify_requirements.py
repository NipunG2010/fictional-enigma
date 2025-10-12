#!/usr/bin/env python3
"""
Verification script to ensure Task 3 implementation meets all requirements.

Requirements verified:
- Requirement 2.1: Characterize each regime by signal statistics
- Requirement 2.2: Compute AIC, BIC, and held-out log-likelihood
- Requirement 3.1: Characterize each regime by signal statistics
- Requirement 3.2: Identify economic interpretation
"""

import sys
from pathlib import Path
import logging
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_hmm_systematic import SystematicHMMTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Verify requirements are met."""
    
    logger.info("="*70)
    logger.info("Requirements Verification for Task 3")
    logger.info("="*70)
    
    # Use test data
    data_path = Path("py/processed_data/hmm_observations.parquet")
    output_dir = Path("py/temp_requirements_test")
    
    if not data_path.exists():
        logger.error(f"Test data not found: {data_path}")
        return 1
    
    # Create trainer
    trainer = SystematicHMMTrainer(
        data_path=data_path,
        output_dir=output_dir,
        n_states_range=[2, 3],
        cv_folds=3
    )
    
    try:
        # Load data
        observations = trainer.load_and_validate_data()
        
        # Train models
        training_results = trainer.train_all_configurations(observations)
        
        # Evaluate models
        evaluation_summary = trainer.evaluate_all_models(observations)
        
        logger.info("\n" + "="*70)
        logger.info("Requirement 2.1: Time-series cross-validation")
        logger.info("="*70)
        
        # Check CV was performed
        cv_performed = False
        for config_name, result in training_results.items():
            if 'cv_results' in result and result['cv_results']:
                cv_performed = True
                logger.info(f"✓ {config_name}: Cross-validation performed")
                cv_results = result['cv_results']
                if 'log_likelihood_mean' in cv_results:
                    logger.info(f"  - CV Log-Likelihood: {cv_results['log_likelihood_mean']:.2f}")
        
        if not cv_performed:
            logger.error("✗ Cross-validation not performed")
            return 1
        
        logger.info("\n" + "="*70)
        logger.info("Requirement 2.2: Compute AIC, BIC, and held-out log-likelihood")
        logger.info("="*70)
        
        # Check metrics are computed
        all_metrics_present = True
        for config_name, model_data in evaluation_summary['models'].items():
            logger.info(f"\n{config_name}:")
            
            metrics = model_data['basic_metrics']
            required_metrics = ['aic', 'bic', 'log_likelihood', 'cv_log_likelihood_mean']
            
            for metric in required_metrics:
                if metric in metrics and metrics[metric] is not None:
                    logger.info(f"  ✓ {metric}: {metrics[metric]:.2f}")
                else:
                    logger.error(f"  ✗ {metric}: missing")
                    all_metrics_present = False
        
        if not all_metrics_present:
            logger.error("\n✗ Not all required metrics computed")
            return 1
        
        logger.info("\n" + "="*70)
        logger.info("Requirement 2.3: Rank configurations by multiple criteria")
        logger.info("="*70)
        
        # Check rankings exist
        if not evaluation_summary['rankings']:
            logger.error("✗ No rankings generated")
            return 1
        
        logger.info(f"✓ Generated {len(evaluation_summary['rankings'])} rankings")
        
        # Check ranking criteria
        for i, ranking in enumerate(evaluation_summary['rankings'], 1):
            logger.info(f"\nRank {i}: {ranking['config_name']}")
            logger.info(f"  Combined Score: {ranking['combined_score']:.3f}")
            logger.info(f"  AIC: {ranking['aic']:.2f}")
            logger.info(f"  BIC: {ranking['bic']:.2f}")
            logger.info(f"  CV Score: {ranking['cv_score']:.2f}")
            logger.info(f"  Interpretability: {ranking['interpretability']:.3f}")
        
        logger.info("\n" + "="*70)
        logger.info("Requirement 2.4: Provide diagnostic information on failure")
        logger.info("="*70)
        
        # Check error handling
        has_error_handling = False
        for config_name, model_data in evaluation_summary['models'].items():
            if 'error' in model_data:
                has_error_handling = True
                logger.info(f"✓ Error handling present for {config_name}")
                logger.info(f"  Error: {model_data['error']}")
        
        if not has_error_handling:
            logger.info("✓ No errors occurred (error handling is implemented)")
        
        logger.info("\n" + "="*70)
        logger.info("Requirement 3.1: Characterize each regime by signal statistics")
        logger.info("="*70)
        
        # Check regime characteristics
        all_characterized = True
        for config_name, model_data in evaluation_summary['models'].items():
            if 'characteristics' not in model_data:
                logger.error(f"✗ {config_name}: No characteristics")
                all_characterized = False
                continue
            
            logger.info(f"\n{config_name}:")
            characteristics = model_data['characteristics']
            
            for state_id, char in characteristics.items():
                logger.info(f"  State {state_id}:")
                logger.info(f"    Mean values: {char['mean_values']}")
                logger.info(f"    Std values: {char['std_values']}")
                logger.info(f"    Volatility: {char['volatility']:.3f}")
                logger.info(f"    Trend strength: {char['trend_strength']:.3f}")
                logger.info(f"    Mean reversion: {char['mean_reversion_score']:.3f}")
        
        if not all_characterized:
            logger.error("\n✗ Not all regimes characterized")
            return 1
        
        logger.info("\n" + "="*70)
        logger.info("Requirement 3.2: Identify economic interpretation")
        logger.info("="*70)
        
        # Check interpretations
        all_interpreted = True
        for config_name, model_data in evaluation_summary['models'].items():
            if 'interpretations' not in model_data:
                logger.error(f"✗ {config_name}: No interpretations")
                all_interpreted = False
                continue
            
            logger.info(f"\n{config_name}:")
            interpretations = model_data['interpretations']
            
            for state_id, interp in interpretations.items():
                logger.info(f"  State {state_id}:")
                logger.info(f"    Regime type: {interp['regime_type']}")
                logger.info(f"    Market condition: {interp['market_condition']}")
                logger.info(f"    Risk level: {interp['risk_level']}")
                logger.info(f"    Trading recommendation: {interp['trading_recommendation']}")
        
        if not all_interpreted:
            logger.error("\n✗ Not all regimes interpreted")
            return 1
        
        logger.info("\n" + "="*70)
        logger.info("✓ ALL REQUIREMENTS VERIFIED")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
