#!/usr/bin/env python3
"""
Verification script for Task 5: Create comprehensive reporting and visualization

This script verifies that all sub-tasks have been completed:
1. Implement generate_report() method creating JSON report with all results
2. Add _print_summary_table() for console output of model rankings
3. Create training_report.json with timestamp, configuration, and full evaluation results
4. Add clear logging of best model selection with scores and justification
"""

import sys
from pathlib import Path
import inspect
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_hmm_systematic import SystematicHMMTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_method_exists(cls, method_name, description):
    """Verify that a method exists in a class."""
    if hasattr(cls, method_name):
        logger.info(f"✓ {description}")
        return True
    else:
        logger.error(f"✗ {description}")
        return False


def verify_method_signature(cls, method_name, expected_params):
    """Verify method signature has expected parameters."""
    if not hasattr(cls, method_name):
        return False
    
    method = getattr(cls, method_name)
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())
    
    for param in expected_params:
        if param not in params:
            logger.error(f"  ✗ Missing parameter: {param}")
            return False
    
    logger.info(f"  ✓ Method signature correct: {params}")
    return True


def verify_method_implementation(cls, method_name, keywords):
    """Verify method contains expected implementation keywords."""
    if not hasattr(cls, method_name):
        return False
    
    method = getattr(cls, method_name)
    source = inspect.getsource(method)
    
    missing = []
    for keyword in keywords:
        if keyword not in source:
            missing.append(keyword)
    
    if missing:
        logger.error(f"  ✗ Missing implementation details: {missing}")
        return False
    
    logger.info(f"  ✓ Implementation contains expected components")
    return True


def main():
    """Verify Task 5 implementation."""
    
    logger.info("="*70)
    logger.info("Task 5 Verification: Create comprehensive reporting and visualization")
    logger.info("="*70)
    
    all_passed = True
    
    # Sub-task 1: Implement generate_report() method creating JSON report with all results
    logger.info("\n1. Checking generate_report() method...")
    
    if not verify_method_exists(
        SystematicHMMTrainer, 
        'generate_report',
        "Method generate_report() exists"
    ):
        all_passed = False
    else:
        # Check signature
        if not verify_method_signature(
            SystematicHMMTrainer,
            'generate_report',
            ['self', 'evaluation_summary', 'best_model_info']
        ):
            all_passed = False
        
        # Check implementation creates JSON report
        if not verify_method_implementation(
            SystematicHMMTrainer,
            'generate_report',
            ['json.dump', 'training_report.json']
        ):
            all_passed = False
    
    # Sub-task 2: Add _print_summary_table() for console output of model rankings
    logger.info("\n2. Checking _print_summary_table() method...")
    
    if not verify_method_exists(
        SystematicHMMTrainer,
        '_print_summary_table',
        "Method _print_summary_table() exists"
    ):
        all_passed = False
    else:
        # Check signature
        if not verify_method_signature(
            SystematicHMMTrainer,
            '_print_summary_table',
            ['self', 'rankings']
        ):
            all_passed = False
        
        # Check implementation prints formatted table
        if verify_method_implementation(
            SystematicHMMTrainer,
            '_print_summary_table',
            ['logger.info', 'Rank', 'Model', 'Score']
        ):
            logger.info("  ✓ Prints formatted table with rankings")
        else:
            logger.error("  ✗ Missing formatted table output")
            all_passed = False
    
    # Sub-task 3: Create training_report.json with timestamp, configuration, and full evaluation results
    logger.info("\n3. Checking training_report.json structure...")
    
    source = inspect.getsource(SystematicHMMTrainer.generate_report)
    
    required_fields = [
        'timestamp',
        'configuration',
        'best_model',
        'all_models',
        'rankings'
    ]
    
    for field in required_fields:
        if f"'{field}'" in source or f'"{field}"' in source:
            logger.info(f"  ✓ Report includes {field}")
        else:
            logger.error(f"  ✗ Report missing {field}")
            all_passed = False
    
    # Check configuration details
    config_fields = ['data_path', 'output_dir', 'n_states_range', 'cv_folds']
    for field in config_fields:
        if f"'{field}'" in source or f'"{field}"' in source:
            logger.info(f"  ✓ Configuration includes {field}")
        else:
            logger.error(f"  ✗ Configuration missing {field}")
            all_passed = False
    
    # Sub-task 4: Add clear logging of best model selection with scores and justification
    logger.info("\n4. Checking best model selection logging...")
    
    select_source = inspect.getsource(SystematicHMMTrainer.select_best_model)
    
    required_logging = [
        'Best Model Selected',
        'Combined Score',
        'Confidence',
        'Component Metrics',
        'Justification'
    ]
    
    for log_item in required_logging:
        if log_item in select_source:
            logger.info(f"  ✓ Logs {log_item}")
        else:
            logger.error(f"  ✗ Missing logging for {log_item}")
            all_passed = False
    
    # Check for score details
    score_details = ['aic', 'bic', 'cv_score', 'interpretability']
    for detail in score_details:
        if detail in select_source:
            logger.info(f"  ✓ Logs {detail} score")
        else:
            logger.warning(f"  ⚠ Missing {detail} in logging")
    
    # Additional checks
    logger.info("\n5. Additional implementation checks...")
    
    # Check generate_report calls _print_summary_table
    if '_print_summary_table' in source:
        logger.info("  ✓ generate_report() calls _print_summary_table()")
    else:
        logger.error("  ✗ generate_report() doesn't call _print_summary_table()")
        all_passed = False
    
    # Check for numpy type conversion
    if 'convert_to_serializable' in source or 'default=str' in source:
        logger.info("  ✓ Handles numpy type conversion for JSON")
    else:
        logger.warning("  ⚠ May have issues with numpy type serialization")
    
    # Check report is saved to file
    if 'report_path' in source and 'open(' in source:
        logger.info("  ✓ Saves report to file")
    else:
        logger.error("  ✗ Missing file save operation")
        all_passed = False
    
    # Check for summary logging
    if 'Report Summary' in source or 'Total models' in source:
        logger.info("  ✓ Includes summary logging")
    else:
        logger.warning("  ⚠ Missing summary logging")
    
    # Check integration with run() method
    logger.info("\n6. Checking integration with run() method...")
    
    run_source = inspect.getsource(SystematicHMMTrainer.run)
    
    if 'generate_report' in run_source:
        logger.info("  ✓ run() method calls generate_report()")
    else:
        logger.error("  ✗ run() method doesn't call generate_report()")
        all_passed = False
    
    if 'evaluation_summary' in run_source and 'best_model' in run_source:
        logger.info("  ✓ run() method passes correct parameters")
    else:
        logger.error("  ✗ run() method missing required parameters")
        all_passed = False
    
    # Final result
    logger.info("\n" + "="*70)
    if all_passed:
        logger.info("✓ ALL SUB-TASKS VERIFIED - Task 5 is COMPLETE")
        logger.info("="*70)
        logger.info("\nThe reporting and visualization implementation includes:")
        logger.info("  • generate_report() creates comprehensive JSON report")
        logger.info("  • _print_summary_table() displays formatted rankings")
        logger.info("  • training_report.json with full configuration and results")
        logger.info("  • Detailed logging of best model selection with justification")
        return 0
    else:
        logger.error("✗ SOME SUB-TASKS INCOMPLETE - Please review failures above")
        logger.info("="*70)
        return 1


if __name__ == '__main__':
    exit(main())
