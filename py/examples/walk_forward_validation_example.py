"""
Example demonstrating walk-forward validation framework.

This example shows how to:
1. Set up walk-forward validation windows
2. Run validation with model retraining
3. Generate comprehensive validation reports
4. Analyze in-sample vs out-of-sample performance
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

from imp.backtesting import (
    WalkForwardValidator,
    ValidationReportGenerator,
    PerformanceAnalyzer,
    PortfolioSnapshot,
    Order,
)
from imp.backtesting.config import WalkForwardConfig, PerformanceConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mock_backtest_function(
    start_date: datetime,
    end_date: datetime
) -> Tuple[List[PortfolioSnapshot], List[Order]]:
    """
    Mock backtest function for demonstration.
    
    In a real implementation, this would:
    1. Load signal data for the date range
    2. Run the backtesting engine
    3. Return portfolio snapshots and orders
    """
    logger.info(f"Running backtest from {start_date} to {end_date}")
    
    # Generate mock data
    num_days = (end_date - start_date).days
    num_snapshots = num_days * 288  # 5-minute bars
    
    snapshots = []
    orders = []
    
    initial_capital = 100000.0
    current_value = initial_capital
    
    # Simulate random walk with drift
    np.random.seed(hash(start_date) % 2**32)  # Deterministic but different per window
    daily_return = np.random.normal(0.0005, 0.02, num_days)
    
    for day in range(num_days):
        date = start_date + timedelta(days=day)
        
        # Apply daily return
        current_value *= (1 + daily_return[day])
        
        # Create snapshot
        snapshot = PortfolioSnapshot(
            timestamp=date,
            cash=current_value * 0.3,
            positions={},  # Empty positions for mock data
            total_value=current_value,
            unrealized_pnl=(current_value - initial_capital) * 0.7,
            realized_pnl=(current_value - initial_capital) * 0.3,
            total_pnl=current_value - initial_capital
        )
        snapshots.append(snapshot)
        
        # Generate some mock orders
        if day % 5 == 0:  # Trade every 5 days
            # Create required imports
            from imp.backtesting.trade_signal_generator import TradeSignal, TradeAction
            from imp.backtesting.signal_processor import ProcessedSignal, SignalDirection
            
            # Create a mock processed signal first
            mock_processed_signal = ProcessedSignal(
                timestamp=date,
                symbol="BTCUSDT",
                direction=SignalDirection.BUY if day % 10 == 0 else SignalDirection.SELL,
                strength=0.7,
                signal_source="ldc",
                raw_signals={"ldc": 0.7, "mr": 0.2, "tsmom": 0.1},
                regime_state=1,
                regime_weights={"ldc": 0.6, "mr": 0.2, "tsmom": 0.2},
                metadata={"source": "mock"}
            )
            
            # Create a mock trade signal
            mock_trade_signal = TradeSignal(
                trade_id=f"trade_{day}",
                timestamp=date,
                symbol="BTCUSDT",
                action=TradeAction.BUY if day % 10 == 0 else TradeAction.SELL,
                confidence=0.8,
                signal_source="ldc",
                reasoning="Mock signal for testing",
                processed_signal=mock_processed_signal,
                metadata={"source": "mock"}
            )
            
            order = Order(
                order_id=f"order_{day}",
                timestamp=date,
                symbol="BTCUSDT",
                side="buy" if day % 10 == 0 else "sell",
                quantity=0.1,
                price=50000.0 * (1 + daily_return[day]),
                order_type="market",
                trade_signal=mock_trade_signal,
                position_size_method="fixed_size",
                risk_checks_passed={"position_limit": True, "exposure_limit": True},
                metadata={}
            )
            orders.append(order)
    
    return snapshots, orders


def mock_retrain_function(
    train_start: datetime,
    train_end: datetime
) -> str:
    """
    Mock model retraining function.
    
    In a real implementation, this would:
    1. Load training data
    2. Retrain HMM or other models
    3. Save model artifacts
    4. Return model version identifier
    """
    logger.info(f"Retraining model on data from {train_start} to {train_end}")
    
    # Generate mock model version
    model_version = f"model_{train_start.strftime('%Y%m%d')}_{train_end.strftime('%Y%m%d')}"
    
    return model_version


def main():
    """Run walk-forward validation example."""
    logger.info("=" * 80)
    logger.info("Walk-Forward Validation Example")
    logger.info("=" * 80)
    
    # 1. Configure walk-forward validation
    logger.info("\n1. Configuring walk-forward validation")
    
    wf_config = WalkForwardConfig(
        enabled=True,
        train_period="6M",      # 6 months training
        test_period="1M",       # 1 month testing
        step_size="2W",         # 2 week steps
        min_train_samples=1000,
        retrain_threshold=0.05  # Retrain if performance drops 5%
    )
    
    logger.info(f"Configuration: train={wf_config.train_period}, "
               f"test={wf_config.test_period}, step={wf_config.step_size}")
    
    # 2. Initialize validator
    logger.info("\n2. Initializing walk-forward validator")
    
    validator = WalkForwardValidator(wf_config)
    
    # 3. Set up validation windows
    logger.info("\n3. Setting up validation windows")
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    windows = validator.setup_windows(start_date, end_date)
    
    logger.info(f"Created {len(windows)} validation windows")
    for i, window in enumerate(windows[:3]):  # Show first 3
        logger.info(f"  Window {i}: train=[{window.train_start.date()}, {window.train_end.date()}], "
                   f"test=[{window.test_start.date()}, {window.test_end.date()}]")
    if len(windows) > 3:
        logger.info(f"  ... and {len(windows) - 3} more windows")
    
    # 4. Validate temporal separation
    logger.info("\n4. Validating temporal separation")
    
    # Create mock data for validation
    train_data = pd.DataFrame({
        'timestamp': pd.date_range(start_date, start_date + timedelta(days=180), freq='5min')
    })
    test_data = pd.DataFrame({
        'timestamp': pd.date_range(start_date + timedelta(days=181), 
                                   start_date + timedelta(days=210), freq='5min')
    })
    
    is_valid = validator.validate_temporal_separation(train_data, test_data)
    logger.info(f"Temporal separation valid: {is_valid}")
    
    # 5. Run walk-forward validation
    logger.info("\n5. Running walk-forward validation")
    
    performance_analyzer = PerformanceAnalyzer(PerformanceConfig())
    
    validation_report = validator.run_validation(
        backtest_func=mock_backtest_function,
        retrain_func=mock_retrain_function,
        performance_analyzer=performance_analyzer,
        initial_capital=100000.0
    )
    
    logger.info(f"\nValidation complete:")
    logger.info(f"  Total windows: {len(validation_report.windows)}")
    logger.info(f"  Avg in-sample return: {validation_report.avg_in_sample_return:.2%}")
    logger.info(f"  Avg out-of-sample return: {validation_report.avg_out_of_sample_return:.2%}")
    logger.info(f"  Performance degradation: {validation_report.performance_degradation:.2%}")
    logger.info(f"  Stability score: {validation_report.stability_score:.2f}")
    logger.info(f"  Number of retrains: {validation_report.num_retrains}")
    
    # 6. Generate comprehensive report
    logger.info("\n6. Generating validation report")
    
    output_dir = Path("./validation_output")
    report_generator = ValidationReportGenerator(output_dir)
    
    comparison_report = report_generator.generate_comparison_report(
        validation_report,
        include_visualizations=True
    )
    
    logger.info(f"Generated comparison report with {len(comparison_report['recommendations'])} recommendations")
    
    # Display recommendations
    if comparison_report['recommendations']:
        logger.info("\nTop Recommendations:")
        for i, rec in enumerate(comparison_report['recommendations'][:3], 1):
            logger.info(f"\n  {i}. [{rec['priority'].upper()}] {rec['category'].title()}")
            logger.info(f"     Issue: {rec['issue']}")
            logger.info(f"     Recommendation: {rec['recommendation']}")
            logger.info(f"     Action: {rec['action']}")
    
    # 7. Export results
    logger.info("\n7. Exporting results")
    
    # Save validation report
    report_path = output_dir / "validation_report.json"
    validation_report.save_to_json(report_path)
    logger.info(f"Saved validation report to {report_path}")
    
    # Export to CSV
    csv_path = report_generator.export_to_csv(validation_report)
    if csv_path:
        logger.info(f"Exported results to {csv_path}")
    
    # 8. Statistical analysis
    logger.info("\n8. Statistical Analysis")
    
    statistical_tests = comparison_report['statistical_tests']
    if 'returns' in statistical_tests:
        returns_test = statistical_tests['returns']
        logger.info(f"\nReturns Analysis:")
        logger.info(f"  In-sample mean: {returns_test['descriptive_stats']['in_sample_mean']:.2%}")
        logger.info(f"  Out-of-sample mean: {returns_test['descriptive_stats']['out_of_sample_mean']:.2%}")
        logger.info(f"  Difference: {returns_test['descriptive_stats']['difference']:.2%}")
        logger.info(f"  T-test p-value: {returns_test['paired_ttest']['p_value']:.4f}")
        logger.info(f"  Significant at 0.05: {returns_test['paired_ttest']['significant_at_0.05']}")
        logger.info(f"  Effect size (Cohen's d): {returns_test['effect_size']['cohens_d']:.3f} "
                   f"({returns_test['effect_size']['interpretation']})")
    
    logger.info("\n" + "=" * 80)
    logger.info("Walk-forward validation example complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
