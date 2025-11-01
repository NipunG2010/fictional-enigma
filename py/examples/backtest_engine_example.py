"""
Example usage of the BacktestEngine for running comprehensive backtests.

This script demonstrates how to:
1. Load configuration from YAML
2. Run a complete backtest
3. Save results in multiple formats
4. Generate comprehensive reports
"""

import logging
from pathlib import Path
from datetime import date

from imp.backtesting import (
    BacktestEngine,
    BacktestConfig,
    create_crypto_config,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_from_yaml():
    """Example: Load configuration from YAML and run backtest."""
    logger.info("=" * 60)
    logger.info("Example 1: Running backtest from YAML configuration")
    logger.info("=" * 60)
    
    # Load configuration from YAML file
    config_path = Path("py/imp/backtesting/example_config.yaml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create the configuration file or adjust the path")
        return
    
    try:
        config = BacktestConfig.from_yaml(config_path)
        
        # Validate configuration
        validation_result = config.validate_configuration()
        
        if not validation_result['is_valid']:
            logger.error("Configuration validation failed:")
            for error in validation_result['errors']:
                logger.error(f"  - {error}")
            return
        
        if validation_result['warnings']:
            logger.warning("Configuration warnings:")
            for warning in validation_result['warnings']:
                logger.warning(f"  - {warning}")
        
        # Print configuration summary
        logger.info("\n" + config.get_summary())
        
        # Create backtest engine
        engine = BacktestEngine(config)
        
        # Run backtest with progress callback
        def progress_callback(step, total, message, progress=None):
            if progress:
                logger.info(f"[{step}/{total}] {message} - {progress:.1%}")
            else:
                logger.info(f"[{step}/{total}] {message}")
        
        results = engine.run_backtest(progress_callback=progress_callback)
        
        # Print results summary
        logger.info("\n" + "=" * 60)
        logger.info("Backtest Results Summary")
        logger.info("=" * 60)
        logger.info(f"Final Portfolio Value: ${results.final_portfolio_value:,.2f}")
        logger.info(f"Total Return: {results.performance_metrics.total_return:.2%}")
        logger.info(f"Annualized Return: {results.performance_metrics.annualized_return:.2%}")
        logger.info(f"Sharpe Ratio: {results.performance_metrics.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.performance_metrics.max_drawdown:.2%}")
        logger.info(f"Number of Trades: {results.performance_metrics.num_trades}")
        logger.info(f"Win Rate: {results.performance_metrics.win_rate:.2%}")
        logger.info(f"Execution Time: {results.execution_time:.2f} seconds")
        
        # Save results in multiple formats
        logger.info("\nSaving results...")
        saved_files = engine.save_results(formats=['json', 'csv', 'html'])
        
        logger.info("Results saved to:")
        for format_name, file_path in saved_files.items():
            logger.info(f"  {format_name}: {file_path}")
        
        logger.info("\nBacktest complete!")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)


def example_programmatic_config():
    """Example: Create configuration programmatically and run backtest."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Running backtest with programmatic configuration")
    logger.info("=" * 60)
    
    # Create configuration programmatically
    config = create_crypto_config(
        symbols=["BTCUSDT"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        signals_path=Path("./data/signals"),
        market_data_path=Path("./data/market")
    )
    
    # Customize configuration
    config.name = "programmatic_backtest"
    config.description = "Backtest created programmatically"
    config.initial_capital = 50000.0
    config.position_sizing.percentage = 0.03  # 3% per trade
    
    # Print configuration
    logger.info("\n" + config.get_summary())
    
    # Validate paths exist before running
    try:
        config.validate_paths()
    except FileNotFoundError as e:
        logger.error(f"Path validation failed: {e}")
        logger.info("Please ensure data paths exist before running backtest")
        return
    
    # Create and run backtest
    try:
        engine = BacktestEngine(config)
        results = engine.run_backtest()
        
        # Print summary
        logger.info(f"\nFinal Value: ${results.final_portfolio_value:,.2f}")
        logger.info(f"Return: {results.performance_metrics.total_return:.2%}")
        logger.info(f"Sharpe: {results.performance_metrics.sharpe_ratio:.2f}")
        
        # Save results
        engine.save_results(formats=['json', 'csv'])
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)


def example_multi_symbol_backtest():
    """Example: Run backtest with multiple symbols."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Multi-symbol backtest")
    logger.info("=" * 60)
    
    # Create configuration for multiple symbols
    config = create_crypto_config(
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        signals_path=Path("./data/signals"),
        market_data_path=Path("./data/market")
    )
    
    config.name = "multi_symbol_backtest"
    config.description = "Backtest with multiple cryptocurrency pairs"
    
    # Adjust position sizing for multiple symbols
    config.position_sizing.percentage = 0.015  # 1.5% per trade
    config.position_sizing.max_total_exposure = 0.8  # 80% max exposure
    
    logger.info(f"Backtesting {len(config.symbols)} symbols: {', '.join(config.symbols)}")
    
    try:
        config.validate_paths()
        
        engine = BacktestEngine(config)
        results = engine.run_backtest()
        
        logger.info(f"\nMulti-symbol backtest complete:")
        logger.info(f"  Total Return: {results.performance_metrics.total_return:.2%}")
        logger.info(f"  Sharpe Ratio: {results.performance_metrics.sharpe_ratio:.2f}")
        logger.info(f"  Total Trades: {results.performance_metrics.num_trades}")
        
        # Save results
        engine.save_results(formats=['json', 'csv', 'html'])
        
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)


def example_with_walk_forward_validation():
    """Example: Run backtest with walk-forward validation."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Backtest with walk-forward validation")
    logger.info("=" * 60)
    
    # Load base configuration
    config_path = Path("py/imp/backtesting/example_config.yaml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    try:
        config = BacktestConfig.from_yaml(config_path)
        
        # Enable walk-forward validation
        config.walk_forward.enabled = True
        config.walk_forward.train_period = "3M"  # 3 months training
        config.walk_forward.test_period = "1M"   # 1 month testing
        config.walk_forward.step_size = "2W"     # 2 week steps
        
        logger.info("Walk-forward validation enabled:")
        logger.info(f"  Train period: {config.walk_forward.train_period}")
        logger.info(f"  Test period: {config.walk_forward.test_period}")
        logger.info(f"  Step size: {config.walk_forward.step_size}")
        
        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run_backtest()
        
        # Print validation results
        if results.validation_report:
            logger.info("\nWalk-forward validation results:")
            logger.info(f"  Number of windows: {len(results.validation_report.windows)}")
            logger.info(f"  Avg in-sample return: {results.validation_report.avg_in_sample_return:.2%}")
            logger.info(f"  Avg out-of-sample return: {results.validation_report.avg_out_of_sample_return:.2%}")
            logger.info(f"  Performance degradation: {results.validation_report.performance_degradation:.2%}")
            logger.info(f"  Stability score: {results.validation_report.stability_score:.2f}")
            logger.info(f"  Number of retrains: {results.validation_report.num_retrains}")
        
        # Save results
        engine.save_results(formats=['json', 'csv', 'html'])
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("IMP Backtesting Framework - Examples")
    logger.info("=" * 60)
    
    # Run examples
    # Note: These examples require actual data files to exist
    # Uncomment the examples you want to run
    
    # Example 1: Load from YAML
    # example_from_yaml()
    
    # Example 2: Programmatic configuration
    # example_programmatic_config()
    
    # Example 3: Multi-symbol backtest
    # example_multi_symbol_backtest()
    
    # Example 4: Walk-forward validation
    # example_with_walk_forward_validation()
    
    logger.info("\nTo run these examples, uncomment the desired example function")
    logger.info("and ensure the required data files exist in the specified paths.")
