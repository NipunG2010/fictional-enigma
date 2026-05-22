"""
Main backtesting engine orchestrator for IMP trading system.

This module implements the BacktestEngine class that orchestrates the complete
backtesting workflow, coordinating all components including data loading, signal
processing, trade generation, cost modeling, and performance analysis.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd
import numpy as np

from .config import BacktestConfig
from .data_loader import DataLoader, DataQualityReport
from .artifact_loader import ArtifactLoader
from .signal_processor import SignalProcessor, ProcessedSignal
from .trade_signal_generator import TradeSignalGenerator, TradeSignal
from .trade_generator import TradeGenerator, Order
from .portfolio_state import PortfolioState, PortfolioSnapshot
from .cost_model import CostModel, TradeCost
from .slippage_model import SlippageModelEngine
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .regime_performance import RegimePerformanceAnalyzer
from .walk_forward_validator import WalkForwardValidator, ValidationReport
from .report_generator import ReportGenerator


logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Container for complete backtest results."""
    config: BacktestConfig
    
    # Performance metrics
    performance_metrics: PerformanceMetrics
    regime_metrics: Optional[Dict[str, Any]] = None
    
    # Portfolio state
    portfolio_snapshots: List[PortfolioSnapshot] = field(default_factory=list)
    final_portfolio_value: float = 0.0
    
    # Trade data
    orders: List[Order] = field(default_factory=list)
    trade_costs: List[TradeCost] = field(default_factory=list)
    
    # Data quality
    data_quality_report: Optional[DataQualityReport] = None
    
    # Walk-forward validation
    validation_report: Optional[ValidationReport] = None
    
    # Execution metadata
    execution_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'config': self.config.model_dump(mode='json'),
            'performance_metrics': self.performance_metrics.to_dict(),
            'regime_metrics': self.regime_metrics,
            'final_portfolio_value': self.final_portfolio_value,
            'num_orders': len(self.orders),
            'num_snapshots': len(self.portfolio_snapshots),
            'data_quality_report': self.data_quality_report.__dict__ if self.data_quality_report else None,
            'validation_report': self.validation_report.to_dict() if self.validation_report else None,
            'execution_time': self.execution_time,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            **self.metadata
        }
    
    def save_to_json(self, output_path: Path) -> None:
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved backtest results to {output_path}")


class BacktestEngine:
    """
    Main orchestrator for backtesting workflow.
    
    This class coordinates all backtesting components to execute complete
    backtests with multi-symbol support, progress tracking, and comprehensive
    result generation.
    
    Requirements: 1.1, 1.2, 1.4
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize BacktestEngine with configuration.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        
        # Validate configuration
        self.config.validate_paths()
        
        # Initialize components
        self._initialize_components()
        
        # Track execution state
        self._current_backtest: Optional[BacktestResults] = None
        self._execution_start_time: Optional[datetime] = None
        
        logger.info(f"Initialized BacktestEngine: {config.name}")
        logger.info(f"  Symbols: {config.symbols}")
        logger.info(f"  Date range: {config.start_date} to {config.end_date}")
        logger.info(f"  Initial capital: ${config.initial_capital:,.2f}")
    
    def _initialize_components(self) -> None:
        """Initialize all backtesting components."""
        logger.info("Initializing backtesting components")
        
        # Data loader
        self.data_loader = DataLoader(self.config.data_source)
        
        # Artifact loader (for HMM weights)
        self.artifact_loader = ArtifactLoader(
            config=self.config.data_source
        )
        
        # Signal processor
        self.signal_processor = SignalProcessor(
            config=self.config.signal_processing,
            artifact_loader=self.artifact_loader
        )
        
        # Trade signal generator
        self.trade_signal_generator = TradeSignalGenerator(
            config=self.config.signal_processing
        )
        
        # Trade generator
        self.trade_generator = TradeGenerator(
            config=self.config.position_sizing,
            initial_capital=self.config.initial_capital
        )
        
        # Cost model
        self.cost_model = CostModel(
            config=self.config.cost_model
        )
        
        # Slippage model
        self.slippage_model = SlippageModelEngine(
            config=self.config.cost_model
        )
        
        # Performance analyzer
        self.performance_analyzer = PerformanceAnalyzer(
            config=self.config.performance
        )
        
        # Regime performance analyzer (if enabled)
        if self.config.performance.regime_analysis:
            self.regime_analyzer = RegimePerformanceAnalyzer(
                config=self.config.performance
            )
        else:
            self.regime_analyzer = None
        
        # Walk-forward validator (if enabled)
        if self.config.walk_forward.enabled:
            self.walk_forward_validator = WalkForwardValidator(
                config=self.config.walk_forward
            )
        else:
            self.walk_forward_validator = None
        
        # Report generator
        self.report_generator = ReportGenerator()
        
        logger.info("All components initialized successfully")
    
    def run_backtest(
        self,
        progress_callback: Optional[callable] = None
    ) -> BacktestResults:
        """
        Execute complete backtest workflow.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete backtest results
            
        Requirements: 1.1, 1.2, 1.4
        """
        self._execution_start_time = datetime.now()
        logger.info(f"Starting backtest: {self.config.name}")
        
        try:
            # Step 1: Load and validate data
            logger.info("Step 1/6: Loading and validating data")
            signal_df, market_df = self._load_data()
            
            if progress_callback:
                progress_callback(step=1, total=6, message="Data loaded")
            
            # Step 2: Validate data quality
            logger.info("Step 2/6: Validating data quality")
            data_quality_report = self.data_loader.validate_data_quality(signal_df, market_df)
            
            if data_quality_report.quality_score < 0.7:
                logger.warning(f"Low data quality score: {data_quality_report.quality_score:.2f}")
                for error in data_quality_report.errors:
                    logger.error(f"  Data quality error: {error}")
            
            if progress_callback:
                progress_callback(step=2, total=6, message="Data validated")
            
            # Step 3: Run backtest (single or multi-symbol)
            logger.info("Step 3/6: Running backtest simulation")
            
            if len(self.config.symbols) == 1:
                # Single symbol backtest
                portfolio_state, orders, trade_costs = self._run_single_symbol_backtest(
                    symbol=self.config.symbols[0],
                    signal_df=signal_df,
                    market_df=market_df,
                    progress_callback=progress_callback
                )
            else:
                # Multi-symbol backtest
                portfolio_state, orders, trade_costs = self._run_multi_symbol_backtest(
                    signal_df=signal_df,
                    market_df=market_df,
                    progress_callback=progress_callback
                )
            
            if progress_callback:
                progress_callback(step=3, total=6, message="Simulation complete")
            
            # Step 4: Calculate performance metrics
            logger.info("Step 4/6: Calculating performance metrics")
            performance_metrics = self.performance_analyzer.calculate_metrics(
                snapshots=portfolio_state.get_snapshots(),
                orders=orders,
                initial_capital=self.config.initial_capital
            )
            
            if progress_callback:
                progress_callback(step=4, total=6, message="Performance calculated")
            
            # Step 5: Regime-specific analysis (if enabled)
            regime_metrics = None
            if self.regime_analyzer:
                logger.info("Step 5/6: Performing regime-specific analysis")
                regime_metrics = self._perform_regime_analysis(
                    portfolio_state=portfolio_state,
                    signal_df=signal_df,
                    orders=orders
                )
            else:
                logger.info("Step 5/6: Skipping regime analysis (disabled)")
            
            if progress_callback:
                progress_callback(step=5, total=6, message="Regime analysis complete")
            
            # Step 6: Walk-forward validation (if enabled)
            validation_report = None
            if self.walk_forward_validator:
                logger.info("Step 6/6: Running walk-forward validation")
                validation_report = self._run_walk_forward_validation()
            else:
                logger.info("Step 6/6: Skipping walk-forward validation (disabled)")
            
            if progress_callback:
                progress_callback(step=6, total=6, message="Validation complete")
            
            # Create results object
            execution_end_time = datetime.now()
            execution_time = (execution_end_time - self._execution_start_time).total_seconds()
            
            results = BacktestResults(
                config=self.config,
                performance_metrics=performance_metrics,
                regime_metrics=regime_metrics,
                portfolio_snapshots=portfolio_state.get_snapshots(),
                final_portfolio_value=portfolio_state.total_value,
                orders=orders,
                trade_costs=trade_costs,
                data_quality_report=data_quality_report,
                validation_report=validation_report,
                execution_time=execution_time,
                start_time=self._execution_start_time,
                end_time=execution_end_time,
                metadata={
                    'num_symbols': len(self.config.symbols),
                    'num_periods': len(portfolio_state.get_snapshots()),
                    'data_quality_score': data_quality_report.quality_score
                }
            )
            
            self._current_backtest = results
            
            logger.info(f"Backtest complete in {execution_time:.2f} seconds")
            logger.info(f"  Final portfolio value: ${results.final_portfolio_value:,.2f}")
            logger.info(f"  Total return: {performance_metrics.total_return:.2%}")
            logger.info(f"  Sharpe ratio: {performance_metrics.sharpe_ratio:.2f}")
            logger.info(f"  Max drawdown: {performance_metrics.max_drawdown:.2%}")
            logger.info(f"  Number of trades: {len(orders)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            raise
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load signal and market data.
        
        Returns:
            Tuple of (signal_df, market_df)
        """
        logger.info(f"Loading data for symbols: {self.config.symbols}")
        
        # Load combined data with alignment
        signal_df, market_df = self.data_loader.load_combined_data(
            symbols=self.config.symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            validate=True
        )
        
        logger.info(f"Loaded {len(signal_df)} signal records and {len(market_df)} market records")
        
        return signal_df, market_df
    
    def _run_single_symbol_backtest(
        self,
        symbol: str,
        signal_df: pd.DataFrame,
        market_df: pd.DataFrame,
        progress_callback: Optional[callable] = None
    ) -> Tuple[PortfolioState, List[Order], List[TradeCost]]:
        """
        Run backtest for a single symbol.
        
        Requirements: 1.1, 1.2
        """
        logger.info(f"Running single-symbol backtest for {symbol}")
        
        # Filter data for symbol
        symbol_signals = signal_df[signal_df['symbol'] == symbol].copy()
        symbol_market = market_df[market_df['symbol'] == symbol].copy()
        
        # Initialize portfolio state
        portfolio_state = PortfolioState(
            initial_capital=self.config.initial_capital,
            start_time=datetime.combine(self.config.start_date, datetime.min.time())
        )
        
        all_orders = []
        all_trade_costs = []
        
        # Process signals chronologically
        timestamps = sorted(symbol_signals.index.unique())
        
        logger.info(f"Processing {len(timestamps)} time periods")
        
        for i, timestamp in enumerate(tqdm(timestamps, desc=f"Backtesting {symbol}")):
            # Get signals for this timestamp
            period_signals = symbol_signals.loc[[timestamp]]
            
            # Process signals
            processed_signals = self.signal_processor.process_signals(
                signal_df=period_signals,
                symbol=symbol,
                current_time=timestamp
            )
            
            if not processed_signals:
                continue
            
            # Generate trade signals
            trade_signals = self.trade_signal_generator.generate_trade_signals(
                processed_signals=processed_signals,
                current_positions=portfolio_state.position_quantities
            )
            
            if not trade_signals:
                continue
            
            # Get current prices
            period_market = symbol_market.loc[[timestamp]]
            if period_market.empty:
                continue
            
            current_prices = {symbol: period_market['close'].iloc[0]}
            
            # Generate orders
            orders = self.trade_generator.generate_orders(
                trade_signals=trade_signals,
                market_data=symbol_market,
                portfolio_value=portfolio_state.total_value,
                current_positions=portfolio_state.position_quantities,
                current_prices=current_prices
            )
            
            # Execute orders
            for order in orders:
                # Calculate costs
                trade_cost = self.cost_model.calculate_trade_cost(
                    order=order,
                    market_data=symbol_market,
                    volume=period_market['volume'].iloc[0] if 'volume' in period_market.columns else None
                )
                
                # Update portfolio
                portfolio_state.update_from_order(
                    order=order,
                    execution_price=trade_cost.execution_price,
                    commission=trade_cost.commission,
                    slippage=trade_cost.slippage
                )
                
                all_orders.append(order)
                all_trade_costs.append(trade_cost)
            
            # Update prices and take snapshot
            portfolio_state.update_prices(current_prices, timestamp)
            portfolio_state.take_snapshot(timestamp)
            
            # Progress callback
            if progress_callback and i % 100 == 0:
                progress = (i + 1) / len(timestamps)
                progress_callback(
                    step=3,
                    total=6,
                    message=f"Processing {symbol}: {progress:.1%}",
                    progress=progress
                )
        
        logger.info(f"Completed backtest for {symbol}: {len(all_orders)} orders executed")
        
        return portfolio_state, all_orders, all_trade_costs
    
    def _run_multi_symbol_backtest(
        self,
        signal_df: pd.DataFrame,
        market_df: pd.DataFrame,
        progress_callback: Optional[callable] = None
    ) -> Tuple[PortfolioState, List[Order], List[TradeCost]]:
        """
        Run backtest for multiple symbols with independent position tracking.
        
        Requirements: 1.2, 1.3
        """
        logger.info(f"Running multi-symbol backtest for {len(self.config.symbols)} symbols")
        
        # Initialize portfolio state
        portfolio_state = PortfolioState(
            initial_capital=self.config.initial_capital,
            start_time=datetime.combine(self.config.start_date, datetime.min.time())
        )
        
        all_orders = []
        all_trade_costs = []
        
        # Get all unique timestamps across all symbols
        timestamps = sorted(signal_df.index.unique())
        
        logger.info(f"Processing {len(timestamps)} time periods across {len(self.config.symbols)} symbols")
        
        for i, timestamp in enumerate(tqdm(timestamps, desc="Backtesting")):
            # Process all symbols for this timestamp
            period_signals = signal_df.loc[[timestamp]]
            period_market = market_df.loc[[timestamp]]
            
            if period_signals.empty or period_market.empty:
                continue
            
            # Collect all trade signals for this period
            all_trade_signals = []
            
            for symbol in self.config.symbols:
                symbol_signals = period_signals[period_signals['symbol'] == symbol]
                
                if symbol_signals.empty:
                    continue
                
                # Process signals for this symbol
                processed_signals = self.signal_processor.process_signals(
                    signal_df=symbol_signals,
                    symbol=symbol,
                    current_time=timestamp
                )
                
                if not processed_signals:
                    continue
                
                # Generate trade signals
                trade_signals = self.trade_signal_generator.generate_trade_signals(
                    processed_signals=processed_signals,
                    current_positions=portfolio_state.position_quantities
                )
                
                all_trade_signals.extend(trade_signals)
            
            if not all_trade_signals:
                # No trades, but still update prices and take snapshot
                current_prices = {}
                for symbol in self.config.symbols:
                    symbol_market = period_market[period_market['symbol'] == symbol]
                    if not symbol_market.empty:
                        current_prices[symbol] = symbol_market['close'].iloc[0]
                
                if current_prices:
                    portfolio_state.update_prices(current_prices, timestamp)
                    portfolio_state.take_snapshot(timestamp)
                
                continue
            
            # Get current prices for all symbols
            current_prices = {}
            for symbol in self.config.symbols:
                symbol_market = period_market[period_market['symbol'] == symbol]
                if not symbol_market.empty:
                    current_prices[symbol] = symbol_market['close'].iloc[0]
            
            # Generate orders for all trade signals
            orders = self.trade_generator.generate_orders(
                trade_signals=all_trade_signals,
                market_data=market_df,
                portfolio_value=portfolio_state.total_value,
                current_positions=portfolio_state.position_quantities,
                current_prices=current_prices
            )
            
            # Execute orders
            for order in orders:
                # Get volume for cost calculation
                symbol_market = period_market[period_market['symbol'] == order.symbol]
                volume = symbol_market['volume'].iloc[0] if not symbol_market.empty and 'volume' in symbol_market.columns else None
                
                # Calculate costs
                trade_cost = self.cost_model.calculate_trade_cost(
                    order=order,
                    market_data=market_df,
                    volume=volume
                )
                
                # Update portfolio
                portfolio_state.update_from_order(
                    order=order,
                    execution_price=trade_cost.execution_price,
                    commission=trade_cost.commission,
                    slippage=trade_cost.slippage
                )
                
                all_orders.append(order)
                all_trade_costs.append(trade_cost)
            
            # Update prices and take snapshot
            portfolio_state.update_prices(current_prices, timestamp)
            portfolio_state.take_snapshot(timestamp)
            
            # Progress callback
            if progress_callback and i % 100 == 0:
                progress = (i + 1) / len(timestamps)
                progress_callback(
                    step=3,
                    total=6,
                    message=f"Processing: {progress:.1%}",
                    progress=progress
                )
        
        logger.info(f"Completed multi-symbol backtest: {len(all_orders)} orders executed")
        
        return portfolio_state, all_orders, all_trade_costs
    
    def _perform_regime_analysis(
        self,
        portfolio_state: PortfolioState,
        signal_df: pd.DataFrame,
        orders: List[Order]
    ) -> Dict[str, Any]:
        """
        Perform regime-specific performance analysis.
        
        Requirements: 4.3, 4.4
        """
        if not self.regime_analyzer:
            return {}
        
        logger.info("Performing regime-specific analysis")
        
        # Analyze performance by regime
        regime_metrics = self.regime_analyzer.analyze_by_regime(
            snapshots=portfolio_state.get_snapshots(),
            signal_df=signal_df,
            orders=orders
        )
        
        # Analyze signal attribution
        attribution = self.regime_analyzer.analyze_signal_attribution(
            orders=orders,
            snapshots=portfolio_state.get_snapshots()
        )
        
        return {
            'regime_metrics': regime_metrics,
            'signal_attribution': attribution
        }
    
    def _run_walk_forward_validation(self) -> ValidationReport:
        """
        Run walk-forward validation.
        
        Requirements: 5.1, 5.2, 5.3, 5.4
        """
        if not self.walk_forward_validator:
            raise ValueError("Walk-forward validator not initialized")
        
        logger.info("Running walk-forward validation")
        
        # Setup validation windows
        self.walk_forward_validator.setup_windows(
            start_date=datetime.combine(self.config.start_date, datetime.min.time()),
            end_date=datetime.combine(self.config.end_date, datetime.min.time())
        )
        
        # Define backtest function for validation
        def backtest_window(start_date: datetime, end_date: datetime) -> Tuple[List[PortfolioSnapshot], List[Order]]:
            # Create temporary config for this window
            window_config = self.config.model_copy(deep=True)
            window_config.start_date = start_date.date()
            window_config.end_date = end_date.date()
            window_config.walk_forward.enabled = False  # Disable nested validation
            
            # Create temporary engine
            temp_engine = BacktestEngine(window_config)
            
            # Run backtest
            results = temp_engine.run_backtest()
            
            return results.portfolio_snapshots, results.orders
        
        # Run validation
        validation_report = self.walk_forward_validator.run_validation(
            backtest_func=backtest_window,
            performance_analyzer=self.performance_analyzer,
            initial_capital=self.config.initial_capital
        )
        
        return validation_report
    
    def get_results(self) -> Optional[BacktestResults]:
        """Get the most recent backtest results."""
        return self._current_backtest
    
    def save_results(
        self,
        output_dir: Optional[Path] = None,
        formats: List[str] = ['json', 'csv']
    ) -> Dict[str, Path]:
        """
        Save backtest results to files.
        
        Args:
            output_dir: Output directory (uses config.output_path if None)
            formats: List of output formats ('json', 'csv', 'html')
            
        Returns:
            Dictionary mapping format to output file path
            
        Requirements: 1.5
        """
        if self._current_backtest is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        output_dir = output_dir or self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save main results as JSON using report generator
        if 'json' in formats:
            json_path = output_dir / f"{self.config.name}_results.json"
            self.report_generator.generate_json_report(self._current_backtest, json_path)
            saved_files['json'] = json_path
        
        # Save CSV reports using report generator
        if 'csv' in formats:
            csv_files = self.report_generator.generate_csv_reports(self._current_backtest, output_dir)
            saved_files.update(csv_files)
        
        # Save HTML report using report generator
        if 'html' in formats and self.config.generate_report:
            html_path = output_dir / f"{self.config.name}_report.html"
            self.report_generator.generate_html_report(
                self._current_backtest,
                html_path,
                include_charts=True
            )
            saved_files['html'] = html_path
        
        logger.info(f"Saved backtest results to {output_dir}")
        for format_name, file_path in saved_files.items():
            logger.info(f"  {format_name}: {file_path}")
        
        return saved_files

