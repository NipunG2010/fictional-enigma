"""
Backtesting Framework for IMP Trading System

This package provides comprehensive backtesting capabilities for validating
trading strategies using historical data. It includes:

- Signal processing and trade generation
- Position sizing and risk management
- Transaction cost and slippage modeling
- Performance analysis and metrics calculation
- Walk-forward validation framework
- Integration with existing IMP infrastructure

Key Components:
- BacktestEngine: Main orchestrator for backtesting workflow
- SignalProcessor: Convert raw signals into actionable trades
- TradeGenerator: Generate sized trades with risk management
- CostModel: Apply realistic transaction costs and slippage
- PerformanceAnalyzer: Calculate comprehensive performance metrics
- WalkForwardValidator: Rolling window validation with retraining

Example Usage:
    from imp.backtesting import BacktestEngine
    from imp.backtesting.config import BacktestConfig
    
    config = BacktestConfig.from_yaml("backtest_config.yaml")
    engine = BacktestEngine(config)
    results = engine.run_backtest(signals_path="signals.parquet")
"""

from .config import (
    BacktestConfig,
    DataSourceConfig,
    SignalProcessingConfig,
    PositionSizingConfig,
    CostModelConfig,
    PerformanceConfig,
    WalkForwardConfig,
    PositionSizingMethod,
    FusionMethod,
    CostStructureType,
    create_crypto_config,
    create_forex_config,
)

from .data_loader import (
    DataLoader,
    DataQualityReport,
    DataValidationError,
)

from .artifact_loader import (
    ArtifactLoader,
    ArtifactCache,
    ArtifactLoadError,
    MinIOConnectionError,
)

from .signal_processor import (
    SignalProcessor,
    ProcessedSignal,
    SignalDirection,
    SignalConflict,
)

from .trade_signal_generator import (
    TradeSignalGenerator,
    TradeSignal,
    TradeAction,
    TradingState,
)

from .trade_generator import (
    TradeGenerator,
    Order,
)

from .portfolio_state import (
    PortfolioState,
    Position,
    PortfolioSnapshot,
)

from .cost_model import (
    CostModel,
    TradeCost,
)

from .slippage_model import (
    SlippageModelEngine,
    SlippageEstimate,
    SlippageModel,
)

from .performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    BenchmarkComparison,
)

from .regime_performance import (
    RegimePerformanceAnalyzer,
    RegimeMetrics,
    SignalAttribution,
    TimeSeriesPerformance,
)

from .walk_forward_validator import (
    WalkForwardValidator,
    ValidationWindow,
    ValidationReport,
)

from .validation_reporting import (
    ValidationReportGenerator,
)

from .backtest_engine import (
    BacktestEngine,
    BacktestResults,
)

from .report_generator import (
    ReportGenerator,
)

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "BacktestConfig",
    "DataSourceConfig", 
    "SignalProcessingConfig",
    "PositionSizingConfig",
    "CostModelConfig",
    "PerformanceConfig",
    "WalkForwardConfig",
    "PositionSizingMethod",
    "FusionMethod", 
    "CostStructureType",
    "create_crypto_config",
    "create_forex_config",
    
    # Data Loading
    "DataLoader",
    "DataQualityReport",
    "DataValidationError",
    
    # Artifact Loading
    "ArtifactLoader",
    "ArtifactCache",
    "ArtifactLoadError",
    "MinIOConnectionError",
    
    # Signal Processing
    "SignalProcessor",
    "ProcessedSignal",
    "SignalDirection",
    "SignalConflict",
    
    # Trade Signal Generation
    "TradeSignalGenerator",
    "TradeSignal",
    "TradeAction",
    "TradingState",
    
    # Trade Generation
    "TradeGenerator",
    "Order",
    
    # Portfolio State
    "PortfolioState",
    "Position",
    "PortfolioSnapshot",
    
    # Cost Model
    "CostModel",
    "TradeCost",
    
    # Slippage Model
    "SlippageModelEngine",
    "SlippageEstimate",
    "SlippageModel",
    
    # Performance Analysis
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "BenchmarkComparison",
    
    # Regime Performance
    "RegimePerformanceAnalyzer",
    "RegimeMetrics",
    "SignalAttribution",
    "TimeSeriesPerformance",
    
    # Walk-Forward Validation
    "WalkForwardValidator",
    "ValidationWindow",
    "ValidationReport",
    "ValidationReportGenerator",
    
    # Backtest Engine
    "BacktestEngine",
    "BacktestResults",
    
    # Report Generator
    "ReportGenerator",
]