# Backtesting Framework Design

## Overview

The backtesting framework is implemented as a Python package using vectorbt for high-performance vectorized backtesting. The system processes historical signals, applies realistic trading costs, and generates comprehensive performance reports. The design emphasizes modularity, performance, and integration with the existing IMP infrastructure.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Loader   │───▶│  Signal Processor │───▶│ Trade Generator │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Market Data     │    │ HMM Weights      │    │ Position Tracker│
│ (Parquet)       │    │ (MinIO/JSON)     │    │ (In-Memory)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Cost Calculator │◀───│ Performance      │◀───│ Trade Executor  │
└─────────────────┘    │ Analyzer         │    └─────────────────┘
         │              └──────────────────┘             │
         ▼                       │                       ▼
┌─────────────────┐              ▼              ┌─────────────────┐
│ Slippage Model  │    ┌──────────────────┐    │ Audit Logger    │
└─────────────────┘    │ Report Generator │    └─────────────────┘
                       └──────────────────┘
```

## Components and Interfaces

### BacktestEngine
**Purpose**: Main orchestrator for backtesting workflow
**Key Methods**:
- `run_backtest(signals_path, config)`: Execute complete backtest
- `load_data(date_range, symbols)`: Load historical data and signals
- `validate_inputs()`: Ensure data quality and completeness

**Interfaces**:
- Input: Parquet signal files, YAML configuration
- Output: BacktestResults object with performance metrics

### SignalProcessor
**Purpose**: Convert raw signals into actionable trading decisions
**Key Methods**:
- `process_signals(signal_df)`: Apply thresholds and filters
- `apply_regime_weights(signals, hmm_weights)`: Apply HMM-based weighting
- `generate_trade_signals()`: Create buy/sell/hold decisions

**Configuration**:
```python
signal_config = {
    'thresholds': {'ldc': 0.6, 'mr': 0.4, 'tsmom': 0.5},
    'filters': {'min_signal_strength': 0.3, 'max_trades_per_day': 10},
    'fusion_method': 'hmm_weighted'  # or 'static_weighted'
}
```

### TradeGenerator
**Purpose**: Convert signals into sized trades with risk management
**Key Methods**:
- `size_positions(signals, portfolio_value)`: Calculate position sizes
- `apply_risk_limits(trades)`: Enforce position and exposure limits
- `generate_orders()`: Create timestamped trade orders

**Position Sizing Options**:
- Fixed dollar amount
- Percentage of portfolio
- Volatility-adjusted (using ATR)
- Kelly criterion-based

### CostModel
**Purpose**: Apply realistic transaction costs and market impact
**Key Methods**:
- `calculate_commission(trade_value, symbol)`: Compute trading fees
- `apply_slippage(order, market_conditions)`: Model execution slippage
- `estimate_market_impact(order_size, volume)`: Calculate price impact

**Cost Structure**:
```python
cost_config = {
    'commission': {'rate': 0.001, 'min_fee': 1.0},  # 0.1% with $1 minimum
    'spread': {'crypto': 0.0005, 'forex': 0.00002},  # Half-spread costs
    'slippage': {'linear_impact': 0.0001, 'sqrt_impact': 0.001}
}
```

### PerformanceAnalyzer
**Purpose**: Calculate comprehensive performance metrics and attribution
**Key Methods**:
- `calculate_returns()`: Compute period and cumulative returns
- `risk_metrics()`: Calculate Sharpe, Sortino, max drawdown, VaR
- `regime_analysis()`: Performance breakdown by market regime
- `attribution_analysis()`: Return contribution by signal source

**Metrics Output**:
```python
performance_metrics = {
    'returns': {'total': 0.15, 'annualized': 0.12, 'volatility': 0.18},
    'risk': {'sharpe': 0.67, 'sortino': 0.89, 'max_dd': -0.08, 'var_95': -0.025},
    'regime': {'bull': 0.18, 'bear': -0.02, 'sideways': 0.08},
    'attribution': {'ldc': 0.06, 'mr': 0.04, 'tsmom': 0.05}
}
```

### WalkForwardValidator
**Purpose**: Implement rolling window validation with model retraining
**Key Methods**:
- `setup_windows(start_date, end_date, train_period, test_period)`: Define validation windows
- `retrain_models(training_data)`: Trigger HMM retraining for each window
- `validate_performance()`: Compare in-sample vs out-of-sample results

**Validation Configuration**:
```python
walkforward_config = {
    'train_period': '6M',  # 6 months training
    'test_period': '1M',   # 1 month testing
    'step_size': '2W',     # 2 week steps
    'min_train_samples': 1000,
    'retrain_threshold': 0.05  # Retrain if performance drops 5%
}
```

## Data Models

### Signal Data Schema
```python
@dataclass
class SignalData:
    timestamp: datetime
    symbol: str
    s_ldc: float
    s_mr: float
    s_tsmom: float
    s_fusion: float
    regime_state: int
    regime_weights: Dict[str, float]
    metadata: Dict[str, Any]
```

### Trade Record Schema
```python
@dataclass
class TradeRecord:
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy', 'sell', 'close'
    quantity: float
    price: float
    signal_source: str
    signal_strength: float
    commission: float
    slippage: float
    total_cost: float
```

### Portfolio State Schema
```python
@dataclass
class PortfolioState:
    timestamp: datetime
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    market_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
```

## Error Handling

### Data Quality Issues
- **Missing Data**: Use configurable fill strategies (forward fill, interpolation, skip)
- **Invalid Signals**: Log warnings and use fallback values or skip periods
- **Price Anomalies**: Detect and handle extreme price movements or gaps

### Model Failures
- **HMM Service Unavailable**: Fall back to static fusion weights
- **Retraining Failures**: Continue with previous model and log errors
- **Memory Issues**: Implement chunked processing for large datasets

### Configuration Errors
- **Invalid Parameters**: Validate all configuration at startup
- **Missing Files**: Provide clear error messages with suggested fixes
- **Version Mismatches**: Check model artifact versions and compatibility

## Testing Strategy

### Unit Tests
- **Signal Processing**: Test threshold application and filtering logic
- **Cost Calculations**: Verify commission and slippage calculations
- **Performance Metrics**: Validate metric calculations against known results
- **Position Sizing**: Test different sizing methods with edge cases

### Integration Tests
- **End-to-End Pipeline**: Test complete backtest workflow with sample data
- **Data Loading**: Test Parquet reading and MinIO integration
- **Walk-Forward Validation**: Test rolling window logic and retraining

### Performance Tests
- **Large Dataset Processing**: Test with 1+ years of 5-minute data
- **Memory Usage**: Monitor memory consumption during processing
- **Execution Speed**: Ensure backtests complete within target timeframes

### Validation Tests
- **Known Results**: Compare against manually calculated results
- **Benchmark Comparison**: Validate against simple buy-and-hold strategies
- **Cross-Validation**: Compare results across different time periods

## Implementation Notes

### Technology Stack
- **Core Framework**: vectorbt for vectorized backtesting operations
- **Data Processing**: polars for fast data manipulation
- **Visualization**: plotly for interactive performance charts
- **Storage**: Integration with existing MinIO/Parquet infrastructure
- **Configuration**: YAML files with pydantic validation

### Performance Optimizations
- **Vectorized Operations**: Use vectorbt's optimized backtesting engine
- **Lazy Loading**: Load data in chunks to manage memory usage
- **Parallel Processing**: Process multiple symbols concurrently
- **Caching**: Cache expensive calculations like regime weights

### Integration Points
- **Signal Data**: Read from existing Parquet signal files
- **HMM Models**: Load weights from MinIO-stored JSON artifacts
- **Market Data**: Use existing OHLCV data infrastructure
- **Reporting**: Generate reports compatible with existing analysis tools