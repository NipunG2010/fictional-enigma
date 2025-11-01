# IMP Backtesting Framework

Comprehensive backtesting framework for validating the IMP trading system using historical data.

## Overview

The backtesting framework provides a complete solution for simulating trading performance with:

- **Signal Processing**: Convert raw signals into actionable trades with regime-aware fusion
- **Position Sizing**: Multiple methods including fixed size, percentage, volatility-adjusted, and Kelly criterion
- **Cost Modeling**: Realistic transaction costs including commissions, spreads, and market impact
- **Performance Analysis**: Comprehensive metrics including Sharpe ratio, drawdown, VaR, and more
- **Walk-Forward Validation**: Rolling window validation with model retraining
- **Multi-Symbol Support**: Independent position tracking across multiple symbols
- **Comprehensive Reporting**: JSON, CSV, and HTML reports with interactive charts

## Quick Start

### 1. Install Dependencies

```bash
cd py
pip install -e ".[dev]"
```

### 2. Create Configuration

Create a YAML configuration file (see `example_config.yaml`):

```yaml
name: "my_backtest"
start_date: "2024-01-01"
end_date: "2024-03-31"
symbols: ["BTCUSDT"]
initial_capital: 100000.0

data_source:
  signals_path: "./data/signals"
  market_data_path: "./data/market"

position_sizing:
  method: "percentage"
  percentage: 0.02  # 2% per trade
```

### 3. Run Backtest

```python
from imp.backtesting import BacktestEngine, BacktestConfig

# Load configuration
config = BacktestConfig.from_yaml("backtest_config.yaml")

# Create engine and run backtest
engine = BacktestEngine(config)
results = engine.run_backtest()

# Save results
engine.save_results(formats=['json', 'csv', 'html'])

# Print summary
print(f"Total Return: {results.performance_metrics.total_return:.2%}")
print(f"Sharpe Ratio: {results.performance_metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.performance_metrics.max_drawdown:.2%}")
```

## Architecture

### Core Components

1. **BacktestEngine**: Main orchestrator coordinating all components
2. **DataLoader**: Loads and validates signal and market data
3. **SignalProcessor**: Processes signals with regime-aware fusion
4. **TradeGenerator**: Generates sized orders with risk management
5. **CostModel**: Applies realistic transaction costs
6. **PerformanceAnalyzer**: Calculates comprehensive metrics
7. **ReportGenerator**: Creates detailed reports in multiple formats

### Data Flow

```
Signal Data → SignalProcessor → TradeSignalGenerator → TradeGenerator
                                                              ↓
Market Data → DataLoader → PortfolioState ← CostModel ← Orders
                                  ↓
                          PerformanceAnalyzer → Results → ReportGenerator
```

## Configuration

### Position Sizing Methods

- **fixed_size**: Fixed dollar amount per trade
- **percentage**: Percentage of portfolio per trade
- **volatility_adjusted**: Size based on target volatility (ATR)
- **kelly_criterion**: Kelly criterion with fractional sizing

### Cost Model

Supports different asset classes with appropriate cost structures:

- **crypto**: Higher spreads, moderate commissions
- **forex**: Tight spreads, low commissions
- **equity**: Moderate spreads and commissions
- **futures**: Variable spreads, exchange fees

### Signal Processing

- **HMM-weighted fusion**: Dynamic weights based on market regime
- **Static-weighted fusion**: Fixed weights for signal combination
- Configurable thresholds and filters
- Trade frequency limits and cooldown periods

## Performance Metrics

### Return Metrics
- Total return, annualized return, cumulative return
- Period-by-period returns

### Risk Metrics
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown, average drawdown
- Value at Risk (VaR), Conditional VaR (CVaR)
- Volatility (annualized)

### Trade Statistics
- Win rate, profit factor
- Average win/loss
- Number of trades

### Regime Analysis (Optional)
- Performance by market regime
- Signal attribution analysis
- Regime transition analysis

## Walk-Forward Validation

Enable rolling window validation to assess out-of-sample performance:

```yaml
walk_forward:
  enabled: true
  train_period: "6M"  # 6 months training
  test_period: "1M"   # 1 month testing
  step_size: "2W"     # 2 week steps
  retrain_threshold: 0.05  # Retrain if performance drops 5%
```

The validator:
- Creates rolling train/test windows
- Maintains strict temporal separation
- Tracks performance degradation
- Triggers model retraining when needed
- Generates comprehensive validation reports

## Output Formats

### JSON Report
Complete backtest results including:
- Configuration
- Performance metrics
- Trade data
- Data quality report
- Validation results (if enabled)

### CSV Reports
- `trades.csv`: Individual trade records
- `portfolio.csv`: Portfolio state over time
- `metrics.csv`: Performance metrics
- `costs.csv`: Transaction cost breakdown

### HTML Report
Interactive report with:
- Summary dashboard
- Performance metrics tables
- Risk analysis
- Trade statistics
- Interactive charts (equity curve, drawdown)
- Configuration details

## Examples

See `examples/backtest_engine_example.py` for complete examples:

1. **Load from YAML**: Standard configuration-based backtest
2. **Programmatic Config**: Create configuration in code
3. **Multi-Symbol**: Backtest multiple symbols simultaneously
4. **Walk-Forward**: Rolling window validation

## Data Requirements

### Signal Data (Parquet)
Required columns:
- `timestamp`: Unix timestamp or datetime
- `symbol`: Trading symbol
- `s_ldc`: LDC signal value
- `s_mr`: Mean reversion signal value
- `s_tsmom`: Momentum signal value
- `regime_state`: (Optional) HMM regime state
- `s_fusion`: (Optional) Pre-computed fusion signal

### Market Data (Parquet)
Required columns:
- `timestamp`: Unix timestamp or datetime
- `symbol`: Trading symbol
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume

### Directory Structure

Supports both flat and partitioned structures:

**Partitioned** (recommended):
```
data/
  signals/
    symbol=BTCUSDT/
      date=2024-01-01/
        interval=5m/
          signals.parquet
  market/
    symbol=BTCUSDT/
      date=2024-01-01/
        interval=5m/
          ohlcv.parquet
```

**Flat**:
```
data/
  signals/
    signals_btcusdt.parquet
  market/
    ohlcv_btcusdt.parquet
```

## Environment Variables

Override configuration with environment variables:

```bash
export BACKTEST_INITIAL_CAPITAL=200000
export BACKTEST_START_DATE=2024-01-01
export BACKTEST_END_DATE=2024-12-31
export BACKTEST_SIGNALS_PATH=/path/to/signals
export BACKTEST_MARKET_DATA_PATH=/path/to/market
export BACKTEST_MINIO_ENDPOINT=localhost:9000
export BACKTEST_MINIO_ACCESS_KEY=minioadmin
export BACKTEST_MINIO_SECRET_KEY=minioadmin
```

## Performance Targets

The framework is designed to meet these performance targets:

- **Backtest Speed**: Process 1+ years of 5-minute data in <60 seconds
- **Memory Efficiency**: Handle large datasets with chunked processing
- **Multi-Symbol**: Support 10+ symbols concurrently
- **Data Quality**: Validate and report data issues before processing

## Requirements Coverage

This implementation satisfies all requirements from the specification:

- **Requirement 1**: Complete backtest execution workflow ✓
- **Requirement 2**: Signal conversion with position sizing ✓
- **Requirement 3**: Realistic transaction costs and slippage ✓
- **Requirement 4**: Comprehensive performance metrics ✓
- **Requirement 5**: Walk-forward validation framework ✓
- **Requirement 6**: Data infrastructure integration ✓

## Troubleshooting

### Common Issues

**FileNotFoundError: Signals path not found**
- Ensure `signals_path` and `market_data_path` exist
- Check file permissions
- Verify path format (absolute or relative to working directory)

**Low data quality score**
- Check for missing values in signal or market data
- Verify timestamp alignment between signals and market data
- Review data quality report for specific issues

**No trades generated**
- Check signal thresholds (may be too high)
- Verify signal values are in expected range
- Review signal filtering rules
- Check minimum signal strength setting

**High transaction costs**
- Review commission rate and spread costs
- Check slippage model parameters
- Verify order sizes are appropriate
- Consider adjusting position sizing method

## Development

### Running Tests

```bash
cd py
pytest tests/test_backtesting.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
make type-check
```

## License

Part of the IMP (Intelligent Market Prediction) trading system.
