# Backtesting Framework Implementation Plan

- [x] 1. Set up backtesting package structure and dependencies
  - Create `py/imp/backtesting/` package directory with `__init__.py`
  - Add vectorbt, plotly, and yaml dependencies to `pyproject.toml`
  - Create configuration schema using pydantic for backtest settings
  - _Requirements: 1.1, 6.3_

- [x] 2. Implement core data loading and validation
- [x] 2.1 Create data loader for signal and market data
  - Write `DataLoader` class to read Parquet signal files with date filtering
  - Implement market data loading from existing OHLCV Parquet files
  - Add data validation methods to check for missing values and anomalies
  - _Requirements: 6.1, 6.4, 6.5_

- [x] 2.2 Implement MinIO integration for HMM artifacts
  - Create `ArtifactLoader` class to fetch HMM weights from MinIO storage
  - Add caching mechanism for frequently accessed model artifacts
  - Implement fallback to local storage when MinIO is unavailable
  - _Requirements: 6.2, 6.3_

- [ ]* 2.3 Write unit tests for data loading components
  - Create test fixtures with sample signal and market data
  - Test data validation logic with various edge cases
  - Verify MinIO integration with mocked storage
  - _Requirements: 6.1, 6.2, 6.5_

- [x] 3. Build signal processing and filtering system
- [x] 3.1 Implement SignalProcessor class
  - Create signal threshold application and filtering logic
  - Implement HMM weight integration for regime-aware signal fusion
  - Add signal strength validation and conflict resolution
  - _Requirements: 2.1, 2.2, 2.4, 2.5_

- [x] 3.2 Create trade signal generation logic
  - Implement buy/sell/hold decision logic based on processed signals
  - Add configurable signal filtering rules to prevent overtrading
  - Create audit trail logging for all signal-to-trade conversions
  - _Requirements: 2.1, 2.3, 2.5_

- [ ]* 3.3 Write unit tests for signal processing
  - Test signal threshold application with various input scenarios
  - Verify HMM weight integration and fallback behavior
  - Test trade signal generation logic and audit trail
  - _Requirements: 2.1, 2.2, 2.5_

- [ ] 4. Implement position sizing and trade generation
- [x] 4.1 Create TradeGenerator class with position sizing methods
  - Implement fixed size, percentage, and volatility-based position sizing
  - Add risk management rules for position limits and exposure controls
  - Create order generation with proper timestamps and metadata
  - _Requirements: 2.1, 2.2, 2.4_

- [x] 4.2 Build position tracking and portfolio state management
  - Implement PortfolioState class to track cash and positions over time
  - Add position carry-forward logic for multi-period backtests
  - Create position reconciliation and validation methods
  - _Requirements: 1.2, 1.3_

- [ ]* 4.3 Write unit tests for trade generation and position tracking
  - Test different position sizing methods with edge cases
  - Verify position tracking accuracy across multiple time periods
  - Test risk management rule enforcement
  - _Requirements: 2.1, 2.2, 2.4_

- [x] 5. Implement transaction cost and slippage modeling
- [x] 5.1 Create CostModel class for transaction cost calculation
  - Implement commission calculation with configurable rate structures
  - Add spread cost modeling for different asset classes
  - Create market impact estimation based on order size and volume
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 5.2 Implement slippage modeling system
  - Create slippage calculation based on order size and market volatility
  - Add linear and square-root market impact models
  - Implement cost breakdown analysis and reporting
  - _Requirements: 3.2, 3.5_

- [ ]* 5.3 Write unit tests for cost modeling
  - Test commission calculations with various trade sizes and rates
  - Verify slippage modeling accuracy with different market conditions
  - Test cost breakdown analysis and aggregation
  - _Requirements: 3.1, 3.2, 3.4, 3.5_

- [ ] 6. Build performance analysis and metrics calculation
- [x] 6.1 Implement PerformanceAnalyzer class
  - Create standard performance metrics calculation (returns, Sharpe, drawdown)
  - Implement risk-adjusted metrics (Sortino, Calmar, VaR)
  - Add benchmark comparison functionality
  - _Requirements: 4.1, 4.2, 4.5_

- [x] 6.2 Create regime-specific performance analysis
  - Implement performance breakdown by HMM-detected market regimes
  - Add performance attribution analysis by signal source
  - Create time-series performance tracking and visualization
  - _Requirements: 4.3, 4.4_

- [ ]* 6.3 Write unit tests for performance calculations
  - Test performance metric calculations against known results
  - Verify regime-specific analysis with sample data
  - Test benchmark comparison and attribution analysis
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 7. Implement walk-forward validation framework
- [x] 7.1 Create WalkForwardValidator class
  - Implement rolling window setup with configurable train/test periods
  - Add temporal data separation to prevent look-ahead bias
  - Create model retraining trigger logic based on performance degradation
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 7.2 Build validation reporting and analysis
  - Implement in-sample vs out-of-sample performance comparison
  - Create model stability tracking and retraining recommendations
  - Add validation report generation with statistical significance tests
  - _Requirements: 5.4, 5.5_

- [ ]* 7.3 Write unit tests for walk-forward validation
  - Test rolling window logic with various time periods
  - Verify temporal separation and data integrity
  - Test retraining trigger logic and performance tracking
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 8. Create main BacktestEngine orchestrator
- [x] 8.1 Implement BacktestEngine class
  - Create main backtest execution workflow orchestrating all components
  - Add multi-symbol processing with independent position tracking
  - Implement progress tracking and logging for long-running backtests
  - _Requirements: 1.1, 1.2, 1.4_

- [x] 8.2 Add configuration management and validation
  - Create comprehensive configuration schema with validation
  - Implement configuration file loading and environment variable support
  - Add configuration validation and error reporting
  - _Requirements: 1.1, 6.5_

- [x] 8.3 Build result output and reporting system
  - Implement standardized result output format compatible with analysis tools
  - Create comprehensive backtest reports with charts and statistics
  - Add result export to various formats (JSON, CSV, HTML)
  - _Requirements: 1.5, 4.1, 4.2_

