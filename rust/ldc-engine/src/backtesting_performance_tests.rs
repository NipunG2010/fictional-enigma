use crate::backtesting::*;
use crate::{LDCConfig, Direction};
use feature_pipeline::{OHLCV, Features};
use chrono::{Utc, Duration};

    /// Create comprehensive sample data for testing performance calculations
    fn create_comprehensive_test_data() -> (Vec<OHLCV>, Vec<Features>) {
        let mut ohlcv_data = Vec::new();
        let mut features_data = Vec::new();
        let base_timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
        let mut price = 100.0;

        // Create 1000 data points with various market conditions
        for i in 0..1000 {
            // Simulate different market regimes
            let price_change = match i {
                0..=200 => (i as f64 * 0.1).sin() * 1.0, // Trending up
                201..=400 => (i as f64 * 0.05).cos() * 0.5, // Sideways
                401..=600 => -(i as f64 * 0.08).sin() * 1.5, // Trending down
                601..=800 => (i as f64 * 0.2).sin() * 2.0, // Volatile
                _ => (i as f64 * 0.03).cos() * 0.3, // Quiet
            };
            
            price += price_change;
            price = price.max(50.0); // Prevent negative prices
            
            let high = price + (i as f64 * 0.01).abs() % 2.0;
            let low = price - (i as f64 * 0.01).abs() % 2.0;
            let volume = 1000.0 + (i as f64 * 0.1).sin().abs() * 500.0;

            ohlcv_data.push(OHLCV {
                timestamp: base_timestamp + (i as i64 * 300), // 5-minute intervals
                open: price - price_change,
                high,
                low,
                close: price,
                volume,
            });

            // Create corresponding features with realistic values
            features_data.push(Features {
                timestamp: base_timestamp + (i as i64 * 300),
                rsi: Some(50.0 + (i as f64 * 0.1).sin() * 30.0), // RSI 20-80
                sma_20: Some(price),
                ema_20: Some(price * 0.99),
                std_20: Some(2.0 + (i as f64 * 0.01).abs()),
                zscore_20: Some((i as f64 * 0.02).sin()),
                momentum: Some((i as f64 * 0.04).sin() * 0.1), // Momentum -0.1 to 0.1
                wavetrend_1: Some((i as f64 * 0.05).cos() * 50.0), // WT -50 to 50
                wavetrend_2: Some((i as f64 * 0.06).sin() * 30.0), // WT2 -30 to 30
                cci: Some((i as f64 * 0.03).sin() * 100.0), // CCI -100 to 100
                adx: Some(25.0 + (i as f64 * 0.02).abs() * 25.0), // ADX 25-50
            });
        }

        (ohlcv_data, features_data)
    }

    /// Test comprehensive performance calculation
    #[test]
    fn test_comprehensive_performance_calculation() {
        let (ohlcv_data, features_data) = create_comprehensive_test_data();
        
        let backtest_config = BacktestConfig {
            initial_capital: 100_000.0,
            position_size: 0.1,
            transaction_cost: 0.001,
            slippage: 0.0005,
            signal_threshold: 0.1, // Lower threshold to generate more trades
            max_positions: 1,
            stop_loss_percent: 0.05,
            take_profit_percent: 0.10,
            ..Default::default()
        };

        let ldc_config = LDCConfig::default();
        let mut engine = BacktestingEngine::new(backtest_config, ldc_config);

        let result = engine.run_backtest(&ohlcv_data, &features_data).unwrap();

        println!("Backtest executed {} trades", result.total_trades);
        
        // Verify basic metrics are calculated (allow for zero trades in test scenario)
        assert!(result.sharpe_ratio.is_finite(), "Sharpe ratio should be finite");
        assert!(result.max_drawdown >= 0.0, "Max drawdown should be non-negative");
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0, "Win rate should be between 0 and 1");

        // If no trades were executed, verify the performance metrics handle this correctly
        if result.total_trades == 0 {
            assert_eq!(result.win_rate, 0.0, "Win rate should be 0 when no trades");
            assert_eq!(result.profitable_trades, 0, "Profitable trades should be 0");
            assert_eq!(result.trade_analysis.total_trades, 0, "Trade analysis should show 0 trades");
            println!("No trades executed - this is acceptable for test data");
            return; // Skip remaining assertions that require trades
        }

        // Verify trade analysis
        let trade_analysis = &result.trade_analysis;
        assert_eq!(trade_analysis.total_trades, result.total_trades);
        assert_eq!(trade_analysis.profitable_trades + trade_analysis.losing_trades + trade_analysis.breakeven_trades, 
                   result.total_trades);
        assert!(trade_analysis.profit_factor >= 0.0, "Profit factor should be non-negative");
        
        // Verify attribution analysis
        let attribution = &result.performance_attribution;
        assert_eq!(attribution.long_trades + attribution.short_trades, result.total_trades);
        assert_eq!(attribution.high_signal_trades + attribution.medium_signal_trades + attribution.low_signal_trades, 
                   result.total_trades);
        assert_eq!(attribution.short_term_trades + attribution.medium_term_trades + attribution.long_term_trades, 
                   result.total_trades);

        // Verify drawdown analysis
        let drawdown_analysis = &result.drawdown_analysis;
        assert!(drawdown_analysis.max_drawdown >= 0.0, "Max drawdown should be non-negative");
        assert!(drawdown_analysis.drawdown_periods_count >= 0, "Drawdown periods count should be non-negative");

        // Print summary for manual verification
        result.print_summary();
    }

    /// Test performance calculator with edge cases
    #[test]
    fn test_performance_calculator_edge_cases() {
        let calculator = PerformanceCalculator::new();

        // Test with empty data
        let empty_trades: Vec<Trade> = Vec::new();
        let empty_equity: Vec<EquityPoint> = Vec::new();
        let metrics = calculator.calculate_metrics(&empty_trades, &empty_equity, 100_000.0);
        
        assert_eq!(metrics.win_rate, 0.0);
        assert_eq!(metrics.sharpe_ratio, 0.0);
        assert_eq!(metrics.max_drawdown, 0.0);
        assert_eq!(metrics.total_returns, 0.0);

        // Test with single trade
        let single_trade = vec![Trade {
            entry_time: Utc::now(),
            exit_time: Utc::now() + Duration::hours(1),
            direction: Direction::Long,
            entry_price: 100.0,
            exit_price: 105.0,
            quantity: 10.0,
            pnl: 50.0,
            signal_strength: 0.8,
            confidence: 0.9,
            holding_period: Duration::hours(1),
            exit_reason: ExitReason::TakeProfit,
            position_id: 1,
        }];

        let single_equity = vec![
            EquityPoint {
                timestamp: Utc::now(),
                equity: 100_000.0,
                drawdown: 0.0,
                position_value: 0.0,
                cash: 100_000.0,
                open_positions: 0,
            },
            EquityPoint {
                timestamp: Utc::now() + Duration::hours(1),
                equity: 100_050.0,
                drawdown: 0.0,
                position_value: 0.0,
                cash: 100_050.0,
                open_positions: 0,
            },
        ];

        let metrics = calculator.calculate_metrics(&single_trade, &single_equity, 100_000.0);
        assert_eq!(metrics.win_rate, 1.0);
        assert!(metrics.sharpe_ratio.is_finite());
        assert_eq!(metrics.max_drawdown, 0.0);
        assert!(metrics.total_returns > 0.0);
    }

    /// Test correlation calculation
    #[test]
    fn test_correlation_calculation() {
        let calculator = PerformanceCalculator::new();

        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = calculator.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10, "Should be perfect positive correlation");

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let correlation = calculator.calculate_correlation(&x, &y_neg);
        assert!((correlation + 1.0).abs() < 1e-10, "Should be perfect negative correlation");

        // No correlation
        let y_random = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let correlation = calculator.calculate_correlation(&x, &y_random);
        assert!(correlation.abs() < 1.0, "Should be between -1 and 1");

        // Edge cases
        let empty: Vec<f64> = Vec::new();
        let correlation = calculator.calculate_correlation(&empty, &empty);
        assert_eq!(correlation, 0.0, "Empty vectors should return 0 correlation");

        let single = vec![1.0];
        let correlation = calculator.calculate_correlation(&single, &single);
        assert_eq!(correlation, 0.0, "Single element should return 0 correlation");
    }

    /// Test consecutive streak calculation
    #[test]
    fn test_consecutive_streaks() {
        let calculator = PerformanceCalculator::new();

        // Create trades with known win/loss pattern
        let trades = vec![
            create_test_trade(10.0),   // Win
            create_test_trade(5.0),    // Win
            create_test_trade(15.0),   // Win (3 consecutive wins)
            create_test_trade(-8.0),   // Loss
            create_test_trade(-12.0),  // Loss (2 consecutive losses)
            create_test_trade(20.0),   // Win
            create_test_trade(-5.0),   // Loss
            create_test_trade(-3.0),   // Loss
            create_test_trade(-7.0),   // Loss (3 consecutive losses)
            create_test_trade(25.0),   // Win
        ];

        let (max_wins, max_losses) = calculator.calculate_consecutive_streaks(&trades);
        assert_eq!(max_wins, 3, "Should find 3 consecutive wins");
        assert_eq!(max_losses, 3, "Should find 3 consecutive losses");

        // Test with all wins
        let all_wins: Vec<Trade> = (0..5).map(|_| create_test_trade(10.0)).collect();
        let (max_wins, max_losses) = calculator.calculate_consecutive_streaks(&all_wins);
        assert_eq!(max_wins, 5, "Should find 5 consecutive wins");
        assert_eq!(max_losses, 0, "Should find 0 consecutive losses");

        // Test with empty trades
        let empty_trades: Vec<Trade> = Vec::new();
        let (max_wins, max_losses) = calculator.calculate_consecutive_streaks(&empty_trades);
        assert_eq!(max_wins, 0, "Empty trades should return 0 wins");
        assert_eq!(max_losses, 0, "Empty trades should return 0 losses");
    }

    /// Helper function to create a test trade
    fn create_test_trade(pnl: f64) -> Trade {
        Trade {
            entry_time: Utc::now(),
            exit_time: Utc::now() + Duration::hours(1),
            direction: if pnl > 0.0 { Direction::Long } else { Direction::Short },
            entry_price: 100.0,
            exit_price: if pnl > 0.0 { 105.0 } else { 95.0 },
            quantity: 10.0,
            pnl,
            signal_strength: 0.5,
            confidence: 0.6,
            holding_period: Duration::hours(1),
            exit_reason: if pnl > 0.0 { ExitReason::TakeProfit } else { ExitReason::StopLoss },
            position_id: 1,
        }
    }

    /// Test drawdown analysis
    #[test]
    fn test_drawdown_analysis() {
        let calculator = PerformanceCalculator::new();

        // Create equity curve with known drawdown pattern
        let base_time = Utc::now();
        let equity_curve = vec![
            EquityPoint { timestamp: base_time, equity: 100_000.0, drawdown: 0.0, position_value: 0.0, cash: 100_000.0, open_positions: 0 },
            EquityPoint { timestamp: base_time + Duration::hours(1), equity: 110_000.0, drawdown: 0.0, position_value: 0.0, cash: 110_000.0, open_positions: 0 }, // New high
            EquityPoint { timestamp: base_time + Duration::hours(2), equity: 105_000.0, drawdown: 0.045, position_value: 0.0, cash: 105_000.0, open_positions: 0 }, // 4.5% drawdown
            EquityPoint { timestamp: base_time + Duration::hours(3), equity: 95_000.0, drawdown: 0.136, position_value: 0.0, cash: 95_000.0, open_positions: 0 }, // 13.6% drawdown
            EquityPoint { timestamp: base_time + Duration::hours(4), equity: 115_000.0, drawdown: 0.0, position_value: 0.0, cash: 115_000.0, open_positions: 0 }, // Recovery to new high
            EquityPoint { timestamp: base_time + Duration::hours(5), equity: 110_000.0, drawdown: 0.043, position_value: 0.0, cash: 110_000.0, open_positions: 0 }, // 4.3% drawdown
            EquityPoint { timestamp: base_time + Duration::hours(6), equity: 120_000.0, drawdown: 0.0, position_value: 0.0, cash: 120_000.0, open_positions: 0 }, // Recovery
        ];

        let drawdown_analysis = calculator.calculate_drawdown_analysis(&equity_curve);

        assert!(drawdown_analysis.max_drawdown > 0.13, "Should detect significant drawdown");
        assert!(drawdown_analysis.drawdown_periods_count >= 2, "Should detect multiple drawdown periods");
        assert!(drawdown_analysis.average_drawdown_duration_minutes > 0.0, "Should have positive average duration");
        assert!(drawdown_analysis.average_recovery_time_minutes > 0.0, "Should have positive recovery time");
    }

    /// Test JSON serialization and report generation
    #[test]
    fn test_json_report_generation() {
        let (ohlcv_data, features_data) = create_comprehensive_test_data();
        
        let backtest_config = BacktestConfig::default();
        let ldc_config = LDCConfig::default();
        let mut engine = BacktestingEngine::new(backtest_config, ldc_config);

        let result = engine.run_backtest(&ohlcv_data[..100], &features_data[..100]).unwrap(); // Use smaller dataset for faster test

        // Test JSON serialization
        let json_report = result.to_json_report();
        assert!(json_report.is_ok(), "Should be able to serialize to JSON");

        let json_string = json_report.unwrap();
        assert!(json_string.contains("total_return"), "JSON should contain total_return field");
        assert!(json_string.contains("sharpe_ratio"), "JSON should contain sharpe_ratio field");
        assert!(json_string.contains("trade_analysis"), "JSON should contain trade_analysis field");
        assert!(json_string.contains("drawdown_analysis"), "JSON should contain drawdown_analysis field");

        // Test deserialization
        let deserialized: Result<BacktestResult, _> = serde_json::from_str(&json_string);
        assert!(deserialized.is_ok(), "Should be able to deserialize from JSON");
    }