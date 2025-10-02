use ldc_engine::backtesting::*;
use ldc_engine::{LDCConfig, Direction};
use feature_pipeline::{OHLCV, Features};
use chrono::{Utc, Duration};

fn main() -> anyhow::Result<()> {
    println!("=== LDC Engine Backtesting Performance Demo ===\n");

    // Create sample market data
    let (ohlcv_data, features_data) = create_sample_market_data(500);
    println!("Generated {} data points for backtesting", ohlcv_data.len());

    // Configure backtesting parameters
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.15, // 15% of equity per position
        transaction_cost: 0.001, // 0.1% transaction cost
        slippage: 0.0005, // 0.05% slippage
        signal_threshold: 0.2, // Lower threshold for more trades
        max_positions: 1,
        stop_loss_percent: 0.03, // 3% stop loss
        take_profit_percent: 0.06, // 6% take profit
        min_holding_period: Duration::minutes(30),
        max_holding_period: Duration::hours(12),
        rebalance_frequency: Duration::hours(1),
    };

    // Configure LDC engine
    let ldc_config = LDCConfig {
        neighbors_count: 8,
        max_bars_back: 2000,
        use_hnsw_index: false, // Use exact search for demo
        use_simd_optimization: true,
        ..Default::default()
    };

    // Create and run backtest
    let mut engine = BacktestingEngine::new(backtest_config, ldc_config);
    println!("Running backtest...");
    
    let result = engine.run_backtest(&ohlcv_data, &features_data)?;
    
    // Display comprehensive results
    result.print_summary();
    
    // Show detailed trade analysis if trades were executed
    if result.total_trades > 0 {
        println!("=== Top 10 Trades ===");
        result.print_detailed_trades(Some(10));
        
        // Save detailed report to file
        let report_path = "backtest_performance_report.json";
        result.save_report(report_path)?;
        println!("Detailed performance report saved to: {}", report_path);
    } else {
        println!("No trades were executed in this backtest scenario.");
        println!("This can happen with conservative signal thresholds or insufficient training data.");
    }

    // Demonstrate performance calculator directly
    println!("\n=== Performance Calculator Demo ===");
    demonstrate_performance_calculator()?;

    Ok(())
}

/// Create realistic sample market data for demonstration
fn create_sample_market_data(count: usize) -> (Vec<OHLCV>, Vec<Features>) {
    let mut ohlcv_data = Vec::new();
    let mut features_data = Vec::new();
    let base_timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
    let mut price = 50000.0; // Starting price (like BTC)

    for i in 0..count {
        // Create realistic price movements with different market phases
        let phase_factor = (i as f64 / 100.0).floor();
        let price_change = match (phase_factor as usize) % 4 {
            0 => (i as f64 * 0.02).sin() * 500.0 + 200.0, // Bull market
            1 => (i as f64 * 0.03).cos() * 300.0, // Sideways
            2 => -(i as f64 * 0.025).sin() * 400.0 - 150.0, // Bear market
            _ => (i as f64 * 0.04).sin() * 800.0, // Volatile
        };
        
        price += price_change;
        price = price.max(10000.0); // Prevent unrealistic low prices
        
        let volatility = 0.02 + (i as f64 * 0.01).sin().abs() * 0.03;
        let high = price * (1.0 + volatility);
        let low = price * (1.0 - volatility);
        let volume = 1000000.0 + (i as f64 * 0.1).cos().abs() * 500000.0;

        ohlcv_data.push(OHLCV {
            timestamp: base_timestamp + (i as i64 * 300), // 5-minute intervals
            open: price - price_change,
            high,
            low,
            close: price,
            volume,
        });

        // Create realistic technical indicators
        let rsi = 50.0 + (i as f64 * 0.05).sin() * 40.0; // RSI 10-90
        let momentum = (i as f64 * 0.02).sin() * 0.05; // ±5% momentum
        let wt1 = (i as f64 * 0.03).cos() * 60.0; // WaveTrend
        let cci = (i as f64 * 0.04).sin() * 150.0; // CCI
        let adx = 20.0 + (i as f64 * 0.01).abs() * 40.0; // ADX 20-60

        features_data.push(Features {
            timestamp: base_timestamp + (i as i64 * 300),
            rsi: Some(rsi.clamp(0.0, 100.0)),
            sma_20: Some(price * 0.98),
            ema_20: Some(price * 0.99),
            std_20: Some(price * volatility),
            zscore_20: Some((i as f64 * 0.02).sin() * 2.0),
            momentum: Some(momentum),
            wavetrend_1: Some(wt1),
            wavetrend_2: Some(wt1 * 0.8),
            cci: Some(cci.clamp(-300.0, 300.0)),
            adx: Some(adx.clamp(0.0, 100.0)),
        });
    }

    (ohlcv_data, features_data)
}

/// Demonstrate the performance calculator with sample trades
fn demonstrate_performance_calculator() -> anyhow::Result<()> {
    let calculator = PerformanceCalculator::with_risk_free_rate(0.03); // 3% risk-free rate

    // Create sample trades
    let base_time = Utc::now();
    let trades = vec![
        Trade {
            entry_time: base_time,
            exit_time: base_time + Duration::hours(2),
            direction: Direction::Long,
            entry_price: 50000.0,
            exit_price: 52000.0,
            quantity: 0.1,
            pnl: 200.0,
            signal_strength: 0.8,
            confidence: 0.9,
            holding_period: Duration::hours(2),
            exit_reason: ExitReason::TakeProfit,
            position_id: 1,
        },
        Trade {
            entry_time: base_time + Duration::hours(3),
            exit_time: base_time + Duration::hours(5),
            direction: Direction::Short,
            entry_price: 51000.0,
            exit_price: 49500.0,
            quantity: 0.1,
            pnl: 150.0,
            signal_strength: 0.6,
            confidence: 0.7,
            holding_period: Duration::hours(2),
            exit_reason: ExitReason::OppositeSignal,
            position_id: 2,
        },
        Trade {
            entry_time: base_time + Duration::hours(6),
            exit_time: base_time + Duration::hours(7),
            direction: Direction::Long,
            entry_price: 49000.0,
            exit_price: 47500.0,
            quantity: 0.1,
            pnl: -150.0,
            signal_strength: 0.4,
            confidence: 0.5,
            holding_period: Duration::hours(1),
            exit_reason: ExitReason::StopLoss,
            position_id: 3,
        },
    ];

    // Create sample equity curve
    let equity_curve = vec![
        EquityPoint {
            timestamp: base_time,
            equity: 100000.0,
            drawdown: 0.0,
            position_value: 0.0,
            cash: 100000.0,
            open_positions: 0,
        },
        EquityPoint {
            timestamp: base_time + Duration::hours(2),
            equity: 100200.0,
            drawdown: 0.0,
            position_value: 0.0,
            cash: 100200.0,
            open_positions: 0,
        },
        EquityPoint {
            timestamp: base_time + Duration::hours(5),
            equity: 100350.0,
            drawdown: 0.0,
            position_value: 0.0,
            cash: 100350.0,
            open_positions: 0,
        },
        EquityPoint {
            timestamp: base_time + Duration::hours(7),
            equity: 100200.0,
            drawdown: 0.0015, // Small drawdown
            position_value: 0.0,
            cash: 100200.0,
            open_positions: 0,
        },
    ];

    // Calculate comprehensive metrics
    let metrics = calculator.calculate_metrics(&trades, &equity_curve, 100000.0);

    println!("Performance Metrics:");
    println!("  Win Rate: {:.1}%", metrics.win_rate * 100.0);
    println!("  Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("  Total Returns: {:.2}%", metrics.total_returns * 100.0);

    println!("\nTrade Analysis:");
    println!("  Total Trades: {}", metrics.trade_analysis.total_trades);
    println!("  Profitable Trades: {}", metrics.trade_analysis.profitable_trades);
    println!("  Profit Factor: {:.2}", metrics.trade_analysis.profit_factor);
    println!("  Average Trade: ${:.2}", metrics.trade_analysis.average_pnl);
    println!("  Largest Winner: ${:.2}", metrics.trade_analysis.largest_winner);
    println!("  Largest Loser: ${:.2}", metrics.trade_analysis.largest_loser);
    println!("  Signal-PnL Correlation: {:.3}", metrics.trade_analysis.signal_pnl_correlation);

    println!("\nAttribution Analysis:");
    println!("  Long Trades: {} (PnL: ${:.2})", metrics.attribution.long_trades, metrics.attribution.long_pnl);
    println!("  Short Trades: {} (PnL: ${:.2})", metrics.attribution.short_trades, metrics.attribution.short_pnl);
    println!("  High Signal Trades: {} (PnL: ${:.2})", metrics.attribution.high_signal_trades, metrics.attribution.high_signal_pnl);

    Ok(())
}