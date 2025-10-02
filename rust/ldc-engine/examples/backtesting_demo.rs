use anyhow::Result;
use chrono::{DateTime, Utc};
use ldc_engine::backtesting::{BacktestConfig, BacktestingEngine};
use ldc_engine::LDCConfig;
use feature_pipeline::{OHLCV, Features};

/// Create sample OHLCV data for demonstration
fn create_demo_ohlcv_data(count: usize) -> Vec<OHLCV> {
    let mut data = Vec::new();
    let base_timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
    let mut price = 100.0;

    for i in 0..count {
        // Simulate realistic price movement with trend and volatility
        let trend = (i as f64 * 0.001).sin() * 0.5; // Long-term trend
        let volatility = (i as f64 * 0.1).sin() * 2.0; // Short-term volatility
        let noise = (i as f64 * 0.3).cos() * 0.5; // Random noise
        
        let price_change = trend + volatility + noise;
        price += price_change;
        price = price.max(50.0); // Prevent negative prices
        
        let high = price + (i as f64 * 0.05).abs().min(5.0);
        let low = price - (i as f64 * 0.05).abs().min(5.0);
        let volume = 1000.0 + (i as f64 * 10.0);

        data.push(OHLCV {
            timestamp: base_timestamp + (i as i64 * 3600), // 1 hour intervals
            open: price - price_change,
            high,
            low,
            close: price,
            volume,
        });
    }

    data
}

/// Create sample features data for demonstration
fn create_demo_features_data(ohlcv_data: &[OHLCV]) -> Vec<Features> {
    ohlcv_data
        .iter()
        .enumerate()
        .map(|(i, ohlcv)| {
            // Create realistic technical indicators
            let rsi = 50.0 + (i as f64 * 0.1).sin() * 30.0; // RSI between 20-80
            let momentum = (ohlcv.close - 100.0) / 100.0; // Simple momentum
            let wavetrend = (i as f64 * 0.2).sin() * 50.0; // Oscillating indicator
            let cci = (i as f64 * 0.15).sin() * 100.0; // CCI oscillator
            let adx = 25.0 + (i as f64 * 0.1).abs() * 25.0; // ADX trend strength
            
            Features {
                timestamp: ohlcv.timestamp,
                rsi: Some(rsi),
                sma_20: Some(ohlcv.close * 0.99), // SMA slightly below close
                ema_20: Some(ohlcv.close * 1.01), // EMA slightly above close
                std_20: Some(2.0),
                zscore_20: Some((ohlcv.close - 100.0) / 2.0),
                momentum: Some(momentum),
                wavetrend_1: Some(wavetrend),
                wavetrend_2: Some(wavetrend * 0.8), // Signal line
                cci: Some(cci),
                adx: Some(adx),
            }
        })
        .collect()
}

fn main() -> Result<()> {
    println!("LDC Backtesting Framework Demo");
    println!("==============================\n");

    // Create backtesting configuration
    let backtest_config = BacktestConfig {
        initial_capital: 50_000.0,
        position_size: 0.2, // 20% of equity per position
        transaction_cost: 0.001, // 0.1% transaction cost
        slippage: 0.0005, // 0.05% slippage
        signal_threshold: 0.3, // Lower threshold for more trades
        max_positions: 2, // Allow up to 2 concurrent positions
        stop_loss_percent: 0.03, // 3% stop loss
        take_profit_percent: 0.06, // 6% take profit
        ..BacktestConfig::default()
    };

    // Create LDC engine configuration
    let ldc_config = LDCConfig {
        neighbors_count: 8,
        max_bars_back: 2000,
        use_multithreading: true,
        use_simd_optimization: true,
        ..LDCConfig::default()
    };

    // Initialize backtesting engine
    let mut engine = BacktestingEngine::new(backtest_config, ldc_config);

    // Generate sample data
    println!("Generating sample market data...");
    let ohlcv_data = create_demo_ohlcv_data(1000); // 1000 hours of data
    let features_data = create_demo_features_data(&ohlcv_data);

    println!("Data generated: {} OHLCV bars, {} feature sets", 
             ohlcv_data.len(), features_data.len());
    println!("Price range: ${:.2} - ${:.2}", 
             ohlcv_data.iter().map(|o| o.close).fold(f64::INFINITY, f64::min),
             ohlcv_data.iter().map(|o| o.close).fold(f64::NEG_INFINITY, f64::max));

    // Run backtest
    println!("\nRunning backtest...");
    let start_time = std::time::Instant::now();
    
    let result = engine.run_backtest(&ohlcv_data, &features_data)?;
    
    let duration = start_time.elapsed();
    println!("Backtest completed in {:.2}s", duration.as_secs_f64());

    // Display results
    result.print_summary();

    // Additional analysis
    println!("=== Detailed Analysis ===");
    
    if !result.trades.is_empty() {
        let avg_holding_period = result.trades.iter()
            .map(|t| t.holding_period.num_hours())
            .sum::<i64>() as f64 / result.trades.len() as f64;
        
        println!("Average holding period: {:.1} hours", avg_holding_period);
        
        let best_trade = result.trades.iter()
            .max_by(|a, b| a.pnl.partial_cmp(&b.pnl).unwrap());
        let worst_trade = result.trades.iter()
            .min_by(|a, b| a.pnl.partial_cmp(&b.pnl).unwrap());
        
        if let Some(trade) = best_trade {
            println!("Best trade: ${:.2} PnL ({:?} position)", trade.pnl, trade.direction);
        }
        if let Some(trade) = worst_trade {
            println!("Worst trade: ${:.2} PnL ({:?} position)", trade.pnl, trade.direction);
        }

        // Exit reason analysis
        let mut exit_reasons = std::collections::HashMap::new();
        for trade in &result.trades {
            *exit_reasons.entry(format!("{:?}", trade.exit_reason)).or_insert(0) += 1;
        }
        
        println!("\nExit reasons:");
        for (reason, count) in exit_reasons {
            println!("  {}: {} trades", reason, count);
        }
    }

    // Equity curve analysis
    if result.equity_curve.len() > 1 {
        let final_equity = result.equity_curve.last().unwrap().equity;
        let peak_equity = result.equity_curve.iter()
            .map(|e| e.equity)
            .fold(f64::NEG_INFINITY, f64::max);
        
        println!("\nEquity Analysis:");
        println!("Final equity: ${:.2}", final_equity);
        println!("Peak equity: ${:.2}", peak_equity);
        println!("Max drawdown: {:.2}%", result.max_drawdown * 100.0);
        
        // Calculate volatility
        let returns: Vec<f64> = result.equity_curve.windows(2)
            .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
            .collect();
        
        if !returns.is_empty() {
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let volatility = variance.sqrt();
            
            println!("Return volatility: {:.4} ({:.2}% annualized)", 
                     volatility, volatility * (365.25 * 24.0_f64).sqrt() * 100.0);
        }
    }

    println!("\n=== Demo Complete ===");
    println!("This demo shows the basic functionality of the LDC backtesting framework.");
    println!("The framework supports:");
    println!("- Configurable trading parameters");
    println!("- Multiple position management");
    println!("- Stop loss and take profit orders");
    println!("- Comprehensive performance metrics");
    println!("- Integration with the LDC prediction engine");

    Ok(())
}