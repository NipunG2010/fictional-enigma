use feature_pipeline::{FeaturePipeline, OHLCV};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debugging Feature Calculations");
    
    // Generate a small sample of synthetic data with more realistic price movements
    let mut ohlcv_data = Vec::new();
    let mut price = 100.0;
    let mut timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
    
    for i in 0..30 {
        // Generate more realistic price movement with both up and down moves
        let trend = (i as f64 * 0.2).sin() * 0.1; // Oscillating trend
        let volatility = 0.005 + (i as f64 * 0.1).cos() * 0.002; // Variable volatility
        let random_move = (i as f64 * 0.3 + 1.5).sin() * volatility; // More random-like movement
        
        price += trend + random_move;
        
        // Generate OHLC from price with realistic spread
        let spread = price * 0.001; // 0.1% spread
        let open = price;
        let high = price + spread * (0.5 + (i as f64 * 0.1).cos() * 0.3);
        let low = price - spread * (0.5 + (i as f64 * 0.15).sin() * 0.3);
        let close = price + spread * (i as f64 * 0.05).sin() * 0.2;
        let volume = 1000.0 + (i as f64 * 50.0) + (i as f64 * 0.2).sin() * 200.0;
        
        ohlcv_data.push(OHLCV {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
        
        timestamp += 300; // 5-minute bars
        
        println!("Bar {}: O={:.4} H={:.4} L={:.4} C={:.4} V={:.2}", 
                 i, open, high, low, close, volume);
    }
    
    // Compute features
    let pipeline = FeaturePipeline::new(20);
    let features = pipeline.compute_features(&ohlcv_data)?;
    
    println!("\n📊 Feature Analysis:");
    println!("Generated {} features from {} OHLCV bars", features.len(), ohlcv_data.len());
    
    // Examine first few features in detail
    for (i, feature) in features.iter().take(10).enumerate() {
        println!("\nFeature {}:", i);
        println!("  RSI: {:?}", feature.rsi);
        println!("  SMA_20: {:?}", feature.sma_20);
        println!("  EMA_20: {:?}", feature.ema_20);
        println!("  STD_20: {:?}", feature.std_20);
        println!("  ZScore_20: {:?}", feature.zscore_20);
        println!("  Momentum: {:?}", feature.momentum);
        println!("  WaveTrend_1: {:?}", feature.wavetrend_1);
        println!("  WaveTrend_2: {:?}", feature.wavetrend_2);
        println!("  CCI: {:?}", feature.cci);
        println!("  ADX: {:?}", feature.adx);
    }
    
    // Check last few features (should be more stable)
    println!("\n📈 Last 5 Features (should be more stable):");
    for (i, feature) in features.iter().rev().take(5).enumerate() {
        let idx = features.len() - 1 - i;
        println!("\nFeature {} (index {}):", idx, idx);
        println!("  RSI: {:?}", feature.rsi);
        println!("  SMA_20: {:?}", feature.sma_20);
        println!("  EMA_20: {:?}", feature.ema_20);
        println!("  STD_20: {:?}", feature.std_20);
        println!("  ZScore_20: {:?}", feature.zscore_20);
        println!("  Momentum: {:?}", feature.momentum);
        println!("  WaveTrend_1: {:?}", feature.wavetrend_1);
        println!("  WaveTrend_2: {:?}", feature.wavetrend_2);
        println!("  CCI: {:?}", feature.cci);
        println!("  ADX: {:?}", feature.adx);
    }
    
    // Analyze specific issues
    println!("\n🔍 Issue Analysis:");
    
    // Check RSI values
    let rsi_values: Vec<f64> = features.iter().filter_map(|f| f.rsi).collect();
    if !rsi_values.is_empty() {
        let rsi_min = rsi_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let rsi_max = rsi_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let rsi_avg = rsi_values.iter().sum::<f64>() / rsi_values.len() as f64;
        println!("RSI: min={:.4}, max={:.4}, avg={:.4}, count={}", rsi_min, rsi_max, rsi_avg, rsi_values.len());
        
        if rsi_values.iter().all(|&x| (x - 100.0).abs() < 1e-6) {
            println!("⚠️  RSI Issue: All RSI values are 100.0 - likely calculation error");
        }
    }
    
    // Check momentum values
    let momentum_values: Vec<f64> = features.iter().filter_map(|f| f.momentum).collect();
    if !momentum_values.is_empty() {
        let momentum_min = momentum_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let momentum_max = momentum_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let momentum_avg = momentum_values.iter().sum::<f64>() / momentum_values.len() as f64;
        println!("Momentum: min={:.6}, max={:.6}, avg={:.6}, count={}", momentum_min, momentum_max, momentum_avg, momentum_values.len());
        
        if momentum_values.iter().all(|&x| x.abs() < 1e-10) {
            println!("⚠️  Momentum Issue: All momentum values are ~0.0 - likely calculation error");
        }
    }
    
    // Check z-score values
    let zscore_values: Vec<f64> = features.iter().filter_map(|f| f.zscore_20).collect();
    if !zscore_values.is_empty() {
        let zscore_min = zscore_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let zscore_max = zscore_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let zscore_avg = zscore_values.iter().sum::<f64>() / zscore_values.len() as f64;
        println!("Z-Score: min={:.4}, max={:.4}, avg={:.4}, count={}", zscore_min, zscore_max, zscore_avg, zscore_values.len());
        
        let zscore_range = (1.61..=1.64);
        if zscore_values.iter().all(|&x| zscore_range.contains(&x)) {
            println!("⚠️  Z-Score Issue: All z-scores in narrow range 1.61-1.64 - likely calculation error");
        }
    }
    
    // Check ADX values
    let adx_values: Vec<f64> = features.iter().filter_map(|f| f.adx).collect();
    if !adx_values.is_empty() {
        let adx_min = adx_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let adx_max = adx_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let adx_avg = adx_values.iter().sum::<f64>() / adx_values.len() as f64;
        println!("ADX: min={:.4}, max={:.4}, avg={:.4}, count={}", adx_min, adx_max, adx_avg, adx_values.len());
        
        if adx_values.iter().all(|&x| (x - 100.0).abs() < 1e-6) {
            println!("⚠️  ADX Issue: All ADX values are 100.0 - likely calculation error");
        }
    }
    
    Ok(())
}