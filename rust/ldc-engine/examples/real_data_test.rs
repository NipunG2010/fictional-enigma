use ldc_engine::{LDCEngine, LDCConfig};
use feature_pipeline::{FeaturePipeline, OHLCV};
use serde::Deserialize;
use std::error::Error;

// Binance kline data comes as array: [open_time, open, high, low, close, volume, close_time, ...]
type BinanceKline = (i64, String, String, String, String, String, i64, String, i64, String, String, String);

async fn fetch_btc_data(limit: u32) -> Result<Vec<OHLCV>, Box<dyn Error>> {
    println!("📡 Fetching BTC/USDT data from Binance...");
    
    let url = format!(
        "https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=5m&limit={}",
        limit
    );
    
    let response = reqwest::get(&url).await?;
    let klines: Vec<BinanceKline> = response.json().await?;
    
    let mut ohlcv_data = Vec::new();
    
    for kline in klines {
        let ohlcv = OHLCV {
            timestamp: kline.0 / 1000, // Convert to seconds
            open: kline.1.parse::<f64>()?,
            high: kline.2.parse::<f64>()?,
            low: kline.3.parse::<f64>()?,
            close: kline.4.parse::<f64>()?,
            volume: kline.5.parse::<f64>()?,
        };
        ohlcv_data.push(ohlcv);
    }
    
    println!("✅ Fetched {} BTC/USDT 5-minute bars", ohlcv_data.len());
    println!("📊 Price range: ${:.2} - ${:.2}", 
             ohlcv_data.iter().map(|x| x.low).fold(f64::INFINITY, f64::min),
             ohlcv_data.iter().map(|x| x.high).fold(f64::NEG_INFINITY, f64::max));
    
    Ok(ohlcv_data)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 LDC Engine Real Data Test - BTC/USDT");
    println!("{}", "=".repeat(50));
    
    // Fetch real BTC data (last 500 5-minute bars ≈ ~42 hours)
    let ohlcv_data = fetch_btc_data(500).await?;
    
    // Compute features using our validated pipeline
    println!("🔧 Computing technical indicators...");
    let pipeline = FeaturePipeline::new(20);
    let features = pipeline.compute_features(&ohlcv_data)?;
    
    // Filter out features that don't have all required fields
    let valid_features: Vec<_> = features.iter()
        .filter(|f| f.rsi.is_some() && f.wavetrend_1.is_some() && f.cci.is_some() && f.adx.is_some() && f.wavetrend_2.is_some())
        .collect();
    
    println!("✅ Computed features for {} bars ({} valid)", features.len(), valid_features.len());
    
    if valid_features.len() < 50 {
        return Err("Not enough valid features for LDC training. Need at least 50 samples.".into());
    }
    
    // Configure LDC engine for real data testing
    let mut config = LDCConfig::default();
    config.max_bars_back = valid_features.len().min(300); // Use available data
    config.neighbors_count = 8;
    config.feature_count = 5;
    config.use_multithreading = true;
    config.enable_debug_logging = false;
    config.log_predictions = true;
    config.log_performance_metrics = true;
    
    let mut engine = LDCEngine::with_config(config);
    
    // Use first 70% for training, last 30% for testing
    let training_size = (valid_features.len() as f64 * 0.7) as usize;
    let training_features: Vec<_> = valid_features[..training_size].iter().map(|f| (*f).clone()).collect();
    let training_ohlcv: Vec<_> = ohlcv_data[..training_size].to_vec();
    
    println!("📚 Training LDC with {} samples...", training_size);
    
    // Create training samples with 4-bar lookahead (Pine Script standard)
    engine.create_training_samples_from_ohlcv(
        &training_ohlcv,
        &training_features,
        4, // 4-bar lookahead for labeling
    )?;
    
    let (total_samples, long_samples, short_samples) = engine.get_training_stats();
    println!("   Training stats: {} total ({} long, {} short)", 
             total_samples, long_samples, short_samples);
    
    if total_samples < 20 {
        return Err("Not enough training samples generated. Need at least 20.".into());
    }
    
    // Test on remaining data
    let test_features: Vec<_> = valid_features[training_size..].iter().map(|f| (*f).clone()).collect();
    let test_ohlcv: Vec<_> = ohlcv_data[training_size..].to_vec();
    
    println!("🔮 Testing LDC on {} out-of-sample bars...", test_features.len());
    
    // Generate predictions for test data
    let predictions = engine.batch_predict_from_features(&test_features)?;
    
    // Analyze results
    println!("\n📊 LDC Results Analysis:");
    println!("{}", "=".repeat(40));
    
    let signals: Vec<f32> = predictions.iter().map(|p| p.signal).collect();
    let confidences: Vec<f32> = predictions.iter().map(|p| p.confidence).collect();
    
    if !signals.is_empty() {
        let signal_mean = signals.iter().sum::<f32>() / signals.len() as f32;
        let signal_std = {
            let variance = signals.iter()
                .map(|x| (x - signal_mean).powi(2))
                .sum::<f32>() / signals.len() as f32;
            variance.sqrt()
        };
        let signal_min = signals.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let signal_max = signals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        println!("Signal Statistics:");
        println!("  Mean: {:.4}", signal_mean);
        println!("  Std:  {:.4}", signal_std);
        println!("  Min:  {:.4}", signal_min);
        println!("  Max:  {:.4}", signal_max);
        
        // Count predictions by direction
        let long_predictions = predictions.iter().filter(|p| p.signal > 0.0).count();
        let short_predictions = predictions.iter().filter(|p| p.signal < 0.0).count();
        let neutral_predictions = predictions.iter().filter(|p| p.signal == 0.0).count();
        
        println!("\nPrediction Distribution:");
        println!("  Long:    {} ({:.1}%)", long_predictions, 100.0 * long_predictions as f32 / predictions.len() as f32);
        println!("  Short:   {} ({:.1}%)", short_predictions, 100.0 * short_predictions as f32 / predictions.len() as f32);
        println!("  Neutral: {} ({:.1}%)", neutral_predictions, 100.0 * neutral_predictions as f32 / predictions.len() as f32);
        
        // Confidence statistics
        if !confidences.is_empty() {
            let conf_mean = confidences.iter().sum::<f32>() / confidences.len() as f32;
            let conf_min = confidences.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let conf_max = confidences.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            println!("\nConfidence Statistics:");
            println!("  Mean: {:.4}", conf_mean);
            println!("  Min:  {:.4}", conf_min);
            println!("  Max:  {:.4}", conf_max);
        }
    }
    
    // Show some sample predictions with price context
    println!("\n📋 Sample Predictions (last 10):");
    println!("{:<8} {:<10} {:<8} {:<8} {:<10} {:<8}", "Bar", "Price", "Signal", "Conf", "Direction", "Neighbors");
    println!("{}", "-".repeat(60));
    
    let sample_count = predictions.len().min(10);
    let start_idx = predictions.len().saturating_sub(sample_count);
    
    for (i, prediction) in predictions[start_idx..].iter().enumerate() {
        let bar_idx = start_idx + i;
        let price = if bar_idx < test_ohlcv.len() {
            format!("{:.2}", test_ohlcv[bar_idx].close)
        } else {
            "N/A".to_string()
        };
        
        let direction = if prediction.signal > 0.0 { "LONG" } 
                       else if prediction.signal < 0.0 { "SHORT" } 
                       else { "NEUTRAL" };
        
        println!("{:<8} ${:<9} {:<8.2} {:<8.3} {:<10} {:<8}", 
                 bar_idx + training_size,
                 price,
                 prediction.signal,
                 prediction.confidence,
                 direction,
                 prediction.k_nearest_distances.len());
    }
    
    // Performance metrics
    let metrics = engine.get_performance_metrics();
    println!("\n⚡ Performance Metrics:");
    println!("  Total predictions: {}", metrics.total_predictions);
    println!("  Average time: {:.2}ms", metrics.average_prediction_time_ms);
    println!("  Parallel predictions: {}", metrics.parallel_predictions);
    
    println!("\n🎉 Real data test complete!");
    println!("💡 Try different time periods or symbols by modifying the fetch_btc_data() call");
    
    Ok(())
}