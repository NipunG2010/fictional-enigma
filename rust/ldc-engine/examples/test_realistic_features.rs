use feature_pipeline::{FeaturePipeline, OHLCV};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing with Realistic Price Movements");
    
    // Create more realistic test data with varying price movements
    let mut test_data = Vec::new();
    let mut price = 100.0;
    
    for i in 0..100 {
        // Create realistic price movements with some volatility
        let change = if i % 10 < 3 {
            // Downward movement
            -2.0 + (i as f64 * 0.1).sin() * 1.5
        } else if i % 10 < 7 {
            // Sideways movement
            (i as f64 * 0.2).sin() * 0.5
        } else {
            // Upward movement
            1.5 + (i as f64 * 0.15).cos() * 1.0
        };
        
        price += change;
        let high = price + 1.0 + (i as f64 * 0.1).sin() * 0.5;
        let low = price - 1.0 - (i as f64 * 0.1).cos() * 0.5;
        let volume = 1000.0 + (i as f64 * 0.05).sin() * 200.0;
        
        test_data.push(OHLCV {
            timestamp: 1000 + i as i64,
            open: price,
            high: high.max(price),
            low: low.min(price),
            close: price + change * 0.8, // Close slightly different from open
            volume: volume.max(100.0),
        });
        
        price = test_data.last().unwrap().close;
    }
    
    println!("📊 Testing with {} bars of realistic data", test_data.len());
    
    let pipeline = FeaturePipeline::new(20);
    let features = pipeline.compute_features(&test_data)?;
    
    println!("\n🔍 Feature Analysis (Last 10 bars):");
    println!("Index | RSI     | WT1     | WT2     | CCI     | ADX     |");
    println!("------|---------|---------|---------|---------|---------|");
    
    let start_idx = features.len().saturating_sub(10);
    for (i, feature) in features.iter().enumerate().skip(start_idx) {
        let rsi_str = match feature.rsi {
            Some(v) => format!("{:.2}", v),
            None => "None".to_string(),
        };
        let wt1_str = match feature.wavetrend_1 {
            Some(v) => format!("{:.2}", v),
            None => "None".to_string(),
        };
        let wt2_str = match feature.wavetrend_2 {
            Some(v) => format!("{:.2}", v),
            None => "None".to_string(),
        };
        let cci_str = match feature.cci {
            Some(v) => format!("{:.2}", v),
            None => "None".to_string(),
        };
        let adx_str = match feature.adx {
            Some(v) => format!("{:.2}", v),
            None => "None".to_string(),
        };
        
        println!("{:4} | {:7} | {:7} | {:7} | {:7} | {:7} |", 
                 i, rsi_str, wt1_str, wt2_str, cci_str, adx_str);
    }
    
    // Check for patterns
    let mut wt1_count = 0;
    let mut wt2_count = 0;
    let mut cci_count = 0;
    let mut adx_count = 0;
    let mut rsi_count = 0;
    
    for feature in &features {
        if feature.wavetrend_1.is_some() { wt1_count += 1; }
        if feature.wavetrend_2.is_some() { wt2_count += 1; }
        if feature.cci.is_some() { cci_count += 1; }
        if feature.adx.is_some() { adx_count += 1; }
        if feature.rsi.is_some() { rsi_count += 1; }
    }
    
    println!("\n📈 Feature Availability:");
    println!("RSI: {}/{} ({:.1}%)", rsi_count, features.len(), 
             (rsi_count as f64 / features.len() as f64) * 100.0);
    println!("WaveTrend1: {}/{} ({:.1}%)", wt1_count, features.len(), 
             (wt1_count as f64 / features.len() as f64) * 100.0);
    println!("WaveTrend2: {}/{} ({:.1}%)", wt2_count, features.len(), 
             (wt2_count as f64 / features.len() as f64) * 100.0);
    println!("CCI: {}/{} ({:.1}%)", cci_count, features.len(), 
             (cci_count as f64 / features.len() as f64) * 100.0);
    println!("ADX: {}/{} ({:.1}%)", adx_count, features.len(), 
             (adx_count as f64 / features.len() as f64) * 100.0);
    
    // Check RSI range
    let rsi_values: Vec<f64> = features.iter()
        .filter_map(|f| f.rsi)
        .collect();
    
    if !rsi_values.is_empty() {
        let min_rsi = rsi_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_rsi = rsi_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        println!("\n📊 RSI Range: {:.2} - {:.2}", min_rsi, max_rsi);
    }
    
    Ok(())
}
