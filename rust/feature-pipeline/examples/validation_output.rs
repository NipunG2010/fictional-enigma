use feature_pipeline::{FeaturePipeline, OHLCV};
use serde_json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate the same test data as Python validation script
    let mut ohlcv_data = Vec::new();
    let mut price = 100.0;
    let mut timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
    
    for i in 0..30 {
        // Same logic as Python validation script
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
    }
    
    // Compute features
    let pipeline = FeaturePipeline::new(20);
    let features = pipeline.compute_features(&ohlcv_data)?;
    
    // Output in JSON format for easy parsing by Python
    let output = serde_json::json!({
        "ohlcv_data": ohlcv_data,
        "features": features,
        "config": {
            "window_size": 20,
            "rsi_period": 14,
            "ma_period": 20
        }
    });
    
    println!("{}", serde_json::to_string_pretty(&output)?);
    
    Ok(())
}