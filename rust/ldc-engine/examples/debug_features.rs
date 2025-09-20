use feature_pipeline::{FeaturePipeline, OHLCV};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debugging Feature Pipeline Issues");
    
    // Create simple test data with known values
    let test_data = vec![
        OHLCV { timestamp: 1000, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000.0 },
        OHLCV { timestamp: 1001, open: 102.0, high: 108.0, low: 98.0, close: 106.0, volume: 1200.0 },
        OHLCV { timestamp: 1002, open: 106.0, high: 110.0, low: 104.0, close: 108.0, volume: 1100.0 },
        OHLCV { timestamp: 1003, open: 108.0, high: 112.0, low: 106.0, close: 110.0, volume: 1300.0 },
        OHLCV { timestamp: 1004, open: 110.0, high: 115.0, low: 108.0, close: 113.0, volume: 1400.0 },
        OHLCV { timestamp: 1005, open: 113.0, high: 118.0, low: 111.0, close: 116.0, volume: 1500.0 },
        OHLCV { timestamp: 1006, open: 116.0, high: 120.0, low: 114.0, close: 118.0, volume: 1600.0 },
        OHLCV { timestamp: 1007, open: 118.0, high: 122.0, low: 116.0, close: 120.0, volume: 1700.0 },
        OHLCV { timestamp: 1008, open: 120.0, high: 125.0, low: 118.0, close: 123.0, volume: 1800.0 },
        OHLCV { timestamp: 1009, open: 123.0, high: 128.0, low: 121.0, close: 126.0, volume: 1900.0 },
        OHLCV { timestamp: 1010, open: 126.0, high: 130.0, low: 124.0, close: 128.0, volume: 2000.0 },
        OHLCV { timestamp: 1011, open: 128.0, high: 132.0, low: 126.0, close: 130.0, volume: 2100.0 },
        OHLCV { timestamp: 1012, open: 130.0, high: 135.0, low: 128.0, close: 133.0, volume: 2200.0 },
        OHLCV { timestamp: 1013, open: 133.0, high: 138.0, low: 131.0, close: 136.0, volume: 2300.0 },
        OHLCV { timestamp: 1014, open: 136.0, high: 140.0, low: 134.0, close: 138.0, volume: 2400.0 },
        OHLCV { timestamp: 1015, open: 138.0, high: 142.0, low: 136.0, close: 140.0, volume: 2500.0 },
        OHLCV { timestamp: 1016, open: 140.0, high: 145.0, low: 138.0, close: 143.0, volume: 2600.0 },
        OHLCV { timestamp: 1017, open: 143.0, high: 148.0, low: 141.0, close: 146.0, volume: 2700.0 },
        OHLCV { timestamp: 1018, open: 146.0, high: 150.0, low: 144.0, close: 148.0, volume: 2800.0 },
        OHLCV { timestamp: 1019, open: 148.0, high: 152.0, low: 146.0, close: 150.0, volume: 2900.0 },
        OHLCV { timestamp: 1020, open: 150.0, high: 155.0, low: 148.0, close: 153.0, volume: 3000.0 },
        OHLCV { timestamp: 1021, open: 153.0, high: 158.0, low: 151.0, close: 156.0, volume: 3100.0 },
        OHLCV { timestamp: 1022, open: 156.0, high: 160.0, low: 154.0, close: 158.0, volume: 3200.0 },
        OHLCV { timestamp: 1023, open: 158.0, high: 162.0, low: 156.0, close: 160.0, volume: 3300.0 },
        OHLCV { timestamp: 1024, open: 160.0, high: 165.0, low: 158.0, close: 163.0, volume: 3400.0 },
    ];
    
    println!("📊 Testing with {} bars of data", test_data.len());
    
    // Test with different window sizes
    let pipeline = FeaturePipeline::new(20);
    let features = pipeline.compute_features(&test_data)?;
    
    println!("\n🔍 Feature Analysis:");
    println!("Index | RSI     | WT1     | WT2     | CCI     | ADX     |");
    println!("------|---------|---------|---------|---------|---------|");
    
    for (i, feature) in features.iter().enumerate() {
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
    
    for feature in &features {
        if feature.wavetrend_1.is_some() { wt1_count += 1; }
        if feature.wavetrend_2.is_some() { wt2_count += 1; }
        if feature.cci.is_some() { cci_count += 1; }
        if feature.adx.is_some() { adx_count += 1; }
    }
    
    println!("\n📈 Feature Availability:");
    println!("WaveTrend1: {}/{} ({:.1}%)", wt1_count, features.len(), 
             (wt1_count as f64 / features.len() as f64) * 100.0);
    println!("WaveTrend2: {}/{} ({:.1}%)", wt2_count, features.len(), 
             (wt2_count as f64 / features.len() as f64) * 100.0);
    println!("CCI: {}/{} ({:.1}%)", cci_count, features.len(), 
             (cci_count as f64 / features.len() as f64) * 100.0);
    println!("ADX: {}/{} ({:.1}%)", adx_count, features.len(), 
             (adx_count as f64 / features.len() as f64) * 100.0);
    
    // Check for constant values
    if let Some(first_adx) = features.iter().find(|f| f.adx.is_some()).and_then(|f| f.adx) {
        let all_same_adx = features.iter()
            .filter(|f| f.adx.is_some())
            .all(|f| f.adx.unwrap() == first_adx);
        println!("\n⚠️  ADX Issue: All values are the same ({:.2})? {}", first_adx, all_same_adx);
    }
    
    if let Some(first_cci) = features.iter().find(|f| f.cci.is_some()).and_then(|f| f.cci) {
        let all_same_cci = features.iter()
            .filter(|f| f.cci.is_some())
            .all(|f| f.cci.unwrap() == first_cci);
        println!("⚠️  CCI Issue: All values are the same ({:.2})? {}", first_cci, all_same_cci);
    }
    
    Ok(())
}
