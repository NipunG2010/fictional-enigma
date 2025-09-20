use feature_pipeline::{FeaturePipeline, OHLCV};
use std::path::Path;

fn generate_synthetic_ohlcv(n: usize) -> Vec<OHLCV> {
    let mut data = Vec::with_capacity(n);
    let mut price = 100.0f64;
    for i in 0..n {
        let ts = 1_700_000_000 + (i as i64) * 60; // 1-minute steps
        // simple random walk-ish
        let step = ((i as f64).sin() * 0.2) + ((i as f64).cos() * 0.1);
        price = (price + step).max(1.0);
        let open = price * 0.999;
        let high = price * 1.006;
        let low = price * 0.994;
        let close = price;
        let volume = 1_000.0 + (i as f64) * 2.0;
        data.push(OHLCV { timestamp: ts, open, high, low, close, volume });
    }
    data
}

fn main() -> anyhow::Result<()> {
    let base_path = "partitioned_data";
    let symbol = "BTCUSDT";
    let interval = "5m";
    
    println!("Generating synthetic OHLCV data...");
    let data = generate_synthetic_ohlcv(1000);
    
    println!("Creating feature pipeline...");
    let pipeline = FeaturePipeline::new(50);
    
    println!("Processing data with partitioned storage...");
    let (ohlcv_path, features_path, signals_path) = pipeline.process_and_save_partitioned(
        &data, 
        base_path, 
        symbol, 
        interval
    )?;
    
    println!("✅ Partitioned storage completed!");
    println!("📁 OHLCV saved to: {}", ohlcv_path);
    println!("📁 Features saved to: {}", features_path);
    println!("📁 Signals saved to: {}", signals_path);
    
    // Show directory structure
    let partition_dir = Path::new(base_path)
        .join(format!("symbol={}", symbol))
        .join("date=2024-09-19") // Today's date
        .join(format!("interval={}", interval));
    
    println!("\n📂 Directory structure created:");
    println!("{}", partition_dir.display());
    
    // List files in the partition
    if partition_dir.exists() {
        println!("\n📄 Files in partition:");
        for entry in std::fs::read_dir(&partition_dir)? {
            let entry = entry?;
            println!("  - {}", entry.file_name().to_string_lossy());
        }
    }
    
    Ok(())
}

