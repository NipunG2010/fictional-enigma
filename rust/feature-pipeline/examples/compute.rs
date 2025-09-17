use feature_pipeline::{FeaturePipeline, OHLCV};
use polars::prelude::*;
use std::fs::{create_dir_all, File};
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

fn write_parquet(df: &mut DataFrame, path: &str) -> PolarsResult<u64> {
    let file = File::create(path).expect("create parquet file");
    ParquetWriter::new(file).finish(df)
}

fn main() -> anyhow::Result<()> {
    let sample_dir = Path::new("sample");
    let input_path = sample_dir.join("ohlcv.parquet");
    let output_path = sample_dir.join("features.parquet");

    if !sample_dir.exists() {
        create_dir_all(sample_dir)?;
    }

    if !input_path.exists() {
        let data = generate_synthetic_ohlcv(500);
        let pipeline = FeaturePipeline::new(50);
        let mut df = pipeline.ohlcv_to_dataframe(&data)?;
        let _ = write_parquet(&mut df, input_path.to_str().unwrap())?;
        println!("Wrote sample Parquet to {}", input_path.display());
    } else {
        println!("Sample Parquet already exists at {}", input_path.display());
    }

    let pipeline = FeaturePipeline::new(50);
    let df = pipeline.read_parquet(input_path.to_str().unwrap())?;
    let mut features_df = pipeline.compute_features_lazy(df)?;

    println!("Features preview:");
    println!("{:?}", features_df.head(Some(10)));

    let _ = write_parquet(&mut features_df, output_path.to_str().unwrap())?;
    println!("Wrote features Parquet to {}", output_path.display());

    Ok(())
}


