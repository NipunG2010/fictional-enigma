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

fn df_to_ohlcv(df: &DataFrame) -> anyhow::Result<Vec<OHLCV>> {
    let ts = df.column("timestamp")?.i64()?.into_no_null_iter();
    let open = df.column("open")?.f64()?.into_no_null_iter();
    let high = df.column("high")?.f64()?.into_no_null_iter();
    let low = df.column("low")?.f64()?.into_no_null_iter();
    let close = df.column("close")?.f64()?.into_no_null_iter();
    let volume = df.column("volume")?.f64()?.into_no_null_iter();

    let mut out = Vec::with_capacity(df.height());
    for (((((t, o), h), l), c), v) in ts.zip(open).zip(high).zip(low).zip(close).zip(volume) {
        out.push(OHLCV {
            timestamp: t,
            open: o,
            high: h,
            low: l,
            close: c,
            volume: v,
        });
    }
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let sample_dir = Path::new("sample");
    let input_path = sample_dir.join("ohlcv.parquet");
    let features_out_path = sample_dir.join("features.parquet");
    let signals_out_path = sample_dir.join("features_cli.parquet");

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

    // Compute features via lazy path for a DataFrame preview/save
    let mut features_df = pipeline.compute_features_lazy(df.clone())?;
    println!("Features preview:");
    println!("{:?}", features_df.head(Some(10)));
    let _ = write_parquet(&mut features_df, features_out_path.to_str().unwrap())?;
    println!("Wrote features Parquet to {}", features_out_path.display());

    // Also compute Features as typed Vec for signal generation
    let ohlcv = df_to_ohlcv(&df)?;
    let features_vec = pipeline.compute_features(&ohlcv)?;

    // Generate both MR and TSMOM signals
    let mut signals = pipeline.generate_signals(&features_vec)?;

    // Apply simple thresholding/filtering
    let mr_threshold = 0.1f64;
    let tsmom_threshold = 0.01f64;
    pipeline.apply_signal_filtering(&mut signals, mr_threshold, tsmom_threshold);

    // Convert signals to a DataFrame and write to Parquet
    let timestamps: Vec<i64> = signals.iter().map(|s| s.timestamp).collect();
    let mr: Vec<f64> = signals.iter().map(|s| s.s_mr.unwrap_or(f64::NAN)).collect();
    let tsmom: Vec<f64> = signals.iter().map(|s| s.s_tsmom.unwrap_or(f64::NAN)).collect();
    let mut signals_df = df![
        "timestamp" => timestamps,
        "s_mr" => mr,
        "s_tsmom" => tsmom,
    ]?;

    println!("Signals preview:");
    println!("{:?}", signals_df.head(Some(10)));

    let _ = write_parquet(&mut signals_df, signals_out_path.to_str().unwrap())?;
    println!("Wrote signals Parquet to {}", signals_out_path.display());

    Ok(())
}


