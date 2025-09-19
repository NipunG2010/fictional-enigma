use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber;
use feature_pipeline::FeaturePipeline;
use polars::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,
    
    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Subcommands
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compute features from a Parquet file and write output Parquet
    ComputeFeatures {
        /// Input Parquet path
        #[arg(short, long, default_value = "sample/ohlcv.parquet")]
        input: String,

        /// Output Parquet path
        #[arg(short, long, default_value = "sample/features_cli.parquet")]
        output: String,

        /// Window size for indicators
        #[arg(long, default_value_t = 50usize)]
        window: usize,

        /// RSI period
        #[arg(long, default_value_t = 14usize)]
        rsi: usize,

        /// Moving average period
        #[arg(long, default_value_t = 20usize)]
        ma: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    let log_level = match args.log_level.as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };
    
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .init();
    
    // Handle CLI subcommands first
    if let Some(command) = args.command {
        match command {
            Command::ComputeFeatures { input, output, window, rsi, ma } => {
                run_compute_features(&input, &output, window, rsi, ma)?;
                return Ok(());
            }
        }
    }

    info!("Starting IMP Inference Engine v{}", env!("CARGO_PKG_VERSION"));
    info!("Configuration file: {}", args.config);
    
    // TODO: Load configuration
    // TODO: Initialize components
    // TODO: Start HTTP server
    // TODO: Start signal processing loop
    
    info!("Inference engine started successfully");
    
    // Keep the service running
    tokio::signal::ctrl_c().await?;
    info!("Shutting down inference engine");
    
    Ok(())
}

fn run_compute_features(input: &str, output: &str, window: usize, rsi: usize, ma: usize) -> anyhow::Result<()> {
    info!("Computing features from {} -> {} (window={}, rsi={}, ma={})", input, output, window, rsi, ma);

    let pipeline = FeaturePipeline::with_periods(window, rsi, ma);
    let df = pipeline.read_parquet(input)?;
    let mut features_df = pipeline.compute_features_lazy(df)?;

    // Print preview
    println!("Features preview (CLI):\n{:?}", features_df.head(Some(10)));

    // Write Parquet
    use std::fs::File;
    let file = File::create(output)?;
    ParquetWriter::new(file).finish(&mut features_df)?;
    info!("Wrote features Parquet to {}", output);
    Ok(())
}
