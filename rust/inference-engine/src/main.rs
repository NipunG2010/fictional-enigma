mod config;
mod hmm;
mod runtime;
mod schema;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use config::RuntimeConfig;
use feature_pipeline::FeaturePipeline;
use polars::prelude::*;
use runtime::InferenceRuntime;
use tracing::{info, Level};

const DEFAULT_RUNTIME_CONFIG: &str = "inference-engine/fixtures/local-smoke.toml";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Runtime configuration file path for the batch runtime commands.
    #[arg(short, long, default_value = DEFAULT_RUNTIME_CONFIG)]
    config: String,

    /// Log level.
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compute features from an OHLCV parquet file and write them to parquet.
    ComputeFeatures {
        #[arg(short, long, default_value = "sample/ohlcv.parquet")]
        input: String,
        #[arg(short, long, default_value = "sample/features_cli.parquet")]
        output: String,
        #[arg(long, default_value_t = 50usize)]
        window: usize,
        #[arg(long, default_value_t = 14usize)]
        rsi: usize,
        #[arg(long, default_value_t = 20usize)]
        ma: usize,
    },

    /// Run the first-class offline batch runtime using the supplied config.
    RunRuntime,

    /// Run the deterministic bundled smoke configuration and compare the output to the expected fixture.
    Smoke {
        #[arg(long)]
        config: Option<String>,
        #[arg(long)]
        output: Option<String>,
    },

    /// Print the supported runtime modes.
    DescribeModes,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing(&args.log_level)?;

    match args.command.unwrap_or(Command::RunRuntime) {
        Command::ComputeFeatures {
            input,
            output,
            window,
            rsi,
            ma,
        } => run_compute_features(&input, &output, window, rsi, ma),
        Command::RunRuntime => run_runtime_command(Path::new(&args.config)).await,
        Command::Smoke { config, output } => {
            let config_path = config.unwrap_or_else(|| args.config.clone());
            run_smoke_command(Path::new(&config_path), output.map(PathBuf::from)).await
        }
        Command::DescribeModes => {
            println!("Supported runtime modes:");
            println!("  - offline_batch   : first-class MVP batch orchestrator");
            println!("  - local_smoke     : deterministic fixture-driven batch run with fallback weights");
            println!("  - integration_hmm : batch runtime with live HMM service integration and explicit fallback handling");
            println!("  - fallback_only   : batch runtime with deterministic static weights and no HMM service dependency");
            Ok(())
        }
    }
}

fn init_tracing(level: &str) -> Result<()> {
    let log_level = match level {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    tracing_subscriber::fmt().with_max_level(log_level).init();
    Ok(())
}

async fn run_runtime_command(config_path: &Path) -> Result<()> {
    let config = RuntimeConfig::load(config_path)
        .with_context(|| format!("failed to load runtime config {}", config_path.display()))?;

    info!(
        mode = %config.runtime.mode,
        symbol = %config.runtime.symbol,
        input = %config.input.market_data.display(),
        output = %config.output.canonical_jsonl.display(),
        "starting runtime"
    );

    let mut runtime = InferenceRuntime::bootstrap(config).await?;
    let summary = runtime.run().await?;
    runtime.shutdown().await?;

    info!(
        rows = summary.output_rows,
        fused_rows = summary.fused_rows,
        emitted_rows = summary.emitted_rows,
        fallback_rows = summary.fallback_rows,
        hash = %summary.canonical_output_sha256,
        "runtime completed successfully"
    );

    println!("Runtime summary:\n{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

async fn run_smoke_command(config_path: &Path, output_override: Option<PathBuf>) -> Result<()> {
    let mut config = RuntimeConfig::load(config_path)
        .with_context(|| format!("failed to load smoke config {}", config_path.display()))?;

    let expected_output = config
        .validation
        .expected_output_jsonl
        .clone()
        .context("smoke config is missing validation.expected_output_jsonl")?;
    let expected_summary = config
        .validation
        .expected_summary_json
        .clone()
        .context("smoke config is missing validation.expected_summary_json")?;

    let actual_output = output_override.unwrap_or_else(|| {
        std::env::temp_dir().join("imp-local-smoke.actual.jsonl")
    });
    let actual_summary = actual_output.with_extension("summary.json");
    config.output.canonical_jsonl = actual_output.clone();
    config.output.summary_json = actual_summary.clone();
    config.output.overwrite = true;

    let mut runtime = InferenceRuntime::bootstrap(config).await?;
    let summary = runtime.run().await?;
    runtime.shutdown().await?;

    let expected_bytes = fs::read(&expected_output)
        .with_context(|| format!("failed to read expected output {}", expected_output.display()))?;
    let actual_bytes = fs::read(&actual_output)
        .with_context(|| format!("failed to read actual output {}", actual_output.display()))?;
    if expected_bytes != actual_bytes {
        anyhow::bail!(
            "smoke output mismatch: expected {} but got {}",
            expected_output.display(),
            actual_output.display()
        );
    }

    let expected_summary_bytes = fs::read(&expected_summary)
        .with_context(|| format!("failed to read expected summary {}", expected_summary.display()))?;
    let actual_summary_bytes = fs::read(&actual_summary)
        .with_context(|| format!("failed to read actual summary {}", actual_summary.display()))?;
    if expected_summary_bytes != actual_summary_bytes {
        anyhow::bail!(
            "smoke summary mismatch: expected {} but got {}",
            expected_summary.display(),
            actual_summary.display()
        );
    }

    println!(
        "Smoke run passed. Output matches {}.\n{}",
        expected_output.display(),
        serde_json::to_string_pretty(&summary)?
    );

    Ok(())
}

fn run_compute_features(
    input: &str,
    output: &str,
    window: usize,
    rsi: usize,
    ma: usize,
) -> Result<()> {
    info!(
        input,
        output,
        window,
        rsi,
        ma,
        "computing features"
    );

    let pipeline = FeaturePipeline::with_periods(window, rsi, ma);
    let df = pipeline.read_parquet(input)?;
    let mut features_df = pipeline.compute_features_lazy(df)?;

    println!("Features preview (CLI):\n{:?}", features_df.head(Some(10)));

    let file = std::fs::File::create(output)
        .with_context(|| format!("failed to create feature output {}", output))?;
    ParquetWriter::new(file).finish(&mut features_df)?;
    info!(output, "wrote features parquet");
    Ok(())
}
