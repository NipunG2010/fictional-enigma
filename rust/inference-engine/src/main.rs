mod config;
mod daemon;
mod hmm;
mod runtime;
mod schema;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use config::RuntimeConfig;
use daemon::DaemonConfig;
use feature_pipeline::FeaturePipeline;
use polars::prelude::*;
use runtime::InferenceRuntime;
use tracing::{info, Level};
use std::time::Duration;

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

    /// Log output format.  "text" (default) is human-readable; "json" emits one JSON object per
    /// line, suitable for structured log aggregators (Loki, Datadog, CloudWatch, etc.).
    #[arg(long, default_value = "text")]
    log_format: String,

    /// Shortcut flag to run in daemon (serve) mode. Equivalent to `serve` subcommand.
    #[arg(long)]
    serve: bool,

    /// Port for the health check HTTP endpoint (used with --serve).
    #[arg(long, default_value_t = 9090)]
    serve_port: u16,

    /// Interval in seconds between pipeline runs (used with --serve).
    #[arg(long, default_value_t = 60u64)]
    serve_interval: u64,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the inference engine as a long-running daemon with periodic pipeline
    /// execution and a health check HTTP endpoint.
    Serve {
        /// Port for the health check HTTP server.
        #[arg(long, default_value_t = 9090)]
        port: u16,

        /// Interval in seconds between successive pipeline runs.
        #[arg(long, default_value_t = 60u64)]
        interval: u64,

        /// Path to the runtime configuration file.
        #[arg(short, long, default_value = DEFAULT_RUNTIME_CONFIG)]
        config: String,
    },

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
    init_tracing(&args.log_level, &args.log_format)?;

    // When --serve is passed as a top-level flag (legacy), treat it as Serve subcommand.
    let effective_command = if args.serve {
        Some(Command::Serve {
            port: args.serve_port,
            interval: args.serve_interval,
            config: args.config.clone(),
        })
    } else {
        args.command
    };

    match effective_command.unwrap_or(Command::RunRuntime) {
        Command::Serve { port, interval, config } => {
            let daemon_config = DaemonConfig {
                port,
                interval: Duration::from_secs(interval),
            };
            daemon::run_daemon(PathBuf::from(&config), daemon_config).await?;
            Ok(())
        }
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
            println!("");
            println!("Daemon commands:");
            println!("  serve             : run as long-running daemon with periodic pipeline execution");
            println!("                      and health check HTTP endpoint (default port 9090)");
            Ok(())
        }
    }
}

fn init_tracing(level: &str, format: &str) -> Result<()> {
    let log_level = match level {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };
    if format == "json" {
        tracing_subscriber::fmt().json().with_max_level(log_level).init();
    } else {
        tracing_subscriber::fmt().with_max_level(log_level).init();
    }
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

// ---------------------------------------------------------------------------
// Integration test — runs the real pipeline against sample data (non-mock).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;

    /// Path to the integration test config (resolved at compile time).
    fn test_config_path() -> PathBuf {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join("fixtures/integration-test.toml")
    }

    /// Path for the temporary output directory.
    fn test_output_dir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push("imp-integration-test");
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[tokio::test]
    async fn test_pipeline_bootstrap_and_run() {
        let config_path = test_config_path();
        assert!(
            config_path.exists(),
            "integration test config not found: {}",
            config_path.display()
        );

        // Load config and verify basic structure
        let config = RuntimeConfig::load(&config_path).unwrap();
        assert_eq!(config.runtime.symbol, "BTCUSDT");
        assert_eq!(config.runtime.max_rows, Some(16));

        // Bootstrap and run the pipeline
        let mut runtime = InferenceRuntime::bootstrap(config).await.unwrap();
        let summary = runtime.run().await.unwrap();
        runtime.shutdown().await.unwrap();

        // Validate summary structure
        assert!(summary.output_rows > 0, "pipeline must produce at least one output row");
        assert!(
            summary.fused_rows > 0 || summary.fallback_rows > 0,
            "pipeline must fuse or fall back on at least one row"
        );
        assert!(
            !summary.canonical_output_sha256.is_empty(),
            "output sha256 must be present"
        );
        assert!(summary.first_timestamp.is_some(), "first_timestamp must be set");
        assert!(summary.last_timestamp.is_some(), "last_timestamp must be set");
        assert_eq!(summary.runtime_mode, config::RuntimeMode::FallbackOnly);

        // Verify the output JSONL file exists and has content
        let output_path = test_output_dir().join("integration-test-output.jsonl");
        assert!(output_path.exists(), "output jsonl must exist");
        let contents = fs::read_to_string(&output_path).unwrap();
        assert!(!contents.is_empty(), "output jsonl must not be empty");

        // Parse first line and verify schema
        let first_line = contents.lines().next().unwrap();
        let record: schema::RuntimeOutputRecord =
            serde_json::from_str(first_line).expect("first output line must be valid JSON");
        assert_eq!(record.schema_version, "imp.runtime.output.v1");
        assert!(!record.audit.run_id.is_empty());
        assert!(!record.audit.correlation_id.is_empty());

        // Verify the summary file exists and matches
        let summary_path = test_output_dir().join("integration-test-summary.json");
        assert!(summary_path.exists(), "summary json must exist");
        let loaded_summary: schema::RuntimeRunSummary =
            serde_json::from_str(&fs::read_to_string(summary_path).unwrap()).unwrap();
        assert_eq!(loaded_summary.output_rows, summary.output_rows);
        assert_eq!(
            loaded_summary.canonical_output_sha256,
            summary.canonical_output_sha256
        );
    }

    #[tokio::test]
    async fn test_pipeline_with_fallback_only_mode() {
        let config_path = test_config_path();
        let config = RuntimeConfig::load(&config_path).unwrap();

        // In fallback_only mode, the HMM resolver should always use static weights
        let mut runtime = InferenceRuntime::bootstrap(config).await.unwrap();
        let summary = runtime.run().await.unwrap();
        runtime.shutdown().await.unwrap();

        // All fused rows use static fallback — no service calls
        assert!(
            summary.fallback_rows >= summary.fused_rows,
            "all fused rows should be counted as fallback in fallback_only mode"
        );
    }
}

#[cfg(test)]
mod log_format_tests {
    use super::*;

    #[test]
    fn test_log_format_arg_defaults_to_text() {
        let args = Args::try_parse_from(["inference-engine"]).unwrap();
        assert_eq!(args.log_format, "text");
    }

    #[test]
    fn test_log_format_arg_accepts_json() {
        let args = Args::try_parse_from(["inference-engine", "--log-format", "json"]).unwrap();
        assert_eq!(args.log_format, "json");
    }
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
