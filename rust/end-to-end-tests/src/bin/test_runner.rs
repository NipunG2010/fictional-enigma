//! End-to-End Test Runner Binary
//! 
//! Command-line interface for running comprehensive end-to-end tests
//! of the IMP trading system pipeline.

use clap::{Arg, Command};
use end_to_end_tests::{TestConfig, TestHarness, DEFAULT_CONFIG_FILE};
use std::path::PathBuf;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let matches = Command::new("test-runner")
        .version(end_to_end_tests::VERSION)
        .about("IMP End-to-End Test Runner")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .default_value(DEFAULT_CONFIG_FILE),
        )
        .arg(
            Arg::new("suite")
                .short('s')
                .long("suite")
                .value_name("SUITE")
                .help("Test suite to run (pipeline, failure, performance, all)")
                .default_value("all"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("DIR")
                .help("Output directory for test results")
                .default_value("test_results"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose logging")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("generate-config")
                .long("generate-config")
                .help("Generate default configuration file and exit")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // Handle config generation
    if matches.get_flag("generate-config") {
        let config_path = matches.get_one::<String>("config").unwrap();
        let default_config = TestConfig::default();
        
        default_config.save_to_file(config_path)?;
        info!("Generated default configuration at: {}", config_path);
        return Ok(());
    }

    // Load configuration
    let config_path = matches.get_one::<String>("config").unwrap();
    let config_path = PathBuf::from(config_path);
    
    let mut config = if config_path.exists() {
        info!("Loading configuration from: {}", config_path.display());
        TestConfig::from_file(&config_path)?
    } else {
        info!("Configuration file not found, using defaults");
        TestConfig::default()
    };

    // Override config with command line arguments
    if let Some(output_dir) = matches.get_one::<String>("output") {
        config.execution.output_dir = output_dir.clone();
    }

    if matches.get_flag("verbose") {
        config.execution.verbose_logging = true;
    }

    // Create test harness
    info!("Initializing test harness...");
    let mut harness = TestHarness::new(config).await?;
    
    info!("Starting test execution with session ID: {}", harness.session_id());

    // Run specified test suite
    let suite = matches.get_one::<String>("suite").unwrap();
    let report = match suite.as_str() {
        "pipeline" => {
            info!("Running pipeline integration tests only");
            let results = harness.run_pipeline_tests().await?;
            harness.generate_report(&[results], std::time::Duration::from_secs(0)).await?
        }
        "failure" => {
            info!("Running failure scenario tests only");
            let results = harness.run_failure_tests().await?;
            harness.generate_report(&[results], std::time::Duration::from_secs(0)).await?
        }
        "performance" => {
            info!("Running performance validation tests only");
            let results = harness.run_performance_tests().await?;
            harness.generate_report(&[results], std::time::Duration::from_secs(0)).await?
        }
        "all" => {
            info!("Running all test suites");
            harness.run_all_tests().await?
        }
        _ => {
            error!("Unknown test suite: {}", suite);
            return Err("Invalid test suite specified".into());
        }
    };

    // Save reports
    let output_dir = PathBuf::from(&harness.config().execution.output_dir);
    std::fs::create_dir_all(&output_dir)?;

    if harness.config().execution.generate_json_reports {
        let json_path = output_dir.join("test_report.json");
        report.save_json(&json_path)?;
        info!("JSON report saved to: {}", json_path.display());
    }

    if harness.config().execution.generate_html_reports {
        let html_path = output_dir.join("test_report.html");
        report.save_html(&html_path)?;
        info!("HTML report saved to: {}", html_path.display());
    }

    // Print summary
    let (total, passed, failed) = report.overall_stats();
    info!("Test execution completed!");
    info!("Total tests: {}, Passed: {}, Failed: {}", total, passed, failed);
    info!("Overall pass rate: {:.1}%", report.summary.overall_pass_rate * 100.0);
    info!("System health score: {:.1}%", report.summary.system_health_score * 100.0);

    if failed > 0 {
        error!("Some tests failed. Check the detailed report for more information.");
        std::process::exit(1);
    }

    Ok(())
}