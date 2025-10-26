//! CI Helper Binary
//! 
//! Utility for CI/CD integration tasks such as generating test configurations,
//! processing test results, and creating status reports.

use clap::{Arg, Command};
use end_to_end_tests::TestReport;
use serde_json;
use std::fs;
use std::path::PathBuf;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let matches = Command::new("ci-helper")
        .version(end_to_end_tests::VERSION)
        .about("CI/CD Integration Helper for End-to-End Tests")
        .subcommand(
            Command::new("generate-config")
                .about("Generate test configuration for CI environment")
                .arg(
                    Arg::new("environment")
                        .short('e')
                        .long("environment")
                        .value_name("ENV")
                        .help("Target environment (ci, local, performance)")
                        .default_value("ci"),
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("FILE")
                        .help("Output configuration file")
                        .default_value("test_config.toml"),
                ),
        )
        .subcommand(
            Command::new("process-results")
                .about("Process test results for CI integration")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("FILE")
                        .help("Input test report JSON file")
                        .required(true),
                )
                .arg(
                    Arg::new("output-dir")
                        .short('o')
                        .long("output-dir")
                        .value_name("DIR")
                        .help("Output directory for processed results")
                        .default_value("ci-artifacts"),
                )
                .arg(
                    Arg::new("baseline")
                        .short('b')
                        .long("baseline")
                        .value_name("FILE")
                        .help("Baseline report for comparison"),
                ),
        )
        .subcommand(
            Command::new("check-regressions")
                .about("Check for performance regressions")
                .arg(
                    Arg::new("current")
                        .short('c')
                        .long("current")
                        .value_name("FILE")
                        .help("Current test report")
                        .required(true),
                )
                .arg(
                    Arg::new("baseline")
                        .short('b')
                        .long("baseline")
                        .value_name("FILE")
                        .help("Baseline test report")
                        .required(true),
                )
                .arg(
                    Arg::new("fail-on-regression")
                        .long("fail-on-regression")
                        .help("Exit with error code if regressions are detected")
                        .action(clap::ArgAction::SetTrue),
                ),
        )
        .subcommand(
            Command::new("generate-status")
                .about("Generate status report for GitHub Actions")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("FILE")
                        .help("Input test report JSON file")
                        .required(true),
                )
                .arg(
                    Arg::new("format")
                        .short('f')
                        .long("format")
                        .value_name("FORMAT")
                        .help("Output format (github, json, text)")
                        .default_value("github"),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("generate-config", sub_matches)) => {
            let environment = sub_matches.get_one::<String>("environment").unwrap();
            let output_file = sub_matches.get_one::<String>("output").unwrap();
            generate_test_config(environment, output_file).await?;
        }
        Some(("process-results", sub_matches)) => {
            let input_file = sub_matches.get_one::<String>("input").unwrap();
            let output_dir = sub_matches.get_one::<String>("output-dir").unwrap();
            let baseline_file = sub_matches.get_one::<String>("baseline");
            process_test_results(input_file, output_dir, baseline_file).await?;
        }
        Some(("check-regressions", sub_matches)) => {
            let current_file = sub_matches.get_one::<String>("current").unwrap();
            let baseline_file = sub_matches.get_one::<String>("baseline").unwrap();
            let fail_on_regression = sub_matches.get_flag("fail-on-regression");
            check_performance_regressions(current_file, baseline_file, fail_on_regression).await?;
        }
        Some(("generate-status", sub_matches)) => {
            let input_file = sub_matches.get_one::<String>("input").unwrap();
            let format = sub_matches.get_one::<String>("format").unwrap();
            generate_status_report(input_file, format).await?;
        }
        _ => {
            error!("No subcommand provided. Use --help for usage information.");
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn generate_test_config(environment: &str, output_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating test configuration for environment: {}", environment);

    let config_content = match environment {
        "ci" => generate_ci_config(),
        "local" => generate_local_config(),
        "performance" => generate_performance_config(),
        _ => {
            error!("Unknown environment: {}. Supported: ci, local, performance", environment);
            return Err("Invalid environment".into());
        }
    };

    fs::write(output_file, config_content)?;
    info!("Test configuration written to: {}", output_file);

    Ok(())
}

fn generate_ci_config() -> String {
    r#"# CI Environment Test Configuration
[pipeline_tests]
test_symbols = ["BTCUSDT", "ETHUSDT"]
test_duration_hours = 1
data_interval = "5m"
include_edge_cases = true
validate_against_reference = false

[failure_tests]
test_hmm_failures = true
test_redis_failures = true
test_kafka_failures = true
test_data_corruption = true
failure_duration_seconds = 10
recovery_timeout_seconds = 30

[performance_tests]
max_end_to_end_latency_ms = 150  # Relaxed for CI environment
min_throughput_signals_per_second = 3.0
max_memory_usage_mb = 1024
concurrent_symbols = 2
test_duration_minutes = 2

[execution]
output_dir = "test-results"
max_parallel_tests = 2  # Conservative for CI
test_timeout_seconds = 300
verbose_logging = true

[validation]
feature_tolerance = 0.001
signal_tolerance = 0.01
performance_tolerance = 0.2  # More lenient for CI
"#.to_string()
}

fn generate_local_config() -> String {
    r#"# Local Development Test Configuration
[pipeline_tests]
test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
test_duration_hours = 2
data_interval = "5m"
include_edge_cases = true
validate_against_reference = true

[failure_tests]
test_hmm_failures = true
test_redis_failures = true
test_kafka_failures = true
test_data_corruption = true
failure_duration_seconds = 30
recovery_timeout_seconds = 60

[performance_tests]
max_end_to_end_latency_ms = 100
min_throughput_signals_per_second = 10.0
max_memory_usage_mb = 512
concurrent_symbols = 5
test_duration_minutes = 5

[execution]
output_dir = "test-results"
max_parallel_tests = 4
test_timeout_seconds = 600
verbose_logging = false

[validation]
feature_tolerance = 0.001
signal_tolerance = 0.01
performance_tolerance = 0.1
"#.to_string()
}

fn generate_performance_config() -> String {
    r#"# Performance Testing Configuration
[pipeline_tests]
test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
test_duration_hours = 4
data_interval = "5m"
include_edge_cases = true
validate_against_reference = true

[failure_tests]
test_hmm_failures = true
test_redis_failures = true
test_kafka_failures = true
test_data_corruption = true
failure_duration_seconds = 60
recovery_timeout_seconds = 120

[performance_tests]
max_end_to_end_latency_ms = 50  # Strict performance requirements
min_throughput_signals_per_second = 20.0
max_memory_usage_mb = 256
concurrent_symbols = 10
test_duration_minutes = 15

[execution]
output_dir = "test-results"
max_parallel_tests = 8
test_timeout_seconds = 1200
verbose_logging = false

[validation]
feature_tolerance = 0.0001  # Strict validation
signal_tolerance = 0.001
performance_tolerance = 0.05
"#.to_string()
}

async fn process_test_results(
    input_file: &str,
    output_dir: &str,
    baseline_file: Option<&String>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Processing test results from: {}", input_file);

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Load test report
    let content = fs::read_to_string(input_file)?;
    let report: TestReport = serde_json::from_str(&content)?;

    // Generate CI summary
    let ci_summary = report.generate_ci_summary();
    let ci_summary_path = PathBuf::from(output_dir).join("ci_summary.json");
    let ci_summary_json = serde_json::to_string_pretty(&ci_summary)?;
    fs::write(&ci_summary_path, ci_summary_json)?;
    info!("CI summary written to: {}", ci_summary_path.display());

    // Generate timestamped reports
    let (json_path, html_path) = report.save_timestamped_reports(output_dir)?;
    info!("Timestamped reports saved to: {} and {}", json_path.display(), html_path.display());

    // Perform comparison if baseline is provided
    if let Some(baseline_path) = baseline_file {
        info!("Comparing against baseline: {}", baseline_path);
        
        let baseline_content = fs::read_to_string(baseline_path)?;
        let baseline_report: TestReport = serde_json::from_str(&baseline_content)?;
        
        let comparison = TestReport::compare_reports(&report, &baseline_report);
        let comparison_path = PathBuf::from(output_dir).join("comparison_report.json");
        let comparison_json = serde_json::to_string_pretty(&comparison)?;
        fs::write(&comparison_path, comparison_json)?;
        
        info!("Comparison report written to: {}", comparison_path.display());
        
        // Log comparison summary
        info!("Comparison Summary:");
        info!("  Status: {:?}", comparison.summary.status);
        info!("  Pass rate change: {:.1}%", comparison.pass_rate_change * 100.0);
        info!("  New failures: {}", comparison.new_failures.len());
        info!("  Resolved failures: {}", comparison.resolved_failures.len());
        info!("  Performance regressions: {}", comparison.performance_regressions.len());
    }

    Ok(())
}

async fn check_performance_regressions(
    current_file: &str,
    baseline_file: &str,
    fail_on_regression: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Checking for performance regressions");
    info!("  Current: {}", current_file);
    info!("  Baseline: {}", baseline_file);

    // Load reports
    let current_content = fs::read_to_string(current_file)?;
    let baseline_content = fs::read_to_string(baseline_file)?;
    
    let current_report: TestReport = serde_json::from_str(&current_content)?;
    let baseline_report: TestReport = serde_json::from_str(&baseline_content)?;

    // Perform comparison
    let comparison = TestReport::compare_reports(&current_report, &baseline_report);

    // Check for regressions
    let mut has_critical_regressions = false;
    let mut has_any_regressions = false;

    if !comparison.performance_regressions.is_empty() {
        has_any_regressions = true;
        warn!("Performance regressions detected:");
        
        for regression in &comparison.performance_regressions {
            let severity_str = match regression.severity {
                end_to_end_tests::RegressionSeverity::Critical => {
                    has_critical_regressions = true;
                    "CRITICAL"
                }
                end_to_end_tests::RegressionSeverity::High => "HIGH",
                end_to_end_tests::RegressionSeverity::Medium => "MEDIUM",
                end_to_end_tests::RegressionSeverity::Low => "LOW",
            };
            
            warn!("  {} ({}): {:.1}% change - {}", 
                  regression.metric_name,
                  severity_str,
                  regression.percentage_change,
                  regression.description);
        }
    }

    // Check pass rate regression
    if comparison.pass_rate_change < -0.05 {
        has_any_regressions = true;
        if comparison.pass_rate_change < -0.15 {
            has_critical_regressions = true;
        }
        warn!("Pass rate regression: {:.1}% decrease", comparison.pass_rate_change.abs() * 100.0);
    }

    // Report results
    if has_any_regressions {
        if has_critical_regressions {
            error!("Critical performance regressions detected!");
        } else {
            warn!("Performance regressions detected (non-critical)");
        }
        
        if fail_on_regression {
            error!("Failing due to --fail-on-regression flag");
            std::process::exit(1);
        }
    } else {
        info!("No performance regressions detected");
    }

    Ok(())
}

async fn generate_status_report(input_file: &str, format: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating status report from: {}", input_file);

    // Load test report
    let content = fs::read_to_string(input_file)?;
    let report: TestReport = serde_json::from_str(&content)?;

    match format {
        "github" => generate_github_status(&report),
        "json" => generate_json_status(&report)?,
        "text" => generate_text_status(&report),
        _ => {
            error!("Unknown format: {}. Supported: github, json, text", format);
            return Err("Invalid format".into());
        }
    }

    Ok(())
}

fn generate_github_status(report: &TestReport) {
    let (total_tests, passed_tests, failed_tests) = report.overall_stats();
    let pass_rate = report.summary.overall_pass_rate * 100.0;
    
    // Generate GitHub Actions output
    println!("::set-output name=total_tests::{}", total_tests);
    println!("::set-output name=passed_tests::{}", passed_tests);
    println!("::set-output name=failed_tests::{}", failed_tests);
    println!("::set-output name=pass_rate::{:.1}", pass_rate);
    println!("::set-output name=duration_minutes::{:.1}", report.summary.total_duration_minutes);
    println!("::set-output name=health_score::{:.1}", report.summary.system_health_score * 100.0);

    // Generate status messages
    if failed_tests == 0 {
        println!("::notice title=Tests Passed::All {} tests passed ({:.1}% pass rate)", total_tests, pass_rate);
    } else {
        println!("::error title=Tests Failed::{} out of {} tests failed ({:.1}% pass rate)", failed_tests, total_tests, pass_rate);
    }

    if report.summary.performance_violations > 0 {
        println!("::warning title=Performance Issues::{} performance violations detected", report.summary.performance_violations);
    }

    if report.summary.system_health_score < 0.8 {
        println!("::warning title=System Health::System health score is {:.1}% (below 80%)", report.summary.system_health_score * 100.0);
    }
}

fn generate_json_status(report: &TestReport) -> Result<(), Box<dyn std::error::Error>> {
    let ci_summary = report.generate_ci_summary();
    let json = serde_json::to_string_pretty(&ci_summary)?;
    println!("{}", json);
    Ok(())
}

fn generate_text_status(report: &TestReport) {
    let (total_tests, passed_tests, failed_tests) = report.overall_stats();
    
    println!("=== End-to-End Test Status Report ===");
    println!("Session ID: {}", report.session_id);
    println!("Generated: {}", chrono::DateTime::from_timestamp(report.generated_at, 0).unwrap_or_default().format("%Y-%m-%d %H:%M:%S UTC"));
    println!();
    println!("Test Results:");
    println!("  Total Tests: {}", total_tests);
    println!("  Passed: {}", passed_tests);
    println!("  Failed: {}", failed_tests);
    println!("  Pass Rate: {:.1}%", report.summary.overall_pass_rate * 100.0);
    println!();
    println!("Execution Summary:");
    println!("  Duration: {:.1} minutes", report.summary.total_duration_minutes);
    println!("  Critical Failures: {}", report.summary.critical_failures);
    println!("  Performance Violations: {}", report.summary.performance_violations);
    println!("  System Health Score: {:.1}%", report.summary.system_health_score * 100.0);
    
    if !report.recommendations.is_empty() {
        println!();
        println!("Recommendations:");
        for (i, recommendation) in report.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, recommendation);
        }
    }
}