//! Test Report Generator Binary
//! 
//! Utility for generating test reports from existing test result files
//! and performing analysis on historical test data.

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

    let matches = Command::new("test-report-generator")
        .version(end_to_end_tests::VERSION)
        .about("IMP Test Report Generator and Analysis Tool")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("FILE_OR_DIR")
                .help("Input test results file or directory")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("DIR")
                .help("Output directory for generated reports")
                .default_value("generated_reports"),
        )
        .arg(
            Arg::new("format")
                .short('f')
                .long("format")
                .value_name("FORMAT")
                .help("Output format (html, json, both)")
                .default_value("both"),
        )
        .arg(
            Arg::new("analyze")
                .short('a')
                .long("analyze")
                .help("Perform trend analysis on multiple test results")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("compare")
                .short('c')
                .long("compare")
                .value_name("BASELINE_FILE")
                .help("Compare results against a baseline file"),
        )
        .get_matches();

    let input_path = PathBuf::from(matches.get_one::<String>("input").unwrap());
    let output_dir = PathBuf::from(matches.get_one::<String>("output").unwrap());
    let format = matches.get_one::<String>("format").unwrap();

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    if input_path.is_file() {
        // Process single test report file
        info!("Processing single test report: {}", input_path.display());
        process_single_report(&input_path, &output_dir, format).await?;
    } else if input_path.is_dir() {
        // Process directory of test results
        info!("Processing test results directory: {}", input_path.display());
        
        if matches.get_flag("analyze") {
            perform_trend_analysis(&input_path, &output_dir, format).await?;
        } else {
            process_results_directory(&input_path, &output_dir, format).await?;
        }
    } else {
        error!("Input path does not exist: {}", input_path.display());
        return Err("Invalid input path".into());
    }

    // Handle comparison if requested
    if let Some(baseline_path) = matches.get_one::<String>("compare") {
        info!("Performing comparison against baseline: {}", baseline_path);
        perform_comparison(&input_path, &PathBuf::from(baseline_path), &output_dir).await?;
    }

    info!("Report generation completed. Results saved to: {}", output_dir.display());
    Ok(())
}

async fn process_single_report(
    input_path: &PathBuf,
    output_dir: &PathBuf,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(input_path)?;
    let report: TestReport = serde_json::from_str(&content)?;

    match format {
        "html" => {
            let html_path = output_dir.join("report.html");
            report.save_html(&html_path)?;
            info!("HTML report saved to: {}", html_path.display());
        }
        "json" => {
            let json_path = output_dir.join("report.json");
            report.save_json(&json_path)?;
            info!("JSON report saved to: {}", json_path.display());
        }
        "both" => {
            let html_path = output_dir.join("report.html");
            let json_path = output_dir.join("report.json");
            report.save_html(&html_path)?;
            report.save_json(&json_path)?;
            info!("Reports saved to: {} and {}", html_path.display(), json_path.display());
        }
        _ => {
            error!("Unknown format: {}", format);
            return Err("Invalid format specified".into());
        }
    }

    // Print summary
    let (total, passed, failed) = report.overall_stats();
    info!("Report Summary:");
    info!("  Total tests: {}", total);
    info!("  Passed: {}", passed);
    info!("  Failed: {}", failed);
    info!("  Pass rate: {:.1}%", report.summary.overall_pass_rate * 100.0);
    info!("  Duration: {:.2} minutes", report.summary.total_duration_minutes);

    Ok(())
}

async fn process_results_directory(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = fs::read_dir(input_dir)?;
    let mut processed_count = 0;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            info!("Processing: {}", path.display());
            
            match fs::read_to_string(&path) {
                Ok(content) => {
                    match serde_json::from_str::<TestReport>(&content) {
                        Ok(report) => {
                            let filename = path.file_stem().unwrap().to_str().unwrap();
                            let report_output_dir = output_dir.join(filename);
                            std::fs::create_dir_all(&report_output_dir)?;
                            
                            match format {
                                "html" => {
                                    let html_path = report_output_dir.join("report.html");
                                    report.save_html(&html_path)?;
                                }
                                "json" => {
                                    let json_path = report_output_dir.join("report.json");
                                    report.save_json(&json_path)?;
                                }
                                "both" => {
                                    let html_path = report_output_dir.join("report.html");
                                    let json_path = report_output_dir.join("report.json");
                                    report.save_html(&html_path)?;
                                    report.save_json(&json_path)?;
                                }
                                _ => {}
                            }
                            
                            processed_count += 1;
                        }
                        Err(e) => {
                            warn!("Failed to parse report {}: {}", path.display(), e);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to read file {}: {}", path.display(), e);
                }
            }
        }
    }

    info!("Processed {} test reports", processed_count);
    Ok(())
}

async fn perform_trend_analysis(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    _format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = fs::read_dir(input_dir)?;
    let mut reports = Vec::new();

    // Collect all test reports
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            match fs::read_to_string(&path) {
                Ok(content) => {
                    match serde_json::from_str::<TestReport>(&content) {
                        Ok(report) => {
                            reports.push(report);
                        }
                        Err(e) => {
                            warn!("Failed to parse report {}: {}", path.display(), e);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to read file {}: {}", path.display(), e);
                }
            }
        }
    }

    if reports.is_empty() {
        warn!("No valid test reports found for trend analysis");
        return Ok(());
    }

    info!("Performing trend analysis on {} reports", reports.len());

    // Use the new trend analysis method from TestReport
    let trend_report = TestReport::analyze_trends(&reports);
    let trend_path = output_dir.join("trend_analysis.json");
    
    let trend_json = serde_json::to_string_pretty(&trend_report)?;
    fs::write(&trend_path, trend_json)?;
    
    info!("Trend analysis saved to: {}", trend_path.display());

    // Print trend summary
    info!("Trend Analysis Summary:");
    info!("  Reports analyzed: {}", reports.len());
    info!("  Time span: {} to {}", 
          chrono::DateTime::from_timestamp(reports.first().unwrap().generated_at, 0).unwrap().format("%Y-%m-%d %H:%M"),
          chrono::DateTime::from_timestamp(reports.last().unwrap().generated_at, 0).unwrap().format("%Y-%m-%d %H:%M"));
    
    if let Some(trend) = trend_report.pass_rate_trend {
        info!("  Pass rate trend: {:.2}% per day", trend * 100.0);
    }
    
    if let Some(trend) = trend_report.duration_trend {
        info!("  Duration trend: {:.2} minutes per day", trend);
    }
    
    if !trend_report.performance_regressions.is_empty() {
        warn!("  Performance regressions detected: {}", trend_report.performance_regressions.len());
        for regression in &trend_report.performance_regressions {
            warn!("    {}: {:.1}% change ({})", 
                  regression.metric_name, 
                  regression.percentage_change,
                  match regression.severity {
                      end_to_end_tests::RegressionSeverity::Critical => "CRITICAL",
                      end_to_end_tests::RegressionSeverity::High => "HIGH",
                      end_to_end_tests::RegressionSeverity::Medium => "MEDIUM",
                      end_to_end_tests::RegressionSeverity::Low => "LOW",
                  });
        }
    }

    Ok(())
}

async fn perform_comparison(
    current_path: &PathBuf,
    baseline_path: &PathBuf,
    output_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let current_content = fs::read_to_string(current_path)?;
    let baseline_content = fs::read_to_string(baseline_path)?;

    let current_report: TestReport = serde_json::from_str(&current_content)?;
    let baseline_report: TestReport = serde_json::from_str(&baseline_content)?;

    // Use the new comparison method from TestReport
    let comparison = TestReport::compare_reports(&current_report, &baseline_report);
    
    let comparison_path = output_dir.join("comparison_report.json");
    let comparison_json = serde_json::to_string_pretty(&comparison)?;
    fs::write(&comparison_path, comparison_json)?;

    info!("Comparison report saved to: {}", comparison_path.display());

    // Print detailed comparison summary
    info!("Comparison Summary:");
    info!("  Status: {:?}", comparison.summary.status);
    info!("  Current pass rate: {:.1}%", current_report.summary.overall_pass_rate * 100.0);
    info!("  Baseline pass rate: {:.1}%", baseline_report.summary.overall_pass_rate * 100.0);
    info!("  Pass rate change: {:.1}%", comparison.pass_rate_change * 100.0);
    info!("  Duration change: {:.1} minutes", comparison.duration_change);
    info!("  Health score change: {:.1}%", comparison.health_score_change * 100.0);
    
    if !comparison.new_failures.is_empty() {
        warn!("New failures detected: {}", comparison.new_failures.len());
        for failure in &comparison.new_failures {
            warn!("  - {}", failure);
        }
    }
    
    if !comparison.resolved_failures.is_empty() {
        info!("Resolved failures: {}", comparison.resolved_failures.len());
        for resolved in &comparison.resolved_failures {
            info!("  + {}", resolved);
        }
    }
    
    if !comparison.performance_regressions.is_empty() {
        warn!("Performance regressions detected: {}", comparison.performance_regressions.len());
        for regression in &comparison.performance_regressions {
            warn!("  {} ({}): {:.1}% change", 
                  regression.metric_name,
                  match regression.severity {
                      end_to_end_tests::RegressionSeverity::Critical => "CRITICAL",
                      end_to_end_tests::RegressionSeverity::High => "HIGH", 
                      end_to_end_tests::RegressionSeverity::Medium => "MEDIUM",
                      end_to_end_tests::RegressionSeverity::Low => "LOW",
                  },
                  regression.percentage_change);
        }
    }
    
    if !comparison.summary.recommendations.is_empty() {
        info!("Recommendations:");
        for recommendation in &comparison.summary.recommendations {
            info!("  • {}", recommendation);
        }
    }

    Ok(())
}

