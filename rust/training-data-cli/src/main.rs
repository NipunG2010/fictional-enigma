use anyhow::Result;
use clap::Parser;
use polars::prelude::*;
use std::fs::{self, File};

mod cli;
mod config;
mod snapshot;
mod validation;
mod utils;

use cli::{Cli, Commands, ValidateArgs, ValidationLevel};
use validation::{DataValidator, ValidationConfig, ValidationReport};
use utils::{init_logging, create_error_message, display_info, display_success, display_warning};

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging based on verbosity
    let verbose = match &cli.command {
        Commands::Create(args) => args.verbose,
        Commands::Validate(args) => args.verbose,
        Commands::Config(_) => false, // Config commands don't have verbose flag
    };
    
    init_logging(verbose);
    
    // Validate arguments before processing
    let validation_result = match &cli.command {
        Commands::Create(args) => args.validate(),
        Commands::Validate(args) => args.validate(),
        Commands::Config(args) => args.validate(),
    };
    
    if let Err(validation_error) = validation_result {
        let error = anyhow::anyhow!("{}", validation_error);
        eprintln!("{}", create_error_message(&error));
        std::process::exit(1);
    }
    
    let result = match cli.command {
        Commands::Create(args) => {
            if args.verbose {
                display_info("Creating training snapshot with configuration:");
                println!("  Input: {}", args.input.display());
                println!("  Output: {}", args.output.display());
                println!("  Horizon: {} periods", args.horizon);
                println!("  Format: {:?}", args.format);
                if let Some(ref start) = args.start_date {
                    println!("  Start date: {}", start);
                }
                if let Some(ref end) = args.end_date {
                    println!("  End date: {}", end);
                }
                if let Some(buy_threshold) = args.buy_threshold {
                    println!("  Buy threshold: {:.4}", buy_threshold);
                }
                if let Some(sell_threshold) = args.sell_threshold {
                    println!("  Sell threshold: {:.4}", sell_threshold);
                }
                if args.skip_validation {
                    display_warning("Validation will be SKIPPED");
                }
            }
            
            display_success("Create command validated successfully");
            display_info("Implementation will be added in task 7-8");
            Ok(())
        }
        Commands::Validate(args) => {
            if args.verbose {
                display_info("Validating market data:");
                println!("  Input: {}", args.input.display());
                println!("  Strictness: {:?}", args.strictness);
                if let Some(ref report) = args.report {
                    println!("  Report output: {}", report.display());
                }
                if let Some(ref checks) = args.checks {
                    println!("  Specific checks: {}", checks.join(", "));
                }
                if let Some(max_missing) = args.max_missing_percent {
                    println!("  Max missing threshold: {:.2}%", max_missing);
                }
            }
            
            validate_data(&args)
        }
        Commands::Config(args) => {
            match &args.action {
                cli::ConfigAction::List { verbose } => {
                    display_info(&format!("Listing configurations (verbose: {})", verbose));
                }
                cli::ConfigAction::Save { name, file, description } => {
                    display_info(&format!("Saving configuration '{}'", name));
                    if let Some(ref file_path) = file {
                        println!("  From file: {}", file_path.display());
                    }
                    if let Some(ref desc) = description {
                        println!("  Description: {}", desc);
                    }
                }
                cli::ConfigAction::Load { name, output } => {
                    display_info(&format!("Loading configuration '{}'", name));
                    if let Some(ref output_path) = output {
                        println!("  Output to: {}", output_path.display());
                    }
                }
                cli::ConfigAction::Delete { name, force } => {
                    display_info(&format!("Deleting configuration '{}' (force: {})", name, force));
                }
                cli::ConfigAction::Template { output } => {
                    display_info("Showing configuration template");
                    if let Some(ref output_path) = output {
                        println!("  Output to: {}", output_path.display());
                    }
                }
            }
            
            display_success("Config command validated successfully");
            display_info("Implementation will be added in task 3");
            Ok(())
        }
    };
    
    // Handle any errors with improved error messages
    if let Err(e) = result {
        eprintln!("{}", create_error_message(&e));
        std::process::exit(1);
    }
    
    Ok(())
}

/// Validate market data quality and generate reports
fn validate_data(args: &ValidateArgs) -> Result<()> {
    // Load the data
    if args.verbose {
        display_info(&format!("Loading data from: {}", args.input.display()));
    }
    
    let file = File::open(&args.input)?;
    let reader = ParquetReader::new(file);
    let data = reader.finish()?;
    
    if args.verbose {
        display_info(&format!("Loaded {} rows and {} columns", data.height(), data.width()));
    }

    // Create validation configuration based on strictness level
    let mut validation_config = match args.strictness {
        ValidationLevel::Strict => ValidationConfig::strict(),
        ValidationLevel::Normal => ValidationConfig::normal(),
        ValidationLevel::Lenient => ValidationConfig::lenient(),
    };

    // Override max missing percentage if specified
    if let Some(max_missing) = args.max_missing_percent {
        validation_config.max_missing_percentage = max_missing;
    }

    // Create validator and run validation
    let validator = DataValidator::new(validation_config);
    
    if args.verbose {
        display_info("Running validation checks...");
    }

    let validation_result = if let Some(ref specific_checks) = args.checks {
        // Run only specific checks
        run_specific_validation_checks(&validator, &data, specific_checks)?
    } else {
        // Run all validation checks
        validator.validate(&data)?
    };

    // Generate validation report
    let data_source = Some(args.input.to_string_lossy().to_string());
    let report = ValidationReport::new(validation_result, data_source);

    // Output results
    if let Some(report_path) = &args.report {
        // Save JSON report to file
        let json_report = report.to_json()?;
        fs::write(&report_path, json_report)?;
        
        if args.verbose {
            display_success(&format!("Validation report saved to: {}", report_path.display()));
        }
        
        // Also display summary to console
        println!("\n📊 Validation Summary:");
        println!("Status: {:?}", report.summary.overall_status);
        println!("Issues: {} warnings, {} critical", report.summary.warnings, report.summary.critical_issues);
        
        if report.summary.critical_issues > 0 {
            display_warning(&format!("Critical issues found. Check the detailed report at: {}", report_path.display()));
        } else {
            display_success("No critical issues found");
        }
    } else {
        // Display human-readable report to console
        println!("{}", report.format_human_readable());
    }

    // Exit with error code if validation failed and we're in strict mode
    if matches!(args.strictness, ValidationLevel::Strict) && 
       matches!(report.summary.overall_status, validation::ValidationStatus::Failed) {
        return Err(anyhow::anyhow!("Validation failed in strict mode"));
    }
    
    if args.verbose {
        display_success("Validation completed successfully");
    }

    Ok(())
}

/// Run specific validation checks based on user selection
fn run_specific_validation_checks(
    validator: &DataValidator, 
    data: &DataFrame, 
    checks: &[String]
) -> Result<validation::ValidationResult> {
    use validation::{ValidationStatus, MissingValueReport, OutlierReport, TimestampReport, DuplicateReport};
    
    let statistics = validator.calculate_statistics(data)?;
    
    // Initialize with default "passed" reports
    let mut missing_values = MissingValueReport {
        total_rows: data.height(),
        columns_with_missing: std::collections::HashMap::new(),
        missing_percentage: 0.0,
        status: ValidationStatus::Passed,
    };
    
    let mut outliers = OutlierReport {
        method_used: validator.config.outlier_method.clone(),
        columns_with_outliers: std::collections::HashMap::new(),
        total_outliers: 0,
        status: ValidationStatus::Passed,
    };
    
    let mut timestamps = TimestampReport {
        total_rows: data.height(),
        sequential: true,
        gaps_found: 0,
        duplicate_timestamps: 0,
        expected_interval_seconds: validator.config.expected_interval_seconds,
        actual_intervals: Vec::new(),
        status: ValidationStatus::Passed,
    };
    
    let mut duplicates = DuplicateReport {
        total_rows: data.height(),
        duplicate_rows: 0,
        duplicate_percentage: 0.0,
        removed: false,
        status: ValidationStatus::Passed,
    };

    // Run only requested checks
    for check in checks {
        match check.as_str() {
            "missing" => {
                missing_values = validator.check_missing_values(data)?;
            }
            "outliers" => {
                outliers = validator.detect_outliers(data)?;
            }
            "timestamps" => {
                timestamps = validator.validate_timestamps(data)?;
            }
            "duplicates" => {
                duplicates = validator.check_duplicates(data)?;
            }
            _ => {
                // Invalid check names should have been caught by CLI validation
                eprintln!("Warning: Unknown check type '{}'", check);
            }
        }
    }

    // Determine overall status
    let overall_status = determine_overall_status(&missing_values, &outliers, &timestamps, &duplicates);

    Ok(validation::ValidationResult {
        overall_status,
        missing_values,
        outliers,
        timestamps,
        duplicates,
        statistics,
    })
}

/// Helper function to determine overall validation status
fn determine_overall_status(
    missing: &validation::MissingValueReport,
    outliers: &validation::OutlierReport,
    timestamps: &validation::TimestampReport,
    duplicates: &validation::DuplicateReport,
) -> validation::ValidationStatus {
    use validation::ValidationStatus;
    
    let statuses = [&missing.status, &outliers.status, &timestamps.status, &duplicates.status];

    if statuses.iter().any(|s| matches!(s, ValidationStatus::Failed)) {
        ValidationStatus::Failed
    } else if statuses.iter().any(|s| matches!(s, ValidationStatus::Warning)) {
        ValidationStatus::Warning
    } else {
        ValidationStatus::Passed
    }
}