// Example demonstrating ValidationReport functionality

use polars::prelude::*;
use training_data_cli::validation::{DataValidator, ValidationReport};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ValidationReport Demo");
    println!("====================\n");

    // Create sample data with some issues
    let data_with_issues = df! {
        "timestamp" => vec![
            1640995200i64, // 2022-01-01 00:00:00
            1640995500i64, // 2022-01-01 00:05:00
            1640995500i64, // Duplicate timestamp
            1640997000i64, // Large gap (25 minutes later)
            1640997300i64, // 2022-01-01 00:35:00
        ],
        "close" => vec![
            Some(100.0),
            Some(101.0),
            None, // Missing value
            Some(1000.0), // Outlier
            Some(98.0),
        ],
        "volume" => vec![1000.0, 1100.0, 1100.0, 1200.0, 800.0],
    }?;

    // Create validator and run validation
    let validator = DataValidator::with_default_config();
    let validation_result = validator.validate(&data_with_issues)?;

    // Create validation report
    let report = ValidationReport::new(
        validation_result,
        Some("sample_market_data.parquet".to_string()),
    );

    // Display human-readable report
    println!("{}", report);

    // Generate JSON report
    println!("\n{}", "=".repeat(80));
    println!("JSON Report (first 500 characters):");
    println!("{}", "=".repeat(80));
    let json_report = report.to_json()?;
    println!("{}...", &json_report[..500.min(json_report.len())]);

    // Show compact JSON
    println!("\n{}", "=".repeat(80));
    println!("Compact JSON Report (first 300 characters):");
    println!("{}", "=".repeat(80));
    let compact_json = report.to_json_compact()?;
    println!("{}...", &compact_json[..300.min(compact_json.len())]);

    Ok(())
}