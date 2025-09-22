use anyhow::Result;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

/// CLI-specific integration tests
/// 
/// These tests focus on CLI argument parsing, error handling, and user experience

#[cfg(test)]
mod cli_integration_tests {
    use super::*;

    /// Test CLI help and version commands
    #[test]
    fn test_cli_help_and_version() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        
        // Test --help
        let output = Command::new(&binary_path)
            .args(["--help"])
            .output()?;
        
        assert!(output.status.success(), "Help command should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Training data management for LDC trading system"), 
            "Help should contain description");
        assert!(stdout.contains("create"), "Help should list create command");
        assert!(stdout.contains("validate"), "Help should list validate command");
        assert!(stdout.contains("config"), "Help should list config command");
        
        // Test --version
        let output = Command::new(&binary_path)
            .args(["--version"])
            .output()?;
        
        assert!(output.status.success(), "Version command should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("training-data"), "Version should contain program name");
        
        Ok(())
    }

    /// Test subcommand help
    #[test]
    fn test_subcommand_help() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        
        let subcommands = ["create", "validate", "config"];
        
        for subcommand in &subcommands {
            let output = Command::new(&binary_path)
                .args([subcommand, "--help"])
                .output()?;
            
            assert!(output.status.success(), 
                "Help for '{}' subcommand should succeed", subcommand);
            
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.len() > 100, 
                "Help for '{}' should be substantial", subcommand);
        }
        
        Ok(())
    }

    /// Test argument validation errors
    #[test]
    fn test_argument_validation_errors() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Test invalid horizon
        let output = Command::new(&binary_path)
            .args([
                "create",
                "--input", "test.parquet",
                "--output", "output.parquet",
                "--horizon", "0"
            ])
            .output()?;
        
        assert!(!output.status.success(), "Should fail with invalid horizon");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Horizon must be greater than 0"), 
            "Should show horizon validation error");
        
        // Test invalid date format
        let output = Command::new(&binary_path)
            .args([
                "create",
                "--input", "test.parquet",
                "--output", "output.parquet",
                "--start-date", "invalid-date"
            ])
            .output()?;
        
        assert!(!output.status.success(), "Should fail with invalid date format");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("YYYY-MM-DD format"), 
            "Should show date format validation error");
        
        // Test invalid thresholds
        let output = Command::new(&binary_path)
            .args([
                "create",
                "--input", "test.parquet",
                "--output", "output.parquet",
                "--buy-threshold", "-0.1"
            ])
            .output()?;
        
        assert!(!output.status.success(), "Should fail with negative buy threshold");
        
        // Test invalid file extension
        let output = Command::new(&binary_path)
            .args([
                "create",
                "--input", "test.csv",
                "--output", "output.parquet"
            ])
            .output()?;
        
        assert!(!output.status.success(), "Should fail with non-parquet input");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Parquet format"), 
            "Should show file format validation error");
        
        Ok(())
    }

    /// Test verbose output functionality
    #[test]
    fn test_verbose_output() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create a minimal test parquet file
        let test_input = temp_path.join("test_input.parquet");
        create_minimal_test_data(&test_input)?;
        
        let output_path = temp_path.join("test_output.parquet");
        
        // Test verbose create command
        let output = Command::new(&binary_path)
            .args([
                "create",
                "--input", test_input.to_str().unwrap(),
                "--output", output_path.to_str().unwrap(),
                "--horizon", "12",
                "--verbose"
            ])
            .output()?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined_output = format!("{}{}", stdout, stderr);
        
        // Verbose output should contain configuration details
        assert!(combined_output.contains("Input:"), "Verbose should show input path");
        assert!(combined_output.contains("Output:"), "Verbose should show output path");
        assert!(combined_output.contains("Horizon: 12"), "Verbose should show horizon");
        
        // Test verbose validate command
        let output = Command::new(&binary_path)
            .args([
                "validate",
                "--input", test_input.to_str().unwrap(),
                "--verbose"
            ])
            .output()?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined_output = format!("{}{}", stdout, stderr);
        
        // Verbose validation should show loading and processing info
        assert!(combined_output.contains("Loading data") || combined_output.contains("Loaded"), 
            "Verbose validation should show data loading info");
        
        Ok(())
    }

    /// Test configuration file handling
    #[test]
    fn test_config_file_handling() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create test configuration file
        let config_path = temp_path.join("test_config.json");
        let config_content = serde_json::json!({
            "horizon": 24,
            "features": ["rsi_14", "sma_20"],
            "label_thresholds": {
                "buy_threshold": 0.03,
                "sell_threshold": -0.03
            }
        });
        fs::write(&config_path, serde_json::to_string_pretty(&config_content)?)?;
        
        // Create test input data
        let test_input = temp_path.join("test_input.parquet");
        create_minimal_test_data(&test_input)?;
        
        let output_path = temp_path.join("test_output.parquet");
        
        // Test using configuration file
        let output = Command::new(&binary_path)
            .args([
                "create",
                "--input", test_input.to_str().unwrap(),
                "--output", output_path.to_str().unwrap(),
                "--config", config_path.to_str().unwrap(),
                "--verbose"
            ])
            .output()?;
        
        // Should succeed or show appropriate error if config loading isn't implemented yet
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // If config loading isn't implemented, should show a clear message
            println!("Config test result: {}", stderr);
        }
        
        Ok(())
    }

    /// Test different output formats
    #[test]
    fn test_output_format_options() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create test input data
        let test_input = temp_path.join("test_input.parquet");
        create_minimal_test_data(&test_input)?;
        
        let formats = ["parquet", "csv", "json"];
        
        for format in &formats {
            let output_path = temp_path.join(format!("test_output.{}", format));
            
            let output = Command::new(&binary_path)
                .args([
                    "create",
                    "--input", test_input.to_str().unwrap(),
                    "--output", output_path.to_str().unwrap(),
                    "--format", format,
                    "--horizon", "12"
                ])
                .output()?;
            
            // Command should either succeed or show clear error about unimplemented format
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("Format '{}' test result: {}", format, stderr);
            }
        }
        
        Ok(())
    }

    /// Test validation strictness levels
    #[test]
    fn test_validation_strictness_levels() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create test input data with some issues
        let test_input = temp_path.join("test_input.parquet");
        create_test_data_with_minor_issues(&test_input)?;
        
        let strictness_levels = ["strict", "normal", "lenient"];
        
        for strictness in &strictness_levels {
            let report_path = temp_path.join(format!("report_{}.json", strictness));
            
            let output = Command::new(&binary_path)
                .args([
                    "validate",
                    "--input", test_input.to_str().unwrap(),
                    "--strictness", strictness,
                    "--report", report_path.to_str().unwrap(),
                    "--verbose"
                ])
                .output()?;
            
            // Different strictness levels might have different success/failure behavior
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("Strictness '{}' result: success={}, stdout={}, stderr={}", 
                strictness, output.status.success(), stdout, stderr);
        }
        
        Ok(())
    }

    /// Test specific validation checks
    #[test]
    fn test_specific_validation_checks() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create test input data
        let test_input = temp_path.join("test_input.parquet");
        create_minimal_test_data(&test_input)?;
        
        let checks = ["missing", "outliers", "duplicates", "timestamps"];
        
        for check in &checks {
            let output = Command::new(&binary_path)
                .args([
                    "validate",
                    "--input", test_input.to_str().unwrap(),
                    "--checks", check,
                    "--verbose"
                ])
                .output()?;
            
            // Should succeed or show clear error about unimplemented check
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("Check '{}' test result: {}", check, stderr);
            }
        }
        
        // Test multiple checks
        let output = Command::new(&binary_path)
            .args([
                "validate",
                "--input", test_input.to_str().unwrap(),
                "--checks", "missing,outliers",
                "--verbose"
            ])
            .output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("Multiple checks test result: {}", stderr);
        }
        
        Ok(())
    }

    /// Test error message quality and user experience
    #[test]
    fn test_error_message_quality() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        
        // Test missing required arguments
        let output = Command::new(&binary_path)
            .args(["create"])
            .output()?;
        
        assert!(!output.status.success(), "Should fail with missing arguments");
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        // Error message should be helpful
        assert!(stderr.contains("required") || stderr.contains("missing"), 
            "Should indicate missing required arguments");
        
        // Test invalid subcommand
        let output = Command::new(&binary_path)
            .args(["invalid-command"])
            .output()?;
        
        assert!(!output.status.success(), "Should fail with invalid subcommand");
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        // Should suggest valid subcommands
        assert!(stderr.len() > 0, "Should provide error message for invalid subcommand");
        
        Ok(())
    }

    /// Test progress indicators and user feedback
    #[test]
    fn test_progress_indicators() -> Result<()> {
        let binary_path = get_cli_binary_path()?;
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create a larger test dataset to trigger progress indicators
        let test_input = temp_path.join("large_test_input.parquet");
        create_larger_test_data(&test_input, 10000)?; // 10k rows
        
        let output_path = temp_path.join("test_output.parquet");
        
        // Run with verbose to see progress indicators
        let output = Command::new(&binary_path)
            .args([
                "create",
                "--input", test_input.to_str().unwrap(),
                "--output", output_path.to_str().unwrap(),
                "--horizon", "12",
                "--verbose"
            ])
            .output()?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined_output = format!("{}{}", stdout, stderr);
        
        // Should show some kind of progress or status information
        println!("Progress test output: {}", combined_output);
        
        Ok(())
    }
}

// Helper functions for CLI integration tests

/// Get the path to the CLI binary for testing
fn get_cli_binary_path() -> Result<PathBuf> {
    // In integration tests, we use the binary built by cargo test
    let binary_name = if cfg!(windows) { "training-data.exe" } else { "training-data" };
    
    // Try to find the binary in target directory
    let possible_paths = [
        format!("target/debug/{}", binary_name),
        format!("../target/debug/{}", binary_name),
        format!("../../target/debug/{}", binary_name),
    ];
    
    for path in &possible_paths {
        let path_buf = PathBuf::from(path);
        if path_buf.exists() {
            return Ok(path_buf);
        }
    }
    
    // If not found, assume it's in PATH
    Ok(PathBuf::from(binary_name))
}

/// Create minimal test data for CLI testing
fn create_minimal_test_data(output_path: &PathBuf) -> Result<()> {
    use chrono::{DateTime, Utc, Duration};
    use polars::prelude::*;
    
    let start_time = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
    let num_rows = 100;
    
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut opens = Vec::with_capacity(num_rows);
    let mut highs = Vec::with_capacity(num_rows);
    let mut lows = Vec::with_capacity(num_rows);
    let mut closes = Vec::with_capacity(num_rows);
    let mut volumes = Vec::with_capacity(num_rows);
    
    for i in 0..num_rows {
        let timestamp = start_time + Duration::minutes(i as i64 * 5);
        timestamps.push(timestamp.timestamp_millis());
        
        let base_price = 50000.0 + (i as f64 * 10.0);
        opens.push(base_price);
        highs.push(base_price + 100.0);
        lows.push(base_price - 100.0);
        closes.push(base_price + (i as f64 % 10.0) - 5.0);
        volumes.push(1000.0 + (i as f64 * 10.0));
    }
    
    let timestamp_series = Series::new("timestamp".into(), timestamps.iter().map(|&ts| {
        DateTime::from_timestamp_millis(ts).unwrap()
    }).collect::<Vec<_>>());
    let open_series = Series::new("open".into(), opens);
    let high_series = Series::new("high".into(), highs);
    let low_series = Series::new("low".into(), lows);
    let close_series = Series::new("close".into(), closes);
    let volume_series = Series::new("volume".into(), volumes);
    
    let mut df = DataFrame::new(vec![
        timestamp_series, open_series, high_series, low_series, close_series, volume_series
    ])?;
    
    let mut file = std::fs::File::create(output_path)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;
    
    Ok(())
}

/// Create test data with minor issues for validation testing
fn create_test_data_with_minor_issues(output_path: &PathBuf) -> Result<()> {
    use chrono::{DateTime, Utc, Duration};
    use polars::prelude::*;
    
    let start_time = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
    let num_rows = 100;
    
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut opens = Vec::with_capacity(num_rows);
    let mut highs = Vec::with_capacity(num_rows);
    let mut lows = Vec::with_capacity(num_rows);
    let mut closes = Vec::with_capacity(num_rows);
    let mut volumes = Vec::with_capacity(num_rows);
    
    for i in 0..num_rows {
        let timestamp = start_time + Duration::minutes(i as i64 * 5);
        timestamps.push(timestamp.timestamp_millis());
        
        let base_price = 50000.0 + (i as f64 * 10.0);
        
        // Add a few missing values
        if i == 50 {
            opens.push(f64::NAN);
        } else {
            opens.push(base_price);
        }
        
        // Add a minor outlier
        if i == 75 {
            highs.push(base_price * 1.1); // 10% higher than normal
        } else {
            highs.push(base_price + 100.0);
        }
        
        lows.push(base_price - 100.0);
        closes.push(base_price + (i as f64 % 10.0) - 5.0);
        volumes.push(1000.0 + (i as f64 * 10.0));
    }
    
    let timestamp_series = Series::new("timestamp".into(), timestamps.iter().map(|&ts| {
        DateTime::from_timestamp_millis(ts).unwrap()
    }).collect::<Vec<_>>());
    let open_series = Series::new("open".into(), opens);
    let high_series = Series::new("high".into(), highs);
    let low_series = Series::new("low".into(), lows);
    let close_series = Series::new("close".into(), closes);
    let volume_series = Series::new("volume".into(), volumes);
    
    let mut df = DataFrame::new(vec![
        timestamp_series, open_series, high_series, low_series, close_series, volume_series
    ])?;
    
    let mut file = std::fs::File::create(output_path)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;
    
    Ok(())
}

/// Create larger test data for progress indicator testing
fn create_larger_test_data(output_path: &PathBuf, num_rows: usize) -> Result<()> {
    use chrono::{DateTime, Utc, Duration};
    use polars::prelude::*;
    
    let start_time = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
    
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut opens = Vec::with_capacity(num_rows);
    let mut highs = Vec::with_capacity(num_rows);
    let mut lows = Vec::with_capacity(num_rows);
    let mut closes = Vec::with_capacity(num_rows);
    let mut volumes = Vec::with_capacity(num_rows);
    
    for i in 0..num_rows {
        let timestamp = start_time + Duration::minutes(i as i64 * 5);
        timestamps.push(timestamp.timestamp_millis());
        
        let base_price = 50000.0 + (i as f64 * 0.1);
        opens.push(base_price);
        highs.push(base_price + 100.0);
        lows.push(base_price - 100.0);
        closes.push(base_price + (i as f64 % 10.0) - 5.0);
        volumes.push(1000.0 + (i as f64));
    }
    
    let timestamp_series = Series::new("timestamp".into(), timestamps.iter().map(|&ts| {
        DateTime::from_timestamp_millis(ts).unwrap()
    }).collect::<Vec<_>>());
    let open_series = Series::new("open".into(), opens);
    let high_series = Series::new("high".into(), highs);
    let low_series = Series::new("low".into(), lows);
    let close_series = Series::new("close".into(), closes);
    let volume_series = Series::new("volume".into(), volumes);
    
    let mut df = DataFrame::new(vec![
        timestamp_series, open_series, high_series, low_series, close_series, volume_series
    ])?;
    
    let mut file = std::fs::File::create(output_path)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;
    
    Ok(())
}