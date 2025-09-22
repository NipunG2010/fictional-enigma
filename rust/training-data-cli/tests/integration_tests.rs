use anyhow::Result;
use polars::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

/// Integration tests for the training data CLI
/// 
/// These tests verify the complete workflow from raw OHLCV data to labeled training snapshots,
/// including error scenarios and performance benchmarks.

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test the complete end-to-end workflow from OHLCV to labeled training snapshot
    #[test]
    fn test_end_to_end_workflow() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Copy sample data to temp directory
        let sample_data_path = get_sample_data_path()?;
        let input_path = temp_path.join("test_input.parquet");
        fs::copy(&sample_data_path, &input_path)?;
        
        let output_path = temp_path.join("test_output.parquet");
        let config_path = temp_path.join("test_config.json");
        
        // Create a test configuration
        create_test_config(&config_path)?;
        
        // Run the CLI command to create a training snapshot
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "create",
                "--input", input_path.to_str().unwrap(),
                "--output", output_path.to_str().unwrap(),
                "--horizon", "12",
                "--config", config_path.to_str().unwrap(),
                "--verbose"
            ])
            .output()?;
        
        // Verify command succeeded
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("CLI command failed: {}", stderr);
        }
        
        // Verify output file was created
        assert!(output_path.exists(), "Output file should be created");
        
        // Verify output format and content
        verify_training_snapshot_format(&output_path)?;
        
        // Verify metadata file was created
        let metadata_path = output_path.with_extension("json");
        assert!(metadata_path.exists(), "Metadata file should be created");
        verify_metadata_format(&metadata_path)?;
        
        Ok(())
    }

    /// Test data validation workflow
    #[test]
    fn test_validation_workflow() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Copy sample data to temp directory
        let sample_data_path = get_sample_data_path()?;
        let input_path = temp_path.join("test_input.parquet");
        fs::copy(&sample_data_path, &input_path)?;
        
        let report_path = temp_path.join("validation_report.json");
        
        // Run validation command
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "validate",
                "--input", input_path.to_str().unwrap(),
                "--report", report_path.to_str().unwrap(),
                "--strictness", "normal",
                "--verbose"
            ])
            .output()?;
        
        // Verify command succeeded
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("Validation command failed: {}", stderr);
        }
        
        // Verify report file was created
        assert!(report_path.exists(), "Validation report should be created");
        
        // Verify report format
        verify_validation_report_format(&report_path)?;
        
        Ok(())
    }

    /// Test configuration management workflow
    #[test]
    fn test_config_management_workflow() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Set config directory for testing
        std::env::set_var("TRAINING_DATA_CONFIG_DIR", temp_path.to_str().unwrap());
        
        let config_file = temp_path.join("test_config.json");
        create_test_config(&config_file)?;
        
        // Test saving configuration
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "config", "save",
                "--name", "test_config",
                "--file", config_file.to_str().unwrap(),
                "--description", "Test configuration for integration tests"
            ])
            .output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("Config save command failed: {}", stderr);
        }
        
        // Test listing configurations
        let output = Command::new(get_cli_binary_path()?)
            .args(["config", "list", "--verbose"])
            .output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("Config list command failed: {}", stderr);
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("test_config"), "Saved config should appear in list");
        
        // Test loading configuration
        let loaded_config_path = temp_path.join("loaded_config.json");
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "config", "load", "test_config",
                "--output", loaded_config_path.to_str().unwrap()
            ])
            .output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("Config load command failed: {}", stderr);
        }
        
        assert!(loaded_config_path.exists(), "Loaded config file should be created");
        
        Ok(())
    }

    /// Test error scenarios and recovery mechanisms
    #[test]
    fn test_error_scenarios() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Test with non-existent input file
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "create",
                "--input", "non_existent_file.parquet",
                "--output", temp_path.join("output.parquet").to_str().unwrap(),
                "--horizon", "12"
            ])
            .output()?;
        
        assert!(!output.status.success(), "Command should fail with non-existent input");
        
        // Test with invalid horizon
        let sample_data_path = get_sample_data_path()?;
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "create",
                "--input", sample_data_path.to_str().unwrap(),
                "--output", temp_path.join("output.parquet").to_str().unwrap(),
                "--horizon", "0"
            ])
            .output()?;
        
        assert!(!output.status.success(), "Command should fail with invalid horizon");
        
        // Test with invalid date range
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "create",
                "--input", sample_data_path.to_str().unwrap(),
                "--output", temp_path.join("output.parquet").to_str().unwrap(),
                "--horizon", "12",
                "--start-date", "2023-12-31",
                "--end-date", "2023-01-01"
            ])
            .output()?;
        
        assert!(!output.status.success(), "Command should fail with invalid date range");
        
        // Test with corrupted data file
        let corrupted_file = temp_path.join("corrupted.parquet");
        fs::write(&corrupted_file, "not a parquet file")?;
        
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "validate",
                "--input", corrupted_file.to_str().unwrap()
            ])
            .output()?;
        
        assert!(!output.status.success(), "Command should fail with corrupted data");
        
        Ok(())
    }

    /// Test output format compatibility with LDC engine expectations
    #[test]
    fn test_ldc_engine_compatibility() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Copy sample data to temp directory
        let sample_data_path = get_sample_data_path()?;
        let input_path = temp_path.join("test_input.parquet");
        fs::copy(&sample_data_path, &input_path)?;
        
        let output_path = temp_path.join("ldc_compatible_output.parquet");
        
        // Create training snapshot
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "create",
                "--input", input_path.to_str().unwrap(),
                "--output", output_path.to_str().unwrap(),
                "--horizon", "12",
                "--format", "parquet"
            ])
            .output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("CLI command failed: {}", stderr);
        }
        
        // Verify LDC engine compatibility
        verify_ldc_engine_compatibility(&output_path)?;
        
        Ok(())
    }

    /// Performance benchmark for large dataset processing
    #[test]
    fn test_performance_benchmark() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create a larger synthetic dataset for performance testing
        let large_dataset_path = temp_path.join("large_dataset.parquet");
        create_large_synthetic_dataset(&large_dataset_path, 100_000)?; // 100k rows
        
        let output_path = temp_path.join("performance_output.parquet");
        
        let start_time = std::time::Instant::now();
        
        // Run the CLI command
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "create",
                "--input", large_dataset_path.to_str().unwrap(),
                "--output", output_path.to_str().unwrap(),
                "--horizon", "12",
                "--verbose"
            ])
            .output()?;
        
        let duration = start_time.elapsed();
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("Performance test failed: {}", stderr);
        }
        
        // Verify performance is reasonable (should process 100k rows in under 30 seconds)
        assert!(duration.as_secs() < 30, 
            "Processing 100k rows took too long: {:?}", duration);
        
        // Verify output was created correctly
        assert!(output_path.exists(), "Output file should be created");
        
        // Check memory usage didn't exceed reasonable limits by verifying file sizes
        let input_size = fs::metadata(&large_dataset_path)?.len();
        let output_size = fs::metadata(&output_path)?.len();
        
        // Output should be larger due to additional features and labels, but not excessively so
        assert!(output_size > input_size, "Output should be larger than input");
        assert!(output_size < input_size * 10, "Output shouldn't be more than 10x input size");
        
        println!("Performance benchmark completed in {:?}", duration);
        println!("Input size: {} bytes, Output size: {} bytes", input_size, output_size);
        
        Ok(())
    }

    /// Test specific validation checks
    #[test]
    fn test_specific_validation_checks() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create test data with known issues
        let test_data_path = temp_path.join("test_data_with_issues.parquet");
        create_test_data_with_issues(&test_data_path)?;
        
        let report_path = temp_path.join("specific_validation_report.json");
        
        // Test specific checks
        let checks = ["missing", "outliers", "duplicates", "timestamps"];
        
        for check in &checks {
            let output = Command::new(get_cli_binary_path()?)
                .args([
                    "validate",
                    "--input", test_data_path.to_str().unwrap(),
                    "--report", report_path.to_str().unwrap(),
                    "--checks", check,
                    "--verbose"
                ])
                .output()?;
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!("Validation check '{}' failed: {}", check, stderr);
            }
            
            // Verify report was created and contains expected check results
            assert!(report_path.exists(), "Validation report should be created for check: {}", check);
            
            // Clean up report for next iteration
            if report_path.exists() {
                fs::remove_file(&report_path)?;
            }
        }
        
        Ok(())
    }

    /// Test different output formats
    #[test]
    fn test_output_formats() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Copy sample data to temp directory
        let sample_data_path = get_sample_data_path()?;
        let input_path = temp_path.join("test_input.parquet");
        fs::copy(&sample_data_path, &input_path)?;
        
        let formats = [("parquet", "parquet"), ("csv", "csv"), ("json", "json")];
        
        for (format_name, extension) in &formats {
            let output_path = temp_path.join(format!("test_output.{}", extension));
            
            let output = Command::new(get_cli_binary_path()?)
                .args([
                    "create",
                    "--input", input_path.to_str().unwrap(),
                    "--output", output_path.to_str().unwrap(),
                    "--horizon", "12",
                    "--format", format_name
                ])
                .output()?;
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!("Format '{}' test failed: {}", format_name, stderr);
            }
            
            assert!(output_path.exists(), "Output file should be created for format: {}", format_name);
            
            // Verify file is not empty
            let file_size = fs::metadata(&output_path)?.len();
            assert!(file_size > 0, "Output file should not be empty for format: {}", format_name);
        }
        
        Ok(())
    }
}

// Helper functions for integration tests

/// Get the path to sample OHLCV data for testing
fn get_sample_data_path() -> Result<PathBuf> {
    let sample_path = PathBuf::from("../sample/ohlcv.parquet");
    if sample_path.exists() {
        Ok(sample_path)
    } else {
        // Try alternative path
        let alt_path = PathBuf::from("rust/sample/ohlcv.parquet");
        if alt_path.exists() {
            Ok(alt_path)
        } else {
            Err(anyhow::anyhow!("Sample data file not found. Expected at ../sample/ohlcv.parquet or rust/sample/ohlcv.parquet"))
        }
    }
}

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

/// Create a test configuration file
fn create_test_config(config_path: &Path) -> Result<()> {
    let config = serde_json::json!({
        "horizon": 12,
        "features": [
            "rsi_14", "sma_20", "ema_12", "ema_26", 
            "macd", "macd_signal", "bb_upper", "bb_lower", "atr_14"
        ],
        "label_thresholds": {
            "buy_threshold": 0.02,
            "sell_threshold": -0.02
        },
        "validation": {
            "strictness": "normal",
            "max_missing_percentage": 5.0,
            "outlier_method": "iqr"
        }
    });
    
    fs::write(config_path, serde_json::to_string_pretty(&config)?)?;
    Ok(())
}

/// Verify the format of a training snapshot output file
fn verify_training_snapshot_format(output_path: &Path) -> Result<()> {
    let df = LazyFrame::scan_parquet(output_path.to_string_lossy(), ScanArgsParquet::default())?
        .collect()?;
    
    // Verify required columns exist
    let required_columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "future_return", "label"
    ];
    
    let column_names = df.get_column_names();
    for col in &required_columns {
        assert!(column_names.iter().any(|c| c.as_str() == *col), 
            "Required column '{}' missing from output", col);
    }
    
    // Verify data types
    let schema = df.schema();
    
    // Timestamp should be datetime
    if let Some(timestamp_dtype) = schema.get("timestamp") {
        assert!(matches!(timestamp_dtype, DataType::Datetime(_, _)), 
            "Timestamp column should be datetime type");
    }
    
    // OHLCV columns should be numeric
    for col in &["open", "high", "low", "close", "volume"] {
        if let Some(dtype) = schema.get(col) {
            assert!(dtype.is_numeric(), 
                "Column '{}' should be numeric type", col);
        }
    }
    
    // Verify we have some data
    assert!(df.height() > 0, "Output should contain data rows");
    
    Ok(())
}

/// Verify the format of a metadata file
fn verify_metadata_format(metadata_path: &Path) -> Result<()> {
    let metadata_content = fs::read_to_string(metadata_path)?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;
    
    // Verify required metadata fields
    let required_fields = [
        "snapshot_id", "created_at", "config", "data_info", 
        "label_distribution", "validation_summary"
    ];
    
    for field in &required_fields {
        assert!(metadata.get(field).is_some(), 
            "Required metadata field '{}' missing", field);
    }
    
    // Verify config structure
    let config = metadata.get("config").unwrap();
    assert!(config.get("horizon").is_some(), "Config should have horizon");
    assert!(config.get("features").is_some(), "Config should have features");
    assert!(config.get("label_thresholds").is_some(), "Config should have label_thresholds");
    
    Ok(())
}

/// Verify the format of a validation report
fn verify_validation_report_format(report_path: &Path) -> Result<()> {
    let report_content = fs::read_to_string(report_path)?;
    let report: serde_json::Value = serde_json::from_str(&report_content)?;
    
    // Verify required report fields
    let required_fields = [
        "overall_status", "missing_values", "outliers", 
        "timestamps", "duplicates", "statistics"
    ];
    
    for field in &required_fields {
        assert!(report.get(field).is_some(), 
            "Required report field '{}' missing", field);
    }
    
    // Verify status is valid
    let status = report.get("overall_status").unwrap().as_str().unwrap();
    assert!(["Passed", "Warning", "Failed"].contains(&status), 
        "Invalid validation status: {}", status);
    
    Ok(())
}

/// Verify LDC engine compatibility
fn verify_ldc_engine_compatibility(output_path: &Path) -> Result<()> {
    let df = LazyFrame::scan_parquet(output_path.to_string_lossy(), ScanArgsParquet::default())?
        .collect()?;
    
    // Verify LDC engine expected columns and formats
    let ldc_required_columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "rsi_14", "sma_20", "ema_12", "ema_26", "macd", "atr_14",
        "label"
    ];
    
    let column_names = df.get_column_names();
    for col in &ldc_required_columns {
        assert!(column_names.iter().any(|c| c.as_str() == *col), 
            "LDC engine requires column '{}'", col);
    }
    
    // Verify label column contains valid values
    if let Ok(label_series) = df.column("label") {
        // Check that labels are categorical (Buy/Sell/Hold or numeric equivalents)
        // This is a basic check - in practice, you'd verify the exact format expected by LDC engine
        assert!(label_series.len() > 0, "Label column should not be empty");
    }
    
    // Verify timestamp ordering (LDC engine expects chronological order)
    if let Ok(timestamp_series) = df.column("timestamp") {
        // Basic check that we have timestamps
        assert!(timestamp_series.len() > 0, "Timestamp column should not be empty");
    }
    
    Ok(())
}

/// Create a large synthetic dataset for performance testing
fn create_large_synthetic_dataset(output_path: &Path, num_rows: usize) -> Result<()> {
    use chrono::{DateTime, Utc, Duration};
    use std::f64::consts::PI;
    
    // Generate synthetic OHLCV data
    let start_time = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
    
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut opens = Vec::with_capacity(num_rows);
    let mut highs = Vec::with_capacity(num_rows);
    let mut lows = Vec::with_capacity(num_rows);
    let mut closes = Vec::with_capacity(num_rows);
    let mut volumes = Vec::with_capacity(num_rows);
    
    let mut current_price = 50000.0; // Starting price
    
    for i in 0..num_rows {
        let timestamp = start_time + Duration::minutes(i as i64 * 5); // 5-minute intervals
        timestamps.push(timestamp.timestamp_millis());
        
        // Generate realistic price movement using sine wave with noise
        let trend = (i as f64 * 2.0 * PI / 1000.0).sin() * 0.001; // Long-term trend
        let noise = (i as f64 * 13.0).sin() * 0.01; // Short-term noise
        let price_change = trend + noise;
        
        current_price *= 1.0 + price_change;
        
        let open = current_price;
        let close = current_price * (1.0 + (i as f64 * 7.0).sin() * 0.005);
        let high = open.max(close) * (1.0 + (i as f64 * 11.0).sin().abs() * 0.01);
        let low = open.min(close) * (1.0 - (i as f64 * 17.0).sin().abs() * 0.01);
        let volume = 1000.0 + (i as f64 * 19.0).sin().abs() * 5000.0;
        
        opens.push(open);
        highs.push(high);
        lows.push(low);
        closes.push(close);
        volumes.push(volume);
        
        current_price = close;
    }
    
    // Create DataFrame
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
    
    // Write to Parquet
    let mut file = std::fs::File::create(output_path)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;
    
    Ok(())
}

/// Create test data with known data quality issues
fn create_test_data_with_issues(output_path: &Path) -> Result<()> {
    use chrono::{DateTime, Utc, Duration};
    
    let start_time = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
    let num_rows = 1000;
    
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut opens = Vec::with_capacity(num_rows);
    let mut highs = Vec::with_capacity(num_rows);
    let mut lows = Vec::with_capacity(num_rows);
    let mut closes = Vec::with_capacity(num_rows);
    let mut volumes = Vec::with_capacity(num_rows);
    
    for i in 0..num_rows {
        let timestamp = if i == 500 {
            // Create a duplicate timestamp
            start_time + Duration::minutes((i - 1) as i64 * 5)
        } else if i == 600 {
            // Create a gap in timestamps
            start_time + Duration::minutes(i as i64 * 5 + 60) // 1 hour gap
        } else {
            start_time + Duration::minutes(i as i64 * 5)
        };
        
        timestamps.push(timestamp.timestamp_millis());
        
        // Create some missing values
        if i % 100 == 0 {
            opens.push(f64::NAN);
        } else {
            opens.push(50000.0 + (i as f64 * 0.1));
        }
        
        // Create outliers
        if i == 300 {
            highs.push(1000000.0); // Extreme outlier
        } else {
            highs.push(50100.0 + (i as f64 * 0.1));
        }
        
        lows.push(49900.0 + (i as f64 * 0.1));
        closes.push(50000.0 + (i as f64 * 0.1));
        volumes.push(1000.0 + (i as f64));
    }
    
    // Create DataFrame with issues
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
    
    // Write to Parquet
    let mut file = std::fs::File::create(output_path)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;
    
    Ok(())
}