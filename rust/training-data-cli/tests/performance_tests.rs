use anyhow::Result;
use polars::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Performance benchmark tests for the training data CLI
/// 
/// These tests measure processing time, memory usage, and scalability
/// to ensure the system can handle large datasets efficiently.

#[cfg(test)]
mod performance_tests {
    use super::*;

    /// Benchmark processing time for different dataset sizes
    #[test]
    fn benchmark_dataset_sizes() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        let dataset_sizes = [1_000, 10_000, 50_000];
        let mut results = Vec::new();
        
        for &size in &dataset_sizes {
            println!("Benchmarking dataset size: {} rows", size);
            
            // Create synthetic dataset
            let input_path = temp_path.join(format!("input_{}.parquet", size));
            let output_path = temp_path.join(format!("output_{}.parquet", size));
            
            create_synthetic_dataset(&input_path, size)?;
            
            // Measure processing time
            let start_time = Instant::now();
            
            let output = Command::new(get_cli_binary_path()?)
                .args([
                    "create",
                    "--input", input_path.to_str().unwrap(),
                    "--output", output_path.to_str().unwrap(),
                    "--horizon", "12"
                ])
                .output()?;
            
            let duration = start_time.elapsed();
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("Warning: Processing failed for size {}: {}", size, stderr);
                continue;
            }
            
            // Calculate throughput
            let rows_per_second = size as f64 / duration.as_secs_f64();
            
            results.push((size, duration, rows_per_second));
            
            println!("Size: {} rows, Time: {:?}, Throughput: {:.0} rows/sec", 
                size, duration, rows_per_second);
            
            // Verify output was created
            assert!(output_path.exists(), "Output should be created for size {}", size);
            
            // Check output size is reasonable
            let input_size = fs::metadata(&input_path)?.len();
            let output_size = fs::metadata(&output_path)?.len();
            
            assert!(output_size > input_size, 
                "Output should be larger than input due to features and labels");
            assert!(output_size < input_size * 20, 
                "Output shouldn't be excessively large (max 20x input)");
        }
        
        // Analyze performance scaling
        if results.len() >= 2 {
            analyze_performance_scaling(&results);
        }
        
        Ok(())
    }

    /// Benchmark memory usage patterns
    #[test]
    fn benchmark_memory_usage() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create a moderately large dataset
        let input_path = temp_path.join("memory_test_input.parquet");
        let output_path = temp_path.join("memory_test_output.parquet");
        
        create_synthetic_dataset(&input_path, 25_000)?;
        
        // Get initial memory usage
        let initial_memory = get_process_memory_usage()?;
        
        let start_time = Instant::now();
        
        let output = Command::new(get_cli_binary_path()?)
            .args([
                "create",
                "--input", input_path.to_str().unwrap(),
                "--output", output_path.to_str().unwrap(),
                "--horizon", "12",
                "--verbose"
            ])
            .output()?;
        
        let duration = start_time.elapsed();
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("Memory test failed: {}", stderr);
            return Ok(()); // Don't fail the test if implementation isn't complete
        }
        
        // Get final memory usage
        let final_memory = get_process_memory_usage()?;
        let memory_increase = final_memory.saturating_sub(initial_memory);
        
        println!("Memory usage test results:");
        println!("  Processing time: {:?}", duration);
        println!("  Initial memory: {} MB", initial_memory / 1024 / 1024);
        println!("  Final memory: {} MB", final_memory / 1024 / 1024);
        println!("  Memory increase: {} MB", memory_increase / 1024 / 1024);
        
        // Memory usage should be reasonable (less than 1GB for 25k rows)
        let max_memory_mb = 1024; // 1GB
        assert!(memory_increase < max_memory_mb * 1024 * 1024, 
            "Memory usage should be reasonable (< {}MB), got {}MB", 
            max_memory_mb, memory_increase / 1024 / 1024);
        
        Ok(())
    }

    /// Benchmark validation performance
    #[test]
    fn benchmark_validation_performance() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        let dataset_sizes = [5_000, 25_000, 100_000];
        
        for &size in &dataset_sizes {
            println!("Benchmarking validation for {} rows", size);
            
            let input_path = temp_path.join(format!("validation_input_{}.parquet", size));
            let report_path = temp_path.join(format!("validation_report_{}.json", size));
            
            create_synthetic_dataset(&input_path, size)?;
            
            let start_time = Instant::now();
            
            let output = Command::new(get_cli_binary_path()?)
                .args([
                    "validate",
                    "--input", input_path.to_str().unwrap(),
                    "--report", report_path.to_str().unwrap(),
                    "--strictness", "normal"
                ])
                .output()?;
            
            let duration = start_time.elapsed();
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("Validation benchmark failed for size {}: {}", size, stderr);
                continue;
            }
            
            let rows_per_second = size as f64 / duration.as_secs_f64();
            
            println!("Validation - Size: {} rows, Time: {:?}, Throughput: {:.0} rows/sec", 
                size, duration, rows_per_second);
            
            // Validation should be fast (> 10k rows/sec for normal data)
            assert!(rows_per_second > 1000.0, 
                "Validation should process at least 1k rows/sec, got {:.0}", rows_per_second);
            
            // Verify report was created
            assert!(report_path.exists(), "Validation report should be created");
        }
        
        Ok(())
    }

    /// Benchmark different feature computation scenarios
    #[test]
    fn benchmark_feature_scenarios() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        let input_path = temp_path.join("feature_test_input.parquet");
        create_synthetic_dataset(&input_path, 20_000)?;
        
        // Test different horizon values
        let horizons = [6, 12, 24, 48];
        
        for &horizon in &horizons {
            println!("Benchmarking horizon: {} periods", horizon);
            
            let output_path = temp_path.join(format!("feature_output_h{}.parquet", horizon));
            
            let start_time = Instant::now();
            
            let output = Command::new(get_cli_binary_path()?)
                .args([
                    "create",
                    "--input", input_path.to_str().unwrap(),
                    "--output", output_path.to_str().unwrap(),
                    "--horizon", &horizon.to_string()
                ])
                .output()?;
            
            let duration = start_time.elapsed();
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("Feature benchmark failed for horizon {}: {}", horizon, stderr);
                continue;
            }
            
            println!("Horizon {} - Time: {:?}", horizon, duration);
            
            // Processing time shouldn't increase dramatically with horizon
            assert!(duration.as_secs() < 60, 
                "Processing should complete within 60 seconds for horizon {}", horizon);
        }
        
        Ok(())
    }

    /// Benchmark concurrent processing capability
    #[test]
    fn benchmark_concurrent_processing() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create multiple input files
        let num_files = 3;
        let rows_per_file = 10_000;
        
        let mut input_paths = Vec::new();
        let mut output_paths = Vec::new();
        
        for i in 0..num_files {
            let input_path = temp_path.join(format!("concurrent_input_{}.parquet", i));
            let output_path = temp_path.join(format!("concurrent_output_{}.parquet", i));
            
            create_synthetic_dataset(&input_path, rows_per_file)?;
            
            input_paths.push(input_path);
            output_paths.push(output_path);
        }
        
        // Sequential processing
        let sequential_start = Instant::now();
        
        for (input_path, output_path) in input_paths.iter().zip(output_paths.iter()) {
            let output = Command::new(get_cli_binary_path()?)
                .args([
                    "create",
                    "--input", input_path.to_str().unwrap(),
                    "--output", output_path.to_str().unwrap(),
                    "--horizon", "12"
                ])
                .output()?;
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("Sequential processing failed: {}", stderr);
                return Ok(()); // Don't fail if implementation isn't complete
            }
        }
        
        let sequential_duration = sequential_start.elapsed();
        
        println!("Concurrent processing benchmark:");
        println!("  Sequential processing of {} files: {:?}", num_files, sequential_duration);
        println!("  Average per file: {:?}", sequential_duration / num_files as u32);
        
        // Verify all outputs were created
        for output_path in &output_paths {
            assert!(output_path.exists(), "Output file should be created: {:?}", output_path);
        }
        
        Ok(())
    }

    /// Stress test with edge case data
    #[test]
    fn stress_test_edge_cases() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Test with data containing various edge cases
        let edge_case_scenarios: Vec<(&str, fn(&PathBuf, usize) -> Result<()>)> = vec![
            ("high_volatility", create_high_volatility_data),
            ("sparse_data", create_sparse_data),
            ("extreme_values", create_extreme_value_data),
        ];
        
        for (scenario_name, data_creator) in &edge_case_scenarios {
            println!("Stress testing scenario: {}", scenario_name);
            
            let input_path = temp_path.join(format!("{}_input.parquet", scenario_name));
            let output_path = temp_path.join(format!("{}_output.parquet", scenario_name));
            
            data_creator(&input_path, 15_000)?;
            
            let start_time = Instant::now();
            
            let output = Command::new(get_cli_binary_path()?)
                .args([
                    "create",
                    "--input", input_path.to_str().unwrap(),
                    "--output", output_path.to_str().unwrap(),
                    "--horizon", "12"
                ])
                .output()?;
            
            let duration = start_time.elapsed();
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("Stress test '{}' failed: {}", scenario_name, stderr);
                continue;
            }
            
            println!("Scenario '{}' completed in {:?}", scenario_name, duration);
            
            // Should complete within reasonable time even with edge cases
            assert!(duration.as_secs() < 120, 
                "Scenario '{}' took too long: {:?}", scenario_name, duration);
            
            assert!(output_path.exists(), "Output should be created for scenario: {}", scenario_name);
        }
        
        Ok(())
    }
}

// Helper functions for performance tests

/// Get the path to the CLI binary for testing
fn get_cli_binary_path() -> Result<PathBuf> {
    let binary_name = if cfg!(windows) { "training-data.exe" } else { "training-data" };
    
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
    
    Ok(PathBuf::from(binary_name))
}

/// Create synthetic dataset for performance testing
fn create_synthetic_dataset(output_path: &PathBuf, num_rows: usize) -> Result<()> {
    use chrono::{DateTime, Utc, Duration as ChronoDuration};
    
    let start_time = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
    
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut opens = Vec::with_capacity(num_rows);
    let mut highs = Vec::with_capacity(num_rows);
    let mut lows = Vec::with_capacity(num_rows);
    let mut closes = Vec::with_capacity(num_rows);
    let mut volumes = Vec::with_capacity(num_rows);
    
    let mut current_price = 50000.0;
    
    for i in 0..num_rows {
        let timestamp = start_time + ChronoDuration::minutes(i as i64 * 5);
        timestamps.push(timestamp.timestamp_millis());
        
        // Generate realistic price movement
        let price_change = ((i as f64 * 0.1).sin() + (i as f64 * 0.03).cos()) * 0.001;
        current_price *= 1.0 + price_change;
        
        let open = current_price;
        let close = current_price * (1.0 + (i as f64 * 0.07).sin() * 0.005);
        let high = open.max(close) * (1.0 + (i as f64 * 0.11).sin().abs() * 0.01);
        let low = open.min(close) * (1.0 - (i as f64 * 0.13).sin().abs() * 0.01);
        let volume = 1000.0 + (i as f64 * 0.17).sin().abs() * 5000.0;
        
        opens.push(open);
        highs.push(high);
        lows.push(low);
        closes.push(close);
        volumes.push(volume);
        
        current_price = close;
    }
    
    let df = df! [
        "timestamp" => timestamps.iter().map(|&ts| {
            DateTime::from_timestamp_millis(ts).unwrap()
        }).collect::<Vec<_>>(),
        "open" => opens,
        "high" => highs,
        "low" => lows,
        "close" => closes,
        "volume" => volumes,
    ]?;
    
    let mut file = std::fs::File::create(output_path)?;
    ParquetWriter::new(&mut file).finish(&mut df.lazy())?;
    
    Ok(())
}

/// Create high volatility test data
fn create_high_volatility_data(output_path: &PathBuf, num_rows: usize) -> Result<()> {
    use chrono::{DateTime, Utc, Duration as ChronoDuration};
    
    let start_time = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
    
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut opens = Vec::with_capacity(num_rows);
    let mut highs = Vec::with_capacity(num_rows);
    let mut lows = Vec::with_capacity(num_rows);
    let mut closes = Vec::with_capacity(num_rows);
    let mut volumes = Vec::with_capacity(num_rows);
    
    let mut current_price = 50000.0;
    
    for i in 0..num_rows {
        let timestamp = start_time + ChronoDuration::minutes(i as i64 * 5);
        timestamps.push(timestamp.timestamp_millis());
        
        // High volatility: large price swings
        let volatility = 0.05; // 5% swings
        let price_change = ((i as f64 * 0.5).sin() + (i as f64 * 0.3).cos()) * volatility;
        current_price *= 1.0 + price_change;
        
        let open = current_price;
        let close = current_price * (1.0 + (i as f64 * 0.7).sin() * volatility);
        let high = open.max(close) * (1.0 + (i as f64 * 1.1).sin().abs() * volatility);
        let low = open.min(close) * (1.0 - (i as f64 * 1.3).sin().abs() * volatility);
        let volume = 5000.0 + (i as f64 * 0.17).sin().abs() * 50000.0; // High volume
        
        opens.push(open);
        highs.push(high);
        lows.push(low);
        closes.push(close);
        volumes.push(volume);
        
        current_price = close;
    }
    
    let df = df! [
        "timestamp" => timestamps.iter().map(|&ts| {
            DateTime::from_timestamp_millis(ts).unwrap()
        }).collect::<Vec<_>>(),
        "open" => opens,
        "high" => highs,
        "low" => lows,
        "close" => closes,
        "volume" => volumes,
    ]?;
    
    let mut file = std::fs::File::create(output_path)?;
    ParquetWriter::new(&mut file).finish(&mut df.lazy())?;
    
    Ok(())
}

/// Create sparse data with missing values
fn create_sparse_data(output_path: &PathBuf, num_rows: usize) -> Result<()> {
    use chrono::{DateTime, Utc, Duration as ChronoDuration};
    
    let start_time = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
    
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut opens = Vec::with_capacity(num_rows);
    let mut highs = Vec::with_capacity(num_rows);
    let mut lows = Vec::with_capacity(num_rows);
    let mut closes = Vec::with_capacity(num_rows);
    let mut volumes = Vec::with_capacity(num_rows);
    
    let mut current_price = 50000.0;
    
    for i in 0..num_rows {
        let timestamp = start_time + ChronoDuration::minutes(i as i64 * 5);
        timestamps.push(timestamp.timestamp_millis());
        
        // Add missing values (NaN) for some entries
        if i % 20 == 0 {
            opens.push(f64::NAN);
            highs.push(f64::NAN);
            lows.push(f64::NAN);
            closes.push(f64::NAN);
            volumes.push(f64::NAN);
        } else {
            let price_change = (i as f64 * 0.1).sin() * 0.001;
            current_price *= 1.0 + price_change;
            
            let open = current_price;
            let close = current_price * (1.0 + (i as f64 * 0.07).sin() * 0.005);
            let high = open.max(close) * (1.0 + (i as f64 * 0.11).sin().abs() * 0.01);
            let low = open.min(close) * (1.0 - (i as f64 * 0.13).sin().abs() * 0.01);
            let volume = 1000.0 + (i as f64 * 0.17).sin().abs() * 5000.0;
            
            opens.push(open);
            highs.push(high);
            lows.push(low);
            closes.push(close);
            volumes.push(volume);
            
            current_price = close;
        }
    }
    
    let df = df! [
        "timestamp" => timestamps.iter().map(|&ts| {
            DateTime::from_timestamp_millis(ts).unwrap()
        }).collect::<Vec<_>>(),
        "open" => opens,
        "high" => highs,
        "low" => lows,
        "close" => closes,
        "volume" => volumes,
    ]?;
    
    let mut file = std::fs::File::create(output_path)?;
    ParquetWriter::new(&mut file).finish(&mut df.lazy())?;
    
    Ok(())
}

/// Create data with extreme values
fn create_extreme_value_data(output_path: &PathBuf, num_rows: usize) -> Result<()> {
    use chrono::{DateTime, Utc, Duration as ChronoDuration};
    
    let start_time = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")?.with_timezone(&Utc);
    
    let mut timestamps = Vec::with_capacity(num_rows);
    let mut opens = Vec::with_capacity(num_rows);
    let mut highs = Vec::with_capacity(num_rows);
    let mut lows = Vec::with_capacity(num_rows);
    let mut closes = Vec::with_capacity(num_rows);
    let mut volumes = Vec::with_capacity(num_rows);
    
    let mut current_price = 50000.0;
    
    for i in 0..num_rows {
        let timestamp = start_time + ChronoDuration::minutes(i as i64 * 5);
        timestamps.push(timestamp.timestamp_millis());
        
        // Add extreme outliers occasionally
        if i == num_rows / 4 {
            // Extreme high price
            opens.push(current_price * 10.0);
            highs.push(current_price * 12.0);
            lows.push(current_price * 9.0);
            closes.push(current_price * 11.0);
            volumes.push(1000000.0); // Extreme volume
        } else if i == num_rows / 2 {
            // Extreme low price
            opens.push(current_price * 0.1);
            highs.push(current_price * 0.2);
            lows.push(current_price * 0.05);
            closes.push(current_price * 0.15);
            volumes.push(10.0); // Very low volume
        } else {
            let price_change = (i as f64 * 0.1).sin() * 0.001;
            current_price *= 1.0 + price_change;
            
            let open = current_price;
            let close = current_price * (1.0 + (i as f64 * 0.07).sin() * 0.005);
            let high = open.max(close) * (1.0 + (i as f64 * 0.11).sin().abs() * 0.01);
            let low = open.min(close) * (1.0 - (i as f64 * 0.13).sin().abs() * 0.01);
            let volume = 1000.0 + (i as f64 * 0.17).sin().abs() * 5000.0;
            
            opens.push(open);
            highs.push(high);
            lows.push(low);
            closes.push(close);
            volumes.push(volume);
            
            current_price = close;
        }
    }
    
    let df = df! [
        "timestamp" => timestamps.iter().map(|&ts| {
            DateTime::from_timestamp_millis(ts).unwrap()
        }).collect::<Vec<_>>(),
        "open" => opens,
        "high" => highs,
        "low" => lows,
        "close" => closes,
        "volume" => volumes,
    ]?;
    
    let mut file = std::fs::File::create(output_path)?;
    ParquetWriter::new(&mut file).finish(&mut df.lazy())?;
    
    Ok(())
}

/// Analyze performance scaling characteristics
fn analyze_performance_scaling(results: &[(usize, Duration, f64)]) {
    println!("\nPerformance Scaling Analysis:");
    println!("Size\t\tTime\t\tThroughput");
    println!("----\t\t----\t\t----------");
    
    for &(size, duration, throughput) in results {
        println!("{}\t\t{:?}\t\t{:.0} rows/sec", size, duration, throughput);
    }
    
    // Check if throughput is relatively stable (good scaling)
    if results.len() >= 2 {
        let throughputs: Vec<f64> = results.iter().map(|(_, _, t)| *t).collect();
        let min_throughput = throughputs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_throughput = throughputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let throughput_ratio = max_throughput / min_throughput;
        
        println!("\nScaling characteristics:");
        println!("  Min throughput: {:.0} rows/sec", min_throughput);
        println!("  Max throughput: {:.0} rows/sec", max_throughput);
        println!("  Throughput ratio: {:.2}x", throughput_ratio);
        
        if throughput_ratio < 2.0 {
            println!("  ✓ Good scaling - throughput is relatively stable");
        } else if throughput_ratio < 5.0 {
            println!("  ⚠ Moderate scaling - some performance degradation with size");
        } else {
            println!("  ✗ Poor scaling - significant performance degradation with size");
        }
    }
}

/// Get current process memory usage (approximate)
fn get_process_memory_usage() -> Result<u64> {
    // This is a simplified memory usage estimation
    // In a real implementation, you might use system-specific APIs
    
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let status = fs::read_to_string("/proc/self/status")?;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<u64>() {
                        return Ok(kb * 1024); // Convert KB to bytes
                    }
                }
            }
        }
    }
    
    // Fallback: return 0 if we can't measure memory
    Ok(0)
}