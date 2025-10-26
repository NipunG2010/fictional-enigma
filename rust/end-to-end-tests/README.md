# End-to-End Testing Framework

A comprehensive testing framework for the IMP trading system that validates the complete signal generation pipeline, tests failure scenarios, and ensures performance requirements are met.

## Features

- **Complete Pipeline Testing**: Validates the entire signal flow from OHLCV data to final signal emission
- **Failure Scenario Testing**: Tests system behavior under various failure conditions (HMM service unavailable, Redis/Kafka failures, data corruption)
- **Performance Validation**: Measures and validates latency, throughput, and memory usage against requirements
- **Test Data Generation**: Creates realistic market scenarios and edge cases for comprehensive testing
- **Comprehensive Reporting**: Generates detailed HTML and JSON reports with test results and recommendations

## Architecture

The framework consists of several key components:

- **TestHarness**: Orchestrates test execution and manages test infrastructure
- **TestDataGenerator**: Generates realistic OHLCV data and market scenarios
- **PerformanceMonitor**: Tracks latency, throughput, and memory usage metrics
- **ResultValidator**: Validates test results against expected values and performance requirements
- **TestReport**: Comprehensive reporting with HTML and JSON output formats

## Configuration

Tests are configured via TOML files. See `test_config.toml` for an example configuration:

```toml
[pipeline_tests]
test_symbols = ["BTCUSDT", "ETHUSDT"]
test_duration_hours = 24
data_interval = "5m"
include_edge_cases = true

[performance_tests]
max_end_to_end_latency_ms = 100
min_throughput_signals_per_second = 10.0
max_memory_usage_mb = 512

[execution]
max_parallel_tests = 4
output_dir = "test_results"
generate_html_reports = true
```

## Usage

### Library Usage

```rust
use end_to_end_tests::{TestConfig, TestHarness};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = TestConfig::from_file("test_config.toml")?;
    
    // Create test harness
    let mut harness = TestHarness::new(config).await?;
    
    // Run all tests
    let report = harness.run_all_tests().await?;
    
    // Save reports
    report.save_html("test_results/report.html")?;
    report.save_json("test_results/report.json")?;
    
    println!("Tests completed with {:.1}% pass rate", 
             report.summary.overall_pass_rate * 100.0);
    
    Ok(())
}
```

### Running Tests

```bash
# Run library tests
cargo test --package end-to-end-tests

# Check compilation
cargo check --package end-to-end-tests
```

## Test Types

### Pipeline Integration Tests
- Complete signal pipeline validation
- Feature computation accuracy testing
- Signal generation validation for LDC, MR, and TSMOM strategies
- HMM integration and regime-aware weight application

### Failure Scenario Tests
- HMM service unavailability and fallback behavior
- Redis/Kafka connection failures and local buffering
- Data corruption scenarios and error handling
- Circuit breaker behavior under repeated failures

### Performance Tests
- End-to-end latency measurement and validation
- Concurrent symbol processing performance
- Throughput and memory usage validation
- System performance under sustained load

## Market Scenarios

The test data generator supports various market scenarios:

- **TrendingUp/TrendingDown**: Strong directional movements
- **Sideways**: Range-bound market conditions
- **HighVolatility/LowVolatility**: Different volatility regimes
- **GapUp/GapDown**: Price gap scenarios
- **FlashCrash**: Sudden market crashes
- **Recovery**: Post-crash recovery patterns
- **Consolidation**: Tight trading ranges

## Edge Cases

The framework tests various edge cases:

- Missing OHLCV values
- Extreme price outliers
- Zero or negative volumes
- Corrupted timestamp sequences
- Duplicate bars
- Extreme volatility spikes

## Performance Requirements

Default performance requirements:
- End-to-end latency: < 100ms
- Minimum throughput: 10 signals/second
- Maximum memory usage: 512MB
- Concurrent symbols: 5+

## Output

The framework generates comprehensive reports including:

- Test execution summary with pass/fail rates
- Individual test case results with detailed error messages
- Performance metrics and requirement validation
- System health score and recommendations
- HTML reports with charts and visualizations
- JSON reports for programmatic analysis

## Dependencies

Core dependencies:
- `tokio`: Async runtime
- `serde`: Serialization/deserialization
- `chrono`: Date/time handling
- `uuid`: Unique identifiers
- `rand`: Random data generation
- `anyhow`: Error handling

Note: Some dependencies (polars, reqwest, local crates) are temporarily disabled due to OpenSSL build issues in the current environment.

## Future Enhancements

- Binary executables for command-line usage (test-runner, test-report-generator)
- Integration with actual IMP system components
- Real-time test execution monitoring
- Test result trend analysis
- Automated performance regression detection