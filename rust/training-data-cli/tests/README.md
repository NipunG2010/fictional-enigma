# Integration Tests for Training Data CLI

This directory contains comprehensive integration tests for the training data management CLI tool.

## Test Structure

### `integration_tests.rs`
End-to-end workflow tests that verify the complete functionality:
- **End-to-end workflow**: Tests the complete pipeline from OHLCV data to labeled training snapshots
- **Validation workflow**: Tests data quality validation and reporting
- **Configuration management**: Tests saving, loading, and managing configurations
- **Error scenarios**: Tests error handling and recovery mechanisms
- **LDC engine compatibility**: Verifies output format compatibility
- **Output formats**: Tests different output formats (Parquet, CSV, JSON)

### `cli_integration_tests.rs`
CLI-specific tests focusing on user experience:
- **Help and version commands**: Tests CLI documentation and version info
- **Argument validation**: Tests input validation and error messages
- **Verbose output**: Tests detailed logging and progress indicators
- **Configuration files**: Tests configuration file handling
- **Error message quality**: Tests user-friendly error reporting

### `performance_tests.rs`
Performance benchmarks and scalability tests:
- **Dataset size scaling**: Tests performance with different dataset sizes
- **Memory usage**: Monitors memory consumption patterns
- **Validation performance**: Benchmarks data validation speed
- **Feature computation**: Tests different feature scenarios
- **Concurrent processing**: Tests parallel processing capabilities
- **Stress testing**: Tests with edge case data (high volatility, sparse data, extreme values)

## Running Tests

### Run All Integration Tests
```bash
cd rust/training-data-cli
cargo test --test integration_tests
cargo test --test cli_integration_tests
cargo test --test performance_tests
```

### Run Specific Test Categories
```bash
# End-to-end workflow tests
cargo test --test integration_tests test_end_to_end_workflow

# CLI argument validation
cargo test --test cli_integration_tests test_argument_validation_errors

# Performance benchmarks
cargo test --test performance_tests benchmark_dataset_sizes
```

### Run Tests with Output
```bash
# Show test output (useful for performance metrics)
cargo test --test performance_tests -- --nocapture

# Run tests in single thread (for accurate performance measurements)
cargo test --test performance_tests -- --test-threads=1
```

## Test Requirements

### Prerequisites
1. **Sample Data**: Tests expect sample OHLCV data at `../sample/ohlcv.parquet`
2. **Built Binary**: Tests require the `training-data` binary to be built
3. **Sufficient Disk Space**: Performance tests create temporary datasets up to 100MB

### Building the Binary
```bash
cd rust/training-data-cli
cargo build
```

### Sample Data Format
The tests expect OHLCV data with the following schema:
```
timestamp: DateTime<Utc>
open: f64
high: f64
low: f64
close: f64
volume: f64
```

## Test Data Generation

The tests include synthetic data generators for various scenarios:

### `create_synthetic_dataset()`
Generates realistic OHLCV data with:
- Configurable number of rows
- 5-minute intervals
- Realistic price movements using sine waves
- Proper OHLCV relationships

### `create_test_data_with_issues()`
Creates data with known quality issues:
- Missing values (NaN)
- Duplicate timestamps
- Timestamp gaps
- Statistical outliers

### Performance Test Data Generators
- **High volatility data**: Large price swings for stress testing
- **Sparse data**: Data with missing values
- **Extreme values**: Data with outliers and edge cases

## Expected Test Results

### Performance Benchmarks
- **Throughput**: Should process at least 1,000 rows/second for normal data
- **Memory usage**: Should use less than 1GB for 25,000 rows
- **Scaling**: Throughput should remain relatively stable across dataset sizes
- **Processing time**: Should complete 100,000 rows in under 30 seconds

### Validation Performance
- **Speed**: Should validate at least 10,000 rows/second
- **Accuracy**: Should detect all injected data quality issues
- **Reporting**: Should generate valid JSON reports

### CLI Usability
- **Help text**: Should provide comprehensive usage information
- **Error messages**: Should be clear and actionable
- **Argument validation**: Should catch invalid inputs before processing

## Troubleshooting

### Common Issues

1. **Binary not found**: Ensure `cargo build` has been run
2. **Sample data missing**: Check that `../sample/ohlcv.parquet` exists
3. **Permission errors**: Ensure write permissions in test directories
4. **Memory issues**: Reduce dataset sizes in performance tests if needed

### Test Failures

If tests fail due to incomplete implementation:
- Tests are designed to be informative rather than blocking
- Many tests will show warnings instead of failures for unimplemented features
- Check test output for specific error messages and implementation status

### Performance Variations

Performance test results may vary based on:
- System specifications (CPU, memory, disk speed)
- System load during test execution
- Debug vs release build configuration

For consistent performance measurements:
- Run tests on an idle system
- Use release builds: `cargo test --release`
- Run performance tests single-threaded: `--test-threads=1`

## Extending Tests

### Adding New Test Cases

1. **Integration tests**: Add to `integration_tests.rs` for new workflow scenarios
2. **CLI tests**: Add to `cli_integration_tests.rs` for new CLI features
3. **Performance tests**: Add to `performance_tests.rs` for new benchmarks

### Test Data Scenarios

To add new test data scenarios:
1. Create a new data generator function
2. Add it to the appropriate test module
3. Include it in stress testing scenarios

### Validation Scenarios

To test new validation rules:
1. Create test data with specific issues
2. Add validation test cases
3. Verify error detection and reporting