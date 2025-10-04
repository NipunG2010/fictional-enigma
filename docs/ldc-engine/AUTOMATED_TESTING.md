
This document describes the automated test execution framework for the LDC Engine, which provides comprehensive testing capabilities with CI/CD integration, performance regression detection, and parallel test execution.

## Overview

The automated test runner supports:

- **Parallel Test Execution**: Run independent test suites in parallel for faster feedback
- **Test Selection Strategies**: Run all tests, specific categories, pattern matching, or tests affected by changed files
- **Performance Regression Detection**: Track performance metrics over time and detect regressions
- **CI/CD Integration**: Machine-readable reports, proper exit codes, and timeout handling
- **Resource Management**: Automatic cleanup and resource monitoring
- **Comprehensive Reporting**: Human-readable and machine-readable test reports

## Quick Start

### Using the CLI Tool

```bash
# Build the test runner
cargo build --release --bin test_runner

# Run all tests
./target/release/test_runner

# Run specific test categories
./target/release/test_runner --categories unit,mathematical

# Run tests with custom timeout
./target/release/test_runner --timeout 600

# Run tests for changed files (useful in CI/CD)
./target/release/test_runner --changed-files src/lib.rs,tests/unit_tests.rs

# Enable verbose output
./target/release/test_runner --verbose

# Generate machine-readable output
./target/release/test_runner --machine-readable
```

### Using the CI/CD Script

```bash
# Run full test suite
./scripts/ci_test_runner.sh

# Run quick tests (unit + mathematical only)
./scripts/ci_test_runner.sh --quick

# Run tests for changed files only
./scripts/ci_test_runner.sh --changed-only

# Run performance tests only
./scripts/ci_test_runner.sh --performance-only

# Stop on first failure
./scripts/ci_test_runner.sh --fail-fast
```

## Configuration

### Configuration File

Create a `test_runner_config.json` file to customize the test runner behavior:

```json
{
  "max_parallel_suites": 4,
  "suite_timeout_seconds": 300,
  "test_timeout_seconds": 60,
  "enable_regression_detection": true,
  "regression_threshold_percent": 10.0,
  "output_directory": "test_reports",
  "machine_readable_output": true,
  "test_selection": "All",
  "cleanup_timeout_seconds": 30,
  "verbose": false
}
```

### Environment Variables

The CI/CD script supports these environment variables:

- `TIMEOUT_SECONDS`: Test execution timeout (default: 1800)
- `MAX_RETRIES`: Maximum retry attempts (default: 2)
- `PARALLEL_JOBS`: Number of parallel test jobs (default: nproc)
- `UPDATE_DEPS`: Update dependencies before testing (default: false)
- `CI_BASE_REF`: Base git reference for changed file detection (default: origin/main)

## Test Categories

The framework organizes tests into categories:

- **Unit**: Fast unit tests for individual components
- **Integration**: Tests that verify component interactions
- **Performance**: Performance benchmarks and validation tests
- **Mathematical**: Mathematical accuracy and precision tests
- **Backtesting**: Historical backtesting framework tests
- **Statistical**: Statistical analysis and validation tests
- **Compatibility**: Pine Script compatibility tests

## Test Selection Strategies

### 1. All Tests
```bash
./target/release/test_runner
```

### 2. Specific Categories
```bash
./target/release/test_runner --categories unit,performance
```

### 3. Pattern Matching
```bash
./target/release/test_runner --pattern mathematical
```

### 4. Changed Files (CI/CD)
```bash
./target/release/test_runner --changed-files src/lib.rs,tests/test.rs
```

## Performance Regression Detection

The framework tracks performance metrics over time and detects regressions:

### Regression Thresholds
- **Minor**: 10-20% performance degradation
- **Major**: 20-50% performance degradation  
- **Critical**: >50% performance degradation

### Baseline Management
- Performance baselines are automatically updated when all tests pass
- Baselines are stored in `test_reports/performance_baseline.json`
- Historical performance data is tracked in `test_reports/test_history.json`

### Example Regression Report
```json
{
  "performance_regressions": [
    {
      "suite_name": "performance_validation",
      "metric_name": "execution_time_ms",
      "baseline_value": 1000.0,
      "current_value": 1300.0,
      "regression_percent": 30.0,
      "severity": "Major"
    }
  ]
}
```

## CI/CD Integration

### GitHub Actions

The framework includes a comprehensive GitHub Actions workflow (`.github/workflows/ci.yml`) that:

- Runs quick tests on pull requests
- Executes full test suites on main branch pushes
- Performs nightly performance testing
- Generates coverage reports
- Detects and reports performance regressions

### Exit Codes

The test runner uses standard exit codes for CI/CD integration:

- `0`: All tests passed
- `1`: Some tests failed
- `2`: Test execution error
- `3`: Tests timed out

### Report Formats

Multiple report formats are generated for different use cases:

- **JSON** (`test_reports/latest.json`): Machine-readable for CI/CD processing
- **Text** (`test_reports/latest.txt`): Human-readable summary
- **JUnit XML** (`test_reports/junit.xml`): Standard format for CI/CD systems
- **Coverage** (`test_reports/cobertura.xml`): Code coverage data

## Parallel Execution

### Parallel-Safe Tests
Most test suites can run in parallel:
- Unit tests
- Integration tests
- Mathematical accuracy tests
- Backtesting tests
- Statistical tests
- Compatibility tests

### Sequential Tests
Some tests must run sequentially:
- Performance tests (to avoid resource contention)
- Benchmarks (for accurate measurements)

### Dependency Resolution
The framework automatically resolves test dependencies:
- Integration tests depend on unit tests
- Backtesting tests depend on unit tests
- Statistical tests depend on unit tests

## Resource Management

### Timeouts
- **Suite Timeout**: Maximum time for a test suite (default: 300s)
- **Test Timeout**: Maximum time for individual tests (default: 60s)
- **Cleanup Timeout**: Maximum time for resource cleanup (default: 30s)

### Resource Monitoring
- Memory usage tracking
- CPU utilization monitoring
- Disk space validation
- Automatic cleanup on timeout or failure

### Error Handling
- Graceful degradation on resource exhaustion
- Automatic retry for transient failures
- Comprehensive error reporting with actionable recommendations

## Troubleshooting

### Common Issues

#### Tests Timing Out
```bash
# Increase timeout
./target/release/test_runner --timeout 600

# Run tests sequentially
./target/release/test_runner --parallel 1
```

#### Performance Regressions
```bash
# Disable regression detection temporarily
./target/release/test_runner --no-regression-detection

# Adjust regression threshold
./target/release/test_runner --regression-threshold 20.0
```

#### Memory Issues
```bash
# Reduce parallel execution
export PARALLEL_JOBS=2
./scripts/ci_test_runner.sh
```

### Debug Mode
```bash
# Enable verbose logging
./target/release/test_runner --verbose

# Check system resources
./scripts/ci_test_runner.sh --help
```

### Log Files
- Execution logs: `test_reports/ci_execution.log`
- Failed test archives: `test_reports/failed_test_logs_*.tar.gz`
- Test history: `test_reports/test_history.json`

## API Usage

### Programmatic Usage

```rust
use ldc_engine::automated_test_runner::{
    AutomatedTestRunner, TestRunnerConfig, TestSelectionStrategy, TestCategory
};

// Create configuration
let config = TestRunnerConfig {
    max_parallel_suites: 4,
    enable_regression_detection: true,
    test_selection: TestSelectionStrategy::Categories(vec![
        TestCategory::Unit,
        TestCategory::Performance
    ]),
    ..Default::default()
};

// Create and run tests
let mut runner = AutomatedTestRunner::new(config)?;
let report = runner.run_all_tests()?;

// Check results
match report.summary.overall_status {
    TestStatus::Passed => println!("All tests passed!"),
    TestStatus::Failed => println!("Some tests failed!"),
    _ => println!("Test execution encountered issues"),
}

// Get exit code for CI/CD
let exit_code = runner.get_exit_code(&report);
std::process::exit(exit_code);
```

### Custom Test Suites

```rust
use ldc_engine::automated_test_runner::{TestSuite, TestCategory};

let custom_suite = TestSuite {
    name: "custom_validation".to_string(),
    category: TestCategory::Integration,
    command: "cargo".to_string(),
    args: vec!["test".to_string(), "custom".to_string()],
    working_directory: None,
    environment: HashMap::new(),
    timeout_seconds: Some(180),
    dependencies: vec!["unit_tests".to_string()],
    affected_by_files: vec!["src/custom/**/*.rs".to_string()],
    parallel_safe: true,
};
```

## Best Practices

### Test Organization
1. Keep unit tests fast (< 1 second each)
2. Use appropriate test categories
3. Make tests deterministic and repeatable
4. Include comprehensive error messages

### CI/CD Integration
1. Use quick tests for pull request validation
2. Run full test suites on main branch
3. Schedule nightly performance tests
4. Monitor performance trends over time

### Performance Testing
1. Run performance tests in isolation
2. Use consistent hardware for benchmarks
3. Establish stable baselines
4. Monitor for gradual performance degradation

### Error Handling
1. Provide actionable error messages
2. Include debugging information
3. Implement proper cleanup
4. Use appropriate retry strategies

## Examples

See the `examples/automated_test_runner_demo.rs` file for comprehensive usage examples covering:

- Basic configuration
- Test selection strategies
- Performance regression detection
- CI/CD integration features

Run the demo with:
```bash
cargo run --example automated_test_runner_demo
```

## Contributing

When adding new tests or modifying the test framework:

1. Follow the existing test suite structure
2. Add appropriate test categories and dependencies
3. Include performance metrics for regression detection
4. Update documentation and examples
5. Test CI/CD integration changes thoroughly

## Support

For issues or questions about the automated test framework:

1. Check the troubleshooting section above
2. Review the test execution logs
3. Run tests with verbose output for debugging
4. Consult the API documentation and examples