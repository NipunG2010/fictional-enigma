# CI/CD Integration Guide

This document describes how to integrate the end-to-end testing framework with CI/CD pipelines.

## Overview

The end-to-end testing framework provides comprehensive CI/CD integration capabilities including:

- **Automated Test Execution**: GitHub Actions workflows for running tests on every PR and push
- **Performance Regression Detection**: Automated comparison against baseline performance metrics
- **Test Report Generation**: HTML and JSON reports with charts and detailed analysis
- **Trend Analysis**: Historical performance tracking and regression detection
- **PR Status Updates**: Automatic comments on pull requests with test results

## GitHub Actions Workflows

### Main End-to-End Tests Workflow

**File**: `.github/workflows/end-to-end-tests.yml`

This workflow runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` branch
- Daily schedule (2 AM UTC)
- Manual dispatch

**Features**:
- Runs complete end-to-end test suite
- Generates HTML and JSON reports
- Uploads test artifacts
- Comments on PRs with results
- Publishes reports to GitHub Pages
- Creates issues for failures on main branches

### Performance Regression Detection

**File**: `.github/workflows/performance-regression.yml`

This workflow runs on:
- Pull requests affecting Rust code
- Manual dispatch with configurable baseline

**Features**:
- Compares performance against baseline branch
- Detects regressions in pass rate, duration, and health score
- Comments on PRs with regression analysis
- Uploads detailed comparison artifacts

## Local Development Tools

### Performance Comparison Script

**File**: `scripts/run-performance-comparison.sh`

Compare performance between two git references locally:

```bash
# Compare current branch against main
./scripts/run-performance-comparison.sh

# Compare specific branches
./scripts/run-performance-comparison.sh -b v1.0.0 -c feature-branch

# Use different test configuration
./scripts/run-performance-comparison.sh -t ci
```

### CI Helper Binary

**Binary**: `rust/end-to-end-tests/src/bin/ci_helper.rs`

Utility for CI/CD integration tasks:

```bash
# Generate test configuration for CI environment
cargo run --bin ci-helper -- generate-config --environment ci

# Process test results for CI integration
cargo run --bin ci-helper -- process-results --input report.json --baseline baseline.json

# Check for performance regressions
cargo run --bin ci-helper -- check-regressions --current current.json --baseline baseline.json --fail-on-regression

# Generate status report for GitHub Actions
cargo run --bin ci-helper -- generate-status --input report.json --format github
```

## Test Configurations

The framework supports different test configurations optimized for different environments:

### CI Configuration
- Reduced test duration and parallelism
- Relaxed performance thresholds
- Optimized for CI environment constraints

### Local Configuration
- Full test coverage with reference validation
- Moderate performance requirements
- Suitable for local development

### Performance Configuration
- Strict performance requirements
- Extended test duration
- Maximum test coverage
- Used for performance regression detection

## Report Generation

### Comprehensive Test Reports

The framework generates detailed test reports including:

- **Test Summary**: Pass rates, duration, health scores
- **Interactive Charts**: Visual representation of test results
- **Detailed Test Cases**: Individual test results with error messages
- **Performance Metrics**: Latency, throughput, and resource usage
- **Recommendations**: Actionable suggestions for improvements

### Trend Analysis

Historical analysis capabilities:

- **Pass Rate Trends**: Track test stability over time
- **Performance Trends**: Monitor system performance evolution
- **Regression Detection**: Automatic identification of performance degradations
- **Comparison Reports**: Detailed comparison between test runs

## Integration Examples

### GitHub Actions Output

The workflows generate structured outputs for integration:

```yaml
- name: Use test results
  run: |
    echo "Pass rate: ${{ steps.tests.outputs.pass_rate }}%"
    echo "Duration: ${{ steps.tests.outputs.duration_minutes }} minutes"
    echo "Failed tests: ${{ steps.tests.outputs.failed_tests }}"
```

### Status Checks

Configure branch protection rules to require test passage:

```yaml
# .github/branch-protection.yml
protection_rules:
  main:
    required_status_checks:
      - "End-to-End Integration Tests"
      - "Performance Regression Detection"
```

### Notifications

The framework can integrate with various notification systems:

- **GitHub Issues**: Automatic issue creation for test failures
- **PR Comments**: Detailed test results and regression analysis
- **Status Updates**: Real-time status updates during test execution

## Configuration

### Environment Variables

Configure the CI environment using these variables:

```bash
# Test execution
RUST_BACKTRACE=1
CARGO_TERM_COLOR=always

# Service endpoints (if using external services)
REDIS_URL=redis://localhost:6379
KAFKA_BROKERS=localhost:9092
HMM_SERVICE_URL=http://localhost:8080
```

### Test Configuration Files

Customize test behavior using TOML configuration files:

```toml
# test_config.toml
[pipeline_tests]
test_symbols = ["BTCUSDT", "ETHUSDT"]
test_duration_hours = 1
validate_against_reference = false

[performance_tests]
max_end_to_end_latency_ms = 150
concurrent_symbols = 2
test_duration_minutes = 2

[execution]
max_parallel_tests = 2
test_timeout_seconds = 300
verbose_logging = true
```

## Troubleshooting

### Common Issues

1. **Test Timeouts**: Increase `test_timeout_seconds` in configuration
2. **Service Connection Failures**: Verify service health checks in workflows
3. **Performance Regressions**: Review baseline selection and thresholds
4. **Report Generation Failures**: Check output directory permissions

### Debug Mode

Enable verbose logging for debugging:

```bash
# Set in test configuration
verbose_logging = true

# Or via environment variable
RUST_LOG=debug cargo run --bin test-runner
```

### Artifact Collection

All workflows collect comprehensive artifacts:

- Test results (JSON and HTML)
- Performance metrics
- Comparison reports
- Debug logs
- Configuration files

Access artifacts through the GitHub Actions interface or API.

## Best Practices

### Test Configuration
- Use appropriate configuration for each environment
- Keep CI tests fast but comprehensive
- Use performance configuration for regression detection

### Baseline Management
- Update baselines regularly
- Use stable releases as baselines for comparison
- Document baseline selection criteria

### Report Analysis
- Review trend analysis regularly
- Act on performance regressions promptly
- Use recommendations to improve test coverage

### CI/CD Pipeline
- Run tests on every PR
- Use branch protection rules
- Monitor test execution times
- Set up notifications for failures

## Support

For issues or questions about CI/CD integration:

1. Check the workflow logs for detailed error messages
2. Review the test configuration for environment-specific settings
3. Use the debug mode for additional logging
4. Consult the main documentation for framework details