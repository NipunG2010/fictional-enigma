# LDC Engine Automation & CI/CD Integration

## Overview

This document covers the automated test runner framework and CI/CD integration examples for the LDC engine, including GitHub Actions, GitLab CI, Jenkins, Docker, pre-commit hooks, and monitoring.

---

## Part 1: Automated Test Runner

The automated test runner supports:
- **Parallel Test Execution**: Run independent test suites in parallel for faster feedback
- **Test Selection Strategies**: All tests, specific categories, pattern matching, or changed-file-based selection
- **Performance Regression Detection**: Track metrics over time and detect regressions
- **CI/CD Integration**: Machine-readable reports, proper exit codes, timeout handling
- **Resource Management**: Automatic cleanup and resource monitoring

### Quick Start

```bash
# Build the test runner
cargo build --release --bin test_runner

# Run all tests
./target/release/test_runner

# Run specific test categories
./target/release/test_runner --categories unit,mathematical

# Run with custom timeout
./target/release/test_runner --timeout 600

# Run tests for changed files (CI/CD)
./target/release/test_runner --changed-files src/lib.rs,tests/unit_tests.rs

# Enable verbose output / machine-readable output
./target/release/test_runner --verbose
./target/release/test_runner --machine-readable
```

#### Using the CI/CD Script

```bash
./scripts/ci_test_runner.sh              # Run full test suite
./scripts/ci_test_runner.sh --quick      # Unit + mathematical only
./scripts/ci_test_runner.sh --changed-only
./scripts/ci_test_runner.sh --performance-only
./scripts/ci_test_runner.sh --fail-fast
```

### Configuration

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

Environment variables: `TIMEOUT_SECONDS` (default 1800), `MAX_RETRIES` (default 2), `PARALLEL_JOBS` (default nproc), `UPDATE_DEPS` (default false), `CI_BASE_REF` (default origin/main).

### Test Categories

- **Unit**: Fast unit tests for individual components
- **Integration**: Tests that verify component interactions
- **Performance**: Performance benchmarks and validation tests
- **Mathematical**: Mathematical accuracy and precision tests
- **Backtesting**: Historical backtesting framework tests
- **Statistical**: Statistical analysis and validation tests
- **Compatibility**: Pine Script compatibility tests

### Performance Regression Detection

```json
{
  "performance_regressions": [{
    "suite_name": "performance_validation",
    "metric_name": "execution_time_ms",
    "baseline_value": 1000.0,
    "current_value": 1300.0,
    "regression_percent": 30.0,
    "severity": "Major"
  }]
}
```

Severity thresholds: Minor (10-20%), Major (20-50%), Critical (>50%)

Baselines are stored in `test_reports/performance_baseline.json` and auto-updated when all tests pass.

### Exit Codes

- `0`: All tests passed
- `1`: Some tests failed
- `2`: Test execution error
- `3`: Tests timed out

### Report Formats

- **JSON** (`test_reports/latest.json`): Machine-readable for CI/CD
- **Text** (`test_reports/latest.txt`): Human-readable summary
- **JUnit XML** (`test_reports/junit.xml`): Standard CI/CD format
- **Coverage** (`test_reports/cobertura.xml`): Code coverage data

### Parallel Execution

Most suites run in parallel (unit, integration, mathematical, backtesting, statistical, compatibility). Performance tests and benchmarks run sequentially to avoid resource contention. Integration tests depend on unit tests; backtesting and statistical tests also depend on unit tests.

### API Usage

```rust
use ldc_engine::automated_test_runner::{
    AutomatedTestRunner, TestRunnerConfig, TestSelectionStrategy, TestCategory
};

let config = TestRunnerConfig {
    max_parallel_suites: 4,
    enable_regression_detection: true,
    test_selection: TestSelectionStrategy::Categories(vec![
        TestCategory::Unit,
        TestCategory::Performance
    ]),
    ..Default::default()
};

let mut runner = AutomatedTestRunner::new(config)?;
let report = runner.run_all_tests()?;

match report.summary.overall_status {
    TestStatus::Passed => println!("All tests passed!"),
    TestStatus::Failed => println!("Some tests failed!"),
    _ => println!("Test execution encountered issues"),
}

let exit_code = runner.get_exit_code(&report);
std::process::exit(exit_code);
```

---

## Part 2: CI/CD Integration Examples

### GitHub Actions

```yaml
# .github/workflows/comprehensive-testing.yml
name: Comprehensive Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  quick-validation:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
    - uses: actions/checkout@v4
    - uses: actions-rs/toolchain@v1
      with: { toolchain: stable, components: rustfmt clippy, override: true }
    - uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    - run: cargo fmt --check
    - run: cargo clippy --all-targets --all-features -- -D warnings
    - run: cargo test --test mathematical_accuracy_tests --verbose

  mathematical-tests:
    runs-on: ubuntu-latest
    needs: quick-validation
    strategy:
      matrix:
        tolerance: [1e-6, 1e-8, 1e-10]
    steps:
    - uses: actions/checkout@v4
    - uses: actions-rs/toolchain@v1
      with: { toolchain: stable }
    - run: |
        cargo test --test mathematical_accuracy_tests -- \
          --tolerance ${{ matrix.tolerance }} --nocapture
    - uses: actions/upload-artifact@v3
      if: always()
      with:
        name: mathematical-test-results-${{ matrix.tolerance }}
        path: test_reports/mathematical_*.json

  performance-tests:
    runs-on: ubuntu-latest
    needs: quick-validation
    strategy:
      matrix:
        dataset-size: [1000, 10000, 50000]
    steps:
    - uses: actions/checkout@v4
    - uses: actions-rs/toolchain@v1
      with: { toolchain: stable }
    - run: |
        cargo run --example performance_validation_demo -- \
          --dataset-size ${{ matrix.dataset-size }} \
          --config ci \
          --output-format json \
          --output-file test_reports/performance_${{ matrix.dataset-size }}.json
    - uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-test-results-${{ matrix.dataset-size }}
        path: test_reports/performance_*.json

  integration-tests:
    runs-on: ubuntu-latest
    needs: [mathematical-tests, performance-tests]
    steps:
    - uses: actions/checkout@v4
    - uses: actions-rs/toolchain@v1
      with: { toolchain: stable }
    - run: cargo test --test comprehensive_integration_tests --verbose
    - run: |
        cargo run --example end_to_end_pipeline -- \
          --config ci --test-data sample_data/btc_5m_sample.csv

  comprehensive-tests:
    runs-on: ubuntu-latest
    needs: [mathematical-tests, performance-tests, integration-tests]
    steps:
    - uses: actions/checkout@v4
    - uses: actions-rs/toolchain@v1
      with: { toolchain: stable }
    - run: |
        cargo run --example automated_test_runner_demo -- \
          --config ci \
          --output-format json \
          --output-file test_reports/comprehensive_results.json \
          --generate-artifacts
    - uses: actions/upload-artifact@v3
      if: always()
      with:
        name: comprehensive-test-results
        path: test_reports/
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = JSON.parse(fs.readFileSync('test_reports/comprehensive_results.json', 'utf8'));
          const comment = `## Test Results\n\n**Status:** ${results.summary.overall_status}\n**Success Rate:** ${results.summary.success_rate.toFixed(1)}%\n**Passed:** ${results.summary.passed_tests} / ${results.summary.total_tests}`;
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });

  nightly-regression:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    timeout-minutes: 120
    steps:
    - uses: actions/checkout@v4
    - uses: actions-rs/toolchain@v1
      with: { toolchain: stable }
    - run: |
        cargo run --example performance_regression_demo -- \
          --baseline baseline_results.json \
          --config production \
          --output-file regression_results.json
    - name: Create issue on regression
      if: failure()
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: 'Performance Regression Detected',
            body: `Performance regression detected in nightly tests.\n\nRun: ${{ github.run_id }}\nCommit: ${{ github.sha }}`,
            labels: ['bug', 'performance', 'regression']
          });
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages: [validate, test, integration, report, deploy]

variables:
  CARGO_HOME: $CI_PROJECT_DIR/.cargo
  RUST_BACKTRACE: "1"

cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths: [.cargo/, target/]

code-quality:
  stage: validate
  image: rust:latest
  script:
    - rustup component add rustfmt clippy
    - cargo fmt --check
    - cargo clippy --all-targets --all-features -- -D warnings

mathematical-tests:
  stage: test
  image: rust:latest
  parallel:
    matrix:
      - TOLERANCE: ["1e-6", "1e-8", "1e-10"]
  script:
    - cargo test --test mathematical_accuracy_tests -- --tolerance $TOLERANCE
  artifacts:
    reports:
      junit: test_reports/mathematical_tests_$TOLERANCE.xml
    expire_in: 1 week

performance-tests:
  stage: test
  image: rust:latest
  parallel:
    matrix:
      - DATASET_SIZE: [1000, 10000, 50000]
  script:
    - |
      cargo run --example performance_validation_demo -- \
        --dataset-size $DATASET_SIZE --config ci \
        --output-format junit --output-file test_reports/performance_$DATASET_SIZE.xml
  artifacts:
    reports:
      junit: test_reports/performance_*.xml
    expire_in: 1 week

integration-tests:
  stage: integration
  image: rust:latest
  script:
    - cargo test --test comprehensive_integration_tests --verbose
  needs: ["mathematical-tests", "performance-tests"]

pages:
  stage: deploy
  script:
    - mkdir public
    - cp -r test_reports/* public/
  artifacts:
    paths: [public]
  only: [main]
  needs: ["integration-tests"]
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    environment {
        CARGO_HOME = "${WORKSPACE}/.cargo"
        RUST_BACKTRACE = "1"
    }
    options {
        timeout(time: 2, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }
    stages {
        stage('Code Quality') {
            parallel {
                stage('Format') { steps { sh 'cargo fmt --check' } }
                stage('Lint')   { steps { sh 'cargo clippy --all-targets --all-features -- -D warnings' } }
            }
        }
        stage('Comprehensive Testing') {
            parallel {
                stage('Mathematical') {
                    steps {
                        sh 'cargo test --test mathematical_accuracy_tests --verbose'
                    }
                }
                stage('Performance') {
                    steps {
                        sh '''
                            cargo run --example performance_validation_demo -- \
                                --config ci --output-format junit \
                                --output-file test_reports/performance.xml
                        '''
                    }
                }
                stage('Integration') {
                    steps {
                        sh 'cargo test --test comprehensive_integration_tests --verbose'
                    }
                }
            }
        }
        stage('Generate Reports') {
            steps {
                sh '''
                    cargo run --example automated_test_runner_demo -- \
                        --config ci --output-format html \
                        --output-file test_reports/report.html --include-charts
                '''
            }
        }
    }
    post {
        always {
            publishTestResults testResultsPattern: 'test_reports/*.xml'
            archiveArtifacts artifacts: 'test_reports/**/*', fingerprint: true
            publishHTML([
                reportDir: 'test_reports',
                reportFiles: 'report.html',
                reportName: 'LDC Test Report'
            ])
        }
        failure {
            emailext(
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Check the build at: ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

### Pre-commit Hooks

```bash
#!/bin/sh
# .git/hooks/pre-commit
set -e

echo "Running pre-commit tests..."

cargo test --test mathematical_accuracy_tests --quiet
if [ $? -ne 0 ]; then echo "Mathematical accuracy tests failed"; exit 1; fi

cargo run --example automated_test_runner_demo -- \
    --tests performance --quick-mode --timeout 30
if [ $? -ne 0 ]; then echo "Performance smoke tests failed"; exit 1; fi

cargo fmt --check
if [ $? -ne 0 ]; then echo "Code formatting check failed. Run 'cargo fmt' to fix."; exit 1; fi

cargo clippy -- -D warnings
if [ $? -ne 0 ]; then echo "Clippy found issues. Please fix them before committing."; exit 1; fi

echo "All pre-commit tests passed!"
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: mathematical-tests
        name: Mathematical Accuracy Tests
        entry: cargo test --test mathematical_accuracy_tests --quiet
        language: system
        pass_filenames: false
      - id: cargo-fmt
        name: Cargo Format
        entry: cargo fmt --check
        language: system
        pass_filenames: false
      - id: cargo-clippy
        name: Cargo Clippy
        entry: cargo clippy -- -D warnings
        language: system
        pass_filenames: false
```

### Docker Integration

```dockerfile
# Multi-stage Dockerfile for LDC testing
FROM rust:1.70 as builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY rust/ ./rust/
RUN cargo build --release
COPY . .
RUN cargo build --release --examples

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y ca-certificates python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install pandas numpy matplotlib seaborn
WORKDIR /app
COPY --from=builder /app/target/release/examples/* /usr/local/bin/
COPY config/ ./config/
COPY scripts/ ./scripts/
RUN mkdir -p test_reports
ENV RUST_LOG=info LDC_TEST_CONFIG=docker
CMD ["/usr/local/bin/automated_test_runner_demo", "--config", "docker"]
```

```yaml
# docker-compose.test.yml
services:
  ldc-test:
    build: { context: ., dockerfile: Dockerfile.test }
    environment: [RUST_LOG=info, LDC_TEST_CONFIG=docker, PARALLEL_JOBS=4]
    volumes: [./test_reports:/app/test_reports, ./config:/app/config:ro]
    command: >
      sh -c "
        /usr/local/bin/automated_test_runner_demo --config docker --output-format json --output-file test_reports/results.json &&
        /usr/local/bin/automated_test_runner_demo --config docker --output-format html --output-file test_reports/report.html --include-charts
      "
```

### Monitoring & Alerting

```rust
// Prometheus metrics for test monitoring
pub struct TestMetrics {
    test_duration: Histogram,
    test_failures: Counter,
    test_successes: Counter,
    performance_latency: Histogram,
    memory_usage: Gauge,
}

impl TestMetrics {
    pub fn record_test_result(&self, duration: f64, success: bool) {
        self.test_duration.observe(duration);
        if success { self.test_successes.inc(); } else { self.test_failures.inc(); }
    }
}
```

Prometheus alerting rules:
```yaml
groups:
  - name: ldc_testing_alerts
    rules:
      - alert: LDCTestFailureRate
        expr: rate(ldc_test_failures_total[5m]) / (rate(ldc_test_successes_total[5m]) + rate(ldc_test_failures_total[5m])) > 0.1
        for: 2m
        labels: { severity: warning }
        annotations:
          summary: "High test failure rate detected"
          
      - alert: LDCPerformanceRegression
        expr: histogram_quantile(0.95, rate(ldc_performance_latency_ms_bucket[5m])) > 10
        for: 5m
        labels: { severity: critical }
        annotations:
          summary: "95th percentile latency exceeds 10ms threshold"
```

### Development Makefile

```makefile
test-quick:
	cargo test --test mathematical_accuracy_tests
	cargo run --example performance_validation_demo -- --dataset-size 1000 --iterations 10

test-math:
	cargo test --test mathematical_accuracy_tests --verbose

test-perf:
	cargo run --example performance_validation_demo

test-integration:
	cargo test --test comprehensive_integration_tests

test-all:
	cargo run --example automated_test_runner_demo -- --config dev

report:
	cargo run --example automated_test_runner_demo -- \
		--config dev --output-format html \
		--output-file test_reports/comprehensive_report.html --include-charts

watch:
	cargo watch -x "test --test mathematical_accuracy_tests" \
		-x "run --example performance_validation_demo -- --dataset-size 1000 --iterations 5"
```
