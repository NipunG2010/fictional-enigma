# Integration Examples

## Overview

This document provides comprehensive examples of integrating the LDC engine testing framework into various development workflows, CI/CD pipelines, and deployment scenarios.

## Development Workflow Integration

### IDE Integration

#### Visual Studio Code Integration

**Configuration Files:**

`.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Mathematical Tests",
            "type": "shell",
            "command": "cargo",
            "args": ["test", "--test", "mathematical_accuracy_tests", "--", "--nocapture"],
            "group": {
                "kind": "test",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": true,
                "clear": false
            },
            "problemMatcher": "$rustc"
        },
        {
            "label": "Run Performance Tests",
            "type": "shell",
            "command": "cargo",
            "args": ["run", "--example", "performance_validation_demo"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run Complete Test Suite",
            "type": "shell",
            "command": "cargo",
            "args": ["run", "--example", "automated_test_runner_demo", "--", "--config", "dev"],
            "group": "test",
            "dependsOrder": "sequence",
            "dependsOn": ["Run Mathematical Tests"],
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": true,
                "panel": "shared"
            }
        },
        {
            "label": "Generate Test Report",
            "type": "shell",
            "command": "cargo",
            "args": [
                "run", "--example", "automated_test_runner_demo", "--",
                "--output-format", "html",
                "--output-file", "test_reports/latest.html",
                "--include-charts"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "silent",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
```

`.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Mathematical Tests",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/target/debug/deps/mathematical_accuracy_tests-${command:rust-analyzer.getTargetExecutable}",
            "args": ["--nocapture"],
            "cwd": "${workspaceFolder}",
            "sourceLanguages": ["rust"],
            "preLaunchTask": "cargo build --test mathematical_accuracy_tests"
        },
        {
            "name": "Debug Performance Tests",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/target/debug/examples/performance_validation_demo",
            "args": ["--dataset-size", "1000", "--iterations", "10"],
            "cwd": "${workspaceFolder}",
            "sourceLanguages": ["rust"],
            "preLaunchTask": "cargo build --example performance_validation_demo"
        }
    ]
}
```

`.vscode/settings.json`:
```json
{
    "rust-analyzer.cargo.features": ["testing"],
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.checkOnSave.extraArgs": ["--", "-W", "clippy::all"],
    "files.watcherExclude": {
        "**/target/**": true,
        "**/test_reports/**": true
    },
    "terminal.integrated.env.linux": {
        "RUST_LOG": "debug",
        "LDC_TEST_CONFIG": "dev"
    }
}
```

#### IntelliJ IDEA / CLion Integration

**Run Configurations:**

Create run configurations for different test scenarios:

1. **Mathematical Tests Configuration:**
   - Name: "Mathematical Accuracy Tests"
   - Command: `cargo test --test mathematical_accuracy_tests`
   - Working directory: `$PROJECT_DIR$`
   - Environment variables: `RUST_LOG=debug`

2. **Performance Tests Configuration:**
   - Name: "Performance Validation"
   - Command: `cargo run --example performance_validation_demo`
   - Program arguments: `--config dev --iterations 50`

3. **Complete Test Suite Configuration:**
   - Name: "Complete Test Suite"
   - Command: `cargo run --example automated_test_runner_demo`
   - Program arguments: `--config dev --output-format json --output-file test_reports/dev_results.json`

### Pre-commit Hooks

**Git Hooks Integration:**

`.git/hooks/pre-commit`:
```bash
#!/bin/sh
# Pre-commit hook for LDC engine testing

set -e

echo "Running pre-commit tests..."

# Run quick mathematical accuracy tests
echo "🧮 Running mathematical accuracy tests..."
cargo test --test mathematical_accuracy_tests --quiet

if [ $? -ne 0 ]; then
    echo "❌ Mathematical accuracy tests failed. Commit aborted."
    exit 1
fi

# Run basic performance validation (quick version)
echo "⚡ Running basic performance validation..."
cargo run --example performance_validation_demo -- \
    --dataset-size 1000 \
    --iterations 10 \
    --timeout 30 \
    --quiet

if [ $? -ne 0 ]; then
    echo "❌ Performance validation failed. Commit aborted."
    exit 1
fi

# Check code formatting
echo "🎨 Checking code formatting..."
cargo fmt --check

if [ $? -ne 0 ]; then
    echo "❌ Code formatting check failed. Run 'cargo fmt' to fix."
    exit 1
fi

# Run clippy for linting
echo "📎 Running clippy..."
cargo clippy -- -D warnings

if [ $? -ne 0 ]; then
    echo "❌ Clippy found issues. Please fix them before committing."
    exit 1
fi

echo "✅ All pre-commit tests passed!"
```

**Using pre-commit framework:**

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
        
      - id: performance-validation
        name: Basic Performance Validation
        entry: cargo run --example performance_validation_demo -- --dataset-size 1000 --iterations 10 --quiet
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

### Development Scripts

**Makefile for common tasks:**

```makefile
# Makefile for LDC Engine Testing

.PHONY: test test-math test-perf test-integration test-all clean setup help

# Default target
help:
	@echo "Available targets:"
	@echo "  test-math       - Run mathematical accuracy tests"
	@echo "  test-perf       - Run performance validation tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-all        - Run complete test suite"
	@echo "  test-quick      - Run quick test suite for development"
	@echo "  benchmark       - Run performance benchmarks"
	@echo "  report          - Generate comprehensive test report"
	@echo "  clean           - Clean test artifacts"
	@echo "  setup           - Setup development environment"

# Quick tests for development
test-quick:
	@echo "🚀 Running quick test suite..."
	cargo test --test mathematical_accuracy_tests
	cargo run --example performance_validation_demo -- --dataset-size 1000 --iterations 10

# Mathematical accuracy tests
test-math:
	@echo "🧮 Running mathematical accuracy tests..."
	cargo test --test mathematical_accuracy_tests --verbose

# Performance validation tests
test-perf:
	@echo "⚡ Running performance validation tests..."
	cargo run --example performance_validation_demo

# Integration tests
test-integration:
	@echo "🔗 Running integration tests..."
	cargo test --test comprehensive_integration_tests

# Complete test suite
test-all:
	@echo "🎯 Running complete test suite..."
	cargo run --example automated_test_runner_demo -- --config dev

# Performance benchmarks
benchmark:
	@echo "📊 Running performance benchmarks..."
	cargo run --example performance_benchmarking_demo

# Generate comprehensive report
report:
	@echo "📋 Generating comprehensive test report..."
	cargo run --example automated_test_runner_demo -- \
		--config dev \
		--output-format html \
		--output-file test_reports/comprehensive_report.html \
		--include-charts
	@echo "Report generated: test_reports/comprehensive_report.html"

# Clean test artifacts
clean:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf test_reports/*
	rm -rf target/debug/deps/*test*
	cargo clean

# Setup development environment
setup:
	@echo "🛠️  Setting up development environment..."
	rustup component add rustfmt clippy
	cargo install cargo-watch
	mkdir -p test_reports
	mkdir -p config
	@echo "✅ Development environment setup complete"

# Watch mode for continuous testing
watch:
	@echo "👀 Starting watch mode..."
	cargo watch -x "test --test mathematical_accuracy_tests" -x "run --example performance_validation_demo -- --dataset-size 1000 --iterations 5"
```

**Development helper scripts:**

`scripts/dev-test.sh`:
```bash
#!/bin/bash
# Development testing script

set -e

# Configuration
CONFIG_FILE="${LDC_TEST_CONFIG:-dev}"
VERBOSE="${VERBOSE:-false}"
QUICK_MODE="${QUICK_MODE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run tests with proper error handling
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    log_info "Running $test_name..."
    
    if [ "$VERBOSE" = "true" ]; then
        eval "$test_command"
    else
        eval "$test_command" > /dev/null 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        log_success "$test_name passed"
        return 0
    else
        log_error "$test_name failed"
        return 1
    fi
}

# Main testing workflow
main() {
    log_info "Starting LDC Engine development tests..."
    log_info "Configuration: $CONFIG_FILE"
    log_info "Quick mode: $QUICK_MODE"
    
    local failed_tests=0
    
    # Mathematical accuracy tests
    if ! run_test "Mathematical Accuracy Tests" "cargo test --test mathematical_accuracy_tests"; then
        ((failed_tests++))
    fi
    
    # Performance tests (quick or full)
    if [ "$QUICK_MODE" = "true" ]; then
        perf_command="cargo run --example performance_validation_demo -- --dataset-size 1000 --iterations 10"
    else
        perf_command="cargo run --example performance_validation_demo"
    fi
    
    if ! run_test "Performance Validation" "$perf_command"; then
        ((failed_tests++))
    fi
    
    # Integration tests (skip in quick mode)
    if [ "$QUICK_MODE" != "true" ]; then
        if ! run_test "Integration Tests" "cargo test --test comprehensive_integration_tests"; then
            ((failed_tests++))
        fi
    fi
    
    # Summary
    if [ $failed_tests -eq 0 ]; then
        log_success "All tests passed! ✨"
        exit 0
    else
        log_error "$failed_tests test(s) failed"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -v, --verbose    Enable verbose output"
            echo "  -q, --quick      Run in quick mode (reduced test scope)"
            echo "  -c, --config     Specify configuration file"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
```

## CI/CD Pipeline Integration

### GitHub Actions

**Complete workflow configuration:**

`.github/workflows/comprehensive-testing.yml`:
```yaml
name: Comprehensive Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run nightly tests at 2 AM UTC
    - cron: '0 2 * * *'

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  # Quick validation job for fast feedback
  quick-validation:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        components: rustfmt, clippy
        override: true
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        restore-keys: |
          ${{ runner.os }}-cargo-
          
    - name: Check formatting
      run: cargo fmt --check
      
    - name: Run clippy
      run: cargo clippy --all-targets --all-features -- -D warnings
      
    - name: Quick mathematical tests
      run: cargo test --test mathematical_accuracy_tests --verbose

  # Mathematical accuracy tests
  mathematical-tests:
    runs-on: ubuntu-latest
    needs: quick-validation
    timeout-minutes: 15
    
    strategy:
      matrix:
        tolerance: [1e-6, 1e-8, 1e-10]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Run mathematical accuracy tests
      run: |
        cargo test --test mathematical_accuracy_tests -- \
          --tolerance ${{ matrix.tolerance }} \
          --nocapture
          
    - name: Upload mathematical test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: mathematical-test-results-${{ matrix.tolerance }}
        path: test_reports/mathematical_*.json

  # Performance validation tests
  performance-tests:
    runs-on: ubuntu-latest
    needs: quick-validation
    timeout-minutes: 30
    
    strategy:
      matrix:
        dataset-size: [1000, 10000, 50000]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Run performance validation
      run: |
        cargo run --example performance_validation_demo -- \
          --dataset-size ${{ matrix.dataset-size }} \
          --config ci \
          --output-format json \
          --output-file test_reports/performance_${{ matrix.dataset-size }}.json
          
    - name: Analyze performance results
      run: |
        python scripts/analyze_performance.py \
          test_reports/performance_${{ matrix.dataset-size }}.json
          
    - name: Upload performance test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-test-results-${{ matrix.dataset-size }}
        path: test_reports/performance_*.json

  # Integration tests
  integration-tests:
    runs-on: ubuntu-latest
    needs: [mathematical-tests, performance-tests]
    timeout-minutes: 45
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Run integration tests
      run: |
        cargo test --test comprehensive_integration_tests --verbose
        
    - name: Run end-to-end pipeline test
      run: |
        cargo run --example end_to_end_pipeline -- \
          --config ci \
          --test-data sample_data/btc_5m_sample.csv
          
    - name: Upload integration test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-results
        path: test_reports/integration_*.json

  # Comprehensive test suite
  comprehensive-tests:
    runs-on: ubuntu-latest
    needs: [mathematical-tests, performance-tests, integration-tests]
    timeout-minutes: 60
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Run comprehensive test suite
      run: |
        cargo run --example automated_test_runner_demo -- \
          --config ci \
          --output-format json \
          --output-file test_reports/comprehensive_results.json \
          --generate-artifacts
          
    - name: Generate HTML report
      run: |
        cargo run --example automated_test_runner_demo -- \
          --config ci \
          --output-format html \
          --output-file test_reports/comprehensive_report.html \
          --include-charts
          
    - name: Upload comprehensive test results
      uses: actions/upload-artifact@v3
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
          
          const comment = `## 🧪 Test Results
          
          **Overall Status:** ${results.summary.overall_status}
          **Success Rate:** ${results.summary.success_rate.toFixed(1)}%
          **Total Tests:** ${results.summary.total_tests}
          **Passed:** ${results.summary.passed_tests}
          **Failed:** ${results.summary.failed_tests}
          
          ### Performance Metrics
          - **Average Latency:** ${results.performance_summary.avg_latency_ms.toFixed(2)}ms
          - **HNSW Accuracy:** ${results.performance_summary.hnsw_accuracy.toFixed(1)}%
          - **Memory Usage:** ${results.performance_summary.peak_memory_mb.toFixed(1)}MB
          
          [📊 View Detailed Report](https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }})
          `;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });

  # Nightly performance regression tests
  nightly-regression:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    timeout-minutes: 120
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Download baseline results
      run: |
        curl -H "Authorization: token ${{ secrets.GITHUB_TOKEN }}" \
          -o baseline_results.json \
          "https://api.github.com/repos/${{ github.repository }}/releases/latest/assets/baseline_results.json"
          
    - name: Run regression tests
      run: |
        cargo run --example performance_regression_demo -- \
          --baseline baseline_results.json \
          --config production \
          --output-file regression_results.json
          
    - name: Upload regression results
      uses: actions/upload-artifact@v3
      with:
        name: regression-test-results
        path: regression_results.json
        
    - name: Create issue on regression
      if: failure()
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: '🚨 Performance Regression Detected',
            body: `Performance regression detected in nightly tests.
            
            **Run:** ${{ github.run_id }}
            **Commit:** ${{ github.sha }}
            
            Please investigate the performance degradation.`,
            labels: ['bug', 'performance', 'regression']
          });
```

### GitLab CI

**GitLab CI configuration:**

`.gitlab-ci.yml`:
```yaml
stages:
  - validate
  - test
  - integration
  - report
  - deploy

variables:
  CARGO_HOME: $CI_PROJECT_DIR/.cargo
  RUST_BACKTRACE: "1"

cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - .cargo/
    - target/

# Validation stage
code-quality:
  stage: validate
  image: rust:latest
  script:
    - rustup component add rustfmt clippy
    - cargo fmt --check
    - cargo clippy --all-targets --all-features -- -D warnings
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

quick-tests:
  stage: validate
  image: rust:latest
  script:
    - cargo test --test mathematical_accuracy_tests --verbose
  artifacts:
    reports:
      junit: test_reports/quick_tests.xml
    paths:
      - test_reports/
    expire_in: 1 week
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# Test stage
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
    paths:
      - test_reports/
    expire_in: 1 week
  needs: ["quick-tests"]

performance-tests:
  stage: test
  image: rust:latest
  parallel:
    matrix:
      - DATASET_SIZE: [1000, 10000, 50000]
  script:
    - |
      cargo run --example performance_validation_demo -- \
        --dataset-size $DATASET_SIZE \
        --config ci \
        --output-format junit \
        --output-file test_reports/performance_$DATASET_SIZE.xml
  artifacts:
    reports:
      junit: test_reports/performance_*.xml
    paths:
      - test_reports/
    expire_in: 1 week
  needs: ["quick-tests"]

# Integration stage
integration-tests:
  stage: integration
  image: rust:latest
  script:
    - cargo test --test comprehensive_integration_tests --verbose
    - |
      cargo run --example end_to_end_pipeline -- \
        --config ci \
        --test-data sample_data/btc_5m_sample.csv
  artifacts:
    reports:
      junit: test_reports/integration_tests.xml
    paths:
      - test_reports/
    expire_in: 1 week
  needs: ["mathematical-tests", "performance-tests"]

# Report stage
comprehensive-report:
  stage: report
  image: rust:latest
  script:
    - |
      cargo run --example automated_test_runner_demo -- \
        --config ci \
        --output-format json \
        --output-file test_reports/comprehensive_results.json
    - |
      cargo run --example automated_test_runner_demo -- \
        --config ci \
        --output-format html \
        --output-file test_reports/comprehensive_report.html \
        --include-charts
  artifacts:
    paths:
      - test_reports/
    expire_in: 1 month
  needs: ["integration-tests"]

# Deploy test results to pages
pages:
  stage: deploy
  script:
    - mkdir public
    - cp -r test_reports/* public/
  artifacts:
    paths:
      - public
  only:
    - main
  needs: ["comprehensive-report"]
```

### Jenkins Pipeline

**Jenkinsfile:**

```groovy
pipeline {
    agent any
    
    environment {
        CARGO_HOME = "${WORKSPACE}/.cargo"
        RUST_BACKTRACE = "1"
        PATH = "${CARGO_HOME}/bin:${PATH}"
    }
    
    options {
        timeout(time: 2, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }
    
    stages {
        stage('Setup') {
            steps {
                script {
                    // Install Rust if not available
                    sh '''
                        if ! command -v cargo &> /dev/null; then
                            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                            source ~/.cargo/env
                        fi
                        rustup component add rustfmt clippy
                    '''
                }
            }
        }
        
        stage('Code Quality') {
            parallel {
                stage('Format Check') {
                    steps {
                        sh 'cargo fmt --check'
                    }
                }
                
                stage('Lint') {
                    steps {
                        sh 'cargo clippy --all-targets --all-features -- -D warnings'
                    }
                }
            }
        }
        
        stage('Quick Validation') {
            steps {
                sh 'cargo test --test mathematical_accuracy_tests --verbose'
            }
            post {
                always {
                    publishTestResults testResultsPattern: 'test_reports/quick_*.xml'
                }
            }
        }
        
        stage('Comprehensive Testing') {
            parallel {
                stage('Mathematical Tests') {
                    steps {
                        script {
                            def tolerances = ['1e-6', '1e-8', '1e-10']
                            for (tolerance in tolerances) {
                                sh """
                                    cargo test --test mathematical_accuracy_tests -- \
                                        --tolerance ${tolerance} \
                                        --output-format junit \
                                        --output-file test_reports/math_${tolerance}.xml
                                """
                            }
                        }
                    }
                }
                
                stage('Performance Tests') {
                    steps {
                        script {
                            def sizes = [1000, 10000, 50000]
                            for (size in sizes) {
                                sh """
                                    cargo run --example performance_validation_demo -- \
                                        --dataset-size ${size} \
                                        --config ci \
                                        --output-format junit \
                                        --output-file test_reports/perf_${size}.xml
                                """
                            }
                        }
                    }
                }
                
                stage('Integration Tests') {
                    steps {
                        sh '''
                            cargo test --test comprehensive_integration_tests --verbose
                            cargo run --example end_to_end_pipeline -- \
                                --config ci \
                                --test-data sample_data/btc_5m_sample.csv
                        '''
                    }
                }
            }
        }
        
        stage('Generate Reports') {
            steps {
                sh '''
                    cargo run --example automated_test_runner_demo -- \
                        --config ci \
                        --output-format json \
                        --output-file test_reports/comprehensive_results.json
                        
                    cargo run --example automated_test_runner_demo -- \
                        --config ci \
                        --output-format html \
                        --output-file test_reports/comprehensive_report.html \
                        --include-charts
                '''
            }
        }
        
        stage('Performance Regression Check') {
            when {
                branch 'main'
            }
            steps {
                script {
                    // Download baseline from previous successful build
                    try {
                        copyArtifacts(
                            projectName: env.JOB_NAME,
                            selector: lastSuccessful(),
                            filter: 'baseline_results.json',
                            optional: true
                        )
                        
                        sh '''
                            if [ -f baseline_results.json ]; then
                                cargo run --example performance_regression_demo -- \
                                    --baseline baseline_results.json \
                                    --current test_reports/comprehensive_results.json \
                                    --threshold 0.1
                            else
                                echo "No baseline found, skipping regression check"
                            fi
                        '''
                    } catch (Exception e) {
                        echo "Regression check failed: ${e.getMessage()}"
                    }
                }
            }
        }
    }
    
    post {
        always {
            // Publish test results
            publishTestResults testResultsPattern: 'test_reports/*.xml'
            
            // Archive artifacts
            archiveArtifacts artifacts: 'test_reports/**/*', fingerprint: true
            
            // Publish HTML reports
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'test_reports',
                reportFiles: 'comprehensive_report.html',
                reportName: 'LDC Test Report'
            ])
        }
        
        success {
            script {
                if (env.BRANCH_NAME == 'main') {
                    // Save current results as baseline for future regression checks
                    sh 'cp test_reports/comprehensive_results.json baseline_results.json'
                    archiveArtifacts artifacts: 'baseline_results.json', fingerprint: true
                }
            }
        }
        
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: """
                    Build failed for ${env.JOB_NAME} - ${env.BUILD_NUMBER}
                    
                    Check the build at: ${env.BUILD_URL}
                    
                    Recent changes:
                    ${env.CHANGE_LOG}
                """,
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
        
        unstable {
            emailext (
                subject: "Build Unstable: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: """
                    Build is unstable for ${env.JOB_NAME} - ${env.BUILD_NUMBER}
                    
                    Some tests may have failed. Check the build at: ${env.BUILD_URL}
                """,
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

## Docker Integration

### Containerized Testing Environment

**Dockerfile for testing:**

```dockerfile
# Multi-stage Dockerfile for LDC testing
FROM rust:1.70 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY Cargo.toml Cargo.lock ./
COPY rust/ ./rust/

# Build dependencies (cached layer)
RUN cargo build --release

# Copy source code
COPY . .

# Build the application
RUN cargo build --release --examples

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for analysis scripts
RUN pip3 install pandas numpy matplotlib seaborn

# Create app user
RUN useradd -m -u 1000 app

# Set working directory
WORKDIR /app

# Copy built binaries and examples
COPY --from=builder /app/target/release/examples/* /usr/local/bin/
COPY --from=builder /app/target/release/deps/*test* /usr/local/bin/

# Copy configuration and scripts
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY sample_data/ ./sample_data/

# Create test reports directory
RUN mkdir -p test_reports && chown -R app:app /app

# Switch to app user
USER app

# Set environment variables
ENV RUST_LOG=info
ENV LDC_TEST_CONFIG=docker

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/local/bin/mathematical_accuracy_tests --quick || exit 1

# Default command
CMD ["/usr/local/bin/automated_test_runner_demo", "--config", "docker"]
```

**Docker Compose for testing environment:**

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  ldc-test:
    build:
      context: .
      dockerfile: Dockerfile.test
    environment:
      - RUST_LOG=info
      - LDC_TEST_CONFIG=docker
      - PARALLEL_JOBS=4
    volumes:
      - ./test_reports:/app/test_reports
      - ./config:/app/config:ro
    command: >
      sh -c "
        echo 'Running comprehensive test suite...' &&
        /usr/local/bin/automated_test_runner_demo --config docker --output-format json --output-file test_reports/docker_results.json &&
        echo 'Generating HTML report...' &&
        /usr/local/bin/automated_test_runner_demo --config docker --output-format html --output-file test_reports/docker_report.html --include-charts
      "
    
  ldc-performance:
    build:
      context: .
      dockerfile: Dockerfile.test
    environment:
      - RUST_LOG=info
      - LDC_TEST_CONFIG=performance
    volumes:
      - ./test_reports:/app/test_reports
    command: >
      /usr/local/bin/performance_validation_demo 
      --config performance 
      --output-format json 
      --output-file test_reports/performance_results.json
    
  ldc-regression:
    build:
      context: .
      dockerfile: Dockerfile.test
    environment:
      - RUST_LOG=info
    volumes:
      - ./test_reports:/app/test_reports
      - ./baselines:/app/baselines:ro
    command: >
      /usr/local/bin/performance_regression_demo 
      --baseline baselines/latest.json 
      --output-file test_reports/regression_results.json
    depends_on:
      - ldc-performance

  test-reporter:
    image: python:3.9-slim
    volumes:
      - ./test_reports:/app/test_reports
      - ./scripts:/app/scripts:ro
    working_dir: /app
    command: >
      sh -c "
        pip install pandas matplotlib seaborn jinja2 &&
        python scripts/generate_summary_report.py test_reports/ test_reports/summary.html
      "
    depends_on:
      - ldc-test
      - ldc-performance
```

**Test execution scripts:**

`scripts/run-docker-tests.sh`:
```bash
#!/bin/bash
# Docker-based test execution script

set -e

# Configuration
COMPOSE_FILE="docker-compose.test.yml"
REPORT_DIR="./test_reports"
CLEANUP=${CLEANUP:-true}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    if [ "$CLEANUP" = "true" ]; then
        log_info "Cleaning up Docker resources..."
        docker-compose -f $COMPOSE_FILE down --volumes --remove-orphans
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    log_info "Starting Docker-based LDC testing..."
    
    # Create report directory
    mkdir -p $REPORT_DIR
    
    # Build test images
    log_info "Building test images..."
    docker-compose -f $COMPOSE_FILE build
    
    # Run comprehensive tests
    log_info "Running comprehensive test suite..."
    if docker-compose -f $COMPOSE_FILE run --rm ldc-test; then
        log_success "Comprehensive tests completed"
    else
        log_error "Comprehensive tests failed"
        exit 1
    fi
    
    # Run performance tests
    log_info "Running performance validation..."
    if docker-compose -f $COMPOSE_FILE run --rm ldc-performance; then
        log_success "Performance tests completed"
    else
        log_error "Performance tests failed"
        exit 1
    fi
    
    # Run regression tests (if baseline exists)
    if [ -f "baselines/latest.json" ]; then
        log_info "Running regression tests..."
        if docker-compose -f $COMPOSE_FILE run --rm ldc-regression; then
            log_success "Regression tests completed"
        else
            log_warning "Regression tests failed or detected regressions"
        fi
    else
        log_warning "No baseline found, skipping regression tests"
    fi
    
    # Generate summary report
    log_info "Generating summary report..."
    if docker-compose -f $COMPOSE_FILE run --rm test-reporter; then
        log_success "Summary report generated"
    else
        log_warning "Failed to generate summary report"
    fi
    
    # Display results
    log_info "Test execution completed!"
    log_info "Reports available in: $REPORT_DIR"
    
    if [ -f "$REPORT_DIR/docker_report.html" ]; then
        log_info "Main report: $REPORT_DIR/docker_report.html"
    fi
    
    if [ -f "$REPORT_DIR/summary.html" ]; then
        log_info "Summary report: $REPORT_DIR/summary.html"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        --report-dir)
            REPORT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-cleanup     Don't clean up Docker resources after execution"
            echo "  --report-dir     Specify report directory (default: ./test_reports)"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
```

## Monitoring and Alerting Integration

### Prometheus Metrics Integration

**Metrics collection:**

```rust
use prometheus::{Counter, Histogram, Gauge, Registry, Encoder, TextEncoder};
use std::sync::Arc;

pub struct TestMetrics {
    registry: Registry,
    test_duration: Histogram,
    test_failures: Counter,
    test_successes: Counter,
    performance_latency: Histogram,
    memory_usage: Gauge,
    cpu_usage: Gauge,
}

impl TestMetrics {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Registry::new();
        
        let test_duration = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "ldc_test_duration_seconds",
                "Duration of test execution in seconds"
            ).buckets(vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0])
        )?;
        
        let test_failures = Counter::with_opts(
            prometheus::CounterOpts::new(
                "ldc_test_failures_total",
                "Total number of test failures"
            )
        )?;
        
        let test_successes = Counter::with_opts(
            prometheus::CounterOpts::new(
                "ldc_test_successes_total",
                "Total number of test successes"
            )
        )?;
        
        let performance_latency = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "ldc_performance_latency_ms",
                "Performance test latency in milliseconds"
            ).buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
        )?;
        
        let memory_usage = Gauge::with_opts(
            prometheus::GaugeOpts::new(
                "ldc_memory_usage_mb",
                "Memory usage during tests in MB"
            )
        )?;
        
        let cpu_usage = Gauge::with_opts(
            prometheus::GaugeOpts::new(
                "ldc_cpu_usage_percent",
                "CPU usage during tests in percent"
            )
        )?;
        
        // Register metrics
        registry.register(Box::new(test_duration.clone()))?;
        registry.register(Box::new(test_failures.clone()))?;
        registry.register(Box::new(test_successes.clone()))?;
        registry.register(Box::new(performance_latency.clone()))?;
        registry.register(Box::new(memory_usage.clone()))?;
        registry.register(Box::new(cpu_usage.clone()))?;
        
        Ok(Self {
            registry,
            test_duration,
            test_failures,
            test_successes,
            performance_latency,
            memory_usage,
            cpu_usage,
        })
    }
    
    pub fn record_test_result(&self, duration: f64, success: bool) {
        self.test_duration.observe(duration);
        if success {
            self.test_successes.inc();
        } else {
            self.test_failures.inc();
        }
    }
    
    pub fn record_performance_latency(&self, latency_ms: f64) {
        self.performance_latency.observe(latency_ms);
    }
    
    pub fn update_resource_usage(&self, memory_mb: f64, cpu_percent: f64) {
        self.memory_usage.set(memory_mb);
        self.cpu_usage.set(cpu_percent);
    }
    
    pub fn export_metrics(&self) -> Result<String, Box<dyn std::error::Error>> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

// HTTP server for metrics endpoint
use warp::Filter;

pub async fn start_metrics_server(metrics: Arc<TestMetrics>, port: u16) {
    let metrics_route = warp::path("metrics")
        .map(move || {
            match metrics.export_metrics() {
                Ok(metrics_text) => warp::reply::with_status(
                    metrics_text,
                    warp::http::StatusCode::OK
                ),
                Err(_) => warp::reply::with_status(
                    "Error exporting metrics".to_string(),
                    warp::http::StatusCode::INTERNAL_SERVER_ERROR
                ),
            }
        });
    
    warp::serve(metrics_route)
        .run(([0, 0, 0, 0], port))
        .await;
}
```

### Grafana Dashboard Configuration

**Dashboard JSON:**

```json
{
  "dashboard": {
    "id": null,
    "title": "LDC Engine Testing Dashboard",
    "tags": ["ldc", "testing", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Test Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(ldc_test_successes_total[5m]) / (rate(ldc_test_successes_total[5m]) + rate(ldc_test_failures_total[5m])) * 100",
            "legendFormat": "Success Rate %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 80},
                {"color": "green", "value": 95}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Test Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ldc_test_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(ldc_test_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "unit": "s",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Performance Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ldc_performance_latency_ms_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(ldc_performance_latency_ms_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "unit": "ms",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "ldc_memory_usage_mb",
            "legendFormat": "Memory (MB)"
          },
          {
            "expr": "ldc_cpu_usage_percent",
            "legendFormat": "CPU (%)"
          }
        ],
        "yAxes": [
          {
            "unit": "short",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### Alerting Rules

**Prometheus alerting rules:**

```yaml
# alerts.yml
groups:
  - name: ldc_testing_alerts
    rules:
      - alert: LDCTestFailureRate
        expr: rate(ldc_test_failures_total[5m]) / (rate(ldc_test_successes_total[5m]) + rate(ldc_test_failures_total[5m])) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High test failure rate detected"
          description: "LDC test failure rate is {{ $value | humanizePercentage }} over the last 5 minutes"
          
      - alert: LDCPerformanceRegression
        expr: histogram_quantile(0.95, rate(ldc_performance_latency_ms_bucket[5m])) > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Performance regression detected"
          description: "95th percentile latency is {{ $value }}ms, exceeding 10ms threshold"
          
      - alert: LDCHighMemoryUsage
        expr: ldc_memory_usage_mb > 2048
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage during testing"
          description: "Memory usage is {{ $value }}MB, exceeding 2GB threshold"
          
      - alert: LDCTestsDown
        expr: up{job="ldc-testing"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LDC testing service is down"
          description: "LDC testing service has been down for more than 1 minute"
```

This comprehensive integration guide provides practical examples for incorporating the LDC engine testing framework into various development workflows, CI/CD pipelines, and monitoring systems, enabling seamless integration across different environments and tools.