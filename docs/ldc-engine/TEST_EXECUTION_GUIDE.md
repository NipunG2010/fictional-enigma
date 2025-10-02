# Test Execution Guide

## Overview

This guide provides detailed instructions for executing tests in the LDC engine testing framework, including usage examples, configuration options, and best practices for different testing scenarios.

## Quick Start

### Running All Tests

```bash
# Run the complete test suite with default configuration
cargo run --example automated_test_runner_demo

# Run with verbose output
cargo run --example automated_test_runner_demo -- --verbose

# Run with specific configuration
cargo run --example automated_test_runner_demo -- --config production
```

### Running Specific Test Categories

```bash
# Mathematical accuracy tests only
cargo test --test mathematical_accuracy_tests

# Performance validation tests only
cargo test --test performance_validation_tests

# Integration tests only
cargo test --test comprehensive_integration_tests

# Backtesting framework tests
cargo run --example backtesting_demo

# Statistical analysis tests
cargo run --example statistical_analysis_demo
```

## Test Categories and Usage

### 1. Mathematical Accuracy Tests

**Purpose**: Verify mathematical correctness of distance calculations and core algorithms.

**Command Line Usage**:
```bash
# Run all mathematical tests
cargo test --test mathematical_accuracy_tests

# Run with custom tolerance
cargo test --test mathematical_accuracy_tests -- --tolerance 1e-8

# Test specific implementations
cargo test --test mathematical_accuracy_tests simd_accuracy
cargo test --test mathematical_accuracy_tests hnsw_compatibility
```

**Programmatic Usage**:
```rust
use ldc_engine::testing::MathematicalTestSuite;

// Basic usage
let test_suite = MathematicalTestSuite::new();
let results = test_suite.run_all_tests()?;

// Custom configuration
let test_suite = MathematicalTestSuite::with_config(MathematicalTestConfig {
    tolerance: 1e-8,
    test_edge_cases: true,
    test_extreme_values: true,
    enable_detailed_logging: true,
});

// Run specific test categories
let simd_results = test_suite.test_simd_accuracy()?;
let hnsw_results = test_suite.test_hnsw_compatibility()?;
```

### 2. Performance Validation Tests

**Purpose**: Ensure the system meets performance requirements under various load conditions.

**Command Line Usage**:
```bash
# Run performance tests with default targets
cargo test --test performance_validation_tests

# Run with custom performance targets
cargo run --example performance_validation_demo -- \
    --target-1k 0.5 \
    --target-10k 1.0 \
    --target-50k 5.0 \
    --iterations 100

# Test specific dataset sizes
cargo run --example performance_validation_demo -- --dataset-size 10000
```

**Programmatic Usage**:
```rust
use ldc_engine::testing::{PerformanceValidator, PerformanceTestConfig};

// Create performance test configuration
let config = PerformanceTestConfig {
    target_latency_1k_samples_ms: 0.5,
    target_latency_10k_samples_ms: 1.0,
    target_latency_50k_samples_ms: 5.0,
    target_hnsw_accuracy_percent: 95.0,
    test_iterations: 100,
    warmup_iterations: 10,
};

// Run performance validation
let validator = PerformanceValidator::new(config);
let engine = LDCEngine::with_config(ldc_config);
let results = validator.validate_query_performance(&engine)?;

// Validate HNSW accuracy
let hnsw_results = validator.validate_hnsw_accuracy(&engine)?;
```

### 3. Integration Tests

**Purpose**: Test complete workflow from data input to signal output.

**Command Line Usage**:
```bash
# Run all integration tests
cargo test --test comprehensive_integration_tests

# Test specific integration scenarios
cargo test --test integration_testing_framework_tests pipeline_integration
cargo test --test integration_testing_framework_tests error_handling_integration
```

**Programmatic Usage**:
```rust
use ldc_engine::testing::IntegrationTestSuite;

// Create integration test suite
let test_suite = IntegrationTestSuite::new();

// Test complete pipeline
let pipeline_results = test_suite.test_complete_pipeline(
    &ohlcv_data,
    &features_data,
    &expected_signals
)?;

// Test error handling
let error_results = test_suite.test_error_handling_scenarios()?;

// Test configuration changes
let config_results = test_suite.test_dynamic_configuration_changes()?;
```

### 4. Backtesting Framework

**Purpose**: Validate trading strategies using historical market data.

**Command Line Usage**:
```bash
# Run backtesting demo
cargo run --example backtesting_demo

# Run with custom configuration
cargo run --example backtesting_demo -- \
    --initial-capital 100000 \
    --position-size 0.1 \
    --transaction-cost 0.001 \
    --signal-threshold 0.6

# Run performance backtesting
cargo run --example backtesting_performance_demo
```

**Programmatic Usage**:
```rust
use ldc_engine::backtesting::{BacktestingEngine, BacktestConfig};

// Create backtest configuration
let config = BacktestConfig {
    initial_capital: 100000.0,
    position_size: 0.1,
    transaction_cost: 0.001,
    signal_threshold: 0.6,
    max_positions: 5,
    rebalance_frequency: Duration::from_secs(3600),
};

// Run backtest
let mut engine = BacktestingEngine::new(config, ldc_config);
let results = engine.run_backtest(&ohlcv_data, &features_data)?;

// Analyze results
println!("Total Return: {:.2}%", results.total_return * 100.0);
println!("Sharpe Ratio: {:.2}", results.sharpe_ratio);
println!("Max Drawdown: {:.2}%", results.max_drawdown * 100.0);
```

### 5. Statistical Analysis

**Purpose**: Validate prediction quality and statistical significance.

**Command Line Usage**:
```bash
# Run statistical analysis demo
cargo run --example statistical_analysis_demo

# Run with custom confidence level
cargo run --example statistical_analysis_demo -- --confidence 0.99

# Run statistical validation demo
cargo run --example statistical_validation_demo
```

**Programmatic Usage**:
```rust
use ldc_engine::testing::{StatisticalAnalyzer, StatisticalConfig};

// Create statistical analyzer
let config = StatisticalConfig {
    confidence_level: 0.95,
    min_sample_size: 1000,
    significance_threshold: 0.05,
};

let analyzer = StatisticalAnalyzer::new(config);

// Analyze predictions
let results = analyzer.analyze_predictions(
    &predictions,
    &actual_outcomes,
    &market_data
)?;

// Check statistical significance
if results.statistical_significance.p_value < 0.05 {
    println!("Results are statistically significant");
} else {
    println!("Results are not statistically significant");
}
```

## Configuration Management

### Configuration Files

#### Development Configuration
```toml
# config/dev_test_config.toml
[mathematical]
tolerance = 1e-6
test_edge_cases = true
test_extreme_values = true

[performance]
target_latency_1k_samples_ms = 1.0
target_latency_10k_samples_ms = 2.0
target_latency_50k_samples_ms = 10.0
target_hnsw_accuracy_percent = 90.0
test_iterations = 50
warmup_iterations = 5

[statistical]
confidence_level = 0.90
min_sample_size = 500
significance_threshold = 0.05

[execution]
enable_detailed_logging = true
enable_performance_profiling = true
parallel_execution = false
timeout_seconds = 600
```

#### Production Configuration
```toml
# config/prod_test_config.toml
[mathematical]
tolerance = 1e-8
test_edge_cases = true
test_extreme_values = true

[performance]
target_latency_1k_samples_ms = 0.5
target_latency_10k_samples_ms = 1.0
target_latency_50k_samples_ms = 5.0
target_hnsw_accuracy_percent = 95.0
test_iterations = 100
warmup_iterations = 10

[statistical]
confidence_level = 0.95
min_sample_size = 1000
significance_threshold = 0.05

[execution]
enable_detailed_logging = false
enable_performance_profiling = false
parallel_execution = true
timeout_seconds = 300
```

#### CI/CD Configuration
```toml
# config/ci_test_config.toml
[mathematical]
tolerance = 1e-6
test_edge_cases = true
test_extreme_values = false  # Skip extreme tests in CI

[performance]
target_latency_1k_samples_ms = 2.0   # Relaxed for CI environment
target_latency_10k_samples_ms = 4.0
target_latency_50k_samples_ms = 15.0
target_hnsw_accuracy_percent = 95.0
test_iterations = 50
warmup_iterations = 5

[statistical]
confidence_level = 0.95
min_sample_size = 500
significance_threshold = 0.05

[execution]
enable_detailed_logging = true
enable_performance_profiling = false
parallel_execution = true
timeout_seconds = 300
```

### Loading Configuration

```rust
use ldc_engine::testing::TestConfig;

// Load from file
let config = TestConfig::from_file("config/prod_test_config.toml")?;

// Load from environment
let config = TestConfig::from_env()?;

// Create programmatically
let config = TestConfig {
    mathematical: MathematicalTestConfig {
        tolerance: 1e-6,
        test_edge_cases: true,
        test_extreme_values: true,
    },
    performance: PerformanceTestConfig {
        target_latency_1k_samples_ms: 0.5,
        target_latency_10k_samples_ms: 1.0,
        target_latency_50k_samples_ms: 5.0,
        target_hnsw_accuracy_percent: 95.0,
        test_iterations: 100,
        warmup_iterations: 10,
    },
    statistical: StatisticalConfig {
        confidence_level: 0.95,
        min_sample_size: 1000,
        significance_threshold: 0.05,
    },
    execution: ExecutionConfig {
        enable_detailed_logging: true,
        enable_performance_profiling: false,
        parallel_execution: true,
        timeout_seconds: 300,
    },
};
```

## Advanced Usage Patterns

### Parallel Test Execution

```rust
use rayon::prelude::*;
use ldc_engine::testing::*;

// Run test categories in parallel
let test_results: Vec<_> = vec![
    TestCategory::Mathematical,
    TestCategory::Performance,
    TestCategory::Integration,
    TestCategory::Statistical,
].into_par_iter()
.map(|category| {
    match category {
        TestCategory::Mathematical => {
            let suite = MathematicalTestSuite::new();
            suite.run_all_tests()
        },
        TestCategory::Performance => {
            let validator = PerformanceValidator::new(perf_config);
            validator.validate_all(&engine)
        },
        TestCategory::Integration => {
            let suite = IntegrationTestSuite::new();
            suite.run_all_tests()
        },
        TestCategory::Statistical => {
            let analyzer = StatisticalAnalyzer::new(stats_config);
            analyzer.analyze_all(&predictions, &outcomes, &market_data)
        },
    }
})
.collect();
```

### Custom Test Suites

```rust
use ldc_engine::testing::{TestSuite, TestResult};

// Create custom test suite
struct CustomTradingStrategyTests {
    strategy_config: StrategyConfig,
}

impl TestSuite for CustomTradingStrategyTests {
    fn run_all_tests(&self) -> Result<TestResult> {
        let mut results = TestResult::new();
        
        // Test strategy initialization
        results.add_test_result(self.test_strategy_initialization()?);
        
        // Test signal generation
        results.add_test_result(self.test_signal_generation()?);
        
        // Test risk management
        results.add_test_result(self.test_risk_management()?);
        
        Ok(results)
    }
}

impl CustomTradingStrategyTests {
    fn test_strategy_initialization(&self) -> Result<UnitTestResult> {
        // Custom test implementation
        Ok(UnitTestResult {
            test_name: "strategy_initialization".to_string(),
            passed: true,
            expected: 1.0,
            actual: 1.0,
            difference: 0.0,
            tolerance: 1e-6,
        })
    }
    
    // Additional custom test methods...
}
```

### Test Data Management

```rust
use ldc_engine::testing::TestDataManager;

// Manage test data lifecycle
let data_manager = TestDataManager::new();

// Generate synthetic data
let synthetic_data = data_manager.generate_synthetic_ohlcv(
    1000,  // number of samples
    Duration::from_secs(300),  // 5-minute intervals
    SyntheticDataConfig {
        volatility: 0.02,
        trend: 0.001,
        noise_level: 0.1,
    }
)?;

// Load historical data
let historical_data = data_manager.load_historical_data(
    "BTCUSDT",
    "2023-01-01",
    "2023-12-31",
    "5m"
)?;

// Cache test data for reuse
data_manager.cache_test_data("test_dataset_1", &synthetic_data)?;
let cached_data = data_manager.load_cached_data("test_dataset_1")?;
```

## Monitoring and Reporting

### Real-time Test Monitoring

```rust
use ldc_engine::testing::{TestMonitor, TestEvent};

// Create test monitor
let monitor = TestMonitor::new();

// Subscribe to test events
monitor.subscribe(|event: TestEvent| {
    match event {
        TestEvent::TestStarted { name, category } => {
            println!("🚀 Started test: {} ({})", name, category);
        },
        TestEvent::TestCompleted { name, result, duration } => {
            let status = if result.passed { "✅" } else { "❌" };
            println!("{} Completed test: {} ({:.2}ms)", status, name, duration.as_millis());
        },
        TestEvent::TestFailed { name, error } => {
            println!("💥 Test failed: {} - {}", name, error);
        },
    }
});

// Run tests with monitoring
let results = monitor.run_monitored_tests(test_suite)?;
```

### Automated Report Generation

```rust
use ldc_engine::testing::{ReportGenerator, ReportFormat};

// Generate comprehensive reports
let report_generator = ReportGenerator::new();

// HTML report with charts
report_generator.generate_report(
    &test_results,
    ReportFormat::Html,
    "test_reports/comprehensive_report.html"
)?;

// JSON report for CI/CD integration
report_generator.generate_report(
    &test_results,
    ReportFormat::Json,
    "test_reports/results.json"
)?;

// XML report for test result parsers
report_generator.generate_report(
    &test_results,
    ReportFormat::Xml,
    "test_reports/results.xml"
)?;

// Custom report with specific metrics
let custom_report = report_generator.create_custom_report()
    .include_performance_metrics()
    .include_statistical_analysis()
    .include_trend_analysis()
    .generate(&test_results)?;
```

## Best Practices

### Test Organization

1. **Separate test categories**: Keep mathematical, performance, integration, and statistical tests in separate modules
2. **Use descriptive test names**: Include the test purpose and expected outcome in the name
3. **Implement proper cleanup**: Ensure tests clean up resources and don't affect other tests
4. **Use appropriate timeouts**: Set reasonable timeouts for different test categories

### Performance Testing

1. **Warm up before measurement**: Run warmup iterations to stabilize performance
2. **Use multiple iterations**: Average results over multiple runs for stability
3. **Account for system variance**: Set realistic targets that account for system differences
4. **Monitor resource usage**: Track CPU, memory, and I/O usage during tests

### Statistical Testing

1. **Ensure adequate sample sizes**: Use sufficient data for statistical significance
2. **Test across market regimes**: Validate performance in different market conditions
3. **Use proper statistical methods**: Apply appropriate statistical tests for your data
4. **Document assumptions**: Clearly state statistical assumptions and limitations

### CI/CD Integration

1. **Use appropriate configurations**: Adjust targets for CI environment constraints
2. **Implement proper error handling**: Ensure tests fail gracefully with useful error messages
3. **Generate machine-readable output**: Provide structured output for automated processing
4. **Cache dependencies**: Use caching to speed up CI/CD pipeline execution

## Troubleshooting

### Common Issues

#### Test Timeouts
```bash
# Increase timeout for slow systems
cargo run --example automated_test_runner_demo -- --timeout 600

# Run tests with reduced dataset sizes
cargo run --example automated_test_runner_demo -- --dataset-size small
```

#### Memory Issues
```bash
# Run with memory constraints
cargo run --example automated_test_runner_demo -- --max-memory 1GB

# Enable memory profiling
cargo run --example automated_test_runner_demo -- --profile-memory
```

#### Performance Variance
```bash
# Run with more iterations for stability
cargo run --example automated_test_runner_demo -- --iterations 200

# Use relaxed performance targets
cargo run --example automated_test_runner_demo -- --config relaxed
```

### Debug Mode

```bash
# Enable debug logging
RUST_LOG=debug cargo run --example automated_test_runner_demo

# Enable trace logging for specific components
RUST_LOG=ldc_engine::testing=trace cargo run --example automated_test_runner_demo

# Save debug output to file
cargo run --example automated_test_runner_demo 2>&1 | tee debug.log
```

## Environment-Specific Considerations

### Development Environment
- Use relaxed performance targets
- Enable detailed logging and profiling
- Run smaller test datasets for faster feedback
- Use single-threaded execution for easier debugging

### CI/CD Environment
- Account for virtualized environment overhead
- Use appropriate timeouts for build systems
- Generate machine-readable reports
- Implement proper error codes for build failure detection

### Production Environment
- Use strict performance targets
- Minimize logging overhead
- Run comprehensive test suites
- Generate detailed reports for analysis

## Integration with Development Workflow

### Pre-commit Hooks
```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run quick mathematical accuracy tests
cargo test --test mathematical_accuracy_tests --quiet

if [ $? -ne 0 ]; then
    echo "Mathematical accuracy tests failed. Commit aborted."
    exit 1
fi

echo "All pre-commit tests passed."
```

### IDE Integration
```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Mathematical Tests",
            "type": "shell",
            "command": "cargo",
            "args": ["test", "--test", "mathematical_accuracy_tests"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run Performance Tests",
            "type": "shell",
            "command": "cargo",
            "args": ["run", "--example", "performance_validation_demo"],
            "group": "test"
        }
    ]
}
```

This comprehensive test execution guide provides detailed instructions for running all types of tests in the LDC engine testing framework, with practical examples and best practices for different scenarios and environments.