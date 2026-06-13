# LDC Engine Testing Guide

## Overview

This guide provides comprehensive documentation for testing the LDC (Lorentzian Distance Classifier) engine, including usage examples, best practices, and troubleshooting information.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Categories](#test-categories)
3. [Configuration Examples](#configuration-examples)
4. [Test Execution](#test-execution)
5. [Result Interpretation](#result-interpretation)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Integration Examples](#integration-examples)
9. [Best Practices](#best-practices)

## Additional Documentation

- **[Performance Testing & Tuning](performance.md)** - Benchmarking framework, performance targets, and system-level optimization
- **[Automation & CI/CD Integration](automation.md)** - Automated test runner and CI/CD pipeline examples
- **[Mathematical Accuracy Testing](mathematical-accuracy.md)** - Lorentzian distance calculation accuracy validation

## Quick Start

### Running All Tests

```bash
# Run the complete test suite
cargo run --example automated_test_runner_demo

# Run specific test categories
cargo test --test mathematical_accuracy_tests
cargo test --test performance_validation_tests
cargo test --test comprehensive_integration_tests
```

### Basic Test Configuration

```rust
use ldc_engine::testing::*;

// Create a basic test configuration
let config = TestConfig {
    mathematical_tolerance: 1e-6,
    performance_targets: PerformanceTargets::default(),
    statistical_confidence: 0.95,
    enable_detailed_logging: true,
};

// Run tests with configuration
let results = run_comprehensive_tests(config).await?;
```

## Test Categories

### 1. Mathematical Accuracy Tests

Tests the mathematical correctness of distance calculations and core algorithms.

**Purpose**: Ensure mathematical accuracy across different implementations (standard, SIMD, HNSW).

**Key Tests**:
- Lorentzian distance calculation accuracy
- SIMD vs standard implementation comparison
- HNSW compatibility verification
- Edge case handling (NaN, infinity, zero values)

**Example Usage**:
```rust
use ldc_engine::testing::MathematicalTestSuite;

let test_suite = MathematicalTestSuite::new();
let results = test_suite.run_all_tests()?;

// Check results
for result in results.test_results {
    if !result.passed {
        println!("FAILED: {} - Expected: {}, Got: {}", 
                result.test_name, result.expected, result.actual);
    }
}
```

### 2. Performance Validation Tests

Validates that the system meets performance requirements under various load conditions.

**Purpose**: Ensure the system can handle production workloads with acceptable latency and throughput.

**Key Metrics**:
- Query latency (1k samples: <0.5ms, 10k samples: <1ms, 50k samples: <5ms)
- HNSW accuracy (>95% vs exact search)
- Memory usage and CPU utilization
- Concurrent performance scaling

**Example Usage**:
```rust
use ldc_engine::testing::PerformanceValidator;

let config = PerformanceTestConfig {
    target_latency_1k_samples_ms: 0.5,
    target_latency_10k_samples_ms: 1.0,
    target_latency_50k_samples_ms: 5.0,
    target_hnsw_accuracy_percent: 95.0,
    test_iterations: 100,
    warmup_iterations: 10,
};

let validator = PerformanceValidator::new(config);
let results = validator.validate_query_performance(&engine)?;
```

### 3. Integration Tests

Tests the complete workflow from data input to signal output.

**Purpose**: Verify that all components work together correctly in realistic scenarios.

**Key Tests**:
- End-to-end OHLCV → Features → LDC → Signals pipeline
- Error handling and recovery mechanisms
- Configuration changes without restart
- Multi-threaded operation

**Example Usage**:
```rust
use ldc_engine::testing::IntegrationTestSuite;

let test_suite = IntegrationTestSuite::new();
let results = test_suite.test_complete_pipeline(
    &ohlcv_data,
    &expected_signals
)?;
```

### 4. Backtesting Framework

Historical strategy validation using real market data.

**Purpose**: Validate trading strategies and measure performance metrics using historical data.

**Key Features**:
- Historical data processing
- Performance metrics calculation (Sharpe ratio, drawdown, win rate)
- Trade-by-trade analysis
- Market regime analysis

**Example Usage**:
```rust
use ldc_engine::backtesting::BacktestingEngine;

let config = BacktestConfig {
    initial_capital: 100000.0,
    position_size: 0.1,
    transaction_cost: 0.001,
    signal_threshold: 0.6,
    max_positions: 5,
    rebalance_frequency: Duration::from_secs(3600),
};

let mut engine = BacktestingEngine::new(config, ldc_config);
let results = engine.run_backtest(&ohlcv_data, &features_data)?;
```

### 5. Statistical Analysis

Validates prediction quality and statistical significance.

**Purpose**: Ensure predictions have statistical validity and measure signal quality.

**Key Metrics**:
- Hit rate, precision, recall, F1 score
- Information coefficient
- Signal-to-noise ratio
- Statistical significance testing

**Example Usage**:
```rust
use ldc_engine::testing::StatisticalAnalyzer;

let analyzer = StatisticalAnalyzer::new(StatisticalConfig {
    confidence_level: 0.95,
    min_sample_size: 1000,
    significance_threshold: 0.05,
});

let results = analyzer.analyze_predictions(
    &predictions,
    &actual_outcomes,
    &market_data
)?;
```

## Configuration Examples

### Development Environment Configuration

```rust
pub fn create_dev_config() -> TestConfig {
    TestConfig {
        mathematical_tolerance: 1e-6,
        performance_targets: PerformanceTargets {
            target_latency_1k_samples_ms: 1.0,
            target_latency_10k_samples_ms: 2.0,
            target_latency_50k_samples_ms: 10.0,
            target_hnsw_accuracy_percent: 90.0,
        },
        statistical_confidence: 0.90,
        enable_detailed_logging: true,
        enable_performance_profiling: true,
        test_data_size: TestDataSize::Small,
        parallel_execution: false,
    }
}
```

### Production Environment Configuration

```rust
pub fn create_prod_config() -> TestConfig {
    TestConfig {
        mathematical_tolerance: 1e-8,
        performance_targets: PerformanceTargets {
            target_latency_1k_samples_ms: 0.5,
            target_latency_10k_samples_ms: 1.0,
            target_latency_50k_samples_ms: 5.0,
            target_hnsw_accuracy_percent: 95.0,
        },
        statistical_confidence: 0.95,
        enable_detailed_logging: false,
        enable_performance_profiling: false,
        test_data_size: TestDataSize::Large,
        parallel_execution: true,
    }
}
```

### CI/CD Configuration

```rust
pub fn create_ci_config() -> TestConfig {
    TestConfig {
        mathematical_tolerance: 1e-6,
        performance_targets: PerformanceTargets {
            target_latency_1k_samples_ms: 2.0,
            target_latency_10k_samples_ms: 4.0,
            target_latency_50k_samples_ms: 15.0,
            target_hnsw_accuracy_percent: 95.0,
        },
        statistical_confidence: 0.95,
        enable_detailed_logging: true,
        enable_performance_profiling: false,
        test_data_size: TestDataSize::Medium,
        parallel_execution: true,
        timeout_seconds: 300,
    }
}
```

## Test Execution

### Command Line Interface

```bash
# Run all tests with default configuration
cargo run --example automated_test_runner_demo

# Run with specific configuration
cargo run --example automated_test_runner_demo -- --config prod

# Run specific test categories
cargo run --example automated_test_runner_demo -- --tests mathematical,performance

# Run with custom parameters
cargo run --example automated_test_runner_demo -- \
    --tolerance 1e-8 \
    --performance-target 0.5 \
    --iterations 1000 \
    --parallel

# Generate detailed reports
cargo run --example automated_test_runner_demo -- \
    --output-format json \
    --output-file test_results.json \
    --include-charts
```

### Programmatic Execution

```rust
use ldc_engine::testing::*;

#[tokio::main]
async fn main() -> Result<()> {
    let config = TestConfig::from_file("config/test_config.toml")?;
    
    let mut runner = TestRunner::new(config);
    runner.add_test_suite(Box::new(CustomMathematicalTests::new()));
    runner.add_test_suite(Box::new(CustomPerformanceTests::new()));
    
    let results = runner.run_all_tests().await?;
    
    let report_generator = ReportGenerator::new();
    report_generator.generate_html_report(&results, "test_reports/latest.html")?;
    report_generator.generate_json_report(&results, "test_reports/latest.json")?;
    
    if !results.all_passed() {
        eprintln!("Some tests failed!");
        std::process::exit(1);
    }
    
    println!("All tests passed!");
    Ok(())
}
```

## Result Interpretation

### Mathematical Accuracy Results

```rust
let math_results = test_suite.run_mathematical_tests()?;

for result in math_results.test_results {
    match result.test_category {
        TestCategory::Standard => {
            if !result.passed {
                println!("Standard calculation failed: {}", result.test_name);
                println!("  Expected: {:.6}, Got: {:.6}, Diff: {:.2e}", 
                        result.expected, result.actual, result.difference);
            }
        },
        TestCategory::EdgeCases => {
            if !result.passed {
                println!("Edge case handling failed: {}", result.test_name);
            }
        },
        TestCategory::Precision => {
            if !result.passed {
                println!("Precision test failed: {}", result.test_name);
                println!("  Consider adjusting tolerance or algorithm");
            }
        }
    }
}
```

### Performance Results

```rust
let perf_results = validator.validate_query_performance(&engine)?;

for result in perf_results.results {
    if !result.passed {
        println!("Performance target missed for {}", result.dataset_name);
        println!("  Target: {:.2}ms, Actual: {:.2}ms (P95: {:.2}ms)", 
                result.target_latency_ms, 
                result.avg_latency_ms,
                result.p95_latency_ms);
        
        if result.avg_latency_ms > result.target_latency_ms * 2.0 {
            println!("  Recommendation: Consider enabling HNSW indexing");
        } else if result.p95_latency_ms > result.avg_latency_ms * 1.5 {
            println!("  Recommendation: Check for memory allocation issues");
        }
    }
}
```

### Statistical Analysis Results

```rust
let stats = analyzer.analyze_predictions(&predictions, &outcomes, &market_data)?;

println!("Prediction Accuracy:");
println!("  Hit Rate: {:.2}%", stats.prediction_accuracy.hit_rate * 100.0);
println!("  Precision: {:.2}%", stats.prediction_accuracy.precision * 100.0);
println!("  F1 Score: {:.3}", stats.prediction_accuracy.f1_score);

if stats.statistical_significance.p_value > 0.05 {
    println!("Results not statistically significant (p={:.3})", 
            stats.statistical_significance.p_value);
}

if stats.signal_quality.information_coefficient < 0.05 {
    println!("Low information coefficient ({:.3}): signals may not be predictive",
            stats.signal_quality.information_coefficient);
}
```

## Performance Tuning

### Identifying Performance Bottlenecks

```rust
use ldc_engine::testing::PerformanceProfiler;

let profiler = PerformanceProfiler::new();
let profile_results = profiler.profile_ldc_operations(&engine, &test_data)?;

for operation in profile_results.operations {
    if operation.cpu_usage_percent > 80.0 {
        println!("High CPU usage in {}: {:.1}%", 
                operation.name, operation.cpu_usage_percent);
    }
}
```

### LDC Engine Parameters

```rust
// Performance-oriented configuration
pub fn create_performance_config() -> LDCConfig {
    LDCConfig {
        neighbors_count: 8,
        max_bars_back: 2000,
        use_hnsw_index: true,
        hnsw_config: tune_hnsw_parameters(50000, 0.95),
        enable_simd: true,
        parallel_processing: true,
        batch_size: 1000,
        cache_size: 10000,
    }
}

// Accuracy-oriented configuration
pub fn create_accuracy_config() -> LDCConfig {
    LDCConfig {
        neighbors_count: 20,
        max_bars_back: 5000,
        use_hnsw_index: false,
        enable_simd: true,
        parallel_processing: false,
        batch_size: 100,
        cache_size: 50000,
    }
}
```

See [performance.md](performance.md) for comprehensive benchmarking and system-level tuning.

## Troubleshooting

### Mathematical Accuracy Failures

**Problem**: SIMD vs Standard calculation differences

**Diagnosis**:
```rust
if result.difference < 1e-6 {
    println!("Difference within acceptable floating-point precision");
} else {
    println!("Significant calculation difference detected");
    println!("  Check SIMD implementation for correctness");
}
```

**Solutions**:
1. Adjust tolerance for floating-point comparisons
2. Verify SIMD implementation matches standard algorithm
3. Check for compiler optimization differences

**Problem**: NaN or Infinity in distance calculations

**Solutions**:
1. Add input validation and sanitization
2. Implement robust handling of edge cases
3. Use numerically stable algorithms

### Performance Test Failures

**Problem**: Latency targets not met

**Solutions**:
1. Enable HNSW indexing for large datasets
2. Optimize distance calculation with SIMD
3. Reduce dataset size or increase targets for CI environments
4. Check for memory allocation issues

### Integration Test Failures

**Problem**: Pipeline component failures

**Diagnosis**:
```rust
let health_check = pipeline.check_component_health();
for component in health_check.components {
    if !component.healthy {
        println!("Unhealthy component: {} - {}", 
                component.name, component.error);
    }
}
```

**Solutions**:
1. Increase timeout values for slow systems
2. Check for deadlocks in multi-threaded code
3. Verify proper resource cleanup

### Debugging Techniques

```rust
// Enable detailed logging
let config = TestConfig {
    enable_detailed_logging: true,
    log_level: LogLevel::Debug,
    log_components: vec![
        "distance_calculation".to_string(),
        "hnsw_index".to_string(),
    ],
};

// Profile specific operations
let profiler = Profiler::new();
profiler.start("distance_calculation");
let distance = engine.calculate_distance(&features1, &features2);
let profile_data = profiler.stop("distance_calculation");
println!("Distance calculation took: {:.3}ms", profile_data.duration_ms);
```

## Integration Examples

See [automation.md](automation.md) for detailed CI/CD pipeline examples including GitHub Actions, GitLab CI, Jenkins, Docker, and pre-commit hooks.

## Best Practices

### Test Organization

```rust
// Hierarchical test structure
pub mod mathematical {
    pub mod distance_calculations;
    pub mod simd_compatibility;
    pub mod edge_cases;
}

pub mod performance {
    pub mod latency_tests;
    pub mod throughput_tests;
    pub mod memory_usage;
}

pub mod integration {
    pub mod pipeline_tests;
    pub mod error_handling;
    pub mod configuration;
}
```

### Test Naming Conventions

```rust
#[test]
fn test_lorentzian_distance_identical_features_returns_zero() { }

#[test]
fn test_simd_distance_calculation_matches_standard_within_tolerance() { }

#[test]
fn test_performance_query_latency_under_1ms_for_10k_samples() { }
```

### Test Data Management

```rust
// Use seeded RNG for reproducible tests
pub fn create_reproducible_test_data(seed: u64, size: usize) -> Vec<TrainingSample> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // ...
}

pub fn validate_test_data(samples: &[TrainingSample]) -> Result<(), TestDataError> {
    for (i, sample) in samples.iter().enumerate() {
        let features = sample.features.to_array();
        for (j, &feature) in features.iter().enumerate() {
            if feature.is_nan() {
                return Err(TestDataError::NaNValue { sample_index: i, feature_index: j });
            }
        }
    }
    Ok(())
}
```

### CI/CD Best Practices

1. Use quick tests for pull request validation
2. Run full test suites on main branch pushes
3. Schedule nightly performance tests
4. Monitor performance trends over time
5. Set up automated alerting for test failures
