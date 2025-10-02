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

For comprehensive information on specific aspects of the testing framework, refer to these detailed guides:

- **[Test Execution Guide](TEST_EXECUTION_GUIDE.md)** - Detailed instructions for running tests in different scenarios
- **[Test Configuration Examples](TEST_CONFIGURATION_EXAMPLES.md)** - Comprehensive configuration examples for various environments
- **[Test Result Interpretation](TEST_RESULT_INTERPRETATION.md)** - Guide for understanding and acting on test results
- **[Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)** - Solutions for common issues and problems
- **[Performance Tuning Guide](PERFORMANCE_TUNING_GUIDE.md)** - Optimization techniques and performance tuning strategies
- **[Integration Examples](INTEGRATION_EXAMPLES.md)** - Examples for integrating testing into development workflows and CI/CD pipelines

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
    position_size: 0.1, // 10% of capital per trade
    transaction_cost: 0.001, // 0.1% transaction cost
    signal_threshold: 0.6,
    max_positions: 5,
    rebalance_frequency: Duration::from_secs(3600), // 1 hour
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
// config/dev_test_config.rs
use ldc_engine::testing::*;

pub fn create_dev_config() -> TestConfig {
    TestConfig {
        mathematical_tolerance: 1e-6,
        performance_targets: PerformanceTargets {
            target_latency_1k_samples_ms: 1.0,  // Relaxed for dev
            target_latency_10k_samples_ms: 2.0,
            target_latency_50k_samples_ms: 10.0,
            target_hnsw_accuracy_percent: 90.0, // Relaxed for dev
        },
        statistical_confidence: 0.90, // Relaxed for dev
        enable_detailed_logging: true,
        enable_performance_profiling: true,
        test_data_size: TestDataSize::Small,
        parallel_execution: false, // Easier debugging
    }
}
```

### Production Environment Configuration

```rust
// config/prod_test_config.rs
use ldc_engine::testing::*;

pub fn create_prod_config() -> TestConfig {
    TestConfig {
        mathematical_tolerance: 1e-8, // Stricter for production
        performance_targets: PerformanceTargets {
            target_latency_1k_samples_ms: 0.5,
            target_latency_10k_samples_ms: 1.0,
            target_latency_50k_samples_ms: 5.0,
            target_hnsw_accuracy_percent: 95.0,
        },
        statistical_confidence: 0.95,
        enable_detailed_logging: false, // Performance optimization
        enable_performance_profiling: false,
        test_data_size: TestDataSize::Large,
        parallel_execution: true,
    }
}
```

### CI/CD Configuration

```rust
// config/ci_test_config.rs
use ldc_engine::testing::*;

pub fn create_ci_config() -> TestConfig {
    TestConfig {
        mathematical_tolerance: 1e-6,
        performance_targets: PerformanceTargets {
            target_latency_1k_samples_ms: 2.0,  // Account for CI overhead
            target_latency_10k_samples_ms: 4.0,
            target_latency_50k_samples_ms: 15.0,
            target_hnsw_accuracy_percent: 95.0,
        },
        statistical_confidence: 0.95,
        enable_detailed_logging: true,
        enable_performance_profiling: false,
        test_data_size: TestDataSize::Medium,
        parallel_execution: true,
        timeout_seconds: 300, // 5 minute timeout
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
    // Create test configuration
    let config = TestConfig::from_file("config/test_config.toml")?;
    
    // Initialize test runner
    let mut runner = TestRunner::new(config);
    
    // Add custom test suites
    runner.add_test_suite(Box::new(CustomMathematicalTests::new()));
    runner.add_test_suite(Box::new(CustomPerformanceTests::new()));
    
    // Execute tests
    let results = runner.run_all_tests().await?;
    
    // Generate reports
    let report_generator = ReportGenerator::new();
    report_generator.generate_html_report(&results, "test_reports/latest.html")?;
    report_generator.generate_json_report(&results, "test_reports/latest.json")?;
    
    // Check for failures
    if !results.all_passed() {
        eprintln!("Some tests failed!");
        std::process::exit(1);
    }
    
    println!("All tests passed!");
    Ok(())
}
```

### Integration with Build Systems

#### Cargo Integration

```toml
# Cargo.toml
[dev-dependencies]
ldc-engine = { path = ".", features = ["testing"] }

[[example]]
name = "run_tests"
path = "examples/run_tests.rs"

[package.metadata.test]
timeout = 300
```

#### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Run Mathematical Tests
      run: cargo test --test mathematical_accuracy_tests
      
    - name: Run Performance Tests
      run: cargo test --test performance_validation_tests
      
    - name: Run Integration Tests
      run: cargo test --test comprehensive_integration_tests
      
    - name: Run Complete Test Suite
      run: cargo run --example automated_test_runner_demo -- --config ci
      
    - name: Upload Test Reports
      uses: actions/upload-artifact@v2
      if: always()
      with:
        name: test-reports
        path: test_reports/
```

## Result Interpretation

### Understanding Test Results

#### Mathematical Accuracy Results

```rust
// Example result interpretation
let math_results = test_suite.run_mathematical_tests()?;

for result in math_results.test_results {
    match result.test_category {
        TestCategory::Standard => {
            if !result.passed {
                println!("⚠️  Standard calculation failed: {}", result.test_name);
                println!("   Expected: {:.6}, Got: {:.6}, Diff: {:.2e}", 
                        result.expected, result.actual, result.difference);
            }
        },
        TestCategory::EdgeCases => {
            if !result.passed {
                println!("🚨 Edge case handling failed: {}", result.test_name);
                println!("   This may indicate numerical instability issues");
            }
        },
        TestCategory::Precision => {
            if !result.passed {
                println!("🔍 Precision test failed: {}", result.test_name);
                println!("   Consider adjusting tolerance or algorithm");
            }
        }
    }
}
```

#### Performance Results

```rust
// Performance result analysis
let perf_results = validator.validate_query_performance(&engine)?;

for result in perf_results.results {
    if !result.passed {
        println!("⏱️  Performance target missed for {}", result.dataset_name);
        println!("   Target: {:.2}ms, Actual: {:.2}ms (P95: {:.2}ms)", 
                result.target_latency_ms, 
                result.avg_latency_ms,
                result.p95_latency_ms);
        
        // Provide actionable recommendations
        if result.avg_latency_ms > result.target_latency_ms * 2.0 {
            println!("   🔧 Recommendation: Consider enabling HNSW indexing");
        } else if result.p95_latency_ms > result.avg_latency_ms * 1.5 {
            println!("   🔧 Recommendation: Check for memory allocation issues");
        }
    }
}
```

#### Statistical Analysis Results

```rust
// Statistical result interpretation
let stats = analyzer.analyze_predictions(&predictions, &outcomes, &market_data)?;

println!("📊 Prediction Accuracy:");
println!("   Hit Rate: {:.2}%", stats.prediction_accuracy.hit_rate * 100.0);
println!("   Precision: {:.2}%", stats.prediction_accuracy.precision * 100.0);
println!("   Recall: {:.2}%", stats.prediction_accuracy.recall * 100.0);
println!("   F1 Score: {:.3}", stats.prediction_accuracy.f1_score);

if stats.statistical_significance.p_value > 0.05 {
    println!("⚠️  Results not statistically significant (p={:.3})", 
            stats.statistical_significance.p_value);
    println!("   Consider collecting more data or adjusting parameters");
}

if stats.signal_quality.information_coefficient < 0.05 {
    println!("📉 Low information coefficient ({:.3})", 
            stats.signal_quality.information_coefficient);
    println!("   Signals may not be predictive of future returns");
}
```

### Actionable Recommendations

#### Performance Optimization

```rust
pub fn analyze_performance_results(results: &PerformanceTestResult) -> Vec<Recommendation> {
    let mut recommendations = Vec::new();
    
    for result in &results.results {
        if !result.passed {
            if result.avg_latency_ms > result.target_latency_ms * 3.0 {
                recommendations.push(Recommendation {
                    priority: Priority::High,
                    category: Category::Performance,
                    description: format!(
                        "Severe performance degradation in {}: {:.2}ms vs {:.2}ms target",
                        result.dataset_name, result.avg_latency_ms, result.target_latency_ms
                    ),
                    actions: vec![
                        "Enable HNSW indexing for large datasets".to_string(),
                        "Consider SIMD optimizations".to_string(),
                        "Profile memory allocation patterns".to_string(),
                    ],
                });
            }
            
            if result.p99_latency_ms > result.avg_latency_ms * 3.0 {
                recommendations.push(Recommendation {
                    priority: Priority::Medium,
                    category: Category::Reliability,
                    description: "High latency variance detected".to_string(),
                    actions: vec![
                        "Investigate garbage collection pauses".to_string(),
                        "Check for memory fragmentation".to_string(),
                        "Consider pre-allocation strategies".to_string(),
                    ],
                });
            }
        }
    }
    
    recommendations
}
```

#### Statistical Significance

```rust
pub fn interpret_statistical_results(results: &StatisticalAnalysisResult) -> Interpretation {
    let mut interpretation = Interpretation::new();
    
    // Check sample size adequacy
    if results.statistical_significance.sample_size < 1000 {
        interpretation.add_warning(
            "Sample size may be insufficient for reliable conclusions"
        );
        interpretation.add_recommendation(
            "Collect at least 1000 samples for statistical significance"
        );
    }
    
    // Evaluate prediction quality
    if results.prediction_accuracy.hit_rate < 0.55 {
        interpretation.add_concern(
            "Hit rate below 55% suggests limited predictive power"
        );
        interpretation.add_recommendation(
            "Consider feature engineering or model parameter tuning"
        );
    }
    
    // Check information coefficient
    if results.signal_quality.information_coefficient < 0.02 {
        interpretation.add_warning(
            "Low information coefficient indicates weak signal quality"
        );
        interpretation.add_recommendation(
            "Review feature selection and signal generation logic"
        );
    }
    
    interpretation
}
```
## P
erformance Tuning

### Identifying Performance Bottlenecks

#### CPU-Bound Operations

```rust
use ldc_engine::testing::PerformanceProfiler;

let profiler = PerformanceProfiler::new();
let profile_results = profiler.profile_ldc_operations(&engine, &test_data)?;

// Analyze CPU usage patterns
for operation in profile_results.operations {
    if operation.cpu_usage_percent > 80.0 {
        println!("🔥 High CPU usage in {}: {:.1}%", 
                operation.name, operation.cpu_usage_percent);
        
        match operation.name.as_str() {
            "distance_calculation" => {
                println!("   💡 Consider enabling SIMD optimizations");
                println!("   💡 Use HNSW indexing for large datasets");
            },
            "feature_computation" => {
                println!("   💡 Cache frequently computed features");
                println!("   💡 Use parallel feature computation");
            },
            "k_nearest_neighbors" => {
                println!("   💡 Tune k value for optimal performance");
                println!("   💡 Consider approximate nearest neighbor algorithms");
            }
        }
    }
}
```

#### Memory Usage Optimization

```rust
use ldc_engine::testing::MemoryProfiler;

let memory_profiler = MemoryProfiler::new();
let memory_results = memory_profiler.analyze_memory_usage(&engine)?;

if memory_results.peak_usage_mb > 1000.0 {
    println!("🧠 High memory usage detected: {:.1} MB", memory_results.peak_usage_mb);
    
    // Analyze allocation patterns
    for allocation in memory_results.allocations {
        if allocation.size_mb > 100.0 {
            println!("   📦 Large allocation: {} ({:.1} MB)", 
                    allocation.location, allocation.size_mb);
        }
    }
    
    // Provide optimization suggestions
    println!("   💡 Optimization suggestions:");
    println!("      - Use object pooling for frequently allocated objects");
    println!("      - Implement streaming processing for large datasets");
    println!("      - Consider memory-mapped files for historical data");
}
```

### Configuration Tuning Guidelines

#### HNSW Index Parameters

```rust
// Tuning HNSW parameters based on dataset characteristics
pub fn tune_hnsw_parameters(dataset_size: usize, accuracy_target: f64) -> HNSWConfig {
    let (m, ef_construction, ef_search) = match dataset_size {
        size if size < 10_000 => {
            // Small datasets: prioritize accuracy over speed
            (16, 200, 100)
        },
        size if size < 100_000 => {
            // Medium datasets: balanced approach
            (16, 200, 50)
        },
        _ => {
            // Large datasets: prioritize speed
            (8, 100, 32)
        }
    };
    
    // Adjust based on accuracy requirements
    let ef_search_adjusted = if accuracy_target > 0.98 {
        ef_search * 2
    } else if accuracy_target < 0.90 {
        ef_search / 2
    } else {
        ef_search
    };
    
    HNSWConfig {
        m,
        ef_construction,
        ef_search: ef_search_adjusted,
        max_m: m,
        max_m0: m * 2,
    }
}
```

#### LDC Engine Parameters

```rust
// Performance-oriented configuration
pub fn create_performance_config() -> LDCConfig {
    LDCConfig {
        neighbors_count: 8,        // Reduced for speed
        max_bars_back: 2000,       // Reasonable history
        use_hnsw_index: true,      // Enable for large datasets
        hnsw_config: tune_hnsw_parameters(50000, 0.95),
        enable_simd: true,         // Enable SIMD optimizations
        parallel_processing: true,  // Use multiple cores
        batch_size: 1000,          // Optimize batch processing
        cache_size: 10000,         // Cache recent calculations
    }
}

// Accuracy-oriented configuration
pub fn create_accuracy_config() -> LDCConfig {
    LDCConfig {
        neighbors_count: 20,       // More neighbors for accuracy
        max_bars_back: 5000,       // Longer history
        use_hnsw_index: false,     // Exact search for maximum accuracy
        enable_simd: true,         // Keep SIMD for speed
        parallel_processing: false, // Deterministic results
        batch_size: 100,           // Smaller batches for precision
        cache_size: 50000,         // Larger cache for accuracy
    }
}
```

### System-Specific Optimizations

#### Multi-Core Systems

```rust
use std::thread;
use rayon::prelude::*;

// Optimize for multi-core systems
pub fn optimize_for_multicore(config: &mut LDCConfig) {
    let num_cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    
    if num_cores >= 8 {
        // High-core count systems
        config.parallel_processing = true;
        config.batch_size = 2000;
        config.worker_threads = Some(num_cores - 1); // Leave one core for OS
    } else if num_cores >= 4 {
        // Medium-core count systems
        config.parallel_processing = true;
        config.batch_size = 1000;
        config.worker_threads = Some(num_cores / 2);
    } else {
        // Low-core count systems
        config.parallel_processing = false;
        config.batch_size = 500;
    }
}
```

#### Memory-Constrained Systems

```rust
// Optimize for systems with limited memory
pub fn optimize_for_low_memory(config: &mut LDCConfig) {
    config.max_bars_back = 1000;      // Reduce history
    config.cache_size = 1000;         // Smaller cache
    config.batch_size = 100;          // Smaller batches
    config.use_memory_mapping = true; // Use memory-mapped files
    config.enable_compression = true; // Compress stored data
}
```

## Troubleshooting

### Common Test Failures

#### Mathematical Accuracy Failures

**Problem**: SIMD vs Standard calculation differences

```
FAILED: SIMD_vs_Standard_normal_features - Expected: 2.345678, Got: 2.345679
```

**Diagnosis**:
```rust
// Check floating-point precision issues
if result.difference < 1e-6 {
    println!("✅ Difference within acceptable floating-point precision");
} else {
    println!("❌ Significant calculation difference detected");
    println!("   Check SIMD implementation for correctness");
}
```

**Solutions**:
1. Adjust tolerance for floating-point comparisons
2. Verify SIMD implementation matches standard algorithm
3. Check for compiler optimization differences

**Problem**: NaN or Infinity in distance calculations

```
FAILED: EdgeCase_extreme_values - Got: NaN, Expected: finite value
```

**Diagnosis**:
```rust
// Check for problematic input values
for feature in &test_case.features1.to_array() {
    if feature.is_nan() || feature.is_infinite() {
        println!("❌ Invalid input feature: {}", feature);
    }
}
```

**Solutions**:
1. Add input validation and sanitization
2. Implement robust handling of edge cases
3. Use numerically stable algorithms

#### Performance Test Failures

**Problem**: Latency targets not met

```
⏱️ Performance target missed for medium_10k
Target: 1.00ms, Actual: 2.34ms (P95: 4.56ms)
```

**Diagnosis**:
```rust
// Analyze performance bottlenecks
let bottlenecks = profiler.identify_bottlenecks(&performance_data);
for bottleneck in bottlenecks {
    println!("🔍 Bottleneck: {} ({:.1}% of total time)", 
            bottleneck.operation, bottleneck.percentage);
}
```

**Solutions**:
1. Enable HNSW indexing for large datasets
2. Optimize distance calculation with SIMD
3. Reduce dataset size or increase targets for CI environments
4. Check for memory allocation issues

**Problem**: High memory usage

```
🧠 Memory usage exceeded limit: 2.1GB used, 1.0GB limit
```

**Solutions**:
1. Implement streaming processing
2. Use memory-mapped files for large datasets
3. Reduce cache sizes and batch sizes
4. Enable data compression

#### Integration Test Failures

**Problem**: Pipeline component failures

```
❌ Integration test failed: feature_pipeline_integration
Error: Feature computation timeout after 30s
```

**Diagnosis**:
```rust
// Check component health
let health_check = pipeline.check_component_health();
for component in health_check.components {
    if !component.healthy {
        println!("🚨 Unhealthy component: {} - {}", 
                component.name, component.error);
    }
}
```

**Solutions**:
1. Increase timeout values for slow systems
2. Check for deadlocks in multi-threaded code
3. Verify proper resource cleanup
4. Monitor system resource usage

### Debugging Techniques

#### Verbose Logging

```rust
// Enable detailed logging for debugging
let config = TestConfig {
    enable_detailed_logging: true,
    log_level: LogLevel::Debug,
    log_components: vec![
        "distance_calculation".to_string(),
        "hnsw_index".to_string(),
        "performance_monitoring".to_string(),
    ],
};

// Run tests with verbose output
let results = run_tests_with_config(config)?;
```

#### Performance Profiling

```rust
use ldc_engine::testing::Profiler;

// Profile specific operations
let profiler = Profiler::new();
profiler.start("distance_calculation");

let distance = engine.calculate_distance(&features1, &features2);

let profile_data = profiler.stop("distance_calculation");
println!("Distance calculation took: {:.3}ms", profile_data.duration_ms);
```

#### Memory Leak Detection

```rust
use ldc_engine::testing::MemoryTracker;

// Track memory usage over time
let tracker = MemoryTracker::new();
tracker.start_tracking();

// Run test operations
for i in 0..1000 {
    let result = engine.process_sample(&samples[i]);
    
    if i % 100 == 0 {
        let memory_usage = tracker.current_usage_mb();
        println!("Memory usage at iteration {}: {:.1} MB", i, memory_usage);
    }
}

let leak_report = tracker.generate_leak_report();
if !leak_report.leaks.is_empty() {
    println!("🚨 Memory leaks detected:");
    for leak in leak_report.leaks {
        println!("   {} bytes at {}", leak.size, leak.location);
    }
}
```

### Error Recovery Strategies

#### Graceful Degradation

```rust
// Implement fallback strategies for test failures
pub fn run_tests_with_fallback(config: TestConfig) -> TestResult {
    let mut results = TestResult::new();
    
    // Try performance tests with strict targets
    match run_performance_tests(&config) {
        Ok(perf_results) => results.add_performance_results(perf_results),
        Err(e) => {
            println!("⚠️  Performance tests failed, trying with relaxed targets");
            let relaxed_config = config.with_relaxed_performance_targets();
            match run_performance_tests(&relaxed_config) {
                Ok(relaxed_results) => {
                    results.add_performance_results(relaxed_results);
                    results.add_warning("Performance targets were relaxed");
                },
                Err(e2) => results.add_error(format!("Performance tests failed: {}", e2)),
            }
        }
    }
    
    results
}
```

#### Test Isolation

```rust
// Isolate failing tests to prevent cascade failures
pub fn run_isolated_tests(test_suite: &TestSuite) -> Vec<TestResult> {
    let mut results = Vec::new();
    
    for test in &test_suite.tests {
        // Run each test in isolation
        let result = std::panic::catch_unwind(|| {
            test.run()
        });
        
        match result {
            Ok(test_result) => results.push(test_result),
            Err(_) => {
                results.push(TestResult::failed(
                    test.name.clone(),
                    "Test panicked".to_string()
                ));
            }
        }
    }
    
    results
}
```

## Integration Examples

### CI/CD Pipeline Integration

#### GitHub Actions Workflow

```yaml
# .github/workflows/comprehensive-testing.yml
name: Comprehensive Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  mathematical-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        components: rustfmt, clippy
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Run Mathematical Accuracy Tests
      run: |
        cargo test --test mathematical_accuracy_tests --verbose
        
    - name: Upload Mathematical Test Results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: mathematical-test-results
        path: test_reports/mathematical_*.json

  performance-tests:
    runs-on: ubuntu-latest
    needs: mathematical-tests
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Run Performance Tests
      run: |
        # Use CI-specific configuration with relaxed targets
        cargo run --example automated_test_runner_demo -- \
          --config ci \
          --tests performance \
          --output-format json \
          --output-file test_reports/performance_results.json
          
    - name: Analyze Performance Results
      run: |
        python scripts/analyze_performance.py test_reports/performance_results.json
        
    - name: Upload Performance Test Results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-test-results
        path: test_reports/performance_*.json

  integration-tests:
    runs-on: ubuntu-latest
    needs: [mathematical-tests, performance-tests]
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Run Integration Tests
      run: |
        cargo test --test comprehensive_integration_tests --verbose
        
    - name: Run End-to-End Pipeline Test
      run: |
        cargo run --example end_to_end_pipeline -- \
          --test-mode \
          --output test_reports/e2e_results.json

  generate-report:
    runs-on: ubuntu-latest
    needs: [mathematical-tests, performance-tests, integration-tests]
    if: always()
    steps:
    - uses: actions/checkout@v3
    
    - name: Download All Test Results
      uses: actions/download-artifact@v3
      
    - name: Generate Comprehensive Report
      run: |
        python scripts/generate_test_report.py \
          --mathematical mathematical-test-results/ \
          --performance performance-test-results/ \
          --integration integration-test-results/ \
          --output test_reports/comprehensive_report.html
          
    - name: Upload Comprehensive Report
      uses: actions/upload-artifact@v3
      with:
        name: comprehensive-test-report
        path: test_reports/comprehensive_report.html
        
    - name: Comment PR with Results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('test_reports/summary.md', 'utf8');
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: report
          });
```

#### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        RUST_BACKTRACE = '1'
        CARGO_TERM_COLOR = 'always'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'rustup update stable'
                sh 'cargo --version'
            }
        }
        
        stage('Mathematical Tests') {
            steps {
                sh 'cargo test --test mathematical_accuracy_tests --verbose'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'test_reports/mathematical_*.json', allowEmptyArchive: true
                }
            }
        }
        
        stage('Performance Tests') {
            steps {
                sh '''
                    cargo run --example automated_test_runner_demo -- \
                        --config ci \
                        --tests performance \
                        --output-format json \
                        --output-file test_reports/performance_results.json
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'test_reports/performance_*.json', allowEmptyArchive: true
                }
            }
        }
        
        stage('Integration Tests') {
            parallel {
                stage('Component Integration') {
                    steps {
                        sh 'cargo test --test comprehensive_integration_tests'
                    }
                }
                stage('End-to-End Pipeline') {
                    steps {
                        sh '''
                            cargo run --example end_to_end_pipeline -- \
                                --test-mode \
                                --output test_reports/e2e_results.json
                        '''
                    }
                }
            }
        }
        
        stage('Generate Reports') {
            steps {
                sh '''
                    python scripts/generate_test_report.py \
                        --all-results test_reports/ \
                        --output test_reports/jenkins_report.html
                '''
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'test_reports',
                        reportFiles: 'jenkins_report.html',
                        reportName: 'Test Report'
                    ])
                }
            }
        }
    }
    
    post {
        failure {
            emailext (
                subject: "Test Failure: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Test suite failed. Check the build logs for details.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

### Development Workflow Integration

#### Pre-commit Hooks

```bash
#!/bin/sh
# .git/hooks/pre-commit

echo "Running pre-commit tests..."

# Run quick mathematical accuracy tests
cargo test --test mathematical_accuracy_tests --quiet
if [ $? -ne 0 ]; then
    echo "❌ Mathematical accuracy tests failed"
    exit 1
fi

# Run basic performance smoke tests
cargo run --example automated_test_runner_demo -- \
    --tests performance \
    --quick-mode \
    --timeout 30
if [ $? -ne 0 ]; then
    echo "❌ Performance smoke tests failed"
    exit 1
fi

echo "✅ Pre-commit tests passed"
```

#### IDE Integration (VS Code)

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
            "args": ["run", "--example", "automated_test_runner_demo", "--", "--tests", "performance"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run All Tests",
            "type": "shell",
            "command": "cargo",
            "args": ["run", "--example", "automated_test_runner_demo"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
```

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Test Runner",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/target/debug/examples/automated_test_runner_demo",
            "args": ["--config", "dev", "--verbose"],
            "cwd": "${workspaceFolder}",
            "sourceLanguages": ["rust"]
        }
    ]
}
```

## Best Practices

### Test Organization

#### Hierarchical Test Structure

```rust
// tests/mod.rs
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

pub mod statistical {
    pub mod accuracy_metrics;
    pub mod significance_tests;
    pub mod regime_analysis;
}
```

#### Test Naming Conventions

```rust
// Use descriptive, hierarchical test names
#[test]
fn test_lorentzian_distance_identical_features_returns_zero() {
    // Test implementation
}

#[test]
fn test_simd_distance_calculation_matches_standard_within_tolerance() {
    // Test implementation
}

#[test]
fn test_hnsw_accuracy_exceeds_95_percent_on_large_dataset() {
    // Test implementation
}

#[test]
fn test_performance_query_latency_under_1ms_for_10k_samples() {
    // Test implementation
}
```

### Test Data Management

#### Reproducible Test Data

```rust
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// Use seeded random number generators for reproducible tests
pub fn create_reproducible_test_data(seed: u64, size: usize) -> Vec<TrainingSample> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(size);
    
    for i in 0..size {
        let features = FeatureSeries {
            f1: rng.gen_range(-100.0..100.0),
            f2: rng.gen_range(-100.0..100.0),
            f3: rng.gen_range(-100.0..100.0),
            f4: rng.gen_range(-100.0..100.0),
            f5: rng.gen_range(-100.0..100.0),
        };
        
        samples.push(TrainingSample {
            features,
            label: Direction::Long, // Will be properly labeled based on future data
            timestamp: i as i64,
            bar_index: i,
        });
    }
    
    samples
}
```

#### Test Data Validation

```rust
pub fn validate_test_data(samples: &[TrainingSample]) -> Result<(), TestDataError> {
    // Check for NaN or infinite values
    for (i, sample) in samples.iter().enumerate() {
        let features = sample.features.to_array();
        for (j, &feature) in features.iter().enumerate() {
            if feature.is_nan() {
                return Err(TestDataError::NaNValue { sample_index: i, feature_index: j });
            }
            if feature.is_infinite() {
                return Err(TestDataError::InfiniteValue { sample_index: i, feature_index: j });
            }
        }
    }
    
    // Check for reasonable value ranges
    let feature_stats = calculate_feature_statistics(samples);
    for (i, stats) in feature_stats.iter().enumerate() {
        if stats.std_dev == 0.0 {
            return Err(TestDataError::ZeroVariance { feature_index: i });
        }
        if stats.range > 1e6 {
            return Err(TestDataError::ExtremeRange { 
                feature_index: i, 
                range: stats.range 
            });
        }
    }
    
    Ok(())
}
```

### Continuous Monitoring

#### Performance Regression Detection

```rust
use std::collections::HashMap;

pub struct PerformanceBaseline {
    baselines: HashMap<String, f64>,
    tolerance: f64,
}

impl PerformanceBaseline {
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let baselines: HashMap<String, f64> = serde_json::from_str(&content)?;
        
        Ok(Self {
            baselines,
            tolerance: 0.1, // 10% tolerance
        })
    }
    
    pub fn check_regression(&self, current_results: &PerformanceTestResult) -> Vec<Regression> {
        let mut regressions = Vec::new();
        
        for result in &current_results.results {
            let test_name = format!("{}_{}", result.dataset_name, "avg_latency");
            
            if let Some(&baseline) = self.baselines.get(&test_name) {
                let regression_ratio = (result.avg_latency_ms - baseline) / baseline;
                
                if regression_ratio > self.tolerance {
                    regressions.push(Regression {
                        test_name: test_name.clone(),
                        baseline_value: baseline,
                        current_value: result.avg_latency_ms,
                        regression_percent: regression_ratio * 100.0,
                    });
                }
            }
        }
        
        regressions
    }
    
    pub fn update_baseline(&mut self, results: &PerformanceTestResult) {
        for result in &results.results {
            let test_name = format!("{}_{}", result.dataset_name, "avg_latency");
            self.baselines.insert(test_name, result.avg_latency_ms);
        }
    }
}

#[derive(Debug)]
pub struct Regression {
    pub test_name: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub regression_percent: f64,
}
```

#### Automated Alerting

```rust
use reqwest;
use serde_json::json;

pub async fn send_test_failure_alert(
    webhook_url: &str,
    test_results: &ComprehensiveTestResult,
) -> Result<()> {
    if test_results.all_passed() {
        return Ok(()); // No alert needed
    }
    
    let failed_tests: Vec<_> = test_results.get_failed_tests();
    let message = format!(
        "🚨 Test Failure Alert\n\n{} tests failed:\n{}",
        failed_tests.len(),
        failed_tests.iter()
            .map(|t| format!("• {}", t.name))
            .collect::<Vec<_>>()
            .join("\n")
    );
    
    let payload = json!({
        "text": message,
        "attachments": [{
            "color": "danger",
            "fields": [
                {
                    "title": "Failed Tests",
                    "value": failed_tests.len().to_string(),
                    "short": true
                },
                {
                    "title": "Success Rate",
                    "value": format!("{:.1}%", test_results.success_rate() * 100.0),
                    "short": true
                }
            ]
        }]
    });
    
    let client = reqwest::Client::new();
    client.post(webhook_url)
        .json(&payload)
        .send()
        .await?;
    
    Ok(())
}
```

This comprehensive testing guide provides developers with all the necessary information to effectively test the LDC engine, troubleshoot issues, and integrate testing into their development workflows.