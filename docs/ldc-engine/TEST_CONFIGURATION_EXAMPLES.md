# Test Configuration Examples

## Overview

This document provides comprehensive examples of test configurations for different scenarios, environments, and use cases in the LDC engine testing framework.

## Configuration Structure

### Base Configuration Format

```toml
# Base test configuration structure
[mathematical]
tolerance = 1e-6
test_edge_cases = true
test_extreme_values = true
enable_simd_tests = true
enable_hnsw_tests = true

[performance]
target_latency_1k_samples_ms = 0.5
target_latency_10k_samples_ms = 1.0
target_latency_50k_samples_ms = 5.0
target_hnsw_accuracy_percent = 95.0
target_cpu_utilization_percent = 90.0
test_iterations = 100
warmup_iterations = 10
enable_memory_profiling = false
enable_cpu_profiling = false

[statistical]
confidence_level = 0.95
min_sample_size = 1000
significance_threshold = 0.05
enable_regime_analysis = true
enable_correlation_analysis = true

[backtesting]
initial_capital = 100000.0
position_size = 0.1
transaction_cost = 0.001
slippage = 0.0005
signal_threshold = 0.6
max_positions = 5
rebalance_frequency_seconds = 3600

[execution]
enable_detailed_logging = true
enable_performance_profiling = false
parallel_execution = true
timeout_seconds = 300
max_memory_mb = 2048
output_format = "json"
report_directory = "test_reports"
```

## Environment-Specific Configurations

### 1. Development Environment

**Purpose**: Fast feedback during development with relaxed targets and detailed debugging information.

```toml
# config/dev_test_config.toml
[mathematical]
tolerance = 1e-6
test_edge_cases = true
test_extreme_values = false  # Skip time-consuming extreme tests
enable_simd_tests = true
enable_hnsw_tests = true

[performance]
target_latency_1k_samples_ms = 2.0   # Relaxed for development
target_latency_10k_samples_ms = 5.0
target_latency_50k_samples_ms = 20.0
target_hnsw_accuracy_percent = 90.0  # Relaxed for faster tests
target_cpu_utilization_percent = 80.0
test_iterations = 20                 # Fewer iterations for speed
warmup_iterations = 3
enable_memory_profiling = true       # Enable for debugging
enable_cpu_profiling = true

[statistical]
confidence_level = 0.90              # Relaxed for development
min_sample_size = 500                # Smaller sample for speed
significance_threshold = 0.05
enable_regime_analysis = false       # Skip complex analysis
enable_correlation_analysis = true

[backtesting]
initial_capital = 10000.0            # Smaller capital for faster tests
position_size = 0.05
transaction_cost = 0.001
slippage = 0.0005
signal_threshold = 0.5               # Lower threshold for more signals
max_positions = 3
rebalance_frequency_seconds = 1800   # More frequent for testing

[execution]
enable_detailed_logging = true       # Full logging for debugging
enable_performance_profiling = true
parallel_execution = false           # Single-threaded for debugging
timeout_seconds = 600                # Longer timeout for debugging
max_memory_mb = 1024
output_format = "json"
report_directory = "test_reports/dev"

[test_data]
dataset_size = "small"               # Use smaller datasets
synthetic_data_points = 1000
historical_data_days = 30
enable_data_caching = true
```

### 2. Production Environment

**Purpose**: Strict validation with production-level performance requirements.

```toml
# config/prod_test_config.toml
[mathematical]
tolerance = 1e-8                     # Stricter tolerance
test_edge_cases = true
test_extreme_values = true
enable_simd_tests = true
enable_hnsw_tests = true

[performance]
target_latency_1k_samples_ms = 0.5   # Production targets
target_latency_10k_samples_ms = 1.0
target_latency_50k_samples_ms = 5.0
target_hnsw_accuracy_percent = 95.0
target_cpu_utilization_percent = 90.0
test_iterations = 200                # More iterations for accuracy
warmup_iterations = 20
enable_memory_profiling = false      # Disabled for performance
enable_cpu_profiling = false

[statistical]
confidence_level = 0.95              # High confidence required
min_sample_size = 2000               # Large sample for reliability
significance_threshold = 0.01        # Stricter significance
enable_regime_analysis = true
enable_correlation_analysis = true

[backtesting]
initial_capital = 1000000.0          # Realistic capital
position_size = 0.1
transaction_cost = 0.0005            # Realistic transaction costs
slippage = 0.0002
signal_threshold = 0.7               # Higher threshold for quality
max_positions = 10
rebalance_frequency_seconds = 3600

[execution]
enable_detailed_logging = false      # Minimal logging for performance
enable_performance_profiling = false
parallel_execution = true
timeout_seconds = 300
max_memory_mb = 4096
output_format = "json"
report_directory = "test_reports/prod"

[test_data]
dataset_size = "large"
synthetic_data_points = 50000
historical_data_days = 365
enable_data_caching = true
```

### 3. CI/CD Environment

**Purpose**: Automated testing with appropriate timeouts and resource constraints.

```toml
# config/ci_test_config.toml
[mathematical]
tolerance = 1e-6
test_edge_cases = true
test_extreme_values = false          # Skip to reduce CI time
enable_simd_tests = true
enable_hnsw_tests = true

[performance]
target_latency_1k_samples_ms = 3.0   # Account for CI overhead
target_latency_10k_samples_ms = 8.0
target_latency_50k_samples_ms = 25.0
target_hnsw_accuracy_percent = 95.0
target_cpu_utilization_percent = 85.0
test_iterations = 50                 # Balanced for CI time
warmup_iterations = 5
enable_memory_profiling = false
enable_cpu_profiling = false

[statistical]
confidence_level = 0.95
min_sample_size = 1000
significance_threshold = 0.05
enable_regime_analysis = false       # Skip complex analysis in CI
enable_correlation_analysis = true

[backtesting]
initial_capital = 100000.0
position_size = 0.1
transaction_cost = 0.001
slippage = 0.0005
signal_threshold = 0.6
max_positions = 5
rebalance_frequency_seconds = 3600

[execution]
enable_detailed_logging = true       # Detailed logs for CI debugging
enable_performance_profiling = false
parallel_execution = true
timeout_seconds = 600                # Longer timeout for CI
max_memory_mb = 2048
output_format = "junit"              # CI-friendly format
report_directory = "test_reports/ci"

[test_data]
dataset_size = "medium"
synthetic_data_points = 10000
historical_data_days = 90
enable_data_caching = true

[ci_specific]
fail_fast = false                    # Continue testing after failures
generate_artifacts = true
upload_reports = true
notify_on_failure = true
```

## Use Case-Specific Configurations

### 1. Performance Benchmarking

**Purpose**: Focus on performance measurement and optimization.

```toml
# config/benchmark_config.toml
[mathematical]
tolerance = 1e-6
test_edge_cases = false              # Skip for pure performance focus
test_extreme_values = false
enable_simd_tests = true
enable_hnsw_tests = true

[performance]
target_latency_1k_samples_ms = 0.3   # Aggressive targets
target_latency_10k_samples_ms = 0.8
target_latency_50k_samples_ms = 3.0
target_hnsw_accuracy_percent = 95.0
target_cpu_utilization_percent = 95.0
test_iterations = 500                # Many iterations for accuracy
warmup_iterations = 50
enable_memory_profiling = true       # Detailed profiling
enable_cpu_profiling = true

[statistical]
confidence_level = 0.99              # High confidence for benchmarks
min_sample_size = 5000
significance_threshold = 0.001       # Very strict
enable_regime_analysis = false
enable_correlation_analysis = false

[execution]
enable_detailed_logging = false      # Minimal logging for performance
enable_performance_profiling = true
parallel_execution = true
timeout_seconds = 1800               # Long timeout for thorough testing
max_memory_mb = 8192
output_format = "benchmark"
report_directory = "benchmarks"

[benchmark_specific]
enable_flamegraph = true
enable_perf_counters = true
measure_cache_performance = true
measure_branch_prediction = true
compare_with_baseline = true
baseline_file = "benchmarks/baseline.json"
```

### 2. Mathematical Accuracy Focus

**Purpose**: Comprehensive mathematical validation with strict tolerances.

```toml
# config/accuracy_config.toml
[mathematical]
tolerance = 1e-10                    # Very strict tolerance
test_edge_cases = true
test_extreme_values = true
enable_simd_tests = true
enable_hnsw_tests = true
enable_precision_tests = true
test_numerical_stability = true

[performance]
target_latency_1k_samples_ms = 10.0  # Relaxed for accuracy focus
target_latency_10k_samples_ms = 50.0
target_latency_50k_samples_ms = 200.0
target_hnsw_accuracy_percent = 99.0  # Very high accuracy
target_cpu_utilization_percent = 70.0
test_iterations = 10                 # Fewer iterations, focus on correctness
warmup_iterations = 2
enable_memory_profiling = false
enable_cpu_profiling = false

[statistical]
confidence_level = 0.999             # Extremely high confidence
min_sample_size = 10000
significance_threshold = 0.0001      # Very strict
enable_regime_analysis = true
enable_correlation_analysis = true

[execution]
enable_detailed_logging = true
enable_performance_profiling = false
parallel_execution = false           # Single-threaded for determinism
timeout_seconds = 3600               # Long timeout for thorough testing
max_memory_mb = 4096
output_format = "detailed"
report_directory = "accuracy_reports"

[accuracy_specific]
test_reference_implementations = true
compare_with_external_libraries = true
test_cross_platform_consistency = true
validate_ieee754_compliance = true
```

### 3. Integration Testing Focus

**Purpose**: Comprehensive end-to-end testing with realistic scenarios.

```toml
# config/integration_config.toml
[mathematical]
tolerance = 1e-6
test_edge_cases = true
test_extreme_values = false
enable_simd_tests = true
enable_hnsw_tests = true

[performance]
target_latency_1k_samples_ms = 1.0
target_latency_10k_samples_ms = 2.0
target_latency_50k_samples_ms = 10.0
target_hnsw_accuracy_percent = 95.0
target_cpu_utilization_percent = 85.0
test_iterations = 50
warmup_iterations = 5
enable_memory_profiling = true
enable_cpu_profiling = false

[statistical]
confidence_level = 0.95
min_sample_size = 1000
significance_threshold = 0.05
enable_regime_analysis = true
enable_correlation_analysis = true

[backtesting]
initial_capital = 100000.0
position_size = 0.1
transaction_cost = 0.001
slippage = 0.0005
signal_threshold = 0.6
max_positions = 5
rebalance_frequency_seconds = 3600

[execution]
enable_detailed_logging = true
enable_performance_profiling = false
parallel_execution = true
timeout_seconds = 900
max_memory_mb = 3072
output_format = "comprehensive"
report_directory = "integration_reports"

[integration_specific]
test_error_scenarios = true
test_recovery_mechanisms = true
test_configuration_changes = true
test_concurrent_operations = true
test_data_corruption_handling = true
simulate_network_failures = true
simulate_disk_failures = true

[test_scenarios]
# Define specific integration test scenarios
[[test_scenarios.scenario]]
name = "normal_operation"
duration_minutes = 30
data_source = "synthetic"
market_conditions = "normal"

[[test_scenarios.scenario]]
name = "high_volatility"
duration_minutes = 15
data_source = "historical"
market_conditions = "volatile"
volatility_multiplier = 3.0

[[test_scenarios.scenario]]
name = "trending_market"
duration_minutes = 45
data_source = "synthetic"
market_conditions = "trending"
trend_strength = 0.8

[[test_scenarios.scenario]]
name = "sideways_market"
duration_minutes = 60
data_source = "synthetic"
market_conditions = "ranging"
volatility_multiplier = 0.5
```

### 4. Memory-Constrained Environment

**Purpose**: Testing in resource-limited environments.

```toml
# config/low_memory_config.toml
[mathematical]
tolerance = 1e-6
test_edge_cases = true
test_extreme_values = false          # Skip memory-intensive tests
enable_simd_tests = true
enable_hnsw_tests = false            # HNSW uses more memory

[performance]
target_latency_1k_samples_ms = 2.0   # Relaxed due to constraints
target_latency_10k_samples_ms = 8.0
target_latency_50k_samples_ms = 30.0
target_hnsw_accuracy_percent = 90.0
target_cpu_utilization_percent = 70.0
test_iterations = 20                 # Fewer iterations
warmup_iterations = 2
enable_memory_profiling = true       # Monitor memory usage
enable_cpu_profiling = false

[statistical]
confidence_level = 0.90              # Relaxed due to smaller samples
min_sample_size = 500                # Smaller samples
significance_threshold = 0.05
enable_regime_analysis = false       # Skip memory-intensive analysis
enable_correlation_analysis = true

[backtesting]
initial_capital = 50000.0
position_size = 0.05                 # Smaller positions
transaction_cost = 0.001
slippage = 0.0005
signal_threshold = 0.6
max_positions = 3                    # Fewer positions
rebalance_frequency_seconds = 7200

[execution]
enable_detailed_logging = false      # Reduce memory usage
enable_performance_profiling = false
parallel_execution = false           # Single-threaded to save memory
timeout_seconds = 1200
max_memory_mb = 512                  # Strict memory limit
output_format = "compact"
report_directory = "low_mem_reports"

[memory_optimization]
enable_streaming_processing = true
use_memory_mapped_files = true
enable_data_compression = true
garbage_collect_frequency = 100
batch_size = 100                     # Smaller batches
cache_size = 1000                    # Smaller cache
```

## Advanced Configuration Patterns

### 1. Multi-Stage Testing Configuration

```toml
# config/multi_stage_config.toml
[stages.quick]
# Quick smoke tests
mathematical_tolerance = 1e-6
performance_iterations = 10
statistical_sample_size = 100
timeout_seconds = 60

[stages.standard]
# Standard validation
mathematical_tolerance = 1e-7
performance_iterations = 50
statistical_sample_size = 1000
timeout_seconds = 300

[stages.comprehensive]
# Full validation
mathematical_tolerance = 1e-8
performance_iterations = 200
statistical_sample_size = 5000
timeout_seconds = 1800

[execution]
default_stage = "standard"
enable_stage_progression = true      # Auto-progress through stages
fail_fast_between_stages = true
```

### 2. A/B Testing Configuration

```toml
# config/ab_testing_config.toml
[baseline]
# Configuration A (baseline)
ldc_neighbors_count = 8
ldc_max_bars_back = 2000
use_hnsw_index = false
enable_simd = true

[variant]
# Configuration B (variant)
ldc_neighbors_count = 12
ldc_max_bars_back = 3000
use_hnsw_index = true
enable_simd = true

[comparison]
statistical_significance_threshold = 0.05
min_effect_size = 0.05               # Minimum meaningful difference
test_duration_minutes = 120
sample_size_per_variant = 2000

[metrics]
primary_metric = "prediction_accuracy"
secondary_metrics = ["latency", "memory_usage", "cpu_usage"]
```

### 3. Regression Testing Configuration

```toml
# config/regression_config.toml
[baseline]
baseline_file = "test_reports/baseline_v1.2.3.json"
performance_tolerance_percent = 5.0   # Allow 5% performance regression
accuracy_tolerance = 1e-7

[regression_detection]
enable_performance_regression = true
enable_accuracy_regression = true
enable_memory_regression = true
alert_on_regression = true

[comparison_metrics]
latency_p50 = { tolerance = 0.05, alert_threshold = 0.10 }
latency_p95 = { tolerance = 0.10, alert_threshold = 0.20 }
latency_p99 = { tolerance = 0.15, alert_threshold = 0.30 }
memory_peak = { tolerance = 0.05, alert_threshold = 0.10 }
accuracy_hit_rate = { tolerance = 0.01, alert_threshold = 0.02 }
```

## Configuration Loading and Management

### Programmatic Configuration

```rust
use ldc_engine::testing::{TestConfig, ConfigBuilder};

// Build configuration programmatically
let config = ConfigBuilder::new()
    .mathematical_tolerance(1e-6)
    .performance_target_1k(0.5)
    .performance_target_10k(1.0)
    .performance_target_50k(5.0)
    .statistical_confidence(0.95)
    .enable_detailed_logging(true)
    .parallel_execution(true)
    .timeout_seconds(300)
    .build()?;

// Load from environment variables
let config = TestConfig::from_env_with_prefix("LDC_TEST")?;

// Merge configurations
let base_config = TestConfig::from_file("config/base_config.toml")?;
let env_overrides = TestConfig::from_env()?;
let final_config = base_config.merge(env_overrides)?;
```

### Environment Variable Overrides

```bash
# Override specific configuration values
export LDC_TEST_MATHEMATICAL_TOLERANCE=1e-8
export LDC_TEST_PERFORMANCE_TARGET_1K=0.3
export LDC_TEST_PERFORMANCE_ITERATIONS=200
export LDC_TEST_ENABLE_DETAILED_LOGGING=true
export LDC_TEST_PARALLEL_EXECUTION=false
export LDC_TEST_TIMEOUT_SECONDS=600

# Run tests with environment overrides
cargo run --example automated_test_runner_demo
```

### Configuration Validation

```rust
use ldc_engine::testing::{ConfigValidator, ValidationError};

// Validate configuration before use
let validator = ConfigValidator::new();
let validation_result = validator.validate(&config)?;

if !validation_result.is_valid() {
    for error in validation_result.errors() {
        match error {
            ValidationError::InvalidTolerance { value } => {
                eprintln!("Invalid tolerance: {} (must be > 0 and < 1)", value);
            },
            ValidationError::InvalidTimeout { value } => {
                eprintln!("Invalid timeout: {}s (must be > 0)", value);
            },
            ValidationError::IncompatibleSettings { setting1, setting2 } => {
                eprintln!("Incompatible settings: {} and {}", setting1, setting2);
            },
        }
    }
    return Err(anyhow::anyhow!("Configuration validation failed"));
}
```

## Best Practices

### 1. Configuration Organization

- **Separate by environment**: Use different configuration files for dev, staging, and production
- **Use inheritance**: Create base configurations and override specific values
- **Document settings**: Include comments explaining the purpose of each setting
- **Version control**: Track configuration changes alongside code changes

### 2. Performance Tuning

- **Start with conservative targets**: Begin with achievable performance targets and tighten over time
- **Account for environment differences**: CI/CD environments typically have different performance characteristics
- **Use appropriate sample sizes**: Balance test accuracy with execution time
- **Monitor resource usage**: Track memory and CPU usage to identify bottlenecks

### 3. Statistical Configuration

- **Choose appropriate confidence levels**: Higher confidence requires larger sample sizes
- **Consider multiple testing corrections**: Adjust significance thresholds when running many tests
- **Validate assumptions**: Ensure statistical tests are appropriate for your data
- **Document methodology**: Clearly document statistical methods and assumptions

### 4. Maintenance

- **Regular review**: Periodically review and update configurations as the system evolves
- **Automated validation**: Use configuration validation to catch errors early
- **Performance baselines**: Maintain performance baselines and update them with significant improvements
- **Documentation**: Keep configuration documentation up to date with changes

This comprehensive guide provides practical examples and patterns for configuring the LDC engine testing framework across different environments and use cases.