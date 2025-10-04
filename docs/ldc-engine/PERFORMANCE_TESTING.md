# LDC Engine Performance Testing Framework

This document describes the comprehensive performance testing and benchmarking framework for the LDC engine optimization features.

## Overview

The performance testing framework validates that all optimization features meet the specified requirements while maintaining Pine Script compatibility. It includes:

- **Micro-benchmarks** using Criterion.rs for detailed performance analysis
- **Integration tests** verifying Pine Script compatibility across all optimizations
- **Performance requirement validation** ensuring sub-millisecond query times
- **Memory efficiency testing** for optimized data structures
- **Concurrent access testing** for thread safety and scalability

## Quick Start

### Running All Tests

```bash
cd rust/ldc-engine
./run_performance_tests.sh
```

This script will:
1. Run unit tests for correctness
2. Verify Pine Script compatibility
3. Validate performance requirements
4. Execute comprehensive benchmarks
5. Generate HTML reports and summary

### Running Individual Test Suites

```bash
# Unit tests
cargo test --lib --release

# Pine Script compatibility tests
cargo test --test pine_script_compatibility_tests --release

# Performance requirements tests
cargo test --test performance_requirements_tests --release

# Benchmarks only
cargo bench --bench performance_benchmarks
```

## Test Categories

### 1. Pine Script Compatibility Tests (`pine_script_compatibility_tests.rs`)

Ensures all optimizations maintain exact Pine Script behavior:

- **Lorentzian Distance Accuracy**: Verifies SIMD and standard calculations match exactly
- **Batch Processing Consistency**: Tests batch operations maintain individual calculation accuracy
- **k-NN Search Consistency**: Validates results across sequential, parallel, and HNSW strategies
- **HNSW Accuracy Requirements**: Ensures 95%+ accuracy compared to exact search
- **Memory Optimization Compatibility**: Verifies optimized data structures preserve data integrity
- **Thread Pool Strategy Consistency**: Tests different threading strategies produce identical results
- **Full Stack Integration**: Validates all optimizations work together without breaking compatibility

### 2. Performance Requirements Tests (`performance_requirements_tests.rs`)

Validates specific performance targets from requirements:

- **Query Time Requirements**: 
  - 10k samples in <1ms (Requirement 1.1)
  - 50k samples in <5ms (Requirement 1.2)
- **CPU Utilization**: 90% utilization during parallel processing (Requirement 2.2)
- **Memory Threshold Monitoring**: Triggers at 80% RAM usage (Requirement 3.4)
- **HNSW Accuracy**: Maintains 95%+ accuracy (Requirement 4.3)
- **Performance Metrics Tracking**: Comprehensive metrics collection (Requirements 5.1-5.5)
- **Concurrent Access**: Thread safety and scalability validation
- **1ms Query Target**: Real-world performance validation

### 3. Comprehensive Benchmarks (`performance_benchmarks.rs`)

Detailed micro-benchmarks using Criterion.rs:

#### k-NN Search Benchmarks
- Compares exact vs HNSW vs parallel strategies
- Tests multiple dataset sizes (1k, 5k, 10k, 25k, 50k samples)
- Measures throughput and latency
- Automatic strategy selection validation

#### SIMD Distance Calculation Benchmarks
- Single distance calculation: standard vs SIMD
- Batch distance calculation: various batch sizes
- Throughput measurement for batch operations
- SIMD optimization effectiveness

#### Memory Usage Benchmarks
- VecDeque vs optimized storage comparison
- Memory pool allocation/deallocation performance
- Memory-mapped storage efficiency
- Memory alignment impact on performance

#### HNSW Operations Benchmarks
- Index construction time vs dataset size
- Search performance vs accuracy trade-offs
- Index rebuild performance
- Memory usage of HNSW structures

#### Thread Pool Strategy Benchmarks
- Global vs dedicated vs adaptive strategies
- Thread utilization efficiency
- Work distribution effectiveness
- Scalability across different core counts

## Benchmark Configuration

### Criterion.rs Settings

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "performance_benchmarks"
harness = false
```

### Environment Variables

```bash
export RAYON_NUM_THREADS=$(nproc)  # Use all available cores
export RUST_LOG=error              # Reduce log noise during benchmarks
```

### Benchmark Parameters

- **Measurement Time**: 5-10 seconds per benchmark
- **Sample Size**: 20-100 samples depending on benchmark complexity
- **Warm-up**: Multiple iterations before measurement
- **Statistical Analysis**: Mean, standard deviation, percentiles

## Performance Targets

### Query Time Targets

| Dataset Size | Target Time | Strategy |
|--------------|-------------|----------|
| 1k samples | <0.1ms | Sequential |
| 10k samples | <1ms | Parallel/HNSW |
| 50k samples | <5ms | HNSW |

### Accuracy Targets

- **HNSW Accuracy**: ≥95% compared to exact search
- **SIMD Accuracy**: Exact match with standard calculations
- **Pine Script Compatibility**: 100% behavioral compatibility

### Resource Utilization Targets

- **CPU Utilization**: ≥90% during parallel processing
- **Memory Efficiency**: <80% RAM usage before compression
- **Thread Efficiency**: Linear scaling with available cores

## Output and Reports

### HTML Reports

Criterion generates interactive HTML reports at:
```
target/criterion/report/index.html
```

Features:
- Interactive charts and graphs
- Performance comparisons over time
- Statistical analysis and confidence intervals
- Regression detection

### Test Logs

All test results are saved with timestamps:
- `performance_results/unit_tests_YYYYMMDD_HHMMSS.log`
- `performance_results/compatibility_tests_YYYYMMDD_HHMMSS.log`
- `performance_results/requirements_tests_YYYYMMDD_HHMMSS.log`
- `performance_results/benchmarks_YYYYMMDD_HHMMSS.log`

### Performance Report

Comprehensive markdown report generated:
- `performance_results/performance_report_YYYYMMDD_HHMMSS.md`

Includes:
- System information
- Test summary
- Performance metrics
- Optimization recommendations
- Detailed results links

## Continuous Integration

### Automated Testing

The framework supports CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Performance Tests
  run: |
    cd rust/ldc-engine
    ./run_performance_tests.sh
    
- name: Upload Performance Reports
  uses: actions/upload-artifact@v3
  with:
    name: performance-reports
    path: rust/ldc-engine/performance_results/
```

### Performance Regression Detection

Criterion automatically detects performance regressions:
- Compares against previous benchmark runs
- Statistical significance testing
- Configurable regression thresholds
- Automated alerts on performance degradation

## Troubleshooting

### Common Issues

1. **Slow Benchmark Execution**
   - Reduce sample sizes in benchmark configuration
   - Use `--quick` flag for faster runs during development
   - Ensure system is not under load during benchmarking

2. **Memory-Related Test Failures**
   - Increase system memory or reduce test dataset sizes
   - Check for memory leaks in optimized data structures
   - Verify memory pool configuration

3. **HNSW Accuracy Issues**
   - Increase `ef_search` parameter for better accuracy
   - Verify distance function implementation
   - Check for numerical precision issues

4. **Thread Pool Issues**
   - Verify system has sufficient CPU cores
   - Check for thread contention or deadlocks
   - Ensure proper thread pool cleanup

### Performance Debugging

1. **Enable Debug Symbols**
   ```toml
   [profile.bench]
   debug = true
   ```

2. **Use Profiling Tools**
   ```bash
   cargo bench --bench performance_benchmarks -- --profile-time=5
   perf record cargo bench
   ```

3. **Memory Profiling**
   ```bash
   valgrind --tool=massif cargo test test_memory_pool_performance
   ```

## Extending the Framework

### Adding New Benchmarks

1. Add benchmark function to `benches/performance_benchmarks.rs`:
   ```rust
   fn benchmark_new_feature(c: &mut Criterion) {
       let mut group = c.benchmark_group("new_feature");
       // Benchmark implementation
       group.finish();
   }
   ```

2. Add to criterion group:
   ```rust
   criterion_group!(benches, ..., benchmark_new_feature);
   ```

### Adding New Tests

1. Create test in appropriate test file:
   ```rust
   #[test]
   fn test_new_requirement() {
       // Test implementation
   }
   ```

2. Update `run_performance_tests.sh` if needed

### Custom Metrics

Add custom performance metrics:
```rust
impl PerformanceMetrics {
    pub fn custom_metric(&self) -> f64 {
        // Custom calculation
    }
}
```

## Best Practices

### Benchmark Design

1. **Reproducible Results**: Use fixed seeds for random data
2. **Realistic Workloads**: Use representative data and query patterns
3. **Proper Warm-up**: Allow JIT compilation and cache warming
4. **Statistical Significance**: Use sufficient sample sizes
5. **Isolation**: Run benchmarks on dedicated systems when possible

### Test Design

1. **Comprehensive Coverage**: Test all optimization paths
2. **Edge Cases**: Include boundary conditions and error cases
3. **Compatibility**: Verify behavior matches reference implementation
4. **Performance**: Validate against specific requirements
5. **Regression**: Detect performance degradation over time

### Maintenance

1. **Regular Updates**: Keep benchmarks current with code changes
2. **Baseline Updates**: Update performance baselines as hardware improves
3. **Documentation**: Keep performance documentation up to date
4. **Monitoring**: Set up automated performance monitoring in CI/CD

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [LDC Engine Requirements](requirements.md)
- [LDC Engine Design](design.md)