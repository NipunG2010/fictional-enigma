# LDC Engine Performance Testing & Tuning

## Overview

This document covers the performance testing framework and tuning guide for the LDC engine, including benchmarking, optimization strategies, and system-level configurations.

---

## Part 1: Performance Testing Framework

The performance testing framework validates that all optimization features meet specified requirements while maintaining Pine Script compatibility. It includes:

- **Micro-benchmarks** using Criterion.rs for detailed performance analysis
- **Integration tests** verifying Pine Script compatibility across all optimizations
- **Performance requirement validation** ensuring sub-millisecond query times
- **Memory efficiency testing** for optimized data structures
- **Concurrent access testing** for thread safety and scalability

### Quick Start

```bash
cd rust/ldc-engine
./run_performance_tests.sh
```

This script runs unit tests, Pine Script compatibility checks, performance requirement validation, comprehensive benchmarks, and generates HTML reports.

#### Running Individual Test Suites

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

### Test Categories

#### Pine Script Compatibility Tests

Ensures all optimizations maintain exact Pine Script behavior:
- Lorentzian distance accuracy (SIMD and standard match exactly)
- Batch processing consistency
- k-NN search consistency across strategies
- HNSW accuracy requirements (≥95% vs exact search)
- Memory optimization compatibility
- Thread pool strategy consistency

#### Performance Requirements Tests

Validates specific performance targets:
- **Query Time**: 10k samples <1ms (Req 1.1), 50k samples <5ms (Req 1.2)
- **CPU Utilization**: 90% during parallel processing (Req 2.2)
- **Memory Threshold**: Monitoring triggers at 80% RAM (Req 3.4)
- **HNSW Accuracy**: ≥95% maintained (Req 4.3)
- **Concurrent Access**: Thread safety and scalability validation

#### Comprehensive Benchmarks (Criterion.rs)

**k-NN Search**: exact vs HNSW vs parallel, dataset sizes 1k–50k

**SIMD Distance Calculation**: standard vs SIMD, batch sizes, throughput

**Memory Usage**: VecDeque vs optimized storage, memory pool, memory-mapped storage

**HNSW Operations**: construction time, search vs accuracy trade-offs, rebuild performance

**Thread Pool Strategy**: global vs dedicated vs adaptive, work distribution

### Benchmark Configuration

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "performance_benchmarks"
harness = false
```

```bash
export RAYON_NUM_THREADS=$(nproc)
export RUST_LOG=error
```

### Performance Targets

| Dataset Size | Target Time | Strategy |
|---|---|---|
| 1k samples | <0.1ms | Sequential |
| 10k samples | <1ms | Parallel/HNSW |
| 50k samples | <5ms | HNSW |

- **HNSW Accuracy**: ≥95% compared to exact search
- **SIMD Accuracy**: Exact match with standard calculations
- **Pine Script Compatibility**: 100% behavioral compatibility
- **CPU Utilization**: ≥90% during parallel processing
- **Memory Efficiency**: <80% RAM usage before compression

### Output and Reports

Criterion generates interactive HTML reports at `target/criterion/report/index.html`.

Test logs are saved with timestamps under `performance_results/`. A comprehensive markdown report is generated at `performance_results/performance_report_YYYYMMDD_HHMMSS.md`.

### CI/CD Integration

```yaml
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

Criterion automatically detects performance regressions with statistical significance testing and configurable thresholds.

### Troubleshooting

1. **Slow benchmarks**: Reduce sample sizes, use `--quick` flag, ensure system is not under load
2. **Memory failures**: Increase system memory, check for leaks in optimized data structures, verify memory pool config
3. **HNSW accuracy issues**: Increase `ef_search`, verify distance function, check numerical precision
4. **Thread pool issues**: Verify sufficient CPU cores, check for contention/deadlocks, ensure proper cleanup

---

## Part 2: Performance Tuning Guide

This section provides instructions for optimizing the LDC engine, including system-level optimizations, algorithm tuning, and environment-specific configurations.

### Performance Analysis and Profiling

#### System Performance Baseline

```rust
fn establish_performance_baseline() -> SystemBaseline {
    let mut system = System::new_all();
    system.refresh_all();
    
    let cpu_info = CpuInfo {
        cores: system.cpus().len(),
        frequency_mhz: system.cpus()[0].frequency(),
        architecture: std::env::consts::ARCH.to_string(),
        features: get_cpu_features(),
    };
    
    let baseline_scores = BaselineScores {
        cpu_score: benchmark_cpu_performance(),
        memory_bandwidth_score: benchmark_memory_bandwidth(),
        storage_iops_score: benchmark_storage_performance(),
        network_latency_score: benchmark_network_latency(),
    };
    
    SystemBaseline { cpu_info, memory_info, storage_info, baseline_scores }
}
```

#### Built-in Profiler

```rust
pub struct PerformanceProfiler {
    timings: HashMap<String, Vec<Duration>>,
    memory_usage: HashMap<String, usize>,
    call_counts: HashMap<String, usize>,
    active_operations: HashMap<String, Instant>,
}

impl PerformanceProfiler {
    pub fn start_operation(&mut self, name: &str) {
        self.active_operations.insert(name.to_string(), Instant::now());
        *self.call_counts.entry(name.to_string()).or_insert(0) += 1;
    }
    
    pub fn end_operation(&mut self, name: &str) {
        if let Some(start_time) = self.active_operations.remove(name) {
            let duration = start_time.elapsed();
            self.timings.entry(name.to_string()).or_default().push(duration);
        }
    }
    
    pub fn generate_performance_report(&self) -> PerformanceReport {
        // Sort operations by total time (highest impact first)
        let mut operations = self.build_operation_profiles();
        operations.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        PerformanceReport { operations }
    }
}
```

### Algorithm-Specific Optimizations

#### SIMD Distance Calculations

```rust
#[cfg(target_arch = "x86_64")]
pub fn lorentzian_distance_avx2(features1: &[f32], features2: &[f32]) -> f32 {
    assert!(features1.len() % 8 == 0, "Feature length must be multiple of 8 for AVX2");
    
    unsafe {
        let mut sum = _mm256_setzero_ps();
        
        for chunk in 0..(features1.len() / 8) {
            let offset = chunk * 8;
            let a = _mm256_loadu_ps(features1.as_ptr().add(offset));
            let b = _mm256_loadu_ps(features2.as_ptr().add(offset));
            let diff = _mm256_sub_ps(a, b);
            let diff_squared = _mm256_mul_ps(diff, diff);
            let ones = _mm256_set1_ps(1.0);
            let term = _mm256_add_ps(ones, diff_squared);
            let log_term = fast_log_avx2(term);
            sum = _mm256_add_ps(sum, log_term);
        }
        
        horizontal_sum_avx2(sum)
    }
}

// Fallback with runtime detection
pub fn lorentzian_distance_optimized(features1: &[f32], features2: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && features1.len() % 8 == 0 {
            return lorentzian_distance_avx2(features1, features2);
        }
    }
    
    features1.iter().zip(features2.iter())
        .map(|(&a, &b)| { let diff = a - b; (1.0 + diff * diff).ln() })
        .sum()
}
```

#### Memory Layout Optimizations

```rust
// Structure of Arrays for better cache performance
#[derive(Clone)]
pub struct SoAFeatures {
    f1: Vec<f32>, f2: Vec<f32>, f3: Vec<f32>, f4: Vec<f32>, f5: Vec<f32>,
}

// Cache-line aligned features
#[repr(C, align(64))]
pub struct AlignedFeatures {
    pub data: [f32; 5],
    _padding: [u8; 44], // Pad to 64 bytes
}
```

### HNSW Index Parameter Tuning

```rust
pub struct HNSWTuner {
    dataset_characteristics: DatasetCharacteristics,
    performance_requirements: PerformanceRequirements,
}

impl HNSWTuner {
    pub fn tune_parameters(&self) -> HNSWConfig {
        HNSWConfig {
            m: self.tune_m_parameter(),
            ef_construction: self.tune_ef_construction(),
            ef_search: self.tune_ef_search(),
            ..Default::default()
        }
    }
    
    fn tune_m_parameter(&self) -> usize {
        match (self.dataset_characteristics.size, self.performance_requirements.target_accuracy) {
            (size, acc) if size < 10_000 && acc >= 0.98 => 32,
            (size, acc) if size < 10_000 && acc >= 0.95 => 24,
            (size, acc) if size < 100_000 && acc >= 0.95 => 16,
            (size, acc) if size < 100_000 && acc >= 0.90 => 12,
            (_, acc) if acc >= 0.95 => 12,
            _ => 8,
        }
    }
}

// Adaptive HNSW configuration that self-tunes based on observed performance
pub struct AdaptiveHNSWConfig {
    config: HNSWConfig,
    performance_history: Vec<PerformanceMetric>,
}

impl AdaptiveHNSWConfig {
    pub fn record_performance(&mut self, metric: PerformanceMetric) {
        self.performance_history.push(metric);
        if self.performance_history.len() > 100 {
            self.performance_history.drain(0..50);
        }
        if self.should_adapt() {
            self.adapt_configuration();
        }
    }
}
```

### Parallel Processing Optimizations

```rust
pub struct OptimizedThreadPool {
    pool: rayon::ThreadPool,
}

impl OptimizedThreadPool {
    pub fn optimal_config() -> ThreadPoolConfig {
        let num_cores = thread::available_parallelism()
            .map(|n| n.get()).unwrap_or(1);
        
        ThreadPoolConfig {
            num_threads: match num_cores {
                1 => 1,
                2..=4 => num_cores,
                5..=8 => num_cores - 1,
                9..=16 => num_cores - 2,
                _ => num_cores - 4,
            },
            stack_size: 8 * 1024 * 1024,
            thread_affinity: num_cores >= 8,
            numa_aware: num_cores >= 16,
        }
    }
    
    pub fn parallel_distance_calculation<F>(&self, 
        features: &[FeatureSeries], 
        query: &FeatureSeries,
        distance_fn: F
    ) -> Vec<f32>
    where F: Fn(&FeatureSeries, &FeatureSeries) -> f32 + Sync + Send {
        self.pool.install(|| {
            features.par_iter()
                .map(|feature| distance_fn(feature, query))
                .collect()
        })
    }
}
```

### Memory Optimizations

#### Memory Pool Management

```rust
pub struct MemoryPool {
    pools: Vec<Mutex<Vec<NonNull<u8>>>>,
    sizes: Vec<usize>, // [64, 128, 256, ..., 32768]
}

impl MemoryPool {
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        let pool_index = self.sizes.iter().position(|&s| s >= size)?;
        let mut pool = self.pools[pool_index].lock().unwrap();
        pool.pop().or_else(|| {
            let layout = Layout::from_size_align(self.sizes[pool_index], 8).ok()?;
            NonNull::new(unsafe { System.alloc(layout) })
        })
    }
}
```

### Adaptive Configuration

```rust
pub struct AdaptivePerformanceConfig {
    current_config: LDCConfig,
    performance_history: VecDeque<PerformanceSnapshot>,
    adaptation_strategy: AdaptationStrategy,
    last_adaptation: Instant,
    adaptation_cooldown: Duration,
}

impl AdaptivePerformanceConfig {
    pub fn record_performance(&mut self, snapshot: PerformanceSnapshot) {
        self.performance_history.push_back(snapshot);
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }
        if self.should_adapt() {
            self.adapt_configuration();
        }
    }
}
```

### I/O Optimizations

```rust
pub struct OptimizedDataLoader {
    use_memory_mapping: bool,
    buffer_size: usize,
}

impl OptimizedDataLoader {
    pub fn load_ohlcv_data(&self, path: &str) -> Result<Vec<OHLCV>> {
        if self.use_memory_mapping {
            self.load_with_mmap(path)
        } else {
            self.load_with_buffered_io(path)
        }
    }
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            use_memory_mapping: true,
            buffer_size: 64 * 1024, // 64KB buffer
            prefetch_enabled: true,
        }
    }
}
```

### Best Practices Summary

1. **Measurement first**: Establish baselines before optimizing; profile to find actual bottlenecks
2. **Algorithm optimization**: Use SIMD for vectorizable ops, cache-friendly layouts, appropriate algorithms per dataset size
3. **System-level**: Configure thread pools for your core count, use NUMA-aware scheduling on large systems
4. **Adaptive configuration**: Monitor and auto-adjust `ef_search` based on observed accuracy/latency tradeoffs
5. **Environment-specific**: Relax CI targets by 2-4x to account for shared runners; tune memory limits per deployment

### Extending the Framework

```rust
fn benchmark_new_feature(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_feature");
    // Benchmark implementation
    group.finish();
}

criterion_group!(benches, existing_benchmarks, benchmark_new_feature);
```
