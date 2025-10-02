# Testing Framework Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for common issues encountered when using the LDC engine testing framework, including diagnostic steps, solutions, and prevention strategies.

## Common Test Failures and Solutions

### Mathematical Accuracy Test Failures

#### Issue 1: SIMD vs Standard Calculation Differences

**Symptoms:**
```
FAILED: SIMD_vs_Standard_normal_features
Expected: 2.345678901234567
Actual:   2.345678901234568
Difference: 1e-15
Tolerance: 1e-6
```

**Diagnosis Steps:**
```rust
// Check if difference is within floating-point precision
fn diagnose_simd_difference(result: &UnitTestResult) -> DiagnosisResult {
    let relative_error = result.difference / result.expected.abs();
    
    if result.difference < f64::EPSILON {
        DiagnosisResult::FloatingPointPrecision
    } else if relative_error < 1e-12 {
        DiagnosisResult::AcceptablePrecision
    } else {
        DiagnosisResult::AlgorithmicDifference
    }
}
```

**Solutions:**
1. **Adjust Tolerance**: If difference is within floating-point precision
   ```rust
   let config = MathematicalTestConfig {
       tolerance: 1e-12, // More lenient for floating-point comparisons
       use_relative_tolerance: true,
   };
   ```

2. **Verify SIMD Implementation**: Check SIMD code matches standard algorithm
   ```rust
   // Ensure SIMD implementation uses same order of operations
   #[cfg(target_arch = "x86_64")]
   fn lorentzian_distance_simd(features1: &[f32], features2: &[f32]) -> f32 {
       // Verify: same precision, same operation order
       unsafe {
           // SIMD implementation should match standard exactly
       }
   }
   ```

3. **Use Relative Tolerance**: For large numbers, use relative comparison
   ```rust
   fn compare_with_relative_tolerance(expected: f64, actual: f64, tolerance: f64) -> bool {
       let diff = (expected - actual).abs();
       let relative_diff = diff / expected.abs().max(actual.abs());
       relative_diff <= tolerance
   }
   ```

**Prevention:**
- Use consistent floating-point precision across implementations
- Test with known reference values
- Implement comprehensive edge case testing

#### Issue 2: NaN or Infinity in Distance Calculations

**Symptoms:**
```
FAILED: EdgeCase_extreme_values
Expected: finite value
Actual: NaN
Input: features1=[f32::MAX, f32::MIN, 1e10, -1e10, 0.0]
```

**Diagnosis Steps:**
```rust
fn diagnose_nan_infinity(features1: &[f32], features2: &[f32]) -> DiagnosisResult {
    // Check input validity
    for (i, &value) in features1.iter().enumerate() {
        if value.is_nan() {
            return DiagnosisResult::InputNaN(i);
        }
        if value.is_infinite() {
            return DiagnosisResult::InputInfinity(i);
        }
        if value.abs() > 1e20 {
            return DiagnosisResult::ExtremeValue(i, value);
        }
    }
    
    // Check for overflow in calculations
    let sum_of_squares: f64 = features1.iter()
        .zip(features2.iter())
        .map(|(&a, &b)| {
            let diff = (a - b) as f64;
            diff * diff
        })
        .sum();
    
    if sum_of_squares.is_infinite() {
        DiagnosisResult::CalculationOverflow
    } else {
        DiagnosisResult::UnknownNumericalIssue
    }
}
```

**Solutions:**
1. **Input Validation and Sanitization**:
   ```rust
   fn sanitize_features(features: &mut [f32]) {
       for feature in features.iter_mut() {
           if feature.is_nan() {
               *feature = 0.0; // or median value
           } else if feature.is_infinite() {
               *feature = if feature.is_sign_positive() { 1e6 } else { -1e6 };
           } else if feature.abs() > 1e6 {
               *feature = feature.signum() * 1e6; // Clamp extreme values
           }
       }
   }
   ```

2. **Numerically Stable Distance Calculation**:
   ```rust
   fn lorentzian_distance_stable(features1: &[f32], features2: &[f32]) -> f32 {
       let mut sum = 0.0f64; // Use double precision for intermediate calculations
       
       for (&a, &b) in features1.iter().zip(features2.iter()) {
           let diff = (a as f64) - (b as f64);
           let term = 1.0 + diff * diff;
           sum += term.ln();
       }
       
       sum as f32
   }
   ```

3. **Robust Edge Case Handling**:
   ```rust
   fn calculate_distance_with_checks(features1: &[f32], features2: &[f32]) -> Result<f32> {
       // Pre-flight checks
       if features1.len() != features2.len() {
           return Err(anyhow::anyhow!("Feature length mismatch"));
       }
       
       // Check for problematic values
       for (i, (&a, &b)) in features1.iter().zip(features2.iter()).enumerate() {
           if a.is_nan() || b.is_nan() {
               return Err(anyhow::anyhow!("NaN detected at index {}", i));
           }
           if a.is_infinite() || b.is_infinite() {
               return Err(anyhow::anyhow!("Infinity detected at index {}", i));
           }
       }
       
       // Perform calculation with overflow protection
       let distance = lorentzian_distance_stable(features1, features2);
       
       if distance.is_nan() || distance.is_infinite() {
           return Err(anyhow::anyhow!("Calculation resulted in NaN/Infinity"));
       }
       
       Ok(distance)
   }
   ```

**Prevention:**
- Implement comprehensive input validation
- Use numerically stable algorithms
- Add overflow detection and handling
- Test with extreme value datasets

### Performance Test Failures

#### Issue 3: Latency Targets Not Met

**Symptoms:**
```
⏱️ Performance target missed for medium_10k
Target: 1.00ms, Actual: 3.45ms (P95: 7.23ms, P99: 12.45ms)
Dataset size: 10,000 samples
```

**Diagnosis Steps:**
```rust
use std::time::Instant;

fn diagnose_performance_issue(engine: &LDCEngine, test_data: &TestDataset) -> PerformanceDiagnosis {
    let mut diagnosis = PerformanceDiagnosis::new();
    
    // Profile individual operations
    let start = Instant::now();
    let _neighbors = engine.find_k_nearest_neighbors_optimized(&test_data.query_features[0]);
    let total_time = start.elapsed();
    
    // Break down timing
    let distance_calc_time = profile_distance_calculations(engine, test_data);
    let indexing_time = profile_indexing_operations(engine, test_data);
    let memory_access_time = profile_memory_access(engine, test_data);
    
    diagnosis.total_time = total_time;
    diagnosis.distance_calculation_percent = (distance_calc_time.as_nanos() * 100) / total_time.as_nanos();
    diagnosis.indexing_percent = (indexing_time.as_nanos() * 100) / total_time.as_nanos();
    diagnosis.memory_access_percent = (memory_access_time.as_nanos() * 100) / total_time.as_nanos();
    
    // Identify bottlenecks
    if diagnosis.distance_calculation_percent > 60 {
        diagnosis.primary_bottleneck = Bottleneck::DistanceCalculation;
    } else if diagnosis.indexing_percent > 40 {
        diagnosis.primary_bottleneck = Bottleneck::Indexing;
    } else if diagnosis.memory_access_percent > 30 {
        diagnosis.primary_bottleneck = Bottleneck::MemoryAccess;
    }
    
    diagnosis
}
```

**Solutions:**

1. **Enable HNSW Indexing for Large Datasets**:
   ```rust
   let mut config = LDCConfig::default();
   if dataset_size > 5000 {
       config.use_hnsw_index = true;
       config.hnsw_config = HNSWConfig {
           m: 16,
           ef_construction: 200,
           ef_search: 50,
           max_m: 16,
           max_m0: 32,
       };
   }
   ```

2. **Implement SIMD Optimizations**:
   ```rust
   #[cfg(target_arch = "x86_64")]
   fn enable_simd_optimizations(config: &mut LDCConfig) {
       if is_x86_feature_detected!("avx2") {
           config.enable_simd = true;
           config.simd_width = 8; // AVX2 can process 8 f32s at once
       } else if is_x86_feature_detected!("sse4.1") {
           config.enable_simd = true;
           config.simd_width = 4; // SSE can process 4 f32s at once
       }
   }
   ```

3. **Optimize Memory Access Patterns**:
   ```rust
   // Use memory-friendly data layouts
   #[repr(C, align(32))] // Align for SIMD
   struct AlignedFeatures {
       data: [f32; 5],
   }
   
   // Pre-allocate and reuse buffers
   struct PerformanceOptimizedEngine {
       distance_buffer: Vec<f32>,
       index_buffer: Vec<usize>,
       temp_features: Vec<f32>,
   }
   ```

4. **Tune Batch Processing**:
   ```rust
   fn optimize_batch_size(dataset_size: usize) -> usize {
       match dataset_size {
           size if size < 1000 => 100,
           size if size < 10000 => 500,
           size if size < 50000 => 1000,
           _ => 2000,
       }
   }
   ```

**Prevention:**
- Profile regularly during development
- Set realistic performance targets for different environments
- Use appropriate algorithms for dataset sizes
- Monitor resource usage patterns

#### Issue 4: High Memory Usage

**Symptoms:**
```
🧠 Memory usage exceeded limit: 3.2GB used, 2.0GB limit
Peak allocation: 3.2GB at HNSW index construction
Allocation rate: 450MB/s during distance calculations
```

**Diagnosis Steps:**
```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// Memory tracking allocator
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let current = ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK_ALLOCATED.fetch_max(current, Ordering::Relaxed);
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

fn diagnose_memory_usage() -> MemoryDiagnosis {
    MemoryDiagnosis {
        current_usage_mb: ALLOCATED.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0,
        peak_usage_mb: PEAK_ALLOCATED.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0,
    }
}
```

**Solutions:**

1. **Implement Streaming Processing**:
   ```rust
   struct StreamingProcessor {
       batch_size: usize,
       buffer: Vec<TrainingSample>,
   }
   
   impl StreamingProcessor {
       fn process_in_batches<F>(&mut self, data: &[TrainingSample], mut processor: F) 
       where F: FnMut(&[TrainingSample]) {
           for chunk in data.chunks(self.batch_size) {
               processor(chunk);
               // Clear intermediate results to free memory
               self.buffer.clear();
           }
       }
   }
   ```

2. **Use Memory-Mapped Files for Large Datasets**:
   ```rust
   use memmap2::MmapOptions;
   use std::fs::File;
   
   fn load_large_dataset_mmap(path: &str) -> Result<Mmap> {
       let file = File::open(path)?;
       let mmap = unsafe { MmapOptions::new().map(&file)? };
       Ok(mmap)
   }
   ```

3. **Implement Object Pooling**:
   ```rust
   use std::collections::VecDeque;
   
   struct ObjectPool<T> {
       objects: VecDeque<T>,
       factory: Box<dyn Fn() -> T>,
   }
   
   impl<T> ObjectPool<T> {
       fn get(&mut self) -> T {
           self.objects.pop_front().unwrap_or_else(|| (self.factory)())
       }
       
       fn return_object(&mut self, obj: T) {
           if self.objects.len() < 100 { // Limit pool size
               self.objects.push_back(obj);
           }
       }
   }
   ```

4. **Optimize Data Structures**:
   ```rust
   // Use smaller data types where possible
   #[derive(Clone, Copy)]
   struct CompactFeatures {
       f1: f16, // Half precision for memory savings
       f2: f16,
       f3: f16,
       f4: f16,
       f5: f16,
   }
   
   // Use bit packing for enums
   #[repr(u8)]
   enum Direction {
       Long = 0,
       Short = 1,
       Neutral = 2,
   }
   ```

**Prevention:**
- Monitor memory usage during development
- Use memory profiling tools regularly
- Implement memory limits and checks
- Design for memory-constrained environments

### Integration Test Failures

#### Issue 5: Pipeline Component Timeouts

**Symptoms:**
```
❌ Integration test failed: complete_pipeline_test
Error: Component timeout after 30s
Component: feature_computation
Last successful operation: OHLCV data loading (1000 samples)
```

**Diagnosis Steps:**
```rust
use std::time::{Duration, Instant};
use tokio::time::timeout;

async fn diagnose_pipeline_timeout(pipeline: &Pipeline) -> TimeoutDiagnosis {
    let mut diagnosis = TimeoutDiagnosis::new();
    
    // Test each component individually
    let components = vec![
        ("data_loading", || pipeline.test_data_loading()),
        ("feature_computation", || pipeline.test_feature_computation()),
        ("ldc_prediction", || pipeline.test_ldc_prediction()),
        ("signal_generation", || pipeline.test_signal_generation()),
    ];
    
    for (name, test_fn) in components {
        let start = Instant::now();
        match timeout(Duration::from_secs(10), test_fn()).await {
            Ok(Ok(_)) => {
                diagnosis.component_timings.insert(name.to_string(), start.elapsed());
            },
            Ok(Err(e)) => {
                diagnosis.component_errors.insert(name.to_string(), e.to_string());
            },
            Err(_) => {
                diagnosis.timeout_components.push(name.to_string());
            }
        }
    }
    
    diagnosis
}
```

**Solutions:**

1. **Increase Timeouts for Slow Systems**:
   ```rust
   let config = IntegrationTestConfig {
       component_timeout_seconds: 60,  // Increased from 30
       pipeline_timeout_seconds: 300,  // Increased from 120
       enable_timeout_scaling: true,   // Auto-adjust based on system performance
   };
   ```

2. **Implement Asynchronous Processing**:
   ```rust
   use tokio::task;
   
   async fn process_pipeline_async(pipeline: &Pipeline, data: &[OHLCV]) -> Result<Vec<Signal>> {
       // Process in parallel where possible
       let feature_task = task::spawn(async move {
           pipeline.compute_features_async(data).await
       });
       
       let features = feature_task.await??;
       
       // Continue with dependent operations
       let predictions = pipeline.generate_predictions_async(&features).await?;
       let signals = pipeline.generate_signals_async(&predictions).await?;
       
       Ok(signals)
   }
   ```

3. **Add Progress Monitoring**:
   ```rust
   struct ProgressMonitor {
       start_time: Instant,
       last_update: Instant,
       processed_items: usize,
       total_items: usize,
   }
   
   impl ProgressMonitor {
       fn update(&mut self, processed: usize) {
           self.processed_items = processed;
           self.last_update = Instant::now();
           
           let progress = processed as f64 / self.total_items as f64;
           let elapsed = self.start_time.elapsed();
           let estimated_total = elapsed.as_secs_f64() / progress;
           let remaining = estimated_total - elapsed.as_secs_f64();
           
           println!("Progress: {:.1}% ({}/{}) - ETA: {:.1}s", 
                   progress * 100.0, processed, self.total_items, remaining);
       }
   }
   ```

4. **Implement Circuit Breaker Pattern**:
   ```rust
   struct CircuitBreaker {
       failure_count: usize,
       failure_threshold: usize,
       timeout_duration: Duration,
       last_failure_time: Option<Instant>,
   }
   
   impl CircuitBreaker {
       fn call<F, T>(&mut self, operation: F) -> Result<T>
       where F: FnOnce() -> Result<T> {
           if self.is_open() {
               return Err(anyhow::anyhow!("Circuit breaker is open"));
           }
           
           match operation() {
               Ok(result) => {
                   self.reset();
                   Ok(result)
               },
               Err(e) => {
                   self.record_failure();
                   Err(e)
               }
           }
       }
   }
   ```

**Prevention:**
- Set realistic timeouts based on system capabilities
- Implement proper error handling and recovery
- Use asynchronous processing where appropriate
- Monitor component performance regularly

### Statistical Test Failures

#### Issue 6: Insufficient Statistical Significance

**Symptoms:**
```
⚠️ Statistical significance test failed
P-value: 0.087 (threshold: 0.05)
Sample size: 456 (minimum: 1000)
Confidence interval: [0.48, 0.62] (target: [0.52, 0.58])
```

**Diagnosis Steps:**
```rust
fn diagnose_statistical_significance(results: &StatisticalAnalysisResult) -> StatisticalDiagnosis {
    let mut diagnosis = StatisticalDiagnosis::new();
    
    // Check sample size adequacy
    let required_sample_size = calculate_required_sample_size(
        results.effect_size,
        results.statistical_power,
        results.significance_level
    );
    
    if results.sample_size < required_sample_size {
        diagnosis.issues.push(StatisticalIssue::InsufficientSampleSize {
            current: results.sample_size,
            required: required_sample_size,
        });
    }
    
    // Check effect size
    if results.effect_size < 0.2 {
        diagnosis.issues.push(StatisticalIssue::SmallEffectSize(results.effect_size));
    }
    
    // Check for multiple testing issues
    if results.number_of_tests > 1 {
        let bonferroni_threshold = results.significance_level / results.number_of_tests as f64;
        if results.p_value > bonferroni_threshold {
            diagnosis.issues.push(StatisticalIssue::MultipleTestingCorrection {
                original_threshold: results.significance_level,
                corrected_threshold: bonferroni_threshold,
            });
        }
    }
    
    diagnosis
}

fn calculate_required_sample_size(effect_size: f64, power: f64, alpha: f64) -> usize {
    // Simplified power analysis calculation
    let z_alpha = 1.96; // For alpha = 0.05
    let z_beta = 0.84;  // For power = 0.8
    
    let n = 2.0 * ((z_alpha + z_beta) / effect_size).powi(2);
    n.ceil() as usize
}
```

**Solutions:**

1. **Increase Sample Size**:
   ```rust
   fn collect_more_data(current_data: &[TrainingSample], target_size: usize) -> Vec<TrainingSample> {
       let mut extended_data = current_data.to_vec();
       
       // Generate additional synthetic data if needed
       if extended_data.len() < target_size {
           let synthetic_generator = SyntheticDataGenerator::new();
           let additional_samples = target_size - extended_data.len();
           let synthetic_data = synthetic_generator.generate_samples(additional_samples);
           extended_data.extend(synthetic_data);
       }
       
       extended_data
   }
   ```

2. **Use Bootstrap Methods for Confidence Intervals**:
   ```rust
   use rand::seq::SliceRandom;
   
   fn bootstrap_confidence_interval(data: &[f64], n_bootstrap: usize, confidence_level: f64) -> (f64, f64) {
       let mut rng = rand::thread_rng();
       let mut bootstrap_means = Vec::new();
       
       for _ in 0..n_bootstrap {
           let bootstrap_sample: Vec<f64> = (0..data.len())
               .map(|_| *data.choose(&mut rng).unwrap())
               .collect();
           
           let mean = bootstrap_sample.iter().sum::<f64>() / bootstrap_sample.len() as f64;
           bootstrap_means.push(mean);
       }
       
       bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
       
       let alpha = 1.0 - confidence_level;
       let lower_idx = (alpha / 2.0 * n_bootstrap as f64) as usize;
       let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
       
       (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
   }
   ```

3. **Apply Multiple Testing Corrections**:
   ```rust
   fn apply_bonferroni_correction(p_values: &[f64], alpha: f64) -> Vec<bool> {
       let corrected_alpha = alpha / p_values.len() as f64;
       p_values.iter().map(|&p| p < corrected_alpha).collect()
   }
   
   fn apply_benjamini_hochberg_correction(p_values: &[f64], alpha: f64) -> Vec<bool> {
       let mut indexed_p_values: Vec<(usize, f64)> = p_values.iter()
           .enumerate()
           .map(|(i, &p)| (i, p))
           .collect();
       
       indexed_p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
       
       let mut results = vec![false; p_values.len()];
       let m = p_values.len() as f64;
       
       for (rank, (original_idx, p_value)) in indexed_p_values.iter().enumerate() {
           let threshold = (rank as f64 + 1.0) / m * alpha;
           if *p_value <= threshold {
               results[*original_idx] = true;
           }
       }
       
       results
   }
   ```

4. **Implement Bayesian Analysis**:
   ```rust
   use statrs::distribution::{Beta, Continuous};
   
   fn bayesian_hit_rate_analysis(successes: usize, total: usize) -> BayesianResult {
       // Use Beta distribution as conjugate prior for binomial likelihood
       let alpha_prior = 1.0; // Uniform prior
       let beta_prior = 1.0;
       
       // Update with observed data
       let alpha_posterior = alpha_prior + successes as f64;
       let beta_posterior = beta_prior + (total - successes) as f64;
       
       let posterior = Beta::new(alpha_posterior, beta_posterior).unwrap();
       
       BayesianResult {
           posterior_mean: posterior.mean().unwrap(),
           credible_interval_95: (
               posterior.inverse_cdf(0.025),
               posterior.inverse_cdf(0.975)
           ),
           probability_above_threshold: 1.0 - posterior.cdf(0.5), // P(hit_rate > 0.5)
       }
   }
   ```

**Prevention:**
- Plan for adequate sample sizes before testing
- Use power analysis to determine required data
- Consider Bayesian methods for small samples
- Document statistical assumptions and limitations

## Environment-Specific Troubleshooting

### CI/CD Environment Issues

#### Issue 7: Tests Pass Locally but Fail in CI

**Symptoms:**
```
✅ Local: All tests pass (Ubuntu 22.04, 16GB RAM, 8 cores)
❌ CI: Performance tests fail (GitHub Actions, 2 cores, 7GB RAM)
Target: 1.0ms, Local: 0.8ms, CI: 2.3ms
```

**Diagnosis Steps:**
```rust
fn diagnose_ci_environment() -> CIDiagnosis {
    let mut diagnosis = CIDiagnosis::new();
    
    // Check system resources
    diagnosis.cpu_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    diagnosis.available_memory = get_available_memory_mb();
    
    // Check for virtualization overhead
    diagnosis.is_virtualized = detect_virtualization();
    
    // Check CPU performance
    let cpu_benchmark_score = run_cpu_benchmark();
    diagnosis.cpu_performance_ratio = cpu_benchmark_score / REFERENCE_CPU_SCORE;
    
    // Check I/O performance
    let io_benchmark_score = run_io_benchmark();
    diagnosis.io_performance_ratio = io_benchmark_score / REFERENCE_IO_SCORE;
    
    diagnosis
}
```

**Solutions:**

1. **Adjust Performance Targets for CI**:
   ```rust
   fn create_ci_performance_config() -> PerformanceTestConfig {
       let base_config = PerformanceTestConfig::default();
       
       // Scale targets based on CI environment
       let ci_scaling_factor = if is_ci_environment() { 3.0 } else { 1.0 };
       
       PerformanceTestConfig {
           target_latency_1k_samples_ms: base_config.target_latency_1k_samples_ms * ci_scaling_factor,
           target_latency_10k_samples_ms: base_config.target_latency_10k_samples_ms * ci_scaling_factor,
           target_latency_50k_samples_ms: base_config.target_latency_50k_samples_ms * ci_scaling_factor,
           ..base_config
       }
   }
   ```

2. **Use Environment Detection**:
   ```rust
   fn is_ci_environment() -> bool {
       std::env::var("CI").is_ok() || 
       std::env::var("GITHUB_ACTIONS").is_ok() ||
       std::env::var("GITLAB_CI").is_ok()
   }
   
   fn get_environment_config() -> TestConfig {
       if is_ci_environment() {
           TestConfig::from_file("config/ci_test_config.toml").unwrap()
       } else {
           TestConfig::from_file("config/dev_test_config.toml").unwrap()
       }
   }
   ```

3. **Implement Resource-Aware Testing**:
   ```rust
   fn adjust_test_parameters_for_resources() -> TestConfig {
       let available_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
       let available_memory_mb = get_available_memory_mb();
       
       let mut config = TestConfig::default();
       
       // Adjust based on available resources
       if available_cores < 4 {
           config.parallel_execution = false;
           config.performance_targets.scale_by(2.0);
       }
       
       if available_memory_mb < 4096 {
           config.test_data_size = TestDataSize::Small;
           config.max_memory_mb = available_memory_mb / 2;
       }
       
       config
   }
   ```

**Prevention:**
- Test in CI-like environments during development
- Use resource-aware configuration
- Implement proper environment detection
- Document environment requirements

### Memory-Constrained Environments

#### Issue 8: Out of Memory Errors

**Symptoms:**
```
💥 Test failed: large_dataset_performance_test
Error: memory allocation of 2147483648 bytes failed
Available memory: 1.2GB
Requested allocation: 2.0GB for HNSW index
```

**Solutions:**

1. **Implement Memory Monitoring**:
   ```rust
   struct MemoryMonitor {
       max_allowed_mb: usize,
       current_usage_mb: AtomicUsize,
   }
   
   impl MemoryMonitor {
       fn check_allocation(&self, size_bytes: usize) -> Result<()> {
           let size_mb = size_bytes / 1024 / 1024;
           let current = self.current_usage_mb.load(Ordering::Relaxed);
           
           if current + size_mb > self.max_allowed_mb {
               return Err(anyhow::anyhow!(
                   "Memory allocation would exceed limit: {}MB + {}MB > {}MB",
                   current, size_mb, self.max_allowed_mb
               ));
           }
           
           Ok(())
       }
   }
   ```

2. **Use Streaming Processing**:
   ```rust
   fn process_large_dataset_streaming(data: &[TrainingSample], batch_size: usize) -> Result<Vec<f32>> {
       let mut results = Vec::new();
       
       for chunk in data.chunks(batch_size) {
           let chunk_results = process_batch(chunk)?;
           results.extend(chunk_results);
           
           // Force garbage collection between batches
           std::hint::black_box(&chunk_results);
       }
       
       Ok(results)
   }
   ```

3. **Implement Memory-Efficient Algorithms**:
   ```rust
   // Use approximate algorithms for memory-constrained environments
   fn create_memory_efficient_config(available_memory_mb: usize) -> LDCConfig {
       let mut config = LDCConfig::default();
       
       if available_memory_mb < 1024 {
           // Very memory constrained
           config.use_hnsw_index = false; // HNSW uses significant memory
           config.max_bars_back = 500;    // Reduce history
           config.batch_size = 50;        // Smaller batches
       } else if available_memory_mb < 2048 {
           // Moderately constrained
           config.hnsw_config.m = 8;      // Smaller HNSW parameters
           config.max_bars_back = 1000;
           config.batch_size = 100;
       }
       
       config
   }
   ```

## Debugging Tools and Techniques

### Performance Profiling

```rust
use std::time::Instant;
use std::collections::HashMap;

struct PerformanceProfiler {
    timings: HashMap<String, Vec<Duration>>,
    current_operations: HashMap<String, Instant>,
}

impl PerformanceProfiler {
    fn start_operation(&mut self, name: &str) {
        self.current_operations.insert(name.to_string(), Instant::now());
    }
    
    fn end_operation(&mut self, name: &str) {
        if let Some(start_time) = self.current_operations.remove(name) {
            let duration = start_time.elapsed();
            self.timings.entry(name.to_string()).or_default().push(duration);
        }
    }
    
    fn generate_report(&self) -> String {
        let mut report = String::new();
        
        for (operation, durations) in &self.timings {
            let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
            let min_duration = durations.iter().min().unwrap();
            let max_duration = durations.iter().max().unwrap();
            
            report.push_str(&format!(
                "{}: avg={:.2}ms, min={:.2}ms, max={:.2}ms, count={}\n",
                operation,
                avg_duration.as_secs_f64() * 1000.0,
                min_duration.as_secs_f64() * 1000.0,
                max_duration.as_secs_f64() * 1000.0,
                durations.len()
            ));
        }
        
        report
    }
}
```

### Memory Leak Detection

```rust
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref ALLOCATION_TRACKER: Mutex<HashMap<usize, AllocationInfo>> = Mutex::new(HashMap::new());
}

struct AllocationInfo {
    size: usize,
    location: String,
    timestamp: std::time::Instant,
}

fn track_allocation(ptr: usize, size: usize, location: &str) {
    let mut tracker = ALLOCATION_TRACKER.lock().unwrap();
    tracker.insert(ptr, AllocationInfo {
        size,
        location: location.to_string(),
        timestamp: std::time::Instant::now(),
    });
}

fn track_deallocation(ptr: usize) {
    let mut tracker = ALLOCATION_TRACKER.lock().unwrap();
    tracker.remove(&ptr);
}

fn generate_leak_report() -> String {
    let tracker = ALLOCATION_TRACKER.lock().unwrap();
    let mut report = String::new();
    
    let total_leaked = tracker.values().map(|info| info.size).sum::<usize>();
    report.push_str(&format!("Total leaked memory: {} bytes\n", total_leaked));
    
    for (ptr, info) in tracker.iter() {
        report.push_str(&format!(
            "Leak: {} bytes at 0x{:x} from {} (age: {:.2}s)\n",
            info.size,
            ptr,
            info.location,
            info.timestamp.elapsed().as_secs_f64()
        ));
    }
    
    report
}
```

### Automated Issue Detection

```rust
struct AutomatedDiagnostics {
    performance_analyzer: PerformanceAnalyzer,
    memory_analyzer: MemoryAnalyzer,
    statistical_analyzer: StatisticalAnalyzer,
}

impl AutomatedDiagnostics {
    fn analyze_test_results(&self, results: &TestResults) -> DiagnosticReport {
        let mut report = DiagnosticReport::new();
        
        // Analyze performance issues
        if let Some(perf_issues) = self.performance_analyzer.detect_issues(&results.performance) {
            report.add_issues(perf_issues);
        }
        
        // Analyze memory issues
        if let Some(memory_issues) = self.memory_analyzer.detect_issues(&results.memory_usage) {
            report.add_issues(memory_issues);
        }
        
        // Analyze statistical issues
        if let Some(stat_issues) = self.statistical_analyzer.detect_issues(&results.statistical) {
            report.add_issues(stat_issues);
        }
        
        // Generate recommendations
        report.recommendations = self.generate_recommendations(&report.issues);
        
        report
    }
    
    fn generate_recommendations(&self, issues: &[Issue]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();
        
        for issue in issues {
            match issue.category {
                IssueCategory::Performance => {
                    recommendations.extend(self.generate_performance_recommendations(issue));
                },
                IssueCategory::Memory => {
                    recommendations.extend(self.generate_memory_recommendations(issue));
                },
                IssueCategory::Statistical => {
                    recommendations.extend(self.generate_statistical_recommendations(issue));
                },
            }
        }
        
        recommendations
    }
}
```

## Best Practices for Troubleshooting

### 1. Systematic Approach
- Start with the most likely causes
- Use diagnostic tools to gather data
- Test one change at a time
- Document findings and solutions

### 2. Environment Awareness
- Understand the testing environment constraints
- Use appropriate configurations for different environments
- Test in production-like conditions when possible
- Account for resource limitations

### 3. Proactive Monitoring
- Implement comprehensive logging
- Use performance monitoring tools
- Set up alerts for critical issues
- Track trends over time

### 4. Documentation and Knowledge Sharing
- Document common issues and solutions
- Create runbooks for troubleshooting procedures
- Share knowledge across the team
- Maintain up-to-date troubleshooting guides

This comprehensive troubleshooting guide provides practical solutions for the most common issues encountered in the LDC engine testing framework, enabling quick resolution and prevention of problems.