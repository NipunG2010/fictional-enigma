# Performance Tuning Guide

## Overview

This guide provides comprehensive instructions for optimizing the performance of the LDC engine testing framework, including system-level optimizations, algorithm tuning, and environment-specific configurations.

## Performance Analysis and Profiling

### System Performance Baseline

Before tuning, establish a performance baseline to measure improvements against:

```rust
use std::time::{Duration, Instant};
use sysinfo::{System, SystemExt, CpuExt};

struct SystemBaseline {
    cpu_info: CpuInfo,
    memory_info: MemoryInfo,
    storage_info: StorageInfo,
    baseline_scores: BaselineScores,
}

#[derive(Debug)]
struct BaselineScores {
    cpu_score: f64,
    memory_bandwidth_score: f64,
    storage_iops_score: f64,
    network_latency_score: f64,
}

fn establish_performance_baseline() -> SystemBaseline {
    let mut system = System::new_all();
    system.refresh_all();
    
    let cpu_info = CpuInfo {
        cores: system.cpus().len(),
        frequency_mhz: system.cpus()[0].frequency(),
        architecture: std::env::consts::ARCH.to_string(),
        features: get_cpu_features(),
    };
    
    let memory_info = MemoryInfo {
        total_mb: system.total_memory() / 1024 / 1024,
        available_mb: system.available_memory() / 1024 / 1024,
        swap_mb: system.total_swap() / 1024 / 1024,
    };
    
    let baseline_scores = run_baseline_benchmarks();
    
    SystemBaseline {
        cpu_info,
        memory_info,
        storage_info: get_storage_info(),
        baseline_scores,
    }
}

fn run_baseline_benchmarks() -> BaselineScores {
    BaselineScores {
        cpu_score: benchmark_cpu_performance(),
        memory_bandwidth_score: benchmark_memory_bandwidth(),
        storage_iops_score: benchmark_storage_performance(),
        network_latency_score: benchmark_network_latency(),
    }
}

fn benchmark_cpu_performance() -> f64 {
    let iterations = 1_000_000;
    let start = Instant::now();
    
    // CPU-intensive calculation
    let mut result = 0.0f64;
    for i in 0..iterations {
        result += (i as f64).sin().cos().tan();
    }
    
    let duration = start.elapsed();
    std::hint::black_box(result); // Prevent optimization
    
    // Return operations per second
    iterations as f64 / duration.as_secs_f64()
}

fn benchmark_memory_bandwidth() -> f64 {
    let size = 100_000_000; // 100MB
    let data: Vec<u64> = (0..size).collect();
    let start = Instant::now();
    
    // Memory bandwidth test
    let sum: u64 = data.iter().sum();
    
    let duration = start.elapsed();
    std::hint::black_box(sum);
    
    // Return MB/s
    (size * 8) as f64 / duration.as_secs_f64() / 1_000_000.0
}
```

### Performance Profiling Tools

#### Built-in Profiler

```rust
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct PerformanceProfiler {
    timings: HashMap<String, Vec<Duration>>,
    memory_usage: HashMap<String, usize>,
    call_counts: HashMap<String, usize>,
    active_operations: HashMap<String, Instant>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            memory_usage: HashMap::new(),
            call_counts: HashMap::new(),
            active_operations: HashMap::new(),
        }
    }
    
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
    
    pub fn record_memory_usage(&mut self, operation: &str, bytes: usize) {
        self.memory_usage.insert(operation.to_string(), bytes);
    }
    
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let mut operations = Vec::new();
        
        for (name, durations) in &self.timings {
            let total_time: Duration = durations.iter().sum();
            let avg_time = total_time / durations.len() as u32;
            let min_time = *durations.iter().min().unwrap();
            let max_time = *durations.iter().max().unwrap();
            
            operations.push(OperationProfile {
                name: name.clone(),
                call_count: *self.call_counts.get(name).unwrap_or(&0),
                total_time,
                avg_time,
                min_time,
                max_time,
                memory_usage: self.memory_usage.get(name).copied().unwrap_or(0),
            });
        }
        
        // Sort by total time (highest impact first)
        operations.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        
        PerformanceReport { operations }
    }
}

#[derive(Debug)]
pub struct OperationProfile {
    pub name: String,
    pub call_count: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub memory_usage: usize,
}

#[derive(Debug)]
pub struct PerformanceReport {
    pub operations: Vec<OperationProfile>,
}

impl PerformanceReport {
    pub fn identify_bottlenecks(&self) -> Vec<&OperationProfile> {
        // Identify operations that take >10% of total time
        let total_time: Duration = self.operations.iter().map(|op| op.total_time).sum();
        let threshold = total_time / 10;
        
        self.operations.iter()
            .filter(|op| op.total_time > threshold)
            .collect()
    }
    
    pub fn print_summary(&self) {
        println!("Performance Profile Summary");
        println!("==========================");
        
        for op in &self.operations {
            println!(
                "{}: {} calls, {:.2}ms avg, {:.2}ms total, {} bytes memory",
                op.name,
                op.call_count,
                op.avg_time.as_secs_f64() * 1000.0,
                op.total_time.as_secs_f64() * 1000.0,
                op.memory_usage
            );
        }
        
        println!("\nBottlenecks:");
        for bottleneck in self.identify_bottlenecks() {
            println!("  - {}: {:.2}ms total", bottleneck.name, bottleneck.total_time.as_secs_f64() * 1000.0);
        }
    }
}
```

#### Integration with External Profilers

```rust
// Integration with perf (Linux)
#[cfg(target_os = "linux")]
pub fn run_with_perf_profiling<F, R>(operation: F) -> R 
where F: FnOnce() -> R {
    use std::process::Command;
    
    // Start perf recording
    let mut perf_process = Command::new("perf")
        .args(&["record", "-g", "--call-graph=dwarf", "-p", &std::process::id().to_string()])
        .spawn()
        .expect("Failed to start perf");
    
    // Run the operation
    let result = operation();
    
    // Stop perf recording
    perf_process.kill().expect("Failed to stop perf");
    
    result
}

// Integration with Instruments (macOS)
#[cfg(target_os = "macos")]
pub fn run_with_instruments_profiling<F, R>(operation: F) -> R 
where F: FnOnce() -> R {
    // Implementation for macOS Instruments integration
    operation()
}
```

## Algorithm-Specific Optimizations

### Distance Calculation Optimizations

#### SIMD Optimizations

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub fn lorentzian_distance_avx2(features1: &[f32], features2: &[f32]) -> f32 {
    assert_eq!(features1.len(), features2.len());
    assert!(features1.len() % 8 == 0, "Feature length must be multiple of 8 for AVX2");
    
    unsafe {
        let mut sum = _mm256_setzero_ps();
        
        for chunk in 0..(features1.len() / 8) {
            let offset = chunk * 8;
            
            // Load 8 f32 values at once
            let a = _mm256_loadu_ps(features1.as_ptr().add(offset));
            let b = _mm256_loadu_ps(features2.as_ptr().add(offset));
            
            // Calculate difference
            let diff = _mm256_sub_ps(a, b);
            
            // Square the difference
            let diff_squared = _mm256_mul_ps(diff, diff);
            
            // Add 1.0
            let ones = _mm256_set1_ps(1.0);
            let term = _mm256_add_ps(ones, diff_squared);
            
            // Natural logarithm (approximation for performance)
            let log_term = fast_log_avx2(term);
            
            // Accumulate
            sum = _mm256_add_ps(sum, log_term);
        }
        
        // Horizontal sum of the 8 values
        horizontal_sum_avx2(sum)
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn fast_log_avx2(x: __m256) -> __m256 {
    // Fast logarithm approximation using polynomial
    // log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 for x near 1
    let ones = _mm256_set1_ps(1.0);
    let x_minus_1 = _mm256_sub_ps(x, ones);
    
    let half = _mm256_set1_ps(0.5);
    let third = _mm256_set1_ps(1.0/3.0);
    
    let x2 = _mm256_mul_ps(x_minus_1, x_minus_1);
    let x3 = _mm256_mul_ps(x2, x_minus_1);
    
    let term2 = _mm256_mul_ps(x2, half);
    let term3 = _mm256_mul_ps(x3, third);
    
    let result = _mm256_sub_ps(x_minus_1, term2);
    _mm256_add_ps(result, term3)
}

#[cfg(target_arch = "x86_64")]
unsafe fn horizontal_sum_avx2(x: __m256) -> f32 {
    let high = _mm256_extractf128_ps(x, 1);
    let low = _mm256_castps256_ps128(x);
    let sum128 = _mm_add_ps(high, low);
    
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    
    _mm_cvtss_f32(sum32)
}

// Fallback for non-AVX2 systems
pub fn lorentzian_distance_optimized(features1: &[f32], features2: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && features1.len() % 8 == 0 {
            return lorentzian_distance_avx2(features1, features2);
        }
    }
    
    // Standard implementation with compiler optimizations
    features1.iter()
        .zip(features2.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            (1.0 + diff * diff).ln()
        })
        .sum()
}
```

#### Memory Layout Optimizations

```rust
// Structure of Arrays (SoA) layout for better cache performance
#[derive(Clone)]
pub struct SoAFeatures {
    f1: Vec<f32>,
    f2: Vec<f32>,
    f3: Vec<f32>,
    f4: Vec<f32>,
    f5: Vec<f32>,
}

impl SoAFeatures {
    pub fn from_aos(aos_features: &[FeatureSeries]) -> Self {
        let capacity = aos_features.len();
        let mut soa = SoAFeatures {
            f1: Vec::with_capacity(capacity),
            f2: Vec::with_capacity(capacity),
            f3: Vec::with_capacity(capacity),
            f4: Vec::with_capacity(capacity),
            f5: Vec::with_capacity(capacity),
        };
        
        for features in aos_features {
            soa.f1.push(features.f1);
            soa.f2.push(features.f2);
            soa.f3.push(features.f3);
            soa.f4.push(features.f4);
            soa.f5.push(features.f5);
        }
        
        soa
    }
    
    pub fn distance_to_index(&self, index: usize, query: &FeatureSeries) -> f32 {
        let diff1 = self.f1[index] - query.f1;
        let diff2 = self.f2[index] - query.f2;
        let diff3 = self.f3[index] - query.f3;
        let diff4 = self.f4[index] - query.f4;
        let diff5 = self.f5[index] - query.f5;
        
        (1.0 + diff1 * diff1).ln() +
        (1.0 + diff2 * diff2).ln() +
        (1.0 + diff3 * diff3).ln() +
        (1.0 + diff4 * diff4).ln() +
        (1.0 + diff5 * diff5).ln()
    }
    
    // Vectorized distance calculation for multiple queries
    pub fn batch_distances(&self, queries: &[FeatureSeries]) -> Vec<Vec<f32>> {
        queries.par_iter()
            .map(|query| {
                (0..self.len())
                    .map(|i| self.distance_to_index(i, query))
                    .collect()
            })
            .collect()
    }
    
    pub fn len(&self) -> usize {
        self.f1.len()
    }
}

// Cache-friendly data alignment
#[repr(C, align(64))] // Align to cache line size
pub struct AlignedFeatures {
    pub data: [f32; 5],
    _padding: [u8; 44], // Pad to 64 bytes
}

impl AlignedFeatures {
    pub fn new(f1: f32, f2: f32, f3: f32, f4: f32, f5: f32) -> Self {
        Self {
            data: [f1, f2, f3, f4, f5],
            _padding: [0; 44],
        }
    }
}
```

### HNSW Index Optimizations

#### Parameter Tuning

```rust
pub struct HNSWTuner {
    dataset_characteristics: DatasetCharacteristics,
    performance_requirements: PerformanceRequirements,
}

#[derive(Debug)]
pub struct DatasetCharacteristics {
    pub size: usize,
    pub dimensionality: usize,
    pub data_distribution: DataDistribution,
    pub query_pattern: QueryPattern,
}

#[derive(Debug)]
pub enum DataDistribution {
    Uniform,
    Clustered,
    Skewed,
}

#[derive(Debug)]
pub enum QueryPattern {
    Random,
    Temporal,
    Clustered,
}

#[derive(Debug)]
pub struct PerformanceRequirements {
    pub target_accuracy: f64,
    pub max_latency_ms: f64,
    pub max_memory_mb: usize,
}

impl HNSWTuner {
    pub fn tune_parameters(&self) -> HNSWConfig {
        let base_config = self.get_base_config();
        let mut config = base_config;
        
        // Tune M parameter based on dataset size and accuracy requirements
        config.m = self.tune_m_parameter();
        
        // Tune ef_construction based on accuracy vs build time tradeoff
        config.ef_construction = self.tune_ef_construction();
        
        // Tune ef_search based on accuracy vs query time tradeoff
        config.ef_search = self.tune_ef_search();
        
        // Validate configuration meets requirements
        if !self.validate_config(&config) {
            config = self.fallback_config();
        }
        
        config
    }
    
    fn tune_m_parameter(&self) -> usize {
        match self.dataset_characteristics.size {
            size if size < 10_000 => {
                // Small datasets: higher M for better accuracy
                match self.performance_requirements.target_accuracy {
                    acc if acc >= 0.98 => 32,
                    acc if acc >= 0.95 => 24,
                    _ => 16,
                }
            },
            size if size < 100_000 => {
                // Medium datasets: balanced approach
                match self.performance_requirements.target_accuracy {
                    acc if acc >= 0.98 => 24,
                    acc if acc >= 0.95 => 16,
                    _ => 12,
                }
            },
            _ => {
                // Large datasets: lower M for memory efficiency
                match self.performance_requirements.target_accuracy {
                    acc if acc >= 0.98 => 16,
                    acc if acc >= 0.95 => 12,
                    _ => 8,
                }
            }
        }
    }
    
    fn tune_ef_construction(&self) -> usize {
        let base_ef = match self.dataset_characteristics.data_distribution {
            DataDistribution::Uniform => 200,
            DataDistribution::Clustered => 300,
            DataDistribution::Skewed => 400,
        };
        
        // Adjust based on accuracy requirements
        let accuracy_multiplier = match self.performance_requirements.target_accuracy {
            acc if acc >= 0.98 => 2.0,
            acc if acc >= 0.95 => 1.5,
            acc if acc >= 0.90 => 1.0,
            _ => 0.8,
        };
        
        (base_ef as f64 * accuracy_multiplier) as usize
    }
    
    fn tune_ef_search(&self) -> usize {
        let base_ef = match self.dataset_characteristics.query_pattern {
            QueryPattern::Random => 50,
            QueryPattern::Temporal => 75,
            QueryPattern::Clustered => 100,
        };
        
        // Adjust based on latency requirements
        let latency_factor = if self.performance_requirements.max_latency_ms < 1.0 {
            0.5 // Aggressive latency requirements
        } else if self.performance_requirements.max_latency_ms < 5.0 {
            1.0 // Standard latency requirements
        } else {
            2.0 // Relaxed latency requirements
        };
        
        // Adjust based on accuracy requirements
        let accuracy_factor = match self.performance_requirements.target_accuracy {
            acc if acc >= 0.98 => 3.0,
            acc if acc >= 0.95 => 2.0,
            acc if acc >= 0.90 => 1.5,
            _ => 1.0,
        };
        
        (base_ef as f64 * latency_factor * accuracy_factor) as usize
    }
    
    fn validate_config(&self, config: &HNSWConfig) -> bool {
        // Estimate memory usage
        let estimated_memory_mb = self.estimate_memory_usage(config);
        if estimated_memory_mb > self.performance_requirements.max_memory_mb {
            return false;
        }
        
        // Estimate query latency
        let estimated_latency_ms = self.estimate_query_latency(config);
        if estimated_latency_ms > self.performance_requirements.max_latency_ms {
            return false;
        }
        
        true
    }
    
    fn estimate_memory_usage(&self, config: &HNSWConfig) -> usize {
        let n = self.dataset_characteristics.size;
        let d = self.dataset_characteristics.dimensionality;
        let m = config.m;
        
        // Rough estimation: each node stores d features + m connections per layer
        let avg_layers = 1.0 / (2.0_f64).ln(); // Expected number of layers
        let memory_per_node = d * 4 + (m as f64 * avg_layers) as usize * 8; // 4 bytes per f32, 8 bytes per connection
        let total_memory_bytes = n * memory_per_node;
        
        total_memory_bytes / 1024 / 1024 // Convert to MB
    }
    
    fn estimate_query_latency(&self, config: &HNSWConfig) -> f64 {
        let n = self.dataset_characteristics.size;
        let ef_search = config.ef_search;
        
        // Rough estimation based on empirical observations
        let base_latency_us = 10.0; // Base latency in microseconds
        let search_factor = (ef_search as f64).ln() * (n as f64).ln();
        
        (base_latency_us * search_factor) / 1000.0 // Convert to milliseconds
    }
    
    fn get_base_config(&self) -> HNSWConfig {
        HNSWConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_m: 16,
            max_m0: 32,
        }
    }
    
    fn fallback_config(&self) -> HNSWConfig {
        // Conservative configuration that should work in most cases
        HNSWConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 32,
            max_m: 8,
            max_m0: 16,
        }
    }
}

// Adaptive HNSW configuration
pub struct AdaptiveHNSWConfig {
    config: HNSWConfig,
    performance_history: Vec<PerformanceMetric>,
    adaptation_enabled: bool,
}

impl AdaptiveHNSWConfig {
    pub fn new(initial_config: HNSWConfig) -> Self {
        Self {
            config: initial_config,
            performance_history: Vec::new(),
            adaptation_enabled: true,
        }
    }
    
    pub fn record_performance(&mut self, metric: PerformanceMetric) {
        self.performance_history.push(metric);
        
        // Keep only recent history
        if self.performance_history.len() > 100 {
            self.performance_history.drain(0..50);
        }
        
        if self.adaptation_enabled && self.should_adapt() {
            self.adapt_configuration();
        }
    }
    
    fn should_adapt(&self) -> bool {
        if self.performance_history.len() < 10 {
            return false;
        }
        
        let recent_metrics = &self.performance_history[self.performance_history.len()-10..];
        let avg_accuracy = recent_metrics.iter().map(|m| m.accuracy).sum::<f64>() / 10.0;
        let avg_latency = recent_metrics.iter().map(|m| m.latency_ms).sum::<f64>() / 10.0;
        
        // Adapt if accuracy is consistently low or latency is consistently high
        avg_accuracy < 0.90 || avg_latency > 5.0
    }
    
    fn adapt_configuration(&mut self) {
        let recent_metrics = &self.performance_history[self.performance_history.len()-10..];
        let avg_accuracy = recent_metrics.iter().map(|m| m.accuracy).sum::<f64>() / 10.0;
        let avg_latency = recent_metrics.iter().map(|m| m.latency_ms).sum::<f64>() / 10.0;
        
        if avg_accuracy < 0.90 {
            // Increase accuracy at the cost of latency
            self.config.ef_search = (self.config.ef_search as f64 * 1.2) as usize;
        } else if avg_latency > 5.0 && avg_accuracy > 0.95 {
            // Decrease latency at the cost of some accuracy
            self.config.ef_search = (self.config.ef_search as f64 * 0.8) as usize;
        }
        
        // Ensure reasonable bounds
        self.config.ef_search = self.config.ef_search.max(16).min(500);
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub accuracy: f64,
    pub latency_ms: f64,
    pub timestamp: std::time::Instant,
}
```

### Parallel Processing Optimizations

#### Thread Pool Configuration

```rust
use rayon::prelude::*;
use std::sync::Arc;
use std::thread;

pub struct OptimizedThreadPool {
    pool: rayon::ThreadPool,
    config: ThreadPoolConfig,
}

#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    pub num_threads: usize,
    pub stack_size: usize,
    pub thread_affinity: bool,
    pub numa_aware: bool,
}

impl OptimizedThreadPool {
    pub fn new(config: ThreadPoolConfig) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .stack_size(config.stack_size)
            .thread_name(|index| format!("ldc-worker-{}", index))
            .build()?;
        
        Ok(Self { pool, config })
    }
    
    pub fn optimal_config() -> ThreadPoolConfig {
        let num_cores = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        
        ThreadPoolConfig {
            num_threads: Self::optimal_thread_count(num_cores),
            stack_size: 8 * 1024 * 1024, // 8MB stack
            thread_affinity: num_cores >= 8,
            numa_aware: num_cores >= 16,
        }
    }
    
    fn optimal_thread_count(num_cores: usize) -> usize {
        match num_cores {
            1 => 1,
            2..=4 => num_cores,
            5..=8 => num_cores - 1, // Leave one core for OS
            9..=16 => num_cores - 2,
            _ => num_cores - 4, // Leave more cores for very large systems
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
    
    pub fn parallel_batch_processing<T, R, F>(&self,
        data: &[T],
        batch_size: usize,
        processor: F
    ) -> Vec<R>
    where 
        T: Sync,
        R: Send,
        F: Fn(&[T]) -> Vec<R> + Sync + Send
    {
        self.pool.install(|| {
            data.par_chunks(batch_size)
                .flat_map(|chunk| processor(chunk))
                .collect()
        })
    }
}

// Work-stealing queue for load balancing
pub struct WorkStealingQueue<T> {
    queues: Vec<crossbeam::deque::Worker<T>>,
    stealers: Vec<crossbeam::deque::Stealer<T>>,
    current_worker: std::sync::atomic::AtomicUsize,
}

impl<T> WorkStealingQueue<T> {
    pub fn new(num_workers: usize) -> Self {
        let mut queues = Vec::new();
        let mut stealers = Vec::new();
        
        for _ in 0..num_workers {
            let (worker, stealer) = crossbeam::deque::deque();
            queues.push(worker);
            stealers.push(stealer);
        }
        
        Self {
            queues,
            stealers,
            current_worker: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    pub fn push(&self, item: T) {
        let worker_id = self.current_worker.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.queues.len();
        self.queues[worker_id].push(item);
    }
    
    pub fn steal(&self, worker_id: usize) -> Option<T> {
        // Try to pop from own queue first
        if let Some(item) = self.queues[worker_id].pop() {
            return Some(item);
        }
        
        // Try to steal from other queues
        for (i, stealer) in self.stealers.iter().enumerate() {
            if i != worker_id {
                if let crossbeam::deque::Steal::Success(item) = stealer.steal() {
                    return Some(item);
                }
            }
        }
        
        None
    }
}
```

## Memory Optimizations

### Memory Pool Management

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::NonNull;
use std::sync::Mutex;

pub struct MemoryPool {
    pools: Vec<Mutex<Vec<NonNull<u8>>>>,
    sizes: Vec<usize>,
}

impl MemoryPool {
    pub fn new() -> Self {
        let sizes = vec![
            64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
        ];
        
        let pools = sizes.iter()
            .map(|_| Mutex::new(Vec::new()))
            .collect();
        
        Self { pools, sizes }
    }
    
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        // Find the appropriate pool
        let pool_index = self.sizes.iter()
            .position(|&pool_size| pool_size >= size)?;
        
        let mut pool = self.pools[pool_index].lock().unwrap();
        
        if let Some(ptr) = pool.pop() {
            Some(ptr)
        } else {
            // Allocate new memory
            let layout = Layout::from_size_align(self.sizes[pool_index], 8).ok()?;
            let ptr = unsafe { System.alloc(layout) };
            NonNull::new(ptr)
        }
    }
    
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        if let Some(pool_index) = self.sizes.iter().position(|&pool_size| pool_size >= size) {
            let mut pool = self.pools[pool_index].lock().unwrap();
            
            // Return to pool if not too many objects
            if pool.len() < 1000 {
                pool.push(ptr);
                return;
            }
        }
        
        // Deallocate if pool is full or no appropriate pool
        let layout = Layout::from_size_align(size, 8).unwrap();
        unsafe { System.dealloc(ptr.as_ptr(), layout) };
    }
}

// Custom allocator for LDC engine
pub struct LDCAllocator {
    pool: MemoryPool,
    stats: Mutex<AllocationStats>,
}

#[derive(Debug, Default)]
struct AllocationStats {
    total_allocated: usize,
    total_deallocated: usize,
    peak_usage: usize,
    current_usage: usize,
}

impl LDCAllocator {
    pub fn new() -> Self {
        Self {
            pool: MemoryPool::new(),
            stats: Mutex::new(AllocationStats::default()),
        }
    }
    
    pub fn get_stats(&self) -> AllocationStats {
        self.stats.lock().unwrap().clone()
    }
}

unsafe impl GlobalAlloc for LDCAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = if layout.size() <= 32768 {
            // Use pool for small allocations
            self.pool.allocate(layout.size())
                .map(|p| p.as_ptr())
                .unwrap_or_else(|| System.alloc(layout))
        } else {
            // Use system allocator for large allocations
            System.alloc(layout)
        };
        
        if !ptr.is_null() {
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocated += layout.size();
            stats.current_usage += layout.size();
            stats.peak_usage = stats.peak_usage.max(stats.current_usage);
        }
        
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() <= 32768 {
            if let Some(non_null_ptr) = NonNull::new(ptr) {
                self.pool.deallocate(non_null_ptr, layout.size());
            }
        } else {
            System.dealloc(ptr, layout);
        }
        
        let mut stats = self.stats.lock().unwrap();
        stats.total_deallocated += layout.size();
        stats.current_usage -= layout.size();
    }
}
```

### Cache-Friendly Data Structures

```rust
// Cache-line aligned data structures
#[repr(C, align(64))]
pub struct CacheAlignedFeatures {
    pub features: [f32; 5],
    pub metadata: FeatureMetadata,
    _padding: [u8; 32],
}

#[derive(Clone, Copy)]
pub struct FeatureMetadata {
    pub timestamp: i64,
    pub bar_index: u32,
    pub label: Direction,
}

// Blocked data layout for better cache utilization
pub struct BlockedFeatureMatrix {
    blocks: Vec<FeatureBlock>,
    block_size: usize,
    total_features: usize,
}

#[repr(C, align(64))]
struct FeatureBlock {
    features: [[f32; 5]; 16], // 16 features per block
    metadata: [FeatureMetadata; 16],
}

impl BlockedFeatureMatrix {
    pub fn new(features: &[FeatureSeries], metadata: &[FeatureMetadata]) -> Self {
        const BLOCK_SIZE: usize = 16;
        let num_blocks = (features.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks);
        
        for chunk_start in (0..features.len()).step_by(BLOCK_SIZE) {
            let chunk_end = (chunk_start + BLOCK_SIZE).min(features.len());
            let chunk_size = chunk_end - chunk_start;
            
            let mut block = FeatureBlock {
                features: [[0.0; 5]; BLOCK_SIZE],
                metadata: [FeatureMetadata {
                    timestamp: 0,
                    bar_index: 0,
                    label: Direction::Neutral,
                }; BLOCK_SIZE],
            };
            
            for (i, feature_idx) in (chunk_start..chunk_end).enumerate() {
                block.features[i] = features[feature_idx].to_array();
                block.metadata[i] = metadata[feature_idx];
            }
            
            blocks.push(block);
        }
        
        Self {
            blocks,
            block_size: BLOCK_SIZE,
            total_features: features.len(),
        }
    }
    
    pub fn distance_to_query(&self, query: &FeatureSeries) -> Vec<f32> {
        let mut distances = Vec::with_capacity(self.total_features);
        let query_array = query.to_array();
        
        for block in &self.blocks {
            for i in 0..self.block_size {
                if distances.len() >= self.total_features {
                    break;
                }
                
                let distance = lorentzian_distance_array(&block.features[i], &query_array);
                distances.push(distance);
            }
        }
        
        distances
    }
}

fn lorentzian_distance_array(features1: &[f32; 5], features2: &[f32; 5]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..5 {
        let diff = features1[i] - features2[i];
        sum += (1.0 + diff * diff).ln();
    }
    sum
}
```

## System-Level Optimizations

### CPU Affinity and NUMA Awareness

```rust
use std::thread;

#[cfg(target_os = "linux")]
mod linux_affinity {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};
    use std::mem;
    
    pub fn set_thread_affinity(cpu_id: usize) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let mut cpu_set: cpu_set_t = mem::zeroed();
            CPU_ZERO(&mut cpu_set);
            CPU_SET(cpu_id, &mut cpu_set);
            
            let result = sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &cpu_set);
            if result != 0 {
                return Err("Failed to set CPU affinity".into());
            }
        }
        
        Ok(())
    }
    
    pub fn get_numa_nodes() -> Vec<Vec<usize>> {
        // Simplified NUMA topology detection
        // In practice, you'd use hwloc or similar library
        let num_cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        
        if num_cores <= 8 {
            // Single NUMA node
            vec![(0..num_cores).collect()]
        } else {
            // Assume two NUMA nodes for simplicity
            let cores_per_node = num_cores / 2;
            vec![
                (0..cores_per_node).collect(),
                (cores_per_node..num_cores).collect(),
            ]
        }
    }
}

pub struct NUMAAwareThreadPool {
    pools: Vec<rayon::ThreadPool>,
    numa_nodes: Vec<Vec<usize>>,
}

impl NUMAAwareThreadPool {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            let numa_nodes = linux_affinity::get_numa_nodes();
            let mut pools = Vec::new();
            
            for (node_id, cpu_ids) in numa_nodes.iter().enumerate() {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(cpu_ids.len())
                    .thread_name(move |index| format!("numa-{}-worker-{}", node_id, index))
                    .spawn_handler(move |thread| {
                        let cpu_id = cpu_ids[thread.index() % cpu_ids.len()];
                        std::thread::spawn(move || {
                            if let Err(e) = linux_affinity::set_thread_affinity(cpu_id) {
                                eprintln!("Failed to set CPU affinity: {}", e);
                            }
                            thread.run()
                        });
                        Ok(())
                    })
                    .build()?;
                
                pools.push(pool);
            }
            
            Ok(Self { pools, numa_nodes })
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for non-Linux systems
            let pool = rayon::ThreadPoolBuilder::new().build()?;
            Ok(Self {
                pools: vec![pool],
                numa_nodes: vec![(0..thread::available_parallelism().map(|n| n.get()).unwrap_or(1)).collect()],
            })
        }
    }
    
    pub fn execute_on_node<F, R>(&self, node_id: usize, work: F) -> R
    where F: FnOnce() -> R + Send, R: Send {
        if node_id < self.pools.len() {
            self.pools[node_id].install(work)
        } else {
            self.pools[0].install(work)
        }
    }
    
    pub fn parallel_work_numa_aware<T, R, F>(&self, data: &[T], work: F) -> Vec<R>
    where 
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync + Send
    {
        use rayon::prelude::*;
        
        // Distribute work across NUMA nodes
        let chunk_size = (data.len() + self.pools.len() - 1) / self.pools.len();
        
        data.par_chunks(chunk_size)
            .enumerate()
            .flat_map(|(chunk_id, chunk)| {
                let node_id = chunk_id % self.pools.len();
                self.execute_on_node(node_id, || {
                    chunk.par_iter().map(&work).collect::<Vec<_>>()
                })
            })
            .collect()
    }
}
```

### I/O Optimizations

```rust
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use memmap2::{Mmap, MmapOptions};

pub struct OptimizedDataLoader {
    use_memory_mapping: bool,
    buffer_size: usize,
    prefetch_enabled: bool,
}

impl OptimizedDataLoader {
    pub fn new(config: DataLoaderConfig) -> Self {
        Self {
            use_memory_mapping: config.use_memory_mapping,
            buffer_size: config.buffer_size,
            prefetch_enabled: config.prefetch_enabled,
        }
    }
    
    pub fn load_ohlcv_data(&self, path: &str) -> Result<Vec<OHLCV>, Box<dyn std::error::Error>> {
        if self.use_memory_mapping {
            self.load_with_mmap(path)
        } else {
            self.load_with_buffered_io(path)
        }
    }
    
    fn load_with_mmap(&self, path: &str) -> Result<Vec<OHLCV>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Advise kernel about access pattern
        #[cfg(target_os = "linux")]
        {
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_SEQUENTIAL | libc::MADV_WILLNEED
                );
            }
        }
        
        // Parse data from memory-mapped file
        self.parse_ohlcv_from_bytes(&mmap)
    }
    
    fn load_with_buffered_io(&self, path: &str) -> Result<Vec<OHLCV>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::with_capacity(self.buffer_size, file);
        
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        
        self.parse_ohlcv_from_bytes(&buffer)
    }
    
    fn parse_ohlcv_from_bytes(&self, data: &[u8]) -> Result<Vec<OHLCV>, Box<dyn std::error::Error>> {
        // Implement efficient parsing logic
        // This is a simplified example
        let content = std::str::from_utf8(data)?;
        let mut ohlcv_data = Vec::new();
        
        for line in content.lines().skip(1) { // Skip header
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 6 {
                let ohlcv = OHLCV {
                    timestamp: fields[0].parse()?,
                    open: fields[1].parse()?,
                    high: fields[2].parse()?,
                    low: fields[3].parse()?,
                    close: fields[4].parse()?,
                    volume: fields[5].parse()?,
                };
                ohlcv_data.push(ohlcv);
            }
        }
        
        Ok(ohlcv_data)
    }
}

#[derive(Debug)]
pub struct DataLoaderConfig {
    pub use_memory_mapping: bool,
    pub buffer_size: usize,
    pub prefetch_enabled: bool,
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

// Asynchronous data loading
pub struct AsyncDataLoader {
    runtime: tokio::runtime::Runtime,
}

impl AsyncDataLoader {
    pub fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2) // Dedicated I/O threads
            .thread_name("async-loader")
            .build()
            .expect("Failed to create async runtime");
        
        Self { runtime }
    }
    
    pub fn load_multiple_files(&self, paths: &[String]) -> Result<Vec<Vec<OHLCV>>, Box<dyn std::error::Error>> {
        self.runtime.block_on(async {
            let tasks: Vec<_> = paths.iter()
                .map(|path| {
                    let path = path.clone();
                    tokio::task::spawn_blocking(move || {
                        let loader = OptimizedDataLoader::new(DataLoaderConfig::default());
                        loader.load_ohlcv_data(&path)
                    })
                })
                .collect();
            
            let mut results = Vec::new();
            for task in tasks {
                let result = task.await??;
                results.push(result);
            }
            
            Ok(results)
        })
    }
}
```

## Configuration Optimization

### Adaptive Configuration

```rust
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub struct AdaptivePerformanceConfig {
    current_config: LDCConfig,
    performance_history: VecDeque<PerformanceSnapshot>,
    adaptation_strategy: AdaptationStrategy,
    last_adaptation: Instant,
    adaptation_cooldown: Duration,
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub accuracy: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub config: LDCConfig,
}

#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    Conservative,  // Small, safe changes
    Aggressive,    // Larger changes for faster optimization
    Balanced,      // Moderate changes
}

impl AdaptivePerformanceConfig {
    pub fn new(initial_config: LDCConfig, strategy: AdaptationStrategy) -> Self {
        Self {
            current_config: initial_config,
            performance_history: VecDeque::with_capacity(100),
            adaptation_strategy: strategy,
            last_adaptation: Instant::now(),
            adaptation_cooldown: Duration::from_secs(60), // 1 minute cooldown
        }
    }
    
    pub fn record_performance(&mut self, snapshot: PerformanceSnapshot) {
        self.performance_history.push_back(snapshot);
        
        // Keep only recent history
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }
        
        // Check if we should adapt
        if self.should_adapt() {
            self.adapt_configuration();
        }
    }
    
    fn should_adapt(&self) -> bool {
        // Don't adapt too frequently
        if self.last_adaptation.elapsed() < self.adaptation_cooldown {
            return false;
        }
        
        // Need sufficient data
        if self.performance_history.len() < 10 {
            return false;
        }
        
        // Check if performance is consistently suboptimal
        let recent_snapshots = self.performance_history.iter()
            .rev()
            .take(5)
            .collect::<Vec<_>>();
        
        let avg_latency = recent_snapshots.iter()
            .map(|s| s.avg_latency_ms)
            .sum::<f64>() / recent_snapshots.len() as f64;
        
        let avg_accuracy = recent_snapshots.iter()
            .map(|s| s.accuracy)
            .sum::<f64>() / recent_snapshots.len() as f64;
        
        // Adapt if latency is high or accuracy is low
        avg_latency > 2.0 || avg_accuracy < 0.90
    }
    
    fn adapt_configuration(&mut self) {
        let recent_performance = self.analyze_recent_performance();
        let new_config = self.generate_optimized_config(&recent_performance);
        
        if self.validate_config_change(&self.current_config, &new_config) {
            self.current_config = new_config;
            self.last_adaptation = Instant::now();
        }
    }
    
    fn analyze_recent_performance(&self) -> PerformanceAnalysis {
        let recent_snapshots = self.performance_history.iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>();
        
        let avg_latency = recent_snapshots.iter()
            .map(|s| s.avg_latency_ms)
            .sum::<f64>() / recent_snapshots.len() as f64;
        
        let avg_accuracy = recent_snapshots.iter()
            .map(|s| s.accuracy)
            .sum::<f64>() / recent_snapshots.len() as f64;
        
        let avg_memory = recent_snapshots.iter()
            .map(|s| s.memory_usage_mb)
            .sum::<f64>() / recent_snapshots.len() as f64;
        
        let avg_cpu = recent_snapshots.iter()
            .map(|s| s.cpu_usage_percent)
            .sum::<f64>() / recent_snapshots.len() as f64;
        
        PerformanceAnalysis {
            avg_latency_ms: avg_latency,
            avg_accuracy: avg_accuracy,
            avg_memory_usage_mb: avg_memory,
            avg_cpu_usage_percent: avg_cpu,
            latency_trend: self.calculate_trend(recent_snapshots.iter().map(|s| s.avg_latency_ms)),
            accuracy_trend: self.calculate_trend(recent_snapshots.iter().map(|s| s.accuracy)),
        }
    }
    
    fn calculate_trend<I>(&self, values: I) -> Trend
    where I: Iterator<Item = f64> {
        let values: Vec<f64> = values.collect();
        if values.len() < 3 {
            return Trend::Stable;
        }
        
        let first_half = &values[0..values.len()/2];
        let second_half = &values[values.len()/2..];
        
        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
        
        let change_percent = (second_avg - first_avg) / first_avg * 100.0;
        
        if change_percent > 5.0 {
            Trend::Increasing
        } else if change_percent < -5.0 {
            Trend::Decreasing
        } else {
            Trend::Stable
        }
    }
    
    fn generate_optimized_config(&self, analysis: &PerformanceAnalysis) -> LDCConfig {
        let mut new_config = self.current_config.clone();
        
        // Adjust based on performance analysis
        if analysis.avg_latency_ms > 2.0 {
            // High latency - optimize for speed
            if new_config.use_hnsw_index {
                // Reduce HNSW search effort
                new_config.hnsw_config.ef_search = (new_config.hnsw_config.ef_search as f64 * 0.8) as usize;
            } else if analysis.avg_memory_usage_mb < 1000.0 {
                // Enable HNSW if memory allows
                new_config.use_hnsw_index = true;
            }
            
            // Reduce batch size for lower latency
            new_config.batch_size = (new_config.batch_size as f64 * 0.8) as usize;
        }
        
        if analysis.avg_accuracy < 0.90 {
            // Low accuracy - optimize for accuracy
            if new_config.use_hnsw_index {
                // Increase HNSW search effort
                new_config.hnsw_config.ef_search = (new_config.hnsw_config.ef_search as f64 * 1.2) as usize;
            }
            
            // Increase neighbor count
            new_config.neighbors_count = (new_config.neighbors_count + 2).min(20);
        }
        
        if analysis.avg_memory_usage_mb > 2000.0 {
            // High memory usage - optimize for memory
            new_config.max_bars_back = (new_config.max_bars_back as f64 * 0.8) as usize;
            new_config.cache_size = (new_config.cache_size as f64 * 0.8) as usize;
        }
        
        // Apply strategy-specific adjustments
        match self.adaptation_strategy {
            AdaptationStrategy::Conservative => {
                // Make smaller changes
                self.apply_conservative_limits(&mut new_config);
            },
            AdaptationStrategy::Aggressive => {
                // Allow larger changes
                self.apply_aggressive_adjustments(&mut new_config);
            },
            AdaptationStrategy::Balanced => {
                // Moderate changes (default behavior)
            },
        }
        
        new_config
    }
    
    fn apply_conservative_limits(&self, config: &mut LDCConfig) {
        // Limit changes to 20% of current values
        let current = &self.current_config;
        
        config.neighbors_count = self.clamp_change(
            current.neighbors_count,
            config.neighbors_count,
            0.2
        );
        
        config.batch_size = self.clamp_change(
            current.batch_size,
            config.batch_size,
            0.2
        );
        
        config.hnsw_config.ef_search = self.clamp_change(
            current.hnsw_config.ef_search,
            config.hnsw_config.ef_search,
            0.2
        );
    }
    
    fn apply_aggressive_adjustments(&self, config: &mut LDCConfig) {
        // Allow up to 50% changes
        // This is the default behavior, so no additional limits
    }
    
    fn clamp_change(&self, original: usize, new: usize, max_change_ratio: f64) -> usize {
        let max_change = (original as f64 * max_change_ratio) as usize;
        let change = if new > original {
            (new - original).min(max_change)
        } else {
            (original - new).min(max_change)
        };
        
        if new > original {
            original + change
        } else {
            original - change
        }
    }
    
    fn validate_config_change(&self, old_config: &LDCConfig, new_config: &LDCConfig) -> bool {
        // Ensure new configuration is reasonable
        new_config.neighbors_count >= 3 &&
        new_config.neighbors_count <= 50 &&
        new_config.batch_size >= 10 &&
        new_config.batch_size <= 10000 &&
        new_config.hnsw_config.ef_search >= 10 &&
        new_config.hnsw_config.ef_search <= 1000
    }
    
    pub fn get_current_config(&self) -> &LDCConfig {
        &self.current_config
    }
}

#[derive(Debug)]
struct PerformanceAnalysis {
    avg_latency_ms: f64,
    avg_accuracy: f64,
    avg_memory_usage_mb: f64,
    avg_cpu_usage_percent: f64,
    latency_trend: Trend,
    accuracy_trend: Trend,
}

#[derive(Debug)]
enum Trend {
    Increasing,
    Decreasing,
    Stable,
}
```

## Best Practices Summary

### 1. Measurement and Monitoring
- Establish performance baselines before optimization
- Use comprehensive profiling tools
- Monitor key metrics continuously
- Track performance trends over time

### 2. Algorithm Optimization
- Use SIMD instructions for vectorizable operations
- Implement cache-friendly data layouts
- Optimize memory access patterns
- Use appropriate algorithms for dataset sizes

### 3. System-Level Optimization
- Configure thread pools appropriately
- Use NUMA-aware scheduling when available
- Optimize I/O operations
- Implement efficient memory management

### 4. Adaptive Configuration
- Monitor performance continuously
- Adjust configuration based on observed performance
- Use appropriate adaptation strategies
- Validate configuration changes

### 5. Environment-Specific Tuning
- Account for different hardware capabilities
- Adjust targets for CI/CD environments
- Use resource-aware configurations
- Test in production-like conditions

This comprehensive performance tuning guide provides the tools and techniques needed to optimize the LDC engine testing framework for maximum performance across different environments and use cases.