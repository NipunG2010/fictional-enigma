# Performance Optimization Design Document

## Overview

The Performance Optimization design enhances the existing LDC engine (rust/ldc-engine/src/lib.rs) with advanced multithreading, efficient data structures, and optional HNSW indexing. The design builds upon the current implementation that already includes basic rayon parallelization, VecDeque ring buffer, and PerformanceMetrics tracking to achieve sub-millisecond query performance for large training datasets while maintaining Pine Script accuracy.

## Architecture

### Current LDC Engine Analysis

The existing implementation provides a solid foundation:

```rust
pub struct LDCEngine {
    training_samples: VecDeque<TrainingSample>,  // Ring buffer (2000 samples default)
    config: LDCConfig,                           // Configuration with basic threading
    last_distance: f32,                          // Pine Script compatibility
    performance_metrics: PerformanceMetrics,    // Basic timing metrics
}
```

**Current Performance Features:**
- Basic rayon parallelization in `find_k_nearest_neighbors_parallel`
- Configurable thread pool with `max_threads` parameter
- Sequential vs parallel decision based on `parallel_threshold`
- Basic performance metrics tracking
- Pine Script compatible Lorentzian distance calculation

### Enhanced Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced LDC Engine                         │
├─────────────────────────────────────────────────────────────────┤
│  Existing Components (Enhanced)          │  New Components      │
├─────────────────────────────────────────┼─────────────────────┤
│  • VecDeque<TrainingSample> (Optimized) │  • HNSW Index       │
│  • LDCConfig (Extended)                  │  • SIMD Operations  │
│  • PerformanceMetrics (Enhanced)        │  • Memory Pool      │
│  • Rayon Threading (Optimized)          │  • Spatial Index    │
└─────────────────────────────────────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Performance Layers                          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: SIMD-Optimized Distance Calculation                  │
│  Layer 2: Enhanced Parallel k-NN Search                       │
│  Layer 3: Optional HNSW Approximate Search                    │
│  Layer 4: Memory-Efficient Data Structures                    │
│  Layer 5: Advanced Performance Monitoring                     │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Enhanced LDCConfig

**Extended Configuration Interface:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDCConfig {
    // Existing fields (unchanged for compatibility)
    pub max_bars_back: usize,
    pub neighbors_count: usize,
    pub feature_count: usize,
    pub use_chronological_spacing: bool,
    pub use_multithreading: bool,
    pub max_threads: Option<usize>,
    pub parallel_threshold: usize,
    pub batch_parallel_threshold: usize,
    
    // New performance optimization fields
    pub use_simd_optimization: bool,
    pub simd_chunk_size: usize,
    pub memory_pool_size: usize,
    pub enable_memory_mapping: bool,
    pub memory_threshold_mb: usize,
    
    // HNSW configuration
    pub use_hnsw_index: bool,
    pub hnsw_m: usize,                    // Number of connections (default: 16)
    pub hnsw_ef_construction: usize,      // Size of dynamic candidate list (default: 200)
    pub hnsw_ef_search: usize,           // Size of search candidate list (default: 50)
    pub hnsw_rebuild_threshold: usize,    // Rebuild index after N new samples
    
    // Advanced threading
    pub thread_pool_strategy: ThreadPoolStrategy,
    pub work_stealing_enabled: bool,
    pub numa_aware_allocation: bool,
    
    // Existing fields (unchanged)
    // ... rest of existing config
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreadPoolStrategy {
    Global,           // Use global rayon thread pool
    Dedicated,        // Create dedicated thread pool for LDC
    Adaptive,         // Switch based on workload
}
```

### 2. Enhanced PerformanceMetrics

**Extended Metrics Interface:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Existing fields (unchanged)
    pub total_predictions: u64,
    pub total_training_samples: u64,
    pub average_prediction_time_ms: f64,
    pub last_prediction_time_ms: f64,
    pub parallel_predictions: u64,
    pub sequential_predictions: u64,
    
    // New detailed timing metrics
    pub distance_calculation_time_ms: f64,
    pub knn_search_time_ms: f64,
    pub data_access_time_ms: f64,
    pub simd_operations_count: u64,
    pub hnsw_queries: u64,
    pub exact_queries: u64,
    
    // Latency percentiles
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub latency_samples: VecDeque<f64>,  // Rolling window for percentile calculation
    
    // Memory metrics
    pub peak_memory_usage_mb: usize,
    pub current_memory_usage_mb: usize,
    pub memory_allocations: u64,
    pub memory_deallocations: u64,
    
    // CPU utilization
    pub cpu_utilization_percent: f32,
    pub thread_efficiency_percent: f32,
    
    // HNSW specific metrics
    pub hnsw_index_size: usize,
    pub hnsw_rebuild_count: u64,
    pub hnsw_accuracy_percent: f32,
}
```

### 3. SIMD-Optimized Distance Calculation

**SIMD Interface for FeatureSeries:**
```rust
impl FeatureSeries {
    /// SIMD-optimized Lorentzian distance calculation
    #[cfg(target_arch = "x86_64")]
    pub fn lorentzian_distance_simd(&self, other: &FeatureSeries) -> f32 {
        use std::arch::x86_64::*;
        
        unsafe {
            // Load feature arrays into SIMD registers
            let features1 = _mm_loadu_ps(&self.to_array()[0]);
            let features2 = _mm_loadu_ps(&other.to_array()[0]);
            
            // Calculate absolute differences
            let diff = _mm_sub_ps(features1, features2);
            let abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0), diff);
            
            // Add 1.0 and calculate natural logarithm
            let one = _mm_set1_ps(1.0);
            let plus_one = _mm_add_ps(abs_diff, one);
            
            // Sum the logarithms (SIMD ln approximation or fallback to scalar)
            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), plus_one);
            
            result.iter().take(4).map(|x| x.ln()).sum::<f32>()
        }
    }
    
    /// Batch SIMD distance calculation for multiple feature vectors
    pub fn batch_lorentzian_distance_simd(
        query: &FeatureSeries,
        targets: &[FeatureSeries],
        chunk_size: usize,
    ) -> Vec<f32> {
        targets
            .chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.iter().map(|target| query.lorentzian_distance_simd(target))
            })
            .collect()
    }
}
```

### 4. HNSW Index Integration

**HNSW Index Interface:**
```rust
use hnsw_rs::prelude::*;

pub struct HNSWIndex {
    index: Hnsw<f32, DistanceFunction>,
    feature_to_sample_map: HashMap<usize, usize>, // HNSW ID -> TrainingSample index
    sample_to_feature_map: HashMap<usize, usize>, // TrainingSample index -> HNSW ID
    next_id: usize,
    config: HNSWConfig,
}

#[derive(Debug, Clone)]
pub struct HNSWConfig {
    pub m: usize,              // Number of connections
    pub ef_construction: usize, // Construction parameter
    pub ef_search: usize,      // Search parameter
    pub max_elements: usize,   // Maximum number of elements
}

impl HNSWIndex {
    pub fn new(config: HNSWConfig) -> Result<Self> {
        let distance_func = DistanceFunction::new(lorentzian_distance_hnsw);
        let index = Hnsw::new(config.m, config.max_elements, 5, config.ef_construction, distance_func);
        
        Ok(Self {
            index,
            feature_to_sample_map: HashMap::new(),
            sample_to_feature_map: HashMap::new(),
            next_id: 0,
            config,
        })
    }
    
    pub fn add_sample(&mut self, sample: &TrainingSample, sample_index: usize) -> Result<()> {
        let features = sample.features.to_array();
        let hnsw_id = self.next_id;
        
        self.index.insert((&features, hnsw_id))?;
        self.feature_to_sample_map.insert(hnsw_id, sample_index);
        self.sample_to_feature_map.insert(sample_index, hnsw_id);
        self.next_id += 1;
        
        Ok(())
    }
    
    pub fn search_knn(&self, query: &FeatureSeries, k: usize) -> Result<Vec<(f32, usize)>> {
        let query_features = query.to_array();
        let results = self.index.search(&query_features, k, self.config.ef_search);
        
        let mut distances_and_indices = Vec::new();
        for (distance, hnsw_id) in results {
            if let Some(&sample_index) = self.feature_to_sample_map.get(&hnsw_id) {
                distances_and_indices.push((distance, sample_index));
            }
        }
        
        Ok(distances_and_indices)
    }
    
    pub fn rebuild(&mut self, samples: &VecDeque<TrainingSample>) -> Result<()> {
        // Clear existing index
        self.index = Hnsw::new(
            self.config.m,
            self.config.max_elements,
            5,
            self.config.ef_construction,
            DistanceFunction::new(lorentzian_distance_hnsw),
        );
        self.feature_to_sample_map.clear();
        self.sample_to_feature_map.clear();
        self.next_id = 0;
        
        // Rebuild with current samples
        for (index, sample) in samples.iter().enumerate() {
            self.add_sample(sample, index)?;
        }
        
        Ok(())
    }
}

// HNSW-compatible distance function
fn lorentzian_distance_hnsw(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (1.0 + (x - y).abs()).ln())
        .sum()
}
```

### 5. Enhanced LDC Engine Methods

**Optimized k-NN Search Interface:**
```rust
impl LDCEngine {
    /// Enhanced k-NN search with multiple optimization strategies
    pub fn find_k_nearest_neighbors_optimized(&self, query_features: &FeatureSeries) -> Vec<(f32, Direction)> {
        let start_time = std::time::Instant::now();
        
        // Choose search strategy based on configuration and data size
        let result = if self.config.use_hnsw_index && self.hnsw_index.is_some() && self.training_samples.len() > 1000 {
            self.find_k_nearest_neighbors_hnsw(query_features)
        } else if self.config.use_multithreading && self.training_samples.len() > self.config.parallel_threshold {
            self.find_k_nearest_neighbors_parallel_optimized(query_features)
        } else {
            self.find_k_nearest_neighbors_sequential_optimized(query_features)
        };
        
        // Update performance metrics
        let duration = start_time.elapsed();
        self.update_performance_metrics(duration, result.len());
        
        result
    }
    
    /// HNSW-based approximate k-NN search
    fn find_k_nearest_neighbors_hnsw(&self, query_features: &FeatureSeries) -> Vec<(f32, Direction)> {
        if let Some(ref hnsw_index) = self.hnsw_index {
            match hnsw_index.search_knn(query_features, self.config.neighbors_count) {
                Ok(results) => {
                    results.into_iter()
                        .filter_map(|(distance, sample_index)| {
                            self.training_samples.get(sample_index)
                                .map(|sample| (distance, sample.label))
                        })
                        .collect()
                }
                Err(_) => {
                    // Fallback to exact search on HNSW error
                    self.find_k_nearest_neighbors_parallel_optimized(query_features)
                }
            }
        } else {
            // Fallback if HNSW not initialized
            self.find_k_nearest_neighbors_parallel_optimized(query_features)
        }
    }
    
    /// Enhanced parallel k-NN search with SIMD optimization
    fn find_k_nearest_neighbors_parallel_optimized(&self, query_features: &FeatureSeries) -> Vec<(f32, Direction)> {
        let training_samples = self.get_training_samples_for_search(None);
        let k = self.config.neighbors_count;
        
        // Configure optimal thread pool
        let thread_pool = self.get_or_create_thread_pool();
        
        thread_pool.install(|| {
            // Use SIMD-optimized distance calculation if enabled
            let distances_and_labels: Vec<(f32, Direction)> = if self.config.use_simd_optimization {
                self.parallel_search_with_simd(query_features, &training_samples)
            } else {
                self.parallel_search_standard(query_features, &training_samples)
            };
            
            // Sort and take k nearest
            let mut sorted_distances = distances_and_labels;
            sorted_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            sorted_distances.truncate(k);
            
            sorted_distances
        })
    }
    
    /// SIMD-optimized parallel search
    fn parallel_search_with_simd(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample]) -> Vec<(f32, Direction)> {
        let chunk_size = self.config.simd_chunk_size;
        
        training_samples
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                // Extract features for SIMD batch processing
                let features: Vec<FeatureSeries> = chunk.iter().map(|s| s.features.clone()).collect();
                let distances = FeatureSeries::batch_lorentzian_distance_simd(query_features, &features, chunk_size);
                
                // Combine with labels and apply chronological spacing
                chunk.iter()
                    .zip(distances.iter())
                    .enumerate()
                    .filter_map(|(i, (sample, &distance))| {
                        if i % 4 == 0 || !self.config.use_chronological_spacing {
                            Some((distance, sample.label))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    
    /// Memory pool for efficient allocation
    fn get_memory_pool(&self) -> &MemoryPool {
        // Implementation for memory pool management
        &self.memory_pool
    }
    
    /// Thread pool management
    fn get_or_create_thread_pool(&self) -> &rayon::ThreadPool {
        match self.config.thread_pool_strategy {
            ThreadPoolStrategy::Global => &rayon::ThreadPoolBuilder::new().build().unwrap(),
            ThreadPoolStrategy::Dedicated => &self.dedicated_thread_pool,
            ThreadPoolStrategy::Adaptive => {
                if self.training_samples.len() > 10000 {
                    &self.dedicated_thread_pool
                } else {
                    &rayon::ThreadPoolBuilder::new().build().unwrap()
                }
            }
        }
    }
}
```

## Data Models

### Enhanced Training Sample Storage

**Memory-Efficient Storage Schema:**
```rust
/// Memory-optimized training sample with data alignment
#[repr(C, align(32))]  // 32-byte alignment for SIMD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedTrainingSample {
    pub features: AlignedFeatureSeries,  // SIMD-aligned features
    pub label: Direction,                // 4 bytes
    pub timestamp: i64,                  // 8 bytes
    pub bar_index: u32,                 // Reduced from usize to u32
    pub _padding: [u8; 12],             // Padding for alignment
}

#[repr(C, align(16))]  // 16-byte alignment for SSE/AVX
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedFeatureSeries {
    pub features: [f32; 8],  // Padded to 8 for better SIMD (original 5 + 3 padding)
}

/// Memory-mapped storage for large datasets
pub struct MemoryMappedStorage {
    mmap: memmap2::Mmap,
    sample_count: usize,
    sample_size: usize,
}

impl MemoryMappedStorage {
    pub fn new(file_path: &Path, max_samples: usize) -> Result<Self> {
        let sample_size = std::mem::size_of::<OptimizedTrainingSample>();
        let file_size = max_samples * sample_size;
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(file_path)?;
        
        file.set_len(file_size as u64)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };
        
        Ok(Self {
            mmap: mmap.make_read_only()?,
            sample_count: 0,
            sample_size,
        })
    }
    
    pub fn get_sample(&self, index: usize) -> Option<&OptimizedTrainingSample> {
        if index < self.sample_count {
            let offset = index * self.sample_size;
            unsafe {
                Some(&*(self.mmap.as_ptr().add(offset) as *const OptimizedTrainingSample))
            }
        } else {
            None
        }
    }
}
```

## Error Handling

### Performance-Aware Error Management

```rust
#[derive(thiserror::Error, Debug)]
pub enum PerformanceOptimizationError {
    #[error("HNSW index error: {0}")]
    HNSWError(String),
    
    #[error("SIMD operation failed: {0}")]
    SIMDError(String),
    
    #[error("Memory allocation failed: requested {requested}MB, available {available}MB")]
    MemoryError { requested: usize, available: usize },
    
    #[error("Thread pool configuration error: {0}")]
    ThreadPoolError(String),
    
    #[error("Performance degradation detected: {component} taking {actual_ms}ms (expected <{expected_ms}ms)")]
    PerformanceDegradation {
        component: String,
        actual_ms: f64,
        expected_ms: f64,
    },
}

/// Performance monitoring with automatic fallback
impl LDCEngine {
    fn monitor_performance<T, F>(&self, operation_name: &str, expected_ms: f64, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        if duration_ms > expected_ms {
            self.log_performance_warning(operation_name, duration_ms, expected_ms);
        }
        
        result
    }
    
    fn log_performance_warning(&self, operation: &str, actual_ms: f64, expected_ms: f64) {
        if self.config.log_performance_metrics {
            eprintln!("⚠️  Performance Warning: {} took {:.2}ms (expected <{:.2}ms)", 
                     operation, actual_ms, expected_ms);
        }
    }
}
```

## Testing Strategy

### Performance Testing Framework

```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn benchmark_knn_search(c: &mut Criterion) {
        let mut engine = create_test_engine_with_samples(10000);
        let query = create_test_feature_series();
        
        c.bench_function("knn_exact_10k", |b| {
            b.iter(|| {
                engine.config.use_hnsw_index = false;
                black_box(engine.find_k_nearest_neighbors_optimized(&query))
            })
        });
        
        c.bench_function("knn_hnsw_10k", |b| {
            b.iter(|| {
                engine.config.use_hnsw_index = true;
                black_box(engine.find_k_nearest_neighbors_optimized(&query))
            })
        });
    }
    
    fn benchmark_simd_distance(c: &mut Criterion) {
        let features1 = create_test_feature_series();
        let features2 = create_test_feature_series();
        
        c.bench_function("distance_standard", |b| {
            b.iter(|| black_box(LDCEngine::lorentzian_distance(&features1, &features2, 5)))
        });
        
        c.bench_function("distance_simd", |b| {
            b.iter(|| black_box(features1.lorentzian_distance_simd(&features2)))
        });
    }
    
    criterion_group!(benches, benchmark_knn_search, benchmark_simd_distance);
    criterion_main!(benches);
}
```

## Implementation Considerations

### Backward Compatibility

1. **Existing API Preservation**: All existing public methods remain unchanged
2. **Configuration Migration**: New LDCConfig fields have sensible defaults
3. **Performance Metrics**: Enhanced PerformanceMetrics maintains existing fields
4. **Pine Script Compatibility**: All optimizations preserve exact Pine Script behavior

### Memory Management

1. **SIMD Alignment**: Use aligned allocations for optimal SIMD performance
2. **Memory Pools**: Reduce allocation overhead for frequent operations
3. **Memory Mapping**: Handle datasets larger than RAM efficiently
4. **Garbage Collection**: Implement smart cleanup for old training samples

### Scalability Considerations

1. **Thread Pool Management**: Optimize thread pool size based on workload
2. **NUMA Awareness**: Consider NUMA topology for large systems
3. **Cache Optimization**: Optimize data layout for CPU cache efficiency
4. **Batch Processing**: Implement efficient batch operations for multiple queries