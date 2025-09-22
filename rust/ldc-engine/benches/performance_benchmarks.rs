use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ldc_engine::*;
use rand::prelude::*;
use std::collections::VecDeque;
use std::time::Duration;

/// Create a test LDC engine with specified number of samples
fn create_test_engine_with_samples(sample_count: usize) -> LDCEngine {
    let mut config = LDCConfig::default();
    config.max_bars_back = sample_count.max(2000);
    config.neighbors_count = 8;
    config.feature_count = 5;
    config.use_multithreading = true;
    config.max_threads = Some(num_cpus::get());
    config.parallel_threshold = 100;
    
    let mut engine = LDCEngine::with_config(config);
    
    // Generate random training samples
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducible benchmarks
    
    for i in 0..sample_count {
        let features = FeatureSeries {
            f1: rng.gen_range(0.0..100.0),   // RSI-like
            f2: rng.gen_range(-100.0..100.0), // WT-like
            f3: rng.gen_range(-100.0..100.0), // CCI-like
            f4: rng.gen_range(0.0..100.0),   // ADX-like
            f5: rng.gen_range(0.0..100.0),   // Additional feature
        };
        
        let label = match rng.gen_range(0..3) {
            0 => Direction::Short,
            1 => Direction::Neutral,
            _ => Direction::Long,
        };
        
        let sample = TrainingSample {
            features,
            label,
            timestamp: i as i64,
            bar_index: i,
        };
        
        engine.add_training_sample(sample);
    }
    
    engine
}

/// Create a test feature series for querying
fn create_test_feature_series() -> FeatureSeries {
    FeatureSeries {
        f1: 50.0,
        f2: 25.0,
        f3: -10.0,
        f4: 75.0,
        f5: 60.0,
    }
}

/// Benchmark k-NN search comparing exact vs HNSW vs parallel strategies
fn benchmark_knn_search(c: &mut Criterion) {
    let sample_sizes = vec![1000, 5000, 10000, 25000, 50000];
    
    for &sample_count in &sample_sizes {
        let mut group = c.benchmark_group(format!("knn_search_{}_samples", sample_count));
        group.throughput(Throughput::Elements(sample_count as u64));
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(20);
        
        // Create engines with different configurations
        let mut config_exact = LDCConfig::default();
        config_exact.max_bars_back = sample_count.max(2000);
        config_exact.neighbors_count = 8;
        config_exact.use_hnsw_index = false;
        config_exact.use_multithreading = false;
        let mut engine_exact = LDCEngine::with_config(config_exact);
        
        let mut config_parallel = LDCConfig::default();
        config_parallel.max_bars_back = sample_count.max(2000);
        config_parallel.neighbors_count = 8;
        config_parallel.use_hnsw_index = false;
        config_parallel.use_multithreading = true;
        config_parallel.parallel_threshold = 100;
        let mut engine_parallel = LDCEngine::with_config(config_parallel);
        
        let mut config_hnsw = LDCConfig::default();
        config_hnsw.max_bars_back = sample_count.max(2000);
        config_hnsw.neighbors_count = 8;
        config_hnsw.use_hnsw_index = true;
        config_hnsw.hnsw_m = 16;
        config_hnsw.hnsw_ef_construction = 200;
        config_hnsw.hnsw_ef_search = 50;
        let mut engine_hnsw = LDCEngine::with_config(config_hnsw);
        
        let query = create_test_feature_series();
        
        // Benchmark exact sequential search
        group.bench_with_input(
            BenchmarkId::new("exact_sequential", sample_count),
            &sample_count,
            |b, _| {
                b.iter(|| {
                    black_box(engine_exact.find_k_nearest_neighbors_sequential_optimized(&query))
                })
            },
        );
        
        // Benchmark parallel search
        if sample_count >= 100 {
            group.bench_with_input(
                BenchmarkId::new("parallel", sample_count),
                &sample_count,
                |b, _| {
                    b.iter(|| {
                        black_box(engine_parallel.find_k_nearest_neighbors_parallel_optimized(&query))
                    })
                },
            );
        }
        
        // Benchmark HNSW approximate search
        if sample_count >= 1000 {
            group.bench_with_input(
                BenchmarkId::new("hnsw_approximate", sample_count),
                &sample_count,
                |b, _| {
                    b.iter(|| {
                        black_box(engine_hnsw.find_k_nearest_neighbors_optimized(&query))
                    })
                },
            );
        }
        
        // Benchmark optimized search (automatic strategy selection)
        group.bench_with_input(
            BenchmarkId::new("optimized_auto", sample_count),
            &sample_count,
            |b, _| {
                b.iter(|| {
                    black_box(engine_hnsw.find_k_nearest_neighbors_optimized(&query))
                })
            },
        );
        
        group.finish();
    }
}

/// Benchmark SIMD distance calculations comparing standard vs SIMD implementations
fn benchmark_simd_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_calculation");
    group.measurement_time(Duration::from_secs(5));
    
    let features1 = create_test_feature_series();
    let features2 = FeatureSeries {
        f1: 45.0,
        f2: 30.0,
        f3: -5.0,
        f4: 80.0,
        f5: 55.0,
    };
    
    // Single distance calculation benchmarks
    group.bench_function("single_distance_standard", |b| {
        b.iter(|| {
            black_box(features1.lorentzian_distance_standard(&features2))
        })
    });
    
    group.bench_function("single_distance_simd", |b| {
        b.iter(|| {
            black_box(features1.lorentzian_distance_simd(&features2))
        })
    });
    
    // Batch distance calculation benchmarks
    let batch_sizes = vec![10, 100, 1000, 10000];
    
    for &batch_size in &batch_sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let target_features: Vec<FeatureSeries> = (0..batch_size)
            .map(|_| FeatureSeries {
                f1: rng.gen_range(0.0..100.0),
                f2: rng.gen_range(-100.0..100.0),
                f3: rng.gen_range(-100.0..100.0),
                f4: rng.gen_range(0.0..100.0),
                f5: rng.gen_range(0.0..100.0),
            })
            .collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_distance_standard", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(FeatureSeries::batch_lorentzian_distance_standard(
                        &features1,
                        &target_features,
                    ))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("batch_distance_simd", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(FeatureSeries::batch_lorentzian_distance_simd(
                        &features1,
                        &target_features,
                        256, // chunk size
                    ))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage and efficiency improvements
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.measurement_time(Duration::from_secs(5));
    
    let sample_sizes = vec![1000, 5000, 10000];
    
    for &sample_count in &sample_sizes {
        // Benchmark standard VecDeque storage
        group.bench_with_input(
            BenchmarkId::new("vecdeque_storage", sample_count),
            &sample_count,
            |b, &size| {
                b.iter(|| {
                    let mut samples = VecDeque::new();
                    let mut rng = StdRng::seed_from_u64(42);
                    
                    for i in 0..size {
                        let sample = TrainingSample {
                            features: FeatureSeries {
                                f1: rng.gen_range(0.0..100.0),
                                f2: rng.gen_range(-100.0..100.0),
                                f3: rng.gen_range(-100.0..100.0),
                                f4: rng.gen_range(0.0..100.0),
                                f5: rng.gen_range(0.0..100.0),
                            },
                            label: Direction::Long,
                            timestamp: i as i64,
                            bar_index: i,
                        };
                        samples.push_back(sample);
                    }
                    black_box(samples)
                })
            },
        );
        
        // Benchmark optimized storage with aligned samples
        group.bench_with_input(
            BenchmarkId::new("optimized_storage", sample_count),
            &sample_count,
            |b, &size| {
                b.iter(|| {
                    let mut samples = Vec::new();
                    let mut rng = StdRng::seed_from_u64(42);
                    
                    for i in 0..size {
                        let sample = TrainingSample {
                            features: FeatureSeries {
                                f1: rng.gen_range(0.0..100.0),
                                f2: rng.gen_range(-100.0..100.0),
                                f3: rng.gen_range(-100.0..100.0),
                                f4: rng.gen_range(0.0..100.0),
                                f5: rng.gen_range(0.0..100.0),
                            },
                            label: Direction::Long,
                            timestamp: i as i64,
                            bar_index: i,
                        };
                        let optimized_sample = OptimizedTrainingSample::from_training_sample(&sample);
                        samples.push(optimized_sample);
                    }
                    black_box(samples)
                })
            },
        );
        
        // Benchmark memory pool allocation
        group.bench_with_input(
            BenchmarkId::new("memory_pool", sample_count),
            &sample_count,
            |b, &size| {
                b.iter(|| {
                    let mut pool = MemoryPool::new(100).expect("Failed to create memory pool");
                    
                    for _ in 0..size {
                        let ptr = pool.allocate(
                            std::mem::size_of::<OptimizedTrainingSample>(),
                            std::mem::align_of::<OptimizedTrainingSample>(),
                        );
                        if let Some(ptr) = ptr {
                            pool.deallocate(ptr);
                        }
                    }
                    black_box(pool)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark HNSW index operations
fn benchmark_hnsw_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_operations");
    group.measurement_time(Duration::from_secs(10));
    
    let sample_sizes = vec![1000, 5000, 10000];
    
    for &sample_count in &sample_sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let samples: Vec<TrainingSample> = (0..sample_count)
            .map(|i| TrainingSample {
                features: FeatureSeries {
                    f1: rng.gen_range(0.0..100.0),
                    f2: rng.gen_range(-100.0..100.0),
                    f3: rng.gen_range(-100.0..100.0),
                    f4: rng.gen_range(0.0..100.0),
                    f5: rng.gen_range(0.0..100.0),
                },
                label: Direction::Long,
                timestamp: i as i64,
                bar_index: i,
            })
            .collect();
        
        // Benchmark HNSW index construction
        group.bench_with_input(
            BenchmarkId::new("hnsw_construction", sample_count),
            &sample_count,
            |b, _| {
                b.iter(|| {
                    let config = HNSWConfig {
                        m: 16,
                        ef_construction: 200,
                        ef_search: 50,
                        max_elements: sample_count * 2,
                    };
                    let mut index = HNSWIndex::new(config).expect("Failed to create HNSW index");
                    
                    for (i, sample) in samples.iter().enumerate() {
                        index.add_sample(sample, i).expect("Failed to add sample");
                    }
                    black_box(index)
                })
            },
        );
        
        // Benchmark HNSW search performance
        if sample_count >= 1000 {
            let config = HNSWConfig {
                m: 16,
                ef_construction: 200,
                ef_search: 50,
                max_elements: sample_count * 2,
            };
            let mut index = HNSWIndex::new(config).expect("Failed to create HNSW index");
            
            for (i, sample) in samples.iter().enumerate() {
                index.add_sample(sample, i).expect("Failed to add sample");
            }
            
            let samples_deque: VecDeque<TrainingSample> = samples.into_iter().collect();
            let query = create_test_feature_series();
            
            group.bench_with_input(
                BenchmarkId::new("hnsw_search", sample_count),
                &sample_count,
                |b, _| {
                    b.iter(|| {
                        black_box(index.search_knn(&query, 8, &samples_deque))
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark thread pool performance and strategies
fn benchmark_thread_pool_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_pool_strategies");
    group.measurement_time(Duration::from_secs(5));
    
    let sample_count = 10000;
    let query = create_test_feature_series();
    
    // Test different thread pool strategies
    let strategies = vec![
        ("global", ThreadPoolStrategy::Global),
        ("dedicated", ThreadPoolStrategy::Dedicated),
        ("adaptive", ThreadPoolStrategy::Adaptive),
    ];
    
    for (strategy_name, strategy) in strategies {
        let mut config = LDCConfig::default();
        config.max_bars_back = sample_count.max(2000);
        config.neighbors_count = 8;
        config.thread_pool_strategy = strategy;
        config.use_multithreading = true;
        config.parallel_threshold = 100;
        let mut engine = LDCEngine::with_config(config);
        
        // Add samples to engine
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..sample_count {
            let features = FeatureSeries {
                f1: rng.gen_range(0.0..100.0),
                f2: rng.gen_range(-100.0..100.0),
                f3: rng.gen_range(-100.0..100.0),
                f4: rng.gen_range(0.0..100.0),
                f5: rng.gen_range(0.0..100.0),
            };
            
            let sample = TrainingSample {
                features,
                label: Direction::Long,
                timestamp: i as i64,
                bar_index: i,
            };
            
            engine.add_training_sample(sample);
        }
        
        group.bench_function(
            &format!("thread_strategy_{}", strategy_name),
            |b| {
                b.iter(|| {
                    black_box(engine.find_k_nearest_neighbors_parallel_optimized(&query))
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_knn_search,
    benchmark_simd_distance,
    benchmark_memory_usage,
    benchmark_hnsw_operations,
    benchmark_thread_pool_strategies
);
criterion_main!(benches);