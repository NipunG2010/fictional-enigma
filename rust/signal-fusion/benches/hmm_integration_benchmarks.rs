//! Performance benchmarks for HMM integration
//!
//! These benchmarks measure the performance of key operations:
//! - Cache hit latency
//! - Cache miss + service call latency (with mock)
//! - Fallback activation latency
//! - Full fusion pipeline
//!
//! Run with: cargo bench --bench hmm_integration_benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use signal_fusion::{
    hmm_client::{HmmClient, HmmClientConfig, HmmIntegration},
    weight_cache::WeightCache,
    FusionWeights, SignalComponents,
};
use std::time::Duration;

/// Benchmark cache hit latency (Requirement 1.2, 2.1)
/// Target: <1ms for cache hits
fn benchmark_cache_hit(c: &mut Criterion) {
    let cache = WeightCache::new(Duration::from_secs(60), 1000);
    let observations = [0.5, 0.3, 0.2];
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    // Pre-populate cache
    cache.insert(observations, weights.clone());
    
    c.bench_function("cache_hit", |b| {
        b.iter(|| {
            let result = cache.get(black_box(&observations));
            assert!(result.is_some());
            result
        });
    });
}

/// Benchmark cache miss latency
/// This measures just the cache lookup overhead when there's no hit
fn benchmark_cache_miss(c: &mut Criterion) {
    let cache = WeightCache::new(Duration::from_secs(60), 1000);
    
    c.bench_function("cache_miss", |b| {
        b.iter(|| {
            let observations = [
                black_box(0.5 + rand::random::<f32>() * 0.01),
                black_box(0.3 + rand::random::<f32>() * 0.01),
                black_box(0.2 + rand::random::<f32>() * 0.01),
            ];
            let result = cache.get(&observations);
            assert!(result.is_none());
            result
        });
    });
}

/// Benchmark cache insertion
fn benchmark_cache_insert(c: &mut Criterion) {
    let cache = WeightCache::new(Duration::from_secs(60), 1000);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    c.bench_function("cache_insert", |b| {
        b.iter(|| {
            let observations = [
                black_box(rand::random::<f32>()),
                black_box(rand::random::<f32>()),
                black_box(rand::random::<f32>()),
            ];
            cache.insert(observations, weights.clone());
        });
    });
}

/// Benchmark fallback activation latency (Requirement 5.4)
/// Target: <1ms for fallback activation
fn benchmark_fallback_activation(c: &mut Criterion) {
    let config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse().unwrap(),
        timeout: Duration::from_millis(100),
        retry_attempts: 0, // No retries for faster benchmark
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        },
        ..Default::default()
    };
    
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("fallback_activation", |b| {
        b.to_async(&runtime).iter(|| async {
            let client = HmmClient::with_config(config.clone()).unwrap();
            let observations = black_box([0.1, 0.2, 0.3]);
            let result = client.get_fusion_weights(observations, None).await;
            assert!(result.is_ok());
            result
        });
    });
}

/// Benchmark cache key generation
fn benchmark_cache_key_generation(c: &mut Criterion) {
    c.bench_function("cache_key_generation", |b| {
        b.iter(|| {
            let observations = black_box([0.123456, 0.789012, -0.345678]);
            // The cache internally generates keys, so we simulate by doing a get
            let cache = WeightCache::new(Duration::from_secs(60), 100);
            cache.get(&observations)
        });
    });
}

/// Benchmark concurrent cache access
fn benchmark_concurrent_cache_access(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;
    
    let cache = Arc::new(WeightCache::new(Duration::from_secs(60), 1000));
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    // Pre-populate cache
    for i in 0..10 {
        cache.insert([i as f32 * 0.1, i as f32 * 0.1, i as f32 * 0.1], weights.clone());
    }
    
    c.bench_function("concurrent_cache_access", |b| {
        b.iter(|| {
            let mut handles = vec![];
            
            for i in 0..10 {
                let cache_clone = Arc::clone(&cache);
                let handle = thread::spawn(move || {
                    let obs = [i as f32 * 0.1, i as f32 * 0.1, i as f32 * 0.1];
                    cache_clone.get(&obs)
                });
                handles.push(handle);
            }
            
            for handle in handles {
                let _ = handle.join();
            }
        });
    });
}

/// Benchmark cache eviction
fn benchmark_cache_eviction(c: &mut Criterion) {
    c.bench_function("cache_eviction", |b| {
        b.iter(|| {
            let cache = WeightCache::new(Duration::from_millis(1), 100);
            let weights = FusionWeights {
                w_ldc: 0.4,
                w_mr: 0.3,
                w_tsmom: 0.3,
            };
            
            // Fill cache
            for i in 0..100 {
                cache.insert([i as f32 * 0.01, i as f32 * 0.01, i as f32 * 0.01], weights.clone());
            }
            
            // Wait for expiration
            std::thread::sleep(Duration::from_millis(2));
            
            // Trigger eviction
            cache.evict_expired();
        });
    });
}

/// Benchmark cache statistics calculation
fn benchmark_cache_stats(c: &mut Criterion) {
    let cache = WeightCache::new(Duration::from_secs(60), 1000);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    // Add some entries and generate hits/misses
    cache.insert([0.1, 0.1, 0.1], weights.clone());
    cache.get(&[0.1, 0.1, 0.1]); // hit
    cache.get(&[0.2, 0.2, 0.2]); // miss
    
    c.bench_function("cache_stats", |b| {
        b.iter(|| {
            black_box(cache.get_stats())
        });
    });
}

/// Benchmark full fusion pipeline with cache
/// This simulates the complete workflow: check cache -> fallback -> cache insert
fn benchmark_full_fusion_pipeline(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("full_fusion_pipeline_cached", |b| {
        b.to_async(&runtime).iter(|| async {
            let config = HmmClientConfig {
                base_url: "http://invalid-host:9999".parse().unwrap(),
                timeout: Duration::from_millis(100),
                retry_attempts: 0,
                enable_fallback: true,
                fallback_weights: FusionWeights {
                    w_ldc: 0.4,
                    w_mr: 0.3,
                    w_tsmom: 0.3,
                },
                ..Default::default()
            };
            
            let mut integration = HmmIntegration::with_config(config).unwrap();
            
            let signals = SignalComponents {
                s_ldc: black_box(0.05),
                s_mr: black_box(-0.02),
                s_tsmom: black_box(0.08),
            };
            
            // First call will miss cache and use fallback
            let result = integration.get_fusion_weights_for_signals(&signals).await;
            assert!(result.is_ok());
            
            // Second call should hit cache
            let result = integration.get_fusion_weights_for_signals(&signals).await;
            assert!(result.is_ok());
            
            result
        });
    });
}

/// Benchmark signal component validation
fn benchmark_signal_validation(c: &mut Criterion) {
    c.bench_function("signal_validation", |b| {
        b.iter(|| {
            let signals = SignalComponents {
                s_ldc: black_box(0.05),
                s_mr: black_box(-0.02),
                s_tsmom: black_box(0.08),
            };
            
            // Validate signal ranges
            let valid = signals.s_ldc >= -1.0 && signals.s_ldc <= 1.0
                && signals.s_mr >= -1.0 && signals.s_mr <= 1.0
                && signals.s_tsmom >= -1.0 && signals.s_tsmom <= 1.0;
            
            assert!(valid);
            valid
        });
    });
}

/// Benchmark weight normalization
fn benchmark_weight_normalization(c: &mut Criterion) {
    c.bench_function("weight_normalization", |b| {
        b.iter(|| {
            let weights = FusionWeights {
                w_ldc: black_box(0.4),
                w_mr: black_box(0.3),
                w_tsmom: black_box(0.3),
            };
            
            let total = weights.w_ldc + weights.w_mr + weights.w_tsmom;
            let normalized = FusionWeights {
                w_ldc: weights.w_ldc / total,
                w_mr: weights.w_mr / total,
                w_tsmom: weights.w_tsmom / total,
            };
            
            black_box(normalized)
        });
    });
}

/// Benchmark cache with varying sizes
fn benchmark_cache_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_sizes");
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    for size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let cache = WeightCache::new(Duration::from_secs(60), size);
            
            // Pre-populate cache to 80% capacity
            let populate_count = (size as f32 * 0.8) as usize;
            for i in 0..populate_count {
                cache.insert(
                    [i as f32 * 0.001, i as f32 * 0.001, i as f32 * 0.001],
                    weights.clone(),
                );
            }
            
            b.iter(|| {
                // Mix of hits and misses
                let obs = if rand::random::<f32>() < 0.8 {
                    // 80% cache hits
                    let i = (rand::random::<f32>() * populate_count as f32) as usize;
                    [i as f32 * 0.001, i as f32 * 0.001, i as f32 * 0.001]
                } else {
                    // 20% cache misses
                    [rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>()]
                };
                
                cache.get(black_box(&obs))
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_cache_hit,
    benchmark_cache_miss,
    benchmark_cache_insert,
    benchmark_fallback_activation,
    benchmark_cache_key_generation,
    benchmark_concurrent_cache_access,
    benchmark_cache_eviction,
    benchmark_cache_stats,
    benchmark_full_fusion_pipeline,
    benchmark_signal_validation,
    benchmark_weight_normalization,
    benchmark_cache_sizes,
);

criterion_main!(benches);
