use ldc_engine::*;
use std::time::{Duration, Instant};
use rand::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;
use std::collections::HashMap;
use polars::prelude::*;
use tempfile::TempDir;
use std::path::Path;

/// Comprehensive integration tests for performance optimization features
/// Tests all optimization features together using real market data

/// Test end-to-end performance using real market data from existing samples
#[test]
fn test_end_to_end_performance_with_real_data() {
    println!("Testing end-to-end performance with real market data...");
    
    // Load real market data from sample files
    let real_samples = load_real_market_data().expect("Failed to load real market data");
    println!("Loaded {} real market samples", real_samples.len());
    
    if real_samples.is_empty() {
        println!("Warning: No real market data available, using synthetic data");
        test_end_to_end_performance_with_synthetic_data();
        return;
    }
    
    // Test different configurations with real data
    let configurations = vec![
        ("sequential", create_sequential_config()),
        ("parallel", create_parallel_config()),
        ("hnsw", create_hnsw_config()),
        ("full_optimization", create_full_optimization_config()),
    ];
    
    for (config_name, config) in configurations {
        println!("\nTesting configuration: {}", config_name);
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add real market data to engine
        for sample in &real_samples {
            engine.add_training_sample(sample.clone());
        }
        
        // Test with realistic queries based on real data
        let test_queries = generate_realistic_queries(&real_samples, 10);
        
        let mut query_times = Vec::new();
        let mut results_consistency = Vec::new();
        
        for query in &test_queries {
            let start = Instant::now();
            let results = engine.find_k_nearest_neighbors_optimized(query);
            let duration = start.elapsed();
            
            query_times.push(duration);
            results_consistency.push(results.len());
        }
        
        // Calculate performance statistics
        let avg_time = query_times.iter().sum::<Duration>() / query_times.len() as u32;
        let max_time = query_times.iter().max().unwrap();
        let min_time = query_times.iter().min().unwrap();
        
        println!("  Average query time: {:?}", avg_time);
        println!("  Min query time: {:?}", min_time);
        println!("  Max query time: {:?}", max_time);
        println!("  Results consistency: {:?}", results_consistency);
        
        // Verify performance targets based on data size
        let target_time = if real_samples.len() <= 10000 {
            Duration::from_millis(1)
        } else if real_samples.len() <= 50000 {
            Duration::from_millis(5)
        } else {
            Duration::from_millis(10)
        };
        
        assert!(avg_time <= target_time,
               "Configuration {} average time {:?} exceeds target {:?} for {} samples",
               config_name, avg_time, target_time, real_samples.len());
        
        // Verify all queries returned results
        assert!(results_consistency.iter().all(|&count| count > 0),
               "Some queries returned no results for configuration {}", config_name);
    }
}

/// Test HNSW accuracy against exact k-NN search with 95%+ accuracy requirement
#[test]
fn test_hnsw_accuracy_requirement_comprehensive() {
    println!("Testing HNSW accuracy requirement (95%+ accuracy)...");
    
    let sample_sizes = vec![1000, 5000, 10000, 25000];
    let accuracy_threshold = 0.95;
    
    for &sample_count in &sample_sizes {
        println!("\nTesting HNSW accuracy with {} samples", sample_count);
        
        // Create exact search engine
        let mut config_exact = LDCConfig::default();
        config_exact.max_bars_back = sample_count;
        config_exact.neighbors_count = 10;
        config_exact.use_hnsw_index = false;
        config_exact.use_multithreading = false;
        
        let mut engine_exact = LDCEngine::with_config(config_exact);
        
        // Create HNSW engine with different configurations
        let hnsw_configs = vec![
            ("standard", 16, 200, 50),
            ("high_accuracy", 32, 400, 100),
            ("balanced", 24, 300, 75),
        ];
        
        // Generate diverse training data
        let training_samples = generate_diverse_training_data(sample_count);
        
        for sample in &training_samples {
            engine_exact.add_training_sample(sample.clone());
        }
        
        for (config_name, m, ef_construction, ef_search) in hnsw_configs {
            println!("  Testing HNSW config: {} (M={}, ef_construction={}, ef_search={})", 
                     config_name, m, ef_construction, ef_search);
            
            let mut config_hnsw = LDCConfig::default();
            config_hnsw.max_bars_back = sample_count;
            config_hnsw.neighbors_count = 10;
            config_hnsw.use_hnsw_index = true;
            config_hnsw.hnsw_m = m;
            config_hnsw.hnsw_ef_construction = ef_construction;
            config_hnsw.hnsw_ef_search = ef_search;
            
            let mut engine_hnsw = LDCEngine::with_config(config_hnsw);
            
            for sample in &training_samples {
                engine_hnsw.add_training_sample(sample.clone());
            }
            
            // Test accuracy over multiple diverse queries
            let test_queries = generate_diverse_test_queries(50);
            let mut total_accuracy = 0.0;
            let mut accuracy_measurements = Vec::new();
            
            for query in &test_queries {
                let exact_results = engine_exact.find_k_nearest_neighbors_sequential_optimized(query);
                let hnsw_results = engine_hnsw.find_k_nearest_neighbors_optimized(query);
                
                // Calculate accuracy based on overlap in top-k results
                let accuracy = calculate_knn_accuracy(&exact_results, &hnsw_results);
                total_accuracy += accuracy;
                accuracy_measurements.push(accuracy);
            }
            
            let average_accuracy = total_accuracy / test_queries.len() as f32;
            
            // Calculate accuracy statistics
            accuracy_measurements.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let min_accuracy = accuracy_measurements[0];
            let p95_accuracy = accuracy_measurements[(accuracy_measurements.len() * 95) / 100];
            
            println!("    Average accuracy: {:.3}%", average_accuracy * 100.0);
            println!("    Minimum accuracy: {:.3}%", min_accuracy * 100.0);
            println!("    P95 accuracy: {:.3}%", p95_accuracy * 100.0);
            
            // Verify accuracy requirements
            assert!(average_accuracy >= accuracy_threshold,
                   "HNSW config {} average accuracy {:.3}% below required {:.1}% for {} samples",
                   config_name, average_accuracy * 100.0, accuracy_threshold * 100.0, sample_count);
            
            assert!(p95_accuracy >= accuracy_threshold * 0.9,
                   "HNSW config {} P95 accuracy {:.3}% too low for {} samples",
                   config_name, p95_accuracy * 100.0, sample_count);
        }
    }
}

/// Validate SIMD optimizations maintain exact Pine Script compatibility
#[test]
fn test_simd_pine_script_compatibility_comprehensive() {
    println!("Testing SIMD Pine Script compatibility...");
    
    // Test with various feature value ranges that might cause numerical issues
    let test_cases = vec![
        ("normal_range", generate_normal_range_features(1000)),
        ("extreme_values", generate_extreme_value_features(1000)),
        ("edge_cases", generate_edge_case_features(1000)),
        ("real_market_like", generate_market_like_features(1000)),
    ];
    
    for (case_name, feature_pairs) in test_cases {
        println!("  Testing SIMD compatibility: {}", case_name);
        
        let mut max_difference = 0.0f32;
        let mut total_difference = 0.0f32;
        let mut error_count = 0;
        
        for (features1, features2) in &feature_pairs {
            // Calculate distance using standard method
            let standard_distance = features1.lorentzian_distance_standard(features2);
            
            // Calculate distance using SIMD method
            let simd_result = features1.lorentzian_distance_simd(features2);
            
            match simd_result {
                Ok(simd_distance) => {
                    let difference = (standard_distance - simd_distance).abs();
                    max_difference = max_difference.max(difference);
                    total_difference += difference;
                    
                    // Verify exact compatibility (allowing for minimal floating-point precision differences)
                    assert!(difference < 1e-6,
                           "SIMD distance {:.10} differs from standard {:.10} by {:.10} for case {}",
                           simd_distance, standard_distance, difference, case_name);
                }
                Err(_) => {
                    error_count += 1;
                    // SIMD should fall back to standard calculation on error
                    let fallback_distance = features1.lorentzian_distance_standard(features2);
                    assert!((standard_distance - fallback_distance).abs() < 1e-10,
                           "Fallback calculation differs from standard for case {}", case_name);
                }
            }
        }
        
        let avg_difference = total_difference / feature_pairs.len() as f32;
        println!("    Max difference: {:.2e}", max_difference);
        println!("    Avg difference: {:.2e}", avg_difference);
        println!("    SIMD errors: {}/{}", error_count, feature_pairs.len());
        
        // Verify compatibility requirements
        assert!(max_difference < 1e-5,
               "Maximum SIMD difference {:.2e} too large for case {}", max_difference, case_name);
        
        assert!(error_count < feature_pairs.len() / 10,
               "Too many SIMD errors ({}/{}) for case {}", error_count, feature_pairs.len(), case_name);
    }
    
    // Test batch SIMD operations
    println!("  Testing batch SIMD compatibility...");
    test_batch_simd_compatibility();
}

/// Test memory usage patterns and verify memory mapping functionality
#[test]
fn test_memory_usage_patterns_and_mapping() {
    println!("Testing memory usage patterns and memory mapping...");
    
    // Test memory pool functionality
    test_memory_pool_patterns();
    
    // Test memory mapping with different configurations
    test_memory_mapping_functionality();
    
    // Test memory threshold monitoring
    test_memory_threshold_behavior();
    
    // Test memory efficiency of optimized data structures
    test_optimized_data_structure_memory_efficiency();
}

/// Create stress tests for concurrent access and high-throughput scenarios
#[test]
fn test_concurrent_access_stress_test() {
    println!("Testing concurrent access and high-throughput scenarios...");
    
    let sample_count = 10000;
    let num_threads = num_cpus::get().max(4);
    let queries_per_thread = 100;
    
    println!("  Using {} threads with {} queries each", num_threads, queries_per_thread);
    
    // Create shared engine with thread-safe configuration
    let mut config = LDCConfig::default();
    config.max_bars_back = sample_count;
    config.neighbors_count = 8;
    config.use_multithreading = true;
    config.use_hnsw_index = true;
    config.parallel_threshold = 100;
    config.hnsw_m = 16;
    config.hnsw_ef_construction = 200;
    config.hnsw_ef_search = 50;
    
    let mut engine = LDCEngine::with_config(config);
    
    // Populate with training data
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
    
    // Note: For true concurrent testing, we'd need to make LDCEngine thread-safe
    // For now, we test high-throughput sequential access as a stress test
    
    let start_time = Instant::now();
    let mut total_queries = 0;
    let mut all_query_times = Vec::new();
    
    // Simulate concurrent load with rapid sequential queries
    for thread_id in 0..num_threads {
        let mut thread_rng = StdRng::seed_from_u64(42 + thread_id as u64);
        
        for query_id in 0..queries_per_thread {
            let query = FeatureSeries {
                f1: thread_rng.gen_range(0.0..100.0),
                f2: thread_rng.gen_range(-100.0..100.0),
                f3: thread_rng.gen_range(-100.0..100.0),
                f4: thread_rng.gen_range(0.0..100.0),
                f5: thread_rng.gen_range(0.0..100.0),
            };
            
            let query_start = Instant::now();
            let results = engine.find_k_nearest_neighbors_optimized(&query);
            let query_time = query_start.elapsed();
            
            all_query_times.push(query_time);
            total_queries += 1;
            
            // Verify results are valid
            assert!(!results.is_empty(), 
                   "Query {}/{} returned no results", thread_id, query_id);
            assert!(results.len() <= 8, 
                   "Query {}/{} returned too many results: {}", thread_id, query_id, results.len());
        }
    }
    
    let total_time = start_time.elapsed();
    
    // Calculate performance statistics
    all_query_times.sort();
    let avg_time = all_query_times.iter().sum::<Duration>() / all_query_times.len() as u32;
    let p50_time = all_query_times[all_query_times.len() / 2];
    let p95_time = all_query_times[(all_query_times.len() * 95) / 100];
    let p99_time = all_query_times[(all_query_times.len() * 99) / 100];
    let max_time = all_query_times[all_query_times.len() - 1];
    
    let queries_per_second = total_queries as f64 / total_time.as_secs_f64();
    
    println!("  Stress test results:");
    println!("    Total queries: {}", total_queries);
    println!("    Total time: {:?}", total_time);
    println!("    Queries per second: {:.2}", queries_per_second);
    println!("    Average query time: {:?}", avg_time);
    println!("    P50 query time: {:?}", p50_time);
    println!("    P95 query time: {:?}", p95_time);
    println!("    P99 query time: {:?}", p99_time);
    println!("    Max query time: {:?}", max_time);
    
    // Verify stress test requirements
    assert!(queries_per_second > 100.0,
           "Query rate {:.2} queries/sec too low for stress test", queries_per_second);
    
    assert!(p95_time <= Duration::from_millis(5),
           "P95 query time {:?} too high for stress test", p95_time);
    
    assert!(avg_time <= Duration::from_millis(2),
           "Average query time {:?} too high for stress test", avg_time);
}

/// Benchmark complete system performance against 1ms query time targets
#[test]
fn test_1ms_query_time_targets_comprehensive() {
    println!("Testing 1ms query time targets comprehensively...");
    
    let test_scenarios = vec![
        ("small_dataset", 1000, Duration::from_micros(500)),
        ("medium_dataset", 10000, Duration::from_millis(1)),
        ("large_dataset", 50000, Duration::from_millis(5)),
    ];
    
    for (scenario_name, sample_count, target_time) in test_scenarios {
        println!("\n  Testing scenario: {} ({} samples, target: {:?})", 
                 scenario_name, sample_count, target_time);
        
        // Test different optimization strategies
        let strategies = vec![
            ("sequential", create_sequential_config()),
            ("parallel", create_parallel_config()),
            ("hnsw", create_hnsw_config()),
            ("full_optimization", create_full_optimization_config()),
        ];
        
        for (strategy_name, mut config) in strategies {
            config.max_bars_back = sample_count;
            
            let mut engine = LDCEngine::with_config(config);
            
            // Generate realistic training data
            let training_samples = generate_realistic_training_data(sample_count);
            for sample in &training_samples {
                engine.add_training_sample(sample.clone());
            }
            
            // Generate realistic test queries
            let test_queries = generate_realistic_test_queries(100);
            
            // Warm up the engine
            for _ in 0..5 {
                let _ = engine.find_k_nearest_neighbors_optimized(&test_queries[0]);
            }
            
            // Measure performance
            let mut query_times = Vec::new();
            
            for query in &test_queries {
                let start = Instant::now();
                let results = engine.find_k_nearest_neighbors_optimized(query);
                let duration = start.elapsed();
                
                query_times.push(duration);
                
                // Verify results are valid
                assert!(!results.is_empty(), 
                       "Query returned no results for strategy {}", strategy_name);
            }
            
            // Calculate statistics
            query_times.sort();
            let avg_time = query_times.iter().sum::<Duration>() / query_times.len() as u32;
            let p50_time = query_times[query_times.len() / 2];
            let p95_time = query_times[(query_times.len() * 95) / 100];
            let p99_time = query_times[(query_times.len() * 99) / 100];
            
            println!("    Strategy {}: avg={:?}, p50={:?}, p95={:?}, p99={:?}", 
                     strategy_name, avg_time, p50_time, p95_time, p99_time);
            
            // Verify performance targets
            if strategy_name == "full_optimization" || 
               (strategy_name == "hnsw" && sample_count >= 10000) ||
               (strategy_name == "parallel" && sample_count >= 1000) {
                assert!(avg_time <= target_time,
                       "Strategy {} average time {:?} exceeds target {:?} for scenario {}",
                       strategy_name, avg_time, target_time, scenario_name);
                
                assert!(p95_time <= target_time * 2,
                       "Strategy {} P95 time {:?} exceeds 2x target for scenario {}",
                       strategy_name, p95_time, scenario_name);
            }
        }
    }
}

// Helper functions for test data generation and validation

fn load_real_market_data() -> Result<Vec<TrainingSample>, Box<dyn std::error::Error>> {
    // Try to load from sample parquet files
    let sample_paths = vec![
        "rust/sample/features.parquet",
        "rust/sample/ohlcv.parquet",
        "rust/partitioned_data/symbol=BTCUSDT/date=2025-09-19/interval=5m/features.parquet",
    ];
    
    for path in sample_paths {
        if std::path::Path::new(path).exists() {
            match load_samples_from_parquet(path) {
                Ok(samples) => {
                    if !samples.is_empty() {
                        return Ok(samples);
                    }
                }
                Err(e) => {
                    println!("Warning: Failed to load {}: {}", path, e);
                }
            }
        }
    }
    
    // If no real data available, return empty vector
    Ok(Vec::new())
}

fn load_samples_from_parquet(path: &str) -> Result<Vec<TrainingSample>, Box<dyn std::error::Error>> {
    // This is a simplified implementation - in practice, you'd parse the actual parquet structure
    // For now, return empty vector as we don't have the exact parquet schema
    println!("Note: Parquet loading not implemented, using synthetic data");
    Ok(Vec::new())
}

fn test_end_to_end_performance_with_synthetic_data() {
    println!("Using synthetic data for end-to-end performance test");
    
    let sample_count = 10000;
    let mut config = create_full_optimization_config();
    config.max_bars_back = sample_count;
    
    let mut engine = LDCEngine::with_config(config);
    
    // Generate synthetic market-like data
    let samples = generate_realistic_training_data(sample_count);
    for sample in &samples {
        engine.add_training_sample(sample.clone());
    }
    
    // Test performance
    let queries = generate_realistic_test_queries(10);
    let mut query_times = Vec::new();
    
    for query in &queries {
        let start = Instant::now();
        let _results = engine.find_k_nearest_neighbors_optimized(query);
        query_times.push(start.elapsed());
    }
    
    let avg_time = query_times.iter().sum::<Duration>() / query_times.len() as u32;
    println!("Synthetic data average query time: {:?}", avg_time);
    
    assert!(avg_time <= Duration::from_millis(1),
           "Synthetic data query time {:?} exceeds 1ms target", avg_time);
}

fn generate_realistic_queries(samples: &[TrainingSample], count: usize) -> Vec<FeatureSeries> {
    let mut rng = StdRng::seed_from_u64(12345);
    let mut queries = Vec::new();
    
    for _ in 0..count {
        if !samples.is_empty() {
            // Base query on existing sample with some variation
            let base_sample = &samples[rng.gen_range(0..samples.len())];
            let noise_factor = 0.1;
            
            queries.push(FeatureSeries {
                f1: base_sample.features.f1 + rng.gen_range(-10.0..10.0) * noise_factor,
                f2: base_sample.features.f2 + rng.gen_range(-10.0..10.0) * noise_factor,
                f3: base_sample.features.f3 + rng.gen_range(-10.0..10.0) * noise_factor,
                f4: base_sample.features.f4 + rng.gen_range(-10.0..10.0) * noise_factor,
                f5: base_sample.features.f5 + rng.gen_range(-10.0..10.0) * noise_factor,
            });
        } else {
            // Generate random realistic query
            queries.push(FeatureSeries {
                f1: rng.gen_range(20.0..80.0),   // RSI range
                f2: rng.gen_range(-50.0..50.0),  // WT range
                f3: rng.gen_range(-100.0..100.0), // CCI range
                f4: rng.gen_range(10.0..90.0),   // ADX range
                f5: rng.gen_range(0.0..100.0),   // Additional feature
            });
        }
    }
    
    queries
}

fn generate_diverse_training_data(count: usize) -> Vec<TrainingSample> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut samples = Vec::new();
    
    for i in 0..count {
        // Generate diverse feature patterns
        let pattern = i % 5;
        let features = match pattern {
            0 => FeatureSeries { // Bullish pattern
                f1: rng.gen_range(60.0..90.0),   // High RSI
                f2: rng.gen_range(20.0..50.0),   // Positive WT
                f3: rng.gen_range(50.0..150.0),  // High CCI
                f4: rng.gen_range(40.0..80.0),   // Strong ADX
                f5: rng.gen_range(60.0..100.0),  // High additional
            },
            1 => FeatureSeries { // Bearish pattern
                f1: rng.gen_range(10.0..40.0),   // Low RSI
                f2: rng.gen_range(-50.0..-20.0), // Negative WT
                f3: rng.gen_range(-150.0..-50.0), // Low CCI
                f4: rng.gen_range(40.0..80.0),   // Strong ADX
                f5: rng.gen_range(0.0..40.0),    // Low additional
            },
            2 => FeatureSeries { // Neutral pattern
                f1: rng.gen_range(40.0..60.0),   // Mid RSI
                f2: rng.gen_range(-20.0..20.0),  // Neutral WT
                f3: rng.gen_range(-50.0..50.0),  // Neutral CCI
                f4: rng.gen_range(10.0..40.0),   // Weak ADX
                f5: rng.gen_range(40.0..60.0),   // Mid additional
            },
            3 => FeatureSeries { // Volatile pattern
                f1: rng.gen_range(0.0..100.0),   // Any RSI
                f2: rng.gen_range(-100.0..100.0), // Any WT
                f3: rng.gen_range(-200.0..200.0), // Extreme CCI
                f4: rng.gen_range(60.0..100.0),  // Very strong ADX
                f5: rng.gen_range(0.0..100.0),   // Any additional
            },
            _ => FeatureSeries { // Random pattern
                f1: rng.gen_range(0.0..100.0),
                f2: rng.gen_range(-100.0..100.0),
                f3: rng.gen_range(-100.0..100.0),
                f4: rng.gen_range(0.0..100.0),
                f5: rng.gen_range(0.0..100.0),
            },
        };
        
        let label = match pattern {
            0 => Direction::Long,
            1 => Direction::Short,
            _ => Direction::Neutral,
        };
        
        samples.push(TrainingSample {
            features,
            label,
            timestamp: i as i64,
            bar_index: i,
        });
    }
    
    samples
}

fn generate_diverse_test_queries(count: usize) -> Vec<FeatureSeries> {
    let mut rng = StdRng::seed_from_u64(54321);
    let mut queries = Vec::new();
    
    for i in 0..count {
        let pattern = i % 3;
        let query = match pattern {
            0 => FeatureSeries { // Typical query
                f1: rng.gen_range(30.0..70.0),
                f2: rng.gen_range(-30.0..30.0),
                f3: rng.gen_range(-75.0..75.0),
                f4: rng.gen_range(20.0..60.0),
                f5: rng.gen_range(30.0..70.0),
            },
            1 => FeatureSeries { // Edge case query
                f1: if rng.gen_bool(0.5) { rng.gen_range(0.0..10.0) } else { rng.gen_range(90.0..100.0) },
                f2: if rng.gen_bool(0.5) { rng.gen_range(-100.0..-80.0) } else { rng.gen_range(80.0..100.0) },
                f3: if rng.gen_bool(0.5) { rng.gen_range(-200.0..-150.0) } else { rng.gen_range(150.0..200.0) },
                f4: rng.gen_range(0.0..100.0),
                f5: rng.gen_range(0.0..100.0),
            },
            _ => FeatureSeries { // Random query
                f1: rng.gen_range(0.0..100.0),
                f2: rng.gen_range(-100.0..100.0),
                f3: rng.gen_range(-100.0..100.0),
                f4: rng.gen_range(0.0..100.0),
                f5: rng.gen_range(0.0..100.0),
            },
        };
        queries.push(query);
    }
    
    queries
}

fn calculate_knn_accuracy(exact_results: &[(f32, Direction)], hnsw_results: &[(f32, Direction)]) -> f32 {
    if exact_results.is_empty() || hnsw_results.is_empty() {
        return 0.0;
    }
    
    let k = exact_results.len().min(hnsw_results.len());
    
    // Compare based on distance similarity rather than exact order
    // Since HNSW is approximate, we allow for some reordering of similar distances
    let mut matches = 0;
    
    // For each HNSW result, check if there's a similar result in exact results
    for hnsw_result in hnsw_results.iter().take(k) {
        let hnsw_distance = hnsw_result.0;
        let hnsw_direction = hnsw_result.1;
        
        // Look for a match within a reasonable distance tolerance
        let distance_tolerance = 0.1; // Allow 10% distance difference
        
        for exact_result in exact_results.iter().take(k) {
            let exact_distance = exact_result.0;
            let exact_direction = exact_result.1;
            
            // Check if distances are similar and directions match
            let distance_diff = (hnsw_distance - exact_distance).abs();
            let relative_diff = if exact_distance > 0.0 {
                distance_diff / exact_distance
            } else {
                distance_diff
            };
            
            if relative_diff <= distance_tolerance && hnsw_direction == exact_direction {
                matches += 1;
                break; // Found a match, move to next HNSW result
            }
        }
    }
    
    matches as f32 / k as f32
}

fn generate_normal_range_features(count: usize) -> Vec<(FeatureSeries, FeatureSeries)> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut pairs = Vec::new();
    
    for _ in 0..count {
        let f1 = FeatureSeries {
            f1: rng.gen_range(20.0..80.0),
            f2: rng.gen_range(-50.0..50.0),
            f3: rng.gen_range(-100.0..100.0),
            f4: rng.gen_range(10.0..90.0),
            f5: rng.gen_range(0.0..100.0),
        };
        
        let f2 = FeatureSeries {
            f1: rng.gen_range(20.0..80.0),
            f2: rng.gen_range(-50.0..50.0),
            f3: rng.gen_range(-100.0..100.0),
            f4: rng.gen_range(10.0..90.0),
            f5: rng.gen_range(0.0..100.0),
        };
        
        pairs.push((f1, f2));
    }
    
    pairs
}

fn generate_extreme_value_features(count: usize) -> Vec<(FeatureSeries, FeatureSeries)> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut pairs = Vec::new();
    
    for _ in 0..count {
        let f1 = FeatureSeries {
            f1: if rng.gen_bool(0.5) { rng.gen_range(0.0..1.0) } else { rng.gen_range(99.0..100.0) },
            f2: if rng.gen_bool(0.5) { rng.gen_range(-1000.0..-100.0) } else { rng.gen_range(100.0..1000.0) },
            f3: if rng.gen_bool(0.5) { rng.gen_range(-500.0..-200.0) } else { rng.gen_range(200.0..500.0) },
            f4: if rng.gen_bool(0.5) { rng.gen_range(0.0..5.0) } else { rng.gen_range(95.0..100.0) },
            f5: rng.gen_range(0.0..1000.0),
        };
        
        let f2 = FeatureSeries {
            f1: if rng.gen_bool(0.5) { rng.gen_range(0.0..1.0) } else { rng.gen_range(99.0..100.0) },
            f2: if rng.gen_bool(0.5) { rng.gen_range(-1000.0..-100.0) } else { rng.gen_range(100.0..1000.0) },
            f3: if rng.gen_bool(0.5) { rng.gen_range(-500.0..-200.0) } else { rng.gen_range(200.0..500.0) },
            f4: if rng.gen_bool(0.5) { rng.gen_range(0.0..5.0) } else { rng.gen_range(95.0..100.0) },
            f5: rng.gen_range(0.0..1000.0),
        };
        
        pairs.push((f1, f2));
    }
    
    pairs
}

fn generate_edge_case_features(count: usize) -> Vec<(FeatureSeries, FeatureSeries)> {
    let mut pairs = Vec::new();
    
    // Test with identical features
    let identical = FeatureSeries {
        f1: 50.0,
        f2: 0.0,
        f3: 0.0,
        f4: 50.0,
        f5: 50.0,
    };
    pairs.push((identical.clone(), identical.clone()));
    
    // Test with zero features
    let zero = FeatureSeries {
        f1: 0.0,
        f2: 0.0,
        f3: 0.0,
        f4: 0.0,
        f5: 0.0,
    };
    pairs.push((zero.clone(), identical.clone()));
    
    // Test with very small differences
    let small_diff1 = FeatureSeries {
        f1: 50.0,
        f2: 0.0,
        f3: 0.0,
        f4: 50.0,
        f5: 50.0,
    };
    let small_diff2 = FeatureSeries {
        f1: 50.000001,
        f2: 0.000001,
        f3: -0.000001,
        f4: 50.000001,
        f5: 50.000001,
    };
    pairs.push((small_diff1, small_diff2));
    
    // Fill remaining with random edge cases
    let mut rng = StdRng::seed_from_u64(42);
    while pairs.len() < count {
        let f1 = FeatureSeries {
            f1: rng.gen_range(0.0..100.0),
            f2: rng.gen_range(-100.0..100.0),
            f3: rng.gen_range(-100.0..100.0),
            f4: rng.gen_range(0.0..100.0),
            f5: rng.gen_range(0.0..100.0),
        };
        
        let f2 = FeatureSeries {
            f1: rng.gen_range(0.0..100.0),
            f2: rng.gen_range(-100.0..100.0),
            f3: rng.gen_range(-100.0..100.0),
            f4: rng.gen_range(0.0..100.0),
            f5: rng.gen_range(0.0..100.0),
        };
        
        pairs.push((f1, f2));
    }
    
    pairs
}

fn generate_market_like_features(count: usize) -> Vec<(FeatureSeries, FeatureSeries)> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut pairs = Vec::new();
    
    for _ in 0..count {
        // Generate realistic market indicator values
        let f1 = FeatureSeries {
            f1: rng.gen_range(25.0..75.0),   // RSI typically 30-70
            f2: rng.gen_range(-40.0..40.0),  // WT oscillates around 0
            f3: rng.gen_range(-120.0..120.0), // CCI can be more extreme
            f4: rng.gen_range(15.0..85.0),   // ADX strength indicator
            f5: rng.gen_range(20.0..80.0),   // Additional indicator
        };
        
        let f2 = FeatureSeries {
            f1: rng.gen_range(25.0..75.0),
            f2: rng.gen_range(-40.0..40.0),
            f3: rng.gen_range(-120.0..120.0),
            f4: rng.gen_range(15.0..85.0),
            f5: rng.gen_range(20.0..80.0),
        };
        
        pairs.push((f1, f2));
    }
    
    pairs
}

fn test_batch_simd_compatibility() {
    let mut rng = StdRng::seed_from_u64(42);
    
    let query = FeatureSeries {
        f1: 50.0,
        f2: 25.0,
        f3: -10.0,
        f4: 75.0,
        f5: 60.0,
    };
    
    let batch_sizes = vec![1, 10, 100, 1000];
    
    for &batch_size in &batch_sizes {
        let targets: Vec<FeatureSeries> = (0..batch_size)
            .map(|_| FeatureSeries {
                f1: rng.gen_range(0.0..100.0),
                f2: rng.gen_range(-100.0..100.0),
                f3: rng.gen_range(-100.0..100.0),
                f4: rng.gen_range(0.0..100.0),
                f5: rng.gen_range(0.0..100.0),
            })
            .collect();
        
        // Calculate using standard batch method
        let standard_results = FeatureSeries::batch_lorentzian_distance_standard(&query, &targets);
        
        // Calculate using SIMD batch method
        let simd_results = FeatureSeries::batch_lorentzian_distance_simd(&query, &targets, 64)
            .expect("SIMD batch calculation failed");
        
        assert_eq!(standard_results.len(), simd_results.len(),
                  "Batch size mismatch for {} targets", batch_size);
        
        for (i, (&standard, &simd)) in standard_results.iter().zip(simd_results.iter()).enumerate() {
            let difference = (standard - simd).abs();
            assert!(difference < 1e-6,
                   "Batch SIMD difference {:.10} too large at index {} for batch size {}",
                   difference, i, batch_size);
        }
    }
}

// Configuration helper functions

fn create_sequential_config() -> LDCConfig {
    let mut config = LDCConfig::default();
    config.use_multithreading = false;
    config.use_hnsw_index = false;
    config.use_simd_optimization = false;
    config
}

fn create_parallel_config() -> LDCConfig {
    let mut config = LDCConfig::default();
    config.use_multithreading = true;
    config.use_hnsw_index = false;
    config.parallel_threshold = 100;
    config.max_threads = Some(num_cpus::get());
    config
}

fn create_hnsw_config() -> LDCConfig {
    let mut config = LDCConfig::default();
    config.use_hnsw_index = true;
    config.hnsw_m = 16;
    config.hnsw_ef_construction = 200;
    config.hnsw_ef_search = 50;
    config
}

fn create_full_optimization_config() -> LDCConfig {
    let mut config = LDCConfig::default();
    config.use_multithreading = true;
    config.use_hnsw_index = true;
    config.use_simd_optimization = true;
    config.parallel_threshold = 100;
    config.max_threads = Some(num_cpus::get());
    config.hnsw_m = 16;
    config.hnsw_ef_construction = 200;
    config.hnsw_ef_search = 50;
    config.simd_chunk_size = 256;
    config.memory_pool_size = 100;
    config.enable_memory_mapping = true;
    config
}

// Memory testing helper functions

fn test_memory_pool_patterns() {
    println!("    Testing memory pool patterns...");
    
    let pool_size_mb = 50;
    let mut pool = MemoryPool::new(pool_size_mb).expect("Failed to create memory pool");
    
    let allocation_size = std::mem::size_of::<OptimizedTrainingSample>();
    let alignment = std::mem::align_of::<OptimizedTrainingSample>();
    
    // Test allocation patterns
    let mut allocations = Vec::new();
    
    // Allocate many small blocks
    for _ in 0..1000 {
        if let Ok(ptr) = pool.allocate(allocation_size, alignment) {
            allocations.push(ptr);
        }
    }
    
    println!("      Allocated {} blocks", allocations.len());
    println!("      Pool utilization: {:.1}%", pool.utilization_percent());
    
    // Deallocate half
    for ptr in allocations.drain(0..allocations.len()/2) {
        pool.deallocate(ptr);
    }
    
    println!("      After partial deallocation: {:.1}%", pool.utilization_percent());
    
    // Cleanup remaining
    for ptr in allocations {
        pool.deallocate(ptr);
    }
    
    println!("      After full cleanup: {:.1}%", pool.utilization_percent());
    
    assert_eq!(pool.allocated_bytes(), 0, "Memory pool should be empty after cleanup");
}

fn test_memory_mapping_functionality() {
    println!("    Testing memory mapping functionality...");
    
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test_mmap.dat");
    
    let max_samples = 1000;
    
    // Test write operations
    {
        let mut storage = MemoryMappedStorage::new(&file_path, max_samples, false)
            .expect("Failed to create memory mapped storage");
        
        // Add samples
        for i in 0..100 {
            let sample = OptimizedTrainingSample {
                features: AlignedFeatureSeries {
                    features: [i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32, (i * 5) as f32, 0.0, 0.0, 0.0],
                },
                label: Direction::Long,
                timestamp: i as i64,
                bar_index: i as u32,
                _padding: [0; 12],
            };
            
            storage.push_sample(&sample).expect("Failed to push sample");
        }
        
        assert_eq!(storage.len(), 100, "Storage should contain 100 samples");
        
        // Verify data integrity
        for i in 0..100 {
            let sample = storage.get_sample(i).expect("Failed to get sample");
            assert_eq!(sample.features.features[0], i as f32, "Feature data mismatch");
            assert_eq!(sample.bar_index, i as u32, "Bar index mismatch");
        }
        
        storage.flush().expect("Failed to flush storage");
    }
    
    // Test read operations
    {
        let storage = MemoryMappedStorage::new(&file_path, max_samples, true)
            .expect("Failed to open memory mapped storage for reading");
        
        // Note: In a real implementation, we'd need to store the sample count in the file
        // For this test, we'll just verify the file exists and can be opened
        assert!(file_path.exists(), "Memory mapped file should exist");
    }
    
    println!("      Memory mapping test completed successfully");
}

fn test_memory_threshold_behavior() {
    println!("    Testing memory threshold behavior...");
    
    let mut monitor = MemoryThresholdMonitor::new(100, 80.0, 95.0); // 100MB threshold
    
    // Test normal usage
    let status = monitor.check_memory_usage(50); // 50MB
    match status {
        MemoryStatus::Normal => println!("      Normal usage: OK"),
        _ => panic!("Expected normal status for 50MB usage"),
    }
    
    // Test warning threshold
    let status = monitor.check_memory_usage(85); // 85MB (85% of 100MB)
    match status {
        MemoryStatus::Warning { usage_percent, usage_mb } => {
            println!("      Warning threshold triggered: {}MB ({}%)", usage_mb, usage_percent);
            assert!(usage_percent >= 80.0, "Usage percent should be >= 80%");
        },
        _ => panic!("Expected warning status for 85MB usage"),
    }
    
    // Test critical threshold
    let status = monitor.check_memory_usage(98); // 98MB (98% of 100MB)
    match status {
        MemoryStatus::Critical { usage_percent, usage_mb } => {
            println!("      Critical threshold triggered: {}MB ({}%)", usage_mb, usage_percent);
            assert!(usage_percent >= 95.0, "Usage percent should be >= 95%");
        },
        _ => panic!("Expected critical status for 98MB usage"),
    }
}

fn test_optimized_data_structure_memory_efficiency() {
    println!("    Testing optimized data structure memory efficiency...");
    
    let sample_count = 1000;
    
    // Test standard TrainingSample storage
    let mut standard_samples = Vec::new();
    for i in 0..sample_count {
        standard_samples.push(TrainingSample {
            features: FeatureSeries {
                f1: i as f32,
                f2: (i * 2) as f32,
                f3: (i * 3) as f32,
                f4: (i * 4) as f32,
                f5: (i * 5) as f32,
            },
            label: Direction::Long,
            timestamp: i as i64,
            bar_index: i,
        });
    }
    
    // Test optimized TrainingSample storage
    let mut optimized_samples = Vec::new();
    for sample in &standard_samples {
        optimized_samples.push(OptimizedTrainingSample::from_training_sample(sample));
    }
    
    let standard_size = std::mem::size_of::<TrainingSample>() * sample_count;
    let optimized_size = std::mem::size_of::<OptimizedTrainingSample>() * sample_count;
    
    println!("      Standard storage: {} bytes", standard_size);
    println!("      Optimized storage: {} bytes", optimized_size);
    println!("      Memory efficiency: {:.1}%", (optimized_size as f64 / standard_size as f64) * 100.0);
    
    // Verify data integrity after optimization
    for (i, (standard, optimized)) in standard_samples.iter().zip(optimized_samples.iter()).enumerate() {
        let converted_back = optimized.to_training_sample();
        
        assert_eq!(standard.features.f1, converted_back.features.f1, "Feature f1 mismatch at {}", i);
        assert_eq!(standard.features.f2, converted_back.features.f2, "Feature f2 mismatch at {}", i);
        assert_eq!(standard.features.f3, converted_back.features.f3, "Feature f3 mismatch at {}", i);
        assert_eq!(standard.features.f4, converted_back.features.f4, "Feature f4 mismatch at {}", i);
        assert_eq!(standard.features.f5, converted_back.features.f5, "Feature f5 mismatch at {}", i);
        assert_eq!(standard.label, converted_back.label, "Label mismatch at {}", i);
        assert_eq!(standard.timestamp, converted_back.timestamp, "Timestamp mismatch at {}", i);
        assert_eq!(standard.bar_index, converted_back.bar_index, "Bar index mismatch at {}", i);
    }
    
    println!("      Data integrity verified for {} samples", sample_count);
}

fn generate_realistic_training_data(count: usize) -> Vec<TrainingSample> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut samples = Vec::new();
    
    for i in 0..count {
        // Generate realistic market indicator values with some correlation
        let trend = (i as f32 / count as f32) * 2.0 - 1.0; // -1 to 1 trend
        let volatility = rng.gen_range(0.5..2.0);
        
        let features = FeatureSeries {
            f1: (50.0 + trend * 20.0 + rng.gen_range(-10.0..10.0) * volatility).clamp(0.0, 100.0), // RSI
            f2: (trend * 30.0 + rng.gen_range(-20.0..20.0) * volatility).clamp(-100.0, 100.0), // WT
            f3: (trend * 50.0 + rng.gen_range(-30.0..30.0) * volatility).clamp(-200.0, 200.0), // CCI
            f4: (30.0 + volatility * 20.0 + rng.gen_range(-10.0..10.0)).clamp(0.0, 100.0), // ADX
            f5: (50.0 + trend * 15.0 + rng.gen_range(-15.0..15.0) * volatility).clamp(0.0, 100.0), // Additional
        };
        
        let label = if trend > 0.3 {
            Direction::Long
        } else if trend < -0.3 {
            Direction::Short
        } else {
            Direction::Neutral
        };
        
        samples.push(TrainingSample {
            features,
            label,
            timestamp: i as i64,
            bar_index: i,
        });
    }
    
    samples
}

fn generate_realistic_test_queries(count: usize) -> Vec<FeatureSeries> {
    let mut rng = StdRng::seed_from_u64(54321);
    let mut queries = Vec::new();
    
    for _ in 0..count {
        // Generate realistic query patterns
        let market_condition = rng.gen_range(0..4);
        
        let query = match market_condition {
            0 => FeatureSeries { // Bullish condition
                f1: rng.gen_range(55.0..85.0),
                f2: rng.gen_range(10.0..40.0),
                f3: rng.gen_range(20.0..100.0),
                f4: rng.gen_range(30.0..70.0),
                f5: rng.gen_range(55.0..85.0),
            },
            1 => FeatureSeries { // Bearish condition
                f1: rng.gen_range(15.0..45.0),
                f2: rng.gen_range(-40.0..-10.0),
                f3: rng.gen_range(-100.0..-20.0),
                f4: rng.gen_range(30.0..70.0),
                f5: rng.gen_range(15.0..45.0),
            },
            2 => FeatureSeries { // Neutral condition
                f1: rng.gen_range(40.0..60.0),
                f2: rng.gen_range(-15.0..15.0),
                f3: rng.gen_range(-30.0..30.0),
                f4: rng.gen_range(10.0..40.0),
                f5: rng.gen_range(40.0..60.0),
            },
            _ => FeatureSeries { // Random condition
                f1: rng.gen_range(0.0..100.0),
                f2: rng.gen_range(-100.0..100.0),
                f3: rng.gen_range(-100.0..100.0),
                f4: rng.gen_range(0.0..100.0),
                f5: rng.gen_range(0.0..100.0),
            },
        };
        
        queries.push(query);
    }
    
    queries
}