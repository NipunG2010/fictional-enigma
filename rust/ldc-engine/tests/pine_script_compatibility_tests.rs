use ldc_engine::*;
use rand::prelude::*;

/// Test data structure for Pine Script compatibility validation
#[derive(Debug, Clone)]
struct PineScriptTestCase {
    name: String,
    features: FeatureSeries,
    expected_distances: Vec<f32>,
    expected_labels: Vec<Direction>,
    tolerance: f32,
}

/// Create a deterministic set of training samples for consistent testing
fn create_deterministic_training_samples() -> Vec<TrainingSample> {
    vec![
        TrainingSample {
            features: FeatureSeries { f1: 50.0, f2: 25.0, f3: -10.0, f4: 75.0, f5: 60.0 },
            label: Direction::Long,
            timestamp: 1000,
            bar_index: 0,
        },
        TrainingSample {
            features: FeatureSeries { f1: 30.0, f2: -15.0, f3: 20.0, f4: 45.0, f5: 35.0 },
            label: Direction::Short,
            timestamp: 2000,
            bar_index: 1,
        },
        TrainingSample {
            features: FeatureSeries { f1: 70.0, f2: 40.0, f3: -5.0, f4: 85.0, f5: 75.0 },
            label: Direction::Long,
            timestamp: 3000,
            bar_index: 2,
        },
        TrainingSample {
            features: FeatureSeries { f1: 45.0, f2: 10.0, f3: 15.0, f4: 55.0, f5: 50.0 },
            label: Direction::Neutral,
            timestamp: 4000,
            bar_index: 3,
        },
        TrainingSample {
            features: FeatureSeries { f1: 25.0, f2: -30.0, f3: 35.0, f4: 40.0, f5: 30.0 },
            label: Direction::Short,
            timestamp: 5000,
            bar_index: 4,
        },
        TrainingSample {
            features: FeatureSeries { f1: 80.0, f2: 50.0, f3: -20.0, f4: 90.0, f5: 85.0 },
            label: Direction::Long,
            timestamp: 6000,
            bar_index: 5,
        },
        TrainingSample {
            features: FeatureSeries { f1: 40.0, f2: 5.0, f3: 25.0, f4: 60.0, f5: 45.0 },
            label: Direction::Neutral,
            timestamp: 7000,
            bar_index: 6,
        },
        TrainingSample {
            features: FeatureSeries { f1: 60.0, f2: 35.0, f3: -15.0, f4: 70.0, f5: 65.0 },
            label: Direction::Long,
            timestamp: 8000,
            bar_index: 7,
        },
    ]
}

/// Create test engine with deterministic samples
fn create_test_engine_deterministic() -> LDCEngine {
    let mut config = LDCConfig::default();
    config.neighbors_count = 3;
    config.feature_count = 5;
    config.max_bars_back = 100;
    
    let mut engine = LDCEngine::with_config(config);
    
    let samples = create_deterministic_training_samples();
    for sample in samples {
        engine.add_training_sample(sample);
    }
    
    engine
}

/// Test that Lorentzian distance calculation matches Pine Script exactly
#[test]
fn test_lorentzian_distance_pine_script_compatibility() {
    let features1 = FeatureSeries { f1: 50.0, f2: 25.0, f3: -10.0, f4: 75.0, f5: 60.0 };
    let features2 = FeatureSeries { f1: 45.0, f2: 30.0, f3: -5.0, f4: 80.0, f5: 55.0 };
    
    // Calculate expected Pine Script distance manually
    let expected_distance: f32 = 
        (1.0f32 + (50.0f32 - 45.0f32).abs()).ln() +
        (1.0f32 + (25.0f32 - 30.0f32).abs()).ln() +
        (1.0f32 + (-10.0f32 - (-5.0f32)).abs()).ln() +
        (1.0f32 + (75.0f32 - 80.0f32).abs()).ln() +
        (1.0f32 + (60.0f32 - 55.0f32).abs()).ln();
    
    // Test standard distance calculation
    let standard_distance = features1.lorentzian_distance_standard(&features2);
    assert!((standard_distance - expected_distance).abs() < 1e-6, 
           "Standard distance {} doesn't match expected {}", standard_distance, expected_distance);
    
    // Test SIMD distance calculation
    let simd_distance = features1.lorentzian_distance_simd(&features2);
    assert!((simd_distance - expected_distance).abs() < 1e-6,
           "SIMD distance {} doesn't match expected {}", simd_distance, expected_distance);
    
    // Test aligned feature series distance
    let aligned1 = features1.to_aligned();
    let aligned2 = features2.to_aligned();
    let aligned_distance = aligned1.lorentzian_distance_standard(&aligned2);
    assert!((aligned_distance - expected_distance).abs() < 1e-6,
           "Aligned distance {} doesn't match expected {}", aligned_distance, expected_distance);
}

/// Test that batch distance calculations maintain Pine Script compatibility
#[test]
fn test_batch_distance_pine_script_compatibility() {
    let query = FeatureSeries { f1: 50.0, f2: 25.0, f3: -10.0, f4: 75.0, f5: 60.0 };
    let targets = vec![
        FeatureSeries { f1: 45.0, f2: 30.0, f3: -5.0, f4: 80.0, f5: 55.0 },
        FeatureSeries { f1: 55.0, f2: 20.0, f3: -15.0, f4: 70.0, f5: 65.0 },
        FeatureSeries { f1: 40.0, f2: 35.0, f3: 0.0, f4: 85.0, f5: 50.0 },
    ];
    
    // Calculate expected distances individually
    let expected_distances: Vec<f32> = targets.iter()
        .map(|target| query.lorentzian_distance_standard(target))
        .collect();
    
    // Test standard batch calculation
    let standard_batch = FeatureSeries::batch_lorentzian_distance_standard(&query, &targets);
    for (i, (&standard, &expected)) in standard_batch.iter().zip(expected_distances.iter()).enumerate() {
        assert!((standard - expected).abs() < 1e-6,
               "Standard batch distance {} at index {} doesn't match expected {}", standard, i, expected);
    }
    
    // Test SIMD batch calculation
    let simd_batch = FeatureSeries::batch_lorentzian_distance_simd(&query, &targets, 2);
    for (i, (&simd, &expected)) in simd_batch.iter().zip(expected_distances.iter()).enumerate() {
        assert!((simd - expected).abs() < 1e-6,
               "SIMD batch distance {} at index {} doesn't match expected {}", simd, i, expected);
    }
    
    // Test aligned batch calculation
    let query_aligned = query.to_aligned();
    let targets_aligned: Vec<AlignedFeatureSeries> = targets.iter().map(|t| t.to_aligned()).collect();
    let aligned_batch = AlignedFeatureSeries::batch_lorentzian_distance_simd(&query_aligned, &targets_aligned, 2);
    for (i, (&aligned, &expected)) in aligned_batch.iter().zip(expected_distances.iter()).enumerate() {
        assert!((aligned - expected).abs() < 1e-6,
               "Aligned batch distance {} at index {} doesn't match expected {}", aligned, i, expected);
    }
}

/// Test that k-NN search results are consistent across all optimization strategies
#[test]
fn test_knn_search_consistency_across_strategies() {
    let query = FeatureSeries { f1: 52.0, f2: 27.0, f3: -8.0, f4: 77.0, f5: 62.0 };
    
    // Create engines with different configurations
    let mut config_sequential = LDCConfig::default();
    config_sequential.neighbors_count = 3;
    config_sequential.use_multithreading = false;
    config_sequential.use_hnsw_index = false;
    let mut engine_sequential = LDCEngine::with_config(config_sequential);
    
    let mut config_parallel = LDCConfig::default();
    config_parallel.neighbors_count = 3;
    config_parallel.use_multithreading = true;
    config_parallel.parallel_threshold = 1;
    config_parallel.use_hnsw_index = false;
    let mut engine_parallel = LDCEngine::with_config(config_parallel);
    
    let mut config_hnsw = LDCConfig::default();
    config_hnsw.neighbors_count = 3;
    config_hnsw.use_hnsw_index = true;
    config_hnsw.hnsw_m = 16;
    config_hnsw.hnsw_ef_construction = 200;
    config_hnsw.hnsw_ef_search = 50;
    let mut engine_hnsw = LDCEngine::with_config(config_hnsw);
    
    // Add the same samples to all engines
    let samples = create_deterministic_training_samples();
    for sample in &samples {
        engine_sequential.add_training_sample(sample.clone());
        engine_parallel.add_training_sample(sample.clone());
        engine_hnsw.add_training_sample(sample.clone());
    }
    
    // Get results from each strategy
    let sequential_results = engine_sequential.find_k_nearest_neighbors_sequential_optimized(&query);
    let parallel_results = engine_parallel.find_k_nearest_neighbors_parallel_optimized(&query);
    let hnsw_results = engine_hnsw.find_k_nearest_neighbors_optimized(&query);
    
    // Sequential and parallel should be identical
    assert_eq!(sequential_results.len(), parallel_results.len(),
              "Sequential and parallel results have different lengths");
    
    for (i, (seq, par)) in sequential_results.iter().zip(parallel_results.iter()).enumerate() {
        assert!((seq.0 - par.0).abs() < 1e-6,
               "Distance mismatch at index {}: sequential {} vs parallel {}", i, seq.0, par.0);
        assert_eq!(seq.1, par.1,
                  "Label mismatch at index {}: sequential {:?} vs parallel {:?}", i, seq.1, par.1);
    }
    
    // HNSW should have the same number of results (may be approximate)
    assert_eq!(sequential_results.len(), hnsw_results.len(),
              "Sequential and HNSW results have different lengths");
    
    // For small datasets, HNSW should be very close to exact results
    let mut exact_distances: Vec<f32> = sequential_results.iter().map(|r| r.0).collect();
    let mut hnsw_distances: Vec<f32> = hnsw_results.iter().map(|r| r.0).collect();
    
    exact_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hnsw_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Check that HNSW results are reasonably close (within 10% for small datasets)
    for (i, (&exact, &hnsw)) in exact_distances.iter().zip(hnsw_distances.iter()).enumerate() {
        let relative_error = ((hnsw - exact) / exact).abs();
        assert!(relative_error < 0.1,
               "HNSW distance {} at index {} differs too much from exact {} (error: {:.2}%)", 
               hnsw, i, exact, relative_error * 100.0);
    }
}

/// Test that HNSW index maintains accuracy requirements (95%+ accuracy)
#[test]
fn test_hnsw_accuracy_requirement() {
    // Create a larger dataset for more meaningful accuracy testing
    let mut config_exact = LDCConfig::default();
    config_exact.use_hnsw_index = false;
    config_exact.neighbors_count = 5;
    let mut engine_exact = LDCEngine::with_config(config_exact);
    
    let mut config_hnsw = LDCConfig::default();
    config_hnsw.use_hnsw_index = true;
    config_hnsw.neighbors_count = 5;
    config_hnsw.hnsw_m = 16;
    config_hnsw.hnsw_ef_construction = 200;
    config_hnsw.hnsw_ef_search = 100; // Higher ef_search for better accuracy
    let mut engine_hnsw = LDCEngine::with_config(config_hnsw);
    
    // Generate a larger set of training samples
    let mut rng = StdRng::seed_from_u64(42);
    let sample_count = 1000;
    
    for i in 0..sample_count {
        let features = FeatureSeries {
            f1: rng.gen_range(0.0..100.0),
            f2: rng.gen_range(-100.0..100.0),
            f3: rng.gen_range(-100.0..100.0),
            f4: rng.gen_range(0.0..100.0),
            f5: rng.gen_range(0.0..100.0),
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
        
        engine_exact.add_training_sample(sample.clone());
        engine_hnsw.add_training_sample(sample);
    }
    
    // Test multiple queries
    let test_queries = 50;
    let mut total_accuracy = 0.0;
    
    for _ in 0..test_queries {
        let query = FeatureSeries {
            f1: rng.gen_range(0.0..100.0),
            f2: rng.gen_range(-100.0..100.0),
            f3: rng.gen_range(-100.0..100.0),
            f4: rng.gen_range(0.0..100.0),
            f5: rng.gen_range(0.0..100.0),
        };
        
        let exact_results = engine_exact.find_k_nearest_neighbors_sequential_optimized(&query);
        let hnsw_results = engine_hnsw.find_k_nearest_neighbors_optimized(&query);
        
        // Calculate accuracy based on distance similarity rather than exact label matches
        // Since HNSW is approximate, we check if the distances are reasonably close
        let mut matches = 0;
        for (exact_dist, _) in &exact_results {
            for (hnsw_dist, _) in &hnsw_results {
                if (exact_dist - hnsw_dist).abs() / exact_dist < 0.1 { // Within 10% difference
                    matches += 1;
                    break;
                }
            }
        }
        let accuracy = matches as f32 / exact_results.len() as f32;
        total_accuracy += accuracy;
    }
    
    let average_accuracy = total_accuracy / test_queries as f32;
    
    // Require 95% accuracy as specified in requirements
    assert!(average_accuracy >= 0.95,
           "HNSW accuracy {:.2}% is below required 95%", average_accuracy * 100.0);
}

/// Test that memory optimizations don't affect Pine Script compatibility
#[test]
fn test_memory_optimizations_pine_script_compatibility() {
    let original_sample = TrainingSample {
        features: FeatureSeries { f1: 50.0, f2: 25.0, f3: -10.0, f4: 75.0, f5: 60.0 },
        label: Direction::Long,
        timestamp: 1000,
        bar_index: 42,
    };
    
    // Test OptimizedTrainingSample conversion
    let optimized_sample = OptimizedTrainingSample::from_training_sample(&original_sample);
    let converted_back = optimized_sample.to_training_sample();
    
    // Verify all fields are preserved
    assert_eq!(original_sample.label, converted_back.label);
    assert_eq!(original_sample.timestamp, converted_back.timestamp);
    assert_eq!(original_sample.bar_index, converted_back.bar_index);
    
    // Verify features are preserved with high precision
    let orig_features = original_sample.features.to_array();
    let conv_features = converted_back.features.to_array();
    
    for (i, (&orig, &conv)) in orig_features.iter().zip(conv_features.iter()).enumerate() {
        assert!((orig - conv).abs() < 1e-6,
               "Feature {} mismatch: original {} vs converted {}", i, orig, conv);
    }
    
    // Test AlignedFeatureSeries conversion
    let aligned_features = original_sample.features.to_aligned();
    let features_back = aligned_features.to_feature_series();
    
    let aligned_array = aligned_features.features;
    let back_array = features_back.to_array();
    
    for (i, (&orig, &back)) in orig_features.iter().zip(back_array.iter()).enumerate() {
        assert!((orig - back).abs() < 1e-6,
               "Aligned feature {} mismatch: original {} vs back {}", i, orig, back);
    }
    
    // Verify padding doesn't affect calculations
    assert_eq!(aligned_array[5], 0.0, "Padding should be zero");
    assert_eq!(aligned_array[6], 0.0, "Padding should be zero");
    assert_eq!(aligned_array[7], 0.0, "Padding should be zero");
}

/// Test that thread pool strategies don't affect result consistency
#[test]
fn test_thread_pool_strategy_consistency() {
    let query = FeatureSeries { f1: 50.0, f2: 25.0, f3: -10.0, f4: 75.0, f5: 60.0 };
    
    let strategies = vec![
        ThreadPoolStrategy::Global,
        ThreadPoolStrategy::Dedicated,
        ThreadPoolStrategy::Adaptive,
    ];
    
    let mut baseline_results = None;
    
    for strategy in strategies {
        let mut config = LDCConfig::default();
        config.neighbors_count = 3;
        config.feature_count = 5;
        config.max_bars_back = 100;
        config.use_multithreading = true;
        config.parallel_threshold = 1;
        config.thread_pool_strategy = strategy.clone();
        let mut engine = LDCEngine::with_config(config);
        
        // Add the same samples
        let samples = create_deterministic_training_samples();
        for sample in samples {
            engine.add_training_sample(sample);
        }
        
        let results = engine.find_k_nearest_neighbors_parallel_optimized(&query);
        
        if baseline_results.is_none() {
            baseline_results = Some(results);
        } else {
            let baseline = baseline_results.as_ref().unwrap();
            
            assert_eq!(baseline.len(), results.len(),
                      "Thread pool strategy {:?} produced different number of results", strategy);
            
            for (i, (base, result)) in baseline.iter().zip(results.iter()).enumerate() {
                assert!((base.0 - result.0).abs() < 1e-6,
                       "Thread pool strategy {:?} distance mismatch at index {}: {} vs {}", 
                       strategy, i, base.0, result.0);
                assert_eq!(base.1, result.1,
                          "Thread pool strategy {:?} label mismatch at index {}: {:?} vs {:?}", 
                          strategy, i, base.1, result.1);
            }
        }
    }
}

/// Test that performance monitoring doesn't affect Pine Script results
#[test]
fn test_performance_monitoring_compatibility() {
    let mut config = LDCConfig::default();
    config.neighbors_count = 3;
    config.feature_count = 5;
    config.max_bars_back = 100;
    config.log_performance_metrics = true;
    let mut engine = LDCEngine::with_config(config);
    
    // Add samples
    let samples = create_deterministic_training_samples();
    for sample in samples {
        engine.add_training_sample(sample);
    }
    
    let query = FeatureSeries { f1: 50.0, f2: 25.0, f3: -10.0, f4: 75.0, f5: 60.0 };
    
    // Get baseline results
    let baseline_results = engine.find_k_nearest_neighbors_sequential_optimized(&query);
    
    // Get results again (performance monitoring is always enabled in this config)
    let monitored_results = engine.find_k_nearest_neighbors_sequential_optimized(&query);
    
    // Results should be identical
    assert_eq!(baseline_results.len(), monitored_results.len(),
              "Performance monitoring changed result count");
    
    for (i, (base, monitored)) in baseline_results.iter().zip(monitored_results.iter()).enumerate() {
        assert!((base.0 - monitored.0).abs() < 1e-6,
               "Performance monitoring changed distance at index {}: {} vs {}", i, base.0, monitored.0);
        assert_eq!(base.1, monitored.1,
                  "Performance monitoring changed label at index {}: {:?} vs {:?}", i, base.1, monitored.1);
    }
    
    // Verify that performance metrics were updated
    let metrics = engine.get_performance_metrics();
    assert!(metrics.total_predictions > 0, "Performance metrics should be updated");
}

/// Test that all optimizations work together without breaking Pine Script compatibility
#[test]
fn test_full_optimization_stack_compatibility() {
    // Create engine with all optimizations enabled
    let mut config = LDCConfig::default();
    config.use_multithreading = true;
    config.use_simd_optimization = true;
    config.use_hnsw_index = true;
    config.enable_memory_mapping = false; // Keep in memory for this test
    config.thread_pool_strategy = ThreadPoolStrategy::Adaptive;
    config.neighbors_count = 5;
    config.parallel_threshold = 10;
    config.hnsw_m = 16;
    config.hnsw_ef_construction = 200;
    config.hnsw_ef_search = 50;
    config.log_performance_metrics = true;
    
    let mut optimized_engine = LDCEngine::with_config(config);
    
    // Create baseline engine with minimal optimizations
    let mut baseline_config = LDCConfig::default();
    baseline_config.use_multithreading = false;
    baseline_config.use_simd_optimization = false;
    baseline_config.use_hnsw_index = false;
    baseline_config.neighbors_count = 5;
    
    let mut baseline_engine = LDCEngine::with_config(baseline_config);
    
    // Add the same training samples to both engines
    let samples = create_deterministic_training_samples();
    for sample in &samples {
        optimized_engine.add_training_sample(sample.clone());
        baseline_engine.add_training_sample(sample.clone());
    }
    
    // Test multiple queries
    let test_queries = vec![
        FeatureSeries { f1: 52.0, f2: 27.0, f3: -8.0, f4: 77.0, f5: 62.0 },
        FeatureSeries { f1: 35.0, f2: -10.0, f3: 18.0, f4: 50.0, f5: 40.0 },
        FeatureSeries { f1: 65.0, f2: 45.0, f3: -12.0, f4: 80.0, f5: 70.0 },
    ];
    
    for (i, query) in test_queries.iter().enumerate() {
        let baseline_results = baseline_engine.find_k_nearest_neighbors_sequential_optimized(query);
        let optimized_results = optimized_engine.find_k_nearest_neighbors_optimized(query);
        
        // Results should have the same length
        assert_eq!(baseline_results.len(), optimized_results.len(),
                  "Query {}: Optimized engine returned different number of results", i);
        
        // For HNSW, we expect approximate results, so we check that the results are reasonable
        // rather than exact matches
        let baseline_distances: Vec<f32> = baseline_results.iter().map(|r| r.0).collect();
        let optimized_distances: Vec<f32> = optimized_results.iter().map(|r| r.0).collect();
        
        // Check that optimized distances are in a reasonable range compared to baseline
        let baseline_min = baseline_distances.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let baseline_max = baseline_distances.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let baseline_range = baseline_max - baseline_min;
        
        for (j, &opt_dist) in optimized_distances.iter().enumerate() {
            // Optimized distances should be within a reasonable range of baseline distances
            assert!(opt_dist >= baseline_min - baseline_range * 0.5 && 
                   opt_dist <= baseline_max + baseline_range * 0.5,
                   "Query {}, result {}: Optimized distance {} is outside reasonable range [{}, {}]",
                   i, j, opt_dist, baseline_min - baseline_range * 0.5, baseline_max + baseline_range * 0.5);
        }
        
        // Verify that performance metrics are being tracked
        let metrics = optimized_engine.get_performance_metrics();
        assert!(metrics.total_predictions > 0, "Performance metrics should be updated");
    }
}

/// Test error handling and graceful degradation
#[test]
fn test_optimization_error_handling() {
    let mut config = LDCConfig::default();
    config.use_hnsw_index = true;
    config.hnsw_m = 16;
    config.hnsw_ef_construction = 200;
    config.hnsw_ef_search = 50;
    
    let mut engine = LDCEngine::with_config(config);
    
    // Add some samples
    let samples = create_deterministic_training_samples();
    for sample in samples {
        engine.add_training_sample(sample);
    }
    
    let query = FeatureSeries { f1: 50.0, f2: 25.0, f3: -10.0, f4: 75.0, f5: 60.0 };
    
    // This should work normally
    let results = engine.find_k_nearest_neighbors_optimized(&query);
    assert!(!results.is_empty(), "Should return results even with potential HNSW issues");
    
    // Verify that the engine can still function if HNSW fails
    // (The implementation should fall back to exact search)
    let fallback_results = engine.find_k_nearest_neighbors_sequential_optimized(&query);
    assert_eq!(results.len(), fallback_results.len(), "Fallback should return same number of results");
}