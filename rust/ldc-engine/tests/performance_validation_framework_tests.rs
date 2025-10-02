use ldc_engine::performance_validation::*;
use ldc_engine::*;

/// Test the performance validation framework components
#[test]
fn test_performance_validator_creation_and_configuration() {
    println!("Testing PerformanceValidator creation and configuration...");
    
    // Test default creation
    let validator = PerformanceValidator::new();
    let config = validator.get_config();
    
    assert_eq!(config.target_latency_1k_samples_ms, 0.5);
    assert_eq!(config.target_latency_10k_samples_ms, 1.0);
    assert_eq!(config.target_latency_50k_samples_ms, 5.0);
    assert_eq!(config.target_hnsw_accuracy_percent, 95.0);
    assert_eq!(config.test_iterations, 100);
    assert_eq!(config.warmup_iterations, 10);
    
    // Test custom configuration
    let custom_config = PerformanceTestConfig {
        target_latency_1k_samples_ms: 0.3,
        target_latency_10k_samples_ms: 0.8,
        target_latency_50k_samples_ms: 4.0,
        target_hnsw_accuracy_percent: 90.0,
        test_iterations: 50,
        warmup_iterations: 5,
        ..Default::default()
    };
    
    let custom_validator = PerformanceValidator::with_config(custom_config.clone());
    let retrieved_config = custom_validator.get_config();
    
    assert_eq!(retrieved_config.target_latency_1k_samples_ms, 0.3);
    assert_eq!(retrieved_config.target_latency_10k_samples_ms, 0.8);
    assert_eq!(retrieved_config.target_latency_50k_samples_ms, 4.0);
    assert_eq!(retrieved_config.target_hnsw_accuracy_percent, 90.0);
    
    println!("  ✓ Configuration validation passed");
}

#[test]
fn test_synthetic_dataset_generation() {
    println!("Testing synthetic dataset generation...");
    
    let validator = PerformanceValidator::new();
    let datasets = validator.get_test_datasets();
    
    // Verify we have the expected datasets
    assert_eq!(datasets.len(), 3);
    
    let expected_sizes = [1000, 10000, 50000];
    let expected_names = ["small_1k", "medium_10k", "large_50k"];
    
    for (i, dataset) in datasets.iter().enumerate() {
        assert_eq!(dataset.name, expected_names[i]);
        assert_eq!(dataset.size, expected_sizes[i]);
        assert_eq!(dataset.samples.len(), expected_sizes[i]);
        assert!(!dataset.query_features.is_empty());
        
        println!("  Dataset '{}': {} samples, {} queries", 
                dataset.name, dataset.samples.len(), dataset.query_features.len());
        
        // Verify feature bounds for a sample of data
        let sample_size = dataset.samples.len().min(100);
        for sample in dataset.samples.iter().take(sample_size) {
            // RSI-like feature (0-100)
            assert!(sample.features.f1 >= 0.0 && sample.features.f1 <= 100.0,
                   "f1 out of bounds: {}", sample.features.f1);
            
            // WaveTrend-like feature (-100 to 100)
            assert!(sample.features.f2 >= -100.0 && sample.features.f2 <= 100.0,
                   "f2 out of bounds: {}", sample.features.f2);
            
            // CCI-like feature (-200 to 200)
            assert!(sample.features.f3 >= -200.0 && sample.features.f3 <= 200.0,
                   "f3 out of bounds: {}", sample.features.f3);
            
            // ADX-like feature (0-100)
            assert!(sample.features.f4 >= 0.0 && sample.features.f4 <= 100.0,
                   "f4 out of bounds: {}", sample.features.f4);
            
            // Additional feature (0-100)
            assert!(sample.features.f5 >= 0.0 && sample.features.f5 <= 100.0,
                   "f5 out of bounds: {}", sample.features.f5);
        }
        
        // Verify query features are also within bounds
        for query in &dataset.query_features {
            assert!(query.f1 >= 0.0 && query.f1 <= 100.0);
            assert!(query.f2 >= -100.0 && query.f2 <= 100.0);
            assert!(query.f3 >= -200.0 && query.f3 <= 200.0);
            assert!(query.f4 >= 0.0 && query.f4 <= 100.0);
            assert!(query.f5 >= 0.0 && query.f5 <= 100.0);
        }
    }
    
    println!("  ✓ Dataset generation validation passed");
}

#[test]
fn test_query_performance_validation_small_dataset() {
    println!("Testing query performance validation on small dataset...");
    
    // Create a performance validator with relaxed targets for testing
    let config = PerformanceTestConfig {
        target_latency_1k_samples_ms: 10.0, // Relaxed target for testing
        target_latency_10k_samples_ms: 20.0,
        target_latency_50k_samples_ms: 50.0,
        test_iterations: 10, // Fewer iterations for faster testing
        warmup_iterations: 2,
        ..Default::default()
    };
    
    let validator = PerformanceValidator::with_config(config);
    
    // Create an LDC engine with optimized configuration
    let mut ldc_config = LDCConfig::default();
    ldc_config.max_bars_back = 1000;
    ldc_config.neighbors_count = 5;
    ldc_config.use_multithreading = true;
    ldc_config.use_simd_optimization = true;
    
    let mut engine = LDCEngine::with_config(ldc_config);
    
    // Add training data from the small dataset
    let datasets = validator.get_test_datasets();
    let small_dataset = &datasets[0]; // small_1k dataset
    
    for sample in &small_dataset.samples {
        let _ = engine.add_training_sample(sample.clone());
    }
    
    // Run performance validation
    let result = validator.validate_query_performance(&mut engine)
        .expect("Performance validation should succeed");
    
    // Verify results structure
    assert!(!result.results.is_empty());
    
    for test_case in &result.results {
        println!("  Dataset: {} ({} samples)", test_case.dataset_name, test_case.dataset_size);
        println!("    Average latency: {:.3}ms (target: {:.3}ms)", 
                test_case.avg_latency_ms, test_case.target_latency_ms);
        println!("    P95 latency: {:.3}ms", test_case.p95_latency_ms);
        println!("    P99 latency: {:.3}ms", test_case.p99_latency_ms);
        println!("    Result: {}", if test_case.passed { "PASS" } else { "FAIL" });
        
        // Verify latency values are reasonable
        assert!(test_case.avg_latency_ms > 0.0, "Average latency should be positive");
        assert!(test_case.p95_latency_ms >= test_case.avg_latency_ms, 
               "P95 should be >= average");
        assert!(test_case.p99_latency_ms >= test_case.p95_latency_ms, 
               "P99 should be >= P95");
    }
    
    println!("  ✓ Query performance validation completed");
}

#[test]
fn test_hnsw_accuracy_validation() {
    println!("Testing HNSW accuracy validation...");
    
    // Create a performance validator
    let config = PerformanceTestConfig {
        target_hnsw_accuracy_percent: 80.0, // Relaxed target for testing
        test_iterations: 5, // Fewer iterations for faster testing
        warmup_iterations: 1,
        ..Default::default()
    };
    
    let validator = PerformanceValidator::with_config(config);
    
    // Create an LDC engine with HNSW enabled
    let mut ldc_config = LDCConfig::default();
    ldc_config.max_bars_back = 5000;
    ldc_config.neighbors_count = 5;
    ldc_config.use_hnsw_index = true;
    ldc_config.hnsw_m = 16;
    ldc_config.hnsw_ef_construction = 100;
    ldc_config.hnsw_ef_search = 50;
    
    let mut engine = LDCEngine::with_config(ldc_config);
    
    // Add training data from the medium dataset (10k samples)
    let datasets = validator.get_test_datasets();
    let medium_dataset = &datasets[1]; // medium_10k dataset
    
    // Add a subset of samples for faster testing
    let sample_count = medium_dataset.samples.len().min(2000);
    for sample in medium_dataset.samples.iter().take(sample_count) {
        let _ = engine.add_training_sample(sample.clone());
    }
    
    // Run HNSW accuracy validation
    let result = validator.validate_hnsw_accuracy(&mut engine)
        .expect("HNSW accuracy validation should succeed");
    
    // Verify results structure
    assert!(!result.results.is_empty());
    
    for accuracy_case in &result.results {
        println!("  Dataset: {} ({} samples)", accuracy_case.dataset_name, accuracy_case.dataset_size);
        println!("    Accuracy: {:.1}% (target: {:.1}%)", 
                accuracy_case.accuracy_percent, accuracy_case.target_accuracy_percent);
        println!("    Result: {}", if accuracy_case.passed { "PASS" } else { "FAIL" });
        
        // Verify accuracy values are reasonable
        assert!(accuracy_case.accuracy_percent >= 0.0 && accuracy_case.accuracy_percent <= 100.0,
               "Accuracy should be between 0% and 100%");
    }
    
    println!("  ✓ HNSW accuracy validation completed");
}

#[test]
fn test_performance_test_result_methods() {
    println!("Testing PerformanceTestResult methods...");
    
    let results = vec![
        PerformanceTestCase {
            dataset_name: "test1".to_string(),
            dataset_size: 1000,
            avg_latency_ms: 0.4,
            p95_latency_ms: 0.6,
            p99_latency_ms: 0.8,
            target_latency_ms: 0.5,
            passed: true,
        },
        PerformanceTestCase {
            dataset_name: "test2".to_string(),
            dataset_size: 10000,
            avg_latency_ms: 1.2,
            p95_latency_ms: 1.5,
            p99_latency_ms: 2.0,
            target_latency_ms: 1.0,
            passed: false,
        },
    ];
    
    let test_result = PerformanceTestResult { results };
    
    assert_eq!(test_result.total_count(), 2);
    assert_eq!(test_result.passed_count(), 1);
    assert!(!test_result.all_passed());
    
    println!("  ✓ PerformanceTestResult methods working correctly");
}

#[test]
fn test_hnsw_accuracy_result_methods() {
    println!("Testing HNSWAccuracyResult methods...");
    
    let results = vec![
        HNSWAccuracyCase {
            dataset_name: "test1".to_string(),
            dataset_size: 5000,
            accuracy_percent: 96.0,
            target_accuracy_percent: 95.0,
            passed: true,
        },
        HNSWAccuracyCase {
            dataset_name: "test2".to_string(),
            dataset_size: 10000,
            accuracy_percent: 92.0,
            target_accuracy_percent: 95.0,
            passed: false,
        },
    ];
    
    let accuracy_result = HNSWAccuracyResult { results };
    
    assert_eq!(accuracy_result.total_count(), 2);
    assert_eq!(accuracy_result.passed_count(), 1);
    assert!(!accuracy_result.all_passed());
    
    println!("  ✓ HNSWAccuracyResult methods working correctly");
}

#[test]
fn test_percentile_calculation_edge_cases() {
    println!("Testing percentile calculation edge cases...");
    
    // Test with single value
    let single_value = vec![5.0];
    let p50 = PerformanceValidator::calculate_percentile(&single_value, 50.0);
    assert_eq!(p50, 5.0);
    
    // Test with two values
    let two_values = vec![1.0, 2.0];
    let p50 = PerformanceValidator::calculate_percentile(&two_values, 50.0);
    assert_eq!(p50, 1.0); // Should return first value for 50th percentile
    
    // Test with many values
    let many_values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let p25 = PerformanceValidator::calculate_percentile(&many_values, 25.0);
    let p75 = PerformanceValidator::calculate_percentile(&many_values, 75.0);
    
    assert!((p25 - 25.0).abs() < 1.0); // Approximately 25th value
    assert!((p75 - 75.0).abs() < 1.0); // Approximately 75th value
    
    println!("  ✓ Percentile calculation edge cases handled correctly");
}

#[test]
fn test_knn_overlap_calculation() {
    println!("Testing k-NN overlap calculation...");
    
    // Test identical results
    let identical_exact = vec![
        (1.0, Direction::Long),
        (2.0, Direction::Short),
        (3.0, Direction::Neutral),
    ];
    let identical_hnsw = identical_exact.clone();
    
    let overlap = PerformanceValidator::calculate_knn_overlap(&identical_exact, &identical_hnsw);
    assert_eq!(overlap, 3);
    
    // Test partial overlap
    let exact = vec![
        (1.0, Direction::Long),
        (2.0, Direction::Short),
        (3.0, Direction::Neutral),
    ];
    let hnsw = vec![
        (1.1, Direction::Long),   // Same direction, different distance
        (2.1, Direction::Long),   // Different direction
        (4.0, Direction::Neutral), // Same direction, different distance
    ];
    
    let overlap = PerformanceValidator::calculate_knn_overlap(&exact, &hnsw);
    assert_eq!(overlap, 2); // Long and Neutral directions match
    
    // Test no overlap
    let no_overlap_exact = vec![(1.0, Direction::Long)];
    let no_overlap_hnsw = vec![(2.0, Direction::Short)];
    
    let overlap = PerformanceValidator::calculate_knn_overlap(&no_overlap_exact, &no_overlap_hnsw);
    assert_eq!(overlap, 0);
    
    println!("  ✓ k-NN overlap calculation working correctly");
}

#[test]
fn test_performance_validation_with_different_engine_configurations() {
    println!("Testing performance validation with different engine configurations...");
    
    let validator = PerformanceValidator::new();
    
    // Test configurations
    let configs = vec![
        ("basic", LDCConfig {
            max_bars_back: 1000,
            neighbors_count: 5,
            use_multithreading: false,
            use_simd_optimization: false,
            use_hnsw_index: false,
            ..Default::default()
        }),
        ("optimized", LDCConfig {
            max_bars_back: 1000,
            neighbors_count: 5,
            use_multithreading: true,
            use_simd_optimization: true,
            use_hnsw_index: false,
            parallel_threshold: 100,
            ..Default::default()
        }),
    ];
    
    for (config_name, ldc_config) in configs {
        println!("  Testing configuration: {}", config_name);
        
        let mut engine = LDCEngine::with_config(ldc_config);
        
        // Add some training data
        let datasets = validator.get_test_datasets();
        let small_dataset = &datasets[0];
        
        // Add a subset for faster testing
        for sample in small_dataset.samples.iter().take(500) {
            let _ = engine.add_training_sample(sample.clone());
        }
        
        // Run a quick performance test
        let config = PerformanceTestConfig {
            test_iterations: 5,
            warmup_iterations: 1,
            target_latency_1k_samples_ms: 50.0, // Very relaxed for testing
            ..Default::default()
        };
        
        let test_validator = PerformanceValidator::with_config(config);
        let result = test_validator.validate_query_performance(&mut engine)
            .expect("Performance validation should succeed");
        
        assert!(!result.results.is_empty());
        
        for test_case in &result.results {
            if test_case.dataset_size <= 1000 {
                println!("    {} - Latency: {:.3}ms", config_name, test_case.avg_latency_ms);
                assert!(test_case.avg_latency_ms > 0.0);
            }
        }
    }
    
    println!("  ✓ Performance validation with different configurations completed");
}