use ldc_engine::performance_validation::*;
use ldc_engine::*;

/// Demonstration of the performance validation framework
fn main() -> anyhow::Result<()> {
    println!("=== LDC Engine Performance Validation Framework Demo ===\n");
    
    // Create a performance validator with default configuration
    let validator = PerformanceValidator::new();
    let config = validator.get_config();
    
    println!("Performance Test Configuration:");
    println!("  Target latency (1k samples): {:.1}ms", config.target_latency_1k_samples_ms);
    println!("  Target latency (10k samples): {:.1}ms", config.target_latency_10k_samples_ms);
    println!("  Target latency (50k samples): {:.1}ms", config.target_latency_50k_samples_ms);
    println!("  Target HNSW accuracy: {:.1}%", config.target_hnsw_accuracy_percent);
    println!("  Test iterations: {}", config.test_iterations);
    println!("  Warmup iterations: {}\n", config.warmup_iterations);
    
    // Display test datasets
    let datasets = validator.get_test_datasets();
    println!("Available Test Datasets:");
    for dataset in datasets {
        println!("  {} - {} samples, {} query features", 
                dataset.name, dataset.size, dataset.query_features.len());
    }
    println!();
    
    // Create an LDC engine with optimized configuration
    let mut ldc_config = LDCConfig::default();
    ldc_config.max_bars_back = 2000;
    ldc_config.neighbors_count = 8;
    ldc_config.use_multithreading = true;
    ldc_config.use_simd_optimization = true;
    ldc_config.parallel_threshold = 100;
    
    let mut engine = LDCEngine::with_config(ldc_config);
    
    // Add training data from the small dataset for demonstration
    println!("Loading training data...");
    let small_dataset = &datasets[0]; // small_1k dataset
    
    for (i, sample) in small_dataset.samples.iter().enumerate() {
        if let Err(e) = engine.add_training_sample(sample.clone()) {
            println!("Warning: Failed to add sample {}: {}", i, e);
        }
    }
    
    println!("Added {} training samples\n", small_dataset.samples.len());
    
    // Run query performance validation
    println!("=== Query Performance Validation ===");
    
    // Create a validator with relaxed targets for demonstration
    let demo_config = PerformanceTestConfig {
        target_latency_1k_samples_ms: 10.0, // Relaxed for demo
        target_latency_10k_samples_ms: 20.0,
        target_latency_50k_samples_ms: 50.0,
        test_iterations: 20, // Fewer iterations for faster demo
        warmup_iterations: 3,
        ..Default::default()
    };
    
    let demo_validator = PerformanceValidator::with_config(demo_config);
    
    match demo_validator.validate_query_performance(&mut engine) {
        Ok(result) => {
            println!("Performance Test Results:");
            for test_case in &result.results {
                if test_case.dataset_size <= 1000 { // Only show small dataset results for demo
                    println!("  Dataset: {} ({} samples)", test_case.dataset_name, test_case.dataset_size);
                    println!("    Average latency: {:.3}ms (target: {:.3}ms)", 
                            test_case.avg_latency_ms, test_case.target_latency_ms);
                    println!("    P95 latency: {:.3}ms", test_case.p95_latency_ms);
                    println!("    P99 latency: {:.3}ms", test_case.p99_latency_ms);
                    println!("    Result: {}", if test_case.passed { "✓ PASS" } else { "✗ FAIL" });
                }
            }
            
            println!("\nOverall Performance Summary:");
            println!("  Total tests: {}", result.total_count());
            println!("  Passed: {}", result.passed_count());
            println!("  Failed: {}", result.total_count() - result.passed_count());
            println!("  Success rate: {:.1}%", 
                    (result.passed_count() as f64 / result.total_count() as f64) * 100.0);
        }
        Err(e) => {
            println!("Performance validation failed: {}", e);
        }
    }
    
    println!("\n=== HNSW Accuracy Validation ===");
    
    // Enable HNSW for accuracy testing
    if let Ok(config) = engine.get_config_mut() {
        config.use_hnsw_index = true;
        config.hnsw_m = 16;
        config.hnsw_ef_construction = 100;
        config.hnsw_ef_search = 50;
    }
    
    match demo_validator.validate_hnsw_accuracy(&mut engine) {
        Ok(result) => {
            if result.results.is_empty() {
                println!("No HNSW accuracy tests run (dataset too small or HNSW not available)");
            } else {
                println!("HNSW Accuracy Test Results:");
                for accuracy_case in &result.results {
                    println!("  Dataset: {} ({} samples)", accuracy_case.dataset_name, accuracy_case.dataset_size);
                    println!("    Accuracy: {:.1}% (target: {:.1}%)", 
                            accuracy_case.accuracy_percent, accuracy_case.target_accuracy_percent);
                    println!("    Result: {}", if accuracy_case.passed { "✓ PASS" } else { "✗ FAIL" });
                }
                
                println!("\nHNSW Accuracy Summary:");
                println!("  Total tests: {}", result.total_count());
                println!("  Passed: {}", result.passed_count());
                println!("  Success rate: {:.1}%", 
                        if result.total_count() > 0 {
                            (result.passed_count() as f64 / result.total_count() as f64) * 100.0
                        } else {
                            0.0
                        });
            }
        }
        Err(e) => {
            println!("HNSW accuracy validation failed: {}", e);
        }
    }
    
    println!("\n=== Utility Functions Demo ===");
    
    // Demonstrate percentile calculation
    let sample_latencies = vec![0.5, 0.8, 1.2, 0.9, 1.5, 0.7, 1.1, 0.6, 1.3, 0.4];
    let p50 = PerformanceValidator::calculate_percentile(&sample_latencies, 50.0);
    let p95 = PerformanceValidator::calculate_percentile(&sample_latencies, 95.0);
    let p99 = PerformanceValidator::calculate_percentile(&sample_latencies, 99.0);
    
    println!("Sample latencies: {:?}", sample_latencies);
    println!("  P50 (median): {:.3}ms", p50);
    println!("  P95: {:.3}ms", p95);
    println!("  P99: {:.3}ms", p99);
    
    // Demonstrate k-NN overlap calculation
    let exact_results = vec![
        (1.0, Direction::Long),
        (2.0, Direction::Short),
        (3.0, Direction::Neutral),
    ];
    
    let hnsw_results = vec![
        (1.1, Direction::Long),   // Same direction
        (2.1, Direction::Long),   // Different direction
        (4.0, Direction::Neutral), // Same direction
    ];
    
    let overlap = PerformanceValidator::calculate_knn_overlap(&exact_results, &hnsw_results);
    println!("\nk-NN Overlap Calculation:");
    println!("  Exact results: {:?}", exact_results);
    println!("  HNSW results: {:?}", hnsw_results);
    println!("  Overlap count: {} out of {}", overlap, exact_results.len());
    println!("  Accuracy: {:.1}%", (overlap as f64 / exact_results.len() as f64) * 100.0);
    
    println!("\n=== Demo Complete ===");
    println!("The performance validation framework provides comprehensive testing");
    println!("capabilities for the LDC engine, including:");
    println!("  • Query latency validation with configurable targets");
    println!("  • HNSW accuracy validation against exact search");
    println!("  • Synthetic dataset generation for consistent testing");
    println!("  • Statistical analysis utilities (percentiles, overlaps)");
    println!("  • Detailed reporting with pass/fail status");
    
    Ok(())
}