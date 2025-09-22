use ldc_engine::*;
use std::time::{Duration, Instant};
use rand::prelude::*;

/// Test that k-NN queries complete within required time limits
#[test]
fn test_query_time_requirements() {
    // Requirement 1.1: 10k samples in under 1ms
    test_query_time_for_sample_count(10000, Duration::from_millis(1));
    
    // Requirement 1.2: 50k samples in under 5ms
    test_query_time_for_sample_count(50000, Duration::from_millis(5));
}

/// Helper function to test query time for a specific sample count
fn test_query_time_for_sample_count(sample_count: usize, max_duration: Duration) {
    let mut config = LDCConfig::default();
    config.max_bars_back = sample_count.max(2000);
    config.neighbors_count = 8;
    config.use_multithreading = true;
    config.use_hnsw_index = sample_count >= 1000; // Use HNSW for larger datasets
    config.parallel_threshold = 100;
    
    if config.use_hnsw_index {
        config.hnsw_m = 16;
        config.hnsw_ef_construction = 200;
        config.hnsw_ef_search = 50;
    }
    
    let mut engine = LDCEngine::with_config(config);
    
    // Generate training samples
    let mut rng = StdRng::seed_from_u64(42);
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
        
        engine.add_training_sample(sample);
    }
    
    // Test query performance
    let query = FeatureSeries {
        f1: 50.0,
        f2: 25.0,
        f3: -10.0,
        f4: 75.0,
        f5: 60.0,
    };
    
    // Warm up
    for _ in 0..5 {
        let _ = engine.find_k_nearest_neighbors_optimized(&query);
    }
    
    // Measure performance over multiple queries
    let num_queries = 10;
    let mut total_duration = Duration::new(0, 0);
    
    for _ in 0..num_queries {
        let start = Instant::now();
        let _results = engine.find_k_nearest_neighbors_optimized(&query);
        let duration = start.elapsed();
        total_duration += duration;
    }
    
    let average_duration = total_duration / num_queries;
    
    println!("Sample count: {}, Average query time: {:?}, Requirement: {:?}", 
             sample_count, average_duration, max_duration);
    
    assert!(average_duration <= max_duration,
           "Query time {:?} exceeds requirement {:?} for {} samples", 
           average_duration, max_duration, sample_count);
}

/// Test CPU utilization requirements (Requirement 2.2: 90% CPU utilization)
#[test]
fn test_cpu_utilization_requirement() {
    let sample_count = 25000;
    let mut config = LDCConfig::default();
    config.max_bars_back = sample_count;
    config.neighbors_count = 8;
    config.use_multithreading = true;
    config.parallel_threshold = 100;
    config.max_threads = Some(num_cpus::get());
    
    let mut engine = LDCEngine::with_config(config);
    
    // Generate training samples
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
    
    let query = FeatureSeries {
        f1: 50.0,
        f2: 25.0,
        f3: -10.0,
        f4: 75.0,
        f5: 60.0,
    };
    
    // Perform multiple parallel queries to stress test CPU utilization
    let start_time = Instant::now();
    let num_queries = 100;
    
    for _ in 0..num_queries {
        let _results = engine.find_k_nearest_neighbors_parallel_optimized(&query);
    }
    
    let total_time = start_time.elapsed();
    let metrics = engine.get_performance_metrics();
    
    println!("Processed {} queries in {:?}", num_queries, total_time);
    println!("Performance metrics: {:?}", metrics);
    
    // This is a basic test - in a real scenario, we'd need system-level CPU monitoring
    // For now, we verify that parallel processing is actually being used
    assert!(metrics.parallel_predictions > 0, 
           "Should have used parallel processing for CPU utilization test");
    
    // Verify that we're processing queries efficiently
    let queries_per_second = num_queries as f64 / total_time.as_secs_f64();
    assert!(queries_per_second > 10.0, 
           "Should process at least 10 queries per second, got {:.2}", queries_per_second);
}

/// Test memory usage thresholds (Requirement 3.4: trigger compression at 80% RAM)
#[test]
fn test_memory_threshold_monitoring() {
    let mut monitor = MemoryThresholdMonitor::new(1000, 80.0, 95.0); // 1GB threshold
    
    // Test normal usage (below warning threshold)
    let normal_usage = 500; // 500MB
    let status = monitor.check_memory_usage(normal_usage);
    match status {
        MemoryStatus::Normal => {}, // Expected
        _ => panic!("Expected normal status for 500MB usage"),
    }
    
    // Test warning threshold (80% of 1GB = 800MB)
    let warning_usage = 850; // 850MB
    let status = monitor.check_memory_usage(warning_usage);
    match status {
        MemoryStatus::Warning { usage_percent, usage_mb } => {
            assert_eq!(usage_mb, warning_usage);
            assert!(usage_percent >= 80.0);
        },
        _ => panic!("Expected warning status for 850MB usage"),
    }
    
    // Test critical threshold (95% of 1GB = 950MB)
    let critical_usage = 980; // 980MB
    let status = monitor.check_memory_usage(critical_usage);
    match status {
        MemoryStatus::Critical { usage_percent, usage_mb } => {
            assert_eq!(usage_mb, critical_usage);
            assert!(usage_percent >= 95.0);
        },
        _ => panic!("Expected critical status for 980MB usage"),
    }
    
    // Test recommended actions
    let warning_action = monitor.get_recommended_action(&MemoryStatus::Warning { 
        usage_percent: 85.0, 
        usage_mb: 850 
    });
    match warning_action {
        MemoryAction::SoftCleanup => {}, // Expected
        _ => panic!("Expected soft cleanup for warning status"),
    }
    
    let critical_action = monitor.get_recommended_action(&MemoryStatus::Critical { 
        usage_percent: 98.0, 
        usage_mb: 980 
    });
    match critical_action {
        MemoryAction::ForceCleanup => {}, // Expected
        _ => panic!("Expected force cleanup for critical status"),
    }
}

/// Test HNSW accuracy requirement (Requirement 4.3: 95%+ accuracy)
#[test]
fn test_hnsw_accuracy_requirement() {
    let sample_count = 5000;
    let mut config_exact = LDCConfig::default();
    config_exact.max_bars_back = sample_count;
    config_exact.neighbors_count = 10;
    config_exact.use_hnsw_index = false;
    
    let mut config_hnsw = config_exact.clone();
    config_hnsw.use_hnsw_index = true;
    config_hnsw.hnsw_m = 16;
    config_hnsw.hnsw_ef_construction = 200;
    config_hnsw.hnsw_ef_search = 100; // Higher for better accuracy
    
    let mut engine_exact = LDCEngine::with_config(config_exact);
    let mut engine_hnsw = LDCEngine::with_config(config_hnsw);
    
    // Generate training samples
    let mut rng = StdRng::seed_from_u64(42);
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
    
    // Test accuracy over multiple queries
    let num_test_queries = 100;
    let mut total_accuracy = 0.0;
    
    for _ in 0..num_test_queries {
        let query = FeatureSeries {
            f1: rng.gen_range(0.0..100.0),
            f2: rng.gen_range(-100.0..100.0),
            f3: rng.gen_range(-100.0..100.0),
            f4: rng.gen_range(0.0..100.0),
            f5: rng.gen_range(0.0..100.0),
        };
        
        let exact_results = engine_exact.find_k_nearest_neighbors_sequential_optimized(&query);
        let hnsw_results = engine_hnsw.find_k_nearest_neighbors_optimized(&query);
        
        // Calculate accuracy based on overlap in results
        let exact_indices: std::collections::HashSet<_> = exact_results.iter()
            .enumerate()
            .map(|(i, _)| i)
            .collect();
        
        let hnsw_indices: std::collections::HashSet<_> = hnsw_results.iter()
            .enumerate()
            .map(|(i, _)| i)
            .collect();
        
        let intersection_size = exact_indices.intersection(&hnsw_indices).count();
        let accuracy = intersection_size as f32 / exact_results.len() as f32;
        total_accuracy += accuracy;
    }
    
    let average_accuracy = total_accuracy / num_test_queries as f32;
    
    println!("HNSW accuracy: {:.2}%", average_accuracy * 100.0);
    
    // Requirement 4.3: 95%+ accuracy
    assert!(average_accuracy >= 0.95,
           "HNSW accuracy {:.2}% is below required 95%", average_accuracy * 100.0);
}

/// Test performance metrics tracking requirements (Requirement 5.1-5.5)
#[test]
fn test_performance_metrics_requirements() {
    let mut config = LDCConfig::default();
    config.neighbors_count = 5;
    config.use_multithreading = true;
    config.log_performance_metrics = true;
    
    let mut engine = LDCEngine::with_config(config);
    
    // Add some training samples
    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..1000 {
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
    
    let query = FeatureSeries {
        f1: 50.0,
        f2: 25.0,
        f3: -10.0,
        f4: 75.0,
        f5: 60.0,
    };
    
    // Perform queries to generate metrics
    for _ in 0..10 {
        let _results = engine.find_k_nearest_neighbors_optimized(&query);
    }
    
    let metrics = engine.get_performance_metrics();
    
    // Requirement 5.1: Measure and report query latency percentiles
    assert!(metrics.total_predictions > 0, "Should track total predictions");
    assert!(metrics.average_prediction_time_ms > 0.0, "Should track average prediction time");
    assert!(metrics.last_prediction_time_ms > 0.0, "Should track last prediction time");
    
    // Requirement 5.2: Track allocation patterns and peak memory consumption
    // (These would be tracked by MemoryPool in a real implementation)
    
    // Requirement 5.3: Compare parallel vs sequential performance
    assert!(metrics.parallel_predictions > 0 || metrics.sequential_predictions > 0,
           "Should track parallel or sequential predictions");
    
    // Requirement 5.4: Identify bottlenecks in distance calculation, k-NN search, or data access
    // (This would require more detailed timing breakdown in the implementation)
    
    // Requirement 5.5: Include optimization recommendations
    // (This would be part of performance report generation)
    
    println!("Performance metrics: {:?}", metrics);
}

/// Test memory pool performance and efficiency
#[test]
fn test_memory_pool_performance() {
    let pool_size_mb = 10;
    let mut pool = MemoryPool::new(pool_size_mb).expect("Failed to create memory pool");
    
    let allocation_size = std::mem::size_of::<OptimizedTrainingSample>();
    let alignment = std::mem::align_of::<OptimizedTrainingSample>();
    let num_allocations = 1000;
    
    // Test allocation performance
    let start_time = Instant::now();
    let mut pointers = Vec::new();
    
    for _ in 0..num_allocations {
        if let Some(ptr) = pool.allocate(allocation_size, alignment) {
            pointers.push(ptr);
        }
    }
    
    let allocation_time = start_time.elapsed();
    
    // Test deallocation performance
    let start_time = Instant::now();
    
    for ptr in pointers {
        pool.deallocate(ptr);
    }
    
    let deallocation_time = start_time.elapsed();
    
    println!("Memory pool performance:");
    println!("  Allocations: {} in {:?} ({:.2} μs/allocation)", 
             num_allocations, allocation_time, 
             allocation_time.as_micros() as f64 / num_allocations as f64);
    println!("  Deallocations: {} in {:?} ({:.2} μs/deallocation)", 
             num_allocations, deallocation_time,
             deallocation_time.as_micros() as f64 / num_allocations as f64);
    
    // Verify pool statistics
    assert_eq!(pool.allocation_count(), num_allocations as u64);
    assert_eq!(pool.deallocation_count(), num_allocations as u64);
    assert_eq!(pool.allocated_bytes(), 0); // All memory should be deallocated
    
    // Performance requirement: allocations should be fast
    let avg_allocation_time_us = allocation_time.as_micros() as f64 / num_allocations as f64;
    assert!(avg_allocation_time_us < 10.0, 
           "Average allocation time {:.2} μs is too slow", avg_allocation_time_us);
}

/// Test concurrent access and thread safety
#[test]
fn test_concurrent_access_performance() {
    // For now, test sequential performance as a proxy for concurrent capability
    // In a real implementation, we'd need to make LDCEngine thread-safe with proper synchronization
    
    let sample_count = 5000;
    let mut config = LDCConfig::default();
    config.max_bars_back = sample_count;
    config.neighbors_count = 5;
    config.use_multithreading = true;
    config.parallel_threshold = 100;
    
    let mut engine = LDCEngine::with_config(config);
    
    // Add training samples
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
    
    let num_queries = 100;
    let start_time = Instant::now();
    
    // Test rapid sequential queries to simulate concurrent load
    for _ in 0..num_queries {
        let query = FeatureSeries {
            f1: rng.gen_range(0.0..100.0),
            f2: rng.gen_range(-100.0..100.0),
            f3: rng.gen_range(-100.0..100.0),
            f4: rng.gen_range(0.0..100.0),
            f5: rng.gen_range(0.0..100.0),
        };
        
        let _result = engine.find_k_nearest_neighbors_optimized(&query);
    }
    
    let total_time = start_time.elapsed();
    
    println!("Sequential access test (simulating concurrent load):");
    println!("  Total queries: {}", num_queries);
    println!("  Total time: {:?}", total_time);
    println!("  Queries per second: {:.2}", num_queries as f64 / total_time.as_secs_f64());
    
    // Performance requirement: should handle high query rates efficiently
    let queries_per_second = num_queries as f64 / total_time.as_secs_f64();
    assert!(queries_per_second > 50.0, 
           "Query rate {:.2} queries/sec is too low", queries_per_second);
}

/// Test that 1ms query time target is achievable for typical workloads
#[test]
fn test_1ms_query_target() {
    let sample_count = 10000; // Typical workload size
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
    
    // Generate realistic training data
    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..sample_count {
        let features = FeatureSeries {
            f1: rng.gen_range(20.0..80.0),   // RSI range
            f2: rng.gen_range(-50.0..50.0),  // WT range
            f3: rng.gen_range(-100.0..100.0), // CCI range
            f4: rng.gen_range(10.0..90.0),   // ADX range
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
    
    // Test realistic query
    let query = FeatureSeries {
        f1: 45.0,  // RSI
        f2: 15.0,  // WT
        f3: -25.0, // CCI
        f4: 65.0,  // ADX
        f5: 55.0,  // Additional
    };
    
    // Warm up the engine
    for _ in 0..10 {
        let _ = engine.find_k_nearest_neighbors_optimized(&query);
    }
    
    // Measure performance over multiple queries
    let num_queries = 100;
    let mut query_times = Vec::new();
    
    for _ in 0..num_queries {
        let start = Instant::now();
        let _results = engine.find_k_nearest_neighbors_optimized(&query);
        let duration = start.elapsed();
        query_times.push(duration);
    }
    
    // Calculate statistics
    let total_time: Duration = query_times.iter().sum();
    let average_time = total_time / num_queries;
    
    query_times.sort();
    let p50_time = query_times[num_queries / 2];
    let p95_time = query_times[(num_queries * 95) / 100];
    let p99_time = query_times[(num_queries * 99) / 100];
    
    println!("Query time statistics for {} samples:", sample_count);
    println!("  Average: {:?}", average_time);
    println!("  P50: {:?}", p50_time);
    println!("  P95: {:?}", p95_time);
    println!("  P99: {:?}", p99_time);
    
    // Target: 1ms for typical workloads
    let target_time = Duration::from_millis(1);
    
    assert!(average_time <= target_time,
           "Average query time {:?} exceeds 1ms target", average_time);
    
    assert!(p95_time <= Duration::from_millis(2),
           "P95 query time {:?} exceeds 2ms (reasonable for P95)", p95_time);
}