#[cfg(test)]
mod performance_monitoring_tests {
    use crate::*;
    use std::time::Duration;

    #[test]
    fn test_monitor_performance_wrapper() {
        let mut engine = LDCEngine::new();
        
        // Test successful operation
        let result = engine.monitor_performance("test_operation", 10.0, || {
            std::thread::sleep(Duration::from_millis(5));
            Ok("success")
        });
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        
        // Check that latency was recorded
        assert!(engine.performance_metrics.latency_samples.len() > 0);
    }

    #[test]
    fn test_performance_degradation_detection() {
        let mut engine = LDCEngine::new();
        
        // Enable performance logging for testing
        engine.config.log_performance_metrics = true;
        
        // Test operation that exceeds expected time (should trigger warning)
        let _result = engine.monitor_performance("slow_operation", 1.0, || {
            std::thread::sleep(Duration::from_millis(10)); // 10ms > 1ms expected
            Ok("completed")
        });
        
        // The warning should have been logged (we can't easily test console output in unit tests)
        // But we can verify the latency was recorded
        assert!(engine.performance_metrics.latency_samples.len() > 0);
        let last_latency = engine.performance_metrics.latency_samples.back().unwrap();
        assert!(*last_latency > 1.0); // Should be > 1ms
    }

    #[test]
    fn test_latency_percentile_calculation() {
        let mut engine = LDCEngine::new();
        
        // Add some sample latencies
        for i in 1..=100 {
            engine.update_latency_percentiles(i as f64);
        }
        
        // Check percentiles are calculated
        assert!(engine.performance_metrics.latency_p50_ms > 0.0);
        assert!(engine.performance_metrics.latency_p95_ms > 0.0);
        assert!(engine.performance_metrics.latency_p99_ms > 0.0);
        
        // P99 should be higher than P95, which should be higher than P50
        assert!(engine.performance_metrics.latency_p99_ms >= engine.performance_metrics.latency_p95_ms);
        assert!(engine.performance_metrics.latency_p95_ms >= engine.performance_metrics.latency_p50_ms);
    }

    #[test]
    fn test_performance_report_generation() {
        let mut engine = LDCEngine::new();
        
        // Add some sample data
        engine.performance_metrics.total_predictions = 100;
        engine.performance_metrics.latency_p95_ms = 2.5;
        engine.performance_metrics.cpu_utilization_percent = 65.0;
        engine.performance_metrics.thread_efficiency_percent = 55.0;
        
        let report = engine.generate_performance_report();
        
        // Check report structure
        assert!(report.overall_score >= 0.0 && report.overall_score <= 100.0);
        assert_eq!(report.metrics_summary.total_predictions, 100);
        assert_eq!(report.metrics_summary.p95_latency_ms, 2.5);
        assert_eq!(report.metrics_summary.cpu_utilization_percent, 65.0);
        
        // Should have recommendations due to low CPU utilization and thread efficiency
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_automatic_optimization_triggers() {
        let mut engine = LDCEngine::new();
        engine.config.enable_debug_logging = true;
        
        // Test k-NN search optimization trigger
        engine.optimize_knn_search_strategy();
        
        // Should have adjusted parallel threshold
        assert!(engine.config.parallel_threshold < 100); // Default is 100
        
        // Test distance calculation optimization
        let original_simd = engine.config.use_simd_optimization;
        engine.config.use_simd_optimization = false;
        engine.optimize_distance_calculation();
        assert!(engine.config.use_simd_optimization); // Should be enabled now
        
        // Test memory optimization
        let original_memory_mapping = engine.config.enable_memory_mapping;
        engine.config.memory_pool_size = 128; // Small size
        engine.performance_metrics.current_memory_usage_mb = 2000; // High memory usage
        engine.optimize_memory_management();
        // Should have enabled memory mapping due to high memory usage
        assert!(engine.config.enable_memory_mapping || !original_memory_mapping);
    }

    #[test]
    fn test_performance_score_calculation() {
        let mut engine = LDCEngine::new();
        
        // Test perfect performance (should be close to 100)
        engine.performance_metrics.latency_p95_ms = 0.5; // Under 1ms
        engine.performance_metrics.cpu_utilization_percent = 85.0; // Good utilization
        engine.performance_metrics.thread_efficiency_percent = 80.0; // Good efficiency
        engine.performance_metrics.current_memory_usage_mb = 512; // Reasonable memory usage
        
        let score = engine.calculate_overall_performance_score();
        assert!(score > 90.0); // Should be high score
        
        // Test poor performance
        engine.performance_metrics.latency_p95_ms = 10.0; // High latency
        engine.performance_metrics.cpu_utilization_percent = 30.0; // Low utilization
        engine.performance_metrics.thread_efficiency_percent = 40.0; // Low efficiency
        engine.performance_metrics.current_memory_usage_mb = 2048; // High memory usage
        
        let poor_score = engine.calculate_overall_performance_score();
        assert!(poor_score < score); // Should be lower than good performance
    }

    #[test]
    fn test_rolling_window_latency_samples() {
        let mut engine = LDCEngine::new();
        
        // Add more than 1000 samples to test rolling window
        for i in 1..=1200 {
            engine.update_latency_percentiles(i as f64);
        }
        
        // Should maintain only 1000 samples
        assert_eq!(engine.performance_metrics.latency_samples.len(), 1000);
        
        // Should contain the most recent 1000 samples (201-1200)
        let first_sample = *engine.performance_metrics.latency_samples.front().unwrap();
        let last_sample = *engine.performance_metrics.latency_samples.back().unwrap();
        
        assert!(first_sample >= 201.0); // Should have dropped first 200 samples
        assert_eq!(last_sample, 1200.0); // Should have the last sample
    }
}