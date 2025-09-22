use ldc_engine::{LDCEngine, LDCConfig, FeatureSeries, TrainingSample, Direction};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LDC Engine Performance Monitoring Demo ===\n");
    
    // Create engine with performance logging enabled
    let mut config = LDCConfig::default();
    config.log_performance_metrics = true;
    config.enable_debug_logging = true;
    
    let mut engine = LDCEngine::with_config(config);
    
    // Add some sample training data
    println!("Adding training samples...");
    for i in 0..100 {
        let features = FeatureSeries {
            f1: (i as f32) * 0.1,
            f2: (i as f32) * 0.2,
            f3: (i as f32) * 0.15,
            f4: (i as f32) * 0.25,
            f5: (i as f32) * 0.3,
        };
        
        let sample = TrainingSample {
            features,
            label: if i % 3 == 0 { Direction::Long } else if i % 3 == 1 { Direction::Short } else { Direction::Neutral },
            timestamp: i as i64,
            bar_index: i,
        };
        
        engine.add_training_sample(sample);
    }
    
    println!("Added {} training samples\n", engine.training_samples_count());
    
    // Demonstrate performance monitoring wrapper
    println!("Testing performance monitoring wrapper...");
    
    let result = engine.monitor_performance("demo_operation", 5.0, || {
        // Simulate some work
        std::thread::sleep(Duration::from_millis(2));
        Ok("Operation completed successfully")
    })?;
    
    println!("Operation result: {}\n", result);
    
    // Demonstrate performance degradation detection
    println!("Testing performance degradation detection...");
    
    let _slow_result = engine.monitor_performance("slow_operation", 1.0, || {
        // Simulate slow operation
        std::thread::sleep(Duration::from_millis(10));
        Ok("Slow operation completed")
    })?;
    
    // Add some latency samples for percentile calculation
    println!("\nAdding latency samples for percentile calculation...");
    for i in 1..=50 {
        engine.update_latency_percentiles(i as f64);
    }
    
    // Update some performance metrics
    engine.update_memory_metrics(512, 100);
    engine.update_cpu_metrics(75.0, 80.0);
    engine.update_hnsw_metrics(1000, 95.5);
    
    // Generate and display performance report
    println!("\nGenerating performance report...");
    let report = engine.generate_performance_report();
    
    println!("Overall Performance Score: {:.1}/100", report.overall_score);
    println!("Total Predictions: {}", report.metrics_summary.total_predictions);
    println!("P95 Latency: {:.2}ms", report.metrics_summary.p95_latency_ms);
    println!("P99 Latency: {:.2}ms", report.metrics_summary.p99_latency_ms);
    println!("CPU Utilization: {:.1}%", report.metrics_summary.cpu_utilization_percent);
    println!("Memory Usage: {}MB", report.metrics_summary.memory_usage_mb);
    
    if !report.recommendations.is_empty() {
        println!("\nOptimization Recommendations:");
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!("{}. [{:?}] {}", i + 1, rec.priority, rec.description);
            println!("   Action: {}", rec.action);
        }
    } else {
        println!("\nNo optimization recommendations at this time.");
    }
    
    // Demonstrate automatic optimization
    println!("\nTesting automatic optimization triggers...");
    
    // Simulate performance degradation that triggers optimization
    let _degraded_result = engine.monitor_performance("degraded_knn_search", 1.0, || {
        std::thread::sleep(Duration::from_millis(20)); // Much slower than expected
        Ok("Degraded operation completed")
    })?;
    
    // Log the final performance report
    println!("\nFinal Performance Report:");
    engine.log_performance_report();
    
    println!("\n=== Demo Complete ===");
    
    Ok(())
}