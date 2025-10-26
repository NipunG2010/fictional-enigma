//! Integration tests for signal emission metrics
//!
//! These tests verify that the metrics collection and export functionality
//! works correctly in isolation from the rest of the signal emission system.

use signal_fusion::emission::{SignalEmissionMetrics, MetricsTimer};

#[test]
fn test_metrics_creation_and_basic_operations() {
    let metrics = SignalEmissionMetrics::new().expect("Failed to create metrics");
    
    // Test signal publishing metrics
    metrics.record_signal_published("BTCUSDT", "redis", "BUY");
    metrics.record_signal_published("ETHUSDT", "kafka", "SELL");
    
    // Test validation error metrics
    metrics.record_validation_error("BTCUSDT", "invalid_strength", "strength");
    metrics.record_validation_error("ETHUSDT", "invalid_side", "side");
    
    // Test publisher error metrics
    metrics.record_publisher_error("redis", "connection_timeout");
    metrics.record_publisher_error("kafka", "authentication_failed");
    
    // Test buffer metrics
    metrics.update_buffer_size(50, 100);
    metrics.record_buffer_operation("push");
    metrics.record_buffer_operation("pop");
    metrics.record_buffer_overflow();
    
    // Test latency metrics
    metrics.record_emission_latency("BTCUSDT", "redis", 0.025);
    metrics.record_validation_latency(0.001);
    metrics.record_publishing_latency("kafka", 0.015);
    
    // Test publisher connection metrics
    metrics.update_publisher_connections("redis", 5);
    metrics.update_publisher_connections("kafka", 3);
    metrics.record_publisher_connection_error("redis", "timeout");
    metrics.record_delivery_confirmation("redis", true);
    metrics.record_delivery_confirmation("kafka", false);
    
    // Test audit metrics
    metrics.record_audit_event("signal_emission");
    metrics.record_audit_event("feature_computation");
    metrics.record_audit_error("file_write");
    metrics.update_audit_file_size(1024000);
    metrics.record_s3_upload(true);
    metrics.record_s3_upload(false);
    
    // Test health check metrics
    metrics.record_health_check("redis", 0.005, true);
    metrics.record_health_check("kafka", 0.010, false);
    metrics.record_health_check("buffer", 0.001, true);
    metrics.record_health_check("audit", 0.002, true);
    
    // Verify registry is accessible
    let registry = metrics.registry();
    let metric_families = registry.gather();
    
    // Should have metrics registered
    assert!(!metric_families.is_empty());
    
    // Check that we have the expected metric families
    let metric_names: Vec<String> = metric_families
        .iter()
        .map(|f| f.get_name().to_string())
        .collect();
    
    assert!(metric_names.contains(&"signal_emission_signals_published_total".to_string()));
    assert!(metric_names.contains(&"signal_emission_validation_errors_total".to_string()));
    assert!(metric_names.contains(&"signal_emission_publisher_errors_total".to_string()));
    assert!(metric_names.contains(&"signal_emission_buffer_overflows_total".to_string()));
    assert!(metric_names.contains(&"signal_emission_duration_seconds".to_string()));
    assert!(metric_names.contains(&"signal_emission_buffer_size_current".to_string()));
    assert!(metric_names.contains(&"signal_emission_audit_events_logged_total".to_string()));
    assert!(metric_names.contains(&"signal_emission_health_check_status".to_string()));
}

#[test]
fn test_metrics_timer() {
    let timer = MetricsTimer::start();
    
    // Simulate some work
    std::thread::sleep(std::time::Duration::from_millis(10));
    
    let elapsed = timer.elapsed_seconds();
    
    // Should be at least 10ms
    assert!(elapsed >= 0.01);
    // Should be less than 1 second (generous upper bound)
    assert!(elapsed < 1.0);
}

#[test]
fn test_metrics_with_different_labels() {
    let metrics = SignalEmissionMetrics::new().expect("Failed to create metrics");
    
    // Test with different symbols
    let symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"];
    let backends = ["redis", "kafka"];
    let sides = ["BUY", "SELL", "HOLD"];
    
    for symbol in &symbols {
        for backend in &backends {
            for side in &sides {
                metrics.record_signal_published(symbol, backend, side);
                metrics.record_emission_latency(symbol, backend, 0.025);
            }
        }
    }
    
    // Test error metrics with different types
    let error_types = ["invalid_strength", "invalid_side", "missing_field"];
    let fields = ["strength", "side", "timestamp"];
    
    for (error_type, field) in error_types.iter().zip(fields.iter()) {
        metrics.record_validation_error("BTCUSDT", error_type, field);
    }
    
    // Verify metrics were recorded
    let registry = metrics.registry();
    let metric_families = registry.gather();
    
    // Find the signals published metric
    let signals_published = metric_families
        .iter()
        .find(|f| f.get_name() == "signal_emission_signals_published_total")
        .expect("Should have signals published metric");
    
    // Should have metrics for each combination of labels
    let expected_combinations = symbols.len() * backends.len() * sides.len();
    assert_eq!(signals_published.get_metric().len(), expected_combinations);
}

#[test]
fn test_buffer_utilization_calculation() {
    let metrics = SignalEmissionMetrics::new().expect("Failed to create metrics");
    
    // Test various buffer sizes
    metrics.update_buffer_size(0, 100);   // 0% utilization
    metrics.update_buffer_size(50, 100);  // 50% utilization
    metrics.update_buffer_size(100, 100); // 100% utilization
    
    // Test edge cases
    metrics.update_buffer_size(0, 0);     // Empty buffer
    metrics.update_buffer_size(75, 150);  // 50% utilization
    
    // The metrics should be recorded without panicking
    let registry = metrics.registry();
    let metric_families = registry.gather();
    
    let buffer_size = metric_families
        .iter()
        .find(|f| f.get_name() == "signal_emission_buffer_size_current")
        .expect("Should have buffer size metric");
    
    assert_eq!(buffer_size.get_metric().len(), 1);
    
    let buffer_utilization = metric_families
        .iter()
        .find(|f| f.get_name() == "signal_emission_buffer_utilization_ratio")
        .expect("Should have buffer utilization metric");
    
    assert_eq!(buffer_utilization.get_metric().len(), 1);
}

#[test]
fn test_health_check_status_values() {
    let metrics = SignalEmissionMetrics::new().expect("Failed to create metrics");
    
    let components = ["redis", "kafka", "buffer", "audit"];
    
    // Test healthy components (should be 1)
    for component in &components {
        metrics.record_health_check(component, 0.005, true);
    }
    
    // Test unhealthy components (should be 0)
    for component in &components {
        metrics.record_health_check(&format!("{}_unhealthy", component), 0.100, false);
    }
    
    let registry = metrics.registry();
    let metric_families = registry.gather();
    
    let health_status = metric_families
        .iter()
        .find(|f| f.get_name() == "signal_emission_health_check_status")
        .expect("Should have health check status metric");
    
    // Should have metrics for all components (healthy and unhealthy)
    assert_eq!(health_status.get_metric().len(), components.len() * 2);
    
    // Verify that healthy components have value 1 and unhealthy have value 0
    for metric in health_status.get_metric() {
        let component_name = metric.get_label()
            .iter()
            .find(|l| l.get_name() == "component")
            .expect("Should have component label")
            .get_value();
        
        let expected_value = if component_name.ends_with("_unhealthy") { 0.0 } else { 1.0 };
        assert_eq!(metric.get_gauge().get_value(), expected_value);
    }
}