//! Signal emission metrics collection and export
//!
//! This module provides comprehensive metrics tracking for all signal emission operations:
//! - Signal publishing metrics (count, latency, errors)
//! - Validation metrics (success/failure rates, error types)
//! - Publisher backend metrics (Redis, Kafka performance)
//! - Buffer utilization metrics
//! - Audit logging metrics
//!
//! Metrics are collected using Prometheus format with proper labeling for detailed analysis.

use prometheus::{
    Gauge, Histogram, HistogramOpts, HistogramVec, IntCounter,
    IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Signal emission metrics collector with Prometheus integration
#[derive(Clone)]
pub struct SignalEmissionMetrics {
    // Signal publishing metrics
    pub signals_published_total: IntCounterVec,
    pub signals_validation_errors_total: IntCounterVec,
    pub signals_publisher_errors_total: IntCounterVec,
    pub signals_buffer_overflows_total: IntCounter,
    
    // Latency histograms
    pub signal_emission_duration_seconds: HistogramVec,
    pub signal_validation_duration_seconds: Histogram,
    pub signal_publishing_duration_seconds: HistogramVec,
    
    // Buffer metrics
    pub buffer_size_current: IntGauge,
    pub buffer_utilization_ratio: Gauge,
    pub buffer_operations_total: IntCounterVec,
    
    // Publisher backend metrics
    pub publisher_connections_active: IntGaugeVec,
    pub publisher_connection_errors_total: IntCounterVec,
    pub publisher_delivery_confirmations_total: IntCounterVec,
    
    // Audit logging metrics
    pub audit_events_logged_total: IntCounterVec,
    pub audit_logging_errors_total: IntCounterVec,
    pub audit_file_size_bytes: IntGauge,
    pub audit_s3_uploads_total: IntCounterVec,
    
    // Health check metrics
    pub health_check_duration_seconds: HistogramVec,
    pub health_check_status: IntGaugeVec,
    
    // Registry for metric collection
    registry: Arc<Registry>,
}

impl SignalEmissionMetrics {
    /// Create a new signal emission metrics collector
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Arc::new(Registry::new());
        
        // Signal publishing metrics
        let signals_published_total = IntCounterVec::new(
            Opts::new(
                "signal_emission_signals_published_total",
                "Total number of signals published to message bus"
            ),
            &["symbol", "backend", "side"]
        )?;
        
        let signals_validation_errors_total = IntCounterVec::new(
            Opts::new(
                "signal_emission_validation_errors_total",
                "Total number of signal validation errors"
            ),
            &["symbol", "error_type", "field"]
        )?;
        
        let signals_publisher_errors_total = IntCounterVec::new(
            Opts::new(
                "signal_emission_publisher_errors_total",
                "Total number of signal publisher errors"
            ),
            &["backend", "error_type"]
        )?;
        
        let signals_buffer_overflows_total = IntCounter::new(
            "signal_emission_buffer_overflows_total",
            "Total number of signal buffer overflow events"
        )?;
        
        // Latency histograms with appropriate buckets for trading systems
        let signal_emission_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "signal_emission_duration_seconds",
                "Time taken for complete signal emission pipeline"
            ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
            &["symbol", "backend"]
        )?;
        
        let signal_validation_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "signal_emission_validation_duration_seconds",
                "Time taken for signal validation"
            ).buckets(vec![0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1])
        )?;
        
        let signal_publishing_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "signal_emission_publishing_duration_seconds",
                "Time taken for signal publishing to backend"
            ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
            &["backend"]
        )?;
        
        // Buffer metrics
        let buffer_size_current = IntGauge::new(
            "signal_emission_buffer_size_current",
            "Current number of signals in buffer"
        )?;
        
        let buffer_utilization_ratio = Gauge::new(
            "signal_emission_buffer_utilization_ratio",
            "Buffer utilization as ratio of current size to maximum size"
        )?;
        
        let buffer_operations_total = IntCounterVec::new(
            Opts::new(
                "signal_emission_buffer_operations_total",
                "Total number of buffer operations"
            ),
            &["operation"] // push, pop, persist, restore
        )?;
        
        // Publisher backend metrics
        let publisher_connections_active = IntGaugeVec::new(
            Opts::new(
                "signal_emission_publisher_connections_active",
                "Number of active connections to publisher backends"
            ),
            &["backend"]
        )?;
        
        let publisher_connection_errors_total = IntCounterVec::new(
            Opts::new(
                "signal_emission_publisher_connection_errors_total",
                "Total number of publisher connection errors"
            ),
            &["backend", "error_type"]
        )?;
        
        let publisher_delivery_confirmations_total = IntCounterVec::new(
            Opts::new(
                "signal_emission_publisher_delivery_confirmations_total",
                "Total number of delivery confirmations from publishers"
            ),
            &["backend", "status"] // success, failure
        )?;
        
        // Audit logging metrics
        let audit_events_logged_total = IntCounterVec::new(
            Opts::new(
                "signal_emission_audit_events_logged_total",
                "Total number of audit events logged"
            ),
            &["event_type"] // signal_emission, feature_computation, validation_error, publisher_error
        )?;
        
        let audit_logging_errors_total = IntCounterVec::new(
            Opts::new(
                "signal_emission_audit_logging_errors_total",
                "Total number of audit logging errors"
            ),
            &["error_type"] // file_write, s3_upload, serialization
        )?;
        
        let audit_file_size_bytes = IntGauge::new(
            "signal_emission_audit_file_size_bytes",
            "Current size of audit log file in bytes"
        )?;
        
        let audit_s3_uploads_total = IntCounterVec::new(
            Opts::new(
                "signal_emission_audit_s3_uploads_total",
                "Total number of audit log S3 uploads"
            ),
            &["status"] // success, failure
        )?;
        
        // Health check metrics
        let health_check_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "signal_emission_health_check_duration_seconds",
                "Time taken for health checks"
            ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
            &["component"] // redis, kafka, buffer, audit
        )?;
        
        let health_check_status = IntGaugeVec::new(
            Opts::new(
                "signal_emission_health_check_status",
                "Health check status (1=healthy, 0=unhealthy)"
            ),
            &["component"]
        )?;
        
        // Register all metrics
        registry.register(Box::new(signals_published_total.clone()))?;
        registry.register(Box::new(signals_validation_errors_total.clone()))?;
        registry.register(Box::new(signals_publisher_errors_total.clone()))?;
        registry.register(Box::new(signals_buffer_overflows_total.clone()))?;
        registry.register(Box::new(signal_emission_duration_seconds.clone()))?;
        registry.register(Box::new(signal_validation_duration_seconds.clone()))?;
        registry.register(Box::new(signal_publishing_duration_seconds.clone()))?;
        registry.register(Box::new(buffer_size_current.clone()))?;
        registry.register(Box::new(buffer_utilization_ratio.clone()))?;
        registry.register(Box::new(buffer_operations_total.clone()))?;
        registry.register(Box::new(publisher_connections_active.clone()))?;
        registry.register(Box::new(publisher_connection_errors_total.clone()))?;
        registry.register(Box::new(publisher_delivery_confirmations_total.clone()))?;
        registry.register(Box::new(audit_events_logged_total.clone()))?;
        registry.register(Box::new(audit_logging_errors_total.clone()))?;
        registry.register(Box::new(audit_file_size_bytes.clone()))?;
        registry.register(Box::new(audit_s3_uploads_total.clone()))?;
        registry.register(Box::new(health_check_duration_seconds.clone()))?;
        registry.register(Box::new(health_check_status.clone()))?;
        
        Ok(Self {
            signals_published_total,
            signals_validation_errors_total,
            signals_publisher_errors_total,
            signals_buffer_overflows_total,
            signal_emission_duration_seconds,
            signal_validation_duration_seconds,
            signal_publishing_duration_seconds,
            buffer_size_current,
            buffer_utilization_ratio,
            buffer_operations_total,
            publisher_connections_active,
            publisher_connection_errors_total,
            publisher_delivery_confirmations_total,
            audit_events_logged_total,
            audit_logging_errors_total,
            audit_file_size_bytes,
            audit_s3_uploads_total,
            health_check_duration_seconds,
            health_check_status,
            registry,
        })
    }
    
    /// Get the Prometheus registry for metric collection
    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
    
    /// Record a successful signal publication
    pub fn record_signal_published(&self, symbol: &str, backend: &str, side: &str) {
        self.signals_published_total
            .with_label_values(&[symbol, backend, side])
            .inc();
    }
    
    /// Record a signal validation error
    pub fn record_validation_error(&self, symbol: &str, error_type: &str, field: &str) {
        self.signals_validation_errors_total
            .with_label_values(&[symbol, error_type, field])
            .inc();
    }
    
    /// Record a publisher error
    pub fn record_publisher_error(&self, backend: &str, error_type: &str) {
        self.signals_publisher_errors_total
            .with_label_values(&[backend, error_type])
            .inc();
    }
    
    /// Record a buffer overflow
    pub fn record_buffer_overflow(&self) {
        self.signals_buffer_overflows_total.inc();
    }
    
    /// Record signal emission latency
    pub fn record_emission_latency(&self, symbol: &str, backend: &str, duration_seconds: f64) {
        self.signal_emission_duration_seconds
            .with_label_values(&[symbol, backend])
            .observe(duration_seconds);
    }
    
    /// Record signal validation latency
    pub fn record_validation_latency(&self, duration_seconds: f64) {
        self.signal_validation_duration_seconds
            .observe(duration_seconds);
    }
    
    /// Record signal publishing latency
    pub fn record_publishing_latency(&self, backend: &str, duration_seconds: f64) {
        self.signal_publishing_duration_seconds
            .with_label_values(&[backend])
            .observe(duration_seconds);
    }
    
    /// Update buffer size metrics
    pub fn update_buffer_size(&self, current_size: i64, max_size: i64) {
        self.buffer_size_current.set(current_size);
        if max_size > 0 {
            let utilization = current_size as f64 / max_size as f64;
            self.buffer_utilization_ratio.set(utilization);
        }
    }
    
    /// Record buffer operation
    pub fn record_buffer_operation(&self, operation: &str) {
        self.buffer_operations_total
            .with_label_values(&[operation])
            .inc();
    }
    
    /// Update publisher connection count
    pub fn update_publisher_connections(&self, backend: &str, count: i64) {
        self.publisher_connections_active
            .with_label_values(&[backend])
            .set(count);
    }
    
    /// Record publisher connection error
    pub fn record_publisher_connection_error(&self, backend: &str, error_type: &str) {
        self.publisher_connection_errors_total
            .with_label_values(&[backend, error_type])
            .inc();
    }
    
    /// Record delivery confirmation
    pub fn record_delivery_confirmation(&self, backend: &str, success: bool) {
        let status = if success { "success" } else { "failure" };
        self.publisher_delivery_confirmations_total
            .with_label_values(&[backend, status])
            .inc();
    }
    
    /// Record audit event
    pub fn record_audit_event(&self, event_type: &str) {
        self.audit_events_logged_total
            .with_label_values(&[event_type])
            .inc();
    }
    
    /// Record audit logging error
    pub fn record_audit_error(&self, error_type: &str) {
        self.audit_logging_errors_total
            .with_label_values(&[error_type])
            .inc();
    }
    
    /// Update audit file size
    pub fn update_audit_file_size(&self, size_bytes: i64) {
        self.audit_file_size_bytes.set(size_bytes);
    }
    
    /// Record S3 upload result
    pub fn record_s3_upload(&self, success: bool) {
        let status = if success { "success" } else { "failure" };
        self.audit_s3_uploads_total
            .with_label_values(&[status])
            .inc();
    }
    
    /// Record health check latency and status
    pub fn record_health_check(&self, component: &str, duration_seconds: f64, healthy: bool) {
        self.health_check_duration_seconds
            .with_label_values(&[component])
            .observe(duration_seconds);
        
        let status = if healthy { 1 } else { 0 };
        self.health_check_status
            .with_label_values(&[component])
            .set(status);
    }
}

impl Default for SignalEmissionMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create default SignalEmissionMetrics")
    }
}

/// Timer helper for measuring operation latency
pub struct MetricsTimer {
    start: Instant,
}

impl MetricsTimer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    
    /// Get elapsed time in seconds
    pub fn elapsed_seconds(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

/// Aggregated metrics snapshot for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalEmissionMetricsSnapshot {
    pub timestamp: i64,
    pub signals_published_count: u64,
    pub validation_errors_count: u64,
    pub publisher_errors_count: u64,
    pub buffer_overflows_count: u64,
    pub buffer_current_size: i64,
    pub buffer_utilization_ratio: f64,
    pub avg_emission_latency_ms: f64,
    pub avg_validation_latency_ms: f64,
    pub active_connections: std::collections::HashMap<String, i64>,
    pub audit_events_count: u64,
    pub health_status: std::collections::HashMap<String, bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_creation() {
        let metrics = SignalEmissionMetrics::new().unwrap();
        
        // Test that metrics can be recorded without panicking
        metrics.record_signal_published("BTCUSDT", "redis", "BUY");
        metrics.record_validation_error("BTCUSDT", "invalid_strength", "strength");
        metrics.record_publisher_error("kafka", "connection_timeout");
        metrics.record_buffer_overflow();
    }
    
    #[test]
    fn test_latency_recording() {
        let metrics = SignalEmissionMetrics::new().unwrap();
        
        metrics.record_emission_latency("BTCUSDT", "redis", 0.025);
        metrics.record_validation_latency(0.001);
        metrics.record_publishing_latency("kafka", 0.015);
        
        // Verify metrics were recorded (would need to check registry in real test)
    }
    
    #[test]
    fn test_buffer_metrics() {
        let metrics = SignalEmissionMetrics::new().unwrap();
        
        metrics.update_buffer_size(50, 100);
        metrics.record_buffer_operation("push");
        metrics.record_buffer_operation("pop");
        
        // Buffer utilization should be 0.5
    }
    
    #[test]
    fn test_publisher_metrics() {
        let metrics = SignalEmissionMetrics::new().unwrap();
        
        metrics.update_publisher_connections("redis", 5);
        metrics.record_publisher_connection_error("kafka", "timeout");
        metrics.record_delivery_confirmation("redis", true);
        metrics.record_delivery_confirmation("kafka", false);
    }
    
    #[test]
    fn test_audit_metrics() {
        let metrics = SignalEmissionMetrics::new().unwrap();
        
        metrics.record_audit_event("signal_emission");
        metrics.record_audit_error("file_write");
        metrics.update_audit_file_size(1024000);
        metrics.record_s3_upload(true);
    }
    
    #[test]
    fn test_health_check_metrics() {
        let metrics = SignalEmissionMetrics::new().unwrap();
        
        metrics.record_health_check("redis", 0.005, true);
        metrics.record_health_check("kafka", 0.010, false);
        metrics.record_health_check("buffer", 0.001, true);
    }
    
    #[test]
    fn test_metrics_timer() {
        let timer = MetricsTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed_seconds();
        
        assert!(elapsed >= 0.01);
        assert!(elapsed < 0.1); // Should be much less than 100ms
    }
}