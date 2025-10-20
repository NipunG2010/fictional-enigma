//! Metrics collection and export for HMM integration monitoring
//!
//! This module provides comprehensive metrics tracking for:
//! - Request metrics (count, duration, errors)
//! - Cache metrics (hits, misses, size, evictions)
//! - Circuit breaker state metrics
//! - Fallback activation metrics
//!
//! Metrics can be exported in various formats for integration with monitoring systems.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Request-level metrics tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Total number of requests made
    pub total_requests: u64,
    /// Number of successful requests
    pub successful_requests: u64,
    /// Number of failed requests
    pub failed_requests: u64,
    /// Total request duration in milliseconds
    pub total_duration_ms: u64,
    /// Average request duration in milliseconds
    pub avg_duration_ms: f64,
    /// Minimum request duration in milliseconds
    pub min_duration_ms: u64,
    /// Maximum request duration in milliseconds
    pub max_duration_ms: u64,
    /// Number of timeout errors
    pub timeout_errors: u64,
    /// Number of network errors
    pub network_errors: u64,
    /// Number of validation errors
    pub validation_errors: u64,
}


/// Fallback activation metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FallbackMetrics {
    /// Total number of fallback activations
    pub total_activations: u64,
    /// Fallback activations due to circuit breaker
    pub circuit_breaker_activations: u64,
    /// Fallback activations due to network errors
    pub network_error_activations: u64,
    /// Fallback activations due to timeout
    pub timeout_activations: u64,
    /// Fallback activations due to service errors
    pub service_error_activations: u64,
    /// Whether fallback is currently active
    pub currently_active: bool,
}

/// Comprehensive metrics for HMM integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmIntegrationMetrics {
    /// Request-level metrics
    pub requests: RequestMetrics,
    /// Cache metrics
    pub cache: crate::CacheStats,
    /// Circuit breaker metrics
    pub circuit_breaker: crate::hmm_client::CircuitBreakerMetrics,
    /// Fallback metrics
    pub fallback: FallbackMetrics,
    /// Timestamp when metrics were collected
    pub timestamp: i64,
    /// Uptime in seconds since metrics collection started
    pub uptime_seconds: u64,
}

/// Thread-safe metrics collector
pub struct MetricsCollector {
    // Request metrics
    total_requests: Arc<AtomicU64>,
    successful_requests: Arc<AtomicU64>,
    failed_requests: Arc<AtomicU64>,
    total_duration_ms: Arc<AtomicU64>,
    min_duration_ms: Arc<AtomicU64>,
    max_duration_ms: Arc<AtomicU64>,
    timeout_errors: Arc<AtomicU64>,
    network_errors: Arc<AtomicU64>,
    validation_errors: Arc<AtomicU64>,
    
    // Fallback metrics
    fallback_activations: Arc<AtomicU64>,
    circuit_breaker_fallbacks: Arc<AtomicU64>,
    network_error_fallbacks: Arc<AtomicU64>,
    timeout_fallbacks: Arc<AtomicU64>,
    service_error_fallbacks: Arc<AtomicU64>,
    fallback_active: Arc<RwLock<bool>>,
    
    // Start time for uptime calculation
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            total_requests: Arc::new(AtomicU64::new(0)),
            successful_requests: Arc::new(AtomicU64::new(0)),
            failed_requests: Arc::new(AtomicU64::new(0)),
            total_duration_ms: Arc::new(AtomicU64::new(0)),
            min_duration_ms: Arc::new(AtomicU64::new(u64::MAX)),
            max_duration_ms: Arc::new(AtomicU64::new(0)),
            timeout_errors: Arc::new(AtomicU64::new(0)),
            network_errors: Arc::new(AtomicU64::new(0)),
            validation_errors: Arc::new(AtomicU64::new(0)),
            fallback_activations: Arc::new(AtomicU64::new(0)),
            circuit_breaker_fallbacks: Arc::new(AtomicU64::new(0)),
            network_error_fallbacks: Arc::new(AtomicU64::new(0)),
            timeout_fallbacks: Arc::new(AtomicU64::new(0)),
            service_error_fallbacks: Arc::new(AtomicU64::new(0)),
            fallback_active: Arc::new(RwLock::new(false)),
            start_time: Instant::now(),
        }
    }
    
    /// Record a successful request
    pub fn record_success(&self, duration: Duration) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.record_duration(duration);
    }
    
    /// Record a failed request
    pub fn record_failure(&self, duration: Duration, error_type: &str) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.record_duration(duration);
        
        // Track specific error types
        match error_type {
            "timeout" => self.timeout_errors.fetch_add(1, Ordering::Relaxed),
            "network" => self.network_errors.fetch_add(1, Ordering::Relaxed),
            "validation" => self.validation_errors.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
    }
    
    /// Record request duration
    fn record_duration(&self, duration: Duration) {
        let duration_ms = duration.as_millis() as u64;
        self.total_duration_ms.fetch_add(duration_ms, Ordering::Relaxed);
        
        // Update min duration
        let mut current_min = self.min_duration_ms.load(Ordering::Relaxed);
        while duration_ms < current_min {
            match self.min_duration_ms.compare_exchange(
                current_min,
                duration_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }
        
        // Update max duration
        let mut current_max = self.max_duration_ms.load(Ordering::Relaxed);
        while duration_ms > current_max {
            match self.max_duration_ms.compare_exchange(
                current_max,
                duration_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }
    
    /// Record a fallback activation
    pub fn record_fallback(&self, reason: &str) {
        self.fallback_activations.fetch_add(1, Ordering::Relaxed);
        
        // Track specific fallback reasons
        match reason {
            "circuit_breaker" => self.circuit_breaker_fallbacks.fetch_add(1, Ordering::Relaxed),
            "network_error" => self.network_error_fallbacks.fetch_add(1, Ordering::Relaxed),
            "timeout" => self.timeout_fallbacks.fetch_add(1, Ordering::Relaxed),
            "service_error" => self.service_error_fallbacks.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
        
        // Mark fallback as active
        if let Ok(mut active) = self.fallback_active.write() {
            *active = true;
        }
    }
    
    /// Clear fallback active status
    pub fn clear_fallback_active(&self) {
        if let Ok(mut active) = self.fallback_active.write() {
            *active = false;
        }
    }
    
    /// Get current request metrics
    pub fn get_request_metrics(&self) -> RequestMetrics {
        let total = self.total_requests.load(Ordering::Relaxed);
        let total_duration = self.total_duration_ms.load(Ordering::Relaxed);
        let avg_duration = if total > 0 {
            total_duration as f64 / total as f64
        } else {
            0.0
        };
        
        let min_duration = self.min_duration_ms.load(Ordering::Relaxed);
        let min_duration = if min_duration == u64::MAX { 0 } else { min_duration };
        
        RequestMetrics {
            total_requests: total,
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            total_duration_ms: total_duration,
            avg_duration_ms: avg_duration,
            min_duration_ms: min_duration,
            max_duration_ms: self.max_duration_ms.load(Ordering::Relaxed),
            timeout_errors: self.timeout_errors.load(Ordering::Relaxed),
            network_errors: self.network_errors.load(Ordering::Relaxed),
            validation_errors: self.validation_errors.load(Ordering::Relaxed),
        }
    }
    
    /// Get current fallback metrics
    pub fn get_fallback_metrics(&self) -> FallbackMetrics {
        let currently_active = self.fallback_active.read()
            .map(|active| *active)
            .unwrap_or(false);
        
        FallbackMetrics {
            total_activations: self.fallback_activations.load(Ordering::Relaxed),
            circuit_breaker_activations: self.circuit_breaker_fallbacks.load(Ordering::Relaxed),
            network_error_activations: self.network_error_fallbacks.load(Ordering::Relaxed),
            timeout_activations: self.timeout_fallbacks.load(Ordering::Relaxed),
            service_error_activations: self.service_error_fallbacks.load(Ordering::Relaxed),
            currently_active,
        }
    }
    
    /// Get uptime in seconds
    pub fn get_uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
    
    /// Reset all metrics (useful for testing)
    #[cfg(test)]
    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
        self.total_duration_ms.store(0, Ordering::Relaxed);
        self.min_duration_ms.store(u64::MAX, Ordering::Relaxed);
        self.max_duration_ms.store(0, Ordering::Relaxed);
        self.timeout_errors.store(0, Ordering::Relaxed);
        self.network_errors.store(0, Ordering::Relaxed);
        self.validation_errors.store(0, Ordering::Relaxed);
        self.fallback_activations.store(0, Ordering::Relaxed);
        self.circuit_breaker_fallbacks.store(0, Ordering::Relaxed);
        self.network_error_fallbacks.store(0, Ordering::Relaxed);
        self.timeout_fallbacks.store(0, Ordering::Relaxed);
        self.service_error_fallbacks.store(0, Ordering::Relaxed);
        if let Ok(mut active) = self.fallback_active.write() {
            *active = false;
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics export format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsFormat {
    /// JSON format
    Json,
    /// Prometheus text format
    Prometheus,
}

/// Export metrics in the specified format
pub fn export_metrics(
    metrics: &HmmIntegrationMetrics,
    format: MetricsFormat,
) -> Result<String, serde_json::Error> {
    match format {
        MetricsFormat::Json => serde_json::to_string_pretty(metrics),
        MetricsFormat::Prometheus => Ok(to_prometheus_format(metrics)),
    }
}

/// Convert metrics to Prometheus text format
fn to_prometheus_format(metrics: &HmmIntegrationMetrics) -> String {
    let mut output = String::new();
    
    // Request metrics
    output.push_str("# HELP hmm_requests_total Total number of HMM service requests\n");
    output.push_str("# TYPE hmm_requests_total counter\n");
    output.push_str(&format!("hmm_requests_total {}\n", metrics.requests.total_requests));
    
    output.push_str("# HELP hmm_requests_successful Number of successful HMM service requests\n");
    output.push_str("# TYPE hmm_requests_successful counter\n");
    output.push_str(&format!("hmm_requests_successful {}\n", metrics.requests.successful_requests));
    
    output.push_str("# HELP hmm_requests_failed Number of failed HMM service requests\n");
    output.push_str("# TYPE hmm_requests_failed counter\n");
    output.push_str(&format!("hmm_requests_failed {}\n", metrics.requests.failed_requests));
    
    output.push_str("# HELP hmm_request_duration_ms_avg Average request duration in milliseconds\n");
    output.push_str("# TYPE hmm_request_duration_ms_avg gauge\n");
    output.push_str(&format!("hmm_request_duration_ms_avg {}\n", metrics.requests.avg_duration_ms));
    
    output.push_str("# HELP hmm_request_duration_ms_min Minimum request duration in milliseconds\n");
    output.push_str("# TYPE hmm_request_duration_ms_min gauge\n");
    output.push_str(&format!("hmm_request_duration_ms_min {}\n", metrics.requests.min_duration_ms));
    
    output.push_str("# HELP hmm_request_duration_ms_max Maximum request duration in milliseconds\n");
    output.push_str("# TYPE hmm_request_duration_ms_max gauge\n");
    output.push_str(&format!("hmm_request_duration_ms_max {}\n", metrics.requests.max_duration_ms));
    
    // Error metrics
    output.push_str("# HELP hmm_errors_timeout Number of timeout errors\n");
    output.push_str("# TYPE hmm_errors_timeout counter\n");
    output.push_str(&format!("hmm_errors_timeout {}\n", metrics.requests.timeout_errors));
    
    output.push_str("# HELP hmm_errors_network Number of network errors\n");
    output.push_str("# TYPE hmm_errors_network counter\n");
    output.push_str(&format!("hmm_errors_network {}\n", metrics.requests.network_errors));
    
    output.push_str("# HELP hmm_errors_validation Number of validation errors\n");
    output.push_str("# TYPE hmm_errors_validation counter\n");
    output.push_str(&format!("hmm_errors_validation {}\n", metrics.requests.validation_errors));
    
    // Cache metrics
    output.push_str("# HELP hmm_cache_hits Total number of cache hits\n");
    output.push_str("# TYPE hmm_cache_hits counter\n");
    output.push_str(&format!("hmm_cache_hits {}\n", metrics.cache.hits));
    
    output.push_str("# HELP hmm_cache_misses Total number of cache misses\n");
    output.push_str("# TYPE hmm_cache_misses counter\n");
    output.push_str(&format!("hmm_cache_misses {}\n", metrics.cache.misses));
    
    output.push_str("# HELP hmm_cache_size Current cache size\n");
    output.push_str("# TYPE hmm_cache_size gauge\n");
    output.push_str(&format!("hmm_cache_size {}\n", metrics.cache.size));
    
    output.push_str("# HELP hmm_cache_evictions Total number of cache evictions\n");
    output.push_str("# TYPE hmm_cache_evictions counter\n");
    output.push_str(&format!("hmm_cache_evictions {}\n", metrics.cache.evictions));
    
    output.push_str("# HELP hmm_cache_hit_rate Cache hit rate (0.0 to 1.0)\n");
    output.push_str("# TYPE hmm_cache_hit_rate gauge\n");
    output.push_str(&format!("hmm_cache_hit_rate {}\n", metrics.cache.hit_rate));
    
    // Circuit breaker metrics
    output.push_str("# HELP hmm_circuit_breaker_total_requests Total requests through circuit breaker\n");
    output.push_str("# TYPE hmm_circuit_breaker_total_requests counter\n");
    output.push_str(&format!("hmm_circuit_breaker_total_requests {}\n", metrics.circuit_breaker.total_requests));
    
    output.push_str("# HELP hmm_circuit_breaker_successful_requests Successful requests through circuit breaker\n");
    output.push_str("# TYPE hmm_circuit_breaker_successful_requests counter\n");
    output.push_str(&format!("hmm_circuit_breaker_successful_requests {}\n", metrics.circuit_breaker.successful_requests));
    
    output.push_str("# HELP hmm_circuit_breaker_failed_requests Failed requests through circuit breaker\n");
    output.push_str("# TYPE hmm_circuit_breaker_failed_requests counter\n");
    output.push_str(&format!("hmm_circuit_breaker_failed_requests {}\n", metrics.circuit_breaker.failed_requests));
    
    output.push_str("# HELP hmm_circuit_breaker_opens Number of times circuit breaker opened\n");
    output.push_str("# TYPE hmm_circuit_breaker_opens counter\n");
    output.push_str(&format!("hmm_circuit_breaker_opens {}\n", metrics.circuit_breaker.circuit_breaker_opens));
    
    output.push_str("# HELP hmm_circuit_breaker_closes Number of times circuit breaker closed\n");
    output.push_str("# TYPE hmm_circuit_breaker_closes counter\n");
    output.push_str(&format!("hmm_circuit_breaker_closes {}\n", metrics.circuit_breaker.circuit_breaker_closes));
    
    output.push_str("# HELP hmm_circuit_breaker_half_open_attempts Number of half-open test attempts\n");
    output.push_str("# TYPE hmm_circuit_breaker_half_open_attempts counter\n");
    output.push_str(&format!("hmm_circuit_breaker_half_open_attempts {}\n", metrics.circuit_breaker.half_open_attempts));
    
    output.push_str("# HELP hmm_circuit_breaker_rejected_requests Number of rejected requests\n");
    output.push_str("# TYPE hmm_circuit_breaker_rejected_requests counter\n");
    output.push_str(&format!("hmm_circuit_breaker_rejected_requests {}\n", metrics.circuit_breaker.rejected_requests));
    
    // Fallback metrics
    output.push_str("# HELP hmm_fallback_activations_total Total number of fallback activations\n");
    output.push_str("# TYPE hmm_fallback_activations_total counter\n");
    output.push_str(&format!("hmm_fallback_activations_total {}\n", metrics.fallback.total_activations));
    
    output.push_str("# HELP hmm_fallback_circuit_breaker Fallback activations due to circuit breaker\n");
    output.push_str("# TYPE hmm_fallback_circuit_breaker counter\n");
    output.push_str(&format!("hmm_fallback_circuit_breaker {}\n", metrics.fallback.circuit_breaker_activations));
    
    output.push_str("# HELP hmm_fallback_network_error Fallback activations due to network errors\n");
    output.push_str("# TYPE hmm_fallback_network_error counter\n");
    output.push_str(&format!("hmm_fallback_network_error {}\n", metrics.fallback.network_error_activations));
    
    output.push_str("# HELP hmm_fallback_timeout Fallback activations due to timeout\n");
    output.push_str("# TYPE hmm_fallback_timeout counter\n");
    output.push_str(&format!("hmm_fallback_timeout {}\n", metrics.fallback.timeout_activations));
    
    output.push_str("# HELP hmm_fallback_service_error Fallback activations due to service errors\n");
    output.push_str("# TYPE hmm_fallback_service_error counter\n");
    output.push_str(&format!("hmm_fallback_service_error {}\n", metrics.fallback.service_error_activations));
    
    output.push_str("# HELP hmm_fallback_active Whether fallback is currently active (1=active, 0=inactive)\n");
    output.push_str("# TYPE hmm_fallback_active gauge\n");
    output.push_str(&format!("hmm_fallback_active {}\n", if metrics.fallback.currently_active { 1 } else { 0 }));
    
    // System metrics
    output.push_str("# HELP hmm_uptime_seconds Uptime in seconds\n");
    output.push_str("# TYPE hmm_uptime_seconds counter\n");
    output.push_str(&format!("hmm_uptime_seconds {}\n", metrics.uptime_seconds));
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        let metrics = collector.get_request_metrics();
        
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.successful_requests, 0);
        assert_eq!(metrics.failed_requests, 0);
    }
    
    #[test]
    fn test_record_success() {
        let collector = MetricsCollector::new();
        
        collector.record_success(Duration::from_millis(100));
        collector.record_success(Duration::from_millis(200));
        
        let metrics = collector.get_request_metrics();
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.successful_requests, 2);
        assert_eq!(metrics.failed_requests, 0);
        assert_eq!(metrics.min_duration_ms, 100);
        assert_eq!(metrics.max_duration_ms, 200);
        assert_eq!(metrics.avg_duration_ms, 150.0);
    }
    
    #[test]
    fn test_record_failure() {
        let collector = MetricsCollector::new();
        
        collector.record_failure(Duration::from_millis(50), "timeout");
        collector.record_failure(Duration::from_millis(75), "network");
        
        let metrics = collector.get_request_metrics();
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.successful_requests, 0);
        assert_eq!(metrics.failed_requests, 2);
        assert_eq!(metrics.timeout_errors, 1);
        assert_eq!(metrics.network_errors, 1);
    }
    
    #[test]
    fn test_record_fallback() {
        let collector = MetricsCollector::new();
        
        collector.record_fallback("circuit_breaker");
        collector.record_fallback("network_error");
        collector.record_fallback("timeout");
        
        let metrics = collector.get_fallback_metrics();
        assert_eq!(metrics.total_activations, 3);
        assert_eq!(metrics.circuit_breaker_activations, 1);
        assert_eq!(metrics.network_error_activations, 1);
        assert_eq!(metrics.timeout_activations, 1);
        assert!(metrics.currently_active);
    }
    
    #[test]
    fn test_clear_fallback_active() {
        let collector = MetricsCollector::new();
        
        collector.record_fallback("circuit_breaker");
        assert!(collector.get_fallback_metrics().currently_active);
        
        collector.clear_fallback_active();
        assert!(!collector.get_fallback_metrics().currently_active);
    }
    
    #[test]
    fn test_uptime() {
        let collector = MetricsCollector::new();
        
        thread::sleep(Duration::from_millis(100));
        
        let uptime = collector.get_uptime_seconds();
        assert!(uptime >= 0);
    }
    
    #[test]
    fn test_duration_tracking() {
        let collector = MetricsCollector::new();
        
        collector.record_success(Duration::from_millis(50));
        collector.record_success(Duration::from_millis(150));
        collector.record_success(Duration::from_millis(100));
        
        let metrics = collector.get_request_metrics();
        assert_eq!(metrics.min_duration_ms, 50);
        assert_eq!(metrics.max_duration_ms, 150);
        assert_eq!(metrics.total_duration_ms, 300);
        assert_eq!(metrics.avg_duration_ms, 100.0);
    }
    
    #[test]
    fn test_export_json() {
        let metrics = HmmIntegrationMetrics {
            requests: RequestMetrics {
                total_requests: 100,
                successful_requests: 95,
                failed_requests: 5,
                total_duration_ms: 5000,
                avg_duration_ms: 50.0,
                min_duration_ms: 10,
                max_duration_ms: 200,
                timeout_errors: 2,
                network_errors: 2,
                validation_errors: 1,
            },
            cache: crate::CacheStats {
                hits: 80,
                misses: 20,
                size: 50,
                evictions: 5,
                hit_rate: 0.8,
            },
            circuit_breaker: crate::hmm_client::CircuitBreakerMetrics {
                total_requests: 100,
                successful_requests: 95,
                failed_requests: 5,
                circuit_breaker_opens: 1,
                circuit_breaker_closes: 1,
                half_open_attempts: 1,
                rejected_requests: 0,
            },
            fallback: FallbackMetrics {
                total_activations: 5,
                circuit_breaker_activations: 2,
                network_error_activations: 2,
                timeout_activations: 1,
                service_error_activations: 0,
                currently_active: false,
            },
            timestamp: 1234567890,
            uptime_seconds: 3600,
        };
        
        let json = export_metrics(&metrics, MetricsFormat::Json).unwrap();
        assert!(json.contains("total_requests"));
        assert!(json.contains("cache"));
        assert!(json.contains("circuit_breaker"));
        assert!(json.contains("fallback"));
    }
    
    #[test]
    fn test_export_prometheus() {
        let metrics = HmmIntegrationMetrics {
            requests: RequestMetrics {
                total_requests: 100,
                successful_requests: 95,
                failed_requests: 5,
                total_duration_ms: 5000,
                avg_duration_ms: 50.0,
                min_duration_ms: 10,
                max_duration_ms: 200,
                timeout_errors: 2,
                network_errors: 2,
                validation_errors: 1,
            },
            cache: crate::CacheStats {
                hits: 80,
                misses: 20,
                size: 50,
                evictions: 5,
                hit_rate: 0.8,
            },
            circuit_breaker: crate::hmm_client::CircuitBreakerMetrics {
                total_requests: 100,
                successful_requests: 95,
                failed_requests: 5,
                circuit_breaker_opens: 1,
                circuit_breaker_closes: 1,
                half_open_attempts: 1,
                rejected_requests: 0,
            },
            fallback: FallbackMetrics {
                total_activations: 5,
                circuit_breaker_activations: 2,
                network_error_activations: 2,
                timeout_activations: 1,
                service_error_activations: 0,
                currently_active: false,
            },
            timestamp: 1234567890,
            uptime_seconds: 3600,
        };
        
        let prometheus = export_metrics(&metrics, MetricsFormat::Prometheus).unwrap();
        assert!(prometheus.contains("hmm_requests_total"));
        assert!(prometheus.contains("hmm_cache_hits"));
        assert!(prometheus.contains("hmm_circuit_breaker_opens"));
        assert!(prometheus.contains("hmm_fallback_activations_total"));
        assert!(prometheus.contains("# HELP"));
        assert!(prometheus.contains("# TYPE"));
    }
}
