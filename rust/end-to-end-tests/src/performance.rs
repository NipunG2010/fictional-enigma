//! Performance monitoring and metrics collection
//! 
//! Provides comprehensive performance monitoring capabilities for measuring
//! latency, throughput, memory usage, and other system metrics during testing.

use crate::{Result, TestFrameworkError, Instant};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Performance monitoring system for collecting and analyzing test metrics
pub struct PerformanceMonitor {
    /// Latency measurements
    latency_tracker: Arc<Mutex<LatencyTracker>>,
    
    /// Throughput measurements
    throughput_monitor: Arc<Mutex<ThroughputMonitor>>,
    
    /// Memory usage tracking
    memory_tracker: Arc<Mutex<MemoryTracker>>,
    
    /// Custom metrics collection
    custom_metrics: Arc<Mutex<HashMap<String, Vec<f64>>>>,
    
    /// Current measurement session
    current_session: Arc<Mutex<Option<MeasurementSession>>>,
}

/// Tracks latency measurements for different operations
#[derive(Debug, Default)]
struct LatencyTracker {
    measurements: HashMap<String, Vec<Duration>>,
    active_measurements: HashMap<String, Instant>,
}

/// Monitors throughput for different operations
#[derive(Debug, Default)]
struct ThroughputMonitor {
    measurements: HashMap<String, Vec<ThroughputMeasurement>>,
}

/// Tracks memory usage patterns
#[derive(Debug, Default)]
struct MemoryTracker {
    measurements: Vec<MemoryMeasurement>,
    baseline_memory: Option<u64>,
}

/// Individual throughput measurement
#[derive(Debug, Clone)]
struct ThroughputMeasurement {
    count: u64,
    duration: Duration,
    timestamp: Instant,
}

/// Individual memory measurement
#[derive(Debug, Clone)]
struct MemoryMeasurement {
    timestamp: Instant,
    memory_usage_bytes: u64,
    operation: String,
}

/// Performance measurement session
#[derive(Debug)]
struct MeasurementSession {
    name: String,
    start_time: Instant,
    end_time: Option<Instant>,
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// End-to-end pipeline latency statistics
    pub end_to_end_latency: LatencyStats,
    
    /// Feature computation latency statistics
    pub feature_computation_latency: LatencyStats,
    
    /// Signal generation latency statistics
    pub signal_generation_latency: LatencyStats,
    
    /// Signal emission latency statistics
    pub signal_emission_latency: LatencyStats,
    
    /// Throughput statistics
    pub throughput_stats: ThroughputStats,
    
    /// Memory usage statistics
    pub memory_usage: MemoryStats,
    
    /// Custom performance metrics
    pub custom_metrics: HashMap<String, MetricStats>,
}

/// Statistical summary of latency measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Mean latency in milliseconds
    pub mean: f64,
    
    /// Median latency in milliseconds
    pub median: f64,
    
    /// 95th percentile latency in milliseconds
    pub p95: f64,
    
    /// 99th percentile latency in milliseconds
    pub p99: f64,
    
    /// Minimum latency in milliseconds
    pub min: f64,
    
    /// Maximum latency in milliseconds
    pub max: f64,
    
    /// Standard deviation in milliseconds
    pub std_dev: f64,
    
    /// Number of measurements
    pub count: usize,
}

/// Throughput measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    /// Average throughput (operations per second)
    pub average_ops_per_second: f64,
    
    /// Peak throughput (operations per second)
    pub peak_ops_per_second: f64,
    
    /// Minimum throughput (operations per second)
    pub min_ops_per_second: f64,
    
    /// Total operations processed
    pub total_operations: u64,
    
    /// Total measurement duration in seconds
    pub total_duration_seconds: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    
    /// Average memory usage in MB
    pub average_memory_mb: f64,
    
    /// Memory usage at start in MB
    pub baseline_memory_mb: f64,
    
    /// Memory growth during test in MB
    pub memory_growth_mb: f64,
    
    /// Number of memory measurements
    pub measurement_count: usize,
}

/// Generic metric statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    /// Mean value
    pub mean: f64,
    
    /// Median value
    pub median: f64,
    
    /// Minimum value
    pub min: f64,
    
    /// Maximum value
    pub max: f64,
    
    /// Standard deviation
    pub std_dev: f64,
    
    /// Number of measurements
    pub count: usize,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            latency_tracker: Arc::new(Mutex::new(LatencyTracker::default())),
            throughput_monitor: Arc::new(Mutex::new(ThroughputMonitor::default())),
            memory_tracker: Arc::new(Mutex::new(MemoryTracker::default())),
            custom_metrics: Arc::new(Mutex::new(HashMap::new())),
            current_session: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Start a new measurement session
    pub fn start_measurement(&self, session_name: &str) -> Result<()> {
        let mut session = self.current_session.lock()
            .map_err(|_| TestFrameworkError::SetupError("Failed to acquire session lock".to_string()))?;
        
        *session = Some(MeasurementSession {
            name: session_name.to_string(),
            start_time: Instant::now(),
            end_time: None,
        });
        
        // Record baseline memory
        if let Ok(memory_usage) = self.get_current_memory_usage() {
            let mut memory_tracker = self.memory_tracker.lock()
                .map_err(|_| TestFrameworkError::SetupError("Failed to acquire memory tracker lock".to_string()))?;
            memory_tracker.baseline_memory = Some(memory_usage);
        }
        
        Ok(())
    }
    
    /// End the current measurement session
    pub fn end_measurement(&self) -> Result<()> {
        let mut session = self.current_session.lock()
            .map_err(|_| TestFrameworkError::SetupError("Failed to acquire session lock".to_string()))?;
        
        if let Some(ref mut current_session) = session.as_mut() {
            current_session.end_time = Some(Instant::now());
        }
        
        Ok(())
    }
    
    /// Start measuring latency for a specific operation
    pub fn start_latency_measurement(&self, operation: &str) -> Result<()> {
        let mut tracker = self.latency_tracker.lock()
            .map_err(|_| TestFrameworkError::SetupError("Failed to acquire latency tracker lock".to_string()))?;
        
        tracker.active_measurements.insert(operation.to_string(), Instant::now());
        Ok(())
    }
    
    /// End latency measurement for a specific operation
    pub fn end_latency_measurement(&self, operation: &str) -> Result<Duration> {
        let mut tracker = self.latency_tracker.lock()
            .map_err(|_| TestFrameworkError::SetupError("Failed to acquire latency tracker lock".to_string()))?;
        
        if let Some(start_time) = tracker.active_measurements.remove(operation) {
            let duration = start_time.elapsed();
            tracker.measurements
                .entry(operation.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
            Ok(duration)
        } else {
            Err(TestFrameworkError::SetupError(format!("No active measurement for operation: {}", operation)).into())
        }
    }
    
    /// Record a completed latency measurement
    pub fn record_latency(&self, operation: &str, latency: Duration) -> Result<()> {
        let mut tracker = self.latency_tracker.lock()
            .map_err(|_| TestFrameworkError::SetupError("Failed to acquire latency tracker lock".to_string()))?;
        
        tracker.measurements
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(latency);
        
        Ok(())
    }
    
    /// Record throughput measurement
    pub fn record_throughput(&self, operation: &str, count: u64, duration: Duration) -> Result<()> {
        let mut monitor = self.throughput_monitor.lock()
            .map_err(|_| TestFrameworkError::SetupError("Failed to acquire throughput monitor lock".to_string()))?;
        
        let measurement = ThroughputMeasurement {
            count,
            duration,
            timestamp: Instant::now(),
        };
        
        monitor.measurements
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(measurement);
        
        Ok(())
    }
    
    /// Record memory usage measurement
    pub fn record_memory_usage(&self, operation: &str) -> Result<()> {
        let memory_usage = self.get_current_memory_usage()?;
        
        let mut tracker = self.memory_tracker.lock()
            .map_err(|_| TestFrameworkError::SetupError("Failed to acquire memory tracker lock".to_string()))?;
        
        tracker.measurements.push(MemoryMeasurement {
            timestamp: Instant::now(),
            memory_usage_bytes: memory_usage,
            operation: operation.to_string(),
        });
        
        Ok(())
    }
    
    /// Record a custom metric value
    pub fn record_custom_metric(&self, metric_name: &str, value: f64) -> Result<()> {
        let mut metrics = self.custom_metrics.lock()
            .map_err(|_| TestFrameworkError::SetupError("Failed to acquire custom metrics lock".to_string()))?;
        
        metrics.entry(metric_name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
        
        Ok(())
    }
    
    /// Generate comprehensive performance report
    pub fn get_performance_report(&self) -> PerformanceReport {
        let latency_tracker = self.latency_tracker.lock().unwrap();
        let throughput_monitor = self.throughput_monitor.lock().unwrap();
        let memory_tracker = self.memory_tracker.lock().unwrap();
        let custom_metrics = self.custom_metrics.lock().unwrap();
        
        PerformanceReport {
            end_to_end_latency: self.calculate_latency_stats(&latency_tracker, "end_to_end"),
            feature_computation_latency: self.calculate_latency_stats(&latency_tracker, "feature_computation"),
            signal_generation_latency: self.calculate_latency_stats(&latency_tracker, "signal_generation"),
            signal_emission_latency: self.calculate_latency_stats(&latency_tracker, "signal_emission"),
            throughput_stats: self.calculate_throughput_stats(&throughput_monitor),
            memory_usage: self.calculate_memory_stats(&memory_tracker),
            custom_metrics: self.calculate_custom_metrics_stats(&custom_metrics),
        }
    }
    
    /// Get current memory usage in bytes
    fn get_current_memory_usage(&self) -> Result<u64> {
        // This is a simplified implementation. In a real system, you would use
        // platform-specific APIs to get actual memory usage.
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status")
                .map_err(|e| TestFrameworkError::SetupError(format!("Failed to read memory info: {}", e)))?;
            
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: u64 = parts[1].parse()
                            .map_err(|e| TestFrameworkError::SetupError(format!("Failed to parse memory value: {}", e)))?;
                        return Ok(kb * 1024); // Convert KB to bytes
                    }
                }
            }
        }
        
        // Fallback for other platforms or if reading fails
        Ok(0)
    }
    
    /// Calculate latency statistics for a specific operation
    fn calculate_latency_stats(&self, tracker: &LatencyTracker, operation: &str) -> LatencyStats {
        if let Some(measurements) = tracker.measurements.get(operation) {
            if measurements.is_empty() {
                return LatencyStats::default();
            }
            
            let mut durations_ms: Vec<f64> = measurements
                .iter()
                .map(|d| d.as_secs_f64() * 1000.0)
                .collect();
            durations_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let count = durations_ms.len();
            let mean = durations_ms.iter().sum::<f64>() / count as f64;
            let median = durations_ms[count / 2];
            let p95 = durations_ms[(count as f64 * 0.95) as usize];
            let p99 = durations_ms[(count as f64 * 0.99) as usize];
            let min = durations_ms[0];
            let max = durations_ms[count - 1];
            
            let variance = durations_ms.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / count as f64;
            let std_dev = variance.sqrt();
            
            LatencyStats {
                mean,
                median,
                p95,
                p99,
                min,
                max,
                std_dev,
                count,
            }
        } else {
            LatencyStats::default()
        }
    }
    
    /// Calculate throughput statistics
    fn calculate_throughput_stats(&self, monitor: &ThroughputMonitor) -> ThroughputStats {
        let all_measurements: Vec<&ThroughputMeasurement> = monitor.measurements
            .values()
            .flat_map(|measurements| measurements.iter())
            .collect();
        
        if all_measurements.is_empty() {
            return ThroughputStats::default();
        }
        
        let total_operations: u64 = all_measurements.iter().map(|m| m.count).sum();
        let total_duration: Duration = all_measurements.iter().map(|m| m.duration).sum();
        let total_duration_seconds = total_duration.as_secs_f64();
        
        let ops_per_second: Vec<f64> = all_measurements
            .iter()
            .map(|m| m.count as f64 / m.duration.as_secs_f64())
            .collect();
        
        let average_ops_per_second = if total_duration_seconds > 0.0 {
            total_operations as f64 / total_duration_seconds
        } else {
            0.0
        };
        
        let peak_ops_per_second = ops_per_second.iter().cloned().fold(0.0, f64::max);
        let min_ops_per_second = ops_per_second.iter().cloned().fold(f64::INFINITY, f64::min);
        
        ThroughputStats {
            average_ops_per_second,
            peak_ops_per_second,
            min_ops_per_second: if min_ops_per_second == f64::INFINITY { 0.0 } else { min_ops_per_second },
            total_operations,
            total_duration_seconds,
        }
    }
    
    /// Calculate memory usage statistics
    fn calculate_memory_stats(&self, tracker: &MemoryTracker) -> MemoryStats {
        if tracker.measurements.is_empty() {
            return MemoryStats::default();
        }
        
        let memory_values_mb: Vec<f64> = tracker.measurements
            .iter()
            .map(|m| m.memory_usage_bytes as f64 / (1024.0 * 1024.0))
            .collect();
        
        let peak_memory_mb = memory_values_mb.iter().cloned().fold(0.0, f64::max);
        let average_memory_mb = memory_values_mb.iter().sum::<f64>() / memory_values_mb.len() as f64;
        let baseline_memory_mb = tracker.baseline_memory
            .map(|b| b as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        let memory_growth_mb = peak_memory_mb - baseline_memory_mb;
        
        MemoryStats {
            peak_memory_mb,
            average_memory_mb,
            baseline_memory_mb,
            memory_growth_mb,
            measurement_count: tracker.measurements.len(),
        }
    }
    
    /// Calculate statistics for custom metrics
    fn calculate_custom_metrics_stats(&self, metrics: &HashMap<String, Vec<f64>>) -> HashMap<String, MetricStats> {
        let mut stats = HashMap::new();
        
        for (name, values) in metrics {
            if !values.is_empty() {
                let mut sorted_values = values.clone();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let count = sorted_values.len();
                let mean = sorted_values.iter().sum::<f64>() / count as f64;
                let median = sorted_values[count / 2];
                let min = sorted_values[0];
                let max = sorted_values[count - 1];
                
                let variance = sorted_values.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / count as f64;
                let std_dev = variance.sqrt();
                
                stats.insert(name.clone(), MetricStats {
                    mean,
                    median,
                    min,
                    max,
                    std_dev,
                    count,
                });
            }
        }
        
        stats
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            p95: 0.0,
            p99: 0.0,
            min: 0.0,
            max: 0.0,
            std_dev: 0.0,
            count: 0,
        }
    }
}

impl Default for ThroughputStats {
    fn default() -> Self {
        Self {
            average_ops_per_second: 0.0,
            peak_ops_per_second: 0.0,
            min_ops_per_second: 0.0,
            total_operations: 0,
            total_duration_seconds: 0.0,
        }
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            peak_memory_mb: 0.0,
            average_memory_mb: 0.0,
            baseline_memory_mb: 0.0,
            memory_growth_mb: 0.0,
            measurement_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration as StdDuration;
    
    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        let report = monitor.get_performance_report();
        assert_eq!(report.end_to_end_latency.count, 0);
    }
    
    #[test]
    fn test_latency_measurement() {
        let monitor = PerformanceMonitor::new();
        
        monitor.start_latency_measurement("test_operation").unwrap();
        thread::sleep(StdDuration::from_millis(10));
        let duration = monitor.end_latency_measurement("test_operation").unwrap();
        
        assert!(duration.as_millis() >= 10);
        
        let report = monitor.get_performance_report();
        // Note: This will be 0 because we're looking for "end_to_end" specifically
        assert_eq!(report.end_to_end_latency.count, 0);
    }
    
    #[test]
    fn test_throughput_measurement() {
        let monitor = PerformanceMonitor::new();
        
        monitor.record_throughput("test_op", 100, StdDuration::from_secs(1)).unwrap();
        monitor.record_throughput("test_op", 200, StdDuration::from_secs(2)).unwrap();
        
        let report = monitor.get_performance_report();
        assert!(report.throughput_stats.total_operations > 0);
        assert!(report.throughput_stats.average_ops_per_second > 0.0);
    }
    
    #[test]
    fn test_custom_metrics() {
        let monitor = PerformanceMonitor::new();
        
        monitor.record_custom_metric("test_metric", 1.0).unwrap();
        monitor.record_custom_metric("test_metric", 2.0).unwrap();
        monitor.record_custom_metric("test_metric", 3.0).unwrap();
        
        let report = monitor.get_performance_report();
        assert!(report.custom_metrics.contains_key("test_metric"));
        
        let stats = &report.custom_metrics["test_metric"];
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean, 2.0);
        assert_eq!(stats.median, 2.0);
    }
    
    #[test]
    fn test_measurement_session() {
        let monitor = PerformanceMonitor::new();
        
        monitor.start_measurement("test_session").unwrap();
        thread::sleep(StdDuration::from_millis(10));
        monitor.end_measurement().unwrap();
        
        // Session should be recorded internally
        let session = monitor.current_session.lock().unwrap();
        assert!(session.is_some());
        assert!(session.as_ref().unwrap().end_time.is_some());
    }
}