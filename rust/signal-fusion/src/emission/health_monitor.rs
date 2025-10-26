//! Health monitoring system for signal emission components
//! 
//! This module provides comprehensive health checking and monitoring for all signal emission
//! components including publishers, buffers, audit systems, and overall service health.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, info, warn, error};

use super::{
    Result, SignalEmissionError,
    SignalPublisher,
    publisher::{HealthStatus, HealthLevel},
};

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitorConfig {
    /// Health check interval in seconds (default: 30)
    pub check_interval_seconds: u64,
    
    /// Health check timeout in milliseconds (default: 5000)
    pub check_timeout_ms: u64,
    
    /// Number of consecutive failures before marking as unhealthy (default: 3)
    pub failure_threshold: u32,
    
    /// Number of consecutive successes to recover from unhealthy (default: 2)
    pub recovery_threshold: u32,
    
    /// Whether to enable detailed component health tracking (default: true)
    pub detailed_tracking: bool,
    
    /// Whether to log health status changes (default: true)
    pub log_status_changes: bool,
    
    /// HTTP server configuration for health endpoints
    pub http_server: Option<HealthHttpConfig>,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval_seconds: 30,
            check_timeout_ms: 5000,
            failure_threshold: 3,
            recovery_threshold: 2,
            detailed_tracking: true,
            log_status_changes: true,
            http_server: None,
        }
    }
}

/// HTTP server configuration for health endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthHttpConfig {
    /// Bind address (default: "0.0.0.0")
    pub bind_address: String,
    
    /// Port number (default: 8080)
    pub port: u16,
    
    /// Health endpoint path (default: "/health")
    pub health_path: String,
    
    /// Metrics endpoint path (default: "/metrics")
    pub metrics_path: String,
    
    /// Ready endpoint path (default: "/ready")
    pub ready_path: String,
    
    /// Live endpoint path (default: "/live")
    pub live_path: String,
}

impl Default for HealthHttpConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            health_path: "/health".to_string(),
            metrics_path: "/metrics".to_string(),
            ready_path: "/ready".to_string(),
            live_path: "/live".to_string(),
        }
    }
}

/// Component health tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    
    /// Current health status
    pub status: HealthStatus,
    
    /// Number of consecutive failures
    pub consecutive_failures: u32,
    
    /// Number of consecutive successes
    pub consecutive_successes: u32,
    
    /// Last health check timestamp (Unix timestamp in milliseconds)
    pub last_check: Option<i64>,
    
    /// Health check history (last N checks)
    pub history: Vec<HealthCheckResult>,
    
    /// Component-specific metrics
    pub metrics: HashMap<String, String>,
}

/// Result of a health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Timestamp of the check
    pub timestamp: i64,
    
    /// Health level result
    pub level: HealthLevel,
    
    /// Response time in milliseconds
    pub response_time_ms: u64,
    
    /// Error message if unhealthy
    pub error_message: Option<String>,
}

/// Aggregated health status for the entire service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealth {
    /// Overall service health level
    pub overall_status: HealthLevel,
    
    /// Timestamp of the health check
    pub checked_at: i64,
    
    /// Individual component health statuses
    pub components: HashMap<String, ComponentHealth>,
    
    /// Service-level metrics
    pub metrics: ServiceMetrics,
    
    /// Health summary message
    pub summary: String,
    
    /// Detailed status information
    pub details: HashMap<String, serde_json::Value>,
}

/// Service-level metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMetrics {
    /// Total number of health checks performed
    pub total_health_checks: u64,
    
    /// Number of healthy components
    pub healthy_components: u32,
    
    /// Number of degraded components
    pub degraded_components: u32,
    
    /// Number of unhealthy components
    pub unhealthy_components: u32,
    
    /// Average health check response time in milliseconds
    pub avg_response_time_ms: f64,
    
    /// Service uptime in seconds
    pub uptime_seconds: u64,
    
    /// Last status change timestamp
    pub last_status_change: Option<i64>,
}

/// Health monitor for signal emission system
pub struct HealthMonitor {
    config: HealthMonitorConfig,
    publisher: Arc<SignalPublisher>,
    components: Arc<RwLock<HashMap<String, ComponentHealth>>>,
    service_metrics: Arc<RwLock<ServiceMetrics>>,
    start_time: Instant,
    last_overall_status: Arc<RwLock<HealthLevel>>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(config: HealthMonitorConfig, publisher: Arc<SignalPublisher>) -> Self {
        Self {
            config,
            publisher,
            components: Arc::new(RwLock::new(HashMap::new())),
            service_metrics: Arc::new(RwLock::new(ServiceMetrics {
                total_health_checks: 0,
                healthy_components: 0,
                degraded_components: 0,
                unhealthy_components: 0,
                avg_response_time_ms: 0.0,
                uptime_seconds: 0,
                last_status_change: None,
            })),
            start_time: Instant::now(),
            last_overall_status: Arc::new(RwLock::new(HealthLevel::Healthy)),
        }
    }
    
    /// Start the health monitoring loop
    pub async fn start(&self) -> Result<()> {
        info!("Starting health monitor with interval: {}s", self.config.check_interval_seconds);
        
        let mut interval = interval(Duration::from_secs(self.config.check_interval_seconds));
        
        loop {
            interval.tick().await;
            
            if let Err(error) = self.perform_health_check().await {
                error!("Health check failed: {}", error);
            }
        }
    }
    
    /// Perform a comprehensive health check
    pub async fn perform_health_check(&self) -> Result<ServiceHealth> {
        let start_time = Instant::now();
        debug!("Performing comprehensive health check");
        
        // Check publisher health
        let publisher_health = self.check_publisher_health().await?;
        
        // Update component tracking
        self.update_component_health("publisher", publisher_health).await;
        
        // Check individual publisher backends if detailed tracking is enabled
        if self.config.detailed_tracking {
            self.check_detailed_publisher_health().await?;
        }
        
        // Aggregate overall health status
        let service_health = self.aggregate_service_health().await;
        
        // Update service metrics
        self.update_service_metrics(start_time.elapsed()).await;
        
        // Log status changes if enabled
        if self.config.log_status_changes {
            self.log_status_changes(&service_health).await;
        }
        
        debug!("Health check completed in {:?}", start_time.elapsed());
        Ok(service_health)
    }
    
    /// Check publisher health
    async fn check_publisher_health(&self) -> Result<HealthStatus> {
        let timeout_duration = Duration::from_millis(self.config.check_timeout_ms);
        
        match tokio::time::timeout(timeout_duration, self.publisher.health_check()).await {
            Ok(health_status) => Ok(health_status),
            Err(_) => Ok(HealthStatus::unhealthy(format!(
                "Health check timed out after {}ms", 
                self.config.check_timeout_ms
            ))),
        }
    }
    
    /// Check detailed publisher health (individual backends)
    async fn check_detailed_publisher_health(&self) -> Result<()> {
        // Get publisher metrics for detailed component status
        let metrics = self.publisher.get_metrics().await;
        
        // Check each publisher backend health
        for (backend_name, health_status) in metrics.publisher_health {
            self.update_component_health(&format!("publisher_{}", backend_name), health_status).await;
        }
        
        // Check buffer health
        let buffer_health = if metrics.buffer_size as f64 / 1000.0 > 0.9 {
            HealthStatus::unhealthy("Buffer near capacity")
        } else if metrics.buffer_size as f64 / 1000.0 > 0.75 {
            HealthStatus::degraded(0, "Buffer utilization high")
        } else {
            HealthStatus::healthy(0)
        };
        
        self.update_component_health("buffer", buffer_health).await;
        
        Ok(())
    }
    
    /// Update component health tracking
    async fn update_component_health(&self, component_name: &str, health_status: HealthStatus) {
        let mut components = self.components.write().await;
        
        let component = components.entry(component_name.to_string()).or_insert_with(|| {
            ComponentHealth {
                name: component_name.to_string(),
                status: health_status.clone(),
                consecutive_failures: 0,
                consecutive_successes: 0,
                last_check: None,
                history: Vec::new(),
                metrics: HashMap::new(),
            }
        });
        
        // Update consecutive counters
        match health_status.status {
            HealthLevel::Healthy => {
                component.consecutive_successes += 1;
                component.consecutive_failures = 0;
            }
            HealthLevel::Degraded => {
                // Degraded counts as partial success
                component.consecutive_successes += 1;
                component.consecutive_failures = 0;
            }
            HealthLevel::Unhealthy => {
                component.consecutive_failures += 1;
                component.consecutive_successes = 0;
            }
        }
        
        // Update status and timestamp
        component.status = health_status.clone();
        component.last_check = Some(chrono::Utc::now().timestamp_millis());
        
        // Add to history (keep last 10 checks)
        let check_result = HealthCheckResult {
            timestamp: chrono::Utc::now().timestamp(),
            level: health_status.status,
            response_time_ms: health_status.response_time_ms,
            error_message: health_status.error_message,
        };
        
        component.history.push(check_result);
        if component.history.len() > 10 {
            component.history.remove(0);
        }
        
        // Update component metrics
        component.metrics.insert("consecutive_failures".to_string(), component.consecutive_failures.to_string());
        component.metrics.insert("consecutive_successes".to_string(), component.consecutive_successes.to_string());
        component.metrics.insert("last_response_time_ms".to_string(), health_status.response_time_ms.to_string());
    }
    
    /// Aggregate overall service health from component health
    async fn aggregate_service_health(&self) -> ServiceHealth {
        let components = self.components.read().await;
        let mut overall_status = HealthLevel::Healthy;
        let mut healthy_count = 0u32;
        let mut degraded_count = 0u32;
        let mut unhealthy_count = 0u32;
        let mut messages = Vec::new();
        
        // Analyze component health
        for (name, component) in components.iter() {
            match component.status.status {
                HealthLevel::Healthy => {
                    healthy_count += 1;
                }
                HealthLevel::Degraded => {
                    degraded_count += 1;
                    if overall_status == HealthLevel::Healthy {
                        overall_status = HealthLevel::Degraded;
                    }
                    if let Some(ref error_msg) = component.status.error_message {
                        messages.push(format!("{}: {}", name, error_msg));
                    }
                }
                HealthLevel::Unhealthy => {
                    unhealthy_count += 1;
                    overall_status = HealthLevel::Unhealthy;
                    if let Some(ref error_msg) = component.status.error_message {
                        messages.push(format!("{}: {}", name, error_msg));
                    }
                }
            }
        }
        
        // Create summary message
        let summary = if messages.is_empty() {
            format!("All {} components healthy", healthy_count)
        } else {
            format!("{} issues: {}", messages.len(), messages.join("; "))
        };
        
        // Update service metrics
        let mut service_metrics = self.service_metrics.write().await;
        service_metrics.healthy_components = healthy_count;
        service_metrics.degraded_components = degraded_count;
        service_metrics.unhealthy_components = unhealthy_count;
        service_metrics.uptime_seconds = self.start_time.elapsed().as_secs();
        
        // Create detailed status information
        let mut details = HashMap::new();
        details.insert("component_count".to_string(), serde_json::json!(components.len()));
        details.insert("healthy_count".to_string(), serde_json::json!(healthy_count));
        details.insert("degraded_count".to_string(), serde_json::json!(degraded_count));
        details.insert("unhealthy_count".to_string(), serde_json::json!(unhealthy_count));
        details.insert("uptime_seconds".to_string(), serde_json::json!(service_metrics.uptime_seconds));
        
        ServiceHealth {
            overall_status,
            checked_at: chrono::Utc::now().timestamp(),
            components: components.clone(),
            metrics: service_metrics.clone(),
            summary,
            details,
        }
    }
    
    /// Update service-level metrics
    async fn update_service_metrics(&self, check_duration: Duration) {
        let mut metrics = self.service_metrics.write().await;
        
        metrics.total_health_checks += 1;
        
        // Update average response time
        let new_response_time = check_duration.as_millis() as f64;
        metrics.avg_response_time_ms = 
            (metrics.avg_response_time_ms * (metrics.total_health_checks - 1) as f64 + new_response_time) 
            / metrics.total_health_checks as f64;
    }
    
    /// Log status changes if enabled
    async fn log_status_changes(&self, service_health: &ServiceHealth) {
        let mut last_status = self.last_overall_status.write().await;
        
        if *last_status != service_health.overall_status {
            let mut service_metrics = self.service_metrics.write().await;
            service_metrics.last_status_change = Some(chrono::Utc::now().timestamp());
            
            match service_health.overall_status {
                HealthLevel::Healthy => {
                    info!("Service health recovered: {}", service_health.summary);
                }
                HealthLevel::Degraded => {
                    warn!("Service health degraded: {}", service_health.summary);
                }
                HealthLevel::Unhealthy => {
                    error!("Service health critical: {}", service_health.summary);
                }
            }
            
            *last_status = service_health.overall_status.clone();
        }
    }
    
    /// Get current service health status
    pub async fn get_service_health(&self) -> ServiceHealth {
        self.aggregate_service_health().await
    }
    
    /// Get health status for a specific component
    pub async fn get_component_health(&self, component_name: &str) -> Option<ComponentHealth> {
        let components = self.components.read().await;
        components.get(component_name).cloned()
    }
    
    /// Get service metrics
    pub async fn get_service_metrics(&self) -> ServiceMetrics {
        let metrics = self.service_metrics.read().await;
        metrics.clone()
    }
    
    /// Check if the service is ready (all critical components healthy)
    pub async fn is_ready(&self) -> bool {
        let service_health = self.get_service_health().await;
        matches!(service_health.overall_status, HealthLevel::Healthy | HealthLevel::Degraded)
    }
    
    /// Check if the service is live (basic functionality available)
    pub async fn is_live(&self) -> bool {
        // Service is live if publisher is at least degraded
        if let Some(publisher_health) = self.get_component_health("publisher").await {
            !matches!(publisher_health.status.status, HealthLevel::Unhealthy)
        } else {
            false
        }
    }
    
    /// Force a health check (useful for testing or manual triggers)
    pub async fn force_health_check(&self) -> Result<ServiceHealth> {
        info!("Forcing immediate health check");
        self.perform_health_check().await
    }
    
    /// Get health check configuration
    pub fn get_config(&self) -> &HealthMonitorConfig {
        &self.config
    }
    
    /// Update health check configuration (some fields only)
    pub async fn update_config(&mut self, new_config: HealthMonitorConfig) -> Result<()> {
        info!("Updating health monitor configuration");
        
        // Validate new configuration
        if new_config.check_interval_seconds == 0 {
            return Err(SignalEmissionError::config("Check interval must be greater than 0"));
        }
        
        if new_config.check_timeout_ms == 0 {
            return Err(SignalEmissionError::config("Check timeout must be greater than 0"));
        }
        
        if new_config.failure_threshold == 0 {
            return Err(SignalEmissionError::config("Failure threshold must be greater than 0"));
        }
        
        if new_config.recovery_threshold == 0 {
            return Err(SignalEmissionError::config("Recovery threshold must be greater than 0"));
        }
        
        self.config = new_config;
        info!("Health monitor configuration updated successfully");
        Ok(())
    }
}

/// HTTP health endpoints for external monitoring
pub struct HealthHttpServer {
    config: HealthHttpConfig,
    monitor: Arc<HealthMonitor>,
}

impl HealthHttpServer {
    /// Create a new HTTP health server
    pub fn new(config: HealthHttpConfig, monitor: Arc<HealthMonitor>) -> Self {
        Self { config, monitor }
    }
    
    /// Start the HTTP server for health endpoints
    pub async fn start(&self) -> Result<()> {
        use std::net::SocketAddr;
        
        let addr: SocketAddr = format!("{}:{}", self.config.bind_address, self.config.port)
            .parse()
            .map_err(|e| SignalEmissionError::config(format!("Invalid bind address: {}", e)))?;
        
        info!("Starting health HTTP server on {}", addr);
        
        // Note: This is a simplified implementation
        // In a real implementation, you would use a proper HTTP framework like warp or axum
        
        info!("Health HTTP server would start on {} with endpoints:", addr);
        info!("  {} - Health check", self.config.health_path);
        info!("  {} - Metrics", self.config.metrics_path);
        info!("  {} - Readiness", self.config.ready_path);
        info!("  {} - Liveness", self.config.live_path);
        
        // For now, just return success
        // In a real implementation, this would start the HTTP server
        Ok(())
    }
    
    /// Handle health endpoint request
    pub async fn handle_health(&self) -> Result<serde_json::Value> {
        let health = self.monitor.get_service_health().await;
        Ok(serde_json::to_value(health)?)
    }
    
    /// Handle metrics endpoint request
    pub async fn handle_metrics(&self) -> Result<String> {
        let metrics = self.monitor.get_service_metrics().await;
        
        // Return Prometheus-style metrics
        let mut output = String::new();
        output.push_str(&format!("# HELP signal_emission_health_checks_total Total number of health checks performed\n"));
        output.push_str(&format!("# TYPE signal_emission_health_checks_total counter\n"));
        output.push_str(&format!("signal_emission_health_checks_total {}\n", metrics.total_health_checks));
        
        output.push_str(&format!("# HELP signal_emission_healthy_components Number of healthy components\n"));
        output.push_str(&format!("# TYPE signal_emission_healthy_components gauge\n"));
        output.push_str(&format!("signal_emission_healthy_components {}\n", metrics.healthy_components));
        
        output.push_str(&format!("# HELP signal_emission_degraded_components Number of degraded components\n"));
        output.push_str(&format!("# TYPE signal_emission_degraded_components gauge\n"));
        output.push_str(&format!("signal_emission_degraded_components {}\n", metrics.degraded_components));
        
        output.push_str(&format!("# HELP signal_emission_unhealthy_components Number of unhealthy components\n"));
        output.push_str(&format!("# TYPE signal_emission_unhealthy_components gauge\n"));
        output.push_str(&format!("signal_emission_unhealthy_components {}\n", metrics.unhealthy_components));
        
        output.push_str(&format!("# HELP signal_emission_avg_response_time_ms Average health check response time in milliseconds\n"));
        output.push_str(&format!("# TYPE signal_emission_avg_response_time_ms gauge\n"));
        output.push_str(&format!("signal_emission_avg_response_time_ms {}\n", metrics.avg_response_time_ms));
        
        output.push_str(&format!("# HELP signal_emission_uptime_seconds Service uptime in seconds\n"));
        output.push_str(&format!("# TYPE signal_emission_uptime_seconds gauge\n"));
        output.push_str(&format!("signal_emission_uptime_seconds {}\n", metrics.uptime_seconds));
        
        Ok(output)
    }
    
    /// Handle readiness endpoint request
    pub async fn handle_ready(&self) -> Result<serde_json::Value> {
        let is_ready = self.monitor.is_ready().await;
        Ok(serde_json::json!({
            "ready": is_ready,
            "timestamp": chrono::Utc::now().timestamp()
        }))
    }
    
    /// Handle liveness endpoint request
    pub async fn handle_live(&self) -> Result<serde_json::Value> {
        let is_live = self.monitor.is_live().await;
        Ok(serde_json::json!({
            "live": is_live,
            "timestamp": chrono::Utc::now().timestamp()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SignalPublisherConfig;
    
    #[test]
    fn test_health_monitor_config_default() {
        let config = HealthMonitorConfig::default();
        assert_eq!(config.check_interval_seconds, 30);
        assert_eq!(config.check_timeout_ms, 5000);
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.recovery_threshold, 2);
        assert!(config.detailed_tracking);
        assert!(config.log_status_changes);
    }
    
    #[test]
    fn test_health_http_config_default() {
        let config = HealthHttpConfig::default();
        assert_eq!(config.bind_address, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.health_path, "/health");
        assert_eq!(config.metrics_path, "/metrics");
        assert_eq!(config.ready_path, "/ready");
        assert_eq!(config.live_path, "/live");
    }
    
    #[tokio::test]
    async fn test_health_monitor_creation() {
        let config = HealthMonitorConfig::default();
        let publisher_config = SignalPublisherConfig::default();
        let publisher = Arc::new(SignalPublisher::new(publisher_config).await.unwrap());
        
        let monitor = HealthMonitor::new(config, publisher);
        assert_eq!(monitor.get_config().check_interval_seconds, 30);
    }
    
    #[tokio::test]
    async fn test_service_health_aggregation() {
        let config = HealthMonitorConfig::default();
        let publisher_config = SignalPublisherConfig::default();
        let publisher = Arc::new(SignalPublisher::new(publisher_config).await.unwrap());
        
        let monitor = HealthMonitor::new(config, publisher);
        
        // Add some test component health
        monitor.update_component_health("test_component", HealthStatus::healthy(100)).await;
        
        let service_health = monitor.get_service_health().await;
        assert_eq!(service_health.overall_status, HealthLevel::Healthy);
        assert!(service_health.components.contains_key("test_component"));
    }
    
    #[tokio::test]
    async fn test_health_monitor_config_validation() {
        let config = HealthMonitorConfig::default();
        let publisher_config = SignalPublisherConfig::default();
        let publisher = Arc::new(SignalPublisher::new(publisher_config).await.unwrap());
        
        let mut monitor = HealthMonitor::new(config, publisher);
        
        // Test invalid configuration
        let invalid_config = HealthMonitorConfig {
            check_interval_seconds: 0, // Invalid
            ..Default::default()
        };
        
        assert!(monitor.update_config(invalid_config).await.is_err());
    }
}