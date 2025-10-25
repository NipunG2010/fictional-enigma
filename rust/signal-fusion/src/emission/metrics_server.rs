//! HTTP metrics server for Prometheus integration
//!
//! This module provides HTTP endpoints for Prometheus metrics scraping:
//! - /metrics - Prometheus text format metrics
//! - /health - Health check endpoint
//! - /metrics/json - JSON format metrics (for debugging)
//!
//! The server runs asynchronously and can be integrated into existing applications
//! or run as a standalone metrics endpoint.

use crate::emission::{SignalEmissionMetrics, SignalEmissionError, Result};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use prometheus::{Encoder, TextEncoder};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{error, info};

/// Configuration for the metrics HTTP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsServerConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// Enable health check endpoint
    pub enable_health_endpoint: bool,
    /// Enable JSON metrics endpoint (for debugging)
    pub enable_json_endpoint: bool,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Maximum concurrent connections
    pub max_connections: usize,
}

impl Default for MetricsServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 9090,
            enable_health_endpoint: true,
            enable_json_endpoint: false,
            request_timeout_seconds: 30,
            max_connections: 100,
        }
    }
}

/// HTTP metrics server for Prometheus integration
pub struct MetricsServer {
    config: MetricsServerConfig,
    metrics: Arc<SignalEmissionMetrics>,
    server_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MetricsServer {
    /// Create a new metrics server
    pub fn new(config: MetricsServerConfig, metrics: Arc<SignalEmissionMetrics>) -> Self {
        Self {
            config,
            metrics,
            server_handle: None,
        }
    }
    
    /// Start the metrics server
    pub async fn start(&mut self) -> Result<()> {
        let addr = format!("{}:{}", self.config.bind_address, self.config.port)
            .parse::<SocketAddr>()
            .map_err(|e| SignalEmissionError::config(format!("Invalid bind address: {}", e)))?;
        
        let metrics = self.metrics.clone();
        let config = self.config.clone();
        
        let make_svc = make_service_fn(move |_conn| {
            let metrics = metrics.clone();
            let config = config.clone();
            
            async move {
                Ok::<_, Infallible>(service_fn(move |req| {
                    handle_request(req, metrics.clone(), config.clone())
                }))
            }
        });
        
        let server = Server::bind(&addr).serve(make_svc);
        
        info!("Starting metrics server on {}", addr);
        
        let server_handle = tokio::spawn(async move {
            if let Err(e) = server.await {
                error!("Metrics server error: {}", e);
            }
        });
        
        self.server_handle = Some(server_handle);
        
        Ok(())
    }
    
    /// Stop the metrics server
    pub async fn stop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
            info!("Metrics server stopped");
        }
    }
    
    /// Check if the server is running
    pub fn is_running(&self) -> bool {
        self.server_handle
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false)
    }
}

impl Drop for MetricsServer {
    fn drop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }
    }
}

/// Handle HTTP requests to the metrics server
async fn handle_request(
    req: Request<Body>,
    metrics: Arc<SignalEmissionMetrics>,
    config: MetricsServerConfig,
) -> std::result::Result<Response<Body>, Infallible> {
    let response = match (req.method(), req.uri().path()) {
        (&Method::GET, "/metrics") => handle_prometheus_metrics(metrics).await,
        (&Method::GET, "/health") if config.enable_health_endpoint => handle_health_check().await,
        (&Method::GET, "/metrics/json") if config.enable_json_endpoint => {
            handle_json_metrics(metrics).await
        }
        _ => handle_not_found().await,
    };
    
    Ok(response)
}

/// Handle Prometheus metrics endpoint
async fn handle_prometheus_metrics(metrics: Arc<SignalEmissionMetrics>) -> Response<Body> {
    match export_prometheus_metrics(&metrics).await {
        Ok(metrics_text) => Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            .body(Body::from(metrics_text))
            .unwrap_or_else(|e| {
                error!("Failed to build metrics response: {}", e);
                internal_server_error("Failed to build response")
            }),
        Err(e) => {
            error!("Failed to export Prometheus metrics: {}", e);
            internal_server_error("Failed to export metrics")
        }
    }
}

/// Handle health check endpoint
async fn handle_health_check() -> Response<Body> {
    let health_response = serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().timestamp(),
        "service": "signal-emission-metrics"
    });
    
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(health_response.to_string()))
        .unwrap_or_else(|e| {
            error!("Failed to build health response: {}", e);
            internal_server_error("Failed to build response")
        })
}

/// Handle JSON metrics endpoint (for debugging)
async fn handle_json_metrics(metrics: Arc<SignalEmissionMetrics>) -> Response<Body> {
    match export_json_metrics(&metrics).await {
        Ok(metrics_json) => Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(metrics_json))
            .unwrap_or_else(|e| {
                error!("Failed to build JSON metrics response: {}", e);
                internal_server_error("Failed to build response")
            }),
        Err(e) => {
            error!("Failed to export JSON metrics: {}", e);
            internal_server_error("Failed to export metrics")
        }
    }
}

/// Handle 404 Not Found
async fn handle_not_found() -> Response<Body> {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("Content-Type", "text/plain")
        .body(Body::from("Not Found"))
        .unwrap_or_else(|e| {
            error!("Failed to build 404 response: {}", e);
            internal_server_error("Failed to build response")
        })
}

/// Create internal server error response
fn internal_server_error(message: &str) -> Response<Body> {
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header("Content-Type", "text/plain")
        .body(Body::from(message.to_string()))
        .unwrap_or_else(|_| {
            // Fallback response if even this fails
            Response::new(Body::from("Internal Server Error"))
        })
}

/// Export metrics in Prometheus text format
async fn export_prometheus_metrics(metrics: &SignalEmissionMetrics) -> Result<String> {
    let registry = metrics.registry();
    let metric_families = registry.gather();
    
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();
    
    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|e| SignalEmissionError::InternalError(anyhow::anyhow!("Prometheus encoding error: {}", e)))?;
    
    String::from_utf8(buffer)
        .map_err(|e| SignalEmissionError::InternalError(anyhow::anyhow!("UTF-8 conversion error: {}", e)))
}

/// Export metrics in JSON format (for debugging)
async fn export_json_metrics(metrics: &SignalEmissionMetrics) -> Result<String> {
    let registry = metrics.registry();
    let metric_families = registry.gather();
    
    // Convert Prometheus metrics to a more readable JSON format
    let mut json_metrics = serde_json::Map::new();
    
    for family in metric_families {
        let family_name = family.get_name();
        let mut family_metrics = Vec::new();
        
        for metric in family.get_metric() {
            let mut metric_obj = serde_json::Map::new();
            
            // Add labels
            if !metric.get_label().is_empty() {
                let mut labels = serde_json::Map::new();
                for label in metric.get_label() {
                    labels.insert(label.get_name().to_string(), serde_json::Value::String(label.get_value().to_string()));
                }
                metric_obj.insert("labels".to_string(), serde_json::Value::Object(labels));
            }
            
            // Add value based on metric type
            if metric.has_counter() {
                metric_obj.insert("value".to_string(), serde_json::Value::Number(
                    serde_json::Number::from_f64(metric.get_counter().get_value()).unwrap_or(serde_json::Number::from(0))
                ));
            } else if metric.has_gauge() {
                metric_obj.insert("value".to_string(), serde_json::Value::Number(
                    serde_json::Number::from_f64(metric.get_gauge().get_value()).unwrap_or(serde_json::Number::from(0))
                ));
            } else if metric.has_histogram() {
                let histogram = metric.get_histogram();
                let mut hist_obj = serde_json::Map::new();
                hist_obj.insert("sample_count".to_string(), serde_json::Value::Number(
                    serde_json::Number::from(histogram.get_sample_count())
                ));
                hist_obj.insert("sample_sum".to_string(), serde_json::Value::Number(
                    serde_json::Number::from_f64(histogram.get_sample_sum()).unwrap_or(serde_json::Number::from(0))
                ));
                
                let mut buckets = Vec::new();
                for bucket in histogram.get_bucket() {
                    buckets.push(serde_json::json!({
                        "upper_bound": bucket.get_upper_bound(),
                        "cumulative_count": bucket.get_cumulative_count()
                    }));
                }
                hist_obj.insert("buckets".to_string(), serde_json::Value::Array(buckets));
                
                metric_obj.insert("histogram".to_string(), serde_json::Value::Object(hist_obj));
            }
            
            family_metrics.push(serde_json::Value::Object(metric_obj));
        }
        
        json_metrics.insert(family_name.to_string(), serde_json::Value::Array(family_metrics));
    }
    
    let response = serde_json::json!({
        "timestamp": chrono::Utc::now().timestamp(),
        "metrics": json_metrics
    });
    
    serde_json::to_string_pretty(&response)
        .map_err(|e| SignalEmissionError::SerializationError(e))
}

/// Standalone metrics server that can be run independently
pub struct StandaloneMetricsServer {
    server: MetricsServer,
}

impl StandaloneMetricsServer {
    /// Create a new standalone metrics server
    pub fn new(config: MetricsServerConfig) -> Result<Self> {
        let metrics = Arc::new(SignalEmissionMetrics::new()
            .map_err(|e| SignalEmissionError::InternalError(anyhow::anyhow!("Failed to create metrics: {}", e)))?);
        
        let server = MetricsServer::new(config, metrics);
        
        Ok(Self { server })
    }
    
    /// Start the server and run until shutdown
    pub async fn run(mut self) -> Result<()> {
        self.server.start().await?;
        
        // Wait for shutdown signal
        tokio::signal::ctrl_c().await
            .map_err(|e| SignalEmissionError::InternalError(anyhow::anyhow!("Failed to listen for shutdown signal: {}", e)))?;
        
        info!("Shutdown signal received, stopping metrics server");
        self.server.stop().await;
        
        Ok(())
    }
    
    /// Get access to the metrics for recording
    pub fn metrics(&self) -> Arc<SignalEmissionMetrics> {
        self.server.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    #[tokio::test]
    async fn test_metrics_server_creation() {
        let config = MetricsServerConfig::default();
        let metrics = Arc::new(SignalEmissionMetrics::new().unwrap());
        let server = MetricsServer::new(config, metrics);
        
        assert!(!server.is_running());
    }
    
    #[tokio::test]
    async fn test_prometheus_export() {
        let metrics = Arc::new(SignalEmissionMetrics::new().unwrap());
        
        // Record some test metrics
        metrics.record_signal_published("BTCUSDT", "redis", "BUY");
        metrics.record_validation_error("BTCUSDT", "invalid_strength", "strength");
        
        let prometheus_text = export_prometheus_metrics(&metrics).await.unwrap();
        
        assert!(prometheus_text.contains("signal_emission_signals_published_total"));
        assert!(prometheus_text.contains("signal_emission_validation_errors_total"));
    }
    
    #[tokio::test]
    async fn test_json_export() {
        let metrics = Arc::new(SignalEmissionMetrics::new().unwrap());
        
        // Record some test metrics
        metrics.record_signal_published("BTCUSDT", "redis", "BUY");
        metrics.update_buffer_size(50, 100);
        
        let json_text = export_json_metrics(&metrics).await.unwrap();
        let json_value: serde_json::Value = serde_json::from_str(&json_text).unwrap();
        
        assert!(json_value["metrics"].is_object());
        assert!(json_value["timestamp"].is_number());
    }
    
    #[tokio::test]
    async fn test_server_lifecycle() {
        let mut config = MetricsServerConfig::default();
        config.port = 9091; // Use different port to avoid conflicts
        
        let metrics = Arc::new(SignalEmissionMetrics::new().unwrap());
        let mut server = MetricsServer::new(config, metrics);
        
        // Start server
        server.start().await.unwrap();
        assert!(server.is_running());
        
        // Give server time to start
        sleep(Duration::from_millis(100)).await;
        
        // Stop server
        server.stop().await;
        assert!(!server.is_running());
    }
    
    #[test]
    fn test_config_default() {
        let config = MetricsServerConfig::default();
        
        assert_eq!(config.bind_address, "0.0.0.0");
        assert_eq!(config.port, 9090);
        assert!(config.enable_health_endpoint);
        assert!(!config.enable_json_endpoint);
    }
}