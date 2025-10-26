//! Metrics server demonstration
//!
//! This example shows how to use the signal emission metrics server
//! to export Prometheus metrics via HTTP endpoints.
//!
//! Run with: cargo run --example metrics_server_demo
//! Then visit: http://localhost:9090/metrics

use signal_fusion::emission::{
    MetricsServer, MetricsServerConfig, SignalEmissionMetrics, MetricsTimer
};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();
    
    info!("Starting signal emission metrics server demo");
    
    // Create metrics collector
    let metrics = Arc::new(SignalEmissionMetrics::new()?);
    
    // Configure metrics server
    let config = MetricsServerConfig {
        bind_address: "0.0.0.0".to_string(),
        port: 9090,
        enable_health_endpoint: true,
        enable_json_endpoint: true,
        request_timeout_seconds: 30,
        max_connections: 100,
    };
    
    // Start metrics server
    let mut server = MetricsServer::new(config, metrics.clone());
    server.start().await?;
    
    info!("Metrics server started on http://localhost:9090");
    info!("Available endpoints:");
    info!("  - http://localhost:9090/metrics (Prometheus format)");
    info!("  - http://localhost:9090/health (Health check)");
    info!("  - http://localhost:9090/metrics/json (JSON format)");
    
    // Simulate signal emission activity
    tokio::spawn(async move {
        let symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"];
        let backends = ["redis", "kafka"];
        let sides = ["BUY", "SELL", "HOLD"];
        
        loop {
            // Simulate signal publishing
            for symbol in &symbols {
                for backend in &backends {
                    for side in &sides {
                        let timer = MetricsTimer::start();
                        
                        // Simulate some processing time
                        sleep(Duration::from_millis(rand::random::<u64>() % 50)).await;
                        
                        // Record successful signal publication
                        metrics.record_signal_published(symbol, backend, side);
                        metrics.record_emission_latency(symbol, backend, timer.elapsed_seconds());
                        
                        // Occasionally simulate validation errors
                        if rand::random::<f64>() < 0.1 {
                            metrics.record_validation_error(symbol, "invalid_strength", "strength");
                        }
                        
                        // Occasionally simulate publisher errors
                        if rand::random::<f64>() < 0.05 {
                            metrics.record_publisher_error(backend, "connection_timeout");
                        }
                    }
                }
            }
            
            // Update buffer metrics
            let buffer_size = (rand::random::<u64>() % 100) as i64;
            metrics.update_buffer_size(buffer_size, 100);
            
            // Record some audit events
            metrics.record_audit_event("signal_emission");
            metrics.record_audit_event("feature_computation");
            
            // Update publisher connections
            for backend in &backends {
                let connections = (rand::random::<u64>() % 10) as i64;
                metrics.update_publisher_connections(backend, connections);
            }
            
            // Record health checks
            for component in &["redis", "kafka", "buffer", "audit"] {
                let healthy = rand::random::<f64>() > 0.1; // 90% healthy
                let latency = rand::random::<f64>() * 0.1; // 0-100ms
                metrics.record_health_check(component, latency, healthy);
            }
            
            sleep(Duration::from_secs(1)).await;
        }
    });
    
    // Wait for shutdown signal
    info!("Press Ctrl+C to stop the server");
    tokio::signal::ctrl_c().await?;
    
    info!("Shutdown signal received, stopping server");
    server.stop().await;
    
    Ok(())
}