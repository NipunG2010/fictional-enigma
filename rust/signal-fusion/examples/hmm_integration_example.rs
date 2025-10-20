//! HMM Service Integration Example
//!
//! This example demonstrates how to integrate the Rust inference engine
//! with the HMM microservice for regime-aware signal fusion.
//!
//! Run with: cargo run --example hmm_integration_example

use anyhow::Result;
use signal_fusion::{
    hmm_client::{HmmClient, HmmClientConfig, HmmIntegration},
    SignalComponents, SignalFusion, FusionWeights,
};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn, error, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting HMM Service Integration Example");

    // Example 1: Basic HMM client usage
    basic_hmm_client_example().await?;

    // Example 2: Signal fusion with HMM integration
    signal_fusion_with_hmm_example().await?;

    // Example 3: Error handling and fallback scenarios
    error_handling_example().await?;

    // Example 4: Production-ready integration pattern
    production_integration_example().await?;

    info!("HMM Service Integration Example completed");
    Ok(())
}

/// Example 1: Basic HMM client usage
async fn basic_hmm_client_example() -> Result<()> {
    info!("=== Example 1: Basic HMM Client Usage ===");

    // Create HMM client with custom configuration
    let config = HmmClientConfig {
        base_url: "http://localhost:8000".parse()?,
        timeout: Duration::from_millis(5000),
        retry_attempts: 3,
        retry_delay: Duration::from_millis(100),
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        },
        circuit_breaker_threshold: 5,
        circuit_breaker_timeout: Duration::from_secs(30),
    };

    let client = HmmClient::with_config(config)?;

    // Check service health
    match client.health_check().await {
        Ok(health) => {
            info!("HMM Service Health: {} (uptime: {:?}s)", 
                  health.status, health.uptime);
        }
        Err(e) => {
            warn!("Health check failed: {}", e);
        }
    }

    // Check service readiness
    match client.readiness_check().await {
        Ok(readiness) => {
            info!("HMM Service Ready: {} (model loaded: {})", 
                  readiness.ready, readiness.model_loaded);
        }
        Err(e) => {
            warn!("Readiness check failed: {}", e);
        }
    }

    // Get model information
    match client.get_model_info().await {
        Ok(model_info) => {
            info!("Model Info: loaded={}, version={:?}, states={:?}", 
                  model_info.loaded, model_info.version, model_info.n_states);
        }
        Err(e) => {
            warn!("Model info request failed: {}", e);
        }
    }

    // Example market observations [s_ldc, s_mr, s_tsmom]
    let observations = [0.015, -0.008, 0.022];
    
    // Get state probabilities
    match client.get_state_probabilities(observations, Some("example_1".to_string())).await {
        Ok(response) => {
            info!("State Probabilities: {:?}", response.state_probabilities);
            info!("Most Likely State: {} (confidence: {:.3})", 
                  response.most_likely_state, response.confidence);
            info!("Processing Time: {:.2}ms", response.processing_time_ms);
        }
        Err(e) => {
            warn!("State probabilities request failed: {}", e);
        }
    }

    // Get fusion weights
    match client.get_fusion_weights(observations, Some("example_1".to_string())).await {
        Ok(response) => {
            info!("Fusion Weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
                  response.weights.w_ldc, response.weights.w_mr, response.weights.w_tsmom);
        }
        Err(e) => {
            warn!("Fusion weights request failed: {}", e);
        }
    }

    // Get complete prediction
    match client.predict(observations, Some("example_1".to_string())).await {
        Ok(response) => {
            info!("Complete Prediction:");
            info!("  State Probabilities: {:?}", response.state_probabilities);
            info!("  Fusion Weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
                  response.fusion_weights.w_ldc, 
                  response.fusion_weights.w_mr, 
                  response.fusion_weights.w_tsmom);
            info!("  Model Version: {}", response.model_version);
        }
        Err(e) => {
            warn!("Complete prediction failed: {}", e);
        }
    }

    Ok(())
}

/// Example 2: Signal fusion with HMM integration
async fn signal_fusion_with_hmm_example() -> Result<()> {
    info!("=== Example 2: Signal Fusion with HMM Integration ===");

    // Create HMM integration
    let mut hmm_integration = HmmIntegration::new()?;

    // Create signal fusion engine
    let mut signal_fusion = SignalFusion::new(0.1, 60); // 0.1 threshold, 60s cooldown

    // Check if HMM service is ready
    let service_ready = hmm_integration.is_service_ready().await;
    info!("HMM Service Ready: {}", service_ready);

    // Simulate market data over time
    let market_scenarios = vec![
        // Bull market scenario
        SignalComponents { s_ldc: 0.08, s_mr: -0.02, s_tsmom: 0.15 },
        // Neutral market scenario  
        SignalComponents { s_ldc: 0.01, s_mr: 0.00, s_tsmom: -0.01 },
        // Bear market scenario
        SignalComponents { s_ldc: -0.12, s_mr: 0.08, s_tsmom: -0.18 },
        // Volatile scenario
        SignalComponents { s_ldc: 0.05, s_mr: 0.03, s_tsmom: 0.02 },
    ];

    for (i, signal_components) in market_scenarios.iter().enumerate() {
        info!("Processing market scenario {}: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
              i + 1, signal_components.s_ldc, signal_components.s_mr, signal_components.s_tsmom);

        // Get regime-aware fusion weights from HMM service
        let fusion_weights = hmm_integration
            .get_fusion_weights_for_signals(signal_components)
            .await?;

        info!("HMM Fusion Weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
              fusion_weights.w_ldc, fusion_weights.w_mr, fusion_weights.w_tsmom);

        // Generate trading signal using fusion weights
        let timestamp = chrono::Utc::now().timestamp() + i as i64;
        
        match signal_fusion.fuse_signals(
            signal_components.clone(),
            fusion_weights,
            timestamp,
            "BTCUSDT",
            "hmm_v2.1.0",
        )? {
            Some(trading_signal) => {
                info!("Generated Trading Signal:");
                info!("  Symbol: {}", trading_signal.symbol);
                info!("  Side: {} (strength: {:.3})", trading_signal.side, trading_signal.strength);
                info!("  Confidence: {:.3}", trading_signal.confidence);
                info!("  Model Version: {}", trading_signal.model_version);
            }
            None => {
                info!("No trading signal generated (below threshold or cooldown)");
            }
        }

        // Small delay between scenarios
        sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}

/// Example 3: Error handling and fallback scenarios
async fn error_handling_example() -> Result<()> {
    info!("=== Example 3: Error Handling and Fallback Scenarios ===");

    // Create client with aggressive circuit breaker for demonstration
    let config = HmmClientConfig {
        base_url: "http://localhost:8000".parse()?,
        timeout: Duration::from_millis(1000),
        retry_attempts: 2,
        retry_delay: Duration::from_millis(50),
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        },
        circuit_breaker_threshold: 2,
        circuit_breaker_timeout: Duration::from_secs(5),
    };

    let _client = HmmClient::with_config(config)?;

    // Test with invalid service URL to trigger fallback
    let invalid_config = HmmClientConfig {
        base_url: "http://invalid-host:9999".parse()?,
        ..HmmClientConfig::default()
    };

    let invalid_client = HmmClient::with_config(invalid_config.clone())?;

    let observations = [0.01, -0.02, 0.03];

    // This should trigger fallback mechanisms
    match invalid_client.get_fusion_weights(observations, Some("error_test".to_string())).await {
        Ok(response) => {
            info!("Fallback weights used: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
                  response.weights.w_ldc, response.weights.w_mr, response.weights.w_tsmom);
        }
        Err(e) => {
            error!("Even fallback failed: {}", e);
        }
    }

    // Check circuit breaker status
    let (cb_state, failure_count) = invalid_client.get_circuit_breaker_status();
    info!("Circuit Breaker State: {} (failures: {})", cb_state, failure_count);

    // Demonstrate HMM integration fallback
    let mut hmm_integration = HmmIntegration::with_config(invalid_config)?;
    
    let signal_components = SignalComponents { 
        s_ldc: 0.05, 
        s_mr: -0.01, 
        s_tsmom: 0.08 
    };

    // This should gracefully fall back to default weights
    let fallback_weights = hmm_integration
        .get_fusion_weights_for_signals(&signal_components)
        .await?;

    info!("Integration fallback weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
          fallback_weights.w_ldc, fallback_weights.w_mr, fallback_weights.w_tsmom);

    Ok(())
}

/// Example 4: Production-ready integration pattern
async fn production_integration_example() -> Result<()> {
    info!("=== Example 4: Production-Ready Integration Pattern ===");

    // Production configuration with conservative settings
    let production_config = HmmClientConfig {
        base_url: "http://localhost:8000".parse()?,
        timeout: Duration::from_millis(2000),
        retry_attempts: 3,
        retry_delay: Duration::from_millis(200),
        enable_fallback: true,
        fallback_weights: FusionWeights {
            w_ldc: 0.35,
            w_mr: 0.35,
            w_tsmom: 0.30,
        },
        circuit_breaker_threshold: 5,
        circuit_breaker_timeout: Duration::from_secs(60),
    };

    let mut hmm_integration = HmmIntegration::with_config(production_config)?;
    let mut signal_fusion = SignalFusion::new(0.05, 30); // Conservative threshold and cooldown

    // Production monitoring loop simulation
    info!("Starting production monitoring simulation...");

    for iteration in 1..=5 {
        info!("Production iteration {}", iteration);

        // Check service health before processing
        let (service_ready, model_version) = hmm_integration.get_service_status().await?;
        
        if !service_ready {
            warn!("HMM service not ready, using fallback weights");
        } else {
            info!("HMM service ready with model version: {:?}", model_version);
        }

        // Simulate real market data
        let market_data = SignalComponents {
            s_ldc: 0.02 + (iteration as f32 * 0.01),
            s_mr: -0.01 + (iteration as f32 * 0.005),
            s_tsmom: 0.03 - (iteration as f32 * 0.008),
        };

        info!("Market signals: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
              market_data.s_ldc, market_data.s_mr, market_data.s_tsmom);

        // Get fusion weights with automatic fallback
        let start_time = std::time::Instant::now();
        let fusion_weights = hmm_integration
            .get_fusion_weights_for_signals(&market_data)
            .await?;
        let request_duration = start_time.elapsed();

        info!("Fusion weights obtained in {:?}: LDC={:.3}, MR={:.3}, TSMOM={:.3}", 
              request_duration, fusion_weights.w_ldc, fusion_weights.w_mr, fusion_weights.w_tsmom);

        // Generate trading signal
        let timestamp = chrono::Utc::now().timestamp();
        
        match signal_fusion.fuse_signals(
            market_data,
            fusion_weights,
            timestamp,
            "BTCUSDT",
            "production_v1.0",
        )? {
            Some(signal) => {
                info!("TRADING SIGNAL: {} {} with strength {:.3} (confidence: {:.3})", 
                      signal.side, signal.symbol, signal.strength, signal.confidence);
                
                // In production, this is where you would:
                // 1. Validate the signal
                // 2. Apply risk management rules
                // 3. Execute the trade
                // 4. Log for audit trail
            }
            None => {
                info!("No signal generated (threshold/cooldown)");
            }
        }

        // Monitor circuit breaker status
        let (cb_state, failure_count) = hmm_integration.get_circuit_breaker_status();
        if failure_count > 0 {
            warn!("Circuit breaker status: {} (failures: {})", cb_state, failure_count);
        }

        // Production delay between iterations
        sleep(Duration::from_millis(500)).await;
    }

    info!("Production simulation completed");
    Ok(())
}

/// Helper function to demonstrate error scenarios
#[allow(dead_code)]
async fn demonstrate_error_scenarios() -> Result<()> {
    info!("=== Error Scenario Demonstrations ===");

    let client = HmmClient::new()?;

    // Test invalid observations
    let invalid_observations = [f32::NAN, 0.0, 0.0];
    match client.get_state_probabilities(invalid_observations, None).await {
        Ok(_) => info!("Unexpected success with NaN observations"),
        Err(e) => info!("Expected error with NaN observations: {}", e),
    }

    // Test service timeout (if service is slow)
    let timeout_config = HmmClientConfig {
        timeout: Duration::from_millis(1),
        ..HmmClientConfig::default()
    };
    
    let timeout_client = HmmClient::with_config(timeout_config)?;
    match timeout_client.get_state_probabilities([0.1, 0.2, 0.3], None).await {
        Ok(_) => info!("Request completed within 1ms (unlikely)"),
        Err(e) => info!("Expected timeout error: {}", e),
    }

    Ok(())
}

/// Configuration examples for different environments
#[allow(dead_code)]
fn get_environment_configs() -> Vec<(&'static str, HmmClientConfig)> {
    vec![
        ("development", HmmClientConfig {
            base_url: "http://localhost:8000".parse().unwrap(),
            timeout: Duration::from_millis(10000),
            retry_attempts: 1,
            retry_delay: Duration::from_millis(100),
            enable_fallback: true,
            fallback_weights: FusionWeights { w_ldc: 0.33, w_mr: 0.33, w_tsmom: 0.34 },
            circuit_breaker_threshold: 10,
            circuit_breaker_timeout: Duration::from_secs(30),
        }),
        ("staging", HmmClientConfig {
            base_url: "http://hmm-service-staging:8000".parse().unwrap(),
            timeout: Duration::from_millis(5000),
            retry_attempts: 2,
            retry_delay: Duration::from_millis(150),
            enable_fallback: true,
            fallback_weights: FusionWeights { w_ldc: 0.4, w_mr: 0.3, w_tsmom: 0.3 },
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(60),
        }),
        ("production", HmmClientConfig {
            base_url: "http://hmm-service:8000".parse().unwrap(),
            timeout: Duration::from_millis(2000),
            retry_attempts: 3,
            retry_delay: Duration::from_millis(200),
            enable_fallback: true,
            fallback_weights: FusionWeights { w_ldc: 0.35, w_mr: 0.35, w_tsmom: 0.30 },
            circuit_breaker_threshold: 3,
            circuit_breaker_timeout: Duration::from_secs(120),
        }),
    ]
}