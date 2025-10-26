//! Signal emission integration for SignalFusion workflow
//! 
//! This module provides the SignalEmitter that integrates with the SignalFusion
//! workflow to optionally emit signals after generation with configurable backends.

use std::time::Instant;
use tracing::{debug, info, warn, error};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

use crate::{TradingSignal, SignalComponents, FusionWeights};
use crate::emission::{
    SignalPublisher, SignalPublisherConfig,
    generate_correlation_id, FeatureComputationEvent, AuditLogger,
    Result as EmissionResult,
};

/// Configuration for signal emission integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalEmitterConfig {
    /// Whether signal emission is enabled
    pub enabled: bool,
    
    /// Signal publisher configuration
    pub publisher: SignalPublisherConfig,
    
    /// Whether to fail signal generation if emission fails
    pub fail_on_emission_error: bool,
    
    /// Model version to include in emitted signals
    pub model_version: String,
    
    /// Whether to generate feature computation audit events
    pub audit_feature_computation: bool,
}

impl Default for SignalEmitterConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for backward compatibility
            publisher: SignalPublisherConfig::default(),
            fail_on_emission_error: false,
            model_version: "v1.0".to_string(),
            audit_feature_computation: true,
        }
    }
}

impl SignalEmitterConfig {
    /// Create a new configuration with emission enabled
    pub fn enabled(publisher_config: SignalPublisherConfig) -> Self {
        Self {
            enabled: true,
            publisher: publisher_config,
            ..Default::default()
        }
    }
    
    /// Set the model version
    pub fn with_model_version(mut self, version: String) -> Self {
        self.model_version = version;
        self
    }
    
    /// Enable/disable failure on emission errors
    pub fn with_fail_on_error(mut self, fail_on_error: bool) -> Self {
        self.fail_on_emission_error = fail_on_error;
        self
    }
    
    /// Enable/disable feature computation auditing
    pub fn with_feature_auditing(mut self, audit: bool) -> Self {
        self.audit_feature_computation = audit;
        self
    }
}

/// Signal emitter that integrates with SignalFusion workflow
pub struct SignalEmitter {
    config: SignalEmitterConfig,
    publisher: Option<SignalPublisher>,
    audit_logger: Option<AuditLogger>,
}

impl SignalEmitter {
    /// Create a new SignalEmitter with the given configuration
    pub async fn new(config: SignalEmitterConfig) -> EmissionResult<Self> {
        if !config.enabled {
            info!("SignalEmitter disabled, no publishers will be initialized");
            return Ok(Self {
                config,
                publisher: None,
                audit_logger: None,
            });
        }
        
        info!("Initializing SignalEmitter with backend: {}", config.publisher.backend);
        
        // Initialize publisher if enabled
        let publisher = if config.publisher.enabled {
            Some(SignalPublisher::new(config.publisher.clone()).await?)
        } else {
            None
        };
        
        // Initialize audit logger if feature auditing is enabled
        let audit_logger = if config.audit_feature_computation {
            Some(AuditLogger::new(config.publisher.audit.clone()).await?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            publisher,
            audit_logger,
        })
    }
    
    /// Create a disabled SignalEmitter for backward compatibility
    pub fn disabled() -> Self {
        Self {
            config: SignalEmitterConfig::default(),
            publisher: None,
            audit_logger: None,
        }
    }
    
    /// Generate a correlation ID for tracking signal lifecycle
    pub fn generate_correlation_id(&self) -> String {
        generate_correlation_id()
    }
    
    /// Generate a feature checksum from signal components and weights
    pub fn generate_feature_checksum(
        &self,
        components: &SignalComponents,
        weights: &FusionWeights,
        symbol: &str,
        timestamp: i64,
    ) -> String {
        let mut hasher = Sha256::new();
        
        // Include all relevant data in the checksum
        hasher.update(symbol.as_bytes());
        hasher.update(timestamp.to_le_bytes());
        hasher.update(components.s_ldc.to_le_bytes());
        hasher.update(components.s_mr.to_le_bytes());
        hasher.update(components.s_tsmom.to_le_bytes());
        hasher.update(weights.w_ldc.to_le_bytes());
        hasher.update(weights.w_mr.to_le_bytes());
        hasher.update(weights.w_tsmom.to_le_bytes());
        
        let result = hasher.finalize();
        format!("{:x}", result)[..16].to_string() // Use first 16 chars for brevity
    }
    
    /// Log feature computation event if auditing is enabled
    pub async fn log_feature_computation(
        &self,
        correlation_id: &str,
        symbol: &str,
        feature_names: Vec<String>,
        computation_latency_ms: u64,
        input_checksum: String,
        output_checksum: String,
        validation_passed: bool,
    ) -> EmissionResult<()> {
        if let Some(ref audit_logger) = self.audit_logger {
            let event = FeatureComputationEvent::new(
                correlation_id.to_string(),
                symbol.to_string(),
                feature_names,
                computation_latency_ms,
                input_checksum,
                output_checksum,
                validation_passed,
            );
            
            audit_logger.log_feature_computation(&event).await?;
            debug!("Feature computation event logged for correlation_id: {}", correlation_id);
        }
        
        Ok(())
    }
    
    /// Emit a trading signal if emission is enabled
    pub async fn emit_signal(&self, signal: TradingSignal) -> EmissionResult<bool> {
        if !self.config.enabled {
            debug!("Signal emission disabled, skipping signal: {}", signal.to_compact_string());
            return Ok(false);
        }
        
        if let Some(ref publisher) = self.publisher {
            let start_time = Instant::now();
            
            debug!("Emitting signal: {}", signal.to_compact_string());
            
            match publisher.publish_signal(signal.clone()).await {
                Ok(results) => {
                    let latency_ms = start_time.elapsed().as_millis() as u64;
                    let backend_names = results.iter()
                        .map(|r| r.backend.clone())
                        .collect::<Vec<_>>()
                        .join(",");
                    
                    info!(
                        "Signal emitted successfully to [{}] in {}ms: {}",
                        backend_names,
                        latency_ms,
                        signal.to_compact_string()
                    );
                    
                    Ok(true)
                }
                Err(error) => {
                    let latency_ms = start_time.elapsed().as_millis() as u64;
                    
                    error!(
                        "Failed to emit signal after {}ms: {} - {}",
                        latency_ms,
                        error,
                        signal.to_compact_string()
                    );
                    
                    if self.config.fail_on_emission_error {
                        return Err(error);
                    } else {
                        warn!("Signal emission failed but continuing due to configuration: {}", error);
                        return Ok(false);
                    }
                }
            }
        } else {
            debug!("No publisher configured, signal emission skipped: {}", signal.to_compact_string());
            Ok(false)
        }
    }
    
    /// Check if emission is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
    
    /// Get the model version
    pub fn model_version(&self) -> &str {
        &self.config.model_version
    }
    
    /// Check if feature auditing is enabled
    pub fn is_feature_auditing_enabled(&self) -> bool {
        self.config.audit_feature_computation
    }
    
    /// Get health status of the emission system
    pub async fn health_check(&self) -> EmissionResult<bool> {
        if !self.config.enabled {
            return Ok(true); // Always healthy when disabled
        }
        
        if let Some(ref publisher) = self.publisher {
            let health = publisher.health_check().await;
            match health.status {
                crate::emission::publisher::HealthLevel::Healthy => Ok(true),
                crate::emission::publisher::HealthLevel::Degraded => {
                    warn!("Signal emission system degraded: {:?}", health.error_message);
                    Ok(true) // Still functional
                }
                crate::emission::publisher::HealthLevel::Unhealthy => {
                    error!("Signal emission system unhealthy: {:?}", health.error_message);
                    Ok(false)
                }
            }
        } else {
            Ok(true) // No publisher means no health issues
        }
    }
    
    /// Shutdown the signal emitter
    pub async fn shutdown(&self) -> EmissionResult<()> {
        if let Some(ref publisher) = self.publisher {
            info!("Shutting down SignalEmitter");
            publisher.shutdown().await?;
            info!("SignalEmitter shutdown complete");
        }
        Ok(())
    }
    
    /// Get the current configuration
    pub fn get_config(&self) -> &SignalEmitterConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SignalSide, emission::PublisherBackend};
    
    #[test]
    fn test_signal_emitter_config_creation() {
        let config = SignalEmitterConfig::default();
        assert!(!config.enabled);
        assert!(!config.fail_on_emission_error);
        assert_eq!(config.model_version, "v1.0");
        assert!(config.audit_feature_computation);
    }
    
    #[test]
    fn test_signal_emitter_config_builder() {
        let publisher_config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            ..Default::default()
        };
        
        let config = SignalEmitterConfig::enabled(publisher_config)
            .with_model_version("v2.0".to_string())
            .with_fail_on_error(true)
            .with_feature_auditing(false);
        
        assert!(config.enabled);
        assert!(config.fail_on_emission_error);
        assert_eq!(config.model_version, "v2.0");
        assert!(!config.audit_feature_computation);
    }
    
    #[tokio::test]
    async fn test_signal_emitter_disabled() {
        let emitter = SignalEmitter::disabled();
        
        assert!(!emitter.is_enabled());
        assert_eq!(emitter.model_version(), "v1.0");
        assert!(emitter.is_feature_auditing_enabled());
        
        // Should be healthy when disabled
        let health = emitter.health_check().await.unwrap();
        assert!(health);
        
        // Should shutdown cleanly
        let result = emitter.shutdown().await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_signal_emitter_with_none_backend() {
        let publisher_config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            ..Default::default()
        };
        
        let config = SignalEmitterConfig::enabled(publisher_config);
        let emitter = SignalEmitter::new(config).await.unwrap();
        
        assert!(emitter.is_enabled());
        
        // Should be healthy with None backend
        let health = emitter.health_check().await.unwrap();
        assert!(health);
    }
    
    #[test]
    fn test_correlation_id_generation() {
        let emitter = SignalEmitter::disabled();
        
        let id1 = emitter.generate_correlation_id();
        let id2 = emitter.generate_correlation_id();
        
        assert_ne!(id1, id2);
        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
    }
    
    #[test]
    fn test_feature_checksum_generation() {
        let emitter = SignalEmitter::disabled();
        
        let components = SignalComponents {
            s_ldc: 0.5,
            s_mr: 0.3,
            s_tsmom: 0.2,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let checksum1 = emitter.generate_feature_checksum(&components, &weights, "BTCUSDT", 1000);
        let checksum2 = emitter.generate_feature_checksum(&components, &weights, "BTCUSDT", 1000);
        let checksum3 = emitter.generate_feature_checksum(&components, &weights, "ETHUSDT", 1000);
        
        // Same inputs should produce same checksum
        assert_eq!(checksum1, checksum2);
        
        // Different symbol should produce different checksum
        assert_ne!(checksum1, checksum3);
        
        // Checksum should be 16 characters (hex)
        assert_eq!(checksum1.len(), 16);
        assert!(checksum1.chars().all(|c| c.is_ascii_hexdigit()));
    }
    
    #[tokio::test]
    async fn test_signal_emission_disabled() {
        let emitter = SignalEmitter::disabled();
        
        let signal = create_test_signal();
        let result = emitter.emit_signal(signal).await.unwrap();
        
        // Should return false when disabled
        assert!(!result);
    }
    
    fn create_test_signal() -> TradingSignal {
        let components = SignalComponents {
            s_ldc: 0.5,
            s_mr: 0.3,
            s_tsmom: 0.2,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        TradingSignal::new(
            chrono::Utc::now().timestamp(),
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components,
            weights,
            "v1.0".to_string(),
            "test-correlation".to_string(),
            "test-checksum".to_string(),
            50,
        )
    }
}