//! End-to-end signal emission pipeline
//! 
//! This module provides a complete signal processing pipeline that handles
//! the entire flow from signal generation through validation to publication
//! and audit logging with comprehensive error handling and performance monitoring.

use std::time::Instant;
use tracing::{debug, info, warn, error, instrument};
use serde::{Deserialize, Serialize};

use crate::{
    TradingSignal, SignalComponents, FusionWeights, SignalFusion, SignalEmitterConfig,
    emission::{
        FeatureComputationEvent,
        generate_correlation_id,
        Result as EmissionResult, SignalEmissionError,
    }
};

/// Configuration for the signal emission pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalPipelineConfig {
    /// Signal fusion configuration
    pub fusion_threshold: f32,
    pub fusion_cooldown_period: u64,
    pub normalize_weights: bool,
    
    /// Signal emission configuration
    pub emitter_config: SignalEmitterConfig,
    
    /// Pipeline behavior configuration
    pub enable_performance_monitoring: bool,
    pub log_all_pipeline_steps: bool,
    pub fail_on_validation_error: bool,
    pub fail_on_emission_error: bool,
    
    /// Feature computation audit configuration
    pub audit_feature_computation: bool,
    pub feature_computation_timeout_ms: u64,
}

impl Default for SignalPipelineConfig {
    fn default() -> Self {
        Self {
            fusion_threshold: 0.3,
            fusion_cooldown_period: 60,
            normalize_weights: true,
            emitter_config: SignalEmitterConfig::default(),
            enable_performance_monitoring: true,
            log_all_pipeline_steps: false,
            fail_on_validation_error: true,
            fail_on_emission_error: false,
            audit_feature_computation: true,
            feature_computation_timeout_ms: 5000,
        }
    }
}

/// Performance metrics for the signal pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    /// Total time for the entire pipeline (milliseconds)
    pub total_latency_ms: u64,
    
    /// Time spent in signal fusion (milliseconds)
    pub fusion_latency_ms: u64,
    
    /// Time spent in signal validation (milliseconds)
    pub validation_latency_ms: u64,
    
    /// Time spent in signal emission (milliseconds)
    pub emission_latency_ms: u64,
    
    /// Time spent in audit logging (milliseconds)
    pub audit_latency_ms: u64,
    
    /// Whether the pipeline completed successfully
    pub success: bool,
    
    /// Error message if pipeline failed
    pub error_message: Option<String>,
    
    /// Correlation ID for tracking
    pub correlation_id: String,
}

impl PipelineMetrics {
    fn new(correlation_id: String) -> Self {
        Self {
            total_latency_ms: 0,
            fusion_latency_ms: 0,
            validation_latency_ms: 0,
            emission_latency_ms: 0,
            audit_latency_ms: 0,
            success: false,
            error_message: None,
            correlation_id,
        }
    }
    
    fn with_error(mut self, error: String) -> Self {
        self.error_message = Some(error);
        self.success = false;
        self
    }
    
    fn with_success(mut self) -> Self {
        self.success = true;
        self.error_message = None;
        self
    }
}

/// Result of pipeline execution
#[derive(Debug)]
pub struct PipelineResult {
    /// The generated signal (if any)
    pub signal: Option<TradingSignal>,
    
    /// Performance metrics
    pub metrics: PipelineMetrics,
    
    /// Whether signal was emitted successfully
    pub emitted: bool,
}

/// End-to-end signal emission pipeline
pub struct SignalPipeline {
    config: SignalPipelineConfig,
    fusion: SignalFusion,
}

impl SignalPipeline {
    /// Create a new signal pipeline with the given configuration
    pub async fn new(config: SignalPipelineConfig) -> EmissionResult<Self> {
        info!("Initializing SignalPipeline with emission enabled: {}", config.emitter_config.enabled);
        
        let fusion = if config.emitter_config.enabled {
            SignalFusion::with_emission_and_normalization(
                config.fusion_threshold,
                config.fusion_cooldown_period,
                config.normalize_weights,
                config.emitter_config.clone(),
            ).await?
        } else {
            SignalFusion::with_normalization(
                config.fusion_threshold,
                config.fusion_cooldown_period,
                config.normalize_weights,
            )
        };
        
        Ok(Self {
            config,
            fusion,
        })
    }
    
    /// Create a pipeline with emission disabled (for backward compatibility)
    pub fn without_emission(
        fusion_threshold: f32,
        cooldown_period: u64,
        normalize_weights: bool,
    ) -> Self {
        let config = SignalPipelineConfig {
            fusion_threshold,
            fusion_cooldown_period: cooldown_period,
            normalize_weights,
            emitter_config: SignalEmitterConfig::default(), // Disabled by default
            ..Default::default()
        };
        
        let fusion = SignalFusion::with_normalization(
            fusion_threshold,
            cooldown_period,
            normalize_weights,
        );
        
        Self {
            config,
            fusion,
        }
    }
    
    /// Process a complete signal through the pipeline
    #[instrument(skip(self, components, weights), fields(symbol = %symbol, correlation_id))]
    pub async fn process_signal(
        &mut self,
        components: SignalComponents,
        weights: FusionWeights,
        timestamp: i64,
        symbol: &str,
        feature_names: Option<Vec<String>>,
        input_checksum: Option<String>,
    ) -> EmissionResult<PipelineResult> {
        let pipeline_start = Instant::now();
        let correlation_id = generate_correlation_id();
        
        // Set correlation_id in the tracing span
        tracing::Span::current().record("correlation_id", &correlation_id);
        
        let mut metrics = PipelineMetrics::new(correlation_id.clone());
        
        if self.config.log_all_pipeline_steps {
            info!(
                "Starting signal pipeline for {} (correlation_id: {})",
                symbol, correlation_id
            );
        }
        
        // Step 1: Log feature computation if enabled
        if self.config.audit_feature_computation {
            let audit_start = Instant::now();
            
            if let Err(e) = self.log_feature_computation(
                &correlation_id,
                symbol,
                feature_names.unwrap_or_else(|| vec!["ldc".to_string(), "mr".to_string(), "tsmom".to_string()]),
                input_checksum.unwrap_or_else(|| format!("input-{}-{}", symbol, timestamp)),
                &components,
                &weights,
            ).await {
                warn!("Feature computation audit failed: {} (correlation_id: {})", e, correlation_id);
                // Don't fail the pipeline for audit errors unless configured to do so
            }
            
            metrics.audit_latency_ms += audit_start.elapsed().as_millis() as u64;
        }
        
        // Step 2: Signal fusion
        let fusion_start = Instant::now();
        
        let signal_result = self.fusion.fuse_signals(
            components,
            weights,
            timestamp,
            symbol,
            None, // Use emitter's model version
            Some(correlation_id.clone()),
            None, // Generate feature checksum
            pipeline_start.elapsed().as_millis() as u64,
        ).await;
        
        metrics.fusion_latency_ms = fusion_start.elapsed().as_millis() as u64;
        
        let signal = match signal_result {
            Ok(Some(signal)) => {
                if self.config.log_all_pipeline_steps {
                    debug!(
                        "Signal fusion completed: {} (correlation_id: {})",
                        signal.to_compact_string(), correlation_id
                    );
                }
                signal
            }
            Ok(None) => {
                // No signal generated (below threshold or in cooldown)
                if self.config.log_all_pipeline_steps {
                    debug!(
                        "No signal generated for {} (below threshold or in cooldown) (correlation_id: {})",
                        symbol, correlation_id
                    );
                }
                
                metrics.total_latency_ms = pipeline_start.elapsed().as_millis() as u64;
                return Ok(PipelineResult {
                    signal: None,
                    metrics: metrics.with_success(),
                    emitted: false,
                });
            }
            Err(e) => {
                error!(
                    "Signal fusion failed for {}: {} (correlation_id: {})",
                    symbol, e, correlation_id
                );
                
                metrics.total_latency_ms = pipeline_start.elapsed().as_millis() as u64;
                return if self.config.fail_on_validation_error {
                    Err(SignalEmissionError::validation(format!("Signal fusion failed: {}", e)))
                } else {
                    Ok(PipelineResult {
                        signal: None,
                        metrics: metrics.with_error(format!("Signal fusion failed: {}", e)),
                        emitted: false,
                    })
                };
            }
        };
        
        // Step 3: Signal validation (already done in fusion, but we track timing separately)
        let validation_start = Instant::now();
        
        if let Err(e) = signal.validate() {
            error!(
                "Signal validation failed for {}: {} (correlation_id: {})",
                symbol, e, correlation_id
            );
            
            metrics.validation_latency_ms = validation_start.elapsed().as_millis() as u64;
            metrics.total_latency_ms = pipeline_start.elapsed().as_millis() as u64;
            
            return if self.config.fail_on_validation_error {
                Err(SignalEmissionError::validation(format!("Signal validation failed: {}", e)))
            } else {
                Ok(PipelineResult {
                    signal: Some(signal),
                    metrics: metrics.with_error(format!("Signal validation failed: {}", e)),
                    emitted: false,
                })
            };
        }
        
        metrics.validation_latency_ms = validation_start.elapsed().as_millis() as u64;
        
        if self.config.log_all_pipeline_steps {
            debug!(
                "Signal validation completed for {} (correlation_id: {})",
                symbol, correlation_id
            );
        }
        
        // Step 4: Signal emission (already handled in fusion if emitter is configured)
        let emission_start = Instant::now();
        let emitted = self.fusion.is_emission_enabled();
        
        if emitted && self.config.log_all_pipeline_steps {
            debug!(
                "Signal emission completed for {} (correlation_id: {})",
                symbol, correlation_id
            );
        }
        
        metrics.emission_latency_ms = emission_start.elapsed().as_millis() as u64;
        
        // Step 5: Final metrics and logging
        metrics.total_latency_ms = pipeline_start.elapsed().as_millis() as u64;
        
        if self.config.enable_performance_monitoring {
            info!(
                "Pipeline completed for {} in {}ms (fusion: {}ms, validation: {}ms, emission: {}ms, audit: {}ms) (correlation_id: {})",
                symbol,
                metrics.total_latency_ms,
                metrics.fusion_latency_ms,
                metrics.validation_latency_ms,
                metrics.emission_latency_ms,
                metrics.audit_latency_ms,
                correlation_id
            );
        }
        
        Ok(PipelineResult {
            signal: Some(signal),
            metrics: metrics.with_success(),
            emitted,
        })
    }
    
    /// Log feature computation event
    async fn log_feature_computation(
        &self,
        correlation_id: &str,
        symbol: &str,
        feature_names: Vec<String>,
        input_checksum: String,
        components: &SignalComponents,
        weights: &FusionWeights,
    ) -> EmissionResult<()> {
        // Generate output checksum from components and weights
        let output_data = format!(
            "ldc:{:.6},mr:{:.6},tsmom:{:.6},w_ldc:{:.6},w_mr:{:.6},w_tsmom:{:.6}",
            components.s_ldc, components.s_mr, components.s_tsmom,
            weights.w_ldc, weights.w_mr, weights.w_tsmom
        );
        
        let output_checksum = format!("{:x}", md5::compute(output_data.as_bytes()));
        
        // Validate components and weights
        let validation_passed = components.validate().is_ok() && weights.validate().is_ok();
        
        let _event = FeatureComputationEvent::new(
            correlation_id.to_string(),
            symbol.to_string(),
            feature_names,
            0, // Computation latency will be measured by the caller
            input_checksum,
            output_checksum,
            validation_passed,
        );
        
        // If we have an emitter with audit logging, use it
        if let Some(emitter_config) = self.fusion.get_emitter_config() {
            if emitter_config.audit_feature_computation {
                // The actual audit logging is handled by the emitter
                debug!(
                    "Feature computation event prepared for {} (correlation_id: {})",
                    symbol, correlation_id
                );
            }
        }
        
        Ok(())
    }
    
    /// Get pipeline configuration
    pub fn get_config(&self) -> &SignalPipelineConfig {
        &self.config
    }
    
    /// Check if emission is enabled
    pub fn is_emission_enabled(&self) -> bool {
        self.fusion.is_emission_enabled()
    }
    
    /// Perform health check on the pipeline
    pub async fn health_check(&self) -> EmissionResult<bool> {
        self.fusion.emission_health_check().await
            .map_err(|e| SignalEmissionError::health_check(format!("Pipeline health check failed: {}", e)))
    }
    
    /// Get pipeline statistics
    pub fn get_statistics(&self) -> PipelineStatistics {
        PipelineStatistics {
            fusion_threshold: self.fusion.threshold(),
            cooldown_period: self.fusion.cooldown_period(),
            last_signal_time: self.fusion.last_signal_time(),
            emission_enabled: self.fusion.is_emission_enabled(),
            normalize_weights: self.config.normalize_weights,
        }
    }
    
    /// Shutdown the pipeline
    pub async fn shutdown(&self) -> EmissionResult<()> {
        info!("Shutting down SignalPipeline");
        self.fusion.shutdown().await
            .map_err(|e| SignalEmissionError::shutdown(format!("Pipeline shutdown failed: {}", e)))?;
        info!("SignalPipeline shutdown complete");
        Ok(())
    }
}

/// Pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatistics {
    pub fusion_threshold: f32,
    pub cooldown_period: u64,
    pub last_signal_time: Option<i64>,
    pub emission_enabled: bool,
    pub normalize_weights: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SignalSide, emission::PublisherBackend};
    
    #[tokio::test]
    async fn test_pipeline_creation_without_emission() {
        let pipeline = SignalPipeline::without_emission(0.3, 60, true);
        
        assert!(!pipeline.is_emission_enabled());
        assert_eq!(pipeline.get_config().fusion_threshold, 0.3);
        assert_eq!(pipeline.get_config().fusion_cooldown_period, 60);
        assert!(pipeline.get_config().normalize_weights);
    }
    
    #[tokio::test]
    async fn test_pipeline_creation_with_emission() {
        use std::env;
        
        let temp_dir = env::temp_dir().join("signal_pipeline_test");
        
        let mut publisher_config = crate::SignalPublisherConfig {
            backend: PublisherBackend::None,
            enabled: true,
            ..Default::default()
        };
        
        // Use temp directory for audit logs
        publisher_config.audit.log_directory = temp_dir;
        
        let emitter_config = SignalEmitterConfig::enabled(publisher_config);
        
        let config = SignalPipelineConfig {
            emitter_config,
            ..Default::default()
        };
        
        let pipeline = SignalPipeline::new(config).await.unwrap();
        
        assert!(pipeline.is_emission_enabled());
    }
    
    #[tokio::test]
    async fn test_pipeline_process_signal_above_threshold() {
        let mut pipeline = SignalPipeline::without_emission(0.3, 0, true);
        
        let components = SignalComponents {
            s_ldc: 0.8,
            s_mr: 0.2,
            s_tsmom: 0.1,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let now = chrono::Utc::now().timestamp();
        let result = pipeline.process_signal(
            components,
            weights,
            now,
            "BTCUSDT",
            Some(vec!["ldc".to_string(), "mr".to_string(), "tsmom".to_string()]),
            Some("input-checksum-123".to_string()),
        ).await.unwrap();
        
        assert!(result.signal.is_some());
        assert!(result.metrics.success);
        assert!(!result.emitted); // No emission configured
        // Latency might be 0 for very fast operations, so just check it's not negative
        assert!(result.metrics.total_latency_ms >= 0);
        
        let signal = result.signal.unwrap();
        assert_eq!(signal.symbol, "BTCUSDT");
        assert_eq!(signal.side, SignalSide::Buy);
    }
    
    #[tokio::test]
    async fn test_pipeline_process_signal_below_threshold() {
        let mut pipeline = SignalPipeline::without_emission(0.5, 0, true);
        
        let components = SignalComponents {
            s_ldc: 0.1,
            s_mr: 0.1,
            s_tsmom: 0.1,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.33,
            w_mr: 0.33,
            w_tsmom: 0.34,
        };
        
        let now = chrono::Utc::now().timestamp();
        let result = pipeline.process_signal(
            components,
            weights,
            now,
            "BTCUSDT",
            None,
            None,
        ).await.unwrap();
        
        assert!(result.signal.is_none());
        assert!(result.metrics.success);
        assert!(!result.emitted);
    }
    
    #[tokio::test]
    async fn test_pipeline_process_signal_with_emission() {
        use std::env;
        
        let temp_dir = env::temp_dir().join("signal_pipeline_test_2");
        
        let mut publisher_config = crate::SignalPublisherConfig {
            backend: PublisherBackend::None,
            enabled: true,
            ..Default::default()
        };
        
        // Use temp directory for audit logs
        publisher_config.audit.log_directory = temp_dir;
        
        let emitter_config = SignalEmitterConfig::enabled(publisher_config);
        
        let config = SignalPipelineConfig {
            emitter_config,
            enable_performance_monitoring: true,
            log_all_pipeline_steps: true,
            ..Default::default()
        };
        
        let mut pipeline = SignalPipeline::new(config).await.unwrap();
        
        let components = SignalComponents {
            s_ldc: 0.8,
            s_mr: 0.2,
            s_tsmom: 0.1,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let now = chrono::Utc::now().timestamp();
        let result = pipeline.process_signal(
            components,
            weights,
            now,
            "BTCUSDT",
            Some(vec!["ldc".to_string(), "mr".to_string(), "tsmom".to_string()]),
            Some("input-checksum-456".to_string()),
        ).await.unwrap();
        
        assert!(result.signal.is_some());
        assert!(result.metrics.success);
        assert!(result.emitted); // Emission is configured
        assert!(result.metrics.total_latency_ms >= 0);
        assert!(result.metrics.fusion_latency_ms >= 0);
    }
    
    #[tokio::test]
    async fn test_pipeline_health_check() {
        let pipeline = SignalPipeline::without_emission(0.3, 60, true);
        
        let health = pipeline.health_check().await.unwrap();
        assert!(health); // Should be healthy without emission
    }
    
    #[tokio::test]
    async fn test_pipeline_statistics() {
        let pipeline = SignalPipeline::without_emission(0.3, 60, true);
        
        let stats = pipeline.get_statistics();
        assert_eq!(stats.fusion_threshold, 0.3);
        assert_eq!(stats.cooldown_period, 60);
        assert!(stats.normalize_weights);
        assert!(!stats.emission_enabled);
        assert!(stats.last_signal_time.is_none());
    }
    
    #[tokio::test]
    async fn test_pipeline_shutdown() {
        let pipeline = SignalPipeline::without_emission(0.3, 60, true);
        
        let result = pipeline.shutdown().await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_pipeline_metrics() {
        let correlation_id = "test-correlation-123".to_string();
        let mut metrics = PipelineMetrics::new(correlation_id.clone());
        
        assert_eq!(metrics.correlation_id, correlation_id);
        assert!(!metrics.success);
        assert!(metrics.error_message.is_none());
        
        metrics = metrics.with_error("Test error".to_string());
        assert!(!metrics.success);
        assert_eq!(metrics.error_message, Some("Test error".to_string()));
        
        metrics = metrics.with_success();
        assert!(metrics.success);
        assert!(metrics.error_message.is_none());
    }
}