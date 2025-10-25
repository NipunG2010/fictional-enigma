use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn, error};

pub mod config;
pub mod hmm_client;
pub mod weight_cache;
pub mod metrics;
pub mod emission;
pub mod signal_emitter;
pub mod signal_pipeline;

// Re-export commonly used types
pub use config::HmmIntegrationConfig;
pub use weight_cache::{WeightCache, CacheStats};
pub use metrics::{MetricsCollector, HmmIntegrationMetrics, MetricsFormat, export_metrics};
pub use emission::{SignalEmissionError, Result as EmissionResult};
pub use emission::publisher::{PublisherTrait, PublishResult, HealthStatus, HealthLevel, RetryConfig};
pub use emission::validation::{SignalValidator, ValidationConfig, ValidationError};
pub use emission::buffer::{SignalBuffer, BufferConfig, BufferedSignal, BufferMetrics, OverflowStrategy, PersistenceConfig};
pub use emission::signal_publisher::{SignalPublisher, SignalPublisherConfig, PublisherBackend, SignalPublisherMetrics};
pub use emission::config::{SignalEmissionConfig, ConfigMetadata, ConfigSource, ConfigSummary, ConfigWatcher};
pub use emission::health_monitor::{HealthMonitor, HealthMonitorConfig, HealthHttpConfig, ComponentHealth, ServiceHealth, ServiceMetrics, HealthHttpServer};
pub use signal_emitter::{SignalEmitter, SignalEmitterConfig};
pub use signal_pipeline::{SignalPipeline, SignalPipelineConfig, PipelineMetrics, PipelineResult, PipelineStatistics};

// Signal validation constants
const SIGNAL_MIN: f32 = -1.0;
const SIGNAL_MAX: f32 = 1.0;
const WEIGHT_MIN: f32 = -1.0;
const WEIGHT_MAX: f32 = 1.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalComponents {
    pub s_ldc: f32,
    pub s_mr: f32,
    pub s_tsmom: f32,
}

impl SignalComponents {
    /// Validate that all signal components are within expected ranges
    pub fn validate(&self) -> Result<()> {
        if !self.s_ldc.is_finite() || self.s_ldc < SIGNAL_MIN || self.s_ldc > SIGNAL_MAX {
            bail!("s_ldc out of range [{}, {}]: {}", SIGNAL_MIN, SIGNAL_MAX, self.s_ldc);
        }
        if !self.s_mr.is_finite() || self.s_mr < SIGNAL_MIN || self.s_mr > SIGNAL_MAX {
            bail!("s_mr out of range [{}, {}]: {}", SIGNAL_MIN, SIGNAL_MAX, self.s_mr);
        }
        if !self.s_tsmom.is_finite() || self.s_tsmom < SIGNAL_MIN || self.s_tsmom > SIGNAL_MAX {
            bail!("s_tsmom out of range [{}, {}]: {}", SIGNAL_MIN, SIGNAL_MAX, self.s_tsmom);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionWeights {
    pub w_ldc: f32,
    pub w_mr: f32,
    pub w_tsmom: f32,
}

impl FusionWeights {
    /// Validate that all weights are within expected ranges
    pub fn validate(&self) -> Result<()> {
        if !self.w_ldc.is_finite() || self.w_ldc < WEIGHT_MIN || self.w_ldc > WEIGHT_MAX {
            bail!("w_ldc out of range [{}, {}]: {}", WEIGHT_MIN, WEIGHT_MAX, self.w_ldc);
        }
        if !self.w_mr.is_finite() || self.w_mr < WEIGHT_MIN || self.w_mr > WEIGHT_MAX {
            bail!("w_mr out of range [{}, {}]: {}", WEIGHT_MIN, WEIGHT_MAX, self.w_mr);
        }
        if !self.w_tsmom.is_finite() || self.w_tsmom < WEIGHT_MIN || self.w_tsmom > WEIGHT_MAX {
            bail!("w_tsmom out of range [{}, {}]: {}", WEIGHT_MIN, WEIGHT_MAX, self.w_tsmom);
        }
        Ok(())
    }

    /// Normalize weights to sum to 1.0 while preserving signs
    /// This ensures proper signal scaling
    pub fn normalize(&self) -> Self {
        let sum_abs = self.w_ldc.abs() + self.w_mr.abs() + self.w_tsmom.abs();
        
        // If sum is zero or very small, return equal weights
        if sum_abs < 1e-6 {
            warn!("Weight sum near zero, using equal weights");
            return FusionWeights {
                w_ldc: 0.33,
                w_mr: 0.33,
                w_tsmom: 0.34,
            };
        }
        
        FusionWeights {
            w_ldc: self.w_ldc / sum_abs,
            w_mr: self.w_mr / sum_abs,
            w_tsmom: self.w_tsmom / sum_abs,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SignalSide {
    #[serde(rename = "BUY")]
    Buy,
    #[serde(rename = "SELL")]
    Sell,
    #[serde(rename = "HOLD")]
    Hold,
}

impl std::fmt::Display for SignalSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalSide::Buy => write!(f, "BUY"),
            SignalSide::Sell => write!(f, "SELL"),
            SignalSide::Hold => write!(f, "HOLD"),
        }
    }
}

impl std::str::FromStr for SignalSide {
    type Err = anyhow::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "BUY" => Ok(SignalSide::Buy),
            "SELL" => Ok(SignalSide::Sell),
            "HOLD" => Ok(SignalSide::Hold),
            _ => bail!("Invalid signal side: {}", s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    // Core signal data
    pub timestamp: i64,
    pub symbol: String,
    pub side: SignalSide,
    pub strength: f32, // -1.0 to 1.0
    pub confidence: f32, // 0.0 to 1.0
    
    // Signal components and weights
    pub components: SignalComponents,
    pub weights: FusionWeights,
    
    // Model metadata
    pub model_version: String,
    
    // Audit fields for traceability
    pub correlation_id: String,
    pub feature_checksum: String,
    pub generation_latency_ms: u64,
    
    // Optional HMM-specific fields
    pub hmm_state_probabilities: Option<Vec<f32>>,
    pub fallback_used: bool,
}

impl TradingSignal {
    /// Create a new TradingSignal with all required fields
    pub fn new(
        timestamp: i64,
        symbol: String,
        side: SignalSide,
        strength: f32,
        confidence: f32,
        components: SignalComponents,
        weights: FusionWeights,
        model_version: String,
        correlation_id: String,
        feature_checksum: String,
        generation_latency_ms: u64,
    ) -> Self {
        Self {
            timestamp,
            symbol,
            side,
            strength,
            confidence,
            components,
            weights,
            model_version,
            correlation_id,
            feature_checksum,
            generation_latency_ms,
            hmm_state_probabilities: None,
            fallback_used: false,
        }
    }
    
    /// Create a TradingSignal with HMM-specific fields
    pub fn with_hmm_data(
        mut self,
        state_probabilities: Vec<f32>,
        fallback_used: bool,
    ) -> Self {
        self.hmm_state_probabilities = Some(state_probabilities);
        self.fallback_used = fallback_used;
        self
    }
    
    /// Validate the trading signal structure and content
    pub fn validate(&self) -> Result<()> {
        // Validate timestamp (not too far in past or future)
        let now = chrono::Utc::now().timestamp();
        let max_age = 3600; // 1 hour
        let max_future = 300; // 5 minutes
        
        if self.timestamp < now - max_age {
            bail!("Signal timestamp too old: {} (current: {})", self.timestamp, now);
        }
        
        if self.timestamp > now + max_future {
            bail!("Signal timestamp too far in future: {} (current: {})", self.timestamp, now);
        }
        
        // Validate symbol format (uppercase, alphanumeric)
        if self.symbol.is_empty() {
            bail!("Symbol cannot be empty");
        }
        
        if !self.symbol.chars().all(|c| c.is_ascii_alphanumeric()) {
            bail!("Symbol must contain only alphanumeric characters: {}", self.symbol);
        }
        
        if self.symbol != self.symbol.to_uppercase() {
            bail!("Symbol must be uppercase: {}", self.symbol);
        }
        
        // Validate strength range
        if !self.strength.is_finite() || self.strength < SIGNAL_MIN || self.strength > SIGNAL_MAX {
            bail!("Strength out of range [{}, {}]: {}", SIGNAL_MIN, SIGNAL_MAX, self.strength);
        }
        
        // Validate confidence range
        if !self.confidence.is_finite() || self.confidence < 0.0 || self.confidence > 1.0 {
            bail!("Confidence out of range [0.0, 1.0]: {}", self.confidence);
        }
        
        // Validate components
        self.components.validate()?;
        
        // Validate weights
        self.weights.validate()?;
        
        // Validate model version format
        if self.model_version.is_empty() {
            bail!("Model version cannot be empty");
        }
        
        // Validate correlation ID format (should be non-empty)
        if self.correlation_id.is_empty() {
            bail!("Correlation ID cannot be empty");
        }
        
        // Validate feature checksum format (should be non-empty)
        if self.feature_checksum.is_empty() {
            bail!("Feature checksum cannot be empty");
        }
        
        // Validate HMM state probabilities if present
        if let Some(ref probs) = self.hmm_state_probabilities {
            if probs.is_empty() {
                bail!("HMM state probabilities cannot be empty if provided");
            }
            
            for (i, &prob) in probs.iter().enumerate() {
                if !prob.is_finite() || prob < 0.0 || prob > 1.0 {
                    bail!("HMM state probability {} out of range [0.0, 1.0]: {}", i, prob);
                }
            }
            
            // Check that probabilities sum to approximately 1.0
            let sum: f32 = probs.iter().sum();
            if (sum - 1.0).abs() > 0.01 {
                bail!("HMM state probabilities should sum to 1.0, got: {}", sum);
            }
        }
        
        Ok(())
    }
    
    /// Validate signal consistency (side matches strength sign)
    pub fn validate_consistency(&self) -> Result<()> {
        match self.side {
            SignalSide::Buy => {
                if self.strength <= 0.0 {
                    bail!("BUY signal should have positive strength, got: {}", self.strength);
                }
            }
            SignalSide::Sell => {
                if self.strength >= 0.0 {
                    bail!("SELL signal should have negative strength, got: {}", self.strength);
                }
            }
            SignalSide::Hold => {
                // HOLD signals can have any strength, but typically should be near zero
                if self.strength.abs() > 0.5 {
                    warn!("HOLD signal has high strength: {}", self.strength);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get the signal as a JSON string for serialization
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| anyhow::anyhow!("Failed to serialize signal: {}", e))
    }
    
    /// Create a signal from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| anyhow::anyhow!("Failed to deserialize signal: {}", e))
    }
    
    /// Get a compact string representation for logging
    pub fn to_compact_string(&self) -> String {
        format!(
            "{}@{}: {} {} str={:.3} conf={:.3} [{}]",
            self.symbol,
            self.timestamp,
            self.side,
            self.model_version,
            self.strength,
            self.confidence,
            &self.correlation_id[..8.min(self.correlation_id.len())]
        )
    }
}

pub struct SignalFusion {
    threshold: f32,
    cooldown_period: u64, // seconds
    last_signal_time: Option<i64>,
    normalize_weights: bool,
    signal_emitter: Option<SignalEmitter>,
}

impl SignalFusion {
    pub fn new(threshold: f32, cooldown_period: u64) -> Self {
        Self {
            threshold,
            cooldown_period,
            last_signal_time: None,
            normalize_weights: true,
            signal_emitter: None,
        }
    }

    pub fn with_normalization(threshold: f32, cooldown_period: u64, normalize_weights: bool) -> Self {
        Self {
            threshold,
            cooldown_period,
            last_signal_time: None,
            normalize_weights,
            signal_emitter: None,
        }
    }
    
    /// Create a new SignalFusion with signal emission enabled
    pub async fn with_emission(
        threshold: f32,
        cooldown_period: u64,
        emitter_config: SignalEmitterConfig,
    ) -> Result<Self> {
        let emitter = SignalEmitter::new(emitter_config).await
            .map_err(|e| anyhow::anyhow!("Failed to initialize signal emitter: {}", e))?;
        
        Ok(Self {
            threshold,
            cooldown_period,
            last_signal_time: None,
            normalize_weights: true,
            signal_emitter: Some(emitter),
        })
    }
    
    /// Create a new SignalFusion with signal emission and custom normalization
    pub async fn with_emission_and_normalization(
        threshold: f32,
        cooldown_period: u64,
        normalize_weights: bool,
        emitter_config: SignalEmitterConfig,
    ) -> Result<Self> {
        let emitter = SignalEmitter::new(emitter_config).await
            .map_err(|e| anyhow::anyhow!("Failed to initialize signal emitter: {}", e))?;
        
        Ok(Self {
            threshold,
            cooldown_period,
            last_signal_time: None,
            normalize_weights,
            signal_emitter: Some(emitter),
        })
    }
    
    /// Set the signal emitter (for existing instances)
    pub fn set_signal_emitter(&mut self, emitter: SignalEmitter) {
        self.signal_emitter = Some(emitter);
    }
    
    /// Remove the signal emitter (disable emission)
    pub fn remove_signal_emitter(&mut self) {
        self.signal_emitter = None;
    }
    
    pub async fn fuse_signals(
        &mut self,
        components: SignalComponents,
        weights: FusionWeights,
        timestamp: i64,
        symbol: &str,
        model_version: Option<&str>,
        correlation_id: Option<String>,
        feature_checksum: Option<String>,
        generation_latency_ms: u64,
    ) -> Result<Option<TradingSignal>> {
        // Generate correlation ID and feature checksum if not provided
        let correlation_id = correlation_id.unwrap_or_else(|| {
            if let Some(ref emitter) = self.signal_emitter {
                emitter.generate_correlation_id()
            } else {
                format!("fusion-{}-{}", symbol, timestamp)
            }
        });
        
        let feature_checksum = feature_checksum.unwrap_or_else(|| {
            if let Some(ref emitter) = self.signal_emitter {
                emitter.generate_feature_checksum(&components, &weights, symbol, timestamp)
            } else {
                format!("checksum-{}-{}", symbol, timestamp)
            }
        });
        
        let model_version = model_version.unwrap_or_else(|| {
            if let Some(ref emitter) = self.signal_emitter {
                emitter.model_version()
            } else {
                "v1.0"
            }
        });

        // Validate input signals (Requirement 5.3)
        if let Err(e) = components.validate() {
            warn!("Invalid signal components for {}: {}", symbol, e);
            bail!("Signal validation failed: {}", e);
        }

        // Validate weights
        if let Err(e) = weights.validate() {
            warn!("Invalid fusion weights for {}: {}", symbol, e);
            bail!("Weight validation failed: {}", e);
        }

        debug!(
            "Fusing signals for {} at timestamp {} (correlation_id: {}): components=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}], weights=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}]",
            symbol, timestamp, correlation_id, components.s_ldc, components.s_mr, components.s_tsmom,
            weights.w_ldc, weights.w_mr, weights.w_tsmom
        );

        // Normalize weights if enabled (Requirement 5.2)
        let final_weights = if self.normalize_weights {
            let normalized = weights.normalize();
            debug!(
                "Normalized weights for {}: [LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}]",
                symbol, normalized.w_ldc, normalized.w_mr, normalized.w_tsmom
            );
            normalized
        } else {
            weights
        };

        // Check cooldown (Requirement 5.4)
        if let Some(last_time) = self.last_signal_time {
            let time_since_last = timestamp - last_time;
            if time_since_last < self.cooldown_period as i64 {
                debug!(
                    "Signal for {} suppressed by cooldown: {}s since last signal (cooldown: {}s)",
                    symbol, time_since_last, self.cooldown_period
                );
                return Ok(None);
            }
        }
        
        // Compute fused signal (Requirement 5.1)
        let fused_signal = 
            components.s_ldc * final_weights.w_ldc +
            components.s_mr * final_weights.w_mr +
            components.s_tsmom * final_weights.w_tsmom;
        
        debug!(
            "Computed fused signal for {}: {:.4}",
            symbol, fused_signal
        );

        // Apply threshold (Requirement 5.4)
        if fused_signal.abs() < self.threshold {
            debug!(
                "Fused signal {:.4} below threshold {:.4} for {}",
                fused_signal.abs(), self.threshold, symbol
            );
            return Ok(None);
        }
        
        // Determine side
        let side = if fused_signal > 0.0 { SignalSide::Buy } else { SignalSide::Sell };
        
        // Calculate confidence (normalized to [0, 1])
        let confidence = fused_signal.abs().min(1.0);
        
        // Log fusion operation (Requirement 5.5)
        info!(
            "Generated {} signal for {}: strength={:.4}, confidence={:.4}, components=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}], weights=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}], correlation_id={}",
            side, symbol, fused_signal, confidence,
            components.s_ldc, components.s_mr, components.s_tsmom,
            final_weights.w_ldc, final_weights.w_mr, final_weights.w_tsmom,
            correlation_id
        );

        let signal = TradingSignal::new(
            timestamp,
            symbol.to_string(),
            side,
            fused_signal,
            confidence,
            components,
            final_weights,
            model_version.to_string(),
            correlation_id.clone(),
            feature_checksum,
            generation_latency_ms,
        );
        
        // Emit signal if emitter is configured
        if let Some(ref emitter) = self.signal_emitter {
            match emitter.emit_signal(signal.clone()).await {
                Ok(emitted) => {
                    if emitted {
                        debug!("Signal emitted successfully: {}", signal.to_compact_string());
                    } else {
                        debug!("Signal emission skipped: {}", signal.to_compact_string());
                    }
                }
                Err(emission_error) => {
                    error!("Signal emission failed: {} - {}", emission_error, signal.to_compact_string());
                    
                    // Check if we should fail the entire operation
                    let emitter_config = emitter.get_config();
                    if emitter_config.fail_on_emission_error {
                        bail!("Signal emission failed: {}", emission_error);
                    }
                    
                    warn!("Continuing despite emission failure due to configuration");
                }
            }
        }
        
        self.last_signal_time = Some(timestamp);
        
        Ok(Some(signal))
    }

    /// Get the current threshold value
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get the cooldown period in seconds
    pub fn cooldown_period(&self) -> u64 {
        self.cooldown_period
    }

    /// Get the timestamp of the last signal
    pub fn last_signal_time(&self) -> Option<i64> {
        self.last_signal_time
    }

    /// Check if a signal would be suppressed by cooldown
    pub fn is_in_cooldown(&self, current_timestamp: i64) -> bool {
        if let Some(last_time) = self.last_signal_time {
            current_timestamp - last_time < self.cooldown_period as i64
        } else {
            false
        }
    }
    
    /// Check if signal emission is enabled
    pub fn is_emission_enabled(&self) -> bool {
        self.signal_emitter.as_ref().map_or(false, |e| e.is_enabled())
    }
    
    /// Get the signal emitter configuration
    pub fn get_emitter_config(&self) -> Option<&SignalEmitterConfig> {
        self.signal_emitter.as_ref().map(|e| e.get_config())
    }
    
    /// Perform health check on the signal emission system
    pub async fn emission_health_check(&self) -> Result<bool> {
        if let Some(ref emitter) = self.signal_emitter {
            emitter.health_check().await
                .map_err(|e| anyhow::anyhow!("Emission health check failed: {}", e))
        } else {
            Ok(true) // No emitter means no health issues
        }
    }
    
    /// Shutdown the signal fusion system including emission
    pub async fn shutdown(&self) -> Result<()> {
        if let Some(ref emitter) = self.signal_emitter {
            emitter.shutdown().await
                .map_err(|e| anyhow::anyhow!("Failed to shutdown signal emitter: {}", e))?;
        }
        Ok(())
    }
    
    /// Backward-compatible synchronous version of fuse_signals (without emission)
    /// 
    /// This method maintains the original API for existing code that doesn't need signal emission.
    /// For new code that wants signal emission, use the async `fuse_signals` method.
    pub fn fuse_signals_sync(
        &mut self,
        components: SignalComponents,
        weights: FusionWeights,
        timestamp: i64,
        symbol: &str,
        model_version: &str,
        correlation_id: String,
        feature_checksum: String,
        generation_latency_ms: u64,
    ) -> Result<Option<TradingSignal>> {
        // This is the original implementation without emission
        // Validate input signals (Requirement 5.3)
        if let Err(e) = components.validate() {
            warn!("Invalid signal components for {}: {}", symbol, e);
            bail!("Signal validation failed: {}", e);
        }

        // Validate weights
        if let Err(e) = weights.validate() {
            warn!("Invalid fusion weights for {}: {}", symbol, e);
            bail!("Weight validation failed: {}", e);
        }

        debug!(
            "Fusing signals for {} at timestamp {}: components=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}], weights=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}]",
            symbol, timestamp, components.s_ldc, components.s_mr, components.s_tsmom,
            weights.w_ldc, weights.w_mr, weights.w_tsmom
        );

        // Normalize weights if enabled (Requirement 5.2)
        let final_weights = if self.normalize_weights {
            let normalized = weights.normalize();
            debug!(
                "Normalized weights for {}: [LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}]",
                symbol, normalized.w_ldc, normalized.w_mr, normalized.w_tsmom
            );
            normalized
        } else {
            weights
        };

        // Check cooldown (Requirement 5.4)
        if let Some(last_time) = self.last_signal_time {
            let time_since_last = timestamp - last_time;
            if time_since_last < self.cooldown_period as i64 {
                debug!(
                    "Signal for {} suppressed by cooldown: {}s since last signal (cooldown: {}s)",
                    symbol, time_since_last, self.cooldown_period
                );
                return Ok(None);
            }
        }
        
        // Compute fused signal (Requirement 5.1)
        let fused_signal = 
            components.s_ldc * final_weights.w_ldc +
            components.s_mr * final_weights.w_mr +
            components.s_tsmom * final_weights.w_tsmom;
        
        debug!(
            "Computed fused signal for {}: {:.4}",
            symbol, fused_signal
        );

        // Apply threshold (Requirement 5.4)
        if fused_signal.abs() < self.threshold {
            debug!(
                "Fused signal {:.4} below threshold {:.4} for {}",
                fused_signal.abs(), self.threshold, symbol
            );
            return Ok(None);
        }
        
        // Determine side
        let side = if fused_signal > 0.0 { SignalSide::Buy } else { SignalSide::Sell };
        
        // Calculate confidence (normalized to [0, 1])
        let confidence = fused_signal.abs().min(1.0);
        
        // Log fusion operation (Requirement 5.5)
        info!(
            "Generated {} signal for {}: strength={:.4}, confidence={:.4}, components=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}], weights=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}], correlation_id={}",
            side, symbol, fused_signal, confidence,
            components.s_ldc, components.s_mr, components.s_tsmom,
            final_weights.w_ldc, final_weights.w_mr, final_weights.w_tsmom,
            correlation_id
        );

        let signal = TradingSignal::new(
            timestamp,
            symbol.to_string(),
            side,
            fused_signal,
            confidence,
            components,
            final_weights,
            model_version.to_string(),
            correlation_id,
            feature_checksum,
            generation_latency_ms,
        );
        
        self.last_signal_time = Some(timestamp);
        
        Ok(Some(signal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_fusion_creation() {
        let fusion = SignalFusion::new(0.5, 60);
        assert_eq!(fusion.threshold(), 0.5);
        assert_eq!(fusion.cooldown_period(), 60);
    }
    
    #[tokio::test]
    async fn test_fuse_signals_above_threshold() {
        let mut fusion = SignalFusion::new(0.3, 0);
        
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
        
        let result = fusion.fuse_signals(
            components,
            weights,
            1000,
            "BTCUSDT",
            Some("v1.0"),
            Some("test-correlation-123".to_string()),
            Some("feature-checksum-abc".to_string()),
            50,
        ).await.unwrap();
        
        assert!(result.is_some());
        let signal = result.unwrap();
        assert_eq!(signal.symbol, "BTCUSDT");
        assert_eq!(signal.side, SignalSide::Buy);
        assert_eq!(signal.correlation_id, "test-correlation-123");
        assert_eq!(signal.feature_checksum, "feature-checksum-abc");
        assert_eq!(signal.generation_latency_ms, 50);
    }
    
    #[test]
    fn test_fuse_signals_sync_above_threshold() {
        let mut fusion = SignalFusion::new(0.3, 0);
        
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
        
        let result = fusion.fuse_signals_sync(
            components,
            weights,
            1000,
            "BTCUSDT",
            "v1.0",
            "test-correlation-123".to_string(),
            "feature-checksum-abc".to_string(),
            50,
        ).unwrap();
        
        assert!(result.is_some());
        let signal = result.unwrap();
        assert_eq!(signal.symbol, "BTCUSDT");
        assert_eq!(signal.side, SignalSide::Buy);
        assert_eq!(signal.correlation_id, "test-correlation-123");
        assert_eq!(signal.feature_checksum, "feature-checksum-abc");
        assert_eq!(signal.generation_latency_ms, 50);
    }
    
    #[tokio::test]
    async fn test_fuse_signals_below_threshold() {
        let mut fusion = SignalFusion::new(0.5, 0);
        
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
        
        let result = fusion.fuse_signals(
            components,
            weights,
            1000,
            "BTCUSDT",
            Some("v1.0"),
            Some("test-correlation-456".to_string()),
            Some("feature-checksum-def".to_string()),
            25,
        ).await.unwrap();
        
        assert!(result.is_none());
    }
    
    #[tokio::test]
    async fn test_fuse_signals_with_emission_disabled() {
        let emitter_config = SignalEmitterConfig::default(); // Disabled by default
        let mut fusion = SignalFusion::with_emission(0.3, 0, emitter_config).await.unwrap();
        
        assert!(!fusion.is_emission_enabled());
        
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
        
        let result = fusion.fuse_signals(
            components,
            weights,
            1000,
            "BTCUSDT",
            None, // Should use emitter's model version
            None, // Should generate correlation ID
            None, // Should generate feature checksum
            50,
        ).await.unwrap();
        
        assert!(result.is_some());
        let signal = result.unwrap();
        assert_eq!(signal.symbol, "BTCUSDT");
        assert_eq!(signal.side, SignalSide::Buy);
        assert_eq!(signal.model_version, "v1.0"); // Default from emitter
        assert!(!signal.correlation_id.is_empty());
        assert!(!signal.feature_checksum.is_empty());
    }
    
    #[tokio::test]
    async fn test_fuse_signals_with_emission_enabled() {
        let publisher_config = SignalPublisherConfig {
            backend: PublisherBackend::None, // Use None backend for testing
            enabled: true,
            ..Default::default()
        };
        
        let emitter_config = SignalEmitterConfig::enabled(publisher_config)
            .with_model_version("v2.0".to_string());
        
        let mut fusion = SignalFusion::with_emission(0.3, 0, emitter_config).await.unwrap();
        
        assert!(fusion.is_emission_enabled());
        
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
        
        let result = fusion.fuse_signals(
            components,
            weights,
            1000,
            "BTCUSDT",
            None,
            None,
            None,
            50,
        ).await.unwrap();
        
        assert!(result.is_some());
        let signal = result.unwrap();
        assert_eq!(signal.symbol, "BTCUSDT");
        assert_eq!(signal.side, SignalSide::Buy);
        assert_eq!(signal.model_version, "v2.0"); // Custom model version
        assert!(!signal.correlation_id.is_empty());
        assert!(!signal.feature_checksum.is_empty());
    }
    
    #[tokio::test]
    async fn test_signal_fusion_health_check() {
        let publisher_config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            enabled: true,
            ..Default::default()
        };
        
        let emitter_config = SignalEmitterConfig::enabled(publisher_config);
        let fusion = SignalFusion::with_emission(0.3, 0, emitter_config).await.unwrap();
        
        let health = fusion.emission_health_check().await.unwrap();
        assert!(health); // Should be healthy with None backend
    }
    
    #[tokio::test]
    async fn test_signal_fusion_shutdown() {
        let publisher_config = SignalPublisherConfig {
            backend: PublisherBackend::None,
            enabled: true,
            ..Default::default()
        };
        
        let emitter_config = SignalEmitterConfig::enabled(publisher_config);
        let fusion = SignalFusion::with_emission(0.3, 0, emitter_config).await.unwrap();
        
        let result = fusion.shutdown().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_validation_out_of_range() {
        let components = SignalComponents {
            s_ldc: 1.5, // Out of range
            s_mr: 0.2,
            s_tsmom: 0.1,
        };
        
        assert!(components.validate().is_err());
    }

    #[test]
    fn test_signal_validation_nan() {
        let components = SignalComponents {
            s_ldc: f32::NAN,
            s_mr: 0.2,
            s_tsmom: 0.1,
        };
        
        assert!(components.validate().is_err());
    }

    #[test]
    fn test_signal_validation_valid() {
        let components = SignalComponents {
            s_ldc: 0.5,
            s_mr: -0.3,
            s_tsmom: 0.8,
        };
        
        assert!(components.validate().is_ok());
    }

    #[test]
    fn test_weight_validation_out_of_range() {
        let weights = FusionWeights {
            w_ldc: 1.5, // Out of range
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        assert!(weights.validate().is_err());
    }

    #[test]
    fn test_weight_validation_valid() {
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        assert!(weights.validate().is_ok());
    }

    #[test]
    fn test_weight_normalization() {
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let normalized = weights.normalize();
        let sum = normalized.w_ldc.abs() + normalized.w_mr.abs() + normalized.w_tsmom.abs();
        
        assert!((sum - 1.0).abs() < 1e-6, "Normalized weights should sum to 1.0");
    }

    #[test]
    fn test_weight_normalization_with_negatives() {
        let weights = FusionWeights {
            w_ldc: 0.6,
            w_mr: -0.3,
            w_tsmom: 0.1,
        };
        
        let normalized = weights.normalize();
        let sum = normalized.w_ldc.abs() + normalized.w_mr.abs() + normalized.w_tsmom.abs();
        
        assert!((sum - 1.0).abs() < 1e-6, "Normalized weights should sum to 1.0");
        assert!(normalized.w_mr < 0.0, "Negative weights should remain negative");
    }

    #[test]
    fn test_weight_normalization_zero_sum() {
        let weights = FusionWeights {
            w_ldc: 0.0,
            w_mr: 0.0,
            w_tsmom: 0.0,
        };
        
        let normalized = weights.normalize();
        
        // Should return equal weights when sum is zero
        assert!((normalized.w_ldc - 0.33).abs() < 0.01);
        assert!((normalized.w_mr - 0.33).abs() < 0.01);
        assert!((normalized.w_tsmom - 0.34).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_cooldown_logic() {
        let mut fusion = SignalFusion::new(0.3, 60);
        
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
        
        // First signal should succeed
        let result1 = fusion.fuse_signals(
            components.clone(),
            weights.clone(),
            1000,
            "BTCUSDT",
            Some("v1.0"),
            Some("test-correlation-1".to_string()),
            Some("checksum-1".to_string()),
            30,
        ).await.unwrap();
        assert!(result1.is_some());
        
        // Second signal within cooldown should be suppressed
        let result2 = fusion.fuse_signals(
            components.clone(),
            weights.clone(),
            1030, // 30 seconds later, within 60s cooldown
            "BTCUSDT",
            Some("v1.0"),
            Some("test-correlation-2".to_string()),
            Some("checksum-2".to_string()),
            25,
        ).await.unwrap();
        assert!(result2.is_none());
        
        // Third signal after cooldown should succeed
        let result3 = fusion.fuse_signals(
            components,
            weights,
            1070, // 70 seconds after first, beyond cooldown
            "BTCUSDT",
            Some("v1.0"),
            Some("test-correlation-3".to_string()),
            Some("checksum-3".to_string()),
            35,
        ).await.unwrap();
        assert!(result3.is_some());
    }

    #[tokio::test]
    async fn test_is_in_cooldown() {
        let mut fusion = SignalFusion::new(0.3, 60);
        
        // No signal yet, not in cooldown
        assert!(!fusion.is_in_cooldown(1000));
        
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
        
        // Generate a signal
        fusion.fuse_signals(
            components, 
            weights, 
            1000, 
            "BTCUSDT", 
            Some("v1.0"),
            Some("test-correlation".to_string()),
            Some("checksum".to_string()),
            20,
        ).await.unwrap();
        
        // Should be in cooldown
        assert!(fusion.is_in_cooldown(1030));
        
        // Should not be in cooldown after period
        assert!(!fusion.is_in_cooldown(1070));
    }

    #[tokio::test]
    async fn test_fuse_signals_with_invalid_components() {
        let mut fusion = SignalFusion::new(0.3, 0);
        
        let components = SignalComponents {
            s_ldc: 2.0, // Out of range
            s_mr: 0.2,
            s_tsmom: 0.1,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let result = fusion.fuse_signals(
            components,
            weights,
            1000,
            "BTCUSDT",
            Some("v1.0"),
            Some("test-correlation".to_string()),
            Some("checksum".to_string()),
            30,
        ).await;
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fuse_signals_with_invalid_weights() {
        let mut fusion = SignalFusion::new(0.3, 0);
        
        let components = SignalComponents {
            s_ldc: 0.8,
            s_mr: 0.2,
            s_tsmom: 0.1,
        };
        
        let weights = FusionWeights {
            w_ldc: 2.0, // Out of range
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let result = fusion.fuse_signals(
            components,
            weights,
            1000,
            "BTCUSDT",
            Some("v1.0"),
            Some("test-correlation".to_string()),
            Some("checksum".to_string()),
            30,
        ).await;
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fusion_without_normalization() {
        let mut fusion = SignalFusion::with_normalization(0.3, 0, false);
        
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
        
        let result = fusion.fuse_signals(
            components,
            weights.clone(),
            1000,
            "BTCUSDT",
            Some("v1.0"),
            Some("test-correlation".to_string()),
            Some("checksum".to_string()),
            30,
        ).await.unwrap();
        
        assert!(result.is_some());
        let signal = result.unwrap();
        
        // Weights should not be normalized
        assert_eq!(signal.weights.w_ldc, weights.w_ldc);
        assert_eq!(signal.weights.w_mr, weights.w_mr);
        assert_eq!(signal.weights.w_tsmom, weights.w_tsmom);
    }

    #[tokio::test]
    async fn test_sell_signal_generation() {
        let mut fusion = SignalFusion::new(0.3, 0);
        
        let components = SignalComponents {
            s_ldc: -0.8,
            s_mr: -0.2,
            s_tsmom: -0.1,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let result = fusion.fuse_signals(
            components,
            weights,
            1000,
            "BTCUSDT",
            Some("v1.0"),
            Some("test-correlation".to_string()),
            Some("checksum".to_string()),
            30,
        ).await.unwrap();
        
        assert!(result.is_some());
        let signal = result.unwrap();
        assert_eq!(signal.side, SignalSide::Sell);
        assert!(signal.strength < 0.0);
    }

    #[test]
    fn test_signal_side_enum() {
        assert_eq!(SignalSide::Buy.to_string(), "BUY");
        assert_eq!(SignalSide::Sell.to_string(), "SELL");
        assert_eq!(SignalSide::Hold.to_string(), "HOLD");
        
        assert_eq!("BUY".parse::<SignalSide>().unwrap(), SignalSide::Buy);
        assert_eq!("SELL".parse::<SignalSide>().unwrap(), SignalSide::Sell);
        assert_eq!("HOLD".parse::<SignalSide>().unwrap(), SignalSide::Hold);
        assert_eq!("buy".parse::<SignalSide>().unwrap(), SignalSide::Buy);
        
        assert!("INVALID".parse::<SignalSide>().is_err());
    }

    #[test]
    fn test_trading_signal_validation() {
        let now = chrono::Utc::now().timestamp();
        
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
        
        // Valid signal
        let signal = TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components.clone(),
            weights.clone(),
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        );
        
        assert!(signal.validate().is_ok());
        assert!(signal.validate_consistency().is_ok());
        
        // Invalid timestamp (too old)
        let old_signal = TradingSignal::new(
            now - 7200, // 2 hours ago
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components.clone(),
            weights.clone(),
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        );
        
        assert!(old_signal.validate().is_err());
        
        // Invalid symbol (lowercase)
        let invalid_symbol_signal = TradingSignal::new(
            now,
            "btcusdt".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components.clone(),
            weights.clone(),
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        );
        
        assert!(invalid_symbol_signal.validate().is_err());
        
        // Invalid strength range
        let invalid_strength_signal = TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            2.0, // Out of range
            0.85,
            components.clone(),
            weights.clone(),
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        );
        
        assert!(invalid_strength_signal.validate().is_err());
        
        // Invalid confidence range
        let invalid_confidence_signal = TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            1.5, // Out of range
            components,
            weights,
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        );
        
        assert!(invalid_confidence_signal.validate().is_err());
    }

    #[test]
    fn test_trading_signal_consistency() {
        let now = chrono::Utc::now().timestamp();
        
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
        
        // Inconsistent BUY signal with negative strength
        let inconsistent_buy = TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            -0.75, // Negative strength for BUY
            0.85,
            components.clone(),
            weights.clone(),
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        );
        
        assert!(inconsistent_buy.validate_consistency().is_err());
        
        // Inconsistent SELL signal with positive strength
        let inconsistent_sell = TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Sell,
            0.75, // Positive strength for SELL
            0.85,
            components,
            weights,
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        );
        
        assert!(inconsistent_sell.validate_consistency().is_err());
    }

    #[test]
    fn test_trading_signal_serialization() {
        let now = chrono::Utc::now().timestamp();
        
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
        
        let signal = TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components,
            weights,
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        ).with_hmm_data(vec![0.7, 0.3], false);
        
        // Test JSON serialization
        let json = signal.to_json().unwrap();
        let deserialized = TradingSignal::from_json(&json).unwrap();
        
        assert_eq!(signal.symbol, deserialized.symbol);
        assert_eq!(signal.side, deserialized.side);
        assert_eq!(signal.strength, deserialized.strength);
        assert_eq!(signal.correlation_id, deserialized.correlation_id);
        assert_eq!(signal.hmm_state_probabilities, deserialized.hmm_state_probabilities);
        
        // Test compact string representation
        let compact = signal.to_compact_string();
        assert!(compact.contains("BTCUSDT"));
        assert!(compact.contains("BUY"));
        assert!(compact.contains("correlat"));
    }

    #[test]
    fn test_trading_signal_hmm_validation() {
        let now = chrono::Utc::now().timestamp();
        
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
        
        // Valid HMM probabilities
        let signal_valid_hmm = TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components.clone(),
            weights.clone(),
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        ).with_hmm_data(vec![0.7, 0.3], false);
        
        assert!(signal_valid_hmm.validate().is_ok());
        
        // Invalid HMM probabilities (don't sum to 1.0)
        let signal_invalid_hmm = TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components.clone(),
            weights.clone(),
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        ).with_hmm_data(vec![0.5, 0.3], false); // Sum = 0.8, not 1.0
        
        assert!(signal_invalid_hmm.validate().is_err());
        
        // Invalid HMM probabilities (out of range)
        let signal_invalid_range = TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components,
            weights,
            "v1.0".to_string(),
            "correlation-123".to_string(),
            "checksum-abc".to_string(),
            50,
        ).with_hmm_data(vec![1.2, -0.2], false); // Out of [0,1] range
        
        assert!(signal_invalid_range.validate().is_err());
    }
}
