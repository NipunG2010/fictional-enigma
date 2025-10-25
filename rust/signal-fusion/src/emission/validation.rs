//! Signal validation framework for comprehensive trading signal validation
//! 
//! This module provides a structured approach to validating trading signals
//! with detailed error reporting and configurable validation rules.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{TradingSignal, SignalComponents, FusionWeights, SignalSide};
use super::{Result, SignalEmissionError};

/// Detailed validation error with context
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    /// Timestamp validation errors
    #[error("Timestamp error: {message}")]
    TimestampError { message: String, timestamp: i64, current_time: i64 },
    
    /// Symbol format validation errors
    #[error("Symbol format error: {message}")]
    SymbolFormatError { message: String, symbol: String },
    
    /// Signal side validation errors
    #[error("Signal side error: {message}")]
    SignalSideError { message: String, side: String },
    
    /// Strength validation errors
    #[error("Strength validation error: {message}")]
    StrengthError { message: String, strength: f32, min: f32, max: f32 },
    
    /// Confidence validation errors
    #[error("Confidence validation error: {message}")]
    ConfidenceError { message: String, confidence: f32, min: f32, max: f32 },
    
    /// Component validation errors
    #[error("Component validation error: {message}")]
    ComponentError { message: String, component: String, value: f32 },
    
    /// Weight validation errors
    #[error("Weight validation error: {message}")]
    WeightError { message: String, weight: String, value: f32 },
    
    /// Model version validation errors
    #[error("Model version error: {message}")]
    ModelVersionError { message: String, version: String },
    
    /// Correlation ID validation errors
    #[error("Correlation ID error: {message}")]
    CorrelationIdError { message: String, correlation_id: String },
    
    /// Feature checksum validation errors
    #[error("Feature checksum error: {message}")]
    FeatureChecksumError { message: String, checksum: String },
    
    /// HMM state probability validation errors
    #[error("HMM state probability error: {message}")]
    HmmStateError { message: String, probabilities: Vec<f32> },
    
    /// Signal consistency validation errors
    #[error("Signal consistency error: {message}")]
    ConsistencyError { message: String, side: String, strength: f32 },
    
    /// Custom validation errors
    #[error("Custom validation error: {message}")]
    CustomError { message: String, context: HashMap<String, String> },
}

impl ValidationError {
    /// Get the error category for metrics and logging
    pub fn category(&self) -> &'static str {
        match self {
            Self::TimestampError { .. } => "timestamp",
            Self::SymbolFormatError { .. } => "symbol_format",
            Self::SignalSideError { .. } => "signal_side",
            Self::StrengthError { .. } => "strength",
            Self::ConfidenceError { .. } => "confidence",
            Self::ComponentError { .. } => "component",
            Self::WeightError { .. } => "weight",
            Self::ModelVersionError { .. } => "model_version",
            Self::CorrelationIdError { .. } => "correlation_id",
            Self::FeatureChecksumError { .. } => "feature_checksum",
            Self::HmmStateError { .. } => "hmm_state",
            Self::ConsistencyError { .. } => "consistency",
            Self::CustomError { .. } => "custom",
        }
    }
    
    /// Get additional context for debugging
    pub fn context(&self) -> HashMap<String, String> {
        let mut context = HashMap::new();
        
        match self {
            Self::TimestampError { timestamp, current_time, .. } => {
                context.insert("timestamp".to_string(), timestamp.to_string());
                context.insert("current_time".to_string(), current_time.to_string());
            }
            Self::SymbolFormatError { symbol, .. } => {
                context.insert("symbol".to_string(), symbol.clone());
            }
            Self::StrengthError { strength, min, max, .. } => {
                context.insert("strength".to_string(), strength.to_string());
                context.insert("min".to_string(), min.to_string());
                context.insert("max".to_string(), max.to_string());
            }
            Self::ConfidenceError { confidence, min, max, .. } => {
                context.insert("confidence".to_string(), confidence.to_string());
                context.insert("min".to_string(), min.to_string());
                context.insert("max".to_string(), max.to_string());
            }
            Self::ComponentError { component, value, .. } => {
                context.insert("component".to_string(), component.clone());
                context.insert("value".to_string(), value.to_string());
            }
            Self::WeightError { weight, value, .. } => {
                context.insert("weight".to_string(), weight.clone());
                context.insert("value".to_string(), value.to_string());
            }
            Self::CustomError { context: custom_context, .. } => {
                context.extend(custom_context.clone());
            }
            _ => {}
        }
        
        context
    }
}

/// Configuration for signal validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum age of signals in seconds (default: 3600 = 1 hour)
    pub max_signal_age_seconds: i64,
    
    /// Maximum future time for signals in seconds (default: 300 = 5 minutes)
    pub max_future_seconds: i64,
    
    /// Minimum strength value (default: -1.0)
    pub min_strength: f32,
    
    /// Maximum strength value (default: 1.0)
    pub max_strength: f32,
    
    /// Minimum confidence value (default: 0.0)
    pub min_confidence: f32,
    
    /// Maximum confidence value (default: 1.0)
    pub max_confidence: f32,
    
    /// Minimum component value (default: -1.0)
    pub min_component_value: f32,
    
    /// Maximum component value (default: 1.0)
    pub max_component_value: f32,
    
    /// Minimum weight value (default: -1.0)
    pub min_weight_value: f32,
    
    /// Maximum weight value (default: 1.0)
    pub max_weight_value: f32,
    
    /// Required symbol format pattern (default: uppercase alphanumeric)
    pub symbol_pattern: String,
    
    /// Minimum symbol length (default: 3)
    pub min_symbol_length: usize,
    
    /// Maximum symbol length (default: 20)
    pub max_symbol_length: usize,
    
    /// Minimum model version length (default: 1)
    pub min_model_version_length: usize,
    
    /// Maximum model version length (default: 50)
    pub max_model_version_length: usize,
    
    /// Minimum correlation ID length (default: 8)
    pub min_correlation_id_length: usize,
    
    /// Maximum correlation ID length (default: 100)
    pub max_correlation_id_length: usize,
    
    /// Minimum feature checksum length (default: 8)
    pub min_feature_checksum_length: usize,
    
    /// Maximum feature checksum length (default: 100)
    pub max_feature_checksum_length: usize,
    
    /// HMM probability sum tolerance (default: 0.01)
    pub hmm_probability_sum_tolerance: f32,
    
    /// Enable strict consistency checking (default: true)
    pub strict_consistency_check: bool,
    
    /// Enable HMM validation (default: true)
    pub enable_hmm_validation: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_signal_age_seconds: 3600,
            max_future_seconds: 300,
            min_strength: -1.0,
            max_strength: 1.0,
            min_confidence: 0.0,
            max_confidence: 1.0,
            min_component_value: -1.0,
            max_component_value: 1.0,
            min_weight_value: -1.0,
            max_weight_value: 1.0,
            symbol_pattern: "^[A-Z0-9]+$".to_string(),
            min_symbol_length: 3,
            max_symbol_length: 20,
            min_model_version_length: 1,
            max_model_version_length: 50,
            min_correlation_id_length: 8,
            max_correlation_id_length: 100,
            min_feature_checksum_length: 8,
            max_feature_checksum_length: 100,
            hmm_probability_sum_tolerance: 0.01,
            strict_consistency_check: true,
            enable_hmm_validation: true,
        }
    }
}

/// Comprehensive signal validator with configurable rules
pub struct SignalValidator {
    config: ValidationConfig,
}

impl SignalValidator {
    /// Create a new signal validator with default configuration
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }
    
    /// Create a signal validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }
    
    /// Get the current validation configuration
    pub fn config(&self) -> &ValidationConfig {
        &self.config
    }
    
    /// Update the validation configuration
    pub fn set_config(&mut self, config: ValidationConfig) {
        self.config = config;
    }
    
    /// Validate a complete trading signal
    pub fn validate(&self, signal: &TradingSignal) -> Result<()> {
        let mut errors = Vec::new();
        
        // Validate timestamp
        if let Err(e) = self.validate_timestamp(signal.timestamp) {
            errors.push(e);
        }
        
        // Validate symbol
        if let Err(e) = self.validate_symbol(&signal.symbol) {
            errors.push(e);
        }
        
        // Validate side
        if let Err(e) = self.validate_side(&signal.side) {
            errors.push(e);
        }
        
        // Validate strength
        if let Err(e) = self.validate_strength(signal.strength) {
            errors.push(e);
        }
        
        // Validate confidence
        if let Err(e) = self.validate_confidence(signal.confidence) {
            errors.push(e);
        }
        
        // Validate components
        if let Err(e) = self.validate_components(&signal.components) {
            errors.push(e);
        }
        
        // Validate weights
        if let Err(e) = self.validate_weights(&signal.weights) {
            errors.push(e);
        }
        
        // Validate model version
        if let Err(e) = self.validate_model_version(&signal.model_version) {
            errors.push(e);
        }
        
        // Validate correlation ID
        if let Err(e) = self.validate_correlation_id(&signal.correlation_id) {
            errors.push(e);
        }
        
        // Validate feature checksum
        if let Err(e) = self.validate_feature_checksum(&signal.feature_checksum) {
            errors.push(e);
        }
        
        // Validate HMM state probabilities if present and enabled
        if self.config.enable_hmm_validation {
            if let Some(ref probs) = signal.hmm_state_probabilities {
                if let Err(e) = self.validate_hmm_probabilities(probs) {
                    errors.push(e);
                }
            }
        }
        
        // Validate signal consistency if enabled
        if self.config.strict_consistency_check {
            if let Err(e) = self.validate_consistency(&signal.side, signal.strength) {
                errors.push(e);
            }
        }
        
        // Return first error if any
        if let Some(error) = errors.into_iter().next() {
            return Err(SignalEmissionError::ValidationError { message: error.to_string() });
        }
        
        Ok(())
    }
    
    /// Validate timestamp ranges
    pub fn validate_timestamp(&self, timestamp: i64) -> std::result::Result<(), ValidationError> {
        let now = chrono::Utc::now().timestamp();
        
        if timestamp < now - self.config.max_signal_age_seconds {
            return Err(ValidationError::TimestampError {
                message: format!(
                    "Signal timestamp too old: {}s ago (max: {}s)",
                    now - timestamp,
                    self.config.max_signal_age_seconds
                ),
                timestamp,
                current_time: now,
            });
        }
        
        if timestamp > now + self.config.max_future_seconds {
            return Err(ValidationError::TimestampError {
                message: format!(
                    "Signal timestamp too far in future: {}s ahead (max: {}s)",
                    timestamp - now,
                    self.config.max_future_seconds
                ),
                timestamp,
                current_time: now,
            });
        }
        
        Ok(())
    }
    
    /// Validate symbol format
    pub fn validate_symbol(&self, symbol: &str) -> std::result::Result<(), ValidationError> {
        if symbol.len() < self.config.min_symbol_length {
            return Err(ValidationError::SymbolFormatError {
                message: format!(
                    "Symbol too short: {} chars (min: {})",
                    symbol.len(),
                    self.config.min_symbol_length
                ),
                symbol: symbol.to_string(),
            });
        }
        
        if symbol.len() > self.config.max_symbol_length {
            return Err(ValidationError::SymbolFormatError {
                message: format!(
                    "Symbol too long: {} chars (max: {})",
                    symbol.len(),
                    self.config.max_symbol_length
                ),
                symbol: symbol.to_string(),
            });
        }
        
        // Check if symbol matches pattern (uppercase alphanumeric by default)
        if !symbol.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(ValidationError::SymbolFormatError {
                message: "Symbol must contain only alphanumeric characters".to_string(),
                symbol: symbol.to_string(),
            });
        }
        
        if symbol != symbol.to_uppercase() {
            return Err(ValidationError::SymbolFormatError {
                message: "Symbol must be uppercase".to_string(),
                symbol: symbol.to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Validate signal side
    pub fn validate_side(&self, _side: &SignalSide) -> std::result::Result<(), ValidationError> {
        // SignalSide enum is already validated by type system
        // This method is here for consistency and future extensibility
        Ok(())
    }
    
    /// Validate strength values
    pub fn validate_strength(&self, strength: f32) -> std::result::Result<(), ValidationError> {
        if !strength.is_finite() {
            return Err(ValidationError::StrengthError {
                message: "Strength must be finite".to_string(),
                strength,
                min: self.config.min_strength,
                max: self.config.max_strength,
            });
        }
        
        if strength < self.config.min_strength || strength > self.config.max_strength {
            return Err(ValidationError::StrengthError {
                message: format!(
                    "Strength out of range [{}, {}]",
                    self.config.min_strength,
                    self.config.max_strength
                ),
                strength,
                min: self.config.min_strength,
                max: self.config.max_strength,
            });
        }
        
        Ok(())
    }
    
    /// Validate confidence ranges
    pub fn validate_confidence(&self, confidence: f32) -> std::result::Result<(), ValidationError> {
        if !confidence.is_finite() {
            return Err(ValidationError::ConfidenceError {
                message: "Confidence must be finite".to_string(),
                confidence,
                min: self.config.min_confidence,
                max: self.config.max_confidence,
            });
        }
        
        if confidence < self.config.min_confidence || confidence > self.config.max_confidence {
            return Err(ValidationError::ConfidenceError {
                message: format!(
                    "Confidence out of range [{}, {}]",
                    self.config.min_confidence,
                    self.config.max_confidence
                ),
                confidence,
                min: self.config.min_confidence,
                max: self.config.max_confidence,
            });
        }
        
        Ok(())
    }
    
    /// Validate signal components
    pub fn validate_components(&self, components: &SignalComponents) -> std::result::Result<(), ValidationError> {
        self.validate_component_value("s_ldc", components.s_ldc)?;
        self.validate_component_value("s_mr", components.s_mr)?;
        self.validate_component_value("s_tsmom", components.s_tsmom)?;
        Ok(())
    }
    
    /// Validate individual component value
    fn validate_component_value(&self, name: &str, value: f32) -> std::result::Result<(), ValidationError> {
        if !value.is_finite() {
            return Err(ValidationError::ComponentError {
                message: format!("Component {} must be finite", name),
                component: name.to_string(),
                value,
            });
        }
        
        if value < self.config.min_component_value || value > self.config.max_component_value {
            return Err(ValidationError::ComponentError {
                message: format!(
                    "Component {} out of range [{}, {}]",
                    name,
                    self.config.min_component_value,
                    self.config.max_component_value
                ),
                component: name.to_string(),
                value,
            });
        }
        
        Ok(())
    }
    
    /// Validate fusion weights
    pub fn validate_weights(&self, weights: &FusionWeights) -> std::result::Result<(), ValidationError> {
        self.validate_weight_value("w_ldc", weights.w_ldc)?;
        self.validate_weight_value("w_mr", weights.w_mr)?;
        self.validate_weight_value("w_tsmom", weights.w_tsmom)?;
        Ok(())
    }
    
    /// Validate individual weight value
    fn validate_weight_value(&self, name: &str, value: f32) -> std::result::Result<(), ValidationError> {
        if !value.is_finite() {
            return Err(ValidationError::WeightError {
                message: format!("Weight {} must be finite", name),
                weight: name.to_string(),
                value,
            });
        }
        
        if value < self.config.min_weight_value || value > self.config.max_weight_value {
            return Err(ValidationError::WeightError {
                message: format!(
                    "Weight {} out of range [{}, {}]",
                    name,
                    self.config.min_weight_value,
                    self.config.max_weight_value
                ),
                weight: name.to_string(),
                value,
            });
        }
        
        Ok(())
    }
    
    /// Validate model version format
    pub fn validate_model_version(&self, version: &str) -> std::result::Result<(), ValidationError> {
        if version.len() < self.config.min_model_version_length {
            return Err(ValidationError::ModelVersionError {
                message: format!(
                    "Model version too short: {} chars (min: {})",
                    version.len(),
                    self.config.min_model_version_length
                ),
                version: version.to_string(),
            });
        }
        
        if version.len() > self.config.max_model_version_length {
            return Err(ValidationError::ModelVersionError {
                message: format!(
                    "Model version too long: {} chars (max: {})",
                    version.len(),
                    self.config.max_model_version_length
                ),
                version: version.to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Validate correlation ID format
    pub fn validate_correlation_id(&self, correlation_id: &str) -> std::result::Result<(), ValidationError> {
        if correlation_id.len() < self.config.min_correlation_id_length {
            return Err(ValidationError::CorrelationIdError {
                message: format!(
                    "Correlation ID too short: {} chars (min: {})",
                    correlation_id.len(),
                    self.config.min_correlation_id_length
                ),
                correlation_id: correlation_id.to_string(),
            });
        }
        
        if correlation_id.len() > self.config.max_correlation_id_length {
            return Err(ValidationError::CorrelationIdError {
                message: format!(
                    "Correlation ID too long: {} chars (max: {})",
                    correlation_id.len(),
                    self.config.max_correlation_id_length
                ),
                correlation_id: correlation_id.to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Validate feature checksum format
    pub fn validate_feature_checksum(&self, checksum: &str) -> std::result::Result<(), ValidationError> {
        if checksum.len() < self.config.min_feature_checksum_length {
            return Err(ValidationError::FeatureChecksumError {
                message: format!(
                    "Feature checksum too short: {} chars (min: {})",
                    checksum.len(),
                    self.config.min_feature_checksum_length
                ),
                checksum: checksum.to_string(),
            });
        }
        
        if checksum.len() > self.config.max_feature_checksum_length {
            return Err(ValidationError::FeatureChecksumError {
                message: format!(
                    "Feature checksum too long: {} chars (max: {})",
                    checksum.len(),
                    self.config.max_feature_checksum_length
                ),
                checksum: checksum.to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Validate HMM state probabilities
    pub fn validate_hmm_probabilities(&self, probabilities: &[f32]) -> std::result::Result<(), ValidationError> {
        if probabilities.is_empty() {
            return Err(ValidationError::HmmStateError {
                message: "HMM state probabilities cannot be empty".to_string(),
                probabilities: probabilities.to_vec(),
            });
        }
        
        // Check individual probability values
        for (i, &prob) in probabilities.iter().enumerate() {
            if !prob.is_finite() {
                return Err(ValidationError::HmmStateError {
                    message: format!("HMM state probability {} is not finite", i),
                    probabilities: probabilities.to_vec(),
                });
            }
            
            if prob < 0.0 || prob > 1.0 {
                return Err(ValidationError::HmmStateError {
                    message: format!("HMM state probability {} out of range [0.0, 1.0]: {}", i, prob),
                    probabilities: probabilities.to_vec(),
                });
            }
        }
        
        // Check that probabilities sum to approximately 1.0
        let sum: f32 = probabilities.iter().sum();
        if (sum - 1.0).abs() > self.config.hmm_probability_sum_tolerance {
            return Err(ValidationError::HmmStateError {
                message: format!(
                    "HMM state probabilities should sum to 1.0 ± {}, got: {}",
                    self.config.hmm_probability_sum_tolerance,
                    sum
                ),
                probabilities: probabilities.to_vec(),
            });
        }
        
        Ok(())
    }
    
    /// Validate signal consistency (side matches strength sign)
    pub fn validate_consistency(&self, side: &SignalSide, strength: f32) -> std::result::Result<(), ValidationError> {
        match side {
            SignalSide::Buy => {
                if strength <= 0.0 {
                    return Err(ValidationError::ConsistencyError {
                        message: "BUY signal should have positive strength".to_string(),
                        side: side.to_string(),
                        strength,
                    });
                }
            }
            SignalSide::Sell => {
                if strength >= 0.0 {
                    return Err(ValidationError::ConsistencyError {
                        message: "SELL signal should have negative strength".to_string(),
                        side: side.to_string(),
                        strength,
                    });
                }
            }
            SignalSide::Hold => {
                // HOLD signals can have any strength, but warn if too high
                // This is handled in the main validation logic
            }
        }
        
        Ok(())
    }
}

impl Default for SignalValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SignalComponents, FusionWeights, TradingSignal, SignalSide};
    
    fn create_valid_signal() -> TradingSignal {
        let now = chrono::Utc::now().timestamp();
        
        TradingSignal::new(
            now,
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            SignalComponents {
                s_ldc: 0.5,
                s_mr: 0.3,
                s_tsmom: 0.2,
            },
            FusionWeights {
                w_ldc: 0.5,
                w_mr: 0.3,
                w_tsmom: 0.2,
            },
            "v1.0".to_string(),
            "correlation-123456".to_string(),
            "checksum-abcdef".to_string(),
            50,
        )
    }
    
    #[test]
    fn test_validator_creation() {
        let validator = SignalValidator::new();
        assert_eq!(validator.config().max_signal_age_seconds, 3600);
        
        let custom_config = ValidationConfig {
            max_signal_age_seconds: 1800,
            ..Default::default()
        };
        let custom_validator = SignalValidator::with_config(custom_config);
        assert_eq!(custom_validator.config().max_signal_age_seconds, 1800);
    }
    
    #[test]
    fn test_valid_signal_validation() {
        let validator = SignalValidator::new();
        let signal = create_valid_signal();
        
        assert!(validator.validate(&signal).is_ok());
    }
    
    #[test]
    fn test_timestamp_validation() {
        let validator = SignalValidator::new();
        let now = chrono::Utc::now().timestamp();
        
        // Valid timestamp
        assert!(validator.validate_timestamp(now).is_ok());
        
        // Too old
        assert!(validator.validate_timestamp(now - 7200).is_err());
        
        // Too far in future
        assert!(validator.validate_timestamp(now + 600).is_err());
    }
    
    #[test]
    fn test_symbol_validation() {
        let validator = SignalValidator::new();
        
        // Valid symbols
        assert!(validator.validate_symbol("BTCUSDT").is_ok());
        assert!(validator.validate_symbol("ETH").is_ok());
        assert!(validator.validate_symbol("AAPL123").is_ok());
        
        // Invalid symbols
        assert!(validator.validate_symbol("bt").is_err()); // Too short
        assert!(validator.validate_symbol("btcusdt").is_err()); // Lowercase
        assert!(validator.validate_symbol("BTC-USDT").is_err()); // Special chars
        assert!(validator.validate_symbol("").is_err()); // Empty
    }
    
    #[test]
    fn test_strength_validation() {
        let validator = SignalValidator::new();
        
        // Valid strengths
        assert!(validator.validate_strength(0.0).is_ok());
        assert!(validator.validate_strength(0.75).is_ok());
        assert!(validator.validate_strength(-0.5).is_ok());
        assert!(validator.validate_strength(1.0).is_ok());
        assert!(validator.validate_strength(-1.0).is_ok());
        
        // Invalid strengths
        assert!(validator.validate_strength(1.5).is_err()); // Too high
        assert!(validator.validate_strength(-1.5).is_err()); // Too low
        assert!(validator.validate_strength(f32::NAN).is_err()); // NaN
        assert!(validator.validate_strength(f32::INFINITY).is_err()); // Infinity
    }
    
    #[test]
    fn test_confidence_validation() {
        let validator = SignalValidator::new();
        
        // Valid confidences
        assert!(validator.validate_confidence(0.0).is_ok());
        assert!(validator.validate_confidence(0.5).is_ok());
        assert!(validator.validate_confidence(1.0).is_ok());
        
        // Invalid confidences
        assert!(validator.validate_confidence(-0.1).is_err()); // Too low
        assert!(validator.validate_confidence(1.1).is_err()); // Too high
        assert!(validator.validate_confidence(f32::NAN).is_err()); // NaN
    }
    
    #[test]
    fn test_components_validation() {
        let validator = SignalValidator::new();
        
        // Valid components
        let valid_components = SignalComponents {
            s_ldc: 0.5,
            s_mr: -0.3,
            s_tsmom: 0.8,
        };
        assert!(validator.validate_components(&valid_components).is_ok());
        
        // Invalid components
        let invalid_components = SignalComponents {
            s_ldc: 2.0, // Out of range
            s_mr: 0.3,
            s_tsmom: 0.2,
        };
        assert!(validator.validate_components(&invalid_components).is_err());
    }
    
    #[test]
    fn test_weights_validation() {
        let validator = SignalValidator::new();
        
        // Valid weights
        let valid_weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        assert!(validator.validate_weights(&valid_weights).is_ok());
        
        // Invalid weights
        let invalid_weights = FusionWeights {
            w_ldc: 2.0, // Out of range
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        assert!(validator.validate_weights(&invalid_weights).is_err());
    }
    
    #[test]
    fn test_model_version_validation() {
        let validator = SignalValidator::new();
        
        // Valid versions
        assert!(validator.validate_model_version("v1.0").is_ok());
        assert!(validator.validate_model_version("production_v2.1.0").is_ok());
        
        // Invalid versions
        assert!(validator.validate_model_version("").is_err()); // Too short
        assert!(validator.validate_model_version(&"x".repeat(100)).is_err()); // Too long
    }
    
    #[test]
    fn test_correlation_id_validation() {
        let validator = SignalValidator::new();
        
        // Valid correlation IDs
        assert!(validator.validate_correlation_id("corr-12345678").is_ok());
        assert!(validator.validate_correlation_id("uuid-abcd-efgh").is_ok());
        
        // Invalid correlation IDs
        assert!(validator.validate_correlation_id("short").is_err()); // Too short
        assert!(validator.validate_correlation_id(&"x".repeat(200)).is_err()); // Too long
    }
    
    #[test]
    fn test_hmm_probabilities_validation() {
        let validator = SignalValidator::new();
        
        // Valid probabilities
        assert!(validator.validate_hmm_probabilities(&[0.7, 0.3]).is_ok());
        assert!(validator.validate_hmm_probabilities(&[0.4, 0.3, 0.3]).is_ok());
        
        // Invalid probabilities
        assert!(validator.validate_hmm_probabilities(&[]).is_err()); // Empty
        assert!(validator.validate_hmm_probabilities(&[0.5, 0.3]).is_err()); // Don't sum to 1.0
        assert!(validator.validate_hmm_probabilities(&[1.2, -0.2]).is_err()); // Out of range
        assert!(validator.validate_hmm_probabilities(&[f32::NAN, 0.5]).is_err()); // NaN
    }
    
    #[test]
    fn test_consistency_validation() {
        let validator = SignalValidator::new();
        
        // Valid consistency
        assert!(validator.validate_consistency(&SignalSide::Buy, 0.75).is_ok());
        assert!(validator.validate_consistency(&SignalSide::Sell, -0.5).is_ok());
        assert!(validator.validate_consistency(&SignalSide::Hold, 0.1).is_ok());
        assert!(validator.validate_consistency(&SignalSide::Hold, -0.1).is_ok());
        
        // Invalid consistency
        assert!(validator.validate_consistency(&SignalSide::Buy, -0.5).is_err());
        assert!(validator.validate_consistency(&SignalSide::Sell, 0.75).is_err());
    }
    
    #[test]
    fn test_validation_error_context() {
        let error = ValidationError::StrengthError {
            message: "Test error".to_string(),
            strength: 2.0,
            min: -1.0,
            max: 1.0,
        };
        
        let context = error.context();
        assert_eq!(context.get("strength"), Some(&"2".to_string()));
        assert_eq!(context.get("min"), Some(&"-1".to_string()));
        assert_eq!(context.get("max"), Some(&"1".to_string()));
        assert_eq!(error.category(), "strength");
    }
    
    #[test]
    fn test_custom_validation_config() {
        let config = ValidationConfig {
            max_signal_age_seconds: 1800,
            min_strength: -0.5,
            max_strength: 0.5,
            strict_consistency_check: false,
            ..Default::default()
        };
        
        let validator = SignalValidator::with_config(config);
        
        // Should pass with relaxed strength limits
        assert!(validator.validate_strength(0.4).is_ok());
        
        // Should fail with tighter strength limits
        assert!(validator.validate_strength(0.8).is_err());
    }
}