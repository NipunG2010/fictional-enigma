//! Comprehensive audit logging system for signal emission
//! 
//! This module provides structured audit logging for all signal emission events,
//! feature computation events, validation errors, and publisher operations.
//! It includes correlation ID tracking across the signal lifecycle for full traceability.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;
use crate::{TradingSignal, FusionWeights};
use super::{SignalEmissionError, ValidationError};

/// Generates a new correlation ID for tracking events across the signal lifecycle
pub fn generate_correlation_id() -> String {
    Uuid::new_v4().to_string()
}

/// Generates a timestamp in milliseconds since Unix epoch
pub fn current_timestamp_ms() -> i64 {
    Utc::now().timestamp_millis()
}

/// Generates an event ID for unique identification of audit events
pub fn generate_event_id() -> String {
    Uuid::new_v4().to_string()
}

/// Base trait for all audit events
pub trait AuditEvent {
    /// Get the event ID
    fn event_id(&self) -> &str;
    
    /// Get the timestamp
    fn timestamp(&self) -> i64;
    
    /// Get the correlation ID
    fn correlation_id(&self) -> &str;
    
    /// Get the event type name
    fn event_type(&self) -> &'static str;
    
    /// Validate the event structure
    fn validate(&self) -> Result<(), SignalEmissionError>;
    
    /// Serialize the event to JSON
    fn to_json(&self) -> Result<String, SignalEmissionError>
    where
        Self: Serialize,
    {
        serde_json::to_string(self).map_err(SignalEmissionError::from)
    }
}

/// Audit event for successful signal emissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalEmissionEvent {
    /// Unique identifier for this audit event
    pub event_id: String,
    
    /// Timestamp when the event occurred (milliseconds since Unix epoch)
    pub timestamp: i64,
    
    /// Correlation ID linking this event to the signal lifecycle
    pub correlation_id: String,
    
    /// The complete trading signal that was emitted
    pub signal: TradingSignal,
    
    /// Publisher backend used (redis, kafka, both)
    pub publisher_backend: String,
    
    /// Time taken to deliver the signal (milliseconds)
    pub delivery_latency_ms: u64,
    
    /// Number of retry attempts made
    pub retry_count: u32,
    
    /// Whether the emission was successful
    pub success: bool,
    
    /// Error message if emission failed
    pub error_message: Option<String>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl SignalEmissionEvent {
    /// Create a new successful signal emission event
    pub fn success(
        correlation_id: String,
        signal: TradingSignal,
        publisher_backend: String,
        delivery_latency_ms: u64,
        retry_count: u32,
    ) -> Self {
        Self {
            event_id: generate_event_id(),
            timestamp: current_timestamp_ms(),
            correlation_id,
            signal,
            publisher_backend,
            delivery_latency_ms,
            retry_count,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Create a new failed signal emission event
    pub fn failure(
        correlation_id: String,
        signal: TradingSignal,
        publisher_backend: String,
        delivery_latency_ms: u64,
        retry_count: u32,
        error_message: String,
    ) -> Self {
        Self {
            event_id: generate_event_id(),
            timestamp: current_timestamp_ms(),
            correlation_id,
            signal,
            publisher_backend,
            delivery_latency_ms,
            retry_count,
            success: false,
            error_message: Some(error_message),
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to the event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl AuditEvent for SignalEmissionEvent {
    fn event_id(&self) -> &str {
        &self.event_id
    }
    
    fn timestamp(&self) -> i64 {
        self.timestamp
    }
    
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }
    
    fn event_type(&self) -> &'static str {
        "signal_emission"
    }
    
    fn validate(&self) -> Result<(), SignalEmissionError> {
        if self.event_id.is_empty() {
            return Err(SignalEmissionError::validation("Event ID cannot be empty"));
        }
        
        if self.correlation_id.is_empty() {
            return Err(SignalEmissionError::validation("Correlation ID cannot be empty"));
        }
        
        if self.publisher_backend.is_empty() {
            return Err(SignalEmissionError::validation("Publisher backend cannot be empty"));
        }
        
        // Validate the embedded signal
        self.signal.validate().map_err(|e| {
            SignalEmissionError::validation(format!("Invalid signal in audit event: {}", e))
        })?;
        
        Ok(())
    }
}

/// Audit event for feature computation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureComputationEvent {
    /// Unique identifier for this audit event
    pub event_id: String,
    
    /// Timestamp when the event occurred (milliseconds since Unix epoch)
    pub timestamp: i64,
    
    /// Correlation ID linking this event to the signal lifecycle
    pub correlation_id: String,
    
    /// Symbol for which features were computed
    pub symbol: String,
    
    /// Names of features that were computed
    pub feature_names: Vec<String>,
    
    /// Time taken to compute features (milliseconds)
    pub computation_latency_ms: u64,
    
    /// Checksum of input data used for feature computation
    pub input_checksum: String,
    
    /// Checksum of computed feature output
    pub output_checksum: String,
    
    /// Whether feature validation passed
    pub validation_passed: bool,
    
    /// Data quality issues found during computation
    pub quality_issues: Vec<String>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl FeatureComputationEvent {
    /// Create a new feature computation event
    pub fn new(
        correlation_id: String,
        symbol: String,
        feature_names: Vec<String>,
        computation_latency_ms: u64,
        input_checksum: String,
        output_checksum: String,
        validation_passed: bool,
    ) -> Self {
        Self {
            event_id: generate_event_id(),
            timestamp: current_timestamp_ms(),
            correlation_id,
            symbol,
            feature_names,
            computation_latency_ms,
            input_checksum,
            output_checksum,
            validation_passed,
            quality_issues: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add a data quality issue
    pub fn with_quality_issue(mut self, issue: String) -> Self {
        self.quality_issues.push(issue);
        self
    }
    
    /// Add metadata to the event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl AuditEvent for FeatureComputationEvent {
    fn event_id(&self) -> &str {
        &self.event_id
    }
    
    fn timestamp(&self) -> i64 {
        self.timestamp
    }
    
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }
    
    fn event_type(&self) -> &'static str {
        "feature_computation"
    }
    
    fn validate(&self) -> Result<(), SignalEmissionError> {
        if self.event_id.is_empty() {
            return Err(SignalEmissionError::validation("Event ID cannot be empty"));
        }
        
        if self.correlation_id.is_empty() {
            return Err(SignalEmissionError::validation("Correlation ID cannot be empty"));
        }
        
        if self.symbol.is_empty() {
            return Err(SignalEmissionError::validation("Symbol cannot be empty"));
        }
        
        if self.feature_names.is_empty() {
            return Err(SignalEmissionError::validation("Feature names cannot be empty"));
        }
        
        if self.input_checksum.is_empty() {
            return Err(SignalEmissionError::validation("Input checksum cannot be empty"));
        }
        
        if self.output_checksum.is_empty() {
            return Err(SignalEmissionError::validation("Output checksum cannot be empty"));
        }
        
        Ok(())
    }
}

/// Audit event for signal validation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationErrorEvent {
    /// Unique identifier for this audit event
    pub event_id: String,
    
    /// Timestamp when the event occurred (milliseconds since Unix epoch)
    pub timestamp: i64,
    
    /// Correlation ID linking this event to the signal lifecycle
    pub correlation_id: String,
    
    /// Partial signal data that failed validation (may be incomplete)
    pub signal_partial: serde_json::Value,
    
    /// List of validation errors encountered
    pub validation_errors: Vec<ValidationErrorDetail>,
    
    /// Additional context about the validation failure
    pub error_context: String,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Detailed validation error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationErrorDetail {
    /// Field name that failed validation
    pub field: String,
    
    /// Expected value or format
    pub expected: String,
    
    /// Actual value that was provided
    pub actual: String,
    
    /// Error message
    pub message: String,
}

impl ValidationErrorEvent {
    /// Create a new validation error event
    pub fn new(
        correlation_id: String,
        signal_partial: serde_json::Value,
        validation_errors: Vec<ValidationErrorDetail>,
        error_context: String,
    ) -> Self {
        Self {
            event_id: generate_event_id(),
            timestamp: current_timestamp_ms(),
            correlation_id,
            signal_partial,
            validation_errors,
            error_context,
            metadata: HashMap::new(),
        }
    }
    
    /// Create from a ValidationError
    pub fn from_validation_error(
        correlation_id: String,
        signal_json: serde_json::Value,
        error: &ValidationError,
        context: String,
    ) -> Self {
        let detail = ValidationErrorDetail {
            field: error.category().to_string(),
            expected: "valid value".to_string(),
            actual: format!("{:?}", error),
            message: error.to_string(),
        };
        
        Self::new(correlation_id, signal_json, vec![detail], context)
    }
    
    /// Add metadata to the event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl AuditEvent for ValidationErrorEvent {
    fn event_id(&self) -> &str {
        &self.event_id
    }
    
    fn timestamp(&self) -> i64 {
        self.timestamp
    }
    
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }
    
    fn event_type(&self) -> &'static str {
        "validation_error"
    }
    
    fn validate(&self) -> Result<(), SignalEmissionError> {
        if self.event_id.is_empty() {
            return Err(SignalEmissionError::validation("Event ID cannot be empty"));
        }
        
        if self.correlation_id.is_empty() {
            return Err(SignalEmissionError::validation("Correlation ID cannot be empty"));
        }
        
        if self.validation_errors.is_empty() {
            return Err(SignalEmissionError::validation("Validation errors cannot be empty"));
        }
        
        Ok(())
    }
}

/// Audit event for publisher operations (connection, health checks, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublisherEvent {
    /// Unique identifier for this audit event
    pub event_id: String,
    
    /// Timestamp when the event occurred (milliseconds since Unix epoch)
    pub timestamp: i64,
    
    /// Correlation ID (may be empty for system-level events)
    pub correlation_id: Option<String>,
    
    /// Publisher backend (redis, kafka)
    pub publisher_backend: String,
    
    /// Type of publisher operation
    pub operation_type: PublisherOperationType,
    
    /// Whether the operation was successful
    pub success: bool,
    
    /// Error message if operation failed
    pub error_message: Option<String>,
    
    /// Operation latency in milliseconds
    pub operation_latency_ms: u64,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of publisher operations that can be audited
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PublisherOperationType {
    /// Connection establishment
    Connect,
    /// Connection health check
    HealthCheck,
    /// Signal publication
    Publish,
    /// Connection retry
    Retry,
    /// Circuit breaker state change
    CircuitBreakerStateChange,
    /// Buffer operation
    BufferOperation,
}

impl PublisherEvent {
    /// Create a new publisher event
    pub fn new(
        correlation_id: Option<String>,
        publisher_backend: String,
        operation_type: PublisherOperationType,
        success: bool,
        operation_latency_ms: u64,
    ) -> Self {
        Self {
            event_id: generate_event_id(),
            timestamp: current_timestamp_ms(),
            correlation_id,
            publisher_backend,
            operation_type,
            success,
            error_message: None,
            operation_latency_ms,
            metadata: HashMap::new(),
        }
    }
    
    /// Create a failed publisher event
    pub fn failure(
        correlation_id: Option<String>,
        publisher_backend: String,
        operation_type: PublisherOperationType,
        operation_latency_ms: u64,
        error_message: String,
    ) -> Self {
        Self {
            event_id: generate_event_id(),
            timestamp: current_timestamp_ms(),
            correlation_id,
            publisher_backend,
            operation_type,
            success: false,
            error_message: Some(error_message),
            operation_latency_ms,
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to the event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl AuditEvent for PublisherEvent {
    fn event_id(&self) -> &str {
        &self.event_id
    }
    
    fn timestamp(&self) -> i64 {
        self.timestamp
    }
    
    fn correlation_id(&self) -> &str {
        self.correlation_id.as_deref().unwrap_or("")
    }
    
    fn event_type(&self) -> &'static str {
        "publisher_operation"
    }
    
    fn validate(&self) -> Result<(), SignalEmissionError> {
        if self.event_id.is_empty() {
            return Err(SignalEmissionError::validation("Event ID cannot be empty"));
        }
        
        if self.publisher_backend.is_empty() {
            return Err(SignalEmissionError::validation("Publisher backend cannot be empty"));
        }
        
        Ok(())
    }
}

/// Audit event for HMM weight retrieval and fallback operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmWeightEvent {
    /// Unique identifier for this audit event
    pub event_id: String,
    
    /// Timestamp when the event occurred (milliseconds since Unix epoch)
    pub timestamp: i64,
    
    /// Correlation ID linking this event to the signal lifecycle
    pub correlation_id: String,
    
    /// Symbol for which weights were retrieved
    pub symbol: String,
    
    /// Whether weights were retrieved successfully
    pub success: bool,
    
    /// Whether fallback weights were used
    pub fallback_used: bool,
    
    /// HMM state probabilities if available
    pub state_probabilities: Option<Vec<f32>>,
    
    /// Retrieved fusion weights
    pub fusion_weights: Option<FusionWeights>,
    
    /// Time taken to retrieve weights (milliseconds)
    pub retrieval_latency_ms: u64,
    
    /// Error message if retrieval failed
    pub error_message: Option<String>,
    
    /// Cache hit/miss information
    pub cache_hit: Option<bool>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl HmmWeightEvent {
    /// Create a new successful HMM weight event
    pub fn success(
        correlation_id: String,
        symbol: String,
        state_probabilities: Option<Vec<f32>>,
        fusion_weights: FusionWeights,
        retrieval_latency_ms: u64,
        fallback_used: bool,
        cache_hit: Option<bool>,
    ) -> Self {
        Self {
            event_id: generate_event_id(),
            timestamp: current_timestamp_ms(),
            correlation_id,
            symbol,
            success: true,
            fallback_used,
            state_probabilities,
            fusion_weights: Some(fusion_weights),
            retrieval_latency_ms,
            error_message: None,
            cache_hit,
            metadata: HashMap::new(),
        }
    }
    
    /// Create a new failed HMM weight event
    pub fn failure(
        correlation_id: String,
        symbol: String,
        retrieval_latency_ms: u64,
        error_message: String,
        fallback_used: bool,
    ) -> Self {
        Self {
            event_id: generate_event_id(),
            timestamp: current_timestamp_ms(),
            correlation_id,
            symbol,
            success: false,
            fallback_used,
            state_probabilities: None,
            fusion_weights: None,
            retrieval_latency_ms,
            error_message: Some(error_message),
            cache_hit: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to the event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl AuditEvent for HmmWeightEvent {
    fn event_id(&self) -> &str {
        &self.event_id
    }
    
    fn timestamp(&self) -> i64 {
        self.timestamp
    }
    
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }
    
    fn event_type(&self) -> &'static str {
        "hmm_weight_retrieval"
    }
    
    fn validate(&self) -> Result<(), SignalEmissionError> {
        if self.event_id.is_empty() {
            return Err(SignalEmissionError::validation("Event ID cannot be empty"));
        }
        
        if self.correlation_id.is_empty() {
            return Err(SignalEmissionError::validation("Correlation ID cannot be empty"));
        }
        
        if self.symbol.is_empty() {
            return Err(SignalEmissionError::validation("Symbol cannot be empty"));
        }
        
        // Validate state probabilities if present
        if let Some(ref probs) = self.state_probabilities {
            for (i, &prob) in probs.iter().enumerate() {
                if !prob.is_finite() || prob < 0.0 || prob > 1.0 {
                    return Err(SignalEmissionError::validation(
                        format!("Invalid state probability at index {}: {}", i, prob)
                    ));
                }
            }
            
            let sum: f32 = probs.iter().sum();
            if (sum - 1.0).abs() > 0.01 {
                return Err(SignalEmissionError::validation(
                    format!("State probabilities should sum to 1.0, got: {}", sum)
                ));
            }
        }
        
        // Validate fusion weights if present
        if let Some(ref weights) = self.fusion_weights {
            weights.validate().map_err(|e| {
                SignalEmissionError::validation(format!("Invalid fusion weights: {}", e))
            })?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SignalSide, SignalComponents};
    
    #[test]
    fn test_correlation_id_generation() {
        let id1 = generate_correlation_id();
        let id2 = generate_correlation_id();
        
        assert_ne!(id1, id2);
        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
    }
    
    #[test]
    fn test_signal_emission_event_creation() {
        let signal = create_test_signal();
        
        let event = SignalEmissionEvent::success(
            "test-correlation".to_string(),
            signal,
            "redis".to_string(),
            50,
            0,
        );
        
        assert!(event.success);
        assert_eq!(event.publisher_backend, "redis");
        assert_eq!(event.delivery_latency_ms, 50);
        assert_eq!(event.retry_count, 0);
        assert!(event.error_message.is_none());
        assert!(event.validate().is_ok());
    }
    
    #[test]
    fn test_signal_emission_event_failure() {
        let signal = create_test_signal();
        
        let event = SignalEmissionEvent::failure(
            "test-correlation".to_string(),
            signal,
            "kafka".to_string(),
            100,
            3,
            "Connection timeout".to_string(),
        );
        
        assert!(!event.success);
        assert_eq!(event.publisher_backend, "kafka");
        assert_eq!(event.delivery_latency_ms, 100);
        assert_eq!(event.retry_count, 3);
        assert_eq!(event.error_message, Some("Connection timeout".to_string()));
        assert!(event.validate().is_ok());
    }
    
    #[test]
    fn test_feature_computation_event() {
        let event = FeatureComputationEvent::new(
            "test-correlation".to_string(),
            "BTCUSDT".to_string(),
            vec!["rsi".to_string(), "ma".to_string()],
            25,
            "input-checksum".to_string(),
            "output-checksum".to_string(),
            true,
        ).with_quality_issue("Missing data point".to_string());
        
        assert_eq!(event.symbol, "BTCUSDT");
        assert_eq!(event.feature_names.len(), 2);
        assert_eq!(event.computation_latency_ms, 25);
        assert!(event.validation_passed);
        assert_eq!(event.quality_issues.len(), 1);
        assert!(event.validate().is_ok());
    }
    
    #[test]
    fn test_validation_error_event() {
        let signal_json = serde_json::json!({
            "symbol": "btcusdt",
            "strength": 2.0
        });
        
        let validation_errors = vec![
            ValidationErrorDetail {
                field: "symbol".to_string(),
                expected: "uppercase".to_string(),
                actual: "btcusdt".to_string(),
                message: "Symbol must be uppercase".to_string(),
            },
            ValidationErrorDetail {
                field: "strength".to_string(),
                expected: "[-1.0, 1.0]".to_string(),
                actual: "2.0".to_string(),
                message: "Strength out of range".to_string(),
            },
        ];
        
        let event = ValidationErrorEvent::new(
            "test-correlation".to_string(),
            signal_json,
            validation_errors,
            "Signal validation failed".to_string(),
        );
        
        assert_eq!(event.validation_errors.len(), 2);
        assert_eq!(event.error_context, "Signal validation failed");
        assert!(event.validate().is_ok());
    }
    
    #[test]
    fn test_publisher_event() {
        let event = PublisherEvent::new(
            Some("test-correlation".to_string()),
            "redis".to_string(),
            PublisherOperationType::Publish,
            true,
            15,
        );
        
        assert!(event.success);
        assert_eq!(event.publisher_backend, "redis");
        assert_eq!(event.operation_latency_ms, 15);
        assert!(event.validate().is_ok());
    }
    
    #[test]
    fn test_hmm_weight_event_success() {
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let event = HmmWeightEvent::success(
            "test-correlation".to_string(),
            "BTCUSDT".to_string(),
            Some(vec![0.7, 0.3]),
            weights,
            30,
            false,
            Some(true),
        );
        
        assert!(event.success);
        assert!(!event.fallback_used);
        assert_eq!(event.cache_hit, Some(true));
        assert!(event.state_probabilities.is_some());
        assert!(event.fusion_weights.is_some());
        assert!(event.validate().is_ok());
    }
    
    #[test]
    fn test_hmm_weight_event_failure() {
        let event = HmmWeightEvent::failure(
            "test-correlation".to_string(),
            "BTCUSDT".to_string(),
            50,
            "HMM service unavailable".to_string(),
            true,
        );
        
        assert!(!event.success);
        assert!(event.fallback_used);
        assert!(event.state_probabilities.is_none());
        assert!(event.fusion_weights.is_none());
        assert_eq!(event.error_message, Some("HMM service unavailable".to_string()));
        assert!(event.validate().is_ok());
    }
    
    #[test]
    fn test_event_serialization() {
        let signal = create_test_signal();
        let event = SignalEmissionEvent::success(
            "test-correlation".to_string(),
            signal,
            "redis".to_string(),
            50,
            0,
        );
        
        let json = event.to_json().unwrap();
        let deserialized: SignalEmissionEvent = serde_json::from_str(&json).unwrap();
        
        assert_eq!(event.event_id, deserialized.event_id);
        assert_eq!(event.correlation_id, deserialized.correlation_id);
        assert_eq!(event.success, deserialized.success);
    }
    
    #[test]
    fn test_invalid_event_validation() {
        let mut event = SignalEmissionEvent::success(
            "test-correlation".to_string(),
            create_test_signal(),
            "redis".to_string(),
            50,
            0,
        );
        
        // Make event invalid
        event.event_id = String::new();
        assert!(event.validate().is_err());
        
        event.event_id = "valid-id".to_string();
        event.correlation_id = String::new();
        assert!(event.validate().is_err());
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