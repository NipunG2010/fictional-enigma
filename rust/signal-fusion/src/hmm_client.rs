//! HMM Microservice Client
//!
//! This module provides a Rust HTTP client for integrating with the HMM microservice.
//! It handles state probability calculation, fusion weight computation, and error handling
//! with fallback mechanisms for production use.

use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, error, info, warn};
use url::Url;

use crate::{FusionWeights, SignalComponents};

/// Errors that can occur during HMM service communication
#[derive(Error, Debug)]
pub enum HmmClientError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Service unavailable: {status}")]
    ServiceUnavailable { status: StatusCode },
    
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },
    
    #[error("Model error: {message}")]
    ModelError { message: String },
    
    #[error("Timeout after {duration:?}")]
    Timeout { duration: Duration },
    
    #[error("Validation error: {field} - {message}")]
    ValidationError { field: String, message: String },
    
    #[error("Fallback activated: {reason}")]
    FallbackActivated { reason: String },
}

/// Request payload for HMM inference endpoints
#[derive(Debug, Clone, Serialize)]
pub struct InferenceRequest {
    pub observations: [f32; 3], // [s_ldc, s_mr, s_tsmom]
    pub timestamp: Option<i64>,
    pub request_id: Option<String>,
}

/// Response from state probabilities endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct StateProbabilitiesResponse {
    pub state_probabilities: Vec<f32>,
    pub most_likely_state: usize,
    pub confidence: f32,
    pub timestamp: i64,
    pub processing_time_ms: f32,
}

/// Response from fusion weights endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct FusionWeightsResponse {
    pub weights: FusionWeights,
    pub state_probabilities: Vec<f32>,
    pub most_likely_state: usize,
    pub timestamp: i64,
    pub processing_time_ms: f32,
}

/// Complete prediction response
#[derive(Debug, Clone, Deserialize)]
pub struct PredictionResponse {
    pub state_probabilities: Vec<f32>,
    pub most_likely_state: usize,
    pub confidence: f32,
    pub fusion_weights: FusionWeights,
    pub timestamp: i64,
    pub processing_time_ms: f32,
    pub model_version: String,
    pub request_id: Option<String>,
}

/// Health check response
#[derive(Debug, Clone, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: i64,
    pub uptime: Option<u64>,
}

/// Readiness check response
#[derive(Debug, Clone, Deserialize)]
pub struct ReadinessResponse {
    pub ready: bool,
    pub model_loaded: bool,
    pub last_inference: Option<i64>,
}

/// Model information response
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfoResponse {
    pub loaded: bool,
    pub version: Option<String>,
    pub experiment_id: Option<String>,
    pub n_states: Option<usize>,
    pub library: Option<String>,
    pub load_time: Option<f64>,
    pub has_fusion_weights: Option<bool>,
    pub has_fallback: Option<bool>,
}

/// Configuration for HMM client
#[derive(Debug, Clone)]
pub struct HmmClientConfig {
    pub base_url: Url,
    pub timeout: Duration,
    pub retry_attempts: usize,
    pub retry_delay: Duration,
    pub enable_fallback: bool,
    pub fallback_weights: FusionWeights,
    pub circuit_breaker_threshold: usize,
    pub circuit_breaker_timeout: Duration,
}

impl Default for HmmClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8000".parse().unwrap(),
            timeout: Duration::from_millis(5000),
            retry_attempts: 3,
            retry_delay: Duration::from_millis(100),
            enable_fallback: true,
            fallback_weights: FusionWeights {
                w_ldc: 0.33,
                w_mr: 0.33,
                w_tsmom: 0.34,
            },
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(30),
        }
    }
}

/// Circuit breaker state for handling service failures
#[derive(Debug, Clone)]
enum CircuitBreakerState {
    Closed,
    Open { opened_at: Instant },
    HalfOpen,
}

/// HMM Microservice HTTP Client
///
/// Provides methods for communicating with the HMM microservice including
/// state probability calculation, fusion weight computation, and health checks.
/// Includes comprehensive error handling, retry logic, and fallback mechanisms.
pub struct HmmClient {
    client: Client,
    config: HmmClientConfig,
    circuit_breaker_state: CircuitBreakerState,
    failure_count: usize,
}

impl HmmClient {
    /// Create a new HMM client with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(HmmClientConfig::default())
    }

    /// Create a new HMM client with custom configuration
    pub fn with_config(config: HmmClientConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            config,
            circuit_breaker_state: CircuitBreakerState::Closed,
            failure_count: 0,
        })
    }

    /// Check if circuit breaker allows requests
    fn is_circuit_breaker_open(&mut self) -> bool {
        match &self.circuit_breaker_state {
            CircuitBreakerState::Closed => false,
            CircuitBreakerState::Open { opened_at } => {
                if opened_at.elapsed() > self.config.circuit_breaker_timeout {
                    debug!("Circuit breaker transitioning to half-open");
                    self.circuit_breaker_state = CircuitBreakerState::HalfOpen;
                    false
                } else {
                    true
                }
            }
            CircuitBreakerState::HalfOpen => false,
        }
    }

    /// Record a successful request
    fn record_success(&mut self) {
        self.failure_count = 0;
        if matches!(self.circuit_breaker_state, CircuitBreakerState::HalfOpen) {
            debug!("Circuit breaker closing after successful request");
            self.circuit_breaker_state = CircuitBreakerState::Closed;
        }
    }

    /// Record a failed request
    fn record_failure(&mut self) {
        self.failure_count += 1;
        if self.failure_count >= self.config.circuit_breaker_threshold {
            warn!(
                "Circuit breaker opening after {} failures",
                self.failure_count
            );
            self.circuit_breaker_state = CircuitBreakerState::Open {
                opened_at: Instant::now(),
            };
        }
    }

    /// Perform HTTP request with retry logic
    async fn request_with_retry<T>(&mut self, request_fn: impl Fn() -> reqwest::RequestBuilder) -> Result<T, HmmClientError>
    where
        T: for<'de> Deserialize<'de>,
    {
        // Check circuit breaker
        if self.is_circuit_breaker_open() {
            return Err(HmmClientError::ServiceUnavailable {
                status: StatusCode::SERVICE_UNAVAILABLE,
            });
        }

        let mut last_error = None;

        for attempt in 0..=self.config.retry_attempts {
            let response = match request_fn().send().await {
                Ok(response) => response,
                Err(e) => {
                    last_error = Some(HmmClientError::Network(e));
                    if attempt < self.config.retry_attempts {
                        debug!("Request attempt {} failed, retrying...", attempt + 1);
                        tokio::time::sleep(self.config.retry_delay * (attempt as u32 + 1)).await;
                        continue;
                    } else {
                        break;
                    }
                }
            };

            match response.status() {
                StatusCode::OK => {
                    self.record_success();
                    return response
                        .json::<T>()
                        .await
                        .map_err(HmmClientError::Network);
                }
                StatusCode::BAD_REQUEST => {
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(HmmClientError::InvalidRequest {
                        message: error_text,
                    });
                }
                StatusCode::UNPROCESSABLE_ENTITY => {
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(HmmClientError::ValidationError {
                        field: "request".to_string(),
                        message: error_text,
                    });
                }
                StatusCode::SERVICE_UNAVAILABLE => {
                    return Err(HmmClientError::ServiceUnavailable {
                        status: response.status(),
                    });
                }
                StatusCode::INTERNAL_SERVER_ERROR => {
                    let error_text = response.text().await.unwrap_or_default();
                    last_error = Some(HmmClientError::ModelError {
                        message: error_text,
                    });
                    if attempt < self.config.retry_attempts {
                        debug!("Server error on attempt {}, retrying...", attempt + 1);
                        tokio::time::sleep(self.config.retry_delay * (attempt as u32 + 1)).await;
                        continue;
                    }
                }
                _ => {
                    return Err(HmmClientError::ServiceUnavailable {
                        status: response.status(),
                    });
                }
            }
        }

        self.record_failure();
        Err(last_error.unwrap_or(HmmClientError::ServiceUnavailable {
            status: StatusCode::INTERNAL_SERVER_ERROR,
        }))
    }

    /// Get state probabilities from HMM service
    pub async fn get_state_probabilities(
        &mut self,
        observations: [f32; 3],
        request_id: Option<String>,
    ) -> Result<StateProbabilitiesResponse, HmmClientError> {
        let request = InferenceRequest {
            observations,
            timestamp: Some(chrono::Utc::now().timestamp()),
            request_id: request_id.clone(),
        };

        debug!(
            "Requesting state probabilities for observations: {:?}",
            observations
        );

        let url = self
            .config
            .base_url
            .join("/inference/state-probabilities")
            .unwrap();

        let result = self
            .request_with_retry(|| self.client.post(url.clone()).json(&request))
            .await;

        match result {
            Ok(response) => {
                debug!("State probabilities computed successfully");
                Ok(response)
            }
            Err(e) if self.config.enable_fallback => {
                warn!("State probabilities failed, using fallback: {}", e);
                // Return uniform distribution as fallback
                Ok(StateProbabilitiesResponse {
                    state_probabilities: vec![0.33, 0.33, 0.34],
                    most_likely_state: 0,
                    confidence: 0.34,
                    timestamp: chrono::Utc::now().timestamp(),
                    processing_time_ms: 0.0,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Get fusion weights from HMM service
    pub async fn get_fusion_weights(
        &mut self,
        observations: [f32; 3],
        request_id: Option<String>,
    ) -> Result<FusionWeightsResponse, HmmClientError> {
        let request = InferenceRequest {
            observations,
            timestamp: Some(chrono::Utc::now().timestamp()),
            request_id: request_id.clone(),
        };

        debug!(
            "Requesting fusion weights for observations: {:?}",
            observations
        );

        let url = self
            .config
            .base_url
            .join("/inference/fusion-weights")
            .unwrap();

        let result = self
            .request_with_retry(|| self.client.post(url.clone()).json(&request))
            .await;

        match result {
            Ok(response) => {
                debug!("Fusion weights computed successfully");
                Ok(response)
            }
            Err(e) if self.config.enable_fallback => {
                warn!("Fusion weights failed, using fallback: {}", e);
                // Return fallback weights
                Ok(FusionWeightsResponse {
                    weights: self.config.fallback_weights.clone(),
                    state_probabilities: vec![0.33, 0.33, 0.34],
                    most_likely_state: 0,
                    timestamp: chrono::Utc::now().timestamp(),
                    processing_time_ms: 0.0,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Get complete prediction from HMM service
    pub async fn predict(
        &mut self,
        observations: [f32; 3],
        request_id: Option<String>,
    ) -> Result<PredictionResponse, HmmClientError> {
        let request = InferenceRequest {
            observations,
            timestamp: Some(chrono::Utc::now().timestamp()),
            request_id: request_id.clone(),
        };

        debug!("Requesting complete prediction for observations: {:?}", observations);

        let url = self.config.base_url.join("/inference/predict").unwrap();

        let result = self
            .request_with_retry(|| self.client.post(url.clone()).json(&request))
            .await;

        match result {
            Ok(response) => {
                debug!("Complete prediction computed successfully");
                Ok(response)
            }
            Err(e) if self.config.enable_fallback => {
                warn!("Complete prediction failed, using fallback: {}", e);
                // Return fallback prediction
                Ok(PredictionResponse {
                    state_probabilities: vec![0.33, 0.33, 0.34],
                    most_likely_state: 0,
                    confidence: 0.34,
                    fusion_weights: self.config.fallback_weights.clone(),
                    timestamp: chrono::Utc::now().timestamp(),
                    processing_time_ms: 0.0,
                    model_version: "fallback".to_string(),
                    request_id,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Check service health
    pub async fn health_check(&mut self) -> Result<HealthResponse, HmmClientError> {
        let url = self.config.base_url.join("/health").unwrap();

        self.request_with_retry(|| self.client.get(url.clone()))
            .await
    }

    /// Check service readiness
    pub async fn readiness_check(&mut self) -> Result<ReadinessResponse, HmmClientError> {
        let url = self.config.base_url.join("/health/ready").unwrap();

        self.request_with_retry(|| self.client.get(url.clone()))
            .await
    }

    /// Get current model information
    pub async fn get_model_info(&mut self) -> Result<ModelInfoResponse, HmmClientError> {
        let url = self.config.base_url.join("/models/current").unwrap();

        self.request_with_retry(|| self.client.get(url.clone()))
            .await
    }

    /// Reload current model
    pub async fn reload_model(&mut self) -> Result<serde_json::Value, HmmClientError> {
        let url = self.config.base_url.join("/models/reload").unwrap();

        self.request_with_retry(|| self.client.post(url.clone()))
            .await
    }

    /// Get circuit breaker status
    pub fn get_circuit_breaker_status(&self) -> (String, usize) {
        let state = match &self.circuit_breaker_state {
            CircuitBreakerState::Closed => "closed",
            CircuitBreakerState::Open { .. } => "open",
            CircuitBreakerState::HalfOpen => "half-open",
        };
        (state.to_string(), self.failure_count)
    }
}

/// High-level integration helper for signal fusion workflow
pub struct HmmIntegration {
    client: HmmClient,
    request_counter: u64,
}

impl HmmIntegration {
    /// Create new HMM integration with default configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: HmmClient::new()?,
            request_counter: 0,
        })
    }

    /// Create new HMM integration with custom client configuration
    pub fn with_config(config: HmmClientConfig) -> Result<Self> {
        Ok(Self {
            client: HmmClient::with_config(config)?,
            request_counter: 0,
        })
    }

    /// Get fusion weights for signal components with automatic error handling
    pub async fn get_fusion_weights_for_signals(
        &mut self,
        signal_components: &SignalComponents,
    ) -> Result<FusionWeights> {
        self.request_counter += 1;
        let request_id = format!("req_{}", self.request_counter);

        let observations = [
            signal_components.s_ldc,
            signal_components.s_mr,
            signal_components.s_tsmom,
        ];

        match self
            .client
            .get_fusion_weights(observations, Some(request_id))
            .await
        {
            Ok(response) => {
                info!(
                    "HMM fusion weights computed: LDC={:.3}, MR={:.3}, TSMOM={:.3}",
                    response.weights.w_ldc, response.weights.w_mr, response.weights.w_tsmom
                );
                Ok(response.weights)
            }
            Err(HmmClientError::FallbackActivated { reason }) => {
                warn!("Using fallback weights: {}", reason);
                Ok(self.client.config.fallback_weights.clone())
            }
            Err(e) => {
                error!("HMM service error: {}", e);
                // Return fallback weights on any error
                Ok(self.client.config.fallback_weights.clone())
            }
        }
    }

    /// Check if HMM service is healthy and ready
    pub async fn is_service_ready(&mut self) -> bool {
        match self.client.readiness_check().await {
            Ok(response) => response.ready && response.model_loaded,
            Err(e) => {
                debug!("HMM service readiness check failed: {}", e);
                false
            }
        }
    }

    /// Get service status information
    pub async fn get_service_status(&mut self) -> Result<(bool, Option<String>)> {
        match self.client.get_model_info().await {
            Ok(info) => Ok((info.loaded, info.version)),
            Err(e) => {
                warn!("Failed to get service status: {}", e);
                Ok((false, None))
            }
        }
    }

    /// Get circuit breaker status
    pub fn get_circuit_breaker_status(&self) -> (String, usize) {
        self.client.get_circuit_breaker_status()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_client_config_default() {
        let config = HmmClientConfig::default();
        assert_eq!(config.base_url.as_str(), "http://localhost:8000/");
        assert_eq!(config.timeout, Duration::from_millis(5000));
        assert_eq!(config.retry_attempts, 3);
        assert!(config.enable_fallback);
    }

    #[test]
    fn test_inference_request_serialization() {
        let request = InferenceRequest {
            observations: [0.1, -0.05, 0.08],
            timestamp: Some(1234567890),
            request_id: Some("test_123".to_string()),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("observations"));
        assert!(json.contains("timestamp"));
        assert!(json.contains("request_id"));
    }

    #[tokio::test]
    async fn test_hmm_integration_creation() {
        // This test just verifies the integration can be created
        // Actual network tests would require a running HMM service
        let integration = HmmIntegration::new();
        assert!(integration.is_ok());
    }

    #[test]
    fn test_circuit_breaker_state_transitions() {
        let mut client = HmmClient::new().unwrap();
        
        // Initially closed
        assert!(!client.is_circuit_breaker_open());
        
        // Record failures to open circuit breaker
        for _ in 0..5 {
            client.record_failure();
        }
        
        // Should be open now
        assert!(client.is_circuit_breaker_open());
        
        // Record success to close it
        client.record_success();
        // Note: In half-open state, success closes the circuit breaker
    }
}