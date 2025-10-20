//! HMM Microservice Client
//!
//! This module provides a Rust HTTP client for integrating with the HMM microservice.
//! It handles state probability calculation, fusion weight computation, and error handling
//! with fallback mechanisms for production use.

use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
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
    
    #[error("Circuit breaker open: {reason}")]
    CircuitBreakerOpen { reason: String },
    
    #[error("Max retries exceeded: {attempts} attempts failed - {last_error}")]
    MaxRetriesExceeded { attempts: usize, last_error: String },
}

impl HmmClientError {
    /// Classify error as transient (retryable) or permanent
    /// 
    /// Transient errors are temporary issues that may resolve with retry:
    /// - Network timeouts
    /// - Connection errors
    /// - 500 Internal Server Error
    /// - 503 Service Unavailable
    /// 
    /// Permanent errors indicate issues that won't resolve with retry:
    /// - 400 Bad Request
    /// - 422 Validation Error
    /// - Invalid JSON response
    pub fn is_transient(&self) -> bool {
        match self {
            // Network errors are generally transient
            HmmClientError::Network(e) => {
                // Check if it's a timeout or connection error
                e.is_timeout() || e.is_connect()
            }
            // Service unavailable is transient
            HmmClientError::ServiceUnavailable { status } => {
                matches!(status, &StatusCode::INTERNAL_SERVER_ERROR | &StatusCode::SERVICE_UNAVAILABLE | &StatusCode::BAD_GATEWAY | &StatusCode::GATEWAY_TIMEOUT)
            }
            // Model errors might be transient (e.g., model loading)
            HmmClientError::ModelError { .. } => true,
            // Timeout is transient
            HmmClientError::Timeout { .. } => true,
            // These are permanent errors
            HmmClientError::InvalidRequest { .. } => false,
            HmmClientError::ValidationError { .. } => false,
            HmmClientError::FallbackActivated { .. } => false,
            HmmClientError::CircuitBreakerOpen { .. } => false,
            HmmClientError::MaxRetriesExceeded { .. } => false,
        }
    }
    
    /// Get structured error context for debugging
    pub fn error_context(&self) -> ErrorContext {
        match self {
            HmmClientError::Network(e) => ErrorContext {
                error_type: "network".to_string(),
                is_transient: self.is_transient(),
                message: e.to_string(),
                details: format!("is_timeout: {}, is_connect: {}, is_request: {}", 
                                e.is_timeout(), e.is_connect(), e.is_request()),
                retry_recommended: self.is_transient(),
            },
            HmmClientError::ServiceUnavailable { status } => ErrorContext {
                error_type: "service_unavailable".to_string(),
                is_transient: self.is_transient(),
                message: format!("Service returned status: {}", status),
                details: format!("status_code: {}, canonical_reason: {:?}", 
                                status.as_u16(), status.canonical_reason()),
                retry_recommended: self.is_transient(),
            },
            HmmClientError::InvalidRequest { message } => ErrorContext {
                error_type: "invalid_request".to_string(),
                is_transient: false,
                message: message.clone(),
                details: "Request validation failed on server".to_string(),
                retry_recommended: false,
            },
            HmmClientError::ModelError { message } => ErrorContext {
                error_type: "model_error".to_string(),
                is_transient: true,
                message: message.clone(),
                details: "HMM model processing error".to_string(),
                retry_recommended: true,
            },
            HmmClientError::Timeout { duration } => ErrorContext {
                error_type: "timeout".to_string(),
                is_transient: true,
                message: format!("Request timed out after {:?}", duration),
                details: format!("timeout_duration_ms: {}", duration.as_millis()),
                retry_recommended: true,
            },
            HmmClientError::ValidationError { field, message } => ErrorContext {
                error_type: "validation_error".to_string(),
                is_transient: false,
                message: message.clone(),
                details: format!("field: {}", field),
                retry_recommended: false,
            },
            HmmClientError::FallbackActivated { reason } => ErrorContext {
                error_type: "fallback_activated".to_string(),
                is_transient: false,
                message: reason.clone(),
                details: "Using fallback weights due to service failure".to_string(),
                retry_recommended: false,
            },
            HmmClientError::CircuitBreakerOpen { reason } => ErrorContext {
                error_type: "circuit_breaker_open".to_string(),
                is_transient: true,
                message: reason.clone(),
                details: "Circuit breaker preventing requests to failing service".to_string(),
                retry_recommended: false, // Don't retry immediately, wait for circuit to close
            },
            HmmClientError::MaxRetriesExceeded { attempts, last_error } => ErrorContext {
                error_type: "max_retries_exceeded".to_string(),
                is_transient: false,
                message: format!("All {} retry attempts failed", attempts),
                details: format!("last_error: {}", last_error),
                retry_recommended: false,
            },
        }
    }
}

/// Structured error context for debugging and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub error_type: String,
    pub is_transient: bool,
    pub message: String,
    pub details: String,
    pub retry_recommended: bool,
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
#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitBreakerState {
    Closed,
    Open { opened_at: Instant },
    HalfOpen,
}

/// Circuit breaker metrics for monitoring
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CircuitBreakerMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub circuit_breaker_opens: u64,
    pub circuit_breaker_closes: u64,
    pub half_open_attempts: u64,
    pub rejected_requests: u64,
}

/// HMM Microservice HTTP Client
///
/// Provides methods for communicating with the HMM microservice including
/// state probability calculation, fusion weight computation, and health checks.
/// Includes comprehensive error handling, retry logic, and fallback mechanisms.
pub struct HmmClient {
    client: Client,
    config: HmmClientConfig,
    circuit_breaker_state: Cell<CircuitBreakerState>,
    failure_count: Cell<usize>,
    metrics: Cell<CircuitBreakerMetrics>,
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

        info!(
            "HMM client initialized with circuit breaker threshold={}, timeout={:?}",
            config.circuit_breaker_threshold, config.circuit_breaker_timeout
        );

        Ok(Self {
            client,
            config,
            circuit_breaker_state: Cell::new(CircuitBreakerState::Closed),
            failure_count: Cell::new(0),
            metrics: Cell::new(CircuitBreakerMetrics::default()),
        })
    }

    /// Check if circuit breaker allows requests
    fn is_circuit_breaker_open(&self) -> bool {
        let current_state = self.circuit_breaker_state.get();
        
        match current_state {
            CircuitBreakerState::Closed => {
                // Circuit is closed, allow requests
                false
            }
            CircuitBreakerState::Open { opened_at } => {
                // Check if timeout has elapsed for recovery attempt
                if opened_at.elapsed() > self.config.circuit_breaker_timeout {
                    info!(
                        "Circuit breaker transitioning from Open to Half-Open after {:?} timeout",
                        self.config.circuit_breaker_timeout
                    );
                    
                    // Transition to half-open state
                    self.circuit_breaker_state.set(CircuitBreakerState::HalfOpen);
                    
                    // Update metrics
                    let mut metrics = self.metrics.get();
                    metrics.half_open_attempts += 1;
                    self.metrics.set(metrics);
                    
                    // Allow the request to proceed
                    false
                } else {
                    // Circuit is still open, reject requests
                    debug!(
                        "Circuit breaker is Open, rejecting request (elapsed: {:?}, timeout: {:?})",
                        opened_at.elapsed(),
                        self.config.circuit_breaker_timeout
                    );
                    
                    // Update metrics
                    let mut metrics = self.metrics.get();
                    metrics.rejected_requests += 1;
                    self.metrics.set(metrics);
                    
                    true
                }
            }
            CircuitBreakerState::HalfOpen => {
                // In half-open state, allow a single test request
                debug!("Circuit breaker is Half-Open, allowing test request");
                false
            }
        }
    }

    /// Record a successful request
    fn record_success(&self) {
        let previous_state = self.circuit_breaker_state.get();
        
        // Reset failure count on success
        self.failure_count.set(0);
        
        // Update metrics
        let mut metrics = self.metrics.get();
        metrics.total_requests += 1;
        metrics.successful_requests += 1;
        self.metrics.set(metrics);
        
        // Handle state transitions
        match previous_state {
            CircuitBreakerState::HalfOpen => {
                // Success in half-open state closes the circuit breaker
                info!(
                    "Circuit breaker transitioning from Half-Open to Closed after successful request"
                );
                self.circuit_breaker_state.set(CircuitBreakerState::Closed);
                
                // Update metrics
                let mut metrics = self.metrics.get();
                metrics.circuit_breaker_closes += 1;
                self.metrics.set(metrics);
            }
            CircuitBreakerState::Closed => {
                // Already closed, just log success
                debug!("Request successful, circuit breaker remains Closed");
            }
            CircuitBreakerState::Open { .. } => {
                // This shouldn't happen, but log it
                warn!("Unexpected success while circuit breaker is Open");
            }
        }
    }

    /// Record a failed request
    fn record_failure(&self) {
        let previous_state = self.circuit_breaker_state.get();
        let new_count = self.failure_count.get() + 1;
        self.failure_count.set(new_count);
        
        // Update metrics
        let mut metrics = self.metrics.get();
        metrics.total_requests += 1;
        metrics.failed_requests += 1;
        self.metrics.set(metrics);
        
        // Handle state transitions based on current state
        match previous_state {
            CircuitBreakerState::Closed => {
                // Check if we've reached the threshold to open the circuit breaker
                if new_count >= self.config.circuit_breaker_threshold {
                    warn!(
                        "Circuit breaker transitioning from Closed to Open after {} consecutive failures (threshold: {})",
                        new_count,
                        self.config.circuit_breaker_threshold
                    );
                    
                    self.circuit_breaker_state.set(CircuitBreakerState::Open {
                        opened_at: Instant::now(),
                    });
                    
                    // Update metrics
                    let mut metrics = self.metrics.get();
                    metrics.circuit_breaker_opens += 1;
                    self.metrics.set(metrics);
                } else {
                    debug!(
                        "Request failed, failure count: {}/{} (circuit breaker remains Closed)",
                        new_count,
                        self.config.circuit_breaker_threshold
                    );
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Failure in half-open state reopens the circuit breaker
                warn!(
                    "Circuit breaker transitioning from Half-Open to Open after failed test request"
                );
                
                self.circuit_breaker_state.set(CircuitBreakerState::Open {
                    opened_at: Instant::now(),
                });
                
                // Update metrics
                let mut metrics = self.metrics.get();
                metrics.circuit_breaker_opens += 1;
                self.metrics.set(metrics);
            }
            CircuitBreakerState::Open { .. } => {
                // Already open, this shouldn't happen but log it
                debug!("Request failed while circuit breaker is already Open");
            }
        }
    }

    /// Perform HTTP request with retry logic and exponential backoff
    /// 
    /// Implements comprehensive error handling with:
    /// - Exponential backoff for transient errors
    /// - Error classification (transient vs permanent)
    /// - Structured error logging with context
    /// - Circuit breaker integration
    async fn request_with_retry<T>(&self, request_fn: impl Fn() -> reqwest::RequestBuilder) -> Result<T, HmmClientError>
    where
        T: for<'de> Deserialize<'de>,
    {
        // Check circuit breaker (Requirement 6.1)
        if self.is_circuit_breaker_open() {
            let error = HmmClientError::CircuitBreakerOpen {
                reason: "Circuit breaker is open, rejecting request".to_string(),
            };
            
            // Log with structured context (Requirement 6.4)
            let context = error.error_context();
            warn!(
                "Request rejected by circuit breaker: type={}, message={}, details={}",
                context.error_type, context.message, context.details
            );
            
            return Err(error);
        }

        let mut last_error: Option<HmmClientError> = None;
        let mut last_error_context: Option<ErrorContext> = None;

        for attempt in 0..=self.config.retry_attempts {
            // Log retry attempt
            if attempt > 0 {
                debug!(
                    "Retry attempt {}/{} after previous failure",
                    attempt, self.config.retry_attempts
                );
            }
            
            let response = match request_fn().send().await {
                Ok(response) => response,
                Err(e) => {
                    let error = HmmClientError::Network(e);
                    let context = error.error_context();
                    
                    // Enhanced error logging (Requirement 6.4)
                    warn!(
                        "Network error on attempt {}/{}: type={}, message={}, details={}, is_transient={}, retry_recommended={}",
                        attempt + 1,
                        self.config.retry_attempts + 1,
                        context.error_type,
                        context.message,
                        context.details,
                        context.is_transient,
                        context.retry_recommended
                    );
                    
                    last_error_context = Some(context.clone());
                    last_error = Some(error);
                    
                    // Only retry if error is transient and we have attempts left (Requirement 6.2)
                    if context.is_transient && attempt < self.config.retry_attempts {
                        // Exponential backoff: base_delay * 2^attempt (Requirement 6.1)
                        let backoff_delay = self.config.retry_delay * (1 << attempt);
                        debug!(
                            "Transient error detected, retrying after {:?} (exponential backoff)",
                            backoff_delay
                        );
                        tokio::time::sleep(backoff_delay).await;
                        continue;
                    } else {
                        break;
                    }
                }
            };

            // Handle response status codes with error classification (Requirement 6.2)
            match response.status() {
                StatusCode::OK => {
                    self.record_success();
                    
                    // Parse JSON response with error handling (Requirement 6.2)
                    return match response.json::<T>().await {
                        Ok(data) => {
                            debug!("Request successful, parsed response");
                            Ok(data)
                        }
                        Err(e) => {
                            let error = HmmClientError::Network(e);
                            let context = error.error_context();
                            
                            // Enhanced JSON parsing error logging (Requirement 6.2, 6.4)
                            error!(
                                "JSON parsing failed: type={}, message={}, details={}",
                                context.error_type, context.message, context.details
                            );
                            
                            self.record_failure();
                            Err(error)
                        }
                    };
                }
                
                // Permanent errors - don't retry (Requirement 6.2)
                StatusCode::BAD_REQUEST => {
                    let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    let error = HmmClientError::InvalidRequest {
                        message: error_text.clone(),
                    };
                    let context = error.error_context();
                    
                    error!(
                        "Invalid request (permanent error): type={}, message={}, details={}, retry_recommended={}",
                        context.error_type, context.message, context.details, context.retry_recommended
                    );
                    
                    self.record_failure();
                    return Err(error);
                }
                
                StatusCode::UNPROCESSABLE_ENTITY => {
                    let error_text = response.text().await.unwrap_or_else(|_| "Validation failed".to_string());
                    let error = HmmClientError::ValidationError {
                        field: "request".to_string(),
                        message: error_text.clone(),
                    };
                    let context = error.error_context();
                    
                    error!(
                        "Validation error (permanent): type={}, field={}, message={}, retry_recommended={}",
                        context.error_type, "request", context.message, context.retry_recommended
                    );
                    
                    self.record_failure();
                    return Err(error);
                }
                
                // Transient errors - retry with exponential backoff (Requirement 6.1, 6.2)
                StatusCode::INTERNAL_SERVER_ERROR | StatusCode::BAD_GATEWAY | StatusCode::GATEWAY_TIMEOUT => {
                    let error_text = response.text().await.unwrap_or_else(|_| "Server error".to_string());
                    let error = HmmClientError::ModelError {
                        message: error_text.clone(),
                    };
                    let context = error.error_context();
                    
                    warn!(
                        "Server error on attempt {}/{}: type={}, message={}, is_transient={}, retry_recommended={}",
                        attempt + 1,
                        self.config.retry_attempts + 1,
                        context.error_type,
                        context.message,
                        context.is_transient,
                        context.retry_recommended
                    );
                    
                    last_error_context = Some(context.clone());
                    last_error = Some(error);
                    
                    if context.is_transient && attempt < self.config.retry_attempts {
                        // Exponential backoff (Requirement 6.1)
                        let backoff_delay = self.config.retry_delay * (1 << attempt);
                        debug!(
                            "Transient server error, retrying after {:?} (exponential backoff)",
                            backoff_delay
                        );
                        tokio::time::sleep(backoff_delay).await;
                        continue;
                    } else {
                        break;
                    }
                }
                
                StatusCode::SERVICE_UNAVAILABLE => {
                    let error = HmmClientError::ServiceUnavailable {
                        status: response.status(),
                    };
                    let context = error.error_context();
                    
                    warn!(
                        "Service unavailable on attempt {}/{}: type={}, message={}, is_transient={}",
                        attempt + 1,
                        self.config.retry_attempts + 1,
                        context.error_type,
                        context.message,
                        context.is_transient
                    );
                    
                    last_error_context = Some(context.clone());
                    last_error = Some(error);
                    
                    if context.is_transient && attempt < self.config.retry_attempts {
                        // Exponential backoff (Requirement 6.1)
                        let backoff_delay = self.config.retry_delay * (1 << attempt);
                        debug!(
                            "Service unavailable, retrying after {:?} (exponential backoff)",
                            backoff_delay
                        );
                        tokio::time::sleep(backoff_delay).await;
                        continue;
                    } else {
                        break;
                    }
                }
                
                // Other status codes
                _ => {
                    let error = HmmClientError::ServiceUnavailable {
                        status: response.status(),
                    };
                    let context = error.error_context();
                    
                    warn!(
                        "Unexpected status code: type={}, message={}, details={}",
                        context.error_type, context.message, context.details
                    );
                    
                    self.record_failure();
                    return Err(error);
                }
            }
        }

        // All retry attempts exhausted (Requirement 6.1)
        self.record_failure();
        
        let final_error = if let Some(_err) = last_error {
            let last_msg = last_error_context
                .map(|c| c.message)
                .unwrap_or_else(|| "Unknown error".to_string());
            
            let max_retries_error = HmmClientError::MaxRetriesExceeded {
                attempts: self.config.retry_attempts + 1,
                last_error: last_msg,
            };
            
            // Enhanced error logging for max retries (Requirement 6.4)
            let context = max_retries_error.error_context();
            error!(
                "Max retries exceeded: type={}, attempts={}, message={}, details={}",
                context.error_type,
                self.config.retry_attempts + 1,
                context.message,
                context.details
            );
            
            max_retries_error
        } else {
            HmmClientError::ServiceUnavailable {
                status: StatusCode::INTERNAL_SERVER_ERROR,
            }
        };
        
        Err(final_error)
    }

    /// Get state probabilities from HMM service
    pub async fn get_state_probabilities(
        &self,
        observations: [f32; 3],
        request_id: Option<String>,
    ) -> Result<StateProbabilitiesResponse, HmmClientError> {
        let request = InferenceRequest {
            observations,
            timestamp: Some(chrono::Utc::now().timestamp()),
            request_id: request_id.clone(),
        };

        debug!(
            "Requesting state probabilities for observations: {:?}, request_id: {:?}",
            observations, request_id
        );

        let url = self
            .config
            .base_url
            .join("/inference/state-probabilities")
            .unwrap();

        let result: Result<StateProbabilitiesResponse, HmmClientError> = self
            .request_with_retry(|| self.client.post(url.clone()).json(&request))
            .await;

        match result {
            Ok(response) => {
                info!(
                    "State probabilities computed successfully: most_likely_state={}, confidence={:.3}, request_id={:?}",
                    response.most_likely_state, response.confidence, request_id
                );
                Ok(response)
            }
            Err(e) if self.config.enable_fallback => {
                // Enhanced fallback activation logging (Requirement 6.3)
                let context = e.error_context();
                warn!(
                    "Fallback activated for state probabilities: error_type={}, is_transient={}, message={}, details={}, request_id={:?}, observations={:?}",
                    context.error_type,
                    context.is_transient,
                    context.message,
                    context.details,
                    request_id,
                    observations
                );
                
                // Return uniform distribution as fallback
                let fallback_response = StateProbabilitiesResponse {
                    state_probabilities: vec![0.33, 0.33, 0.34],
                    most_likely_state: 0,
                    confidence: 0.34,
                    timestamp: chrono::Utc::now().timestamp(),
                    processing_time_ms: 0.0,
                };
                
                info!(
                    "Using fallback state probabilities: uniform distribution, request_id={:?}",
                    request_id
                );
                
                Ok(fallback_response)
            }
            Err(e) => {
                // Log error with full context (Requirement 6.4)
                let context = e.error_context();
                error!(
                    "State probabilities request failed (no fallback): error_type={}, message={}, details={}, request_id={:?}",
                    context.error_type, context.message, context.details, request_id
                );
                Err(e)
            }
        }
    }

    /// Get fusion weights from HMM service
    pub async fn get_fusion_weights(
        &self,
        observations: [f32; 3],
        request_id: Option<String>,
    ) -> Result<FusionWeightsResponse, HmmClientError> {
        let request = InferenceRequest {
            observations,
            timestamp: Some(chrono::Utc::now().timestamp()),
            request_id: request_id.clone(),
        };

        debug!(
            "Requesting fusion weights for observations: {:?}, request_id: {:?}",
            observations, request_id
        );

        let url = self
            .config
            .base_url
            .join("/inference/fusion-weights")
            .unwrap();

        let result: Result<FusionWeightsResponse, HmmClientError> = self
            .request_with_retry(|| self.client.post(url.clone()).json(&request))
            .await;

        match result {
            Ok(response) => {
                info!(
                    "Fusion weights computed successfully: LDC={:.3}, MR={:.3}, TSMOM={:.3}, most_likely_state={}, request_id={:?}",
                    response.weights.w_ldc,
                    response.weights.w_mr,
                    response.weights.w_tsmom,
                    response.most_likely_state,
                    request_id
                );
                Ok(response)
            }
            Err(e) if self.config.enable_fallback => {
                // Enhanced fallback activation logging (Requirement 6.3)
                let context = e.error_context();
                warn!(
                    "Fallback activated for fusion weights: error_type={}, is_transient={}, message={}, details={}, request_id={:?}, observations={:?}",
                    context.error_type,
                    context.is_transient,
                    context.message,
                    context.details,
                    request_id,
                    observations
                );
                
                // Return fallback weights
                let fallback_response = FusionWeightsResponse {
                    weights: self.config.fallback_weights.clone(),
                    state_probabilities: vec![0.33, 0.33, 0.34],
                    most_likely_state: 0,
                    timestamp: chrono::Utc::now().timestamp(),
                    processing_time_ms: 0.0,
                };
                
                info!(
                    "Using fallback fusion weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}, request_id={:?}",
                    fallback_response.weights.w_ldc,
                    fallback_response.weights.w_mr,
                    fallback_response.weights.w_tsmom,
                    request_id
                );
                
                Ok(fallback_response)
            }
            Err(e) => {
                // Log error with full context (Requirement 6.4)
                let context = e.error_context();
                error!(
                    "Fusion weights request failed (no fallback): error_type={}, message={}, details={}, request_id={:?}",
                    context.error_type, context.message, context.details, request_id
                );
                Err(e)
            }
        }
    }

    /// Get complete prediction from HMM service
    pub async fn predict(
        &self,
        observations: [f32; 3],
        request_id: Option<String>,
    ) -> Result<PredictionResponse, HmmClientError> {
        let request = InferenceRequest {
            observations,
            timestamp: Some(chrono::Utc::now().timestamp()),
            request_id: request_id.clone(),
        };

        debug!(
            "Requesting complete prediction for observations: {:?}, request_id: {:?}",
            observations, request_id
        );

        let url = self.config.base_url.join("/inference/predict").unwrap();

        let result: Result<PredictionResponse, HmmClientError> = self
            .request_with_retry(|| self.client.post(url.clone()).json(&request))
            .await;

        match result {
            Ok(response) => {
                info!(
                    "Complete prediction computed successfully: state={}, confidence={:.3}, weights=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}], model_version={}, request_id={:?}",
                    response.most_likely_state,
                    response.confidence,
                    response.fusion_weights.w_ldc,
                    response.fusion_weights.w_mr,
                    response.fusion_weights.w_tsmom,
                    response.model_version,
                    request_id
                );
                Ok(response)
            }
            Err(e) if self.config.enable_fallback => {
                // Enhanced fallback activation logging (Requirement 6.3)
                let context = e.error_context();
                warn!(
                    "Fallback activated for complete prediction: error_type={}, is_transient={}, message={}, details={}, request_id={:?}, observations={:?}",
                    context.error_type,
                    context.is_transient,
                    context.message,
                    context.details,
                    request_id,
                    observations
                );
                
                // Return fallback prediction
                let fallback_response = PredictionResponse {
                    state_probabilities: vec![0.33, 0.33, 0.34],
                    most_likely_state: 0,
                    confidence: 0.34,
                    fusion_weights: self.config.fallback_weights.clone(),
                    timestamp: chrono::Utc::now().timestamp(),
                    processing_time_ms: 0.0,
                    model_version: "fallback".to_string(),
                    request_id: request_id.clone(),
                };
                
                info!(
                    "Using fallback prediction: weights=[LDC:{:.3}, MR:{:.3}, TSMOM:{:.3}], model_version=fallback, request_id={:?}",
                    fallback_response.fusion_weights.w_ldc,
                    fallback_response.fusion_weights.w_mr,
                    fallback_response.fusion_weights.w_tsmom,
                    request_id
                );
                
                Ok(fallback_response)
            }
            Err(e) => {
                // Log error with full context (Requirement 6.4)
                let context = e.error_context();
                error!(
                    "Complete prediction request failed (no fallback): error_type={}, message={}, details={}, request_id={:?}",
                    context.error_type, context.message, context.details, request_id
                );
                Err(e)
            }
        }
    }

    /// Check service health
    pub async fn health_check(&self) -> Result<HealthResponse, HmmClientError> {
        let url = self.config.base_url.join("/health").unwrap();

        self.request_with_retry(|| self.client.get(url.clone()))
            .await
    }

    /// Check service readiness
    pub async fn readiness_check(&self) -> Result<ReadinessResponse, HmmClientError> {
        let url = self.config.base_url.join("/health/ready").unwrap();

        self.request_with_retry(|| self.client.get(url.clone()))
            .await
    }

    /// Get current model information
    pub async fn get_model_info(&self) -> Result<ModelInfoResponse, HmmClientError> {
        let url = self.config.base_url.join("/models/current").unwrap();

        self.request_with_retry(|| self.client.get(url.clone()))
            .await
    }

    /// Reload current model
    pub async fn reload_model(&self) -> Result<serde_json::Value, HmmClientError> {
        let url = self.config.base_url.join("/models/reload").unwrap();

        self.request_with_retry(|| self.client.post(url.clone()))
            .await
    }

    /// Get circuit breaker status
    pub fn get_circuit_breaker_status(&self) -> (String, usize) {
        let state = match self.circuit_breaker_state.get() {
            CircuitBreakerState::Closed => "closed",
            CircuitBreakerState::Open { .. } => "open",
            CircuitBreakerState::HalfOpen => "half-open",
        };
        (state.to_string(), self.failure_count.get())
    }

    /// Get detailed circuit breaker metrics
    pub fn get_circuit_breaker_metrics(&self) -> CircuitBreakerMetrics {
        self.metrics.get()
    }

    /// Reset circuit breaker state and metrics (useful for testing)
    #[cfg(test)]
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker_state.set(CircuitBreakerState::Closed);
        self.failure_count.set(0);
        self.metrics.set(CircuitBreakerMetrics::default());
        info!("Circuit breaker reset to initial state");
    }
}

/// High-level integration helper for signal fusion workflow
pub struct HmmIntegration {
    client: HmmClient,
    cache: crate::WeightCache,
    request_counter: u64,
    last_cleanup: Instant,
    cleanup_interval: Duration,
    metrics: crate::metrics::MetricsCollector,
}

impl HmmIntegration {
    /// Create new HMM integration with default configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: HmmClient::new()?,
            cache: crate::WeightCache::default(),
            request_counter: 0,
            last_cleanup: Instant::now(),
            cleanup_interval: Duration::from_secs(10),
            metrics: crate::metrics::MetricsCollector::new(),
        })
    }

    /// Create new HMM integration with custom client configuration
    pub fn with_config(config: HmmClientConfig) -> Result<Self> {
        Ok(Self {
            client: HmmClient::with_config(config)?,
            cache: crate::WeightCache::default(),
            request_counter: 0,
            last_cleanup: Instant::now(),
            cleanup_interval: Duration::from_secs(10),
            metrics: crate::metrics::MetricsCollector::new(),
        })
    }

    /// Create new HMM integration with custom client and cache configuration
    pub fn with_config_and_cache(
        config: HmmClientConfig,
        cache_ttl: Duration,
        cache_max_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            client: HmmClient::with_config(config)?,
            cache: crate::WeightCache::new(cache_ttl, cache_max_size),
            request_counter: 0,
            last_cleanup: Instant::now(),
            cleanup_interval: Duration::from_secs(10),
            metrics: crate::metrics::MetricsCollector::new(),
        })
    }

    /// Perform periodic cache cleanup if interval has elapsed
    fn maybe_cleanup_cache(&mut self) {
        if self.last_cleanup.elapsed() >= self.cleanup_interval {
            debug!("Performing periodic cache cleanup");
            self.cache.evict_expired();
            self.last_cleanup = Instant::now();
        }
    }

    /// Get fusion weights for signal components with automatic error handling and caching
    pub async fn get_fusion_weights_for_signals(
        &mut self,
        signal_components: &SignalComponents,
    ) -> Result<FusionWeights> {
        let observations = [
            signal_components.s_ldc,
            signal_components.s_mr,
            signal_components.s_tsmom,
        ];

        // Perform periodic cache cleanup
        self.maybe_cleanup_cache();

        // Try cache first
        if let Some(cached_weights) = self.cache.get(&observations) {
            debug!(
                "Cache hit for observations: {:?}, weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}",
                observations, cached_weights.w_ldc, cached_weights.w_mr, cached_weights.w_tsmom
            );
            return Ok(cached_weights);
        }

        // Cache miss - fetch from service
        debug!("Cache miss for observations: {:?}, fetching from HMM service", observations);
        
        self.request_counter += 1;
        let request_id = format!("req_{}", self.request_counter);
        
        // Track request start time for metrics
        let start_time = Instant::now();

        match self
            .client
            .get_fusion_weights(observations, Some(request_id.clone()))
            .await
        {
            Ok(response) => {
                // Record successful request metrics
                let duration = start_time.elapsed();
                self.metrics.record_success(duration);
                self.metrics.clear_fallback_active();
                
                info!(
                    "HMM fusion weights computed: LDC={:.3}, MR={:.3}, TSMOM={:.3}, duration={:?}, request_id={}",
                    response.weights.w_ldc, response.weights.w_mr, response.weights.w_tsmom, duration, request_id
                );
                
                // Insert into cache for future requests
                self.cache.insert(observations, response.weights.clone());
                debug!("Cached weights for observations: {:?}, request_id={}", observations, request_id);
                
                Ok(response.weights)
            }
            Err(e) => {
                // Record failed request metrics
                let duration = start_time.elapsed();
                let context = e.error_context();
                self.metrics.record_failure(duration, &context.error_type);
                
                // Log error with full context (Requirement 6.4)
                error!(
                    "HMM service error: error_type={}, is_transient={}, message={}, details={}, retry_recommended={}, duration={:?}, request_id={}",
                    context.error_type,
                    context.is_transient,
                    context.message,
                    context.details,
                    context.retry_recommended,
                    duration,
                    request_id
                );
                
                // Use fallback weights if enabled (Requirement 6.3)
                if self.client.config.enable_fallback {
                    let fallback_weights = self.client.config.fallback_weights.clone();
                    
                    // Record fallback activation metrics
                    self.metrics.record_fallback(&context.error_type);
                    
                    warn!(
                        "Using fallback weights due to error: LDC={:.3}, MR={:.3}, TSMOM={:.3}, error_type={}, request_id={}",
                        fallback_weights.w_ldc,
                        fallback_weights.w_mr,
                        fallback_weights.w_tsmom,
                        context.error_type,
                        request_id
                    );
                    
                    // Cache fallback weights to avoid repeated service calls
                    self.cache.insert(observations, fallback_weights.clone());
                    debug!("Cached fallback weights for observations: {:?}, request_id={}", observations, request_id);
                    
                    Ok(fallback_weights)
                } else {
                    // No fallback enabled, propagate error
                    error!(
                        "Fallback disabled, propagating error: error_type={}, request_id={}",
                        context.error_type, request_id
                    );
                    Err(anyhow::anyhow!("HMM service error: {}", e))
                }
            }
        }
    }

    /// Check if HMM service is healthy and ready
    pub async fn is_service_ready(&self) -> bool {
        match self.client.readiness_check().await {
            Ok(response) => response.ready && response.model_loaded,
            Err(e) => {
                debug!("HMM service readiness check failed: {}", e);
                false
            }
        }
    }

    /// Get service status information
    pub async fn get_service_status(&self) -> Result<(bool, Option<String>)> {
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

    /// Get cache statistics for monitoring
    pub fn get_cache_stats(&self) -> crate::CacheStats {
        self.cache.get_stats()
    }

    /// Manually trigger cache cleanup (evict expired entries)
    pub fn cleanup_cache(&mut self) {
        self.cache.evict_expired();
        self.last_cleanup = Instant::now();
    }

    /// Clear all cache entries
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get comprehensive metrics for monitoring
    ///
    /// Returns all metrics including:
    /// - Request metrics (count, duration, errors)
    /// - Cache metrics (hits, misses, size, evictions)
    /// - Circuit breaker metrics (state, transitions)
    /// - Fallback metrics (activations, reasons)
    pub fn get_metrics(&self) -> crate::metrics::HmmIntegrationMetrics {
        crate::metrics::HmmIntegrationMetrics {
            requests: self.metrics.get_request_metrics(),
            cache: self.cache.get_stats(),
            circuit_breaker: self.client.get_circuit_breaker_metrics(),
            fallback: self.metrics.get_fallback_metrics(),
            timestamp: chrono::Utc::now().timestamp(),
            uptime_seconds: self.metrics.get_uptime_seconds(),
        }
    }

    /// Export metrics in the specified format
    ///
    /// # Arguments
    /// * `format` - Export format (Json or Prometheus)
    ///
    /// # Returns
    /// * Formatted metrics string
    pub fn export_metrics(&self, format: crate::metrics::MetricsFormat) -> Result<String, serde_json::Error> {
        let metrics = self.get_metrics();
        crate::metrics::export_metrics(&metrics, format)
    }

    /// Get metrics collector for advanced usage
    pub fn metrics_collector(&self) -> &crate::metrics::MetricsCollector {
        &self.metrics
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
    fn test_hmm_integration_with_cache() {
        let integration = HmmIntegration::new().unwrap();
        
        // Verify cache is initialized
        let stats = integration.get_cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.size, 0);
    }

    #[test]
    fn test_hmm_integration_with_custom_cache() {
        let config = HmmClientConfig::default();
        let integration = HmmIntegration::with_config_and_cache(
            config,
            Duration::from_secs(30),
            500,
        ).unwrap();
        
        // Verify cache is initialized
        let stats = integration.get_cache_stats();
        assert_eq!(stats.size, 0);
    }

    #[test]
    fn test_hmm_integration_cache_operations() {
        let mut integration = HmmIntegration::new().unwrap();
        
        // Manually insert into cache
        let observations = [0.5, 0.3, 0.2];
        let weights = FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        };
        integration.cache.insert(observations, weights.clone());
        
        // Verify cache size
        let stats = integration.get_cache_stats();
        assert_eq!(stats.size, 1);
        
        // Clear cache
        integration.clear_cache();
        let stats = integration.get_cache_stats();
        assert_eq!(stats.size, 0);
    }

    #[test]
    fn test_hmm_integration_manual_cleanup() {
        let mut integration = HmmIntegration::new().unwrap();
        
        // Insert some entries
        let weights = FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        };
        integration.cache.insert([0.1, 0.1, 0.1], weights.clone());
        integration.cache.insert([0.2, 0.2, 0.2], weights.clone());
        
        assert_eq!(integration.cache.size(), 2);
        
        // Manual cleanup (won't remove non-expired entries)
        integration.cleanup_cache();
        assert_eq!(integration.cache.size(), 2);
    }

    #[test]
    fn test_circuit_breaker_state_transitions() {
        let client = HmmClient::new().unwrap();
        
        // Initially closed
        assert!(!client.is_circuit_breaker_open());
        let (state, count) = client.get_circuit_breaker_status();
        assert_eq!(state, "closed");
        assert_eq!(count, 0);
        
        // Record failures to open circuit breaker (threshold is 5)
        for i in 0..5 {
            client.record_failure();
            let (state, count) = client.get_circuit_breaker_status();
            if i < 4 {
                assert_eq!(state, "closed", "Should remain closed until threshold");
                assert_eq!(count, i + 1);
            } else {
                assert_eq!(state, "open", "Should open at threshold");
                assert_eq!(count, 5);
            }
        }
        
        // Should be open now
        assert!(client.is_circuit_breaker_open());
        
        // Verify metrics
        let metrics = client.get_circuit_breaker_metrics();
        assert_eq!(metrics.failed_requests, 5);
        assert_eq!(metrics.circuit_breaker_opens, 1);
    }

    #[test]
    fn test_circuit_breaker_half_open_success() {
        let config = HmmClientConfig {
            circuit_breaker_threshold: 2,
            circuit_breaker_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let client = HmmClient::with_config(config).unwrap();
        
        // Open the circuit breaker
        client.record_failure();
        client.record_failure();
        
        let (state, _) = client.get_circuit_breaker_status();
        assert_eq!(state, "open");
        
        // Wait for timeout to transition to half-open
        std::thread::sleep(Duration::from_millis(150));
        
        // Check should transition to half-open
        assert!(!client.is_circuit_breaker_open());
        let (state, _) = client.get_circuit_breaker_status();
        assert_eq!(state, "half-open");
        
        // Success in half-open should close the circuit
        client.record_success();
        let (state, count) = client.get_circuit_breaker_status();
        assert_eq!(state, "closed");
        assert_eq!(count, 0);
        
        // Verify metrics
        let metrics = client.get_circuit_breaker_metrics();
        assert_eq!(metrics.circuit_breaker_opens, 1);
        assert_eq!(metrics.circuit_breaker_closes, 1);
        assert_eq!(metrics.half_open_attempts, 1);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure() {
        let config = HmmClientConfig {
            circuit_breaker_threshold: 2,
            circuit_breaker_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let client = HmmClient::with_config(config).unwrap();
        
        // Open the circuit breaker
        client.record_failure();
        client.record_failure();
        
        let (state, _) = client.get_circuit_breaker_status();
        assert_eq!(state, "open");
        
        // Wait for timeout to transition to half-open
        std::thread::sleep(Duration::from_millis(150));
        
        // Check should transition to half-open
        assert!(!client.is_circuit_breaker_open());
        let (state, _) = client.get_circuit_breaker_status();
        assert_eq!(state, "half-open");
        
        // Failure in half-open should reopen the circuit
        client.record_failure();
        let (state, _) = client.get_circuit_breaker_status();
        assert_eq!(state, "open");
        
        // Verify metrics
        let metrics = client.get_circuit_breaker_metrics();
        assert_eq!(metrics.circuit_breaker_opens, 2); // Initial open + reopen
        assert_eq!(metrics.half_open_attempts, 1);
    }

    #[test]
    fn test_circuit_breaker_metrics() {
        let client = HmmClient::new().unwrap();
        
        // Record some successes and failures
        client.record_success();
        client.record_success();
        client.record_failure();
        client.record_success();
        
        let metrics = client.get_circuit_breaker_metrics();
        assert_eq!(metrics.total_requests, 4);
        assert_eq!(metrics.successful_requests, 3);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.circuit_breaker_opens, 0);
    }

    #[test]
    fn test_circuit_breaker_rejected_requests() {
        let config = HmmClientConfig {
            circuit_breaker_threshold: 2,
            circuit_breaker_timeout: Duration::from_secs(10), // Long timeout
            ..Default::default()
        };
        let client = HmmClient::with_config(config).unwrap();
        
        // Open the circuit breaker
        client.record_failure();
        client.record_failure();
        
        // Multiple checks while open should increment rejected count
        assert!(client.is_circuit_breaker_open());
        assert!(client.is_circuit_breaker_open());
        assert!(client.is_circuit_breaker_open());
        
        let metrics = client.get_circuit_breaker_metrics();
        assert_eq!(metrics.rejected_requests, 3);
    }
}