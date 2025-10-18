# Requirements Document

## Introduction

The HMM Microservice is a FastAPI-based service that provides Hidden Markov Model inference capabilities for the IMP trading system. This service calculates state probabilities and fusion weights based on trained HMM models, enabling regime-aware signal fusion in the Rust inference engine. The service must be highly available, performant, and integrate seamlessly with the existing system architecture.

## Glossary

- **HMM_Service**: The FastAPI microservice that provides HMM inference endpoints
- **State_Probability**: The probability distribution over HMM states given current observations
- **Fusion_Weight**: Per-state weight values used to combine trading signals based on current regime
- **Observation_Vector**: Input data containing [s_LDC, s_MR, s_TSMOM] signal values
- **Model_Artifact**: JSON file containing HMM parameters (A, mu, sigma, weights)
- **Health_Check**: Service endpoint that reports operational status and readiness
- **Cache_Layer**: In-memory storage for frequently accessed model parameters and computed results
- **Inference_Request**: HTTP request containing observation data for state probability calculation
- **Weight_Response**: HTTP response containing fusion weights for current market regime

## Requirements

### Requirement 1

**User Story:** As a Rust inference engine, I want to query HMM state probabilities, so that I can apply regime-aware fusion weights to trading signals.

#### Acceptance Criteria

1. WHEN the Rust inference engine sends an observation vector, THE HMM_Service SHALL return state probabilities within 20ms
2. THE HMM_Service SHALL accept observation vectors containing [s_LDC, s_MR, s_TSMOM] values
3. THE HMM_Service SHALL return normalized state probabilities that sum to 1.0
4. IF the observation vector contains invalid values, THEN THE HMM_Service SHALL return a validation error with specific details
5. THE HMM_Service SHALL log all inference requests with timestamps and input values for audit purposes

### Requirement 2

**User Story:** As a trading system operator, I want the HMM service to compute fusion weights, so that signals are properly weighted based on current market regime.

#### Acceptance Criteria

1. THE HMM_Service SHALL compute fusion weights using state probabilities and per-state weight matrices
2. WHEN state probabilities are calculated, THE HMM_Service SHALL return corresponding fusion weights for [LDC, MR, TSMOM] signals
3. THE HMM_Service SHALL ensure fusion weights are within valid ranges [-1.0, 1.0]
4. WHERE multiple HMM models are available, THE HMM_Service SHALL use the currently active model version
5. THE HMM_Service SHALL cache computed weights for identical observation vectors to improve performance

### Requirement 3

**User Story:** As a system administrator, I want health check endpoints, so that I can monitor service availability and performance.

#### Acceptance Criteria

1. THE HMM_Service SHALL provide a health check endpoint that returns service status within 5ms
2. THE HMM_Service SHALL report model loading status and last successful inference timestamp
3. WHEN the service is unhealthy, THE HMM_Service SHALL return appropriate HTTP status codes and error details
4. THE HMM_Service SHALL provide metrics endpoint exposing inference latency and request counts
5. THE HMM_Service SHALL implement readiness checks that verify model artifacts are loaded successfully

### Requirement 4

**User Story:** As a DevOps engineer, I want model management capabilities, so that I can deploy updated HMM models without service downtime.

#### Acceptance Criteria

1. THE HMM_Service SHALL load HMM model artifacts from MinIO storage on startup
2. THE HMM_Service SHALL support hot-reloading of model artifacts through API endpoints
3. WHEN a new model version is deployed, THE HMM_Service SHALL validate model parameters before activation
4. THE HMM_Service SHALL maintain backward compatibility with existing model artifact formats
5. IF model loading fails, THEN THE HMM_Service SHALL continue operating with the previous valid model

### Requirement 5

**User Story:** As a Rust inference engine, I want reliable service communication, so that signal generation continues even during temporary service issues.

#### Acceptance Criteria

1. THE HMM_Service SHALL implement connection pooling and keep-alive for HTTP connections
2. THE HMM_Service SHALL return appropriate HTTP status codes for different error conditions
3. WHEN the service is overloaded, THE HMM_Service SHALL implement request queuing with timeout handling
4. THE HMM_Service SHALL provide circuit breaker patterns for graceful degradation
5. THE HMM_Service SHALL support concurrent requests from multiple Rust inference instances

### Requirement 6

**User Story:** As a system operator, I want comprehensive monitoring and logging, so that I can troubleshoot issues and optimize performance.

#### Acceptance Criteria

1. THE HMM_Service SHALL log all API requests with request ID, timestamp, and processing duration
2. THE HMM_Service SHALL expose Prometheus metrics for request rates, latency percentiles, and error counts
3. THE HMM_Service SHALL implement structured logging with configurable log levels
4. WHEN errors occur, THE HMM_Service SHALL log detailed error information including stack traces
5. THE HMM_Service SHALL provide performance metrics for model inference and cache hit rates