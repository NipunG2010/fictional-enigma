# Requirements Document

## Introduction

The Rust HMM Integration component enables the Rust inference engine to communicate with the HMM microservice for regime-aware signal fusion. This component provides HTTP client functionality, weight caching, fallback mechanisms, and signal fusion logic to combine LDC, MR, and TSMOM signals based on current market regime.

## Glossary

- **HMM_Client**: Rust HTTP client that communicates with the HMM microservice
- **Weight_Cache**: In-memory cache storing recently fetched fusion weights
- **Fusion_Engine**: Component that combines trading signals using regime-aware weights
- **Circuit_Breaker**: Pattern that prevents cascading failures by detecting service issues
- **Fallback_Weights**: Static default weights used when HMM service is unavailable
- **Signal_Vector**: Trading signals [s_LDC, s_MR, s_TSMOM] used as HMM observations
- **Fused_Signal**: Final combined signal output from weighted signal fusion
- **Connection_Pool**: Reusable HTTP connections for efficient service communication

## Requirements

### Requirement 1

**User Story:** As a Rust inference engine, I want to fetch fusion weights from the HMM service, so that I can apply regime-aware weighting to trading signals.

#### Acceptance Criteria

1. THE HMM_Client SHALL send HTTP POST requests to the HMM service with signal observations
2. WHEN the HMM service responds, THE HMM_Client SHALL parse fusion weights within 5ms
3. THE HMM_Client SHALL validate that received weights are within valid ranges [-1.0, 1.0]
4. THE HMM_Client SHALL include request timeouts of 50ms to prevent blocking
5. THE HMM_Client SHALL use connection pooling to minimize connection overhead

### Requirement 2

**User Story:** As a trading system, I want to cache fusion weights, so that I can reduce latency and HMM service load.

#### Acceptance Criteria

1. THE Weight_Cache SHALL store fusion weights with 60-second TTL
2. WHEN identical signal vectors are observed, THE Weight_Cache SHALL return cached weights without service calls
3. THE Weight_Cache SHALL implement thread-safe access for concurrent requests
4. THE Weight_Cache SHALL evict expired entries automatically
5. THE Weight_Cache SHALL track cache hit rates for monitoring

### Requirement 3

**User Story:** As a trading system, I want fallback to static weights, so that signal generation continues during HMM service failures.

#### Acceptance Criteria

1. WHEN the HMM service is unreachable, THE Fusion_Engine SHALL use pre-configured static weights
2. THE Fusion_Engine SHALL log all fallback activations with timestamps and reasons
3. THE Fallback_Weights SHALL be configurable via environment variables or config files
4. THE Fusion_Engine SHALL attempt service recovery after fallback activation
5. THE Fusion_Engine SHALL emit metrics indicating fallback mode status

### Requirement 4

**User Story:** As a system operator, I want circuit breaker protection, so that the system handles service degradation gracefully.

#### Acceptance Criteria

1. THE Circuit_Breaker SHALL open after 3 consecutive HMM service failures
2. WHILE the circuit is open, THE Circuit_Breaker SHALL use fallback weights without attempting service calls
3. THE Circuit_Breaker SHALL attempt service recovery after 30 seconds in open state
4. WHEN service recovery succeeds, THE Circuit_Breaker SHALL close and resume normal operation
5. THE Circuit_Breaker SHALL track state transitions for monitoring

### Requirement 5

**User Story:** As a trading system, I want to fuse signals using regime-aware weights, so that final signals adapt to market conditions.

#### Acceptance Criteria

1. THE Fusion_Engine SHALL compute fused signals as weighted sum of [s_LDC, s_MR, s_TSMOM]
2. THE Fusion_Engine SHALL normalize weights to ensure proper signal scaling
3. THE Fusion_Engine SHALL validate input signals are within expected ranges
4. THE Fusion_Engine SHALL complete fusion computation within 5ms
5. THE Fusion_Engine SHALL log fusion operations with input signals and output values

### Requirement 6

**User Story:** As a system operator, I want comprehensive error handling, so that I can diagnose and resolve integration issues.

#### Acceptance Criteria

1. THE HMM_Client SHALL handle network errors with appropriate retry logic
2. WHEN JSON parsing fails, THE HMM_Client SHALL log detailed error information
3. THE HMM_Client SHALL distinguish between transient and permanent failures
4. THE HMM_Client SHALL emit structured error logs with request context
5. THE HMM_Client SHALL provide error metrics for monitoring and alerting
