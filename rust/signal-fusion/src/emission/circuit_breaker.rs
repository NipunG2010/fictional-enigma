//! Circuit breaker implementation for signal emission resilience
//! 
//! This module provides a circuit breaker pattern to prevent cascade failures
//! during Redis/Kafka outages and improve system resilience.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, warn, info};

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed, allowing all requests
    Closed,
    
    /// Circuit is open, rejecting all requests
    Open,
    
    /// Circuit is half-open, allowing limited requests to test recovery
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of failures required to open the circuit (default: 5)
    pub failure_threshold: u32,
    
    /// Time to wait before transitioning from Open to HalfOpen (default: 30s)
    pub recovery_timeout: Duration,
    
    /// Number of successful requests required to close the circuit from HalfOpen (default: 3)
    pub success_threshold: u32,
    
    /// Time window for counting failures (default: 60s)
    pub failure_window: Duration,
    
    /// Maximum number of requests allowed in HalfOpen state (default: 5)
    pub half_open_max_requests: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            success_threshold: 3,
            failure_window: Duration::from_secs(60),
            half_open_max_requests: 5,
        }
    }
}

/// Circuit breaker implementation
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Current circuit state
    state: CircuitState,
    
    /// Configuration
    config: CircuitBreakerConfig,
    
    /// Failure count in current window
    failure_count: u32,
    
    /// Success count in HalfOpen state
    success_count: u32,
    
    /// Request count in HalfOpen state
    half_open_requests: u32,
    
    /// Timestamp of last failure
    last_failure_time: Option<Instant>,
    
    /// Timestamp when circuit was opened
    opened_at: Option<Instant>,
    
    /// Timestamp of first failure in current window
    failure_window_start: Option<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default configuration
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }
    
    /// Create a circuit breaker with custom configuration
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            state: CircuitState::Closed,
            config,
            failure_count: 0,
            success_count: 0,
            half_open_requests: 0,
            last_failure_time: None,
            opened_at: None,
            failure_window_start: None,
        }
    }
    
    /// Get the current circuit state
    pub fn state(&self) -> &CircuitState {
        &self.state
    }
    
    /// Get the current failure count
    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }
    
    /// Get the current success count (in HalfOpen state)
    pub fn success_count(&self) -> u32 {
        self.success_count
    }
    
    /// Check if a request is allowed through the circuit breaker
    pub fn is_request_allowed(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if recovery timeout has elapsed
                if let Some(opened_at) = self.opened_at {
                    if opened_at.elapsed() >= self.config.recovery_timeout {
                        debug!("Circuit breaker transitioning from Open to HalfOpen after recovery timeout");
                        self.transition_to_half_open();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited requests in HalfOpen state
                if self.half_open_requests < self.config.half_open_max_requests {
                    self.half_open_requests += 1;
                    true
                } else {
                    false
                }
            }
        }
    }
    
    /// Record a successful operation
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.reset_failure_count();
            }
            CircuitState::HalfOpen => {
                self.success_count += 1;
                debug!("Circuit breaker recorded success in HalfOpen state: {}/{}", 
                      self.success_count, self.config.success_threshold);
                
                // Check if we have enough successes to close the circuit
                if self.success_count >= self.config.success_threshold {
                    info!("Circuit breaker closing after {} successful requests", self.success_count);
                    self.transition_to_closed();
                }
            }
            CircuitState::Open => {
                // This shouldn't happen if is_request_allowed is used correctly
                warn!("Recorded success while circuit breaker is Open");
            }
        }
    }
    
    /// Record a failed operation
    pub fn record_failure(&mut self) {
        let now = Instant::now();
        self.last_failure_time = Some(now);
        
        match self.state {
            CircuitState::Closed => {
                // Check if we need to start a new failure window
                if let Some(window_start) = self.failure_window_start {
                    if window_start.elapsed() >= self.config.failure_window {
                        // Start new failure window
                        self.failure_window_start = Some(now);
                        self.failure_count = 1;
                    } else {
                        // Increment failure count in current window
                        self.failure_count += 1;
                    }
                } else {
                    // First failure, start the window
                    self.failure_window_start = Some(now);
                    self.failure_count = 1;
                }
                
                debug!("Circuit breaker recorded failure: {}/{} in window", 
                      self.failure_count, self.config.failure_threshold);
                
                // Check if we should open the circuit
                if self.failure_count >= self.config.failure_threshold {
                    warn!("Circuit breaker opening after {} failures", self.failure_count);
                    self.transition_to_open();
                }
            }
            CircuitState::HalfOpen => {
                warn!("Circuit breaker recorded failure in HalfOpen state, reopening");
                self.transition_to_open();
            }
            CircuitState::Open => {
                // Already open, just update the timestamp
                debug!("Circuit breaker recorded failure while Open");
            }
        }
    }
    
    /// Get time until recovery (if circuit is open)
    pub fn time_until_recovery(&self) -> Option<Duration> {
        if let (CircuitState::Open, Some(opened_at)) = (&self.state, self.opened_at) {
            let elapsed = opened_at.elapsed();
            if elapsed < self.config.recovery_timeout {
                Some(self.config.recovery_timeout - elapsed)
            } else {
                Some(Duration::ZERO)
            }
        } else {
            None
        }
    }
    
    /// Reset the circuit breaker to closed state
    pub fn reset(&mut self) {
        info!("Circuit breaker manually reset to Closed state");
        self.transition_to_closed();
    }
    
    /// Get circuit breaker metrics
    pub fn metrics(&self) -> CircuitBreakerMetrics {
        CircuitBreakerMetrics {
            state: self.state.clone(),
            failure_count: self.failure_count,
            success_count: self.success_count,
            half_open_requests: self.half_open_requests,
            last_failure_seconds_ago: self.last_failure_time.map(|t| t.elapsed().as_secs()),
            opened_seconds_ago: self.opened_at.map(|t| t.elapsed().as_secs()),
            time_until_recovery_seconds: self.time_until_recovery().map(|d| d.as_secs()),
        }
    }
    
    /// Transition to Closed state
    fn transition_to_closed(&mut self) {
        self.state = CircuitState::Closed;
        self.reset_failure_count();
        self.success_count = 0;
        self.half_open_requests = 0;
        self.opened_at = None;
    }
    
    /// Transition to Open state
    fn transition_to_open(&mut self) {
        self.state = CircuitState::Open;
        self.opened_at = Some(Instant::now());
        self.success_count = 0;
        self.half_open_requests = 0;
    }
    
    /// Transition to HalfOpen state
    fn transition_to_half_open(&mut self) {
        self.state = CircuitState::HalfOpen;
        self.success_count = 0;
        self.half_open_requests = 0;
    }
    
    /// Reset failure count and window
    fn reset_failure_count(&mut self) {
        self.failure_count = 0;
        self.failure_window_start = None;
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerMetrics {
    pub state: CircuitState,
    pub failure_count: u32,
    pub success_count: u32,
    pub half_open_requests: u32,
    pub last_failure_seconds_ago: Option<u64>,
    pub opened_seconds_ago: Option<u64>,
    pub time_until_recovery_seconds: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_circuit_breaker_creation() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), &CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.success_count(), 0);
    }
    
    #[test]
    fn test_circuit_breaker_closed_state() {
        let mut cb = CircuitBreaker::new();
        
        // Should allow requests when closed
        assert!(cb.is_request_allowed());
        
        // Record some successes
        cb.record_success();
        cb.record_success();
        assert_eq!(cb.state(), &CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
    }
    
    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let mut cb = CircuitBreaker::with_config(config);
        
        // Record failures up to threshold
        cb.record_failure();
        assert_eq!(cb.state(), &CircuitState::Closed);
        
        cb.record_failure();
        assert_eq!(cb.state(), &CircuitState::Closed);
        
        cb.record_failure();
        assert_eq!(cb.state(), &CircuitState::Open);
        
        // Should not allow requests when open
        assert!(!cb.is_request_allowed());
    }
    
    #[test]
    fn test_circuit_breaker_half_open_transition() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let mut cb = CircuitBreaker::with_config(config);
        
        // Open the circuit
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), &CircuitState::Open);
        
        // Wait for recovery timeout
        thread::sleep(Duration::from_millis(150));
        
        // Should transition to HalfOpen on next request
        assert!(cb.is_request_allowed());
        assert_eq!(cb.state(), &CircuitState::HalfOpen);
    }
    
    #[test]
    fn test_circuit_breaker_half_open_to_closed() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            recovery_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let mut cb = CircuitBreaker::with_config(config);
        
        // Open the circuit
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), &CircuitState::Open);
        
        // Wait and transition to HalfOpen
        thread::sleep(Duration::from_millis(100));
        assert!(cb.is_request_allowed());
        assert_eq!(cb.state(), &CircuitState::HalfOpen);
        
        // Record successes to close the circuit
        cb.record_success();
        assert_eq!(cb.state(), &CircuitState::HalfOpen);
        
        cb.record_success();
        assert_eq!(cb.state(), &CircuitState::Closed);
    }
    
    #[test]
    fn test_circuit_breaker_half_open_to_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let mut cb = CircuitBreaker::with_config(config);
        
        // Open the circuit
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), &CircuitState::Open);
        
        // Wait and transition to HalfOpen
        thread::sleep(Duration::from_millis(100));
        assert!(cb.is_request_allowed());
        assert_eq!(cb.state(), &CircuitState::HalfOpen);
        
        // Record failure to reopen the circuit
        cb.record_failure();
        assert_eq!(cb.state(), &CircuitState::Open);
    }
    
    #[test]
    fn test_circuit_breaker_failure_window() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            failure_window: Duration::from_millis(100),
            ..Default::default()
        };
        let mut cb = CircuitBreaker::with_config(config);
        
        // Record failures within window
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count(), 2);
        
        // Wait for window to expire
        thread::sleep(Duration::from_millis(150));
        
        // Next failure should start new window
        cb.record_failure();
        assert_eq!(cb.failure_count(), 1);
        assert_eq!(cb.state(), &CircuitState::Closed);
    }
    
    #[test]
    fn test_circuit_breaker_reset() {
        let mut cb = CircuitBreaker::new();
        
        // Open the circuit
        for _ in 0..5 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), &CircuitState::Open);
        
        // Reset should close the circuit
        cb.reset();
        assert_eq!(cb.state(), &CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
    }
    
    #[test]
    fn test_circuit_breaker_metrics() {
        let mut cb = CircuitBreaker::new();
        
        cb.record_failure();
        cb.record_failure();
        
        let metrics = cb.metrics();
        assert_eq!(metrics.state, CircuitState::Closed);
        assert_eq!(metrics.failure_count, 2);
        assert!(metrics.last_failure_seconds_ago.is_some());
    }
}