//! Failure Simulator for End-to-End Testing
//! 
//! This module provides infrastructure for simulating various failure conditions
//! in external services and system components to test resilience and fallback mechanisms.

use crate::{Result, TestFrameworkError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Types of failures that can be simulated
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FailureType {
    /// HMM service unavailable
    HmmServiceUnavailable,
    /// HMM service slow response
    HmmServiceSlowResponse,
    /// HMM service invalid response
    HmmServiceInvalidResponse,
    /// Redis connection failure
    RedisConnectionFailure,
    /// Redis timeout
    RedisTimeout,
    /// Kafka connection failure
    KafkaConnectionFailure,
    /// Kafka publish failure
    KafkaPublishFailure,
    /// Network partition
    NetworkPartition,
    /// Data corruption
    DataCorruption,
    /// Memory pressure
    MemoryPressure,
    /// CPU overload
    CpuOverload,
}

/// Recovery behavior after failure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecoveryBehavior {
    /// Immediate recovery
    Immediate,
    /// Gradual recovery over time
    Gradual { recovery_time: Duration },
    /// Manual recovery (requires explicit reset)
    Manual,
    /// Intermittent failures
    Intermittent { failure_rate: f64 },
}

/// Context for a simulated failure
#[derive(Debug, Clone)]
pub struct FailureContext {
    /// Type of failure being simulated
    pub failure_type: FailureType,
    /// Duration of the failure
    pub duration: Duration,
    /// Recovery behavior
    pub recovery_behavior: RecoveryBehavior,
    /// When the failure started
    pub start_time: Instant,
    /// Additional failure parameters
    pub parameters: HashMap<String, String>,
    /// Whether the failure is currently active
    pub active: bool,
}

impl FailureContext {
    /// Create a new failure context
    pub fn new(
        failure_type: FailureType,
        duration: Duration,
        recovery_behavior: RecoveryBehavior,
    ) -> Self {
        Self {
            failure_type,
            duration,
            recovery_behavior,
            start_time: Instant::now(),
            parameters: HashMap::new(),
            active: true,
        }
    }
    
    /// Add a parameter to the failure context
    pub fn with_parameter(mut self, key: String, value: String) -> Self {
        self.parameters.insert(key, value);
        self
    }
    
    /// Check if the failure should still be active
    pub fn is_active(&self) -> bool {
        if !self.active {
            return false;
        }
        
        let elapsed = self.start_time.elapsed();
        
        match &self.recovery_behavior {
            RecoveryBehavior::Immediate => false,
            RecoveryBehavior::Gradual { recovery_time } => elapsed < self.duration + *recovery_time,
            RecoveryBehavior::Manual => true,
            RecoveryBehavior::Intermittent { failure_rate } => {
                // Simulate intermittent failures based on time and rate
                let cycle_time = Duration::from_secs(10); // 10-second cycles
                let cycle_position = (elapsed.as_millis() % cycle_time.as_millis()) as f64 / cycle_time.as_millis() as f64;
                cycle_position < *failure_rate
            }
        }
    }
    
    /// Get the current failure intensity (0.0 = no failure, 1.0 = full failure)
    pub fn get_intensity(&self) -> f64 {
        if !self.is_active() {
            return 0.0;
        }
        
        let elapsed = self.start_time.elapsed();
        
        match &self.recovery_behavior {
            RecoveryBehavior::Immediate => 0.0,
            RecoveryBehavior::Gradual { recovery_time } => {
                if elapsed < self.duration {
                    1.0 // Full failure during failure period
                } else {
                    // Gradual recovery
                    let recovery_elapsed = elapsed - self.duration;
                    let recovery_progress = recovery_elapsed.as_secs_f64() / recovery_time.as_secs_f64();
                    (1.0 - recovery_progress).max(0.0)
                }
            }
            RecoveryBehavior::Manual => 1.0,
            RecoveryBehavior::Intermittent { failure_rate } => {
                if self.is_active() { *failure_rate } else { 0.0 }
            }
        }
    }
    
    /// Manually stop the failure
    pub fn stop(&mut self) {
        self.active = false;
    }
}

/// Mock HMM service for testing failure scenarios
#[derive(Debug)]
pub struct MockHMMService {
    /// Current failure context
    failure_context: Arc<RwLock<Option<FailureContext>>>,
    /// Response delay simulation
    base_delay: Duration,
    /// Call count for testing
    call_count: Arc<Mutex<u64>>,
}

impl MockHMMService {
    /// Create a new mock HMM service
    pub fn new() -> Self {
        Self {
            failure_context: Arc::new(RwLock::new(None)),
            base_delay: Duration::from_millis(20), // Normal HMM response time
            call_count: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Set a failure context for the service
    pub async fn set_failure(&self, context: FailureContext) {
        let mut failure = self.failure_context.write().await;
        *failure = Some(context);
        info!("MockHMMService: Failure context set: {:?}", failure.as_ref().unwrap().failure_type);
    }
    
    /// Clear any active failure
    pub async fn clear_failure(&self) {
        let mut failure = self.failure_context.write().await;
        if let Some(ref mut ctx) = failure.as_mut() {
            ctx.stop();
        }
        *failure = None;
        info!("MockHMMService: Failure context cleared");
    }
    
    /// Get current failure status
    pub async fn get_failure_status(&self) -> Option<FailureContext> {
        let failure = self.failure_context.read().await;
        failure.clone()
    }
    
    /// Simulate getting HMM weights (main service method)
    pub async fn get_weights(&self, symbol: &str) -> Result<HashMap<String, f64>> {
        // Increment call count
        {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
        }
        
        // Check for active failures
        let failure_context = {
            let failure = self.failure_context.read().await;
            failure.clone()
        };
        
        if let Some(context) = failure_context {
            if context.is_active() {
                let intensity = context.get_intensity();
                
                match context.failure_type {
                    FailureType::HmmServiceUnavailable => {
                        debug!("MockHMMService: Simulating service unavailable for {}", symbol);
                        return Err(TestFrameworkError::SystemInitError(
                            "HMM service unavailable".to_string()
                        ).into());
                    }
                    FailureType::HmmServiceSlowResponse => {
                        let delay = self.base_delay.mul_f64(1.0 + intensity * 10.0); // Up to 10x slower
                        debug!("MockHMMService: Simulating slow response ({:?}) for {}", delay, symbol);
                        tokio::time::sleep(delay).await;
                    }
                    FailureType::HmmServiceInvalidResponse => {
                        debug!("MockHMMService: Simulating invalid response for {}", symbol);
                        return Ok(HashMap::from([
                            ("w_ldc".to_string(), f64::NAN),
                            ("w_mr".to_string(), f64::INFINITY),
                            ("w_tsmom".to_string(), -1.5), // Invalid weight
                        ]));
                    }
                    _ => {
                        // Other failure types don't affect HMM service directly
                        tokio::time::sleep(self.base_delay).await;
                    }
                }
            }
        } else {
            // Normal operation
            tokio::time::sleep(self.base_delay).await;
        }
        
        // Return normal weights
        Ok(HashMap::from([
            ("w_ldc".to_string(), 0.5),
            ("w_mr".to_string(), 0.3),
            ("w_tsmom".to_string(), 0.2),
        ]))
    }
    
    /// Get the number of calls made to this service
    pub fn get_call_count(&self) -> u64 {
        *self.call_count.lock().unwrap()
    }
    
    /// Reset call count
    pub fn reset_call_count(&self) {
        let mut count = self.call_count.lock().unwrap();
        *count = 0;
    }
}

impl Default for MockHMMService {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock Redis service for testing connection failures
#[derive(Debug)]
pub struct MockRedisService {
    /// Current failure context
    failure_context: Arc<RwLock<Option<FailureContext>>>,
    /// Simulated data store
    data_store: Arc<RwLock<HashMap<String, String>>>,
    /// Connection status
    connected: Arc<RwLock<bool>>,
    /// Operation count
    operation_count: Arc<Mutex<u64>>,
}

impl MockRedisService {
    /// Create a new mock Redis service
    pub fn new() -> Self {
        Self {
            failure_context: Arc::new(RwLock::new(None)),
            data_store: Arc::new(RwLock::new(HashMap::new())),
            connected: Arc::new(RwLock::new(true)),
            operation_count: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Set a failure context for the service
    pub async fn set_failure(&self, context: FailureContext) {
        let mut failure = self.failure_context.write().await;
        *failure = Some(context);
        
        // Update connection status based on failure type
        if let Some(ref ctx) = *failure {
            match ctx.failure_type {
                FailureType::RedisConnectionFailure => {
                    let mut connected = self.connected.write().await;
                    *connected = false;
                }
                _ => {}
            }
        }
        
        info!("MockRedisService: Failure context set: {:?}", failure.as_ref().unwrap().failure_type);
    }
    
    /// Clear any active failure
    pub async fn clear_failure(&self) {
        let mut failure = self.failure_context.write().await;
        if let Some(ref mut ctx) = failure.as_mut() {
            ctx.stop();
        }
        *failure = None;
        
        // Restore connection
        let mut connected = self.connected.write().await;
        *connected = true;
        
        info!("MockRedisService: Failure context cleared");
    }
    
    /// Check if Redis is connected
    pub async fn is_connected(&self) -> bool {
        let connected = self.connected.read().await;
        *connected
    }
    
    /// Simulate Redis SET operation
    pub async fn set(&self, key: &str, value: &str) -> Result<()> {
        self.increment_operation_count();
        
        if !self.is_connected().await {
            return Err(TestFrameworkError::SystemInitError(
                "Redis connection failed".to_string()
            ).into());
        }
        
        // Check for active failures
        let failure_context = {
            let failure = self.failure_context.read().await;
            failure.clone()
        };
        
        if let Some(context) = failure_context {
            if context.is_active() {
                match context.failure_type {
                    FailureType::RedisConnectionFailure => {
                        debug!("MockRedisService: Simulating connection failure for SET {}", key);
                        return Err(TestFrameworkError::SystemInitError(
                            "Redis connection lost".to_string()
                        ).into());
                    }
                    FailureType::RedisTimeout => {
                        debug!("MockRedisService: Simulating timeout for SET {}", key);
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        return Err(TestFrameworkError::ExecutionTimeout { timeout_ms: 5000 }.into());
                    }
                    _ => {}
                }
            }
        }
        
        // Normal operation
        let mut store = self.data_store.write().await;
        store.insert(key.to_string(), value.to_string());
        debug!("MockRedisService: SET {} = {}", key, value);
        
        Ok(())
    }
    
    /// Simulate Redis GET operation
    pub async fn get(&self, key: &str) -> Result<Option<String>> {
        self.increment_operation_count();
        
        if !self.is_connected().await {
            return Err(TestFrameworkError::SystemInitError(
                "Redis connection failed".to_string()
            ).into());
        }
        
        // Check for active failures
        let failure_context = {
            let failure = self.failure_context.read().await;
            failure.clone()
        };
        
        if let Some(context) = failure_context {
            if context.is_active() {
                match context.failure_type {
                    FailureType::RedisConnectionFailure => {
                        debug!("MockRedisService: Simulating connection failure for GET {}", key);
                        return Err(TestFrameworkError::SystemInitError(
                            "Redis connection lost".to_string()
                        ).into());
                    }
                    FailureType::RedisTimeout => {
                        debug!("MockRedisService: Simulating timeout for GET {}", key);
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        return Err(TestFrameworkError::ExecutionTimeout { timeout_ms: 5000 }.into());
                    }
                    _ => {}
                }
            }
        }
        
        // Normal operation
        let store = self.data_store.read().await;
        let result = store.get(key).cloned();
        debug!("MockRedisService: GET {} = {:?}", key, result);
        
        Ok(result)
    }
    
    /// Get operation count
    pub fn get_operation_count(&self) -> u64 {
        *self.operation_count.lock().unwrap()
    }
    
    /// Reset operation count
    pub fn reset_operation_count(&self) {
        let mut count = self.operation_count.lock().unwrap();
        *count = 0;
    }
    
    /// Clear all stored data
    pub async fn clear_data(&self) {
        let mut store = self.data_store.write().await;
        store.clear();
    }
    
    fn increment_operation_count(&self) {
        let mut count = self.operation_count.lock().unwrap();
        *count += 1;
    }
}

impl Default for MockRedisService {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock Kafka service for testing message publishing failures
#[derive(Debug)]
pub struct MockKafkaService {
    /// Current failure context
    failure_context: Arc<RwLock<Option<FailureContext>>>,
    /// Published messages
    published_messages: Arc<RwLock<Vec<(String, String)>>>, // (topic, message)
    /// Connection status
    connected: Arc<RwLock<bool>>,
    /// Operation count
    operation_count: Arc<Mutex<u64>>,
}

impl MockKafkaService {
    /// Create a new mock Kafka service
    pub fn new() -> Self {
        Self {
            failure_context: Arc::new(RwLock::new(None)),
            published_messages: Arc::new(RwLock::new(Vec::new())),
            connected: Arc::new(RwLock::new(true)),
            operation_count: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Set a failure context for the service
    pub async fn set_failure(&self, context: FailureContext) {
        let mut failure = self.failure_context.write().await;
        *failure = Some(context);
        
        // Update connection status based on failure type
        if let Some(ref ctx) = *failure {
            match ctx.failure_type {
                FailureType::KafkaConnectionFailure => {
                    let mut connected = self.connected.write().await;
                    *connected = false;
                }
                _ => {}
            }
        }
        
        info!("MockKafkaService: Failure context set: {:?}", failure.as_ref().unwrap().failure_type);
    }
    
    /// Clear any active failure
    pub async fn clear_failure(&self) {
        let mut failure = self.failure_context.write().await;
        if let Some(ref mut ctx) = failure.as_mut() {
            ctx.stop();
        }
        *failure = None;
        
        // Restore connection
        let mut connected = self.connected.write().await;
        *connected = true;
        
        info!("MockKafkaService: Failure context cleared");
    }
    
    /// Check if Kafka is connected
    pub async fn is_connected(&self) -> bool {
        let connected = self.connected.read().await;
        *connected
    }
    
    /// Simulate publishing a message to Kafka
    pub async fn publish(&self, topic: &str, message: &str) -> Result<()> {
        self.increment_operation_count();
        
        if !self.is_connected().await {
            return Err(TestFrameworkError::SystemInitError(
                "Kafka connection failed".to_string()
            ).into());
        }
        
        // Check for active failures
        let failure_context = {
            let failure = self.failure_context.read().await;
            failure.clone()
        };
        
        if let Some(context) = failure_context {
            if context.is_active() {
                match context.failure_type {
                    FailureType::KafkaConnectionFailure => {
                        debug!("MockKafkaService: Simulating connection failure for publish to {}", topic);
                        return Err(TestFrameworkError::SystemInitError(
                            "Kafka connection lost".to_string()
                        ).into());
                    }
                    FailureType::KafkaPublishFailure => {
                        debug!("MockKafkaService: Simulating publish failure to {}", topic);
                        return Err(TestFrameworkError::SystemInitError(
                            "Kafka publish failed".to_string()
                        ).into());
                    }
                    _ => {}
                }
            }
        }
        
        // Normal operation - store the message
        let mut messages = self.published_messages.write().await;
        messages.push((topic.to_string(), message.to_string()));
        debug!("MockKafkaService: Published to {} - {}", topic, message);
        
        Ok(())
    }
    
    /// Get all published messages
    pub async fn get_published_messages(&self) -> Vec<(String, String)> {
        let messages = self.published_messages.read().await;
        messages.clone()
    }
    
    /// Get messages for a specific topic
    pub async fn get_messages_for_topic(&self, topic: &str) -> Vec<String> {
        let messages = self.published_messages.read().await;
        messages.iter()
            .filter(|(t, _)| t == topic)
            .map(|(_, m)| m.clone())
            .collect()
    }
    
    /// Clear all published messages
    pub async fn clear_messages(&self) {
        let mut messages = self.published_messages.write().await;
        messages.clear();
    }
    
    /// Get operation count
    pub fn get_operation_count(&self) -> u64 {
        *self.operation_count.lock().unwrap()
    }
    
    /// Reset operation count
    pub fn reset_operation_count(&self) {
        let mut count = self.operation_count.lock().unwrap();
        *count = 0;
    }
    
    fn increment_operation_count(&self) {
        let mut count = self.operation_count.lock().unwrap();
        *count += 1;
    }
}

impl Default for MockKafkaService {
    fn default() -> Self {
        Self::new()
    }
}

/// Main failure simulator that coordinates all mock services
#[derive(Debug)]
pub struct FailureSimulator {
    /// Mock HMM service
    pub hmm_service: MockHMMService,
    /// Mock Redis service
    pub redis_service: MockRedisService,
    /// Mock Kafka service
    pub kafka_service: MockKafkaService,
    /// Active failure scenarios
    active_failures: Arc<RwLock<Vec<FailureContext>>>,
}

impl FailureSimulator {
    /// Create a new failure simulator
    pub fn new() -> Self {
        Self {
            hmm_service: MockHMMService::new(),
            redis_service: MockRedisService::new(),
            kafka_service: MockKafkaService::new(),
            active_failures: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Simulate HMM service unavailable
    pub async fn simulate_hmm_unavailable(&self, duration: Duration) -> Result<FailureContext> {
        let context = FailureContext::new(
            FailureType::HmmServiceUnavailable,
            duration,
            RecoveryBehavior::Manual,
        );
        
        self.hmm_service.set_failure(context.clone()).await;
        self.add_active_failure(context.clone()).await;
        
        info!("FailureSimulator: HMM service unavailable simulation started for {:?}", duration);
        Ok(context)
    }
    
    /// Simulate HMM service slow response
    pub async fn simulate_hmm_slow_response(&self, duration: Duration, slowdown_factor: f64) -> Result<FailureContext> {
        let context = FailureContext::new(
            FailureType::HmmServiceSlowResponse,
            duration,
            RecoveryBehavior::Gradual { recovery_time: Duration::from_secs(10) },
        ).with_parameter("slowdown_factor".to_string(), slowdown_factor.to_string());
        
        self.hmm_service.set_failure(context.clone()).await;
        self.add_active_failure(context.clone()).await;
        
        info!("FailureSimulator: HMM service slow response simulation started ({}x slower) for {:?}", 
              slowdown_factor, duration);
        Ok(context)
    }
    
    /// Simulate Redis connection failure
    pub async fn simulate_redis_failure(&self, duration: Duration) -> Result<FailureContext> {
        let context = FailureContext::new(
            FailureType::RedisConnectionFailure,
            duration,
            RecoveryBehavior::Manual,
        );
        
        self.redis_service.set_failure(context.clone()).await;
        self.add_active_failure(context.clone()).await;
        
        info!("FailureSimulator: Redis connection failure simulation started for {:?}", duration);
        Ok(context)
    }
    
    /// Simulate Kafka connection failure
    pub async fn simulate_kafka_failure(&self, duration: Duration) -> Result<FailureContext> {
        let context = FailureContext::new(
            FailureType::KafkaConnectionFailure,
            duration,
            RecoveryBehavior::Manual,
        );
        
        self.kafka_service.set_failure(context.clone()).await;
        self.add_active_failure(context.clone()).await;
        
        info!("FailureSimulator: Kafka connection failure simulation started for {:?}", duration);
        Ok(context)
    }
    
    /// Simulate network partition
    pub async fn simulate_network_partition(&self, duration: Duration) -> Result<FailureContext> {
        let context = FailureContext::new(
            FailureType::NetworkPartition,
            duration,
            RecoveryBehavior::Manual,
        );
        
        // Network partition affects all services - create specific failure contexts for each service
        let hmm_context = FailureContext::new(
            FailureType::HmmServiceUnavailable,
            duration,
            RecoveryBehavior::Manual,
        );
        let redis_context = FailureContext::new(
            FailureType::RedisConnectionFailure,
            duration,
            RecoveryBehavior::Manual,
        );
        let kafka_context = FailureContext::new(
            FailureType::KafkaConnectionFailure,
            duration,
            RecoveryBehavior::Manual,
        );
        
        self.hmm_service.set_failure(hmm_context).await;
        self.redis_service.set_failure(redis_context).await;
        self.kafka_service.set_failure(kafka_context).await;
        self.add_active_failure(context.clone()).await;
        
        info!("FailureSimulator: Network partition simulation started for {:?}", duration);
        Ok(context)
    }
    
    /// Simulate data corruption
    pub async fn simulate_data_corruption(&self, corruption_rate: f64) -> Result<FailureContext> {
        let context = FailureContext::new(
            FailureType::DataCorruption,
            Duration::from_secs(60), // 1 minute of corruption
            RecoveryBehavior::Intermittent { failure_rate: corruption_rate },
        ).with_parameter("corruption_rate".to_string(), corruption_rate.to_string());
        
        self.add_active_failure(context.clone()).await;
        
        info!("FailureSimulator: Data corruption simulation started (rate: {:.2})", corruption_rate);
        Ok(context)
    }
    
    /// Stop a specific failure
    pub async fn stop_failure(&self, failure_type: FailureType) -> Result<()> {
        // Clear failure from services
        match failure_type {
            FailureType::HmmServiceUnavailable | FailureType::HmmServiceSlowResponse | FailureType::HmmServiceInvalidResponse => {
                self.hmm_service.clear_failure().await;
            }
            FailureType::RedisConnectionFailure | FailureType::RedisTimeout => {
                self.redis_service.clear_failure().await;
            }
            FailureType::KafkaConnectionFailure | FailureType::KafkaPublishFailure => {
                self.kafka_service.clear_failure().await;
            }
            FailureType::NetworkPartition => {
                self.hmm_service.clear_failure().await;
                self.redis_service.clear_failure().await;
                self.kafka_service.clear_failure().await;
            }
            _ => {}
        }
        
        // Remove from active failures
        let mut failures = self.active_failures.write().await;
        failures.retain(|f| f.failure_type != failure_type);
        
        info!("FailureSimulator: Stopped failure: {:?}", failure_type);
        Ok(())
    }
    
    /// Stop all active failures
    pub async fn stop_all_failures(&self) -> Result<()> {
        self.hmm_service.clear_failure().await;
        self.redis_service.clear_failure().await;
        self.kafka_service.clear_failure().await;
        
        let mut failures = self.active_failures.write().await;
        failures.clear();
        
        info!("FailureSimulator: All failures stopped");
        Ok(())
    }
    
    /// Get all active failures
    pub async fn get_active_failures(&self) -> Vec<FailureContext> {
        let failures = self.active_failures.read().await;
        failures.iter().filter(|f| f.is_active()).cloned().collect()
    }
    
    /// Get failure statistics
    pub async fn get_failure_statistics(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        
        stats.insert("hmm_calls".to_string(), self.hmm_service.get_call_count());
        stats.insert("redis_operations".to_string(), self.redis_service.get_operation_count());
        stats.insert("kafka_operations".to_string(), self.kafka_service.get_operation_count());
        
        let failures = self.active_failures.read().await;
        stats.insert("active_failures".to_string(), failures.len() as u64);
        
        stats
    }
    
    /// Reset all statistics
    pub async fn reset_statistics(&self) {
        self.hmm_service.reset_call_count();
        self.redis_service.reset_operation_count();
        self.kafka_service.reset_operation_count();
        
        self.redis_service.clear_data().await;
        self.kafka_service.clear_messages().await;
    }
    
    async fn add_active_failure(&self, context: FailureContext) {
        let mut failures = self.active_failures.write().await;
        failures.push(context);
    }
}

impl Default for FailureSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_failure_context_creation() {
        let context = FailureContext::new(
            FailureType::HmmServiceUnavailable,
            Duration::from_secs(10),
            RecoveryBehavior::Manual,
        );
        
        assert_eq!(context.failure_type, FailureType::HmmServiceUnavailable);
        assert_eq!(context.duration, Duration::from_secs(10));
        assert!(context.is_active());
    }
    
    #[tokio::test]
    async fn test_failure_context_intensity() {
        let context = FailureContext::new(
            FailureType::HmmServiceSlowResponse,
            Duration::from_millis(100),
            RecoveryBehavior::Gradual { recovery_time: Duration::from_millis(100) },
        );
        
        assert_eq!(context.get_intensity(), 1.0); // Should be full intensity initially
        
        // Wait for failure period to pass
        sleep(Duration::from_millis(150)).await;
        
        let intensity = context.get_intensity();
        assert!(intensity < 1.0 && intensity > 0.0); // Should be in recovery phase
    }
    
    #[tokio::test]
    async fn test_mock_hmm_service_normal_operation() {
        let service = MockHMMService::new();
        
        let weights = service.get_weights("BTCUSDT").await.unwrap();
        
        assert_eq!(weights.get("w_ldc"), Some(&0.5));
        assert_eq!(weights.get("w_mr"), Some(&0.3));
        assert_eq!(weights.get("w_tsmom"), Some(&0.2));
        assert_eq!(service.get_call_count(), 1);
    }
    
    #[tokio::test]
    async fn test_mock_hmm_service_unavailable() {
        let service = MockHMMService::new();
        
        let context = FailureContext::new(
            FailureType::HmmServiceUnavailable,
            Duration::from_secs(10),
            RecoveryBehavior::Manual,
        );
        
        service.set_failure(context).await;
        
        let result = service.get_weights("BTCUSDT").await;
        assert!(result.is_err());
        assert_eq!(service.get_call_count(), 1);
    }
    
    #[tokio::test]
    async fn test_mock_redis_service_normal_operation() {
        let service = MockRedisService::new();
        
        service.set("test_key", "test_value").await.unwrap();
        let value = service.get("test_key").await.unwrap();
        
        assert_eq!(value, Some("test_value".to_string()));
        assert_eq!(service.get_operation_count(), 2); // SET + GET
    }
    
    #[tokio::test]
    async fn test_mock_redis_service_connection_failure() {
        let service = MockRedisService::new();
        
        let context = FailureContext::new(
            FailureType::RedisConnectionFailure,
            Duration::from_secs(10),
            RecoveryBehavior::Manual,
        );
        
        service.set_failure(context).await;
        
        let result = service.set("test_key", "test_value").await;
        assert!(result.is_err());
        assert!(!service.is_connected().await);
    }
    
    #[tokio::test]
    async fn test_mock_kafka_service_normal_operation() {
        let service = MockKafkaService::new();
        
        service.publish("test_topic", "test_message").await.unwrap();
        
        let messages = service.get_messages_for_topic("test_topic").await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], "test_message");
        assert_eq!(service.get_operation_count(), 1);
    }
    
    #[tokio::test]
    async fn test_mock_kafka_service_publish_failure() {
        let service = MockKafkaService::new();
        
        let context = FailureContext::new(
            FailureType::KafkaPublishFailure,
            Duration::from_secs(10),
            RecoveryBehavior::Manual,
        );
        
        service.set_failure(context).await;
        
        let result = service.publish("test_topic", "test_message").await;
        assert!(result.is_err());
        
        let messages = service.get_messages_for_topic("test_topic").await;
        assert_eq!(messages.len(), 0); // No message should be published
    }
    
    #[tokio::test]
    async fn test_failure_simulator_hmm_unavailable() {
        let simulator = FailureSimulator::new();
        
        let context = simulator.simulate_hmm_unavailable(Duration::from_secs(5)).await.unwrap();
        assert_eq!(context.failure_type, FailureType::HmmServiceUnavailable);
        
        // Test that HMM service is now unavailable
        let result = simulator.hmm_service.get_weights("BTCUSDT").await;
        assert!(result.is_err());
        
        // Stop the failure
        simulator.stop_failure(FailureType::HmmServiceUnavailable).await.unwrap();
        
        // Test that HMM service is now available again
        let result = simulator.hmm_service.get_weights("BTCUSDT").await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_failure_simulator_network_partition() {
        let simulator = FailureSimulator::new();
        
        let context = simulator.simulate_network_partition(Duration::from_secs(5)).await.unwrap();
        assert_eq!(context.failure_type, FailureType::NetworkPartition);
        
        // All services should be affected
        assert!(simulator.hmm_service.get_weights("BTCUSDT").await.is_err());
        assert!(simulator.redis_service.set("key", "value").await.is_err());
        assert!(simulator.kafka_service.publish("topic", "message").await.is_err());
        
        // Stop all failures
        simulator.stop_all_failures().await.unwrap();
        
        // All services should be available again
        assert!(simulator.hmm_service.get_weights("BTCUSDT").await.is_ok());
        assert!(simulator.redis_service.set("key", "value").await.is_ok());
        assert!(simulator.kafka_service.publish("topic", "message").await.is_ok());
    }
    
    #[tokio::test]
    async fn test_failure_simulator_statistics() {
        let simulator = FailureSimulator::new();
        
        // Perform some operations
        let _ = simulator.hmm_service.get_weights("BTCUSDT").await;
        let _ = simulator.redis_service.set("key", "value").await;
        let _ = simulator.kafka_service.publish("topic", "message").await;
        
        let stats = simulator.get_failure_statistics().await;
        
        assert_eq!(stats.get("hmm_calls"), Some(&1));
        assert_eq!(stats.get("redis_operations"), Some(&1));
        assert_eq!(stats.get("kafka_operations"), Some(&1));
        assert_eq!(stats.get("active_failures"), Some(&0));
        
        // Reset and verify
        simulator.reset_statistics().await;
        
        let stats = simulator.get_failure_statistics().await;
        assert_eq!(stats.get("hmm_calls"), Some(&0));
        assert_eq!(stats.get("redis_operations"), Some(&0));
        assert_eq!(stats.get("kafka_operations"), Some(&0));
    }
}