use crate::testing_error::*;
use crate::test_diagnostics::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::timeout;

/// Graceful error recovery system for test suite continuation
pub struct GracefulRecoverySystem {
    config: RecoveryConfig,
    diagnostics_engine: Arc<Mutex<TestDiagnosticsEngine>>,
    recovery_strategies: HashMap<String, RecoveryStrategy>,
    failure_history: Arc<Mutex<Vec<TestFailureRecord>>>,
    circuit_breakers: HashMap<String, CircuitBreaker>,
}

/// Configuration for graceful recovery system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Maximum number of retry attempts per test
    pub max_retry_attempts: u32,
    /// Base delay for exponential backoff (milliseconds)
    pub base_retry_delay_ms: u64,
    /// Maximum delay for exponential backoff (milliseconds)
    pub max_retry_delay_ms: u64,
    /// Enable circuit breaker pattern
    pub enable_circuit_breaker: bool,
    /// Circuit breaker failure threshold
    pub circuit_breaker_failure_threshold: u32,
    /// Circuit breaker timeout (seconds)
    pub circuit_breaker_timeout_seconds: u64,
    /// Enable test isolation
    pub enable_test_isolation: bool,
    /// Maximum concurrent recovery operations
    pub max_concurrent_recoveries: usize,
    /// Enable adaptive recovery strategies
    pub enable_adaptive_strategies: bool,
    /// Failure rate threshold for adaptive behavior
    pub adaptive_failure_threshold: f32,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_retry_attempts: 3,
            base_retry_delay_ms: 100,
            max_retry_delay_ms: 5000,
            enable_circuit_breaker: true,
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_timeout_seconds: 60,
            enable_test_isolation: true,
            max_concurrent_recoveries: 4,
            enable_adaptive_strategies: true,
            adaptive_failure_threshold: 0.3, // 30% failure rate
        }
    }
}

/// Recovery strategy for specific error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStrategy {
    pub strategy_name: String,
    pub applicable_errors: Vec<String>, // Error type patterns
    pub recovery_actions: Vec<RecoveryAction>,
    pub success_rate: f32,
    pub average_recovery_time_ms: u64,
    pub side_effects: Vec<String>,
}

/// Circuit breaker for preventing cascading failures
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub name: String,
    pub state: CircuitBreakerState,
    pub failure_count: u32,
    pub failure_threshold: u32,
    pub last_failure_time: Option<Instant>,
    pub timeout_duration: Duration,
    pub success_count_since_half_open: u32,
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Failing fast
    HalfOpen, // Testing if service recovered
}

/// Test failure record for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFailureRecord {
    pub test_name: String,
    pub error_type: String,
    pub failure_time: chrono::DateTime<chrono::Utc>,
    pub recovery_attempted: bool,
    pub recovery_successful: bool,
    pub recovery_time_ms: Option<u64>,
    pub context: TestContext,
}

/// Test execution wrapper with recovery capabilities
pub struct RecoverableTestExecution {
    pub test_name: String,
    pub test_function: Box<dyn Fn() -> Result<TestResult> + Send + Sync>,
    pub recovery_system: Arc<GracefulRecoverySystem>,
    pub isolation_enabled: bool,
}

/// Test execution result with recovery information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverableTestResult {
    pub test_name: String,
    pub final_result: TestExecutionStatus,
    pub attempts_made: u32,
    pub total_execution_time_ms: u64,
    pub recovery_actions_taken: Vec<RecoveryActionResult>,
    pub errors_encountered: Vec<TestingError>,
    pub recovery_successful: bool,
}

/// Test execution status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestExecutionStatus {
    Success,
    Failed,
    Recovered,
    Skipped,
    CircuitBreakerOpen,
    IsolationFailure,
}

/// Basic test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub passed: bool,
    pub message: String,
    pub execution_time_ms: u64,
    pub details: HashMap<String, String>,
}

impl GracefulRecoverySystem {
    /// Create a new graceful recovery system
    pub fn new(config: RecoveryConfig, diagnostics_engine: TestDiagnosticsEngine) -> Self {
        let mut recovery_strategies = HashMap::new();
        
        // Initialize default recovery strategies
        recovery_strategies.insert(
            "performance_failure".to_string(),
            RecoveryStrategy {
                strategy_name: "Performance Optimization Recovery".to_string(),
                applicable_errors: vec!["PerformanceTestError".to_string()],
                recovery_actions: vec![
                    RecoveryAction {
                        action_type: RecoveryActionType::ReduceResourceUsage,
                        description: "Reduce resource usage to improve performance".to_string(),
                        priority: ActionPriority::High,
                        estimated_time_minutes: 2,
                        success_probability: 0.7,
                        side_effects: vec!["May reduce test accuracy".to_string()],
                    },
                    RecoveryAction {
                        action_type: RecoveryActionType::FallbackToAlternative,
                        description: "Use alternative algorithm implementation".to_string(),
                        priority: ActionPriority::Medium,
                        estimated_time_minutes: 1,
                        success_probability: 0.8,
                        side_effects: vec!["Different performance characteristics".to_string()],
                    },
                ],
                success_rate: 0.75,
                average_recovery_time_ms: 2000,
                side_effects: vec!["May affect subsequent test performance".to_string()],
            }
        );

        recovery_strategies.insert(
            "statistical_failure".to_string(),
            RecoveryStrategy {
                strategy_name: "Statistical Test Recovery".to_string(),
                applicable_errors: vec!["StatisticalTestError".to_string()],
                recovery_actions: vec![
                    RecoveryAction {
                        action_type: RecoveryActionType::UpdateTestData,
                        description: "Generate additional test data for statistical power".to_string(),
                        priority: ActionPriority::High,
                        estimated_time_minutes: 5,
                        success_probability: 0.9,
                        side_effects: vec!["Increased test execution time".to_string()],
                    },
                ],
                success_rate: 0.85,
                average_recovery_time_ms: 5000,
                side_effects: vec!["Longer test execution time".to_string()],
            }
        );

        Self {
            config,
            diagnostics_engine: Arc::new(Mutex::new(diagnostics_engine)),
            recovery_strategies,
            failure_history: Arc::new(Mutex::new(Vec::new())),
            circuit_breakers: HashMap::new(),
        }
    }

    /// Execute a test with graceful recovery capabilities
    pub async fn execute_test_with_recovery<F, Fut>(
        &mut self,
        test_name: String,
        test_function: F,
        test_context: TestContext,
    ) -> Result<RecoverableTestResult>
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<TestResult>> + Send + 'static,
    {
        let start_time = Instant::now();
        let mut attempts_made = 0;
        let mut recovery_actions_taken = Vec::new();
        let mut errors_encountered = Vec::new();
        let mut _recovery_successful = false;

        // Check circuit breaker
        if self.config.enable_circuit_breaker {
            if let Some(circuit_breaker) = self.circuit_breakers.get_mut(&test_name) {
                if circuit_breaker.state == CircuitBreakerState::Open {
                    if let Some(last_failure) = circuit_breaker.last_failure_time {
                        if last_failure.elapsed() < circuit_breaker.timeout_duration {
                            return Ok(RecoverableTestResult {
                                test_name,
                                final_result: TestExecutionStatus::CircuitBreakerOpen,
                                attempts_made: 0,
                                total_execution_time_ms: start_time.elapsed().as_millis() as u64,
                                recovery_actions_taken,
                                errors_encountered,
                                recovery_successful: false,
                            });
                        } else {
                            // Transition to half-open
                            circuit_breaker.state = CircuitBreakerState::HalfOpen;
                            circuit_breaker.success_count_since_half_open = 0;
                        }
                    }
                }
            }
        }

        // Main execution loop with retry logic
        loop {
            attempts_made += 1;

            // Execute test with timeout and isolation
            let execution_result = if self.config.enable_test_isolation {
                self.execute_test_isolated(&test_function, &test_context).await
            } else {
                self.execute_test_direct(&test_function).await
            };

            match execution_result {
                Ok(test_result) => {
                    // Test succeeded
                    self.handle_test_success(&test_name).await?;
                    
                    return Ok(RecoverableTestResult {
                        test_name,
                        final_result: if recovery_actions_taken.is_empty() {
                            TestExecutionStatus::Success
                        } else {
                            TestExecutionStatus::Recovered
                        },
                        attempts_made,
                        total_execution_time_ms: start_time.elapsed().as_millis() as u64,
                        recovery_actions_taken: recovery_actions_taken.clone(),
                        errors_encountered: errors_encountered.clone(),
                        recovery_successful: !recovery_actions_taken.is_empty(),
                    });
                },
                Err(error) => {
                    // Test failed - attempt recovery
                    let testing_error = self.convert_to_testing_error(error, &test_context)?;
                    errors_encountered.push(testing_error.clone());

                    // Record failure
                    self.record_test_failure(&test_name, &testing_error, &test_context).await?;

                    // Check if we should attempt recovery
                    if attempts_made >= self.config.max_retry_attempts {
                        // Max attempts reached
                        self.handle_test_failure(&test_name).await?;
                        
                        return Ok(RecoverableTestResult {
                            test_name,
                            final_result: TestExecutionStatus::Failed,
                            attempts_made,
                            total_execution_time_ms: start_time.elapsed().as_millis() as u64,
                            recovery_actions_taken,
                            errors_encountered,
                            recovery_successful: false,
                        });
                    }

                    // Attempt recovery
                    match self.attempt_recovery(&testing_error, &test_context).await {
                        Ok(recovery_result) => {
                            recovery_actions_taken.extend(recovery_result.attempted_actions);
                            
                            if recovery_result.success {
                                _recovery_successful = true;
                                // Wait before retry with exponential backoff
                                let delay = self.calculate_retry_delay(attempts_made);
                                tokio::time::sleep(delay).await;
                                continue; // Retry the test
                            } else {
                                // Recovery failed, but continue to next attempt if we have retries left
                                let delay = self.calculate_retry_delay(attempts_made);
                                tokio::time::sleep(delay).await;
                                continue;
                            }
                        },
                        Err(recovery_error) => {
                            eprintln!("Recovery attempt failed: {}", recovery_error);
                            // Continue to next attempt if we have retries left
                            let delay = self.calculate_retry_delay(attempts_made);
                            tokio::time::sleep(delay).await;
                            continue;
                        }
                    }
                }
            }
        }
    }

    /// Execute test in isolated environment
    async fn execute_test_isolated<F, Fut>(
        &self,
        test_function: &F,
        _test_context: &TestContext,
    ) -> Result<TestResult>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<TestResult>> + Send,
    {
        // In a real implementation, this would set up process isolation,
        // resource limits, etc. For now, we'll just add a timeout.
        let test_timeout = Duration::from_secs(300); // 5 minutes
        
        match timeout(test_timeout, test_function()).await {
            Ok(result) => result,
            Err(_) => Err(anyhow::anyhow!("Test execution timed out")),
        }
    }

    /// Execute test directly without isolation
    async fn execute_test_direct<F, Fut>(&self, test_function: &F) -> Result<TestResult>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<TestResult>> + Send,
    {
        test_function().await
    }

    /// Attempt recovery from a test failure
    pub async fn attempt_recovery(
        &mut self,
        error: &TestingError,
        test_context: &TestContext,
    ) -> Result<RecoveryResult> {
        let mut diagnostics = self.diagnostics_engine.lock().unwrap();
        let _error_report = diagnostics.analyze_error(error.clone(), test_context.clone())?;
        drop(diagnostics); // Release lock

        // Find applicable recovery strategy
        let strategy = self.find_recovery_strategy(error)?;
        
        if let Some(strategy) = strategy {
            println!("Applying recovery strategy: {}", strategy.strategy_name);
            
            let mut attempted_actions = Vec::new();
            let start_time = Instant::now();

            for action in &strategy.recovery_actions {
                let action_result = self.execute_recovery_action(action).await?;
                attempted_actions.push(action_result.clone());

                if action_result.success {
                    return Ok(RecoveryResult {
                        success: true,
                        attempted_actions,
                        recovery_time_seconds: start_time.elapsed().as_secs(),
                        error_message: None,
                    });
                }
            }

            Ok(RecoveryResult {
                success: false,
                attempted_actions,
                recovery_time_seconds: start_time.elapsed().as_secs(),
                error_message: Some("All recovery actions failed".to_string()),
            })
        } else {
            // No specific strategy found, try generic recovery
            self.attempt_generic_recovery(error).await
        }
    }

    /// Find applicable recovery strategy for an error
    fn find_recovery_strategy(&self, error: &TestingError) -> Result<Option<RecoveryStrategy>> {
        let error_type = match error {
            TestingError::PerformanceTestError { .. } => "PerformanceTestError",
            TestingError::StatisticalTestError { .. } => "StatisticalTestError",
            TestingError::MathematicalAccuracyError { .. } => "MathematicalAccuracyError",
            TestingError::IntegrationTestError { .. } => "IntegrationTestError",
            TestingError::TestDataValidationError { .. } => "TestDataValidationError",
            TestingError::ResourceExhaustionError { .. } => "ResourceExhaustionError",
            _ => "GenericError",
        };

        for strategy in self.recovery_strategies.values() {
            if strategy.applicable_errors.iter().any(|pattern| pattern == error_type) {
                return Ok(Some(strategy.clone()));
            }
        }

        Ok(None)
    }

    /// Execute a specific recovery action
    async fn execute_recovery_action(&self, action: &RecoveryAction) -> Result<RecoveryActionResult> {
        let start_time = Instant::now();

        match action.action_type {
            RecoveryActionType::RetryWithBackoff => {
                // Implement exponential backoff
                let delay = Duration::from_millis(self.config.base_retry_delay_ms);
                tokio::time::sleep(delay).await;
                
                Ok(RecoveryActionResult {
                    action_type: action.action_type.clone(),
                    success: true,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error_message: None,
                })
            },

            RecoveryActionType::FallbackToAlternative => {
                // Simulate fallback to alternative implementation
                tokio::time::sleep(Duration::from_millis(100)).await;
                
                Ok(RecoveryActionResult {
                    action_type: action.action_type.clone(),
                    success: true,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error_message: None,
                })
            },

            RecoveryActionType::ReduceResourceUsage => {
                // Simulate resource usage reduction
                tokio::time::sleep(Duration::from_millis(200)).await;
                
                Ok(RecoveryActionResult {
                    action_type: action.action_type.clone(),
                    success: true,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error_message: None,
                })
            },

            RecoveryActionType::CleanupResources => {
                // Simulate resource cleanup
                tokio::time::sleep(Duration::from_millis(500)).await;
                
                Ok(RecoveryActionResult {
                    action_type: action.action_type.clone(),
                    success: true,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error_message: None,
                })
            },

            RecoveryActionType::UpdateTestData => {
                // Simulate test data update
                tokio::time::sleep(Duration::from_millis(1000)).await;
                
                Ok(RecoveryActionResult {
                    action_type: action.action_type.clone(),
                    success: true,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error_message: None,
                })
            },

            _ => {
                // For other action types, simulate partial success
                Ok(RecoveryActionResult {
                    action_type: action.action_type.clone(),
                    success: false,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error_message: Some("Recovery action not fully implemented".to_string()),
                })
            }
        }
    }

    /// Attempt generic recovery when no specific strategy is available
    async fn attempt_generic_recovery(&self, _error: &TestingError) -> Result<RecoveryResult> {
        let start_time = Instant::now();
        
        // Generic recovery: wait and retry
        let delay = Duration::from_millis(self.config.base_retry_delay_ms);
        tokio::time::sleep(delay).await;

        let generic_action = RecoveryActionResult {
            action_type: RecoveryActionType::RetryWithBackoff,
            success: true,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            error_message: None,
        };

        Ok(RecoveryResult {
            success: true,
            attempted_actions: vec![generic_action],
            recovery_time_seconds: start_time.elapsed().as_secs(),
            error_message: None,
        })
    }

    /// Calculate retry delay with exponential backoff
    fn calculate_retry_delay(&self, attempt: u32) -> Duration {
        let delay_ms = self.config.base_retry_delay_ms * (2_u64.pow(attempt.saturating_sub(1)));
        let capped_delay = delay_ms.min(self.config.max_retry_delay_ms);
        Duration::from_millis(capped_delay)
    }

    /// Handle test success for circuit breaker management
    async fn handle_test_success(&mut self, test_name: &str) -> Result<()> {
        if self.config.enable_circuit_breaker {
            if let Some(circuit_breaker) = self.circuit_breakers.get_mut(test_name) {
                match circuit_breaker.state {
                    CircuitBreakerState::HalfOpen => {
                        circuit_breaker.success_count_since_half_open += 1;
                        if circuit_breaker.success_count_since_half_open >= 3 {
                            // Transition back to closed
                            circuit_breaker.state = CircuitBreakerState::Closed;
                            circuit_breaker.failure_count = 0;
                        }
                    },
                    CircuitBreakerState::Closed => {
                        // Reset failure count on success
                        circuit_breaker.failure_count = 0;
                    },
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Handle test failure for circuit breaker management
    async fn handle_test_failure(&mut self, test_name: &str) -> Result<()> {
        if self.config.enable_circuit_breaker {
            let circuit_breaker = self.circuit_breakers.entry(test_name.to_string())
                .or_insert_with(|| CircuitBreaker {
                    name: test_name.to_string(),
                    state: CircuitBreakerState::Closed,
                    failure_count: 0,
                    failure_threshold: self.config.circuit_breaker_failure_threshold,
                    last_failure_time: None,
                    timeout_duration: Duration::from_secs(self.config.circuit_breaker_timeout_seconds),
                    success_count_since_half_open: 0,
                });

            circuit_breaker.failure_count += 1;
            circuit_breaker.last_failure_time = Some(Instant::now());

            if circuit_breaker.failure_count >= circuit_breaker.failure_threshold {
                circuit_breaker.state = CircuitBreakerState::Open;
                println!("Circuit breaker opened for test: {}", test_name);
            }
        }
        Ok(())
    }

    /// Record test failure for pattern analysis
    async fn record_test_failure(
        &self,
        test_name: &str,
        error: &TestingError,
        test_context: &TestContext,
    ) -> Result<()> {
        let failure_record = TestFailureRecord {
            test_name: test_name.to_string(),
            error_type: format!("{:?}", error).split('(').next().unwrap_or("Unknown").to_string(),
            failure_time: chrono::Utc::now(),
            recovery_attempted: false, // Will be updated if recovery is attempted
            recovery_successful: false,
            recovery_time_ms: None,
            context: test_context.clone(),
        };

        let mut history = self.failure_history.lock().unwrap();
        history.push(failure_record);

        // Maintain history size (keep last 1000 failures)
        if history.len() > 1000 {
            history.remove(0);
        }

        Ok(())
    }

    /// Convert generic error to TestingError
    fn convert_to_testing_error(&self, error: anyhow::Error, context: &TestContext) -> Result<TestingError> {
        let error_message = error.to_string();
        
        // Try to classify the error based on its message
        if error_message.contains("timeout") {
            Ok(TestingError::TestTimeoutError {
                test_name: context.test_suite.clone(),
                timeout_seconds: 300, // Default timeout
                execution_phase: context.test_phase.clone(),
                progress_percent: 50.0, // Estimate
                suggestions: vec![
                    "Increase test timeout".to_string(),
                    "Optimize test performance".to_string(),
                ],
            })
        } else if error_message.contains("memory") || error_message.contains("allocation") {
            Ok(TestingError::ResourceExhaustionError {
                resource: "Memory".to_string(),
                usage_percent: 95.0, // Estimate
                threshold_percent: 90.0,
                adaptive_actions: vec![
                    "Reduce memory usage".to_string(),
                    "Enable garbage collection".to_string(),
                ],
            })
        } else {
            // Generic integration test error
            Ok(TestingError::IntegrationTestError {
                test_name: context.test_suite.clone(),
                component: "Unknown".to_string(),
                error_message,
                interactions: Vec::new(),
                recovery_suggestions: vec![
                    "Check component dependencies".to_string(),
                    "Verify configuration".to_string(),
                ],
            })
        }
    }

    /// Get failure statistics for adaptive behavior
    pub fn get_failure_statistics(&self) -> Result<FailureStatistics> {
        let history = self.failure_history.lock().unwrap();
        let total_failures = history.len();
        
        if total_failures == 0 {
            return Ok(FailureStatistics {
                total_failures: 0,
                failure_rate: 0.0,
                most_common_errors: Vec::new(),
                recovery_success_rate: 0.0,
                average_recovery_time_ms: 0.0,
            });
        }

        let recent_failures = history.iter()
            .filter(|f| {
                let now = chrono::Utc::now();
                (now - f.failure_time).num_hours() < 24 // Last 24 hours
            })
            .collect::<Vec<_>>();

        let failure_rate = recent_failures.len() as f32 / 100.0; // Assume 100 tests per day

        // Count error types
        let mut error_counts: HashMap<String, u32> = HashMap::new();
        for failure in &recent_failures {
            *error_counts.entry(failure.error_type.clone()).or_insert(0) += 1;
        }

        let mut most_common_errors: Vec<(String, u32)> = error_counts.into_iter().collect();
        most_common_errors.sort_by(|a, b| b.1.cmp(&a.1));
        most_common_errors.truncate(5); // Top 5

        // Calculate recovery statistics
        let recovery_attempts = recent_failures.iter().filter(|f| f.recovery_attempted).count();
        let successful_recoveries = recent_failures.iter().filter(|f| f.recovery_successful).count();
        
        let recovery_success_rate = if recovery_attempts > 0 {
            successful_recoveries as f32 / recovery_attempts as f32
        } else {
            0.0
        };

        let average_recovery_time_ms = if successful_recoveries > 0 {
            let total_recovery_time: u64 = recent_failures.iter()
                .filter_map(|f| f.recovery_time_ms)
                .sum();
            total_recovery_time as f64 / successful_recoveries as f64
        } else {
            0.0
        };

        Ok(FailureStatistics {
            total_failures,
            failure_rate,
            most_common_errors,
            recovery_success_rate,
            average_recovery_time_ms,
        })
    }

    /// Update recovery strategies based on success rates
    pub fn update_adaptive_strategies(&mut self) -> Result<()> {
        if !self.config.enable_adaptive_strategies {
            return Ok(());
        }

        let stats = self.get_failure_statistics()?;
        
        // If failure rate is too high, adjust strategies
        if stats.failure_rate > self.config.adaptive_failure_threshold {
            println!("High failure rate detected ({:.1}%), adapting recovery strategies", 
                     stats.failure_rate * 100.0);
            
            // Increase retry attempts for high-failure scenarios
            // This is a simplified adaptation - in practice, you'd use more sophisticated ML
            for strategy in self.recovery_strategies.values_mut() {
                if strategy.success_rate < 0.5 {
                    // Add more aggressive recovery actions for low-success strategies
                    strategy.recovery_actions.push(RecoveryAction {
                        action_type: RecoveryActionType::RestartComponent,
                        description: "Restart component due to repeated failures".to_string(),
                        priority: ActionPriority::High,
                        estimated_time_minutes: 5,
                        success_probability: 0.8,
                        side_effects: vec!["May affect other tests".to_string()],
                    });
                }
            }
        }

        Ok(())
    }
}

/// Failure statistics for adaptive behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureStatistics {
    pub total_failures: usize,
    pub failure_rate: f32,
    pub most_common_errors: Vec<(String, u32)>,
    pub recovery_success_rate: f32,
    pub average_recovery_time_ms: f64,
}

impl CircuitBreaker {
    /// Check if the circuit breaker allows execution
    pub fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.timeout_duration {
                        self.state = CircuitBreakerState::HalfOpen;
                        self.success_count_since_half_open = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            },
            CircuitBreakerState::HalfOpen => true,
        }
    }

    /// Record a successful execution
    pub fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::HalfOpen => {
                self.success_count_since_half_open += 1;
                if self.success_count_since_half_open >= 3 {
                    self.state = CircuitBreakerState::Closed;
                    self.failure_count = 0;
                }
            },
            CircuitBreakerState::Closed => {
                self.failure_count = 0;
            },
            _ => {}
        }
    }

    /// Record a failed execution
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        if self.failure_count >= self.failure_threshold {
            self.state = CircuitBreakerState::Open;
        }
    }
}