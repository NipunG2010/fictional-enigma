//! Test harness for orchestrating end-to-end test execution
//! 
//! The TestHarness manages test lifecycle, coordinates test execution,
//! and aggregates results from different test suites.

use crate::{
    config::TestConfig,
    data_generator::TestDataGenerator,
    performance::PerformanceMonitor,
    reporting::{TestReport, TestResults, TestCaseResult, TestSummary},
    validation::ResultValidator,
    Result, TestFrameworkError, TestStatus, Uuid, Instant,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{info, warn, error, debug};

/// Main test harness for orchestrating end-to-end tests
pub struct TestHarness {
    /// Test configuration
    config: TestConfig,
    
    /// Test data generator
    data_generator: TestDataGenerator,
    
    /// Performance monitoring
    performance_monitor: PerformanceMonitor,
    
    /// Result validator
    validator: ResultValidator,
    
    /// Test execution semaphore for controlling parallelism
    execution_semaphore: Arc<Semaphore>,
    
    /// Test session ID
    session_id: Uuid,
}

/// Test execution context
#[derive(Debug, Clone)]
pub struct TestContext {
    /// Test case name
    pub name: String,
    
    /// Test case ID
    pub id: Uuid,
    
    /// Test start time
    pub start_time: Instant,
    
    /// Test configuration
    pub config: TestConfig,
    
    /// Test metadata
    pub metadata: HashMap<String, String>,
}

/// Test suite execution result
#[derive(Debug)]
pub struct TestSuiteResult {
    /// Suite name
    pub suite_name: String,
    
    /// Individual test results
    pub test_results: Vec<TestCaseResult>,
    
    /// Suite execution duration
    pub duration_ms: u64,
    
    /// Suite-level metrics
    pub metrics: HashMap<String, f64>,
}

impl TestHarness {
    /// Create a new test harness with the given configuration
    pub async fn new(config: TestConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        // Initialize logging if verbose mode is enabled
        if config.execution.verbose_logging {
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::DEBUG)
                .init();
        }
        
        // Create output directory
        std::fs::create_dir_all(&config.execution.output_dir)
            .map_err(|e| TestFrameworkError::SetupError(format!("Failed to create output directory: {}", e)))?;
        
        // Initialize components
        let data_generator = TestDataGenerator::new(config.data_generation.clone())?;
        let performance_monitor = PerformanceMonitor::new();
        let validator = ResultValidator::new(config.validation.clone())?;
        
        let execution_semaphore = Arc::new(Semaphore::new(config.execution.max_parallel_tests as usize));
        let session_id = Uuid::new_v4();
        
        info!("Test harness initialized with session ID: {}", session_id);
        
        Ok(Self {
            config,
            data_generator,
            performance_monitor,
            validator,
            execution_semaphore,
            session_id,
        })
    }
    
    /// Create test harness from configuration file
    pub async fn from_config_file<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let config = TestConfig::from_file(config_path)?;
        Self::new(config).await
    }
    
    /// Get the test session ID
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }
    
    /// Get reference to the configuration
    pub fn config(&self) -> &TestConfig {
        &self.config
    }
    
    /// Get reference to the data generator
    pub fn data_generator(&self) -> &TestDataGenerator {
        &self.data_generator
    }
    
    /// Get reference to the performance monitor
    pub fn performance_monitor(&self) -> &PerformanceMonitor {
        &self.performance_monitor
    }
    
    /// Get reference to the validator
    pub fn validator(&self) -> &ResultValidator {
        &self.validator
    }
    
    /// Run all test suites and generate comprehensive report
    pub async fn run_all_tests(&mut self) -> Result<TestReport> {
        info!("Starting comprehensive test execution for session {}", self.session_id);
        let overall_start = Instant::now();
        
        let mut all_results = Vec::new();
        
        // Run pipeline integration tests
        info!("Running pipeline integration tests...");
        match self.run_pipeline_tests().await {
            Ok(results) => {
                info!("Pipeline tests completed: {} passed, {} failed", 
                      results.passed_tests, results.failed_tests);
                all_results.push(results);
            }
            Err(e) => {
                error!("Pipeline tests failed: {}", e);
                all_results.push(TestResults::failed_suite("pipeline_tests".to_string(), e.to_string()));
            }
        }
        
        // Run failure scenario tests
        info!("Running failure scenario tests...");
        match self.run_failure_tests().await {
            Ok(results) => {
                info!("Failure tests completed: {} passed, {} failed", 
                      results.passed_tests, results.failed_tests);
                all_results.push(results);
            }
            Err(e) => {
                error!("Failure tests failed: {}", e);
                all_results.push(TestResults::failed_suite("failure_tests".to_string(), e.to_string()));
            }
        }
        
        // Run performance validation tests
        info!("Running performance validation tests...");
        match self.run_performance_tests().await {
            Ok(results) => {
                info!("Performance tests completed: {} passed, {} failed", 
                      results.passed_tests, results.failed_tests);
                all_results.push(results);
            }
            Err(e) => {
                error!("Performance tests failed: {}", e);
                all_results.push(TestResults::failed_suite("performance_tests".to_string(), e.to_string()));
            }
        }
        
        let overall_duration = overall_start.elapsed();
        
        // Generate comprehensive report
        let report = self.generate_report_internal(&all_results, overall_duration).await?;
        
        info!("Test execution completed in {:.2}s. Overall pass rate: {:.1}%", 
              overall_duration.as_secs_f64(), 
              report.summary.overall_pass_rate * 100.0);
        
        Ok(report)
    }
    
    /// Run pipeline integration tests
    pub async fn run_pipeline_tests(&mut self) -> Result<TestResults> {
        info!("Executing pipeline integration test suite");
        let start_time = chrono::Utc::now().timestamp();
        let suite_start = Instant::now();
        
        let mut test_cases = Vec::new();
        
        // Test complete signal pipeline for each symbol
        for symbol in &self.config.pipeline_tests.test_symbols {
            let test_name = format!("complete_pipeline_{}", symbol);
            let test_result = self.execute_test_case(&test_name, || {
                self.test_complete_signal_pipeline(symbol)
            }).await;
            test_cases.push(test_result);
        }
        
        // Test feature computation accuracy
        if self.config.pipeline_tests.validate_against_reference {
            let test_result = self.execute_test_case("feature_computation_accuracy", || {
                self.test_feature_computation_accuracy()
            }).await;
            test_cases.push(test_result);
        }
        
        // Test signal generation validation
        let test_result = self.execute_test_case("signal_generation_validation", || {
            self.test_signal_generation_validation()
        }).await;
        test_cases.push(test_result);
        
        let end_time = chrono::Utc::now().timestamp();
        let suite_duration = suite_start.elapsed();
        
        let (passed, failed) = count_test_results(&test_cases);
        
        Ok(TestResults {
            test_suite: "pipeline_tests".to_string(),
            start_time,
            end_time,
            total_tests: test_cases.len() as u32,
            passed_tests: passed,
            failed_tests: failed,
            test_cases,
            performance_metrics: Some(self.performance_monitor.get_performance_report()),
            suite_duration_ms: suite_duration.as_millis() as u64,
        })
    }
    
    /// Run failure scenario tests
    pub async fn run_failure_tests(&mut self) -> Result<TestResults> {
        info!("Executing failure scenario test suite");
        let start_time = chrono::Utc::now().timestamp();
        let suite_start = Instant::now();
        
        let mut test_cases = Vec::new();
        
        // Test HMM service failures
        if self.config.failure_tests.test_hmm_failures {
            let test_result = self.execute_test_case("hmm_service_failure", || {
                self.test_hmm_service_failure()
            }).await;
            test_cases.push(test_result);
        }
        
        // Test Redis connection failures
        if self.config.failure_tests.test_redis_failures {
            let test_result = self.execute_test_case("redis_connection_failure", || {
                self.test_redis_connection_failure()
            }).await;
            test_cases.push(test_result);
        }
        
        // Test Kafka connection failures
        if self.config.failure_tests.test_kafka_failures {
            let test_result = self.execute_test_case("kafka_connection_failure", || {
                self.test_kafka_connection_failure()
            }).await;
            test_cases.push(test_result);
        }
        
        // Test data corruption scenarios
        if self.config.failure_tests.test_data_corruption {
            let test_result = self.execute_test_case("data_corruption_handling", || {
                self.test_data_corruption_handling()
            }).await;
            test_cases.push(test_result);
        }
        
        let end_time = chrono::Utc::now().timestamp();
        let suite_duration = suite_start.elapsed();
        
        let (passed, failed) = count_test_results(&test_cases);
        
        Ok(TestResults {
            test_suite: "failure_tests".to_string(),
            start_time,
            end_time,
            total_tests: test_cases.len() as u32,
            passed_tests: passed,
            failed_tests: failed,
            test_cases,
            performance_metrics: None,
            suite_duration_ms: suite_duration.as_millis() as u64,
        })
    }
    
    /// Run performance validation tests
    pub async fn run_performance_tests(&mut self) -> Result<TestResults> {
        info!("Executing performance validation test suite");
        let start_time = chrono::Utc::now().timestamp();
        let suite_start = Instant::now();
        
        let mut test_cases = Vec::new();
        
        // Test end-to-end latency
        let test_result = self.execute_test_case("end_to_end_latency", || {
            self.test_end_to_end_latency()
        }).await;
        test_cases.push(test_result);
        
        // Test concurrent processing
        let test_result = self.execute_test_case("concurrent_processing", || {
            self.test_concurrent_processing()
        }).await;
        test_cases.push(test_result);
        
        // Test throughput validation
        let test_result = self.execute_test_case("throughput_validation", || {
            self.test_throughput_validation()
        }).await;
        test_cases.push(test_result);
        
        let end_time = chrono::Utc::now().timestamp();
        let suite_duration = suite_start.elapsed();
        
        let (passed, failed) = count_test_results(&test_cases);
        
        Ok(TestResults {
            test_suite: "performance_tests".to_string(),
            start_time,
            end_time,
            total_tests: test_cases.len() as u32,
            passed_tests: passed,
            failed_tests: failed,
            test_cases,
            performance_metrics: Some(self.performance_monitor.get_performance_report()),
            suite_duration_ms: suite_duration.as_millis() as u64,
        })
    }
    
    /// Execute a single test case with proper error handling and timing
    async fn execute_test_case<F, Fut>(&self, test_name: &str, test_fn: F) -> TestCaseResult
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<HashMap<String, f64>>>,
    {
        let _permit = self.execution_semaphore.acquire().await.unwrap();
        let test_id = Uuid::new_v4();
        let start_time = Instant::now();
        
        debug!("Starting test case: {} (ID: {})", test_name, test_id);
        
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.execution.test_timeout_seconds),
            test_fn()
        ).await;
        
        let duration = start_time.elapsed();
        
        match result {
            Ok(Ok(metrics)) => {
                debug!("Test case {} completed successfully in {:.2}ms", test_name, duration.as_millis());
                TestCaseResult {
                    name: test_name.to_string(),
                    id: test_id,
                    status: TestStatus::Passed,
                    duration_ms: duration.as_millis() as u64,
                    error_message: None,
                    metrics,
                    validation_details: Vec::new(),
                }
            }
            Ok(Err(e)) => {
                warn!("Test case {} failed: {}", test_name, e);
                TestCaseResult {
                    name: test_name.to_string(),
                    id: test_id,
                    status: TestStatus::Failed,
                    duration_ms: duration.as_millis() as u64,
                    error_message: Some(e.to_string()),
                    metrics: HashMap::new(),
                    validation_details: Vec::new(),
                }
            }
            Err(_) => {
                warn!("Test case {} timed out after {}s", test_name, self.config.execution.test_timeout_seconds);
                TestCaseResult {
                    name: test_name.to_string(),
                    id: test_id,
                    status: TestStatus::Timeout,
                    duration_ms: duration.as_millis() as u64,
                    error_message: Some(format!("Test timed out after {}s", self.config.execution.test_timeout_seconds)),
                    metrics: HashMap::new(),
                    validation_details: Vec::new(),
                }
            }
        }
    }
    
    /// Generate comprehensive test report (public method)
    pub async fn generate_report(&self, results: &[TestResults], overall_duration: std::time::Duration) -> Result<TestReport> {
        self.generate_report_internal(results, overall_duration).await
    }
    
    /// Generate comprehensive test report (internal implementation)
    async fn generate_report_internal(&self, results: &[TestResults], overall_duration: std::time::Duration) -> Result<TestReport> {
        let total_tests: u32 = results.iter().map(|r| r.total_tests).sum();
        let total_passed: u32 = results.iter().map(|r| r.passed_tests).sum();
        let _total_failed: u32 = results.iter().map(|r| r.failed_tests).sum();
        
        let overall_pass_rate = if total_tests > 0 {
            total_passed as f64 / total_tests as f64
        } else {
            0.0
        };
        
        let critical_failures = results.iter()
            .map(|r| r.test_cases.iter().filter(|tc| tc.status == TestStatus::Failed).count() as u32)
            .sum();
        
        let performance_violations = results.iter()
            .filter_map(|r| r.performance_metrics.as_ref())
            .map(|pm| self.count_performance_violations(pm))
            .sum();
        
        let system_health_score = self.calculate_system_health_score(results);
        
        let summary = TestSummary {
            total_duration_minutes: overall_duration.as_secs_f64() / 60.0,
            overall_pass_rate,
            critical_failures,
            performance_violations,
            system_health_score,
        };
        
        let recommendations = self.generate_recommendations(results);
        
        Ok(TestReport {
            session_id: self.session_id,
            summary,
            results: results.to_vec(),
            recommendations,
            generated_at: chrono::Utc::now().timestamp(),
        })
    }
    
    // Placeholder test implementations - these will be implemented in subsequent tasks
    async fn test_complete_signal_pipeline(&self, _symbol: &str) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 2.1
        Ok(HashMap::new())
    }
    
    async fn test_feature_computation_accuracy(&self) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 2.2
        Ok(HashMap::new())
    }
    
    async fn test_signal_generation_validation(&self) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 2.3
        Ok(HashMap::new())
    }
    
    async fn test_hmm_service_failure(&self) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 3.2
        Ok(HashMap::new())
    }
    
    async fn test_redis_connection_failure(&self) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 3.3
        Ok(HashMap::new())
    }
    
    async fn test_kafka_connection_failure(&self) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 3.3
        Ok(HashMap::new())
    }
    
    async fn test_data_corruption_handling(&self) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 3.1
        Ok(HashMap::new())
    }
    
    async fn test_end_to_end_latency(&self) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 4.1
        Ok(HashMap::new())
    }
    
    async fn test_concurrent_processing(&self) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 4.2
        Ok(HashMap::new())
    }
    
    async fn test_throughput_validation(&self) -> Result<HashMap<String, f64>> {
        // TODO: Implement in task 4.3
        Ok(HashMap::new())
    }
    
    fn count_performance_violations(&self, _performance_metrics: &crate::performance::PerformanceReport) -> u32 {
        // TODO: Implement performance violation counting
        0
    }
    
    fn calculate_system_health_score(&self, results: &[TestResults]) -> f64 {
        let total_tests: u32 = results.iter().map(|r| r.total_tests).sum();
        let total_passed: u32 = results.iter().map(|r| r.passed_tests).sum();
        
        if total_tests == 0 {
            return 0.0;
        }
        
        // Base score from pass rate
        let pass_rate = total_passed as f64 / total_tests as f64;
        
        // Adjust for critical failures and timeouts
        let critical_penalty = results.iter()
            .flat_map(|r| &r.test_cases)
            .filter(|tc| matches!(tc.status, TestStatus::Failed | TestStatus::Timeout))
            .count() as f64 * 0.1;
        
        (pass_rate - critical_penalty).max(0.0).min(1.0)
    }
    
    fn generate_recommendations(&self, results: &[TestResults]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Check overall pass rate
        let total_tests: u32 = results.iter().map(|r| r.total_tests).sum();
        let total_passed: u32 = results.iter().map(|r| r.passed_tests).sum();
        
        if total_tests > 0 {
            let pass_rate = total_passed as f64 / total_tests as f64;
            if pass_rate < 0.9 {
                recommendations.push("Overall test pass rate is below 90%. Review failed tests and address underlying issues.".to_string());
            }
        }
        
        // Check for timeout issues
        let timeout_count = results.iter()
            .flat_map(|r| &r.test_cases)
            .filter(|tc| tc.status == TestStatus::Timeout)
            .count();
        
        if timeout_count > 0 {
            recommendations.push(format!("{} tests timed out. Consider increasing timeout values or optimizing test performance.", timeout_count));
        }
        
        // Check performance metrics
        for result in results {
            if let Some(perf_metrics) = &result.performance_metrics {
                if perf_metrics.end_to_end_latency.mean > self.config.performance_tests.max_end_to_end_latency_ms as f64 {
                    recommendations.push("End-to-end latency exceeds requirements. Review pipeline performance.".to_string());
                }
            }
        }
        
        if recommendations.is_empty() {
            recommendations.push("All tests are performing well. No immediate action required.".to_string());
        }
        
        recommendations
    }
}

/// Count passed and failed tests from a list of test case results
fn count_test_results(test_cases: &[TestCaseResult]) -> (u32, u32) {
    let passed = test_cases.iter().filter(|tc| tc.status == TestStatus::Passed).count() as u32;
    let failed = test_cases.iter().filter(|tc| tc.status != TestStatus::Passed).count() as u32;
    (passed, failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_harness_creation() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        assert_eq!(harness.config().pipeline_tests.test_symbols.len(), 2);
    }
    
    #[tokio::test]
    async fn test_harness_from_config_file() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        let config = TestConfig::default();
        config.save_to_file(&config_path).unwrap();
        
        let harness = TestHarness::from_config_file(&config_path).await.unwrap();
        assert_eq!(harness.config().pipeline_tests.test_symbols.len(), 2);
    }
    
    #[test]
    fn test_count_test_results() {
        let test_cases = vec![
            TestCaseResult {
                name: "test1".to_string(),
                id: Uuid::new_v4(),
                status: TestStatus::Passed,
                duration_ms: 100,
                error_message: None,
                metrics: HashMap::new(),
                validation_details: Vec::new(),
            },
            TestCaseResult {
                name: "test2".to_string(),
                id: Uuid::new_v4(),
                status: TestStatus::Failed,
                duration_ms: 200,
                error_message: Some("Test failed".to_string()),
                metrics: HashMap::new(),
                validation_details: Vec::new(),
            },
        ];
        
        let (passed, failed) = count_test_results(&test_cases);
        assert_eq!(passed, 1);
        assert_eq!(failed, 1);
    }
}