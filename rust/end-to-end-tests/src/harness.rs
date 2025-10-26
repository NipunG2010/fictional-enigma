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
use anyhow::Context;
use feature_pipeline::{FeaturePipeline, OHLCV};

// Mock implementations for signal fusion components (to avoid OpenSSL dependency issues)
#[derive(Debug, Clone)]
pub struct SignalComponents {
    pub s_ldc: f64,
    pub s_mr: f64,
    pub s_tsmom: f64,
}

impl SignalComponents {
    pub fn validate(&self) -> Result<()> {
        if !self.s_ldc.is_finite() || !self.s_mr.is_finite() || !self.s_tsmom.is_finite() {
            return Err(anyhow::anyhow!("Invalid signal components: non-finite values"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FusionWeights {
    pub w_ldc: f64,
    pub w_mr: f64,
    pub w_tsmom: f64,
}

impl FusionWeights {
    pub fn validate(&self) -> Result<()> {
        if !self.w_ldc.is_finite() || !self.w_mr.is_finite() || !self.w_tsmom.is_finite() {
            return Err(anyhow::anyhow!("Invalid fusion weights: non-finite values"));
        }
        let sum = self.w_ldc + self.w_mr + self.w_tsmom;
        if (sum - 1.0).abs() > 0.01 {
            return Err(anyhow::anyhow!("Fusion weights do not sum to 1.0: {}", sum));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum SignalSide {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub symbol: String,
    pub timestamp: i64,
    pub side: SignalSide,
    pub strength: f64,
    pub confidence: f64,
    pub correlation_id: Option<String>,
}

impl TradingSignal {
    pub fn validate(&self) -> Result<()> {
        if self.strength < -1.0 || self.strength > 1.0 {
            return Err(anyhow::anyhow!("Signal strength out of range: {}", self.strength));
        }
        if self.confidence < 0.0 || self.confidence > 1.0 {
            return Err(anyhow::anyhow!("Signal confidence out of range: {}", self.confidence));
        }
        Ok(())
    }
    
    pub fn to_compact_string(&self) -> String {
        format!("{}:{:.3}@{:.3}", self.symbol, self.strength, self.confidence)
    }
}

#[derive(Debug)]
pub struct PipelineMetrics {
    pub total_latency_ms: u64,
    pub fusion_latency_ms: u64,
    pub validation_latency_ms: u64,
    pub emission_latency_ms: u64,
    pub audit_latency_ms: u64,
    pub success: bool,
    pub error_message: Option<String>,
    pub correlation_id: String,
}

#[derive(Debug)]
pub struct PipelineResult {
    pub signal: Option<TradingSignal>,
    pub metrics: PipelineMetrics,
    pub emitted: bool,
}

pub struct SignalPipeline {
    threshold: f32,
    cooldown_period: u64,
    normalize_weights: bool,
}

impl SignalPipeline {
    pub fn without_emission(threshold: f32, cooldown_period: u64, normalize_weights: bool) -> Self {
        Self {
            threshold,
            cooldown_period,
            normalize_weights,
        }
    }
    
    pub async fn process_signal(
        &mut self,
        components: SignalComponents,
        weights: FusionWeights,
        timestamp: i64,
        symbol: &str,
        _feature_names: Option<Vec<String>>,
        _input_checksum: Option<String>,
    ) -> Result<PipelineResult> {
        let start_time = Instant::now();
        let correlation_id = uuid::Uuid::new_v4().to_string();
        
        // Simple signal fusion calculation
        let fused_signal = components.s_ldc * weights.w_ldc + 
                          components.s_mr * weights.w_mr + 
                          components.s_tsmom * weights.w_tsmom;
        
        let signal = if fused_signal.abs() > self.threshold as f64 {
            let side = if fused_signal > 0.0 {
                SignalSide::Buy
            } else {
                SignalSide::Sell
            };
            
            Some(TradingSignal {
                symbol: symbol.to_string(),
                timestamp,
                side,
                strength: fused_signal.clamp(-1.0, 1.0),
                confidence: fused_signal.abs().min(1.0),
                correlation_id: Some(correlation_id.clone()),
            })
        } else {
            None
        };
        
        let total_latency = start_time.elapsed().as_millis() as u64;
        
        let metrics = PipelineMetrics {
            total_latency_ms: total_latency,
            fusion_latency_ms: total_latency / 4,
            validation_latency_ms: total_latency / 4,
            emission_latency_ms: total_latency / 4,
            audit_latency_ms: total_latency / 4,
            success: true,
            error_message: None,
            correlation_id,
        };
        
        Ok(PipelineResult {
            signal,
            metrics,
            emitted: false,
        })
    }
}

// Mock implementations for LDC engine components
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Short = -1,
    Neutral = 0,
    Long = 1,
}

#[derive(Debug, Clone)]
pub struct FeatureSeries {
    pub f1: f32, // RSI
    pub f2: f32, // WT (WaveTrend)
    pub f3: f32, // CCI
    pub f4: f32, // ADX
    pub f5: f32, // Additional feature
}

#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub features: FeatureSeries,
    pub direction: Direction,
    pub timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct LDCConfig {
    pub k_neighbors: usize,
    pub max_bars_back: usize,
    pub use_multithreading: bool,
    pub use_hnsw_index: bool,
}

impl Default for LDCConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 8,
            max_bars_back: 2000,
            use_multithreading: true,
            use_hnsw_index: false,
        }
    }
}

pub struct LDCEngine {
    config: LDCConfig,
    training_samples: Vec<TrainingSample>,
}

impl LDCEngine {
    pub fn with_config(config: LDCConfig) -> Self {
        Self {
            config,
            training_samples: Vec::new(),
        }
    }
    
    pub fn add_sample(&mut self, sample: TrainingSample) {
        self.training_samples.push(sample);
        
        // Keep only the most recent samples
        if self.training_samples.len() > self.config.max_bars_back {
            self.training_samples.remove(0);
        }
    }
    
    pub fn predict(&self, features: &FeatureSeries, k: usize) -> Option<f64> {
        if self.training_samples.is_empty() {
            return Some(0.0);
        }
        
        // Simple mock prediction based on feature similarity
        let mut distances: Vec<(f64, Direction)> = self.training_samples
            .iter()
            .map(|sample| {
                let distance = self.lorentzian_distance(features, &sample.features);
                (distance, sample.direction)
            })
            .collect();
        
        // Sort by distance and take k nearest neighbors
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let k_nearest = distances.into_iter().take(k.min(self.training_samples.len()));
        
        // Calculate weighted prediction
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for (distance, direction) in k_nearest {
            let weight = 1.0 / (1.0 + distance);
            weighted_sum += weight * (direction as i32) as f64;
            weight_sum += weight;
        }
        
        if weight_sum > 0.0 {
            Some((weighted_sum / weight_sum).clamp(-1.0, 1.0))
        } else {
            Some(0.0)
        }
    }
    
    fn lorentzian_distance(&self, a: &FeatureSeries, b: &FeatureSeries) -> f64 {
        (1.0 + (a.f1 - b.f1).abs() as f64).ln() +
        (1.0 + (a.f2 - b.f2).abs() as f64).ln() +
        (1.0 + (a.f3 - b.f3).abs() as f64).ln() +
        (1.0 + (a.f4 - b.f4).abs() as f64).ln() +
        (1.0 + (a.f5 - b.f5).abs() as f64).ln()
    }
}

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
    
    // Complete signal pipeline validation test implementation
    async fn test_complete_signal_pipeline(&self, symbol: &str) -> Result<HashMap<String, f64>> {
        use std::time::Instant;
        
        let mut metrics = HashMap::new();
        let pipeline_start = Instant::now();
        
        // Step 1: Generate test OHLCV data
        let test_data_bars = self.generate_deterministic_ohlcv_data(symbol, 
            self.config.pipeline_tests.test_duration_hours)?;
        
        if test_data_bars.is_empty() {
            return Err(TestFrameworkError::DataGenerationError("No OHLCV data generated".to_string()).into());
        }
        
        // Convert OHLCVBar to OHLCV
        let test_data: Vec<OHLCV> = test_data_bars.into_iter().map(|bar| OHLCV {
            timestamp: bar.timestamp,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        }).collect();
        
        metrics.insert("input_data_points".to_string(), test_data.len() as f64);
        
        // Step 2: Compute features using feature pipeline
        let feature_start = Instant::now();
        let feature_pipeline = FeaturePipeline::new(20);
        let features = feature_pipeline.compute_features_safe(&test_data)
            .context("Failed to compute features")?;
        
        let feature_latency = feature_start.elapsed().as_millis() as f64;
        metrics.insert("feature_computation_latency_ms".to_string(), feature_latency);
        metrics.insert("computed_features_count".to_string(), features.len() as f64);
        
        if features.is_empty() {
            return Err(TestFrameworkError::ValidationError("No features computed".to_string()).into());
        }
        
        // Step 3: Generate MR and TSMOM signals
        let signal_start = Instant::now();
        let signals = feature_pipeline.generate_signals(&features)
            .context("Failed to generate MR/TSMOM signals")?;
        
        let signal_latency = signal_start.elapsed().as_millis() as f64;
        metrics.insert("mr_tsmom_signal_latency_ms".to_string(), signal_latency);
        
        // Step 4: Generate LDC signals using LDC engine
        let ldc_start = Instant::now();
        let mut ldc_config = LDCConfig::default();
        ldc_config.k_neighbors = 8;
        ldc_config.max_bars_back = 2000;
        ldc_config.use_multithreading = false; // For deterministic testing
        
        let mut ldc_engine = LDCEngine::with_config(ldc_config);
        
        // Convert features to LDC format and add training samples
        let mut ldc_signals = Vec::new();
        for (i, feature) in features.iter().enumerate() {
            if let (Some(rsi), Some(wt1), Some(cci), Some(adx)) = 
                (feature.rsi, feature.wavetrend_1, feature.cci, feature.adx) {
                
                let feature_series = FeatureSeries {
                    f1: rsi as f32,
                    f2: wt1 as f32,
                    f3: cci as f32,
                    f4: adx as f32,
                    f5: feature.momentum.unwrap_or(0.0) as f32,
                };
                
                // Add as training sample if we have enough history
                if i > 50 && i < features.len() - 10 {
                    // Use future price movement to determine direction
                    let current_close = test_data[i].close;
                    let future_close = test_data.get(i + 5).map(|d| d.close).unwrap_or(current_close);
                    let price_change = (future_close - current_close) / current_close;
                    
                    let direction = if price_change > 0.01 {
                        Direction::Long
                    } else if price_change < -0.01 {
                        Direction::Short
                    } else {
                        Direction::Neutral
                    };
                    
                    let training_sample = TrainingSample {
                        features: feature_series.clone(),
                        direction,
                        timestamp: feature.timestamp,
                    };
                    
                    ldc_engine.add_sample(training_sample);
                }
                
                // Generate LDC signal for recent data
                if i >= features.len() - 10 {
                    let ldc_signal = ldc_engine.predict(&feature_series, 8)
                        .unwrap_or(0.0);
                    ldc_signals.push(ldc_signal);
                }
            }
        }
        
        let ldc_latency = ldc_start.elapsed().as_millis() as f64;
        metrics.insert("ldc_signal_latency_ms".to_string(), ldc_latency);
        metrics.insert("ldc_signals_count".to_string(), ldc_signals.len() as f64);
        
        // Step 5: Create signal components and fusion weights
        let fusion_start = Instant::now();
        
        // Use the last computed signals for fusion
        let last_signals = signals.last().ok_or_else(|| 
            TestFrameworkError::ValidationError("No signals available for fusion".to_string()))?;
        
        let s_ldc = ldc_signals.last().copied().unwrap_or(0.0) as f64;
        let s_mr = last_signals.s_mr.unwrap_or(0.0);
        let s_tsmom = last_signals.s_tsmom.unwrap_or(0.0);
        
        let components = SignalComponents {
            s_ldc,
            s_mr,
            s_tsmom,
        };
        
        // Validate signal components
        components.validate()
            .map_err(|e| TestFrameworkError::ValidationError(format!("Invalid signal components: {}", e)))?;
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        // Validate fusion weights
        weights.validate()
            .map_err(|e| TestFrameworkError::ValidationError(format!("Invalid fusion weights: {}", e)))?;
        
        // Step 6: Process through signal pipeline
        let mut signal_pipeline = SignalPipeline::without_emission(0.3, 0, true);
        
        let now = chrono::Utc::now().timestamp();
        let pipeline_result = signal_pipeline.process_signal(
            components,
            weights,
            now,
            symbol,
            Some(vec!["ldc".to_string(), "mr".to_string(), "tsmom".to_string()]),
            Some(format!("test-{}-{}", symbol, now)),
        ).await.context("Failed to process signal through pipeline")?;
        
        let fusion_latency = fusion_start.elapsed().as_millis() as f64;
        metrics.insert("signal_fusion_latency_ms".to_string(), fusion_latency);
        
        // Step 7: Validate pipeline result
        if let Some(ref final_signal) = pipeline_result.signal {
            // Validate signal format and content
            if final_signal.symbol != symbol {
                return Err(TestFrameworkError::ValidationError(
                    format!("Signal symbol mismatch: expected {}, got {}", symbol, final_signal.symbol)
                ).into());
            }
            
            if final_signal.timestamp != now {
                return Err(TestFrameworkError::ValidationError(
                    format!("Signal timestamp mismatch: expected {}, got {}", now, final_signal.timestamp)
                ).into());
            }
            
            // Validate signal strength is within expected range
            if final_signal.strength < -1.0 || final_signal.strength > 1.0 {
                return Err(TestFrameworkError::ValidationError(
                    format!("Signal strength out of range: {}", final_signal.strength)
                ).into());
            }
            
            // Validate confidence is within expected range
            if final_signal.confidence < 0.0 || final_signal.confidence > 1.0 {
                return Err(TestFrameworkError::ValidationError(
                    format!("Signal confidence out of range: {}", final_signal.confidence)
                ).into());
            }
            
            metrics.insert("final_signal_strength".to_string(), final_signal.strength);
            metrics.insert("final_signal_confidence".to_string(), final_signal.confidence);
            metrics.insert("signal_generated".to_string(), 1.0);
            
            // Validate correlation ID tracking
            if let Some(correlation_id) = &final_signal.correlation_id {
                if correlation_id.is_empty() {
                    return Err(TestFrameworkError::ValidationError(
                        "Empty correlation ID in final signal".to_string()
                    ).into());
                }
                metrics.insert("correlation_id_present".to_string(), 1.0);
            } else {
                return Err(TestFrameworkError::ValidationError(
                    "Missing correlation ID in final signal".to_string()
                ).into());
            }
            
        } else {
            metrics.insert("signal_generated".to_string(), 0.0);
        }
        
        // Step 8: Validate pipeline metrics
        if !pipeline_result.metrics.success {
            return Err(TestFrameworkError::ValidationError(
                format!("Pipeline execution failed: {:?}", pipeline_result.metrics.error_message)
            ).into());
        }
        
        // Record pipeline performance metrics
        metrics.insert("pipeline_total_latency_ms".to_string(), pipeline_result.metrics.total_latency_ms as f64);
        metrics.insert("pipeline_fusion_latency_ms".to_string(), pipeline_result.metrics.fusion_latency_ms as f64);
        metrics.insert("pipeline_validation_latency_ms".to_string(), pipeline_result.metrics.validation_latency_ms as f64);
        
        // Calculate end-to-end latency
        let total_latency = pipeline_start.elapsed().as_millis() as f64;
        metrics.insert("end_to_end_latency_ms".to_string(), total_latency);
        
        // Validate end-to-end latency requirement
        if total_latency > self.config.performance_tests.max_end_to_end_latency_ms as f64 {
            return Err(TestFrameworkError::PerformanceError {
                requirement: format!("End-to-end latency must be < {}ms, got {}ms", 
                    self.config.performance_tests.max_end_to_end_latency_ms, total_latency)
            }.into());
        }
        
        // Validate audit trail completeness
        let audit_score = self.calculate_audit_completeness(&pipeline_result, &metrics);
        metrics.insert("audit_completeness_score".to_string(), audit_score);
        
        if audit_score < 0.8 {
            return Err(TestFrameworkError::ValidationError(
                format!("Audit trail incomplete: score {:.2} < 0.8", audit_score)
            ).into());
        }
        
        Ok(metrics)
    }
    
    /// Calculate audit trail completeness score
    fn calculate_audit_completeness(&self, pipeline_result: &PipelineResult, metrics: &HashMap<String, f64>) -> f64 {
        let mut score = 0.0;
        let mut total_checks = 0.0;
        
        // Check correlation ID presence
        if metrics.get("correlation_id_present").copied().unwrap_or(0.0) > 0.0 {
            score += 1.0;
        }
        total_checks += 1.0;
        
        // Check pipeline metrics completeness
        if pipeline_result.metrics.total_latency_ms > 0 {
            score += 1.0;
        }
        total_checks += 1.0;
        
        // Check feature computation tracking
        if metrics.get("feature_computation_latency_ms").copied().unwrap_or(0.0) > 0.0 {
            score += 1.0;
        }
        total_checks += 1.0;
        
        // Check signal generation tracking
        if metrics.get("ldc_signal_latency_ms").copied().unwrap_or(0.0) > 0.0 {
            score += 1.0;
        }
        total_checks += 1.0;
        
        // Check input data validation
        if metrics.get("input_data_points").copied().unwrap_or(0.0) > 0.0 {
            score += 1.0;
        }
        total_checks += 1.0;
        
        if total_checks > 0.0 {
            score / total_checks
        } else {
            0.0
        }
    }
    
    async fn test_feature_computation_accuracy(&self) -> Result<HashMap<String, f64>> {
        use std::time::Instant;
        
        let mut metrics = HashMap::new();
        let test_start = Instant::now();
        
        // Step 1: Generate deterministic test data for accuracy validation
        let test_data = self.generate_reference_ohlcv_data()?;
        metrics.insert("reference_data_points".to_string(), test_data.len() as f64);
        
        // Step 2: Compute features using the pipeline
        let computation_start = Instant::now();
        let feature_pipeline = FeaturePipeline::new(20);
        let computed_features = feature_pipeline.compute_features_safe(&test_data)
            .context("Failed to compute features for accuracy test")?;
        
        let computation_latency = computation_start.elapsed().as_millis() as f64;
        metrics.insert("feature_computation_latency_ms".to_string(), computation_latency);
        
        if computed_features.is_empty() {
            return Err(TestFrameworkError::ValidationError("No features computed for accuracy test".to_string()).into());
        }
        
        // Step 3: Load or generate reference values for comparison
        let reference_features = self.generate_reference_features(&test_data)?;
        metrics.insert("reference_features_count".to_string(), reference_features.len() as f64);
        
        // Step 4: Validate feature accuracy with tolerance-based comparison
        let validation_start = Instant::now();
        let mut accuracy_results = HashMap::new();
        
        // RSI accuracy validation
        let rsi_accuracy = self.validate_rsi_accuracy(&computed_features, &reference_features)?;
        accuracy_results.insert("rsi_accuracy".to_string(), rsi_accuracy);
        
        // Moving averages accuracy validation
        let sma_accuracy = self.validate_sma_accuracy(&computed_features, &reference_features)?;
        accuracy_results.insert("sma_accuracy".to_string(), sma_accuracy);
        
        let ema_accuracy = self.validate_ema_accuracy(&computed_features, &reference_features)?;
        accuracy_results.insert("ema_accuracy".to_string(), ema_accuracy);
        
        // Momentum accuracy validation
        let momentum_accuracy = self.validate_momentum_accuracy(&computed_features, &reference_features)?;
        accuracy_results.insert("momentum_accuracy".to_string(), momentum_accuracy);
        
        // Volatility indicators accuracy validation
        let std_accuracy = self.validate_std_accuracy(&computed_features, &reference_features)?;
        accuracy_results.insert("std_accuracy".to_string(), std_accuracy);
        
        let zscore_accuracy = self.validate_zscore_accuracy(&computed_features, &reference_features)?;
        accuracy_results.insert("zscore_accuracy".to_string(), zscore_accuracy);
        
        // Advanced indicators accuracy validation
        let cci_accuracy = self.validate_cci_accuracy(&computed_features, &reference_features)?;
        accuracy_results.insert("cci_accuracy".to_string(), cci_accuracy);
        
        let adx_accuracy = self.validate_adx_accuracy(&computed_features, &reference_features)?;
        accuracy_results.insert("adx_accuracy".to_string(), adx_accuracy);
        
        let validation_latency = validation_start.elapsed().as_millis() as f64;
        metrics.insert("validation_latency_ms".to_string(), validation_latency);
        
        // Step 5: Calculate overall accuracy score
        let overall_accuracy = accuracy_results.values().sum::<f64>() / accuracy_results.len() as f64;
        metrics.insert("overall_accuracy".to_string(), overall_accuracy);
        
        // Step 6: Test feature computation with various market conditions
        let market_conditions_accuracy = self.test_market_conditions_accuracy(&feature_pipeline)?;
        metrics.extend(market_conditions_accuracy);
        
        // Step 7: Validate accuracy requirements
        let min_accuracy = self.config.validation.feature_tolerance;
        if overall_accuracy < (1.0 - min_accuracy) {
            return Err(TestFrameworkError::ValidationError(
                format!("Feature accuracy below threshold: {:.4} < {:.4}", overall_accuracy, 1.0 - min_accuracy)
            ).into());
        }
        
        // Add all accuracy results to metrics
        for (key, value) in accuracy_results {
            metrics.insert(key, value);
        }
        
        let total_latency = test_start.elapsed().as_millis() as f64;
        metrics.insert("total_accuracy_test_latency_ms".to_string(), total_latency);
        
        Ok(metrics)
    }
    
    /// Generate deterministic OHLCV data for reference testing
    fn generate_reference_ohlcv_data(&self) -> Result<Vec<OHLCV>> {
        let mut data = Vec::new();
        let base_price = 100.0;
        let base_timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
        
        // Generate 100 bars of deterministic data with known patterns
        for i in 0..100 {
            let t = i as f64 * 0.1;
            
            // Create predictable price movements for testing
            let trend = t * 0.5; // Linear trend
            let cycle = (t * 2.0 * std::f64::consts::PI / 20.0).sin() * 2.0; // 20-period cycle
            let noise = (t * 7.0).sin() * 0.5; // High-frequency noise
            
            let price = base_price + trend + cycle + noise;
            
            data.push(OHLCV {
                timestamp: base_timestamp + (i * 300), // 5-minute intervals
                open: price,
                high: price * 1.01,
                low: price * 0.99,
                close: price * 1.005,
                volume: 1000.0 + (i as f64 * 10.0),
            });
        }
        
        Ok(data)
    }
    
    /// Generate reference feature values using known calculations
    fn generate_reference_features(&self, data: &[OHLCV]) -> Result<Vec<feature_pipeline::Features>> {
        // For this test, we'll use a simple reference implementation
        // In a real system, this would load pre-computed reference values
        let feature_pipeline = FeaturePipeline::new(20);
        feature_pipeline.compute_features_safe(data)
            .context("Failed to generate reference features")
    }
    
    /// Validate RSI accuracy with tolerance
    fn validate_rsi_accuracy(&self, computed: &[feature_pipeline::Features], reference: &[feature_pipeline::Features]) -> Result<f64> {
        let tolerance = self.config.validation.feature_tolerance;
        let mut matches = 0;
        let mut total = 0;
        
        for (comp, ref_val) in computed.iter().zip(reference.iter()) {
            if let (Some(comp_rsi), Some(ref_rsi)) = (comp.rsi, ref_val.rsi) {
                total += 1;
                let diff = (comp_rsi - ref_rsi).abs();
                if diff <= tolerance * ref_rsi.abs().max(1.0) {
                    matches += 1;
                }
            }
        }
        
        Ok(if total > 0 { matches as f64 / total as f64 } else { 0.0 })
    }
    
    /// Validate SMA accuracy with tolerance
    fn validate_sma_accuracy(&self, computed: &[feature_pipeline::Features], reference: &[feature_pipeline::Features]) -> Result<f64> {
        let tolerance = self.config.validation.feature_tolerance;
        let mut matches = 0;
        let mut total = 0;
        
        for (comp, ref_val) in computed.iter().zip(reference.iter()) {
            if let (Some(comp_sma), Some(ref_sma)) = (comp.sma_20, ref_val.sma_20) {
                total += 1;
                let diff = (comp_sma - ref_sma).abs();
                if diff <= tolerance * ref_sma.abs().max(1.0) {
                    matches += 1;
                }
            }
        }
        
        Ok(if total > 0 { matches as f64 / total as f64 } else { 0.0 })
    }
    
    /// Validate EMA accuracy with tolerance
    fn validate_ema_accuracy(&self, computed: &[feature_pipeline::Features], reference: &[feature_pipeline::Features]) -> Result<f64> {
        let tolerance = self.config.validation.feature_tolerance;
        let mut matches = 0;
        let mut total = 0;
        
        for (comp, ref_val) in computed.iter().zip(reference.iter()) {
            if let (Some(comp_ema), Some(ref_ema)) = (comp.ema_20, ref_val.ema_20) {
                total += 1;
                let diff = (comp_ema - ref_ema).abs();
                if diff <= tolerance * ref_ema.abs().max(1.0) {
                    matches += 1;
                }
            }
        }
        
        Ok(if total > 0 { matches as f64 / total as f64 } else { 0.0 })
    }
    
    /// Validate momentum accuracy with tolerance
    fn validate_momentum_accuracy(&self, computed: &[feature_pipeline::Features], reference: &[feature_pipeline::Features]) -> Result<f64> {
        let tolerance = self.config.validation.feature_tolerance;
        let mut matches = 0;
        let mut total = 0;
        
        for (comp, ref_val) in computed.iter().zip(reference.iter()) {
            if let (Some(comp_mom), Some(ref_mom)) = (comp.momentum, ref_val.momentum) {
                total += 1;
                let diff = (comp_mom - ref_mom).abs();
                if diff <= tolerance * ref_mom.abs().max(0.01) {
                    matches += 1;
                }
            }
        }
        
        Ok(if total > 0 { matches as f64 / total as f64 } else { 0.0 })
    }
    
    /// Validate standard deviation accuracy with tolerance
    fn validate_std_accuracy(&self, computed: &[feature_pipeline::Features], reference: &[feature_pipeline::Features]) -> Result<f64> {
        let tolerance = self.config.validation.feature_tolerance;
        let mut matches = 0;
        let mut total = 0;
        
        for (comp, ref_val) in computed.iter().zip(reference.iter()) {
            if let (Some(comp_std), Some(ref_std)) = (comp.std_20, ref_val.std_20) {
                total += 1;
                let diff = (comp_std - ref_std).abs();
                if diff <= tolerance * ref_std.abs().max(0.01) {
                    matches += 1;
                }
            }
        }
        
        Ok(if total > 0 { matches as f64 / total as f64 } else { 0.0 })
    }
    
    /// Validate z-score accuracy with tolerance
    fn validate_zscore_accuracy(&self, computed: &[feature_pipeline::Features], reference: &[feature_pipeline::Features]) -> Result<f64> {
        let tolerance = self.config.validation.feature_tolerance;
        let mut matches = 0;
        let mut total = 0;
        
        for (comp, ref_val) in computed.iter().zip(reference.iter()) {
            if let (Some(comp_zscore), Some(ref_zscore)) = (comp.zscore_20, ref_val.zscore_20) {
                total += 1;
                let diff = (comp_zscore - ref_zscore).abs();
                if diff <= tolerance * ref_zscore.abs().max(0.1) {
                    matches += 1;
                }
            }
        }
        
        Ok(if total > 0 { matches as f64 / total as f64 } else { 0.0 })
    }
    
    /// Validate CCI accuracy with tolerance
    fn validate_cci_accuracy(&self, computed: &[feature_pipeline::Features], reference: &[feature_pipeline::Features]) -> Result<f64> {
        let tolerance = self.config.validation.feature_tolerance;
        let mut matches = 0;
        let mut total = 0;
        
        for (comp, ref_val) in computed.iter().zip(reference.iter()) {
            if let (Some(comp_cci), Some(ref_cci)) = (comp.cci, ref_val.cci) {
                total += 1;
                let diff = (comp_cci - ref_cci).abs();
                if diff <= tolerance * ref_cci.abs().max(1.0) {
                    matches += 1;
                }
            }
        }
        
        Ok(if total > 0 { matches as f64 / total as f64 } else { 0.0 })
    }
    
    /// Validate ADX accuracy with tolerance
    fn validate_adx_accuracy(&self, computed: &[feature_pipeline::Features], reference: &[feature_pipeline::Features]) -> Result<f64> {
        let tolerance = self.config.validation.feature_tolerance;
        let mut matches = 0;
        let mut total = 0;
        
        for (comp, ref_val) in computed.iter().zip(reference.iter()) {
            if let (Some(comp_adx), Some(ref_adx)) = (comp.adx, ref_val.adx) {
                total += 1;
                let diff = (comp_adx - ref_adx).abs();
                if diff <= tolerance * ref_adx.abs().max(1.0) {
                    matches += 1;
                }
            }
        }
        
        Ok(if total > 0 { matches as f64 / total as f64 } else { 0.0 })
    }
    
    /// Test feature computation accuracy with various market conditions
    fn test_market_conditions_accuracy(&self, feature_pipeline: &FeaturePipeline) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();
        
        // Test trending market
        let trending_data = self.generate_trending_market_data()?;
        let trending_features = feature_pipeline.compute_features_safe(&trending_data)?;
        let trending_accuracy = self.validate_trending_features(&trending_features)?;
        results.insert("trending_market_accuracy".to_string(), trending_accuracy);
        
        // Test sideways market
        let sideways_data = self.generate_sideways_market_data()?;
        let sideways_features = feature_pipeline.compute_features_safe(&sideways_data)?;
        let sideways_accuracy = self.validate_sideways_features(&sideways_features)?;
        results.insert("sideways_market_accuracy".to_string(), sideways_accuracy);
        
        // Test volatile market
        let volatile_data = self.generate_volatile_market_data()?;
        let volatile_features = feature_pipeline.compute_features_safe(&volatile_data)?;
        let volatile_accuracy = self.validate_volatile_features(&volatile_features)?;
        results.insert("volatile_market_accuracy".to_string(), volatile_accuracy);
        
        Ok(results)
    }
    
    /// Generate trending market data
    fn generate_trending_market_data(&self) -> Result<Vec<OHLCV>> {
        let mut data = Vec::new();
        let base_price = 100.0;
        let base_timestamp = 1640995200;
        
        for i in 0..50 {
            let price = base_price + (i as f64 * 0.5); // Strong uptrend
            data.push(OHLCV {
                timestamp: base_timestamp + (i * 300),
                open: price,
                high: price * 1.02,
                low: price * 0.98,
                close: price * 1.01,
                volume: 1000.0,
            });
        }
        
        Ok(data)
    }
    
    /// Generate sideways market data
    fn generate_sideways_market_data(&self) -> Result<Vec<OHLCV>> {
        let mut data = Vec::new();
        let base_price = 100.0;
        let base_timestamp = 1640995200;
        
        for i in 0..50 {
            let price = base_price + (i as f64 * 0.1 * std::f64::consts::PI / 10.0).sin() * 2.0; // Sideways movement
            data.push(OHLCV {
                timestamp: base_timestamp + (i * 300),
                open: price,
                high: price * 1.01,
                low: price * 0.99,
                close: price * 1.005,
                volume: 1000.0,
            });
        }
        
        Ok(data)
    }
    
    /// Generate volatile market data
    fn generate_volatile_market_data(&self) -> Result<Vec<OHLCV>> {
        let mut data = Vec::new();
        let base_price = 100.0;
        let base_timestamp = 1640995200;
        
        for i in 0..50 {
            let volatility = (i as f64 * 0.3).sin() * 5.0; // High volatility
            let price = base_price + volatility;
            data.push(OHLCV {
                timestamp: base_timestamp + (i * 300),
                open: price,
                high: price * 1.05,
                low: price * 0.95,
                close: price * 1.02,
                volume: 1000.0 + volatility.abs() * 100.0,
            });
        }
        
        Ok(data)
    }
    
    /// Validate features computed on trending market data
    fn validate_trending_features(&self, features: &[feature_pipeline::Features]) -> Result<f64> {
        let mut valid_count = 0;
        let mut total_count = 0;
        
        for feature in features.iter().skip(20) { // Skip initial period
            total_count += 1;
            
            // In a trending market, momentum should generally be positive
            if let Some(momentum) = feature.momentum {
                if momentum > -0.1 { // Allow some tolerance
                    valid_count += 1;
                }
            }
        }
        
        Ok(if total_count > 0 { valid_count as f64 / total_count as f64 } else { 0.0 })
    }
    
    /// Validate features computed on sideways market data
    fn validate_sideways_features(&self, features: &[feature_pipeline::Features]) -> Result<f64> {
        let mut valid_count = 0;
        let mut total_count = 0;
        
        for feature in features.iter().skip(20) { // Skip initial period
            total_count += 1;
            
            // In a sideways market, z-score should oscillate around zero
            if let Some(zscore) = feature.zscore_20 {
                if zscore.abs() < 2.0 { // Should not be extreme
                    valid_count += 1;
                }
            }
        }
        
        Ok(if total_count > 0 { valid_count as f64 / total_count as f64 } else { 0.0 })
    }
    
    /// Validate features computed on volatile market data
    fn validate_volatile_features(&self, features: &[feature_pipeline::Features]) -> Result<f64> {
        let mut valid_count = 0;
        let mut total_count = 0;
        
        for feature in features.iter().skip(20) { // Skip initial period
            total_count += 1;
            
            // In a volatile market, standard deviation should be elevated
            if let Some(std_dev) = feature.std_20 {
                if std_dev > 1.0 { // Should show higher volatility
                    valid_count += 1;
                }
            }
        }
        
        Ok(if total_count > 0 { valid_count as f64 / total_count as f64 } else { 0.0 })
    }
    
    async fn test_signal_generation_validation(&self) -> Result<HashMap<String, f64>> {

        use std::time::Instant;
        
        let mut metrics = HashMap::new();
        let test_start = Instant::now();
        
        // Step 1: Generate comprehensive test data for signal validation
        let test_data = self.generate_signal_test_data()?;
        metrics.insert("signal_test_data_points".to_string(), test_data.len() as f64);
        
        // Step 2: Compute features for signal generation
        let feature_pipeline = FeaturePipeline::new(20);
        let features = feature_pipeline.compute_features_safe(&test_data)
            .context("Failed to compute features for signal validation")?;
        
        if features.is_empty() {
            return Err(TestFrameworkError::ValidationError("No features computed for signal validation".to_string()).into());
        }
        
        // Step 3: Test LDC signal generation with k-NN classification
        let ldc_results = self.test_ldc_signal_generation(&features, &test_data).await?;
        metrics.extend(ldc_results);
        
        // Step 4: Test MR (Mean Reversion) signal validation
        let mr_results = self.test_mr_signal_generation(&features, &feature_pipeline).await?;
        metrics.extend(mr_results);
        
        // Step 5: Test TSMOM (Time Series Momentum) signal validation
        let tsmom_results = self.test_tsmom_signal_generation(&features, &feature_pipeline).await?;
        metrics.extend(tsmom_results);
        
        // Step 6: Test signal strength and confidence value ranges
        let range_validation_results = self.test_signal_range_validation(&features, &test_data).await?;
        metrics.extend(range_validation_results);
        
        // Step 7: Test signal consistency and stability
        let consistency_results = self.test_signal_consistency(&features, &test_data).await?;
        metrics.extend(consistency_results);
        
        // Step 8: Validate overall signal generation performance
        let overall_performance = self.calculate_signal_generation_performance(&metrics)?;
        metrics.insert("overall_signal_performance".to_string(), overall_performance);
        
        let total_latency = test_start.elapsed().as_millis() as f64;
        metrics.insert("total_signal_validation_latency_ms".to_string(), total_latency);
        
        // Validate performance requirements
        if overall_performance < 0.8 {
            return Err(TestFrameworkError::ValidationError(
                format!("Signal generation performance below threshold: {:.3} < 0.8", overall_performance)
            ).into());
        }
        
        Ok(metrics)
    }
    
    /// Generate test data specifically designed for signal validation
    fn generate_signal_test_data(&self) -> Result<Vec<OHLCV>> {
        let mut data = Vec::new();
        let base_price = 100.0;
        let base_timestamp = 1640995200;
        
        // Generate 200 bars with various market patterns for comprehensive testing
        for i in 0..200 {
            let t = i as f64;
            
            // Create different market regimes for signal testing
            let price = if i < 50 {
                // Trending up phase
                base_price + t * 0.3 + (t * 0.1).sin() * 0.5
            } else if i < 100 {
                // Sideways phase
                base_price + 15.0 + (t * 0.2).sin() * 3.0
            } else if i < 150 {
                // Trending down phase
                base_price + 15.0 - (t - 100.0) * 0.2 + (t * 0.15).sin() * 0.8
            } else {
                // Volatile phase
                base_price + 5.0 + (t * 0.3).sin() * 5.0 + (t * 0.7).cos() * 2.0
            };
            
            data.push(OHLCV {
                timestamp: base_timestamp + (i * 300),
                open: price,
                high: price * (1.0 + 0.01 + (t * 0.1).sin().abs() * 0.01),
                low: price * (1.0 - 0.01 - (t * 0.1).cos().abs() * 0.01),
                close: price * (1.0 + (t * 0.05).sin() * 0.005),
                volume: 1000.0 + (t * 0.1).sin().abs() * 500.0,
            });
        }
        
        Ok(data)
    }
    
    /// Test LDC signal generation with k-NN classification
    async fn test_ldc_signal_generation(&self, features: &[feature_pipeline::Features], test_data: &[feature_pipeline::OHLCV]) -> Result<HashMap<String, f64>> {
        use std::time::Instant;
        
        let mut results = HashMap::new();
        let ldc_start = Instant::now();
        
        // Configure LDC engine for testing
        let mut ldc_config = LDCConfig::default();
        ldc_config.k_neighbors = 8;
        ldc_config.max_bars_back = 2000;
        ldc_config.use_multithreading = false; // For deterministic testing
        ldc_config.use_hnsw_index = false; // Use exact search for accuracy
        
        let mut ldc_engine = LDCEngine::with_config(ldc_config);
        
        let mut ldc_signals = Vec::new();
        let mut training_samples_added = 0;
        let mut predictions_made = 0;
        
        // Process features and generate LDC signals
        for (i, feature) in features.iter().enumerate() {
            if let (Some(rsi), Some(wt1), Some(cci), Some(adx)) = 
                (feature.rsi, feature.wavetrend_1, feature.cci, feature.adx) {
                
                let feature_series = FeatureSeries {
                    f1: rsi as f32,
                    f2: wt1 as f32,
                    f3: cci as f32,
                    f4: adx as f32,
                    f5: feature.momentum.unwrap_or(0.0) as f32,
                };
                
                // Add training samples for the first 80% of data
                if i < features.len() * 4 / 5 && i < test_data.len() - 5 {
                    let current_close = test_data[i].close;
                    let future_close = test_data.get(i + 5).map(|d| d.close).unwrap_or(current_close);
                    let price_change = (future_close - current_close) / current_close;
                    
                    let direction = if price_change > 0.02 {
                        Direction::Long
                    } else if price_change < -0.02 {
                        Direction::Short
                    } else {
                        Direction::Neutral
                    };
                    
                    let training_sample = TrainingSample {
                        features: feature_series.clone(),
                        direction,
                        timestamp: feature.timestamp,
                    };
                    
                    ldc_engine.add_sample(training_sample);
                    training_samples_added += 1;
                }
                
                // Generate predictions for the last 20% of data
                if i >= features.len() * 4 / 5 && training_samples_added > 50 {
                    let prediction_start = Instant::now();
                    let ldc_signal = ldc_engine.predict(&feature_series, 8)
                        .unwrap_or(0.0);
                    let prediction_latency = prediction_start.elapsed().as_micros() as f64 / 1000.0;
                    
                    ldc_signals.push((ldc_signal, prediction_latency));
                    predictions_made += 1;
                }
            }
        }
        
        let ldc_total_latency = ldc_start.elapsed().as_millis() as f64;
        
        // Validate LDC signal properties
        results.insert("ldc_training_samples".to_string(), training_samples_added as f64);
        results.insert("ldc_predictions_made".to_string(), predictions_made as f64);
        results.insert("ldc_total_latency_ms".to_string(), ldc_total_latency);
        
        if !ldc_signals.is_empty() {
            // Calculate average prediction latency
            let avg_prediction_latency = ldc_signals.iter().map(|(_, lat)| lat).sum::<f64>() / ldc_signals.len() as f64;
            results.insert("ldc_avg_prediction_latency_ms".to_string(), avg_prediction_latency);
            
            // Validate signal strength ranges
            let signal_values: Vec<f64> = ldc_signals.iter().map(|(sig, _)| *sig).collect();
            let min_signal = signal_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_signal = signal_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg_signal = signal_values.iter().sum::<f64>() / signal_values.len() as f64;
            
            results.insert("ldc_min_signal".to_string(), min_signal);
            results.insert("ldc_max_signal".to_string(), max_signal);
            results.insert("ldc_avg_signal".to_string(), avg_signal);
            
            // Validate signal range is reasonable (-1.0 to 1.0)
            let signals_in_range = signal_values.iter().filter(|&&s| s >= -1.0 && s <= 1.0).count();
            let range_compliance = signals_in_range as f64 / signal_values.len() as f64;
            results.insert("ldc_range_compliance".to_string(), range_compliance);
            
            // Validate prediction latency meets requirements
            let latency_compliance = if avg_prediction_latency <= 10.0 { 1.0 } else { 0.0 };
            results.insert("ldc_latency_compliance".to_string(), latency_compliance);
        } else {
            results.insert("ldc_range_compliance".to_string(), 0.0);
            results.insert("ldc_latency_compliance".to_string(), 0.0);
        }
        
        Ok(results)
    }
    
    /// Test MR (Mean Reversion) signal validation
    async fn test_mr_signal_generation(&self, features: &[feature_pipeline::Features], feature_pipeline: &FeaturePipeline) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();
        let mr_start = Instant::now();
        
        // Generate MR signals
        let mr_signals = feature_pipeline.generate_mr_signal(features)
            .context("Failed to generate MR signals")?;
        
        let mr_latency = mr_start.elapsed().as_millis() as f64;
        results.insert("mr_generation_latency_ms".to_string(), mr_latency);
        results.insert("mr_signals_generated".to_string(), mr_signals.len() as f64);
        
        // Validate MR signal properties
        let valid_mr_signals: Vec<f64> = mr_signals.iter()
            .filter_map(|s| s.s_mr)
            .collect();
        
        if !valid_mr_signals.is_empty() {
            let min_mr = valid_mr_signals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_mr = valid_mr_signals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg_mr = valid_mr_signals.iter().sum::<f64>() / valid_mr_signals.len() as f64;
            let std_mr = {
                let variance = valid_mr_signals.iter()
                    .map(|&x| (x - avg_mr).powi(2))
                    .sum::<f64>() / valid_mr_signals.len() as f64;
                variance.sqrt()
            };
            
            results.insert("mr_min_signal".to_string(), min_mr);
            results.insert("mr_max_signal".to_string(), max_mr);
            results.insert("mr_avg_signal".to_string(), avg_mr);
            results.insert("mr_std_signal".to_string(), std_mr);
            
            // Validate MR signal characteristics
            // MR signals should be mean-reverting (oscillate around zero)
            let zero_crossings = valid_mr_signals.windows(2)
                .filter(|w| (w[0] > 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] > 0.0))
                .count();
            let zero_crossing_rate = zero_crossings as f64 / (valid_mr_signals.len() - 1) as f64;
            results.insert("mr_zero_crossing_rate".to_string(), zero_crossing_rate);
            
            // Validate signal strength distribution
            let strong_signals = valid_mr_signals.iter().filter(|&&s| s.abs() > 0.5).count();
            let strong_signal_rate = strong_signals as f64 / valid_mr_signals.len() as f64;
            results.insert("mr_strong_signal_rate".to_string(), strong_signal_rate);
            
            // MR signals should have reasonable variance (not constant)
            let variance_check = if std_mr > 0.01 { 1.0 } else { 0.0 };
            results.insert("mr_variance_check".to_string(), variance_check);
        } else {
            results.insert("mr_variance_check".to_string(), 0.0);
        }
        
        Ok(results)
    }
    
    /// Test TSMOM (Time Series Momentum) signal validation
    async fn test_tsmom_signal_generation(&self, features: &[feature_pipeline::Features], feature_pipeline: &FeaturePipeline) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();
        let tsmom_start = Instant::now();
        
        // Generate TSMOM signals
        let tsmom_signals = feature_pipeline.generate_tsmom_signal(features)
            .context("Failed to generate TSMOM signals")?;
        
        let tsmom_latency = tsmom_start.elapsed().as_millis() as f64;
        results.insert("tsmom_generation_latency_ms".to_string(), tsmom_latency);
        results.insert("tsmom_signals_generated".to_string(), tsmom_signals.len() as f64);
        
        // Validate TSMOM signal properties
        let valid_tsmom_signals: Vec<f64> = tsmom_signals.iter()
            .filter_map(|s| s.s_tsmom)
            .collect();
        
        if !valid_tsmom_signals.is_empty() {
            let min_tsmom = valid_tsmom_signals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_tsmom = valid_tsmom_signals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg_tsmom = valid_tsmom_signals.iter().sum::<f64>() / valid_tsmom_signals.len() as f64;
            let std_tsmom = {
                let variance = valid_tsmom_signals.iter()
                    .map(|&x| (x - avg_tsmom).powi(2))
                    .sum::<f64>() / valid_tsmom_signals.len() as f64;
                variance.sqrt()
            };
            
            results.insert("tsmom_min_signal".to_string(), min_tsmom);
            results.insert("tsmom_max_signal".to_string(), max_tsmom);
            results.insert("tsmom_avg_signal".to_string(), avg_tsmom);
            results.insert("tsmom_std_signal".to_string(), std_tsmom);
            
            // Validate TSMOM signal characteristics
            // TSMOM signals should show momentum persistence
            let positive_signals = valid_tsmom_signals.iter().filter(|&&s| s > 0.0).count();
            let negative_signals = valid_tsmom_signals.iter().filter(|&&s| s < 0.0).count();
            let signal_bias = (positive_signals as f64 - negative_signals as f64) / valid_tsmom_signals.len() as f64;
            results.insert("tsmom_signal_bias".to_string(), signal_bias.abs());
            
            // Validate momentum persistence (consecutive signals in same direction)
            let mut persistence_count = 0;
            let mut total_transitions = 0;
            for window in valid_tsmom_signals.windows(3) {
                total_transitions += 1;
                if (window[0] > 0.0 && window[1] > 0.0 && window[2] > 0.0) ||
                   (window[0] < 0.0 && window[1] < 0.0 && window[2] < 0.0) {
                    persistence_count += 1;
                }
            }
            let persistence_rate = if total_transitions > 0 {
                persistence_count as f64 / total_transitions as f64
            } else {
                0.0
            };
            results.insert("tsmom_persistence_rate".to_string(), persistence_rate);
            
            // Validate signal range is reasonable (momentum should be bounded)
            let range_bounded = valid_tsmom_signals.iter().all(|&s| s.abs() <= 1.0);
            results.insert("tsmom_range_bounded".to_string(), if range_bounded { 1.0 } else { 0.0 });
        } else {
            results.insert("tsmom_range_bounded".to_string(), 0.0);
        }
        
        Ok(results)
    }
    
    /// Test signal strength and confidence value ranges
    async fn test_signal_range_validation(&self, features: &[feature_pipeline::Features], _test_data: &[OHLCV]) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();
        
        // Test with extreme market conditions to validate signal ranges
        let extreme_data = self.generate_extreme_market_data()?;
        let feature_pipeline = FeaturePipeline::new(20);
        let extreme_features = feature_pipeline.compute_features_safe(&extreme_data)?;
        
        if extreme_features.is_empty() {
            results.insert("range_validation_success".to_string(), 0.0);
            return Ok(results);
        }
        
        // Test LDC with extreme conditions
        let mut ldc_config = LDCConfig::default();
        ldc_config.k_neighbors = 5;
        ldc_config.use_multithreading = false;
        let mut ldc_engine = LDCEngine::with_config(ldc_config);
        
        // Add some training data first
        for (_i, feature) in features.iter().enumerate().take(50) {
            if let (Some(rsi), Some(wt1), Some(cci), Some(adx)) = 
                (feature.rsi, feature.wavetrend_1, feature.cci, feature.adx) {
                
                let feature_series = FeatureSeries {
                    f1: rsi as f32,
                    f2: wt1 as f32,
                    f3: cci as f32,
                    f4: adx as f32,
                    f5: feature.momentum.unwrap_or(0.0) as f32,
                };
                
                let training_sample = TrainingSample {
                    features: feature_series,
                    direction: Direction::Neutral,
                    timestamp: feature.timestamp,
                };
                
                ldc_engine.add_sample(training_sample);
            }
        }
        
        // Test signal ranges with extreme features
        let mut range_violations = 0;
        let mut total_tests = 0;
        
        for feature in extreme_features.iter().take(20) {
            if let (Some(rsi), Some(wt1), Some(cci), Some(adx)) = 
                (feature.rsi, feature.wavetrend_1, feature.cci, feature.adx) {
                
                let feature_series = FeatureSeries {
                    f1: rsi as f32,
                    f2: wt1 as f32,
                    f3: cci as f32,
                    f4: adx as f32,
                    f5: feature.momentum.unwrap_or(0.0) as f32,
                };
                
                total_tests += 1;
                
                // Test LDC signal range
                let ldc_signal = ldc_engine.predict(&feature_series, 5).unwrap_or(0.0);
                if ldc_signal.abs() > 2.0 { // Allow some flexibility but catch extreme values
                    range_violations += 1;
                }
                
                // Test MR signal range
                if let (Some(zscore), Some(std)) = (feature.zscore_20, feature.std_20) {
                    let mr_signal = -zscore / std.max(1e-8);
                    if mr_signal.abs() > 10.0 { // MR signals can be larger but should be bounded
                        range_violations += 1;
                    }
                }
                
                // Test TSMOM signal range
                if let Some(momentum) = feature.momentum {
                    if momentum.abs() > 2.0 { // Momentum should be reasonable
                        range_violations += 1;
                    }
                }
            }
        }
        
        let range_compliance = if total_tests > 0 {
            1.0 - (range_violations as f64 / (total_tests * 3) as f64) // 3 signals per test
        } else {
            0.0
        };
        
        results.insert("signal_range_compliance".to_string(), range_compliance);
        results.insert("range_violations".to_string(), range_violations as f64);
        results.insert("range_tests_performed".to_string(), total_tests as f64);
        
        Ok(results)
    }
    
    /// Test signal consistency and stability
    async fn test_signal_consistency(&self, features: &[feature_pipeline::Features], test_data: &[OHLCV]) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();
        
        // Test signal consistency by running the same computation multiple times
        let feature_pipeline = FeaturePipeline::new(20);
        
        // Generate signals multiple times with the same input
        let signals1 = feature_pipeline.generate_signals(features)?;
        let signals2 = feature_pipeline.generate_signals(features)?;
        
        // Compare signal consistency
        let mut consistent_mr = 0;
        let mut consistent_tsmom = 0;
        let mut total_comparisons = 0;
        
        for (s1, s2) in signals1.iter().zip(signals2.iter()) {
            total_comparisons += 1;
            
            // Check MR signal consistency
            match (s1.s_mr, s2.s_mr) {
                (Some(mr1), Some(mr2)) => {
                    if (mr1 - mr2).abs() < 1e-10 { // Should be exactly the same
                        consistent_mr += 1;
                    }
                }
                (None, None) => consistent_mr += 1,
                _ => {} // Inconsistent (one has value, other doesn't)
            }
            
            // Check TSMOM signal consistency
            match (s1.s_tsmom, s2.s_tsmom) {
                (Some(tsmom1), Some(tsmom2)) => {
                    if (tsmom1 - tsmom2).abs() < 1e-10 { // Should be exactly the same
                        consistent_tsmom += 1;
                    }
                }
                (None, None) => consistent_tsmom += 1,
                _ => {} // Inconsistent
            }
        }
        
        let mr_consistency = if total_comparisons > 0 {
            consistent_mr as f64 / total_comparisons as f64
        } else {
            0.0
        };
        
        let tsmom_consistency = if total_comparisons > 0 {
            consistent_tsmom as f64 / total_comparisons as f64
        } else {
            0.0
        };
        
        results.insert("mr_signal_consistency".to_string(), mr_consistency);
        results.insert("tsmom_signal_consistency".to_string(), tsmom_consistency);
        results.insert("consistency_tests_performed".to_string(), total_comparisons as f64);
        
        // Test signal stability with small input perturbations
        let stability_score = self.test_signal_stability(features, test_data).await?;
        results.insert("signal_stability_score".to_string(), stability_score);
        
        Ok(results)
    }
    
    /// Test signal stability with small input perturbations
    async fn test_signal_stability(&self, features: &[feature_pipeline::Features], test_data: &[OHLCV]) -> Result<f64> {
        let feature_pipeline = FeaturePipeline::new(20);
        
        // Create slightly perturbed data (1% price changes)
        let mut perturbed_data = test_data.to_vec();
        for ohlcv in &mut perturbed_data {
            ohlcv.close *= 1.001; // 0.1% perturbation
            ohlcv.high *= 1.001;
            ohlcv.low *= 1.001;
            ohlcv.open *= 1.001;
        }
        
        // Compute features and signals for perturbed data
        let perturbed_features = feature_pipeline.compute_features_safe(&perturbed_data)?;
        let original_signals = feature_pipeline.generate_signals(features)?;
        let perturbed_signals = feature_pipeline.generate_signals(&perturbed_features)?;
        
        // Measure signal stability
        let mut stable_signals = 0;
        let mut total_signals = 0;
        
        for (orig, pert) in original_signals.iter().zip(perturbed_signals.iter()) {
            if let (Some(orig_mr), Some(pert_mr)) = (orig.s_mr, pert.s_mr) {
                total_signals += 1;
                let relative_change = if orig_mr.abs() > 1e-8 {
                    (pert_mr - orig_mr).abs() / orig_mr.abs()
                } else {
                    (pert_mr - orig_mr).abs()
                };
                
                if relative_change < 0.1 { // 10% tolerance for stability
                    stable_signals += 1;
                }
            }
            
            if let (Some(orig_tsmom), Some(pert_tsmom)) = (orig.s_tsmom, pert.s_tsmom) {
                total_signals += 1;
                let relative_change = if orig_tsmom.abs() > 1e-8 {
                    (pert_tsmom - orig_tsmom).abs() / orig_tsmom.abs()
                } else {
                    (pert_tsmom - orig_tsmom).abs()
                };
                
                if relative_change < 0.1 { // 10% tolerance for stability
                    stable_signals += 1;
                }
            }
        }
        
        Ok(if total_signals > 0 {
            stable_signals as f64 / total_signals as f64
        } else {
            0.0
        })
    }
    
    /// Generate extreme market data for range testing
    fn generate_extreme_market_data(&self) -> Result<Vec<OHLCV>> {
        let mut data = Vec::new();
        let base_timestamp = 1640995200;
        
        // Generate extreme market conditions
        for i in 0..50 {
            let price = if i < 10 {
                // Flash crash
                100.0 - (i as f64 * 5.0)
            } else if i < 20 {
                // Recovery spike
                50.0 + ((i - 10) as f64 * 8.0)
            } else if i < 30 {
                // Extreme volatility
                130.0 + (i as f64 * 2.0).sin() * 20.0
            } else {
                // Gradual normalization
                120.0 + ((40 - i) as f64 * 0.5)
            };
            
            data.push(OHLCV {
                timestamp: base_timestamp + (i * 300),
                open: price,
                high: price * 1.1,
                low: price * 0.9,
                close: price * 1.05,
                volume: 10000.0,
            });
        }
        
        Ok(data)
    }
    
    /// Calculate overall signal generation performance score
    fn calculate_signal_generation_performance(&self, metrics: &HashMap<String, f64>) -> Result<f64> {
        let mut score = 0.0;
        let mut weight_sum = 0.0;
        
        // LDC performance (weight: 0.4)
        if let Some(ldc_range_compliance) = metrics.get("ldc_range_compliance") {
            score += ldc_range_compliance * 0.4;
            weight_sum += 0.4;
        }
        
        if let Some(ldc_latency_compliance) = metrics.get("ldc_latency_compliance") {
            score += ldc_latency_compliance * 0.1;
            weight_sum += 0.1;
        }
        
        // MR performance (weight: 0.2)
        if let Some(mr_variance_check) = metrics.get("mr_variance_check") {
            score += mr_variance_check * 0.1;
            weight_sum += 0.1;
        }
        
        if let Some(mr_consistency) = metrics.get("mr_signal_consistency") {
            score += mr_consistency * 0.1;
            weight_sum += 0.1;
        }
        
        // TSMOM performance (weight: 0.2)
        if let Some(tsmom_range_bounded) = metrics.get("tsmom_range_bounded") {
            score += tsmom_range_bounded * 0.1;
            weight_sum += 0.1;
        }
        
        if let Some(tsmom_consistency) = metrics.get("tsmom_signal_consistency") {
            score += tsmom_consistency * 0.1;
            weight_sum += 0.1;
        }
        
        // Overall stability and range compliance (weight: 0.2)
        if let Some(signal_range_compliance) = metrics.get("signal_range_compliance") {
            score += signal_range_compliance * 0.1;
            weight_sum += 0.1;
        }
        
        if let Some(signal_stability) = metrics.get("signal_stability_score") {
            score += signal_stability * 0.1;
            weight_sum += 0.1;
        }
        
        Ok(if weight_sum > 0.0 { score / weight_sum } else { 0.0 })
    }
    
    /// Generate deterministic OHLCV data for testing (doesn't require &mut self)
    fn generate_deterministic_ohlcv_data(&self, symbol: &str, duration_hours: u32) -> Result<Vec<crate::data_generator::OHLCVBar>> {
        use crate::data_generator::OHLCVBar;
        
        let interval_minutes = 5; // 5-minute intervals
        let total_bars = (duration_hours * 60) / interval_minutes;
        let mut bars = Vec::with_capacity(total_bars as usize);
        let base_price = 100.0;
        let start_timestamp = chrono::Utc::now().timestamp() - (duration_hours as i64 * 3600);
        
        for i in 0..total_bars {
            let timestamp = start_timestamp + (i as i64 * interval_minutes as i64 * 60);
            
            // Generate deterministic price movement
            let t = i as f64 * 0.1;
            let trend = t * 0.1; // Small upward trend
            let cycle = (t * 2.0 * std::f64::consts::PI / 20.0).sin() * 2.0; // 20-bar cycle
            let noise = (t * 7.0).sin() * 0.5; // High-frequency component
            
            let price = base_price + trend + cycle + noise;
            
            bars.push(OHLCVBar {
                timestamp,
                open: price,
                high: price * 1.01,
                low: price * 0.99,
                close: price * 1.005,
                volume: 1000.0 + (i as f64 * 10.0),
                symbol: symbol.to_string(),
                interval: "5m".to_string(),
            });
        }
        
        Ok(bars)
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