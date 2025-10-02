use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;
use chrono::{DateTime, Utc};

// Import error handling modules
use crate::test_diagnostics::*;
use crate::graceful_recovery::*;

/// Automated test runner supporting parallel execution and CI/CD integration
pub struct AutomatedTestRunner {
    pub config: TestRunnerConfig,
    pub test_suites: Vec<TestSuite>,
    pub performance_baseline: Option<PerformanceBaseline>,
    pub test_history: TestHistory,
    pub diagnostics_engine: Option<TestDiagnosticsEngine>,
    pub recovery_system: Option<GracefulRecoverySystem>,
}

/// Configuration for the automated test runner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRunnerConfig {
    /// Maximum number of parallel test suites to run
    pub max_parallel_suites: usize,
    /// Timeout for individual test suites in seconds
    pub suite_timeout_seconds: u64,
    /// Timeout for individual tests in seconds
    pub test_timeout_seconds: u64,
    /// Enable performance regression detection
    pub enable_regression_detection: bool,
    /// Performance regression threshold (percentage)
    pub regression_threshold_percent: f64,
    /// Output directory for test reports
    pub output_directory: PathBuf,
    /// Enable machine-readable output
    pub machine_readable_output: bool,
    /// Test selection strategy
    pub test_selection: TestSelectionStrategy,
    /// Resource cleanup timeout in seconds
    pub cleanup_timeout_seconds: u64,
    /// Enable verbose logging
    pub verbose: bool,
    /// Stop on first test failure
    pub fail_fast: bool,
    /// Enable comprehensive error diagnostics
    pub enable_error_diagnostics: bool,
    /// Enable graceful recovery from test failures
    pub enable_graceful_recovery: bool,
}

impl Default for TestRunnerConfig {
    fn default() -> Self {
        Self {
            max_parallel_suites: num_cpus::get(),
            suite_timeout_seconds: 300, // 5 minutes
            test_timeout_seconds: 60,   // 1 minute
            enable_regression_detection: true,
            regression_threshold_percent: 10.0, // 10% regression threshold
            output_directory: PathBuf::from("test_reports"),
            machine_readable_output: true,
            test_selection: TestSelectionStrategy::All,
            cleanup_timeout_seconds: 30,
            verbose: false,
            fail_fast: false,
            enable_error_diagnostics: true,
            enable_graceful_recovery: true,
        }
    }
}

/// Test selection strategy for running relevant tests based on changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestSelectionStrategy {
    /// Run all tests
    All,
    /// Run only tests affected by changed files
    ChangedFiles(Vec<PathBuf>),
    /// Run specific test categories
    Categories(Vec<TestCategory>),
    /// Run tests matching a pattern
    Pattern(String),
}

/// Test categories for selective execution
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TestCategory {
    Unit,
    Integration,
    Performance,
    Mathematical,
    Backtesting,
    Statistical,
    Compatibility,
}

/// Individual test suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    pub name: String,
    pub category: TestCategory,
    pub command: String,
    pub args: Vec<String>,
    pub working_directory: Option<PathBuf>,
    pub environment: HashMap<String, String>,
    pub timeout_seconds: Option<u64>,
    pub dependencies: Vec<String>, // Other test suites this depends on
    pub affected_by_files: Vec<String>, // File patterns that affect this test
    pub parallel_safe: bool, // Whether this test can run in parallel with others
}

/// Test execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteResult {
    pub suite_name: String,
    pub category: TestCategory,
    pub status: TestStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_ms: u64,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub performance_metrics: Option<PerformanceMetrics>,
    pub error_details: Option<String>,
}

/// Test execution status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Timeout,
    Skipped,
    Error,
}

/// Performance metrics for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub execution_time_ms: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub custom_metrics: HashMap<String, f64>,
}

/// Performance baseline for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub timestamp: DateTime<Utc>,
    pub suite_baselines: HashMap<String, PerformanceMetrics>,
    pub git_commit: Option<String>,
}

/// Test execution history for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestHistory {
    pub executions: Vec<TestExecutionRecord>,
    pub max_history_entries: usize,
}

/// Individual test execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionRecord {
    pub timestamp: DateTime<Utc>,
    pub git_commit: Option<String>,
    pub results: Vec<TestSuiteResult>,
    pub overall_status: TestStatus,
    pub total_duration_ms: u64,
}

/// Comprehensive test execution report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionReport {
    pub execution_id: String,
    pub timestamp: DateTime<Utc>,
    pub git_commit: Option<String>,
    pub config: TestRunnerConfig,
    pub results: Vec<TestSuiteResult>,
    pub summary: TestSummary,
    pub performance_regressions: Vec<PerformanceRegression>,
    pub recommendations: Vec<String>,
}

/// Test execution summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub total_suites: usize,
    pub passed_suites: usize,
    pub failed_suites: usize,
    pub skipped_suites: usize,
    pub error_suites: usize,
    pub total_duration_ms: u64,
    pub success_rate: f64,
    pub overall_status: TestStatus,
}

/// Performance regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    pub suite_name: String,
    pub metric_name: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub regression_percent: f64,
    pub severity: RegressionSeverity,
}

/// Severity level for performance regressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,   // < 20% regression
    Major,   // 20-50% regression
    Critical, // > 50% regression
}

impl AutomatedTestRunner {
    /// Create a new automated test runner with configuration
    pub fn new(config: TestRunnerConfig) -> Result<Self> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&config.output_directory)?;
        
        let test_suites = Self::discover_test_suites()?;
        let performance_baseline = Self::load_performance_baseline(&config.output_directory)?;
        let test_history = Self::load_test_history(&config.output_directory)?;
        
        // Initialize error handling components
        let diagnostics_engine = if config.enable_error_diagnostics {
            Some(TestDiagnosticsEngine::new(DiagnosticsConfig::default()))
        } else {
            None
        };

        let recovery_system = if config.enable_graceful_recovery {
            diagnostics_engine.as_ref().map(|de| {
                GracefulRecoverySystem::new(
                    RecoveryConfig::default(),
                    de.clone(),
                )
            })
        } else {
            None
        };

        Ok(Self {
            config,
            test_suites,
            performance_baseline,
            test_history,
            diagnostics_engine,
            recovery_system,
        })
    }
    
    /// Run all tests with parallel execution and comprehensive reporting
    pub fn run_all_tests(&mut self) -> Result<TestExecutionReport> {
        let execution_id = format!("test_run_{}", 
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        let start_time = Utc::now();
        
        if self.config.verbose {
            println!("Starting test execution: {}", execution_id);
            println!("Configuration: {:#?}", self.config);
        }
        
        // Select tests based on strategy
        let selected_suites = self.select_test_suites()?;
        
        if self.config.verbose {
            println!("Selected {} test suites for execution", selected_suites.len());
        }
        
        // Execute tests with dependency resolution and parallel execution
        let results = self.execute_test_suites_parallel(&selected_suites)?;
        
        let end_time = Utc::now();
        let total_duration = (end_time - start_time).num_milliseconds() as u64;
        
        // Generate summary
        let summary = self.generate_summary(&results, total_duration);
        
        // Detect performance regressions
        let performance_regressions = if self.config.enable_regression_detection {
            self.detect_performance_regressions(&results)?
        } else {
            Vec::new()
        };
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&results, &performance_regressions);
        
        // Get git commit if available
        let git_commit = self.get_git_commit().ok();
        
        let report = TestExecutionReport {
            execution_id: execution_id.clone(),
            timestamp: start_time,
            git_commit: git_commit.clone(),
            config: self.config.clone(),
            results: results.clone(),
            summary,
            performance_regressions,
            recommendations,
        };
        
        // Save report
        self.save_test_report(&report)?;
        
        // Update test history
        self.update_test_history(TestExecutionRecord {
            timestamp: start_time,
            git_commit,
            results,
            overall_status: report.summary.overall_status.clone(),
            total_duration_ms: total_duration,
        })?;
        
        // Update performance baseline if tests passed
        if report.summary.overall_status == TestStatus::Passed {
            self.update_performance_baseline(&report.results)?;
        }
        
        if self.config.verbose {
            println!("Test execution completed: {}", execution_id);
            self.print_summary(&report.summary);
        }
        
        Ok(report)
    }
    
    /// Discover available test suites from the project structure
    pub fn discover_test_suites() -> Result<Vec<TestSuite>> {
        let mut suites = Vec::new();
        
        // Unit tests
        suites.push(TestSuite {
            name: "unit_tests".to_string(),
            category: TestCategory::Unit,
            command: "cargo".to_string(),
            args: vec!["test".to_string(), "--lib".to_string()],
            working_directory: None,
            environment: HashMap::new(),
            timeout_seconds: Some(120),
            dependencies: Vec::new(),
            affected_by_files: vec!["src/**/*.rs".to_string()],
            parallel_safe: true,
        });
        
        // Mathematical accuracy tests
        suites.push(TestSuite {
            name: "mathematical_accuracy".to_string(),
            category: TestCategory::Mathematical,
            command: "cargo".to_string(),
            args: vec!["test".to_string(), "mathematical_accuracy".to_string()],
            working_directory: None,
            environment: HashMap::new(),
            timeout_seconds: Some(180),
            dependencies: Vec::new(),
            affected_by_files: vec!["src/**/*.rs".to_string(), "tests/mathematical_accuracy_tests.rs".to_string()],
            parallel_safe: true,
        });
        
        // Performance validation tests
        suites.push(TestSuite {
            name: "performance_validation".to_string(),
            category: TestCategory::Performance,
            command: "cargo".to_string(),
            args: vec!["test".to_string(), "performance_validation".to_string(), "--release".to_string()],
            working_directory: None,
            environment: HashMap::new(),
            timeout_seconds: Some(300),
            dependencies: Vec::new(),
            affected_by_files: vec!["src/**/*.rs".to_string(), "tests/performance_validation_tests.rs".to_string()],
            parallel_safe: false, // Performance tests should run alone
        });
        
        // Integration tests
        suites.push(TestSuite {
            name: "integration_tests".to_string(),
            category: TestCategory::Integration,
            command: "cargo".to_string(),
            args: vec!["test".to_string(), "integration".to_string()],
            working_directory: None,
            environment: HashMap::new(),
            timeout_seconds: Some(240),
            dependencies: vec!["unit_tests".to_string()],
            affected_by_files: vec!["src/**/*.rs".to_string(), "tests/**/*.rs".to_string()],
            parallel_safe: true,
        });
        
        // Backtesting tests
        suites.push(TestSuite {
            name: "backtesting_tests".to_string(),
            category: TestCategory::Backtesting,
            command: "cargo".to_string(),
            args: vec!["test".to_string(), "backtesting".to_string()],
            working_directory: None,
            environment: HashMap::new(),
            timeout_seconds: Some(300),
            dependencies: vec!["unit_tests".to_string()],
            affected_by_files: vec!["src/backtesting.rs".to_string(), "src/**/*.rs".to_string()],
            parallel_safe: true,
        });
        
        // Statistical analysis tests
        suites.push(TestSuite {
            name: "statistical_analysis".to_string(),
            category: TestCategory::Statistical,
            command: "cargo".to_string(),
            args: vec!["test".to_string(), "statistical".to_string()],
            working_directory: None,
            environment: HashMap::new(),
            timeout_seconds: Some(180),
            dependencies: vec!["unit_tests".to_string()],
            affected_by_files: vec!["src/statistical_analysis.rs".to_string(), "src/**/*.rs".to_string()],
            parallel_safe: true,
        });
        
        // Compatibility tests
        suites.push(TestSuite {
            name: "pine_script_compatibility".to_string(),
            category: TestCategory::Compatibility,
            command: "cargo".to_string(),
            args: vec!["test".to_string(), "pine_script_compatibility".to_string()],
            working_directory: None,
            environment: HashMap::new(),
            timeout_seconds: Some(120),
            dependencies: vec!["unit_tests".to_string()],
            affected_by_files: vec!["src/**/*.rs".to_string(), "tests/pine_script_compatibility_tests.rs".to_string()],
            parallel_safe: true,
        });
        
        // Benchmarks (optional, only run if requested)
        suites.push(TestSuite {
            name: "benchmarks".to_string(),
            category: TestCategory::Performance,
            command: "cargo".to_string(),
            args: vec!["bench".to_string()],
            working_directory: None,
            environment: HashMap::new(),
            timeout_seconds: Some(600), // Benchmarks can take longer
            dependencies: Vec::new(),
            affected_by_files: vec!["src/**/*.rs".to_string(), "benches/**/*.rs".to_string()],
            parallel_safe: false, // Benchmarks should run alone
        });
        
        Ok(suites)
    }
    
    /// Select test suites based on the configured strategy
    pub fn select_test_suites(&self) -> Result<Vec<TestSuite>> {
        match &self.config.test_selection {
            TestSelectionStrategy::All => Ok(self.test_suites.clone()),
            
            TestSelectionStrategy::Categories(categories) => {
                Ok(self.test_suites.iter()
                    .filter(|suite| categories.contains(&suite.category))
                    .cloned()
                    .collect())
            },
            
            TestSelectionStrategy::Pattern(pattern) => {
                Ok(self.test_suites.iter()
                    .filter(|suite| suite.name.contains(pattern))
                    .cloned()
                    .collect())
            },
            
            TestSelectionStrategy::ChangedFiles(changed_files) => {
                let mut selected = Vec::new();
                
                for suite in &self.test_suites {
                    let is_affected = suite.affected_by_files.iter().any(|pattern| {
                        changed_files.iter().any(|file| {
                            self.matches_pattern(file, pattern)
                        })
                    });
                    
                    if is_affected {
                        selected.push(suite.clone());
                    }
                }
                
                // Always include unit tests if any files changed
                if !changed_files.is_empty() && !selected.iter().any(|s| s.name == "unit_tests") {
                    if let Some(unit_tests) = self.test_suites.iter().find(|s| s.name == "unit_tests") {
                        selected.push(unit_tests.clone());
                    }
                }
                
                Ok(selected)
            }
        }
    }
    
    /// Check if a file path matches a glob pattern
    pub fn matches_pattern(&self, file_path: &Path, pattern: &str) -> bool {
        // Simple pattern matching - in a real implementation, use a proper glob library
        let file_str = file_path.to_string_lossy();
        
        if pattern.contains("**") {
            let parts: Vec<&str> = pattern.split("**").collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1].trim_start_matches('/');
                
                // For "src/**/*.rs", check if file starts with "src/" and ends with ".rs"
                if file_str.starts_with(prefix) {
                    if suffix.is_empty() {
                        return true;
                    }
                    // Handle the suffix part after **
                    if suffix.starts_with("*.") {
                        let extension = suffix.trim_start_matches("*.");
                        return file_str.ends_with(&format!(".{}", extension));
                    }
                    return file_str.ends_with(suffix);
                }
            }
        }
        
        if pattern.contains("*") && !pattern.contains("**") {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return file_str.starts_with(prefix) && file_str.ends_with(suffix);
            }
        }
        
        // For patterns without wildcards, check exact match or substring
        if !pattern.contains("*") {
            return file_str == pattern || file_str.contains(pattern);
        }
        
        false
    }
    
    /// Execute test suites in parallel with dependency resolution
    fn execute_test_suites_parallel(&self, suites: &[TestSuite]) -> Result<Vec<TestSuiteResult>> {
        let results = Arc::new(Mutex::new(Vec::new()));
        let completed_suites = Arc::new(Mutex::new(std::collections::HashSet::new()));
        
        // Separate parallel-safe and non-parallel-safe suites
        let (parallel_suites, sequential_suites): (Vec<_>, Vec<_>) = 
            suites.iter().partition(|suite| suite.parallel_safe);
        
        // Execute non-parallel-safe suites first (sequentially)
        for suite in sequential_suites {
            if self.can_execute_suite(suite, &completed_suites)? {
                let result = self.execute_single_suite(suite)?;
                
                {
                    let mut results_lock = results.lock().unwrap();
                    results_lock.push(result);
                    
                    let mut completed_lock = completed_suites.lock().unwrap();
                    completed_lock.insert(suite.name.clone());
                }
            }
        }
        
        // Execute parallel-safe suites in parallel (or sequentially if fail-fast)
        let parallel_results: Vec<TestSuiteResult> = if self.config.fail_fast {
            // Execute sequentially for fail-fast
            let mut results = Vec::new();
            for suite in parallel_suites {
                if self.can_execute_suite(suite, &completed_suites).unwrap_or(false) {
                    match self.execute_single_suite(suite) {
                        Ok(result) => {
                            let failed = result.status == TestStatus::Failed || result.status == TestStatus::Error;
                            
                            // Mark as completed
                            {
                                let mut completed_lock = completed_suites.lock().unwrap();
                                completed_lock.insert(suite.name.clone());
                            }
                            
                            results.push(result);
                            
                            // Stop on first failure if fail-fast is enabled
                            if failed {
                                if self.config.verbose {
                                    println!("Stopping execution due to test failure (fail-fast mode)");
                                }
                                break;
                            }
                        },
                        Err(e) => {
                            eprintln!("Error executing suite {}: {}", suite.name, e);
                            break; // Stop on error in fail-fast mode
                        }
                    }
                }
            }
            results
        } else {
            // Execute in parallel
            parallel_suites
                .par_iter()
                .filter_map(|suite| {
                    if self.can_execute_suite(suite, &completed_suites).unwrap_or(false) {
                        match self.execute_single_suite(suite) {
                            Ok(result) => {
                                // Mark as completed
                                {
                                    let mut completed_lock = completed_suites.lock().unwrap();
                                    completed_lock.insert(suite.name.clone());
                                }
                                Some(result)
                            },
                            Err(e) => {
                                eprintln!("Error executing suite {}: {}", suite.name, e);
                                None
                            }
                        }
                    } else {
                        None
                    }
                })
                .collect()
        };
        
        // Combine results
        let mut final_results = results.lock().unwrap().clone();
        final_results.extend(parallel_results);
        
        Ok(final_results)
    }
    
    /// Check if a test suite can be executed (dependencies satisfied)
    fn can_execute_suite(
        &self, 
        suite: &TestSuite, 
        completed_suites: &Arc<Mutex<std::collections::HashSet<String>>>
    ) -> Result<bool> {
        let completed = completed_suites.lock().unwrap();
        
        for dependency in &suite.dependencies {
            if !completed.contains(dependency) {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Execute a single test suite with timeout and resource monitoring
    fn execute_single_suite(&self, suite: &TestSuite) -> Result<TestSuiteResult> {
        let start_time = Utc::now();
        let start_instant = Instant::now();
        
        if self.config.verbose {
            println!("Executing test suite: {}", suite.name);
        }
        
        // Set up command
        let mut command = Command::new(&suite.command);
        command.args(&suite.args);
        
        if let Some(ref working_dir) = suite.working_directory {
            command.current_dir(working_dir);
        }
        
        for (key, value) in &suite.environment {
            command.env(key, value);
        }
        
        // Set timeout
        let timeout = Duration::from_secs(
            suite.timeout_seconds.unwrap_or(self.config.suite_timeout_seconds)
        );
        
        // Execute with timeout
        let execution_result = self.execute_with_timeout(command, timeout)?;
        
        let end_time = Utc::now();
        let duration_ms = start_instant.elapsed().as_millis() as u64;
        
        // Determine status
        let status = match execution_result {
            ExecutionResult::Success(exit_status) => {
                if exit_status.success() {
                    TestStatus::Passed
                } else {
                    TestStatus::Failed
                }
            },
            ExecutionResult::Timeout => TestStatus::Timeout,
            ExecutionResult::Error(_) => TestStatus::Error,
        };
        
        // Extract output and error details
        let (stdout, stderr, exit_code, error_details) = match execution_result {
            ExecutionResult::Success(exit_status) => {
                (String::new(), String::new(), exit_status.code(), None)
            },
            ExecutionResult::Timeout => {
                (String::new(), String::new(), None, Some("Test suite timed out".to_string()))
            },
            ExecutionResult::Error(e) => {
                (String::new(), String::new(), None, Some(e.to_string()))
            },
        };
        
        // Collect performance metrics (simplified for this implementation)
        let performance_metrics = Some(PerformanceMetrics {
            execution_time_ms: duration_ms,
            memory_usage_mb: 0.0, // Would be collected from system monitoring
            cpu_usage_percent: 0.0, // Would be collected from system monitoring
            custom_metrics: HashMap::new(),
        });
        
        Ok(TestSuiteResult {
            suite_name: suite.name.clone(),
            category: suite.category.clone(),
            status,
            start_time,
            end_time,
            duration_ms,
            exit_code,
            stdout,
            stderr,
            performance_metrics,
            error_details,
        })
    }
    
    /// Execute command with timeout handling
    fn execute_with_timeout(&self, mut command: Command, timeout: Duration) -> Result<ExecutionResult> {
        use std::sync::mpsc;
        use std::thread;
        
        let (tx, rx) = mpsc::channel();
        
        // Spawn command execution in a separate thread
        let handle = thread::spawn(move || {
            let result = command.status();
            let _ = tx.send(result);
        });
        
        // Wait for completion or timeout
        match rx.recv_timeout(timeout) {
            Ok(Ok(exit_status)) => Ok(ExecutionResult::Success(exit_status)),
            Ok(Err(e)) => Ok(ExecutionResult::Error(e.into())),
            Err(_) => {
                // Timeout occurred - attempt to clean up
                drop(handle); // This doesn't actually kill the process, but signals our intent
                Ok(ExecutionResult::Timeout)
            }
        }
    }
    
    /// Generate test execution summary
    pub fn generate_summary(&self, results: &[TestSuiteResult], total_duration_ms: u64) -> TestSummary {
        let total_suites = results.len();
        let passed_suites = results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let failed_suites = results.iter().filter(|r| r.status == TestStatus::Failed).count();
        let skipped_suites = results.iter().filter(|r| r.status == TestStatus::Skipped).count();
        let error_suites = results.iter().filter(|r| r.status == TestStatus::Error).count();
        
        let success_rate = if total_suites > 0 {
            (passed_suites as f64 / total_suites as f64) * 100.0
        } else {
            0.0
        };
        
        let overall_status = if failed_suites > 0 || error_suites > 0 {
            TestStatus::Failed
        } else if total_suites == passed_suites {
            TestStatus::Passed
        } else {
            TestStatus::Error
        };
        
        TestSummary {
            total_suites,
            passed_suites,
            failed_suites,
            skipped_suites,
            error_suites,
            total_duration_ms,
            success_rate,
            overall_status,
        }
    }
    
    /// Detect performance regressions compared to baseline
    pub fn detect_performance_regressions(&self, results: &[TestSuiteResult]) -> Result<Vec<PerformanceRegression>> {
        let mut regressions = Vec::new();
        
        if let Some(ref baseline) = self.performance_baseline {
            for result in results {
                if let Some(ref metrics) = result.performance_metrics {
                    if let Some(baseline_metrics) = baseline.suite_baselines.get(&result.suite_name) {
                        // Check execution time regression
                        let regression = self.calculate_regression(
                            baseline_metrics.execution_time_ms as f64,
                            metrics.execution_time_ms as f64,
                        );
                        
                        if regression > self.config.regression_threshold_percent {
                            let severity = if regression > 50.0 {
                                RegressionSeverity::Critical
                            } else if regression > 20.0 {
                                RegressionSeverity::Major
                            } else {
                                RegressionSeverity::Minor
                            };
                            
                            regressions.push(PerformanceRegression {
                                suite_name: result.suite_name.clone(),
                                metric_name: "execution_time_ms".to_string(),
                                baseline_value: baseline_metrics.execution_time_ms as f64,
                                current_value: metrics.execution_time_ms as f64,
                                regression_percent: regression,
                                severity,
                            });
                        }
                        
                        // Check custom metrics regressions
                        for (metric_name, current_value) in &metrics.custom_metrics {
                            if let Some(baseline_value) = baseline_metrics.custom_metrics.get(metric_name) {
                                let regression = self.calculate_regression(*baseline_value, *current_value);
                                
                                if regression > self.config.regression_threshold_percent {
                                    let severity = if regression > 50.0 {
                                        RegressionSeverity::Critical
                                    } else if regression > 20.0 {
                                        RegressionSeverity::Major
                                    } else {
                                        RegressionSeverity::Minor
                                    };
                                    
                                    regressions.push(PerformanceRegression {
                                        suite_name: result.suite_name.clone(),
                                        metric_name: metric_name.clone(),
                                        baseline_value: *baseline_value,
                                        current_value: *current_value,
                                        regression_percent: regression,
                                        severity,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(regressions)
    }
    
    /// Calculate regression percentage
    pub fn calculate_regression(&self, baseline: f64, current: f64) -> f64 {
        if baseline == 0.0 {
            return 0.0;
        }
        
        ((current - baseline) / baseline) * 100.0
    }
    
    /// Generate actionable recommendations based on test results
    pub fn generate_recommendations(&self, results: &[TestSuiteResult], regressions: &[PerformanceRegression]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Check for failed tests
        let failed_tests: Vec<_> = results.iter()
            .filter(|r| r.status == TestStatus::Failed)
            .collect();
        
        if !failed_tests.is_empty() {
            recommendations.push(format!(
                "Fix {} failed test suite(s): {}",
                failed_tests.len(),
                failed_tests.iter()
                    .map(|t| t.suite_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        
        // Check for timeout issues
        let timeout_tests: Vec<_> = results.iter()
            .filter(|r| r.status == TestStatus::Timeout)
            .collect();
        
        if !timeout_tests.is_empty() {
            recommendations.push(format!(
                "Investigate timeout issues in {} test suite(s): {}. Consider increasing timeout or optimizing test performance.",
                timeout_tests.len(),
                timeout_tests.iter()
                    .map(|t| t.suite_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        
        // Check for performance regressions
        if !regressions.is_empty() {
            let critical_regressions = regressions.iter()
                .filter(|r| r.severity == RegressionSeverity::Critical)
                .count();
            
            if critical_regressions > 0 {
                recommendations.push(format!(
                    "Address {} critical performance regression(s) immediately. Performance has degraded by more than 50%.",
                    critical_regressions
                ));
            }
            
            let major_regressions = regressions.iter()
                .filter(|r| r.severity == RegressionSeverity::Major)
                .count();
            
            if major_regressions > 0 {
                recommendations.push(format!(
                    "Investigate {} major performance regression(s). Performance has degraded by 20-50%.",
                    major_regressions
                ));
            }
        }
        
        // Check for slow tests
        let slow_tests: Vec<_> = results.iter()
            .filter(|r| r.duration_ms > 60000) // Tests taking more than 1 minute
            .collect();
        
        if !slow_tests.is_empty() {
            recommendations.push(format!(
                "Consider optimizing {} slow test suite(s): {}",
                slow_tests.len(),
                slow_tests.iter()
                    .map(|t| format!("{} ({}ms)", t.suite_name, t.duration_ms))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        
        // Check overall success rate
        let success_rate = (results.iter().filter(|r| r.status == TestStatus::Passed).count() as f64 
            / results.len() as f64) * 100.0;
        
        if success_rate < 90.0 {
            recommendations.push(format!(
                "Test success rate is {}%. Aim for >95% success rate in CI/CD pipeline.",
                success_rate as u32
            ));
        }
        
        recommendations
    }
    
    /// Save test report in multiple formats (JSON, XML, HTML)
    pub fn save_test_report(&self, report: &TestExecutionReport) -> Result<()> {
        let output_dir = &self.config.output_directory;
        
        // Save JSON report (machine-readable)
        if self.config.machine_readable_output {
            let json_path = output_dir.join(format!("{}.json", report.execution_id));
            let json_content = serde_json::to_string_pretty(report)?;
            std::fs::write(&json_path, json_content)?;
            
            if self.config.verbose {
                println!("Saved JSON report: {}", json_path.display());
            }
        }
        
        // Save JUnit XML format for CI/CD integration
        let xml_path = output_dir.join(format!("{}.xml", report.execution_id));
        let xml_content = self.generate_junit_xml(report)?;
        std::fs::write(&xml_path, xml_content)?;
        
        if self.config.verbose {
            println!("Saved JUnit XML report: {}", xml_path.display());
        }
        
        // Save human-readable HTML report
        let html_path = output_dir.join(format!("{}.html", report.execution_id));
        let html_content = self.generate_html_report(report)?;
        std::fs::write(&html_path, html_content)?;
        
        if self.config.verbose {
            println!("Saved HTML report: {}", html_path.display());
        }
        
        // Save latest report as symlink/copy for easy access
        let latest_json = output_dir.join("latest.json");
        let latest_xml = output_dir.join("latest.xml");
        let latest_html = output_dir.join("latest.html");
        
        if self.config.machine_readable_output {
            let _ = std::fs::copy(
                output_dir.join(format!("{}.json", report.execution_id)),
                latest_json
            );
        }
        let _ = std::fs::copy(
            output_dir.join(format!("{}.xml", report.execution_id)),
            latest_xml
        );
        let _ = std::fs::copy(
            output_dir.join(format!("{}.html", report.execution_id)),
            latest_html
        );
        
        Ok(())
    }
    
    /// Generate JUnit XML format for CI/CD integration
    pub fn generate_junit_xml(&self, report: &TestExecutionReport) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!(
            "<testsuites name=\"LDC Engine Tests\" tests=\"{}\" failures=\"{}\" errors=\"{}\" time=\"{:.3}\">\n",
            report.summary.total_suites,
            report.summary.failed_suites,
            report.summary.error_suites,
            report.summary.total_duration_ms as f64 / 1000.0
        ));
        
        for result in &report.results {
            xml.push_str(&format!(
                "  <testsuite name=\"{}\" tests=\"1\" failures=\"{}\" errors=\"{}\" time=\"{:.3}\">\n",
                result.suite_name,
                if result.status == TestStatus::Failed { 1 } else { 0 },
                if result.status == TestStatus::Error { 1 } else { 0 },
                result.duration_ms as f64 / 1000.0
            ));
            
            xml.push_str(&format!(
                "    <testcase name=\"{}\" classname=\"{}\" time=\"{:.3}\"",
                result.suite_name,
                format!("{:?}", result.category),
                result.duration_ms as f64 / 1000.0
            ));
            
            match result.status {
                TestStatus::Passed => {
                    xml.push_str(" />\n");
                },
                TestStatus::Failed => {
                    xml.push_str(">\n");
                    xml.push_str(&format!(
                        "      <failure message=\"Test suite failed\" type=\"TestFailure\">{}</failure>\n",
                        result.stderr.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
                    ));
                    xml.push_str("    </testcase>\n");
                },
                TestStatus::Error => {
                    xml.push_str(">\n");
                    xml.push_str(&format!(
                        "      <error message=\"Test suite error\" type=\"TestError\">{}</error>\n",
                        result.error_details.as_deref().unwrap_or("Unknown error").replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
                    ));
                    xml.push_str("    </testcase>\n");
                },
                TestStatus::Timeout => {
                    xml.push_str(">\n");
                    xml.push_str("      <error message=\"Test suite timeout\" type=\"TestTimeout\">Test suite exceeded timeout limit</error>\n");
                    xml.push_str("    </testcase>\n");
                },
                TestStatus::Skipped => {
                    xml.push_str(">\n");
                    xml.push_str("      <skipped />\n");
                    xml.push_str("    </testcase>\n");
                },
            }
            
            xml.push_str("  </testsuite>\n");
        }
        
        xml.push_str("</testsuites>\n");
        Ok(xml)
    }
    
    /// Generate HTML report for human consumption
    pub fn generate_html_report(&self, report: &TestExecutionReport) -> Result<String> {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>LDC Engine Test Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(r#"
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; color: #333; }
h1, h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h1 { font-size: 2.5em; margin-bottom: 20px; }
h2 { font-size: 1.8em; margin-top: 30px; margin-bottom: 15px; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }
th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
th { background-color: #34495e; color: white; font-weight: 600; text-transform: uppercase; font-size: 0.9em; letter-spacing: 0.5px; }
tr:nth-child(even) { background-color: #f8f9fa; }
tr:hover { background-color: #e8f4fd; }
.passed { color: #27ae60; font-weight: bold; }
.failed { color: #e74c3c; font-weight: bold; }
.error { color: #e67e22; font-weight: bold; }
.timeout { color: #f39c12; font-weight: bold; }
.skipped { color: #95a5a6; font-weight: bold; }
.minor { color: #f39c12; font-weight: bold; }
.major { color: #e67e22; font-weight: bold; }
.critical { color: #e74c3c; font-weight: bold; background-color: #fdf2f2; }
.summary { max-width: 500px; margin: 20px 0; }
.summary td:first-child { font-weight: 600; color: #2c3e50; }
.summary td:last-child { text-align: right; font-weight: bold; }
ul { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
li { margin: 10px 0; padding: 10px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #856404; }
"#);
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");
        
        // Header
        html.push_str(&format!(
            "<h1>LDC Engine Test Report</h1>\n<p>Execution ID: {}</p>\n<p>Timestamp: {}</p>\n",
            report.execution_id,
            report.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        
        if let Some(ref commit) = report.git_commit {
            html.push_str(&format!("<p>Git Commit: {}</p>\n", commit));
        }
        
        // Summary
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<table class=\"summary\">\n");
        html.push_str(&format!("<tr><td>Total Suites</td><td>{}</td></tr>\n", report.summary.total_suites));
        html.push_str(&format!("<tr><td>Passed</td><td class=\"passed\">{}</td></tr>\n", report.summary.passed_suites));
        html.push_str(&format!("<tr><td>Failed</td><td class=\"failed\">{}</td></tr>\n", report.summary.failed_suites));
        html.push_str(&format!("<tr><td>Errors</td><td class=\"error\">{}</td></tr>\n", report.summary.error_suites));
        html.push_str(&format!("<tr><td>Success Rate</td><td>{:.1}%</td></tr>\n", report.summary.success_rate));
        html.push_str(&format!("<tr><td>Total Duration</td><td>{:.2}s</td></tr>\n", report.summary.total_duration_ms as f64 / 1000.0));
        html.push_str("</table>\n");
        
        // Test Results
        html.push_str("<h2>Test Results</h2>\n");
        html.push_str("<table class=\"results\">\n");
        html.push_str("<tr><th>Suite</th><th>Category</th><th>Status</th><th>Duration</th><th>Details</th></tr>\n");
        
        for result in &report.results {
            let status_class = match result.status {
                TestStatus::Passed => "passed",
                TestStatus::Failed => "failed",
                TestStatus::Error => "error",
                TestStatus::Timeout => "timeout",
                TestStatus::Skipped => "skipped",
            };
            
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:?}</td><td class=\"{}\">{:?}</td><td>{:.2}s</td><td>{}</td></tr>\n",
                result.suite_name,
                result.category,
                status_class,
                result.status,
                result.duration_ms as f64 / 1000.0,
                result.error_details.as_deref().unwrap_or("")
            ));
        }
        
        html.push_str("</table>\n");
        
        // Performance Regressions
        if !report.performance_regressions.is_empty() {
            html.push_str("<h2>Performance Regressions</h2>\n");
            html.push_str("<table class=\"regressions\">\n");
            html.push_str("<tr><th>Suite</th><th>Metric</th><th>Baseline</th><th>Current</th><th>Regression</th><th>Severity</th></tr>\n");
            
            for regression in &report.performance_regressions {
                let severity_class = match regression.severity {
                    RegressionSeverity::Minor => "minor",
                    RegressionSeverity::Major => "major",
                    RegressionSeverity::Critical => "critical",
                };
                
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.1}%</td><td class=\"{}\">{:?}</td></tr>\n",
                    regression.suite_name,
                    regression.metric_name,
                    regression.baseline_value,
                    regression.current_value,
                    regression.regression_percent,
                    severity_class,
                    regression.severity
                ));
            }
            
            html.push_str("</table>\n");
        }
        
        // Recommendations
        if !report.recommendations.is_empty() {
            html.push_str("<h2>Recommendations</h2>\n");
            html.push_str("<ul>\n");
            for recommendation in &report.recommendations {
                html.push_str(&format!("<li>{}</li>\n", recommendation.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")));
            }
            html.push_str("</ul>\n");
        }
        
        html.push_str("</body>\n</html>\n");
        Ok(html)
    }
    
    /// Load performance baseline from disk
    pub fn load_performance_baseline(output_dir: &Path) -> Result<Option<PerformanceBaseline>> {
        let baseline_path = output_dir.join("performance_baseline.json");
        
        if baseline_path.exists() {
            let content = std::fs::read_to_string(&baseline_path)?;
            let baseline: PerformanceBaseline = serde_json::from_str(&content)?;
            Ok(Some(baseline))
        } else {
            Ok(None)
        }
    }
    
    /// Update performance baseline with new successful test results
    pub fn update_performance_baseline(&mut self, results: &[TestSuiteResult]) -> Result<()> {
        let mut suite_baselines = HashMap::new();
        
        for result in results {
            if result.status == TestStatus::Passed {
                if let Some(ref metrics) = result.performance_metrics {
                    suite_baselines.insert(result.suite_name.clone(), metrics.clone());
                }
            }
        }
        
        let baseline = PerformanceBaseline {
            timestamp: Utc::now(),
            suite_baselines,
            git_commit: self.get_git_commit().ok(),
        };
        
        // Save baseline
        let baseline_path = self.config.output_directory.join("performance_baseline.json");
        let content = serde_json::to_string_pretty(&baseline)?;
        std::fs::write(&baseline_path, content)?;
        
        self.performance_baseline = Some(baseline);
        
        if self.config.verbose {
            println!("Updated performance baseline");
        }
        
        Ok(())
    }
    
    /// Load test history from disk
    pub fn load_test_history(output_dir: &Path) -> Result<TestHistory> {
        let history_path = output_dir.join("test_history.json");
        
        if history_path.exists() {
            let content = std::fs::read_to_string(&history_path)?;
            let history: TestHistory = serde_json::from_str(&content)?;
            Ok(history)
        } else {
            Ok(TestHistory {
                executions: Vec::new(),
                max_history_entries: 100, // Keep last 100 executions
            })
        }
    }
    
    /// Update test history with new execution record
    pub fn update_test_history(&mut self, record: TestExecutionRecord) -> Result<()> {
        self.test_history.executions.push(record);
        
        // Trim history if it exceeds max entries
        if self.test_history.executions.len() > self.test_history.max_history_entries {
            let excess = self.test_history.executions.len() - self.test_history.max_history_entries;
            self.test_history.executions.drain(0..excess);
        }
        
        // Save history
        let history_path = self.config.output_directory.join("test_history.json");
        let content = serde_json::to_string_pretty(&self.test_history)?;
        std::fs::write(&history_path, content)?;
        
        Ok(())
    }
    
    /// Get current git commit hash
    pub fn get_git_commit(&self) -> Result<String> {
        let output = Command::new("git")
            .args(&["rev-parse", "HEAD"])
            .output()?;
        
        if output.status.success() {
            Ok(String::from_utf8(output.stdout)?.trim().to_string())
        } else {
            Err(anyhow::anyhow!("Failed to get git commit"))
        }
    }
    
    /// Print summary to console
    pub fn print_summary(&self, summary: &TestSummary) {
        println!("\n=== Test Execution Summary ===");
        println!("Total Suites: {}", summary.total_suites);
        println!("Passed: {}", summary.passed_suites);
        println!("Failed: {}", summary.failed_suites);
        println!("Errors: {}", summary.error_suites);
        println!("Skipped: {}", summary.skipped_suites);
        println!("Success Rate: {:.1}%", summary.success_rate);
        println!("Total Duration: {:.2}s", summary.total_duration_ms as f64 / 1000.0);
        println!("Overall Status: {:?}", summary.overall_status);
        println!("==============================\n");
    }
    
    /// Get proper exit code for CI/CD integration
    pub fn get_exit_code(&self, report: &TestExecutionReport) -> i32 {
        match report.summary.overall_status {
            TestStatus::Passed => 0,
            TestStatus::Failed => 1,
            TestStatus::Error => 2,
            TestStatus::Timeout => 3,
            TestStatus::Skipped => 0, // Skipped tests don't fail the build
        }
    }
    
    /// Cleanup resources and temporary files
    pub fn cleanup(&self) -> Result<()> {
        // Clean up old test reports (keep last 10)
        let mut report_files: Vec<_> = std::fs::read_dir(&self.config.output_directory)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()? == "json" && 
                   path.file_stem()?.to_str()?.starts_with("test_run_") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by modification time (newest first)
        report_files.sort_by_key(|path| {
            std::fs::metadata(path)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        });
        report_files.reverse();
        
        // Remove old reports (keep last 10)
        for old_report in report_files.iter().skip(10) {
            if let Err(e) = std::fs::remove_file(old_report) {
                eprintln!("Warning: Failed to remove old report {}: {}", old_report.display(), e);
            }
        }
        
        if self.config.verbose {
            println!("Cleanup completed");
        }
        
        Ok(())
    }
}

/// Result of command execution with timeout
#[derive(Debug)]
enum ExecutionResult {
    Success(ExitStatus),
    Timeout,
    Error(anyhow::Error),
}

impl TestHistory {
    /// Get test trend analysis
    pub fn get_trend_analysis(&self, suite_name: &str) -> Option<TrendAnalysis> {
        let recent_executions: Vec<_> = self.executions
            .iter()
            .rev()
            .take(10) // Last 10 executions
            .collect();
        
        if recent_executions.len() < 2 {
            return None;
        }
        
        let mut durations = Vec::new();
        let mut success_rates = Vec::new();
        
        for execution in &recent_executions {
            if let Some(result) = execution.results.iter().find(|r| r.suite_name == suite_name) {
                durations.push(result.duration_ms as f64);
                success_rates.push(if result.status == TestStatus::Passed { 1.0 } else { 0.0 });
            }
        }
        
        if durations.len() < 2 {
            return None;
        }
        
        // Calculate trends
        let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let recent_avg = durations.iter().take(3).sum::<f64>() / (3.0_f64).min(durations.len() as f64);
        let duration_trend = ((recent_avg - avg_duration) / avg_duration) * 100.0;
        
        let avg_success_rate = success_rates.iter().sum::<f64>() / success_rates.len() as f64;
        
        Some(TrendAnalysis {
            suite_name: suite_name.to_string(),
            average_duration_ms: avg_duration,
            duration_trend_percent: duration_trend,
            success_rate: avg_success_rate,
            execution_count: durations.len(),
        })
    }
}

/// Trend analysis for a test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub suite_name: String,
    pub average_duration_ms: f64,
    pub duration_trend_percent: f64, // Positive = getting slower, negative = getting faster
    pub success_rate: f64,
    pub execution_count: usize,
}