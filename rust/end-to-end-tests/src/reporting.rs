//! Test reporting and result aggregation
//! 
//! Handles collection, aggregation, and formatting of test results
//! for various output formats including JSON and HTML reports.

use crate::{performance::PerformanceReport, TestStatus, Uuid};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive test report containing all test results and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    /// Test session identifier
    pub session_id: Uuid,
    
    /// Test execution summary
    pub summary: TestSummary,
    
    /// Results from all test suites
    pub results: Vec<TestResults>,
    
    /// Recommendations based on test results
    pub recommendations: Vec<String>,
    
    /// Report generation timestamp
    pub generated_at: i64,
}

/// High-level test execution summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    /// Total test execution duration in minutes
    pub total_duration_minutes: f64,
    
    /// Overall pass rate (0.0 to 1.0)
    pub overall_pass_rate: f64,
    
    /// Number of critical test failures
    pub critical_failures: u32,
    
    /// Number of performance requirement violations
    pub performance_violations: u32,
    
    /// Overall system health score (0.0 to 1.0)
    pub system_health_score: f64,
}

/// Test results for a specific test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    /// Name of the test suite
    pub test_suite: String,
    
    /// Test suite start timestamp
    pub start_time: i64,
    
    /// Test suite end timestamp
    pub end_time: i64,
    
    /// Total number of tests executed
    pub total_tests: u32,
    
    /// Number of tests that passed
    pub passed_tests: u32,
    
    /// Number of tests that failed
    pub failed_tests: u32,
    
    /// Individual test case results
    pub test_cases: Vec<TestCaseResult>,
    
    /// Performance metrics for the suite (if applicable)
    pub performance_metrics: Option<PerformanceReport>,
    
    /// Suite execution duration in milliseconds
    pub suite_duration_ms: u64,
}

/// Result of an individual test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseResult {
    /// Test case name
    pub name: String,
    
    /// Test case unique identifier
    pub id: Uuid,
    
    /// Test execution status
    pub status: TestStatus,
    
    /// Test execution duration in milliseconds
    pub duration_ms: u64,
    
    /// Error message if test failed
    pub error_message: Option<String>,
    
    /// Test-specific metrics
    pub metrics: HashMap<String, f64>,
    
    /// Detailed validation results
    pub validation_details: Vec<ValidationDetail>,
}

/// Detailed validation result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationDetail {
    /// Validation check name
    pub check_name: String,
    
    /// Whether the validation passed
    pub passed: bool,
    
    /// Expected value (if applicable)
    pub expected: Option<String>,
    
    /// Actual value (if applicable)
    pub actual: Option<String>,
    
    /// Validation message
    pub message: String,
}

/// CI/CD integration summary for automated testing
#[derive(Debug, Serialize, Deserialize)]
pub struct CISummary {
    /// Test session identifier
    pub session_id: Uuid,
    
    /// Total number of tests executed
    pub total_tests: u32,
    
    /// Number of tests that passed
    pub passed_tests: u32,
    
    /// Number of tests that failed
    pub failed_tests: u32,
    
    /// Overall pass rate (0.0 to 1.0)
    pub pass_rate: f64,
    
    /// Total execution duration in minutes
    pub duration_minutes: f64,
    
    /// Number of critical failures
    pub critical_failures: u32,
    
    /// Number of performance issues detected
    pub performance_issues: u32,
    
    /// System health score (0.0 to 1.0)
    pub health_score: f64,
    
    /// Slowest test cases
    pub slowest_tests: Vec<SlowTestInfo>,
    
    /// Names of failed tests
    pub failed_test_names: Vec<String>,
    
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    
    /// Report generation timestamp
    pub generated_at: i64,
}

/// Information about slow test cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowTestInfo {
    /// Test suite name
    pub suite_name: String,
    
    /// Test case name
    pub test_name: String,
    
    /// Test duration in milliseconds
    pub duration_ms: u64,
}

/// Trend analysis data for performance regression detection
#[derive(Debug, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Analysis timestamp
    pub analysis_timestamp: i64,
    
    /// Number of reports analyzed
    pub report_count: usize,
    
    /// Time span of analysis in days
    pub time_span_days: f64,
    
    /// Pass rate trend (change per day)
    pub pass_rate_trend: Option<f64>,
    
    /// Duration trend (change per day in minutes)
    pub duration_trend: Option<f64>,
    
    /// Health score trend (change per day)
    pub health_score_trend: Option<f64>,
    
    /// Performance regression indicators
    pub performance_regressions: Vec<PerformanceRegression>,
    
    /// Historical data points
    pub data_points: Vec<TrendDataPoint>,
}

/// Individual data point for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendDataPoint {
    /// Timestamp of the test report
    pub timestamp: i64,
    
    /// Pass rate at this point
    pub pass_rate: f64,
    
    /// Total number of tests
    pub total_tests: u32,
    
    /// Test duration in minutes
    pub duration_minutes: f64,
    
    /// System health score
    pub health_score: f64,
    
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

/// Performance regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Metric name that regressed
    pub metric_name: String,
    
    /// Regression severity (Low, Medium, High, Critical)
    pub severity: RegressionSeverity,
    
    /// Current value
    pub current_value: f64,
    
    /// Baseline value
    pub baseline_value: f64,
    
    /// Percentage change
    pub percentage_change: f64,
    
    /// Description of the regression
    pub description: String,
}

/// Severity levels for performance regressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Comparison report between two test runs
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Comparison timestamp
    pub comparison_timestamp: i64,
    
    /// Current test report session ID
    pub current_session_id: Uuid,
    
    /// Baseline test report session ID
    pub baseline_session_id: Uuid,
    
    /// Pass rate change
    pub pass_rate_change: f64,
    
    /// Duration change in minutes
    pub duration_change: f64,
    
    /// Health score change
    pub health_score_change: f64,
    
    /// New test failures
    pub new_failures: Vec<String>,
    
    /// Resolved test failures
    pub resolved_failures: Vec<String>,
    
    /// Performance regressions detected
    pub performance_regressions: Vec<PerformanceRegression>,
    
    /// Overall comparison summary
    pub summary: ComparisonSummary,
}

/// Summary of comparison results
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Overall status (Improved, Degraded, Stable)
    pub status: ComparisonStatus,
    
    /// Key changes detected
    pub key_changes: Vec<String>,
    
    /// Recommendations based on comparison
    pub recommendations: Vec<String>,
}

/// Status of comparison between test runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonStatus {
    Improved,
    Degraded,
    Stable,
}

impl TestResults {
    /// Create a failed test suite result
    pub fn failed_suite(suite_name: String, error_message: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        
        Self {
            test_suite: suite_name,
            start_time: now,
            end_time: now,
            total_tests: 1,
            passed_tests: 0,
            failed_tests: 1,
            test_cases: vec![TestCaseResult {
                name: "suite_initialization".to_string(),
                id: Uuid::new_v4(),
                status: TestStatus::Failed,
                duration_ms: 0,
                error_message: Some(error_message),
                metrics: HashMap::new(),
                validation_details: Vec::new(),
            }],
            performance_metrics: None,
            suite_duration_ms: 0,
        }
    }
    
    /// Calculate pass rate for this test suite
    pub fn pass_rate(&self) -> f64 {
        if self.total_tests == 0 {
            0.0
        } else {
            self.passed_tests as f64 / self.total_tests as f64
        }
    }
    
    /// Get average test duration in milliseconds
    pub fn average_test_duration_ms(&self) -> f64 {
        if self.test_cases.is_empty() {
            0.0
        } else {
            let total_duration: u64 = self.test_cases.iter().map(|tc| tc.duration_ms).sum();
            total_duration as f64 / self.test_cases.len() as f64
        }
    }
    
    /// Get tests that failed or timed out
    pub fn failed_tests(&self) -> Vec<&TestCaseResult> {
        self.test_cases
            .iter()
            .filter(|tc| tc.status != TestStatus::Passed)
            .collect()
    }
    
    /// Get tests that exceeded expected duration
    pub fn slow_tests(&self, threshold_ms: u64) -> Vec<&TestCaseResult> {
        self.test_cases
            .iter()
            .filter(|tc| tc.duration_ms > threshold_ms)
            .collect()
    }
}

impl TestCaseResult {
    /// Create a successful test case result
    pub fn success(name: String, duration_ms: u64, metrics: HashMap<String, f64>) -> Self {
        Self {
            name,
            id: Uuid::new_v4(),
            status: TestStatus::Passed,
            duration_ms,
            error_message: None,
            metrics,
            validation_details: Vec::new(),
        }
    }
    
    /// Create a failed test case result
    pub fn failure(name: String, duration_ms: u64, error_message: String) -> Self {
        Self {
            name,
            id: Uuid::new_v4(),
            status: TestStatus::Failed,
            duration_ms,
            error_message: Some(error_message),
            metrics: HashMap::new(),
            validation_details: Vec::new(),
        }
    }
    
    /// Create a timed out test case result
    pub fn timeout(name: String, timeout_ms: u64) -> Self {
        Self {
            name,
            id: Uuid::new_v4(),
            status: TestStatus::Timeout,
            duration_ms: timeout_ms,
            error_message: Some(format!("Test timed out after {}ms", timeout_ms)),
            metrics: HashMap::new(),
            validation_details: Vec::new(),
        }
    }
    
    /// Add a validation detail to this test case
    pub fn add_validation_detail(&mut self, detail: ValidationDetail) {
        self.validation_details.push(detail);
    }
    
    /// Check if this test case has any validation failures
    pub fn has_validation_failures(&self) -> bool {
        self.validation_details.iter().any(|vd| !vd.passed)
    }
}

impl ValidationDetail {
    /// Create a successful validation detail
    pub fn success(check_name: String, message: String) -> Self {
        Self {
            check_name,
            passed: true,
            expected: None,
            actual: None,
            message,
        }
    }
    
    /// Create a failed validation detail with expected and actual values
    pub fn failure(
        check_name: String,
        expected: String,
        actual: String,
        message: String,
    ) -> Self {
        Self {
            check_name,
            passed: false,
            expected: Some(expected),
            actual: Some(actual),
            message,
        }
    }
    
    /// Create a failed validation detail with just a message
    pub fn failure_with_message(check_name: String, message: String) -> Self {
        Self {
            check_name,
            passed: false,
            expected: None,
            actual: None,
            message,
        }
    }
}

impl TestReport {
    /// Save the report as JSON to the specified path
    pub fn save_json<P: AsRef<std::path::Path>>(&self, path: P) -> crate::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    /// Save the report as HTML to the specified path
    pub fn save_html<P: AsRef<std::path::Path>>(&self, path: P) -> crate::Result<()> {
        let html = self.generate_html_report();
        std::fs::write(path, html)?;
        Ok(())
    }
    
    /// Save both JSON and HTML reports with timestamped filenames
    pub fn save_timestamped_reports<P: AsRef<std::path::Path>>(&self, output_dir: P) -> crate::Result<(std::path::PathBuf, std::path::PathBuf)> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        
        let timestamp = chrono::DateTime::from_timestamp(self.generated_at, 0)
            .unwrap_or_default()
            .format("%Y%m%d_%H%M%S");
        
        let json_path = output_dir.join(format!("test_report_{}.json", timestamp));
        let html_path = output_dir.join(format!("test_report_{}.html", timestamp));
        
        self.save_json(&json_path)?;
        self.save_html(&html_path)?;
        
        Ok((json_path, html_path))
    }
    
    /// Generate a summary report for CI/CD integration
    pub fn generate_ci_summary(&self) -> CISummary {
        let (total_tests, passed_tests, failed_tests) = self.overall_stats();
        
        let critical_failures = self.results.iter()
            .flat_map(|suite| &suite.test_cases)
            .filter(|tc| tc.status == crate::TestStatus::Failed)
            .count();
        
        let performance_issues = self.results.iter()
            .filter_map(|suite| suite.performance_metrics.as_ref())
            .map(|pm| self.count_performance_violations(pm))
            .sum::<u32>();
        
        let slowest_tests = self.get_slowest_tests(5);
        let failed_test_names = self.get_failed_test_names();
        
        CISummary {
            session_id: self.session_id,
            total_tests,
            passed_tests,
            failed_tests,
            pass_rate: self.summary.overall_pass_rate,
            duration_minutes: self.summary.total_duration_minutes,
            critical_failures: critical_failures as u32,
            performance_issues,
            health_score: self.summary.system_health_score,
            slowest_tests,
            failed_test_names,
            recommendations: self.recommendations.clone(),
            generated_at: self.generated_at,
        }
    }
    
    /// Get the slowest test cases across all suites
    pub fn get_slowest_tests(&self, limit: usize) -> Vec<SlowTestInfo> {
        let mut all_tests: Vec<SlowTestInfo> = self.results
            .iter()
            .flat_map(|suite| {
                suite.test_cases.iter().map(|tc| SlowTestInfo {
                    suite_name: suite.test_suite.clone(),
                    test_name: tc.name.clone(),
                    duration_ms: tc.duration_ms,
                })
            })
            .collect();
        
        all_tests.sort_by(|a, b| b.duration_ms.cmp(&a.duration_ms));
        all_tests.into_iter().take(limit).collect()
    }
    
    /// Get names of all failed tests
    pub fn get_failed_test_names(&self) -> Vec<String> {
        self.results
            .iter()
            .flat_map(|suite| {
                suite.test_cases.iter()
                    .filter(|tc| tc.status != crate::TestStatus::Passed)
                    .map(|tc| format!("{}::{}", suite.test_suite, tc.name))
            })
            .collect()
    }
    
    /// Count performance violations in a performance report
    fn count_performance_violations(&self, _performance_report: &crate::performance::PerformanceReport) -> u32 {
        // This would be implemented based on the actual PerformanceReport structure
        // For now, return 0 as a placeholder
        0
    }
    
    /// Generate HTML report content
    fn generate_html_report(&self) -> String {
        let mut html = String::new();
        
        // HTML header
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        html.push_str("    <title>IMP End-to-End Test Report</title>\n");
        html.push_str("    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n");
        html.push_str("    <style>\n");
        html.push_str(include_str!("../assets/report_styles.css"));
        html.push_str("    </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");
        
        // Report header
        html.push_str("    <div class=\"header\">\n");
        html.push_str("        <h1>IMP End-to-End Test Report</h1>\n");
        html.push_str(&format!("        <p>Session ID: {}</p>\n", self.session_id));
        html.push_str(&format!("        <p>Generated: {}</p>\n", 
            chrono::DateTime::from_timestamp(self.generated_at, 0)
                .unwrap_or_default()
                .format("%Y-%m-%d %H:%M:%S UTC")));
        html.push_str("    </div>\n");
        
        // Summary section with charts
        html.push_str("    <div class=\"summary\">\n");
        html.push_str("        <h2>Test Summary</h2>\n");
        html.push_str("        <div class=\"summary-grid\">\n");
        html.push_str("            <div class=\"summary-stats\">\n");
        html.push_str(&format!("                <p><strong>Duration:</strong> {:.2} minutes</p>\n", self.summary.total_duration_minutes));
        html.push_str(&format!("                <p><strong>Pass Rate:</strong> {:.1}%</p>\n", self.summary.overall_pass_rate * 100.0));
        html.push_str(&format!("                <p><strong>Critical Failures:</strong> {}</p>\n", self.summary.critical_failures));
        html.push_str(&format!("                <p><strong>Performance Violations:</strong> {}</p>\n", self.summary.performance_violations));
        html.push_str(&format!("                <p><strong>System Health Score:</strong> {:.1}%</p>\n", self.summary.system_health_score * 100.0));
        html.push_str("            </div>\n");
        html.push_str("            <div class=\"summary-charts\">\n");
        html.push_str("                <canvas id=\"passRateChart\" width=\"300\" height=\"200\"></canvas>\n");
        html.push_str("            </div>\n");
        html.push_str("        </div>\n");
        html.push_str("    </div>\n");
        
        // Test results section
        html.push_str("    <div class=\"results\">\n");
        html.push_str("        <h2>Test Results</h2>\n");
        
        for result in &self.results {
            html.push_str("        <div class=\"test-suite\">\n");
            html.push_str(&format!("            <h3>{}</h3>\n", result.test_suite));
            html.push_str(&format!("            <p>Tests: {} | Passed: {} | Failed: {} | Pass Rate: {:.1}%</p>\n",
                result.total_tests, result.passed_tests, result.failed_tests, result.pass_rate() * 100.0));
            
            // Test cases table
            html.push_str("            <table class=\"test-cases\">\n");
            html.push_str("                <thead>\n");
            html.push_str("                    <tr><th>Test Case</th><th>Status</th><th>Duration (ms)</th><th>Error</th></tr>\n");
            html.push_str("                </thead>\n");
            html.push_str("                <tbody>\n");
            
            for test_case in &result.test_cases {
                let status_class = match test_case.status {
                    TestStatus::Passed => "passed",
                    TestStatus::Failed => "failed",
                    TestStatus::Timeout => "timeout",
                    TestStatus::Skipped => "skipped",
                };
                
                html.push_str(&format!("                    <tr class=\"{}\">\n", status_class));
                html.push_str(&format!("                        <td>{}</td>\n", test_case.name));
                html.push_str(&format!("                        <td>{:?}</td>\n", test_case.status));
                html.push_str(&format!("                        <td>{}</td>\n", test_case.duration_ms));
                html.push_str(&format!("                        <td>{}</td>\n", 
                    test_case.error_message.as_deref().unwrap_or("")));
                html.push_str("                    </tr>\n");
            }
            
            html.push_str("                </tbody>\n");
            html.push_str("            </table>\n");
            html.push_str("        </div>\n");
        }
        
        html.push_str("    </div>\n");
        
        // Recommendations section
        if !self.recommendations.is_empty() {
            html.push_str("    <div class=\"recommendations\">\n");
            html.push_str("        <h2>Recommendations</h2>\n");
            html.push_str("        <ul>\n");
            for recommendation in &self.recommendations {
                html.push_str(&format!("            <li>{}</li>\n", recommendation));
            }
            html.push_str("        </ul>\n");
            html.push_str("    </div>\n");
        }
        
        // JavaScript for charts
        html.push_str("    <script>\n");
        html.push_str("        // Pass Rate Pie Chart\n");
        html.push_str("        const ctx = document.getElementById('passRateChart').getContext('2d');\n");
        
        let (total_tests, passed_tests, failed_tests) = self.overall_stats();
        html.push_str(&format!("        const passedTests = {};\n", passed_tests));
        html.push_str(&format!("        const failedTests = {};\n", failed_tests));
        
        html.push_str("        new Chart(ctx, {\n");
        html.push_str("            type: 'doughnut',\n");
        html.push_str("            data: {\n");
        html.push_str("                labels: ['Passed', 'Failed'],\n");
        html.push_str("                datasets: [{\n");
        html.push_str("                    data: [passedTests, failedTests],\n");
        html.push_str("                    backgroundColor: ['#27ae60', '#e74c3c'],\n");
        html.push_str("                    borderWidth: 2,\n");
        html.push_str("                    borderColor: '#fff'\n");
        html.push_str("                }]\n");
        html.push_str("            },\n");
        html.push_str("            options: {\n");
        html.push_str("                responsive: true,\n");
        html.push_str("                plugins: {\n");
        html.push_str("                    title: {\n");
        html.push_str("                        display: true,\n");
        html.push_str("                        text: 'Test Results Distribution'\n");
        html.push_str("                    },\n");
        html.push_str("                    legend: {\n");
        html.push_str("                        position: 'bottom'\n");
        html.push_str("                    }\n");
        html.push_str("                }\n");
        html.push_str("            }\n");
        html.push_str("        });\n");
        
        // Add suite performance chart if we have multiple suites
        if self.results.len() > 1 {
            html.push_str("        \n");
            html.push_str("        // Suite Performance Chart\n");
            html.push_str("        const suiteNames = [");
            for (i, result) in self.results.iter().enumerate() {
                if i > 0 { html.push_str(", "); }
                html.push_str(&format!("'{}'", result.test_suite));
            }
            html.push_str("];\n");
            
            html.push_str("        const suitePassRates = [");
            for (i, result) in self.results.iter().enumerate() {
                if i > 0 { html.push_str(", "); }
                html.push_str(&format!("{:.1}", result.pass_rate() * 100.0));
            }
            html.push_str("];\n");
            
            html.push_str("        const suiteDurations = [");
            for (i, result) in self.results.iter().enumerate() {
                if i > 0 { html.push_str(", "); }
                html.push_str(&format!("{:.2}", result.suite_duration_ms as f64 / 1000.0));
            }
            html.push_str("];\n");
        }
        
        html.push_str("    </script>\n");
        
        // HTML footer
        html.push_str("</body>\n");
        html.push_str("</html>\n");
        
        html
    }
    
    /// Get overall statistics across all test suites
    pub fn overall_stats(&self) -> (u32, u32, u32) {
        let total_tests: u32 = self.results.iter().map(|r| r.total_tests).sum();
        let total_passed: u32 = self.results.iter().map(|r| r.passed_tests).sum();
        let total_failed: u32 = self.results.iter().map(|r| r.failed_tests).sum();
        (total_tests, total_passed, total_failed)
    }
    
    /// Get all failed test cases across all suites
    pub fn all_failed_tests(&self) -> Vec<(&str, &TestCaseResult)> {
        self.results
            .iter()
            .flat_map(|suite| {
                suite.failed_tests()
                    .into_iter()
                    .map(|tc| (suite.test_suite.as_str(), tc))
            })
            .collect()
    }
    
    /// Perform trend analysis on multiple test reports
    pub fn analyze_trends(reports: &[TestReport]) -> TrendAnalysis {
        if reports.is_empty() {
            return TrendAnalysis {
                analysis_timestamp: chrono::Utc::now().timestamp(),
                report_count: 0,
                time_span_days: 0.0,
                pass_rate_trend: None,
                duration_trend: None,
                health_score_trend: None,
                performance_regressions: Vec::new(),
                data_points: Vec::new(),
            };
        }
        
        // Sort reports by timestamp
        let mut sorted_reports = reports.to_vec();
        sorted_reports.sort_by_key(|r| r.generated_at);
        
        // Convert to data points
        let data_points: Vec<TrendDataPoint> = sorted_reports
            .iter()
            .map(|report| {
                let (total_tests, _, _) = report.overall_stats();
                TrendDataPoint {
                    timestamp: report.generated_at,
                    pass_rate: report.summary.overall_pass_rate,
                    total_tests,
                    duration_minutes: report.summary.total_duration_minutes,
                    health_score: report.summary.system_health_score,
                    performance_metrics: HashMap::new(), // Could be populated from performance reports
                }
            })
            .collect();
        
        let time_span_days = if data_points.len() > 1 {
            let first = data_points.first().unwrap().timestamp;
            let last = data_points.last().unwrap().timestamp;
            (last - first) as f64 / (24.0 * 3600.0)
        } else {
            0.0
        };
        
        // Calculate trends using linear regression
        let pass_rate_trend = Self::calculate_trend(&data_points, |dp| dp.pass_rate);
        let duration_trend = Self::calculate_trend(&data_points, |dp| dp.duration_minutes);
        let health_score_trend = Self::calculate_trend(&data_points, |dp| dp.health_score);
        
        // Detect performance regressions
        let performance_regressions = Self::detect_performance_regressions(&data_points);
        
        TrendAnalysis {
            analysis_timestamp: chrono::Utc::now().timestamp(),
            report_count: reports.len(),
            time_span_days,
            pass_rate_trend,
            duration_trend,
            health_score_trend,
            performance_regressions,
            data_points,
        }
    }
    
    /// Compare two test reports
    pub fn compare_reports(current: &TestReport, baseline: &TestReport) -> ComparisonReport {
        let pass_rate_change = current.summary.overall_pass_rate - baseline.summary.overall_pass_rate;
        let duration_change = current.summary.total_duration_minutes - baseline.summary.total_duration_minutes;
        let health_score_change = current.summary.system_health_score - baseline.summary.system_health_score;
        
        // Identify new and resolved failures
        let current_failures: std::collections::HashSet<String> = current.all_failed_tests()
            .into_iter()
            .map(|(suite, test)| format!("{}::{}", suite, test.name))
            .collect();
        
        let baseline_failures: std::collections::HashSet<String> = baseline.all_failed_tests()
            .into_iter()
            .map(|(suite, test)| format!("{}::{}", suite, test.name))
            .collect();
        
        let new_failures: Vec<String> = current_failures.difference(&baseline_failures)
            .cloned()
            .collect();
        
        let resolved_failures: Vec<String> = baseline_failures.difference(&current_failures)
            .cloned()
            .collect();
        
        // Detect performance regressions
        let performance_regressions = Self::detect_comparison_regressions(current, baseline);
        
        // Determine overall status
        let status = if pass_rate_change < -0.05 || health_score_change < -0.1 || !performance_regressions.is_empty() {
            ComparisonStatus::Degraded
        } else if pass_rate_change > 0.05 || health_score_change > 0.1 {
            ComparisonStatus::Improved
        } else {
            ComparisonStatus::Stable
        };
        
        // Generate key changes and recommendations
        let mut key_changes = Vec::new();
        let mut recommendations = Vec::new();
        
        if pass_rate_change.abs() > 0.01 {
            key_changes.push(format!("Pass rate changed by {:.1}%", pass_rate_change * 100.0));
        }
        
        if duration_change.abs() > 0.5 {
            key_changes.push(format!("Test duration changed by {:.1} minutes", duration_change));
        }
        
        if !new_failures.is_empty() {
            key_changes.push(format!("{} new test failures", new_failures.len()));
            recommendations.push("Investigate new test failures and fix underlying issues".to_string());
        }
        
        if !performance_regressions.is_empty() {
            key_changes.push(format!("{} performance regressions detected", performance_regressions.len()));
            recommendations.push("Review performance regressions and optimize affected components".to_string());
        }
        
        let summary = ComparisonSummary {
            status,
            key_changes,
            recommendations,
        };
        
        ComparisonReport {
            comparison_timestamp: chrono::Utc::now().timestamp(),
            current_session_id: current.session_id,
            baseline_session_id: baseline.session_id,
            pass_rate_change,
            duration_change,
            health_score_change,
            new_failures,
            resolved_failures,
            performance_regressions,
            summary,
        }
    }
    
    /// Calculate linear trend for a given metric
    fn calculate_trend<F>(data_points: &[TrendDataPoint], value_fn: F) -> Option<f64>
    where
        F: Fn(&TrendDataPoint) -> f64,
    {
        if data_points.len() < 2 {
            return None;
        }
        
        let n = data_points.len() as f64;
        let sum_x: f64 = (0..data_points.len()).map(|i| i as f64).sum();
        let sum_y: f64 = data_points.iter().map(&value_fn).sum();
        let sum_xy: f64 = data_points.iter().enumerate()
            .map(|(i, dp)| i as f64 * value_fn(dp))
            .sum();
        let sum_x2: f64 = (0..data_points.len()).map(|i| (i as f64).powi(2)).sum();
        
        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < f64::EPSILON {
            return None;
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        Some(slope)
    }
    
    /// Detect performance regressions in trend data
    fn detect_performance_regressions(data_points: &[TrendDataPoint]) -> Vec<PerformanceRegression> {
        let mut regressions = Vec::new();
        
        if data_points.len() < 2 {
            return regressions;
        }
        
        let recent = data_points.last().unwrap();
        let baseline = data_points.first().unwrap();
        
        // Check pass rate regression
        let pass_rate_change = (recent.pass_rate - baseline.pass_rate) / baseline.pass_rate;
        if pass_rate_change < -0.1 {
            regressions.push(PerformanceRegression {
                metric_name: "pass_rate".to_string(),
                severity: if pass_rate_change < -0.2 { RegressionSeverity::Critical } else { RegressionSeverity::High },
                current_value: recent.pass_rate,
                baseline_value: baseline.pass_rate,
                percentage_change: pass_rate_change * 100.0,
                description: "Test pass rate has significantly decreased".to_string(),
            });
        }
        
        // Check duration regression
        let duration_change = (recent.duration_minutes - baseline.duration_minutes) / baseline.duration_minutes;
        if duration_change > 0.2 {
            regressions.push(PerformanceRegression {
                metric_name: "test_duration".to_string(),
                severity: if duration_change > 0.5 { RegressionSeverity::High } else { RegressionSeverity::Medium },
                current_value: recent.duration_minutes,
                baseline_value: baseline.duration_minutes,
                percentage_change: duration_change * 100.0,
                description: "Test execution time has significantly increased".to_string(),
            });
        }
        
        regressions
    }
    
    /// Detect performance regressions between two specific reports
    fn detect_comparison_regressions(current: &TestReport, baseline: &TestReport) -> Vec<PerformanceRegression> {
        let mut regressions = Vec::new();
        
        // Check pass rate regression
        let pass_rate_change = (current.summary.overall_pass_rate - baseline.summary.overall_pass_rate) / baseline.summary.overall_pass_rate;
        if pass_rate_change < -0.05 {
            regressions.push(PerformanceRegression {
                metric_name: "pass_rate".to_string(),
                severity: if pass_rate_change < -0.15 { RegressionSeverity::Critical } else { RegressionSeverity::Medium },
                current_value: current.summary.overall_pass_rate,
                baseline_value: baseline.summary.overall_pass_rate,
                percentage_change: pass_rate_change * 100.0,
                description: "Test pass rate decreased compared to baseline".to_string(),
            });
        }
        
        // Check duration regression
        if baseline.summary.total_duration_minutes > 0.0 {
            let duration_change = (current.summary.total_duration_minutes - baseline.summary.total_duration_minutes) / baseline.summary.total_duration_minutes;
            if duration_change > 0.15 {
                regressions.push(PerformanceRegression {
                    metric_name: "test_duration".to_string(),
                    severity: if duration_change > 0.3 { RegressionSeverity::High } else { RegressionSeverity::Medium },
                    current_value: current.summary.total_duration_minutes,
                    baseline_value: baseline.summary.total_duration_minutes,
                    percentage_change: duration_change * 100.0,
                    description: "Test execution time increased compared to baseline".to_string(),
                });
            }
        }
        
        regressions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_test_results_pass_rate() {
        let results = TestResults {
            test_suite: "test_suite".to_string(),
            start_time: 0,
            end_time: 100,
            total_tests: 10,
            passed_tests: 8,
            failed_tests: 2,
            test_cases: Vec::new(),
            performance_metrics: None,
            suite_duration_ms: 1000,
        };
        
        assert_eq!(results.pass_rate(), 0.8);
    }
    
    #[test]
    fn test_validation_detail_creation() {
        let success = ValidationDetail::success("test_check".to_string(), "All good".to_string());
        assert!(success.passed);
        
        let failure = ValidationDetail::failure(
            "test_check".to_string(),
            "expected".to_string(),
            "actual".to_string(),
            "Values don't match".to_string(),
        );
        assert!(!failure.passed);
        assert_eq!(failure.expected, Some("expected".to_string()));
        assert_eq!(failure.actual, Some("actual".to_string()));
    }
    
    #[test]
    fn test_test_case_result_creation() {
        let success = TestCaseResult::success("test1".to_string(), 100, HashMap::new());
        assert_eq!(success.status, TestStatus::Passed);
        assert!(success.error_message.is_none());
        
        let failure = TestCaseResult::failure("test2".to_string(), 200, "Error occurred".to_string());
        assert_eq!(failure.status, TestStatus::Failed);
        assert!(failure.error_message.is_some());
        
        let timeout = TestCaseResult::timeout("test3".to_string(), 5000);
        assert_eq!(timeout.status, TestStatus::Timeout);
        assert!(timeout.error_message.is_some());
    }
    
    #[test]
    fn test_report_json_serialization() {
        let report = TestReport {
            session_id: Uuid::new_v4(),
            summary: TestSummary {
                total_duration_minutes: 5.0,
                overall_pass_rate: 0.9,
                critical_failures: 1,
                performance_violations: 0,
                system_health_score: 0.85,
            },
            results: Vec::new(),
            recommendations: vec!["Test recommendation".to_string()],
            generated_at: chrono::Utc::now().timestamp(),
        };
        
        let json = serde_json::to_string(&report).unwrap();
        let deserialized: TestReport = serde_json::from_str(&json).unwrap();
        
        assert_eq!(report.session_id, deserialized.session_id);
        assert_eq!(report.summary.overall_pass_rate, deserialized.summary.overall_pass_rate);
    }
    
    #[test]
    fn test_report_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let json_path = temp_dir.path().join("report.json");
        let html_path = temp_dir.path().join("report.html");
        
        let report = TestReport {
            session_id: Uuid::new_v4(),
            summary: TestSummary {
                total_duration_minutes: 5.0,
                overall_pass_rate: 0.9,
                critical_failures: 1,
                performance_violations: 0,
                system_health_score: 0.85,
            },
            results: Vec::new(),
            recommendations: vec!["Test recommendation".to_string()],
            generated_at: chrono::Utc::now().timestamp(),
        };
        
        // Test JSON save
        report.save_json(&json_path).unwrap();
        assert!(json_path.exists());
        
        // Test HTML save
        report.save_html(&html_path).unwrap();
        assert!(html_path.exists());
        
        // Verify HTML content contains expected elements
        let html_content = std::fs::read_to_string(&html_path).unwrap();
        assert!(html_content.contains("IMP End-to-End Test Report"));
        assert!(html_content.contains(&report.session_id.to_string()));
    }
}