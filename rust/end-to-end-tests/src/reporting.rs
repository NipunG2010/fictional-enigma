//! Test reporting and result aggregation
//! 
//! Handles collection, aggregation, and formatting of test results
//! for various output formats including JSON and HTML reports.

use crate::{performance::PerformanceReport, TestStatus, Uuid};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive test report containing all test results and analysis
#[derive(Debug, Serialize, Deserialize)]
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
#[derive(Debug, Serialize, Deserialize)]
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
        
        // Summary section
        html.push_str("    <div class=\"summary\">\n");
        html.push_str("        <h2>Test Summary</h2>\n");
        html.push_str(&format!("        <p><strong>Duration:</strong> {:.2} minutes</p>\n", self.summary.total_duration_minutes));
        html.push_str(&format!("        <p><strong>Pass Rate:</strong> {:.1}%</p>\n", self.summary.overall_pass_rate * 100.0));
        html.push_str(&format!("        <p><strong>Critical Failures:</strong> {}</p>\n", self.summary.critical_failures));
        html.push_str(&format!("        <p><strong>Performance Violations:</strong> {}</p>\n", self.summary.performance_violations));
        html.push_str(&format!("        <p><strong>System Health Score:</strong> {:.1}%</p>\n", self.summary.system_health_score * 100.0));
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