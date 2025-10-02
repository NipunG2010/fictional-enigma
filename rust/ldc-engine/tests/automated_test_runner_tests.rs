use ldc_engine::automated_test_runner::{
    AutomatedTestRunner, TestRunnerConfig, TestSelectionStrategy, TestCategory
};
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_automated_test_runner_creation() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = TestRunnerConfig::default();
    config.output_directory = temp_dir.path().to_path_buf();
    config.verbose = false;
    
    let runner = AutomatedTestRunner::new(config);
    assert!(runner.is_ok());
    
    let runner = runner.unwrap();
    assert!(!runner.test_suites.is_empty());
}

#[test]
fn test_test_suite_discovery() {
    let suites = AutomatedTestRunner::discover_test_suites().unwrap();
    
    // Should discover standard test suites
    assert!(!suites.is_empty());
    
    // Check for expected test suites
    let suite_names: Vec<_> = suites.iter().map(|s| s.name.as_str()).collect();
    assert!(suite_names.contains(&"unit_tests"));
    assert!(suite_names.contains(&"integration_tests"));
    assert!(suite_names.contains(&"performance_validation"));
}

#[test]
fn test_test_selection_strategies() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = TestRunnerConfig::default();
    config.output_directory = temp_dir.path().to_path_buf();
    
    // Test category selection
    config.test_selection = TestSelectionStrategy::Categories(vec![TestCategory::Unit]);
    let runner = AutomatedTestRunner::new(config.clone()).unwrap();
    let selected = runner.select_test_suites().unwrap();
    assert!(selected.iter().all(|s| s.category == TestCategory::Unit));
    
    // Test pattern selection
    config.test_selection = TestSelectionStrategy::Pattern("unit".to_string());
    let runner = AutomatedTestRunner::new(config.clone()).unwrap();
    let selected = runner.select_test_suites().unwrap();
    assert!(selected.iter().all(|s| s.name.contains("unit")));
    
    // Test changed files selection
    config.test_selection = TestSelectionStrategy::ChangedFiles(vec![
        PathBuf::from("src/lib.rs")
    ]);
    let runner = AutomatedTestRunner::new(config).unwrap();
    let selected = runner.select_test_suites().unwrap();
    // Should include unit tests at minimum
    assert!(selected.iter().any(|s| s.name == "unit_tests"));
}

#[test]
fn test_pattern_matching() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = TestRunnerConfig::default();
    config.output_directory = temp_dir.path().to_path_buf();
    let runner = AutomatedTestRunner::new(config).unwrap();
    
    // Test glob pattern matching
    assert!(runner.matches_pattern(&PathBuf::from("src/lib.rs"), "src/**/*.rs"));
    assert!(runner.matches_pattern(&PathBuf::from("src/main.rs"), "src/*.rs"));
    assert!(!runner.matches_pattern(&PathBuf::from("tests/test.rs"), "src/*.rs"));
    assert!(runner.matches_pattern(&PathBuf::from("src/automated_test_runner.rs"), "src/automated_test_runner.rs"));
}

#[test]
fn test_performance_regression_calculation() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = TestRunnerConfig::default();
    config.output_directory = temp_dir.path().to_path_buf();
    let runner = AutomatedTestRunner::new(config).unwrap();
    
    // Test regression calculation
    assert_eq!(runner.calculate_regression(100.0, 110.0), 10.0);
    assert_eq!(runner.calculate_regression(100.0, 90.0), -10.0);
    assert_eq!(runner.calculate_regression(0.0, 10.0), 0.0); // Handle division by zero
}

#[test]
fn test_report_generation() {
    use ldc_engine::automated_test_runner::{
        TestExecutionReport, TestSummary, TestStatus, TestSuiteResult, TestCategory,
        PerformanceMetrics
    };
    use chrono::Utc;
    use std::collections::HashMap;
    
    let temp_dir = TempDir::new().unwrap();
    let mut config = TestRunnerConfig::default();
    config.output_directory = temp_dir.path().to_path_buf();
    let runner = AutomatedTestRunner::new(config).unwrap();
    
    // Create a mock test result
    let result = TestSuiteResult {
        suite_name: "test_suite".to_string(),
        category: TestCategory::Unit,
        status: TestStatus::Passed,
        start_time: Utc::now(),
        end_time: Utc::now(),
        duration_ms: 1000,
        exit_code: Some(0),
        stdout: "Test output".to_string(),
        stderr: "".to_string(),
        performance_metrics: Some(PerformanceMetrics {
            execution_time_ms: 1000,
            memory_usage_mb: 10.0,
            cpu_usage_percent: 50.0,
            custom_metrics: HashMap::new(),
        }),
        error_details: None,
    };
    
    let summary = TestSummary {
        total_suites: 1,
        passed_suites: 1,
        failed_suites: 0,
        skipped_suites: 0,
        error_suites: 0,
        total_duration_ms: 1000,
        success_rate: 100.0,
        overall_status: TestStatus::Passed,
    };
    
    let report = TestExecutionReport {
        execution_id: "test_123".to_string(),
        timestamp: Utc::now(),
        git_commit: None,
        config: runner.config.clone(),
        results: vec![result],
        summary,
        performance_regressions: Vec::new(),
        recommendations: Vec::new(),
    };
    
    // Test JUnit XML generation
    let xml = runner.generate_junit_xml(&report).unwrap();
    assert!(xml.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assert!(xml.contains("<testsuites"));
    assert!(xml.contains("test_suite"));
    
    // Test HTML generation
    let html = runner.generate_html_report(&report).unwrap();
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<title>LDC Engine Test Report</title>"));
    assert!(html.contains("test_suite"));
}