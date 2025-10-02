use anyhow::Result;
use ldc_engine::automated_test_runner::{
    AutomatedTestRunner, TestRunnerConfig, TestSelectionStrategy, TestCategory
};
use std::path::PathBuf;

/// Demonstration of the automated test runner functionality
/// 
/// This example shows how to:
/// 1. Configure the automated test runner
/// 2. Run different types of test selections
/// 3. Handle test results and reports
/// 4. Integrate with CI/CD pipelines
fn main() -> Result<()> {
    println!("LDC Engine Automated Test Runner Demo");
    println!("=====================================\n");

    // Example 1: Basic test runner configuration
    demo_basic_configuration()?;
    
    // Example 2: Different test selection strategies
    demo_test_selection_strategies()?;
    
    // Example 3: Performance regression detection
    demo_performance_regression_detection()?;
    
    // Example 4: CI/CD integration
    demo_ci_cd_integration()?;

    println!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demonstrate basic test runner configuration
fn demo_basic_configuration() -> Result<()> {
    println!("🔧 Demo 1: Basic Configuration");
    println!("------------------------------");

    // Create a basic configuration
    let config = TestRunnerConfig {
        max_parallel_suites: 4,
        suite_timeout_seconds: 300,
        test_timeout_seconds: 60,
        enable_regression_detection: true,
        regression_threshold_percent: 10.0,
        output_directory: PathBuf::from("demo_test_reports"),
        machine_readable_output: true,
        test_selection: TestSelectionStrategy::All,
        cleanup_timeout_seconds: 30,
        verbose: true,
    };

    println!("Configuration created:");
    println!("  - Max parallel suites: {}", config.max_parallel_suites);
    println!("  - Suite timeout: {}s", config.suite_timeout_seconds);
    println!("  - Regression detection: {}", config.enable_regression_detection);
    println!("  - Output directory: {}", config.output_directory.display());

    // Create the test runner
    let runner = AutomatedTestRunner::new(config)?;
    println!("  - Discovered {} test suites", runner.test_suites.len());

    // List discovered test suites
    println!("\nDiscovered test suites:");
    for suite in &runner.test_suites {
        println!("  - {} ({:?})", suite.name, suite.category);
    }

    println!("✅ Basic configuration demo completed\n");
    Ok(())
}

/// Demonstrate different test selection strategies
fn demo_test_selection_strategies() -> Result<()> {
    println!("🎯 Demo 2: Test Selection Strategies");
    println!("-----------------------------------");

    let base_config = TestRunnerConfig {
        output_directory: PathBuf::from("demo_test_reports"),
        verbose: false,
        ..Default::default()
    };

    // Strategy 1: Run all tests
    println!("Strategy 1: All tests");
    let config_all = TestRunnerConfig {
        test_selection: TestSelectionStrategy::All,
        ..base_config.clone()
    };
    let runner_all = AutomatedTestRunner::new(config_all)?;
    let selected_all = runner_all.select_test_suites()?;
    println!("  - Selected {} out of {} test suites", selected_all.len(), runner_all.test_suites.len());

    // Strategy 2: Run specific categories
    println!("\nStrategy 2: Specific categories (Unit + Mathematical)");
    let config_categories = TestRunnerConfig {
        test_selection: TestSelectionStrategy::Categories(vec![
            TestCategory::Unit,
            TestCategory::Mathematical
        ]),
        ..base_config.clone()
    };
    let runner_categories = AutomatedTestRunner::new(config_categories)?;
    let selected_categories = runner_categories.select_test_suites()?;
    println!("  - Selected {} test suites", selected_categories.len());
    for suite in &selected_categories {
        println!("    - {} ({:?})", suite.name, suite.category);
    }

    // Strategy 3: Pattern matching
    println!("\nStrategy 3: Pattern matching ('unit')");
    let config_pattern = TestRunnerConfig {
        test_selection: TestSelectionStrategy::Pattern("unit".to_string()),
        ..base_config.clone()
    };
    let runner_pattern = AutomatedTestRunner::new(config_pattern)?;
    let selected_pattern = runner_pattern.select_test_suites()?;
    println!("  - Selected {} test suites matching 'unit'", selected_pattern.len());
    for suite in &selected_pattern {
        println!("    - {}", suite.name);
    }

    // Strategy 4: Changed files
    println!("\nStrategy 4: Changed files");
    let changed_files = vec![
        PathBuf::from("src/lib.rs"),
        PathBuf::from("src/automated_test_runner.rs"),
        PathBuf::from("tests/unit_tests.rs"),
    ];
    let config_changed = TestRunnerConfig {
        test_selection: TestSelectionStrategy::ChangedFiles(changed_files.clone()),
        ..base_config
    };
    let runner_changed = AutomatedTestRunner::new(config_changed)?;
    let selected_changed = runner_changed.select_test_suites()?;
    println!("  - Changed files: {:?}", changed_files);
    println!("  - Selected {} affected test suites", selected_changed.len());
    for suite in &selected_changed {
        println!("    - {} (affected by file changes)", suite.name);
    }

    println!("✅ Test selection strategies demo completed\n");
    Ok(())
}

/// Demonstrate performance regression detection
fn demo_performance_regression_detection() -> Result<()> {
    println!("📊 Demo 3: Performance Regression Detection");
    println!("-------------------------------------------");

    let config = TestRunnerConfig {
        output_directory: PathBuf::from("demo_test_reports"),
        enable_regression_detection: true,
        regression_threshold_percent: 15.0,
        verbose: true,
        ..Default::default()
    };

    let runner = AutomatedTestRunner::new(config)?;

    // Simulate regression calculation
    println!("Regression calculation examples:");
    
    let baseline = 1000.0; // 1 second baseline
    let scenarios = vec![
        ("No change", 1000.0),
        ("5% improvement", 950.0),
        ("10% regression", 1100.0),
        ("20% regression", 1200.0),
        ("50% regression", 1500.0),
    ];

    for (description, current) in scenarios {
        let regression = runner.calculate_regression(baseline, current);
        let status = if regression > runner.config.regression_threshold_percent {
            "⚠️  REGRESSION DETECTED"
        } else if regression < 0.0 {
            "✅ IMPROVEMENT"
        } else {
            "✅ ACCEPTABLE"
        };
        
        println!("  - {}: {:.1}% change - {}", description, regression, status);
    }

    // Demonstrate regression severity classification
    println!("\nRegression severity classification:");
    let regression_values = vec![5.0, 15.0, 25.0, 60.0];
    
    for regression in regression_values {
        let severity = if regression > 50.0 {
            "🔴 CRITICAL"
        } else if regression > 20.0 {
            "🟡 MAJOR"
        } else if regression > 10.0 {
            "🟢 MINOR"
        } else {
            "✅ ACCEPTABLE"
        };
        
        println!("  - {:.1}% regression: {}", regression, severity);
    }

    println!("✅ Performance regression detection demo completed\n");
    Ok(())
}

/// Demonstrate CI/CD integration features
fn demo_ci_cd_integration() -> Result<()> {
    println!("🚀 Demo 4: CI/CD Integration");
    println!("----------------------------");

    let config = TestRunnerConfig {
        output_directory: PathBuf::from("demo_test_reports"),
        machine_readable_output: true,
        verbose: false,
        ..Default::default()
    };

    let runner = AutomatedTestRunner::new(config)?;

    // Demonstrate exit code mapping
    println!("Exit code mapping for CI/CD:");
    
    use ldc_engine::automated_test_runner::{TestExecutionReport, TestSummary, TestStatus};
    
    let test_scenarios = vec![
        ("All tests passed", TestStatus::Passed, 0),
        ("Some tests failed", TestStatus::Failed, 1),
        ("Test execution error", TestStatus::Error, 2),
        ("Tests timed out", TestStatus::Timeout, 3),
        ("Tests skipped", TestStatus::Skipped, 0),
    ];

    for (description, status, expected_code) in test_scenarios {
        let mock_report = TestExecutionReport {
            execution_id: "demo".to_string(),
            timestamp: chrono::Utc::now(),
            git_commit: Some("abc123".to_string()),
            config: runner.config.clone(),
            results: vec![],
            summary: TestSummary {
                total_suites: 1,
                passed_suites: if status == TestStatus::Passed { 1 } else { 0 },
                failed_suites: if status == TestStatus::Failed { 1 } else { 0 },
                skipped_suites: if status == TestStatus::Skipped { 1 } else { 0 },
                error_suites: if status == TestStatus::Error { 1 } else { 0 },
                total_duration_ms: 5000,
                success_rate: if status == TestStatus::Passed { 100.0 } else { 0.0 },
                overall_status: status,
            },
            performance_regressions: vec![],
            recommendations: vec![],
        };

        let exit_code = runner.get_exit_code(&mock_report);
        println!("  - {}: Exit code {} ✅", description, exit_code);
        assert_eq!(exit_code, expected_code);
    }

    // Demonstrate report formats
    println!("\nSupported report formats:");
    println!("  - JSON (machine-readable): test_reports/latest.json");
    println!("  - Text (human-readable): test_reports/latest.txt");
    println!("  - JUnit XML: test_reports/junit.xml");
    println!("  - Test history: test_reports/test_history.json");
    println!("  - Performance baseline: test_reports/performance_baseline.json");

    // Demonstrate CI/CD environment detection
    println!("\nCI/CD environment detection:");
    let ci_indicators = vec![
        ("GITHUB_ACTIONS", std::env::var("GITHUB_ACTIONS").is_ok()),
        ("GITLAB_CI", std::env::var("GITLAB_CI").is_ok()),
        ("JENKINS_URL", std::env::var("JENKINS_URL").is_ok()),
        ("TRAVIS", std::env::var("TRAVIS").is_ok()),
        ("CIRCLECI", std::env::var("CIRCLECI").is_ok()),
    ];

    for (env_var, detected) in ci_indicators {
        let status = if detected { "✅ DETECTED" } else { "❌ Not detected" };
        println!("  - {}: {}", env_var, status);
    }

    // Demonstrate timeout and resource management
    println!("\nTimeout and resource management:");
    println!("  - Suite timeout: {}s", runner.config.suite_timeout_seconds);
    println!("  - Test timeout: {}s", runner.config.test_timeout_seconds);
    println!("  - Cleanup timeout: {}s", runner.config.cleanup_timeout_seconds);
    println!("  - Max parallel suites: {}", runner.config.max_parallel_suites);

    println!("✅ CI/CD integration demo completed\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_functions() -> Result<()> {
        // Test that all demo functions can run without errors
        demo_basic_configuration()?;
        demo_test_selection_strategies()?;
        demo_performance_regression_detection()?;
        demo_ci_cd_integration()?;
        Ok(())
    }
}