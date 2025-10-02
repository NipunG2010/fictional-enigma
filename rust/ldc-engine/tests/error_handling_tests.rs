use ldc_engine::testing_error::*;
use ldc_engine::test_diagnostics::*;
use ldc_engine::graceful_recovery::*;
use anyhow::Result;
use std::collections::HashMap;
use tokio;

/// Test comprehensive error handling and diagnostics functionality
#[tokio::test]
async fn test_mathematical_accuracy_error_handling() -> Result<()> {
    let mut diagnostics_engine = TestDiagnosticsEngine::new(DiagnosticsConfig::default());
    
    // Create a mathematical accuracy error
    let error = TestingError::mathematical_accuracy_error(
        "simd_vs_standard_distance".to_string(),
        1.23456789,
        1.23457000,
        0.000001,
    );
    
    let test_context = TestContext {
        test_suite: "mathematical_accuracy".to_string(),
        test_category: "unit_tests".to_string(),
        test_phase: "distance_calculation".to_string(),
        configuration: {
            let mut config = HashMap::new();
            config.insert("tolerance".to_string(), "1e-6".to_string());
            config.insert("algorithm".to_string(), "simd".to_string());
            config
        },
        environment_variables: HashMap::new(),
        input_parameters: {
            let mut params = HashMap::new();
            params.insert("feature_count".to_string(), "5".to_string());
            params.insert("sample_size".to_string(), "1000".to_string());
            params
        },
    };
    
    // Analyze the error
    let error_report = diagnostics_engine.analyze_error(error, test_context)?;
    
    // Verify error report structure
    assert!(matches!(error_report.error, TestingError::MathematicalAccuracyError { .. }));
    assert_eq!(error_report.test_context.test_suite, "mathematical_accuracy");
    assert!(!error_report.recovery_actions.is_empty());
    assert!(!error_report.debugging_info.stack_trace.is_empty());
    
    println!("Mathematical accuracy error handling test passed");
    Ok(())
}

#[tokio::test]
async fn test_performance_failure_analysis() -> Result<()> {
    let mut diagnostics_engine = TestDiagnosticsEngine::new(DiagnosticsConfig::default());
    
    // Create detailed performance metrics
    let mut detailed_metrics = HashMap::new();
    detailed_metrics.insert("execution_time_ms".to_string(), 2500.0); // Slow
    detailed_metrics.insert("memory_usage_mb".to_string(), 1200.0); // High memory
    detailed_metrics.insert("cpu_usage_percent".to_string(), 95.0); // High CPU
    detailed_metrics.insert("cache_miss_rate".to_string(), 25.0);
    detailed_metrics.insert("thread_contention".to_string(), 15.0);
    
    // Analyze performance failure
    let analysis = diagnostics_engine.analyze_performance_failure(
        "k_nearest_neighbors_10k",
        1000.0, // Target: 1 second
        2500.0, // Actual: 2.5 seconds
        detailed_metrics,
    )?;
    
    // Verify analysis results
    assert_eq!(analysis.test_name, "k_nearest_neighbors_10k");
    assert!(!analysis.bottlenecks.is_empty());
    assert!(!analysis.optimization_recommendations.is_empty());
    
    // Check for CPU bottleneck detection
    let cpu_bottleneck = analysis.bottlenecks.iter()
        .find(|b| matches!(b.bottleneck_type, BottleneckType::CPU));
    assert!(cpu_bottleneck.is_some());
    
    // Check for memory bottleneck detection
    let memory_bottleneck = analysis.bottlenecks.iter()
        .find(|b| matches!(b.bottleneck_type, BottleneckType::Memory));
    assert!(memory_bottleneck.is_some());
    
    // Verify optimization recommendations
    let parallelization_rec = analysis.optimization_recommendations.iter()
        .find(|r| matches!(r.recommendation_type, OptimizationType::ParallelizationImprovement));
    assert!(parallelization_rec.is_some());
    
    println!("Performance failure analysis test passed");
    Ok(())
}

#[tokio::test]
async fn test_statistical_failure_diagnostics() -> Result<()> {
    let diagnostics_engine = TestDiagnosticsEngine::new(DiagnosticsConfig::default());
    
    // Analyze statistical test failure
    let diagnostics = diagnostics_engine.analyze_statistical_failure(
        "prediction_accuracy_test",
        50,    // Small sample size
        0.15,  // High p-value
        0.05,  // Standard significance threshold
        0.3,   // Medium effect size
    )?;
    
    // Verify diagnostics
    assert_eq!(diagnostics.test_name, "prediction_accuracy_test");
    assert!(matches!(diagnostics.sample_size_analysis.adequacy_assessment, SampleAdequacy::Inadequate));
    assert!(diagnostics.sample_size_analysis.current_sample_size < diagnostics.sample_size_analysis.minimum_required_size);
    assert!(diagnostics.power_analysis.observed_power < diagnostics.power_analysis.target_power);
    assert!(!diagnostics.recommendations.is_empty());
    
    // Check for sample size increase recommendation
    let sample_size_rec = diagnostics.recommendations.iter()
        .find(|r| matches!(r.recommendation_type, StatisticalRecommendationType::IncreaseSampleSize));
    assert!(sample_size_rec.is_some());
    
    println!("Statistical failure diagnostics test passed");
    Ok(())
}

#[tokio::test]
async fn test_data_validation_error_handling() -> Result<()> {
    let diagnostics_engine = TestDiagnosticsEngine::new(DiagnosticsConfig::default());
    
    // Create test data with various quality issues
    let test_data = vec![1, 2, 3]; // Too small dataset
    let validation_rules = DataValidationRules {
        minimum_sample_size: 100,
        check_duplicates: true,
        check_ranges: true,
        check_consistency: true,
        outlier_detection: true,
    };
    
    // Validate test data
    let validation_result = diagnostics_engine.validate_test_data(&test_data, &validation_rules)?;
    
    // Verify validation results
    assert!(validation_result.quality_score < 100);
    assert!(!validation_result.issues.is_empty());
    assert!(!validation_result.recommendations.is_empty());
    
    // Check for missing values issue
    let missing_values_issue = validation_result.issues.iter()
        .find(|i| matches!(i.issue_type, DataQualityIssueType::MissingValues));
    assert!(missing_values_issue.is_some());
    
    println!("Data validation error handling test passed");
    Ok(())
}

#[tokio::test]
async fn test_graceful_recovery_system() -> Result<()> {
    let diagnostics_engine = TestDiagnosticsEngine::new(DiagnosticsConfig::default());
    let mut recovery_system = GracefulRecoverySystem::new(
        RecoveryConfig::default(),
        diagnostics_engine,
    );
    
    let test_context = TestContext {
        test_suite: "integration_tests".to_string(),
        test_category: "integration".to_string(),
        test_phase: "component_interaction".to_string(),
        configuration: HashMap::new(),
        environment_variables: HashMap::new(),
        input_parameters: HashMap::new(),
    };
    
    // Create a test function that fails initially but can be recovered
    let attempt_count = std::sync::Arc::new(std::sync::Mutex::new(0));
    let attempt_count_clone = attempt_count.clone();
    let test_function = move || {
        let attempt_count = attempt_count_clone.clone();
        async move {
            let mut count = attempt_count.lock().unwrap();
            *count += 1;
            let current_count = *count;
            drop(count); // Release the lock
            
            if current_count < 3 {
                // Fail first two attempts
                Err(anyhow::anyhow!("Simulated test failure"))
            } else {
                // Succeed on third attempt
                Ok(TestResult {
                    passed: true,
                    message: "Test passed after recovery".to_string(),
                    execution_time_ms: 100,
                    details: HashMap::new(),
                })
            }
        }
    };
    
    // Execute test with recovery
    let result = recovery_system.execute_test_with_recovery(
        "recoverable_integration_test".to_string(),
        test_function,
        test_context,
    ).await?;
    
    // Verify recovery worked
    assert!(matches!(result.final_result, TestExecutionStatus::Recovered));
    assert!(result.attempts_made >= 2);
    assert!(!result.recovery_actions_taken.is_empty());
    assert!(result.recovery_successful);
    
    println!("Graceful recovery system test passed");
    Ok(())
}

#[tokio::test]
async fn test_circuit_breaker_functionality() -> Result<()> {
    let diagnostics_engine = TestDiagnosticsEngine::new(DiagnosticsConfig::default());
    let mut recovery_system = GracefulRecoverySystem::new(
        RecoveryConfig {
            circuit_breaker_failure_threshold: 2, // Low threshold for testing
            ..RecoveryConfig::default()
        },
        diagnostics_engine,
    );
    
    let test_context = TestContext {
        test_suite: "circuit_breaker_test".to_string(),
        test_category: "reliability".to_string(),
        test_phase: "failure_handling".to_string(),
        configuration: HashMap::new(),
        environment_variables: HashMap::new(),
        input_parameters: HashMap::new(),
    };
    
    // Execute the failing test multiple times to trigger circuit breaker
    for i in 0..5 {
        let failing_test = || async {
            Err(anyhow::anyhow!("Persistent test failure"))
        };
        
        let result = recovery_system.execute_test_with_recovery(
            "persistent_failing_test".to_string(),
            failing_test,
            test_context.clone(),
        ).await?;
        
        if i >= 2 {
            // After threshold failures, circuit breaker should be open
            assert!(matches!(result.final_result, TestExecutionStatus::CircuitBreakerOpen | TestExecutionStatus::Failed));
        }
    }
    
    println!("Circuit breaker functionality test passed");
    Ok(())
}

#[tokio::test]
async fn test_error_recovery_strategies() -> Result<()> {
    let diagnostics_engine = TestDiagnosticsEngine::new(DiagnosticsConfig::default());
    let mut recovery_system = GracefulRecoverySystem::new(
        RecoveryConfig::default(),
        diagnostics_engine,
    );
    
    // Test performance error recovery
    let performance_error = TestingError::PerformanceTestError {
        test_name: "performance_test".to_string(),
        target_ms: 1000.0,
        actual_ms: 2500.0,
        regression_percent: 150.0,
        bottleneck: "CPU".to_string(),
        recommendations: vec!["Optimize algorithm".to_string()],
    };
    
    let test_context = TestContext {
        test_suite: "performance_tests".to_string(),
        test_category: "performance".to_string(),
        test_phase: "execution".to_string(),
        configuration: HashMap::new(),
        environment_variables: HashMap::new(),
        input_parameters: HashMap::new(),
    };
    
    let recovery_result = recovery_system.attempt_recovery(&performance_error, &test_context).await?;
    
    // Verify recovery was attempted
    assert!(!recovery_result.attempted_actions.is_empty());
    
    // Test statistical error recovery
    let statistical_error = TestingError::StatisticalTestError {
        test_name: "statistical_test".to_string(),
        sample_size: 30,
        required_sample_size: 100,
        p_value: 0.15,
        significance_threshold: 0.05,
        diagnosis: "Insufficient sample size".to_string(),
    };
    
    let recovery_result = recovery_system.attempt_recovery(&statistical_error, &test_context).await?;
    
    // Verify recovery was attempted
    assert!(!recovery_result.attempted_actions.is_empty());
    
    println!("Error recovery strategies test passed");
    Ok(())
}

#[tokio::test]
async fn test_adaptive_recovery_behavior() -> Result<()> {
    let diagnostics_engine = TestDiagnosticsEngine::new(DiagnosticsConfig::default());
    let mut recovery_system = GracefulRecoverySystem::new(
        RecoveryConfig {
            enable_adaptive_strategies: true,
            adaptive_failure_threshold: 0.2, // 20% failure rate threshold
            ..RecoveryConfig::default()
        },
        diagnostics_engine,
    );
    
    // Get initial failure statistics
    let initial_stats = recovery_system.get_failure_statistics()?;
    assert_eq!(initial_stats.total_failures, 0);
    assert_eq!(initial_stats.failure_rate, 0.0);
    
    // Update adaptive strategies (should not change much with no failures)
    recovery_system.update_adaptive_strategies()?;
    
    println!("Adaptive recovery behavior test passed");
    Ok(())
}

#[tokio::test]
async fn test_comprehensive_error_reporting() -> Result<()> {
    let mut diagnostics_engine = TestDiagnosticsEngine::new(DiagnosticsConfig {
        enable_detailed_logging: true,
        max_error_history: 100,
        ..DiagnosticsConfig::default()
    });
    
    // Create multiple types of errors and analyze them
    let errors = vec![
        TestingError::MathematicalAccuracyError {
            test_name: "distance_calculation".to_string(),
            expected: 1.0,
            actual: 1.1,
            difference: 0.1,
            tolerance: 0.01,
            recommendation: "Check algorithm implementation".to_string(),
        },
        TestingError::ResourceExhaustionError {
            resource: "Memory".to_string(),
            usage_percent: 95.0,
            threshold_percent: 90.0,
            adaptive_actions: vec!["Cleanup resources".to_string()],
        },
        TestingError::TestTimeoutError {
            test_name: "long_running_test".to_string(),
            timeout_seconds: 300,
            execution_phase: "data_processing".to_string(),
            progress_percent: 75.0,
            suggestions: vec!["Increase timeout".to_string()],
        },
    ];
    
    let test_context = TestContext {
        test_suite: "comprehensive_tests".to_string(),
        test_category: "mixed".to_string(),
        test_phase: "execution".to_string(),
        configuration: HashMap::new(),
        environment_variables: HashMap::new(),
        input_parameters: HashMap::new(),
    };
    
    // Analyze each error
    for error in errors {
        let error_report = diagnostics_engine.analyze_error(error, test_context.clone())?;
        
        // Verify comprehensive reporting
        assert!(!error_report.recovery_actions.is_empty());
        assert!(!error_report.debugging_info.stack_trace.is_empty());
        assert!(!error_report.debugging_info.performance_metrics.is_empty());
        assert!(error_report.system_state.memory_usage_mb > 0.0);
    }
    
    println!("Comprehensive error reporting test passed");
    Ok(())
}

/// Helper function to create test context
fn create_test_context(suite: &str, category: &str, phase: &str) -> TestContext {
    TestContext {
        test_suite: suite.to_string(),
        test_category: category.to_string(),
        test_phase: phase.to_string(),
        configuration: HashMap::new(),
        environment_variables: HashMap::new(),
        input_parameters: HashMap::new(),
    }
}

/// Integration test for the complete error handling pipeline
#[tokio::test]
async fn test_complete_error_handling_pipeline() -> Result<()> {
    // Initialize the complete error handling system
    let diagnostics_config = DiagnosticsConfig {
        enable_detailed_logging: true,
        enable_automatic_recovery: true,
        ..DiagnosticsConfig::default()
    };
    
    let diagnostics_engine = TestDiagnosticsEngine::new(diagnostics_config);
    let mut recovery_system = GracefulRecoverySystem::new(
        RecoveryConfig::default(),
        diagnostics_engine,
    );
    
    // Simulate a complex test scenario with multiple failure modes
    let test_context = create_test_context("integration_pipeline", "end_to_end", "full_workflow");
    
    let execution_count = std::sync::Arc::new(std::sync::Mutex::new(0));
    let execution_count_clone = execution_count.clone();
    let complex_test = move || {
        let execution_count = execution_count_clone.clone();
        async move {
            let mut count = execution_count.lock().unwrap();
            *count += 1;
            let current_count = *count;
            drop(count); // Release the lock
            
            match current_count {
                1 => Err(anyhow::anyhow!("Memory allocation failed")), // Resource error
                2 => Err(anyhow::anyhow!("Operation timed out")),       // Timeout error
                3 => Ok(TestResult {                                    // Success after recovery
                    passed: true,
                    message: "Test completed successfully after recovery".to_string(),
                    execution_time_ms: 1500,
                    details: {
                        let mut details = HashMap::new();
                        details.insert("recovery_attempts".to_string(), "2".to_string());
                        details.insert("final_status".to_string(), "recovered".to_string());
                        details
                    },
                }),
                _ => Err(anyhow::anyhow!("Unexpected execution")),
            }
        }
    };
    
    // Execute the test with full error handling pipeline
    let result = recovery_system.execute_test_with_recovery(
        "complex_integration_test".to_string(),
        complex_test,
        test_context,
    ).await?;
    
    // Verify the complete pipeline worked
    assert!(matches!(result.final_result, TestExecutionStatus::Recovered));
    assert_eq!(result.attempts_made, 3);
    assert!(result.recovery_successful);
    assert!(!result.recovery_actions_taken.is_empty());
    assert_eq!(result.errors_encountered.len(), 2); // Two failures before success
    
    // Verify error types were properly classified
    let resource_error = result.errors_encountered.iter()
        .any(|e| matches!(e, TestingError::ResourceExhaustionError { .. }));
    let timeout_error = result.errors_encountered.iter()
        .any(|e| matches!(e, TestingError::TestTimeoutError { .. }));
    
    assert!(resource_error || timeout_error); // At least one should be properly classified
    
    println!("Complete error handling pipeline test passed");
    println!("Final result: {:?}", result.final_result);
    println!("Attempts made: {}", result.attempts_made);
    println!("Recovery actions: {}", result.recovery_actions_taken.len());
    println!("Errors encountered: {}", result.errors_encountered.len());
    
    Ok(())
}