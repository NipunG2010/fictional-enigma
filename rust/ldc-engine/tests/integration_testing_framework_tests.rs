use ldc_engine::integration_testing::{TestExecutionPipeline, IntegrationTestConfig};
use std::time::Duration;

/// Test the comprehensive integration testing framework
#[test]
fn test_integration_testing_framework() {
    println!("Testing the comprehensive integration testing framework...");
    
    // Create a test configuration with reduced scope for faster testing
    let config = IntegrationTestConfig {
        run_unit_tests: true,
        run_performance_tests: true,
        run_integration_tests: true,
        run_backtest_tests: true,
        run_statistical_tests: true,
        test_error_handling: true,
        test_configuration_changes: true,
        integration_test_data_size: 100, // Smaller dataset for testing
        test_iterations: 10, // Fewer iterations for testing
        test_timeout_seconds: 60, // 1 minute timeout for testing
    };
    
    // Create and run the test pipeline
    let mut pipeline = TestExecutionPipeline::with_config(config)
        .expect("Failed to create test execution pipeline");
    
    let results = pipeline.run_all_tests()
        .expect("Failed to run comprehensive tests");
    
    // Verify test execution
    println!("Test execution completed:");
    println!("  Total categories: {}", results.summary.total_test_categories);
    println!("  Passed categories: {}", results.summary.passed_categories);
    println!("  Failed categories: {}", results.summary.failed_categories);
    println!("  Skipped categories: {}", results.summary.skipped_categories);
    println!("  Overall success: {}", results.summary.overall_success);
    println!("  Total execution time: {:?}", results.summary.total_execution_time);
    
    // Verify that tests were actually run
    assert!(results.summary.total_test_categories > 0, "No test categories were executed");
    assert!(results.summary.passed_categories > 0, "No test categories passed");
    
    // Verify individual test category results
    if let Some(ref unit_results) = results.unit_test_results {
        println!("  Unit tests: {} passed", unit_results.overall_passed);
        assert!(unit_results.simd_accuracy_results.total_tests > 0, "No SIMD accuracy tests run");
        assert!(unit_results.hnsw_compatibility_results.total_tests > 0, "No HNSW compatibility tests run");
    }
    
    if let Some(ref perf_results) = results.performance_test_results {
        println!("  Performance tests: {}/{} passed", perf_results.passed_count(), perf_results.total_count());
        assert!(perf_results.total_count() > 0, "No performance tests run");
    }
    
    if let Some(ref integration_results) = results.integration_test_results {
        println!("  Integration tests: {} passed", integration_results.overall_passed);
        assert!(integration_results.ohlcv_to_signals_workflow.total_operations > 0, "No workflow tests run");
    }
    
    if let Some(ref backtest_results) = results.backtest_results {
        println!("  Backtest: {} trades executed", backtest_results.total_trades);
        // Backtest should execute some trades or handle gracefully
    }
    
    if let Some(ref statistical_results) = results.statistical_results {
        println!("  Statistical analysis: hit rate {:.2}%", 
                statistical_results.prediction_accuracy.hit_rate * 100.0);
        assert!(statistical_results.prediction_accuracy.hit_rate >= 0.0, "Invalid hit rate");
        assert!(statistical_results.prediction_accuracy.hit_rate <= 1.0, "Invalid hit rate");
    }
    
    if let Some(ref error_results) = results.error_handling_results {
        println!("  Error handling tests: {} passed", error_results.overall_passed);
        assert!(!error_results.graceful_degradation_tests.is_empty(), "No graceful degradation tests run");
    }
    
    if let Some(ref config_results) = results.configuration_test_results {
        println!("  Configuration tests: {} passed", config_results.overall_passed);
        assert!(!config_results.dynamic_reconfiguration_tests.is_empty(), "No reconfiguration tests run");
    }
    
    // Verify execution metadata
    assert!(results.execution_metadata.test_environment.cpu_count > 0, "Invalid CPU count");
    assert!(results.execution_metadata.test_environment.test_data_size > 0, "Invalid test data size");
    
    // The framework should complete without panicking, even if some tests fail
    println!("Integration testing framework test completed successfully!");
}

/// Test individual components of the integration testing framework
#[test]
fn test_integration_framework_components() {
    println!("Testing individual integration framework components...");
    
    // Test with minimal configuration
    let config = IntegrationTestConfig {
        run_unit_tests: true,
        run_performance_tests: false,
        run_integration_tests: false,
        run_backtest_tests: false,
        run_statistical_tests: false,
        test_error_handling: false,
        test_configuration_changes: false,
        integration_test_data_size: 50,
        test_iterations: 5,
        test_timeout_seconds: 30,
    };
    
    let mut pipeline = TestExecutionPipeline::with_config(config)
        .expect("Failed to create minimal test pipeline");
    
    let results = pipeline.run_all_tests()
        .expect("Failed to run minimal tests");
    
    // Should have run only unit tests (but total includes skipped categories)
    assert_eq!(results.summary.passed_categories, 1, "Should have passed only 1 test category");
    assert_eq!(results.summary.skipped_categories, 6, "Should have skipped 6 test categories");
    assert!(results.unit_test_results.is_some(), "Unit test results should be present");
    assert!(results.performance_test_results.is_none(), "Performance test results should be absent");
    assert!(results.integration_test_results.is_none(), "Integration test results should be absent");
    
    println!("Individual component test completed successfully!");
}

/// Test error handling in the integration testing framework
#[test]
fn test_integration_framework_error_handling() {
    println!("Testing integration framework error handling...");
    
    // Test with configuration that might cause issues
    let config = IntegrationTestConfig {
        run_unit_tests: true,
        run_performance_tests: true,
        run_integration_tests: true,
        run_backtest_tests: false, // Skip potentially slow backtest
        run_statistical_tests: true,
        test_error_handling: true,
        test_configuration_changes: true,
        integration_test_data_size: 10, // Very small dataset
        test_iterations: 3, // Minimal iterations
        test_timeout_seconds: 15, // Short timeout
    };
    
    let mut pipeline = TestExecutionPipeline::with_config(config)
        .expect("Failed to create error-prone test pipeline");
    
    // This should not panic even if individual tests fail
    let results = pipeline.run_all_tests()
        .expect("Test pipeline should handle errors gracefully");
    
    // Verify that the framework handled errors gracefully
    assert!(results.summary.total_test_categories > 0, "Some test categories should have been attempted");
    
    // Check that error information is captured
    for (category_name, category_result) in &results.summary.category_results {
        println!("  Category {}: passed={}, error={:?}", 
                category_name, category_result.passed, category_result.error_message);
        
        // Each category should have execution time recorded
        assert!(category_result.execution_time >= Duration::from_nanos(0), 
               "Execution time should be non-negative for {}", category_name);
    }
    
    println!("Error handling test completed successfully!");
}

/// Test the mathematical test suite independently
#[test]
fn test_mathematical_test_suite() {
    use ldc_engine::integration_testing::MathematicalTestSuite;
    
    println!("Testing mathematical test suite...");
    
    let test_suite = MathematicalTestSuite::new();
    
    // Test SIMD accuracy
    let simd_results = test_suite.test_simd_accuracy()
        .expect("SIMD accuracy test should not fail");
    
    println!("  SIMD accuracy: {}/{} tests passed ({:.1}%)", 
            simd_results.passed_tests, simd_results.total_tests, 
            simd_results.success_rate * 100.0);
    
    assert!(simd_results.total_tests > 0, "Should have SIMD tests");
    assert!(simd_results.success_rate >= 0.8, "SIMD accuracy should be at least 80%");
    
    // Test HNSW compatibility
    let hnsw_results = test_suite.test_hnsw_compatibility()
        .expect("HNSW compatibility test should not fail");
    
    println!("  HNSW compatibility: {}/{} tests passed ({:.1}%)", 
            hnsw_results.passed_tests, hnsw_results.total_tests, 
            hnsw_results.success_rate * 100.0);
    
    assert!(hnsw_results.total_tests > 0, "Should have HNSW tests");
    assert!(hnsw_results.success_rate >= 0.8, "HNSW compatibility should be at least 80%");
    
    // Test distance calculations
    let distance_results = test_suite.test_distance_calculations()
        .expect("Distance calculation test should not fail");
    
    println!("  Distance calculations: {}/{} tests passed ({:.1}%)", 
            distance_results.passed_tests, distance_results.total_tests, 
            distance_results.success_rate * 100.0);
    
    assert!(distance_results.total_tests > 0, "Should have distance calculation tests");
    assert!(distance_results.success_rate >= 0.9, "Distance calculations should be at least 90% accurate");
    
    // Test edge cases
    let edge_case_results = test_suite.test_edge_cases()
        .expect("Edge case test should not fail");
    
    println!("  Edge cases: {}/{} tests passed ({:.1}%)", 
            edge_case_results.passed_tests, edge_case_results.total_tests, 
            edge_case_results.success_rate * 100.0);
    
    assert!(edge_case_results.total_tests > 0, "Should have edge case tests");
    assert!(edge_case_results.success_rate >= 0.7, "Edge cases should be at least 70% handled");
    
    println!("Mathematical test suite completed successfully!");
}