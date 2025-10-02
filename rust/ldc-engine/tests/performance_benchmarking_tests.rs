use ldc_engine::{LDCConfig, LDCEngine};
use ldc_engine::performance_benchmarking::{
    BenchmarkingFramework, BenchmarkConfiguration, BenchmarkTestParameters,
    ParameterSweepUtility, ABTestingFramework, ParameterValue
};
use ldc_engine::performance_reporting::{PerformanceReporter, ReportConfig, ReportFormat};
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn test_benchmarking_framework_basic_functionality() {
    let baseline_config = LDCConfig {
        neighbors_count: 5,
        max_bars_back: 1000,
        use_hnsw_index: false,
        ..Default::default()
    };

    let mut framework = BenchmarkingFramework::new(baseline_config.clone());
    
    // Add a test configuration
    let test_config = BenchmarkConfiguration {
        name: "test_config".to_string(),
        description: "Test configuration".to_string(),
        config: LDCConfig {
            neighbors_count: 10,
            ..baseline_config
        },
        test_parameters: BenchmarkTestParameters {
            iterations: 5,
            warmup_iterations: 1,
            dataset_sizes: vec![100],
            k_values: vec![5],
            enable_memory_profiling: false,
            enable_cpu_profiling: false,
        },
    };

    framework.add_configuration(test_config);
    assert_eq!(framework.test_configurations.len(), 1);

    // Establish baseline
    let baseline_result = framework.establish_baseline();
    assert!(baseline_result.is_ok());
    assert!(framework.baseline_results.is_some());
}

#[test]
fn test_parameter_sweep_utility() {
    let base_config = LDCConfig::default();
    let mut sweep = ParameterSweepUtility::new(base_config);

    // Add parameters to sweep
    sweep.add_parameter("k_values".to_string(), vec![
        ParameterValue::Integer(5),
        ParameterValue::Integer(10),
        ParameterValue::Integer(15),
    ]);

    sweep.add_parameter("hnsw_enabled".to_string(), vec![
        ParameterValue::Boolean(true),
        ParameterValue::Boolean(false),
    ]);

    let configurations = sweep.generate_configurations();
    assert!(!configurations.is_empty());
    
    // Should generate configurations for different parameter combinations
    assert!(configurations.len() >= 6); // At least k variations + hnsw variations
}

#[test]
fn test_ab_testing_framework() {
    let control_config = LDCConfig {
        neighbors_count: 5,
        max_bars_back: 500,
        use_hnsw_index: false,
        ..Default::default()
    };

    let mut ab_framework = ABTestingFramework::new(
        control_config.clone(),
        Duration::from_secs(10),
        50, // Small sample size for testing
    );

    // Add treatment configurations
    ab_framework.add_treatment("treatment_hnsw".to_string(), LDCConfig {
        use_hnsw_index: true,
        ..control_config.clone()
    });

    ab_framework.add_treatment("treatment_high_k".to_string(), LDCConfig {
        neighbors_count: 10,
        ..control_config
    });

    // Run A/B test
    let ab_results = ab_framework.run_ab_test();
    assert!(ab_results.is_ok());

    let results = ab_results.unwrap();
    assert_eq!(results.treatment_results.len(), 2);
    assert_eq!(results.statistical_analysis.len(), 2);
    
    // Verify control results
    assert_eq!(results.control_results.configuration_name, "control");
    assert!(results.control_results.sample_size > 0);
}

#[test]
fn test_benchmark_comparison() {
    let config1 = LDCConfig {
        neighbors_count: 5,
        max_bars_back: 500,
        use_hnsw_index: false,
        ..Default::default()
    };

    let config2 = LDCConfig {
        neighbors_count: 10,
        max_bars_back: 500,
        use_hnsw_index: true,
        ..Default::default()
    };

    let mut framework = BenchmarkingFramework::new(config1);
    
    let test_config = BenchmarkConfiguration {
        name: "optimized".to_string(),
        description: "Optimized configuration".to_string(),
        config: config2,
        test_parameters: BenchmarkTestParameters {
            iterations: 3,
            warmup_iterations: 1,
            dataset_sizes: vec![100],
            k_values: vec![5],
            enable_memory_profiling: false,
            enable_cpu_profiling: false,
        },
    };

    framework.add_configuration(test_config);

    // Run benchmarks
    let baseline_result = framework.establish_baseline();
    assert!(baseline_result.is_ok());

    let all_results = framework.run_all_benchmarks();
    assert!(all_results.is_ok());

    let results = all_results.unwrap();
    assert_eq!(results.len(), 2); // Baseline + 1 test configuration

    // Compare results
    let comparison = framework.compare_results(&results[0], &results[1]);
    assert_eq!(comparison.baseline_name, "baseline");
    assert_eq!(comparison.comparison_name, "optimized");
    assert!(comparison.statistical_significance.p_value >= 0.0);
    assert!(comparison.statistical_significance.p_value <= 1.0);
}

#[test]
fn test_performance_reporter() {
    let mut reporter = PerformanceReporter::new();
    
    // Create sample benchmark results
    let result1 = create_sample_benchmark_result("config1", 2.5, 85.0);
    let result2 = create_sample_benchmark_result("config2", 1.8, 90.0);
    let result3 = create_sample_benchmark_result("config3", 3.2, 80.0);

    reporter.add_results(result1);
    reporter.add_results(result2);
    reporter.add_results(result3);

    assert_eq!(reporter.results_history().len(), 3);

    // Test dashboard creation
    let dashboard = reporter.create_dashboard();
    assert_eq!(dashboard.summary_metrics.total_configurations_tested, 3);
    assert_eq!(dashboard.summary_metrics.best_latency_ms, 1.8);
    assert_eq!(dashboard.summary_metrics.best_accuracy_percent, 90.0);
    assert_eq!(dashboard.configuration_comparison.len(), 3);

    // Verify ranking
    assert_eq!(dashboard.configuration_comparison[0].rank, 1);
    assert_eq!(dashboard.configuration_comparison[1].rank, 2);
    assert_eq!(dashboard.configuration_comparison[2].rank, 3);
}

#[test]
fn test_report_generation() {
    let mut reporter = PerformanceReporter::with_config(ReportConfig {
        include_charts: false,
        include_detailed_metrics: true,
        include_recommendations: true,
        output_format: ReportFormat::Json,
        chart_width: 600,
        chart_height: 300,
    });

    // Add sample results
    reporter.add_results(create_sample_benchmark_result("baseline", 3.0, 85.0));
    reporter.add_results(create_sample_benchmark_result("optimized", 2.0, 88.0));

    // Test JSON report generation
    let temp_dir = tempdir().unwrap();
    let json_path = temp_dir.path().join("report.json");
    let result = reporter.generate_report(&json_path);
    assert!(result.is_ok());
    assert!(json_path.exists());

    // Test Markdown report generation
    let mut md_reporter = PerformanceReporter::with_config(ReportConfig {
        output_format: ReportFormat::Markdown,
        ..Default::default()
    });
    md_reporter.add_results(create_sample_benchmark_result("test", 2.5, 87.0));
    
    let md_path = temp_dir.path().join("report.md");
    let md_result = md_reporter.generate_report(&md_path);
    assert!(md_result.is_ok());
    assert!(md_path.exists());

    // Test CSV report generation
    let mut csv_reporter = PerformanceReporter::with_config(ReportConfig {
        output_format: ReportFormat::Csv,
        ..Default::default()
    });
    csv_reporter.add_results(create_sample_benchmark_result("test", 2.5, 87.0));
    
    let csv_path = temp_dir.path().join("report.csv");
    let csv_result = csv_reporter.generate_report(&csv_path);
    assert!(csv_result.is_ok());
    assert!(csv_path.exists());
}

#[test]
fn test_performance_trend_analysis() {
    let mut reporter = PerformanceReporter::new();
    
    // Add results with improving trend
    let mut timestamp = chrono::Utc::now() - chrono::Duration::hours(3);
    
    for i in 0..5 {
        let latency = 5.0 - (i as f64 * 0.5); // Improving latency
        let accuracy = 80.0 + (i as f64 * 2.0); // Improving accuracy
        
        let mut result = create_sample_benchmark_result(&format!("v{}", i), latency, accuracy);
        result.test_timestamp = timestamp;
        
        reporter.add_results(result);
        timestamp = timestamp + chrono::Duration::hours(1);
    }

    let trends = reporter.analyze_performance_trends();
    assert!(!trends.is_empty());
    
    // Should detect trends for both latency and accuracy
    let latency_trend = trends.iter().find(|t| t.metric_name == "Average Latency");
    let accuracy_trend = trends.iter().find(|t| t.metric_name == "Prediction Accuracy");
    
    assert!(latency_trend.is_some());
    assert!(accuracy_trend.is_some());
    
    // Verify trend data points
    if let Some(trend) = latency_trend {
        assert_eq!(trend.data_points.len(), 5);
        assert!(trend.change_percent != 0.0); // Should detect change
    }
}

#[test]
fn test_alert_generation() {
    let mut reporter = PerformanceReporter::new();
    
    // Add result with high latency (should trigger alert)
    reporter.add_results(create_sample_benchmark_result("high_latency", 12.0, 85.0));
    
    // Add result with low accuracy (should trigger alert)
    reporter.add_results(create_sample_benchmark_result("low_accuracy", 2.0, 75.0));
    
    let dashboard = reporter.create_dashboard();
    assert!(!dashboard.alerts.is_empty());
    
    // Should have alerts for high latency and low accuracy
    let latency_alert = dashboard.alerts.iter()
        .any(|a| matches!(a.alert_type, ldc_engine::performance_reporting::AlertType::LatencyRegression));
    let accuracy_alert = dashboard.alerts.iter()
        .any(|a| matches!(a.alert_type, ldc_engine::performance_reporting::AlertType::AccuracyDrop));
    
    assert!(latency_alert);
    assert!(accuracy_alert);
}

#[test]
fn test_recommendation_generation() {
    let mut reporter = PerformanceReporter::new();
    
    // Add results with high latency to trigger HNSW recommendation
    reporter.add_results(create_sample_benchmark_result("slow_config", 8.0, 85.0));
    
    let dashboard = reporter.create_dashboard();
    assert!(!dashboard.recommendations.is_empty());
    
    // Should recommend HNSW indexing for high latency
    let hnsw_recommendation = dashboard.recommendations.iter()
        .any(|r| r.title.contains("HNSW"));
    
    assert!(hnsw_recommendation);
}

// Helper function to create sample benchmark results
fn create_sample_benchmark_result(
    name: &str, 
    latency: f64, 
    accuracy: f64
) -> ldc_engine::performance_benchmarking::BenchmarkResults {
    use ldc_engine::performance_benchmarking::{BenchmarkResults, PerformanceMetrics, MemoryMetrics, AccuracyMetrics};
    
    BenchmarkResults {
        configuration_name: name.to_string(),
        test_timestamp: chrono::Utc::now(),
        performance_metrics: PerformanceMetrics {
            avg_query_latency_ms: latency,
            p50_latency_ms: latency * 0.9,
            p95_latency_ms: latency * 1.5,
            p99_latency_ms: latency * 2.0,
            throughput_queries_per_second: 1000.0 / latency,
            cpu_utilization_percent: 75.0,
            parallel_efficiency: 0.85,
        },
        memory_metrics: MemoryMetrics {
            peak_memory_usage_mb: 200.0,
            avg_memory_usage_mb: 150.0,
            memory_efficiency_percent: 85.0,
            allocation_count: 1000,
            deallocation_count: 950,
        },
        accuracy_metrics: AccuracyMetrics {
            prediction_accuracy_percent: accuracy,
            hnsw_accuracy_percent: accuracy * 0.95,
            signal_quality_score: accuracy * 0.8,
            consistency_score: 90.0,
        },
        detailed_results: Vec::new(),
    }
}

#[test]
fn test_statistical_significance_calculation() {
    let baseline_config = LDCConfig::default();
    let framework = BenchmarkingFramework::new(baseline_config);
    
    let baseline_result = create_sample_benchmark_result("baseline", 5.0, 80.0);
    let comparison_result = create_sample_benchmark_result("comparison", 2.5, 85.0);
    
    let comparison = framework.compare_results(&baseline_result, &comparison_result);
    
    // Should detect significant improvement in latency
    assert!(comparison.performance_improvement.latency_improvement_percent > 0.0);
    assert!(comparison.performance_improvement.accuracy_change_percent > 0.0);
    
    // Statistical significance should be calculated
    assert!(comparison.statistical_significance.p_value >= 0.0);
    assert!(comparison.statistical_significance.p_value <= 1.0);
    assert!(comparison.statistical_significance.effect_size >= 0.0);
}

#[test]
fn test_benchmark_configuration_validation() {
    let base_config = LDCConfig::default();
    let mut framework = BenchmarkingFramework::new(base_config.clone());
    
    // Test valid configuration
    let valid_config = BenchmarkConfiguration {
        name: "valid_test".to_string(),
        description: "Valid test configuration".to_string(),
        config: base_config,
        test_parameters: BenchmarkTestParameters {
            iterations: 10,
            warmup_iterations: 2,
            dataset_sizes: vec![100, 500],
            k_values: vec![5, 10],
            enable_memory_profiling: true,
            enable_cpu_profiling: false,
        },
    };
    
    framework.add_configuration(valid_config);
    assert_eq!(framework.test_configurations.len(), 1);
    
    // Verify configuration was added correctly
    let added_config = &framework.test_configurations[0];
    assert_eq!(added_config.name, "valid_test");
    assert_eq!(added_config.test_parameters.iterations, 10);
    assert_eq!(added_config.test_parameters.dataset_sizes.len(), 2);
    assert_eq!(added_config.test_parameters.k_values.len(), 2);
}