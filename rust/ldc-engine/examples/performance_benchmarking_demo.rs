use ldc_engine::{LDCConfig, LDCEngine};
use ldc_engine::performance_benchmarking::{
    BenchmarkingFramework, BenchmarkConfiguration, BenchmarkTestParameters,
    ParameterSweepUtility, ABTestingFramework, ParameterValue
};
use std::time::Duration;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== LDC Engine Performance Benchmarking Demo ===\n");

    // 1. Basic benchmarking framework demonstration
    demo_basic_benchmarking()?;
    
    // 2. Parameter sweep demonstration
    demo_parameter_sweep()?;
    
    // 3. A/B testing demonstration
    demo_ab_testing()?;
    
    // 4. Performance regression detection
    demo_regression_detection()?;

    println!("\n=== Benchmarking Demo Complete ===");
    Ok(())
}

/// Demonstrate basic benchmarking framework
fn demo_basic_benchmarking() -> Result<()> {
    println!("1. Basic Benchmarking Framework");
    println!("================================");

    // Create baseline configuration
    let baseline_config = LDCConfig {
        neighbors_count: 10,
        max_bars_back: 2000,
        use_hnsw_index: false,
        ..Default::default()
    };

    // Initialize benchmarking framework
    let mut framework = BenchmarkingFramework::new(baseline_config.clone());

    // Add test configurations
    let optimized_config = BenchmarkConfiguration {
        name: "optimized_hnsw".to_string(),
        description: "Configuration with HNSW indexing enabled".to_string(),
        config: LDCConfig {
            use_hnsw_index: true,
            ..baseline_config.clone()
        },
        test_parameters: BenchmarkTestParameters {
            iterations: 10,
            warmup_iterations: 2,
            dataset_sizes: vec![50, 100],
            k_values: vec![5],
            enable_memory_profiling: false,
            enable_cpu_profiling: false,
        },
    };

    let high_k_config = BenchmarkConfiguration {
        name: "high_k_value".to_string(),
        description: "Configuration with higher K value".to_string(),
        config: LDCConfig {
            neighbors_count: 20,
            ..baseline_config.clone()
        },
        test_parameters: BenchmarkTestParameters::default(),
    };

    framework.add_configuration(optimized_config);
    framework.add_configuration(high_k_config);

    // Establish baseline
    println!("Establishing baseline performance...");
    let baseline_results = framework.establish_baseline()?;
    println!("Baseline: {:.3}ms avg latency, {:.1}% accuracy\n",
            baseline_results.performance_metrics.avg_query_latency_ms,
            baseline_results.accuracy_metrics.prediction_accuracy_percent);

    // Run all benchmarks
    println!("Running benchmark configurations...");
    let all_results = framework.run_all_benchmarks()?;

    // Display results
    for result in &all_results {
        println!("Configuration: {}", result.configuration_name);
        println!("  Avg Latency: {:.3}ms", result.performance_metrics.avg_query_latency_ms);
        println!("  P95 Latency: {:.3}ms", result.performance_metrics.p95_latency_ms);
        println!("  Throughput: {:.1} queries/sec", result.performance_metrics.throughput_queries_per_second);
        println!("  Accuracy: {:.1}%", result.accuracy_metrics.prediction_accuracy_percent);
        println!("  Memory Usage: {:.1}MB", result.memory_metrics.avg_memory_usage_mb);
        println!();
    }

    // Compare configurations
    if all_results.len() >= 2 {
        let comparison = framework.compare_results(&all_results[0], &all_results[1]);
        println!("Comparison: {} vs {}", comparison.baseline_name, comparison.comparison_name);
        println!("  Latency Improvement: {:.1}%", comparison.performance_improvement.latency_improvement_percent);
        println!("  Accuracy Change: {:.1}%", comparison.performance_improvement.accuracy_change_percent);
        println!("  Recommendation: {}", comparison.recommendation.recommended_configuration);
        println!("  Reasoning: {}", comparison.recommendation.reasoning);
        println!("  Confidence: {:.1}%", comparison.recommendation.confidence_level * 100.0);
        println!();
    }

    Ok(())
}

/// Demonstrate parameter sweep functionality
fn demo_parameter_sweep() -> Result<()> {
    println!("2. Parameter Sweep Optimization");
    println!("===============================");

    let base_config = LDCConfig::default();
    let mut sweep_utility = ParameterSweepUtility::new(base_config.clone());

    // Add parameters to sweep
    sweep_utility.add_parameter("k_values".to_string(), vec![
        ParameterValue::Integer(5),
        ParameterValue::Integer(10),
        ParameterValue::Integer(15),
        ParameterValue::Integer(20),
    ]);

    sweep_utility.add_parameter("hnsw_enabled".to_string(), vec![
        ParameterValue::Boolean(true),
        ParameterValue::Boolean(false),
    ]);

    // Generate parameter combinations
    let configurations = sweep_utility.generate_configurations();
    println!("Generated {} parameter combinations:", configurations.len());

    for (name, config) in &configurations {
        println!("  {}: neighbors_count={}, hnsw={}, max_bars_back={}", 
                name, config.neighbors_count, config.use_hnsw_index, config.max_bars_back);
    }

    // Create benchmarking framework for parameter sweep
    let mut framework = BenchmarkingFramework::new(base_config);
    framework.add_parameter_sweep("param_sweep", configurations);

    println!("\nRunning parameter sweep (simplified)...");
    
    // In a real scenario, you would run the full benchmark
    // For demo purposes, we'll just show the setup
    println!("Parameter sweep configured with {} variations", framework.test_configurations.len());
    println!("Each variation will be tested against baseline performance");
    println!("Results would identify optimal parameter combinations\n");

    Ok(())
}

/// Demonstrate A/B testing framework
fn demo_ab_testing() -> Result<()> {
    println!("3. A/B Testing Framework");
    println!("========================");

    // Control configuration (current production)
    let control_config = LDCConfig {
        neighbors_count: 10,
        max_bars_back: 2000,
        use_hnsw_index: false,
        ..Default::default()
    };

    // Create A/B testing framework
    let mut ab_framework = ABTestingFramework::new(
        control_config.clone(),
        Duration::from_secs(60), // 1 minute test
        100, // 100 sample queries
    );

    // Add treatment configurations
    ab_framework.add_treatment("treatment_hnsw".to_string(), LDCConfig {
        use_hnsw_index: true,
        ..control_config.clone()
    });

    ab_framework.add_treatment("treatment_high_neighbors".to_string(), LDCConfig {
        neighbors_count: 20,
        ..control_config.clone()
    });

    println!("Running A/B test with control and 2 treatments...");
    
    // Run A/B test
    let ab_results = ab_framework.run_ab_test()?;

    // Display results
    println!("\nA/B Test Results:");
    println!("Control ({}): {:.3}ms avg latency, {:.3} accuracy",
            ab_results.control_results.configuration_name,
            ab_results.control_results.avg_latency_ms,
            ab_results.control_results.accuracy_score);

    for (i, treatment) in ab_results.treatment_results.iter().enumerate() {
        let stats = &ab_results.statistical_analysis[i];
        println!("Treatment ({}): {:.3}ms avg latency, {:.3} accuracy",
                treatment.configuration_name,
                treatment.avg_latency_ms,
                treatment.accuracy_score);
        println!("  Latency difference: {:.3}ms", stats.latency_difference_ms);
        println!("  Statistical significance: {} (p={:.3})", 
                if stats.is_significant { "YES" } else { "NO" }, stats.p_value);
        println!("  Effect size: {:.3}", stats.effect_size);
    }

    println!();
    Ok(())
}

/// Demonstrate performance regression detection
fn demo_regression_detection() -> Result<()> {
    println!("4. Performance Regression Detection");
    println!("===================================");

    let base_config = LDCConfig::default();
    let mut framework = BenchmarkingFramework::new(base_config.clone());

    // Simulate baseline performance (previous version)
    println!("Establishing baseline from previous version...");
    let baseline = framework.establish_baseline()?;

    // Simulate new version with potential regression
    let regression_config = BenchmarkConfiguration {
        name: "new_version".to_string(),
        description: "New version with potential performance regression".to_string(),
        config: LDCConfig {
            neighbors_count: 15, // Slightly higher neighbors_count might cause regression
            max_bars_back: 3000, // More bars might use more memory
            ..base_config
        },
        test_parameters: BenchmarkTestParameters {
            iterations: 5,
            warmup_iterations: 1,
            dataset_sizes: vec![50, 100],
            k_values: vec![5],
            enable_memory_profiling: false,
            enable_cpu_profiling: false,
        },
    };

    framework.add_configuration(regression_config);
    let results = framework.run_all_benchmarks()?;

    // Analyze for regressions
    if results.len() >= 2 {
        let comparison = framework.compare_results(&results[0], &results[1]);
        
        println!("Regression Analysis:");
        println!("  Baseline: {:.3}ms", results[0].performance_metrics.avg_query_latency_ms);
        println!("  New Version: {:.3}ms", results[1].performance_metrics.avg_query_latency_ms);
        
        let latency_regression = comparison.performance_improvement.latency_improvement_percent < -5.0;
        let memory_regression = comparison.performance_improvement.memory_improvement_percent < -10.0;
        
        if latency_regression {
            println!("  ⚠️  LATENCY REGRESSION DETECTED: {:.1}% slower", 
                    -comparison.performance_improvement.latency_improvement_percent);
        }
        
        if memory_regression {
            println!("  ⚠️  MEMORY REGRESSION DETECTED: {:.1}% more memory", 
                    -comparison.performance_improvement.memory_improvement_percent);
        }
        
        if !latency_regression && !memory_regression {
            println!("  ✅ No significant performance regressions detected");
        }
        
        println!("  Statistical significance: {}", 
                if comparison.statistical_significance.is_significant { "YES" } else { "NO" });
    }

    println!();
    Ok(())
}

/// Helper function to create a sample configuration for testing
fn create_sample_config(name: &str, neighbors_count: usize, use_hnsw: bool) -> BenchmarkConfiguration {
    BenchmarkConfiguration {
        name: name.to_string(),
        description: format!("Sample configuration: neighbors_count={}, hnsw={}", neighbors_count, use_hnsw),
        config: LDCConfig {
            neighbors_count,
            use_hnsw_index: use_hnsw,
            max_bars_back: 2000,
            ..Default::default()
        },
        test_parameters: BenchmarkTestParameters {
            iterations: 5,
            warmup_iterations: 1,
            dataset_sizes: vec![50],
            k_values: vec![neighbors_count],
            enable_memory_profiling: false,
            enable_cpu_profiling: false,
        },
    }
}