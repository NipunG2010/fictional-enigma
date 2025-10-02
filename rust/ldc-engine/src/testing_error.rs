use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// Comprehensive testing error types with specific error scenarios and actionable debugging information
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum TestingError {
    /// Mathematical accuracy test failures with detailed analysis
    #[error("Mathematical accuracy test failed: {test_name}. Expected: {expected}, Actual: {actual}, Difference: {difference}, Tolerance: {tolerance}. {recommendation}")]
    MathematicalAccuracyError {
        test_name: String,
        expected: f64,
        actual: f64,
        difference: f64,
        tolerance: f64,
        recommendation: String,
    },

    /// Performance test failures with bottleneck analysis and optimization recommendations
    #[error("Performance test failed: {test_name}. Target: {target_ms}ms, Actual: {actual_ms}ms, Regression: {regression_percent}%. Bottleneck: {bottleneck}. Recommendations: {recommendations:?}")]
    PerformanceTestError {
        test_name: String,
        target_ms: f64,
        actual_ms: f64,
        regression_percent: f64,
        bottleneck: String,
        recommendations: Vec<String>,
    },

    /// Statistical test failures with sample size and significance analysis
    #[error("Statistical test failed: {test_name}. Sample size: {sample_size}, Required: {required_sample_size}, P-value: {p_value}, Significance threshold: {significance_threshold}. {diagnosis}")]
    StatisticalTestError {
        test_name: String,
        sample_size: usize,
        required_sample_size: usize,
        p_value: f64,
        significance_threshold: f64,
        diagnosis: String,
    },

    /// Integration test failures with component interaction analysis
    #[error("Integration test failed: {test_name}. Failed component: {component}, Error: {error_message}. Component interactions: {interactions:?}. Recovery suggestions: {recovery_suggestions:?}")]
    IntegrationTestError {
        test_name: String,
        component: String,
        error_message: String,
        interactions: Vec<ComponentInteraction>,
        recovery_suggestions: Vec<String>,
    },

    /// Test data validation errors with clear quality issue descriptions
    #[error("Test data validation failed: {validation_type}. Issues found: {issues:?}. Data quality score: {quality_score}/100. Recommendations: {recommendations:?}")]
    TestDataValidationError {
        validation_type: String,
        issues: Vec<DataQualityIssue>,
        quality_score: u32,
        recommendations: Vec<String>,
    },

    /// Backtesting test failures with strategy and market condition analysis
    #[error("Backtesting test failed: {test_name}. Strategy: {strategy_name}, Market conditions: {market_conditions}, Performance metrics: {performance_summary}. Analysis: {analysis}")]
    BacktestingTestError {
        test_name: String,
        strategy_name: String,
        market_conditions: String,
        performance_summary: String,
        analysis: String,
    },

    /// Configuration validation errors with specific parameter issues
    #[error("Configuration validation failed: {parameter} = {value} is invalid. Valid range: {valid_range}, Default: {default_value}. Impact: {impact}")]
    ConfigurationValidationError {
        parameter: String,
        value: String,
        valid_range: String,
        default_value: String,
        impact: String,
    },

    /// Resource exhaustion errors with adaptive behavior recommendations
    #[error("Resource exhaustion detected: {resource} usage at {usage_percent}%. Threshold: {threshold_percent}%. Adaptive actions: {adaptive_actions:?}")]
    ResourceExhaustionError {
        resource: String,
        usage_percent: f32,
        threshold_percent: f32,
        adaptive_actions: Vec<String>,
    },

    /// Test timeout errors with execution analysis
    #[error("Test timeout: {test_name} exceeded {timeout_seconds}s. Execution phase: {execution_phase}, Progress: {progress_percent}%. Suggestions: {suggestions:?}")]
    TestTimeoutError {
        test_name: String,
        timeout_seconds: u64,
        execution_phase: String,
        progress_percent: f32,
        suggestions: Vec<String>,
    },

    /// Test dependency errors with resolution guidance
    #[error("Test dependency error: {test_name} depends on {dependency}, which failed with: {dependency_error}. Resolution: {resolution}")]
    TestDependencyError {
        test_name: String,
        dependency: String,
        dependency_error: String,
        resolution: String,
    },
}

/// Component interaction information for integration test diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInteraction {
    pub from_component: String,
    pub to_component: String,
    pub interaction_type: String,
    pub status: InteractionStatus,
    pub error_details: Option<String>,
}

/// Status of component interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionStatus {
    Success,
    Failed,
    Timeout,
    NotTested,
}

/// Data quality issues for test data validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityIssue {
    pub issue_type: DataQualityIssueType,
    pub description: String,
    pub severity: IssueSeverity,
    pub affected_records: usize,
    pub suggested_fix: String,
}

/// Types of data quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataQualityIssueType {
    MissingValues,
    InvalidRange,
    Duplicates,
    Inconsistency,
    OutlierDetection,
    TemporalInconsistency,
    FeatureCorrelation,
}

/// Severity levels for data quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Detailed error reporting with actionable debugging information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestErrorReport {
    pub error: TestingError,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub test_context: TestContext,
    pub system_state: SystemState,
    pub debugging_info: DebuggingInfo,
    pub recovery_actions: Vec<RecoveryAction>,
}

/// Test execution context for error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestContext {
    pub test_suite: String,
    pub test_category: String,
    pub test_phase: String,
    pub configuration: HashMap<String, String>,
    pub environment_variables: HashMap<String, String>,
    pub input_parameters: HashMap<String, String>,
}

/// System state at the time of error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub thread_count: usize,
    pub open_file_descriptors: usize,
    pub network_connections: usize,
    pub system_load_average: f64,
}

/// Debugging information for error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebuggingInfo {
    pub stack_trace: Vec<String>,
    pub log_entries: Vec<LogEntry>,
    pub performance_metrics: HashMap<String, f64>,
    pub data_samples: Vec<String>,
    pub configuration_dump: HashMap<String, String>,
}

/// Log entry for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: LogLevel,
    pub message: String,
    pub component: String,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// Recovery action for error resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAction {
    pub action_type: RecoveryActionType,
    pub description: String,
    pub priority: ActionPriority,
    pub estimated_time_minutes: u32,
    pub success_probability: f32,
    pub side_effects: Vec<String>,
}

/// Types of recovery actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryActionType {
    RetryWithBackoff,
    FallbackToAlternative,
    ReduceResourceUsage,
    SkipNonCriticalTests,
    ReconfigureParameters,
    RestartComponent,
    CleanupResources,
    UpdateTestData,
}

/// Priority levels for recovery actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance test failure analysis with specific bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFailureAnalysis {
    pub test_name: String,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    pub resource_analysis: ResourceAnalysis,
    pub comparison_with_baseline: BaselineComparison,
}

/// Performance bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub component: String,
    pub bottleneck_type: BottleneckType,
    pub impact_percent: f32,
    pub description: String,
    pub measurement_details: HashMap<String, f64>,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
    Algorithm,
    Synchronization,
    GarbageCollection,
}

/// Optimization recommendation with specific actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_type: OptimizationType,
    pub description: String,
    pub expected_improvement_percent: f32,
    pub implementation_effort: ImplementationEffort,
    pub code_changes_required: Vec<String>,
    pub configuration_changes: HashMap<String, String>,
}

/// Types of optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    AlgorithmOptimization,
    DataStructureChange,
    CachingStrategy,
    ParallelizationImprovement,
    MemoryOptimization,
    IOOptimization,
    ConfigurationTuning,
}

/// Implementation effort estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,      // < 1 day
    Medium,   // 1-3 days
    High,     // 1-2 weeks
    VeryHigh, // > 2 weeks
}

/// Resource usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAnalysis {
    pub memory_analysis: MemoryAnalysis,
    pub cpu_analysis: CpuAnalysis,
    pub io_analysis: IoAnalysis,
    pub thread_analysis: ThreadAnalysis,
}

/// Memory usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    pub peak_usage_mb: f64,
    pub average_usage_mb: f64,
    pub allocation_rate_mb_per_sec: f64,
    pub garbage_collection_overhead_percent: f32,
    pub memory_leaks_detected: bool,
    pub fragmentation_percent: f32,
}

/// CPU usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuAnalysis {
    pub average_usage_percent: f64,
    pub peak_usage_percent: f64,
    pub core_utilization: Vec<f64>,
    pub context_switches_per_sec: f64,
    pub cache_miss_rate_percent: f32,
    pub instruction_efficiency: f64,
}

/// IO analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoAnalysis {
    pub read_operations_per_sec: f64,
    pub write_operations_per_sec: f64,
    pub average_read_latency_ms: f64,
    pub average_write_latency_ms: f64,
    pub throughput_mb_per_sec: f64,
    pub io_wait_percent: f32,
}

/// Thread analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAnalysis {
    pub thread_count: usize,
    pub active_threads: usize,
    pub blocked_threads: usize,
    pub thread_contention_events: u64,
    pub deadlock_detected: bool,
    pub thread_pool_efficiency_percent: f32,
}

/// Baseline comparison for performance regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_performance: HashMap<String, f64>,
    pub current_performance: HashMap<String, f64>,
    pub regression_analysis: Vec<RegressionAnalysis>,
    pub improvement_areas: Vec<String>,
}

/// Regression analysis for specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub metric_name: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub change_percent: f64,
    pub regression_severity: RegressionSeverity,
    pub trend_analysis: TrendAnalysis,
}

/// Regression severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    None,
    Minor,    // < 10% regression
    Moderate, // 10-25% regression
    Major,    // 25-50% regression
    Critical, // > 50% regression
}

/// Trend analysis for performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub trend_direction: TrendDirection,
    pub confidence_level: f32,
    pub historical_data_points: usize,
    pub prediction_next_period: f64,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Statistical test failure diagnostics with sample size and significance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFailureDiagnostics {
    pub test_name: String,
    pub sample_size_analysis: SampleSizeAnalysis,
    pub significance_analysis: SignificanceAnalysis,
    pub power_analysis: PowerAnalysis,
    pub effect_size_analysis: EffectSizeAnalysis,
    pub recommendations: Vec<StatisticalRecommendation>,
}

/// Sample size analysis for statistical tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleSizeAnalysis {
    pub current_sample_size: usize,
    pub minimum_required_size: usize,
    pub recommended_size: usize,
    pub power_achieved: f64,
    pub confidence_level: f64,
    pub adequacy_assessment: SampleAdequacy,
}

/// Sample adequacy assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SampleAdequacy {
    Adequate,
    Marginal,
    Inadequate,
    SeverelyInadequate,
}

/// Significance analysis for statistical tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceAnalysis {
    pub p_value: f64,
    pub significance_threshold: f64,
    pub confidence_interval: (f64, f64),
    pub effect_size: f64,
    pub statistical_power: f64,
    pub multiple_testing_correction: Option<String>,
}

/// Power analysis for statistical tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    pub observed_power: f64,
    pub target_power: f64,
    pub sample_size_for_target_power: usize,
    pub minimum_detectable_effect: f64,
    pub power_curve_data: Vec<(usize, f64)>, // (sample_size, power) pairs
}

/// Effect size analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeAnalysis {
    pub effect_size: f64,
    pub effect_size_interpretation: EffectSizeInterpretation,
    pub practical_significance: bool,
    pub confidence_interval: (f64, f64),
}

/// Effect size interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectSizeInterpretation {
    Negligible,
    Small,
    Medium,
    Large,
    VeryLarge,
}

/// Statistical recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalRecommendation {
    pub recommendation_type: StatisticalRecommendationType,
    pub description: String,
    pub priority: ActionPriority,
    pub implementation_steps: Vec<String>,
}

/// Types of statistical recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalRecommendationType {
    IncreaseSampleSize,
    AdjustSignificanceLevel,
    UseNonParametricTest,
    ApplyMultipleTestingCorrection,
    CollectMoreData,
    ChangeTestDesign,
    ConsiderPracticalSignificance,
}

impl TestingError {
    /// Create a mathematical accuracy error with detailed analysis
    pub fn mathematical_accuracy_error(
        test_name: String,
        expected: f64,
        actual: f64,
        tolerance: f64,
    ) -> Self {
        let difference = (expected - actual).abs();
        let recommendation = Self::generate_mathematical_recommendation(difference, tolerance, expected, actual);
        
        Self::MathematicalAccuracyError {
            test_name,
            expected,
            actual,
            difference,
            tolerance,
            recommendation,
        }
    }

    /// Create a performance test error with bottleneck analysis
    pub fn performance_test_error(
        test_name: String,
        target_ms: f64,
        actual_ms: f64,
        bottleneck: String,
    ) -> Self {
        let regression_percent = ((actual_ms - target_ms) / target_ms) * 100.0;
        let recommendations = Self::generate_performance_recommendations(regression_percent, &bottleneck);
        
        Self::PerformanceTestError {
            test_name,
            target_ms,
            actual_ms,
            regression_percent,
            bottleneck,
            recommendations,
        }
    }

    /// Create a statistical test error with sample size analysis
    pub fn statistical_test_error(
        test_name: String,
        sample_size: usize,
        p_value: f64,
        significance_threshold: f64,
    ) -> Self {
        let required_sample_size = Self::calculate_required_sample_size(p_value, significance_threshold);
        let diagnosis = Self::generate_statistical_diagnosis(sample_size, required_sample_size, p_value, significance_threshold);
        
        Self::StatisticalTestError {
            test_name,
            sample_size,
            required_sample_size,
            p_value,
            significance_threshold,
            diagnosis,
        }
    }

    /// Generate mathematical accuracy recommendations
    fn generate_mathematical_recommendation(difference: f64, tolerance: f64, _expected: f64, _actual: f64) -> String {
        let ratio = difference / tolerance;
        
        if ratio > 10.0 {
            format!("Critical accuracy failure ({}x tolerance). Check algorithm implementation and input data validity.", ratio.round())
        } else if ratio > 5.0 {
            format!("Significant accuracy issue ({}x tolerance). Verify numerical precision and calculation order.", ratio.round())
        } else if ratio > 2.0 {
            format!("Moderate accuracy issue ({}x tolerance). Consider increasing precision or adjusting tolerance.", ratio.round())
        } else {
            format!("Minor accuracy issue ({}x tolerance). May be acceptable depending on use case.", ratio.round())
        }
    }

    /// Generate performance optimization recommendations
    fn generate_performance_recommendations(regression_percent: f64, bottleneck: &str) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match bottleneck.to_lowercase().as_str() {
            "cpu" => {
                recommendations.push("Consider algorithm optimization or parallelization".to_string());
                recommendations.push("Profile CPU-intensive functions for optimization opportunities".to_string());
                if regression_percent > 50.0 {
                    recommendations.push("Critical: Investigate recent code changes that may have introduced inefficiencies".to_string());
                }
            },
            "memory" => {
                recommendations.push("Optimize memory allocation patterns and reduce memory footprint".to_string());
                recommendations.push("Consider using memory pools or object recycling".to_string());
                recommendations.push("Check for memory leaks or excessive allocations".to_string());
            },
            "io" => {
                recommendations.push("Optimize I/O operations with batching or caching".to_string());
                recommendations.push("Consider asynchronous I/O or memory-mapped files".to_string());
                recommendations.push("Reduce disk access frequency through better data structures".to_string());
            },
            _ => {
                recommendations.push("Profile the application to identify specific bottlenecks".to_string());
                recommendations.push("Consider general optimization techniques like caching and batching".to_string());
            }
        }
        
        if regression_percent > 25.0 {
            recommendations.push("High priority: Performance regression requires immediate attention".to_string());
        }
        
        recommendations
    }

    /// Calculate required sample size for statistical significance
    fn calculate_required_sample_size(p_value: f64, significance_threshold: f64) -> usize {
        // Simplified calculation - in practice, this would depend on effect size and power
        if p_value > significance_threshold {
            let ratio = p_value / significance_threshold;
            (100.0 * ratio * ratio) as usize
        } else {
            100 // Minimum reasonable sample size
        }
    }

    /// Generate statistical test diagnosis
    fn generate_statistical_diagnosis(
        sample_size: usize,
        required_sample_size: usize,
        p_value: f64,
        significance_threshold: f64,
    ) -> String {
        if sample_size < required_sample_size {
            format!(
                "Insufficient sample size. Need {} more samples for adequate power. Current p-value {} exceeds threshold {}.",
                required_sample_size - sample_size,
                p_value,
                significance_threshold
            )
        } else if p_value > significance_threshold {
            format!(
                "Test not statistically significant (p={:.4} > {:.4}). Consider effect size and practical significance.",
                p_value,
                significance_threshold
            )
        } else {
            format!(
                "Test failed despite adequate sample size and significance. Check test assumptions and data quality."
            )
        }
    }
}

impl fmt::Display for TestContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Test Suite: {}, Category: {}, Phase: {}", 
               self.test_suite, self.test_category, self.test_phase)
    }
}

impl fmt::Display for SystemState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Memory: {:.1}MB, CPU: {:.1}%, Threads: {}", 
               self.memory_usage_mb, self.cpu_usage_percent, self.thread_count)
    }
}