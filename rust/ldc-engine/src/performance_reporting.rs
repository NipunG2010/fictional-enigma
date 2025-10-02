use crate::performance_benchmarking::{BenchmarkResults, PerformanceMetrics};
use anyhow::Result;
use serde::{Deserialize, Serialize};

use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Performance visualization and reporting tools
pub struct PerformanceReporter {
    results_history: Vec<BenchmarkResults>,
    report_config: ReportConfig,
}

/// Configuration for report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub include_charts: bool,
    pub include_detailed_metrics: bool,
    pub include_recommendations: bool,
    pub output_format: ReportFormat,
    pub chart_width: usize,
    pub chart_height: usize,
}

/// Supported report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Html,
    Markdown,
    Json,
    Csv,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            include_charts: true,
            include_detailed_metrics: true,
            include_recommendations: true,
            output_format: ReportFormat::Html,
            chart_width: 800,
            chart_height: 400,
        }
    }
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub change_percent: f64,
    pub confidence: f64,
    pub data_points: Vec<TrendDataPoint>,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Individual trend data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendDataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub value: f64,
    pub configuration: String,
}

/// Performance dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDashboard {
    pub summary_metrics: DashboardSummary,
    pub performance_trends: Vec<PerformanceTrend>,
    pub configuration_comparison: Vec<ConfigurationComparison>,
    pub alerts: Vec<PerformanceAlert>,
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Dashboard summary metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    pub total_configurations_tested: usize,
    pub best_latency_ms: f64,
    pub best_accuracy_percent: f64,
    pub memory_efficiency_score: f64,
    pub overall_performance_score: f64,
}

/// Configuration comparison for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationComparison {
    pub name: String,
    pub latency_score: f64,
    pub accuracy_score: f64,
    pub memory_score: f64,
    pub overall_score: f64,
    pub rank: usize,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub configuration: String,
    pub metric_value: f64,
    pub threshold: f64,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    LatencyRegression,
    MemoryUsageHigh,
    AccuracyDrop,
    ThroughputDrop,
    ErrorRateHigh,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

/// Performance recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    pub title: String,
    pub description: String,
    pub expected_improvement: String,
    pub implementation_effort: EffortLevel,
    pub priority: Priority,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

impl PerformanceReporter {
    /// Create new performance reporter
    pub fn new() -> Self {
        Self::with_config(ReportConfig::default())
    }

    /// Create performance reporter with custom configuration
    pub fn with_config(config: ReportConfig) -> Self {
        Self {
            results_history: Vec::new(),
            report_config: config,
        }
    }

    /// Add benchmark results to history
    pub fn add_results(&mut self, results: BenchmarkResults) {
        self.results_history.push(results);
    }

    /// Get results history (for testing)
    pub fn results_history(&self) -> &[BenchmarkResults] {
        &self.results_history
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self, output_path: &Path) -> Result<()> {
        match self.report_config.output_format {
            ReportFormat::Html => self.generate_html_report(output_path),
            ReportFormat::Markdown => self.generate_markdown_report(output_path),
            ReportFormat::Json => self.generate_json_report(output_path),
            ReportFormat::Csv => self.generate_csv_report(output_path),
        }
    }

    /// Generate HTML report with charts and interactive elements
    fn generate_html_report(&self, output_path: &Path) -> Result<()> {
        let dashboard = self.create_dashboard();
        let html_content = self.create_html_content(&dashboard)?;
        
        let mut file = File::create(output_path)?;
        file.write_all(html_content.as_bytes())?;
        
        println!("HTML report generated: {}", output_path.display());
        Ok(())
    }

    /// Generate Markdown report
    fn generate_markdown_report(&self, output_path: &Path) -> Result<()> {
        let dashboard = self.create_dashboard();
        let markdown_content = self.create_markdown_content(&dashboard)?;
        
        let mut file = File::create(output_path)?;
        file.write_all(markdown_content.as_bytes())?;
        
        println!("Markdown report generated: {}", output_path.display());
        Ok(())
    }

    /// Generate JSON report
    fn generate_json_report(&self, output_path: &Path) -> Result<()> {
        let dashboard = self.create_dashboard();
        let json_content = serde_json::to_string_pretty(&dashboard)?;
        
        let mut file = File::create(output_path)?;
        file.write_all(json_content.as_bytes())?;
        
        println!("JSON report generated: {}", output_path.display());
        Ok(())
    }

    /// Generate CSV report
    fn generate_csv_report(&self, output_path: &Path) -> Result<()> {
        let mut csv_content = String::new();
        
        // CSV header
        csv_content.push_str("Configuration,Timestamp,Avg_Latency_ms,P95_Latency_ms,Throughput_QPS,Accuracy_Percent,Memory_MB\n");
        
        // CSV data
        for result in &self.results_history {
            csv_content.push_str(&format!(
                "{},{},{:.3},{:.3},{:.1},{:.1},{:.1}\n",
                result.configuration_name,
                result.test_timestamp.format("%Y-%m-%d %H:%M:%S"),
                result.performance_metrics.avg_query_latency_ms,
                result.performance_metrics.p95_latency_ms,
                result.performance_metrics.throughput_queries_per_second,
                result.accuracy_metrics.prediction_accuracy_percent,
                result.memory_metrics.avg_memory_usage_mb
            ));
        }
        
        let mut file = File::create(output_path)?;
        file.write_all(csv_content.as_bytes())?;
        
        println!("CSV report generated: {}", output_path.display());
        Ok(())
    }

    /// Create performance dashboard
    pub fn create_dashboard(&self) -> PerformanceDashboard {
        let summary_metrics = self.calculate_summary_metrics();
        let performance_trends = self.analyze_performance_trends();
        let configuration_comparison = self.compare_configurations();
        let alerts = self.generate_alerts();
        let recommendations = self.generate_recommendations();

        PerformanceDashboard {
            summary_metrics,
            performance_trends,
            configuration_comparison,
            alerts,
            recommendations,
        }
    }

    /// Calculate summary metrics for dashboard
    fn calculate_summary_metrics(&self) -> DashboardSummary {
        if self.results_history.is_empty() {
            return DashboardSummary {
                total_configurations_tested: 0,
                best_latency_ms: 0.0,
                best_accuracy_percent: 0.0,
                memory_efficiency_score: 0.0,
                overall_performance_score: 0.0,
            };
        }

        let best_latency = self.results_history.iter()
            .map(|r| r.performance_metrics.avg_query_latency_ms)
            .fold(f64::INFINITY, f64::min);

        let best_accuracy = self.results_history.iter()
            .map(|r| r.accuracy_metrics.prediction_accuracy_percent)
            .fold(0.0, f64::max);

        let avg_memory_efficiency = self.results_history.iter()
            .map(|r| r.memory_metrics.memory_efficiency_percent)
            .sum::<f64>() / self.results_history.len() as f64;

        let overall_score = self.calculate_overall_performance_score();

        DashboardSummary {
            total_configurations_tested: self.results_history.len(),
            best_latency_ms: best_latency,
            best_accuracy_percent: best_accuracy,
            memory_efficiency_score: avg_memory_efficiency,
            overall_performance_score: overall_score,
        }
    }

    /// Analyze performance trends over time
    pub fn analyze_performance_trends(&self) -> Vec<PerformanceTrend> {
        let mut trends = Vec::new();

        // Latency trend
        let latency_points: Vec<TrendDataPoint> = self.results_history.iter()
            .map(|r| TrendDataPoint {
                timestamp: r.test_timestamp,
                value: r.performance_metrics.avg_query_latency_ms,
                configuration: r.configuration_name.clone(),
            })
            .collect();

        if latency_points.len() >= 2 {
            let trend_direction = self.calculate_trend_direction(&latency_points, false); // Lower is better
            let change_percent = self.calculate_change_percent(&latency_points);
            
            trends.push(PerformanceTrend {
                metric_name: "Average Latency".to_string(),
                trend_direction,
                change_percent,
                confidence: 0.85,
                data_points: latency_points,
            });
        }

        // Accuracy trend
        let accuracy_points: Vec<TrendDataPoint> = self.results_history.iter()
            .map(|r| TrendDataPoint {
                timestamp: r.test_timestamp,
                value: r.accuracy_metrics.prediction_accuracy_percent,
                configuration: r.configuration_name.clone(),
            })
            .collect();

        if accuracy_points.len() >= 2 {
            let trend_direction = self.calculate_trend_direction(&accuracy_points, true); // Higher is better
            let change_percent = self.calculate_change_percent(&accuracy_points);
            
            trends.push(PerformanceTrend {
                metric_name: "Prediction Accuracy".to_string(),
                trend_direction,
                change_percent,
                confidence: 0.80,
                data_points: accuracy_points,
            });
        }

        trends
    }

    /// Calculate trend direction
    fn calculate_trend_direction(&self, data_points: &[TrendDataPoint], higher_is_better: bool) -> TrendDirection {
        if data_points.len() < 2 {
            return TrendDirection::Stable;
        }

        let first_value = data_points[0].value;
        let last_value = data_points[data_points.len() - 1].value;
        let change_percent = ((last_value - first_value) / first_value) * 100.0;

        // Calculate volatility
        let values: Vec<f64> = data_points.iter().map(|p| p.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let coefficient_of_variation = variance.sqrt() / mean;

        if coefficient_of_variation > 0.2 {
            return TrendDirection::Volatile;
        }

        match change_percent.abs() {
            x if x < 2.0 => TrendDirection::Stable,
            _ => {
                if higher_is_better {
                    if change_percent > 0.0 { TrendDirection::Improving } else { TrendDirection::Degrading }
                } else {
                    if change_percent < 0.0 { TrendDirection::Improving } else { TrendDirection::Degrading }
                }
            }
        }
    }

    /// Calculate percentage change between first and last data points
    fn calculate_change_percent(&self, data_points: &[TrendDataPoint]) -> f64 {
        if data_points.len() < 2 {
            return 0.0;
        }

        let first_value = data_points[0].value;
        let last_value = data_points[data_points.len() - 1].value;
        
        if first_value == 0.0 {
            return 0.0;
        }

        ((last_value - first_value) / first_value) * 100.0
    }

    /// Compare configurations and rank them
    fn compare_configurations(&self) -> Vec<ConfigurationComparison> {
        let mut comparisons: Vec<ConfigurationComparison> = self.results_history.iter()
            .map(|result| {
                let latency_score = self.calculate_latency_score(&result.performance_metrics);
                let accuracy_score = result.accuracy_metrics.prediction_accuracy_percent;
                let memory_score = result.memory_metrics.memory_efficiency_percent;
                let overall_score = (latency_score + accuracy_score + memory_score) / 3.0;

                ConfigurationComparison {
                    name: result.configuration_name.clone(),
                    latency_score,
                    accuracy_score,
                    memory_score,
                    overall_score,
                    rank: 0, // Will be set after sorting
                }
            })
            .collect();

        // Sort by overall score (descending)
        comparisons.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());

        // Assign ranks
        for (i, comparison) in comparisons.iter_mut().enumerate() {
            comparison.rank = i + 1;
        }

        comparisons
    }

    /// Calculate latency score (lower latency = higher score)
    fn calculate_latency_score(&self, metrics: &PerformanceMetrics) -> f64 {
        // Convert latency to score (lower is better)
        let max_acceptable_latency = 10.0; // 10ms
        let score = ((max_acceptable_latency - metrics.avg_query_latency_ms) / max_acceptable_latency) * 100.0;
        score.max(0.0f64).min(100.0f64)
    }

    /// Generate performance alerts
    fn generate_alerts(&self) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();

        for result in &self.results_history {
            // Latency regression alert
            if result.performance_metrics.avg_query_latency_ms > 5.0 {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::LatencyRegression,
                    severity: if result.performance_metrics.avg_query_latency_ms > 10.0 {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    message: format!("High latency detected: {:.3}ms", result.performance_metrics.avg_query_latency_ms),
                    configuration: result.configuration_name.clone(),
                    metric_value: result.performance_metrics.avg_query_latency_ms,
                    threshold: 5.0,
                });
            }

            // Memory usage alert
            if result.memory_metrics.avg_memory_usage_mb > 500.0 {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::MemoryUsageHigh,
                    severity: AlertSeverity::Warning,
                    message: format!("High memory usage: {:.1}MB", result.memory_metrics.avg_memory_usage_mb),
                    configuration: result.configuration_name.clone(),
                    metric_value: result.memory_metrics.avg_memory_usage_mb,
                    threshold: 500.0,
                });
            }

            // Accuracy drop alert
            if result.accuracy_metrics.prediction_accuracy_percent < 80.0 {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::AccuracyDrop,
                    severity: AlertSeverity::Warning,
                    message: format!("Low accuracy: {:.1}%", result.accuracy_metrics.prediction_accuracy_percent),
                    configuration: result.configuration_name.clone(),
                    metric_value: result.accuracy_metrics.prediction_accuracy_percent,
                    threshold: 80.0,
                });
            }
        }

        alerts
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();

        if self.results_history.is_empty() {
            return recommendations;
        }

        // Analyze results for recommendations
        let avg_latency = self.results_history.iter()
            .map(|r| r.performance_metrics.avg_query_latency_ms)
            .sum::<f64>() / self.results_history.len() as f64;

        let avg_memory = self.results_history.iter()
            .map(|r| r.memory_metrics.avg_memory_usage_mb)
            .sum::<f64>() / self.results_history.len() as f64;

        // High latency recommendation
        if avg_latency > 3.0 {
            recommendations.push(PerformanceRecommendation {
                title: "Enable HNSW Indexing".to_string(),
                description: "Consider enabling HNSW indexing to reduce query latency for large datasets".to_string(),
                expected_improvement: "30-50% latency reduction".to_string(),
                implementation_effort: EffortLevel::Low,
                priority: Priority::High,
            });
        }

        // High memory usage recommendation
        if avg_memory > 300.0 {
            recommendations.push(PerformanceRecommendation {
                title: "Optimize Memory Usage".to_string(),
                description: "Implement memory-mapped storage or reduce max_samples to lower memory footprint".to_string(),
                expected_improvement: "20-40% memory reduction".to_string(),
                implementation_effort: EffortLevel::Medium,
                priority: Priority::Medium,
            });
        }

        // SIMD optimization recommendation
        recommendations.push(PerformanceRecommendation {
            title: "Enable SIMD Optimizations".to_string(),
            description: "Use SIMD instructions for batch distance calculations to improve throughput".to_string(),
            expected_improvement: "15-25% throughput increase".to_string(),
            implementation_effort: EffortLevel::Medium,
            priority: Priority::Medium,
        });

        recommendations
    }

    /// Calculate overall performance score
    fn calculate_overall_performance_score(&self) -> f64 {
        if self.results_history.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;
        
        for result in &self.results_history {
            let latency_score = self.calculate_latency_score(&result.performance_metrics);
            let accuracy_score = result.accuracy_metrics.prediction_accuracy_percent;
            let memory_score = result.memory_metrics.memory_efficiency_percent;
            
            // Weighted average: latency 40%, accuracy 40%, memory 20%
            let weighted_score = (latency_score * 0.4) + (accuracy_score * 0.4) + (memory_score * 0.2);
            total_score += weighted_score;
        }

        total_score / self.results_history.len() as f64
    }

    /// Create HTML content for the report
    fn create_html_content(&self, dashboard: &PerformanceDashboard) -> Result<String> {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>LDC Engine Performance Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(include_str!("../assets/report_styles.css"));
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");
        
        // Header
        html.push_str("<h1>LDC Engine Performance Report</h1>\n");
        html.push_str(&format!("<p>Generated: {}</p>\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        // Summary section
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<div class='summary-grid'>\n");
        html.push_str(&format!("<div class='metric-card'><h3>Configurations Tested</h3><p>{}</p></div>\n", 
                              dashboard.summary_metrics.total_configurations_tested));
        html.push_str(&format!("<div class='metric-card'><h3>Best Latency</h3><p>{:.3}ms</p></div>\n", 
                              dashboard.summary_metrics.best_latency_ms));
        html.push_str(&format!("<div class='metric-card'><h3>Best Accuracy</h3><p>{:.1}%</p></div>\n", 
                              dashboard.summary_metrics.best_accuracy_percent));
        html.push_str(&format!("<div class='metric-card'><h3>Overall Score</h3><p>{:.1}/100</p></div>\n", 
                              dashboard.summary_metrics.overall_performance_score));
        html.push_str("</div>\n");
        
        // Configuration comparison
        html.push_str("<h2>Configuration Comparison</h2>\n");
        html.push_str("<table class='comparison-table'>\n");
        html.push_str("<tr><th>Rank</th><th>Configuration</th><th>Latency Score</th><th>Accuracy Score</th><th>Memory Score</th><th>Overall Score</th></tr>\n");
        
        for config in &dashboard.configuration_comparison {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{:.1}</td><td>{:.1}</td><td>{:.1}</td><td>{:.1}</td></tr>\n",
                config.rank, config.name, config.latency_score, config.accuracy_score, config.memory_score, config.overall_score
            ));
        }
        html.push_str("</table>\n");
        
        // Alerts section
        if !dashboard.alerts.is_empty() {
            html.push_str("<h2>Alerts</h2>\n");
            for alert in &dashboard.alerts {
                let severity_class = match alert.severity {
                    AlertSeverity::Critical => "alert-critical",
                    AlertSeverity::Warning => "alert-warning",
                    AlertSeverity::Info => "alert-info",
                };
                html.push_str(&format!("<div class='alert {}'>{}</div>\n", severity_class, alert.message));
            }
        }
        
        // Recommendations section
        if !dashboard.recommendations.is_empty() {
            html.push_str("<h2>Recommendations</h2>\n");
            for rec in &dashboard.recommendations {
                html.push_str("<div class='recommendation'>\n");
                html.push_str(&format!("<h3>{}</h3>\n", rec.title));
                html.push_str(&format!("<p>{}</p>\n", rec.description));
                html.push_str(&format!("<p><strong>Expected Improvement:</strong> {}</p>\n", rec.expected_improvement));
                html.push_str("</div>\n");
            }
        }
        
        html.push_str("</body>\n</html>");
        Ok(html)
    }

    /// Create Markdown content for the report
    fn create_markdown_content(&self, dashboard: &PerformanceDashboard) -> Result<String> {
        let mut md = String::new();
        
        md.push_str("# LDC Engine Performance Report\n\n");
        md.push_str(&format!("Generated: {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!("- **Configurations Tested:** {}\n", dashboard.summary_metrics.total_configurations_tested));
        md.push_str(&format!("- **Best Latency:** {:.3}ms\n", dashboard.summary_metrics.best_latency_ms));
        md.push_str(&format!("- **Best Accuracy:** {:.1}%\n", dashboard.summary_metrics.best_accuracy_percent));
        md.push_str(&format!("- **Overall Performance Score:** {:.1}/100\n\n", dashboard.summary_metrics.overall_performance_score));
        
        // Configuration comparison
        md.push_str("## Configuration Comparison\n\n");
        md.push_str("| Rank | Configuration | Latency Score | Accuracy Score | Memory Score | Overall Score |\n");
        md.push_str("|------|---------------|---------------|----------------|--------------|---------------|\n");
        
        for config in &dashboard.configuration_comparison {
            md.push_str(&format!(
                "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
                config.rank, config.name, config.latency_score, config.accuracy_score, config.memory_score, config.overall_score
            ));
        }
        md.push_str("\n");
        
        // Alerts
        if !dashboard.alerts.is_empty() {
            md.push_str("## Alerts\n\n");
            for alert in &dashboard.alerts {
                let severity_emoji = match alert.severity {
                    AlertSeverity::Critical => "🔴",
                    AlertSeverity::Warning => "🟡",
                    AlertSeverity::Info => "🔵",
                };
                md.push_str(&format!("{} {}\n", severity_emoji, alert.message));
            }
            md.push_str("\n");
        }
        
        // Recommendations
        if !dashboard.recommendations.is_empty() {
            md.push_str("## Recommendations\n\n");
            for rec in &dashboard.recommendations {
                md.push_str(&format!("### {}\n\n", rec.title));
                md.push_str(&format!("{}\n\n", rec.description));
                md.push_str(&format!("**Expected Improvement:** {}\n\n", rec.expected_improvement));
            }
        }
        
        Ok(md)
    }
}

impl Default for PerformanceReporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance_benchmarking::{PerformanceMetrics, MemoryMetrics, AccuracyMetrics};

    fn create_sample_benchmark_result(name: &str, latency: f64, accuracy: f64) -> BenchmarkResults {
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
    fn test_performance_reporter_creation() {
        let reporter = PerformanceReporter::new();
        assert_eq!(reporter.results_history.len(), 0);
    }

    #[test]
    fn test_add_results() {
        let mut reporter = PerformanceReporter::new();
        let result = create_sample_benchmark_result("test", 2.5, 85.0);
        
        reporter.add_results(result);
        assert_eq!(reporter.results_history.len(), 1);
    }

    #[test]
    fn test_dashboard_creation() {
        let mut reporter = PerformanceReporter::new();
        reporter.add_results(create_sample_benchmark_result("config1", 2.0, 90.0));
        reporter.add_results(create_sample_benchmark_result("config2", 3.0, 85.0));
        
        let dashboard = reporter.create_dashboard();
        assert_eq!(dashboard.summary_metrics.total_configurations_tested, 2);
        assert_eq!(dashboard.summary_metrics.best_latency_ms, 2.0);
        assert_eq!(dashboard.summary_metrics.best_accuracy_percent, 90.0);
    }

    #[test]
    fn test_trend_analysis() {
        let mut reporter = PerformanceReporter::new();
        
        // Add results with improving latency trend
        reporter.add_results(create_sample_benchmark_result("v1", 5.0, 80.0));
        reporter.add_results(create_sample_benchmark_result("v2", 3.0, 85.0));
        reporter.add_results(create_sample_benchmark_result("v3", 2.0, 90.0));
        
        let trends = reporter.analyze_performance_trends();
        assert!(!trends.is_empty());
        
        // Should detect improving latency trend (lower is better)
        let latency_trend = trends.iter().find(|t| t.metric_name == "Average Latency");
        assert!(latency_trend.is_some());
    }
}