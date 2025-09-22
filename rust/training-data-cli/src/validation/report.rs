// Validation report generation

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::{ValidationResult, ValidationStatus};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub report_id: String,
    pub created_at: DateTime<Utc>,
    pub data_source: Option<String>,
    pub validation_result: ValidationResult,
    pub summary: ValidationSummary,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub overall_status: ValidationStatus,
    pub total_issues: usize,
    pub critical_issues: usize,
    pub warnings: usize,
    pub passed_checks: usize,
    pub total_checks: usize,
}

impl ValidationReport {
    pub fn new(validation_result: ValidationResult, data_source: Option<String>) -> Self {
        let report_id = format!("validation_{}", Utc::now().timestamp());
        let created_at = Utc::now();
        
        let summary = Self::create_summary(&validation_result);
        let recommendations = Self::generate_recommendations(&validation_result);
        
        Self {
            report_id,
            created_at,
            data_source,
            validation_result,
            summary,
            recommendations,
        }
    }

    /// Generate a human-readable report for CLI output
    pub fn format_human_readable(&self) -> String {
        let mut output = String::new();
        
        // Header
        output.push_str(&format!("╔═══════════════════════════════════════════════════════════════════════════════╗\n"));
        output.push_str(&format!("║                           VALIDATION REPORT                                  ║\n"));
        output.push_str(&format!("╠═══════════════════════════════════════════════════════════════════════════════╣\n"));
        output.push_str(&format!("║ Report ID: {:<66} ║\n", self.report_id));
        output.push_str(&format!("║ Created:   {:<66} ║\n", self.created_at.format("%Y-%m-%d %H:%M:%S UTC")));
        if let Some(source) = &self.data_source {
            output.push_str(&format!("║ Source:    {:<66} ║\n", source));
        }
        output.push_str(&format!("╚═══════════════════════════════════════════════════════════════════════════════╝\n\n"));

        // Overall Status
        let status_symbol = match self.summary.overall_status {
            ValidationStatus::Passed => "✅",
            ValidationStatus::Warning => "⚠️ ",
            ValidationStatus::Failed => "❌",
        };
        
        output.push_str(&format!("Overall Status: {} {:?}\n\n", status_symbol, self.summary.overall_status));

        // Summary Statistics
        output.push_str("SUMMARY\n");
        output.push_str("═══════\n");
        output.push_str(&format!("• Total Checks:     {}\n", self.summary.total_checks));
        output.push_str(&format!("• Passed:           {}\n", self.summary.passed_checks));
        output.push_str(&format!("• Warnings:         {}\n", self.summary.warnings));
        output.push_str(&format!("• Critical Issues:  {}\n", self.summary.critical_issues));
        output.push_str(&format!("• Total Issues:     {}\n\n", self.summary.total_issues));

        // Data Statistics
        output.push_str("DATA OVERVIEW\n");
        output.push_str("═════════════\n");
        let stats = &self.validation_result.statistics;
        output.push_str(&format!("• Rows:             {}\n", stats.row_count));
        output.push_str(&format!("• Columns:          {}\n", stats.column_count));
        output.push_str(&format!("• Numeric Columns:  {}\n", stats.numeric_columns.len()));
        output.push_str(&format!("• Memory Usage:     {:.2} MB\n\n", stats.memory_usage_bytes as f64 / 1_048_576.0));

        // Detailed Results
        output.push_str("DETAILED RESULTS\n");
        output.push_str("════════════════\n\n");

        // Missing Values
        output.push_str(&self.format_missing_values_section());
        output.push_str(&self.format_outliers_section());
        output.push_str(&self.format_timestamps_section());
        output.push_str(&self.format_duplicates_section());

        // Recommendations
        if !self.recommendations.is_empty() {
            output.push_str("RECOMMENDATIONS\n");
            output.push_str("═══════════════\n");
            for (i, rec) in self.recommendations.iter().enumerate() {
                output.push_str(&format!("{}. {}\n", i + 1, rec));
            }
            output.push_str("\n");
        }

        output
    }

    /// Generate JSON report for programmatic consumption
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Generate compact JSON report
    pub fn to_json_compact(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    // Private helper methods for formatting sections

    fn format_missing_values_section(&self) -> String {
        let mut output = String::new();
        let missing = &self.validation_result.missing_values;
        
        let status_symbol = match missing.status {
            ValidationStatus::Passed => "✅",
            ValidationStatus::Warning => "⚠️ ",
            ValidationStatus::Failed => "❌",
        };

        output.push_str(&format!("Missing Values {}\n", status_symbol));
        output.push_str("──────────────────\n");
        output.push_str(&format!("• Missing Percentage: {:.2}%\n", missing.missing_percentage));
        
        if missing.columns_with_missing.is_empty() {
            output.push_str("• No missing values found\n");
        } else {
            output.push_str("• Columns with missing values:\n");
            for (column, count) in &missing.columns_with_missing {
                let percentage = (*count as f64 / missing.total_rows as f64) * 100.0;
                output.push_str(&format!("  - {}: {} ({:.2}%)\n", column, count, percentage));
            }
        }
        output.push_str("\n");
        output
    }

    fn format_outliers_section(&self) -> String {
        let mut output = String::new();
        let outliers = &self.validation_result.outliers;
        
        let status_symbol = match outliers.status {
            ValidationStatus::Passed => "✅",
            ValidationStatus::Warning => "⚠️ ",
            ValidationStatus::Failed => "❌",
        };

        output.push_str(&format!("Outliers {} (Method: {:?})\n", status_symbol, outliers.method_used));
        output.push_str("─────────────────────────────────\n");
        output.push_str(&format!("• Total Outliers: {}\n", outliers.total_outliers));
        
        if outliers.columns_with_outliers.is_empty() {
            output.push_str("• No outliers detected\n");
        } else {
            output.push_str("• Columns with outliers:\n");
            for (column, stats) in &outliers.columns_with_outliers {
                output.push_str(&format!("  - {}: {} ({:.2}%)\n", column, stats.count, stats.percentage));
                if let (Some(lower), Some(upper)) = (stats.threshold_lower, stats.threshold_upper) {
                    output.push_str(&format!("    Thresholds: [{:.4}, {:.4}]\n", lower, upper));
                }
            }
        }
        output.push_str("\n");
        output
    }

    fn format_timestamps_section(&self) -> String {
        let mut output = String::new();
        let timestamps = &self.validation_result.timestamps;
        
        let status_symbol = match timestamps.status {
            ValidationStatus::Passed => "✅",
            ValidationStatus::Warning => "⚠️ ",
            ValidationStatus::Failed => "❌",
        };

        output.push_str(&format!("Timestamps {}\n", status_symbol));
        output.push_str("──────────────\n");
        output.push_str(&format!("• Sequential: {}\n", if timestamps.sequential { "Yes" } else { "No" }));
        output.push_str(&format!("• Gaps Found: {}\n", timestamps.gaps_found));
        output.push_str(&format!("• Duplicate Timestamps: {}\n", timestamps.duplicate_timestamps));
        
        if let Some(expected) = timestamps.expected_interval_seconds {
            output.push_str(&format!("• Expected Interval: {} seconds\n", expected));
        }
        
        if !timestamps.actual_intervals.is_empty() {
            let avg_interval = timestamps.actual_intervals.iter().sum::<i64>() as f64 / timestamps.actual_intervals.len() as f64;
            output.push_str(&format!("• Average Interval: {:.1} seconds\n", avg_interval));
        }
        output.push_str("\n");
        output
    }

    fn format_duplicates_section(&self) -> String {
        let mut output = String::new();
        let duplicates = &self.validation_result.duplicates;
        
        let status_symbol = match duplicates.status {
            ValidationStatus::Passed => "✅",
            ValidationStatus::Warning => "⚠️ ",
            ValidationStatus::Failed => "❌",
        };

        output.push_str(&format!("Duplicates {}\n", status_symbol));
        output.push_str("──────────────\n");
        output.push_str(&format!("• Duplicate Rows: {} ({:.2}%)\n", 
            duplicates.duplicate_rows, duplicates.duplicate_percentage));
        output.push_str(&format!("• Removed: {}\n", if duplicates.removed { "Yes" } else { "No" }));
        output.push_str("\n");
        output
    }

    fn create_summary(validation_result: &ValidationResult) -> ValidationSummary {
        let checks = [
            &validation_result.missing_values.status,
            &validation_result.outliers.status,
            &validation_result.timestamps.status,
            &validation_result.duplicates.status,
        ];

        let total_checks = checks.len();
        let passed_checks = checks.iter().filter(|s| matches!(s, ValidationStatus::Passed)).count();
        let warnings = checks.iter().filter(|s| matches!(s, ValidationStatus::Warning)).count();
        let critical_issues = checks.iter().filter(|s| matches!(s, ValidationStatus::Failed)).count();
        let total_issues = warnings + critical_issues;

        ValidationSummary {
            overall_status: validation_result.overall_status.clone(),
            total_issues,
            critical_issues,
            warnings,
            passed_checks,
            total_checks,
        }
    }

    fn generate_recommendations(validation_result: &ValidationResult) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Missing values recommendations
        if matches!(validation_result.missing_values.status, ValidationStatus::Warning | ValidationStatus::Failed) {
            if validation_result.missing_values.missing_percentage > 10.0 {
                recommendations.push("Consider investigating the source of missing values - high percentage detected".to_string());
            }
            recommendations.push("Consider using interpolation or forward-fill for missing values in time series data".to_string());
        }

        // Outliers recommendations
        if matches!(validation_result.outliers.status, ValidationStatus::Warning) {
            recommendations.push("Review detected outliers to determine if they represent data errors or genuine extreme values".to_string());
            if validation_result.outliers.total_outliers > 100 {
                recommendations.push("Consider using robust scaling or outlier removal techniques for model training".to_string());
            }
        }

        // Timestamp recommendations
        if matches!(validation_result.timestamps.status, ValidationStatus::Warning | ValidationStatus::Failed) {
            if !validation_result.timestamps.sequential {
                recommendations.push("Sort data by timestamp to ensure chronological order".to_string());
            }
            if validation_result.timestamps.gaps_found > 0 {
                recommendations.push("Consider filling timestamp gaps with interpolated values or explicit missing markers".to_string());
            }
            if validation_result.timestamps.duplicate_timestamps > 0 {
                recommendations.push("Remove or aggregate duplicate timestamps before model training".to_string());
            }
        }

        // Duplicates recommendations
        if matches!(validation_result.duplicates.status, ValidationStatus::Warning) {
            recommendations.push("Remove duplicate rows to prevent data leakage in model training".to_string());
        }

        // General recommendations
        if validation_result.statistics.row_count < 1000 {
            recommendations.push("Dataset is relatively small - consider gathering more data for robust model training".to_string());
        }

        recommendations
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_human_readable())
    }
}