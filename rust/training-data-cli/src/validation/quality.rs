// Data quality validation implementation

use anyhow::{anyhow, Result};

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub outlier_method: OutlierMethod,
    pub outlier_threshold: f64,
    pub max_missing_percentage: f64,
    pub require_sequential_timestamps: bool,
    pub expected_interval_seconds: Option<i64>,
    pub remove_duplicates: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            outlier_method: OutlierMethod::IQR,
            outlier_threshold: 3.0, // 3 standard deviations or 1.5 * IQR
            max_missing_percentage: 5.0, // 5% missing values allowed
            require_sequential_timestamps: true,
            expected_interval_seconds: Some(300), // 5 minutes default
            remove_duplicates: true,
        }
    }
}

impl ValidationConfig {
    /// Create a strict validation configuration
    pub fn strict() -> Self {
        Self {
            outlier_method: OutlierMethod::ZScore,
            outlier_threshold: 2.0, // Stricter threshold
            max_missing_percentage: 1.0, // Only 1% missing allowed
            require_sequential_timestamps: true,
            expected_interval_seconds: Some(300),
            remove_duplicates: true,
        }
    }

    /// Create a normal validation configuration (same as default)
    pub fn normal() -> Self {
        Self::default()
    }

    /// Create a lenient validation configuration
    pub fn lenient() -> Self {
        Self {
            outlier_method: OutlierMethod::IQR,
            outlier_threshold: 5.0, // More lenient threshold
            max_missing_percentage: 10.0, // Allow more missing values
            require_sequential_timestamps: false,
            expected_interval_seconds: None,
            remove_duplicates: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OutlierMethod {
    IQR,
    ZScore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Passed,
    Warning,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingValueReport {
    pub total_rows: usize,
    pub columns_with_missing: HashMap<String, usize>,
    pub missing_percentage: f64,
    pub status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierReport {
    pub method_used: OutlierMethod,
    pub columns_with_outliers: HashMap<String, OutlierStats>,
    pub total_outliers: usize,
    pub status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierStats {
    pub count: usize,
    pub percentage: f64,
    pub threshold_lower: Option<f64>,
    pub threshold_upper: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampReport {
    pub total_rows: usize,
    pub sequential: bool,
    pub gaps_found: usize,
    pub duplicate_timestamps: usize,
    pub expected_interval_seconds: Option<i64>,
    pub actual_intervals: Vec<i64>,
    pub status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateReport {
    pub total_rows: usize,
    pub duplicate_rows: usize,
    pub duplicate_percentage: f64,
    pub removed: bool,
    pub status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics {
    pub row_count: usize,
    pub column_count: usize,
    pub numeric_columns: Vec<String>,
    pub timestamp_columns: Vec<String>,
    pub memory_usage_bytes: usize,
}

#[derive(Debug)]
pub struct DataValidator {
    pub config: ValidationConfig,
}

impl DataValidator {
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Perform comprehensive validation on the DataFrame
    pub fn validate(&self, data: &DataFrame) -> Result<ValidationResult> {
        let statistics = self.calculate_statistics(data)?;
        let missing_values = self.check_missing_values(data)?;
        let outliers = self.detect_outliers(data)?;
        let timestamps = self.validate_timestamps(data)?;
        let duplicates = self.check_duplicates(data)?;

        let overall_status = self.determine_overall_status(&missing_values, &outliers, &timestamps, &duplicates);

        Ok(ValidationResult {
            overall_status,
            missing_values,
            outliers,
            timestamps,
            duplicates,
            statistics,
        })
    }

    /// Check for missing values in all columns
    pub fn check_missing_values(&self, data: &DataFrame) -> Result<MissingValueReport> {
        let total_rows = data.height();
        let mut columns_with_missing = HashMap::new();
        let mut total_missing = 0;

        for column in data.get_columns() {
            let null_count = column.null_count();
            if null_count > 0 {
                columns_with_missing.insert(column.name().to_string(), null_count);
                total_missing += null_count;
            }
        }

        let missing_percentage = if total_rows > 0 {
            (total_missing as f64 / (total_rows * data.width()) as f64) * 100.0
        } else {
            0.0
        };

        let status = if missing_percentage > self.config.max_missing_percentage {
            ValidationStatus::Failed
        } else if missing_percentage > 0.0 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Passed
        };

        Ok(MissingValueReport {
            total_rows,
            columns_with_missing,
            missing_percentage,
            status,
        })
    }

    /// Detect outliers using statistical methods
    pub fn detect_outliers(&self, data: &DataFrame) -> Result<OutlierReport> {
        let mut columns_with_outliers = HashMap::new();
        let mut total_outliers = 0;

        // Get numeric columns only
        let numeric_columns: Vec<_> = data
            .get_columns()
            .iter()
            .filter(|col| col.dtype().is_numeric())
            .collect();

        for column in numeric_columns {
            let outlier_stats = match self.config.outlier_method {
                OutlierMethod::IQR => self.detect_outliers_iqr(column)?,
                OutlierMethod::ZScore => self.detect_outliers_zscore(column)?,
            };

            if outlier_stats.count > 0 {
                total_outliers += outlier_stats.count;
                columns_with_outliers.insert(column.name().to_string(), outlier_stats);
            }
        }

        let status = if total_outliers > 0 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Passed
        };

        Ok(OutlierReport {
            method_used: self.config.outlier_method.clone(),
            columns_with_outliers,
            total_outliers,
            status,
        })
    }

    /// Validate timestamp column for sequential and complete time series
    pub fn validate_timestamps(&self, data: &DataFrame) -> Result<TimestampReport> {
        // Look for timestamp column (common names)
        let timestamp_col = data
            .get_columns()
            .iter()
            .find(|col| {
                let name = col.name().to_lowercase();
                name.contains("timestamp") || name.contains("time") || name.contains("date")
            })
            .ok_or_else(|| anyhow!("No timestamp column found"))?;

        let total_rows = data.height();
        
        // Handle different timestamp formats
        let timestamps: Vec<Option<i64>> = match timestamp_col.dtype() {
            DataType::Datetime(_, _) => {
                timestamp_col
                    .datetime()?
                    .as_datetime_iter()
                    .map(|opt| opt.map(|dt| dt.and_utc().timestamp()))
                    .collect()
            }
            DataType::Int64 => {
                // Already Unix timestamps in seconds
                timestamp_col
                    .i64()?
                    .into_iter()
                    .collect()
            }
            _ => return Err(anyhow!("Timestamp column must be datetime or int64")),
        };

        let mut gaps_found = 0;
        let mut duplicate_timestamps = 0;
        let mut actual_intervals = Vec::new();
        let mut sequential = true;

        // Check for duplicates and calculate intervals
        for i in 1..timestamps.len() {
            if let (Some(prev), Some(curr)) = (timestamps[i - 1], timestamps[i]) {
                let interval = curr - prev;
                
                if interval == 0 {
                    duplicate_timestamps += 1;
                } else if interval < 0 {
                    sequential = false;
                } else {
                    actual_intervals.push(interval);
                    
                    // Check for gaps if expected interval is configured
                    if let Some(expected) = self.config.expected_interval_seconds {
                        // Allow some tolerance - consider it a gap only if interval is significantly larger
                        if interval > expected * 3 {
                            gaps_found += 1;
                        }
                    }
                }
            }
        }

        let status = if !sequential || (self.config.require_sequential_timestamps && gaps_found > 0) {
            ValidationStatus::Failed
        } else if gaps_found > 0 || duplicate_timestamps > 0 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Passed
        };

        Ok(TimestampReport {
            total_rows,
            sequential,
            gaps_found,
            duplicate_timestamps,
            expected_interval_seconds: self.config.expected_interval_seconds,
            actual_intervals,
            status,
        })
    }

    /// Check for and optionally remove duplicate rows
    pub fn check_duplicates(&self, data: &DataFrame) -> Result<DuplicateReport> {
        let total_rows = data.height();
        
        // Count duplicates by comparing with deduplicated version
        let deduplicated = data.unique::<Vec<String>, String>(None, UniqueKeepStrategy::First, None)?;
        let unique_rows = deduplicated.height();
        let duplicate_rows = total_rows - unique_rows;
        
        let duplicate_percentage = if total_rows > 0 {
            (duplicate_rows as f64 / total_rows as f64) * 100.0
        } else {
            0.0
        };

        let status = if duplicate_rows > 0 {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Passed
        };

        Ok(DuplicateReport {
            total_rows,
            duplicate_rows,
            duplicate_percentage,
            removed: false, // This will be updated if removal is performed
            status,
        })
    }

    /// Remove duplicates from DataFrame
    pub fn remove_duplicates(&self, data: DataFrame) -> Result<(DataFrame, DuplicateReport)> {
        let mut duplicate_report = self.check_duplicates(&data)?;
        
        if duplicate_report.duplicate_rows > 0 && self.config.remove_duplicates {
            let cleaned_data = data.unique::<Vec<String>, String>(None, UniqueKeepStrategy::First, None)?;
            duplicate_report.removed = true;
            Ok((cleaned_data, duplicate_report))
        } else {
            Ok((data, duplicate_report))
        }
    }

    // Private helper methods

    fn detect_outliers_iqr(&self, column: &Column) -> Result<OutlierStats> {
        let values: Vec<f64> = column
            .cast(&DataType::Float64)?
            .f64()?
            .into_no_null_iter()
            .collect();

        if values.is_empty() {
            return Ok(OutlierStats {
                count: 0,
                percentage: 0.0,
                threshold_lower: None,
                threshold_upper: None,
            });
        }

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = sorted_values.len() / 4;
        let q3_idx = 3 * sorted_values.len() / 4;
        
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;
        
        let threshold_multiplier = self.config.outlier_threshold;
        let lower_bound = q1 - threshold_multiplier * iqr;
        let upper_bound = q3 + threshold_multiplier * iqr;

        let outlier_count = values
            .iter()
            .filter(|&&val| val < lower_bound || val > upper_bound)
            .count();

        let percentage = (outlier_count as f64 / values.len() as f64) * 100.0;

        Ok(OutlierStats {
            count: outlier_count,
            percentage,
            threshold_lower: Some(lower_bound),
            threshold_upper: Some(upper_bound),
        })
    }

    fn detect_outliers_zscore(&self, column: &Column) -> Result<OutlierStats> {
        let values: Vec<f64> = column
            .cast(&DataType::Float64)?
            .f64()?
            .into_no_null_iter()
            .collect();

        if values.is_empty() {
            return Ok(OutlierStats {
                count: 0,
                percentage: 0.0,
                threshold_lower: None,
                threshold_upper: None,
            });
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|val| (val - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let threshold = self.config.outlier_threshold;
        let outlier_count = values
            .iter()
            .filter(|&&val| ((val - mean) / std_dev).abs() > threshold)
            .count();

        let percentage = (outlier_count as f64 / values.len() as f64) * 100.0;

        Ok(OutlierStats {
            count: outlier_count,
            percentage,
            threshold_lower: Some(mean - threshold * std_dev),
            threshold_upper: Some(mean + threshold * std_dev),
        })
    }

    pub fn calculate_statistics(&self, data: &DataFrame) -> Result<DataStatistics> {
        let row_count = data.height();
        let column_count = data.width();
        
        let numeric_columns: Vec<String> = data
            .get_columns()
            .iter()
            .filter(|col| col.dtype().is_numeric())
            .map(|col| col.name().to_string())
            .collect();

        let timestamp_columns: Vec<String> = data
            .get_columns()
            .iter()
            .filter(|col| matches!(col.dtype(), DataType::Datetime(_, _)))
            .map(|col| col.name().to_string())
            .collect();

        // Estimate memory usage (rough calculation)
        let memory_usage_bytes = data
            .get_columns()
            .iter()
            .map(|col| {
                let base_size = match col.dtype() {
                    DataType::Int8 => 1,
                    DataType::Int16 => 2,
                    DataType::Int32 => 4,
                    DataType::Int64 => 8,
                    DataType::UInt8 => 1,
                    DataType::UInt16 => 2,
                    DataType::UInt32 => 4,
                    DataType::UInt64 => 8,
                    DataType::Float32 => 4,
                    DataType::Float64 => 8,
                    DataType::Boolean => 1,
                    DataType::String => 16, // Rough estimate for string pointers
                    DataType::Datetime(_, _) => 8,
                    _ => 8, // Default estimate
                };
                col.len() * base_size
            })
            .sum();

        Ok(DataStatistics {
            row_count,
            column_count,
            numeric_columns,
            timestamp_columns,
            memory_usage_bytes,
        })
    }

    fn determine_overall_status(
        &self,
        missing: &MissingValueReport,
        outliers: &OutlierReport,
        timestamps: &TimestampReport,
        duplicates: &DuplicateReport,
    ) -> ValidationStatus {
        let statuses = [
            &missing.status,
            &outliers.status,
            &timestamps.status,
            &duplicates.status,
        ];

        if statuses.iter().any(|s| matches!(s, ValidationStatus::Failed)) {
            ValidationStatus::Failed
        } else if statuses.iter().any(|s| matches!(s, ValidationStatus::Warning)) {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Passed
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub overall_status: ValidationStatus,
    pub missing_values: MissingValueReport,
    pub outliers: OutlierReport,
    pub timestamps: TimestampReport,
    pub duplicates: DuplicateReport,
    pub statistics: DataStatistics,
}

impl ValidationResult {
    /// Get the overall validation status
    pub fn overall_status(&self) -> &ValidationStatus {
        &self.overall_status
    }

    /// Get a summary string of the validation results
    pub fn summary(&self) -> String {
        format!(
            "Validation Status: {:?}, Missing: {:.1}%, Outliers: {}, Timestamp Issues: {}, Duplicates: {}",
            self.overall_status,
            self.missing_values.missing_percentage,
            self.outliers.total_outliers,
            self.timestamps.gaps_found + self.timestamps.duplicate_timestamps,
            self.duplicates.duplicate_rows
        )
    }

    /// Count the number of warnings
    pub fn warning_count(&self) -> usize {
        let mut count = 0;
        if matches!(self.missing_values.status, ValidationStatus::Warning) {
            count += 1;
        }
        if matches!(self.outliers.status, ValidationStatus::Warning) {
            count += 1;
        }
        if matches!(self.timestamps.status, ValidationStatus::Warning) {
            count += 1;
        }
        if matches!(self.duplicates.status, ValidationStatus::Warning) {
            count += 1;
        }
        count
    }

    /// Count the number of errors
    pub fn error_count(&self) -> usize {
        let mut count = 0;
        if matches!(self.missing_values.status, ValidationStatus::Failed) {
            count += 1;
        }
        if matches!(self.outliers.status, ValidationStatus::Failed) {
            count += 1;
        }
        if matches!(self.timestamps.status, ValidationStatus::Failed) {
            count += 1;
        }
        if matches!(self.duplicates.status, ValidationStatus::Failed) {
            count += 1;
        }
        count
    }
}