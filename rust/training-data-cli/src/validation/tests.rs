#[cfg(test)]
mod tests {
    use crate::validation::*;
    use polars::prelude::*;

    fn create_test_dataframe() -> DataFrame {
        let timestamps = vec![
            1640995200i64, // 2022-01-01 00:00:00
            1640995500i64, // 2022-01-01 00:05:00 (+300 seconds)
            1640995800i64, // 2022-01-01 00:10:00 (+300 seconds)
            1640996100i64, // 2022-01-01 00:15:00 (+300 seconds)
            1640996400i64, // 2022-01-01 00:20:00 (+300 seconds)
        ];

        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0];
        let volumes = vec![1000.0, 1100.0, 900.0, 1200.0, 800.0];

        df! {
            "timestamp" => timestamps,
            "close" => prices,
            "volume" => volumes,
        }
        .unwrap()
    }

    fn create_test_dataframe_with_issues() -> DataFrame {
        let timestamps = vec![
            1640995200i64, // 2022-01-01 00:00:00
            1640995500i64, // 2022-01-01 00:05:00
            1640995500i64, // Duplicate timestamp
            1640997000i64, // Large gap (25 minutes later, should trigger gap detection)
            1640997300i64, // 2022-01-01 00:35:00
        ];

        let prices = vec![
            Some(100.0),
            Some(101.0),
            None, // Missing value
            Some(1000.0), // Outlier
            Some(98.0),
        ];

        let volumes = vec![1000.0, 1100.0, 1100.0, 1200.0, 800.0]; // Duplicate row data

        df! {
            "timestamp" => timestamps,
            "close" => prices,
            "volume" => volumes,
        }
        .unwrap()
    }

    #[test]
    fn test_validator_creation() {
        let validator = DataValidator::with_default_config();
        assert_eq!(validator.config.outlier_method, OutlierMethod::IQR);
        assert_eq!(validator.config.outlier_threshold, 3.0);
        assert_eq!(validator.config.max_missing_percentage, 5.0);
        assert!(validator.config.require_sequential_timestamps);
        assert_eq!(validator.config.expected_interval_seconds, Some(300));
        assert!(validator.config.remove_duplicates);
    }

    #[test]
    fn test_custom_validator_config() {
        let config = ValidationConfig {
            outlier_method: OutlierMethod::ZScore,
            outlier_threshold: 2.5,
            max_missing_percentage: 10.0,
            require_sequential_timestamps: false,
            expected_interval_seconds: Some(60),
            remove_duplicates: false,
        };

        let validator = DataValidator::new(config.clone());
        assert_eq!(validator.config.outlier_method, OutlierMethod::ZScore);
        assert_eq!(validator.config.outlier_threshold, 2.5);
        assert_eq!(validator.config.max_missing_percentage, 10.0);
        assert!(!validator.config.require_sequential_timestamps);
        assert_eq!(validator.config.expected_interval_seconds, Some(60));
        assert!(!validator.config.remove_duplicates);
    }

    #[test]
    fn test_missing_values_detection_clean_data() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        
        let report = validator.check_missing_values(&data).unwrap();
        
        assert_eq!(report.total_rows, 5);
        assert!(report.columns_with_missing.is_empty());
        assert_eq!(report.missing_percentage, 0.0);
        assert!(matches!(report.status, ValidationStatus::Passed));
    }

    #[test]
    fn test_missing_values_detection_with_nulls() {
        let mut config = ValidationConfig::default();
        config.max_missing_percentage = 10.0; // Allow up to 10% missing values
        let validator = DataValidator::new(config);
        let data = create_test_dataframe_with_issues();
        
        let report = validator.check_missing_values(&data).unwrap();
        
        assert_eq!(report.total_rows, 5);
        assert_eq!(report.columns_with_missing.get("close"), Some(&1));
        assert!(report.missing_percentage > 0.0);
        assert!(matches!(report.status, ValidationStatus::Warning));
    }

    #[test]
    fn test_missing_values_exceeds_threshold() {
        let mut config = ValidationConfig::default();
        config.max_missing_percentage = 1.0; // Very strict threshold
        let validator = DataValidator::new(config);
        let data = create_test_dataframe_with_issues();
        
        let report = validator.check_missing_values(&data).unwrap();
        
        assert!(matches!(report.status, ValidationStatus::Failed));
    }

    #[test]
    fn test_outlier_detection_iqr_clean_data() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        
        let report = validator.detect_outliers(&data).unwrap();
        
        assert_eq!(report.total_outliers, 0);
        assert!(matches!(report.method_used, OutlierMethod::IQR));
        assert!(matches!(report.status, ValidationStatus::Passed));
    }

    #[test]
    fn test_outlier_detection_iqr_with_outliers() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe_with_issues();
        
        let report = validator.detect_outliers(&data).unwrap();
        
        assert!(report.total_outliers > 0);
        assert!(matches!(report.method_used, OutlierMethod::IQR));
        assert!(matches!(report.status, ValidationStatus::Warning));
        assert!(report.columns_with_outliers.contains_key("close"));
    }

    #[test]
    fn test_outlier_detection_zscore() {
        let mut config = ValidationConfig::default();
        config.outlier_method = OutlierMethod::ZScore;
        config.outlier_threshold = 1.5; // Lower threshold to catch the outlier
        let validator = DataValidator::new(config);
        
        // Create data with clear outlier
        let data = df! {
            "values" => vec![1.0, 2.0, 3.0, 4.0, 100.0], // 100.0 is clear outlier
        }
        .unwrap();
        
        let report = validator.detect_outliers(&data).unwrap();
        
        assert!(report.total_outliers > 0);
        assert!(matches!(report.method_used, OutlierMethod::ZScore));
        assert!(matches!(report.status, ValidationStatus::Warning));
    }

    #[test]
    fn test_timestamp_validation_sequential() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        
        let report = validator.validate_timestamps(&data).unwrap();
        
        assert_eq!(report.total_rows, 5);
        assert!(report.sequential);
        assert_eq!(report.gaps_found, 0);
        assert_eq!(report.duplicate_timestamps, 0);
        assert!(matches!(report.status, ValidationStatus::Passed));
    }

    #[test]
    fn test_timestamp_validation_with_issues() {
        let mut config = ValidationConfig::default();
        config.require_sequential_timestamps = false; // Allow gaps but still report them
        let validator = DataValidator::new(config);
        let data = create_test_dataframe_with_issues();
        
        let report = validator.validate_timestamps(&data).unwrap();
        
        assert_eq!(report.total_rows, 5);
        assert!(report.sequential); // Still sequential despite duplicates
        assert!(report.gaps_found > 0); // Should detect the gap
        assert!(report.duplicate_timestamps > 0); // Should detect duplicate
        assert!(matches!(report.status, ValidationStatus::Warning));
    }

    #[test]
    fn test_timestamp_validation_non_sequential() {
        let validator = DataValidator::with_default_config();
        
        // Create data with non-sequential timestamps
        let data = df! {
            "timestamp" => vec![1640995800i64, 1640995500i64, 1640996100i64], // Out of order
            "close" => vec![100.0, 101.0, 102.0],
        }
        .unwrap();
        
        let report = validator.validate_timestamps(&data).unwrap();
        
        assert!(!report.sequential);
        assert!(matches!(report.status, ValidationStatus::Failed));
    }

    #[test]
    fn test_duplicate_detection_clean_data() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        
        let report = validator.check_duplicates(&data).unwrap();
        
        assert_eq!(report.total_rows, 5);
        assert_eq!(report.duplicate_rows, 0);
        assert_eq!(report.duplicate_percentage, 0.0);
        assert!(!report.removed);
        assert!(matches!(report.status, ValidationStatus::Passed));
    }

    #[test]
    fn test_duplicate_detection_with_duplicates() {
        let validator = DataValidator::with_default_config();
        
        // Create data with exact duplicate rows
        let data = df! {
            "timestamp" => vec![1640995200i64, 1640995500i64, 1640995500i64],
            "close" => vec![100.0, 101.0, 101.0],
            "volume" => vec![1000.0, 1100.0, 1100.0],
        }
        .unwrap();
        
        let report = validator.check_duplicates(&data).unwrap();
        
        assert_eq!(report.total_rows, 3);
        assert_eq!(report.duplicate_rows, 1);
        assert!(report.duplicate_percentage > 0.0);
        assert!(!report.removed);
        assert!(matches!(report.status, ValidationStatus::Warning));
    }

    #[test]
    fn test_remove_duplicates() {
        let validator = DataValidator::with_default_config();
        
        // Create data with exact duplicate rows
        let data = df! {
            "timestamp" => vec![1640995200i64, 1640995500i64, 1640995500i64],
            "close" => vec![100.0, 101.0, 101.0],
            "volume" => vec![1000.0, 1100.0, 1100.0],
        }
        .unwrap();
        
        let (cleaned_data, report) = validator.remove_duplicates(data).unwrap();
        
        assert_eq!(cleaned_data.height(), 2); // Should have 2 rows after deduplication
        assert!(report.removed);
        assert_eq!(report.duplicate_rows, 1);
    }

    #[test]
    fn test_statistics_calculation() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        
        let stats = validator.calculate_statistics(&data).unwrap();
        
        assert_eq!(stats.row_count, 5);
        assert_eq!(stats.column_count, 3);
        assert_eq!(stats.numeric_columns.len(), 3); // timestamp, close, volume are all numeric
        assert_eq!(stats.timestamp_columns.len(), 0); // No datetime columns in test data
        assert!(stats.memory_usage_bytes > 0);
    }

    #[test]
    fn test_full_validation_clean_data() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        
        let result = validator.validate(&data).unwrap();
        
        assert!(matches!(result.overall_status, ValidationStatus::Passed));
        assert!(matches!(result.missing_values.status, ValidationStatus::Passed));
        assert!(matches!(result.outliers.status, ValidationStatus::Passed));
        assert!(matches!(result.duplicates.status, ValidationStatus::Passed));
        assert_eq!(result.statistics.row_count, 5);
    }

    #[test]
    fn test_full_validation_with_issues() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe_with_issues();
        
        let result = validator.validate(&data).unwrap();
        
        // Should be warning or failed due to various issues
        assert!(matches!(
            result.overall_status,
            ValidationStatus::Warning | ValidationStatus::Failed
        ));
        
        // Check that individual components detected issues
        assert!(matches!(
            result.missing_values.status,
            ValidationStatus::Warning | ValidationStatus::Failed
        ));
        assert!(matches!(
            result.duplicates.status,
            ValidationStatus::Warning | ValidationStatus::Passed
        ));
    }

    #[test]
    fn test_iqr_outlier_calculation() {
        let validator = DataValidator::with_default_config();
        
        // Create dataframe with known outliers
        let data = df! {
            "test_values" => vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0], // 100.0 is clear outlier
        }
        .unwrap();
        
        let report = validator.detect_outliers(&data).unwrap();
        
        assert!(report.total_outliers > 0);
        assert!(matches!(report.method_used, OutlierMethod::IQR));
        assert!(report.columns_with_outliers.contains_key("test_values"));
        
        let stats = &report.columns_with_outliers["test_values"];
        assert!(stats.count > 0);
        assert!(stats.percentage > 0.0);
        assert!(stats.threshold_lower.is_some());
        assert!(stats.threshold_upper.is_some());
    }

    #[test]
    fn test_zscore_outlier_calculation() {
        let mut config = ValidationConfig::default();
        config.outlier_method = OutlierMethod::ZScore;
        config.outlier_threshold = 1.5; // Lower threshold to catch outliers
        let validator = DataValidator::new(config);
        
        // Create dataframe with known outliers
        let data = df! {
            "test_values" => vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0], // 100.0 is clear outlier
        }
        .unwrap();
        
        let report = validator.detect_outliers(&data).unwrap();
        
        assert!(report.total_outliers > 0);
        assert!(matches!(report.method_used, OutlierMethod::ZScore));
        assert!(report.columns_with_outliers.contains_key("test_values"));
        
        let stats = &report.columns_with_outliers["test_values"];
        assert!(stats.count > 0);
        assert!(stats.percentage > 0.0);
        assert!(stats.threshold_lower.is_some());
        assert!(stats.threshold_upper.is_some());
    }

    #[test]
    fn test_empty_data_handling() {
        let validator = DataValidator::with_default_config();
        
        let empty_data = df! {
            "timestamp" => Vec::<i64>::new(),
            "close" => Vec::<f64>::new(),
        }
        .unwrap();
        
        let result = validator.validate(&empty_data).unwrap();
        
        assert_eq!(result.statistics.row_count, 0);
        assert!(matches!(result.overall_status, ValidationStatus::Passed));
    }

    #[test]
    fn test_single_row_data() {
        let validator = DataValidator::with_default_config();
        
        let single_row_data = df! {
            "timestamp" => vec![1640995200i64],
            "close" => vec![100.0],
        }
        .unwrap();
        
        let result = validator.validate(&single_row_data).unwrap();
        
        assert_eq!(result.statistics.row_count, 1);
        // Single row should pass most validations
        assert!(matches!(
            result.overall_status,
            ValidationStatus::Passed | ValidationStatus::Warning
        ));
    }

    // ValidationReport tests
    #[test]
    fn test_validation_report_creation() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        let validation_result = validator.validate(&data).unwrap();
        
        let report = ValidationReport::new(validation_result, Some("test_data.parquet".to_string()));
        
        assert!(report.report_id.starts_with("validation_"));
        assert_eq!(report.data_source, Some("test_data.parquet".to_string()));
        assert!(matches!(report.summary.overall_status, ValidationStatus::Passed));
        assert_eq!(report.summary.total_checks, 4); // missing, outliers, timestamps, duplicates
        assert_eq!(report.summary.passed_checks, 4);
        assert_eq!(report.summary.warnings, 0);
        assert_eq!(report.summary.critical_issues, 0);
        assert_eq!(report.summary.total_issues, 0);
    }

    #[test]
    fn test_validation_report_with_issues() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe_with_issues();
        let validation_result = validator.validate(&data).unwrap();
        
        let report = ValidationReport::new(validation_result, None);
        
        assert!(report.data_source.is_none());
        assert!(matches!(
            report.summary.overall_status,
            ValidationStatus::Warning | ValidationStatus::Failed
        ));
        assert!(report.summary.total_issues > 0);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_validation_report_json_serialization() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        let validation_result = validator.validate(&data).unwrap();
        
        let report = ValidationReport::new(validation_result, Some("test.parquet".to_string()));
        
        // Test JSON serialization
        let json_output = report.to_json().unwrap();
        assert!(json_output.contains("validation_"));
        assert!(json_output.contains("test.parquet"));
        
        // Test compact JSON
        let compact_json = report.to_json_compact().unwrap();
        assert!(compact_json.len() < json_output.len()); // Compact should be smaller
        
        // Test deserialization
        let deserialized: ValidationReport = serde_json::from_str(&json_output).unwrap();
        assert_eq!(deserialized.report_id, report.report_id);
        assert_eq!(deserialized.data_source, report.data_source);
    }

    #[test]
    fn test_validation_report_human_readable_format() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        let validation_result = validator.validate(&data).unwrap();
        
        let report = ValidationReport::new(validation_result, Some("test.parquet".to_string()));
        
        let human_readable = report.format_human_readable();
        
        // Check that key sections are present
        assert!(human_readable.contains("VALIDATION REPORT"));
        assert!(human_readable.contains("Overall Status"));
        assert!(human_readable.contains("SUMMARY"));
        assert!(human_readable.contains("DATA OVERVIEW"));
        assert!(human_readable.contains("DETAILED RESULTS"));
        assert!(human_readable.contains("Missing Values"));
        assert!(human_readable.contains("Outliers"));
        assert!(human_readable.contains("Timestamps"));
        assert!(human_readable.contains("Duplicates"));
        assert!(human_readable.contains("test.parquet"));
        
        // Check for status symbols
        assert!(human_readable.contains("✅")); // Should have passed checks
    }

    #[test]
    fn test_validation_report_human_readable_with_issues() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe_with_issues();
        let validation_result = validator.validate(&data).unwrap();
        
        let report = ValidationReport::new(validation_result, None);
        
        let human_readable = report.format_human_readable();
        
        // Should contain warning or error symbols
        assert!(human_readable.contains("⚠️") || human_readable.contains("❌"));
        
        // Should contain recommendations section
        assert!(human_readable.contains("RECOMMENDATIONS"));
        
        // Should show specific issues
        assert!(human_readable.contains("Missing Percentage"));
        assert!(human_readable.contains("Duplicate"));
    }

    #[test]
    fn test_validation_report_recommendations() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe_with_issues();
        let validation_result = validator.validate(&data).unwrap();
        
        let report = ValidationReport::new(validation_result, None);
        
        // Should have recommendations for the issues found
        assert!(!report.recommendations.is_empty());
        
        // Check for specific recommendation types
        let recommendations_text = report.recommendations.join(" ");
        assert!(
            recommendations_text.contains("missing") ||
            recommendations_text.contains("outlier") ||
            recommendations_text.contains("duplicate") ||
            recommendations_text.contains("timestamp")
        );
    }

    #[test]
    fn test_validation_report_display_trait() {
        let validator = DataValidator::with_default_config();
        let data = create_test_dataframe();
        let validation_result = validator.validate(&data).unwrap();
        
        let report = ValidationReport::new(validation_result, None);
        
        // Test Display trait implementation
        let display_output = format!("{}", report);
        let human_readable = report.format_human_readable();
        
        assert_eq!(display_output, human_readable);
    }

    #[test]
    fn test_validation_summary_calculation() {
        let validator = DataValidator::with_default_config();
        
        // Test with clean data
        let clean_data = create_test_dataframe();
        let clean_result = validator.validate(&clean_data).unwrap();
        let clean_report = ValidationReport::new(clean_result, None);
        
        assert_eq!(clean_report.summary.total_checks, 4);
        assert_eq!(clean_report.summary.passed_checks, 4);
        assert_eq!(clean_report.summary.warnings, 0);
        assert_eq!(clean_report.summary.critical_issues, 0);
        assert_eq!(clean_report.summary.total_issues, 0);
        
        // Test with problematic data
        let problem_data = create_test_dataframe_with_issues();
        let problem_result = validator.validate(&problem_data).unwrap();
        let problem_report = ValidationReport::new(problem_result, None);
        
        assert_eq!(problem_report.summary.total_checks, 4);
        assert!(problem_report.summary.total_issues > 0);
        assert!(problem_report.summary.passed_checks < 4);
    }
}