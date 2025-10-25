//! Feature computation audit logging
//! 
//! This module provides specialized audit logging for feature computation events,
//! including input/output checksums, timing measurements, data quality issues,
//! and HMM weight retrieval events.

use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use tracing::{debug, warn};

use crate::FusionWeights;
use super::Result;
use super::audit::{FeatureComputationEvent, HmmWeightEvent, generate_correlation_id};
use super::audit_logger::AuditLogger;

/// Configuration for feature audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureAuditConfig {
    /// Whether to enable feature computation audit logging
    pub enable_feature_audit: bool,
    
    /// Whether to enable HMM weight audit logging
    pub enable_hmm_audit: bool,
    
    /// Whether to calculate input/output checksums (can be expensive)
    pub enable_checksums: bool,
    
    /// Whether to log data quality issues
    pub enable_quality_logging: bool,
    
    /// Minimum computation time to log (milliseconds, default: 10ms)
    pub min_computation_time_ms: u64,
    
    /// Maximum number of quality issues to log per event (default: 10)
    pub max_quality_issues_per_event: usize,
    
    /// Whether to include detailed timing breakdowns
    pub enable_detailed_timing: bool,
}

impl Default for FeatureAuditConfig {
    fn default() -> Self {
        Self {
            enable_feature_audit: true,
            enable_hmm_audit: true,
            enable_checksums: true,
            enable_quality_logging: true,
            min_computation_time_ms: 10,
            max_quality_issues_per_event: 10,
            enable_detailed_timing: false,
        }
    }
}

/// Data quality issue information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityIssue {
    /// Type of quality issue (e.g., "missing_data", "outlier", "invalid_value")
    pub issue_type: String,
    
    /// Field or feature name affected
    pub field_name: String,
    
    /// Description of the issue
    pub description: String,
    
    /// Severity level (e.g., "warning", "error", "critical")
    pub severity: String,
    
    /// Value that caused the issue (if applicable)
    pub problematic_value: Option<String>,
    
    /// Expected value or range
    pub expected_value: Option<String>,
    
    /// Additional context
    pub context: HashMap<String, String>,
}

impl DataQualityIssue {
    /// Create a new data quality issue
    pub fn new(
        issue_type: String,
        field_name: String,
        description: String,
        severity: String,
    ) -> Self {
        Self {
            issue_type,
            field_name,
            description,
            severity,
            problematic_value: None,
            expected_value: None,
            context: HashMap::new(),
        }
    }
    
    /// Add problematic value information
    pub fn with_problematic_value(mut self, value: String) -> Self {
        self.problematic_value = Some(value);
        self
    }
    
    /// Add expected value information
    pub fn with_expected_value(mut self, value: String) -> Self {
        self.expected_value = Some(value);
        self
    }
    
    /// Add context information
    pub fn with_context(mut self, key: String, value: String) -> Self {
        self.context.insert(key, value);
        self
    }
}

/// Feature computation timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureTimingInfo {
    /// Total computation time in milliseconds
    pub total_time_ms: u64,
    
    /// Data loading time in milliseconds
    pub data_loading_time_ms: Option<u64>,
    
    /// Feature calculation time in milliseconds
    pub calculation_time_ms: Option<u64>,
    
    /// Validation time in milliseconds
    pub validation_time_ms: Option<u64>,
    
    /// Serialization time in milliseconds
    pub serialization_time_ms: Option<u64>,
}

/// Feature audit logger for tracking feature computation events
pub struct FeatureAuditor {
    config: FeatureAuditConfig,
    audit_logger: Option<AuditLogger>,
}

impl FeatureAuditor {
    /// Create a new feature auditor
    pub fn new(config: FeatureAuditConfig, audit_logger: Option<AuditLogger>) -> Self {
        Self {
            config,
            audit_logger,
        }
    }
    
    /// Start timing a feature computation operation
    pub fn start_timing(&self) -> FeatureComputationTimer {
        FeatureComputationTimer::new()
    }
    
    /// Log a feature computation event
    pub async fn log_feature_computation(
        &self,
        correlation_id: String,
        symbol: String,
        feature_names: Vec<String>,
        timing_info: FeatureTimingInfo,
        input_data: Option<&[u8]>,
        output_data: Option<&[u8]>,
        quality_issues: Vec<DataQualityIssue>,
        validation_passed: bool,
    ) -> Result<()> {
        if !self.config.enable_feature_audit {
            return Ok(());
        }
        
        // Skip logging if computation time is below threshold
        if timing_info.total_time_ms < self.config.min_computation_time_ms {
            return Ok(());
        }
        
        // Calculate checksums if enabled
        let input_checksum = if self.config.enable_checksums && input_data.is_some() {
            calculate_checksum(input_data.unwrap())
        } else {
            "disabled".to_string()
        };
        
        let output_checksum = if self.config.enable_checksums && output_data.is_some() {
            calculate_checksum(output_data.unwrap())
        } else {
            "disabled".to_string()
        };
        
        // Limit quality issues if configured
        let limited_quality_issues = if self.config.enable_quality_logging {
            quality_issues
                .into_iter()
                .take(self.config.max_quality_issues_per_event)
                .map(|issue| issue.description)
                .collect()
        } else {
            Vec::new()
        };
        
        // Create feature computation event
        let mut event = FeatureComputationEvent::new(
            correlation_id.clone(),
            symbol.clone(),
            feature_names.clone(),
            timing_info.total_time_ms,
            input_checksum,
            output_checksum,
            validation_passed,
        );
        
        // Add quality issues
        for issue in limited_quality_issues {
            event = event.with_quality_issue(issue);
        }
        
        // Add timing metadata if detailed timing is enabled
        if self.config.enable_detailed_timing {
            if let Some(data_loading_time) = timing_info.data_loading_time_ms {
                event = event.with_metadata("data_loading_time_ms".to_string(), data_loading_time.to_string());
            }
            if let Some(calculation_time) = timing_info.calculation_time_ms {
                event = event.with_metadata("calculation_time_ms".to_string(), calculation_time.to_string());
            }
            if let Some(validation_time) = timing_info.validation_time_ms {
                event = event.with_metadata("validation_time_ms".to_string(), validation_time.to_string());
            }
            if let Some(serialization_time) = timing_info.serialization_time_ms {
                event = event.with_metadata("serialization_time_ms".to_string(), serialization_time.to_string());
            }
        }
        
        // Log the event
        if let Some(ref logger) = self.audit_logger {
            logger.log_feature_computation(&event).await?;
            
            debug!(
                correlation_id = %correlation_id,
                symbol = %symbol,
                feature_count = feature_names.len(),
                computation_time_ms = timing_info.total_time_ms,
                validation_passed = validation_passed,
                quality_issues = event.quality_issues.len(),
                "Feature computation event logged"
            );
        }
        
        Ok(())
    }
    
    /// Log an HMM weight retrieval event
    pub async fn log_hmm_weight_retrieval(
        &self,
        correlation_id: String,
        symbol: String,
        retrieval_start: Instant,
        result: Result<(Option<Vec<f32>>, FusionWeights, bool, Option<bool>)>,
    ) -> Result<()> {
        if !self.config.enable_hmm_audit {
            return Ok(());
        }
        
        let retrieval_latency_ms = retrieval_start.elapsed().as_millis() as u64;
        
        let event = match result {
            Ok((state_probabilities, fusion_weights, fallback_used, cache_hit)) => {
                HmmWeightEvent::success(
                    correlation_id.clone(),
                    symbol.clone(),
                    state_probabilities,
                    fusion_weights,
                    retrieval_latency_ms,
                    fallback_used,
                    cache_hit,
                )
            }
            Err(ref error) => {
                HmmWeightEvent::failure(
                    correlation_id.clone(),
                    symbol.clone(),
                    retrieval_latency_ms,
                    error.to_string(),
                    true, // Assume fallback was used on error
                )
            }
        };
        
        // Log the event
        if let Some(ref logger) = self.audit_logger {
            logger.log_hmm_weight_event(&event).await?;
            
            debug!(
                correlation_id = %correlation_id,
                symbol = %symbol,
                retrieval_latency_ms = retrieval_latency_ms,
                success = event.success,
                fallback_used = event.fallback_used,
                "HMM weight retrieval event logged"
            );
        }
        
        Ok(())
    }
    
    /// Log a data quality issue
    pub async fn log_data_quality_issue(
        &self,
        correlation_id: String,
        symbol: String,
        issue: DataQualityIssue,
    ) -> Result<()> {
        if !self.config.enable_quality_logging {
            return Ok(());
        }
        
        // Create a feature computation event for the quality issue
        let event = FeatureComputationEvent::new(
            correlation_id.clone(),
            symbol.clone(),
            vec!["quality_check".to_string()],
            0, // No computation time for quality issues
            "n/a".to_string(),
            "n/a".to_string(),
            false, // Quality issue means validation failed
        )
        .with_quality_issue(format!(
            "{}: {} ({})",
            issue.issue_type,
            issue.description,
            issue.severity
        ))
        .with_metadata("issue_type".to_string(), issue.issue_type.clone())
        .with_metadata("field_name".to_string(), issue.field_name.clone())
        .with_metadata("severity".to_string(), issue.severity.clone());
        
        // Log the event
        if let Some(ref logger) = self.audit_logger {
            logger.log_feature_computation(&event).await?;
            
            warn!(
                correlation_id = %correlation_id,
                symbol = %symbol,
                issue_type = %issue.issue_type,
                field_name = %issue.field_name,
                severity = %issue.severity,
                description = %issue.description,
                "Data quality issue logged"
            );
        }
        
        Ok(())
    }
    
    /// Create a correlation ID for a new feature computation session
    pub fn create_correlation_id(&self) -> String {
        generate_correlation_id()
    }
    
    /// Check if feature audit logging is enabled
    pub fn is_feature_audit_enabled(&self) -> bool {
        self.config.enable_feature_audit
    }
    
    /// Check if HMM audit logging is enabled
    pub fn is_hmm_audit_enabled(&self) -> bool {
        self.config.enable_hmm_audit
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &FeatureAuditConfig {
        &self.config
    }
}

/// Timer for measuring feature computation performance
pub struct FeatureComputationTimer {
    start_time: Instant,
    data_loading_start: Option<Instant>,
    calculation_start: Option<Instant>,
    validation_start: Option<Instant>,
    serialization_start: Option<Instant>,
    data_loading_time: Option<u64>,
    calculation_time: Option<u64>,
    validation_time: Option<u64>,
    serialization_time: Option<u64>,
}

impl FeatureComputationTimer {
    /// Create a new timer
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            data_loading_start: None,
            calculation_start: None,
            validation_start: None,
            serialization_start: None,
            data_loading_time: None,
            calculation_time: None,
            validation_time: None,
            serialization_time: None,
        }
    }
    
    /// Start timing data loading phase
    pub fn start_data_loading(&mut self) {
        self.data_loading_start = Some(Instant::now());
    }
    
    /// End timing data loading phase
    pub fn end_data_loading(&mut self) {
        if let Some(start) = self.data_loading_start.take() {
            self.data_loading_time = Some(start.elapsed().as_millis() as u64);
        }
    }
    
    /// Start timing calculation phase
    pub fn start_calculation(&mut self) {
        self.calculation_start = Some(Instant::now());
    }
    
    /// End timing calculation phase
    pub fn end_calculation(&mut self) {
        if let Some(start) = self.calculation_start.take() {
            self.calculation_time = Some(start.elapsed().as_millis() as u64);
        }
    }
    
    /// Start timing validation phase
    pub fn start_validation(&mut self) {
        self.validation_start = Some(Instant::now());
    }
    
    /// End timing validation phase
    pub fn end_validation(&mut self) {
        if let Some(start) = self.validation_start.take() {
            self.validation_time = Some(start.elapsed().as_millis() as u64);
        }
    }
    
    /// Start timing serialization phase
    pub fn start_serialization(&mut self) {
        self.serialization_start = Some(Instant::now());
    }
    
    /// End timing serialization phase
    pub fn end_serialization(&mut self) {
        if let Some(start) = self.serialization_start.take() {
            self.serialization_time = Some(start.elapsed().as_millis() as u64);
        }
    }
    
    /// Get the timing information
    pub fn get_timing_info(&self) -> FeatureTimingInfo {
        FeatureTimingInfo {
            total_time_ms: self.start_time.elapsed().as_millis() as u64,
            data_loading_time_ms: self.data_loading_time,
            calculation_time_ms: self.calculation_time,
            validation_time_ms: self.validation_time,
            serialization_time_ms: self.serialization_time,
        }
    }
}

impl Default for FeatureComputationTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate SHA256 checksum of data
fn calculate_checksum(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

/// Helper function to create a feature auditor with default configuration
pub fn create_feature_auditor(audit_logger: Option<AuditLogger>) -> FeatureAuditor {
    FeatureAuditor::new(FeatureAuditConfig::default(), audit_logger)
}

/// Helper function to create a feature auditor with custom configuration
pub fn create_feature_auditor_with_config(
    config: FeatureAuditConfig,
    audit_logger: Option<AuditLogger>,
) -> FeatureAuditor {
    FeatureAuditor::new(config, audit_logger)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::emission::audit_logger::{AuditLogger, AuditConfig};
    
    async fn create_test_audit_logger() -> AuditLogger {
        let temp_dir = TempDir::new().unwrap();
        let config = AuditConfig {
            log_directory: temp_dir.path().to_path_buf(),
            log_filename: "test_feature_audit".to_string(),
            max_file_size_bytes: 1024 * 1024,
            max_file_age_seconds: 3600,
            max_files_to_keep: 5,
            compress_rotated_files: false,
            enable_integrity_verification: true,
            write_buffer_size: 1024,
            flush_after_write: true,
            file_permissions: 0o644,
            create_directories: true,
            s3_config: None,
        };
        
        AuditLogger::new(config).await.unwrap()
    }
    
    #[test]
    fn test_feature_auditor_creation() {
        let config = FeatureAuditConfig::default();
        let auditor = FeatureAuditor::new(config, None);
        
        assert!(auditor.is_feature_audit_enabled());
        assert!(auditor.is_hmm_audit_enabled());
    }
    
    #[test]
    fn test_data_quality_issue_creation() {
        let issue = DataQualityIssue::new(
            "missing_data".to_string(),
            "price".to_string(),
            "Price data is missing for timestamp".to_string(),
            "warning".to_string(),
        )
        .with_problematic_value("null".to_string())
        .with_expected_value("positive number".to_string())
        .with_context("timestamp".to_string(), "1640995200".to_string());
        
        assert_eq!(issue.issue_type, "missing_data");
        assert_eq!(issue.field_name, "price");
        assert_eq!(issue.severity, "warning");
        assert_eq!(issue.problematic_value, Some("null".to_string()));
        assert_eq!(issue.expected_value, Some("positive number".to_string()));
        assert_eq!(issue.context.get("timestamp"), Some(&"1640995200".to_string()));
    }
    
    #[test]
    fn test_feature_computation_timer() {
        let mut timer = FeatureComputationTimer::new();
        
        timer.start_data_loading();
        std::thread::sleep(std::time::Duration::from_millis(10));
        timer.end_data_loading();
        
        timer.start_calculation();
        std::thread::sleep(std::time::Duration::from_millis(20));
        timer.end_calculation();
        
        let timing_info = timer.get_timing_info();
        
        assert!(timing_info.total_time_ms >= 30);
        assert!(timing_info.data_loading_time_ms.unwrap() >= 10);
        assert!(timing_info.calculation_time_ms.unwrap() >= 20);
        assert!(timing_info.validation_time_ms.is_none());
        assert!(timing_info.serialization_time_ms.is_none());
    }
    
    #[tokio::test]
    async fn test_feature_computation_logging() {
        let audit_logger = create_test_audit_logger().await;
        let config = FeatureAuditConfig::default();
        let auditor = FeatureAuditor::new(config, Some(audit_logger));
        
        let correlation_id = auditor.create_correlation_id();
        let timing_info = FeatureTimingInfo {
            total_time_ms: 50,
            data_loading_time_ms: Some(10),
            calculation_time_ms: Some(30),
            validation_time_ms: Some(5),
            serialization_time_ms: Some(5),
        };
        
        let input_data = b"test input data";
        let output_data = b"test output data";
        let quality_issues = vec![
            DataQualityIssue::new(
                "outlier".to_string(),
                "volume".to_string(),
                "Volume spike detected".to_string(),
                "warning".to_string(),
            )
        ];
        
        let result = auditor.log_feature_computation(
            correlation_id,
            "BTCUSDT".to_string(),
            vec!["rsi".to_string(), "ma".to_string()],
            timing_info,
            Some(input_data),
            Some(output_data),
            quality_issues,
            true,
        ).await;
        
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_hmm_weight_logging() {
        let audit_logger = create_test_audit_logger().await;
        let config = FeatureAuditConfig::default();
        let auditor = FeatureAuditor::new(config, Some(audit_logger));
        
        let correlation_id = auditor.create_correlation_id();
        let retrieval_start = Instant::now();
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let result = Ok((Some(vec![0.7, 0.3]), weights, false, Some(true)));
        
        let log_result = auditor.log_hmm_weight_retrieval(
            correlation_id,
            "BTCUSDT".to_string(),
            retrieval_start,
            result,
        ).await;
        
        assert!(log_result.is_ok());
    }
    
    #[tokio::test]
    async fn test_data_quality_issue_logging() {
        let audit_logger = create_test_audit_logger().await;
        let config = FeatureAuditConfig::default();
        let auditor = FeatureAuditor::new(config, Some(audit_logger));
        
        let correlation_id = auditor.create_correlation_id();
        let issue = DataQualityIssue::new(
            "invalid_value".to_string(),
            "price".to_string(),
            "Negative price detected".to_string(),
            "error".to_string(),
        );
        
        let result = auditor.log_data_quality_issue(
            correlation_id,
            "BTCUSDT".to_string(),
            issue,
        ).await;
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_checksum_calculation() {
        let data = b"test data for checksum";
        let checksum = calculate_checksum(data);
        
        assert!(!checksum.is_empty());
        assert_eq!(checksum.len(), 64); // SHA256 hex string length
        
        // Same data should produce same checksum
        let checksum2 = calculate_checksum(data);
        assert_eq!(checksum, checksum2);
        
        // Different data should produce different checksum
        let different_data = b"different test data";
        let different_checksum = calculate_checksum(different_data);
        assert_ne!(checksum, different_checksum);
    }
    
    #[test]
    fn test_feature_audit_config() {
        let config = FeatureAuditConfig {
            enable_feature_audit: false,
            enable_hmm_audit: true,
            enable_checksums: false,
            enable_quality_logging: true,
            min_computation_time_ms: 50,
            max_quality_issues_per_event: 5,
            enable_detailed_timing: true,
        };
        
        let auditor = FeatureAuditor::new(config, None);
        
        assert!(!auditor.is_feature_audit_enabled());
        assert!(auditor.is_hmm_audit_enabled());
        assert_eq!(auditor.config().min_computation_time_ms, 50);
        assert_eq!(auditor.config().max_quality_issues_per_event, 5);
        assert!(auditor.config().enable_detailed_timing);
    }
    
    #[test]
    fn test_helper_functions() {
        let auditor1 = create_feature_auditor(None);
        assert!(auditor1.is_feature_audit_enabled());
        
        let custom_config = FeatureAuditConfig {
            enable_feature_audit: false,
            ..Default::default()
        };
        let auditor2 = create_feature_auditor_with_config(custom_config, None);
        assert!(!auditor2.is_feature_audit_enabled());
    }
}