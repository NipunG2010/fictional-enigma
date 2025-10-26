//! Result validation and comparison utilities
//! 
//! Provides comprehensive validation capabilities for comparing test results
//! against expected values, reference data, and performance requirements.

use crate::{
    config::ValidationConfig,
    performance::PerformanceReport,
    reporting::ValidationDetail,
    Result, TestFrameworkError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result validator for comparing test outputs against expected values
pub struct ResultValidator {
    /// Validation configuration
    config: ValidationConfig,
    
    /// Reference data for comparison
    reference_data: Option<ReferenceDataSet>,
}

/// Reference data set for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceDataSet {
    /// Reference feature values
    pub features: HashMap<String, Vec<f64>>,
    
    /// Reference signal values
    pub signals: HashMap<String, Vec<TradingSignal>>,
    
    /// Reference performance metrics
    pub performance_metrics: HashMap<String, f64>,
    
    /// Metadata about the reference data
    pub metadata: ReferenceMetadata,
}

/// Metadata for reference data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceMetadata {
    /// Version of the reference data
    pub version: String,
    
    /// Creation timestamp
    pub created_at: i64,
    
    /// Description of the reference data
    pub description: String,
    
    /// Source of the reference data
    pub source: String,
}

/// Trading signal for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Signal timestamp
    pub timestamp: i64,
    
    /// Signal strength (-1.0 to 1.0)
    pub strength: f64,
    
    /// Signal confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Signal type
    pub signal_type: SignalType,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Type of trading signal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalType {
    /// Long Direction Classifier signal
    LDC,
    
    /// Mean Reversion signal
    MR,
    
    /// Time Series Momentum signal
    TSMOM,
    
    /// Fused signal combining multiple strategies
    Fused,
}

/// Feature values for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Features {
    /// RSI values
    pub rsi: Vec<f64>,
    
    /// Moving average values
    pub moving_averages: HashMap<String, Vec<f64>>,
    
    /// Momentum indicators
    pub momentum: Vec<f64>,
    
    /// Volatility measures
    pub volatility: Vec<f64>,
    
    /// Custom features
    pub custom: HashMap<String, Vec<f64>>,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    
    /// Validation details
    pub details: Vec<ValidationDetail>,
    
    /// Overall validation score (0.0 to 1.0)
    pub score: f64,
    
    /// Validation summary message
    pub summary: String,
}

/// Fallback behavior for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackBehavior {
    /// Whether fallback was triggered
    pub fallback_triggered: bool,
    
    /// Reason for fallback
    pub fallback_reason: String,
    
    /// Fallback strategy used
    pub fallback_strategy: String,
    
    /// Performance during fallback
    pub fallback_performance: HashMap<String, f64>,
}

impl ResultValidator {
    /// Create a new result validator
    pub fn new(config: ValidationConfig) -> Result<Self> {
        Ok(Self {
            config,
            reference_data: None,
        })
    }
    
    /// Create validator with reference data
    pub fn with_reference_data(config: ValidationConfig, reference_data: ReferenceDataSet) -> Result<Self> {
        Ok(Self {
            config,
            reference_data: Some(reference_data),
        })
    }
    
    /// Load reference data from file
    pub fn load_reference_data<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| TestFrameworkError::ValidationError(format!("Failed to read reference data: {}", e)))?;
        
        let reference_data: ReferenceDataSet = serde_json::from_str(&content)
            .map_err(|e| TestFrameworkError::ValidationError(format!("Failed to parse reference data: {}", e)))?;
        
        self.reference_data = Some(reference_data);
        Ok(())
    }
    
    /// Validate computed features against reference data
    pub fn validate_features(&self, computed: &Features, expected: &Features) -> ValidationResult {
        let mut details = Vec::new();
        let mut passed_checks = 0;
        let mut total_checks = 0;
        
        // Validate RSI values
        let rsi_result = self.validate_numeric_array(
            &computed.rsi,
            &expected.rsi,
            "RSI",
            self.config.feature_tolerance,
        );
        details.extend(rsi_result.details);
        if rsi_result.passed {
            passed_checks += 1;
        }
        total_checks += 1;
        
        // Validate moving averages
        for (ma_type, expected_values) in &expected.moving_averages {
            if let Some(computed_values) = computed.moving_averages.get(ma_type) {
                let ma_result = self.validate_numeric_array(
                    computed_values,
                    expected_values,
                    &format!("Moving Average ({})", ma_type),
                    self.config.feature_tolerance,
                );
                details.extend(ma_result.details);
                if ma_result.passed {
                    passed_checks += 1;
                }
            } else {
                details.push(ValidationDetail::failure_with_message(
                    format!("Moving Average ({})", ma_type),
                    format!("Missing computed values for {}", ma_type),
                ));
            }
            total_checks += 1;
        }
        
        // Validate momentum
        let momentum_result = self.validate_numeric_array(
            &computed.momentum,
            &expected.momentum,
            "Momentum",
            self.config.feature_tolerance,
        );
        details.extend(momentum_result.details);
        if momentum_result.passed {
            passed_checks += 1;
        }
        total_checks += 1;
        
        // Validate volatility
        let volatility_result = self.validate_numeric_array(
            &computed.volatility,
            &expected.volatility,
            "Volatility",
            self.config.feature_tolerance,
        );
        details.extend(volatility_result.details);
        if volatility_result.passed {
            passed_checks += 1;
        }
        total_checks += 1;
        
        let score = if total_checks > 0 {
            passed_checks as f64 / total_checks as f64
        } else {
            0.0
        };
        
        let passed = score >= 0.8; // 80% of checks must pass
        let summary = format!(
            "Feature validation: {}/{} checks passed (score: {:.1}%)",
            passed_checks, total_checks, score * 100.0
        );
        
        ValidationResult {
            passed,
            details,
            score,
            summary,
        }
    }
    
    /// Validate generated signals against expected signals
    pub fn validate_signals(&self, generated: &TradingSignal, expected: &TradingSignal) -> ValidationResult {
        let mut details = Vec::new();
        let mut passed_checks = 0;
        let mut total_checks = 0;
        
        // Validate signal strength
        let strength_diff = (generated.strength - expected.strength).abs();
        if strength_diff <= self.config.signal_tolerance {
            details.push(ValidationDetail::success(
                "Signal Strength".to_string(),
                format!("Strength within tolerance: {} vs {} (diff: {:.4})", 
                       generated.strength, expected.strength, strength_diff),
            ));
            passed_checks += 1;
        } else {
            details.push(ValidationDetail::failure(
                "Signal Strength".to_string(),
                expected.strength.to_string(),
                generated.strength.to_string(),
                format!("Strength difference {} exceeds tolerance {}", 
                       strength_diff, self.config.signal_tolerance),
            ));
        }
        total_checks += 1;
        
        // Validate signal confidence
        let confidence_diff = (generated.confidence - expected.confidence).abs();
        if confidence_diff <= self.config.signal_tolerance {
            details.push(ValidationDetail::success(
                "Signal Confidence".to_string(),
                format!("Confidence within tolerance: {} vs {} (diff: {:.4})", 
                       generated.confidence, expected.confidence, confidence_diff),
            ));
            passed_checks += 1;
        } else {
            details.push(ValidationDetail::failure(
                "Signal Confidence".to_string(),
                expected.confidence.to_string(),
                generated.confidence.to_string(),
                format!("Confidence difference {} exceeds tolerance {}", 
                       confidence_diff, self.config.signal_tolerance),
            ));
        }
        total_checks += 1;
        
        // Validate signal type
        if generated.signal_type == expected.signal_type {
            details.push(ValidationDetail::success(
                "Signal Type".to_string(),
                format!("Signal type matches: {:?}", generated.signal_type),
            ));
            passed_checks += 1;
        } else {
            details.push(ValidationDetail::failure(
                "Signal Type".to_string(),
                format!("{:?}", expected.signal_type),
                format!("{:?}", generated.signal_type),
                "Signal type mismatch".to_string(),
            ));
        }
        total_checks += 1;
        
        let score = passed_checks as f64 / total_checks as f64;
        let passed = score >= 0.8;
        let summary = format!(
            "Signal validation: {}/{} checks passed (score: {:.1}%)",
            passed_checks, total_checks, score * 100.0
        );
        
        ValidationResult {
            passed,
            details,
            score,
            summary,
        }
    }
    
    /// Validate performance metrics against requirements
    pub fn validate_performance(&self, metrics: &PerformanceReport) -> ValidationResult {
        let mut details = Vec::new();
        let mut passed_checks = 0;
        let mut total_checks = 0;
        
        // Validate end-to-end latency
        let max_latency = 100.0; // 100ms requirement
        if metrics.end_to_end_latency.mean <= max_latency {
            details.push(ValidationDetail::success(
                "End-to-End Latency".to_string(),
                format!("Mean latency {:.2}ms is within {:.2}ms requirement", 
                       metrics.end_to_end_latency.mean, max_latency),
            ));
            passed_checks += 1;
        } else {
            details.push(ValidationDetail::failure(
                "End-to-End Latency".to_string(),
                format!("<= {:.2}ms", max_latency),
                format!("{:.2}ms", metrics.end_to_end_latency.mean),
                format!("Mean latency exceeds requirement by {:.2}ms", 
                       metrics.end_to_end_latency.mean - max_latency),
            ));
        }
        total_checks += 1;
        
        // Validate P95 latency
        if metrics.end_to_end_latency.p95 <= max_latency * 1.5 {
            details.push(ValidationDetail::success(
                "P95 Latency".to_string(),
                format!("P95 latency {:.2}ms is within {:.2}ms threshold", 
                       metrics.end_to_end_latency.p95, max_latency * 1.5),
            ));
            passed_checks += 1;
        } else {
            details.push(ValidationDetail::failure(
                "P95 Latency".to_string(),
                format!("<= {:.2}ms", max_latency * 1.5),
                format!("{:.2}ms", metrics.end_to_end_latency.p95),
                "P95 latency exceeds threshold".to_string(),
            ));
        }
        total_checks += 1;
        
        // Validate throughput
        let min_throughput = 10.0; // 10 ops/sec requirement
        if metrics.throughput_stats.average_ops_per_second >= min_throughput {
            details.push(ValidationDetail::success(
                "Throughput".to_string(),
                format!("Average throughput {:.2} ops/sec meets {:.2} ops/sec requirement", 
                       metrics.throughput_stats.average_ops_per_second, min_throughput),
            ));
            passed_checks += 1;
        } else {
            details.push(ValidationDetail::failure(
                "Throughput".to_string(),
                format!(">= {:.2} ops/sec", min_throughput),
                format!("{:.2} ops/sec", metrics.throughput_stats.average_ops_per_second),
                "Throughput below requirement".to_string(),
            ));
        }
        total_checks += 1;
        
        // Validate memory usage
        let max_memory = 512.0; // 512MB requirement
        if metrics.memory_usage.peak_memory_mb <= max_memory {
            details.push(ValidationDetail::success(
                "Memory Usage".to_string(),
                format!("Peak memory {:.2}MB is within {:.2}MB limit", 
                       metrics.memory_usage.peak_memory_mb, max_memory),
            ));
            passed_checks += 1;
        } else {
            details.push(ValidationDetail::failure(
                "Memory Usage".to_string(),
                format!("<= {:.2}MB", max_memory),
                format!("{:.2}MB", metrics.memory_usage.peak_memory_mb),
                format!("Memory usage exceeds limit by {:.2}MB", 
                       metrics.memory_usage.peak_memory_mb - max_memory),
            ));
        }
        total_checks += 1;
        
        let score = passed_checks as f64 / total_checks as f64;
        let passed = score >= 0.8;
        let summary = format!(
            "Performance validation: {}/{} checks passed (score: {:.1}%)",
            passed_checks, total_checks, score * 100.0
        );
        
        ValidationResult {
            passed,
            details,
            score,
            summary,
        }
    }
    
    /// Validate fallback behavior
    pub fn validate_fallback_behavior(&self, behavior: &FallbackBehavior) -> ValidationResult {
        let mut details = Vec::new();
        let mut passed_checks = 0;
        let mut total_checks = 0;
        
        // Check if fallback was properly triggered
        if behavior.fallback_triggered {
            details.push(ValidationDetail::success(
                "Fallback Trigger".to_string(),
                format!("Fallback properly triggered: {}", behavior.fallback_reason),
            ));
            passed_checks += 1;
        } else {
            details.push(ValidationDetail::failure_with_message(
                "Fallback Trigger".to_string(),
                "Fallback was not triggered when expected".to_string(),
            ));
        }
        total_checks += 1;
        
        // Validate fallback strategy
        if !behavior.fallback_strategy.is_empty() {
            details.push(ValidationDetail::success(
                "Fallback Strategy".to_string(),
                format!("Fallback strategy applied: {}", behavior.fallback_strategy),
            ));
            passed_checks += 1;
        } else {
            details.push(ValidationDetail::failure_with_message(
                "Fallback Strategy".to_string(),
                "No fallback strategy specified".to_string(),
            ));
        }
        total_checks += 1;
        
        // Validate fallback performance
        if let Some(degradation) = behavior.fallback_performance.get("performance_degradation") {
            if *degradation <= 0.5 { // Max 50% performance degradation
                details.push(ValidationDetail::success(
                    "Fallback Performance".to_string(),
                    format!("Performance degradation {:.1}% is acceptable", degradation * 100.0),
                ));
                passed_checks += 1;
            } else {
                details.push(ValidationDetail::failure(
                    "Fallback Performance".to_string(),
                    "<= 50%".to_string(),
                    format!("{:.1}%", degradation * 100.0),
                    "Performance degradation exceeds acceptable threshold".to_string(),
                ));
            }
        }
        total_checks += 1;
        
        let score = passed_checks as f64 / total_checks as f64;
        let passed = score >= 0.8;
        let summary = format!(
            "Fallback validation: {}/{} checks passed (score: {:.1}%)",
            passed_checks, total_checks, score * 100.0
        );
        
        ValidationResult {
            passed,
            details,
            score,
            summary,
        }
    }
    
    /// Validate numeric array with tolerance
    fn validate_numeric_array(
        &self,
        computed: &[f64],
        expected: &[f64],
        name: &str,
        tolerance: f64,
    ) -> ValidationResult {
        let mut details = Vec::new();
        
        // Check array lengths
        if computed.len() != expected.len() {
            return ValidationResult {
                passed: false,
                details: vec![ValidationDetail::failure(
                    format!("{} Length", name),
                    expected.len().to_string(),
                    computed.len().to_string(),
                    "Array length mismatch".to_string(),
                )],
                score: 0.0,
                summary: format!("{} validation failed: length mismatch", name),
            };
        }
        
        let mut within_tolerance = 0;
        let total_values = computed.len();
        
        for (i, (comp, exp)) in computed.iter().zip(expected.iter()).enumerate() {
            let diff = (comp - exp).abs();
            let relative_tolerance = if exp.abs() > 1e-10 {
                diff / exp.abs()
            } else {
                diff
            };
            
            if relative_tolerance <= tolerance {
                within_tolerance += 1;
            } else if details.len() < 5 { // Limit error details to first 5 mismatches
                details.push(ValidationDetail::failure(
                    format!("{} Value [{}]", name, i),
                    exp.to_string(),
                    comp.to_string(),
                    format!("Relative difference {:.6} exceeds tolerance {:.6}", 
                           relative_tolerance, tolerance),
                ));
            }
        }
        
        let score = within_tolerance as f64 / total_values as f64;
        let passed = score >= 0.95; // 95% of values must be within tolerance
        
        if passed {
            details.insert(0, ValidationDetail::success(
                name.to_string(),
                format!("{}/{} values within tolerance (score: {:.1}%)", 
                       within_tolerance, total_values, score * 100.0),
            ));
        }
        
        let summary = format!(
            "{} validation: {}/{} values within tolerance (score: {:.1}%)",
            name, within_tolerance, total_values, score * 100.0
        );
        
        ValidationResult {
            passed,
            details,
            score,
            summary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_result_validator_creation() {
        let config = ValidationConfig {
            feature_tolerance: 0.01,
            signal_tolerance: 0.05,
            performance_tolerance: 0.1,
            strict_mode: false,
            validation_timeout_seconds: 30,
        };
        
        let validator = ResultValidator::new(config).unwrap();
        assert!(validator.reference_data.is_none());
    }
    
    #[test]
    fn test_numeric_array_validation() {
        let config = ValidationConfig {
            feature_tolerance: 0.01,
            signal_tolerance: 0.05,
            performance_tolerance: 0.1,
            strict_mode: false,
            validation_timeout_seconds: 30,
        };
        
        let validator = ResultValidator::new(config).unwrap();
        
        let computed = vec![1.0, 2.0, 3.0];
        let expected = vec![1.001, 2.001, 3.001];
        
        let result = validator.validate_numeric_array(&computed, &expected, "Test", 0.01);
        assert!(result.passed);
        assert!(result.score > 0.9);
    }
    
    #[test]
    fn test_signal_validation() {
        let config = ValidationConfig {
            feature_tolerance: 0.01,
            signal_tolerance: 0.05,
            performance_tolerance: 0.1,
            strict_mode: false,
            validation_timeout_seconds: 30,
        };
        
        let validator = ResultValidator::new(config).unwrap();
        
        let generated = TradingSignal {
            timestamp: 1234567890,
            strength: 0.75,
            confidence: 0.85,
            signal_type: SignalType::LDC,
            metadata: HashMap::new(),
        };
        
        let expected = TradingSignal {
            timestamp: 1234567890,
            strength: 0.76,
            confidence: 0.84,
            signal_type: SignalType::LDC,
            metadata: HashMap::new(),
        };
        
        let result = validator.validate_signals(&generated, &expected);
        assert!(result.passed);
    }
    
    #[test]
    fn test_fallback_behavior_validation() {
        let config = ValidationConfig {
            feature_tolerance: 0.01,
            signal_tolerance: 0.05,
            performance_tolerance: 0.1,
            strict_mode: false,
            validation_timeout_seconds: 30,
        };
        
        let validator = ResultValidator::new(config).unwrap();
        
        let mut performance = HashMap::new();
        performance.insert("performance_degradation".to_string(), 0.3);
        
        let behavior = FallbackBehavior {
            fallback_triggered: true,
            fallback_reason: "HMM service unavailable".to_string(),
            fallback_strategy: "Use cached weights".to_string(),
            fallback_performance: performance,
        };
        
        let result = validator.validate_fallback_behavior(&behavior);
        assert!(result.passed);
    }
}