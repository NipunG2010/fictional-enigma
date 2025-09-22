// Configuration settings structures

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotConfig {
    pub horizon: usize,
    pub features: Vec<FeatureType>,
    pub label_thresholds: LabelThresholds,
    pub validation_strictness: ValidationLevel,
    pub date_range: Option<DateRange>,
    pub output_format: OutputFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FeatureType {
    Rsi14,
    Sma20,
    Ema12,
    Ema26,
    Macd,
    MacdSignal,
    BbUpper,
    BbMiddle,
    BbLower,
    Atr14,
    Cci14,
    Adx14,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LabelThresholds {
    pub buy_threshold: f64,
    pub sell_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationLevel {
    Strict,
    Normal,
    Lenient,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OutputFormat {
    Parquet,
    Csv,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedConfig {
    pub name: String,
    pub config: SnapshotConfig,
    pub created_at: DateTime<Utc>,
    pub version: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConfigInfo {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub version: String,
    pub description: Option<String>,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            horizon: 12,
            features: vec![
                FeatureType::Rsi14,
                FeatureType::Sma20,
                FeatureType::Ema12,
                FeatureType::Ema26,
                FeatureType::Macd,
                FeatureType::MacdSignal,
                FeatureType::BbUpper,
                FeatureType::BbLower,
                FeatureType::Atr14,
            ],
            label_thresholds: LabelThresholds::default(),
            validation_strictness: ValidationLevel::Normal,
            date_range: None,
            output_format: OutputFormat::Parquet,
        }
    }
}

impl Default for LabelThresholds {
    fn default() -> Self {
        Self {
            buy_threshold: 0.02,   // 2%
            sell_threshold: -0.02, // -2%
        }
    }
}

impl SnapshotConfig {
    /// Validate the configuration and return any validation errors
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate horizon
        if self.horizon == 0 {
            errors.push("Horizon must be greater than 0".to_string());
        }
        if self.horizon > 1000 {
            errors.push("Horizon must be less than or equal to 1000".to_string());
        }

        // Validate features
        if self.features.is_empty() {
            errors.push("At least one feature must be specified".to_string());
        }

        // Validate label thresholds
        if self.label_thresholds.buy_threshold <= self.label_thresholds.sell_threshold {
            errors.push("Buy threshold must be greater than sell threshold".to_string());
        }
        if self.label_thresholds.buy_threshold <= 0.0 {
            errors.push("Buy threshold must be positive".to_string());
        }
        if self.label_thresholds.sell_threshold >= 0.0 {
            errors.push("Sell threshold must be negative".to_string());
        }

        // Validate date range
        if let Some(ref date_range) = self.date_range {
            if date_range.start >= date_range.end {
                errors.push("Start date must be before end date".to_string());
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}