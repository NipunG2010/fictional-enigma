use anyhow::{Result, Context};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::create_dir_all;

use feature_pipeline::{OHLCV, Features};
use crate::{FeatureSeries, TrainingSample, Direction};

/// Configuration for test data generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataConfig {
    /// Number of samples to generate
    pub sample_count: usize,
    /// Starting timestamp (Unix timestamp)
    pub start_timestamp: i64,
    /// Time interval between samples in seconds
    pub interval_seconds: i64,
    /// Base price for synthetic data
    pub base_price: f64,
    /// Volatility factor (0.0 to 1.0)
    pub volatility: f64,
    /// Trend factor (-1.0 to 1.0, negative for downtrend)
    pub trend: f64,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    /// Market regime parameters
    pub market_regime: MarketRegime,
}

impl Default for TestDataConfig {
    fn default() -> Self {
        Self {
            sample_count: 1000,
            start_timestamp: 1640995200, // 2022-01-01 00:00:00 UTC
            interval_seconds: 300, // 5 minutes
            base_price: 50000.0, // BTC-like price
            volatility: 0.02, // 2% volatility
            trend: 0.0, // No trend
            seed: Some(42),
            market_regime: MarketRegime::Normal,
        }
    }
}

/// Market regime types for different testing scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Normal market conditions
    Normal,
    /// High volatility trending market
    Trending { direction: f64, strength: f64 },
    /// Low volatility ranging market
    Ranging { range_factor: f64 },
    /// Extreme volatility crisis conditions
    Crisis { volatility_multiplier: f64 },
    /// Flash crash scenario
    FlashCrash { crash_magnitude: f64, recovery_time: usize },
}

/// Test data generator for creating realistic synthetic market data
pub struct TestDataGenerator {
    pub config: TestDataConfig,
    rng: StdRng,
}

impl TestDataGenerator {
    /// Create a new test data generator with configuration
    pub fn new(config: TestDataConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        Self { config, rng }
    }

    /// Create a generator with default configuration
    pub fn default() -> Self {
        Self::new(TestDataConfig::default())
    }

    /// Generate synthetic OHLCV data with proper statistical properties
    pub fn create_synthetic_dataset(&mut self) -> Result<Vec<OHLCV>> {
        let mut data = Vec::with_capacity(self.config.sample_count);
        let mut current_price = self.config.base_price;
        let mut current_timestamp = self.config.start_timestamp;

        // Create price distribution based on volatility
        let price_change_dist = Normal::new(0.0, self.config.volatility)
            .context("Failed to create normal distribution for price changes")?;
        
        let volume_dist = Normal::new(1000000.0, 200000.0)
            .context("Failed to create normal distribution for volume")?;

        for i in 0..self.config.sample_count {
            // Generate price movement based on market regime
            let price_change = self.generate_price_change(i, &price_change_dist)?;
            current_price *= 1.0 + price_change;

            // Ensure price doesn't go negative
            current_price = current_price.max(0.01);

            // Generate OHLC from close price with realistic spread
            let (open, high, low, close) = self.generate_ohlc_from_price(current_price)?;

            // Generate realistic volume
            let volume = self.generate_volume(&volume_dist, price_change.abs())?;

            data.push(OHLCV {
                timestamp: current_timestamp,
                open,
                high,
                low,
                close,
                volume,
            });

            current_timestamp += self.config.interval_seconds;
            current_price = close; // Use close as next period's reference
        }

        Ok(data)
    }

    /// Generate realistic technical indicator values from OHLCV data
    pub fn generate_features_data(&mut self, ohlcv_data: &[OHLCV]) -> Result<Vec<Features>> {
        if ohlcv_data.is_empty() {
            return Ok(Vec::new());
        }

        let mut features = Vec::with_capacity(ohlcv_data.len());

        for (i, ohlcv) in ohlcv_data.iter().enumerate() {
            let feature = if i < 20 {
                // For early periods, use simplified calculations or None
                Features {
                    timestamp: ohlcv.timestamp,
                    rsi: None,
                    sma_20: None,
                    ema_20: None,
                    std_20: None,
                    zscore_20: None,
                    momentum: if i > 0 {
                        Some((ohlcv.close - ohlcv_data[i-1].close) / ohlcv_data[i-1].close)
                    } else {
                        None
                    },
                    wavetrend_1: None,
                    wavetrend_2: None,
                    cci: None,
                    adx: None,
                }
            } else {
                // Calculate realistic technical indicators
                let window_start = i.saturating_sub(19);
                let window_data = &ohlcv_data[window_start..=i];
                
                Features {
                    timestamp: ohlcv.timestamp,
                    rsi: Some(self.calculate_synthetic_rsi(window_data)?),
                    sma_20: Some(self.calculate_sma(window_data)),
                    ema_20: Some(self.calculate_ema(window_data)),
                    std_20: Some(self.calculate_std(window_data)),
                    zscore_20: Some(self.calculate_zscore(window_data)),
                    momentum: Some((ohlcv.close - ohlcv_data[i-1].close) / ohlcv_data[i-1].close),
                    wavetrend_1: Some(self.calculate_synthetic_wavetrend(window_data, 1)?),
                    wavetrend_2: Some(self.calculate_synthetic_wavetrend(window_data, 2)?),
                    cci: Some(self.calculate_synthetic_cci(window_data)?),
                    adx: Some(self.calculate_synthetic_adx(window_data)?),
                }
            };

            features.push(feature);
        }

        Ok(features)
    }

    /// Generate edge case data for testing boundary conditions
    pub fn generate_edge_case_data(&mut self) -> Result<Vec<OHLCV>> {
        let mut edge_cases = Vec::new();
        let base_timestamp = self.config.start_timestamp;

        // Case 1: Zero values (should be handled gracefully)
        edge_cases.push(OHLCV {
            timestamp: base_timestamp,
            open: 0.0001, // Minimum positive value
            high: 0.0001,
            low: 0.0001,
            close: 0.0001,
            volume: 0.0,
        });

        // Case 2: Extreme high values
        edge_cases.push(OHLCV {
            timestamp: base_timestamp + 300,
            open: 1_000_000.0,
            high: 1_000_000.0,
            low: 1_000_000.0,
            close: 1_000_000.0,
            volume: 1_000_000_000.0,
        });

        // Case 3: Identical OHLC values (no price movement)
        let flat_price = 50000.0;
        for i in 0..10 {
            edge_cases.push(OHLCV {
                timestamp: base_timestamp + 600 + (i * 300),
                open: flat_price,
                high: flat_price,
                low: flat_price,
                close: flat_price,
                volume: 100000.0,
            });
        }

        // Case 4: Maximum volatility (large price swings)
        let mut volatile_price = 50000.0;
        for i in 0..10 {
            let change = if i % 2 == 0 { 0.1 } else { -0.1 }; // ±10% swings
            volatile_price *= 1.0 + change;
            
            let (open, high, low, close) = self.generate_ohlc_from_price(volatile_price)?;
            edge_cases.push(OHLCV {
                timestamp: base_timestamp + 3600 + (i * 300),
                open,
                high,
                low,
                close,
                volume: 500000.0,
            });
        }

        // Case 5: Precision edge cases (very small price differences)
        let precision_base = 0.00001;
        for i in 0..5 {
            let price = precision_base + (i as f64 * 0.000001);
            edge_cases.push(OHLCV {
                timestamp: base_timestamp + 6600 + (i * 300),
                open: price,
                high: price * 1.0001,
                low: price * 0.9999,
                close: price * 1.00005,
                volume: 1.0,
            });
        }

        Ok(edge_cases)
    }

    /// Generate error scenario data for testing error handling
    pub fn generate_error_scenarios(&mut self) -> Result<Vec<TestScenario>> {
        let mut scenarios = Vec::new();

        // Scenario 1: Invalid OHLC relationships
        scenarios.push(TestScenario {
            name: "invalid_ohlc_high_low".to_string(),
            description: "High price lower than low price".to_string(),
            data: vec![OHLCV {
                timestamp: self.config.start_timestamp,
                open: 50000.0,
                high: 49000.0, // High < Low (invalid)
                low: 50000.0,
                close: 49500.0,
                volume: 100000.0,
            }],
            expected_error: Some("Invalid OHLC data".to_string()),
        });

        // Scenario 2: Negative prices
        scenarios.push(TestScenario {
            name: "negative_prices".to_string(),
            description: "Negative price values".to_string(),
            data: vec![OHLCV {
                timestamp: self.config.start_timestamp,
                open: -100.0,
                high: -50.0,
                low: -150.0,
                close: -75.0,
                volume: 100000.0,
            }],
            expected_error: Some("Invalid price data".to_string()),
        });

        // Scenario 3: Invalid timestamps
        scenarios.push(TestScenario {
            name: "invalid_timestamps".to_string(),
            description: "Non-increasing timestamps".to_string(),
            data: vec![
                OHLCV {
                    timestamp: self.config.start_timestamp + 300,
                    open: 50000.0,
                    high: 50100.0,
                    low: 49900.0,
                    close: 50050.0,
                    volume: 100000.0,
                },
                OHLCV {
                    timestamp: self.config.start_timestamp, // Earlier timestamp (invalid)
                    open: 50050.0,
                    high: 50150.0,
                    low: 49950.0,
                    close: 50100.0,
                    volume: 100000.0,
                },
            ],
            expected_error: Some("Non-increasing timestamps".to_string()),
        });

        // Scenario 4: Extreme outliers
        scenarios.push(TestScenario {
            name: "extreme_outliers".to_string(),
            description: "Prices with extreme outlier values".to_string(),
            data: vec![
                OHLCV {
                    timestamp: self.config.start_timestamp,
                    open: 50000.0,
                    high: 50100.0,
                    low: 49900.0,
                    close: 50050.0,
                    volume: 100000.0,
                },
                OHLCV {
                    timestamp: self.config.start_timestamp + 300,
                    open: 50050.0,
                    high: 10_000_000.0, // Extreme outlier
                    low: 49950.0,
                    close: 50100.0,
                    volume: 100000.0,
                },
            ],
            expected_error: Some("Suspicious price range".to_string()),
        });

        Ok(scenarios)
    }

    /// Load and preprocess historical market data
    pub fn load_historical_data(&self, file_path: &Path) -> Result<Vec<OHLCV>> {
        let file_extension = file_path.extension()
            .and_then(|ext| ext.to_str())
            .context("Unable to determine file extension")?;

        match file_extension.to_lowercase().as_str() {
            "csv" => self.load_csv_data(file_path),
            "parquet" => self.load_parquet_data(file_path),
            _ => Err(anyhow::anyhow!("Unsupported file format: {}", file_extension)),
        }
    }

    /// Validate test data quality and consistency
    pub fn validate_test_data(&self, data: &[OHLCV]) -> Result<DataQualityReport> {
        let mut report = DataQualityReport::new();

        if data.is_empty() {
            report.add_error("Dataset is empty".to_string());
            return Ok(report);
        }

        // Check data completeness
        report.total_samples = data.len();
        
        // Validate each sample
        for (i, ohlcv) in data.iter().enumerate() {
            self.validate_single_sample(ohlcv, i, &mut report)?;
        }

        // Check temporal consistency
        self.validate_temporal_consistency(data, &mut report)?;

        // Calculate quality metrics
        self.calculate_quality_metrics(data, &mut report)?;

        Ok(report)
    }

    /// Create training samples from OHLCV and features data
    pub fn create_training_samples(
        &mut self,
        ohlcv_data: &[OHLCV],
        features_data: &[Features],
        horizon: usize,
    ) -> Result<Vec<TrainingSample>> {
        if ohlcv_data.len() != features_data.len() {
            return Err(anyhow::anyhow!("OHLCV and features data length mismatch"));
        }

        if ohlcv_data.len() < horizon + 1 {
            return Err(anyhow::anyhow!("Insufficient data for horizon {}", horizon));
        }

        let mut training_samples = Vec::new();

        for i in 0..(ohlcv_data.len() - horizon) {
            let current_features = &features_data[i];
            let future_ohlcv = &ohlcv_data[i + horizon];
            let current_ohlcv = &ohlcv_data[i];

            // Convert Features to FeatureSeries
            let feature_series = self.convert_features_to_series(current_features)?;

            // Determine label based on future price movement
            let price_change = (future_ohlcv.close - current_ohlcv.close) / current_ohlcv.close;
            let label = self.determine_label(price_change);

            training_samples.push(TrainingSample {
                features: feature_series,
                label,
                timestamp: current_ohlcv.timestamp,
                bar_index: i,
            });
        }

        Ok(training_samples)
    }

    // Private helper methods

    fn generate_price_change(&mut self, index: usize, base_dist: &Normal<f64>) -> Result<f64> {
        let base_change = base_dist.sample(&mut self.rng);
        
        match &self.config.market_regime {
            MarketRegime::Normal => Ok(base_change + self.config.trend * 0.001),
            
            MarketRegime::Trending { direction, strength } => {
                Ok(base_change + direction * strength * 0.01)
            },
            
            MarketRegime::Ranging { range_factor } => {
                // Reduce volatility and add mean reversion
                let reduced_change = base_change * range_factor;
                let mean_reversion = -reduced_change * 0.1; // Slight mean reversion
                Ok(reduced_change + mean_reversion)
            },
            
            MarketRegime::Crisis { volatility_multiplier } => {
                Ok(base_change * volatility_multiplier)
            },
            
            MarketRegime::FlashCrash { crash_magnitude, recovery_time } => {
                if index == 100 { // Flash crash at sample 100
                    Ok(-crash_magnitude)
                } else if index > 100 && index < 100 + recovery_time {
                    // Gradual recovery
                    let recovery_progress = (index - 100) as f64 / *recovery_time as f64;
                    Ok(base_change + crash_magnitude * 0.1 * (1.0 - recovery_progress))
                } else {
                    Ok(base_change)
                }
            },
        }
    }

    fn generate_ohlc_from_price(&mut self, close_price: f64) -> Result<(f64, f64, f64, f64)> {
        let spread_factor = 0.001; // 0.1% typical spread
        let uniform_dist = Uniform::new(-1.0, 1.0);

        // Generate open price (slight variation from close)
        let open_variation = uniform_dist.sample(&mut self.rng) * spread_factor;
        let open = close_price * (1.0 + open_variation);

        // Generate high and low with realistic relationships
        let high_variation = uniform_dist.sample(&mut self.rng).abs() * spread_factor * 2.0;
        let low_variation = uniform_dist.sample(&mut self.rng).abs() * spread_factor * 2.0;

        let high = open.max(close_price) * (1.0 + high_variation);
        let low = open.min(close_price) * (1.0 - low_variation);

        Ok((open, high, low, close_price))
    }

    fn generate_volume(&mut self, base_dist: &Normal<f64>, price_volatility: f64) -> Result<f64> {
        let base_volume = base_dist.sample(&mut self.rng).max(1000.0);
        
        // Volume tends to increase with volatility
        let volatility_factor = 1.0 + price_volatility * 10.0;
        
        Ok(base_volume * volatility_factor)
    }

    // Simplified technical indicator calculations for synthetic data
    fn calculate_synthetic_rsi(&mut self, window_data: &[OHLCV]) -> Result<f64> {
        if window_data.len() < 2 {
            return Ok(50.0); // Neutral RSI
        }

        let mut gains = 0.0;
        let mut losses = 0.0;
        let mut count = 0;

        for i in 1..window_data.len() {
            let change = window_data[i].close - window_data[i-1].close;
            if change > 0.0 {
                gains += change;
            } else {
                losses += -change;
            }
            count += 1;
        }

        if count == 0 {
            return Ok(50.0);
        }

        let avg_gain = gains / count as f64;
        let avg_loss = losses / count as f64;

        if avg_loss == 0.0 {
            return Ok(100.0);
        }

        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));

        Ok(rsi.clamp(0.0, 100.0))
    }

    fn calculate_sma(&self, window_data: &[OHLCV]) -> f64 {
        let sum: f64 = window_data.iter().map(|d| d.close).sum();
        sum / window_data.len() as f64
    }

    fn calculate_ema(&self, window_data: &[OHLCV]) -> f64 {
        if window_data.is_empty() {
            return 0.0;
        }

        let alpha = 2.0 / (window_data.len() as f64 + 1.0);
        let mut ema = window_data[0].close;

        for data in window_data.iter().skip(1) {
            ema = alpha * data.close + (1.0 - alpha) * ema;
        }

        ema
    }

    fn calculate_std(&self, window_data: &[OHLCV]) -> f64 {
        let mean = self.calculate_sma(window_data);
        let variance: f64 = window_data.iter()
            .map(|d| (d.close - mean).powi(2))
            .sum::<f64>() / window_data.len() as f64;
        variance.sqrt()
    }

    fn calculate_zscore(&self, window_data: &[OHLCV]) -> f64 {
        let mean = self.calculate_sma(window_data);
        let std = self.calculate_std(window_data);
        
        if std == 0.0 {
            return 0.0;
        }

        let current_price = window_data.last().unwrap().close;
        (current_price - mean) / std
    }

    fn calculate_synthetic_wavetrend(&mut self, window_data: &[OHLCV], variant: u8) -> Result<f64> {
        // Simplified wavetrend calculation
        let typical_prices: Vec<f64> = window_data.iter()
            .map(|d| (d.high + d.low + d.close) / 3.0)
            .collect();

        if typical_prices.is_empty() {
            return Ok(0.0);
        }

        let mean = typical_prices.iter().sum::<f64>() / typical_prices.len() as f64;
        let current = typical_prices.last().unwrap();
        
        let wt = (current - mean) / mean * 100.0;
        
        match variant {
            1 => Ok(wt),
            2 => Ok(wt * 0.8), // Signal line (smoothed)
            _ => Ok(0.0),
        }
    }

    fn calculate_synthetic_cci(&mut self, window_data: &[OHLCV]) -> Result<f64> {
        let typical_prices: Vec<f64> = window_data.iter()
            .map(|d| (d.high + d.low + d.close) / 3.0)
            .collect();

        let mean = typical_prices.iter().sum::<f64>() / typical_prices.len() as f64;
        let current = typical_prices.last().unwrap();
        
        let mean_deviation = typical_prices.iter()
            .map(|&tp| (tp - mean).abs())
            .sum::<f64>() / typical_prices.len() as f64;

        if mean_deviation == 0.0 {
            return Ok(0.0);
        }

        let cci = (current - mean) / (0.015 * mean_deviation);
        Ok(cci.clamp(-500.0, 500.0))
    }

    fn calculate_synthetic_adx(&mut self, window_data: &[OHLCV]) -> Result<f64> {
        if window_data.len() < 2 {
            return Ok(25.0); // Neutral ADX
        }

        // Simplified ADX calculation
        let mut dm_sum = 0.0;
        let mut tr_sum = 0.0;

        for i in 1..window_data.len() {
            let current = &window_data[i];
            let previous = &window_data[i-1];

            let tr = (current.high - current.low)
                .max((current.high - previous.close).abs())
                .max((current.low - previous.close).abs());

            let dm_plus = if current.high - previous.high > previous.low - current.low {
                (current.high - previous.high).max(0.0)
            } else {
                0.0
            };

            dm_sum += dm_plus;
            tr_sum += tr;
        }

        if tr_sum == 0.0 {
            return Ok(25.0);
        }

        let adx = (dm_sum / tr_sum) * 100.0;
        Ok(adx.clamp(0.0, 100.0))
    }

    fn load_csv_data(&self, file_path: &Path) -> Result<Vec<OHLCV>> {
        // Implementation would use CSV parsing library
        // For now, return empty vector as placeholder
        Ok(Vec::new())
    }

    fn load_parquet_data(&self, file_path: &Path) -> Result<Vec<OHLCV>> {
        // Implementation would use Parquet parsing library
        // For now, return empty vector as placeholder
        Ok(Vec::new())
    }

    fn validate_single_sample(&self, ohlcv: &OHLCV, index: usize, report: &mut DataQualityReport) -> Result<()> {
        // Validate timestamp
        if ohlcv.timestamp <= 0 {
            report.add_error(format!("Invalid timestamp at index {}: {}", index, ohlcv.timestamp));
        }

        // Validate prices
        if ohlcv.open <= 0.0 || ohlcv.high <= 0.0 || ohlcv.low <= 0.0 || ohlcv.close <= 0.0 {
            report.add_error(format!("Non-positive prices at index {}", index));
        }

        // Validate OHLC relationships
        if ohlcv.high < ohlcv.low {
            report.add_error(format!("High < Low at index {}", index));
        }

        if ohlcv.high < ohlcv.open.max(ohlcv.close) {
            report.add_error(format!("High is not highest price at index {}", index));
        }

        if ohlcv.low > ohlcv.open.min(ohlcv.close) {
            report.add_error(format!("Low is not lowest price at index {}", index));
        }

        // Validate volume
        if ohlcv.volume < 0.0 {
            report.add_error(format!("Negative volume at index {}", index));
        }

        Ok(())
    }

    fn validate_temporal_consistency(&self, data: &[OHLCV], report: &mut DataQualityReport) -> Result<()> {
        for i in 1..data.len() {
            if data[i].timestamp <= data[i-1].timestamp {
                report.add_error(format!("Non-increasing timestamps at indices {} and {}", i-1, i));
            }
        }
        Ok(())
    }

    fn calculate_quality_metrics(&self, data: &[OHLCV], report: &mut DataQualityReport) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        // Calculate price statistics
        let prices: Vec<f64> = data.iter().map(|d| d.close).collect();
        let mean_price = prices.iter().sum::<f64>() / prices.len() as f64;
        
        let price_variance = prices.iter()
            .map(|&p| (p - mean_price).powi(2))
            .sum::<f64>() / prices.len() as f64;
        
        report.mean_price = Some(mean_price);
        report.price_volatility = Some(price_variance.sqrt() / mean_price);

        // Calculate volume statistics
        let volumes: Vec<f64> = data.iter().map(|d| d.volume).collect();
        let mean_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        report.mean_volume = Some(mean_volume);

        // Calculate data completeness
        report.completeness_ratio = 1.0 - (report.errors.len() as f64 / data.len() as f64);

        Ok(())
    }

    pub fn convert_features_to_series(&self, features: &Features) -> Result<FeatureSeries> {
        Ok(FeatureSeries {
            f1: features.rsi.unwrap_or(50.0) as f32,
            f2: features.wavetrend_1.unwrap_or(0.0) as f32,
            f3: features.cci.unwrap_or(0.0) as f32,
            f4: features.adx.unwrap_or(25.0) as f32,
            f5: features.momentum.unwrap_or(0.0) as f32,
        })
    }

    pub fn determine_label(&self, price_change: f64) -> Direction {
        const THRESHOLD: f64 = 0.001; // 0.1% threshold
        
        if price_change > THRESHOLD {
            Direction::Long
        } else if price_change < -THRESHOLD {
            Direction::Short
        } else {
            Direction::Neutral
        }
    }
}

/// Test scenario for error condition testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    pub name: String,
    pub description: String,
    pub data: Vec<OHLCV>,
    pub expected_error: Option<String>,
}

/// Data quality report for validation results
#[derive(Debug, Clone)]
pub struct DataQualityReport {
    pub total_samples: usize,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub completeness_ratio: f64,
    pub mean_price: Option<f64>,
    pub price_volatility: Option<f64>,
    pub mean_volume: Option<f64>,
}

impl DataQualityReport {
    pub fn new() -> Self {
        Self {
            total_samples: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
            completeness_ratio: 0.0,
            mean_price: None,
            price_volatility: None,
            mean_volume: None,
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn quality_score(&self) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }

        let error_penalty = self.errors.len() as f64 / self.total_samples as f64;
        let warning_penalty = self.warnings.len() as f64 / self.total_samples as f64 * 0.5;
        
        (1.0 - error_penalty - warning_penalty).max(0.0)
    }
}

/// Utility functions for test data management
pub struct TestDataManager;

impl TestDataManager {
    /// Save test dataset to file
    pub fn save_dataset(data: &[OHLCV], file_path: &Path) -> Result<()> {
        if let Some(parent) = file_path.parent() {
            create_dir_all(parent)?;
        }

        let json_data = serde_json::to_string_pretty(data)?;
        std::fs::write(file_path, json_data)?;
        
        Ok(())
    }

    /// Load test dataset from file
    pub fn load_dataset(file_path: &Path) -> Result<Vec<OHLCV>> {
        let json_data = std::fs::read_to_string(file_path)?;
        let data: Vec<OHLCV> = serde_json::from_str(&json_data)?;
        Ok(data)
    }

    /// Create test data directory structure
    pub fn create_test_directories(base_path: &Path) -> Result<HashMap<String, PathBuf>> {
        let mut directories = HashMap::new();

        let subdirs = [
            "synthetic",
            "edge_cases", 
            "error_scenarios",
            "historical",
            "validation_reports",
        ];

        for subdir in &subdirs {
            let dir_path = base_path.join(subdir);
            create_dir_all(&dir_path)?;
            directories.insert(subdir.to_string(), dir_path);
        }

        Ok(directories)
    }

    /// Generate comprehensive test suite
    pub fn generate_test_suite(base_path: &Path) -> Result<TestSuite> {
        let directories = Self::create_test_directories(base_path)?;
        
        let mut generator = TestDataGenerator::default();
        
        // Generate different dataset sizes
        let dataset_configs = vec![
            (TestDataConfig { sample_count: 1000, ..Default::default() }, "small_1k"),
            (TestDataConfig { sample_count: 10000, ..Default::default() }, "medium_10k"),
            (TestDataConfig { sample_count: 50000, ..Default::default() }, "large_50k"),
        ];

        let mut datasets = HashMap::new();
        
        for (config, name) in dataset_configs {
            generator.config = config;
            let data = generator.create_synthetic_dataset()?;
            let file_path = directories["synthetic"].join(format!("{}.json", name));
            Self::save_dataset(&data, &file_path)?;
            datasets.insert(name.to_string(), file_path);
        }

        // Generate edge cases
        let edge_cases = generator.generate_edge_case_data()?;
        let edge_cases_path = directories["edge_cases"].join("edge_cases.json");
        Self::save_dataset(&edge_cases, &edge_cases_path)?;

        // Generate error scenarios
        let error_scenarios = generator.generate_error_scenarios()?;
        let scenarios_path = directories["error_scenarios"].join("error_scenarios.json");
        let scenarios_json = serde_json::to_string_pretty(&error_scenarios)?;
        std::fs::write(&scenarios_path, scenarios_json)?;

        Ok(TestSuite {
            base_path: base_path.to_path_buf(),
            directories,
            datasets,
            edge_cases_path,
            error_scenarios_path: scenarios_path,
        })
    }
}

/// Complete test suite with all generated data
#[derive(Debug)]
pub struct TestSuite {
    pub base_path: PathBuf,
    pub directories: HashMap<String, PathBuf>,
    pub datasets: HashMap<String, PathBuf>,
    pub edge_cases_path: PathBuf,
    pub error_scenarios_path: PathBuf,
}

impl TestSuite {
    /// Get dataset by name
    pub fn get_dataset(&self, name: &str) -> Result<Vec<OHLCV>> {
        let path = self.datasets.get(name)
            .ok_or_else(|| anyhow::anyhow!("Dataset '{}' not found", name))?;
        TestDataManager::load_dataset(path)
    }

    /// Get edge cases data
    pub fn get_edge_cases(&self) -> Result<Vec<OHLCV>> {
        TestDataManager::load_dataset(&self.edge_cases_path)
    }

    /// Get error scenarios
    pub fn get_error_scenarios(&self) -> Result<Vec<TestScenario>> {
        let json_data = std::fs::read_to_string(&self.error_scenarios_path)?;
        let scenarios: Vec<TestScenario> = serde_json::from_str(&json_data)?;
        Ok(scenarios)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_synthetic_data_generation() {
        let mut generator = TestDataGenerator::default();
        let data = generator.create_synthetic_dataset().unwrap();
        
        assert_eq!(data.len(), 1000);
        assert!(data.iter().all(|d| d.open > 0.0 && d.high > 0.0 && d.low > 0.0 && d.close > 0.0));
        
        // Check OHLC relationships
        for ohlcv in &data {
            assert!(ohlcv.high >= ohlcv.open.max(ohlcv.close));
            assert!(ohlcv.low <= ohlcv.open.min(ohlcv.close));
        }
    }

    #[test]
    fn test_features_generation() {
        let mut generator = TestDataGenerator::default();
        let ohlcv_data = generator.create_synthetic_dataset().unwrap();
        let features = generator.generate_features_data(&ohlcv_data).unwrap();
        
        assert_eq!(features.len(), ohlcv_data.len());
        
        // Check that later features have calculated values
        for feature in features.iter().skip(20) {
            assert!(feature.rsi.is_some());
            assert!(feature.sma_20.is_some());
        }
    }

    #[test]
    fn test_edge_case_generation() {
        let mut generator = TestDataGenerator::default();
        let edge_cases = generator.generate_edge_case_data().unwrap();
        
        assert!(!edge_cases.is_empty());
        
        // Should include various edge cases
        let has_zero_volume = edge_cases.iter().any(|d| d.volume == 0.0);
        let has_high_prices = edge_cases.iter().any(|d| d.close > 100000.0);
        
        assert!(has_zero_volume);
        assert!(has_high_prices);
    }

    #[test]
    fn test_data_validation() {
        let generator = TestDataGenerator::default();
        
        // Valid data
        let valid_data = vec![OHLCV {
            timestamp: 1640995200,
            open: 50000.0,
            high: 50100.0,
            low: 49900.0,
            close: 50050.0,
            volume: 100000.0,
        }];
        
        let report = generator.validate_test_data(&valid_data).unwrap();
        assert!(report.is_valid());
        
        // Invalid data
        let invalid_data = vec![OHLCV {
            timestamp: 1640995200,
            open: 50000.0,
            high: 49000.0, // High < Low
            low: 50000.0,
            close: 50050.0,
            volume: 100000.0,
        }];
        
        let report = generator.validate_test_data(&invalid_data).unwrap();
        assert!(!report.is_valid());
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_test_suite_generation() {
        let temp_dir = TempDir::new().unwrap();
        let test_suite = TestDataManager::generate_test_suite(temp_dir.path()).unwrap();
        
        assert!(test_suite.directories.contains_key("synthetic"));
        assert!(test_suite.directories.contains_key("edge_cases"));
        assert!(!test_suite.datasets.is_empty());
        
        // Test loading a dataset
        let small_dataset = test_suite.get_dataset("small_1k").unwrap();
        assert_eq!(small_dataset.len(), 1000);
    }
}