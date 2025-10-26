//! Test data generation for realistic market scenarios
//! 
//! Generates OHLCV data, market scenarios, edge cases, and reference data
//! for comprehensive testing of the trading system pipeline.

use crate::{
    config::DataGenConfig,
    validation::{Features, ReferenceDataSet, ReferenceMetadata, TradingSignal, SignalType},
    Result, TestFrameworkError,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Test data generator for creating realistic market scenarios
pub struct TestDataGenerator {
    /// Configuration for data generation
    config: DataGenConfig,
    
    /// Random number generator with fixed seed for reproducibility
    rng: ChaCha8Rng,
}

/// OHLCV bar data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVBar {
    /// Timestamp in seconds since epoch
    pub timestamp: i64,
    
    /// Opening price
    pub open: f64,
    
    /// Highest price
    pub high: f64,
    
    /// Lowest price
    pub low: f64,
    
    /// Closing price
    pub close: f64,
    
    /// Volume
    pub volume: f64,
    
    /// Symbol identifier
    pub symbol: String,
    
    /// Time interval (e.g., "5m", "1h")
    pub interval: String,
}

/// Market scenario types for testing different conditions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MarketScenario {
    /// Strong upward trend
    TrendingUp,
    
    /// Strong downward trend
    TrendingDown,
    
    /// Sideways/ranging market
    Sideways,
    
    /// High volatility conditions
    HighVolatility,
    
    /// Low volatility conditions
    LowVolatility,
    
    /// Price gap up
    GapUp,
    
    /// Price gap down
    GapDown,
    
    /// Flash crash scenario
    FlashCrash,
    
    /// Market recovery after crash
    Recovery,
    
    /// Consolidation pattern
    Consolidation,
}

/// Complete test data set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataSet {
    /// OHLCV data
    pub ohlcv_data: Vec<OHLCVBar>,
    
    /// Market scenario type
    pub scenario: MarketScenario,
    
    /// Expected features for validation
    pub expected_features: Option<Features>,
    
    /// Expected signals for validation
    pub expected_signals: Option<Vec<TradingSignal>>,
    
    /// Metadata about the test data
    pub metadata: TestDataMetadata,
}

/// Metadata for test data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDataMetadata {
    /// Data generation timestamp
    pub generated_at: i64,
    
    /// Symbol used
    pub symbol: String,
    
    /// Time interval
    pub interval: String,
    
    /// Number of bars
    pub bar_count: usize,
    
    /// Duration covered
    pub duration_hours: f64,
    
    /// Random seed used
    pub random_seed: u64,
    
    /// Description of the scenario
    pub description: String,
}

/// Edge case data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeCaseType {
    /// Missing OHLCV values
    MissingValues,
    
    /// Extreme price outliers
    PriceOutliers,
    
    /// Zero or negative volumes
    InvalidVolume,
    
    /// Corrupted timestamp sequence
    CorruptedTimestamps,
    
    /// Duplicate bars
    DuplicateBars,
    
    /// Extreme volatility spikes
    VolatilitySpikes,
}

impl TestDataGenerator {
    /// Create a new test data generator
    pub fn new(config: DataGenConfig) -> Result<Self> {
        let seed = config.random_seed.unwrap_or(42);
        let rng = ChaCha8Rng::seed_from_u64(seed);
        
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&config.output_dir)
            .map_err(|e| TestFrameworkError::DataGenerationError(format!("Failed to create output directory: {}", e)))?;
        
        Ok(Self { config, rng })
    }
    
    /// Generate OHLCV data for a specific symbol and duration
    pub fn generate_ohlcv_data(&mut self, symbol: &str, duration_hours: u32, interval: &str) -> Result<Vec<OHLCVBar>> {
        let interval_minutes = self.parse_interval_minutes(interval)?;
        let total_bars = (duration_hours * 60) / interval_minutes;
        
        let mut bars = Vec::with_capacity(total_bars as usize);
        let mut current_price = self.config.base_price;
        let start_timestamp = chrono::Utc::now().timestamp() - (duration_hours as i64 * 3600);
        
        for i in 0..total_bars {
            let timestamp = start_timestamp + (i as i64 * interval_minutes as i64 * 60);
            
            // Generate price movement
            let price_change = self.generate_price_change();
            let open = current_price;
            let close = open * (1.0 + price_change);
            
            // Generate high and low based on volatility
            let volatility = self.config.volatility_factor;
            let high_factor = 1.0 + self.rng.gen::<f64>() * volatility;
            let low_factor = 1.0 - self.rng.gen::<f64>() * volatility;
            
            let high = open.max(close) * high_factor;
            let low = open.min(close) * low_factor;
            
            // Generate volume
            let base_volume = 1000.0;
            let volume_factor = 0.5 + self.rng.gen::<f64>() * 1.5;
            let volume = base_volume * volume_factor;
            
            bars.push(OHLCVBar {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                symbol: symbol.to_string(),
                interval: interval.to_string(),
            });
            
            current_price = close;
        }
        
        Ok(bars)
    }
    
    /// Generate data for a specific market scenario
    pub fn generate_market_scenario(&mut self, scenario: MarketScenario, symbol: &str, duration_hours: u32) -> Result<TestDataSet> {
        let interval = "5m";
        let mut base_data = self.generate_ohlcv_data(symbol, duration_hours, interval)?;
        
        // Apply scenario-specific modifications
        match scenario {
            MarketScenario::TrendingUp => {
                self.apply_trend(&mut base_data, 0.02); // 2% upward trend
            }
            MarketScenario::TrendingDown => {
                self.apply_trend(&mut base_data, -0.02); // 2% downward trend
            }
            MarketScenario::Sideways => {
                self.apply_sideways_movement(&mut base_data);
            }
            MarketScenario::HighVolatility => {
                self.apply_high_volatility(&mut base_data);
            }
            MarketScenario::LowVolatility => {
                self.apply_low_volatility(&mut base_data);
            }
            MarketScenario::GapUp => {
                self.apply_gap(&mut base_data, 0.05, true); // 5% gap up
            }
            MarketScenario::GapDown => {
                self.apply_gap(&mut base_data, -0.05, false); // 5% gap down
            }
            MarketScenario::FlashCrash => {
                self.apply_flash_crash(&mut base_data);
            }
            MarketScenario::Recovery => {
                self.apply_recovery_pattern(&mut base_data);
            }
            MarketScenario::Consolidation => {
                self.apply_consolidation(&mut base_data);
            }
        }
        
        let metadata = TestDataMetadata {
            generated_at: chrono::Utc::now().timestamp(),
            symbol: symbol.to_string(),
            interval: interval.to_string(),
            bar_count: base_data.len(),
            duration_hours: duration_hours as f64,
            random_seed: self.config.random_seed.unwrap_or(42),
            description: format!("{:?} market scenario", scenario),
        };
        
        Ok(TestDataSet {
            ohlcv_data: base_data,
            scenario,
            expected_features: None,
            expected_signals: None,
            metadata,
        })
    }
    
    /// Generate edge case data for testing error handling
    pub fn generate_edge_cases(&mut self) -> Result<Vec<TestDataSet>> {
        let mut edge_cases = Vec::new();
        
        // Generate missing values case
        let mut missing_values_data = self.generate_ohlcv_data("TESTUSDT", 24, "5m")?;
        self.introduce_missing_values(&mut missing_values_data);
        
        edge_cases.push(TestDataSet {
            ohlcv_data: missing_values_data,
            scenario: MarketScenario::Sideways, // Base scenario
            expected_features: None,
            expected_signals: None,
            metadata: TestDataMetadata {
                generated_at: chrono::Utc::now().timestamp(),
                symbol: "TESTUSDT".to_string(),
                interval: "5m".to_string(),
                bar_count: 0,
                duration_hours: 24.0,
                random_seed: self.config.random_seed.unwrap_or(42),
                description: "Edge case: Missing values".to_string(),
            },
        });
        
        // Generate price outliers case
        let mut outliers_data = self.generate_ohlcv_data("TESTUSDT", 24, "5m")?;
        self.introduce_price_outliers(&mut outliers_data);
        
        edge_cases.push(TestDataSet {
            ohlcv_data: outliers_data,
            scenario: MarketScenario::HighVolatility,
            expected_features: None,
            expected_signals: None,
            metadata: TestDataMetadata {
                generated_at: chrono::Utc::now().timestamp(),
                symbol: "TESTUSDT".to_string(),
                interval: "5m".to_string(),
                bar_count: 0,
                duration_hours: 24.0,
                random_seed: self.config.random_seed.unwrap_or(42),
                description: "Edge case: Price outliers".to_string(),
            },
        });
        
        // Generate invalid volume case
        let mut invalid_volume_data = self.generate_ohlcv_data("TESTUSDT", 24, "5m")?;
        self.introduce_invalid_volumes(&mut invalid_volume_data);
        
        edge_cases.push(TestDataSet {
            ohlcv_data: invalid_volume_data,
            scenario: MarketScenario::Sideways,
            expected_features: None,
            expected_signals: None,
            metadata: TestDataMetadata {
                generated_at: chrono::Utc::now().timestamp(),
                symbol: "TESTUSDT".to_string(),
                interval: "5m".to_string(),
                bar_count: 0,
                duration_hours: 24.0,
                random_seed: self.config.random_seed.unwrap_or(42),
                description: "Edge case: Invalid volumes".to_string(),
            },
        });
        
        Ok(edge_cases)
    }
    
    /// Load reference data from file for validation
    pub fn load_reference_data<P: AsRef<Path>>(&self, path: P) -> Result<ReferenceDataSet> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| TestFrameworkError::DataGenerationError(format!("Failed to read reference data: {}", e)))?;
        
        let reference_data: ReferenceDataSet = serde_json::from_str(&content)
            .map_err(|e| TestFrameworkError::DataGenerationError(format!("Failed to parse reference data: {}", e)))?;
        
        Ok(reference_data)
    }
    
    /// Generate reference data for validation purposes
    pub fn generate_reference_data(&mut self, symbol: &str) -> Result<ReferenceDataSet> {
        // Generate base OHLCV data
        let ohlcv_data = self.generate_ohlcv_data(symbol, 24, "5m")?;
        
        // Calculate reference features
        let features = self.calculate_reference_features(&ohlcv_data)?;
        
        // Generate reference signals
        let signals = self.generate_reference_signals(&ohlcv_data, &features)?;
        
        // Create performance metrics
        let mut performance_metrics = HashMap::new();
        performance_metrics.insert("end_to_end_latency_ms".to_string(), 85.0);
        performance_metrics.insert("feature_computation_latency_ms".to_string(), 25.0);
        performance_metrics.insert("signal_generation_latency_ms".to_string(), 35.0);
        performance_metrics.insert("throughput_ops_per_sec".to_string(), 12.5);
        
        let metadata = ReferenceMetadata {
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now().timestamp(),
            description: format!("Reference data for {} testing", symbol),
            source: "TestDataGenerator".to_string(),
        };
        
        Ok(ReferenceDataSet {
            features: self.features_to_hashmap(&features),
            signals: self.signals_to_hashmap(&signals),
            performance_metrics,
            metadata,
        })
    }
    
    /// Save test data set to file
    pub fn save_test_data<P: AsRef<Path>>(&self, data: &TestDataSet, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(data)
            .map_err(|e| TestFrameworkError::DataGenerationError(format!("Failed to serialize test data: {}", e)))?;
        
        std::fs::write(path, json)
            .map_err(|e| TestFrameworkError::DataGenerationError(format!("Failed to write test data: {}", e)))?;
        
        Ok(())
    }
    
    /// Save reference data to file
    pub fn save_reference_data<P: AsRef<Path>>(&self, data: &ReferenceDataSet, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(data)
            .map_err(|e| TestFrameworkError::DataGenerationError(format!("Failed to serialize reference data: {}", e)))?;
        
        std::fs::write(path, json)
            .map_err(|e| TestFrameworkError::DataGenerationError(format!("Failed to write reference data: {}", e)))?;
        
        Ok(())
    }
    
    // Private helper methods
    
    fn parse_interval_minutes(&self, interval: &str) -> Result<u32> {
        match interval {
            "1m" => Ok(1),
            "5m" => Ok(5),
            "15m" => Ok(15),
            "30m" => Ok(30),
            "1h" => Ok(60),
            "4h" => Ok(240),
            "1d" => Ok(1440),
            _ => Err(TestFrameworkError::DataGenerationError(format!("Unsupported interval: {}", interval)).into()),
        }
    }
    
    fn generate_price_change(&mut self) -> f64 {
        let base_change = (self.rng.gen::<f64>() - 0.5) * 2.0; // -1 to 1
        base_change * self.config.volatility_factor
    }
    
    fn apply_trend(&mut self, bars: &mut [OHLCVBar], trend_factor: f64) {
        let trend_per_bar = trend_factor / bars.len() as f64;
        
        for (i, bar) in bars.iter_mut().enumerate() {
            let cumulative_trend = trend_per_bar * i as f64;
            let multiplier = 1.0 + cumulative_trend;
            
            bar.open *= multiplier;
            bar.high *= multiplier;
            bar.low *= multiplier;
            bar.close *= multiplier;
        }
    }
    
    fn apply_sideways_movement(&mut self, bars: &mut [OHLCVBar]) {
        if bars.is_empty() {
            return;
        }
        
        let base_price = bars[0].close;
        let range = base_price * 0.02; // 2% range
        
        for bar in bars.iter_mut() {
            let offset = (self.rng.gen::<f64>() - 0.5) * range;
            let target_price = base_price + offset;
            let adjustment = target_price / bar.close;
            
            bar.open *= adjustment;
            bar.high *= adjustment;
            bar.low *= adjustment;
            bar.close *= adjustment;
        }
    }
    
    fn apply_high_volatility(&mut self, bars: &mut [OHLCVBar]) {
        for bar in bars.iter_mut() {
            let volatility_multiplier = 1.0 + self.rng.gen::<f64>() * 0.1; // Up to 10% extra volatility
            let range = bar.high - bar.low;
            let new_range = range * volatility_multiplier;
            let mid_price = (bar.high + bar.low) / 2.0;
            
            bar.high = mid_price + new_range / 2.0;
            bar.low = mid_price - new_range / 2.0;
            
            // Adjust volume for high volatility
            bar.volume *= 1.5 + self.rng.gen::<f64>();
        }
    }
    
    fn apply_low_volatility(&mut self, bars: &mut [OHLCVBar]) {
        for bar in bars.iter_mut() {
            let volatility_multiplier = 0.3 + self.rng.gen::<f64>() * 0.4; // 30-70% of original volatility
            let range = bar.high - bar.low;
            let new_range = range * volatility_multiplier;
            let mid_price = (bar.high + bar.low) / 2.0;
            
            bar.high = mid_price + new_range / 2.0;
            bar.low = mid_price - new_range / 2.0;
            
            // Adjust volume for low volatility
            bar.volume *= 0.5 + self.rng.gen::<f64>() * 0.3;
        }
    }
    
    fn apply_gap(&mut self, bars: &mut [OHLCVBar], gap_percentage: f64, _is_gap_up: bool) {
        if bars.len() < 10 {
            return;
        }
        
        let gap_index = bars.len() / 2; // Apply gap in the middle
        let gap_multiplier = 1.0 + gap_percentage;
        
        for bar in bars.iter_mut().skip(gap_index) {
            bar.open *= gap_multiplier;
            bar.high *= gap_multiplier;
            bar.low *= gap_multiplier;
            bar.close *= gap_multiplier;
        }
    }
    
    fn apply_flash_crash(&mut self, bars: &mut [OHLCVBar]) {
        if bars.len() < 20 {
            return;
        }
        
        let crash_start = bars.len() * 2 / 3; // Start crash at 2/3 through
        let crash_duration = 5; // 5 bars for crash
        let crash_magnitude = 0.15; // 15% crash
        
        for (i, bar) in bars.iter_mut().enumerate().skip(crash_start).take(crash_duration) {
            let crash_progress = (i - crash_start) as f64 / crash_duration as f64;
            let crash_factor = 1.0 - (crash_magnitude * crash_progress);
            
            bar.open *= crash_factor;
            bar.high *= crash_factor;
            bar.low *= crash_factor;
            bar.close *= crash_factor;
            
            // Increase volume during crash
            bar.volume *= 3.0 + self.rng.gen::<f64>() * 2.0;
        }
    }
    
    fn apply_recovery_pattern(&mut self, bars: &mut [OHLCVBar]) {
        if bars.len() < 20 {
            return;
        }
        
        // First apply a crash
        self.apply_flash_crash(bars);
        
        // Then apply recovery
        let recovery_start = bars.len() * 2 / 3 + 5;
        let recovery_factor = 0.1; // 10% recovery
        
        let bars_len = bars.len();
        for (i, bar) in bars.iter_mut().enumerate().skip(recovery_start) {
            let recovery_progress = (i - recovery_start) as f64 / (bars_len - recovery_start) as f64;
            let recovery_multiplier = 1.0 + (recovery_factor * recovery_progress);
            
            bar.open *= recovery_multiplier;
            bar.high *= recovery_multiplier;
            bar.low *= recovery_multiplier;
            bar.close *= recovery_multiplier;
        }
    }
    
    fn apply_consolidation(&mut self, bars: &mut [OHLCVBar]) {
        if bars.is_empty() {
            return;
        }
        
        let base_price = bars[0].close;
        let consolidation_range = base_price * 0.01; // 1% range
        
        for bar in bars.iter_mut() {
            let target_price = base_price + (self.rng.gen::<f64>() - 0.5) * consolidation_range;
            let adjustment = target_price / bar.close;
            
            bar.open *= adjustment;
            bar.high = bar.open.max(target_price) * (1.0 + self.rng.gen::<f64>() * 0.005);
            bar.low = bar.open.min(target_price) * (1.0 - self.rng.gen::<f64>() * 0.005);
            bar.close = target_price;
            
            // Lower volume during consolidation
            bar.volume *= 0.7 + self.rng.gen::<f64>() * 0.3;
        }
    }
    
    fn introduce_missing_values(&mut self, bars: &mut Vec<OHLCVBar>) {
        let missing_count = (bars.len() as f64 * 0.05) as usize; // 5% missing values
        
        for _ in 0..missing_count {
            if !bars.is_empty() {
                let index = self.rng.gen_range(0..bars.len());
                bars[index].close = f64::NAN;
                bars[index].volume = f64::NAN;
            }
        }
    }
    
    fn introduce_price_outliers(&mut self, bars: &mut [OHLCVBar]) {
        let outlier_count = (bars.len() as f64 * 0.02) as usize; // 2% outliers
        
        for _ in 0..outlier_count {
            if !bars.is_empty() {
                let index = self.rng.gen_range(0..bars.len());
                let outlier_multiplier = if self.rng.gen::<bool>() { 10.0 } else { 0.1 };
                
                bars[index].high *= outlier_multiplier;
                bars[index].close *= outlier_multiplier;
            }
        }
    }
    
    fn introduce_invalid_volumes(&mut self, bars: &mut [OHLCVBar]) {
        let invalid_count = (bars.len() as f64 * 0.03) as usize; // 3% invalid volumes
        
        for _ in 0..invalid_count {
            if !bars.is_empty() {
                let index = self.rng.gen_range(0..bars.len());
                bars[index].volume = if self.rng.gen::<bool>() { 0.0 } else { -100.0 };
            }
        }
    }
    
    fn calculate_reference_features(&self, ohlcv_data: &[OHLCVBar]) -> Result<Features> {
        // Simplified feature calculation for reference data
        let closes: Vec<f64> = ohlcv_data.iter().map(|bar| bar.close).collect();
        
        // Calculate RSI (simplified)
        let rsi = self.calculate_simple_rsi(&closes, 14);
        
        // Calculate moving averages
        let mut moving_averages = HashMap::new();
        moving_averages.insert("sma_20".to_string(), self.calculate_sma(&closes, 20));
        moving_averages.insert("ema_12".to_string(), self.calculate_ema(&closes, 12));
        
        // Calculate momentum
        let momentum = self.calculate_momentum(&closes, 10);
        
        // Calculate volatility
        let volatility = self.calculate_volatility(&closes, 20);
        
        Ok(Features {
            rsi,
            moving_averages,
            momentum,
            volatility,
            custom: HashMap::new(),
        })
    }
    
    fn calculate_simple_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut rsi_values = Vec::new();
        
        for i in period..prices.len() {
            let mut gains = 0.0;
            let mut losses = 0.0;
            
            for j in (i - period + 1)..=i {
                let change = prices[j] - prices[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses -= change;
                }
            }
            
            let avg_gain = gains / period as f64;
            let avg_loss = losses / period as f64;
            
            let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 100.0 };
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            
            rsi_values.push(rsi);
        }
        
        rsi_values
    }
    
    fn calculate_sma(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut sma_values = Vec::new();
        
        for i in period..=prices.len() {
            let sum: f64 = prices[(i - period)..i].iter().sum();
            sma_values.push(sum / period as f64);
        }
        
        sma_values
    }
    
    fn calculate_ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut ema_values = Vec::new();
        let multiplier = 2.0 / (period as f64 + 1.0);
        
        if !prices.is_empty() {
            ema_values.push(prices[0]);
            
            for &price in prices.iter().skip(1) {
                let ema = (price * multiplier) + (ema_values.last().unwrap() * (1.0 - multiplier));
                ema_values.push(ema);
            }
        }
        
        ema_values
    }
    
    fn calculate_momentum(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut momentum_values = Vec::new();
        
        for i in period..prices.len() {
            let momentum = prices[i] - prices[i - period];
            momentum_values.push(momentum);
        }
        
        momentum_values
    }
    
    fn calculate_volatility(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut volatility_values = Vec::new();
        
        for i in period..=prices.len() {
            let window = &prices[(i - period)..i];
            let mean = window.iter().sum::<f64>() / period as f64;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            volatility_values.push(variance.sqrt());
        }
        
        volatility_values
    }
    
    fn generate_reference_signals(&mut self, _ohlcv_data: &[OHLCVBar], _features: &Features) -> Result<Vec<TradingSignal>> {
        // Generate sample reference signals
        let mut signals = Vec::new();
        
        for i in 0..10 {
            signals.push(TradingSignal {
                timestamp: chrono::Utc::now().timestamp() + (i * 300), // 5-minute intervals
                strength: (self.rng.gen::<f64>() - 0.5) * 2.0, // -1 to 1
                confidence: 0.5 + self.rng.gen::<f64>() * 0.5, // 0.5 to 1.0
                signal_type: match i % 3 {
                    0 => SignalType::LDC,
                    1 => SignalType::MR,
                    _ => SignalType::TSMOM,
                },
                metadata: HashMap::new(),
            });
        }
        
        Ok(signals)
    }
    
    fn features_to_hashmap(&self, features: &Features) -> HashMap<String, Vec<f64>> {
        let mut map = HashMap::new();
        map.insert("rsi".to_string(), features.rsi.clone());
        map.insert("momentum".to_string(), features.momentum.clone());
        map.insert("volatility".to_string(), features.volatility.clone());
        
        for (key, values) in &features.moving_averages {
            map.insert(key.clone(), values.clone());
        }
        
        map
    }
    
    fn signals_to_hashmap(&self, signals: &[TradingSignal]) -> HashMap<String, Vec<TradingSignal>> {
        let mut map = HashMap::new();
        map.insert("reference_signals".to_string(), signals.to_vec());
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_data_generator_creation() {
        let config = DataGenConfig {
            market_scenarios: vec!["trending_up".to_string()],
            include_gaps: true,
            include_outliers: true,
            base_price: 50000.0,
            volatility_factor: 0.02,
            random_seed: Some(42),
            output_dir: "test_output".to_string(),
        };
        
        let temp_dir = TempDir::new().unwrap();
        let mut config = config;
        config.output_dir = temp_dir.path().to_string_lossy().to_string();
        
        let generator = TestDataGenerator::new(config).unwrap();
        assert_eq!(generator.config.base_price, 50000.0);
    }
    
    #[test]
    fn test_ohlcv_data_generation() {
        let config = DataGenConfig {
            market_scenarios: vec!["trending_up".to_string()],
            include_gaps: true,
            include_outliers: true,
            base_price: 50000.0,
            volatility_factor: 0.02,
            random_seed: Some(42),
            output_dir: "test_output".to_string(),
        };
        
        let temp_dir = TempDir::new().unwrap();
        let mut config = config;
        config.output_dir = temp_dir.path().to_string_lossy().to_string();
        
        let mut generator = TestDataGenerator::new(config).unwrap();
        let data = generator.generate_ohlcv_data("BTCUSDT", 1, "5m").unwrap();
        
        assert_eq!(data.len(), 12); // 1 hour / 5 minutes = 12 bars
        assert!(data.iter().all(|bar| bar.symbol == "BTCUSDT"));
        assert!(data.iter().all(|bar| bar.interval == "5m"));
    }
    
    #[test]
    fn test_market_scenario_generation() {
        let config = DataGenConfig {
            market_scenarios: vec!["trending_up".to_string()],
            include_gaps: true,
            include_outliers: true,
            base_price: 50000.0,
            volatility_factor: 0.02,
            random_seed: Some(42),
            output_dir: "test_output".to_string(),
        };
        
        let temp_dir = TempDir::new().unwrap();
        let mut config = config;
        config.output_dir = temp_dir.path().to_string_lossy().to_string();
        
        let mut generator = TestDataGenerator::new(config).unwrap();
        let data = generator.generate_market_scenario(MarketScenario::TrendingUp, "BTCUSDT", 1).unwrap();
        
        assert_eq!(data.scenario, MarketScenario::TrendingUp);
        assert!(!data.ohlcv_data.is_empty());
        assert_eq!(data.metadata.symbol, "BTCUSDT");
    }
    
    #[test]
    fn test_edge_cases_generation() {
        let config = DataGenConfig {
            market_scenarios: vec!["trending_up".to_string()],
            include_gaps: true,
            include_outliers: true,
            base_price: 50000.0,
            volatility_factor: 0.02,
            random_seed: Some(42),
            output_dir: "test_output".to_string(),
        };
        
        let temp_dir = TempDir::new().unwrap();
        let mut config = config;
        config.output_dir = temp_dir.path().to_string_lossy().to_string();
        
        let mut generator = TestDataGenerator::new(config).unwrap();
        let edge_cases = generator.generate_edge_cases().unwrap();
        
        assert_eq!(edge_cases.len(), 3); // Missing values, outliers, invalid volumes
        assert!(edge_cases.iter().all(|case| case.metadata.description.contains("Edge case")));
    }
    
    #[test]
    fn test_reference_data_generation() {
        let config = DataGenConfig {
            market_scenarios: vec!["trending_up".to_string()],
            include_gaps: true,
            include_outliers: true,
            base_price: 50000.0,
            volatility_factor: 0.02,
            random_seed: Some(42),
            output_dir: "test_output".to_string(),
        };
        
        let temp_dir = TempDir::new().unwrap();
        let mut config = config;
        config.output_dir = temp_dir.path().to_string_lossy().to_string();
        
        let mut generator = TestDataGenerator::new(config).unwrap();
        let reference_data = generator.generate_reference_data("BTCUSDT").unwrap();
        
        assert!(reference_data.features.contains_key("rsi"));
        assert!(reference_data.features.contains_key("sma_20"));
        assert!(reference_data.signals.contains_key("reference_signals"));
        assert!(!reference_data.performance_metrics.is_empty());
    }
}