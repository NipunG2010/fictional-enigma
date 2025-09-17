use anyhow::Result;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use polars::prelude::{RollingOptionsFixedWindow, EWMOptions};
use std::fs::File;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Features {
    pub timestamp: i64,
    pub rsi: Option<f64>,
    pub sma_20: Option<f64>,
    pub ema_20: Option<f64>,
    pub std_20: Option<f64>,
    pub zscore_20: Option<f64>,
    pub momentum: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signals {
    pub timestamp: i64,
    pub s_mr: Option<f64>,    // Mean Reversion signal
    pub s_tsmom: Option<f64>, // Time Series Momentum signal
}

pub struct FeaturePipeline {
    window_size: usize,
    rsi_period: usize,
    ma_period: usize,
}

impl FeaturePipeline {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            rsi_period: 14,
            ma_period: 20,
        }
    }

    pub fn with_periods(window_size: usize, rsi_period: usize, ma_period: usize) -> Self {
        Self {
            window_size,
            rsi_period,
            ma_period,
        }
    }
    // In read_csv
    pub fn read_csv(&self, file_path: &str) -> Result<DataFrame> {
        Ok(CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(file_path.into()))?
            .finish()?
        )
    }

    // In read_parquet
    pub fn read_parquet(&self, file_path: &str) -> Result<DataFrame> {
        let r = File::open(file_path).unwrap();
        let reader = ParquetReader::new(r);
        Ok(reader.finish()?)
    }

    /// Convert OHLCV slice to Polars DataFrame
    pub fn ohlcv_to_dataframe(&self, data: &[OHLCV]) -> Result<DataFrame> {
        let timestamps: Vec<i64> = data.iter().map(|x| x.timestamp).collect();
        let opens: Vec<f64> = data.iter().map(|x| x.open).collect();
        let highs: Vec<f64> = data.iter().map(|x| x.high).collect();
        let lows: Vec<f64> = data.iter().map(|x| x.low).collect();
        let closes: Vec<f64> = data.iter().map(|x| x.close).collect();
        let volumes: Vec<f64> = data.iter().map(|x| x.volume).collect();

        let df = df! [
            "timestamp" => timestamps,
            "open" => opens,
            "high" => highs,
            "low" => lows,
            "close" => closes,
            "volume" => volumes,
        ]?;

        Ok(df)
    }

    /// Compute all features using Polars lazy evaluation
    pub fn compute_features_lazy(&self, df: DataFrame) -> Result<DataFrame> {
        let lazy_df = df.lazy();

        let features_df = lazy_df
            .with_columns([
                // RSI calculation
                self.compute_rsi_expr(),
                // Simple Moving Average
                self.compute_sma_expr(),
                // Exponential Moving Average
                self.compute_ema_expr(),
                // Standard Deviation
                self.compute_std_expr(),
                // Z-Score
                self.compute_zscore_expr(),
                // Momentum
                self.compute_momentum_expr(),
            ])
            .collect()?;

        Ok(features_df)
    }

    /// Compute features from OHLCV data slice
    pub fn compute_features(&self, data: &[OHLCV]) -> Result<Vec<Features>> {
        if data.len() < self.window_size {
            return Ok(vec![]);
        }

        let df = self.ohlcv_to_dataframe(data)?;
        let features_df = self.compute_features_lazy(df)?;

        let mut features = Vec::new();
        for i in 0..features_df.height() {
            let row = features_df.get_row(i).unwrap();
            let timestamp = row.0[0].extract::<i64>().unwrap();
            let rsi = row.0[6].extract::<f64>();
            let sma_20 = row.0[7].extract::<f64>();
            let ema_20 = row.0[8].extract::<f64>();
            let std_20 = row.0[9].extract::<f64>();
            let zscore_20 = row.0[10].extract::<f64>();
            let momentum = row.0[11].extract::<f64>();

            features.push(Features {
                timestamp,
                rsi,
                sma_20,
                ema_20,
                std_20,
                zscore_20,
                momentum,
            });
        }

        Ok(features)
    }

    /// Compute RSI using Polars expressions
    // In compute_rsi_expr
    fn compute_rsi_expr(&self) -> Expr {
        let period = self.rsi_period as f64;
        // Use shift() to get price changes
        let price_change = col("close") - col("close").shift(lit(1));

        let gains = when(price_change.clone().gt(lit(0.0))).then(price_change.clone()).otherwise(lit(0.0));
        let losses = when(price_change.clone().lt(lit(0.0))).then(-price_change).otherwise(lit(0.0));
        // Use .ewm(options).mean()
        let avg_gains = gains.ewm_mean(EWMOptions { alpha: 1.0 / period, adjust: false, min_periods: 1, ..Default::default() });
        let avg_losses = losses.ewm_mean(EWMOptions { alpha: 1.0 / period, adjust: false, min_periods: 1, ..Default::default() });

        let rs = avg_gains / avg_losses;
        (lit(100.0) - (lit(100.0) / (lit(1.0) + rs))).alias("rsi")
    }

    fn compute_sma_expr(&self) -> Expr {
        let options = RollingOptionsFixedWindow {
            window_size: self.ma_period, // usize, not Duration
            ..Default::default()
        };
    
        col("close").rolling_mean(options).alias("sma_20")
    }
    
    fn compute_ema_expr(&self) -> Expr {
        // Use ewm_mean with EWMOptions
        col("close")
            .ewm_mean(EWMOptions { 
                alpha: 2.0 / (self.ma_period as f64 + 1.0), 
                adjust: false, 
                min_periods: 1, 
                ..Default::default() 
            })
            .alias("ema_20")
    }

    // In compute_std_expr
    fn compute_std_expr(&self) -> Expr {
        let options = RollingOptionsFixedWindow {
            window_size: self.ma_period,
            ..Default::default()
        };
    
        col("close").rolling_std(options).alias("std_20")
    }
    // In compute_zscore_expr
    fn compute_zscore_expr(&self) -> Expr {
        let options = RollingOptionsFixedWindow {
            window_size: self.ma_period,
            ..Default::default()
        };
    
        ((col("close") - col("close").rolling_mean(options.clone()))
            / col("close").rolling_std(options))
            .alias("zscore_20")
    }
    // In compute_momentum_expr
    fn compute_momentum_expr(&self) -> Expr {
        // Calculate percentage change manually using shift
        let prev_close = col("close").shift(lit(1));
        ((col("close") - prev_close.clone()) / prev_close).alias("momentum")
    }
    /// Generate Mean Reversion signal
    pub fn generate_mr_signal(&self, features: &[Features]) -> Result<Vec<Signals>> {
        let mut signals = Vec::new();
        
        for feature in features {
            let s_mr = if let (Some(zscore), Some(std)) = (feature.zscore_20, feature.std_20) {
                // Mean reversion signal: negative z-score suggests oversold (buy), positive suggests overbought (sell)
                // Scale by standard deviation for normalization
                Some(-zscore / std.max(1e-8)) // Negative because we want to buy when oversold
            } else {
                None
            };

            signals.push(Signals {
                timestamp: feature.timestamp,
                s_mr,
                s_tsmom: None,
            });
        }

        Ok(signals)
    }

    /// Generate Time Series Momentum signal
    pub fn generate_tsmom_signal(&self, features: &[Features]) -> Result<Vec<Signals>> {
        let mut signals = Vec::new();
        
        for feature in features {
            let s_tsmom = if let Some(momentum) = feature.momentum {
                // Time series momentum: positive momentum suggests continuation (buy), negative suggests reversal (sell)
                Some(momentum)
            } else {
                None
            };

            signals.push(Signals {
                timestamp: feature.timestamp,
                s_mr: None,
                s_tsmom,
            });
        }

        Ok(signals)
    }

    /// Generate both MR and TSMOM signals
    pub fn generate_signals(&self, features: &[Features]) -> Result<Vec<Signals>> {
        let mut signals = Vec::new();
        
        for feature in features {
            let s_mr = if let (Some(zscore), Some(std)) = (feature.zscore_20, feature.std_20) {
                Some(-zscore / std.max(1e-8))
            } else {
                None
            };

            let s_tsmom = feature.momentum;

            signals.push(Signals {
                timestamp: feature.timestamp,
                s_mr,
                s_tsmom,
            });
        }

        Ok(signals)
    }

    /// Apply signal thresholding and filtering
    pub fn apply_signal_filtering(&self, signals: &mut [Signals], mr_threshold: f64, tsmom_threshold: f64) {
        for signal in signals.iter_mut() {
            if let Some(s_mr) = signal.s_mr {
                if s_mr.abs() < mr_threshold {
                    signal.s_mr = Some(0.0);
                }
            }
            
            if let Some(s_tsmom) = signal.s_tsmom {
                if s_tsmom.abs() < tsmom_threshold {
                    signal.s_tsmom = Some(0.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_sample_data() -> Vec<OHLCV> {
        let mut data = Vec::new();
        let base_price = 100.0;
        
        for i in 0..50 {
            let price = base_price + (i as f64 * 0.1) + (i as f64 * 0.05).sin();
            data.push(OHLCV {
                timestamp: 1000 + i * 300, // 5-minute intervals
                open: price,
                high: price * 1.02,
                low: price * 0.98,
                close: price * 1.01,
                volume: 1000.0 + i as f64 * 10.0,
            });
        }
        data
    }
    
    #[test]
    fn test_feature_pipeline_creation() {
        let pipeline = FeaturePipeline::new(20);
        assert_eq!(pipeline.window_size, 20);
        assert_eq!(pipeline.rsi_period, 14);
        assert_eq!(pipeline.ma_period, 20);
    }
    
    #[test]
    fn test_feature_pipeline_with_periods() {
        let pipeline = FeaturePipeline::with_periods(30, 21, 25);
        assert_eq!(pipeline.window_size, 30);
        assert_eq!(pipeline.rsi_period, 21);
        assert_eq!(pipeline.ma_period, 25);
    }
    
    #[test]
    fn test_ohlcv_to_dataframe() {
        let pipeline = FeaturePipeline::new(20);
        let data = create_sample_data();
        let df = pipeline.ohlcv_to_dataframe(&data).unwrap();
        
        assert_eq!(df.height(), 50);
        assert_eq!(df.width(), 6);
    }
    
    #[test]
    fn test_feature_computation() {
        let pipeline = FeaturePipeline::new(20);
        let data = create_sample_data();
        
        let features = pipeline.compute_features(&data).unwrap();
        
        // Should have features for all data points
        assert_eq!(features.len(), 50);
        
        // Check that we have some valid features after the window period
        let valid_features: Vec<_> = features.iter()
            .filter(|f| f.rsi.is_some() || f.sma_20.is_some())
            .collect();
        
        // Should have valid features for most data points after the initial window
        assert!(valid_features.len() > 30);
    }
    
    #[test]
    fn test_signal_generation() {
        let pipeline = FeaturePipeline::new(20);
        let data = create_sample_data();
        let features = pipeline.compute_features(&data).unwrap();
        
        let signals = pipeline.generate_signals(&features).unwrap();
        
        assert_eq!(signals.len(), features.len());
        
        // Check that we have some valid signals
        let valid_signals: Vec<_> = signals.iter()
            .filter(|s| s.s_mr.is_some() || s.s_tsmom.is_some())
            .collect();
        
        assert!(valid_signals.len() > 0);
    }
    
    #[test]
    fn test_signal_filtering() {
        let pipeline = FeaturePipeline::new(20);
        let data = create_sample_data();
        let features = pipeline.compute_features(&data).unwrap();
        let mut signals = pipeline.generate_signals(&features).unwrap();
        
        // Apply filtering
        pipeline.apply_signal_filtering(&mut signals, 0.1, 0.01);
        
        // Check that small signals are zeroed out
        for signal in &signals {
            if let Some(s_mr) = signal.s_mr {
                if s_mr.abs() < 0.1 {
                    assert_eq!(s_mr, 0.0);
                }
            }
            if let Some(s_tsmom) = signal.s_tsmom {
                if s_tsmom.abs() < 0.01 {
                    assert_eq!(s_tsmom, 0.0);
                }
            }
        }
    }
    
    #[test]
    fn test_empty_data() {
        let pipeline = FeaturePipeline::new(20);
        let data = vec![];
        
        let features = pipeline.compute_features(&data).unwrap();
        assert_eq!(features.len(), 0);
    }
    
    #[test]
    fn test_insufficient_data() {
        let pipeline = FeaturePipeline::new(20);
        let data = create_sample_data()[..10].to_vec(); // Only 10 data points
        
        let features = pipeline.compute_features(&data).unwrap();
        assert_eq!(features.len(), 0); // Should return empty because window_size is 20
    }
}