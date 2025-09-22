use anyhow::{Result, Context};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use polars::prelude::{RollingOptionsFixedWindow, EWMOptions};
use std::fs::{File, create_dir_all};
use std::path::{Path, PathBuf};
use chrono::Utc;

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
    pub wavetrend_1: Option<f64>,
    pub wavetrend_2: Option<f64>,
    pub cci: Option<f64>,
    pub adx: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signals {
    pub timestamp: i64,
    pub s_mr: Option<f64>,    // Mean Reversion signal
    pub s_tsmom: Option<f64>, // Time Series Momentum signal
}

#[derive(Debug)]
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
                // Wavetrend WT1, WT2
                self.compute_wavetrend1_expr(),
                self.compute_wavetrend2_expr(),
                // CCI
                self.compute_cci_expr(),
                // ADX
                self.compute_adx_expr(),
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
            let wavetrend_1 = if row.0.len() > 12 { row.0[12].extract::<f64>() } else { None };
            let wavetrend_2 = if row.0.len() > 13 { row.0[13].extract::<f64>() } else { None };
            let cci = if row.0.len() > 14 { row.0[14].extract::<f64>() } else { None };
            let adx = if row.0.len() > 15 { row.0[15].extract::<f64>() } else { None };

            features.push(Features {
                timestamp,
                rsi,
                sma_20,
                ema_20,
                std_20,
                zscore_20,
                momentum,
                wavetrend_1,
                wavetrend_2,
                cci,
                adx,
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
        
        // Use Wilder's smoothing (alpha = 1/period) for RSI
        let alpha = 1.0 / period;
        let avg_gains = gains.ewm_mean(EWMOptions { 
            alpha, 
            adjust: false, 
            min_periods: 1, // Start calculating from first period
            ..Default::default() 
        });
        let avg_losses = losses.ewm_mean(EWMOptions { 
            alpha, 
            adjust: false, 
            min_periods: 1, // Start calculating from first period
            ..Default::default() 
        });

        // Calculate RS (Relative Strength)
        let rs = when(avg_losses.clone().gt(lit(1e-10)))
            .then(avg_gains.clone() / avg_losses.clone())
            .otherwise(lit(f64::INFINITY)); // If no losses, RS is infinite (RSI = 100)
            
        // RSI = 100 - (100 / (1 + RS))
        // Handle special cases: if RS is infinite, RSI = 100; if RS is 0, RSI = 0
        when(rs.clone().is_infinite())
            .then(lit(100.0))
            .when(rs.clone().is_nan().or(avg_gains.is_null()).or(avg_losses.is_null()))
            .then(lit(f64::NAN))
            .otherwise(lit(100.0) - (lit(100.0) / (lit(1.0) + rs)))
            .alias("rsi")
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
        // Handle the first row by using 0.0 momentum when prev_close is null
        when(prev_close.clone().is_null().or(prev_close.clone().eq(lit(0.0))))
            .then(lit(0.0))
            .otherwise((col("close") - prev_close.clone()) / prev_close)
            .alias("momentum")
    }

    // Typical price helper: (high+low+close)/3
    fn compute_typical_price_expr(&self) -> Expr {
        (col("high") + col("low") + col("close")) / lit(3.0)
    }

    // Wavetrend WT1
    fn compute_wavetrend1_expr(&self) -> Expr {
        // Parameters often n1=10, n2=21; we reuse ma_period for n1 and rsi_period for n2 by default
        let n1_alpha = 2.0 / (self.ma_period as f64 + 1.0);
        let n2_alpha = 2.0 / (self.rsi_period as f64 + 1.0);
        let tp = self.compute_typical_price_expr();
        let esa = tp.clone().ewm_mean(EWMOptions { alpha: n1_alpha, adjust: false, min_periods: 1, ..Default::default() });
        let d = (tp.clone() - esa.clone()).abs().ewm_mean(EWMOptions {
            alpha: n1_alpha,
            adjust: false,
            min_periods: 1, // Changed from 0 to 1 to avoid division by zero
            bias: false,
            ignore_nulls: false,
        });
        // Add small epsilon to prevent division by zero
        let ci = when(d.clone().gt(lit(1e-8)))
            .then((tp - esa) / (lit(0.015) * d))
            .otherwise(lit(0.0));
        ci.ewm_mean(EWMOptions { alpha: n2_alpha, adjust: false, min_periods: 1, ..Default::default() }).alias("wavetrend_1")
    }

    // Wavetrend WT2 (signal) - compute directly from the same logic as WT1
    fn compute_wavetrend2_expr(&self) -> Expr {
        let n1_alpha = 2.0 / (self.ma_period as f64 + 1.0);
        let n2_alpha = 2.0 / (self.rsi_period as f64 + 1.0);
        let tp = self.compute_typical_price_expr();
        let esa = tp.clone().ewm_mean(EWMOptions { alpha: n1_alpha, adjust: false, min_periods: 1, ..Default::default() });
        let d = (tp.clone() - esa.clone()).abs().ewm_mean(EWMOptions {
            alpha: n1_alpha,
            adjust: false,
            min_periods: 1, // Changed from 0 to 1 to avoid division by zero
            bias: false,
            ignore_nulls: false,
        });
        let ci = when(d.clone().gt(lit(1e-8)))
            .then((tp - esa) / (lit(0.015) * d))
            .otherwise(lit(0.0));
        let wt1 = ci.ewm_mean(EWMOptions { alpha: n2_alpha, adjust: false, min_periods: 1, ..Default::default() });
        wt1.ewm_mean(EWMOptions { alpha: 2.0 / (4.0 + 1.0), adjust: false, min_periods: 1, ..Default::default() }).alias("wavetrend_2")
    }

    // CCI - Commodity Channel Index
    fn compute_cci_expr(&self) -> Expr {
        let options = RollingOptionsFixedWindow {
            window_size: self.ma_period,
            min_periods: self.ma_period.min(10), // Require at least 10 periods for reasonable CCI
            ..Default::default()
        };
        
        let tp = self.compute_typical_price_expr();
        let tp_sma = tp.clone().rolling_mean(options.clone());
        
        // Mean Deviation calculation (as per standard CCI formula):
        // 1. For each period: |TP - SMA(TP)|
        // 2. Take the mean of these absolute deviations over the window
        let mean_deviation = (tp.clone() - tp_sma.clone()).abs().rolling_mean(options);
        
        // CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
        // The constant 0.015 scales values to typically fall within ±100
        when(tp_sma.clone().is_not_null()
            .and(mean_deviation.clone().is_not_null())
            .and(mean_deviation.clone().gt(lit(1e-10))))
            .then((tp - tp_sma) / (lit(0.015) * mean_deviation))
            .otherwise(lit(f64::NAN))
            .alias("cci")
    }

    // ADX - Average Directional Index (following standard Wilder's method)
    fn compute_adx_expr(&self) -> Expr {
        let prev_high = col("high").shift(lit(1));
        let prev_low = col("low").shift(lit(1));
        let prev_close = col("close").shift(lit(1));

        // True Range calculation: max(H-L, |H-PC|, |L-PC|)
        let tr1 = col("high") - col("low");
        let tr2 = (col("high") - prev_close.clone()).abs();
        let tr3 = (col("low") - prev_close.clone()).abs();
        
        // Handle null values in the first row
        let tr = when(prev_close.clone().is_null())
            .then(tr1.clone()) // For first row, just use high-low
            .otherwise(
                // Calculate True Range: max(H-L, |H-PC|, |L-PC|)
                when(tr1.clone().gt(tr2.clone()))
                    .then(when(tr1.clone().gt(tr3.clone()))
                        .then(tr1)
                        .otherwise(tr3.clone()))
                    .otherwise(when(tr2.clone().gt(tr3.clone()))
                        .then(tr2)
                        .otherwise(tr3))
            );

        // Directional Movement calculation
        // +DM = Current High - Previous High (when > 0 and > -DM)
        // -DM = Previous Low - Current Low (when > 0 and > +DM)
        let up_move = when(prev_high.clone().is_null())
            .then(lit(0.0))
            .otherwise(col("high") - prev_high.clone());
        let down_move = when(prev_low.clone().is_null())
            .then(lit(0.0))
            .otherwise(prev_low.clone() - col("low"));
            
        // Apply the standard +DM/-DM rules
        let plus_dm = when(up_move.clone().gt(down_move.clone()).and(up_move.clone().gt(lit(0.0))))
            .then(up_move.clone())
            .otherwise(lit(0.0));
        let minus_dm = when(down_move.clone().gt(up_move.clone()).and(down_move.clone().gt(lit(0.0))))
            .then(down_move.clone())
            .otherwise(lit(0.0));

        // Wilder's smoothing approximation using EWM
        // Note: True Wilder's smoothing would be: New = Old - (Old/Period) + Current
        // EWM with alpha=1/period is a close approximation
        let alpha = 1.0 / self.ma_period as f64;
        let tr_s = tr.ewm_mean(EWMOptions { 
            alpha, 
            adjust: false, 
            min_periods: self.ma_period, 
            ..Default::default() 
        });
        let plus_dm_s = plus_dm.ewm_mean(EWMOptions { 
            alpha, 
            adjust: false, 
            min_periods: self.ma_period, 
            ..Default::default() 
        });
        let minus_dm_s = minus_dm.ewm_mean(EWMOptions { 
            alpha, 
            adjust: false, 
            min_periods: self.ma_period, 
            ..Default::default() 
        });

        // Directional Indicators: +DI = (+DM_smoothed / TR_smoothed) * 100
        let di_plus = when(tr_s.clone().gt(lit(1e-10)))
            .then(lit(100.0) * plus_dm_s.clone() / tr_s.clone())
            .otherwise(lit(0.0));
        let di_minus = when(tr_s.clone().gt(lit(1e-10)))
            .then(lit(100.0) * minus_dm_s.clone() / tr_s)
            .otherwise(lit(0.0));
            
        // Directional Index: DX = |+DI - -DI| / (+DI + -DI) * 100
        let di_sum = di_plus.clone() + di_minus.clone();
        let dx = when(di_sum.clone().gt(lit(1e-10)))
            .then(lit(100.0) * (di_plus - di_minus).abs() / di_sum)
            .otherwise(lit(0.0));
            
        // ADX is the smoothed DX using the same Wilder's smoothing
        dx.ewm_mean(EWMOptions { 
            alpha, 
            adjust: false, 
            min_periods: self.ma_period, 
            ..Default::default() 
        }).alias("adx")
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

    /// Create partitioned directory structure for data storage
    pub fn create_partitioned_path(&self, base_path: &str, symbol: &str, interval: &str) -> Result<PathBuf> {
        let now = Utc::now();
        let date_str = now.format("%Y-%m-%d").to_string();
        
        let partition_path = Path::new(base_path)
            .join(format!("symbol={}", symbol))
            .join(format!("date={}", date_str))
            .join(format!("interval={}", interval));
        
        create_dir_all(&partition_path)?;
        Ok(partition_path)
    }

    /// Save OHLCV data to partitioned Parquet storage
    pub fn save_ohlcv_partitioned(&self, data: &[OHLCV], base_path: &str, symbol: &str, interval: &str) -> Result<String> {
        let partition_path = self.create_partitioned_path(base_path, symbol, interval)?;
        let file_path = partition_path.join("ohlcv.parquet");
        
        let df = self.ohlcv_to_dataframe(data)?;
        let mut file = File::create(&file_path)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        
        Ok(file_path.to_string_lossy().to_string())
    }

    /// Save features to partitioned Parquet storage
    pub fn save_features_partitioned(&self, features: &[Features], base_path: &str, symbol: &str, interval: &str) -> Result<String> {
        let partition_path = self.create_partitioned_path(base_path, symbol, interval)?;
        let file_path = partition_path.join("features.parquet");
        
        // Convert features to DataFrame
        let timestamps: Vec<i64> = features.iter().map(|f| f.timestamp).collect();
        let rsi: Vec<Option<f64>> = features.iter().map(|f| f.rsi).collect();
        let sma_20: Vec<Option<f64>> = features.iter().map(|f| f.sma_20).collect();
        let ema_20: Vec<Option<f64>> = features.iter().map(|f| f.ema_20).collect();
        let std_20: Vec<Option<f64>> = features.iter().map(|f| f.std_20).collect();
        let zscore_20: Vec<Option<f64>> = features.iter().map(|f| f.zscore_20).collect();
        let momentum: Vec<Option<f64>> = features.iter().map(|f| f.momentum).collect();
        let wavetrend_1: Vec<Option<f64>> = features.iter().map(|f| f.wavetrend_1).collect();
        let wavetrend_2: Vec<Option<f64>> = features.iter().map(|f| f.wavetrend_2).collect();
        let cci: Vec<Option<f64>> = features.iter().map(|f| f.cci).collect();
        let adx: Vec<Option<f64>> = features.iter().map(|f| f.adx).collect();

        let df = df![
            "timestamp" => timestamps,
            "rsi" => rsi,
            "sma_20" => sma_20,
            "ema_20" => ema_20,
            "std_20" => std_20,
            "zscore_20" => zscore_20,
            "momentum" => momentum,
            "wavetrend_1" => wavetrend_1,
            "wavetrend_2" => wavetrend_2,
            "cci" => cci,
            "adx" => adx,
        ]?;

        let mut file = File::create(&file_path)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        
        Ok(file_path.to_string_lossy().to_string())
    }

    /// Save signals to partitioned Parquet storage
    pub fn save_signals_partitioned(&self, signals: &[Signals], base_path: &str, symbol: &str, interval: &str) -> Result<String> {
        let partition_path = self.create_partitioned_path(base_path, symbol, interval)?;
        let file_path = partition_path.join("signals.parquet");
        
        // Convert signals to DataFrame
        let timestamps: Vec<i64> = signals.iter().map(|s| s.timestamp).collect();
        let s_mr: Vec<Option<f64>> = signals.iter().map(|s| s.s_mr).collect();
        let s_tsmom: Vec<Option<f64>> = signals.iter().map(|s| s.s_tsmom).collect();

        let df = df![
            "timestamp" => timestamps,
            "s_mr" => s_mr,
            "s_tsmom" => s_tsmom,
        ]?;

        let mut file = File::create(&file_path)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        
        Ok(file_path.to_string_lossy().to_string())
    }

    /// Process and save all data with partitioning
    pub fn process_and_save_partitioned(&self, data: &[OHLCV], base_path: &str, symbol: &str, interval: &str) -> Result<(String, String, String)> {
        // Validate input data first
        self.validate_ohlcv_data(data)?;
        
        // Compute features
        let features = self.compute_features(data)?;
        
        // Validate computed features
        self.validate_features(&features)?;
        
        // Generate signals
        let mut signals = self.generate_signals(&features)?;
        
        // Apply filtering
        self.apply_signal_filtering(&mut signals, 0.1, 0.01);
        
        // Save all data with partitioning
        let ohlcv_path = self.save_ohlcv_partitioned(data, base_path, symbol, interval)?;
        let features_path = self.save_features_partitioned(&features, base_path, symbol, interval)?;
        let signals_path = self.save_signals_partitioned(&signals, base_path, symbol, interval)?;
        
        Ok((ohlcv_path, features_path, signals_path))
    }

    /// Validate OHLCV data for consistency and quality
    pub fn validate_ohlcv_data(&self, data: &[OHLCV]) -> Result<()> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("OHLCV data is empty"));
        }

        for (i, ohlcv) in data.iter().enumerate() {
            // Validate timestamp
            if ohlcv.timestamp <= 0 {
                return Err(anyhow::anyhow!("Invalid timestamp at index {}: {}", i, ohlcv.timestamp));
            }

            // Validate price data
            if ohlcv.open <= 0.0 || ohlcv.high <= 0.0 || ohlcv.low <= 0.0 || ohlcv.close <= 0.0 {
                return Err(anyhow::anyhow!("Invalid price data at index {}: prices must be positive", i));
            }

            // Validate OHLC relationships
            if ohlcv.high < ohlcv.low {
                return Err(anyhow::anyhow!("Invalid OHLC data at index {}: high ({}) < low ({})", i, ohlcv.high, ohlcv.low));
            }

            if ohlcv.high < ohlcv.open || ohlcv.high < ohlcv.close {
                return Err(anyhow::anyhow!("Invalid OHLC data at index {}: high ({}) is not the highest price", i, ohlcv.high));
            }

            if ohlcv.low > ohlcv.open || ohlcv.low > ohlcv.close {
                return Err(anyhow::anyhow!("Invalid OHLC data at index {}: low ({}) is not the lowest price", i, ohlcv.low));
            }

            // Validate volume
            if ohlcv.volume < 0.0 {
                return Err(anyhow::anyhow!("Invalid volume at index {}: volume ({}) cannot be negative", i, ohlcv.volume));
            }

            // Check for reasonable price ranges (prevent extreme outliers)
            let price_avg = (ohlcv.open + ohlcv.high + ohlcv.low + ohlcv.close) / 4.0;
            if price_avg > 1_000_000.0 || price_avg < 0.0001 {
                return Err(anyhow::anyhow!("Suspicious price range at index {}: average price {}", i, price_avg));
            }
        }

        // Validate timestamp ordering
        for i in 1..data.len() {
            if data[i].timestamp <= data[i-1].timestamp {
                return Err(anyhow::anyhow!("Non-increasing timestamps at indices {} and {}: {} <= {}", 
                    i-1, i, data[i-1].timestamp, data[i].timestamp));
            }
        }

        Ok(())
    }

    /// Validate computed features for reasonable ranges
    pub fn validate_features(&self, features: &[Features]) -> Result<()> {
        for (i, feature) in features.iter().enumerate() {
            // Validate RSI range (0-100)
            if let Some(rsi) = feature.rsi {
                if rsi < 0.0 || rsi > 100.0 {
                    return Err(anyhow::anyhow!("Invalid RSI at index {}: {} (must be 0-100)", i, rsi));
                }
            }

            // Validate moving averages are positive
            if let Some(sma) = feature.sma_20 {
                if sma <= 0.0 {
                    return Err(anyhow::anyhow!("Invalid SMA at index {}: {} (must be positive)", i, sma));
                }
            }

            if let Some(ema) = feature.ema_20 {
                if ema <= 0.0 {
                    return Err(anyhow::anyhow!("Invalid EMA at index {}: {} (must be positive)", i, ema));
                }
            }

            // Validate standard deviation is non-negative
            if let Some(std) = feature.std_20 {
                if std < 0.0 {
                    return Err(anyhow::anyhow!("Invalid standard deviation at index {}: {} (must be non-negative)", i, std));
                }
            }

            // Validate z-score is reasonable (not infinite or NaN)
            if let Some(zscore) = feature.zscore_20 {
                if zscore.is_infinite() || zscore.is_nan() {
                    return Err(anyhow::anyhow!("Invalid z-score at index {}: {} (infinite or NaN)", i, zscore));
                }
                if zscore.abs() > 10.0 {
                    return Err(anyhow::anyhow!("Extreme z-score at index {}: {} (abs value > 10)", i, zscore));
                }
            }

            // Validate momentum is reasonable
            if let Some(momentum) = feature.momentum {
                if momentum.is_infinite() || momentum.is_nan() {
                    return Err(anyhow::anyhow!("Invalid momentum at index {}: {} (infinite or NaN)", i, momentum));
                }
                if momentum.abs() > 1.0 {
                    return Err(anyhow::anyhow!("Extreme momentum at index {}: {} (abs value > 1.0)", i, momentum));
                }
            }

            // Validate Wavetrend values (allow NaN for early periods)
            if let Some(wt1) = feature.wavetrend_1 {
                if wt1.is_infinite() {
                    return Err(anyhow::anyhow!("Invalid Wavetrend1 at index {}: {} (infinite)", i, wt1));
                }
                // Allow NaN for early periods where calculation might not be possible
            }

            if let Some(wt2) = feature.wavetrend_2 {
                if wt2.is_infinite() {
                    return Err(anyhow::anyhow!("Invalid Wavetrend2 at index {}: {} (infinite)", i, wt2));
                }
                // Allow NaN for early periods where calculation might not be possible
            }

            // Validate CCI is reasonable (allow NaN for early periods)
            if let Some(cci) = feature.cci {
                if cci.is_infinite() {
                    return Err(anyhow::anyhow!("Invalid CCI at index {}: {} (infinite)", i, cci));
                }
                if !cci.is_nan() && cci.abs() > 1000.0 {
                    return Err(anyhow::anyhow!("Extreme CCI at index {}: {} (abs value > 1000)", i, cci));
                }
            }

            // Validate ADX range (0-100, allow NaN for early periods, allow small floating point errors)
            if let Some(adx) = feature.adx {
                if !adx.is_nan() && (adx < -0.001 || adx > 100.001) {
                    return Err(anyhow::anyhow!("Invalid ADX at index {}: {} (must be 0-100)", i, adx));
                }
            }
        }

        Ok(())
    }

    /// Enhanced error handling for feature computation
    pub fn compute_features_safe(&self, data: &[OHLCV]) -> Result<Vec<Features>> {
        // Validate input data
        self.validate_ohlcv_data(data)
            .context("Failed to validate OHLCV data")?;

        if data.len() < self.window_size {
            return Err(anyhow::anyhow!(
                "Insufficient data: {} samples, need at least {} for window size {}", 
                data.len(), self.window_size, self.window_size
            ));
        }

        // Compute features with error handling
        let features = self.compute_features(data)
            .context("Failed to compute features")?;

        // Validate computed features
        self.validate_features(&features)
            .context("Failed to validate computed features")?;

        Ok(features)
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
            .filter(|f| f.rsi.is_some() || f.sma_20.is_some() || f.wavetrend_1.is_some() || f.cci.is_some() || f.adx.is_some())
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
    
    #[test]
    fn test_partitioned_storage() {
        let pipeline = FeaturePipeline::new(20);
        let data = create_sample_data();
        
        // Test partitioned path creation
        let partition_path = pipeline.create_partitioned_path("test_data", "BTCUSDT", "5m").unwrap();
        assert!(partition_path.to_string_lossy().contains("symbol=BTCUSDT"));
        assert!(partition_path.to_string_lossy().contains("interval=5m"));
        assert!(partition_path.to_string_lossy().contains("date="));
        
        // Test saving OHLCV data
        let ohlcv_path = pipeline.save_ohlcv_partitioned(&data, "test_data", "BTCUSDT", "5m").unwrap();
        assert!(ohlcv_path.contains("ohlcv.parquet"));
        assert!(std::path::Path::new(&ohlcv_path).exists());
        
        // Test saving features
        let features = pipeline.compute_features(&data).unwrap();
        let features_path = pipeline.save_features_partitioned(&features, "test_data", "BTCUSDT", "5m").unwrap();
        assert!(features_path.contains("features.parquet"));
        assert!(std::path::Path::new(&features_path).exists());
        
        // Test saving signals
        let signals = pipeline.generate_signals(&features).unwrap();
        let signals_path = pipeline.save_signals_partitioned(&signals, "test_data", "BTCUSDT", "5m").unwrap();
        assert!(signals_path.contains("signals.parquet"));
        assert!(std::path::Path::new(&signals_path).exists());
        
        // Clean up test files
        let _ = std::fs::remove_dir_all("test_data");
    }
    
    #[test]
    fn test_process_and_save_partitioned() {
        let pipeline = FeaturePipeline::new(20);
        let data = create_sample_data();
        
        let (ohlcv_path, features_path, signals_path) = pipeline.process_and_save_partitioned(
            &data, 
            "test_data_full", 
            "ETHUSDT", 
            "1m"
        ).unwrap();
        
        // Verify all files were created
        assert!(std::path::Path::new(&ohlcv_path).exists());
        assert!(std::path::Path::new(&features_path).exists());
        assert!(std::path::Path::new(&signals_path).exists());
        
        // Verify paths contain correct partitioning
        assert!(ohlcv_path.contains("symbol=ETHUSDT"));
        assert!(features_path.contains("symbol=ETHUSDT"));
        assert!(signals_path.contains("symbol=ETHUSDT"));
        
        // Clean up test files
        let _ = std::fs::remove_dir_all("test_data_full");
    }
    
    #[test]
    fn test_ohlcv_validation() {
        let pipeline = FeaturePipeline::new(20);
        
        // Test valid data
        let valid_data = create_sample_data();
        assert!(pipeline.validate_ohlcv_data(&valid_data).is_ok());
        
        // Test empty data
        let empty_data = vec![];
        assert!(pipeline.validate_ohlcv_data(&empty_data).is_err());
        
        // Test invalid OHLC data
        let mut invalid_data = create_sample_data();
        invalid_data[0].high = invalid_data[0].low - 1.0; // high < low
        assert!(pipeline.validate_ohlcv_data(&invalid_data).is_err());
        
        // Test negative prices
        let mut invalid_data = create_sample_data();
        invalid_data[0].open = -1.0;
        assert!(pipeline.validate_ohlcv_data(&invalid_data).is_err());
        
        // Test negative volume
        let mut invalid_data = create_sample_data();
        invalid_data[0].volume = -1.0;
        assert!(pipeline.validate_ohlcv_data(&invalid_data).is_err());
        
        // Test non-increasing timestamps
        let mut invalid_data = create_sample_data();
        invalid_data[1].timestamp = invalid_data[0].timestamp - 1;
        assert!(pipeline.validate_ohlcv_data(&invalid_data).is_err());
    }
    
    #[test]
    fn test_feature_validation() {
        let pipeline = FeaturePipeline::new(20);
        let data = create_sample_data();
        let features = pipeline.compute_features(&data).unwrap();
        
        // Valid features should pass
        assert!(pipeline.validate_features(&features).is_ok());
        
        // Test invalid RSI
        let mut invalid_features = features.clone();
        invalid_features[0].rsi = Some(150.0); // RSI > 100
        assert!(pipeline.validate_features(&invalid_features).is_err());
        
        // Test invalid SMA
        let mut invalid_features = features.clone();
        invalid_features[0].sma_20 = Some(-1.0); // Negative SMA
        assert!(pipeline.validate_features(&invalid_features).is_err());
        
        // Test invalid z-score
        let mut invalid_features = features.clone();
        invalid_features[0].zscore_20 = Some(f64::INFINITY);
        assert!(pipeline.validate_features(&invalid_features).is_err());
    }
    
    #[test]
    fn test_compute_features_safe() {
        let pipeline = FeaturePipeline::new(20);
        let data = create_sample_data();
        
        // Valid data should work
        let features = pipeline.compute_features_safe(&data).unwrap();
        assert!(!features.is_empty());
        
        // Insufficient data should fail
        let insufficient_data = &data[..10]; // Only 10 samples, need 20
        assert!(pipeline.compute_features_safe(insufficient_data).is_err());
        
        // Invalid data should fail
        let mut invalid_data = data.clone();
        invalid_data[0].high = invalid_data[0].low - 1.0;
        assert!(pipeline.compute_features_safe(&invalid_data).is_err());
    }
}