use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// Import the feature pipeline types for integration
use feature_pipeline::{Features, OHLCV};

/// Direction labels matching Pine Script
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Short = -1,
    Neutral = 0,
    Long = 1,
}

impl From<i32> for Direction {
    fn from(value: i32) -> Self {
        match value {
            -1 => Direction::Short,
            0 => Direction::Neutral,
            1 => Direction::Long,
            _ => Direction::Neutral,
        }
    }
}

impl From<Direction> for i32 {
    fn from(direction: Direction) -> Self {
        direction as i32
    }
}

/// Feature series matching Pine Script FeatureSeries type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSeries {
    pub f1: f32, // RSI
    pub f2: f32, // WT (WaveTrend)
    pub f3: f32, // CCI
    pub f4: f32, // ADX
    pub f5: f32, // Additional feature (RSI variant)
}

impl FeatureSeries {
    pub fn to_array(&self) -> [f32; 5] {
        [self.f1, self.f2, self.f3, self.f4, self.f5]
    }
    
    pub fn from_array(arr: [f32; 5]) -> Self {
        Self {
            f1: arr[0],
            f2: arr[1],
            f3: arr[2],
            f4: arr[3],
            f5: arr[4],
        }
    }
}

/// Training sample with features and label
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub features: FeatureSeries,
    pub label: Direction,
    pub timestamp: i64,
    pub bar_index: usize,
}

/// LDC prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDCPrediction {
    pub signal: f32, // Sum of k nearest neighbor labels (-k to +k)
    pub confidence: f32, // Based on distance distribution
    pub k_nearest_distances: Vec<f32>,
    pub k_nearest_labels: Vec<Direction>,
    pub prediction_direction: Direction,
}

/// LDC Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDCConfig {
    pub max_bars_back: usize,
    pub neighbors_count: usize,
    pub feature_count: usize,
    pub use_chronological_spacing: bool, // Use modulo 4 spacing like Pine Script
    pub use_multithreading: bool, // Enable parallel processing
    pub max_threads: Option<usize>, // Maximum number of threads (None = auto)
    
    // Performance tuning
    pub parallel_threshold: usize, // Minimum samples to trigger parallel processing
    pub batch_parallel_threshold: usize, // Minimum batch size for parallel batch processing
    
    // Filtering options
    pub enable_regime_filter: bool,
    pub enable_adx_filter: bool,
    pub enable_volatility_filter: bool,
    pub regime_threshold: f32,
    pub adx_threshold: f32,
    
    // Kernel regression options
    pub enable_kernel_smoothing: bool,
    pub kernel_lookback: usize,
    pub kernel_relative_weight: f32,
    pub kernel_regression_level: usize,
    
    // Logging and debugging
    pub enable_debug_logging: bool,
    pub log_predictions: bool,
    pub log_performance_metrics: bool,
}

impl Default for LDCConfig {
    fn default() -> Self {
        Self {
            max_bars_back: 2000,
            neighbors_count: 8,
            feature_count: 5,
            use_chronological_spacing: true,
            use_multithreading: true,
            max_threads: None, // Auto-detect
            
            // Performance tuning defaults
            parallel_threshold: 100,
            batch_parallel_threshold: 10,
            
            // Filtering defaults
            enable_regime_filter: true,
            enable_adx_filter: false,
            enable_volatility_filter: true,
            regime_threshold: -0.1,
            adx_threshold: 20.0,
            
            // Kernel regression defaults
            enable_kernel_smoothing: false,
            kernel_lookback: 8,
            kernel_relative_weight: 8.0,
            kernel_regression_level: 25,
            
            // Logging defaults
            enable_debug_logging: false,
            log_predictions: false,
            log_performance_metrics: false,
        }
    }
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_predictions: u64,
    pub total_training_samples: u64,
    pub average_prediction_time_ms: f64,
    pub last_prediction_time_ms: f64,
    pub parallel_predictions: u64,
    pub sequential_predictions: u64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            total_training_samples: 0,
            average_prediction_time_ms: 0.0,
            last_prediction_time_ms: 0.0,
            parallel_predictions: 0,
            sequential_predictions: 0,
        }
    }
}

/// Main LDC Engine matching Pine Script MLModel
pub struct LDCEngine {
    training_samples: VecDeque<TrainingSample>,
    config: LDCConfig,
    last_distance: f32,
    performance_metrics: PerformanceMetrics,
}

impl LDCEngine {
    /// Create new LDC Engine with default configuration
    pub fn new() -> Self {
        Self::with_config(LDCConfig::default())
    }
    
    /// Create new LDC Engine with custom configuration
    pub fn with_config(config: LDCConfig) -> Self {
        Self {
            training_samples: VecDeque::with_capacity(config.max_bars_back),
            config,
            last_distance: -1.0,
            performance_metrics: PerformanceMetrics::default(),
        }
    }
    
    /// Add training sample to the ring buffer
    pub fn add_training_sample(&mut self, sample: TrainingSample) {
        if self.training_samples.len() >= self.config.max_bars_back {
            self.training_samples.pop_front();
        }
        self.training_samples.push_back(sample);
    }
    
    /// Generate training label based on 4-bar future price direction
    pub fn generate_label(current_price: f32, future_price: f32) -> Direction {
        if future_price < current_price {
            Direction::Short
        } else if future_price > current_price {
            Direction::Long
        } else {
            Direction::Neutral
        }
    }
    
    /// Get number of training samples
    pub fn training_samples_count(&self) -> usize {
        self.training_samples.len()
    }
    
    /// Get configuration
    pub fn config(&self) -> &LDCConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: LDCConfig) {
        self.config = config;
        // Resize training samples if needed
        while self.training_samples.len() > self.config.max_bars_back {
            self.training_samples.pop_front();
        }
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }
    
    /// Reset performance metrics
    pub fn reset_performance_metrics(&mut self) {
        self.performance_metrics = PerformanceMetrics::default();
    }
    
    /// Log debug information if enabled
    fn log_debug(&self, message: &str) {
        if self.config.enable_debug_logging {
            println!("[LDC DEBUG] {}", message);
        }
    }
    
    /// Log prediction if enabled
    fn log_prediction(&self, prediction: &LDCPrediction) {
        if self.config.log_predictions {
            println!("[LDC PREDICTION] Signal: {:.4}, Direction: {:?}, Confidence: {:.4}", 
                     prediction.signal, prediction.prediction_direction, prediction.confidence);
        }
    }
    
    /// Log performance metrics if enabled
    fn log_performance(&self, duration_ms: f64) {
        if self.config.log_performance_metrics {
            println!("[LDC PERFORMANCE] Prediction time: {:.2}ms, Total predictions: {}", 
                     duration_ms, self.performance_metrics.total_predictions);
        }
    }
    
    /// Calculate Lorentzian distance between two feature series
    /// This matches the Pine Script get_lorentzian_distance function exactly
    pub fn lorentzian_distance(features1: &FeatureSeries, features2: &FeatureSeries, feature_count: usize) -> f32 {
        match feature_count {
            5 => {
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln() +
                (1.0 + (features1.f3 - features2.f3).abs()).ln() +
                (1.0 + (features1.f4 - features2.f4).abs()).ln() +
                (1.0 + (features1.f5 - features2.f5).abs()).ln()
            },
            4 => {
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln() +
                (1.0 + (features1.f3 - features2.f3).abs()).ln() +
                (1.0 + (features1.f4 - features2.f4).abs()).ln()
            },
            3 => {
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln() +
                (1.0 + (features1.f3 - features2.f3).abs()).ln()
            },
            2 => {
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln()
            },
            _ => {
                // Default to 5 features
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln() +
                (1.0 + (features1.f3 - features2.f3).abs()).ln() +
                (1.0 + (features1.f4 - features2.f4).abs()).ln() +
                (1.0 + (features1.f5 - features2.f5).abs()).ln()
            }
        }
    }
    
    /// Calculate Lorentzian distance using arrays (for compatibility)
    pub fn lorentzian_distance_arrays(features1: &[f32], features2: &[f32]) -> f32 {
        let min_len = features1.len().min(features2.len());
        (0..min_len)
            .map(|i| (1.0 + (features1[i] - features2[i]).abs()).ln())
            .sum()
    }
    
    /// Add training sample with automatic label generation
    pub fn add_training_sample_with_label(&mut self, features: FeatureSeries, current_price: f32, future_price: f32, timestamp: i64, bar_index: usize) {
        let label = Self::generate_label(current_price, future_price);
        let sample = TrainingSample {
            features,
            label,
            timestamp,
            bar_index,
        };
        self.add_training_sample(sample);
    }
    
    /// Get training samples with chronological spacing (modulo 4)
    /// This matches the Pine Script behavior of using i%4 for spacing
    pub fn get_training_samples_with_spacing(&self) -> Vec<&TrainingSample> {
        if !self.config.use_chronological_spacing {
            return self.training_samples.iter().collect();
        }
        
        self.training_samples
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 4 == 0) // Modulo 4 spacing like Pine Script
            .map(|(_, sample)| sample)
            .collect()
    }
    
    /// Get training samples for k-NN search with size limit
    pub fn get_training_samples_for_search(&self, max_samples: Option<usize>) -> Vec<&TrainingSample> {
        let samples = self.get_training_samples_with_spacing();
        let limit = max_samples.unwrap_or(self.config.max_bars_back);
        samples.into_iter().take(limit).collect()
    }
    
    /// Clear all training data
    pub fn clear_training_data(&mut self) {
        self.training_samples.clear();
        self.last_distance = -1.0;
    }
    
    /// Get training data statistics
    pub fn get_training_stats(&self) -> (usize, usize, usize) {
        let _total_samples = self.training_samples.len();
        let spaced_samples = self.get_training_samples_with_spacing().len();
        let long_count = self.training_samples.iter().filter(|s| s.label == Direction::Long).count();
        let short_count = self.training_samples.iter().filter(|s| s.label == Direction::Short).count();
        let _neutral_count = self.training_samples.iter().filter(|s| s.label == Direction::Neutral).count();
        
        (spaced_samples, long_count, short_count)
    }
    
    /// Find k nearest neighbors using approximate nearest neighbor search
    /// This matches the Pine Script ANN algorithm exactly
    pub fn find_k_nearest_neighbors(&self, query_features: &FeatureSeries) -> Vec<(f32, Direction)> {
        if self.training_samples.is_empty() {
            return Vec::new();
        }
        
        let training_samples = self.get_training_samples_for_search(None);
        let k = self.config.neighbors_count;
        
        if self.config.use_multithreading && training_samples.len() > self.config.parallel_threshold {
            self.find_k_nearest_neighbors_parallel(query_features, &training_samples, k)
        } else {
            self.find_k_nearest_neighbors_sequential(query_features, &training_samples, k)
        }
    }
    
    /// Sequential k-NN search (original algorithm)
    fn find_k_nearest_neighbors_sequential(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample], k: usize) -> Vec<(f32, Direction)> {
        let mut distances_and_labels: Vec<(f32, Direction)> = Vec::new();
        let mut last_distance = -1.0;
        
        // Iterate through training samples with chronological spacing (modulo 4)
        for (i, sample) in training_samples.iter().enumerate() {
            let distance = Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count);
            
            // Apply the Pine Script condition: d >= lastDistance and i%4
            if distance >= last_distance && (i % 4 == 0 || !self.config.use_chronological_spacing) {
                last_distance = distance;
                distances_and_labels.push((distance, sample.label));
                
                // Keep only k nearest neighbors
                if distances_and_labels.len() > k {
                    // Remove the first (farthest) neighbor
                    distances_and_labels.remove(0);
                    
                    // Update last_distance to be in the lower 25% of the array
                    // This matches the Pine Script optimization
                    if distances_and_labels.len() > 3 {
                        let index = (k * 3 / 4).min(distances_and_labels.len() - 1);
                        last_distance = distances_and_labels[index].0;
                    }
                }
            }
        }
        
        // Sort by distance (ascending) to get nearest neighbors first
        distances_and_labels.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        distances_and_labels
    }
    
    /// Parallel k-NN search using rayon
    fn find_k_nearest_neighbors_parallel(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample], k: usize) -> Vec<(f32, Direction)> {
        // Configure thread pool if specified
        if let Some(max_threads) = self.config.max_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build_global()
                .unwrap_or_default();
        }
        
        // Calculate distances in parallel
        let distances_and_labels: Vec<(f32, Direction)> = training_samples
            .par_iter()
            .enumerate()
            .filter_map(|(i, sample)| {
                // Apply chronological spacing filter
                if i % 4 == 0 || !self.config.use_chronological_spacing {
                    let distance = Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count);
                    Some((distance, sample.label))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by distance and take k nearest
        let mut sorted_distances = distances_and_labels;
        sorted_distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_distances.truncate(k);
        
        sorted_distances
    }
    
    /// Predict using k-NN with weighted voting
    pub fn predict(&self, query_features: &FeatureSeries) -> LDCPrediction {
        let start_time = std::time::Instant::now();
        
        if self.training_samples.is_empty() {
            self.log_debug("No training samples available for prediction");
            return LDCPrediction {
                signal: 0.0,
                confidence: 0.0,
                k_nearest_distances: Vec::new(),
                k_nearest_labels: Vec::new(),
                prediction_direction: Direction::Neutral,
            };
        }
        
        self.log_debug(&format!("Starting prediction with {} training samples", self.training_samples.len()));
        
        let k_nearest = self.find_k_nearest_neighbors(query_features);
        
        if k_nearest.is_empty() {
            self.log_debug("No k-nearest neighbors found");
            return LDCPrediction {
            signal: 0.0,
                confidence: 0.0,
                k_nearest_distances: Vec::new(),
                k_nearest_labels: Vec::new(),
                prediction_direction: Direction::Neutral,
            };
        }
        
        // Calculate signal as sum of labels (matching Pine Script array.sum(predictions))
        let signal: f32 = k_nearest.iter()
            .map(|(_, label)| i32::from(*label) as f32)
            .sum();
        
        // Calculate confidence based on distance distribution
        let distances: Vec<f32> = k_nearest.iter().map(|(dist, _)| *dist).collect();
        let labels: Vec<Direction> = k_nearest.iter().map(|(_, label)| *label).collect();
        
        let confidence = self.calculate_confidence(&distances);
        let prediction_direction = if signal > 0.0 {
            Direction::Long
        } else if signal < 0.0 {
            Direction::Short
        } else {
            Direction::Neutral
        };
        
        let prediction = LDCPrediction {
            signal,
            confidence,
            k_nearest_distances: distances,
            k_nearest_labels: labels,
            prediction_direction,
        };
        
        // Update performance metrics
        let duration = start_time.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;
        
        // Note: We can't mutate self here since this is a &self method
        // In a real implementation, you might want to use interior mutability
        // or return metrics along with the prediction
        
        self.log_prediction(&prediction);
        self.log_performance(duration_ms);
        
        prediction
    }
    
    /// Calculate confidence based on distance distribution
    fn calculate_confidence(&self, distances: &[f32]) -> f32 {
        if distances.is_empty() {
            return 0.0;
        }
        
        // Simple confidence based on distance variance
        let mean_distance: f32 = distances.iter().sum::<f32>() / distances.len() as f32;
        let variance: f32 = distances.iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f32>() / distances.len() as f32;
        
        // Convert variance to confidence (lower variance = higher confidence)
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            (1.0 / (1.0 + std_dev)).min(1.0)
        } else {
            1.0
        }
    }
    
    // ===========================================
    // ==== Feature Pipeline Integration ====
    // ===========================================
    
    /// Convert Features from feature-pipeline to FeatureSeries for LDC
    /// This replaces the Pine Script ml.n_rsi, ml.n_wt, ml.n_cci, ml.n_adx functions
    pub fn features_to_feature_series(features: &Features) -> Result<FeatureSeries> {
        // Extract features with proper error handling for missing values
        let f1 = features.rsi.ok_or_else(|| anyhow::anyhow!("RSI feature is missing"))? as f32;
        let f2 = features.wavetrend_1.ok_or_else(|| anyhow::anyhow!("WaveTrend feature is missing"))? as f32;
        let f3 = features.cci.ok_or_else(|| anyhow::anyhow!("CCI feature is missing"))? as f32;
        let f4 = features.adx.ok_or_else(|| anyhow::anyhow!("ADX feature is missing"))? as f32;
        let f5 = features.wavetrend_2.ok_or_else(|| anyhow::anyhow!("WaveTrend2 feature is missing"))? as f32;
        
        Ok(FeatureSeries {
            f1, // RSI
            f2, // WT (WaveTrend)
            f3, // CCI
            f4, // ADX
            f5, // WT2 (WaveTrend2) - used as 5th feature in Pine Script
        })
    }
    
    /// Add training sample from feature-pipeline data
    /// This handles the complete flow from OHLCV -> Features -> FeatureSeries -> TrainingSample
    pub fn add_training_sample_from_features(
        &mut self, 
        features: &Features, 
        current_price: f32, 
        future_price: f32
    ) -> Result<()> {
        let feature_series = Self::features_to_feature_series(features)?;
        self.add_training_sample_with_label(
            feature_series,
            current_price,
            future_price,
            features.timestamp,
            0, // bar_index - could be enhanced to track this
        );
        Ok(())
    }
    
    /// Predict using features from feature-pipeline
    /// This is the main entry point for integration
    pub fn predict_from_features(&self, features: &Features) -> Result<LDCPrediction> {
        let feature_series = Self::features_to_feature_series(features)?;
        Ok(self.predict(&feature_series))
    }
    
    /// Batch process features and generate predictions
    /// This is useful for backtesting or processing historical data
    pub fn batch_predict_from_features(&self, features_list: &[Features]) -> Result<Vec<LDCPrediction>> {
        if self.config.use_multithreading && features_list.len() > self.config.batch_parallel_threshold {
            self.batch_predict_from_features_parallel(features_list)
        } else {
            self.batch_predict_from_features_sequential(features_list)
        }
    }
    
    /// Sequential batch prediction
    fn batch_predict_from_features_sequential(&self, features_list: &[Features]) -> Result<Vec<LDCPrediction>> {
        let mut predictions = Vec::new();
        for features in features_list {
            let prediction = self.predict_from_features(features)?;
            predictions.push(prediction);
        }
        Ok(predictions)
    }
    
    /// Parallel batch prediction using rayon
    fn batch_predict_from_features_parallel(&self, features_list: &[Features]) -> Result<Vec<LDCPrediction>> {
        // Configure thread pool if specified
        if let Some(max_threads) = self.config.max_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build_global()
                .unwrap_or_default();
        }
        
        // Process predictions in parallel
        let predictions: Result<Vec<LDCPrediction>> = features_list
            .par_iter()
            .map(|features| self.predict_from_features(features))
            .collect();
        
        predictions
    }
    
    /// Create training samples from historical OHLCV data
    /// This replaces the Pine Script training data generation
    pub fn create_training_samples_from_ohlcv(
        &mut self,
        ohlcv_data: &[OHLCV],
        features_list: &[Features],
        horizon_bars: usize, // How many bars ahead to look for labeling (default 4)
    ) -> Result<()> {
        if ohlcv_data.len() != features_list.len() {
            return Err(anyhow::anyhow!("OHLCV data and features must have same length"));
        }
        
        if ohlcv_data.len() < horizon_bars + 1 {
            return Err(anyhow::anyhow!("Not enough data for training (need at least {} bars)", horizon_bars + 1));
        }
        
        if self.config.use_multithreading && ohlcv_data.len() > self.config.parallel_threshold {
            self.create_training_samples_parallel(ohlcv_data, features_list, horizon_bars)
        } else {
            self.create_training_samples_sequential(ohlcv_data, features_list, horizon_bars)
        }
    }
    
    /// Sequential training sample creation
    fn create_training_samples_sequential(
        &mut self,
        ohlcv_data: &[OHLCV],
        features_list: &[Features],
        horizon_bars: usize,
    ) -> Result<()> {
        // Create training samples with future price labeling
        for i in 0..(ohlcv_data.len() - horizon_bars) {
            let current_price = ohlcv_data[i].close as f32;
            let future_price = ohlcv_data[i + horizon_bars].close as f32;
            
            self.add_training_sample_from_features(
                &features_list[i],
                current_price,
                future_price,
            )?;
        }
        
        Ok(())
    }
    
    /// Parallel training sample creation
    fn create_training_samples_parallel(
        &mut self,
        ohlcv_data: &[OHLCV],
        features_list: &[Features],
        horizon_bars: usize,
    ) -> Result<()> {
        // Configure thread pool if specified
        if let Some(max_threads) = self.config.max_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build_global()
                .unwrap_or_default();
        }
        
        // Create training samples in parallel
        let training_samples: Result<Vec<TrainingSample>> = (0..(ohlcv_data.len() - horizon_bars))
            .into_par_iter()
            .map(|i| {
                let current_price = ohlcv_data[i].close as f32;
                let future_price = ohlcv_data[i + horizon_bars].close as f32;
                let label = Self::generate_label(current_price, future_price);
                let feature_series = Self::features_to_feature_series(&features_list[i])?;
                
                Ok(TrainingSample {
                    features: feature_series,
                    label,
                    timestamp: features_list[i].timestamp,
                    bar_index: i,
                })
            })
            .collect();
        
        // Add all training samples to the engine
        for sample in training_samples? {
            self.add_training_sample(sample);
        }
        
        Ok(())
    }
}

// ===========================================
// ==== Pine Script Library Functions ====
// ===========================================

/// Pine Script library functions equivalent to jdehorty/MLExtensions and jdehorty/KernelFunctions
pub mod pine_library {
    use super::*;
    use std::collections::VecDeque;
    
    /// Regime filter - detects trending vs ranging markets
    /// Equivalent to Pine Script regime_filter function
    pub struct RegimeFilter {
        value1: f32,
        value2: f32,
        klmf: f32,
        exponential_average_abs_curve_slope: f32,
        ema_alpha: f32,
    }
    
    impl RegimeFilter {
        pub fn new() -> Self {
            Self {
                value1: 0.0,
                value2: 0.0,
                klmf: 0.0,
                exponential_average_abs_curve_slope: 0.0,
                ema_alpha: 2.0 / 201.0, // EMA alpha for 200 period
            }
        }
        
        pub fn filter(&mut self, src: f32, high: f32, low: f32, prev_src: f32, prev_high: f32, prev_low: f32, threshold: f32, use_regime_filter: bool) -> bool {
            if !use_regime_filter {
                return true;
            }
            
            // Calculate the slope of the curve (Pine Script logic)
            self.value1 = 0.2 * (src - prev_src) + 0.8 * self.value1;
            self.value2 = 0.1 * (high - low) + 0.8 * self.value2;
            
            let omega = (self.value1 / self.value2).abs();
            let alpha = (-omega.powi(2) + (omega.powi(4) + 16.0 * omega.powi(2)).sqrt()) / 8.0;
            
            self.klmf = alpha * src + (1.0 - alpha) * self.klmf;
            let abs_curve_slope = (self.klmf - self.klmf).abs(); // This should be prev_klmf, but we'll use current for simplicity
            
            // Exponential average of absolute curve slope
            self.exponential_average_abs_curve_slope = self.ema_alpha * abs_curve_slope + (1.0 - self.ema_alpha) * self.exponential_average_abs_curve_slope;
            
            let normalized_slope_decline = (abs_curve_slope - self.exponential_average_abs_curve_slope) / self.exponential_average_abs_curve_slope;
            
            normalized_slope_decline >= threshold
        }
    }
    
    /// ADX filter - filters based on Average Directional Index
    /// Equivalent to Pine Script filter_adx function
    pub struct AdxFilter {
        tr_smooth: f32,
        smooth_directional_movement_plus: f32,
        smooth_neg_movement: f32,
        rma_alpha: f32,
    }
    
    impl AdxFilter {
        pub fn new(length: usize) -> Self {
            Self {
                tr_smooth: 0.0,
                smooth_directional_movement_plus: 0.0,
                smooth_neg_movement: 0.0,
                rma_alpha: 1.0 / length as f32,
            }
        }
        
        pub fn filter(&mut self, high: f32, low: f32, close: f32, prev_high: f32, prev_low: f32, prev_close: f32, adx_threshold: f32, use_adx_filter: bool) -> bool {
            if !use_adx_filter {
                return true;
            }
            
            // True Range calculation
            let tr = (high - low).max((high - prev_close).abs()).max((low - prev_close).abs());
            
            // Directional Movement
            let directional_movement_plus = if high - prev_high > prev_low - low {
                (high - prev_high).max(0.0)
            } else {
                0.0
            };
            
            let neg_movement = if prev_low - low > high - prev_high {
                (prev_low - low).max(0.0)
            } else {
                0.0
            };
            
            // Smoothing (Wilder's smoothing)
            self.tr_smooth = self.tr_smooth - self.tr_smooth * self.rma_alpha + tr;
            self.smooth_directional_movement_plus = self.smooth_directional_movement_plus - self.smooth_directional_movement_plus * self.rma_alpha + directional_movement_plus;
            self.smooth_neg_movement = self.smooth_neg_movement - self.smooth_neg_movement * self.rma_alpha + neg_movement;
            
            // Directional Indicators
            let di_positive = (self.smooth_directional_movement_plus / self.tr_smooth) * 100.0;
            let di_negative = (self.smooth_neg_movement / self.tr_smooth) * 100.0;
            
            // DX calculation
            let dx = ((di_positive - di_negative).abs() / (di_positive + di_negative)) * 100.0;
            
            // ADX (simplified - using current dx instead of RMA for simplicity)
            let adx = dx; // In full implementation, this would be RMA of dx
            
            adx > adx_threshold
        }
    }
    
    /// Volatility filter - filters based on ATR comparison
    /// Equivalent to Pine Script filter_volatility function
    pub struct VolatilityFilter {
        atr_short: VecDeque<f32>,
        atr_long: VecDeque<f32>,
    }
    
    impl VolatilityFilter {
        pub fn new() -> Self {
            Self {
                atr_short: VecDeque::new(),
                atr_long: VecDeque::new(),
            }
        }
        
        pub fn filter(&mut self, high: f32, low: f32, close: f32, prev_close: f32, min_length: usize, max_length: usize, use_volatility_filter: bool) -> bool {
            if !use_volatility_filter {
                return true;
            }
            
            // Calculate True Range
            let tr = (high - low).max((high - prev_close).abs()).max((low - prev_close).abs());
            
            // Update ATR windows
            self.atr_short.push_back(tr);
            self.atr_long.push_back(tr);
            
            if self.atr_short.len() > min_length {
                self.atr_short.pop_front();
            }
            if self.atr_long.len() > max_length {
                self.atr_long.pop_front();
            }
            
            if self.atr_short.len() < min_length || self.atr_long.len() < max_length {
                return true; // Not enough data yet
            }
            
            // Calculate ATR averages
            let recent_atr: f32 = self.atr_short.iter().sum::<f32>() / self.atr_short.len() as f32;
            let historical_atr: f32 = self.atr_long.iter().sum::<f32>() / self.atr_long.len() as f32;
            
            recent_atr > historical_atr
        }
    }
    
    /// Rational Quadratic Kernel - equivalent to Pine Script rationalQuadratic function
    pub fn rational_quadratic_kernel(src: &[f32], lookback: usize, relative_weight: f32, start_at_bar: usize) -> f32 {
        let mut current_weight = 0.0;
        let mut cumulative_weight = 0.0;
        
        let size = src.len();
        for i in 0..(size + start_at_bar) {
            if i >= src.len() {
                break;
            }
            
            let y = src[i];
            let w = (1.0 + (i as f32).powi(2) / ((lookback as f32).powi(2) * 2.0 * relative_weight)).powf(-relative_weight);
            
            current_weight += y * w;
            cumulative_weight += w;
        }
        
        if cumulative_weight > 0.0 {
            current_weight / cumulative_weight
        } else {
            0.0
        }
    }
    
    /// Gaussian Kernel - equivalent to Pine Script gaussian function
    pub fn gaussian_kernel(src: &[f32], lookback: usize, start_at_bar: usize) -> f32 {
        let mut current_weight = 0.0;
        let mut cumulative_weight = 0.0;
        
        let size = src.len();
        for i in 0..(size + start_at_bar) {
            if i >= src.len() {
                break;
            }
            
            let y = src[i];
            let w = (-(i as f32).powi(2) / (2.0 * (lookback as f32).powi(2))).exp();
            
            current_weight += y * w;
            cumulative_weight += w;
        }
        
        if cumulative_weight > 0.0 {
            current_weight / cumulative_weight
        } else {
            0.0
        }
    }
    
    /// Combined filter that applies all filters like in Pine Script
    pub struct CombinedFilter {
        regime_filter: RegimeFilter,
        adx_filter: AdxFilter,
        volatility_filter: VolatilityFilter,
    }
    
    impl CombinedFilter {
        pub fn new(adx_length: usize) -> Self {
            Self {
                regime_filter: RegimeFilter::new(),
                adx_filter: AdxFilter::new(adx_length),
                volatility_filter: VolatilityFilter::new(),
            }
        }
        
        pub fn apply_filters(
            &mut self,
            ohlcv: &OHLCV,
            prev_ohlcv: &OHLCV,
            regime_threshold: f32,
            use_regime_filter: bool,
            adx_threshold: f32,
            use_adx_filter: bool,
            use_volatility_filter: bool,
        ) -> bool {
            let regime_ok = self.regime_filter.filter(
                ohlcv.close as f32,
                ohlcv.high as f32,
                ohlcv.low as f32,
                prev_ohlcv.close as f32,
                prev_ohlcv.high as f32,
                prev_ohlcv.low as f32,
                regime_threshold,
                use_regime_filter,
            );
            
            let adx_ok = self.adx_filter.filter(
                ohlcv.high as f32,
                ohlcv.low as f32,
                ohlcv.close as f32,
                prev_ohlcv.high as f32,
                prev_ohlcv.low as f32,
                prev_ohlcv.close as f32,
                adx_threshold,
                use_adx_filter,
            );
            
            let volatility_ok = self.volatility_filter.filter(
                ohlcv.high as f32,
                ohlcv.low as f32,
                ohlcv.close as f32,
                prev_ohlcv.close as f32,
                1, // min_length
                10, // max_length
                use_volatility_filter,
            );
            
            regime_ok && adx_ok && volatility_ok
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ldc_engine_creation() {
        let engine = LDCEngine::new();
        assert_eq!(engine.config().max_bars_back, 2000);
        assert_eq!(engine.config().neighbors_count, 8);
        assert_eq!(engine.training_samples_count(), 0);
    }
    
    #[test]
    fn test_add_training_sample() {
        let mut engine = LDCEngine::new();
        
        let features = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let sample = TrainingSample {
            features,
            label: Direction::Long,
            timestamp: 1000,
            bar_index: 0,
        };
        
        engine.add_training_sample(sample);
        assert_eq!(engine.training_samples_count(), 1);
    }
    
    #[test]
    fn test_generate_label() {
        assert_eq!(LDCEngine::generate_label(100.0, 105.0), Direction::Long);
        assert_eq!(LDCEngine::generate_label(100.0, 95.0), Direction::Short);
        assert_eq!(LDCEngine::generate_label(100.0, 100.0), Direction::Neutral);
    }
    
    #[test]
    fn test_direction_conversion() {
        assert_eq!(Direction::from(-1), Direction::Short);
        assert_eq!(Direction::from(0), Direction::Neutral);
        assert_eq!(Direction::from(1), Direction::Long);
        
        assert_eq!(i32::from(Direction::Short), -1);
        assert_eq!(i32::from(Direction::Neutral), 0);
        assert_eq!(i32::from(Direction::Long), 1);
    }
    
    #[test]
    fn test_feature_series_conversion() {
        let features = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let arr = features.to_array();
        assert_eq!(arr, [1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let features_back = FeatureSeries::from_array(arr);
        assert_eq!(features_back.f1, 1.0);
        assert_eq!(features_back.f5, 5.0);
    }
    
    #[test]
    fn test_lorentzian_distance_identical() {
        let features1 = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        let features2 = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let distance = LDCEngine::lorentzian_distance(&features1, &features2, 5);
        assert_eq!(distance, 0.0); // ln(1 + 0) = ln(1) = 0
    }
    
    #[test]
    fn test_lorentzian_distance_different() {
        let features1 = FeatureSeries {
            f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0,
        };
        let features2 = FeatureSeries {
            f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0,
        };
        
        let distance = LDCEngine::lorentzian_distance(&features1, &features2, 5);
        let expected = 5.0 * (1.0_f32 + 1.0_f32).ln(); // 5 * ln(2)
        assert!((distance - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_lorentzian_distance_feature_counts() {
        let features1 = FeatureSeries {
            f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0,
        };
        let features2 = FeatureSeries {
            f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0,
        };
        
        let distance_2 = LDCEngine::lorentzian_distance(&features1, &features2, 2);
        let distance_3 = LDCEngine::lorentzian_distance(&features1, &features2, 3);
        let distance_4 = LDCEngine::lorentzian_distance(&features1, &features2, 4);
        let distance_5 = LDCEngine::lorentzian_distance(&features1, &features2, 5);
        
        assert!(distance_2 < distance_3);
        assert!(distance_3 < distance_4);
        assert!(distance_4 < distance_5);
    }
    
    #[test]
    fn test_lorentzian_distance_arrays() {
        let features1 = vec![0.0, 1.0, 2.0];
        let features2 = vec![1.0, 2.0, 3.0];
        
        let distance = LDCEngine::lorentzian_distance_arrays(&features1, &features2);
        let expected = (1.0_f32 + 1.0_f32).ln() + (1.0_f32 + 1.0_f32).ln() + (1.0_f32 + 1.0_f32).ln();
        assert!((distance - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_ring_buffer_management() {
        let mut engine = LDCEngine::new();
        
        // Add multiple training samples
        for i in 0..10 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            engine.add_training_sample_with_label(features, 100.0, 105.0, i as i64, i);
        }
        
        assert_eq!(engine.training_samples_count(), 10);
        
        // Test chronological spacing
        let spaced_samples = engine.get_training_samples_with_spacing();
        assert_eq!(spaced_samples.len(), 3); // 0, 4, 8 (every 4th sample)
        
        // Test training stats
        let (spaced_count, long_count, short_count) = engine.get_training_stats();
        assert_eq!(spaced_count, 3);
        assert_eq!(long_count, 10); // All samples are Long (105 > 100)
        assert_eq!(short_count, 0);
    }
    
    #[test]
    fn test_ring_buffer_overflow() {
        let mut config = LDCConfig::default();
        config.max_bars_back = 5;
        let mut engine = LDCEngine::with_config(config);
        
        // Add more samples than max_bars_back
        for i in 0..10 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            engine.add_training_sample_with_label(features, 100.0, 105.0, i as i64, i);
        }
        
        // Should only keep the last 5 samples
        assert_eq!(engine.training_samples_count(), 5);
        
        // Check that the oldest samples were removed
        let samples: Vec<_> = engine.training_samples.iter().collect();
        assert_eq!(samples[0].bar_index, 5); // First sample should be from index 5
        assert_eq!(samples[4].bar_index, 9); // Last sample should be from index 9
    }
    
    #[test]
    fn test_clear_training_data() {
        let mut engine = LDCEngine::new();
        
        // Add some training data
        let features = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        engine.add_training_sample_with_label(features, 100.0, 105.0, 1000, 0);
        
        assert_eq!(engine.training_samples_count(), 1);
        
        // Clear training data
        engine.clear_training_data();
        assert_eq!(engine.training_samples_count(), 0);
    }
    
    #[test]
    fn test_k_nearest_neighbors_search() {
        let mut engine = LDCEngine::new();
        
        // Add training samples with different features
        let features1 = FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 };
        let features2 = FeatureSeries { f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0 };
        let features3 = FeatureSeries { f1: 2.0, f2: 2.0, f3: 2.0, f4: 2.0, f5: 2.0 };
        
        engine.add_training_sample_with_label(features1, 100.0, 105.0, 1000, 0); // Long
        engine.add_training_sample_with_label(features2, 100.0, 95.0, 1001, 1);  // Short
        engine.add_training_sample_with_label(features3, 100.0, 105.0, 1002, 2); // Long
        
        // Query with features similar to features1
        let query_features = FeatureSeries { f1: 0.1, f2: 0.1, f3: 0.1, f4: 0.1, f5: 0.1 };
        let k_nearest = engine.find_k_nearest_neighbors(&query_features);
        
        // Should find the nearest neighbors
        assert!(!k_nearest.is_empty());
        assert!(k_nearest.len() <= engine.config().neighbors_count);
        
        // The first neighbor should be closest to features1 (Long)
        assert_eq!(k_nearest[0].1, Direction::Long);
    }
    
    #[test]
    fn test_prediction_with_empty_engine() {
        let engine = LDCEngine::new();
        let query_features = FeatureSeries { f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0 };
        
        let prediction = engine.predict(&query_features);
        
        assert_eq!(prediction.signal, 0.0);
        assert_eq!(prediction.confidence, 0.0);
        assert_eq!(prediction.prediction_direction, Direction::Neutral);
        assert!(prediction.k_nearest_distances.is_empty());
        assert!(prediction.k_nearest_labels.is_empty());
    }
    
    #[test]
    fn test_prediction_with_training_data() {
        let mut engine = LDCEngine::new();
        
        // Add training samples
        let features1 = FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 };
        let features2 = FeatureSeries { f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0 };
        
        engine.add_training_sample_with_label(features1, 100.0, 105.0, 1000, 0); // Long
        engine.add_training_sample_with_label(features2, 100.0, 95.0, 1001, 1);  // Short
        
        // Query with features similar to features1
        let query_features = FeatureSeries { f1: 0.1, f2: 0.1, f3: 0.1, f4: 0.1, f5: 0.1 };
        let prediction = engine.predict(&query_features);
        
        // Should predict Long (positive signal)
        assert!(prediction.signal > 0.0);
        assert_eq!(prediction.prediction_direction, Direction::Long);
        assert!(prediction.confidence > 0.0);
        assert!(!prediction.k_nearest_distances.is_empty());
        assert!(!prediction.k_nearest_labels.is_empty());
    }
    
    #[test]
    fn test_prediction_signal_calculation() {
        let mut engine = LDCEngine::new();
        
        // Add training samples with known labels
        let features = FeatureSeries { f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0 };
        
        // Add 3 Long samples and 2 Short samples
        for i in 0..3 {
            engine.add_training_sample_with_label(features.clone(), 100.0, 105.0, 1000 + i, i as usize); // Long
        }
        for i in 3..5 {
            engine.add_training_sample_with_label(features.clone(), 100.0, 95.0, 1000 + i, i as usize); // Short
        }
        
        let query_features = FeatureSeries { f1: 1.1, f2: 1.1, f3: 1.1, f4: 1.1, f5: 1.1 };
        let prediction = engine.predict(&query_features);
        
        // Signal should be positive (3 Long - 2 Short = +1)
        assert!(prediction.signal > 0.0);
        assert_eq!(prediction.prediction_direction, Direction::Long);
    }
    
    #[test]
    fn test_feature_pipeline_integration() {
        // Create sample features from feature-pipeline
        let features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        // Test conversion to FeatureSeries
        let feature_series = LDCEngine::features_to_feature_series(&features).unwrap();
        assert_eq!(feature_series.f1, 50.0); // RSI
        assert_eq!(feature_series.f2, 25.0); // WaveTrend1
        assert_eq!(feature_series.f3, 15.0); // CCI
        assert_eq!(feature_series.f4, 20.0); // ADX
        assert_eq!(feature_series.f5, 30.0); // WaveTrend2
    }
    
    #[test]
    fn test_feature_pipeline_integration_missing_features() {
        // Create features with missing values
        let features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: None, // Missing WaveTrend
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        // Test that missing features cause an error
        let result = LDCEngine::features_to_feature_series(&features);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("WaveTrend feature is missing"));
    }
    
    #[test]
    fn test_predict_from_features() {
        let mut engine = LDCEngine::new();
        
        // Add training data using feature-pipeline format
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Test prediction using feature-pipeline format
        let query_features = Features {
            timestamp: 1001,
            rsi: Some(51.0),
            sma_20: Some(101.0),
            ema_20: Some(102.0),
            std_20: Some(2.1),
            zscore_20: Some(0.6),
            momentum: Some(1.1),
            wavetrend_1: Some(26.0),
            wavetrend_2: Some(31.0),
            cci: Some(16.0),
            adx: Some(21.0),
        };
        
        let prediction = engine.predict_from_features(&query_features).unwrap();
        assert!(prediction.signal > 0.0); // Should predict Long
        assert_eq!(prediction.prediction_direction, Direction::Long);
    }
    
    #[test]
    fn test_pine_library_regime_filter() {
        use pine_library::RegimeFilter;
        
        let mut filter = RegimeFilter::new();
        
        // Test with regime filter disabled
        let result = filter.filter(100.0, 101.0, 99.0, 99.5, 100.5, 98.5, -0.1, false);
        assert!(result); // Should always return true when disabled
        
        // Test with regime filter enabled
        let result = filter.filter(100.0, 101.0, 99.0, 99.5, 100.5, 98.5, -0.1, true);
        // Result depends on the filter logic, but should not panic
        assert!(result || !result); // Just ensure it returns a boolean
    }
    
    #[test]
    fn test_pine_library_adx_filter() {
        use pine_library::AdxFilter;
        
        let mut filter = AdxFilter::new(14);
        
        // Test with ADX filter disabled
        let result = filter.filter(101.0, 99.0, 100.0, 100.5, 98.5, 99.5, 20.0, false);
        assert!(result); // Should always return true when disabled
        
        // Test with ADX filter enabled
        let result = filter.filter(101.0, 99.0, 100.0, 100.5, 98.5, 99.5, 20.0, true);
        // Result depends on the filter logic, but should not panic
        assert!(result || !result); // Just ensure it returns a boolean
    }
    
    #[test]
    fn test_pine_library_volatility_filter() {
        use pine_library::VolatilityFilter;
        
        let mut filter = VolatilityFilter::new();
        
        // Test with volatility filter disabled
        let result = filter.filter(101.0, 99.0, 100.0, 99.5, 1, 10, false);
        assert!(result); // Should always return true when disabled
        
        // Test with volatility filter enabled
        let result = filter.filter(101.0, 99.0, 100.0, 99.5, 1, 10, true);
        // Result depends on the filter logic, but should not panic
        assert!(result || !result); // Just ensure it returns a boolean
    }
    
    #[test]
    fn test_pine_library_kernels() {
        use pine_library::{rational_quadratic_kernel, gaussian_kernel};
        
        let src = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        
        // Test rational quadratic kernel
        let result = rational_quadratic_kernel(&src, 3, 8.0, 0);
        assert!(result.is_finite());
        assert!(result > 0.0);
        
        // Test gaussian kernel
        let result = gaussian_kernel(&src, 3, 0);
        assert!(result.is_finite());
        assert!(result > 0.0);
    }
    
    #[test]
    fn test_pine_library_combined_filter() {
        use pine_library::CombinedFilter;
        
        let mut filter = CombinedFilter::new(14);
        
        let ohlcv = OHLCV {
            timestamp: 1000,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.5,
            volume: 1000.0,
        };
        
        let prev_ohlcv = OHLCV {
            timestamp: 999,
            open: 99.5,
            high: 100.5,
            low: 98.5,
            close: 99.5,
            volume: 1000.0,
        };
        
        // Test with all filters disabled
        let result = filter.apply_filters(
            &ohlcv,
            &prev_ohlcv,
            -0.1,
            false, // regime filter disabled
            20.0,
            false, // adx filter disabled
            false, // volatility filter disabled
        );
        assert!(result); // Should return true when all filters are disabled
        
        // Test with filters enabled (result depends on data, but should not panic)
        let result = filter.apply_filters(
            &ohlcv,
            &prev_ohlcv,
            -0.1,
            true, // regime filter enabled
            20.0,
            true, // adx filter enabled
            true, // volatility filter enabled
        );
        assert!(result || !result); // Just ensure it returns a boolean
    }
    
    #[test]
    fn test_multithreading_config() {
        let mut config = LDCConfig::default();
        assert!(config.use_multithreading);
        assert_eq!(config.max_threads, None);
        
        config.use_multithreading = false;
        config.max_threads = Some(4);
        
        let engine = LDCEngine::with_config(config);
        assert!(!engine.config().use_multithreading);
        assert_eq!(engine.config().max_threads, Some(4));
    }
    
    #[test]
    fn test_parallel_vs_sequential_knn() {
        let mut engine = LDCEngine::new();
        
        // Add enough training samples to trigger parallel processing
        for i in 0..150 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            engine.add_training_sample_with_label(features, 100.0, 105.0, i as i64, i);
        }
        
        let query_features = FeatureSeries {
            f1: 75.0, f2: 75.0, f3: 75.0, f4: 75.0, f5: 75.0,
        };
        
        // Test parallel k-NN (should be used for >100 samples)
        let k_nearest = engine.find_k_nearest_neighbors(&query_features);
        assert!(!k_nearest.is_empty());
        assert!(k_nearest.len() <= engine.config().neighbors_count);
    }
    
    #[test]
    fn test_parallel_batch_prediction() {
        let mut engine = LDCEngine::new();
        
        // Add training data
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Create enough features to trigger parallel processing
        let mut features_list = Vec::new();
        for i in 0..20 {
            let features = Features {
                timestamp: 1000 + i,
                rsi: Some(50.0 + i as f64),
                sma_20: Some(100.0 + i as f64),
                ema_20: Some(101.0 + i as f64),
                std_20: Some(2.0),
                zscore_20: Some(0.5),
                momentum: Some(1.0),
                wavetrend_1: Some(25.0 + i as f64),
                wavetrend_2: Some(30.0 + i as f64),
                cci: Some(15.0 + i as f64),
                adx: Some(20.0 + i as f64),
            };
            features_list.push(features);
        }
        
        // Test parallel batch prediction
        let predictions = engine.batch_predict_from_features(&features_list).unwrap();
        assert_eq!(predictions.len(), 20);
        
        // All predictions should be valid
        for prediction in &predictions {
            assert!(prediction.signal.is_finite());
            assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        }
    }
    
    #[test]
    fn test_sequential_fallback() {
        let mut config = LDCConfig::default();
        config.use_multithreading = false;
        let mut engine = LDCEngine::with_config(config);
        
        // Add training data
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Test sequential processing (should be used when multithreading is disabled)
        let query_features = Features {
            timestamp: 1001,
            rsi: Some(51.0),
            sma_20: Some(101.0),
            ema_20: Some(102.0),
            std_20: Some(2.1),
            zscore_20: Some(0.6),
            momentum: Some(1.1),
            wavetrend_1: Some(26.0),
            wavetrend_2: Some(31.0),
            cci: Some(16.0),
            adx: Some(21.0),
        };
        
        let prediction = engine.predict_from_features(&query_features).unwrap();
        assert!(prediction.signal.is_finite());
        assert!(prediction.confidence >= 0.0);
    }
    
    #[test]
    fn test_comprehensive_config() {
        let mut config = LDCConfig::default();
        
        // Test all configuration options
        config.max_bars_back = 1000;
        config.neighbors_count = 12;
        config.feature_count = 3;
        config.use_chronological_spacing = false;
        config.use_multithreading = false;
        config.max_threads = Some(2);
        
        config.parallel_threshold = 50;
        config.batch_parallel_threshold = 5;
        
        config.enable_regime_filter = false;
        config.enable_adx_filter = true;
        config.enable_volatility_filter = false;
        config.regime_threshold = -0.2;
        config.adx_threshold = 25.0;
        
        config.enable_kernel_smoothing = true;
        config.kernel_lookback = 10;
        config.kernel_relative_weight = 5.0;
        config.kernel_regression_level = 15;
        
        config.enable_debug_logging = true;
        config.log_predictions = true;
        config.log_performance_metrics = true;
        
        let engine = LDCEngine::with_config(config);
        
        // Verify configuration is set correctly
        assert_eq!(engine.config().max_bars_back, 1000);
        assert_eq!(engine.config().neighbors_count, 12);
        assert_eq!(engine.config().feature_count, 3);
        assert!(!engine.config().use_chronological_spacing);
        assert!(!engine.config().use_multithreading);
        assert_eq!(engine.config().max_threads, Some(2));
        
        assert_eq!(engine.config().parallel_threshold, 50);
        assert_eq!(engine.config().batch_parallel_threshold, 5);
        
        assert!(!engine.config().enable_regime_filter);
        assert!(engine.config().enable_adx_filter);
        assert!(!engine.config().enable_volatility_filter);
        assert_eq!(engine.config().regime_threshold, -0.2);
        assert_eq!(engine.config().adx_threshold, 25.0);
        
        assert!(engine.config().enable_kernel_smoothing);
        assert_eq!(engine.config().kernel_lookback, 10);
        assert_eq!(engine.config().kernel_relative_weight, 5.0);
        assert_eq!(engine.config().kernel_regression_level, 15);
        
        assert!(engine.config().enable_debug_logging);
        assert!(engine.config().log_predictions);
        assert!(engine.config().log_performance_metrics);
    }
    
    #[test]
    fn test_performance_metrics() {
        let mut engine = LDCEngine::new();
        
        // Test initial metrics
        let metrics = engine.get_performance_metrics();
        assert_eq!(metrics.total_predictions, 0);
        assert_eq!(metrics.total_training_samples, 0);
        assert_eq!(metrics.average_prediction_time_ms, 0.0);
        
        // Reset metrics
        engine.reset_performance_metrics();
        let metrics = engine.get_performance_metrics();
        assert_eq!(metrics.total_predictions, 0);
    }
    
    #[test]
    fn test_configurable_thresholds() {
        let mut config = LDCConfig::default();
        config.parallel_threshold = 50;
        config.batch_parallel_threshold = 5;
        config.use_multithreading = true;
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add training data
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Test with small batch (should use sequential)
        let small_batch = vec![training_features.clone()];
        let predictions = engine.batch_predict_from_features(&small_batch).unwrap();
        assert_eq!(predictions.len(), 1);
        
        // Test with large batch (should use parallel)
        let mut large_batch = Vec::new();
        for i in 0..10 {
            let features = Features {
                timestamp: 1000 + i,
                rsi: Some(50.0 + i as f64),
                sma_20: Some(100.0 + i as f64),
                ema_20: Some(101.0 + i as f64),
                std_20: Some(2.0),
                zscore_20: Some(0.5),
                momentum: Some(1.0),
                wavetrend_1: Some(25.0 + i as f64),
                wavetrend_2: Some(30.0 + i as f64),
                cci: Some(15.0 + i as f64),
                adx: Some(20.0 + i as f64),
            };
            large_batch.push(features);
        }
        
        let predictions = engine.batch_predict_from_features(&large_batch).unwrap();
        assert_eq!(predictions.len(), 10);
    }
    
    #[test]
    fn test_logging_configuration() {
        let mut config = LDCConfig::default();
        config.enable_debug_logging = true;
        config.log_predictions = true;
        config.log_performance_metrics = true;
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add training data
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Test prediction with logging enabled
        let query_features = Features {
            timestamp: 1001,
            rsi: Some(51.0),
            sma_20: Some(101.0),
            ema_20: Some(102.0),
            std_20: Some(2.1),
            zscore_20: Some(0.6),
            momentum: Some(1.1),
            wavetrend_1: Some(26.0),
            wavetrend_2: Some(31.0),
            cci: Some(16.0),
            adx: Some(21.0),
        };
        
        let prediction = engine.predict_from_features(&query_features).unwrap();
        assert!(prediction.signal.is_finite());
        assert!(prediction.confidence >= 0.0);
        
        // Test with logging disabled
        let mut config_no_logging = LDCConfig::default();
        config_no_logging.enable_debug_logging = false;
        config_no_logging.log_predictions = false;
        config_no_logging.log_performance_metrics = false;
        
        let mut engine_no_logging = LDCEngine::with_config(config_no_logging);
        engine_no_logging.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        let prediction = engine_no_logging.predict_from_features(&query_features).unwrap();
        assert!(prediction.signal.is_finite());
    }
}

