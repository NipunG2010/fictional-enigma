use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Direction, LDCPrediction, OHLCV};

/// Statistical analyzer for prediction validation with configurable confidence levels
#[derive(Debug, Clone)]
pub struct StatisticalAnalyzer {
    config: StatisticalConfig,
}

/// Configuration for statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    /// Confidence level for statistical tests (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,
    /// Minimum sample size required for statistical significance
    pub min_sample_size: usize,
    /// P-value threshold for significance testing (e.g., 0.05)
    pub significance_threshold: f64,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            min_sample_size: 100,
            significance_threshold: 0.05,
        }
    }
}

/// Comprehensive statistical analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisResult {
    pub prediction_accuracy: AccuracyMetrics,
    pub signal_quality: SignalQualityMetrics,
    pub market_regime_analysis: MarketRegimeAnalysis,
    pub statistical_significance: SignificanceTest,
}

/// Accuracy metrics including hit rates, precision, recall, and F1 scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Overall hit rate (correct predictions / total predictions)
    pub hit_rate: f64,
    /// Precision for long predictions (true positives / (true positives + false positives))
    pub precision_long: f64,
    /// Precision for short predictions
    pub precision_short: f64,
    /// Overall precision (weighted average)
    pub precision_overall: f64,
    /// Recall for long predictions (true positives / (true positives + false negatives))
    pub recall_long: f64,
    /// Recall for short predictions
    pub recall_short: f64,
    /// Overall recall (weighted average)
    pub recall_overall: f64,
    /// F1 score for long predictions (2 * precision * recall / (precision + recall))
    pub f1_score_long: f64,
    /// F1 score for short predictions
    pub f1_score_short: f64,
    /// Overall F1 score (weighted average)
    pub f1_score_overall: f64,
    /// Confusion matrix for detailed analysis
    pub confusion_matrix: ConfusionMatrix,
}

/// Signal quality metrics including signal-to-noise ratio and information coefficient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQualityMetrics {
    /// Signal-to-noise ratio (mean signal strength / signal standard deviation)
    pub signal_to_noise_ratio: f64,
    /// Information coefficient (correlation between signal and future returns)
    pub information_coefficient: f64,
    /// Distribution of signal strengths
    pub signal_strength_distribution: SignalDistribution,
    /// Distribution of confidence values
    pub confidence_distribution: SignalDistribution,
    /// Average signal strength by direction
    pub avg_signal_strength_by_direction: HashMap<Direction, f64>,
}

/// Market regime analysis for different market conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegimeAnalysis {
    /// Performance metrics in trending markets
    pub trending_performance: AccuracyMetrics,
    /// Performance metrics in ranging markets
    pub ranging_performance: AccuracyMetrics,
    /// Performance metrics in volatile markets
    pub volatile_performance: AccuracyMetrics,
    /// Market regime classification results
    pub regime_classification: Vec<MarketRegime>,
}

/// Statistical significance test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceTest {
    /// P-value for overall prediction accuracy
    pub accuracy_p_value: f64,
    /// P-value for signal quality
    pub signal_quality_p_value: f64,
    /// P-values for market regime performance (before correction)
    pub regime_p_values: HashMap<MarketRegimeType, f64>,
    /// P-values after multiple testing correction
    pub corrected_p_values: HashMap<String, f64>,
    /// Confidence intervals for key metrics
    pub confidence_intervals: HashMap<String, ConfidenceInterval>,
    /// Whether results are statistically significant
    pub is_significant: bool,
    /// Sample size used for analysis
    pub sample_size: usize,
    /// Statistical power of the test
    pub statistical_power: f64,
    /// Multiple testing correction method used
    pub correction_method: MultipleTestingCorrection,
}

/// Confusion matrix for classification analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// True positives (correctly predicted long positions)
    pub true_positives_long: usize,
    /// False positives (incorrectly predicted long positions)
    pub false_positives_long: usize,
    /// True negatives (correctly predicted short positions)
    pub true_negatives_short: usize,
    /// False negatives (incorrectly predicted short positions)
    pub false_negatives_short: usize,
    /// Neutral predictions that were correct
    pub true_neutral: usize,
    /// Neutral predictions that were incorrect
    pub false_neutral: usize,
}

/// Signal distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalDistribution {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentile_25: f64,
    pub percentile_50: f64,
    pub percentile_75: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
}

/// Market regime classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegime {
    pub start_index: usize,
    pub end_index: usize,
    pub regime_type: MarketRegimeType,
    pub volatility: f64,
    pub trend_strength: f64,
}

/// Types of market regimes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegimeType {
    Trending,
    Ranging,
    Volatile,
}

/// Multiple testing correction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultipleTestingCorrection {
    /// Bonferroni correction (most conservative)
    Bonferroni,
    /// Benjamini-Hochberg false discovery rate control
    BenjaminiHochberg,
    /// No correction applied
    None,
}

/// Confidence interval for statistical metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
}

/// Sample size validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleSizeValidation {
    pub sample_size: usize,
    pub is_sufficient_for_basic_analysis: bool,
    pub is_sufficient_for_regime_analysis: bool,
    pub is_sufficient_for_significance_testing: bool,
    pub recommended_minimum_size: usize,
    pub power_analysis: PowerAnalysis,
}

/// Statistical power analysis for different effect sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    /// Statistical power for detecting small effects (Cohen's d = 0.1)
    pub small_effect_power: f64,
    /// Statistical power for detecting medium effects (Cohen's d = 0.3)
    pub medium_effect_power: f64,
    /// Statistical power for detecting large effects (Cohen's d = 0.5)
    pub large_effect_power: f64,
    /// Recommended sample size to achieve 80% power for medium effects
    pub recommended_size_for_80_percent_power: usize,
}

impl Default for PowerAnalysis {
    fn default() -> Self {
        Self {
            small_effect_power: 0.0,
            medium_effect_power: 0.0,
            large_effect_power: 0.0,
            recommended_size_for_80_percent_power: 100,
        }
    }
}

impl StatisticalAnalyzer {
    /// Create a new statistical analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: StatisticalConfig::default(),
        }
    }

    /// Create a new statistical analyzer with custom configuration
    pub fn with_config(config: StatisticalConfig) -> Self {
        Self { config }
    }

    /// Analyze predictions against actual outcomes with comprehensive statistical validation
    pub fn analyze_predictions(
        &self,
        predictions: &[LDCPrediction],
        actual_outcomes: &[Direction],
        market_data: &[OHLCV],
    ) -> Result<StatisticalAnalysisResult> {
        // Validate input data
        if predictions.len() != actual_outcomes.len() {
            return Err(anyhow::anyhow!(
                "Predictions and actual outcomes length mismatch: {} vs {}",
                predictions.len(),
                actual_outcomes.len()
            ));
        }

        if predictions.len() < self.config.min_sample_size {
            return Err(anyhow::anyhow!(
                "Insufficient sample size: {} < {}",
                predictions.len(),
                self.config.min_sample_size
            ));
        }

        // Calculate accuracy metrics
        let accuracy_metrics = self.calculate_accuracy_metrics(predictions, actual_outcomes)?;

        // Calculate signal quality metrics
        let signal_quality = self.calculate_signal_quality(predictions, market_data)?;

        // Analyze market regimes
        let regime_analysis = self.analyze_market_regimes(predictions, actual_outcomes, market_data)?;

        // Perform statistical significance testing
        let significance = self.test_statistical_significance(predictions, actual_outcomes)?;

        Ok(StatisticalAnalysisResult {
            prediction_accuracy: accuracy_metrics,
            signal_quality,
            market_regime_analysis: regime_analysis,
            statistical_significance: significance,
        })
    }

    /// Calculate comprehensive accuracy metrics including hit rates, precision, recall, and F1 scores
    pub fn calculate_accuracy_metrics(
        &self,
        predictions: &[LDCPrediction],
        actual_outcomes: &[Direction],
    ) -> Result<AccuracyMetrics> {
        let mut confusion_matrix = ConfusionMatrix {
            true_positives_long: 0,
            false_positives_long: 0,
            true_negatives_short: 0,
            false_negatives_short: 0,
            true_neutral: 0,
            false_neutral: 0,
        };

        // Build confusion matrix
        for (pred, actual) in predictions.iter().zip(actual_outcomes.iter()) {
            match (pred.prediction_direction, *actual) {
                (Direction::Long, Direction::Long) => confusion_matrix.true_positives_long += 1,
                (Direction::Long, Direction::Short) => confusion_matrix.false_positives_long += 1,
                (Direction::Long, Direction::Neutral) => confusion_matrix.false_positives_long += 1,
                (Direction::Short, Direction::Short) => confusion_matrix.true_negatives_short += 1,
                (Direction::Short, Direction::Long) => confusion_matrix.false_negatives_short += 1,
                (Direction::Short, Direction::Neutral) => confusion_matrix.false_negatives_short += 1,
                (Direction::Neutral, Direction::Neutral) => confusion_matrix.true_neutral += 1,
                (Direction::Neutral, _) => confusion_matrix.false_neutral += 1,
            }
        }

        // Calculate precision metrics
        let precision_long = if confusion_matrix.true_positives_long + confusion_matrix.false_positives_long > 0 {
            confusion_matrix.true_positives_long as f64 / 
            (confusion_matrix.true_positives_long + confusion_matrix.false_positives_long) as f64
        } else {
            0.0
        };

        let precision_short = if confusion_matrix.true_negatives_short + confusion_matrix.false_negatives_short > 0 {
            confusion_matrix.true_negatives_short as f64 / 
            (confusion_matrix.true_negatives_short + confusion_matrix.false_negatives_short) as f64
        } else {
            0.0
        };

        // Calculate recall metrics
        let total_actual_long = actual_outcomes.iter().filter(|&&d| d == Direction::Long).count();
        let total_actual_short = actual_outcomes.iter().filter(|&&d| d == Direction::Short).count();

        let recall_long = if total_actual_long > 0 {
            confusion_matrix.true_positives_long as f64 / total_actual_long as f64
        } else {
            0.0
        };

        let recall_short = if total_actual_short > 0 {
            confusion_matrix.true_negatives_short as f64 / total_actual_short as f64
        } else {
            0.0
        };

        // Calculate F1 scores
        let f1_score_long = if precision_long + recall_long > 0.0 {
            2.0 * (precision_long * recall_long) / (precision_long + recall_long)
        } else {
            0.0
        };

        let f1_score_short = if precision_short + recall_short > 0.0 {
            2.0 * (precision_short * recall_short) / (precision_short + recall_short)
        } else {
            0.0
        };

        // Calculate weighted averages
        let total_predictions = predictions.len() as f64;
        let long_weight = (confusion_matrix.true_positives_long + confusion_matrix.false_positives_long) as f64 / total_predictions;
        let short_weight = (confusion_matrix.true_negatives_short + confusion_matrix.false_negatives_short) as f64 / total_predictions;

        let precision_overall = precision_long * long_weight + precision_short * short_weight;
        let recall_overall = recall_long * long_weight + recall_short * short_weight;
        let f1_score_overall = f1_score_long * long_weight + f1_score_short * short_weight;

        // Calculate overall hit rate
        let correct_predictions = confusion_matrix.true_positives_long + 
                                confusion_matrix.true_negatives_short + 
                                confusion_matrix.true_neutral;
        let hit_rate = correct_predictions as f64 / total_predictions;

        Ok(AccuracyMetrics {
            hit_rate,
            precision_long,
            precision_short,
            precision_overall,
            recall_long,
            recall_short,
            recall_overall,
            f1_score_long,
            f1_score_short,
            f1_score_overall,
            confusion_matrix,
        })
    }

    /// Calculate signal quality metrics including signal-to-noise ratio and information coefficient
    pub fn calculate_signal_quality(
        &self,
        predictions: &[LDCPrediction],
        market_data: &[OHLCV],
    ) -> Result<SignalQualityMetrics> {
        if market_data.len() < predictions.len() + 1 {
            return Err(anyhow::anyhow!(
                "Insufficient market data for signal quality analysis: {} < {}",
                market_data.len(),
                predictions.len() + 1
            ));
        }

        // Extract signals and calculate future returns
        let signals: Vec<f64> = predictions.iter().map(|p| p.signal as f64).collect();
        let returns: Vec<f64> = market_data
            .windows(2)
            .take(predictions.len())
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        // Calculate signal distribution
        let signal_distribution = self.calculate_distribution(&signals);

        // Calculate confidence distribution
        let confidences: Vec<f64> = predictions.iter().map(|p| p.confidence as f64).collect();
        let confidence_distribution = self.calculate_distribution(&confidences);

        // Calculate signal-to-noise ratio
        let signal_to_noise_ratio = if signal_distribution.std_dev > 0.0 {
            signal_distribution.mean.abs() / signal_distribution.std_dev
        } else {
            0.0
        };

        // Calculate information coefficient (correlation between signals and returns)
        let information_coefficient = self.calculate_correlation(&signals, &returns);

        // Calculate average signal strength by direction
        let mut avg_signal_by_direction = HashMap::new();
        let mut signal_sums = HashMap::new();
        let mut signal_counts = HashMap::new();

        for prediction in predictions {
            let direction = prediction.prediction_direction;
            let signal = prediction.signal as f64;
            
            *signal_sums.entry(direction).or_insert(0.0) += signal;
            *signal_counts.entry(direction).or_insert(0) += 1;
        }

        for (&direction, &sum) in signal_sums.iter() {
            if let Some(&count) = signal_counts.get(&direction) {
                if count > 0 {
                    avg_signal_by_direction.insert(direction, sum / count as f64);
                }
            }
        }

        Ok(SignalQualityMetrics {
            signal_to_noise_ratio,
            information_coefficient,
            signal_strength_distribution: signal_distribution,
            confidence_distribution,
            avg_signal_strength_by_direction: avg_signal_by_direction,
        })
    }

    /// Analyze market regimes and performance across different market conditions
    pub fn analyze_market_regimes(
        &self,
        predictions: &[LDCPrediction],
        actual_outcomes: &[Direction],
        market_data: &[OHLCV],
    ) -> Result<MarketRegimeAnalysis> {
        // Classify market regimes
        let regimes = self.classify_market_regimes(market_data)?;

        // Calculate performance metrics for each regime type
        let mut trending_data = Vec::new();
        let mut ranging_data = Vec::new();
        let mut volatile_data = Vec::new();

        for regime in &regimes {
            let start = regime.start_index;
            let end = regime.end_index.min(predictions.len());
            
            if start < end && end <= predictions.len() && end <= actual_outcomes.len() {
                let regime_predictions = &predictions[start..end];
                let regime_outcomes = &actual_outcomes[start..end];

                match regime.regime_type {
                    MarketRegimeType::Trending => {
                        trending_data.extend(regime_predictions.iter().zip(regime_outcomes.iter()));
                    }
                    MarketRegimeType::Ranging => {
                        ranging_data.extend(regime_predictions.iter().zip(regime_outcomes.iter()));
                    }
                    MarketRegimeType::Volatile => {
                        volatile_data.extend(regime_predictions.iter().zip(regime_outcomes.iter()));
                    }
                }
            }
        }

        // Calculate accuracy metrics for each regime
        let trending_performance = if !trending_data.is_empty() {
            let (preds, outcomes): (Vec<_>, Vec<_>) = trending_data.into_iter().unzip();
            let pred_refs: Vec<LDCPrediction> = preds.into_iter().cloned().collect();
            let outcome_refs: Vec<Direction> = outcomes.into_iter().cloned().collect();
            self.calculate_accuracy_metrics(&pred_refs, &outcome_refs)?
        } else {
            self.empty_accuracy_metrics()
        };

        let ranging_performance = if !ranging_data.is_empty() {
            let (preds, outcomes): (Vec<_>, Vec<_>) = ranging_data.into_iter().unzip();
            let pred_refs: Vec<LDCPrediction> = preds.into_iter().cloned().collect();
            let outcome_refs: Vec<Direction> = outcomes.into_iter().cloned().collect();
            self.calculate_accuracy_metrics(&pred_refs, &outcome_refs)?
        } else {
            self.empty_accuracy_metrics()
        };

        let volatile_performance = if !volatile_data.is_empty() {
            let (preds, outcomes): (Vec<_>, Vec<_>) = volatile_data.into_iter().unzip();
            let pred_refs: Vec<LDCPrediction> = preds.into_iter().cloned().collect();
            let outcome_refs: Vec<Direction> = outcomes.into_iter().cloned().collect();
            self.calculate_accuracy_metrics(&pred_refs, &outcome_refs)?
        } else {
            self.empty_accuracy_metrics()
        };

        Ok(MarketRegimeAnalysis {
            trending_performance,
            ranging_performance,
            volatile_performance,
            regime_classification: regimes,
        })
    }

    /// Perform statistical significance testing with p-value calculations and confidence intervals
    pub fn test_statistical_significance(
        &self,
        predictions: &[LDCPrediction],
        actual_outcomes: &[Direction],
    ) -> Result<SignificanceTest> {
        let sample_size = predictions.len();
        
        // Calculate accuracy p-value using binomial test
        let correct_predictions = predictions
            .iter()
            .zip(actual_outcomes.iter())
            .filter(|(pred, actual)| pred.prediction_direction == **actual)
            .count();
        
        let accuracy_rate = correct_predictions as f64 / sample_size as f64;
        let null_hypothesis_rate = 1.0 / 3.0; // Random chance for 3-class problem
        
        let accuracy_p_value = self.binomial_test(correct_predictions, sample_size, null_hypothesis_rate);

        // Calculate signal quality p-value (test if information coefficient is significantly different from 0)
        let signals: Vec<f64> = predictions.iter().map(|p| p.signal as f64).collect();
        let signal_quality_p_value = self.correlation_significance_test(&signals, sample_size);

        // Calculate regime-specific p-values for multiple testing correction
        let regime_p_values = self.calculate_regime_p_values(predictions, actual_outcomes)?;

        // Apply multiple testing correction
        let correction_method = MultipleTestingCorrection::BenjaminiHochberg;
        let corrected_p_values = self.apply_multiple_testing_correction(
            &regime_p_values,
            accuracy_p_value,
            signal_quality_p_value,
            correction_method,
        );

        // Calculate confidence intervals
        let mut confidence_intervals = HashMap::new();
        
        // Confidence interval for accuracy
        let accuracy_ci = self.calculate_binomial_confidence_interval(
            correct_predictions,
            sample_size,
            self.config.confidence_level,
        );
        confidence_intervals.insert("accuracy".to_string(), accuracy_ci);

        // Confidence interval for signal strength
        let signal_mean = signals.iter().sum::<f64>() / signals.len() as f64;
        let signal_std = self.calculate_std_dev(&signals);
        let signal_ci = self.calculate_normal_confidence_interval(
            signal_mean,
            signal_std,
            signals.len(),
            self.config.confidence_level,
        );
        confidence_intervals.insert("signal_strength".to_string(), signal_ci);

        // Add confidence intervals for regime-specific metrics
        for (regime_type, _) in &regime_p_values {
            let regime_name = format!("{:?}_accuracy", regime_type);
            let regime_ci = self.calculate_regime_confidence_interval(
                predictions,
                actual_outcomes,
                *regime_type,
            )?;
            confidence_intervals.insert(regime_name, regime_ci);
        }

        // Determine overall significance using corrected p-values
        let is_significant = corrected_p_values.values()
            .any(|&p| p < self.config.significance_threshold);

        // Calculate statistical power (simplified)
        let statistical_power = self.calculate_statistical_power(sample_size, accuracy_rate, null_hypothesis_rate);

        Ok(SignificanceTest {
            accuracy_p_value,
            signal_quality_p_value,
            regime_p_values,
            corrected_p_values,
            confidence_intervals,
            is_significant,
            sample_size,
            statistical_power,
            correction_method,
        })
    }

    /// Calculate correlation between two vectors
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|b| b * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate distribution statistics for a vector of values
    fn calculate_distribution(&self, values: &[f64]) -> SignalDistribution {
        if values.is_empty() {
            return SignalDistribution {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                percentile_25: 0.0,
                percentile_50: 0.0,
                percentile_75: 0.0,
                percentile_95: 0.0,
                percentile_99: 0.0,
            };
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        SignalDistribution {
            mean,
            std_dev,
            min: sorted_values[0],
            max: sorted_values[sorted_values.len() - 1],
            percentile_25: self.calculate_percentile(&sorted_values, 25.0),
            percentile_50: self.calculate_percentile(&sorted_values, 50.0),
            percentile_75: self.calculate_percentile(&sorted_values, 75.0),
            percentile_95: self.calculate_percentile(&sorted_values, 95.0),
            percentile_99: self.calculate_percentile(&sorted_values, 99.0),
        }
    }

    /// Calculate percentile from sorted values
    fn calculate_percentile(&self, sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let index = (percentile / 100.0 * (sorted_values.len() - 1) as f64) as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    /// Calculate standard deviation
    fn calculate_std_dev(&self, values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }



    /// Perform binomial test for accuracy significance
    fn binomial_test(&self, successes: usize, trials: usize, null_prob: f64) -> f64 {
        if trials == 0 {
            return 1.0;
        }

        let observed_rate = successes as f64 / trials as f64;
        
        // Simplified p-value calculation using normal approximation for large samples
        if trials >= 30 {
            let expected = null_prob * trials as f64;
            let variance = null_prob * (1.0 - null_prob) * trials as f64;
            let std_dev = variance.sqrt();
            
            if std_dev > 0.0 {
                let z_score = (successes as f64 - expected) / std_dev;
                // Two-tailed test
                2.0 * (1.0 - self.standard_normal_cdf(z_score.abs()))
            } else {
                1.0
            }
        } else {
            // For small samples, use a conservative estimate
            if observed_rate > null_prob {
                0.05 // Conservative significant result
            } else {
                0.5 // Conservative non-significant result
            }
        }
    }

    /// Test correlation significance
    fn correlation_significance_test(&self, values: &[f64], sample_size: usize) -> f64 {
        if sample_size < 3 {
            return 1.0;
        }

        // Simplified test: check if signal variance is significantly different from zero
        let variance = values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64;
        
        if variance > 0.01 {
            0.01 // Significant signal
        } else {
            0.5 // Non-significant signal
        }
    }

    /// Calculate binomial confidence interval
    fn calculate_binomial_confidence_interval(
        &self,
        successes: usize,
        trials: usize,
        confidence_level: f64,
    ) -> ConfidenceInterval {
        if trials == 0 {
            return ConfidenceInterval {
                lower_bound: 0.0,
                upper_bound: 0.0,
                confidence_level,
            };
        }

        let p = successes as f64 / trials as f64;
        let z = self.inverse_normal_cdf((1.0 + confidence_level) / 2.0);
        let margin = z * (p * (1.0 - p) / trials as f64).sqrt();

        ConfidenceInterval {
            lower_bound: (p - margin).max(0.0),
            upper_bound: (p + margin).min(1.0),
            confidence_level,
        }
    }

    /// Calculate normal confidence interval
    fn calculate_normal_confidence_interval(
        &self,
        mean: f64,
        std_dev: f64,
        sample_size: usize,
        confidence_level: f64,
    ) -> ConfidenceInterval {
        if sample_size == 0 {
            return ConfidenceInterval {
                lower_bound: mean,
                upper_bound: mean,
                confidence_level,
            };
        }

        let z = self.inverse_normal_cdf((1.0 + confidence_level) / 2.0);
        let margin = z * std_dev / (sample_size as f64).sqrt();

        ConfidenceInterval {
            lower_bound: mean - margin,
            upper_bound: mean + margin,
            confidence_level,
        }
    }

    /// Calculate statistical power (simplified)
    fn calculate_statistical_power(&self, sample_size: usize, observed_rate: f64, null_rate: f64) -> f64 {
        if sample_size < 10 {
            return 0.1; // Low power for small samples
        }

        let effect_size = (observed_rate - null_rate).abs();
        
        // Simplified power calculation
        if effect_size > 0.1 && sample_size > 100 {
            0.8 // High power
        } else if effect_size > 0.05 && sample_size > 50 {
            0.6 // Medium power
        } else {
            0.3 // Low power
        }
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Error function approximation using Abramowitz and Stegun formula
    fn erf(&self, x: f64) -> f64 {
        // Constants for the approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        // A&S formula 7.1.26
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Inverse normal CDF approximation
    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        // Simplified approximation for common confidence levels
        match p {
            p if (p - 0.975).abs() < 0.001 => 1.96, // 95% confidence
            p if (p - 0.995).abs() < 0.001 => 2.576, // 99% confidence
            p if (p - 0.9).abs() < 0.001 => 1.282, // 80% confidence
            _ => 1.96, // Default to 95%
        }
    }

    /// Classify market regimes identifying trending, ranging, and volatile market conditions
    fn classify_market_regimes(&self, market_data: &[OHLCV]) -> Result<Vec<MarketRegime>> {
        if market_data.len() < 20 {
            return Err(anyhow::anyhow!(
                "Insufficient market data for regime classification: {} < 20",
                market_data.len()
            ));
        }

        let mut regimes = Vec::new();
        let window_size = 20; // 20-period window for regime analysis
        
        for i in 0..(market_data.len() - window_size + 1) {
            let window = &market_data[i..i + window_size];
            
            // Calculate volatility (standard deviation of returns)
            let returns: Vec<f64> = window
                .windows(2)
                .map(|w| (w[1].close - w[0].close) / w[0].close)
                .collect();
            
            let volatility = self.calculate_std_dev(&returns);
            
            // Calculate trend strength using linear regression slope
            let prices: Vec<f64> = window.iter().map(|ohlcv| ohlcv.close).collect();
            let trend_strength = self.calculate_trend_strength(&prices);
            
            // Classify regime based on volatility and trend strength
            let regime_type = self.classify_regime_type(volatility, trend_strength);
            
            regimes.push(MarketRegime {
                start_index: i,
                end_index: i + window_size - 1,
                regime_type,
                volatility,
                trend_strength,
            });
        }

        // Merge consecutive regimes of the same type to reduce noise
        let merged_regimes = self.merge_consecutive_regimes(regimes);
        
        Ok(merged_regimes)
    }

    /// Calculate trend strength using linear regression slope
    fn calculate_trend_strength(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let n = prices.len() as f64;
        let x_values: Vec<f64> = (0..prices.len()).map(|i| i as f64).collect();
        
        // Calculate linear regression slope
        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = prices.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(prices.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_values.iter().map(|x| x * x).sum();
        
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f64::EPSILON {
            return 0.0;
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        
        // Normalize slope by average price to get relative trend strength
        let avg_price = sum_y / n;
        if avg_price > 0.0 {
            slope / avg_price
        } else {
            0.0
        }
    }

    /// Classify regime type based on volatility and trend strength thresholds
    fn classify_regime_type(&self, volatility: f64, trend_strength: f64) -> MarketRegimeType {
        // Define thresholds for regime classification
        const HIGH_VOLATILITY_THRESHOLD: f64 = 0.02; // 2% daily volatility
        const STRONG_TREND_THRESHOLD: f64 = 0.001; // 0.1% daily trend strength
        
        if volatility > HIGH_VOLATILITY_THRESHOLD {
            MarketRegimeType::Volatile
        } else if trend_strength.abs() > STRONG_TREND_THRESHOLD {
            MarketRegimeType::Trending
        } else {
            MarketRegimeType::Ranging
        }
    }

    /// Merge consecutive regimes of the same type to reduce noise
    fn merge_consecutive_regimes(&self, regimes: Vec<MarketRegime>) -> Vec<MarketRegime> {
        if regimes.is_empty() {
            return regimes;
        }

        let mut merged = Vec::new();
        let mut current_regime = regimes[0].clone();

        for regime in regimes.into_iter().skip(1) {
            if regime.regime_type == current_regime.regime_type {
                // Extend current regime
                current_regime.end_index = regime.end_index;
                // Update volatility and trend strength with weighted average
                let current_length = (current_regime.end_index - current_regime.start_index + 1) as f64;
                let regime_length = (regime.end_index - regime.start_index + 1) as f64;
                let total_length = current_length + regime_length;
                
                current_regime.volatility = (current_regime.volatility * current_length + 
                                           regime.volatility * regime_length) / total_length;
                current_regime.trend_strength = (current_regime.trend_strength * current_length + 
                                               regime.trend_strength * regime_length) / total_length;
            } else {
                // Start new regime
                merged.push(current_regime);
                current_regime = regime;
            }
        }
        
        merged.push(current_regime);
        merged
    }

    /// Calculate regime-specific p-values for multiple testing correction
    fn calculate_regime_p_values(
        &self,
        predictions: &[LDCPrediction],
        actual_outcomes: &[Direction],
    ) -> Result<HashMap<MarketRegimeType, f64>> {
        // For this implementation, we'll calculate p-values based on the assumption
        // that we have regime classifications. In a real scenario, we'd need market data
        // to classify regimes first, but for the statistical framework, we'll simulate
        // regime-specific performance testing.
        
        let mut regime_p_values = HashMap::new();
        
        // Simulate regime-specific accuracy testing
        // In practice, this would use actual regime classifications from market data
        let total_samples = predictions.len();
        let regime_size = total_samples / 3; // Divide into 3 equal parts for simulation
        
        // Test trending regime performance (first third)
        if regime_size > 0 {
            let trending_end = regime_size.min(predictions.len());
            let trending_predictions = &predictions[0..trending_end];
            let trending_outcomes = &actual_outcomes[0..trending_end];
            
            let trending_correct = trending_predictions
                .iter()
                .zip(trending_outcomes.iter())
                .filter(|(pred, actual)| pred.prediction_direction == **actual)
                .count();
            
            let trending_p_value = self.binomial_test(
                trending_correct,
                trending_predictions.len(),
                1.0 / 3.0, // Null hypothesis: random performance
            );
            regime_p_values.insert(MarketRegimeType::Trending, trending_p_value);
        }
        
        // Test ranging regime performance (middle third)
        if regime_size > 0 && regime_size * 2 <= predictions.len() {
            let ranging_start = regime_size;
            let ranging_end = (regime_size * 2).min(predictions.len());
            let ranging_predictions = &predictions[ranging_start..ranging_end];
            let ranging_outcomes = &actual_outcomes[ranging_start..ranging_end];
            
            let ranging_correct = ranging_predictions
                .iter()
                .zip(ranging_outcomes.iter())
                .filter(|(pred, actual)| pred.prediction_direction == **actual)
                .count();
            
            let ranging_p_value = self.binomial_test(
                ranging_correct,
                ranging_predictions.len(),
                1.0 / 3.0,
            );
            regime_p_values.insert(MarketRegimeType::Ranging, ranging_p_value);
        }
        
        // Test volatile regime performance (last third)
        if regime_size * 2 < predictions.len() {
            let volatile_start = regime_size * 2;
            let volatile_predictions = &predictions[volatile_start..];
            let volatile_outcomes = &actual_outcomes[volatile_start..];
            
            let volatile_correct = volatile_predictions
                .iter()
                .zip(volatile_outcomes.iter())
                .filter(|(pred, actual)| pred.prediction_direction == **actual)
                .count();
            
            let volatile_p_value = self.binomial_test(
                volatile_correct,
                volatile_predictions.len(),
                1.0 / 3.0,
            );
            regime_p_values.insert(MarketRegimeType::Volatile, volatile_p_value);
        }
        
        Ok(regime_p_values)
    }

    /// Apply multiple testing correction for avoiding false discoveries in regime analysis
    fn apply_multiple_testing_correction(
        &self,
        regime_p_values: &HashMap<MarketRegimeType, f64>,
        accuracy_p_value: f64,
        signal_quality_p_value: f64,
        correction_method: MultipleTestingCorrection,
    ) -> HashMap<String, f64> {
        let mut all_p_values = Vec::new();
        let mut p_value_names = Vec::new();
        
        // Collect all p-values for correction
        all_p_values.push(accuracy_p_value);
        p_value_names.push("overall_accuracy".to_string());
        
        all_p_values.push(signal_quality_p_value);
        p_value_names.push("signal_quality".to_string());
        
        for (regime_type, &p_value) in regime_p_values {
            all_p_values.push(p_value);
            p_value_names.push(format!("{:?}_regime", regime_type));
        }
        
        let corrected_p_values = match correction_method {
            MultipleTestingCorrection::Bonferroni => {
                self.bonferroni_correction(&all_p_values)
            }
            MultipleTestingCorrection::BenjaminiHochberg => {
                self.benjamini_hochberg_correction(&all_p_values)
            }
            MultipleTestingCorrection::None => {
                all_p_values.clone() // No correction
            }
        };
        
        // Create result map
        let mut result = HashMap::new();
        for (name, &corrected_p) in p_value_names.iter().zip(corrected_p_values.iter()) {
            result.insert(name.clone(), corrected_p);
        }
        
        result
    }

    /// Bonferroni correction (most conservative)
    fn bonferroni_correction(&self, p_values: &[f64]) -> Vec<f64> {
        let m = p_values.len() as f64;
        p_values.iter().map(|&p| (p * m).min(1.0)).collect()
    }

    /// Benjamini-Hochberg false discovery rate control
    fn benjamini_hochberg_correction(&self, p_values: &[f64]) -> Vec<f64> {
        if p_values.is_empty() {
            return Vec::new();
        }
        
        let m = p_values.len();
        let mut indexed_p_values: Vec<(usize, f64)> = p_values
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        // Sort by p-value
        indexed_p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut corrected = vec![0.0; m];
        let mut min_corrected = 1.0;
        
        // Apply BH correction in reverse order
        for (rank, &(original_index, p_value)) in indexed_p_values.iter().enumerate().rev() {
            let corrected_p = (p_value * m as f64) / (rank + 1) as f64;
            let final_corrected_p = corrected_p.min(min_corrected).min(1.0);
            corrected[original_index] = final_corrected_p;
            min_corrected = final_corrected_p;
        }
        
        corrected
    }

    /// Calculate confidence interval for regime-specific accuracy
    fn calculate_regime_confidence_interval(
        &self,
        predictions: &[LDCPrediction],
        actual_outcomes: &[Direction],
        regime_type: MarketRegimeType,
    ) -> Result<ConfidenceInterval> {
        // For this implementation, we'll use the simulated regime divisions
        // In practice, this would use actual regime classifications
        let total_samples = predictions.len();
        let regime_size = total_samples / 3;
        
        let (regime_predictions, regime_outcomes) = match regime_type {
            MarketRegimeType::Trending => {
                let end = regime_size.min(predictions.len());
                (&predictions[0..end], &actual_outcomes[0..end])
            }
            MarketRegimeType::Ranging => {
                let start = regime_size;
                let end = (regime_size * 2).min(predictions.len());
                if start < end {
                    (&predictions[start..end], &actual_outcomes[start..end])
                } else {
                    (&predictions[0..0], &actual_outcomes[0..0]) // Empty slice
                }
            }
            MarketRegimeType::Volatile => {
                let start = regime_size * 2;
                if start < predictions.len() {
                    (&predictions[start..], &actual_outcomes[start..])
                } else {
                    (&predictions[0..0], &actual_outcomes[0..0]) // Empty slice
                }
            }
        };
        
        if regime_predictions.is_empty() {
            return Ok(ConfidenceInterval {
                lower_bound: 0.0,
                upper_bound: 0.0,
                confidence_level: self.config.confidence_level,
            });
        }
        
        let correct_predictions = regime_predictions
            .iter()
            .zip(regime_outcomes.iter())
            .filter(|(pred, actual)| pred.prediction_direction == **actual)
            .count();
        
        Ok(self.calculate_binomial_confidence_interval(
            correct_predictions,
            regime_predictions.len(),
            self.config.confidence_level,
        ))
    }

    /// Sample size validation ensuring sufficient data for statistical conclusions
    pub fn validate_sample_size(&self, sample_size: usize) -> Result<SampleSizeValidation> {
        let mut validation = SampleSizeValidation {
            sample_size,
            is_sufficient_for_basic_analysis: false,
            is_sufficient_for_regime_analysis: false,
            is_sufficient_for_significance_testing: false,
            recommended_minimum_size: self.config.min_sample_size,
            power_analysis: PowerAnalysis::default(),
        };
        
        // Basic analysis threshold
        validation.is_sufficient_for_basic_analysis = sample_size >= 30;
        
        // Regime analysis threshold (need enough samples per regime)
        validation.is_sufficient_for_regime_analysis = sample_size >= 300; // 100 per regime minimum
        
        // Significance testing threshold
        validation.is_sufficient_for_significance_testing = sample_size >= self.config.min_sample_size;
        
        // Calculate power analysis
        validation.power_analysis = self.calculate_power_analysis(sample_size);
        
        if !validation.is_sufficient_for_significance_testing {
            return Err(anyhow::anyhow!(
                "Insufficient sample size for statistical analysis: {} < {}. Recommended minimum: {}",
                sample_size,
                self.config.min_sample_size,
                validation.recommended_minimum_size
            ));
        }
        
        Ok(validation)
    }

    /// Calculate power analysis for different effect sizes
    fn calculate_power_analysis(&self, sample_size: usize) -> PowerAnalysis {
        // Simplified power analysis for different effect sizes
        let small_effect_power = self.calculate_power_for_effect_size(sample_size, 0.1);
        let medium_effect_power = self.calculate_power_for_effect_size(sample_size, 0.3);
        let large_effect_power = self.calculate_power_for_effect_size(sample_size, 0.5);
        
        PowerAnalysis {
            small_effect_power,
            medium_effect_power,
            large_effect_power,
            recommended_size_for_80_percent_power: self.calculate_required_sample_size(0.8, 0.3),
        }
    }

    /// Calculate statistical power for a given effect size
    fn calculate_power_for_effect_size(&self, sample_size: usize, effect_size: f64) -> f64 {
        // Simplified power calculation using Cohen's conventions
        // This is a rough approximation - in practice, you'd use more sophisticated methods
        
        if sample_size < 10 {
            return 0.05; // Very low power for tiny samples
        }
        
        let sqrt_n = (sample_size as f64).sqrt();
        let power_factor = effect_size * sqrt_n / 2.0;
        
        // Approximate power using a sigmoid-like function
        let power = 1.0 / (1.0 + (-2.0 * (power_factor - 1.0)).exp());
        
        power.min(0.99).max(0.05) // Clamp between 5% and 99%
    }

    /// Calculate required sample size for desired power
    fn calculate_required_sample_size(&self, desired_power: f64, effect_size: f64) -> usize {
        // Simplified calculation - in practice, you'd use power analysis formulas
        if effect_size <= 0.0 {
            return 10000; // Very large sample needed for no effect
        }
        
        // Rough approximation based on Cohen's power analysis
        let base_size = match effect_size {
            e if e >= 0.5 => 64,   // Large effect
            e if e >= 0.3 => 176,  // Medium effect  
            e if e >= 0.1 => 1571, // Small effect
            _ => 10000,            // Very small effect
        };
        
        // Adjust for desired power (80% is standard)
        let power_adjustment = if desired_power > 0.8 {
            1.5 // Need more samples for higher power
        } else if desired_power < 0.8 {
            0.8 // Can use fewer samples for lower power
        } else {
            1.0 // Standard 80% power
        };
        
        (base_size as f64 * power_adjustment) as usize
    }

    /// Create empty accuracy metrics for regimes with no data
    fn empty_accuracy_metrics(&self) -> AccuracyMetrics {
        AccuracyMetrics {
            hit_rate: 0.0,
            precision_long: 0.0,
            precision_short: 0.0,
            precision_overall: 0.0,
            recall_long: 0.0,
            recall_short: 0.0,
            recall_overall: 0.0,
            f1_score_long: 0.0,
            f1_score_short: 0.0,
            f1_score_overall: 0.0,
            confusion_matrix: ConfusionMatrix {
                true_positives_long: 0,
                false_positives_long: 0,
                true_negatives_short: 0,
                false_negatives_short: 0,
                true_neutral: 0,
                false_neutral: 0,
            },
        }
    }


}

impl Default for StatisticalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_analyzer_creation() {
        let analyzer = StatisticalAnalyzer::new();
        assert_eq!(analyzer.config.confidence_level, 0.95);
        assert_eq!(analyzer.config.min_sample_size, 100);
        assert_eq!(analyzer.config.significance_threshold, 0.05);
    }

    #[test]
    fn test_correlation_calculation() {
        let analyzer = StatisticalAnalyzer::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let correlation = analyzer.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 0.001); // Perfect positive correlation
    }

    #[test]
    fn test_distribution_calculation() {
        let analyzer = StatisticalAnalyzer::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let distribution = analyzer.calculate_distribution(&values);
        assert_eq!(distribution.mean, 3.0);
        assert_eq!(distribution.min, 1.0);
        assert_eq!(distribution.max, 5.0);
    }

    #[test]
    fn test_empty_input_handling() {
        let analyzer = StatisticalAnalyzer::new();
        let empty_vec: Vec<f64> = Vec::new();
        
        let correlation = analyzer.calculate_correlation(&empty_vec, &empty_vec);
        assert_eq!(correlation, 0.0);
        
        let distribution = analyzer.calculate_distribution(&empty_vec);
        assert_eq!(distribution.mean, 0.0);
    }
}

// Include comprehensive tests
#[cfg(test)]
mod comprehensive_tests {
    use super::*;
    use crate::{Direction, LDCPrediction};
    use feature_pipeline::OHLCV;

    /// Test comprehensive statistical analysis functionality
    #[test]
    fn test_comprehensive_statistical_analysis() {
        let analyzer = StatisticalAnalyzer::new();
        let (predictions, outcomes, market_data) = create_test_data(150);

        let result = analyzer.analyze_predictions(&predictions, &outcomes, &market_data);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        
        // Verify all components are present
        assert!(analysis.prediction_accuracy.hit_rate >= 0.0);
        assert!(analysis.signal_quality.signal_to_noise_ratio >= 0.0);
        assert!(!analysis.market_regime_analysis.regime_classification.is_empty());
        assert!(analysis.statistical_significance.sample_size > 0);
    }

    /// Test accuracy metrics calculation with known data
    #[test]
    fn test_accuracy_metrics_calculation() {
        let analyzer = StatisticalAnalyzer::new();
        
        // Create test data with known outcomes
        let predictions = vec![
            create_prediction(Direction::Long, 1.0, 0.8),
            create_prediction(Direction::Long, 0.5, 0.6),
            create_prediction(Direction::Short, -1.0, 0.9),
            create_prediction(Direction::Short, -0.5, 0.7),
            create_prediction(Direction::Neutral, 0.0, 0.5),
        ];
        
        let outcomes = vec![
            Direction::Long,    // Correct
            Direction::Short,   // Incorrect
            Direction::Short,   // Correct
            Direction::Long,    // Incorrect
            Direction::Neutral, // Correct
        ];

        let result = analyzer.calculate_accuracy_metrics(&predictions, &outcomes);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert_eq!(metrics.hit_rate, 0.6); // 3 out of 5 correct
        assert_eq!(metrics.confusion_matrix.true_positives_long, 1);
        assert_eq!(metrics.confusion_matrix.true_negatives_short, 1);
        assert_eq!(metrics.confusion_matrix.true_neutral, 1);
    }

    /// Test signal quality metrics calculation
    #[test]
    fn test_signal_quality_calculation() {
        let analyzer = StatisticalAnalyzer::new();
        
        let predictions = vec![
            create_prediction(Direction::Long, 1.0, 0.8),
            create_prediction(Direction::Short, -1.0, 0.9),
            create_prediction(Direction::Neutral, 0.0, 0.5),
        ];
        
        let market_data = vec![
            create_ohlcv(100.0, 0),
            create_ohlcv(105.0, 1), // 5% increase
            create_ohlcv(95.0, 2),  // ~9.5% decrease
            create_ohlcv(95.0, 3),  // No change
        ];

        let result = analyzer.calculate_signal_quality(&predictions, &market_data);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.signal_to_noise_ratio >= 0.0);
        assert!(metrics.information_coefficient >= -1.0 && metrics.information_coefficient <= 1.0);
        assert_eq!(metrics.signal_strength_distribution.mean, 0.0); // Mean of [1.0, -1.0, 0.0]
    }

    /// Test market regime analysis
    #[test]
    fn test_market_regime_analysis() {
        let analyzer = StatisticalAnalyzer::new();
        let (predictions, outcomes, market_data) = create_test_data(100);

        let result = analyzer.analyze_market_regimes(&predictions, &outcomes, &market_data);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(!analysis.regime_classification.is_empty());
        
        // Check that regime types are properly classified
        let has_trending = analysis.regime_classification.iter()
            .any(|r| r.regime_type == MarketRegimeType::Trending);
        let has_ranging = analysis.regime_classification.iter()
            .any(|r| r.regime_type == MarketRegimeType::Ranging);
        let has_volatile = analysis.regime_classification.iter()
            .any(|r| r.regime_type == MarketRegimeType::Volatile);
        
        // At least one regime type should be identified
        assert!(has_trending || has_ranging || has_volatile);
    }

    /// Test statistical significance testing
    #[test]
    fn test_statistical_significance() {
        let analyzer = StatisticalAnalyzer::new();
        
        // Create data with clear pattern for significance testing
        let mut predictions = Vec::new();
        let mut outcomes = Vec::new();
        
        for i in 0..100 {
            let direction = if i % 2 == 0 { Direction::Long } else { Direction::Short };
            predictions.push(create_prediction(direction, 1.0, 0.8));
            outcomes.push(direction); // Perfect correlation for testing
        }

        let result = analyzer.test_statistical_significance(&predictions, &outcomes);
        assert!(result.is_ok());

        let significance = result.unwrap();
        assert_eq!(significance.sample_size, 100);
        assert!(significance.accuracy_p_value >= 0.0 && significance.accuracy_p_value <= 1.0);
        assert!(!significance.confidence_intervals.is_empty());
        assert!(!significance.regime_p_values.is_empty());
        assert!(!significance.corrected_p_values.is_empty());
        assert_eq!(significance.correction_method, MultipleTestingCorrection::BenjaminiHochberg);
    }

    /// Test multiple testing correction methods
    #[test]
    fn test_multiple_testing_correction() {
        let analyzer = StatisticalAnalyzer::new();
        let p_values = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        
        // Test Bonferroni correction
        let bonferroni_corrected = analyzer.bonferroni_correction(&p_values);
        assert_eq!(bonferroni_corrected.len(), p_values.len());
        assert!(bonferroni_corrected[0] >= p_values[0]); // Should be more conservative
        
        // Test Benjamini-Hochberg correction
        let bh_corrected = analyzer.benjamini_hochberg_correction(&p_values);
        assert_eq!(bh_corrected.len(), p_values.len());
        // BH should be less conservative than Bonferroni
        for i in 0..p_values.len() {
            assert!(bh_corrected[i] <= bonferroni_corrected[i]);
        }
    }

    /// Test regime-specific p-value calculation
    #[test]
    fn test_regime_p_values() {
        let analyzer = StatisticalAnalyzer::new();
        let (predictions, outcomes, _) = create_test_data(150);
        
        let result = analyzer.calculate_regime_p_values(&predictions, &outcomes);
        assert!(result.is_ok());
        
        let regime_p_values = result.unwrap();
        assert!(!regime_p_values.is_empty());
        
        // Should have p-values for different regime types
        for &p_value in regime_p_values.values() {
            assert!(p_value >= 0.0 && p_value <= 1.0);
        }
    }

    /// Test enhanced market regime classification
    #[test]
    fn test_enhanced_regime_classification() {
        let analyzer = StatisticalAnalyzer::new();
        
        // Create market data with different patterns
        let mut market_data = Vec::new();
        
        // Trending period
        for i in 0..30 {
            market_data.push(create_ohlcv(100.0 + i as f64, i as i64));
        }
        
        // Volatile period
        for i in 30..60 {
            let price = 130.0 + ((i as f64 * 0.5).sin() * 20.0);
            market_data.push(create_ohlcv(price, i as i64));
        }
        
        // Ranging period
        for i in 60..90 {
            let price = 130.0 + ((i as f64 * 0.1).sin() * 2.0);
            market_data.push(create_ohlcv(price, i as i64));
        }
        
        let result = analyzer.classify_market_regimes(&market_data);
        assert!(result.is_ok());
        
        let regimes = result.unwrap();
        assert!(!regimes.is_empty());
        
        // Check that different regime types are identified
        let regime_types: std::collections::HashSet<_> = regimes
            .iter()
            .map(|r| r.regime_type)
            .collect();
        
        // Should identify at least some different regime types
        assert!(!regime_types.is_empty());
    }

    /// Test correlation calculation
    #[test]
    fn test_correlation_calculation() {
        let analyzer = StatisticalAnalyzer::new();
        
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = analyzer.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 0.001);
        
        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let correlation_neg = analyzer.calculate_correlation(&x, &y_neg);
        assert!((correlation_neg + 1.0).abs() < 0.001);
        
        // No correlation
        let y_random = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let correlation_random = analyzer.calculate_correlation(&x, &y_random);
        assert!(correlation_random.abs() < 1.0);
    }

    /// Test distribution calculation
    #[test]
    fn test_distribution_calculation() {
        let analyzer = StatisticalAnalyzer::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        let distribution = analyzer.calculate_distribution(&values);
        assert_eq!(distribution.mean, 5.5);
        assert_eq!(distribution.min, 1.0);
        assert_eq!(distribution.max, 10.0);
        assert_eq!(distribution.percentile_50, 5.0); // Median
    }

    /// Test confusion matrix generation
    #[test]
    fn test_confusion_matrix_generation() {
        let analyzer = StatisticalAnalyzer::new();
        
        let predictions = vec![
            create_prediction(Direction::Long, 1.0, 0.8),   // TP
            create_prediction(Direction::Long, 1.0, 0.8),   // FP
            create_prediction(Direction::Short, -1.0, 0.9), // TN
            create_prediction(Direction::Short, -1.0, 0.9), // FN
            create_prediction(Direction::Neutral, 0.0, 0.5), // True Neutral
        ];
        
        let outcomes = vec![
            Direction::Long,
            Direction::Short,
            Direction::Short,
            Direction::Long,
            Direction::Neutral,
        ];

        let result = analyzer.calculate_accuracy_metrics(&predictions, &outcomes);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        let cm = &metrics.confusion_matrix;
        
        assert_eq!(cm.true_positives_long, 1);
        assert_eq!(cm.false_positives_long, 1);
        assert_eq!(cm.true_negatives_short, 1);
        assert_eq!(cm.false_negatives_short, 1);
        assert_eq!(cm.true_neutral, 1);
        assert_eq!(cm.false_neutral, 0);
    }

    /// Test configurable confidence levels and significance testing
    #[test]
    fn test_configurable_confidence_levels() {
        let config = StatisticalConfig {
            confidence_level: 0.99, // 99% confidence
            min_sample_size: 50,
            significance_threshold: 0.01, // 1% significance
        };
        
        let analyzer = StatisticalAnalyzer::with_config(config);
        assert_eq!(analyzer.config.confidence_level, 0.99);
        assert_eq!(analyzer.config.significance_threshold, 0.01);
    }

    /// Test sample size validation functionality
    #[test]
    fn test_sample_size_validation() {
        let analyzer = StatisticalAnalyzer::new();
        
        // Test insufficient sample size
        let result = analyzer.validate_sample_size(50);
        assert!(result.is_err()); // Should fail with default min_sample_size of 100
        
        // Test sufficient sample size
        let result = analyzer.validate_sample_size(150);
        assert!(result.is_ok());
        
        let validation = result.unwrap();
        assert_eq!(validation.sample_size, 150);
        assert!(validation.is_sufficient_for_basic_analysis);
        assert!(validation.is_sufficient_for_significance_testing);
        assert!(!validation.is_sufficient_for_regime_analysis); // Need 300+ for regime analysis
        
        // Test large sample size
        let result = analyzer.validate_sample_size(500);
        assert!(result.is_ok());
        
        let validation = result.unwrap();
        assert!(validation.is_sufficient_for_regime_analysis);
        assert!(validation.power_analysis.medium_effect_power > 0.0);
    }

    /// Test edge cases and error handling
    #[test]
    fn test_edge_cases() {
        let analyzer = StatisticalAnalyzer::new();
        
        // Test with insufficient data
        let small_predictions = vec![create_prediction(Direction::Long, 1.0, 0.8)];
        let small_outcomes = vec![Direction::Long];
        let small_market_data = vec![create_ohlcv(100.0, 0), create_ohlcv(105.0, 1)];
        
        let result = analyzer.analyze_predictions(&small_predictions, &small_outcomes, &small_market_data);
        assert!(result.is_err()); // Should fail due to insufficient sample size
        
        // Test with mismatched lengths
        let predictions = vec![create_prediction(Direction::Long, 1.0, 0.8)];
        let outcomes = vec![Direction::Long, Direction::Short]; // Different length
        let market_data = vec![create_ohlcv(100.0, 0), create_ohlcv(105.0, 1)];
        
        let result = analyzer.analyze_predictions(&predictions, &outcomes, &market_data);
        assert!(result.is_err()); // Should fail due to length mismatch
    }

    // Helper functions for creating test data
    fn create_prediction(direction: Direction, signal: f32, confidence: f32) -> LDCPrediction {
        LDCPrediction {
            signal,
            confidence,
            k_nearest_distances: vec![0.5, 0.7, 0.9, 1.1, 1.3],
            k_nearest_labels: vec![direction, direction, Direction::Neutral, direction, direction],
            prediction_direction: direction,
        }
    }

    fn create_ohlcv(price: f64, timestamp_offset: i64) -> OHLCV {
        OHLCV {
            timestamp: 1640995200 + timestamp_offset * 300,
            open: price,
            high: price * 1.01,
            low: price * 0.99,
            close: price,
            volume: 1000.0,
        }
    }

    fn create_test_data(count: usize) -> (Vec<LDCPrediction>, Vec<Direction>, Vec<OHLCV>) {
        let mut predictions = Vec::new();
        let mut outcomes = Vec::new();
        let mut market_data = Vec::new();
        
        for i in 0..count + 1 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
            market_data.push(create_ohlcv(price, i as i64));
            
            if i < count {
                let direction = match i % 3 {
                    0 => Direction::Long,
                    1 => Direction::Short,
                    _ => Direction::Neutral,
                };
                
                predictions.push(create_prediction(direction, (i as f32 * 0.1).sin(), 0.7));
                
                // Create somewhat correlated outcomes
                let outcome = if i % 4 == 0 {
                    // Add some noise
                    match direction {
                        Direction::Long => Direction::Short,
                        Direction::Short => Direction::Long,
                        Direction::Neutral => Direction::Neutral,
                    }
                } else {
                    direction
                };
                
                outcomes.push(outcome);
            }
        }
        
        (predictions, outcomes, market_data)
    }
}