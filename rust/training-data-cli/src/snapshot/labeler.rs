// Future returns labeler implementation

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Label {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelThresholds {
    pub buy_threshold: f64,   // e.g., 0.02 (2%)
    pub sell_threshold: f64,  // e.g., -0.02 (-2%)
}

impl Default for LabelThresholds {
    fn default() -> Self {
        Self {
            buy_threshold: 0.02,
            sell_threshold: -0.02,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LabelDistribution {
    pub buy_count: usize,
    pub sell_count: usize,
    pub hold_count: usize,
    pub total_count: usize,
}

impl LabelDistribution {
    pub fn buy_percentage(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.buy_count as f64 / self.total_count as f64) * 100.0
        }
    }

    pub fn sell_percentage(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.sell_count as f64 / self.total_count as f64) * 100.0
        }
    }

    pub fn hold_percentage(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.hold_count as f64 / self.total_count as f64) * 100.0
        }
    }

    /// Check if the distribution is reasonably balanced
    /// Returns true if no single class dominates more than 70%
    pub fn is_balanced(&self) -> bool {
        if self.total_count == 0 {
            return false;
        }
        
        let max_percentage = [
            self.buy_percentage(),
            self.sell_percentage(),
            self.hold_percentage(),
        ]
        .iter()
        .fold(0.0f64, |a, &b| a.max(b));
        
        max_percentage <= 70.0
    }
}

#[derive(Debug)]
pub struct FutureReturnsLabeler {
    horizon: usize,
    thresholds: LabelThresholds,
}

impl FutureReturnsLabeler {
    /// Create a new FutureReturnsLabeler with specified horizon and thresholds
    pub fn new(horizon: usize, thresholds: LabelThresholds) -> Result<Self> {
        if horizon == 0 {
            return Err(anyhow!("Horizon must be greater than 0"));
        }
        
        if thresholds.buy_threshold <= thresholds.sell_threshold {
            return Err(anyhow!(
                "Buy threshold ({}) must be greater than sell threshold ({})",
                thresholds.buy_threshold,
                thresholds.sell_threshold
            ));
        }

        Ok(Self { horizon, thresholds })
    }

    /// Create a new FutureReturnsLabeler with default thresholds
    pub fn with_horizon(horizon: usize) -> Result<Self> {
        Self::new(horizon, LabelThresholds::default())
    }

    /// Calculate future returns using the formula: (close[t+h] - close[t]) / close[t]
    /// Returns None for positions where future data is not available
    pub fn calculate_returns(&self, prices: &[f64]) -> Vec<Option<f64>> {
        if prices.is_empty() {
            return Vec::new();
        }

        let mut returns = Vec::with_capacity(prices.len());
        
        for i in 0..prices.len() {
            if i + self.horizon < prices.len() {
                let current_price = prices[i];
                let future_price = prices[i + self.horizon];
                
                if current_price == 0.0 {
                    returns.push(None); // Avoid division by zero
                } else {
                    let return_value = (future_price - current_price) / current_price;
                    returns.push(Some(return_value));
                }
            } else {
                returns.push(None); // Not enough future data
            }
        }
        
        returns
    }

    /// Classify returns into Buy/Sell/Hold labels based on thresholds
    pub fn classify_returns(&self, returns: &[f64]) -> Vec<Label> {
        returns
            .iter()
            .map(|&return_value| {
                if return_value > self.thresholds.buy_threshold {
                    Label::Buy
                } else if return_value < self.thresholds.sell_threshold {
                    Label::Sell
                } else {
                    Label::Hold
                }
            })
            .collect()
    }

    /// Generate labels from price data, combining return calculation and classification
    pub fn generate_labels(&self, prices: &[f64]) -> Result<Vec<Option<Label>>> {
        if prices.is_empty() {
            return Ok(Vec::new());
        }

        let returns = self.calculate_returns(prices);
        let mut labels = Vec::with_capacity(returns.len());
        
        for return_opt in returns {
            match return_opt {
                Some(return_value) => {
                    let label = if return_value > self.thresholds.buy_threshold {
                        Label::Buy
                    } else if return_value < self.thresholds.sell_threshold {
                        Label::Sell
                    } else {
                        Label::Hold
                    };
                    labels.push(Some(label));
                }
                None => labels.push(None),
            }
        }
        
        Ok(labels)
    }

    /// Calculate label distribution from generated labels
    pub fn calculate_distribution(&self, labels: &[Option<Label>]) -> LabelDistribution {
        let mut buy_count = 0;
        let mut sell_count = 0;
        let mut hold_count = 0;
        let mut total_count = 0;

        for label_opt in labels {
            if let Some(label) = label_opt {
                total_count += 1;
                match label {
                    Label::Buy => buy_count += 1,
                    Label::Sell => sell_count += 1,
                    Label::Hold => hold_count += 1,
                }
            }
        }

        LabelDistribution {
            buy_count,
            sell_count,
            hold_count,
            total_count,
        }
    }

    /// Validate that the label distribution is reasonable (not heavily skewed)
    pub fn validate_distribution(&self, labels: &[Option<Label>]) -> Result<LabelDistribution> {
        let distribution = self.calculate_distribution(labels);
        
        if distribution.total_count == 0 {
            return Err(anyhow!("No valid labels generated - insufficient future data"));
        }

        if !distribution.is_balanced() {
            return Err(anyhow!(
                "Label distribution is heavily skewed: Buy: {:.1}%, Sell: {:.1}%, Hold: {:.1}%. Consider adjusting thresholds.",
                distribution.buy_percentage(),
                distribution.sell_percentage(),
                distribution.hold_percentage()
            ));
        }

        Ok(distribution)
    }

    /// Get the horizon value
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// Get the thresholds
    pub fn thresholds(&self) -> &LabelThresholds {
        &self.thresholds
    }
}
#[cfg
(test)]
mod tests {
    use super::*;

    fn create_test_labeler() -> FutureReturnsLabeler {
        FutureReturnsLabeler::new(
            3,
            LabelThresholds {
                buy_threshold: 0.02,
                sell_threshold: -0.02,
            },
        )
        .unwrap()
    }

    #[test]
    fn test_new_labeler_valid_params() {
        let labeler = FutureReturnsLabeler::new(
            5,
            LabelThresholds {
                buy_threshold: 0.05,
                sell_threshold: -0.03,
            },
        );
        assert!(labeler.is_ok());
        
        let labeler = labeler.unwrap();
        assert_eq!(labeler.horizon(), 5);
        assert_eq!(labeler.thresholds().buy_threshold, 0.05);
        assert_eq!(labeler.thresholds().sell_threshold, -0.03);
    }

    #[test]
    fn test_new_labeler_invalid_horizon() {
        let result = FutureReturnsLabeler::new(
            0,
            LabelThresholds::default(),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Horizon must be greater than 0"));
    }

    #[test]
    fn test_new_labeler_invalid_thresholds() {
        let result = FutureReturnsLabeler::new(
            5,
            LabelThresholds {
                buy_threshold: -0.02,  // This should be greater than sell_threshold
                sell_threshold: -0.01, // This should be less than buy_threshold
            },
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Buy threshold"));
    }

    #[test]
    fn test_with_horizon() {
        let labeler = FutureReturnsLabeler::with_horizon(10).unwrap();
        assert_eq!(labeler.horizon(), 10);
        assert_eq!(labeler.thresholds().buy_threshold, 0.02);
        assert_eq!(labeler.thresholds().sell_threshold, -0.02);
    }

    #[test]
    fn test_calculate_returns_basic() {
        let labeler = create_test_labeler();
        let prices = vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0];
        let returns = labeler.calculate_returns(&prices);
        
        // Expected returns with horizon 3:
        // [0]: (105.0 - 100.0) / 100.0 = 0.05
        // [1]: (103.0 - 102.0) / 102.0 ≈ 0.0098
        // [2]: (108.0 - 101.0) / 101.0 ≈ 0.0693
        // [3]: None (not enough future data)
        // [4]: None
        // [5]: None
        
        assert_eq!(returns.len(), 6);
        assert!(returns[0].is_some());
        assert!((returns[0].unwrap() - 0.05).abs() < 1e-10);
        
        assert!(returns[1].is_some());
        assert!((returns[1].unwrap() - (1.0 / 102.0)).abs() < 1e-10);
        
        assert!(returns[2].is_some());
        assert!((returns[2].unwrap() - (7.0 / 101.0)).abs() < 1e-10);
        
        assert!(returns[3].is_none());
        assert!(returns[4].is_none());
        assert!(returns[5].is_none());
    }

    #[test]
    fn test_calculate_returns_empty_input() {
        let labeler = create_test_labeler();
        let returns = labeler.calculate_returns(&[]);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_calculate_returns_insufficient_data() {
        let labeler = create_test_labeler();
        let prices = vec![100.0, 102.0]; // Less than horizon
        let returns = labeler.calculate_returns(&prices);
        
        assert_eq!(returns.len(), 2);
        assert!(returns[0].is_none());
        assert!(returns[1].is_none());
    }

    #[test]
    fn test_calculate_returns_zero_price() {
        let labeler = create_test_labeler();
        let prices = vec![0.0, 102.0, 101.0, 105.0];
        let returns = labeler.calculate_returns(&prices);
        
        assert_eq!(returns.len(), 4);
        assert!(returns[0].is_none()); // Division by zero avoided
        assert!(returns[1].is_none()); // Not enough future data
    }

    #[test]
    fn test_classify_returns() {
        let labeler = create_test_labeler();
        let returns = vec![0.03, -0.03, 0.01, -0.01, 0.025, -0.025];
        let labels = labeler.classify_returns(&returns);
        
        assert_eq!(labels.len(), 6);
        assert_eq!(labels[0], Label::Buy);    // 0.03 > 0.02
        assert_eq!(labels[1], Label::Sell);   // -0.03 < -0.02
        assert_eq!(labels[2], Label::Hold);   // -0.02 <= 0.01 <= 0.02
        assert_eq!(labels[3], Label::Hold);   // -0.02 <= -0.01 <= 0.02
        assert_eq!(labels[4], Label::Buy);    // 0.025 > 0.02
        assert_eq!(labels[5], Label::Sell);   // -0.025 < -0.02
    }

    #[test]
    fn test_classify_returns_edge_cases() {
        let labeler = create_test_labeler();
        let returns = vec![0.02, -0.02]; // Exactly at thresholds
        let labels = labeler.classify_returns(&returns);
        
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], Label::Hold);   // 0.02 == buy_threshold, not >
        assert_eq!(labels[1], Label::Hold);   // -0.02 == sell_threshold, not <
    }

    #[test]
    fn test_generate_labels() {
        let labeler = create_test_labeler();
        let prices = vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0];
        let labels = labeler.generate_labels(&prices).unwrap();
        
        assert_eq!(labels.len(), 6);
        
        // [0]: return = 0.05 > 0.02 -> Buy
        assert_eq!(labels[0], Some(Label::Buy));
        
        // [1]: return ≈ 0.0098 -> Hold (between thresholds)
        assert_eq!(labels[1], Some(Label::Hold));
        
        // [2]: return ≈ 0.0693 > 0.02 -> Buy
        assert_eq!(labels[2], Some(Label::Buy));
        
        // [3-5]: None (insufficient future data)
        assert_eq!(labels[3], None);
        assert_eq!(labels[4], None);
        assert_eq!(labels[5], None);
    }

    #[test]
    fn test_generate_labels_empty_input() {
        let labeler = create_test_labeler();
        let labels = labeler.generate_labels(&[]).unwrap();
        assert!(labels.is_empty());
    }

    #[test]
    fn test_calculate_distribution() {
        let labeler = create_test_labeler();
        let labels = vec![
            Some(Label::Buy),
            Some(Label::Buy),
            Some(Label::Sell),
            Some(Label::Hold),
            Some(Label::Hold),
            Some(Label::Hold),
            None,
            None,
        ];
        
        let distribution = labeler.calculate_distribution(&labels);
        
        assert_eq!(distribution.buy_count, 2);
        assert_eq!(distribution.sell_count, 1);
        assert_eq!(distribution.hold_count, 3);
        assert_eq!(distribution.total_count, 6);
        
        assert!((distribution.buy_percentage() - 33.33).abs() < 0.1);
        assert!((distribution.sell_percentage() - 16.67).abs() < 0.1);
        assert!((distribution.hold_percentage() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_distribution_is_balanced() {
        // Balanced distribution
        let balanced = LabelDistribution {
            buy_count: 30,
            sell_count: 35,
            hold_count: 35,
            total_count: 100,
        };
        assert!(balanced.is_balanced());
        
        // Unbalanced distribution (one class > 70%)
        let unbalanced = LabelDistribution {
            buy_count: 80,
            sell_count: 10,
            hold_count: 10,
            total_count: 100,
        };
        assert!(!unbalanced.is_balanced());
        
        // Edge case: exactly 70%
        let edge_case = LabelDistribution {
            buy_count: 70,
            sell_count: 15,
            hold_count: 15,
            total_count: 100,
        };
        assert!(edge_case.is_balanced());
        
        // Empty distribution
        let empty = LabelDistribution {
            buy_count: 0,
            sell_count: 0,
            hold_count: 0,
            total_count: 0,
        };
        assert!(!empty.is_balanced());
    }

    #[test]
    fn test_validate_distribution_success() {
        let labeler = create_test_labeler();
        let labels = vec![
            Some(Label::Buy),
            Some(Label::Buy),
            Some(Label::Sell),
            Some(Label::Sell),
            Some(Label::Hold),
            Some(Label::Hold),
        ];
        
        let result = labeler.validate_distribution(&labels);
        assert!(result.is_ok());
        
        let distribution = result.unwrap();
        assert_eq!(distribution.total_count, 6);
        assert!(distribution.is_balanced());
    }

    #[test]
    fn test_validate_distribution_no_labels() {
        let labeler = create_test_labeler();
        let labels = vec![None, None, None];
        
        let result = labeler.validate_distribution(&labels);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No valid labels generated"));
    }

    #[test]
    fn test_validate_distribution_skewed() {
        let labeler = create_test_labeler();
        let mut labels = vec![Some(Label::Buy); 80]; // 80% buy labels
        labels.extend(vec![Some(Label::Sell); 10]);  // 10% sell labels
        labels.extend(vec![Some(Label::Hold); 10]);  // 10% hold labels
        
        let result = labeler.validate_distribution(&labels);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("heavily skewed"));
    }

    #[test]
    fn test_integration_realistic_scenario() {
        // Test with realistic price data that might produce a trending market
        let labeler = FutureReturnsLabeler::new(
            5, // 5-period horizon
            LabelThresholds {
                buy_threshold: 0.01,  // 1% threshold
                sell_threshold: -0.01, // -1% threshold
            },
        ).unwrap();
        
        // Simulate trending upward price data
        let prices: Vec<f64> = (0..20)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.1).sin()) // Slight upward trend with noise
            .collect();
        
        let labels = labeler.generate_labels(&prices).unwrap();
        
        // Should have labels for first 15 prices (20 - 5 horizon)
        let valid_labels: Vec<_> = labels.iter().filter_map(|l| l.as_ref()).collect();
        assert_eq!(valid_labels.len(), 15);
        
        // In an upward trending market, we should see some buy signals
        let buy_count = valid_labels.iter().filter(|&l| **l == Label::Buy).count();
        assert!(buy_count > 0, "Expected some buy signals in upward trending market");
    }

    #[test]
    fn test_different_horizons() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        
        // Test horizon 1
        let labeler1 = FutureReturnsLabeler::with_horizon(1).unwrap();
        let labels1 = labeler1.generate_labels(&prices).unwrap();
        assert_eq!(labels1.iter().filter(|l| l.is_some()).count(), 7);
        
        // Test horizon 3
        let labeler3 = FutureReturnsLabeler::with_horizon(3).unwrap();
        let labels3 = labeler3.generate_labels(&prices).unwrap();
        assert_eq!(labels3.iter().filter(|l| l.is_some()).count(), 5);
        
        // Test horizon 5
        let labeler5 = FutureReturnsLabeler::with_horizon(5).unwrap();
        let labels5 = labeler5.generate_labels(&prices).unwrap();
        assert_eq!(labels5.iter().filter(|l| l.is_some()).count(), 3);
    }

    #[test]
    fn test_extreme_price_movements() {
        let labeler = create_test_labeler();
        
        // Test with extreme price movements
        let prices = vec![100.0, 100.0, 100.0, 150.0, 50.0, 100.0]; // Large jumps
        let labels = labeler.generate_labels(&prices).unwrap();
        
        // [0]: (150.0 - 100.0) / 100.0 = 0.5 > 0.02 -> Buy
        assert_eq!(labels[0], Some(Label::Buy));
        
        // [1]: (50.0 - 100.0) / 100.0 = -0.5 < -0.02 -> Sell
        assert_eq!(labels[1], Some(Label::Sell));
        
        // [2]: (100.0 - 100.0) / 100.0 = 0.0 -> Hold
        assert_eq!(labels[2], Some(Label::Hold));
    }
}