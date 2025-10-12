# Implementation Plan

- [x] 1. Create StateWeightOptimizer class with Sharpe ratio optimization
  - Implement OptimizationConfig dataclass with method selection and constraints
  - Create StateWeightOptimizer with scipy SLSQP optimization method
  - Add grid search optimization as alternative method
  - Implement Sharpe ratio calculation with proper annualization
  - Add portfolio returns computation from weighted signals
  - Include fallback to equal weights for edge cases
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2_

- [X] 2. Implement enhanced compute_state_weights in trainer
  - Replace TODO placeholder with actual optimization logic
  - Add state sequence prediction and data filtering per state
  - Integrate StateWeightOptimizer for each HMM state
  - Compute per-state Sharpe ratios and aggregate metrics
  - Return FusionWeights with proper training_metrics populated
  - Add error handling and logging for optimization failures
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Build WeightValidator class for constraint and performance validation
  - Create constraint validation checking sum=1 and non-negative weights
  - Implement performance comparison between optimized and equal-weight baseline
  - Add statistical significance testing using paired t-test
  - Build comprehensive validation report generation
  - Include improvement metrics and percentage calculations
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4. Add walk-forward validation for robustness testing
  - Implement time-series cross-validation for weight optimization
  - Test optimized weights on out-of-sample data
  - Compare in-sample vs out-of-sample Sharpe ratios
  - Flag overfitting when out-of-sample performance degrades significantly
  - Generate robustness report with multiple validation folds
  - _Requirements: 3.3, 4.3, 4.4_

- [x] 5. Create comprehensive testing suite for weight optimization
  - Unit tests for StateWeightOptimizer with synthetic data
  - Test constraint enforcement (sum=1, non-negative, bounds)
  - Test both scipy and grid search optimization methods
  - Integration tests with real HMM models and signal data
  - Performance regression tests ensuring optimization improves Sharpe
  - Edge case tests (insufficient data, constant signals, optimization failures)
  - _Requirements: 1.4, 2.4, 3.4, 4.4_

- [x] 6. Create notebook demonstrating fusion weight optimization workflow
  - Load trained HMM model and signal data
  - Run weight optimization with both methods (scipy, grid search)
  - Visualize optimized weights per state with bar charts
  - Compare performance metrics (Sharpe, drawdown, win rate)
  - Show statistical significance of improvements
  - Demonstrate walk-forward validation results
  - _Requirements: 1.5, 2.5, 3.5, 4.5_
