# Requirements Document

## Introduction

Fusion Weight Optimization implements per-state weight optimization for combining the three signals [s_LDC, s_MR, s_TSMOM] based on detected market regimes. This task completes the HMM-based adaptive signal fusion by optimizing weights using risk-adjusted metrics (Sharpe ratio) and validating their effectiveness.

## Current Implementation Status

**✅ Already Implemented:**
- FusionWeights model with validation (py/imp/hmm/models.py)
- Placeholder compute_state_weights method in trainer (py/imp/hmm/trainer.py)
- Sharpe ratio calculation in Rust backtesting (rust/ldc-engine/src/backtesting.rs)
- Artifact export for FusionWeights (py/imp/hmm/artifact_management.py)

**🔄 Needs Implementation:**
- Actual per-state weight optimization using Sharpe ratio
- Weight validation framework with constraints
- Optimization methods (grid search, scipy.optimize)
- Performance comparison between optimized and equal weights

## Requirements

### Requirement 1

**User Story:** As a quantitative researcher, I want to optimize fusion weights per HMM state using Sharpe ratio, so that each regime uses the best signal combination for risk-adjusted returns.

#### Acceptance Criteria

1. WHEN optimizing weights THEN the system SHALL use historical returns data aligned with state sequences
2. WHEN calculating Sharpe ratio THEN the system SHALL properly annualize returns and handle risk-free rate
3. WHEN optimizing per state THEN the system SHALL compute separate optimal weights for each HMM state
4. IF optimization fails THEN the system SHALL fall back to equal weights and log warnings
5. WHEN optimization completes THEN the system SHALL return FusionWeights with training_metrics including achieved Sharpe ratios

### Requirement 2

**User Story:** As a risk manager, I want weight constraints and validation, so that fusion weights remain sensible and don't create extreme portfolio allocations.

#### Acceptance Criteria

1. WHEN optimizing weights THEN the system SHALL enforce weights sum to 1.0 constraint
2. WHEN validating weights THEN the system SHALL ensure all weights are non-negative (long-only constraint)
3. WHEN applying constraints THEN the system SHALL optionally support min/max weight bounds per signal
4. IF weights violate constraints THEN the system SHALL reject them and provide clear error messages
5. WHEN weights are valid THEN the system SHALL pass all FusionWeights model validations

### Requirement 3

**User Story:** As a machine learning engineer, I want multiple optimization methods, so that I can compare approaches and select the most robust weight optimization strategy.

#### Acceptance Criteria

1. WHEN selecting method THEN the system SHALL support grid search for exhaustive exploration
2. WHEN using scipy THEN the system SHALL support SLSQP optimizer for constrained optimization
3. WHEN comparing methods THEN the system SHALL evaluate both in-sample and out-of-sample performance
4. IF method fails THEN the system SHALL try alternative methods before falling back to defaults
5. WHEN optimization completes THEN the system SHALL report which method was used and convergence status

### Requirement 4

**User Story:** As a portfolio manager, I want performance validation comparing optimized vs baseline weights, so that I can verify the optimization actually improves risk-adjusted returns.

#### Acceptance Criteria

1. WHEN validating THEN the system SHALL compare optimized weights against equal-weight baseline
2. WHEN calculating metrics THEN the system SHALL compute Sharpe ratio, max drawdown, and win rate for both
3. WHEN testing robustness THEN the system SHALL use walk-forward validation or cross-validation
4. IF optimized weights underperform THEN the system SHALL flag this and recommend investigation
5. WHEN validation passes THEN the system SHALL generate performance comparison report with statistical significance
