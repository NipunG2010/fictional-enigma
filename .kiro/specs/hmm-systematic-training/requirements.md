# Requirements Document

## Introduction

This spec completes Phase 3 Task 2: HMM Model Development by implementing systematic training and evaluation of HMM models with 2-4 states using the specific signal observations [s_LDC, s_MR, s_TSMOM]. The existing HMM infrastructure provides the foundation, but we need to add systematic training workflows, comprehensive evaluation, and production-ready model selection.

## Current State

**✅ Already Implemented:**
- HMM training framework with hmmlearn (py/imp/hmm/trainer.py)
- RegimeAnalyzer for state analysis (py/imp/hmm/regime_analysis.py)
- Basic training comparison notebook (notebooks/02_hmm_training_comparison.ipynb)
- Model artifacts and storage (py/imp/hmm/models.py)

**🎯 Gaps to Address:**
- No systematic training script for 2-4 states with [s_LDC, s_MR, s_TSMOM]
- Limited model comparison and selection framework
- Need comprehensive evaluation with held-out likelihood
- Missing production-ready model selection pipeline

## Requirements

### Requirement 1

**User Story:** As a quantitative researcher, I want a systematic training script that trains HMM models with 2-4 states on [s_LDC, s_MR, s_TSMOM] observations, so that I can efficiently evaluate different model configurations.

#### Acceptance Criteria

1. WHEN running the training script THEN the system SHALL train models with 2, 3, and 4 states automatically
2. WHEN loading data THEN the system SHALL use exactly [s_LDC, s_MR, s_TSMOM] as multivariate observations
3. WHEN training completes THEN the system SHALL save all model artifacts with proper metadata
4. IF training fails for any configuration THEN the system SHALL log errors and continue with remaining configurations
5. WHEN all training completes THEN the system SHALL generate a comparison report with AIC/BIC scores

### Requirement 2

**User Story:** As a machine learning engineer, I want comprehensive model evaluation using cross-validation and held-out likelihood, so that I can objectively select the best performing HMM configuration.

#### Acceptance Criteria

1. WHEN evaluating models THEN the system SHALL use time-series cross-validation with proper train/test splits
2. WHEN calculating metrics THEN the system SHALL compute AIC, BIC, and held-out log-likelihood for each model
3. WHEN comparing models THEN the system SHALL rank configurations by multiple criteria
4. IF evaluation fails THEN the system SHALL provide diagnostic information
5. WHEN evaluation completes THEN the system SHALL output a ranked model comparison table

### Requirement 3

**User Story:** As a portfolio manager, I want enhanced state interpretability analysis with economic meaning, so that I can understand what market regimes the HMM has detected and their trading implications.

#### Acceptance Criteria

1. WHEN analyzing states THEN the system SHALL characterize each regime by signal statistics
2. WHEN examining regimes THEN the system SHALL identify economic interpretation (trending, mean-reverting, etc.)
3. WHEN generating reports THEN the system SHALL provide actionable trading insights for each regime
4. IF states lack clear interpretation THEN the system SHALL flag this in the report
5. WHEN analysis completes THEN the system SHALL generate a comprehensive regime report

### Requirement 4

**User Story:** As a system integrator, I want automated model selection that chooses the best HMM configuration, so that I can deploy the optimal model to production with confidence.

#### Acceptance Criteria

1. WHEN selecting models THEN the system SHALL use weighted scoring combining statistical and interpretability metrics
2. WHEN generating artifacts THEN the system SHALL create production-ready HMMArtifact with all metadata
3. WHEN validation runs THEN the system SHALL ensure artifact compatibility with existing inference code
4. IF no model meets quality thresholds THEN the system SHALL provide recommendations for improvement
5. WHEN selection completes THEN the system SHALL output the best model with confidence scores
