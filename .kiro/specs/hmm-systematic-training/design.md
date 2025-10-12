# HMM Systematic Training Design Document

## Overview

This design completes Phase 3 Task 2 by adding systematic training, evaluation, and model selection capabilities to the existing HMM infrastructure. We'll build on the existing trainer and regime analyzer to create a production-ready pipeline for training HMM models with 2-4 states on [s_LDC, s_MR, s_TSMOM] observations.

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────┐
│              Existing HMM Infrastructure                    │
├─────────────────────────────────────────────────────────────┤
│  • HMMLearnTrainer (py/imp/hmm/trainer.py)                 │
│  • RegimeAnalyzer (py/imp/hmm/regime_analysis.py)          │
│  • HMMArtifact models (py/imp/hmm/models.py)               │
│  • Training notebooks (notebooks/02_*.ipynb)                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           New Systematic Training Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│  1. Data Loading & Validation                               │
│     └─ Load [s_LDC, s_MR, s_TSMOM] from Parquet            │
│                                                             │
│  2. Systematic Training                                     │
│     └─ Train 2, 3, 4 state models with CV                  │
│                                                             │
│  3. Comprehensive Evaluation                                │
│     └─ AIC/BIC, held-out likelihood, interpretability      │
│                                                             │
│  4. Model Selection                                         │
│     └─ Weighted scoring and artifact generation            │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Systematic Training Script

**File**: `py/scripts/train_hmm_systematic.py`

This is the main entry point that orchestrates the entire training pipeline.

```python
"""
Systematic HMM training script for Phase 3 Task 2.

Trains HMM models with 2-4 states on [s_LDC, s_MR, s_TSMOM] observations,
evaluates model quality, and selects the best configuration.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Import existing infrastructure
from imp.hmm.trainer import EnhancedHMMTrainer
from imp.hmm.regime_analysis import RegimeAnalyzer
from imp.hmm.models import HMMArtifact

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystematicHMMTrainer:
    """Orchestrates systematic HMM training and evaluation."""
    
    def __init__(self, 
                 data_path: Path,
                 output_dir: Path,
                 n_states_range: List[int] = [2, 3, 4],
                 cv_folds: int = 5):
        """
        Initialize systematic trainer.
        
        Args:
            data_path: Path to Parquet file with [s_LDC, s_MR, s_TSMOM]
            output_dir: Directory for saving artifacts and reports
            n_states_range: List of state counts to try
            cv_folds: Number of cross-validation folds
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.n_states_range = n_states_range
        self.cv_folds = cv_folds
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.training_results = {}
        self.evaluation_results = {}
        
    def load_and_validate_data(self) -> np.ndarray:
        """
        Load and validate [s_LDC, s_MR, s_TSMOM] observations.
        
        Returns:
            Numpy array of shape (n_samples, 3)
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Load Parquet file
        df = pd.read_parquet(self.data_path)
        
        # Validate required columns
        required_cols = ['s_LDC', 's_MR', 's_TSMOM']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Extract observations
        observations = df[required_cols].values
        
        # Validate data quality
        if np.any(np.isnan(observations)):
            nan_pct = np.isnan(observations).mean() * 100
            logger.warning(f"Data contains {nan_pct:.2f}% NaN values")
            # Remove NaN rows
            observations = observations[~np.isnan(observations).any(axis=1)]
        
        logger.info(f"Loaded {len(observations)} observations with 3 features")
        return observations
    
    def train_all_configurations(self, observations: np.ndarray) -> Dict[str, Any]:
        """
        Train HMM models for all state configurations.
        
        Args:
            observations: Array of shape (n_samples, 3)
            
        Returns:
            Dictionary of training results
        """
        logger.info("Starting systematic training...")
        
        for n_states in self.n_states_range:
            config_name = f"{n_states}_states"
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {config_name}")
            logger.info(f"{'='*60}")
            
            try:
                # Create trainer
                trainer = EnhancedHMMTrainer(
                    n_states=n_states,
                    library="hmmlearn",
                    covariance_type="full",
                    random_state=42
                )
                
                # Train with cross-validation
                cv_results = trainer.cross_validate(
                    observations,
                    cv_folds=self.cv_folds,
                    n_iterations=100
                )
                
                # Train final model on full data
                artifact = trainer.train(observations, n_iterations=100)
                
                # Store results
                self.training_results[config_name] = {
                    'artifact': artifact,
                    'cv_results': cv_results,
                    'n_states': n_states
                }
                
                # Save artifact
                artifact_path = self.output_dir / f"hmm_{config_name}.json"
                with open(artifact_path, 'w') as f:
                    json.dump(artifact.model_dump(), f, indent=2)
                
                logger.info(f"✓ {config_name} training complete")
                logger.info(f"  AIC: {artifact.metadata['aic']:.2f}")
                logger.info(f"  BIC: {artifact.metadata['bic']:.2f}")
                logger.info(f"  CV Log-Likelihood: {cv_results.get('log_likelihood_mean', 'N/A')}")
                
            except Exception as e:
                logger.error(f"✗ {config_name} training failed: {e}")
                self.training_results[config_name] = {
                    'error': str(e),
                    'n_states': n_states
                }
        
        return self.training_results
    
    def evaluate_all_models(self, observations: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate all trained models comprehensively.
        
        Args:
            observations: Array of shape (n_samples, 3)
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("\n" + "="*60)
        logger.info("Evaluating Models")
        logger.info("="*60)
        
        evaluation_summary = {
            'models': {},
            'rankings': []
        }
        
        for config_name, result in self.training_results.items():
            if 'error' in result:
                logger.warning(f"Skipping {config_name} (training failed)")
                continue
            
            logger.info(f"\nEvaluating {config_name}...")
            
            artifact = result['artifact']
            cv_results = result['cv_results']
            
            # Basic metrics from training
            basic_metrics = {
                'aic': artifact.metadata['aic'],
                'bic': artifact.metadata['bic'],
                'log_likelihood': artifact.metadata['convergence_log_likelihood'],
                'cv_log_likelihood_mean': cv_results.get('log_likelihood_mean', None),
                'cv_log_likelihood_std': cv_results.get('log_likelihood_std', None)
            }
            
            # Regime analysis
            try:
                analyzer = RegimeAnalyzer(artifact)
                
                # Decode state sequence
                from hmmlearn import hmm
                model = hmm.GaussianHMM(
                    n_components=artifact.n_states,
                    covariance_type="full"
                )
                model.transmat_ = np.array(artifact.transition_matrix)
                model.startprob_ = np.array(artifact.initial_probabilities)
                model.means_ = np.array(artifact.means)
                model.covars_ = np.array(artifact.covariances)
                
                state_sequence = model.predict(observations)
                
                # Characterize regimes
                characteristics = analyzer.characterize_regimes(observations, state_sequence)
                persistence = analyzer.analyze_state_persistence(state_sequence)
                interpretations = analyzer.interpret_regimes(characteristics, persistence)
                
                # Calculate interpretability score
                interpretability_score = self._calculate_interpretability_score(
                    characteristics, persistence
                )
                
                evaluation_summary['models'][config_name] = {
                    'basic_metrics': basic_metrics,
                    'interpretability_score': interpretability_score,
                    'n_states': artifact.n_states,
                    'characteristics': {k: v.to_dict() for k, v in characteristics.items()},
                    'persistence': {k: v.to_dict() for k, v in persistence.items()},
                    'interpretations': {k: v.to_dict() for k, v in interpretations.items()}
                }
                
                logger.info(f"  Interpretability Score: {interpretability_score:.3f}")
                
            except Exception as e:
                logger.error(f"  Regime analysis failed: {e}")
                evaluation_summary['models'][config_name] = {
                    'basic_metrics': basic_metrics,
                    'interpretability_score': 0.0,
                    'error': str(e)
                }
        
        # Rank models
        evaluation_summary['rankings'] = self._rank_models(evaluation_summary['models'])
        
        return evaluation_summary
    
    def select_best_model(self, evaluation_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best model based on weighted scoring.
        
        Args:
            evaluation_summary: Results from evaluate_all_models()
            
        Returns:
            Best model information
        """
        logger.info("\n" + "="*60)
        logger.info("Model Selection")
        logger.info("="*60)
        
        if not evaluation_summary['rankings']:
            raise ValueError("No models available for selection")
        
        best_model = evaluation_summary['rankings'][0]
        
        logger.info(f"\n🏆 Best Model: {best_model['config_name']}")
        logger.info(f"   Combined Score: {best_model['combined_score']:.3f}")
        logger.info(f"   AIC: {best_model['aic']:.2f}")
        logger.info(f"   BIC: {best_model['bic']:.2f}")
        logger.info(f"   Interpretability: {best_model['interpretability']:.3f}")
        
        # Get the artifact
        best_config = best_model['config_name']
        best_artifact = self.training_results[best_config]['artifact']
        
        # Save best model
        best_model_path = self.output_dir / "hmm_best.json"
        with open(best_model_path, 'w') as f:
            json.dump(best_artifact.model_dump(), f, indent=2)
        
        logger.info(f"\n✓ Best model saved to {best_model_path}")
        
        return {
            'config_name': best_config,
            'artifact': best_artifact,
            'scores': best_model
        }
    
    def generate_report(self, evaluation_summary: Dict[str, Any]) -> None:
        """
        Generate comprehensive training report.
        
        Args:
            evaluation_summary: Results from evaluate_all_models()
        """
        logger.info("\n" + "="*60)
        logger.info("Generating Report")
        logger.info("="*60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'n_states_range': self.n_states_range,
            'cv_folds': self.cv_folds,
            'evaluation_summary': evaluation_summary
        }
        
        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✓ Report saved to {report_path}")
        
        # Print summary table
        self._print_summary_table(evaluation_summary['rankings'])
    
    def _calculate_interpretability_score(self, 
                                         characteristics: Dict,
                                         persistence: Dict) -> float:
        """Calculate overall interpretability score."""
        if not characteristics:
            return 0.0
        
        scores = []
        for state_id in characteristics.keys():
            char = characteristics[state_id]
            pers = persistence.get(state_id)
            
            # State distinctiveness (higher volatility difference = more distinct)
            volatility_score = char.volatility
            
            # Persistence (longer duration = more interpretable)
            persistence_score = 0.0
            if pers:
                # Normalize by max reasonable duration (e.g., 20)
                persistence_score = min(1.0, pers.mean_duration / 20.0)
            
            # Combined score
            state_score = (volatility_score * 0.5 + persistence_score * 0.5)
            scores.append(state_score)
        
        return float(np.mean(scores))
    
    def _rank_models(self, models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank models by combined criteria."""
        rankings = []
        
        for config_name, model_data in models.items():
            if 'error' in model_data:
                continue
            
            metrics = model_data['basic_metrics']
            interpretability = model_data['interpretability_score']
            
            # Normalize metrics (lower AIC/BIC is better)
            # Use negative values so higher is better
            aic_score = -metrics['aic'] / 1000.0
            bic_score = -metrics['bic'] / 1000.0
            
            # CV score (higher is better)
            cv_score = metrics.get('cv_log_likelihood_mean', 0) / 100.0
            
            # Combined score with weights
            combined_score = (
                aic_score * 0.3 +
                bic_score * 0.3 +
                cv_score * 0.2 +
                interpretability * 0.2
            )
            
            rankings.append({
                'config_name': config_name,
                'combined_score': combined_score,
                'aic': metrics['aic'],
                'bic': metrics['bic'],
                'cv_score': metrics.get('cv_log_likelihood_mean', 0),
                'interpretability': interpretability
            })
        
        # Sort by combined score (descending)
        rankings.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return rankings
    
    def _print_summary_table(self, rankings: List[Dict[str, Any]]) -> None:
        """Print summary table of model rankings."""
        logger.info("\n" + "="*60)
        logger.info("Model Comparison Summary")
        logger.info("="*60)
        
        # Header
        logger.info(f"{'Rank':<6} {'Model':<12} {'Score':<10} {'AIC':<12} {'BIC':<12} {'Interp':<10}")
        logger.info("-" * 60)
        
        # Rows
        for i, model in enumerate(rankings, 1):
            logger.info(
                f"{i:<6} "
                f"{model['config_name']:<12} "
                f"{model['combined_score']:<10.3f} "
                f"{model['aic']:<12.2f} "
                f"{model['bic']:<12.2f} "
                f"{model['interpretability']:<10.3f}"
            )
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete systematic training pipeline.
        
        Returns:
            Dictionary with best model and evaluation results
        """
        logger.info("="*60)
        logger.info("HMM Systematic Training Pipeline")
        logger.info("="*60)
        
        # 1. Load data
        observations = self.load_and_validate_data()
        
        # 2. Train all configurations
        self.train_all_configurations(observations)
        
        # 3. Evaluate all models
        evaluation_summary = self.evaluate_all_models(observations)
        
        # 4. Select best model
        best_model = self.select_best_model(evaluation_summary)
        
        # 5. Generate report
        self.generate_report(evaluation_summary)
        
        logger.info("\n" + "="*60)
        logger.info("✓ Systematic Training Complete")
        logger.info("="*60)
        
        return {
            'best_model': best_model,
            'evaluation_summary': evaluation_summary
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Systematic HMM training for Phase 3 Task 2"
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        required=True,
        help='Path to Parquet file with [s_LDC, s_MR, s_TSMOM]'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output/hmm_training'),
        help='Output directory for artifacts and reports'
    )
    parser.add_argument(
        '--n-states',
        type=int,
        nargs='+',
        default=[2, 3, 4],
        help='List of state counts to try'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SystematicHMMTrainer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_states_range=args.n_states,
        cv_folds=args.cv_folds
    )
    
    # Run pipeline
    results = trainer.run()
    
    logger.info(f"\n✓ Best model: {results['best_model']['config_name']}")
    logger.info(f"✓ Artifacts saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
```

### 2. Enhanced Notebook

**File**: `notebooks/07_systematic_hmm_training.ipynb`

A Jupyter notebook that provides an interactive interface to the systematic training pipeline with visualizations.

Key sections:
1. **Data Loading**: Load and visualize [s_LDC, s_MR, s_TSMOM] signals
2. **Training**: Run systematic training with progress tracking
3. **Evaluation**: Interactive model comparison with charts
4. **Regime Analysis**: Detailed regime characteristics and interpretations
5. **Model Selection**: Visual comparison and selection interface

## Data Flow

```
Input: Parquet file with [s_LDC, s_MR, s_TSMOM]
   │
   ├─> Load & Validate
   │      └─> Check for NaN, validate shape
   │
   ├─> Systematic Training (2, 3, 4 states)
   │      ├─> Train with cross-validation
   │      ├─> Calculate AIC/BIC
   │      └─> Save artifacts
   │
   ├─> Comprehensive Evaluation
   │      ├─> Regime characterization
   │      ├─> Persistence analysis
   │      ├─> Economic interpretation
   │      └─> Interpretability scoring
   │
   ├─> Model Ranking
   │      └─> Weighted scoring (AIC, BIC, CV, interpretability)
   │
   └─> Output
          ├─> Best model artifact (hmm_best.json)
          ├─> All model artifacts (hmm_2_states.json, etc.)
          └─> Training report (training_report.json)
```

## Implementation Considerations

### 1. Leveraging Existing Infrastructure

We build directly on existing components:
- **EnhancedHMMTrainer**: Already has cross-validation support
- **RegimeAnalyzer**: Already provides comprehensive regime analysis
- **HMMArtifact**: Already has proper validation and serialization

This minimizes new code and ensures consistency.

### 2. Model Selection Criteria

Weighted scoring combines:
- **Statistical fit** (30%): AIC score
- **Model complexity** (30%): BIC score  
- **Generalization** (20%): Cross-validation log-likelihood
- **Interpretability** (20%): Regime distinctiveness and persistence

### 3. Error Handling

- Training failures for individual configurations don't stop the pipeline
- All errors are logged with diagnostic information
- Reports include both successful and failed configurations

### 4. Output Structure

```
output/hmm_training/
├── hmm_2_states.json          # Individual model artifacts
├── hmm_3_states.json
├── hmm_4_states.json
├── hmm_best.json              # Best model (copy)
└── training_report.json       # Comprehensive report
```

## Testing Strategy

### Unit Tests

Test individual components:
- Data loading and validation
- Interpretability score calculation
- Model ranking logic

### Integration Tests

Test the full pipeline:
- Run with synthetic data
- Verify all artifacts are created
- Check report format

### Validation Tests

Verify model quality:
- Check AIC/BIC values are reasonable
- Verify state probabilities sum to 1
- Ensure artifacts are valid HMMArtifact objects

## Success Criteria

1. **Completeness**: All 2-4 state models train successfully
2. **Evaluation**: Comprehensive metrics calculated for all models
3. **Selection**: Best model selected with clear justification
4. **Artifacts**: All artifacts saved in correct format
5. **Report**: Comprehensive report generated with rankings
