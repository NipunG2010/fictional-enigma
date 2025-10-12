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
from imp.hmm.models import HMMArtifact
from imp.hmm.regime_analysis import RegimeAnalyzer

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
        
        # Validate shape
        if observations.shape[1] != 3:
            raise ValueError(f"Expected 3 features [s_LDC, s_MR, s_TSMOM], got {observations.shape[1]}")
        
        if len(observations) == 0:
            raise ValueError("No valid observations after removing NaN values")
        
        logger.info(f"Loaded {len(observations)} observations with 3 features")
        logger.info(f"Data shape: {observations.shape}")
        logger.info(f"Data range - s_LDC: [{observations[:, 0].min():.3f}, {observations[:, 0].max():.3f}]")
        logger.info(f"Data range - s_MR: [{observations[:, 1].min():.3f}, {observations[:, 1].max():.3f}]")
        logger.info(f"Data range - s_TSMOM: [{observations[:, 2].min():.3f}, {observations[:, 2].max():.3f}]")
        
        return observations
    
    def train_all_configurations(self, observations: np.ndarray) -> Dict[str, Any]:
        """
        Train HMM models for all state configurations.
        
        Args:
            observations: Array of shape (n_samples, 3)
            
        Returns:
            Dictionary of training results
        """
        logger.info("\n" + "="*60)
        logger.info("Starting Systematic Training")
        logger.info("="*60)
        logger.info(f"Training configurations: {self.n_states_range} states")
        logger.info(f"Cross-validation folds: {self.cv_folds}")
        
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
                
                logger.info(f"Running {self.cv_folds}-fold cross-validation...")
                
                # Train with cross-validation
                cv_results = trainer.cross_validate(
                    observations,
                    cv_folds=self.cv_folds,
                    n_iterations=100
                )
                
                logger.info(f"Training final model on full dataset...")
                
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
                
                logger.info(f"\n✓ {config_name} training complete")
                logger.info(f"  AIC: {artifact.metadata['aic']:.2f}")
                logger.info(f"  BIC: {artifact.metadata['bic']:.2f}")
                logger.info(f"  Log-Likelihood: {artifact.metadata['convergence_log_likelihood']:.2f}")
                
                # Log CV results if available
                if 'log_likelihood_mean' in cv_results:
                    logger.info(f"  CV Log-Likelihood: {cv_results['log_likelihood_mean']:.2f} ± {cv_results.get('log_likelihood_std', 0):.2f}")
                if 'aic_mean' in cv_results:
                    logger.info(f"  CV AIC: {cv_results['aic_mean']:.2f} ± {cv_results.get('aic_std', 0):.2f}")
                if 'bic_mean' in cv_results:
                    logger.info(f"  CV BIC: {cv_results['bic_mean']:.2f} ± {cv_results.get('bic_std', 0):.2f}")
                
                logger.info(f"  Artifact saved to: {artifact_path}")
                
            except Exception as e:
                logger.error(f"\n✗ {config_name} training failed: {e}")
                logger.error(f"  Error type: {type(e).__name__}")
                logger.error(f"  Continuing with remaining configurations...")
                
                # Store error information
                self.training_results[config_name] = {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'n_states': n_states
                }
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Training Summary")
        logger.info(f"{'='*60}")
        
        successful = sum(1 for r in self.training_results.values() if 'artifact' in r)
        failed = sum(1 for r in self.training_results.values() if 'error' in r)
        
        logger.info(f"Successful: {successful}/{len(self.n_states_range)}")
        logger.info(f"Failed: {failed}/{len(self.n_states_range)}")
        
        if successful == 0:
            logger.error("All training configurations failed!")
        else:
            logger.info(f"\n✓ Successfully trained {successful} model(s)")
        
        return self.training_results
    
    def evaluate_all_models(self, observations: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate all trained models comprehensively.
        
        Uses RegimeAnalyzer to decode state sequences and calculate regime characteristics,
        persistence, and interpretations for each model. Combines with basic metrics
        (AIC, BIC, CV scores) to create comprehensive evaluation results.
        
        Args:
            observations: Array of shape (n_samples, 3) with [s_LDC, s_MR, s_TSMOM]
            
        Returns:
            Dictionary with evaluation results for all models and rankings
        """
        logger.info("\n" + "="*60)
        logger.info("Comprehensive Model Evaluation")
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
            
            # Extract basic metrics from training
            basic_metrics = {
                'aic': artifact.metadata['aic'],
                'bic': artifact.metadata['bic'],
                'log_likelihood': artifact.metadata['convergence_log_likelihood'],
                'cv_log_likelihood_mean': cv_results.get('log_likelihood_mean', None),
                'cv_log_likelihood_std': cv_results.get('log_likelihood_std', None),
                'cv_aic_mean': cv_results.get('aic_mean', None),
                'cv_bic_mean': cv_results.get('bic_mean', None)
            }
            
            logger.info(f"  Basic Metrics:")
            logger.info(f"    AIC: {basic_metrics['aic']:.2f}")
            logger.info(f"    BIC: {basic_metrics['bic']:.2f}")
            logger.info(f"    Log-Likelihood: {basic_metrics['log_likelihood']:.2f}")
            
            # Perform regime analysis
            try:
                logger.info(f"  Performing regime analysis...")
                
                # Create RegimeAnalyzer
                analyzer = RegimeAnalyzer(artifact)
                
                # Decode state sequence using hmmlearn
                from hmmlearn import hmm
                model = hmm.GaussianHMM(
                    n_components=artifact.n_states,
                    covariance_type="full",
                    random_state=42
                )
                
                # Set model parameters from artifact
                model.transmat_ = np.array(artifact.transition_matrix)
                model.startprob_ = np.array(artifact.initial_probabilities)
                model.means_ = np.array(artifact.means)
                model.covars_ = np.array(artifact.covariances)
                
                # Decode state sequence
                state_sequence = model.predict(observations)
                logger.info(f"    State sequence decoded: {len(state_sequence)} states")
                
                # Characterize regimes
                logger.info(f"    Characterizing regimes...")
                characteristics = analyzer.characterize_regimes(observations, state_sequence)
                logger.info(f"    ✓ Characterized {len(characteristics)} regimes")
                
                # Analyze persistence
                logger.info(f"    Analyzing state persistence...")
                persistence = analyzer.analyze_state_persistence(state_sequence)
                logger.info(f"    ✓ Analyzed persistence for {len(persistence)} states")
                
                # Generate interpretations
                logger.info(f"    Generating economic interpretations...")
                interpretations = analyzer.interpret_regimes(characteristics, persistence)
                logger.info(f"    ✓ Generated interpretations for {len(interpretations)} regimes")
                
                # Calculate interpretability score
                interpretability_score = self._calculate_interpretability_score(
                    characteristics, persistence
                )
                logger.info(f"  Interpretability Score: {interpretability_score:.3f}")
                
                # Log regime summaries
                logger.info(f"  Regime Summary:")
                for state_id in sorted(characteristics.keys()):
                    char = characteristics[state_id]
                    pers = persistence.get(state_id)
                    interp = interpretations.get(state_id)
                    
                    logger.info(f"    State {state_id}:")
                    logger.info(f"      Type: {interp.regime_type if interp else 'Unknown'}")
                    logger.info(f"      Volatility: {char.volatility:.3f}")
                    logger.info(f"      Mean Duration: {pers.mean_duration:.1f} periods" if pers else "      Mean Duration: N/A")
                    logger.info(f"      Sample Count: {char.sample_count}")
                
                # Store comprehensive evaluation results
                evaluation_summary['models'][config_name] = {
                    'basic_metrics': basic_metrics,
                    'interpretability_score': interpretability_score,
                    'n_states': artifact.n_states,
                    'characteristics': {k: v.to_dict() for k, v in characteristics.items()},
                    'persistence': {k: v.to_dict() for k, v in persistence.items()},
                    'interpretations': {k: v.to_dict() for k, v in interpretations.items()}
                }
                
                logger.info(f"  ✓ Evaluation complete for {config_name}")
                
            except Exception as e:
                logger.error(f"  ✗ Regime analysis failed: {e}")
                logger.error(f"    Error type: {type(e).__name__}")
                
                # Store partial results with error
                evaluation_summary['models'][config_name] = {
                    'basic_metrics': basic_metrics,
                    'interpretability_score': 0.0,
                    'n_states': artifact.n_states,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        
        # Rank models based on combined criteria
        logger.info(f"\n{'='*60}")
        logger.info("Ranking Models")
        logger.info(f"{'='*60}")
        
        evaluation_summary['rankings'] = self._rank_models(evaluation_summary['models'])
        
        logger.info(f"✓ Ranked {len(evaluation_summary['rankings'])} models")
        
        return evaluation_summary
    
    def _calculate_interpretability_score(self, 
                                         characteristics: Dict,
                                         persistence: Dict) -> float:
        """
        Calculate overall interpretability score for a model.
        
        Combines volatility distinctiveness and state persistence to measure
        how interpretable and actionable the detected regimes are.
        
        Args:
            characteristics: Dictionary mapping state_id to RegimeCharacteristics
            persistence: Dictionary mapping state_id to StatePersistence
            
        Returns:
            Interpretability score between 0 and 1 (higher is better)
        """
        if not characteristics:
            return 0.0
        
        scores = []
        
        for state_id in characteristics.keys():
            char = characteristics[state_id]
            pers = persistence.get(state_id)
            
            # Component 1: State distinctiveness
            # Higher volatility difference from mean indicates more distinct regimes
            # Normalize volatility to 0-1 range (assuming max volatility ~2.0)
            volatility_score = min(1.0, char.volatility / 2.0)
            
            # Component 2: Persistence
            # Longer duration = more interpretable and tradeable
            # Normalize by max reasonable duration (20 periods)
            persistence_score = 0.0
            if pers:
                persistence_score = min(1.0, pers.mean_duration / 20.0)
            
            # Component 3: Sample adequacy
            # Need sufficient samples for statistical significance
            # Normalize by reasonable minimum (100 samples)
            sample_score = min(1.0, char.sample_count / 100.0)
            
            # Combined score with weights
            # Volatility: 40% - distinctiveness is key
            # Persistence: 40% - need stable regimes
            # Sample size: 20% - need adequate data
            state_score = (
                volatility_score * 0.4 +
                persistence_score * 0.4 +
                sample_score * 0.2
            )
            
            scores.append(state_score)
        
        # Return average score across all states
        return float(np.mean(scores))
    
    def _rank_models(self, models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank models by combined criteria with confidence scores and justification.
        
        Uses weighted scoring combining statistical fit (AIC, BIC),
        generalization (CV scores), and interpretability. Adds confidence
        scores based on metric consistency and justification for rankings.
        
        Args:
            models: Dictionary of model evaluation results
            
        Returns:
            List of ranked models (best first) with confidence and justification
        """
        rankings = []
        
        # Collect all metrics for normalization
        all_aic = []
        all_bic = []
        all_cv = []
        all_interp = []
        
        for config_name, model_data in models.items():
            if 'error' in model_data:
                continue
            
            metrics = model_data['basic_metrics']
            all_aic.append(metrics['aic'])
            all_bic.append(metrics['bic'])
            
            cv_score = metrics.get('cv_log_likelihood_mean', None)
            if cv_score is not None:
                all_cv.append(cv_score)
            
            all_interp.append(model_data['interpretability_score'])
        
        # Calculate normalization ranges
        aic_range = max(all_aic) - min(all_aic) if len(all_aic) > 1 else 1.0
        bic_range = max(all_bic) - min(all_bic) if len(all_bic) > 1 else 1.0
        cv_range = max(all_cv) - min(all_cv) if len(all_cv) > 1 else 1.0
        
        for config_name, model_data in models.items():
            if 'error' in model_data:
                continue
            
            metrics = model_data['basic_metrics']
            interpretability = model_data['interpretability_score']
            
            # Normalize AIC (lower is better, so invert)
            if aic_range > 0:
                aic_score = 1.0 - (metrics['aic'] - min(all_aic)) / aic_range
            else:
                aic_score = 1.0
            
            # Normalize BIC (lower is better, so invert)
            if bic_range > 0:
                bic_score = 1.0 - (metrics['bic'] - min(all_bic)) / bic_range
            else:
                bic_score = 1.0
            
            # Normalize CV score (higher is better)
            cv_score_raw = metrics.get('cv_log_likelihood_mean', None)
            if cv_score_raw is not None and cv_range > 0:
                cv_score = (cv_score_raw - min(all_cv)) / cv_range
            else:
                cv_score = 0.5  # Neutral if not available
            
            # Interpretability is already 0-1
            interp_score = interpretability
            
            # Combined score with weights
            # AIC: 30% - statistical fit
            # BIC: 30% - model complexity penalty
            # CV: 20% - generalization
            # Interpretability: 20% - practical usability
            combined_score = (
                aic_score * 0.30 +
                bic_score * 0.30 +
                cv_score * 0.20 +
                interp_score * 0.20
            )
            
            # Calculate confidence score based on metric consistency
            # High confidence when all metrics agree (all high or all low)
            component_scores = [aic_score, bic_score, cv_score, interp_score]
            score_std = np.std(component_scores)
            score_mean = np.mean(component_scores)
            
            # Confidence is higher when:
            # 1. Low variance (metrics agree)
            # 2. High mean score (all metrics are good)
            # 3. CV data is available (more reliable)
            consistency_factor = 1.0 - min(1.0, score_std / 0.5)  # Normalize std
            quality_factor = score_mean
            cv_availability_factor = 1.0 if cv_score_raw is not None else 0.7
            
            confidence_score = (
                consistency_factor * 0.4 +
                quality_factor * 0.4 +
                cv_availability_factor * 0.2
            )
            
            # Generate justification based on strengths and weaknesses
            strengths = []
            weaknesses = []
            
            if aic_score > 0.7:
                strengths.append(f"excellent statistical fit (AIC: {metrics['aic']:.2f})")
            elif aic_score < 0.3:
                weaknesses.append(f"poor statistical fit (AIC: {metrics['aic']:.2f})")
            
            if bic_score > 0.7:
                strengths.append(f"good complexity balance (BIC: {metrics['bic']:.2f})")
            elif bic_score < 0.3:
                weaknesses.append(f"complexity penalty (BIC: {metrics['bic']:.2f})")
            
            if cv_score_raw is not None:
                if cv_score > 0.7:
                    strengths.append(f"strong generalization (CV: {cv_score_raw:.2f})")
                elif cv_score < 0.3:
                    weaknesses.append(f"weak generalization (CV: {cv_score_raw:.2f})")
            else:
                weaknesses.append("CV score unavailable")
            
            if interp_score > 0.7:
                strengths.append(f"highly interpretable regimes ({interp_score:.2f})")
            elif interp_score < 0.3:
                weaknesses.append(f"low interpretability ({interp_score:.2f})")
            
            # Build justification text
            justification_parts = []
            if strengths:
                justification_parts.append("Strengths: " + "; ".join(strengths))
            if weaknesses:
                justification_parts.append("Weaknesses: " + "; ".join(weaknesses))
            
            justification = ". ".join(justification_parts) if justification_parts else "Balanced performance across all metrics"
            
            rankings.append({
                'config_name': config_name,
                'combined_score': combined_score,
                'confidence_score': confidence_score,
                'aic': metrics['aic'],
                'bic': metrics['bic'],
                'cv_score': cv_score_raw if cv_score_raw is not None else 0.0,
                'interpretability': interpretability,
                'n_states': model_data['n_states'],
                'component_scores': {
                    'aic_score': aic_score,
                    'bic_score': bic_score,
                    'cv_score': cv_score,
                    'interpretability_score': interp_score
                },
                'justification': justification
            })
        
        # Sort by combined score (descending - higher is better)
        rankings.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return rankings
    
    def select_best_model(self, evaluation_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best model based on weighted scoring and save as production artifact.
        
        Chooses the top-ranked configuration and saves it as hmm_best.json for
        easy production deployment. Includes confidence scores and justification.
        
        Args:
            evaluation_summary: Results from evaluate_all_models()
            
        Returns:
            Dictionary with best model information, scores, and justification
            
        Raises:
            ValueError: If no models are available for selection
        """
        logger.info("\n" + "="*60)
        logger.info("Model Selection")
        logger.info("="*60)
        
        if not evaluation_summary['rankings']:
            raise ValueError("No models available for selection - all training failed")
        
        # Get top-ranked model
        best_model = evaluation_summary['rankings'][0]
        best_config = best_model['config_name']
        
        # Log selection details
        logger.info(f"\n🏆 Best Model Selected: {best_config}")
        logger.info(f"{'='*60}")
        
        logger.info(f"\nOverall Scores:")
        logger.info(f"  Combined Score: {best_model['combined_score']:.3f}")
        logger.info(f"  Confidence: {best_model['confidence_score']:.3f}")
        
        logger.info(f"\nComponent Metrics:")
        logger.info(f"  AIC: {best_model['aic']:.2f} (normalized: {best_model['component_scores']['aic_score']:.3f})")
        logger.info(f"  BIC: {best_model['bic']:.2f} (normalized: {best_model['component_scores']['bic_score']:.3f})")
        logger.info(f"  CV Log-Likelihood: {best_model['cv_score']:.2f} (normalized: {best_model['component_scores']['cv_score']:.3f})")
        logger.info(f"  Interpretability: {best_model['interpretability']:.3f}")
        
        logger.info(f"\nJustification:")
        logger.info(f"  {best_model['justification']}")
        
        # Get the artifact
        best_artifact = self.training_results[best_config]['artifact']
        
        # Add selection metadata to artifact
        selection_metadata = {
            'selection_timestamp': datetime.now().isoformat(),
            'selection_method': 'weighted_scoring',
            'combined_score': best_model['combined_score'],
            'confidence_score': best_model['confidence_score'],
            'ranking_position': 1,
            'total_candidates': len(evaluation_summary['rankings']),
            'component_scores': best_model['component_scores'],
            'justification': best_model['justification']
        }
        
        # Save best model artifact
        best_model_path = self.output_dir / "hmm_best.json"
        
        # Create enhanced artifact dict with selection metadata
        artifact_dict = best_artifact.model_dump()
        artifact_dict['metadata']['selection'] = selection_metadata
        
        with open(best_model_path, 'w') as f:
            json.dump(artifact_dict, f, indent=2)
        
        logger.info(f"\n✓ Best model saved to: {best_model_path}")
        
        # Log comparison with other models if available
        if len(evaluation_summary['rankings']) > 1:
            logger.info(f"\nComparison with Alternatives:")
            for i, model in enumerate(evaluation_summary['rankings'][1:], 2):
                score_diff = best_model['combined_score'] - model['combined_score']
                logger.info(f"  #{i} {model['config_name']}: {model['combined_score']:.3f} (Δ {score_diff:.3f})")
        
        # Check if confidence is low and provide recommendations
        if best_model['confidence_score'] < 0.5:
            logger.warning(f"\n⚠️  Low confidence score ({best_model['confidence_score']:.3f})")
            logger.warning("Recommendations:")
            logger.warning("  - Consider collecting more training data")
            logger.warning("  - Review model assumptions and data quality")
            logger.warning("  - Try different covariance types or initialization strategies")
        elif best_model['confidence_score'] < 0.7:
            logger.info(f"\nℹ️  Moderate confidence score ({best_model['confidence_score']:.3f})")
            logger.info("Consider validating on additional out-of-sample data before production deployment")
        else:
            logger.info(f"\n✓ High confidence score ({best_model['confidence_score']:.3f})")
            logger.info("Model is ready for production deployment")
        
        return {
            'config_name': best_config,
            'artifact': best_artifact,
            'artifact_path': str(best_model_path),
            'scores': {
                'combined_score': best_model['combined_score'],
                'confidence_score': best_model['confidence_score'],
                'aic': best_model['aic'],
                'bic': best_model['bic'],
                'cv_score': best_model['cv_score'],
                'interpretability': best_model['interpretability']
            },
            'component_scores': best_model['component_scores'],
            'justification': best_model['justification'],
            'selection_metadata': selection_metadata
        }


    def _print_summary_table(self, rankings: List[Dict[str, Any]]) -> None:
        """
        Print formatted summary table of model rankings.
        
        Args:
            rankings: List of ranked models from _rank_models()
        """
        logger.info("\n" + "="*80)
        logger.info("Model Comparison Summary")
        logger.info("="*80)
        
        if not rankings:
            logger.warning("No models to display")
            return
        
        # Header
        header = f"{'Rank':<6} {'Model':<14} {'Score':<8} {'Conf':<8} {'AIC':<10} {'BIC':<10} {'Interp':<8}"
        logger.info(header)
        logger.info("-" * 80)
        
        # Rows
        for i, model in enumerate(rankings, 1):
            row = (
                f"{i:<6} "
                f"{model['config_name']:<14} "
                f"{model['combined_score']:<8.3f} "
                f"{model['confidence_score']:<8.3f} "
                f"{model['aic']:<10.2f} "
                f"{model['bic']:<10.2f} "
                f"{model['interpretability']:<8.3f}"
            )
            logger.info(row)
        
        logger.info("="*80)
        logger.info("\nLegend:")
        logger.info("  Score: Combined weighted score (higher is better)")
        logger.info("  Conf: Confidence score based on metric consistency")
        logger.info("  AIC/BIC: Information criteria (lower is better)")
        logger.info("  Interp: Interpretability score (higher is better)")
    
    def generate_report(self, evaluation_summary: Dict[str, Any], 
                       best_model_info: Dict[str, Any]) -> None:
        """
        Generate comprehensive training report with all results.
        
        Args:
            evaluation_summary: Results from evaluate_all_models()
            best_model_info: Results from select_best_model()
        """
        logger.info("\n" + "="*60)
        logger.info("Generating Comprehensive Report")
        logger.info("="*60)
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'data_path': str(self.data_path),
                'output_dir': str(self.output_dir),
                'n_states_range': self.n_states_range,
                'cv_folds': self.cv_folds
            },
            'best_model': {
                'config_name': best_model_info['config_name'],
                'artifact_path': best_model_info['artifact_path'],
                'scores': best_model_info['scores'],
                'component_scores': best_model_info['component_scores'],
                'justification': best_model_info['justification'],
                'selection_metadata': best_model_info['selection_metadata']
            },
            'all_models': evaluation_summary['models'],
            'rankings': evaluation_summary['rankings']
        }
        
        # Convert numpy types to native Python types
        report = convert_to_serializable(report)
        
        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✓ Report saved to: {report_path}")
        
        # Print summary table
        self._print_summary_table(evaluation_summary['rankings'])
        
        logger.info(f"\n{'='*60}")
        logger.info("Report Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total models evaluated: {len(evaluation_summary['models'])}")
        logger.info(f"Best model: {best_model_info['config_name']}")
        logger.info(f"Best model path: {best_model_info['artifact_path']}")
        logger.info(f"Combined score: {best_model_info['scores']['combined_score']:.3f}")
        logger.info(f"Confidence: {best_model_info['scores']['confidence_score']:.3f}")
    
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
        best_model_info = self.select_best_model(evaluation_summary)
        
        # 5. Generate report
        self.generate_report(evaluation_summary, best_model_info)
        
        logger.info("\n" + "="*60)
        logger.info("✓ Systematic Training Complete")
        logger.info("="*60)
        
        return {
            'best_model': best_model_info,
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
    
    # Run complete pipeline
    try:
        results = trainer.run()
        
        logger.info(f"\n✓ Pipeline completed successfully")
        logger.info(f"✓ Best model: {results['best_model']['config_name']}")
        logger.info(f"✓ Artifacts saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        logger.error(f"  Error type: {type(e).__name__}")
        import traceback
        logger.error(f"  Traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == '__main__':
    exit(main())
