"""
Ensemble model evaluation and comparison framework.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from scipy import stats

from ..hmm.trainer import EnhancedHMMTrainer
from ..hmm.models import HMMArtifact

logger = logging.getLogger(__name__)


@dataclass
class EnsembleMember:
    """Individual model in an ensemble."""
    config: Dict[str, Any]
    artifact: HMMArtifact
    weight: float
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config,
            'artifact': self.artifact.model_dump(),
            'weight': self.weight,
            'performance_metrics': self.performance_metrics
        }


@dataclass
class EnsembleResult:
    """Result from ensemble evaluation."""
    members: List[EnsembleMember]
    ensemble_predictions: np.ndarray
    ensemble_performance: Dict[str, float]
    diversity_metrics: Dict[str, float]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'members': [m.to_dict() for m in self.members],
            'ensemble_predictions': self.ensemble_predictions.tolist(),
            'ensemble_performance': self.ensemble_performance,
            'diversity_metrics': self.diversity_metrics,
            'timestamp': self.timestamp
        }
    
    def save(self, filepath: Path):
        """Save results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Ensemble results saved to {filepath}")


class EnsembleEvaluator:
    """
    Ensemble model evaluation and comparison framework.
    
    Supports multiple ensemble strategies:
    - Weighted averaging of state probabilities
    - Voting-based ensemble
    - Stacking with meta-learner
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ensemble evaluator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.ensemble_members: List[EnsembleMember] = []
    
    def create_ensemble(self,
                       observations: np.ndarray,
                       configs: List[Dict[str, Any]],
                       n_iterations: int = 100,
                       weighting_strategy: str = 'performance',
                       validation_split: float = 0.2) -> EnsembleResult:
        """
        Create and evaluate an ensemble of HMM models.
        
        Args:
            observations: Training data
            configs: List of model configurations
            n_iterations: Number of training iterations
            weighting_strategy: 'uniform', 'performance', or 'diversity'
            validation_split: Fraction of data for validation
            
        Returns:
            EnsembleResult with ensemble predictions and metrics
        """
        logger.info(f"Creating ensemble with {len(configs)} models")
        
        # Split data
        split_idx = int(len(observations) * (1 - validation_split))
        train_data = observations[:split_idx]
        val_data = observations[split_idx:]
        
        # Train all models
        members = []
        for i, config in enumerate(configs):
            logger.info(f"Training ensemble member {i+1}/{len(configs)}")
            
            try:
                # Train model
                trainer = EnhancedHMMTrainer(**config)
                artifact = trainer.train(train_data, n_iterations)
                
                # Evaluate on validation set
                metrics = trainer.trainer.evaluate(val_data)
                
                # Create member
                member = EnsembleMember(
                    config=config,
                    artifact=artifact,
                    weight=1.0 / len(configs),  # Initial uniform weight
                    performance_metrics=metrics
                )
                members.append(member)
            
            except Exception as e:
                logger.warning(f"Failed to train member {i+1}: {str(e)}")
                continue
        
        if not members:
            raise ValueError("No models successfully trained for ensemble")
        
        # Calculate weights based on strategy
        if weighting_strategy == 'performance':
            members = self._calculate_performance_weights(members)
        elif weighting_strategy == 'diversity':
            members = self._calculate_diversity_weights(members, val_data)
        # else: keep uniform weights
        
        self.ensemble_members = members
        
        # Generate ensemble predictions
        ensemble_preds = self._generate_ensemble_predictions(members, observations)
        
        # Evaluate ensemble performance
        ensemble_perf = self._evaluate_ensemble(members, observations, ensemble_preds)
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(members, observations)
        
        result = EnsembleResult(
            members=members,
            ensemble_predictions=ensemble_preds,
            ensemble_performance=ensemble_perf,
            diversity_metrics=diversity_metrics,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Ensemble created with {len(members)} members")
        
        return result
    
    def _calculate_performance_weights(self, members: List[EnsembleMember]) -> List[EnsembleMember]:
        """Calculate weights based on validation performance."""
        # Use log-likelihood as performance metric
        log_likelihoods = [m.performance_metrics.get('log_likelihood', 0.0) for m in members]
        
        # Convert to positive weights (shift if necessary)
        min_ll = min(log_likelihoods)
        if min_ll < 0:
            log_likelihoods = [ll - min_ll + 1 for ll in log_likelihoods]
        
        # Normalize to sum to 1
        total = sum(log_likelihoods)
        if total > 0:
            weights = [ll / total for ll in log_likelihoods]
        else:
            weights = [1.0 / len(members)] * len(members)
        
        # Update member weights
        for member, weight in zip(members, weights):
            member.weight = weight
        
        logger.info(f"Performance-based weights: {[f'{w:.3f}' for w in weights]}")
        
        return members
    
    def _calculate_diversity_weights(self,
                                    members: List[EnsembleMember],
                                    observations: np.ndarray) -> List[EnsembleMember]:
        """Calculate weights based on prediction diversity."""
        # Get predictions from all members
        predictions = []
        for member in members:
            try:
                # Reconstruct the model from artifact
                from hmmlearn import hmm as hmmlearn_hmm
                
                model = hmmlearn_hmm.GaussianHMM(
                    n_components=member.artifact.n_states,
                    covariance_type=member.config.get('covariance_type', 'diag')
                )
                model.startprob_ = np.array(member.artifact.initial_probabilities)
                model.transmat_ = np.array(member.artifact.transition_matrix)
                model.means_ = np.array(member.artifact.means)
                model.covars_ = np.array(member.artifact.covariances)
                
                # Get state probabilities
                state_probs = model.predict_proba(observations)
                predictions.append(state_probs)
            except Exception as e:
                logger.warning(f"Failed to get predictions for diversity calculation: {str(e)}")
                predictions.append(None)
        
        # Calculate pairwise diversity
        n_members = len(members)
        diversity_scores = np.zeros(n_members)
        
        for i in range(n_members):
            if predictions[i] is None:
                continue
            
            total_diversity = 0.0
            count = 0
            
            for j in range(n_members):
                if i != j and predictions[j] is not None:
                    # Calculate disagreement (1 - correlation)
                    pred_i = np.argmax(predictions[i], axis=1)
                    pred_j = np.argmax(predictions[j], axis=1)
                    disagreement = np.mean(pred_i != pred_j)
                    total_diversity += disagreement
                    count += 1
            
            if count > 0:
                diversity_scores[i] = total_diversity / count
        
        # Combine with performance
        log_likelihoods = [m.performance_metrics.get('log_likelihood', 0.0) for m in members]
        min_ll = min(log_likelihoods)
        if min_ll < 0:
            log_likelihoods = [ll - min_ll + 1 for ll in log_likelihoods]
        
        # Normalize both
        if max(log_likelihoods) > 0:
            norm_perf = [ll / max(log_likelihoods) for ll in log_likelihoods]
        else:
            norm_perf = [1.0] * n_members
        
        if max(diversity_scores) > 0:
            norm_div = [d / max(diversity_scores) for d in diversity_scores]
        else:
            norm_div = [1.0] * n_members
        
        # Combine (50% performance, 50% diversity)
        combined_scores = [0.5 * p + 0.5 * d for p, d in zip(norm_perf, norm_div)]
        
        # Normalize to sum to 1
        total = sum(combined_scores)
        if total > 0:
            weights = [s / total for s in combined_scores]
        else:
            weights = [1.0 / n_members] * n_members
        
        # Update member weights
        for member, weight in zip(members, weights):
            member.weight = weight
        
        logger.info(f"Diversity-based weights: {[f'{w:.3f}' for w in weights]}")
        
        return members
    
    def _generate_ensemble_predictions(self,
                                      members: List[EnsembleMember],
                                      observations: np.ndarray) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        # Get predictions from all members
        all_predictions = []
        valid_weights = []
        
        for member in members:
            try:
                # Use the trainer to get predictions properly
                trainer = EnhancedHMMTrainer(**member.config)
                
                # Reconstruct the model from artifact using the trainer's method
                from hmmlearn import hmm as hmmlearn_hmm
                
                model = hmmlearn_hmm.GaussianHMM(
                    n_components=member.artifact.n_states,
                    covariance_type=member.config.get('covariance_type', 'diag')
                )
                model.startprob_ = np.array(member.artifact.initial_probabilities)
                model.transmat_ = np.array(member.artifact.transition_matrix)
                model.means_ = np.array(member.artifact.means)
                
                # Handle covariances based on type
                cov_type = member.config.get('covariance_type', 'diag')
                covars = np.array(member.artifact.covariances)
                
                if cov_type == 'diag':
                    # For diag, extract diagonal elements
                    if covars.ndim == 3:
                        model.covars_ = np.array([np.diag(cov) for cov in covars])
                    else:
                        model.covars_ = covars
                elif cov_type == 'full':
                    model.covars_ = covars
                elif cov_type == 'spherical':
                    # For spherical, take mean of diagonal
                    if covars.ndim == 3:
                        model.covars_ = np.array([np.mean(np.diag(cov)) for cov in covars])
                    else:
                        model.covars_ = covars
                
                # Get state probabilities
                state_probs = model.predict_proba(observations)
                all_predictions.append(state_probs)
                valid_weights.append(member.weight)
            
            except Exception as e:
                logger.warning(f"Failed to get predictions from member: {str(e)}")
                continue
        
        if not all_predictions:
            raise ValueError("No valid predictions from ensemble members")
        
        # Normalize weights
        total_weight = sum(valid_weights)
        valid_weights = [w / total_weight for w in valid_weights]
        
        # Weighted average of state probabilities
        ensemble_preds = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, valid_weights):
            ensemble_preds += weight * pred
        
        return ensemble_preds
    
    def _evaluate_ensemble(self,
                          members: List[EnsembleMember],
                          observations: np.ndarray,
                          ensemble_preds: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        # Calculate ensemble log-likelihood
        # Use the most likely state sequence
        most_likely_states = np.argmax(ensemble_preds, axis=1)
        
        # Calculate metrics
        n_states = ensemble_preds.shape[1]
        n_samples = len(observations)
        
        # Average individual model performance
        avg_ll = np.mean([m.performance_metrics.get('log_likelihood', 0.0) for m in members])
        avg_aic = np.mean([m.performance_metrics.get('aic', 0.0) for m in members])
        avg_bic = np.mean([m.performance_metrics.get('bic', 0.0) for m in members])
        
        # Ensemble-specific metrics
        state_entropy = -np.sum(ensemble_preds * np.log(ensemble_preds + 1e-10), axis=1).mean()
        prediction_confidence = np.max(ensemble_preds, axis=1).mean()
        
        return {
            'avg_member_log_likelihood': float(avg_ll),
            'avg_member_aic': float(avg_aic),
            'avg_member_bic': float(avg_bic),
            'ensemble_state_entropy': float(state_entropy),
            'ensemble_prediction_confidence': float(prediction_confidence),
            'n_members': len(members),
            'n_states': n_states
        }
    
    def _calculate_diversity_metrics(self,
                                    members: List[EnsembleMember],
                                    observations: np.ndarray) -> Dict[str, float]:
        """Calculate diversity metrics for the ensemble."""
        # Get predictions from all members
        predictions = []
        for member in members:
            try:
                # Reconstruct the model from artifact
                from hmmlearn import hmm as hmmlearn_hmm
                
                model = hmmlearn_hmm.GaussianHMM(
                    n_components=member.artifact.n_states,
                    covariance_type=member.config.get('covariance_type', 'diag')
                )
                model.startprob_ = np.array(member.artifact.initial_probabilities)
                model.transmat_ = np.array(member.artifact.transition_matrix)
                model.means_ = np.array(member.artifact.means)
                model.covars_ = np.array(member.artifact.covariances)
                
                state_probs = model.predict_proba(observations)
                predictions.append(np.argmax(state_probs, axis=1))
            except Exception as e:
                logger.warning(f"Failed to get predictions for diversity: {str(e)}")
                continue
        
        if len(predictions) < 2:
            return {'pairwise_disagreement': 0.0, 'diversity_score': 0.0}
        
        # Calculate pairwise disagreement
        disagreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreements.append(disagreement)
        
        avg_disagreement = np.mean(disagreements)
        
        # Calculate Q-statistic (measure of diversity)
        q_statistics = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # Calculate agreement matrix
                n11 = np.sum((predictions[i] == predictions[j]))
                n00 = np.sum((predictions[i] != predictions[j]))
                n10 = np.sum((predictions[i] != predictions[j]))
                n01 = np.sum((predictions[i] == predictions[j]))
                
                # Q-statistic
                numerator = n11 * n00 - n01 * n10
                denominator = n11 * n00 + n01 * n10
                
                if denominator > 0:
                    q = numerator / denominator
                    q_statistics.append(q)
        
        avg_q = np.mean(q_statistics) if q_statistics else 0.0
        
        return {
            'pairwise_disagreement': float(avg_disagreement),
            'q_statistic': float(avg_q),
            'diversity_score': float(1.0 - avg_q)  # Higher is more diverse
        }
    
    def compare_ensemble_vs_individual(self,
                                      ensemble_result: EnsembleResult,
                                      observations: np.ndarray) -> pd.DataFrame:
        """
        Compare ensemble performance against individual models.
        
        Args:
            ensemble_result: Result from create_ensemble
            observations: Test data
            
        Returns:
            DataFrame with comparison results
        """
        comparisons = []
        
        # Individual model performance
        for i, member in enumerate(ensemble_result.members):
            comparisons.append({
                'model': f'Member_{i+1}',
                'type': 'individual',
                'weight': member.weight,
                'log_likelihood': member.performance_metrics.get('log_likelihood', np.nan),
                'aic': member.performance_metrics.get('aic', np.nan),
                'bic': member.performance_metrics.get('bic', np.nan),
                'n_states': member.config.get('n_states', np.nan)
            })
        
        # Ensemble performance
        comparisons.append({
            'model': 'Ensemble',
            'type': 'ensemble',
            'weight': 1.0,
            'log_likelihood': ensemble_result.ensemble_performance.get('avg_member_log_likelihood', np.nan),
            'aic': ensemble_result.ensemble_performance.get('avg_member_aic', np.nan),
            'bic': ensemble_result.ensemble_performance.get('avg_member_bic', np.nan),
            'n_states': ensemble_result.ensemble_performance.get('n_states', np.nan)
        })
        
        df = pd.DataFrame(comparisons)
        
        return df
    
    def get_ensemble_report(self, ensemble_result: EnsembleResult) -> str:
        """Generate human-readable ensemble report."""
        report = []
        report.append("="*70)
        report.append("ENSEMBLE EVALUATION REPORT")
        report.append("="*70)
        report.append("")
        
        # Ensemble composition
        report.append(f"Ensemble Members: {len(ensemble_result.members)}")
        report.append("")
        
        for i, member in enumerate(ensemble_result.members):
            report.append(f"Member {i+1}:")
            report.append(f"  Configuration: {member.config.get('n_states')} states, "
                         f"{member.config.get('library')}, {member.config.get('covariance_type')}")
            report.append(f"  Weight: {member.weight:.4f}")
            report.append(f"  Log-Likelihood: {member.performance_metrics.get('log_likelihood', 'N/A')}")
        
        report.append("")
        report.append("Ensemble Performance:")
        for metric, value in ensemble_result.ensemble_performance.items():
            if isinstance(value, float):
                report.append(f"  {metric}: {value:.4f}")
            else:
                report.append(f"  {metric}: {value}")
        
        report.append("")
        report.append("Diversity Metrics:")
        for metric, value in ensemble_result.diversity_metrics.items():
            report.append(f"  {metric}: {value:.4f}")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
