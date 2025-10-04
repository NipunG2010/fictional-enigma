"""
Advanced regime analysis and economic interpretation tools for HMM models.

This module provides comprehensive tools for analyzing market regimes detected by HMM models,
including statistical characterization, persistence analysis, economic interpretation,
and actionable trading insights.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

from .models import HMMArtifact, HMMPrediction


@dataclass
class RegimeCharacteristics:
    """Statistical characteristics of a market regime."""
    
    state_id: int
    mean_values: Dict[str, float]
    std_values: Dict[str, float]
    volatility: float
    trend_strength: float
    mean_reversion_score: float
    sample_count: int
    feature_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'state_id': self.state_id,
            'mean_values': self.mean_values,
            'std_values': self.std_values,
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'mean_reversion_score': self.mean_reversion_score,
            'sample_count': self.sample_count,
            'feature_statistics': self.feature_statistics
        }


@dataclass
class StatePersistence:
    """State persistence and transition statistics."""
    
    state_id: int
    mean_duration: float
    median_duration: float
    max_duration: int
    min_duration: int
    total_occurrences: int
    stable_periods: int  # Periods lasting >= threshold
    transition_frequencies: Dict[int, int]
    transition_probabilities: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'state_id': self.state_id,
            'mean_duration': self.mean_duration,
            'median_duration': self.median_duration,
            'max_duration': self.max_duration,
            'min_duration': self.min_duration,
            'total_occurrences': self.total_occurrences,
            'stable_periods': self.stable_periods,
            'transition_frequencies': self.transition_frequencies,
            'transition_probabilities': self.transition_probabilities
        }


@dataclass
class EconomicInterpretation:
    """Economic interpretation of a market regime."""
    
    state_id: int
    regime_type: str  # e.g., "High Volatility", "Trending", "Mean Reverting"
    market_condition: str  # e.g., "Bull", "Bear", "Sideways"
    risk_level: str  # "Low", "Medium", "High"
    trading_recommendation: str
    key_characteristics: List[str]
    correlated_events: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'state_id': self.state_id,
            'regime_type': self.regime_type,
            'market_condition': self.market_condition,
            'risk_level': self.risk_level,
            'trading_recommendation': self.trading_recommendation,
            'key_characteristics': self.key_characteristics,
            'correlated_events': self.correlated_events
        }


@dataclass
class FeatureImportance:
    """Feature importance for regime characterization."""
    
    state_id: int
    feature_scores: Dict[str, float]
    ranked_features: List[Tuple[str, float]]
    dominant_features: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'state_id': self.state_id,
            'feature_scores': self.feature_scores,
            'ranked_features': self.ranked_features,
            'dominant_features': self.dominant_features
        }


class RegimeAnalyzer:
    """
    Comprehensive regime analysis tool for HMM models.
    
    Provides statistical characterization, persistence analysis, economic interpretation,
    and actionable trading insights for detected market regimes.
    """
    
    def __init__(self, artifact: HMMArtifact):
        """
        Initialize regime analyzer.
        
        Args:
            artifact: Trained HMM artifact
        """
        self.artifact = artifact
        self.n_states = artifact.n_states
        self.feature_names = ['s_ldc', 's_mr', 's_tsmom']
        
    def characterize_regimes(
        self,
        observations: np.ndarray,
        state_sequence: np.ndarray
    ) -> Dict[int, RegimeCharacteristics]:
        """
        Characterize each regime with statistical analysis.
        
        Args:
            observations: Observation data (n_samples, n_features)
            state_sequence: Decoded state sequence (n_samples,)
            
        Returns:
            Dictionary mapping state_id to RegimeCharacteristics
        """
        characteristics = {}
        
        for state_id in range(self.n_states):
            # Get observations for this state
            state_mask = state_sequence == state_id
            state_obs = observations[state_mask]
            
            if len(state_obs) == 0:
                continue
            
            # Calculate basic statistics
            mean_values = {
                name: float(np.mean(state_obs[:, i]))
                for i, name in enumerate(self.feature_names)
            }
            
            std_values = {
                name: float(np.std(state_obs[:, i]))
                for i, name in enumerate(self.feature_names)
            }
            
            # Calculate volatility (average standard deviation across features)
            volatility = float(np.mean([std_values[name] for name in self.feature_names]))
            
            # Calculate trend strength (using autocorrelation)
            trend_strength = self._calculate_trend_strength(state_obs)
            
            # Calculate mean reversion score
            mean_reversion_score = self._calculate_mean_reversion_score(state_obs)
            
            # Detailed feature statistics
            feature_statistics = {}
            for i, name in enumerate(self.feature_names):
                feature_data = state_obs[:, i]
                feature_statistics[name] = {
                    'mean': float(np.mean(feature_data)),
                    'std': float(np.std(feature_data)),
                    'min': float(np.min(feature_data)),
                    'max': float(np.max(feature_data)),
                    'median': float(np.median(feature_data)),
                    'skewness': float(stats.skew(feature_data)),
                    'kurtosis': float(stats.kurtosis(feature_data))
                }
            
            characteristics[state_id] = RegimeCharacteristics(
                state_id=state_id,
                mean_values=mean_values,
                std_values=std_values,
                volatility=volatility,
                trend_strength=trend_strength,
                mean_reversion_score=mean_reversion_score,
                sample_count=len(state_obs),
                feature_statistics=feature_statistics
            )
        
        return characteristics
    
    def analyze_state_persistence(
        self,
        state_sequence: np.ndarray,
        min_stable_duration: int = 5
    ) -> Dict[int, StatePersistence]:
        """
        Analyze state persistence and transition patterns.
        
        Args:
            state_sequence: Decoded state sequence (n_samples,)
            min_stable_duration: Minimum duration to consider a period "stable"
            
        Returns:
            Dictionary mapping state_id to StatePersistence
        """
        persistence_stats = {}
        
        # Track state durations and transitions
        state_durations = defaultdict(list)
        state_transitions = defaultdict(lambda: defaultdict(int))
        
        current_state = state_sequence[0]
        current_duration = 1
        
        for i in range(1, len(state_sequence)):
            if state_sequence[i] == current_state:
                current_duration += 1
            else:
                # Record duration
                state_durations[current_state].append(current_duration)
                
                # Record transition
                next_state = state_sequence[i]
                state_transitions[current_state][next_state] += 1
                
                # Reset for new state
                current_state = next_state
                current_duration = 1
        
        # Record final duration
        state_durations[current_state].append(current_duration)
        
        # Calculate statistics for each state
        for state_id in range(self.n_states):
            durations = state_durations.get(state_id, [])
            
            if not durations:
                continue
            
            # Duration statistics
            mean_duration = float(np.mean(durations))
            median_duration = float(np.median(durations))
            max_duration = int(np.max(durations))
            min_duration = int(np.min(durations))
            total_occurrences = len(durations)
            stable_periods = sum(1 for d in durations if d >= min_stable_duration)
            
            # Transition statistics
            transitions = state_transitions[state_id]
            total_transitions = sum(transitions.values())
            
            transition_frequencies = dict(transitions)
            transition_probabilities = {
                to_state: count / total_transitions
                for to_state, count in transitions.items()
            } if total_transitions > 0 else {}
            
            persistence_stats[state_id] = StatePersistence(
                state_id=state_id,
                mean_duration=mean_duration,
                median_duration=median_duration,
                max_duration=max_duration,
                min_duration=min_duration,
                total_occurrences=total_occurrences,
                stable_periods=stable_periods,
                transition_frequencies=transition_frequencies,
                transition_probabilities=transition_probabilities
            )
        
        return persistence_stats
    
    def interpret_regimes(
        self,
        characteristics: Dict[int, RegimeCharacteristics],
        persistence: Dict[int, StatePersistence]
    ) -> Dict[int, EconomicInterpretation]:
        """
        Provide economic interpretation of detected regimes.
        
        Args:
            characteristics: Regime characteristics from characterize_regimes()
            persistence: Persistence statistics from analyze_state_persistence()
            
        Returns:
            Dictionary mapping state_id to EconomicInterpretation
        """
        interpretations = {}
        
        for state_id in range(self.n_states):
            if state_id not in characteristics:
                continue
            
            char = characteristics[state_id]
            pers = persistence.get(state_id)
            
            # Determine regime type based on characteristics
            regime_type = self._classify_regime_type(char)
            
            # Determine market condition
            market_condition = self._classify_market_condition(char)
            
            # Assess risk level
            risk_level = self._assess_risk_level(char)
            
            # Generate trading recommendation
            trading_recommendation = self._generate_trading_recommendation(
                regime_type, market_condition, risk_level, pers
            )
            
            # Identify key characteristics
            key_characteristics = self._identify_key_characteristics(char, pers)
            
            interpretations[state_id] = EconomicInterpretation(
                state_id=state_id,
                regime_type=regime_type,
                market_condition=market_condition,
                risk_level=risk_level,
                trading_recommendation=trading_recommendation,
                key_characteristics=key_characteristics
            )
        
        return interpretations
    
    def analyze_feature_importance(
        self,
        observations: np.ndarray,
        state_sequence: np.ndarray
    ) -> Dict[int, FeatureImportance]:
        """
        Analyze feature importance for regime characterization.
        
        Args:
            observations: Observation data (n_samples, n_features)
            state_sequence: Decoded state sequence (n_samples,)
            
        Returns:
            Dictionary mapping state_id to FeatureImportance
        """
        importance_results = {}
        
        for state_id in range(self.n_states):
            state_mask = state_sequence == state_id
            state_obs = observations[state_mask]
            
            if len(state_obs) == 0:
                continue
            
            # Calculate feature importance scores
            feature_scores = {}
            
            for i, name in enumerate(self.feature_names):
                # Use coefficient of variation as importance metric
                mean_val = np.mean(state_obs[:, i])
                std_val = np.std(state_obs[:, i])
                
                # Avoid division by zero
                if abs(mean_val) > 1e-10:
                    cv = abs(std_val / mean_val)
                else:
                    cv = std_val
                
                # Also consider absolute magnitude
                magnitude = abs(mean_val)
                
                # Combined score (normalized)
                score = cv * 0.5 + magnitude * 0.5
                feature_scores[name] = float(score)
            
            # Normalize scores to sum to 1
            total_score = sum(feature_scores.values())
            if total_score > 0:
                feature_scores = {
                    name: score / total_score
                    for name, score in feature_scores.items()
                }
            
            # Rank features
            ranked_features = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Identify dominant features (top contributors)
            dominant_features = [
                name for name, score in ranked_features
                if score > 0.25  # More than 25% contribution
            ]
            
            importance_results[state_id] = FeatureImportance(
                state_id=state_id,
                feature_scores=feature_scores,
                ranked_features=ranked_features,
                dominant_features=dominant_features
            )
        
        return importance_results
    
    def validate_regimes(
        self,
        observations: np.ndarray,
        state_sequence: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        known_events: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Validate detected regimes against known market conditions.
        
        Args:
            observations: Observation data (n_samples, n_features)
            state_sequence: Decoded state sequence (n_samples,)
            timestamps: Optional timestamps for observations
            known_events: Optional list of known market events with timestamps
            
        Returns:
            Validation report with metrics and correlations
        """
        validation_report = {
            'regime_stability': self._assess_regime_stability(state_sequence),
            'state_separability': self._assess_state_separability(observations, state_sequence),
            'temporal_consistency': self._assess_temporal_consistency(state_sequence)
        }
        
        # Correlate with known events if provided
        if known_events and timestamps is not None:
            event_correlations = self._correlate_with_events(
                state_sequence, timestamps, known_events
            )
            validation_report['event_correlations'] = event_correlations
        
        return validation_report
    
    def generate_regime_report(
        self,
        observations: np.ndarray,
        state_sequence: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive regime analysis report.
        
        Args:
            observations: Observation data (n_samples, n_features)
            state_sequence: Decoded state sequence (n_samples,)
            timestamps: Optional timestamps for observations
            
        Returns:
            Comprehensive report with all analysis results
        """
        # Perform all analyses
        characteristics = self.characterize_regimes(observations, state_sequence)
        persistence = self.analyze_state_persistence(state_sequence)
        interpretations = self.interpret_regimes(characteristics, persistence)
        feature_importance = self.analyze_feature_importance(observations, state_sequence)
        validation = self.validate_regimes(observations, state_sequence, timestamps)
        
        # Compile report
        report = {
            'metadata': {
                'n_states': self.n_states,
                'n_observations': len(observations),
                'feature_names': self.feature_names,
                'generated_at': datetime.now().isoformat()
            },
            'regime_characteristics': {
                state_id: char.to_dict()
                for state_id, char in characteristics.items()
            },
            'state_persistence': {
                state_id: pers.to_dict()
                for state_id, pers in persistence.items()
            },
            'economic_interpretations': {
                state_id: interp.to_dict()
                for state_id, interp in interpretations.items()
            },
            'feature_importance': {
                state_id: imp.to_dict()
                for state_id, imp in feature_importance.items()
            },
            'validation': validation,
            'summary': self._generate_summary(
                characteristics, persistence, interpretations
            )
        }
        
        return report
    
    # Helper methods
    
    def _calculate_trend_strength(self, observations: np.ndarray) -> float:
        """Calculate trend strength using autocorrelation."""
        if len(observations) < 2:
            return 0.0
        
        # Use first feature (s_ldc) for trend calculation
        data = observations[:, 0]
        
        # Calculate lag-1 autocorrelation
        if len(data) > 1:
            autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
            return float(abs(autocorr)) if not np.isnan(autocorr) else 0.0
        
        return 0.0
    
    def _calculate_mean_reversion_score(self, observations: np.ndarray) -> float:
        """Calculate mean reversion score."""
        if len(observations) < 3:
            return 0.0
        
        # Use first feature (s_ldc)
        data = observations[:, 0]
        
        # Calculate Hurst exponent approximation
        # H < 0.5 indicates mean reversion
        # H > 0.5 indicates trending
        
        lags = range(2, min(20, len(data) // 2))
        tau = []
        
        for lag in lags:
            # Calculate standard deviation of differences
            diff = data[lag:] - data[:-lag]
            tau.append(np.std(diff))
        
        if len(tau) > 1:
            # Fit power law
            log_lags = np.log(list(lags))
            log_tau = np.log(tau)
            
            # Linear regression
            slope, _ = np.polyfit(log_lags, log_tau, 1)
            hurst = slope
            
            # Convert to mean reversion score (0 = trending, 1 = mean reverting)
            mean_reversion_score = max(0.0, min(1.0, 1.0 - 2 * hurst))
            return float(mean_reversion_score)
        
        return 0.5  # Neutral
    
    def _classify_regime_type(self, char: RegimeCharacteristics) -> str:
        """Classify regime type based on characteristics."""
        if char.volatility > 0.5:
            return "High Volatility"
        elif char.trend_strength > 0.6:
            return "Trending"
        elif char.mean_reversion_score > 0.6:
            return "Mean Reverting"
        else:
            return "Neutral"
    
    def _classify_market_condition(self, char: RegimeCharacteristics) -> str:
        """Classify market condition."""
        # Use average signal value to determine condition
        avg_signal = np.mean(list(char.mean_values.values()))
        
        if avg_signal > 0.3:
            return "Bull"
        elif avg_signal < -0.3:
            return "Bear"
        else:
            return "Sideways"
    
    def _assess_risk_level(self, char: RegimeCharacteristics) -> str:
        """Assess risk level."""
        if char.volatility > 0.7:
            return "High"
        elif char.volatility > 0.3:
            return "Medium"
        else:
            return "Low"
    
    def _generate_trading_recommendation(
        self,
        regime_type: str,
        market_condition: str,
        risk_level: str,
        persistence: Optional[StatePersistence]
    ) -> str:
        """Generate trading recommendation."""
        recommendations = []
        
        # Base recommendation on regime type
        if regime_type == "Trending":
            recommendations.append("Favor momentum strategies")
        elif regime_type == "Mean Reverting":
            recommendations.append("Favor mean reversion strategies")
        elif regime_type == "High Volatility":
            recommendations.append("Reduce position sizes, increase stop losses")
        
        # Add market condition guidance
        if market_condition == "Bull":
            recommendations.append("Consider long bias")
        elif market_condition == "Bear":
            recommendations.append("Consider short bias or defensive positioning")
        
        # Add risk management
        if risk_level == "High":
            recommendations.append("Implement strict risk controls")
        
        # Add persistence insight
        if persistence and persistence.mean_duration > 10:
            recommendations.append("Regime shows persistence - suitable for longer holding periods")
        elif persistence and persistence.mean_duration < 5:
            recommendations.append("Regime is transient - favor shorter timeframes")
        
        return "; ".join(recommendations) if recommendations else "Monitor market conditions"
    
    def _identify_key_characteristics(
        self,
        char: RegimeCharacteristics,
        persistence: Optional[StatePersistence]
    ) -> List[str]:
        """Identify key characteristics of regime."""
        characteristics = []
        
        # Volatility
        if char.volatility > 0.7:
            characteristics.append("Very high volatility")
        elif char.volatility > 0.5:
            characteristics.append("High volatility")
        elif char.volatility < 0.2:
            characteristics.append("Low volatility")
        
        # Trend
        if char.trend_strength > 0.7:
            characteristics.append("Strong trending behavior")
        elif char.trend_strength > 0.5:
            characteristics.append("Moderate trending")
        
        # Mean reversion
        if char.mean_reversion_score > 0.7:
            characteristics.append("Strong mean reversion")
        
        # Persistence
        if persistence:
            if persistence.mean_duration > 15:
                characteristics.append("Highly persistent regime")
            elif persistence.mean_duration < 5:
                characteristics.append("Transient regime")
        
        return characteristics
    
    def _assess_regime_stability(self, state_sequence: np.ndarray) -> float:
        """Assess overall regime stability."""
        # Calculate transition frequency
        transitions = np.sum(state_sequence[1:] != state_sequence[:-1])
        transition_rate = transitions / len(state_sequence)
        
        # Stability score (lower transition rate = higher stability)
        stability = 1.0 - min(1.0, transition_rate * 2)
        return float(stability)
    
    def _assess_state_separability(
        self,
        observations: np.ndarray,
        state_sequence: np.ndarray
    ) -> float:
        """Assess how well-separated the states are."""
        # Calculate between-state variance vs within-state variance
        overall_mean = np.mean(observations, axis=0)
        
        # Between-state variance
        state_means = []
        state_counts = []
        for state_id in range(self.n_states):
            state_mask = state_sequence == state_id
            if np.any(state_mask):
                state_mean = np.mean(observations[state_mask], axis=0)
                state_means.append(state_mean)
                state_counts.append(np.sum(state_mask))
        
        if not state_means:
            return 0.0
        
        state_means = np.array(state_means)
        state_counts = np.array(state_counts)
        
        between_var = np.sum(
            state_counts[:, np.newaxis] * (state_means - overall_mean) ** 2
        ) / len(observations)
        
        # Within-state variance
        within_var = 0.0
        for state_id in range(self.n_states):
            state_mask = state_sequence == state_id
            if np.any(state_mask):
                state_obs = observations[state_mask]
                state_mean = np.mean(state_obs, axis=0)
                within_var += np.sum((state_obs - state_mean) ** 2)
        
        within_var /= len(observations)
        
        # Separability score (higher is better)
        if within_var > 0:
            separability = between_var / (between_var + within_var)
            return float(separability)
        
        return 1.0
    
    def _assess_temporal_consistency(self, state_sequence: np.ndarray) -> float:
        """Assess temporal consistency of state assignments."""
        # Calculate entropy of state distribution
        state_counts = np.bincount(state_sequence, minlength=self.n_states)
        state_probs = state_counts / len(state_sequence)
        
        # Remove zero probabilities
        state_probs = state_probs[state_probs > 0]
        
        # Calculate entropy
        entropy = -np.sum(state_probs * np.log(state_probs))
        max_entropy = np.log(self.n_states)
        
        # Consistency score (lower entropy = higher consistency)
        if max_entropy > 0:
            consistency = 1.0 - (entropy / max_entropy)
            return float(consistency)
        
        return 1.0
    
    def _correlate_with_events(
        self,
        state_sequence: np.ndarray,
        timestamps: np.ndarray,
        known_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Correlate regime changes with known market events."""
        correlations = []
        
        # Find regime transitions
        transitions = []
        for i in range(1, len(state_sequence)):
            if state_sequence[i] != state_sequence[i-1]:
                transitions.append({
                    'timestamp': timestamps[i],
                    'from_state': int(state_sequence[i-1]),
                    'to_state': int(state_sequence[i])
                })
        
        # Match with known events (within time window)
        time_window = 5  # Allow 5 time units tolerance
        
        for event in known_events:
            event_time = event.get('timestamp')
            if event_time is None:
                continue
            
            # Find nearby transitions
            nearby_transitions = [
                t for t in transitions
                if abs(t['timestamp'] - event_time) <= time_window
            ]
            
            if nearby_transitions:
                correlations.append({
                    'event': event,
                    'transitions': nearby_transitions,
                    'correlation_strength': 'strong' if len(nearby_transitions) > 1 else 'moderate'
                })
        
        return correlations
    
    def _generate_summary(
        self,
        characteristics: Dict[int, RegimeCharacteristics],
        persistence: Dict[int, StatePersistence],
        interpretations: Dict[int, EconomicInterpretation]
    ) -> Dict[str, Any]:
        """Generate executive summary of regime analysis."""
        summary = {
            'total_states': self.n_states,
            'dominant_regime': None,
            'most_persistent_regime': None,
            'highest_risk_regime': None,
            'key_insights': []
        }
        
        # Find dominant regime (most occurrences)
        if persistence:
            dominant_state = max(
                persistence.items(),
                key=lambda x: x[1].total_occurrences
            )[0]
            summary['dominant_regime'] = {
                'state_id': dominant_state,
                'interpretation': interpretations[dominant_state].regime_type if dominant_state in interpretations else 'Unknown'
            }
        
        # Find most persistent regime
        if persistence:
            persistent_state = max(
                persistence.items(),
                key=lambda x: x[1].mean_duration
            )[0]
            summary['most_persistent_regime'] = {
                'state_id': persistent_state,
                'mean_duration': persistence[persistent_state].mean_duration
            }
        
        # Find highest risk regime
        if characteristics:
            risky_state = max(
                characteristics.items(),
                key=lambda x: x[1].volatility
            )[0]
            summary['highest_risk_regime'] = {
                'state_id': risky_state,
                'volatility': characteristics[risky_state].volatility
            }
        
        # Generate key insights
        for state_id in range(self.n_states):
            if state_id in interpretations:
                interp = interpretations[state_id]
                summary['key_insights'].append(
                    f"State {state_id}: {interp.regime_type} - {interp.trading_recommendation}"
                )
        
        return summary
