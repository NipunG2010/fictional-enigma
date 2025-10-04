"""
Tests for regime analysis and economic interpretation tools.
"""

import pytest
import numpy as np
from datetime import datetime

from imp.hmm.models import HMMArtifact
from imp.hmm.regime_analysis import (
    RegimeAnalyzer,
    RegimeCharacteristics,
    StatePersistence,
    EconomicInterpretation,
    FeatureImportance
)


@pytest.fixture
def sample_artifact():
    """Create a sample HMM artifact for testing."""
    return HMMArtifact(
        version="1.0.0",
        n_states=3,
        transition_matrix=[
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6]
        ],
        initial_probabilities=[0.33, 0.33, 0.34],
        means=[
            [0.5, 0.3, 0.2],
            [-0.2, 0.1, 0.4],
            [0.1, -0.3, 0.5]
        ],
        covariances=[
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
            [[0.15, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.15]]
        ],
        training_window_start=0,
        training_window_end=1000,
        metadata={"description": "Test HMM model"}
    )


@pytest.fixture
def sample_observations():
    """Create sample observation data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # Create observations with distinct patterns for different states
    observations = []
    
    # State 0: High positive values
    obs_state0 = np.random.randn(30, n_features) * 0.3 + np.array([0.5, 0.3, 0.2])
    observations.append(obs_state0)
    
    # State 1: Mixed values
    obs_state1 = np.random.randn(40, n_features) * 0.4 + np.array([-0.2, 0.1, 0.4])
    observations.append(obs_state1)
    
    # State 2: Different pattern
    obs_state2 = np.random.randn(30, n_features) * 0.35 + np.array([0.1, -0.3, 0.5])
    observations.append(obs_state2)
    
    return np.vstack(observations)


@pytest.fixture
def sample_state_sequence():
    """Create sample state sequence."""
    # Create a sequence with clear regime periods
    sequence = []
    
    # State 0 for 30 samples
    sequence.extend([0] * 30)
    
    # State 1 for 40 samples
    sequence.extend([1] * 40)
    
    # State 2 for 30 samples
    sequence.extend([2] * 30)
    
    return np.array(sequence)


class TestRegimeAnalyzer:
    """Test RegimeAnalyzer class."""
    
    def test_initialization(self, sample_artifact):
        """Test analyzer initialization."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        assert analyzer.n_states == 3
        assert analyzer.feature_names == ['s_ldc', 's_mr', 's_tsmom']
        assert analyzer.artifact == sample_artifact
    
    def test_characterize_regimes(self, sample_artifact, sample_observations, sample_state_sequence):
        """Test regime characterization."""
        analyzer = RegimeAnalyzer(sample_artifact)
        characteristics = analyzer.characterize_regimes(sample_observations, sample_state_sequence)
        
        # Check all states are characterized
        assert len(characteristics) == 3
        
        for state_id in range(3):
            assert state_id in characteristics
            char = characteristics[state_id]
            
            # Check basic attributes
            assert isinstance(char, RegimeCharacteristics)
            assert char.state_id == state_id
            assert char.sample_count > 0
            
            # Check statistics
            assert 's_ldc' in char.mean_values
            assert 's_mr' in char.mean_values
            assert 's_tsmom' in char.mean_values
            
            assert 's_ldc' in char.std_values
            assert char.volatility >= 0
            assert 0 <= char.trend_strength <= 1
            assert 0 <= char.mean_reversion_score <= 1
            
            # Check feature statistics
            assert 's_ldc' in char.feature_statistics
            assert 'mean' in char.feature_statistics['s_ldc']
            assert 'std' in char.feature_statistics['s_ldc']
            assert 'skewness' in char.feature_statistics['s_ldc']
    
    def test_analyze_state_persistence(self, sample_artifact, sample_state_sequence):
        """Test state persistence analysis."""
        analyzer = RegimeAnalyzer(sample_artifact)
        persistence = analyzer.analyze_state_persistence(sample_state_sequence)
        
        # Check all states have persistence stats
        assert len(persistence) == 3
        
        for state_id in range(3):
            assert state_id in persistence
            pers = persistence[state_id]
            
            assert isinstance(pers, StatePersistence)
            assert pers.state_id == state_id
            assert pers.mean_duration > 0
            assert pers.median_duration > 0
            assert pers.max_duration >= pers.min_duration
            assert pers.total_occurrences > 0
    
    def test_state_persistence_with_transitions(self, sample_artifact):
        """Test persistence analysis with multiple transitions."""
        # Create sequence with multiple transitions
        state_sequence = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 1])
        
        analyzer = RegimeAnalyzer(sample_artifact)
        persistence = analyzer.analyze_state_persistence(state_sequence)
        
        # State 0 appears twice: duration 3 and 2
        assert persistence[0].total_occurrences == 2
        assert persistence[0].mean_duration == 2.5
        
        # State 1 appears twice: duration 2 and 3
        assert persistence[1].total_occurrences == 2
        
        # State 2 appears once: duration 4
        assert persistence[2].total_occurrences == 1
        assert persistence[2].mean_duration == 4.0
    
    def test_interpret_regimes(self, sample_artifact, sample_observations, sample_state_sequence):
        """Test regime interpretation."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        characteristics = analyzer.characterize_regimes(sample_observations, sample_state_sequence)
        persistence = analyzer.analyze_state_persistence(sample_state_sequence)
        interpretations = analyzer.interpret_regimes(characteristics, persistence)
        
        # Check all states have interpretations
        assert len(interpretations) == 3
        
        for state_id in range(3):
            assert state_id in interpretations
            interp = interpretations[state_id]
            
            assert isinstance(interp, EconomicInterpretation)
            assert interp.state_id == state_id
            assert interp.regime_type in ["High Volatility", "Trending", "Mean Reverting", "Neutral"]
            assert interp.market_condition in ["Bull", "Bear", "Sideways"]
            assert interp.risk_level in ["Low", "Medium", "High"]
            assert len(interp.trading_recommendation) > 0
            assert isinstance(interp.key_characteristics, list)
    
    def test_analyze_feature_importance(self, sample_artifact, sample_observations, sample_state_sequence):
        """Test feature importance analysis."""
        analyzer = RegimeAnalyzer(sample_artifact)
        importance = analyzer.analyze_feature_importance(sample_observations, sample_state_sequence)
        
        # Check all states have importance scores
        assert len(importance) == 3
        
        for state_id in range(3):
            assert state_id in importance
            imp = importance[state_id]
            
            assert isinstance(imp, FeatureImportance)
            assert imp.state_id == state_id
            
            # Check feature scores
            assert 's_ldc' in imp.feature_scores
            assert 's_mr' in imp.feature_scores
            assert 's_tsmom' in imp.feature_scores
            
            # Scores should sum to approximately 1
            total_score = sum(imp.feature_scores.values())
            assert abs(total_score - 1.0) < 0.01
            
            # Check ranked features
            assert len(imp.ranked_features) == 3
            assert imp.ranked_features[0][1] >= imp.ranked_features[1][1]
            assert imp.ranked_features[1][1] >= imp.ranked_features[2][1]
            
            # Check dominant features
            assert isinstance(imp.dominant_features, list)
    
    def test_validate_regimes(self, sample_artifact, sample_observations, sample_state_sequence):
        """Test regime validation."""
        analyzer = RegimeAnalyzer(sample_artifact)
        validation = analyzer.validate_regimes(sample_observations, sample_state_sequence)
        
        # Check validation metrics
        assert 'regime_stability' in validation
        assert 'state_separability' in validation
        assert 'temporal_consistency' in validation
        
        # Check metric ranges
        assert 0 <= validation['regime_stability'] <= 1
        assert 0 <= validation['state_separability'] <= 1
        assert 0 <= validation['temporal_consistency'] <= 1
    
    def test_validate_regimes_with_events(self, sample_artifact, sample_observations, sample_state_sequence):
        """Test regime validation with known events."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        timestamps = np.arange(len(sample_state_sequence))
        known_events = [
            {'timestamp': 30, 'event': 'Market crash', 'type': 'volatility'},
            {'timestamp': 70, 'event': 'Policy change', 'type': 'regime_shift'}
        ]
        
        validation = analyzer.validate_regimes(
            sample_observations,
            sample_state_sequence,
            timestamps,
            known_events
        )
        
        # Check event correlations
        assert 'event_correlations' in validation
        assert isinstance(validation['event_correlations'], list)
    
    def test_generate_regime_report(self, sample_artifact, sample_observations, sample_state_sequence):
        """Test comprehensive regime report generation."""
        analyzer = RegimeAnalyzer(sample_artifact)
        report = analyzer.generate_regime_report(sample_observations, sample_state_sequence)
        
        # Check report structure
        assert 'metadata' in report
        assert 'regime_characteristics' in report
        assert 'state_persistence' in report
        assert 'economic_interpretations' in report
        assert 'feature_importance' in report
        assert 'validation' in report
        assert 'summary' in report
        
        # Check metadata
        assert report['metadata']['n_states'] == 3
        assert report['metadata']['n_observations'] == len(sample_observations)
        assert 'generated_at' in report['metadata']
        
        # Check summary
        summary = report['summary']
        assert 'total_states' in summary
        assert 'dominant_regime' in summary
        assert 'most_persistent_regime' in summary
        assert 'highest_risk_regime' in summary
        assert 'key_insights' in summary
        
        # Check that all states are included
        assert len(report['regime_characteristics']) == 3
        assert len(report['state_persistence']) == 3
        assert len(report['economic_interpretations']) == 3
        assert len(report['feature_importance']) == 3
    
    def test_regime_stability_assessment(self, sample_artifact):
        """Test regime stability assessment."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        # Stable sequence (few transitions)
        stable_sequence = np.array([0] * 50 + [1] * 50)
        stability_stable = analyzer._assess_regime_stability(stable_sequence)
        
        # Unstable sequence (many transitions)
        unstable_sequence = np.array([0, 1, 0, 1, 0, 1] * 10)
        stability_unstable = analyzer._assess_regime_stability(unstable_sequence)
        
        # Stable should have higher score
        assert stability_stable > stability_unstable
        assert 0 <= stability_stable <= 1
        assert 0 <= stability_unstable <= 1
    
    def test_state_separability_assessment(self, sample_artifact, sample_observations, sample_state_sequence):
        """Test state separability assessment."""
        analyzer = RegimeAnalyzer(sample_artifact)
        separability = analyzer._assess_state_separability(sample_observations, sample_state_sequence)
        
        assert 0 <= separability <= 1
        # With distinct state patterns, separability should be reasonable
        assert separability > 0.1
    
    def test_temporal_consistency_assessment(self, sample_artifact):
        """Test temporal consistency assessment."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        # Balanced distribution
        balanced_sequence = np.array([0] * 30 + [1] * 30 + [2] * 30)
        consistency_balanced = analyzer._assess_temporal_consistency(balanced_sequence)
        
        # Imbalanced distribution
        imbalanced_sequence = np.array([0] * 80 + [1] * 10 + [2] * 10)
        consistency_imbalanced = analyzer._assess_temporal_consistency(imbalanced_sequence)
        
        # Both should be valid
        assert 0 <= consistency_balanced <= 1
        assert 0 <= consistency_imbalanced <= 1
    
    def test_to_dict_methods(self, sample_artifact, sample_observations, sample_state_sequence):
        """Test to_dict conversion methods."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        # Test RegimeCharacteristics.to_dict()
        characteristics = analyzer.characterize_regimes(sample_observations, sample_state_sequence)
        char_dict = characteristics[0].to_dict()
        assert isinstance(char_dict, dict)
        assert 'state_id' in char_dict
        assert 'volatility' in char_dict
        
        # Test StatePersistence.to_dict()
        persistence = analyzer.analyze_state_persistence(sample_state_sequence)
        pers_dict = persistence[0].to_dict()
        assert isinstance(pers_dict, dict)
        assert 'state_id' in pers_dict
        assert 'mean_duration' in pers_dict
        
        # Test EconomicInterpretation.to_dict()
        interpretations = analyzer.interpret_regimes(characteristics, persistence)
        interp_dict = interpretations[0].to_dict()
        assert isinstance(interp_dict, dict)
        assert 'state_id' in interp_dict
        assert 'regime_type' in interp_dict
        
        # Test FeatureImportance.to_dict()
        importance = analyzer.analyze_feature_importance(sample_observations, sample_state_sequence)
        imp_dict = importance[0].to_dict()
        assert isinstance(imp_dict, dict)
        assert 'state_id' in imp_dict
        assert 'feature_scores' in imp_dict


class TestRegimeClassification:
    """Test regime classification logic."""
    
    def test_classify_regime_type(self, sample_artifact):
        """Test regime type classification."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        # High volatility regime
        high_vol_char = RegimeCharacteristics(
            state_id=0,
            mean_values={'s_ldc': 0.5, 's_mr': 0.3, 's_tsmom': 0.2},
            std_values={'s_ldc': 0.6, 's_mr': 0.5, 's_tsmom': 0.4},
            volatility=0.8,
            trend_strength=0.3,
            mean_reversion_score=0.2,
            sample_count=100
        )
        regime_type = analyzer._classify_regime_type(high_vol_char)
        assert regime_type == "High Volatility"
        
        # Trending regime
        trending_char = RegimeCharacteristics(
            state_id=1,
            mean_values={'s_ldc': 0.5, 's_mr': 0.3, 's_tsmom': 0.2},
            std_values={'s_ldc': 0.2, 's_mr': 0.2, 's_tsmom': 0.2},
            volatility=0.2,
            trend_strength=0.8,
            mean_reversion_score=0.2,
            sample_count=100
        )
        regime_type = analyzer._classify_regime_type(trending_char)
        assert regime_type == "Trending"
        
        # Mean reverting regime
        mr_char = RegimeCharacteristics(
            state_id=2,
            mean_values={'s_ldc': 0.5, 's_mr': 0.3, 's_tsmom': 0.2},
            std_values={'s_ldc': 0.2, 's_mr': 0.2, 's_tsmom': 0.2},
            volatility=0.2,
            trend_strength=0.3,
            mean_reversion_score=0.8,
            sample_count=100
        )
        regime_type = analyzer._classify_regime_type(mr_char)
        assert regime_type == "Mean Reverting"
    
    def test_classify_market_condition(self, sample_artifact):
        """Test market condition classification."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        # Bull market
        bull_char = RegimeCharacteristics(
            state_id=0,
            mean_values={'s_ldc': 0.5, 's_mr': 0.4, 's_tsmom': 0.3},
            std_values={'s_ldc': 0.2, 's_mr': 0.2, 's_tsmom': 0.2},
            volatility=0.2,
            trend_strength=0.5,
            mean_reversion_score=0.5,
            sample_count=100
        )
        condition = analyzer._classify_market_condition(bull_char)
        assert condition == "Bull"
        
        # Bear market
        bear_char = RegimeCharacteristics(
            state_id=1,
            mean_values={'s_ldc': -0.5, 's_mr': -0.4, 's_tsmom': -0.3},
            std_values={'s_ldc': 0.2, 's_mr': 0.2, 's_tsmom': 0.2},
            volatility=0.2,
            trend_strength=0.5,
            mean_reversion_score=0.5,
            sample_count=100
        )
        condition = analyzer._classify_market_condition(bear_char)
        assert condition == "Bear"
    
    def test_assess_risk_level(self, sample_artifact):
        """Test risk level assessment."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        # High risk
        high_risk_char = RegimeCharacteristics(
            state_id=0,
            mean_values={'s_ldc': 0.5, 's_mr': 0.3, 's_tsmom': 0.2},
            std_values={'s_ldc': 0.8, 's_mr': 0.7, 's_tsmom': 0.6},
            volatility=0.8,
            trend_strength=0.5,
            mean_reversion_score=0.5,
            sample_count=100
        )
        risk = analyzer._assess_risk_level(high_risk_char)
        assert risk == "High"
        
        # Low risk
        low_risk_char = RegimeCharacteristics(
            state_id=1,
            mean_values={'s_ldc': 0.5, 's_mr': 0.3, 's_tsmom': 0.2},
            std_values={'s_ldc': 0.1, 's_mr': 0.1, 's_tsmom': 0.1},
            volatility=0.1,
            trend_strength=0.5,
            mean_reversion_score=0.5,
            sample_count=100
        )
        risk = analyzer._assess_risk_level(low_risk_char)
        assert risk == "Low"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_state_observations(self, sample_artifact):
        """Test handling of states with no observations."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        # Create observations and sequence where state 2 never appears
        observations = np.random.randn(50, 3)
        state_sequence = np.array([0] * 25 + [1] * 25)
        
        characteristics = analyzer.characterize_regimes(observations, state_sequence)
        
        # Only states 0 and 1 should be characterized
        assert len(characteristics) == 2
        assert 0 in characteristics
        assert 1 in characteristics
        assert 2 not in characteristics
    
    def test_single_state_sequence(self, sample_artifact):
        """Test with sequence containing only one state."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        observations = np.random.randn(50, 3)
        state_sequence = np.array([0] * 50)
        
        persistence = analyzer.analyze_state_persistence(state_sequence)
        
        # Only state 0 should have persistence stats
        assert len(persistence) == 1
        assert 0 in persistence
        assert persistence[0].total_occurrences == 1
        assert persistence[0].mean_duration == 50
    
    def test_very_short_sequence(self, sample_artifact):
        """Test with very short observation sequence."""
        analyzer = RegimeAnalyzer(sample_artifact)
        
        observations = np.random.randn(3, 3)
        state_sequence = np.array([0, 1, 2])
        
        # Should not crash
        characteristics = analyzer.characterize_regimes(observations, state_sequence)
        persistence = analyzer.analyze_state_persistence(state_sequence)
        
        assert len(characteristics) == 3
        assert len(persistence) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
