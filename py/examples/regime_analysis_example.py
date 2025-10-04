"""
Example demonstrating advanced regime analysis and economic interpretation.

This example shows how to:
1. Characterize market regimes detected by HMM
2. Analyze state persistence and transitions
3. Generate economic interpretations
4. Analyze feature importance
5. Validate regimes and generate comprehensive reports
"""

import numpy as np
from pathlib import Path
import json

from imp.hmm.models import HMMArtifact
from imp.hmm.trainer import HMMTrainer
from imp.hmm.regime_analysis import RegimeAnalyzer


def generate_synthetic_market_data(n_samples: int = 500) -> np.ndarray:
    """
    Generate synthetic market data with distinct regime patterns.
    
    Returns:
        Array of shape (n_samples, 3) with [s_ldc, s_mr, s_tsmom]
    """
    np.random.seed(42)
    observations = []
    
    # Regime 1: Bull market - high trending, low volatility
    n_bull = n_samples // 3
    bull_data = np.random.randn(n_bull, 3) * 0.2 + np.array([0.6, 0.4, 0.5])
    observations.append(bull_data)
    
    # Regime 2: High volatility - large swings
    n_volatile = n_samples // 3
    volatile_data = np.random.randn(n_volatile, 3) * 0.8 + np.array([0.0, 0.0, 0.0])
    observations.append(volatile_data)
    
    # Regime 3: Mean reverting - oscillating around mean
    n_mr = n_samples - n_bull - n_volatile
    mr_data = np.random.randn(n_mr, 3) * 0.3 + np.array([-0.2, 0.3, 0.1])
    # Add mean reverting pattern
    for i in range(1, len(mr_data)):
        mr_data[i] = mr_data[i] * 0.7 - mr_data[i-1] * 0.3
    observations.append(mr_data)
    
    return np.vstack(observations)


def main():
    """Run regime analysis example."""
    print("=" * 80)
    print("Advanced Regime Analysis Example")
    print("=" * 80)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic market data...")
    observations = generate_synthetic_market_data(n_samples=500)
    print(f"   Generated {len(observations)} observations with 3 features")
    print(f"   Feature means: {np.mean(observations, axis=0)}")
    print(f"   Feature stds: {np.std(observations, axis=0)}")
    
    # Step 2: Train HMM model
    print("\n2. Training HMM model...")
    trainer = HMMTrainer(n_states=3)
    artifact = trainer.train(observations, n_iterations=100)
    print(f"   Trained {artifact.n_states}-state HMM model")
    print(f"   Training window: {artifact.training_window_start} - {artifact.training_window_end}")
    
    # Step 3: Decode state sequence
    print("\n3. Decoding state sequence...")
    from hmmlearn import hmm
    
    model = hmm.GaussianHMM(n_components=artifact.n_states, covariance_type='full')
    model.startprob_ = np.array(artifact.initial_probabilities)
    model.transmat_ = np.array(artifact.transition_matrix)
    model.means_ = np.array(artifact.means)
    model.covars_ = np.array(artifact.covariances)
    
    state_sequence = model.predict(observations)
    print(f"   Decoded {len(state_sequence)} states")
    print(f"   State distribution: {np.bincount(state_sequence)}")
    
    # Step 4: Initialize regime analyzer
    print("\n4. Initializing regime analyzer...")
    analyzer = RegimeAnalyzer(artifact)
    print(f"   Analyzer ready for {analyzer.n_states} states")
    
    # Step 5: Characterize regimes
    print("\n5. Characterizing market regimes...")
    characteristics = analyzer.characterize_regimes(observations, state_sequence)
    
    for state_id, char in characteristics.items():
        print(f"\n   State {state_id}:")
        print(f"      Sample count: {char.sample_count}")
        print(f"      Mean values: {char.mean_values}")
        print(f"      Volatility: {char.volatility:.4f}")
        print(f"      Trend strength: {char.trend_strength:.4f}")
        print(f"      Mean reversion score: {char.mean_reversion_score:.4f}")
    
    # Step 6: Analyze state persistence
    print("\n6. Analyzing state persistence...")
    persistence = analyzer.analyze_state_persistence(state_sequence, min_stable_duration=10)
    
    for state_id, pers in persistence.items():
        print(f"\n   State {state_id}:")
        print(f"      Mean duration: {pers.mean_duration:.2f}")
        print(f"      Median duration: {pers.median_duration:.2f}")
        print(f"      Max duration: {pers.max_duration}")
        print(f"      Total occurrences: {pers.total_occurrences}")
        print(f"      Stable periods (>= 10): {pers.stable_periods}")
        print(f"      Transition probabilities: {pers.transition_probabilities}")
    
    # Step 7: Generate economic interpretations
    print("\n7. Generating economic interpretations...")
    interpretations = analyzer.interpret_regimes(characteristics, persistence)
    
    for state_id, interp in interpretations.items():
        print(f"\n   State {state_id}:")
        print(f"      Regime type: {interp.regime_type}")
        print(f"      Market condition: {interp.market_condition}")
        print(f"      Risk level: {interp.risk_level}")
        print(f"      Trading recommendation: {interp.trading_recommendation}")
        print(f"      Key characteristics: {', '.join(interp.key_characteristics)}")
    
    # Step 8: Analyze feature importance
    print("\n8. Analyzing feature importance...")
    feature_importance = analyzer.analyze_feature_importance(observations, state_sequence)
    
    for state_id, imp in feature_importance.items():
        print(f"\n   State {state_id}:")
        print(f"      Feature scores: {imp.feature_scores}")
        print(f"      Ranked features: {imp.ranked_features}")
        print(f"      Dominant features: {imp.dominant_features}")
    
    # Step 9: Validate regimes
    print("\n9. Validating regime detection...")
    validation = analyzer.validate_regimes(observations, state_sequence)
    
    print(f"   Regime stability: {validation['regime_stability']:.4f}")
    print(f"   State separability: {validation['state_separability']:.4f}")
    print(f"   Temporal consistency: {validation['temporal_consistency']:.4f}")
    
    # Step 10: Generate comprehensive report
    print("\n10. Generating comprehensive regime report...")
    report = analyzer.generate_regime_report(observations, state_sequence)
    
    print("\n   Report Summary:")
    summary = report['summary']
    print(f"      Total states: {summary['total_states']}")
    
    if summary['dominant_regime']:
        print(f"      Dominant regime: State {summary['dominant_regime']['state_id']} "
              f"({summary['dominant_regime']['interpretation']})")
    
    if summary['most_persistent_regime']:
        print(f"      Most persistent: State {summary['most_persistent_regime']['state_id']} "
              f"(mean duration: {summary['most_persistent_regime']['mean_duration']:.2f})")
    
    if summary['highest_risk_regime']:
        print(f"      Highest risk: State {summary['highest_risk_regime']['state_id']} "
              f"(volatility: {summary['highest_risk_regime']['volatility']:.4f})")
    
    print("\n   Key Insights:")
    for insight in summary['key_insights']:
        print(f"      - {insight}")
    
    # Step 11: Save report to file
    print("\n11. Saving report to file...")
    output_dir = Path("py/processed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    report_serializable = convert_numpy_types(report)
    
    report_path = output_dir / "regime_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report_serializable, f, indent=2, default=str)
    
    print(f"   Report saved to: {report_path}")
    
    # Step 12: Demonstrate regime validation with events
    print("\n12. Demonstrating event correlation...")
    timestamps = np.arange(len(observations))
    known_events = [
        {'timestamp': 150, 'event': 'Market volatility spike', 'type': 'volatility'},
        {'timestamp': 300, 'event': 'Trend reversal', 'type': 'regime_shift'},
        {'timestamp': 450, 'event': 'Policy announcement', 'type': 'external'}
    ]
    
    validation_with_events = analyzer.validate_regimes(
        observations,
        state_sequence,
        timestamps,
        known_events
    )
    
    if validation_with_events.get('event_correlations'):
        print(f"   Found {len(validation_with_events['event_correlations'])} event correlations")
        for corr in validation_with_events['event_correlations']:
            event = corr['event']
            print(f"      - {event['event']} at t={event['timestamp']}: "
                  f"{corr['correlation_strength']} correlation")
    else:
        print("   No significant event correlations found")
    
    print("\n" + "=" * 80)
    print("Regime Analysis Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Regime characterization provides statistical insights into market states")
    print("2. Persistence analysis reveals regime stability and transition patterns")
    print("3. Economic interpretation translates statistics into actionable insights")
    print("4. Feature importance identifies key drivers of each regime")
    print("5. Validation ensures regime detection quality and reliability")
    print("\nNext Steps:")
    print("- Use regime insights to optimize trading strategies")
    print("- Adjust position sizing based on risk levels")
    print("- Align strategy selection with detected market conditions")
    print("- Monitor regime transitions for early warning signals")


if __name__ == "__main__":
    main()
