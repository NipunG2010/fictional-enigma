#!/usr/bin/env python3
"""
Demonstration of the RegimeVisualizer functionality.

This script shows how to use the RegimeVisualizer class to create
various plots and analyses for HMM regime detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from imp.hmm.models import HMMArtifact
from imp.visualization.regime_visualizer import RegimeVisualizer


def create_sample_hmm_artifact():
    """Create a sample HMM artifact for demonstration."""
    return HMMArtifact(
        version="1.0.0",
        n_states=3,
        transition_matrix=[
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.1, 0.3, 0.6]
        ],
        initial_probabilities=[0.33, 0.33, 0.34],
        means=[
            [0.1, 0.2],
            [0.5, 0.6],
            [-0.2, -0.1]
        ],
        covariances=[
            [[0.1, 0.05], [0.05, 0.1]],
            [[0.2, 0.1], [0.1, 0.2]],
            [[0.15, 0.08], [0.08, 0.15]]
        ],
        training_window_start=0,
        training_window_end=1000,
        metadata={"demo": True}
    )


def generate_sample_data(n_timesteps=200, n_features=2, n_states=3):
    """Generate sample market data and state probabilities."""
    np.random.seed(42)
    
    # Generate sample observations (simulating market signals)
    observations = np.random.randn(n_timesteps, n_features)
    
    # Add some regime-like behavior
    regime_changes = [0, 50, 120, 180]  # Approximate regime change points
    for i in range(len(regime_changes) - 1):
        start, end = regime_changes[i], regime_changes[i + 1]
        # Add different mean and volatility for each regime
        observations[start:end, 0] += i * 0.5  # Different means
        observations[start:end, 1] *= (1 + i * 0.3)  # Different volatilities
    
    # Generate state probabilities with some persistence
    state_probs = np.zeros((n_timesteps, n_states))
    current_state = 0
    
    for t in range(n_timesteps):
        # Add some regime persistence
        if t in regime_changes[1:]:
            current_state = (current_state + 1) % n_states
        
        # Generate probabilities with noise
        probs = np.random.dirichlet([3, 1, 1])  # Favor current state
        if current_state == 1:
            probs = np.random.dirichlet([1, 3, 1])
        elif current_state == 2:
            probs = np.random.dirichlet([1, 1, 3])
        
        state_probs[t] = probs
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = np.array([
        (start_date + timedelta(hours=i)).timestamp() 
        for i in range(n_timesteps)
    ])
    
    return observations, state_probs, timestamps


def main():
    """Run the visualization demonstration."""
    print("HMM Regime Visualization Demo")
    print("=" * 40)
    
    # Create sample data
    artifact = create_sample_hmm_artifact()
    observations, state_probs, timestamps = generate_sample_data()
    
    # Initialize visualizer
    visualizer = RegimeVisualizer(artifact)
    
    print(f"Created visualizer for {artifact.n_states} states")
    print(f"Generated {len(observations)} observations with {observations.shape[1]} features")
    
    # 1. Plot state probabilities (static)
    print("\n1. Creating static state probability plot...")
    fig1 = visualizer.plot_state_probabilities(
        state_probs, timestamps, interactive=False,
        title="Market Regime Probabilities Over Time"
    )
    plt.savefig('regime_probabilities_static.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Plot transition matrix
    print("\n2. Creating transition matrix heatmap...")
    fig2 = visualizer.plot_transition_matrix(
        title="HMM State Transition Matrix"
    )
    plt.savefig('transition_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Calculate and display regime statistics
    print("\n3. Calculating regime statistics...")
    stats = visualizer.calculate_regime_statistics(observations, state_probs, timestamps)
    
    print(f"Total observations: {stats['total_observations']}")
    print(f"Number of states: {stats['n_states']}")
    
    print("\nState frequencies:")
    for state in range(stats['n_states']):
        state_key = f'state_{state}'
        if state_key in stats['state_statistics']:
            freq = stats['state_statistics'][state_key]['frequency']
            print(f"  State {state}: {freq:.3f}")
    
    print("\nRegime persistence:")
    for state in range(stats['n_states']):
        state_key = f'state_{state}'
        if state_key in stats['regime_persistence']:
            persistence = stats['regime_persistence'][state_key]
            print(f"  State {state}: mean duration = {persistence['mean_duration']:.1f}, "
                  f"episodes = {persistence['total_episodes']}")
    
    # 4. Format statistics for display
    print("\n4. Formatted statistics:")
    formatted_stats = visualizer.format_regime_statistics(stats)
    # Note: This would normally be displayed in a Jupyter notebook
    print("(HTML formatted statistics created - would display in Jupyter)")
    
    # 5. Create regime comparison plot
    print("\n5. Creating regime comparison plot...")
    
    # Generate second set of state probabilities for comparison
    np.random.seed(123)
    state_probs_2 = np.random.dirichlet([1, 1, 1], size=len(state_probs))
    
    fig3 = visualizer.plot_regime_comparison(
        observations[:, 0],  # Use first feature only
        [state_probs, state_probs_2],
        ['HMM Model 1', 'HMM Model 2'],
        timestamps
    )
    plt.savefig('regime_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n6. Testing interactive features...")
    try:
        # Try to create interactive plot (will fall back to static if plotly not available)
        fig4 = visualizer.plot_state_probabilities(
            state_probs, timestamps, interactive=True,
            title="Interactive Market Regime Probabilities"
        )
        if hasattr(fig4, 'show'):
            print("Interactive plot created successfully!")
            # fig4.show()  # Uncomment to display in browser
        else:
            print("Plotly not available, created static plot instead")
            plt.show()
    except Exception as e:
        print(f"Interactive plotting failed: {e}")
    
    print("\n7. Testing dashboard creation...")
    try:
        dashboard = visualizer.create_regime_dashboard(
            observations, state_probs, timestamps,
            title="HMM Regime Analysis Dashboard"
        )
        if isinstance(dashboard, str):
            print("Dashboard creation note:", dashboard)
        else:
            print("Interactive dashboard created successfully!")
            print("(Dashboard would be displayed in Jupyter notebook)")
    except Exception as e:
        print(f"Dashboard creation failed: {e}")
    
    print("\nDemo completed successfully!")
    print("Generated files:")
    print("  - regime_probabilities_static.png")
    print("  - transition_matrix.png") 
    print("  - regime_comparison.png")


if __name__ == "__main__":
    main()