"""
Advanced visualization tools for HMM regime analysis.
"""

from typing import Union, Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Core plotting libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Interactive plotting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Jupyter widgets
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

from ..hmm.models import HMMArtifact


class RegimeVisualizer:
    """Advanced visualization tools for HMM regime analysis."""
    
    def __init__(self, artifact: HMMArtifact):
        """
        Initialize the RegimeVisualizer.
        
        Args:
            artifact: HMMArtifact containing the trained HMM model
        """
        self.artifact = artifact
        self.n_states = artifact.n_states
        self.transition_matrix = np.array(artifact.transition_matrix)
        self.initial_probabilities = np.array(artifact.initial_probabilities)
        self.means = np.array(artifact.means)
        self.covariances = np.array(artifact.covariances)
        
        # Set up color palette for states
        self.state_colors = self._generate_state_colors()
        
    def _generate_state_colors(self) -> List[str]:
        """Generate distinct colors for each state."""
        if self.n_states <= 10:
            # Use seaborn's tab10 palette for up to 10 states
            colors = sns.color_palette("tab10", n_colors=self.n_states)
            return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                   for r, g, b in colors]
        else:
            # Use husl palette for more states
            colors = sns.color_palette("husl", n_colors=self.n_states)
            return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                   for r, g, b in colors]
    
    def plot_state_probabilities(self, 
                               state_probs: np.ndarray,
                               timestamps: Optional[np.ndarray] = None,
                               interactive: bool = True,
                               figsize: Tuple[int, int] = (12, 6),
                               title: str = "HMM State Probabilities Over Time") -> Union[plt.Figure, go.Figure]:
        """
        Plot state probabilities over time.
        
        Args:
            state_probs: Array of shape (n_timesteps, n_states) with state probabilities
            timestamps: Optional array of timestamps or indices
            interactive: Whether to create interactive plotly plot or static matplotlib
            figsize: Figure size for matplotlib plots
            title: Plot title
            
        Returns:
            matplotlib Figure or plotly Figure depending on interactive parameter
        """
        if state_probs.shape[1] != self.n_states:
            raise ValueError(f"State probabilities must have {self.n_states} columns, got {state_probs.shape[1]}")
        
        if timestamps is None:
            timestamps = np.arange(len(state_probs))
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_interactive_states(state_probs, timestamps, title)
        else:
            return self._plot_static_states(state_probs, timestamps, figsize, title)
    
    def _plot_static_states(self, 
                          state_probs: np.ndarray, 
                          timestamps: np.ndarray,
                          figsize: Tuple[int, int],
                          title: str) -> plt.Figure:
        """Create static matplotlib plot of state probabilities."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert timestamps to appropriate format for plotting
        x_values = timestamps
        x_label = 'Time'
        
        # Check if timestamps look like Unix timestamps and convert to dates
        if len(timestamps) > 0 and timestamps[0] > 1e9:  # Unix timestamp
            try:
                # Convert to datetime objects for better plotting
                dates = [datetime.fromtimestamp(ts) for ts in timestamps]
                x_values = dates
                x_label = 'Date'
            except (ValueError, OverflowError, OSError):
                # If conversion fails, use original timestamps
                pass
        
        # Create stacked area plot
        ax.stackplot(x_values, *[state_probs[:, i] for i in range(self.n_states)],
                    labels=[f'State {i}' for i in range(self.n_states)],
                    colors=self.state_colors,
                    alpha=0.7)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('State Probability')
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        if isinstance(x_values[0], datetime):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(x_values)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _plot_interactive_states(self, 
                               state_probs: np.ndarray, 
                               timestamps: np.ndarray,
                               title: str) -> go.Figure:
        """Create interactive plotly plot of state probabilities."""
        fig = go.Figure()
        
        # Add traces for each state
        for i in range(self.n_states):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=state_probs[:, i],
                mode='lines',
                name=f'State {i}',
                line=dict(color=self.state_colors[i]),
                fill='tonexty' if i > 0 else 'tozeroy',
                stackgroup='one'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='State Probability',
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_transition_matrix(self, 
                             annotate: bool = True,
                             cmap: str = "Blues",
                             figsize: Tuple[int, int] = (8, 6),
                             title: str = "HMM State Transition Matrix") -> plt.Figure:
        """
        Visualize transition matrix as heatmap.
        
        Args:
            annotate: Whether to annotate cells with probability values
            cmap: Colormap for the heatmap
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(self.transition_matrix, 
                   annot=annotate,
                   fmt='.3f',
                   cmap=cmap,
                   square=True,
                   ax=ax,
                   cbar_kws={'label': 'Transition Probability'},
                   xticklabels=[f'State {i}' for i in range(self.n_states)],
                   yticklabels=[f'State {i}' for i in range(self.n_states)])
        
        ax.set_title(title)
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        
        plt.tight_layout()
        return fig
    
    def calculate_regime_statistics(self, 
                                  observations: np.ndarray,
                                  state_probs: np.ndarray,
                                  timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive regime statistics.
        
        Args:
            observations: Market observations used for training
            state_probs: State probabilities over time
            timestamps: Optional timestamps for duration calculations
            
        Returns:
            Dictionary containing regime statistics
        """
        if len(observations) != len(state_probs):
            raise ValueError("Observations and state probabilities must have same length")
        
        # Decode most likely state sequence
        most_likely_states = np.argmax(state_probs, axis=1)
        
        stats = {
            'n_states': self.n_states,
            'total_observations': len(observations),
            'state_statistics': {},
            'transition_statistics': {},
            'regime_persistence': {}
        }
        
        # Calculate per-state statistics
        for state in range(self.n_states):
            state_mask = most_likely_states == state
            state_obs = observations[state_mask]
            
            if len(state_obs) > 0:
                state_stats = {
                    'frequency': np.sum(state_mask) / len(observations),
                    'mean_probability': np.mean(state_probs[state_mask, state]),
                    'observation_count': len(state_obs)
                }
                
                # Calculate observation statistics if multivariate
                if observations.ndim > 1:
                    state_stats['mean_values'] = np.mean(state_obs, axis=0).tolist()
                    state_stats['std_values'] = np.std(state_obs, axis=0).tolist()
                    state_stats['correlation_matrix'] = np.corrcoef(state_obs.T).tolist()
                else:
                    state_stats['mean_value'] = np.mean(state_obs)
                    state_stats['std_value'] = np.std(state_obs)
                
                stats['state_statistics'][f'state_{state}'] = state_stats
        
        # Calculate transition statistics
        transitions = np.zeros((self.n_states, self.n_states))
        for i in range(len(most_likely_states) - 1):
            from_state = most_likely_states[i]
            to_state = most_likely_states[i + 1]
            transitions[from_state, to_state] += 1
        
        # Normalize to get empirical transition probabilities
        empirical_transitions = transitions / (transitions.sum(axis=1, keepdims=True) + 1e-8)
        stats['transition_statistics']['empirical_matrix'] = empirical_transitions.tolist()
        stats['transition_statistics']['theoretical_matrix'] = self.transition_matrix.tolist()
        
        # Calculate regime persistence
        stats['regime_persistence'] = self._calculate_persistence_metrics(
            most_likely_states, timestamps)
        
        return stats
    
    def _calculate_persistence_metrics(self, 
                                     states: np.ndarray,
                                     timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate regime persistence and duration statistics."""
        persistence = {}
        
        # Calculate state durations
        state_durations = {i: [] for i in range(self.n_states)}
        current_state = states[0]
        current_duration = 1
        
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_duration += 1
            else:
                state_durations[current_state].append(current_duration)
                current_state = states[i]
                current_duration = 1
        
        # Add final duration
        state_durations[current_state].append(current_duration)
        
        # Calculate statistics for each state
        for state in range(self.n_states):
            durations = state_durations[state]
            if durations:
                persistence[f'state_{state}'] = {
                    'mean_duration': float(np.mean(durations)),
                    'median_duration': float(np.median(durations)),
                    'max_duration': int(np.max(durations)),
                    'min_duration': int(np.min(durations)),
                    'total_episodes': len(durations),
                    'duration_std': float(np.std(durations))
                }
                
                # Convert to time units if timestamps provided
                if timestamps is not None and len(timestamps) > 1:
                    time_unit = timestamps[1] - timestamps[0]  # Assume regular intervals
                    persistence[f'state_{state}']['mean_duration_time'] = persistence[f'state_{state}']['mean_duration'] * time_unit
                    persistence[f'state_{state}']['median_duration_time'] = persistence[f'state_{state}']['median_duration'] * time_unit
        
        return persistence
    
    def format_regime_statistics(self, stats: Dict[str, Any]) -> str:
        """
        Format regime statistics for display.
        
        Args:
            stats: Statistics dictionary from calculate_regime_statistics
            
        Returns:
            Formatted string representation of statistics
        """
        output = []
        output.append(f"<h4>Regime Analysis Summary</h4>")
        output.append(f"<p><strong>Total Observations:</strong> {stats['total_observations']}</p>")
        output.append(f"<p><strong>Number of States:</strong> {stats['n_states']}</p>")
        
        output.append("<h5>State Frequencies</h5>")
        output.append("<table style='border-collapse: collapse; width: 100%;'>")
        output.append("<tr style='border: 1px solid #ddd;'><th style='padding: 8px; border: 1px solid #ddd;'>State</th><th style='padding: 8px; border: 1px solid #ddd;'>Frequency</th><th style='padding: 8px; border: 1px solid #ddd;'>Mean Duration</th></tr>")
        
        for state in range(stats['n_states']):
            state_key = f'state_{state}'
            if state_key in stats['state_statistics']:
                freq = stats['state_statistics'][state_key]['frequency']
                if state_key in stats['regime_persistence']:
                    mean_dur = stats['regime_persistence'][state_key]['mean_duration']
                    output.append(f"<tr style='border: 1px solid #ddd;'><td style='padding: 8px; border: 1px solid #ddd;'>State {state}</td><td style='padding: 8px; border: 1px solid #ddd;'>{freq:.3f}</td><td style='padding: 8px; border: 1px solid #ddd;'>{mean_dur:.1f}</td></tr>")
                else:
                    output.append(f"<tr style='border: 1px solid #ddd;'><td style='padding: 8px; border: 1px solid #ddd;'>State {state}</td><td style='padding: 8px; border: 1px solid #ddd;'>{freq:.3f}</td><td style='padding: 8px; border: 1px solid #ddd;'>N/A</td></tr>")
        
        output.append("</table>")
        
        return "".join(output)
    
    def create_regime_dashboard(self, 
                              observations: np.ndarray,
                              state_probs: np.ndarray,
                              timestamps: Optional[np.ndarray] = None,
                              title: str = "HMM Regime Analysis Dashboard") -> Union[widgets.VBox, str]:
        """
        Create interactive dashboard for regime analysis.
        
        Args:
            observations: Market observations
            state_probs: State probabilities over time
            timestamps: Optional timestamps
            title: Dashboard title
            
        Returns:
            IPython widget VBox or error message string
        """
        if not WIDGETS_AVAILABLE:
            return "IPython widgets not available. Please install ipywidgets to use the dashboard."
        
        # Calculate regime statistics
        regime_stats = self.calculate_regime_statistics(observations, state_probs, timestamps)
        
        # Create output widget for plots
        plot_output = widgets.Output()
        stats_output = widgets.Output()
        
        # Create control widgets
        plot_type = widgets.Dropdown(
            options=[
                ('State Probabilities', 'probabilities'),
                ('Transition Matrix', 'transitions'),
                ('State Statistics', 'statistics')
            ],
            value='probabilities',
            description='Plot Type:'
        )
        
        interactive_toggle = widgets.Checkbox(
            value=True,
            description='Interactive Plot',
            disabled=not PLOTLY_AVAILABLE
        )
        
        state_selector = widgets.Dropdown(
            options=[(f'State {i}', i) for i in range(self.n_states)] + [('All States', -1)],
            value=-1,
            description='Focus State:'
        )
        
        def update_display(change=None):
            """Update the display based on widget values."""
            with plot_output:
                clear_output(wait=True)
                
                if plot_type.value == 'probabilities':
                    fig = self.plot_state_probabilities(
                        state_probs, 
                        timestamps, 
                        interactive=interactive_toggle.value
                    )
                    if interactive_toggle.value and PLOTLY_AVAILABLE:
                        fig.show()
                    else:
                        plt.show()
                        
                elif plot_type.value == 'transitions':
                    fig = self.plot_transition_matrix()
                    plt.show()
                    
                elif plot_type.value == 'statistics':
                    # Create state-specific statistics plot
                    self._plot_state_statistics(observations, state_probs, state_selector.value)
                    plt.show()
            
            with stats_output:
                clear_output(wait=True)
                formatted_stats = self.format_regime_statistics(regime_stats)
                display(widgets.HTML(formatted_stats))
        
        # Connect widget events
        plot_type.observe(update_display, names='value')
        interactive_toggle.observe(update_display, names='value')
        state_selector.observe(update_display, names='value')
        
        # Initial display
        update_display()
        
        # Create dashboard layout
        controls = widgets.HBox([plot_type, interactive_toggle, state_selector])
        
        dashboard = widgets.VBox([
            widgets.HTML(f"<h3>{title}</h3>"),
            controls,
            widgets.HBox([plot_output, stats_output])
        ])
        
        return dashboard
    
    def _plot_state_statistics(self, 
                             observations: np.ndarray,
                             state_probs: np.ndarray,
                             focus_state: int = -1):
        """Plot state-specific statistics."""
        most_likely_states = np.argmax(state_probs, axis=1)
        
        if focus_state == -1:
            # Plot all states
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            # State frequency bar plot
            state_counts = np.bincount(most_likely_states, minlength=self.n_states)
            axes[0].bar(range(self.n_states), state_counts, color=self.state_colors)
            axes[0].set_title('State Frequencies')
            axes[0].set_xlabel('State')
            axes[0].set_ylabel('Count')
            
            # State probability distributions
            for state in range(self.n_states):
                state_mask = most_likely_states == state
                if np.any(state_mask):
                    axes[1].hist(state_probs[state_mask, state], 
                               alpha=0.7, label=f'State {state}', 
                               color=self.state_colors[state], bins=20)
            axes[1].set_title('State Probability Distributions')
            axes[1].set_xlabel('Probability')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            
            # Observation statistics by state (if univariate)
            if observations.ndim == 1:
                for state in range(self.n_states):
                    state_mask = most_likely_states == state
                    if np.any(state_mask):
                        axes[2].hist(observations[state_mask], 
                                   alpha=0.7, label=f'State {state}',
                                   color=self.state_colors[state], bins=20)
                axes[2].set_title('Observation Distributions by State')
                axes[2].set_xlabel('Observation Value')
                axes[2].set_ylabel('Frequency')
                axes[2].legend()
            
            # State duration histogram
            durations = self._get_state_durations(most_likely_states)
            all_durations = []
            for state, state_durations in durations.items():
                all_durations.extend(state_durations)
            
            if all_durations:
                axes[3].hist(all_durations, bins=20, alpha=0.7)
                axes[3].set_title('State Duration Distribution')
                axes[3].set_xlabel('Duration (timesteps)')
                axes[3].set_ylabel('Frequency')
            
            plt.tight_layout()
            
        else:
            # Plot specific state
            state_mask = most_likely_states == focus_state
            if not np.any(state_mask):
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f'No observations for State {focus_state}', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'State {focus_state} Analysis')
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            # State probability over time
            axes[0].plot(state_probs[:, focus_state], color=self.state_colors[focus_state])
            axes[0].fill_between(range(len(state_probs)), 0, state_probs[:, focus_state], 
                               alpha=0.3, color=self.state_colors[focus_state])
            axes[0].set_title(f'State {focus_state} Probability Over Time')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Probability')
            
            # State probability distribution
            axes[1].hist(state_probs[state_mask, focus_state], 
                        bins=20, color=self.state_colors[focus_state], alpha=0.7)
            axes[1].set_title(f'State {focus_state} Probability Distribution')
            axes[1].set_xlabel('Probability')
            axes[1].set_ylabel('Frequency')
            
            # Observations in this state
            if observations.ndim == 1:
                axes[2].hist(observations[state_mask], 
                           bins=20, color=self.state_colors[focus_state], alpha=0.7)
                axes[2].set_title(f'Observations in State {focus_state}')
                axes[2].set_xlabel('Observation Value')
                axes[2].set_ylabel('Frequency')
            
            # State transitions
            transition_counts = np.zeros(self.n_states)
            for i in range(len(most_likely_states) - 1):
                if most_likely_states[i] == focus_state:
                    transition_counts[most_likely_states[i + 1]] += 1
            
            axes[3].bar(range(self.n_states), transition_counts, color=self.state_colors)
            axes[3].set_title(f'Transitions from State {focus_state}')
            axes[3].set_xlabel('To State')
            axes[3].set_ylabel('Count')
            
            plt.tight_layout()
    
    def _get_state_durations(self, states: np.ndarray) -> Dict[int, List[int]]:
        """Get duration statistics for each state."""
        durations = {i: [] for i in range(self.n_states)}
        
        if len(states) == 0:
            return durations
        
        current_state = states[0]
        current_duration = 1
        
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_duration += 1
            else:
                durations[current_state].append(current_duration)
                current_state = states[i]
                current_duration = 1
        
        # Add final duration
        durations[current_state].append(current_duration)
        
        return durations
    
    def plot_regime_comparison(self, 
                             observations: np.ndarray,
                             state_probs_list: List[np.ndarray],
                             model_names: List[str],
                             timestamps: Optional[np.ndarray] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compare regime detection across multiple models.
        
        Args:
            observations: Market observations
            state_probs_list: List of state probability arrays from different models
            model_names: Names of the models being compared
            timestamps: Optional timestamps
            figsize: Figure size
            
        Returns:
            matplotlib Figure with comparison plots
        """
        n_models = len(state_probs_list)
        fig, axes = plt.subplots(n_models + 1, 1, figsize=figsize, sharex=True)
        
        if timestamps is None:
            timestamps = np.arange(len(observations))
        
        # Plot observations
        if observations.ndim == 1:
            axes[0].plot(timestamps, observations, 'k-', alpha=0.7, label='Observations')
        else:
            # Plot first dimension if multivariate
            axes[0].plot(timestamps, observations[:, 0], 'k-', alpha=0.7, label='Observations (dim 0)')
        
        axes[0].set_ylabel('Value')
        axes[0].set_title('Market Observations')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot state probabilities for each model
        for i, (state_probs, model_name) in enumerate(zip(state_probs_list, model_names)):
            ax = axes[i + 1]
            
            # Most likely state as background
            most_likely = np.argmax(state_probs, axis=1)
            for state in range(state_probs.shape[1]):
                state_mask = most_likely == state
                if np.any(state_mask):
                    ax.fill_between(timestamps, 0, 1, where=state_mask, 
                                  alpha=0.3, color=self.state_colors[state % len(self.state_colors)])
            
            # Plot state probabilities
            for state in range(min(state_probs.shape[1], len(self.state_colors))):
                ax.plot(timestamps, state_probs[:, state], 
                       color=self.state_colors[state], 
                       label=f'State {state}', linewidth=1.5)
            
            ax.set_ylabel('Probability')
            ax.set_title(f'{model_name} - State Probabilities')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        return fig