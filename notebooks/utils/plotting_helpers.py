"""
Plotting utilities for HMM research environment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots will be disabled.")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_interactive_plot(df: pd.DataFrame, 
                          title: str = "Signal Data",
                          height: int = 600) -> Union[go.Figure, plt.Figure]:
    """
    Create interactive plot of signal data.
    
    Args:
        df: DataFrame with signal data
        title: Plot title
        height: Plot height in pixels
    
    Returns:
        Plotly figure if available, otherwise matplotlib figure
    """
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(width=1.5)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Signal Value",
            height=height,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    else:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=1.5)
        
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def plot_signal_correlation(df: pd.DataFrame, 
                          method: str = 'pearson',
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot correlation matrix of signals.
    
    Args:
        df: DataFrame with signal data
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    
    corr_matrix = df.corr(method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                ax=ax)
    
    ax.set_title(f'Signal Correlation Matrix ({method.title()})')
    plt.tight_layout()
    
    return fig


def plot_signal_distributions(df: pd.DataFrame, 
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot distributions of all signals.
    
    Args:
        df: DataFrame with signal data
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    
    n_signals = len(df.columns)
    n_cols = min(3, n_signals)
    n_rows = (n_signals + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_signals == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        ax = axes[i]
        
        # Histogram
        ax.hist(df[col].dropna(), bins=50, alpha=0.7, density=True, color='skyblue')
        
        # Overlay normal distribution for comparison
        mu, sigma = df[col].mean(), df[col].std()
        x = np.linspace(df[col].min(), df[col].max(), 100)
        normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, normal_dist, 'r--', alpha=0.8, label='Normal')
        
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_signals, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_state_probabilities(state_probs: np.ndarray,
                           timestamps: Optional[pd.DatetimeIndex] = None,
                           state_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """
    Plot HMM state probabilities over time.
    
    Args:
        state_probs: Array of state probabilities (n_samples, n_states)
        timestamps: Optional timestamp index
        state_names: Optional state names
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    
    n_states = state_probs.shape[1]
    
    if timestamps is None:
        timestamps = np.arange(len(state_probs))
    
    if state_names is None:
        state_names = [f'State {i}' for i in range(n_states)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create stacked area plot
    ax.stackplot(timestamps, *[state_probs[:, i] for i in range(n_states)],
                labels=state_names, alpha=0.8)
    
    ax.set_title('HMM State Probabilities Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def plot_transition_matrix(transition_matrix: np.ndarray,
                         state_names: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot HMM transition matrix as heatmap.
    
    Args:
        transition_matrix: Transition matrix (n_states, n_states)
        state_names: Optional state names
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    
    n_states = transition_matrix.shape[0]
    
    if state_names is None:
        state_names = [f'State {i}' for i in range(n_states)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(transition_matrix,
                annot=True,
                fmt='.3f',
                cmap='Blues',
                square=True,
                xticklabels=state_names,
                yticklabels=state_names,
                cbar_kws={'label': 'Transition Probability'},
                ax=ax)
    
    ax.set_title('HMM State Transition Matrix')
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    
    plt.tight_layout()
    return fig


def plot_regime_comparison(df: pd.DataFrame,
                         state_sequence: np.ndarray,
                         state_names: Optional[List[str]] = None,
                         signal_col: str = None,
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot signal data colored by regime states.
    
    Args:
        df: DataFrame with signal data
        state_sequence: Most likely state sequence
        state_names: Optional state names
        signal_col: Specific signal column to plot (if None, plot all)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    
    n_states = len(np.unique(state_sequence))
    
    if state_names is None:
        state_names = [f'State {i}' for i in range(n_states)]
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_states))
    
    if signal_col:
        signals_to_plot = [signal_col]
    else:
        signals_to_plot = df.columns[:min(3, len(df.columns))]  # Plot max 3 signals
    
    n_plots = len(signals_to_plot)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, col in enumerate(signals_to_plot):
        ax = axes[i]
        
        # Plot signal data colored by state
        for state in range(n_states):
            mask = state_sequence == state
            if mask.any():
                ax.scatter(df.index[mask], df[col].iloc[mask], 
                          c=[colors[state]], label=state_names[state], 
                          alpha=0.6, s=10)
        
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time')
    plt.suptitle('Signal Data Colored by HMM States')
    plt.tight_layout()
    
    return fig


def plot_model_comparison(results: Dict[str, Dict[str, float]],
                        metrics: List[str] = ['log_likelihood', 'aic', 'bic'],
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot comparison of different HMM models.
    
    Args:
        results: Dictionary of model results
        metrics: List of metrics to compare
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    model_names = list(results.keys())
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        values = [results[model].get(metric, np.nan) for model in model_names]
        
        bars = ax.bar(model_names, values, alpha=0.7)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Value')
        
        # Rotate x-axis labels if they're long
        if max(len(name) for name in model_names) > 10:
            ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def format_regime_stats(regime_stats: Dict[str, Any]) -> str:
    """
    Format regime statistics for display.
    
    Args:
        regime_stats: Dictionary of regime statistics
    
    Returns:
        Formatted string
    """
    
    formatted = "📊 Regime Statistics\n"
    formatted += "=" * 50 + "\n"
    
    for state, stats in regime_stats.items():
        formatted += f"\n{state.replace('_', ' ').title()}:\n"
        formatted += f"  Mean Duration: {stats.get('mean_duration', 'N/A'):.2f} periods\n"
        formatted += f"  Median Duration: {stats.get('median_duration', 'N/A'):.2f} periods\n"
        formatted += f"  Max Duration: {stats.get('max_duration', 'N/A')} periods\n"
        formatted += f"  Stable Periods: {stats.get('stable_periods', 'N/A')}\n"
        formatted += f"  Total Periods: {stats.get('total_periods', 'N/A')}\n"
    
    return formatted


def create_diagnostic_plots(df: pd.DataFrame, 
                          figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """
    Create comprehensive diagnostic plots for signal data.
    
    Args:
        df: DataFrame with signal data
        figsize: Figure size
    
    Returns:
        Matplotlib figure with multiple subplots
    """
    
    fig = plt.figure(figsize=figsize)
    
    # Time series plot
    ax1 = plt.subplot(2, 2, 1)
    for col in df.columns:
        ax1.plot(df.index, df[col], label=col, alpha=0.8)
    ax1.set_title('Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Correlation heatmap
    ax2 = plt.subplot(2, 2, 2)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax2)
    ax2.set_title('Correlation Matrix')
    
    # Distribution plot
    ax3 = plt.subplot(2, 2, 3)
    df.plot(kind='hist', bins=30, alpha=0.7, ax=ax3)
    ax3.set_title('Signal Distributions')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    
    # Box plot
    ax4 = plt.subplot(2, 2, 4)
    df.boxplot(ax=ax4)
    ax4.set_title('Signal Box Plots')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    return fig