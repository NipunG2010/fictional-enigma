"""
Interactive parameter tuning framework with ipywidgets interface.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings

# Core libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Jupyter widgets
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None

# HMM components
from ..hmm.trainer import EnhancedHMMTrainer, HMMTrainingError
from ..hmm.models import HMMArtifact
from ..visualization.regime_visualizer import RegimeVisualizer


@dataclass
class TuningConfig:
    """Configuration for parameter tuning."""
    n_states: int = 3
    library: str = "hmmlearn"
    covariance_type: str = "full"
    n_iterations: int = 100
    validation_split: float = 0.2
    random_state: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TuningConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TuningResult:
    """Result from a tuning experiment."""
    config: TuningConfig
    artifact: HMMArtifact
    metrics: Dict[str, float]
    timestamp: str
    experiment_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config': self.config.to_dict(),
            'artifact': self.artifact.model_dump(),
            'metrics': self.metrics,
            'timestamp': self.timestamp,
            'experiment_id': self.experiment_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TuningResult':
        """Create from dictionary."""
        return cls(
            config=TuningConfig.from_dict(data['config']),
            artifact=HMMArtifact(**data['artifact']),
            metrics=data['metrics'],
            timestamp=data['timestamp'],
            experiment_id=data['experiment_id']
        )


class HMMParameterTuner:
    """Interactive parameter tuning interface for HMM models."""
    
    def __init__(self, 
                 observations: np.ndarray,
                 config_dir: Optional[Path] = None):
        """
        Initialize the parameter tuner.
        
        Args:
            observations: Training data of shape (n_samples, n_features)
            config_dir: Directory to save/load configurations (default: ./tuning_configs)
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError(
                "IPython widgets not available. Install with: pip install ipywidgets jupyter"
            )
        
        self.observations = observations
        self.config_dir = config_dir or Path("./tuning_configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.results: Dict[str, TuningResult] = {}
        self.current_artifact: Optional[HMMArtifact] = None
        self.current_metrics: Optional[Dict[str, float]] = None
        
        # Create widgets
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all UI widgets."""
        # Parameter widgets
        self.n_states_slider = widgets.IntSlider(
            value=3,
            min=2,
            max=10,
            step=1,
            description='States:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.library_dropdown = widgets.Dropdown(
            options=['hmmlearn', 'pomegranate'],
            value='hmmlearn',
            description='Library:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.covariance_dropdown = widgets.Dropdown(
            options=['full', 'diag', 'spherical'],
            value='full',
            description='Covariance:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.iterations_slider = widgets.IntSlider(
            value=100,
            min=10,
            max=1000,
            step=10,
            description='Iterations:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.validation_slider = widgets.FloatSlider(
            value=0.2,
            min=0.1,
            max=0.5,
            step=0.05,
            description='Val Split:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.random_state_input = widgets.IntText(
            value=42,
            description='Random Seed:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Action buttons
        self.train_button = widgets.Button(
            description='Train Model',
            button_style='success',
            icon='play',
            layout=widgets.Layout(width='150px')
        )
        
        self.save_button = widgets.Button(
            description='Save Config',
            button_style='info',
            icon='save',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        
        self.load_button = widgets.Button(
            description='Load Config',
            button_style='warning',
            icon='upload',
            layout=widgets.Layout(width='150px')
        )
        
        self.compare_button = widgets.Button(
            description='Compare Results',
            button_style='primary',
            icon='bar-chart',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        
        # Output areas
        self.status_output = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='10px',
                margin='10px 0'
            )
        )
        
        self.metrics_output = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='10px',
                margin='10px 0'
            )
        )
        
        self.plot_output = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='10px',
                margin='10px 0'
            )
        )
        
        # Progress indicator
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Training:',
            bar_style='info',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', visibility='hidden')
        )
        
        # Connect button callbacks
        self.train_button.on_click(self._on_train_clicked)
        self.save_button.on_click(self._on_save_clicked)
        self.load_button.on_click(self._on_load_clicked)
        self.compare_button.on_click(self._on_compare_clicked)
    
    def _get_current_config(self) -> TuningConfig:
        """Get current configuration from widgets."""
        return TuningConfig(
            n_states=self.n_states_slider.value,
            library=self.library_dropdown.value,
            covariance_type=self.covariance_dropdown.value,
            n_iterations=self.iterations_slider.value,
            validation_split=self.validation_slider.value,
            random_state=self.random_state_input.value
        )
    
    def _set_config(self, config: TuningConfig):
        """Set widget values from configuration."""
        self.n_states_slider.value = config.n_states
        self.library_dropdown.value = config.library
        self.covariance_dropdown.value = config.covariance_type
        self.iterations_slider.value = config.n_iterations
        self.validation_slider.value = config.validation_split
        self.random_state_input.value = config.random_state
    
    def _on_train_clicked(self, button):
        """Handle train button click."""
        config = self._get_current_config()
        
        # Clear previous outputs
        self.status_output.clear_output()
        self.metrics_output.clear_output()
        self.plot_output.clear_output()
        
        # Show progress bar
        self.progress_bar.layout.visibility = 'visible'
        self.progress_bar.value = 0
        
        # Disable train button during training
        self.train_button.disabled = True
        
        with self.status_output:
            print("🚀 Starting model training...")
            print(f"Configuration: {config.n_states} states, {config.library}, {config.covariance_type}")
        
        try:
            # Update progress
            self.progress_bar.value = 20
            
            # Create trainer
            trainer = EnhancedHMMTrainer(
                n_states=config.n_states,
                library=config.library,
                covariance_type=config.covariance_type,
                random_state=config.random_state
            )
            
            self.progress_bar.value = 40
            
            # Train with validation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                artifact, metrics = trainer.train_with_validation(
                    self.observations,
                    validation_split=config.validation_split,
                    n_iterations=config.n_iterations
                )
            
            self.progress_bar.value = 80
            
            # Store results
            self.current_artifact = artifact
            self.current_metrics = metrics
            
            # Create experiment ID
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store in results
            result = TuningResult(
                config=config,
                artifact=artifact,
                metrics=metrics,
                timestamp=datetime.now().isoformat(),
                experiment_id=experiment_id
            )
            self.results[experiment_id] = result
            
            self.progress_bar.value = 100
            
            # Display results
            self._display_training_results(config, artifact, metrics)
            
            # Enable save and compare buttons
            self.save_button.disabled = False
            if len(self.results) > 1:
                self.compare_button.disabled = False
            
            with self.status_output:
                print(f"✅ Training completed successfully!")
                print(f"Experiment ID: {experiment_id}")
        
        except Exception as e:
            with self.status_output:
                print(f"❌ Training failed: {str(e)}")
                print("\nSuggestions:")
                print("1. Check data quality and preprocessing")
                print("2. Try different initialization parameters")
                print("3. Reduce model complexity (fewer states)")
                print("4. Increase number of iterations")
        
        finally:
            # Hide progress bar and re-enable button
            self.progress_bar.layout.visibility = 'hidden'
            self.train_button.disabled = False
    
    def _display_training_results(self, 
                                 config: TuningConfig,
                                 artifact: HMMArtifact,
                                 metrics: Dict[str, float]):
        """Display training results."""
        # Display metrics
        with self.metrics_output:
            clear_output(wait=True)
            
            # Create metrics table
            html = "<h4>Training Metrics</h4>"
            html += "<table style='border-collapse: collapse; width: 100%;'>"
            html += "<tr style='background-color: #f2f2f2;'>"
            html += "<th style='padding: 8px; border: 1px solid #ddd; text-align: left;'>Metric</th>"
            html += "<th style='padding: 8px; border: 1px solid #ddd; text-align: right;'>Value</th>"
            html += "</tr>"
            
            # Training metrics from artifact
            if 'convergence_log_likelihood' in artifact.metadata:
                html += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'>Train Log-Likelihood</td>"
                html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: right;'>{artifact.metadata['convergence_log_likelihood']:.4f}</td></tr>"
            
            if 'aic' in artifact.metadata:
                html += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'>Train AIC</td>"
                html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: right;'>{artifact.metadata['aic']:.4f}</td></tr>"
            
            if 'bic' in artifact.metadata:
                html += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'>Train BIC</td>"
                html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: right;'>{artifact.metadata['bic']:.4f}</td></tr>"
            
            # Validation metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    html += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'>Val {metric_name.replace('_', ' ').title()}</td>"
                    html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: right;'>{value:.4f}</td></tr>"
            
            html += "</table>"
            
            # Convergence info
            if 'converged' in artifact.metadata:
                converged = artifact.metadata['converged']
                status = "✅ Converged" if converged else "⚠️ Did not converge"
                html += f"<p style='margin-top: 10px;'><strong>Convergence:</strong> {status}</p>"
            
            display(HTML(html))
        
        # Display visualizations
        with self.plot_output:
            clear_output(wait=True)
            
            # Create visualizer
            visualizer = RegimeVisualizer(artifact)
            
            # Plot transition matrix
            fig = visualizer.plot_transition_matrix(
                title=f"Transition Matrix ({config.n_states} states)"
            )
            plt.show()
            
            # If we have enough data, show state probabilities
            if len(self.observations) > 0:
                try:
                    # Get state probabilities
                    trainer = EnhancedHMMTrainer(
                        n_states=config.n_states,
                        library=config.library,
                        covariance_type=config.covariance_type,
                        random_state=config.random_state
                    )
                    trainer.trainer.model = artifact  # Set the trained model
                    
                    # For hmmlearn, we need to reconstruct the model
                    if config.library == "hmmlearn":
                        from hmmlearn import hmm as hmmlearn_hmm
                        model = hmmlearn_hmm.GaussianHMM(
                            n_components=config.n_states,
                            covariance_type=config.covariance_type
                        )
                        model.startprob_ = np.array(artifact.initial_probabilities)
                        model.transmat_ = np.array(artifact.transition_matrix)
                        model.means_ = np.array(artifact.means)
                        model.covars_ = np.array(artifact.covariances)
                        
                        state_probs = model.predict_proba(self.observations)
                        
                        # Plot state probabilities (sample if too large)
                        sample_size = min(500, len(self.observations))
                        indices = np.linspace(0, len(self.observations)-1, sample_size, dtype=int)
                        
                        fig = visualizer.plot_state_probabilities(
                            state_probs[indices],
                            timestamps=indices,
                            interactive=False,
                            title=f"State Probabilities (sampled {sample_size} points)"
                        )
                        plt.show()
                
                except Exception as e:
                    print(f"Could not generate state probability plot: {str(e)}")
    
    def _on_save_clicked(self, button):
        """Handle save button click."""
        if self.current_artifact is None:
            with self.status_output:
                print("❌ No model to save. Train a model first.")
            return
        
        config = self._get_current_config()
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"config_{config.library}_{config.n_states}states_{timestamp}.json"
        filepath = self.config_dir / filename
        
        # Save configuration and results
        result = TuningResult(
            config=config,
            artifact=self.current_artifact,
            metrics=self.current_metrics or {},
            timestamp=datetime.now().isoformat(),
            experiment_id=f"exp_{timestamp}"
        )
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            with self.status_output:
                print(f"✅ Configuration saved to: {filepath}")
        
        except Exception as e:
            with self.status_output:
                print(f"❌ Failed to save configuration: {str(e)}")
    
    def _on_load_clicked(self, button):
        """Handle load button click."""
        # Get list of saved configurations
        config_files = sorted(self.config_dir.glob("config_*.json"))
        
        if not config_files:
            with self.status_output:
                print("❌ No saved configurations found.")
            return
        
        # Create file selector
        file_selector = widgets.Dropdown(
            options=[(f.name, f) for f in config_files],
            description='Select:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='500px')
        )
        
        load_confirm_button = widgets.Button(
            description='Load',
            button_style='success',
            icon='check'
        )
        
        cancel_button = widgets.Button(
            description='Cancel',
            button_style='danger',
            icon='times'
        )
        
        load_output = widgets.Output()
        
        def on_load_confirm(b):
            selected_file = file_selector.value
            
            try:
                with open(selected_file, 'r') as f:
                    data = json.load(f)
                
                result = TuningResult.from_dict(data)
                
                # Set configuration
                self._set_config(result.config)
                
                # Store loaded result
                self.current_artifact = result.artifact
                self.current_metrics = result.metrics
                self.results[result.experiment_id] = result
                
                with load_output:
                    clear_output()
                    print(f"✅ Configuration loaded from: {selected_file.name}")
                
                # Enable buttons
                self.save_button.disabled = False
                if len(self.results) > 1:
                    self.compare_button.disabled = False
            
            except Exception as e:
                with load_output:
                    clear_output()
                    print(f"❌ Failed to load configuration: {str(e)}")
        
        def on_cancel(b):
            with load_output:
                clear_output()
        
        load_confirm_button.on_click(on_load_confirm)
        cancel_button.on_click(on_cancel)
        
        # Display load interface
        with self.status_output:
            clear_output()
            display(widgets.VBox([
                widgets.HTML("<h4>Load Configuration</h4>"),
                file_selector,
                widgets.HBox([load_confirm_button, cancel_button]),
                load_output
            ]))
    
    def _on_compare_clicked(self, button):
        """Handle compare button click."""
        if len(self.results) < 2:
            with self.status_output:
                print("❌ Need at least 2 results to compare.")
            return
        
        with self.plot_output:
            clear_output(wait=True)
            
            # Create comparison plots
            self._plot_comparison()
    
    def _plot_comparison(self):
        """Plot comparison of all results."""
        if len(self.results) == 0:
            print("No results to compare.")
            return
        
        # Extract data for comparison
        experiment_ids = list(self.results.keys())
        configs = [self.results[eid].config for eid in experiment_ids]
        metrics_list = [self.results[eid].metrics for eid in experiment_ids]
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Log-likelihood comparison
        ax = axes[0, 0]
        train_ll = [self.results[eid].artifact.metadata.get('convergence_log_likelihood', np.nan) 
                   for eid in experiment_ids]
        val_ll = [m.get('log_likelihood', np.nan) for m in metrics_list]
        
        x = np.arange(len(experiment_ids))
        width = 0.35
        
        ax.bar(x - width/2, train_ll, width, label='Train', alpha=0.8)
        ax.bar(x + width/2, val_ll, width, label='Validation', alpha=0.8)
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Log-Likelihood')
        ax.set_title('Log-Likelihood Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Exp {i+1}" for i in range(len(experiment_ids))], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: AIC/BIC comparison
        ax = axes[0, 1]
        aic_values = [self.results[eid].artifact.metadata.get('aic', np.nan) 
                     for eid in experiment_ids]
        bic_values = [self.results[eid].artifact.metadata.get('bic', np.nan) 
                     for eid in experiment_ids]
        
        ax.bar(x - width/2, aic_values, width, label='AIC', alpha=0.8)
        ax.bar(x + width/2, bic_values, width, label='BIC', alpha=0.8)
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Information Criterion')
        ax.set_title('AIC/BIC Comparison (lower is better)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Exp {i+1}" for i in range(len(experiment_ids))], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Configuration comparison
        ax = axes[1, 0]
        n_states_list = [c.n_states for c in configs]
        libraries = [c.library for c in configs]
        
        # Color by library
        colors = ['blue' if lib == 'hmmlearn' else 'orange' for lib in libraries]
        ax.scatter(x, n_states_list, c=colors, s=100, alpha=0.6)
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Number of States')
        ax.set_title('Model Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Exp {i+1}" for i in range(len(experiment_ids))], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add legend for libraries
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label='hmmlearn'),
            Patch(facecolor='orange', alpha=0.6, label='pomegranate')
        ]
        ax.legend(handles=legend_elements)
        
        # Plot 4: Summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary table
        table_data = []
        for i, eid in enumerate(experiment_ids):
            config = configs[i]
            metrics = metrics_list[i]
            val_ll = metrics.get('log_likelihood', 'N/A')
            if isinstance(val_ll, float):
                val_ll = f"{val_ll:.2f}"
            
            table_data.append([
                f"Exp {i+1}",
                f"{config.n_states}",
                config.library[:4],  # Abbreviate
                config.covariance_type[:4],  # Abbreviate
                val_ll
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['ID', 'States', 'Lib', 'Cov', 'Val LL'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Experiment Summary', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed comparison
        print("\n" + "="*60)
        print("DETAILED COMPARISON")
        print("="*60)
        
        for i, eid in enumerate(experiment_ids):
            result = self.results[eid]
            print(f"\nExperiment {i+1} ({eid}):")
            print(f"  States: {result.config.n_states}")
            print(f"  Library: {result.config.library}")
            print(f"  Covariance: {result.config.covariance_type}")
            print(f"  Iterations: {result.config.n_iterations}")
            print(f"  Train LL: {result.artifact.metadata.get('convergence_log_likelihood', 'N/A')}")
            print(f"  Val LL: {result.metrics.get('log_likelihood', 'N/A')}")
            print(f"  AIC: {result.artifact.metadata.get('aic', 'N/A')}")
            print(f"  BIC: {result.artifact.metadata.get('bic', 'N/A')}")
    
    def create_tuning_interface(self) -> widgets.VBox:
        """
        Create the complete interactive tuning interface.
        
        Returns:
            IPython widget VBox containing the full interface
        """
        # Header
        header = widgets.HTML(
            "<h2>🎛️ HMM Parameter Tuning Interface</h2>"
            "<p>Interactively tune HMM parameters and compare different configurations.</p>"
        )
        
        # Parameter controls
        param_box1 = widgets.HBox([self.n_states_slider, self.library_dropdown])
        param_box2 = widgets.HBox([self.covariance_dropdown, self.iterations_slider])
        param_box3 = widgets.HBox([self.validation_slider, self.random_state_input])
        
        params_section = widgets.VBox([
            widgets.HTML("<h3>Model Parameters</h3>"),
            param_box1,
            param_box2,
            param_box3,
            self.progress_bar
        ])
        
        # Action buttons
        button_box = widgets.HBox([
            self.train_button,
            self.save_button,
            self.load_button,
            self.compare_button
        ])
        
        actions_section = widgets.VBox([
            widgets.HTML("<h3>Actions</h3>"),
            button_box
        ])
        
        # Results section
        results_section = widgets.VBox([
            widgets.HTML("<h3>Training Status</h3>"),
            self.status_output,
            widgets.HTML("<h3>Metrics</h3>"),
            self.metrics_output,
            widgets.HTML("<h3>Visualizations</h3>"),
            self.plot_output
        ])
        
        # Complete interface
        interface = widgets.VBox([
            header,
            widgets.HTML("<hr>"),
            params_section,
            widgets.HTML("<hr>"),
            actions_section,
            widgets.HTML("<hr>"),
            results_section
        ])
        
        return interface
    
    def get_best_result(self, 
                       metric: str = 'log_likelihood',
                       higher_is_better: bool = True) -> Optional[TuningResult]:
        """
        Get the best result based on a specific metric.
        
        Args:
            metric: Metric name to use for comparison
            higher_is_better: Whether higher values are better
            
        Returns:
            Best TuningResult or None if no results
        """
        if not self.results:
            return None
        
        best_result = None
        best_value = float('-inf') if higher_is_better else float('inf')
        
        for result in self.results.values():
            value = result.metrics.get(metric)
            if value is None:
                continue
            
            if higher_is_better:
                if value > best_value:
                    best_value = value
                    best_result = result
            else:
                if value < best_value:
                    best_value = value
                    best_result = result
        
        return best_result
    
    def export_results(self, filepath: Path):
        """
        Export all results to a JSON file.
        
        Args:
            filepath: Path to save results
        """
        export_data = {
            'results': {eid: result.to_dict() for eid, result in self.results.items()},
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Results exported to: {filepath}")
