"""
Model performance tracking and regression detection for production monitoring.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, field
from scipy import stats
import warnings

from ..hmm.models import HMMArtifact
from ..hmm.trainer import EnhancedHMMTrainer

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of model performance at a point in time."""
    timestamp: str
    model_id: str
    model_version: str
    metrics: Dict[str, float]
    data_stats: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'metrics': self.metrics,
            'data_stats': self.data_stats,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceSnapshot':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RegressionAlert:
    """Alert for detected performance regression."""
    timestamp: str
    model_id: str
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    severity: str  # 'warning', 'critical'
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'model_id': self.model_id,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'change_percent': self.change_percent,
            'severity': self.severity,
            'message': self.message
        }


class PerformanceTracker:
    """
    Track model performance over time and detect regressions.
    
    Monitors key metrics and alerts when performance degrades
    beyond acceptable thresholds.
    """
    
    def __init__(self,
                 tracking_dir: Path,
                 baseline_window: int = 10,
                 warning_threshold: float = 0.05,
                 critical_threshold: float = 0.10):
        """
        Initialize performance tracker.
        
        Args:
            tracking_dir: Directory to store tracking data
            baseline_window: Number of recent snapshots to use for baseline
            warning_threshold: Threshold for warning alerts (5% degradation)
            critical_threshold: Threshold for critical alerts (10% degradation)
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_window = baseline_window
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self.snapshots: Dict[str, List[PerformanceSnapshot]] = {}
        self.alerts: List[RegressionAlert] = []
        
        # Load existing tracking data
        self._load_tracking_data()
    
    def record_performance(self,
                          model_id: str,
                          model_version: str,
                          artifact: HMMArtifact,
                          observations: np.ndarray,
                          metadata: Optional[Dict[str, Any]] = None) -> PerformanceSnapshot:
        """
        Record model performance snapshot.
        
        Args:
            model_id: Unique identifier for the model
            model_version: Version string for the model
            artifact: HMM artifact to evaluate
            observations: Data to evaluate on
            metadata: Optional additional metadata
            
        Returns:
            PerformanceSnapshot with recorded metrics
        """
        logger.info(f"Recording performance for model {model_id} v{model_version}")
        
        # Evaluate model by reconstructing it
        # Get library and covariance_type from metadata if available
        library = artifact.metadata.get('library', 'hmmlearn')
        covariance_type = artifact.metadata.get('covariance_type', 'diag')
        
        # Reconstruct the model from artifact
        from hmmlearn import hmm as hmmlearn_hmm
        
        model = hmmlearn_hmm.GaussianHMM(
            n_components=artifact.n_states,
            covariance_type=covariance_type
        )
        model.startprob_ = np.array(artifact.initial_probabilities)
        model.transmat_ = np.array(artifact.transition_matrix)
        model.means_ = np.array(artifact.means)
        
        # Handle covariances based on type
        covars = np.array(artifact.covariances)
        if covariance_type == 'diag':
            if covars.ndim == 3:
                model.covars_ = np.array([np.diag(cov) for cov in covars])
            else:
                model.covars_ = covars
        elif covariance_type == 'full':
            model.covars_ = covars
        elif covariance_type == 'spherical':
            if covars.ndim == 3:
                model.covars_ = np.array([np.mean(np.diag(cov)) for cov in covars])
            else:
                model.covars_ = covars
        
        # Calculate metrics
        log_likelihood = model.score(observations)
        n_params = (artifact.n_states ** 2 + 2 * artifact.n_states * observations.shape[1] - 1)
        n_samples = observations.shape[0]
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        perplexity = np.exp(-log_likelihood / n_samples)
        
        metrics = {
            'log_likelihood': float(log_likelihood),
            'aic': float(aic),
            'bic': float(bic),
            'perplexity': float(perplexity)
        }
        
        # Calculate data statistics
        data_stats = {
            'n_samples': int(observations.shape[0]),
            'n_features': int(observations.shape[1]),
            'mean': float(np.mean(observations)),
            'std': float(np.std(observations)),
            'min': float(np.min(observations)),
            'max': float(np.max(observations))
        }
        
        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            model_id=model_id,
            model_version=model_version,
            metrics=metrics,
            data_stats=data_stats,
            metadata=metadata or {}
        )
        
        # Store snapshot
        if model_id not in self.snapshots:
            self.snapshots[model_id] = []
        self.snapshots[model_id].append(snapshot)
        
        # Save to disk
        self._save_snapshot(snapshot)
        
        # Check for regressions
        alerts = self.detect_regression(model_id)
        if alerts:
            logger.warning(f"Detected {len(alerts)} performance regressions for model {model_id}")
            for alert in alerts:
                logger.warning(f"  {alert.severity.upper()}: {alert.message}")
        
        return snapshot
    
    def detect_regression(self,
                         model_id: str,
                         metrics_to_check: Optional[List[str]] = None) -> List[RegressionAlert]:
        """
        Detect performance regressions for a model.
        
        Args:
            model_id: Model identifier
            metrics_to_check: List of metrics to check (default: all)
            
        Returns:
            List of RegressionAlerts
        """
        if model_id not in self.snapshots or len(self.snapshots[model_id]) < 2:
            return []
        
        snapshots = self.snapshots[model_id]
        current_snapshot = snapshots[-1]
        
        # Get baseline snapshots
        baseline_snapshots = snapshots[-(self.baseline_window + 1):-1]
        if not baseline_snapshots:
            return []
        
        # Metrics to check
        if metrics_to_check is None:
            metrics_to_check = list(current_snapshot.metrics.keys())
        
        alerts = []
        
        for metric_name in metrics_to_check:
            # Get baseline values
            baseline_values = [
                s.metrics.get(metric_name)
                for s in baseline_snapshots
                if metric_name in s.metrics
            ]
            
            if not baseline_values:
                continue
            
            baseline_mean = np.mean(baseline_values)
            baseline_std = np.std(baseline_values)
            
            # Get current value
            current_value = current_snapshot.metrics.get(metric_name)
            if current_value is None:
                continue
            
            # Calculate change
            if baseline_mean != 0:
                change_percent = (current_value - baseline_mean) / abs(baseline_mean)
            else:
                change_percent = 0.0
            
            # Determine if this is a regression
            # For metrics like log_likelihood, higher is better
            # For metrics like AIC/BIC, lower is better
            is_regression = False
            severity = None
            
            if metric_name in ['log_likelihood', 'perplexity']:
                # Higher is better
                if change_percent < -self.critical_threshold:
                    is_regression = True
                    severity = 'critical'
                elif change_percent < -self.warning_threshold:
                    is_regression = True
                    severity = 'warning'
            elif metric_name in ['aic', 'bic']:
                # Lower is better
                if change_percent > self.critical_threshold:
                    is_regression = True
                    severity = 'critical'
                elif change_percent > self.warning_threshold:
                    is_regression = True
                    severity = 'warning'
            
            # Statistical significance test
            if is_regression and len(baseline_values) >= 3:
                # Perform t-test
                t_stat, p_value = stats.ttest_1samp(baseline_values, current_value)
                
                # Only alert if statistically significant
                if p_value > 0.05:
                    is_regression = False
            
            if is_regression:
                message = (f"{metric_name} degraded by {abs(change_percent)*100:.1f}%: "
                          f"{baseline_mean:.4f} → {current_value:.4f}")
                
                alert = RegressionAlert(
                    timestamp=current_snapshot.timestamp,
                    model_id=model_id,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_mean,
                    change_percent=change_percent,
                    severity=severity,
                    message=message
                )
                
                alerts.append(alert)
                self.alerts.append(alert)
        
        # Save alerts
        if alerts:
            self._save_alerts(alerts)
        
        return alerts
    
    def get_performance_history(self,
                               model_id: str,
                               metric_name: Optional[str] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get performance history for a model.
        
        Args:
            model_id: Model identifier
            metric_name: Optional specific metric to retrieve
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with performance history
        """
        if model_id not in self.snapshots:
            return pd.DataFrame()
        
        snapshots = self.snapshots[model_id]
        
        # Filter by date if specified
        if start_date or end_date:
            filtered_snapshots = []
            for snapshot in snapshots:
                snapshot_date = datetime.fromisoformat(snapshot.timestamp)
                if start_date and snapshot_date < start_date:
                    continue
                if end_date and snapshot_date > end_date:
                    continue
                filtered_snapshots.append(snapshot)
            snapshots = filtered_snapshots
        
        # Convert to DataFrame
        rows = []
        for snapshot in snapshots:
            row = {
                'timestamp': snapshot.timestamp,
                'model_version': snapshot.model_version
            }
            
            if metric_name:
                if metric_name in snapshot.metrics:
                    row[metric_name] = snapshot.metrics[metric_name]
            else:
                row.update(snapshot.metrics)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def get_alert_history(self,
                         model_id: Optional[str] = None,
                         severity: Optional[str] = None,
                         start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get alert history.
        
        Args:
            model_id: Optional model filter
            severity: Optional severity filter ('warning' or 'critical')
            start_date: Optional start date filter
            
        Returns:
            DataFrame with alert history
        """
        alerts = self.alerts
        
        # Apply filters
        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if start_date:
            alerts = [
                a for a in alerts
                if datetime.fromisoformat(a.timestamp) >= start_date
            ]
        
        # Convert to DataFrame
        if not alerts:
            return pd.DataFrame()
        
        df = pd.DataFrame([a.to_dict() for a in alerts])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        return df
    
    def plot_performance_trend(self,
                              model_id: str,
                              metric_name: str,
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[Path] = None):
        """
        Plot performance trend over time.
        
        Args:
            model_id: Model identifier
            metric_name: Metric to plot
            figsize: Figure size
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        
        df = self.get_performance_history(model_id, metric_name)
        
        if df.empty:
            logger.warning(f"No performance history for model {model_id}")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot metric over time
        ax.plot(df['timestamp'], df[metric_name], 'o-', linewidth=2, markersize=6)
        
        # Add baseline reference
        if len(df) > self.baseline_window:
            baseline_values = df[metric_name].iloc[-(self.baseline_window + 1):-1]
            baseline_mean = baseline_values.mean()
            ax.axhline(baseline_mean, color='green', linestyle='--',
                      label=f'Baseline Mean: {baseline_mean:.4f}')
            
            # Add threshold lines
            if metric_name in ['log_likelihood', 'perplexity']:
                warning_line = baseline_mean * (1 - self.warning_threshold)
                critical_line = baseline_mean * (1 - self.critical_threshold)
            else:  # AIC, BIC
                warning_line = baseline_mean * (1 + self.warning_threshold)
                critical_line = baseline_mean * (1 + self.critical_threshold)
            
            ax.axhline(warning_line, color='orange', linestyle=':',
                      label=f'Warning Threshold')
            ax.axhline(critical_line, color='red', linestyle=':',
                      label=f'Critical Threshold')
        
        # Mark alerts
        alerts = [a for a in self.alerts if a.model_id == model_id and a.metric_name == metric_name]
        if alerts:
            alert_times = [datetime.fromisoformat(a.timestamp) for a in alerts]
            alert_values = [a.current_value for a in alerts]
            alert_colors = ['orange' if a.severity == 'warning' else 'red' for a in alerts]
            
            ax.scatter(alert_times, alert_values, c=alert_colors, s=100,
                      marker='X', zorder=5, label='Alerts')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Performance Trend: {model_id} - {metric_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance trend plot saved to {save_path}")
        
        plt.show()
    
    def generate_monitoring_report(self, model_id: str) -> str:
        """Generate monitoring report for a model."""
        report = []
        report.append("="*70)
        report.append(f"PERFORMANCE MONITORING REPORT: {model_id}")
        report.append("="*70)
        report.append("")
        
        if model_id not in self.snapshots:
            report.append("No performance data available for this model.")
            return "\n".join(report)
        
        snapshots = self.snapshots[model_id]
        
        # Summary statistics
        report.append(f"Total Snapshots: {len(snapshots)}")
        report.append(f"First Recorded: {snapshots[0].timestamp}")
        report.append(f"Last Recorded: {snapshots[-1].timestamp}")
        report.append("")
        
        # Current performance
        current = snapshots[-1]
        report.append("Current Performance:")
        for metric, value in current.metrics.items():
            report.append(f"  {metric}: {value:.4f}")
        report.append("")
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if a.model_id == model_id]
        recent_alerts = sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)[:5]
        
        if recent_alerts:
            report.append("Recent Alerts:")
            for alert in recent_alerts:
                report.append(f"  [{alert.severity.upper()}] {alert.timestamp}")
                report.append(f"    {alert.message}")
            report.append("")
        else:
            report.append("No recent alerts.")
            report.append("")
        
        # Trend analysis
        if len(snapshots) >= 5:
            report.append("Trend Analysis (last 5 snapshots):")
            recent_snapshots = snapshots[-5:]
            
            for metric in current.metrics.keys():
                values = [s.metrics.get(metric) for s in recent_snapshots if metric in s.metrics]
                if len(values) >= 2:
                    trend = "improving" if values[-1] > values[0] else "declining"
                    change = ((values[-1] - values[0]) / abs(values[0])) * 100 if values[0] != 0 else 0
                    report.append(f"  {metric}: {trend} ({change:+.1f}%)")
            report.append("")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    def _save_snapshot(self, snapshot: PerformanceSnapshot):
        """Save snapshot to disk."""
        model_dir = self.tracking_dir / snapshot.model_id
        model_dir.mkdir(exist_ok=True)
        
        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = model_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)
    
    def _save_alerts(self, alerts: List[RegressionAlert]):
        """Save alerts to disk."""
        alerts_file = self.tracking_dir / 'alerts.jsonl'
        
        with open(alerts_file, 'a') as f:
            for alert in alerts:
                f.write(json.dumps(alert.to_dict()) + '\n')
    
    def _load_tracking_data(self):
        """Load existing tracking data from disk."""
        if not self.tracking_dir.exists():
            return
        
        # Load snapshots
        for model_dir in self.tracking_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_id = model_dir.name
            self.snapshots[model_id] = []
            
            for snapshot_file in sorted(model_dir.glob('snapshot_*.json')):
                try:
                    with open(snapshot_file, 'r') as f:
                        data = json.load(f)
                    snapshot = PerformanceSnapshot.from_dict(data)
                    self.snapshots[model_id].append(snapshot)
                except Exception as e:
                    logger.warning(f"Failed to load snapshot {snapshot_file}: {str(e)}")
        
        # Load alerts
        alerts_file = self.tracking_dir / 'alerts.jsonl'
        if alerts_file.exists():
            with open(alerts_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        alert = RegressionAlert(**data)
                        self.alerts.append(alert)
                    except Exception as e:
                        logger.warning(f"Failed to load alert: {str(e)}")
        
        logger.info(f"Loaded tracking data for {len(self.snapshots)} models")
