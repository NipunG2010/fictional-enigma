"""
Walk-forward validation framework for backtesting.

This module implements rolling window validation with model retraining to assess
out-of-sample performance and model stability over time.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

import pandas as pd
import numpy as np
from scipy import stats

from .config import WalkForwardConfig
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .portfolio_state import PortfolioSnapshot
from .trade_generator import Order


logger = logging.getLogger(__name__)


@dataclass
class ValidationWindow:
    """Represents a single walk-forward validation window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Performance metrics
    train_metrics: Optional[PerformanceMetrics] = None
    test_metrics: Optional[PerformanceMetrics] = None
    
    # Model information
    model_version: Optional[str] = None
    retrained: bool = False
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'window_id': self.window_id,
            'train_start': self.train_start.isoformat(),
            'train_end': self.train_end.isoformat(),
            'test_start': self.test_start.isoformat(),
            'test_end': self.test_end.isoformat(),
            'train_metrics': self.train_metrics.to_dict() if self.train_metrics else None,
            'test_metrics': self.test_metrics.to_dict() if self.test_metrics else None,
            'model_version': self.model_version,
            'retrained': self.retrained,
            **self.metadata
        }


@dataclass
class ValidationReport:
    """Comprehensive walk-forward validation report."""
    windows: List[ValidationWindow]
    
    # Aggregate statistics
    avg_in_sample_return: float
    avg_out_of_sample_return: float
    performance_degradation: float
    
    # Statistical tests
    in_sample_vs_out_sample_ttest: Dict[str, float]
    stability_score: float
    
    # Retraining analysis
    num_retrains: int
    retrain_windows: List[int]
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'windows': [w.to_dict() for w in self.windows],
            'avg_in_sample_return': self.avg_in_sample_return,
            'avg_out_of_sample_return': self.avg_out_of_sample_return,
            'performance_degradation': self.performance_degradation,
            'in_sample_vs_out_sample_ttest': self.in_sample_vs_out_sample_ttest,
            'stability_score': self.stability_score,
            'num_retrains': self.num_retrains,
            'retrain_windows': self.retrain_windows,
            **self.metadata
        }
    
    def save_to_json(self, output_path: Path) -> None:
        """Save report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved validation report to {output_path}")


class WalkForwardValidator:
    """
    Implement rolling window validation with model retraining.
    
    This class performs walk-forward analysis by:
    1. Splitting data into rolling train/test windows
    2. Training models on each training window
    3. Testing on subsequent out-of-sample periods
    4. Tracking performance degradation and triggering retraining
    5. Generating comprehensive validation reports
    
    Requirements: 5.1, 5.2, 5.3, 5.4
    """
    
    def __init__(self, config: WalkForwardConfig):
        """
        Initialize WalkForwardValidator.
        
        Args:
            config: Walk-forward validation configuration
        """
        self.config = config
        self.windows: List[ValidationWindow] = []
        
        logger.info(f"Initialized WalkForwardValidator with train={config.train_period}, "
                   f"test={config.test_period}, step={config.step_size}")
    
    def setup_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[ValidationWindow]:
        """
        Create rolling validation windows with configurable train/test periods.
        
        Args:
            start_date: Start date for validation
            end_date: End date for validation
            
        Returns:
            List of validation windows
            
        Requirements: 5.1
        """
        logger.info(f"Setting up walk-forward windows from {start_date} to {end_date}")
        
        # Parse period strings to timedeltas
        train_delta = self._parse_period(self.config.train_period)
        test_delta = self._parse_period(self.config.test_period)
        step_delta = self._parse_period(self.config.step_size)
        
        windows = []
        window_id = 0
        current_start = start_date
        
        while True:
            # Calculate window boundaries
            train_start = current_start
            train_end = train_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta
            
            # Check if we've exceeded the end date
            if test_end > end_date:
                logger.info(f"Reached end date, created {len(windows)} windows")
                break
            
            # Create window
            window = ValidationWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            
            windows.append(window)
            logger.debug(f"Created window {window_id}: train=[{train_start}, {train_end}], "
                        f"test=[{test_start}, {test_end}]")
            
            # Move to next window
            current_start += step_delta
            window_id += 1
        
        self.windows = windows
        logger.info(f"Created {len(windows)} validation windows")
        
        return windows
    
    def _parse_period(self, period_str: str) -> timedelta:
        """
        Parse period string to timedelta.
        
        Supports formats like: '1W', '2W', '1M', '3M', '6M', '1Y'
        
        Args:
            period_str: Period string (e.g., '6M', '1W')
            
        Returns:
            Timedelta object
        """
        period_str = period_str.strip().upper()
        
        # Extract number and unit
        if len(period_str) < 2:
            raise ValueError(f"Invalid period format: {period_str}")
        
        try:
            value = int(period_str[:-1])
            unit = period_str[-1]
        except ValueError:
            raise ValueError(f"Invalid period format: {period_str}")
        
        # Convert to timedelta
        if unit == 'D':
            return timedelta(days=value)
        elif unit == 'W':
            return timedelta(weeks=value)
        elif unit == 'M':
            # Approximate month as 30 days
            return timedelta(days=value * 30)
        elif unit == 'Y':
            # Approximate year as 365 days
            return timedelta(days=value * 365)
        else:
            raise ValueError(f"Unsupported period unit: {unit}")
    
    def validate_temporal_separation(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> bool:
        """
        Validate temporal separation to prevent look-ahead bias.
        
        Ensures that all training data comes before all testing data.
        
        Args:
            train_data: Training dataset
            test_data: Testing dataset
            timestamp_col: Name of timestamp column
            
        Returns:
            True if temporal separation is valid
            
        Requirements: 5.2
        """
        if train_data.empty or test_data.empty:
            logger.warning("Empty dataset provided for temporal validation")
            return False
        
        # Get max training timestamp and min testing timestamp
        max_train_time = train_data[timestamp_col].max()
        min_test_time = test_data[timestamp_col].min()
        
        # Check temporal separation
        is_valid = max_train_time < min_test_time
        
        if not is_valid:
            logger.error(f"Temporal separation violated: max_train={max_train_time}, "
                        f"min_test={min_test_time}")
        else:
            logger.debug(f"Temporal separation validated: gap={(min_test_time - max_train_time).total_seconds() / 3600:.1f} hours")
        
        return is_valid
    
    def check_retraining_needed(
        self,
        current_performance: PerformanceMetrics,
        baseline_performance: PerformanceMetrics
    ) -> bool:
        """
        Determine if model retraining is needed based on performance degradation.
        
        Args:
            current_performance: Current window performance
            baseline_performance: Baseline performance to compare against
            
        Returns:
            True if retraining is recommended
            
        Requirements: 5.3, 5.4
        """
        # Calculate performance degradation
        baseline_return = baseline_performance.total_return
        current_return = current_performance.total_return
        
        if baseline_return == 0:
            logger.warning("Baseline return is zero, cannot calculate degradation")
            return False
        
        degradation = (baseline_return - current_return) / abs(baseline_return)
        
        # Check if degradation exceeds threshold
        needs_retrain = degradation > self.config.retrain_threshold
        
        if needs_retrain:
            logger.warning(f"Performance degradation detected: {degradation:.2%} "
                          f"(threshold: {self.config.retrain_threshold:.2%})")
        else:
            logger.debug(f"Performance degradation: {degradation:.2%} (within threshold)")
        
        return needs_retrain
    
    def run_validation(
        self,
        backtest_func: Callable[[datetime, datetime], Tuple[List[PortfolioSnapshot], List[Order]]],
        retrain_func: Optional[Callable[[datetime, datetime], str]] = None,
        performance_analyzer: Optional[PerformanceAnalyzer] = None,
        initial_capital: float = 100000.0
    ) -> ValidationReport:
        """
        Run complete walk-forward validation.
        
        Args:
            backtest_func: Function that runs backtest for a date range
                          Returns (snapshots, orders)
            retrain_func: Optional function that retrains model for a date range
                         Returns model version string
            performance_analyzer: Performance analyzer instance
            initial_capital: Initial capital for backtesting
            
        Returns:
            Validation report with all windows and statistics
            
        Requirements: 5.1, 5.2, 5.3, 5.4
        """
        if not self.windows:
            raise ValueError("No validation windows configured. Call setup_windows() first.")
        
        if performance_analyzer is None:
            from .config import PerformanceConfig
            performance_analyzer = PerformanceAnalyzer(PerformanceConfig())
        
        logger.info(f"Starting walk-forward validation with {len(self.windows)} windows")
        
        baseline_performance = None
        retrain_windows = []
        
        for window in self.windows:
            logger.info(f"Processing window {window.window_id}")
            
            # Check if retraining is needed
            if retrain_func and (window.window_id == 0 or 
                                (baseline_performance and 
                                 self.check_retraining_needed(window.test_metrics or baseline_performance, 
                                                              baseline_performance))):
                logger.info(f"Retraining model for window {window.window_id}")
                try:
                    model_version = retrain_func(window.train_start, window.train_end)
                    window.model_version = model_version
                    window.retrained = True
                    retrain_windows.append(window.window_id)
                except Exception as e:
                    logger.error(f"Model retraining failed: {e}")
                    window.retrained = False
            
            # Run backtest on training period
            try:
                train_snapshots, train_orders = backtest_func(window.train_start, window.train_end)
                
                # Validate minimum samples
                if len(train_snapshots) < self.config.min_train_samples:
                    logger.warning(f"Window {window.window_id} has insufficient training samples: "
                                 f"{len(train_snapshots)} < {self.config.min_train_samples}")
                    continue
                
                # Calculate training metrics
                window.train_metrics = performance_analyzer.calculate_metrics(
                    train_snapshots, train_orders, initial_capital
                )
                
                logger.info(f"Window {window.window_id} training: "
                          f"Return={window.train_metrics.total_return:.2%}, "
                          f"Sharpe={window.train_metrics.sharpe_ratio:.2f}")
                
            except Exception as e:
                logger.error(f"Training backtest failed for window {window.window_id}: {e}")
                continue
            
            # Run backtest on testing period
            try:
                test_snapshots, test_orders = backtest_func(window.test_start, window.test_end)
                
                # Calculate testing metrics
                window.test_metrics = performance_analyzer.calculate_metrics(
                    test_snapshots, test_orders, initial_capital
                )
                
                logger.info(f"Window {window.window_id} testing: "
                          f"Return={window.test_metrics.total_return:.2%}, "
                          f"Sharpe={window.test_metrics.sharpe_ratio:.2f}")
                
                # Update baseline if this is the first window or after retraining
                if baseline_performance is None or window.retrained:
                    baseline_performance = window.test_metrics
                
            except Exception as e:
                logger.error(f"Testing backtest failed for window {window.window_id}: {e}")
                continue
        
        # Generate validation report
        report = self._generate_report(retrain_windows)
        
        logger.info(f"Walk-forward validation complete: {len(self.windows)} windows, "
                   f"{report.num_retrains} retrains")
        
        return report
    
    def _generate_report(self, retrain_windows: List[int]) -> ValidationReport:
        """
        Generate comprehensive validation report.
        
        Requirements: 5.4, 5.5
        """
        # Filter windows with both train and test metrics
        valid_windows = [w for w in self.windows if w.train_metrics and w.test_metrics]
        
        if not valid_windows:
            logger.warning("No valid windows for report generation")
            return ValidationReport(
                windows=self.windows,
                avg_in_sample_return=0.0,
                avg_out_of_sample_return=0.0,
                performance_degradation=0.0,
                in_sample_vs_out_sample_ttest={},
                stability_score=0.0,
                num_retrains=len(retrain_windows),
                retrain_windows=retrain_windows
            )
        
        # Calculate aggregate statistics
        in_sample_returns = [w.train_metrics.total_return for w in valid_windows]
        out_of_sample_returns = [w.test_metrics.total_return for w in valid_windows]
        
        avg_in_sample = np.mean(in_sample_returns)
        avg_out_of_sample = np.mean(out_of_sample_returns)
        
        # Performance degradation
        if avg_in_sample != 0:
            degradation = (avg_in_sample - avg_out_of_sample) / abs(avg_in_sample)
        else:
            degradation = 0.0
        
        # Statistical significance test (paired t-test)
        if len(in_sample_returns) > 1:
            t_stat, p_value = stats.ttest_rel(in_sample_returns, out_of_sample_returns)
            ttest_results = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        else:
            ttest_results = {
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant': False
            }
        
        # Stability score (coefficient of variation of out-of-sample returns)
        if avg_out_of_sample != 0:
            stability_score = 1.0 - (np.std(out_of_sample_returns) / abs(avg_out_of_sample))
            stability_score = max(0.0, min(1.0, stability_score))  # Clamp to [0, 1]
        else:
            stability_score = 0.0
        
        report = ValidationReport(
            windows=self.windows,
            avg_in_sample_return=avg_in_sample,
            avg_out_of_sample_return=avg_out_of_sample,
            performance_degradation=degradation,
            in_sample_vs_out_sample_ttest=ttest_results,
            stability_score=stability_score,
            num_retrains=len(retrain_windows),
            retrain_windows=retrain_windows,
            metadata={
                'num_valid_windows': len(valid_windows),
                'num_total_windows': len(self.windows)
            }
        )
        
        logger.info(f"Generated validation report: "
                   f"In-sample={avg_in_sample:.2%}, "
                   f"Out-of-sample={avg_out_of_sample:.2%}, "
                   f"Degradation={degradation:.2%}, "
                   f"Stability={stability_score:.2f}")
        
        return report
    
    def get_window_by_id(self, window_id: int) -> Optional[ValidationWindow]:
        """Get validation window by ID."""
        for window in self.windows:
            if window.window_id == window_id:
                return window
        return None
    
    def get_windows_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[ValidationWindow]:
        """Get all windows that overlap with the specified date range."""
        overlapping_windows = []
        
        for window in self.windows:
            # Check if window overlaps with date range
            if (window.train_start <= end_date and window.test_end >= start_date):
                overlapping_windows.append(window)
        
        return overlapping_windows
