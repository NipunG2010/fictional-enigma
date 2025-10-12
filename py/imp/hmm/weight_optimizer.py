"""
Fusion weight optimization for HMM-based signal combination.

This module implements per-state weight optimization using Sharpe ratio
maximization with support for multiple optimization methods, as well as
comprehensive validation of optimized weights and walk-forward validation
for robustness testing.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import warnings


@dataclass
class OptimizationConfig:
    """Configuration for weight optimization.
    
    Attributes:
        method: Optimization method ('SLSQP' or 'grid_search')
        risk_free_rate: Annualized risk-free rate for Sharpe calculation
        min_weight: Minimum weight constraint per signal
        max_weight: Maximum weight constraint per signal
        grid_points: Number of grid points per dimension for grid search
        min_observations: Minimum observations required per state
    """
    method: str = "SLSQP"
    risk_free_rate: float = 0.02
    min_weight: float = 0.0
    max_weight: float = 1.0
    grid_points: int = 11
    min_observations: int = 30


class StateWeightOptimizer:
    """Optimize fusion weights per HMM state using Sharpe ratio.
    
    This class implements weight optimization for combining multiple signals
    within each market regime (HMM state). It supports both scipy-based
    constrained optimization and exhaustive grid search.
    
    The optimization objective is to maximize the Sharpe ratio of the
    combined signal strategy, subject to constraints:
    - Weights sum to 1.0
    - Weights are non-negative (long-only)
    - Optional min/max bounds per weight
    """
    
    def __init__(self, config: OptimizationConfig):
        """Initialize optimizer with configuration.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        
    def optimize_state_weights(
        self,
        state_returns: np.ndarray,
        state_signals: np.ndarray,
        signal_names: List[str] = ['s_LDC', 's_MR', 's_TSMOM']
    ) -> Tuple[Dict[str, float], float]:
        """Optimize weights for a single state.
        
        Args:
            state_returns: Returns when this state is active, shape (T,)
            state_signals: Signal values when state is active, shape (T, n_signals)
            signal_names: Names of the signals
            
        Returns:
            Tuple of (optimal_weights_dict, achieved_sharpe)
            
        Raises:
            ValueError: If optimization method is unknown
        """
        # Check minimum data requirement
        if len(state_returns) < self.config.min_observations:
            warnings.warn(
                f"Insufficient data for optimization: {len(state_returns)} < "
                f"{self.config.min_observations}. Using equal weights."
            )
            return self._equal_weights(signal_names), 0.0
        
        # Check for valid data
        if np.any(np.isnan(state_returns)) or np.any(np.isnan(state_signals)):
            warnings.warn("NaN values detected in data. Using equal weights.")
            return self._equal_weights(signal_names), 0.0
            
        if np.std(state_returns) == 0:
            warnings.warn("Zero variance in returns. Using equal weights.")
            return self._equal_weights(signal_names), 0.0
        
        # Route to appropriate optimization method
        if self.config.method == "SLSQP":
            return self._optimize_scipy(state_returns, state_signals, signal_names)
        elif self.config.method == "grid_search":
            return self._optimize_grid(state_returns, state_signals, signal_names)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}")
    
    def _optimize_scipy(
        self,
        returns: np.ndarray,
        signals: np.ndarray,
        signal_names: List[str]
    ) -> Tuple[Dict[str, float], float]:
        """Optimize using scipy SLSQP constrained optimization.
        
        Args:
            returns: Historical returns, shape (T,)
            signals: Signal values, shape (T, n_signals)
            signal_names: Names of the signals
            
        Returns:
            Tuple of (optimal_weights_dict, achieved_sharpe)
        """
        n_signals = len(signal_names)
        
        def objective(weights):
            """Negative Sharpe ratio (minimize negative = maximize positive)."""
            portfolio_returns = self._compute_portfolio_returns(weights, signals, returns)
            sharpe = self._calculate_sharpe(portfolio_returns)
            return -sharpe
        
        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds: [min_weight, max_weight] for each weight
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_signals
        
        # Initial guess: equal weights
        x0 = np.array([1.0 / n_signals] * n_signals)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-9}
        )
        
        if not result.success:
            warnings.warn(
                f"Optimization did not converge: {result.message}. "
                f"Using equal weights."
            )
            return self._equal_weights(signal_names), 0.0
        
        optimal_weights = result.x
        achieved_sharpe = -result.fun
        
        # Normalize to ensure exact sum to 1 (numerical precision)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Convert to dict
        weights_dict = {
            name: float(w) for name, w in zip(signal_names, optimal_weights)
        }
        
        return weights_dict, achieved_sharpe
    
    def _optimize_grid(
        self,
        returns: np.ndarray,
        signals: np.ndarray,
        signal_names: List[str]
    ) -> Tuple[Dict[str, float], float]:
        """Optimize using exhaustive grid search.
        
        For 3 signals, generates a 2D grid where the third weight is
        determined by the constraint that weights sum to 1.
        
        Args:
            returns: Historical returns, shape (T,)
            signals: Signal values, shape (T, n_signals)
            signal_names: Names of the signals
            
        Returns:
            Tuple of (optimal_weights_dict, achieved_sharpe)
        """
        n_signals = len(signal_names)
        grid_points = self.config.grid_points
        
        if n_signals != 3:
            raise NotImplementedError(
                f"Grid search currently only supports 3 signals, got {n_signals}"
            )
        
        best_sharpe = -np.inf
        best_weights = None
        
        # Generate grid for first two weights (third is determined by constraint)
        for w1 in np.linspace(self.config.min_weight, self.config.max_weight, grid_points):
            for w2 in np.linspace(self.config.min_weight, self.config.max_weight, grid_points):
                w3 = 1.0 - w1 - w2
                
                # Check if third weight satisfies bounds
                if w3 < self.config.min_weight or w3 > self.config.max_weight:
                    continue
                
                weights = np.array([w1, w2, w3])
                
                # Compute Sharpe ratio for this weight combination
                portfolio_returns = self._compute_portfolio_returns(weights, signals, returns)
                sharpe = self._calculate_sharpe(portfolio_returns)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = weights
        
        if best_weights is None:
            warnings.warn(
                "Grid search found no valid weight combinations. Using equal weights."
            )
            return self._equal_weights(signal_names), 0.0
        
        # Normalize to ensure exact sum to 1
        best_weights = best_weights / np.sum(best_weights)
        
        weights_dict = {
            name: float(w) for name, w in zip(signal_names, best_weights)
        }
        
        return weights_dict, best_sharpe
    
    def _compute_portfolio_returns(
        self,
        weights: np.ndarray,
        signals: np.ndarray,
        returns: np.ndarray
    ) -> np.ndarray:
        """Compute portfolio returns given weights and signals.
        
        Strategy logic:
        1. Combine signals using weighted sum
        2. Take position based on sign of combined signal
        3. Portfolio return = position * actual return
        
        Args:
            weights: Signal weights, shape (n_signals,)
            signals: Signal values, shape (T, n_signals)
            returns: Actual returns, shape (T,)
            
        Returns:
            Portfolio returns, shape (T,)
        """
        # Combined signal = weighted sum of individual signals
        combined_signal = signals @ weights
        
        # Position: +1 if signal positive, -1 if negative, 0 if zero
        positions = np.sign(combined_signal)
        
        # Portfolio return = position * actual return
        portfolio_returns = positions * returns
        
        return portfolio_returns
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio.
        
        Assumes daily returns and annualizes using 252 trading days.
        
        Args:
            returns: Portfolio returns, shape (T,)
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        
        # Annualize (assuming daily returns)
        annual_return = mean_return * 252
        annual_std = std_return * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (annual_return - self.config.risk_free_rate) / annual_std
        
        return sharpe
    
    def _equal_weights(self, signal_names: List[str]) -> Dict[str, float]:
        """Return equal weights as fallback.
        
        Args:
            signal_names: Names of the signals
            
        Returns:
            Dictionary with equal weights for all signals
        """
        n = len(signal_names)
        return {name: 1.0 / n for name in signal_names}



class WeightValidator:
    """Validate fusion weights and compare performance against baseline.
    
    This class provides comprehensive validation of optimized fusion weights:
    - Constraint validation (sum to 1, non-negative, bounds)
    - Performance comparison vs equal-weight baseline
    - Statistical significance testing
    - Robustness metrics
    
    The validator ensures that optimized weights meet all requirements and
    actually improve performance over naive strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize validator.
        
        Args:
            risk_free_rate: Annualized risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
        self.validation_results = {}
    
    def validate_weights(
        self,
        state_weights: List[Dict[str, float]],
        observations: np.ndarray,
        returns: np.ndarray,
        state_sequence: np.ndarray,
        signal_names: List[str] = ['s_LDC', 's_MR', 's_TSMOM']
    ) -> Dict[str, Any]:
        """Comprehensive validation of fusion weights.
        
        Performs three types of validation:
        1. Constraint validation (mathematical requirements)
        2. Performance comparison (vs baseline)
        3. Statistical significance testing
        
        Args:
            state_weights: List of weight dicts per state
            observations: Signal observations, shape (T, n_signals)
            returns: Actual returns, shape (T,)
            state_sequence: HMM state sequence, shape (T,)
            signal_names: Names of the signals
            
        Returns:
            Comprehensive validation report dictionary with:
            - constraints_valid: bool
            - constraint_errors: List[str]
            - performance_comparison: Dict with metrics
            - statistical_tests: Dict with significance tests
            - recommendation: str
        """
        validation_report = {
            'constraints_valid': True,
            'constraint_errors': [],
            'performance_comparison': {},
            'statistical_tests': {},
            'recommendation': ''
        }
        
        # 1. Validate constraints
        constraint_check = self._validate_constraints(state_weights, signal_names)
        validation_report['constraints_valid'] = constraint_check['valid']
        validation_report['constraint_errors'] = constraint_check['errors']
        
        # If constraints fail, stop here
        if not constraint_check['valid']:
            validation_report['recommendation'] = (
                "REJECT: Weights violate constraints. Fix errors before proceeding."
            )
            return validation_report
        
        # 2. Compare performance vs baseline
        perf_comparison = self._compare_performance(
            state_weights,
            observations,
            returns,
            state_sequence,
            signal_names
        )
        validation_report['performance_comparison'] = perf_comparison
        
        # 3. Statistical significance testing
        if perf_comparison['optimized_sharpe'] > perf_comparison['baseline_sharpe']:
            sig_test = self._test_significance(
                perf_comparison['optimized_returns'],
                perf_comparison['baseline_returns']
            )
            validation_report['statistical_tests'] = sig_test
            
            # Generate recommendation
            if sig_test['significant_at_5pct']:
                validation_report['recommendation'] = (
                    f"ACCEPT: Optimized weights show statistically significant "
                    f"improvement ({perf_comparison['improvement_pct']:.2f}% better Sharpe, "
                    f"p={sig_test['p_value']:.4f})"
                )
            else:
                validation_report['recommendation'] = (
                    f"CAUTION: Improvement not statistically significant "
                    f"(p={sig_test['p_value']:.4f}). Consider using equal weights or "
                    f"gathering more data."
                )
        else:
            validation_report['statistical_tests'] = {
                'note': 'Optimized weights underperform baseline, no significance test needed'
            }
            validation_report['recommendation'] = (
                f"REJECT: Optimized weights underperform equal-weight baseline "
                f"({perf_comparison['improvement_pct']:.2f}% worse). Use equal weights instead."
            )
        
        return validation_report
    
    def _validate_constraints(
        self,
        state_weights: List[Dict[str, float]],
        signal_names: List[str]
    ) -> Dict[str, Any]:
        """Validate weight constraints.
        
        Checks:
        - All states have weights for all signals
        - Weights sum to 1.0 (within tolerance)
        - All weights are non-negative
        - Weights are within [0, 1] bounds
        
        Args:
            state_weights: List of weight dicts per state
            signal_names: Expected signal names
            
        Returns:
            Dictionary with 'valid' (bool) and 'errors' (List[str])
        """
        errors = []
        
        if not state_weights:
            errors.append("No state weights provided")
            return {'valid': False, 'errors': errors}
        
        for i, state_weight in enumerate(state_weights):
            # Check all signals present
            missing_signals = set(signal_names) - set(state_weight.keys())
            if missing_signals:
                errors.append(
                    f"State {i}: missing weights for signals {missing_signals}"
                )
            
            extra_signals = set(state_weight.keys()) - set(signal_names)
            if extra_signals:
                errors.append(
                    f"State {i}: unexpected signals {extra_signals}"
                )
            
            # Check sum to 1
            weight_sum = sum(state_weight.values())
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                errors.append(
                    f"State {i}: weights sum to {weight_sum:.6f}, not 1.0 "
                    f"(difference: {abs(weight_sum - 1.0):.6e})"
                )
            
            # Check non-negative and within bounds
            for signal, weight in state_weight.items():
                if weight < 0:
                    errors.append(
                        f"State {i}, {signal}: negative weight {weight:.6f}"
                    )
                if weight > 1.0:
                    errors.append(
                        f"State {i}, {signal}: weight {weight:.6f} exceeds 1.0"
                    )
                if np.isnan(weight) or np.isinf(weight):
                    errors.append(
                        f"State {i}, {signal}: invalid weight {weight}"
                    )
        
        return {'valid': len(errors) == 0, 'errors': errors}
    
    def _compare_performance(
        self,
        state_weights: List[Dict[str, float]],
        observations: np.ndarray,
        returns: np.ndarray,
        state_sequence: np.ndarray,
        signal_names: List[str]
    ) -> Dict[str, Any]:
        """Compare optimized weights vs equal-weight baseline.
        
        Computes comprehensive performance metrics for both strategies:
        - Sharpe ratio
        - Total return
        - Volatility
        - Max drawdown
        - Win rate
        
        Args:
            state_weights: Optimized weights per state
            observations: Signal observations, shape (T, n_signals)
            returns: Actual returns, shape (T,)
            state_sequence: HMM state sequence, shape (T,)
            signal_names: Names of the signals
            
        Returns:
            Dictionary with performance metrics for both strategies
        """
        # Compute returns with optimized weights
        optimized_returns = self._compute_strategy_returns(
            state_weights,
            observations,
            returns,
            state_sequence,
            signal_names
        )
        
        # Compute returns with equal weights
        n_states = len(state_weights)
        n_signals = len(signal_names)
        equal_weight = 1.0 / n_signals
        equal_weights = [
            {signal: equal_weight for signal in signal_names}
            for _ in range(n_states)
        ]
        baseline_returns = self._compute_strategy_returns(
            equal_weights,
            observations,
            returns,
            state_sequence,
            signal_names
        )
        
        # Calculate metrics for both strategies
        opt_metrics = self._calculate_metrics(optimized_returns)
        base_metrics = self._calculate_metrics(baseline_returns)
        
        # Calculate improvements
        sharpe_improvement = opt_metrics['sharpe'] - base_metrics['sharpe']
        sharpe_improvement_pct = (
            (sharpe_improvement / abs(base_metrics['sharpe']) * 100)
            if base_metrics['sharpe'] != 0 else 0.0
        )
        
        return {
            'optimized_sharpe': opt_metrics['sharpe'],
            'baseline_sharpe': base_metrics['sharpe'],
            'optimized_total_return': opt_metrics['total_return'],
            'baseline_total_return': base_metrics['total_return'],
            'optimized_volatility': opt_metrics['volatility'],
            'baseline_volatility': base_metrics['volatility'],
            'optimized_max_drawdown': opt_metrics['max_drawdown'],
            'baseline_max_drawdown': base_metrics['max_drawdown'],
            'optimized_win_rate': opt_metrics['win_rate'],
            'baseline_win_rate': base_metrics['win_rate'],
            'improvement': sharpe_improvement,
            'improvement_pct': sharpe_improvement_pct,
            'optimized_returns': optimized_returns,
            'baseline_returns': baseline_returns
        }
    
    def _compute_strategy_returns(
        self,
        state_weights: List[Dict[str, float]],
        observations: np.ndarray,
        returns: np.ndarray,
        state_sequence: np.ndarray,
        signal_names: List[str]
    ) -> np.ndarray:
        """Compute strategy returns given state-dependent weights.
        
        For each time step:
        1. Determine active state
        2. Get weights for that state
        3. Combine signals using weights
        4. Take position based on combined signal
        5. Compute return
        
        Args:
            state_weights: Weights per state
            observations: Signal observations, shape (T, n_signals)
            returns: Actual returns, shape (T,)
            state_sequence: HMM state sequence, shape (T,)
            signal_names: Names of the signals
            
        Returns:
            Portfolio returns, shape (T,)
        """
        T = len(returns)
        portfolio_returns = np.zeros(T)
        
        for t in range(T):
            state = state_sequence[t]
            weights = state_weights[state]
            
            # Combined signal (weighted sum)
            combined = 0.0
            for i, signal_name in enumerate(signal_names):
                combined += weights[signal_name] * observations[t, i]
            
            # Position based on signal
            position = np.sign(combined)
            
            # Portfolio return
            portfolio_returns[t] = position * returns[t]
        
        return portfolio_returns
    
    def _calculate_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics.
        
        Args:
            returns: Portfolio returns, shape (T,)
            
        Returns:
            Dictionary with performance metrics
        """
        if len(returns) == 0:
            return {
                'sharpe': 0.0,
                'total_return': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        # Sharpe ratio
        sharpe = self._calculate_sharpe(returns)
        
        # Total return (cumulative)
        total_return = np.sum(returns)
        
        # Volatility (annualized)
        volatility = np.std(returns, ddof=1) * np.sqrt(252)
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0
        
        return {
            'sharpe': float(sharpe),
            'total_return': float(total_return),
            'volatility': float(volatility),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate)
        }
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio.
        
        Args:
            returns: Portfolio returns, shape (T,)
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        
        # Annualize (assuming daily returns)
        annual_return = mean_return * 252
        annual_std = std_return * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (annual_return - self.risk_free_rate) / annual_std
        
        return sharpe
    
    def _test_significance(
        self,
        optimized_returns: np.ndarray,
        baseline_returns: np.ndarray
    ) -> Dict[str, Any]:
        """Test statistical significance of performance difference.
        
        Uses paired t-test to determine if the difference in returns
        between optimized and baseline strategies is statistically
        significant.
        
        Args:
            optimized_returns: Returns from optimized strategy, shape (T,)
            baseline_returns: Returns from baseline strategy, shape (T,)
            
        Returns:
            Dictionary with test results:
            - t_statistic: t-test statistic
            - p_value: two-tailed p-value
            - significant_at_5pct: bool
            - significant_at_1pct: bool
            - degrees_of_freedom: int
            - mean_difference: float
        """
        # Paired t-test (tests if mean difference is significantly different from 0)
        t_stat, p_value = stats.ttest_rel(optimized_returns, baseline_returns)
        
        # Mean difference
        mean_diff = np.mean(optimized_returns - baseline_returns)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_5pct': bool(p_value < 0.05),
            'significant_at_1pct': bool(p_value < 0.01),
            'degrees_of_freedom': len(optimized_returns) - 1,
            'mean_difference': float(mean_diff),
            'interpretation': (
                f"The optimized strategy {'significantly' if p_value < 0.05 else 'does not significantly'} "
                f"outperform the baseline (p={p_value:.4f})"
            )
        }



@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.
    
    Attributes:
        n_folds: Number of validation folds
        train_ratio: Ratio of data to use for training in each fold (0-1)
        min_train_size: Minimum training samples required
        min_test_size: Minimum test samples required
        overfitting_threshold: Threshold for flagging overfitting (out-of-sample degradation %)
    """
    n_folds: int = 5
    train_ratio: float = 0.7
    min_train_size: int = 100
    min_test_size: int = 30
    overfitting_threshold: float = 50.0  # Flag if out-of-sample Sharpe drops >50%


class WalkForwardValidator:
    """Walk-forward validation for testing weight optimization robustness.
    
    This class implements time-series cross-validation to assess whether
    optimized weights generalize to out-of-sample data. It helps detect
    overfitting by comparing in-sample vs out-of-sample performance.
    
    The walk-forward approach:
    1. Split data into sequential folds
    2. For each fold: train on past data, test on future data
    3. Compare in-sample vs out-of-sample Sharpe ratios
    4. Flag overfitting if out-of-sample performance degrades significantly
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
        optimization_config: OptimizationConfig
    ):
        """Initialize walk-forward validator.
        
        Args:
            config: Walk-forward validation configuration
            optimization_config: Configuration for weight optimization
        """
        self.config = config
        self.optimization_config = optimization_config
        self.optimizer = StateWeightOptimizer(optimization_config)
    
    def validate_robustness(
        self,
        observations: np.ndarray,
        returns: np.ndarray,
        state_sequence: np.ndarray,
        n_states: int,
        signal_names: List[str] = ['s_LDC', 's_MR', 's_TSMOM']
    ) -> Dict[str, Any]:
        """Perform walk-forward validation for weight optimization.
        
        Tests whether optimized weights generalize to out-of-sample data
        by using time-series cross-validation.
        
        Args:
            observations: Signal observations, shape (T, n_signals)
            returns: Actual returns, shape (T,)
            state_sequence: HMM state sequence, shape (T,)
            n_states: Number of HMM states
            signal_names: Names of the signals
            
        Returns:
            Comprehensive robustness report with:
            - fold_results: List of results per fold
            - aggregate_metrics: Overall performance metrics
            - overfitting_detected: bool
            - recommendation: str
        """
        T = len(returns)
        
        # Check minimum data requirements
        if T < self.config.min_train_size + self.config.min_test_size:
            return {
                'error': f'Insufficient data for walk-forward validation: {T} samples',
                'min_required': self.config.min_train_size + self.config.min_test_size,
                'overfitting_detected': False,
                'recommendation': 'Gather more data before performing walk-forward validation'
            }
        
        # Generate fold splits
        fold_splits = self._generate_fold_splits(T)
        
        if len(fold_splits) == 0:
            return {
                'error': 'Could not generate valid fold splits',
                'overfitting_detected': False,
                'recommendation': 'Adjust walk-forward configuration or gather more data'
            }
        
        # Run validation for each fold
        fold_results = []
        for fold_idx, (train_indices, test_indices) in enumerate(fold_splits):
            fold_result = self._validate_fold(
                fold_idx,
                train_indices,
                test_indices,
                observations,
                returns,
                state_sequence,
                n_states,
                signal_names
            )
            fold_results.append(fold_result)
        
        # Aggregate results across folds
        aggregate_metrics = self._aggregate_fold_results(fold_results)
        
        # Detect overfitting
        overfitting_detected, overfitting_details = self._detect_overfitting(
            fold_results,
            aggregate_metrics
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            aggregate_metrics,
            overfitting_detected,
            overfitting_details
        )
        
        return {
            'fold_results': fold_results,
            'aggregate_metrics': aggregate_metrics,
            'overfitting_detected': overfitting_detected,
            'overfitting_details': overfitting_details,
            'recommendation': recommendation,
            'n_folds': len(fold_results),
            'config': {
                'n_folds': self.config.n_folds,
                'train_ratio': self.config.train_ratio,
                'overfitting_threshold': self.config.overfitting_threshold
            }
        }
    
    def _generate_fold_splits(self, T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time-series fold splits for walk-forward validation.
        
        Uses expanding window approach where training set grows and
        test set slides forward in time.
        
        Args:
            T: Total number of samples
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        fold_splits = []
        
        # Calculate fold size
        fold_size = T // self.config.n_folds
        
        if fold_size < self.config.min_test_size:
            warnings.warn(
                f"Fold size {fold_size} is smaller than min_test_size "
                f"{self.config.min_test_size}. Reducing number of folds."
            )
            # Adjust number of folds
            adjusted_n_folds = max(1, T // self.config.min_test_size)
            fold_size = T // adjusted_n_folds
        
        # Generate folds with expanding training window
        for fold_idx in range(1, self.config.n_folds + 1):
            # Test set: next fold_size samples
            test_end = min(fold_idx * fold_size, T)
            test_start = max(0, test_end - fold_size)
            
            # Training set: all data before test set
            train_start = 0
            train_end = test_start
            
            # Check minimum sizes
            train_size = train_end - train_start
            test_size = test_end - test_start
            
            if train_size < self.config.min_train_size:
                continue
            
            if test_size < self.config.min_test_size:
                continue
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            fold_splits.append((train_indices, test_indices))
        
        return fold_splits
    
    def _validate_fold(
        self,
        fold_idx: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        observations: np.ndarray,
        returns: np.ndarray,
        state_sequence: np.ndarray,
        n_states: int,
        signal_names: List[str]
    ) -> Dict[str, Any]:
        """Validate a single fold.
        
        Args:
            fold_idx: Fold index
            train_indices: Training data indices
            test_indices: Test data indices
            observations: Signal observations
            returns: Actual returns
            state_sequence: HMM state sequence
            n_states: Number of HMM states
            signal_names: Names of the signals
            
        Returns:
            Fold validation results
        """
        # Split data
        train_obs = observations[train_indices]
        train_returns = returns[train_indices]
        train_states = state_sequence[train_indices]
        
        test_obs = observations[test_indices]
        test_returns = returns[test_indices]
        test_states = state_sequence[test_indices]
        
        # Optimize weights on training data
        train_weights = []
        train_sharpes = []
        
        for state in range(n_states):
            state_mask = train_states == state
            state_returns = train_returns[state_mask]
            state_signals = train_obs[state_mask]
            
            if len(state_returns) < self.optimization_config.min_observations:
                # Use equal weights for states with insufficient data
                n_signals = len(signal_names)
                weights = {name: 1.0 / n_signals for name in signal_names}
                sharpe = 0.0
            else:
                weights, sharpe = self.optimizer.optimize_state_weights(
                    state_returns,
                    state_signals,
                    signal_names
                )
            
            train_weights.append(weights)
            train_sharpes.append(sharpe)
        
        # Evaluate on training data (in-sample)
        train_portfolio_returns = self._compute_portfolio_returns(
            train_weights,
            train_obs,
            train_returns,
            train_states,
            signal_names
        )
        in_sample_sharpe = self._calculate_sharpe(train_portfolio_returns)
        
        # Evaluate on test data (out-of-sample)
        test_portfolio_returns = self._compute_portfolio_returns(
            train_weights,
            test_obs,
            test_returns,
            test_states,
            signal_names
        )
        out_of_sample_sharpe = self._calculate_sharpe(test_portfolio_returns)
        
        # Calculate degradation
        if in_sample_sharpe != 0:
            degradation_pct = (
                (in_sample_sharpe - out_of_sample_sharpe) / abs(in_sample_sharpe) * 100
            )
        else:
            degradation_pct = 0.0
        
        return {
            'fold_idx': fold_idx,
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'optimized_weights': train_weights,
            'in_sample_sharpe': float(in_sample_sharpe),
            'out_of_sample_sharpe': float(out_of_sample_sharpe),
            'degradation_pct': float(degradation_pct),
            'state_sharpes': [float(s) for s in train_sharpes]
        }
    
    def _compute_portfolio_returns(
        self,
        state_weights: List[Dict[str, float]],
        observations: np.ndarray,
        returns: np.ndarray,
        state_sequence: np.ndarray,
        signal_names: List[str]
    ) -> np.ndarray:
        """Compute portfolio returns given state-dependent weights.
        
        Args:
            state_weights: Weights per state
            observations: Signal observations
            returns: Actual returns
            state_sequence: HMM state sequence
            signal_names: Names of the signals
            
        Returns:
            Portfolio returns
        """
        T = len(returns)
        portfolio_returns = np.zeros(T)
        
        for t in range(T):
            state = state_sequence[t]
            weights = state_weights[state]
            
            # Combined signal
            combined = 0.0
            for i, signal_name in enumerate(signal_names):
                combined += weights[signal_name] * observations[t, i]
            
            # Position and return
            position = np.sign(combined)
            portfolio_returns[t] = position * returns[t]
        
        return portfolio_returns
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio.
        
        Args:
            returns: Portfolio returns
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        
        # Annualize (assuming daily returns)
        annual_return = mean_return * 252
        annual_std = std_return * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (annual_return - self.optimization_config.risk_free_rate) / annual_std
        
        return sharpe
    
    def _aggregate_fold_results(
        self,
        fold_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results across all folds.
        
        Args:
            fold_results: List of fold validation results
            
        Returns:
            Aggregated metrics
        """
        if not fold_results:
            return {}
        
        in_sample_sharpes = [f['in_sample_sharpe'] for f in fold_results]
        out_of_sample_sharpes = [f['out_of_sample_sharpe'] for f in fold_results]
        degradations = [f['degradation_pct'] for f in fold_results]
        
        return {
            'mean_in_sample_sharpe': float(np.mean(in_sample_sharpes)),
            'mean_out_of_sample_sharpe': float(np.mean(out_of_sample_sharpes)),
            'std_in_sample_sharpe': float(np.std(in_sample_sharpes, ddof=1)),
            'std_out_of_sample_sharpe': float(np.std(out_of_sample_sharpes, ddof=1)),
            'mean_degradation_pct': float(np.mean(degradations)),
            'max_degradation_pct': float(np.max(degradations)),
            'min_degradation_pct': float(np.min(degradations)),
            'std_degradation_pct': float(np.std(degradations, ddof=1)),
            'consistency_ratio': float(
                np.mean(out_of_sample_sharpes) / np.mean(in_sample_sharpes)
                if np.mean(in_sample_sharpes) != 0 else 0.0
            )
        }
    
    def _detect_overfitting(
        self,
        fold_results: List[Dict[str, Any]],
        aggregate_metrics: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detect overfitting based on performance degradation.
        
        Args:
            fold_results: List of fold validation results
            aggregate_metrics: Aggregated metrics
            
        Returns:
            Tuple of (overfitting_detected, details)
        """
        if not fold_results or not aggregate_metrics:
            return False, {}
        
        # Check if mean degradation exceeds threshold
        mean_degradation = aggregate_metrics['mean_degradation_pct']
        max_degradation = aggregate_metrics['max_degradation_pct']
        
        overfitting_detected = (
            mean_degradation > self.config.overfitting_threshold or
            max_degradation > self.config.overfitting_threshold * 1.5
        )
        
        # Count folds with significant degradation
        folds_with_degradation = sum(
            1 for f in fold_results
            if f['degradation_pct'] > self.config.overfitting_threshold
        )
        
        # Check for negative out-of-sample Sharpe
        negative_oos_folds = sum(
            1 for f in fold_results
            if f['out_of_sample_sharpe'] < 0
        )
        
        details = {
            'mean_degradation_pct': mean_degradation,
            'max_degradation_pct': max_degradation,
            'threshold': self.config.overfitting_threshold,
            'folds_with_degradation': folds_with_degradation,
            'total_folds': len(fold_results),
            'negative_oos_folds': negative_oos_folds,
            'consistency_ratio': aggregate_metrics['consistency_ratio']
        }
        
        return overfitting_detected, details
    
    def _generate_recommendation(
        self,
        aggregate_metrics: Dict[str, Any],
        overfitting_detected: bool,
        overfitting_details: Dict[str, Any]
    ) -> str:
        """Generate recommendation based on validation results.
        
        Args:
            aggregate_metrics: Aggregated metrics
            overfitting_detected: Whether overfitting was detected
            overfitting_details: Details about overfitting
            
        Returns:
            Recommendation string
        """
        if not aggregate_metrics:
            return "ERROR: No validation results available"
        
        mean_oos_sharpe = aggregate_metrics['mean_out_of_sample_sharpe']
        consistency_ratio = aggregate_metrics['consistency_ratio']
        
        if overfitting_detected:
            return (
                f"⚠️ OVERFITTING DETECTED: Out-of-sample performance degrades by "
                f"{overfitting_details['mean_degradation_pct']:.1f}% on average "
                f"(max: {overfitting_details['max_degradation_pct']:.1f}%). "
                f"{overfitting_details['folds_with_degradation']}/{overfitting_details['total_folds']} "
                f"folds exceed degradation threshold. "
                f"Recommendation: Use simpler optimization method, add regularization, "
                f"or fall back to equal weights."
            )
        
        if mean_oos_sharpe < 0:
            return (
                f"⚠️ POOR GENERALIZATION: Mean out-of-sample Sharpe is negative "
                f"({mean_oos_sharpe:.3f}). Optimized weights do not generalize well. "
                f"Recommendation: Use equal weights or gather more training data."
            )
        
        if consistency_ratio < 0.5:
            return (
                f"⚠️ LOW CONSISTENCY: Out-of-sample performance is only "
                f"{consistency_ratio*100:.1f}% of in-sample performance. "
                f"Recommendation: Consider using more conservative optimization or "
                f"increasing training data size."
            )
        
        if consistency_ratio >= 0.8:
            return (
                f"✅ ROBUST: Optimized weights generalize well with consistency ratio "
                f"of {consistency_ratio:.2f}. Mean out-of-sample Sharpe: {mean_oos_sharpe:.3f}. "
                f"Recommendation: Weights are robust and suitable for production use."
            )
        
        return (
            f"✓ ACCEPTABLE: Optimized weights show reasonable generalization "
            f"(consistency ratio: {consistency_ratio:.2f}, mean OOS Sharpe: {mean_oos_sharpe:.3f}). "
            f"Recommendation: Weights can be used but monitor performance closely."
        )
