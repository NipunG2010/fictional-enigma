# Fusion Weight Optimization Design Document

## Overview

Fusion Weight Optimization implements per-state weight optimization for the three signals [s_LDC, s_MR, s_TSMOM] using risk-adjusted metrics. The design focuses on practical optimization methods, robust validation, and seamless integration with existing HMM infrastructure.

## Architecture

### Current Foundation

```python
# Existing Components
├── FusionWeights model (py/imp/hmm/models.py)          # ✅ Complete
├── compute_state_weights stub (py/imp/hmm/trainer.py) # 🔄 Needs implementation
├── Sharpe calculation (Rust backtesting)               # ✅ Complete
└── Artifact export (py/imp/hmm/artifact_management.py) # ✅ Complete
```

### Enhanced Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           Fusion Weight Optimization Pipeline               │
├─────────────────────────────────────────────────────────────┤
│  State Sequence  │  Returns Data  │  Signal Components      │
│  from HMM        │  Aligned       │  [s_LDC, s_MR, s_TSMOM] │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Per-State Weight Optimizer                     │
├─────────────────────────────────────────────────────────────┤
│  For each state:                                            │
│  • Filter returns where state is active                     │
│  • Optimize weights to maximize Sharpe ratio                │
│  • Apply constraints (sum=1, non-negative, bounds)          │
│  • Validate convergence and performance                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Weight Validation Framework                    │
├─────────────────────────────────────────────────────────────┤
│  • Constraint validation (sum, bounds, non-negative)        │
│  • Performance comparison vs baseline                       │
│  • Statistical significance testing                         │
│  • Robustness checks (walk-forward validation)              │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Weight Optimizer

```python
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration for weight optimization."""
    method: str = "SLSQP"  # 'SLSQP', 'grid_search'
    risk_free_rate: float = 0.02  # Annualized
    min_weight: float = 0.0
    max_weight: float = 1.0
    grid_points: int = 11  # For grid search
    
class StateWeightOptimizer:
    """Optimize fusion weights per HMM state using Sharpe ratio."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def optimize_state_weights(
        self,
        state_returns: np.ndarray,
        state_signals: np.ndarray,
        signal_names: List[str] = ['s_LDC', 's_MR', 's_TSMOM']
    ) -> Tuple[Dict[str, float], float]:
        """
        Optimize weights for a single state.
        
        Args:
            state_returns: Returns when this state is active (T,)
            state_signals: Signal values when state is active (T, 3)
            signal_names: Names of the three signals
            
        Returns:
            (optimal_weights_dict, achieved_sharpe)
        """
        if len(state_returns) < 30:  # Minimum data requirement
            return self._equal_weights(signal_names), 0.0
            
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
        """Optimize using scipy SLSQP."""
        
        def objective(weights):
            """Negative Sharpe ratio (minimize)."""
            portfolio_returns = self._compute_portfolio_returns(weights, signals, returns)
            sharpe = self._calculate_sharpe(portfolio_returns)
            return -sharpe  # Minimize negative = maximize positive
        
        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds: [min_weight, max_weight] for each weight
        bounds = [(self.config.min_weight, self.config.max_weight)] * len(signal_names)
        
        # Initial guess: equal weights
        x0 = np.array([1.0 / len(signal_names)] * len(signal_names))
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
            return self._equal_weights(signal_names), 0.0
        
        optimal_weights = result.x
        achieved_sharpe = -result.fun
        
        # Convert to dict
        weights_dict = {name: float(w) for name, w in zip(signal_names, optimal_weights)}
        
        return weights_dict, achieved_sharpe
    
    def _optimize_grid(
        self,
        returns: np.ndarray,
        signals: np.ndarray,
        signal_names: List[str]
    ) -> Tuple[Dict[str, float], float]:
        """Optimize using grid search."""
        
        # Generate grid of weights that sum to 1
        grid_points = self.config.grid_points
        best_sharpe = -np.inf
        best_weights = None
        
        # For 3 signals, we need 2D grid (third weight determined by constraint)
        for w1 in np.linspace(self.config.min_weight, self.config.max_weight, grid_points):
            for w2 in np.linspace(self.config.min_weight, self.config.max_weight, grid_points):
                w3 = 1.0 - w1 - w2
                
                # Check if valid
                if w3 < self.config.min_weight or w3 > self.config.max_weight:
                    continue
                
                weights = np.array([w1, w2, w3])
                portfolio_returns = self._compute_portfolio_returns(weights, signals, returns)
                sharpe = self._calculate_sharpe(portfolio_returns)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = weights
        
        if best_weights is None:
            return self._equal_weights(signal_names), 0.0
        
        weights_dict = {name: float(w) for name, w in zip(signal_names, best_weights)}
        return weights_dict, best_sharpe
    
    def _compute_portfolio_returns(
        self,
        weights: np.ndarray,
        signals: np.ndarray,
        returns: np.ndarray
    ) -> np.ndarray:
        """
        Compute portfolio returns given weights and signals.
        
        Assumes signals predict next-period returns.
        Combined signal = weighted sum of signals.
        Position = sign(combined_signal).
        Portfolio return = position * actual_return.
        """
        combined_signal = signals @ weights
        positions = np.sign(combined_signal)
        portfolio_returns = positions * returns
        return portfolio_returns
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Annualize (assuming daily returns)
        annual_return = mean_return * 252
        annual_std = std_return * np.sqrt(252)
        
        sharpe = (annual_return - self.config.risk_free_rate) / annual_std
        return sharpe
    
    def _equal_weights(self, signal_names: List[str]) -> Dict[str, float]:
        """Return equal weights as fallback."""
        n = len(signal_names)
        return {name: 1.0 / n for name in signal_names}
```

### 2. Enhanced compute_state_weights Implementation

```python
# In py/imp/hmm/trainer.py

def compute_state_weights(
    self, 
    observations: np.ndarray,
    artifact: HMMArtifact,
    returns: np.ndarray,
    optimization_config: Optional[OptimizationConfig] = None
) -> FusionWeights:
    """
    Compute optimal fusion weights for each state.
    
    Args:
        observations: Signal observations (T, 3) for [s_LDC, s_MR, s_TSMOM]
        artifact: Trained HMM artifact
        returns: Future returns for optimization (T,)
        optimization_config: Configuration for optimization
        
    Returns:
        FusionWeights with per-state optimal weights
    """
    if optimization_config is None:
        optimization_config = OptimizationConfig()
    
    optimizer = StateWeightOptimizer(optimization_config)
    
    # Get state sequence from trained model
    state_sequence = self.model.predict(observations)
    
    # Optimize weights for each state
    state_weights = []
    state_sharpes = []
    
    for state in range(artifact.n_states):
        # Filter data for this state
        state_mask = state_sequence == state
        state_returns = returns[state_mask]
        state_signals = observations[state_mask]
        
        # Optimize
        optimal_weights, sharpe = optimizer.optimize_state_weights(
            state_returns,
            state_signals
        )
        
        state_weights.append(optimal_weights)
        state_sharpes.append(sharpe)
        
        print(f"State {state}: Sharpe={sharpe:.3f}, Weights={optimal_weights}")
    
    # Calculate overall metrics
    avg_sharpe = np.mean(state_sharpes)
    
    return FusionWeights(
        version="v1.0",
        state_weights=state_weights,
        model_version=artifact.version,
        training_metrics={
            "sharpe_ratio": float(avg_sharpe),
            "state_sharpes": [float(s) for s in state_sharpes],
            "optimization_method": optimization_config.method
        },
        metadata={
            "optimization_method": optimization_config.method,
            "n_states": artifact.n_states,
            "n_observations": len(observations),
            "risk_free_rate": optimization_config.risk_free_rate
        }
    )
```

### 3. Weight Validation Framework

```python
from typing import Dict, Any
import numpy as np
from scipy import stats

class WeightValidator:
    """Validate fusion weights and compare performance."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_weights(
        self,
        fusion_weights: FusionWeights,
        observations: np.ndarray,
        returns: np.ndarray,
        state_sequence: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of fusion weights.
        
        Returns validation report with:
        - Constraint validation
        - Performance vs baseline
        - Statistical significance
        """
        validation_report = {
            'constraints_valid': True,
            'constraint_errors': [],
            'performance_comparison': {},
            'statistical_tests': {}
        }
        
        # 1. Validate constraints
        constraint_check = self._validate_constraints(fusion_weights)
        validation_report['constraints_valid'] = constraint_check['valid']
        validation_report['constraint_errors'] = constraint_check['errors']
        
        # 2. Compare performance vs baseline
        perf_comparison = self._compare_performance(
            fusion_weights,
            observations,
            returns,
            state_sequence
        )
        validation_report['performance_comparison'] = perf_comparison
        
        # 3. Statistical significance
        if perf_comparison['optimized_sharpe'] > perf_comparison['baseline_sharpe']:
            sig_test = self._test_significance(
                perf_comparison['optimized_returns'],
                perf_comparison['baseline_returns']
            )
            validation_report['statistical_tests'] = sig_test
        
        return validation_report
    
    def _validate_constraints(self, fusion_weights: FusionWeights) -> Dict[str, Any]:
        """Validate weight constraints."""
        errors = []
        
        for i, state_weight in enumerate(fusion_weights.state_weights):
            # Check sum to 1
            weight_sum = sum(state_weight.values())
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                errors.append(f"State {i}: weights sum to {weight_sum}, not 1.0")
            
            # Check non-negative
            for signal, weight in state_weight.items():
                if weight < 0:
                    errors.append(f"State {i}, {signal}: negative weight {weight}")
        
        return {'valid': len(errors) == 0, 'errors': errors}
    
    def _compare_performance(
        self,
        fusion_weights: FusionWeights,
        observations: np.ndarray,
        returns: np.ndarray,
        state_sequence: np.ndarray
    ) -> Dict[str, Any]:
        """Compare optimized weights vs equal-weight baseline."""
        
        # Compute returns with optimized weights
        optimized_returns = self._compute_strategy_returns(
            fusion_weights.state_weights,
            observations,
            returns,
            state_sequence
        )
        
        # Compute returns with equal weights
        n_states = len(fusion_weights.state_weights)
        equal_weights = [{'s_LDC': 1/3, 's_MR': 1/3, 's_TSMOM': 1/3}] * n_states
        baseline_returns = self._compute_strategy_returns(
            equal_weights,
            observations,
            returns,
            state_sequence
        )
        
        # Calculate metrics
        opt_sharpe = self._calculate_sharpe(optimized_returns)
        base_sharpe = self._calculate_sharpe(baseline_returns)
        
        return {
            'optimized_sharpe': opt_sharpe,
            'baseline_sharpe': base_sharpe,
            'improvement': opt_sharpe - base_sharpe,
            'improvement_pct': ((opt_sharpe - base_sharpe) / abs(base_sharpe) * 100) if base_sharpe != 0 else 0,
            'optimized_returns': optimized_returns,
            'baseline_returns': baseline_returns
        }
    
    def _compute_strategy_returns(
        self,
        state_weights: List[Dict[str, float]],
        observations: np.ndarray,
        returns: np.ndarray,
        state_sequence: np.ndarray
    ) -> np.ndarray:
        """Compute strategy returns given state-dependent weights."""
        
        portfolio_returns = np.zeros(len(returns))
        
        for t in range(len(returns)):
            state = state_sequence[t]
            weights = state_weights[state]
            
            # Combined signal
            combined = (weights['s_LDC'] * observations[t, 0] +
                       weights['s_MR'] * observations[t, 1] +
                       weights['s_TSMOM'] * observations[t, 2])
            
            # Position and return
            position = np.sign(combined)
            portfolio_returns[t] = position * returns[t]
        
        return portfolio_returns
    
    def _calculate_sharpe(self, returns: np.ndarray, rf_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        annual_std = np.std(returns) * np.sqrt(252)
        return (annual_return - rf_rate) / annual_std
    
    def _test_significance(
        self,
        optimized_returns: np.ndarray,
        baseline_returns: np.ndarray
    ) -> Dict[str, Any]:
        """Test statistical significance of performance difference."""
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(optimized_returns, baseline_returns)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_5pct': p_value < 0.05,
            'significant_at_1pct': p_value < 0.01
        }
```

## Implementation Considerations

### Optimization Challenges

1. **Limited Data Per State**: Some states may have few observations
2. **Overfitting Risk**: Optimizing on same data used for HMM training
3. **Non-Stationarity**: Market regimes may shift over time

### Solutions

1. **Minimum Data Threshold**: Require at least 30 observations per state
2. **Walk-Forward Validation**: Use out-of-sample testing
3. **Regularization**: Consider adding penalty for extreme weights
4. **Fallback Strategy**: Use equal weights when optimization fails

### Integration Points

- Seamlessly integrates with existing `FusionWeights` model
- Works with current `HMMTrainer` and `HMMArtifact` infrastructure
- Compatible with Rust inference engine expectations
- Extends artifact export functionality
