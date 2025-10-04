"""
Parameter optimization utilities with grid search and Bayesian optimization.
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
from itertools import product
import json
from pathlib import Path
from datetime import datetime
import warnings

# Try to import scikit-optimize for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Integer, Categorical, Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from ..hmm.trainer import EnhancedHMMTrainer, HMMTrainingError
from ..hmm.models import HMMArtifact


class OptimizationResult:
    """Result from parameter optimization."""
    
    def __init__(self,
                 best_params: Dict[str, Any],
                 best_score: float,
                 all_results: List[Dict[str, Any]],
                 optimization_time: float):
        self.best_params = best_params
        self.best_score = best_score
        self.all_results = all_results
        self.optimization_time = optimization_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.all_results,
            'optimization_time': self.optimization_time
        }
    
    def save(self, filepath: Path):
        """Save results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class GridSearchOptimizer:
    """Grid search parameter optimization for HMM models."""
    
    def __init__(self,
                 observations: np.ndarray,
                 param_grid: Dict[str, List[Any]],
                 scoring_metric: str = 'log_likelihood',
                 higher_is_better: bool = True,
                 validation_split: float = 0.2,
                 n_iterations: int = 100,
                 verbose: bool = True):
        """
        Initialize grid search optimizer.
        
        Args:
            observations: Training data
            param_grid: Dictionary of parameter names to lists of values to try
            scoring_metric: Metric to optimize ('log_likelihood', 'aic', 'bic')
            higher_is_better: Whether higher metric values are better
            validation_split: Fraction of data for validation
            n_iterations: Number of EM iterations per model
            verbose: Whether to print progress
        """
        self.observations = observations
        self.param_grid = param_grid
        self.scoring_metric = scoring_metric
        self.higher_is_better = higher_is_better
        self.validation_split = validation_split
        self.n_iterations = n_iterations
        self.verbose = verbose
        
        self.results: List[Dict[str, Any]] = []
    
    def fit(self) -> OptimizationResult:
        """
        Run grid search optimization.
        
        Returns:
            OptimizationResult with best parameters and all results
        """
        start_time = datetime.now()
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        
        if self.verbose:
            print(f"Starting grid search with {total_combinations} parameter combinations...")
        
        best_score = float('-inf') if self.higher_is_better else float('inf')
        best_params = None
        
        for i, param_combo in enumerate(param_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            if self.verbose:
                print(f"\n[{i+1}/{total_combinations}] Testing: {params}")
            
            try:
                # Train model with these parameters
                score, artifact, metrics = self._evaluate_params(params)
                
                # Store result
                result = {
                    'params': params,
                    'score': score,
                    'metrics': metrics,
                    'artifact_metadata': artifact.metadata
                }
                self.results.append(result)
                
                # Update best
                is_better = (score > best_score) if self.higher_is_better else (score < best_score)
                if is_better:
                    best_score = score
                    best_params = params
                    if self.verbose:
                        print(f"  ✅ New best score: {score:.4f}")
                else:
                    if self.verbose:
                        print(f"  Score: {score:.4f}")
            
            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Failed: {str(e)}")
                
                # Store failed result
                self.results.append({
                    'params': params,
                    'score': None,
                    'error': str(e)
                })
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Grid search completed in {optimization_time:.2f} seconds")
            print(f"Best parameters: {best_params}")
            print(f"Best score: {best_score:.4f}")
            print(f"{'='*60}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=self.results,
            optimization_time=optimization_time
        )
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Tuple[float, HMMArtifact, Dict[str, float]]:
        """Evaluate a parameter configuration."""
        # Extract parameters
        n_states = params.get('n_states', 3)
        library = params.get('library', 'hmmlearn')
        covariance_type = params.get('covariance_type', 'full')
        random_state = params.get('random_state', 42)
        
        # Create trainer
        trainer = EnhancedHMMTrainer(
            n_states=n_states,
            library=library,
            covariance_type=covariance_type,
            random_state=random_state
        )
        
        # Train with validation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            artifact, metrics = trainer.train_with_validation(
                self.observations,
                validation_split=self.validation_split,
                n_iterations=self.n_iterations
            )
        
        # Extract score
        if self.scoring_metric in metrics:
            score = metrics[self.scoring_metric]
        elif self.scoring_metric in artifact.metadata:
            score = artifact.metadata[self.scoring_metric]
        else:
            raise ValueError(f"Scoring metric '{self.scoring_metric}' not found in results")
        
        return score, artifact, metrics


class BayesianOptimizer:
    """Bayesian optimization for HMM hyperparameters using Gaussian Processes."""
    
    def __init__(self,
                 observations: np.ndarray,
                 param_space: Dict[str, Any],
                 scoring_metric: str = 'log_likelihood',
                 higher_is_better: bool = True,
                 validation_split: float = 0.2,
                 n_iterations: int = 100,
                 n_calls: int = 20,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize Bayesian optimizer.
        
        Args:
            observations: Training data
            param_space: Dictionary defining parameter search space
            scoring_metric: Metric to optimize
            higher_is_better: Whether higher metric values are better
            validation_split: Fraction of data for validation
            n_iterations: Number of EM iterations per model
            n_calls: Number of optimization iterations
            random_state: Random seed
            verbose: Whether to print progress
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize not available. Install with: pip install scikit-optimize"
            )
        
        self.observations = observations
        self.param_space = param_space
        self.scoring_metric = scoring_metric
        self.higher_is_better = higher_is_better
        self.validation_split = validation_split
        self.n_iterations = n_iterations
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        
        self.results: List[Dict[str, Any]] = []
        self.iteration = 0
    
    def fit(self) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Returns:
            OptimizationResult with best parameters and all results
        """
        if self.verbose:
            print(f"Starting Bayesian optimization with {self.n_calls} iterations...")
        
        start_time = datetime.now()
        
        # Convert param_space to skopt format
        dimensions, param_names = self._create_search_space()
        
        # Define objective function
        @use_named_args(dimensions=dimensions)
        def objective(**params):
            self.iteration += 1
            
            if self.verbose:
                print(f"\n[{self.iteration}/{self.n_calls}] Testing: {params}")
            
            try:
                score, artifact, metrics = self._evaluate_params(params)
                
                # Store result
                result = {
                    'params': params,
                    'score': score,
                    'metrics': metrics,
                    'artifact_metadata': artifact.metadata
                }
                self.results.append(result)
                
                if self.verbose:
                    print(f"  Score: {score:.4f}")
                
                # Return negative score if we want to maximize (gp_minimize minimizes)
                return -score if self.higher_is_better else score
            
            except Exception as e:
                if self.verbose:
                    print(f"  ❌ Failed: {str(e)}")
                
                # Store failed result
                self.results.append({
                    'params': params,
                    'score': None,
                    'error': str(e)
                })
                
                # Return worst possible score
                return float('inf')
        
        # Run optimization
        result = gp_minimize(
            objective,
            dimensions=dimensions,
            n_calls=self.n_calls,
            random_state=self.random_state,
            verbose=False  # We handle our own verbosity
        )
        
        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun if self.higher_is_better else result.fun
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Bayesian optimization completed in {optimization_time:.2f} seconds")
            print(f"Best parameters: {best_params}")
            print(f"Best score: {best_score:.4f}")
            print(f"{'='*60}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=self.results,
            optimization_time=optimization_time
        )
    
    def _create_search_space(self) -> Tuple[List, List[str]]:
        """Create skopt search space from param_space."""
        dimensions = []
        param_names = []
        
        for param_name, param_spec in self.param_space.items():
            param_names.append(param_name)
            
            if param_spec['type'] == 'integer':
                dimensions.append(Integer(
                    param_spec['low'],
                    param_spec['high'],
                    name=param_name
                ))
            elif param_spec['type'] == 'categorical':
                dimensions.append(Categorical(
                    param_spec['categories'],
                    name=param_name
                ))
            elif param_spec['type'] == 'real':
                dimensions.append(Real(
                    param_spec['low'],
                    param_spec['high'],
                    name=param_name
                ))
            else:
                raise ValueError(f"Unknown parameter type: {param_spec['type']}")
        
        return dimensions, param_names
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Tuple[float, HMMArtifact, Dict[str, float]]:
        """Evaluate a parameter configuration."""
        # Extract parameters
        n_states = int(params.get('n_states', 3))
        library = params.get('library', 'hmmlearn')
        covariance_type = params.get('covariance_type', 'full')
        random_state = params.get('random_state', self.random_state)
        
        # Create trainer
        trainer = EnhancedHMMTrainer(
            n_states=n_states,
            library=library,
            covariance_type=covariance_type,
            random_state=random_state
        )
        
        # Train with validation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            artifact, metrics = trainer.train_with_validation(
                self.observations,
                validation_split=self.validation_split,
                n_iterations=self.n_iterations
            )
        
        # Extract score
        if self.scoring_metric in metrics:
            score = metrics[self.scoring_metric]
        elif self.scoring_metric in artifact.metadata:
            score = artifact.metadata[self.scoring_metric]
        else:
            raise ValueError(f"Scoring metric '{self.scoring_metric}' not found in results")
        
        return score, artifact, metrics


def create_default_param_grid() -> Dict[str, List[Any]]:
    """
    Create a default parameter grid for grid search.
    
    Returns:
        Dictionary with default parameter ranges
    """
    return {
        'n_states': [2, 3, 4, 5],
        'library': ['hmmlearn'],
        'covariance_type': ['full', 'diag'],
        'random_state': [42]
    }


def create_default_param_space() -> Dict[str, Dict[str, Any]]:
    """
    Create a default parameter space for Bayesian optimization.
    
    Returns:
        Dictionary with default parameter space definitions
    """
    return {
        'n_states': {
            'type': 'integer',
            'low': 2,
            'high': 10
        },
        'library': {
            'type': 'categorical',
            'categories': ['hmmlearn']
        },
        'covariance_type': {
            'type': 'categorical',
            'categories': ['full', 'diag', 'spherical']
        }
    }


def quick_grid_search(observations: np.ndarray,
                     n_states_range: List[int] = [2, 3, 4, 5],
                     covariance_types: List[str] = ['full', 'diag'],
                     verbose: bool = True) -> OptimizationResult:
    """
    Quick grid search with common parameters.
    
    Args:
        observations: Training data
        n_states_range: List of state counts to try
        covariance_types: List of covariance types to try
        verbose: Whether to print progress
        
    Returns:
        OptimizationResult with best configuration
    """
    param_grid = {
        'n_states': n_states_range,
        'library': ['hmmlearn'],
        'covariance_type': covariance_types,
        'random_state': [42]
    }
    
    optimizer = GridSearchOptimizer(
        observations=observations,
        param_grid=param_grid,
        scoring_metric='log_likelihood',
        higher_is_better=True,
        verbose=verbose
    )
    
    return optimizer.fit()


def quick_bayesian_search(observations: np.ndarray,
                         n_calls: int = 20,
                         verbose: bool = True) -> OptimizationResult:
    """
    Quick Bayesian optimization with default parameters.
    
    Args:
        observations: Training data
        n_calls: Number of optimization iterations
        verbose: Whether to print progress
        
    Returns:
        OptimizationResult with best configuration
    """
    if not SKOPT_AVAILABLE:
        raise ImportError(
            "scikit-optimize not available. Install with: pip install scikit-optimize"
        )
    
    param_space = create_default_param_space()
    
    optimizer = BayesianOptimizer(
        observations=observations,
        param_space=param_space,
        scoring_metric='log_likelihood',
        higher_is_better=True,
        n_calls=n_calls,
        verbose=verbose
    )
    
    return optimizer.fit()
