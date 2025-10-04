"""
IMP Python Package

This package contains the Python components for the IMP trading system:
- HMM training and inference
- HMM evaluation and comparison
- Research notebooks and experimentation
- Model training services
- Utility functions
"""

__version__ = "0.1.0"
__author__ = "IMP Team"

# Import key components for easy access
from .hmm import HMMTrainer, HMMInference, HMMArtifact, FusionWeights
from .evaluation import HMMEvaluator, ModelComparison, EvaluationMetrics

__all__ = [
    "HMMTrainer",
    "HMMInference", 
    "HMMArtifact",
    "FusionWeights",
    "HMMEvaluator",
    "ModelComparison",
    "EvaluationMetrics"
]
