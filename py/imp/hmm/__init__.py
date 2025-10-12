"""
HMM (Hidden Markov Model) components for regime detection and signal fusion.
"""

from .trainer import HMMTrainer
from .inference import HMMInference
from .models import HMMArtifact, FusionWeights
from .artifact_management import (
    ResearchArtifact,
    ResearchMetadata,
    ExperimentTracker,
    ArtifactValidator,
    ArtifactExporter,
    ArtifactManagementError,
    ValidationError,
    ExportError
)
from .regime_analysis import (
    RegimeAnalyzer,
    RegimeCharacteristics,
    StatePersistence,
    EconomicInterpretation,
    FeatureImportance
)
from .weight_optimizer import (
    StateWeightOptimizer,
    OptimizationConfig,
    WeightValidator,
    WalkForwardValidator,
    WalkForwardConfig
)

__all__ = [
    "HMMTrainer",
    "HMMInference",
    "HMMArtifact",
    "FusionWeights",
    "ResearchArtifact",
    "ResearchMetadata",
    "ExperimentTracker",
    "ArtifactValidator",
    "ArtifactExporter",
    "ArtifactManagementError",
    "ValidationError",
    "ExportError",
    "RegimeAnalyzer",
    "RegimeCharacteristics",
    "StatePersistence",
    "EconomicInterpretation",
    "FeatureImportance",
    "StateWeightOptimizer",
    "OptimizationConfig",
    "WeightValidator",
    "WalkForwardValidator",
    "WalkForwardConfig"
]
