"""
Production integration and artifact management for HMM research environment.

This module provides enhanced artifact management with research-specific metadata,
experiment tracking, versioning, and validation utilities for production deployment.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import json
import hashlib
import logging
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np

from .models import HMMArtifact, FusionWeights

logger = logging.getLogger(__name__)


class ArtifactManagementError(Exception):
    """Base exception for artifact management errors."""
    pass


class ValidationError(ArtifactManagementError):
    """Exception raised during artifact validation."""
    pass


class ExportError(ArtifactManagementError):
    """Exception raised during artifact export."""
    pass


class ResearchMetadata(BaseModel):
    """Research-specific metadata for HMM artifacts."""
    
    experiment_id: str = Field(description="Unique experiment identifier")
    researcher: str = Field(description="Name or ID of researcher")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    notebook_path: Optional[str] = Field(None, description="Path to source notebook")
    
    # Training configuration
    training_config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")
    library_used: str = Field(description="HMM library used (hmmlearn/pomegranate)")
    
    # Evaluation metrics
    evaluation_metrics: Dict[str, float] = Field(default_factory=dict, description="Evaluation metrics")
    cross_validation_scores: Optional[List[float]] = Field(None, description="CV scores")
    
    # Model comparison
    comparison_baseline: Optional[str] = Field(None, description="Baseline model for comparison")
    performance_ranking: Optional[int] = Field(None, description="Performance rank among compared models")
    
    # Additional notes
    notes: str = Field(default="", description="Additional research notes")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResearchArtifact(BaseModel):
    """Enhanced artifact for research environment with experiment tracking."""
    
    # Core HMM artifact
    base_artifact: HMMArtifact = Field(description="Base HMM artifact")
    
    # Research metadata
    research_metadata: ResearchMetadata = Field(description="Research-specific metadata")
    
    # Versioning
    artifact_version: str = Field(description="Artifact version (semantic versioning)")
    artifact_hash: str = Field(description="Hash of artifact for integrity checking")
    
    # Production compatibility
    production_ready: bool = Field(default=False, description="Whether artifact is production-ready")
    validation_status: Dict[str, bool] = Field(default_factory=dict, description="Validation checks status")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def from_hmm_artifact(cls,
                         artifact: HMMArtifact,
                         experiment_id: str,
                         researcher: str,
                         training_config: Dict[str, Any],
                         evaluation_metrics: Optional[Dict[str, float]] = None,
                         **kwargs) -> "ResearchArtifact":
        """
        Create ResearchArtifact from HMMArtifact.
        
        Args:
            artifact: Base HMM artifact
            experiment_id: Unique experiment identifier
            researcher: Researcher name/ID
            training_config: Training configuration used
            evaluation_metrics: Evaluation metrics
            **kwargs: Additional metadata fields
            
        Returns:
            ResearchArtifact instance
        """
        # Create research metadata
        research_metadata = ResearchMetadata(
            experiment_id=experiment_id,
            researcher=researcher,
            training_config=training_config,
            library_used=artifact.metadata.get("library", "unknown"),
            evaluation_metrics=evaluation_metrics or {},
            **kwargs
        )
        
        # Calculate artifact hash
        artifact_hash = cls._calculate_artifact_hash(artifact)
        
        # Generate version
        artifact_version = kwargs.get("artifact_version", "1.0.0")
        
        return cls(
            base_artifact=artifact,
            research_metadata=research_metadata,
            artifact_version=artifact_version,
            artifact_hash=artifact_hash,
            production_ready=False,
            validation_status={}
        )
    
    @staticmethod
    def _calculate_artifact_hash(artifact: HMMArtifact) -> str:
        """Calculate hash of artifact for integrity checking."""
        # Create deterministic string representation
        artifact_str = json.dumps(artifact.model_dump(), sort_keys=True)
        return hashlib.sha256(artifact_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify artifact integrity using hash."""
        current_hash = self._calculate_artifact_hash(self.base_artifact)
        return current_hash == self.artifact_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchArtifact":
        """Create from dictionary."""
        return cls(**data)


class ExperimentTracker:
    """Track and manage research experiments with versioning."""
    
    def __init__(self, experiment_dir: Union[str, Path]):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory for storing experiment artifacts
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file for tracking experiments
        self.index_file = self.experiment_dir / "experiment_index.json"
        self.experiments = self._load_index()
        
        logger.info(f"Experiment tracker initialized at {self.experiment_dir}")
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load experiment index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load experiment index: {e}")
                return {}
        else:
            # Create empty index file
            self._save_index_internal({})
        return {}
    
    def _save_index(self) -> None:
        """Save experiment index to disk."""
        self._save_index_internal(self.experiments)
    
    def _save_index_internal(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Internal method to save index data to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save experiment index: {e}")
            raise ArtifactManagementError(f"Failed to save index: {e}")
    
    def log_experiment(self,
                      artifact: ResearchArtifact,
                      notes: str = "") -> str:
        """
        Log research experiment with versioning.
        
        Args:
            artifact: Research artifact to log
            notes: Additional notes about the experiment
            
        Returns:
            Experiment ID
        """
        experiment_id = artifact.research_metadata.experiment_id
        
        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save artifact
        artifact_path = exp_dir / f"artifact_v{artifact.artifact_version}.json"
        try:
            with open(artifact_path, 'w') as f:
                json.dump(artifact.to_dict(), f, indent=2, default=str)
        except Exception as e:
            raise ExportError(f"Failed to save artifact: {e}")
        
        # Update index
        self.experiments[experiment_id] = {
            "experiment_id": experiment_id,
            "researcher": artifact.research_metadata.researcher,
            "created_at": artifact.research_metadata.created_at.isoformat(),
            "artifact_version": artifact.artifact_version,
            "artifact_path": str(artifact_path),
            "artifact_hash": artifact.artifact_hash,
            "production_ready": artifact.production_ready,
            "notes": notes,
            "tags": artifact.research_metadata.tags,
            "library": artifact.research_metadata.library_used,
            "n_states": artifact.base_artifact.n_states
        }
        
        self._save_index()
        logger.info(f"Experiment {experiment_id} logged successfully")
        
        return experiment_id
    
    def load_experiment(self, experiment_id: str, version: Optional[str] = None) -> ResearchArtifact:
        """
        Load experiment artifact.
        
        Args:
            experiment_id: Experiment identifier
            version: Specific version to load (None for latest)
            
        Returns:
            ResearchArtifact
        """
        if experiment_id not in self.experiments:
            raise ArtifactManagementError(f"Experiment {experiment_id} not found")
        
        exp_info = self.experiments[experiment_id]
        exp_dir = self.experiment_dir / experiment_id
        
        # Find artifact file
        if version:
            artifact_path = exp_dir / f"artifact_v{version}.json"
        else:
            artifact_path = Path(exp_info["artifact_path"])
        
        if not artifact_path.exists():
            raise ArtifactManagementError(f"Artifact file not found: {artifact_path}")
        
        try:
            with open(artifact_path, 'r') as f:
                data = json.load(f)
            artifact = ResearchArtifact.from_dict(data)
            
            # Verify integrity
            if not artifact.verify_integrity():
                logger.warning(f"Artifact integrity check failed for {experiment_id}")
            
            return artifact
        except Exception as e:
            raise ArtifactManagementError(f"Failed to load artifact: {e}")
    
    def list_experiments(self,
                        researcher: Optional[str] = None,
                        tags: Optional[List[str]] = None,
                        production_ready: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.
        
        Args:
            researcher: Filter by researcher
            tags: Filter by tags (any match)
            production_ready: Filter by production readiness
            
        Returns:
            List of experiment info dictionaries
        """
        experiments = list(self.experiments.values())
        
        # Apply filters
        if researcher:
            experiments = [e for e in experiments if e.get("researcher") == researcher]
        
        if tags:
            experiments = [e for e in experiments 
                          if any(tag in e.get("tags", []) for tag in tags)]
        
        if production_ready is not None:
            experiments = [e for e in experiments 
                          if e.get("production_ready") == production_ready]
        
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            "experiments": [],
            "metrics_comparison": {},
            "best_by_metric": {}
        }
        
        for exp_id in experiment_ids:
            try:
                artifact = self.load_experiment(exp_id)
                exp_info = {
                    "experiment_id": exp_id,
                    "n_states": artifact.base_artifact.n_states,
                    "library": artifact.research_metadata.library_used,
                    "metrics": artifact.research_metadata.evaluation_metrics,
                    "production_ready": artifact.production_ready
                }
                comparison["experiments"].append(exp_info)
            except Exception as e:
                logger.warning(f"Failed to load experiment {exp_id}: {e}")
        
        # Compare metrics
        if comparison["experiments"]:
            # Collect all metric names
            all_metrics = set()
            for exp in comparison["experiments"]:
                all_metrics.update(exp["metrics"].keys())
            
            # Compare each metric
            for metric in all_metrics:
                values = []
                for exp in comparison["experiments"]:
                    if metric in exp["metrics"]:
                        values.append({
                            "experiment_id": exp["experiment_id"],
                            "value": exp["metrics"][metric]
                        })
                
                comparison["metrics_comparison"][metric] = values
                
                # Find best (assuming higher is better for most metrics except AIC/BIC)
                if values:
                    if metric.lower() in ["aic", "bic", "perplexity"]:
                        # Lower is better
                        best = min(values, key=lambda x: x["value"])
                    else:
                        # Higher is better
                        best = max(values, key=lambda x: x["value"])
                    
                    comparison["best_by_metric"][metric] = best["experiment_id"]
        
        return comparison
    
    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete experiment and its artifacts.
        
        Args:
            experiment_id: Experiment to delete
        """
        if experiment_id not in self.experiments:
            raise ArtifactManagementError(f"Experiment {experiment_id} not found")
        
        # Delete experiment directory
        exp_dir = self.experiment_dir / experiment_id
        if exp_dir.exists():
            import shutil
            shutil.rmtree(exp_dir)
        
        # Remove from index
        del self.experiments[experiment_id]
        self._save_index()
        
        logger.info(f"Experiment {experiment_id} deleted")


class ArtifactValidator:
    """Validate artifacts for production deployment compatibility."""
    
    @staticmethod
    def validate_for_production(artifact: ResearchArtifact) -> Dict[str, bool]:
        """
        Validate artifact for production deployment.
        
        Args:
            artifact: Research artifact to validate
            
        Returns:
            Dictionary of validation checks and their status
        """
        validation_results = {}
        
        # Check 1: Artifact integrity
        validation_results["integrity_check"] = artifact.verify_integrity()
        
        # Check 2: HMM artifact validation
        try:
            base_artifact = artifact.base_artifact
            validation_results["hmm_artifact_valid"] = True
        except Exception as e:
            logger.error(f"HMM artifact validation failed: {e}")
            validation_results["hmm_artifact_valid"] = False
        
        # Check 3: Transition matrix properties
        try:
            transition_matrix = np.array(base_artifact.transition_matrix)
            # Check row sums
            row_sums = transition_matrix.sum(axis=1)
            validation_results["transition_matrix_valid"] = np.allclose(row_sums, 1.0, atol=1e-6)
        except Exception as e:
            logger.error(f"Transition matrix validation failed: {e}")
            validation_results["transition_matrix_valid"] = False
        
        # Check 4: Covariance matrices are positive definite
        try:
            all_positive_definite = True
            for cov in base_artifact.covariances:
                cov_matrix = np.array(cov)
                try:
                    np.linalg.cholesky(cov_matrix)
                except np.linalg.LinAlgError:
                    all_positive_definite = False
                    break
            validation_results["covariances_positive_definite"] = all_positive_definite
        except Exception as e:
            logger.error(f"Covariance validation failed: {e}")
            validation_results["covariances_positive_definite"] = False
        
        # Check 5: Evaluation metrics present
        validation_results["has_evaluation_metrics"] = bool(
            artifact.research_metadata.evaluation_metrics
        )
        
        # Check 6: Reasonable number of states
        validation_results["reasonable_n_states"] = 2 <= base_artifact.n_states <= 10
        
        # Check 7: Training window validity
        validation_results["valid_training_window"] = (
            base_artifact.training_window_end > base_artifact.training_window_start
        )
        
        # Check 8: Metadata completeness
        required_metadata = ["library", "algorithm", "covariance_type"]
        validation_results["metadata_complete"] = all(
            key in base_artifact.metadata for key in required_metadata
        )
        
        return validation_results
    
    @staticmethod
    def validate_rust_compatibility(artifact: ResearchArtifact) -> Dict[str, bool]:
        """
        Validate compatibility with Rust inference engine.
        
        Args:
            artifact: Research artifact to validate
            
        Returns:
            Dictionary of compatibility checks
        """
        compatibility_results = {}
        
        base_artifact = artifact.base_artifact
        
        # Check 1: Data types are JSON-serializable
        try:
            json.dumps(base_artifact.model_dump())
            compatibility_results["json_serializable"] = True
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            compatibility_results["json_serializable"] = False
        
        # Check 2: Array dimensions are consistent
        try:
            n_states = base_artifact.n_states
            n_features = len(base_artifact.means[0]) if base_artifact.means else 0
            
            # Check transition matrix
            trans_shape_ok = (
                len(base_artifact.transition_matrix) == n_states and
                all(len(row) == n_states for row in base_artifact.transition_matrix)
            )
            
            # Check means
            means_shape_ok = (
                len(base_artifact.means) == n_states and
                all(len(mean) == n_features for mean in base_artifact.means)
            )
            
            # Check covariances
            covs_shape_ok = (
                len(base_artifact.covariances) == n_states and
                all(len(cov) == n_features and all(len(row) == n_features for row in cov)
                    for cov in base_artifact.covariances)
            )
            
            compatibility_results["dimensions_consistent"] = (
                trans_shape_ok and means_shape_ok and covs_shape_ok
            )
        except Exception as e:
            logger.error(f"Dimension check failed: {e}")
            compatibility_results["dimensions_consistent"] = False
        
        # Check 3: No NaN or Inf values
        try:
            def has_invalid_values(data):
                if isinstance(data, (list, tuple)):
                    return any(has_invalid_values(item) for item in data)
                elif isinstance(data, (int, float)):
                    return np.isnan(data) or np.isinf(data)
                return False
            
            no_invalid = not any([
                has_invalid_values(base_artifact.transition_matrix),
                has_invalid_values(base_artifact.initial_probabilities),
                has_invalid_values(base_artifact.means),
                has_invalid_values(base_artifact.covariances)
            ])
            
            compatibility_results["no_invalid_values"] = no_invalid
        except Exception as e:
            logger.error(f"Invalid value check failed: {e}")
            compatibility_results["no_invalid_values"] = False
        
        # Check 4: Version format is valid
        compatibility_results["valid_version_format"] = bool(
            base_artifact.version and isinstance(base_artifact.version, str)
        )
        
        return compatibility_results
    
    @staticmethod
    def run_all_validations(artifact: ResearchArtifact) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Args:
            artifact: Research artifact to validate
            
        Returns:
            Complete validation report
        """
        production_checks = ArtifactValidator.validate_for_production(artifact)
        rust_checks = ArtifactValidator.validate_rust_compatibility(artifact)
        
        all_checks = {**production_checks, **rust_checks}
        all_passed = all(all_checks.values())
        
        return {
            "all_passed": all_passed,
            "production_checks": production_checks,
            "rust_compatibility_checks": rust_checks,
            "failed_checks": [k for k, v in all_checks.items() if not v]
        }


class ArtifactExporter:
    """Export artifacts for production deployment."""
    
    @staticmethod
    def export_for_production(artifact: ResearchArtifact,
                             output_dir: Union[str, Path],
                             validate: bool = True) -> Dict[str, Path]:
        """
        Export artifact for production deployment.
        
        Args:
            artifact: Research artifact to export
            output_dir: Output directory for exported files
            validate: Whether to validate before export
            
        Returns:
            Dictionary of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate if requested
        if validate:
            validation_results = ArtifactValidator.run_all_validations(artifact)
            if not validation_results["all_passed"]:
                raise ValidationError(
                    f"Artifact validation failed: {validation_results['failed_checks']}"
                )
        
        exported_files = {}
        
        # Export base HMM artifact
        hmm_artifact_path = output_dir / "hmm_artifact.json"
        try:
            with open(hmm_artifact_path, 'w') as f:
                json.dump(artifact.base_artifact.model_dump(), f, indent=2)
            exported_files["hmm_artifact"] = hmm_artifact_path
        except Exception as e:
            raise ExportError(f"Failed to export HMM artifact: {e}")
        
        # Export research metadata
        metadata_path = output_dir / "research_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(artifact.research_metadata.model_dump(), f, indent=2, default=str)
            exported_files["metadata"] = metadata_path
        except Exception as e:
            raise ExportError(f"Failed to export metadata: {e}")
        
        # Export validation report
        validation_path = output_dir / "validation_report.json"
        try:
            validation_results = ArtifactValidator.run_all_validations(artifact)
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            exported_files["validation_report"] = validation_path
        except Exception as e:
            logger.warning(f"Failed to export validation report: {e}")
        
        # Create deployment manifest
        manifest_path = output_dir / "deployment_manifest.json"
        try:
            manifest = {
                "artifact_version": artifact.artifact_version,
                "artifact_hash": artifact.artifact_hash,
                "experiment_id": artifact.research_metadata.experiment_id,
                "created_at": artifact.research_metadata.created_at.isoformat(),
                "researcher": artifact.research_metadata.researcher,
                "n_states": artifact.base_artifact.n_states,
                "library": artifact.research_metadata.library_used,
                "production_ready": artifact.production_ready,
                "exported_files": {k: str(v) for k, v in exported_files.items()}
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            exported_files["manifest"] = manifest_path
        except Exception as e:
            logger.warning(f"Failed to create deployment manifest: {e}")
        
        logger.info(f"Artifact exported to {output_dir}")
        return exported_files
    
    @staticmethod
    def export_fusion_weights(fusion_weights: FusionWeights,
                             output_path: Union[str, Path]) -> Path:
        """
        Export fusion weights for production.
        
        Args:
            fusion_weights: Fusion weights to export
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(fusion_weights.model_dump(), f, indent=2)
            logger.info(f"Fusion weights exported to {output_path}")
            return output_path
        except Exception as e:
            raise ExportError(f"Failed to export fusion weights: {e}")
