"""
Tests for HMM artifact management and production integration.
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from imp.hmm import (
    HMMArtifact,
    FusionWeights,
    ResearchArtifact,
    ResearchMetadata,
    ExperimentTracker,
    ArtifactValidator,
    ArtifactExporter,
    ArtifactManagementError,
    ValidationError,
    ExportError
)


@pytest.fixture
def sample_hmm_artifact():
    """Create a sample HMM artifact for testing."""
    return HMMArtifact(
        version="v1.0",
        n_states=3,
        transition_matrix=[
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6]
        ],
        initial_probabilities=[0.33, 0.34, 0.33],
        means=[
            [0.5, 0.3],
            [-0.2, 0.8],
            [0.1, -0.4]
        ],
        covariances=[
            [[1.0, 0.1], [0.1, 1.0]],
            [[0.8, 0.0], [0.0, 0.9]],
            [[1.2, -0.1], [-0.1, 1.1]]
        ],
        training_window_start=0,
        training_window_end=1000,
        metadata={
            "library": "hmmlearn",
            "algorithm": "baum-welch",
            "covariance_type": "full",
            "aic": 1234.5,
            "bic": 1250.0
        }
    )


@pytest.fixture
def sample_fusion_weights():
    """Create sample fusion weights for testing."""
    return FusionWeights(
        version="v1.0",
        state_weights=[
            {"w_ldc": 0.5, "w_mr": 0.3, "w_tsmom": 0.2},
            {"w_ldc": 0.3, "w_mr": 0.4, "w_tsmom": 0.3},
            {"w_ldc": 0.4, "w_mr": 0.2, "w_tsmom": 0.4}
        ],
        model_version="v1.0",
        training_metrics={"sharpe_ratio": 1.5, "max_drawdown": -0.15},
        metadata={"n_states": 3}
    )


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for experiments."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestResearchMetadata:
    """Tests for ResearchMetadata model."""
    
    def test_create_research_metadata(self):
        """Test creating research metadata."""
        metadata = ResearchMetadata(
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={"n_states": 3, "library": "hmmlearn"},
            library_used="hmmlearn",
            evaluation_metrics={"aic": 1234.5, "bic": 1250.0}
        )
        
        assert metadata.experiment_id == "exp_001"
        assert metadata.researcher == "test_researcher"
        assert metadata.library_used == "hmmlearn"
        assert "aic" in metadata.evaluation_metrics
    
    def test_metadata_with_optional_fields(self):
        """Test metadata with optional fields."""
        metadata = ResearchMetadata(
            experiment_id="exp_002",
            researcher="researcher2",
            training_config={},
            library_used="pomegranate",
            notebook_path="/path/to/notebook.ipynb",
            comparison_baseline="exp_001",
            performance_ranking=1,
            notes="Best model so far",
            tags=["production", "optimized"]
        )
        
        assert metadata.notebook_path == "/path/to/notebook.ipynb"
        assert metadata.comparison_baseline == "exp_001"
        assert metadata.performance_ranking == 1
        assert "production" in metadata.tags


class TestResearchArtifact:
    """Tests for ResearchArtifact model."""
    
    def test_create_from_hmm_artifact(self, sample_hmm_artifact):
        """Test creating ResearchArtifact from HMMArtifact."""
        research_artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={"n_states": 3, "library": "hmmlearn"},
            evaluation_metrics={"aic": 1234.5, "bic": 1250.0}
        )
        
        assert research_artifact.base_artifact == sample_hmm_artifact
        assert research_artifact.research_metadata.experiment_id == "exp_001"
        assert research_artifact.artifact_version == "1.0.0"
        assert research_artifact.artifact_hash is not None
        assert not research_artifact.production_ready
    
    def test_artifact_hash_calculation(self, sample_hmm_artifact):
        """Test artifact hash calculation."""
        artifact1 = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="researcher1",
            training_config={}
        )
        
        artifact2 = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_002",
            researcher="researcher2",
            training_config={}
        )
        
        # Same base artifact should produce same hash
        assert artifact1.artifact_hash == artifact2.artifact_hash
    
    def test_verify_integrity(self, sample_hmm_artifact):
        """Test artifact integrity verification."""
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={}
        )
        
        # Should pass integrity check
        assert artifact.verify_integrity()
        
        # Modify artifact and check again
        artifact.base_artifact.n_states = 5
        assert not artifact.verify_integrity()
    
    def test_to_dict_and_from_dict(self, sample_hmm_artifact):
        """Test serialization and deserialization."""
        original = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={"n_states": 3}
        )
        
        # Convert to dict and back
        artifact_dict = original.to_dict()
        restored = ResearchArtifact.from_dict(artifact_dict)
        
        assert restored.research_metadata.experiment_id == original.research_metadata.experiment_id
        assert restored.artifact_hash == original.artifact_hash
        assert restored.base_artifact.n_states == original.base_artifact.n_states


class TestExperimentTracker:
    """Tests for ExperimentTracker."""
    
    def test_initialize_tracker(self, temp_experiment_dir):
        """Test initializing experiment tracker."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        assert tracker.experiment_dir == temp_experiment_dir
        assert tracker.index_file.exists()
    
    def test_log_experiment(self, temp_experiment_dir, sample_hmm_artifact):
        """Test logging an experiment."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={"n_states": 3}
        )
        
        exp_id = tracker.log_experiment(artifact, notes="Test experiment")
        
        assert exp_id == "exp_001"
        assert exp_id in tracker.experiments
        assert (temp_experiment_dir / exp_id).exists()
    
    def test_load_experiment(self, temp_experiment_dir, sample_hmm_artifact):
        """Test loading an experiment."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        # Log experiment
        original = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={"n_states": 3}
        )
        tracker.log_experiment(original)
        
        # Load experiment
        loaded = tracker.load_experiment("exp_001")
        
        assert loaded.research_metadata.experiment_id == "exp_001"
        assert loaded.base_artifact.n_states == 3
    
    def test_list_experiments(self, temp_experiment_dir, sample_hmm_artifact):
        """Test listing experiments with filters."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        # Log multiple experiments
        for i in range(3):
            artifact = ResearchArtifact.from_hmm_artifact(
                artifact=sample_hmm_artifact,
                experiment_id=f"exp_{i:03d}",
                researcher=f"researcher_{i % 2}",
                training_config={},
                tags=["test"] if i % 2 == 0 else ["production"]
            )
            tracker.log_experiment(artifact)
        
        # List all experiments
        all_exps = tracker.list_experiments()
        assert len(all_exps) == 3
        
        # Filter by researcher
        researcher_0_exps = tracker.list_experiments(researcher="researcher_0")
        assert len(researcher_0_exps) == 2
        
        # Filter by tags
        test_exps = tracker.list_experiments(tags=["test"])
        assert len(test_exps) == 2
    
    def test_compare_experiments(self, temp_experiment_dir, sample_hmm_artifact):
        """Test comparing experiments."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        # Log experiments with different metrics
        for i in range(2):
            artifact = ResearchArtifact.from_hmm_artifact(
                artifact=sample_hmm_artifact,
                experiment_id=f"exp_{i:03d}",
                researcher="test_researcher",
                training_config={},
                evaluation_metrics={
                    "aic": 1200 + i * 10,
                    "bic": 1250 + i * 10,
                    "log_likelihood": -600 - i * 5
                }
            )
            tracker.log_experiment(artifact)
        
        # Compare experiments
        comparison = tracker.compare_experiments(["exp_000", "exp_001"])
        
        assert len(comparison["experiments"]) == 2
        assert "aic" in comparison["metrics_comparison"]
        assert "best_by_metric" in comparison
    
    def test_delete_experiment(self, temp_experiment_dir, sample_hmm_artifact):
        """Test deleting an experiment."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        # Log experiment
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={}
        )
        tracker.log_experiment(artifact)
        
        # Delete experiment
        tracker.delete_experiment("exp_001")
        
        assert "exp_001" not in tracker.experiments
        assert not (temp_experiment_dir / "exp_001").exists()
    
    def test_load_nonexistent_experiment(self, temp_experiment_dir):
        """Test loading non-existent experiment raises error."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        with pytest.raises(ArtifactManagementError):
            tracker.load_experiment("nonexistent")


class TestArtifactValidator:
    """Tests for ArtifactValidator."""
    
    def test_validate_for_production_valid_artifact(self, sample_hmm_artifact):
        """Test validation of valid artifact."""
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={},
            evaluation_metrics={"aic": 1234.5}
        )
        
        results = ArtifactValidator.validate_for_production(artifact)
        
        assert results["integrity_check"]
        assert results["hmm_artifact_valid"]
        assert results["transition_matrix_valid"]
        assert results["covariances_positive_definite"]
        assert results["has_evaluation_metrics"]
        assert results["reasonable_n_states"]
        assert results["valid_training_window"]
        assert results["metadata_complete"]
    
    def test_validate_invalid_transition_matrix(self, sample_hmm_artifact):
        """Test validation with invalid transition matrix."""
        # Create artifact with invalid transition matrix
        sample_hmm_artifact.transition_matrix = [
            [0.5, 0.3, 0.1],  # Doesn't sum to 1
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6]
        ]
        
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={}
        )
        
        # Update hash to match modified artifact
        artifact.artifact_hash = artifact._calculate_artifact_hash(artifact.base_artifact)
        
        results = ArtifactValidator.validate_for_production(artifact)
        
        assert not results["transition_matrix_valid"]
    
    def test_validate_rust_compatibility(self, sample_hmm_artifact):
        """Test Rust compatibility validation."""
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={}
        )
        
        results = ArtifactValidator.validate_rust_compatibility(artifact)
        
        assert results["json_serializable"]
        assert results["dimensions_consistent"]
        assert results["no_invalid_values"]
        assert results["valid_version_format"]
    
    def test_run_all_validations(self, sample_hmm_artifact):
        """Test running all validations."""
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={},
            evaluation_metrics={"aic": 1234.5}
        )
        
        results = ArtifactValidator.run_all_validations(artifact)
        
        assert "all_passed" in results
        assert "production_checks" in results
        assert "rust_compatibility_checks" in results
        assert "failed_checks" in results
        assert results["all_passed"]


class TestArtifactExporter:
    """Tests for ArtifactExporter."""
    
    def test_export_for_production(self, temp_experiment_dir, sample_hmm_artifact):
        """Test exporting artifact for production."""
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={},
            evaluation_metrics={"aic": 1234.5}
        )
        
        output_dir = temp_experiment_dir / "export"
        exported_files = ArtifactExporter.export_for_production(
            artifact, output_dir, validate=True
        )
        
        assert "hmm_artifact" in exported_files
        assert "metadata" in exported_files
        assert "validation_report" in exported_files
        assert "manifest" in exported_files
        
        # Check files exist
        for file_path in exported_files.values():
            assert file_path.exists()
    
    def test_export_without_validation(self, temp_experiment_dir, sample_hmm_artifact):
        """Test exporting without validation."""
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={}
        )
        
        output_dir = temp_experiment_dir / "export_no_val"
        exported_files = ArtifactExporter.export_for_production(
            artifact, output_dir, validate=False
        )
        
        assert len(exported_files) > 0
    
    def test_export_fusion_weights(self, temp_experiment_dir, sample_fusion_weights):
        """Test exporting fusion weights."""
        output_path = temp_experiment_dir / "fusion_weights.json"
        
        exported_path = ArtifactExporter.export_fusion_weights(
            sample_fusion_weights, output_path
        )
        
        assert exported_path.exists()
        
        # Verify content
        with open(exported_path, 'r') as f:
            data = json.load(f)
        
        assert data["version"] == "v1.0"
        assert len(data["state_weights"]) == 3
    
    def test_export_invalid_artifact_raises_error(self, temp_experiment_dir, sample_hmm_artifact):
        """Test that exporting invalid artifact raises error."""
        # Create artifact with invalid transition matrix
        sample_hmm_artifact.transition_matrix = [
            [0.5, 0.3, 0.1],  # Invalid
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6]
        ]
        
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_001",
            researcher="test_researcher",
            training_config={}
        )
        
        # Update hash
        artifact.artifact_hash = artifact._calculate_artifact_hash(artifact.base_artifact)
        
        output_dir = temp_experiment_dir / "export_invalid"
        
        with pytest.raises(ValidationError):
            ArtifactExporter.export_for_production(
                artifact, output_dir, validate=True
            )


class TestIntegrationWorkflow:
    """Integration tests for complete workflow."""
    
    def test_complete_research_to_production_workflow(self, temp_experiment_dir, sample_hmm_artifact):
        """Test complete workflow from research to production."""
        # Step 1: Create research artifact
        artifact = ResearchArtifact.from_hmm_artifact(
            artifact=sample_hmm_artifact,
            experiment_id="exp_prod_001",
            researcher="production_researcher",
            training_config={"n_states": 3, "library": "hmmlearn"},
            evaluation_metrics={"aic": 1234.5, "bic": 1250.0},
            tags=["production-candidate"]
        )
        
        # Step 2: Log experiment
        tracker = ExperimentTracker(temp_experiment_dir)
        exp_id = tracker.log_experiment(artifact, notes="Production candidate")
        
        # Step 3: Validate artifact
        validation_results = ArtifactValidator.run_all_validations(artifact)
        assert validation_results["all_passed"]
        
        # Step 4: Mark as production ready
        artifact.production_ready = True
        artifact.validation_status = validation_results["production_checks"]
        
        # Step 5: Export for production
        export_dir = temp_experiment_dir / "production_export"
        exported_files = ArtifactExporter.export_for_production(
            artifact, export_dir, validate=True
        )
        
        # Verify all steps completed successfully
        assert exp_id == "exp_prod_001"
        assert artifact.production_ready
        assert len(exported_files) >= 3
        
        # Verify exported artifact can be loaded
        with open(exported_files["hmm_artifact"], 'r') as f:
            exported_data = json.load(f)
        
        loaded_artifact = HMMArtifact(**exported_data)
        assert loaded_artifact.n_states == 3
    
    def test_experiment_comparison_workflow(self, temp_experiment_dir, sample_hmm_artifact):
        """Test comparing multiple experiments."""
        tracker = ExperimentTracker(temp_experiment_dir)
        
        # Create multiple experiments with different configurations
        configs = [
            {"n_states": 2, "aic": 1300, "bic": 1320},
            {"n_states": 3, "aic": 1234, "bic": 1250},
            {"n_states": 4, "aic": 1250, "bic": 1280}
        ]
        
        exp_ids = []
        for i, config in enumerate(configs):
            # Modify artifact for different states
            test_artifact = sample_hmm_artifact.model_copy(deep=True)
            
            artifact = ResearchArtifact.from_hmm_artifact(
                artifact=test_artifact,
                experiment_id=f"exp_compare_{i:03d}",
                researcher="comparison_researcher",
                training_config={"n_states": config["n_states"]},
                evaluation_metrics={"aic": config["aic"], "bic": config["bic"]}
            )
            
            exp_id = tracker.log_experiment(artifact)
            exp_ids.append(exp_id)
        
        # Compare experiments
        comparison = tracker.compare_experiments(exp_ids)
        
        assert len(comparison["experiments"]) == 3
        assert "aic" in comparison["best_by_metric"]
        
        # Best model should be the one with lowest AIC
        best_exp_id = comparison["best_by_metric"]["aic"]
        assert best_exp_id == "exp_compare_001"  # Middle one has lowest AIC


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
