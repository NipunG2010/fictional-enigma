# HMM Artifact Management

Production integration and artifact management for HMM research environment.

## Overview

The artifact management module provides comprehensive tools for managing HMM research artifacts, tracking experiments, validating models for production deployment, and exporting artifacts for the Rust inference engine.

## Key Features

- **Research Artifact Management**: Enhanced artifacts with research-specific metadata and versioning
- **Experiment Tracking**: Track and compare multiple experiments with automatic versioning
- **Production Validation**: Comprehensive validation checks for production readiness
- **Rust Compatibility**: Ensure artifacts work with the Rust inference engine
- **Export Utilities**: Export artifacts with proper versioning and metadata

## Components

### 1. ResearchArtifact

Enhanced artifact that wraps `HMMArtifact` with research-specific metadata.

```python
from imp.hmm import ResearchArtifact, HMMTrainer

# Train model
trainer = HMMTrainer(n_states=3)
hmm_artifact = trainer.train(observations)

# Create research artifact
research_artifact = ResearchArtifact.from_hmm_artifact(
    artifact=hmm_artifact,
    experiment_id="exp_001",
    researcher="researcher_name",
    training_config={"n_states": 3, "library": "hmmlearn"},
    evaluation_metrics={"aic": 1234.5, "bic": 1250.0},
    notes="Initial experiment",
    tags=["baseline", "production-candidate"]
)

# Verify integrity
assert research_artifact.verify_integrity()
```

### 2. ExperimentTracker

Track and manage research experiments with versioning.

```python
from imp.hmm import ExperimentTracker
from pathlib import Path

# Initialize tracker
tracker = ExperimentTracker(Path("experiments"))

# Log experiment
exp_id = tracker.log_experiment(
    research_artifact,
    notes="Baseline model with 3 states"
)

# Load experiment
loaded_artifact = tracker.load_experiment("exp_001")

# List experiments
all_experiments = tracker.list_experiments()
production_ready = tracker.list_experiments(production_ready=True)
by_researcher = tracker.list_experiments(researcher="researcher_name")

# Compare experiments
comparison = tracker.compare_experiments(["exp_001", "exp_002", "exp_003"])
print(f"Best by AIC: {comparison['best_by_metric']['aic']}")

# Delete experiment
tracker.delete_experiment("exp_001")
```

### 3. ArtifactValidator

Validate artifacts for production deployment and Rust compatibility.

```python
from imp.hmm import ArtifactValidator

# Validate for production
production_checks = ArtifactValidator.validate_for_production(research_artifact)
print(f"Integrity check: {production_checks['integrity_check']}")
print(f"Transition matrix valid: {production_checks['transition_matrix_valid']}")

# Validate Rust compatibility
rust_checks = ArtifactValidator.validate_rust_compatibility(research_artifact)
print(f"JSON serializable: {rust_checks['json_serializable']}")
print(f"Dimensions consistent: {rust_checks['dimensions_consistent']}")

# Run all validations
all_results = ArtifactValidator.run_all_validations(research_artifact)
if all_results["all_passed"]:
    print("✓ All validation checks passed")
    research_artifact.production_ready = True
else:
    print(f"✗ Failed checks: {all_results['failed_checks']}")
```

### 4. ArtifactExporter

Export artifacts for production deployment.

```python
from imp.hmm import ArtifactExporter
from pathlib import Path

# Export for production
exported_files = ArtifactExporter.export_for_production(
    research_artifact,
    output_dir=Path("production_export"),
    validate=True  # Validate before export
)

# Exported files include:
# - hmm_artifact.json: Base HMM artifact
# - research_metadata.json: Research metadata
# - validation_report.json: Validation results
# - deployment_manifest.json: Deployment manifest

# Export fusion weights separately
from imp.hmm import FusionWeights

fusion_weights = FusionWeights(
    version="v1.0",
    state_weights=[
        {"w_ldc": 0.5, "w_mr": 0.3, "w_tsmom": 0.2},
        {"w_ldc": 0.3, "w_mr": 0.4, "w_tsmom": 0.3},
        {"w_ldc": 0.4, "w_mr": 0.2, "w_tsmom": 0.4}
    ],
    model_version="v1.0",
    training_metrics={"sharpe_ratio": 1.5},
    metadata={"n_states": 3}
)

ArtifactExporter.export_fusion_weights(
    fusion_weights,
    output_path=Path("production_export/fusion_weights.json")
)
```

## Complete Workflow Example

```python
from pathlib import Path
import numpy as np
from imp.hmm import (
    HMMTrainer,
    ResearchArtifact,
    ExperimentTracker,
    ArtifactValidator,
    ArtifactExporter
)

# Step 1: Initialize experiment tracker
tracker = ExperimentTracker(Path("experiments"))

# Step 2: Train multiple models
best_artifact = None
best_aic = float('inf')

for n_states in [2, 3, 4]:
    # Train model
    trainer = HMMTrainer(n_states=n_states)
    hmm_artifact = trainer.train(observations, n_iterations=100)
    
    # Create research artifact
    research_artifact = ResearchArtifact.from_hmm_artifact(
        artifact=hmm_artifact,
        experiment_id=f"exp_{n_states}states",
        researcher="researcher_name",
        training_config={"n_states": n_states},
        evaluation_metrics={
            "aic": hmm_artifact.metadata["aic"],
            "bic": hmm_artifact.metadata["bic"]
        }
    )
    
    # Log experiment
    tracker.log_experiment(research_artifact)
    
    # Track best model
    aic = hmm_artifact.metadata["aic"]
    if aic < best_aic:
        best_aic = aic
        best_artifact = research_artifact

# Step 3: Validate best model
validation_results = ArtifactValidator.run_all_validations(best_artifact)

if validation_results["all_passed"]:
    best_artifact.production_ready = True
    
    # Step 4: Export for production
    exported_files = ArtifactExporter.export_for_production(
        best_artifact,
        output_dir=Path("production_export"),
        validate=True
    )
    
    print(f"✓ Model ready for deployment: {best_artifact.research_metadata.experiment_id}")
else:
    print(f"✗ Validation failed: {validation_results['failed_checks']}")

# Step 5: Compare all experiments
comparison = tracker.compare_experiments([f"exp_{n}states" for n in [2, 3, 4]])
print(f"Best model by AIC: {comparison['best_by_metric']['aic']}")
```

## Validation Checks

### Production Validation Checks

1. **integrity_check**: Artifact hash matches content
2. **hmm_artifact_valid**: Base HMM artifact is valid
3. **transition_matrix_valid**: Transition matrix rows sum to 1
4. **covariances_positive_definite**: All covariance matrices are positive definite
5. **has_evaluation_metrics**: Evaluation metrics are present
6. **reasonable_n_states**: Number of states is between 2 and 10
7. **valid_training_window**: Training window is valid
8. **metadata_complete**: Required metadata fields are present

### Rust Compatibility Checks

1. **json_serializable**: Artifact can be serialized to JSON
2. **dimensions_consistent**: Array dimensions are consistent
3. **no_invalid_values**: No NaN or Inf values present
4. **valid_version_format**: Version format is valid

## Research Metadata Fields

- **experiment_id**: Unique experiment identifier
- **researcher**: Name or ID of researcher
- **created_at**: Creation timestamp
- **notebook_path**: Path to source notebook (optional)
- **training_config**: Training configuration used
- **library_used**: HMM library used (hmmlearn/pomegranate)
- **evaluation_metrics**: Evaluation metrics (AIC, BIC, etc.)
- **cross_validation_scores**: CV scores (optional)
- **comparison_baseline**: Baseline model for comparison (optional)
- **performance_ranking**: Performance rank (optional)
- **notes**: Additional research notes
- **tags**: Tags for categorization

## Deployment Manifest

When exporting for production, a deployment manifest is created with:

```json
{
  "artifact_version": "1.0.0",
  "artifact_hash": "abc123...",
  "experiment_id": "exp_001",
  "created_at": "2025-10-04T12:00:00",
  "researcher": "researcher_name",
  "n_states": 3,
  "library": "hmmlearn",
  "production_ready": true,
  "exported_files": {
    "hmm_artifact": "hmm_artifact.json",
    "metadata": "research_metadata.json",
    "validation_report": "validation_report.json",
    "manifest": "deployment_manifest.json"
  }
}
```

## Integration with Rust Inference Engine

The exported `hmm_artifact.json` is compatible with the Rust inference engine:

1. **JSON Format**: All data is in JSON format
2. **Consistent Dimensions**: Array dimensions match expected format
3. **No Invalid Values**: No NaN or Inf values
4. **Validated Structure**: All required fields are present and valid

## Error Handling

The module provides specific exceptions:

- **ArtifactManagementError**: Base exception for artifact management
- **ValidationError**: Raised during validation failures
- **ExportError**: Raised during export failures

```python
from imp.hmm import ValidationError, ExportError

try:
    exported_files = ArtifactExporter.export_for_production(
        artifact, output_dir, validate=True
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
except ExportError as e:
    print(f"Export failed: {e}")
```

## Best Practices

1. **Always validate before production**: Use `validate=True` when exporting
2. **Track all experiments**: Log every experiment for reproducibility
3. **Use meaningful experiment IDs**: Include date, configuration, or purpose
4. **Add descriptive tags**: Tag experiments for easy filtering
5. **Document training config**: Include all relevant training parameters
6. **Verify integrity**: Check artifact integrity after loading
7. **Compare experiments**: Use comparison tools to select best model
8. **Version artifacts**: Use semantic versioning for artifacts

## Examples

See `py/examples/artifact_management_example.py` for complete examples demonstrating:

1. Basic artifact creation
2. Experiment tracking
3. Artifact validation
4. Export for production
5. Complete research-to-production workflow

## Testing

Run tests with:

```bash
cd py
python -m pytest tests/test_artifact_management.py -v
```

## Requirements

This module satisfies the following requirements from the HMM Research Environment spec:

- **6.1**: Export models as HMMArtifact and FusionWeights compatible with production
- **6.2**: Validate artifact format and Rust inference engine compatibility
- **6.3**: Provide model versioning and metadata tracking
- **6.4**: Validate artifacts and provide correction guidance
- **6.5**: Support automated testing before production deployment
