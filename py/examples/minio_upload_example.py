"""
Example demonstrating MinIO artifact upload with versioning.

This example shows how to:
1. Create a research artifact
2. Upload it to MinIO with versioning
3. Handle upload with retry logic
"""

from datetime import datetime
from imp.hmm.models import HMMArtifact, FusionWeights
from imp.hmm.artifact_management import (
    MinIOConfig,
    MinIOArtifactStore,
    ResearchArtifact,
    ResearchMetadata
)


def main():
    """Demonstrate MinIO artifact upload."""
    
    # Create HMM artifact
    hmm_artifact = HMMArtifact(
        version="1.0.0",
        n_states=3,
        n_features=2,
        transition_matrix=[
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6]
        ],
        initial_probabilities=[0.33, 0.33, 0.34],
        means=[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ],
        covariances=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]]
        ],
        training_window_start=int(datetime(2024, 1, 1).timestamp()),
        training_window_end=int(datetime(2024, 12, 31).timestamp()),
        metadata={
            "library": "hmmlearn",
            "algorithm": "baum-welch",
            "covariance_type": "full"
        }
    )
    
    # Create research metadata
    research_metadata = ResearchMetadata(
        experiment_id="demo_experiment_001",
        researcher="demo_user",
        training_config={
            "n_iter": 100,
            "tol": 1e-4,
            "algorithm": "baum-welch"
        },
        library_used="hmmlearn",
        evaluation_metrics={
            "log_likelihood": -1234.5,
            "aic": 2500.0,
            "bic": 2550.0
        },
        tags=["demo", "example"]
    )
    
    # Create research artifact
    research_artifact = ResearchArtifact(
        base_artifact=hmm_artifact,
        research_metadata=research_metadata,
        artifact_version="v1.0.0",
        artifact_hash="demo_hash_123",
        production_ready=False
    )
    
    # Create fusion weights (optional)
    fusion_weights = FusionWeights(
        version="1.0.0",
        state_weights=[
            {"signal_a": 0.6, "signal_b": 0.4},
            {"signal_a": 0.5, "signal_b": 0.5},
            {"signal_a": 0.7, "signal_b": 0.3}
        ],
        model_version="1.0.0",
        training_metrics={"sharpe_ratio": 1.5},
        metadata={
            "optimization_method": "grid_search",
            "n_states": 3
        }
    )
    
    # Initialize MinIO store
    # Note: This requires MinIO to be running (e.g., via docker-compose)
    try:
        config = MinIOConfig.from_env()
        store = MinIOArtifactStore(config)
        
        print("MinIO connection established successfully!")
        print(f"Bucket: {config.bucket_name}")
        print(f"Endpoint: {config.endpoint}")
        
        # Upload artifact with versioning
        print("\nUploading artifact...")
        result = store.upload_artifact(
            research_artifact,
            fusion_weights=fusion_weights,
            tags=["demo", "validated"],
            max_retries=3,
            retry_delay=1.0
        )
        
        print("\nUpload successful!")
        print("Uploaded files:")
        for key, path in result.items():
            print(f"  - {key}: {path}")
        
        # Example of structured paths
        print("\nStructured path format:")
        print(f"  {research_metadata.experiment_id}/{research_artifact.artifact_version}/")
        print(f"    ├── hmm_artifact.json")
        print(f"    ├── fusion_weights.json")
        print(f"    └── metadata.json")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure MinIO is running (docker-compose up -d)")


if __name__ == "__main__":
    main()
