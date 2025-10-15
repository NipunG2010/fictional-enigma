#!/usr/bin/env python3
"""
Verification script for Task 4: Tagging and Deployment Workflow Support

This script tests:
1. tag_artifact method with production validation
2. Validation ensuring only validated artifacts can be tagged "production"
3. Deployment helper methods for production artifact retrieval
4. Metadata tracking for deployment timestamps and versions
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imp.hmm.artifact_management import (
    MinIOArtifactStore,
    MinIOConfig,
    ResearchArtifact,
    ResearchMetadata,
    ArtifactValidator,
    ValidationError,
    MinIODownloadError
)
from imp.hmm.models import HMMArtifact, FusionWeights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_artifact(experiment_id: str, production_ready: bool = False) -> ResearchArtifact:
    """Create a test artifact for verification."""
    # Create a simple HMM artifact
    hmm_artifact = HMMArtifact(
        version="1.0.0",
        n_states=2,
        n_features=3,
        transition_matrix=[[0.7, 0.3], [0.4, 0.6]],
        initial_probabilities=[0.6, 0.4],
        means=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        covariances=[
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ],
        training_window_start=datetime(2024, 1, 1),
        training_window_end=datetime(2024, 12, 31),
        metadata={
            "library": "hmmlearn",
            "algorithm": "baum-welch",
            "covariance_type": "full"
        }
    )
    
    # Create research artifact
    research_artifact = ResearchArtifact.from_hmm_artifact(
        artifact=hmm_artifact,
        experiment_id=experiment_id,
        researcher="test_researcher",
        training_config={"n_iter": 100, "tol": 1e-4},
        evaluation_metrics={"log_likelihood": -1234.56, "aic": 2500.0},
        artifact_version="1.0.0",
        library_used="hmmlearn",
        notes="Test artifact for task 4 verification"
    )
    
    # Set production ready status
    research_artifact.production_ready = production_ready
    
    if production_ready:
        # Run validation to populate validation_status
        validation_results = ArtifactValidator.run_all_validations(research_artifact)
        research_artifact.validation_status = {
            **validation_results["production_checks"],
            **validation_results["rust_compatibility_checks"]
        }
    
    return research_artifact


def test_tag_artifact_validation():
    """Test 1: Verify tag_artifact validates production tags."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Tag Artifact with Production Validation")
    logger.info("="*80)
    
    try:
        # Initialize MinIO store
        config = MinIOConfig.from_env()
        store = MinIOArtifactStore(config)
        
        # Create and upload a non-production-ready artifact
        logger.info("Creating non-production-ready artifact...")
        artifact = create_test_artifact("test_task4_validation", production_ready=False)
        
        logger.info("Uploading artifact to MinIO...")
        store.upload_artifact(artifact)
        
        # Try to tag as production (should fail)
        logger.info("Attempting to tag non-validated artifact as 'production' (should fail)...")
        try:
            store.tag_artifact("test_task4_validation", "1.0.0", "production")
            logger.error("❌ FAILED: Should have raised ValidationError for non-validated artifact")
            return False
        except ValidationError as e:
            logger.info(f"✅ PASSED: Correctly rejected non-validated artifact: {e}")
        
        # Tag as staging (should succeed)
        logger.info("Tagging as 'staging' (should succeed)...")
        store.tag_artifact("test_task4_validation", "1.0.0", "staging", validate_production=False)
        logger.info("✅ PASSED: Successfully tagged as 'staging'")
        
        # Create and upload a production-ready artifact
        logger.info("\nCreating production-ready artifact...")
        prod_artifact = create_test_artifact("test_task4_prod", production_ready=True)
        
        logger.info("Uploading production-ready artifact to MinIO...")
        store.upload_artifact(prod_artifact)
        
        # Tag as production (should succeed)
        logger.info("Tagging validated artifact as 'production' (should succeed)...")
        store.tag_artifact("test_task4_prod", "1.0.0", "production")
        logger.info("✅ PASSED: Successfully tagged validated artifact as 'production'")
        
        # Verify tagging history
        logger.info("\nVerifying tagging history...")
        artifact_data = store.download_artifact("test_task4_prod", "1.0.0")
        metadata = artifact_data["metadata"]
        
        if "tagging_history" in metadata:
            logger.info(f"Tagging history: {json.dumps(metadata['tagging_history'], indent=2)}")
            logger.info("✅ PASSED: Tagging history tracked correctly")
        else:
            logger.error("❌ FAILED: Tagging history not found in metadata")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FAILED: Unexpected error: {e}", exc_info=True)
        return False


def test_deployment_workflow():
    """Test 2: Verify deployment helper methods and metadata tracking."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Deployment Workflow and Metadata Tracking")
    logger.info("="*80)
    
    try:
        # Initialize MinIO store
        config = MinIOConfig.from_env()
        store = MinIOArtifactStore(config)
        
        # Create and upload production-ready artifact
        logger.info("Creating and uploading production-ready artifact...")
        artifact = create_test_artifact("test_task4_deploy", production_ready=True)
        store.upload_artifact(artifact)
        
        # Tag as production
        logger.info("Tagging as production...")
        store.tag_artifact("test_task4_deploy", "1.0.0", "production")
        
        # Deploy artifact
        logger.info("Deploying artifact to production environment...")
        deployment_record = store.deploy_artifact(
            "test_task4_deploy",
            "1.0.0",
            deployment_environment="production",
            deployment_notes="Initial production deployment for task 4 verification"
        )
        
        logger.info(f"Deployment record: {json.dumps(deployment_record, indent=2)}")
        
        # Verify deployment record structure
        required_fields = [
            "experiment_id", "version", "deployment_environment",
            "deployed_at", "deployment_notes", "artifact_hash"
        ]
        
        for field in required_fields:
            if field not in deployment_record:
                logger.error(f"❌ FAILED: Missing field '{field}' in deployment record")
                return False
        
        logger.info("✅ PASSED: Deployment record has all required fields")
        
        # Verify deployment timestamp
        deployed_at = deployment_record["deployed_at"]
        try:
            datetime.fromisoformat(deployed_at)
            logger.info(f"✅ PASSED: Deployment timestamp is valid ISO format: {deployed_at}")
        except ValueError:
            logger.error(f"❌ FAILED: Invalid deployment timestamp format: {deployed_at}")
            return False
        
        # Get deployment history
        logger.info("\nRetrieving deployment history...")
        history = store.get_deployment_history("test_task4_deploy")
        
        if not history:
            logger.error("❌ FAILED: Deployment history is empty")
            return False
        
        logger.info(f"Found {len(history)} deployment record(s)")
        logger.info(f"Latest deployment: {json.dumps(history[0], indent=2)}")
        logger.info("✅ PASSED: Deployment history retrieved successfully")
        
        # Deploy to staging
        logger.info("\nDeploying to staging environment...")
        staging_record = store.deploy_artifact(
            "test_task4_deploy",
            "1.0.0",
            deployment_environment="staging",
            deployment_notes="Staging deployment"
        )
        
        # Get deployment history filtered by environment
        logger.info("Retrieving production-only deployment history...")
        prod_history = store.get_deployment_history(
            "test_task4_deploy",
            environment="production"
        )
        
        if len(prod_history) != 1:
            logger.error(f"❌ FAILED: Expected 1 production deployment, found {len(prod_history)}")
            return False
        
        logger.info("✅ PASSED: Environment filtering works correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FAILED: Unexpected error: {e}", exc_info=True)
        return False


def test_production_artifact_fallback():
    """Test 3: Verify production artifact retrieval with fallback."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Production Artifact Retrieval with Fallback")
    logger.info("="*80)
    
    try:
        # Initialize MinIO store
        config = MinIOConfig.from_env()
        store = MinIOArtifactStore(config)
        
        # Create and upload multiple production artifacts
        logger.info("Creating multiple production artifacts...")
        
        # First artifact (older)
        artifact1 = create_test_artifact("test_task4_fallback_v1", production_ready=True)
        artifact1.artifact_version = "1.0.0"
        artifact1.research_metadata.created_at = datetime(2024, 1, 1)
        store.upload_artifact(artifact1)
        store.tag_artifact("test_task4_fallback_v1", "1.0.0", "production")
        logger.info("Uploaded and tagged artifact v1.0.0")
        
        # Second artifact (newer)
        artifact2 = create_test_artifact("test_task4_fallback_v2", production_ready=True)
        artifact2.artifact_version = "2.0.0"
        artifact2.research_metadata.created_at = datetime(2024, 6, 1)
        store.upload_artifact(artifact2)
        store.tag_artifact("test_task4_fallback_v2", "2.0.0", "production")
        logger.info("Uploaded and tagged artifact v2.0.0")
        
        # Get production artifact (should get latest)
        logger.info("\nRetrieving latest production artifact...")
        prod_artifact = store.get_production_artifact(fallback_to_last_known=True)
        
        if not prod_artifact:
            logger.error("❌ FAILED: No production artifact returned")
            return False
        
        logger.info(f"Retrieved artifact: {prod_artifact['metadata']['experiment_id']} "
                   f"v{prod_artifact['metadata']['version']}")
        
        # Verify it's the latest one
        if prod_artifact['metadata']['experiment_id'] == "test_task4_fallback_v2":
            logger.info("✅ PASSED: Retrieved latest production artifact")
        else:
            logger.error("❌ FAILED: Did not retrieve latest production artifact")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FAILED: Unexpected error: {e}", exc_info=True)
        return False


def test_artifact_lineage():
    """Test 4: Verify artifact lineage tracking."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Artifact Lineage Tracking")
    logger.info("="*80)
    
    try:
        # Initialize MinIO store
        config = MinIOConfig.from_env()
        store = MinIOArtifactStore(config)
        
        # Create and upload artifact
        logger.info("Creating artifact with comprehensive metadata...")
        artifact = create_test_artifact("test_task4_lineage", production_ready=True)
        store.upload_artifact(artifact)
        
        # Tag and deploy
        store.tag_artifact("test_task4_lineage", "1.0.0", "staging", validate_production=False)
        store.tag_artifact("test_task4_lineage", "1.0.0", "production")
        store.deploy_artifact("test_task4_lineage", "1.0.0", "production")
        
        # Get lineage
        logger.info("\nRetrieving artifact lineage...")
        lineage = store.get_artifact_lineage("test_task4_lineage", "1.0.0")
        
        logger.info(f"Lineage information:\n{json.dumps(lineage, indent=2, default=str)}")
        
        # Verify lineage contains required information
        required_fields = [
            "experiment_id", "version", "created_at", "researcher",
            "training_config", "library_used", "evaluation_metrics",
            "tags", "tagging_history", "deployment_history", "artifact_hash"
        ]
        
        for field in required_fields:
            if field not in lineage:
                logger.error(f"❌ FAILED: Missing field '{field}' in lineage")
                return False
        
        logger.info("✅ PASSED: Lineage contains all required fields")
        
        # Verify tagging history
        if len(lineage["tagging_history"]) < 2:
            logger.error("❌ FAILED: Tagging history incomplete")
            return False
        
        logger.info(f"✅ PASSED: Tagging history has {len(lineage['tagging_history'])} entries")
        
        # Verify deployment history
        if len(lineage["deployment_history"]) < 1:
            logger.error("❌ FAILED: Deployment history incomplete")
            return False
        
        logger.info(f"✅ PASSED: Deployment history has {len(lineage['deployment_history'])} entries")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FAILED: Unexpected error: {e}", exc_info=True)
        return False


def test_remove_tag():
    """Test 5: Verify tag removal functionality."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Tag Removal")
    logger.info("="*80)
    
    try:
        # Initialize MinIO store
        config = MinIOConfig.from_env()
        store = MinIOArtifactStore(config)
        
        # Create and upload artifact
        logger.info("Creating and uploading artifact...")
        artifact = create_test_artifact("test_task4_remove_tag", production_ready=False)
        store.upload_artifact(artifact)
        
        # Add tags
        logger.info("Adding tags...")
        store.tag_artifact("test_task4_remove_tag", "1.0.0", "experimental", validate_production=False)
        store.tag_artifact("test_task4_remove_tag", "1.0.0", "staging", validate_production=False)
        
        # Verify tags were added
        artifact_data = store.download_artifact("test_task4_remove_tag", "1.0.0")
        tags = artifact_data["metadata"].get("tags", [])
        
        if "experimental" not in tags or "staging" not in tags:
            logger.error("❌ FAILED: Tags were not added correctly")
            return False
        
        logger.info(f"Tags after adding: {tags}")
        
        # Remove a tag
        logger.info("Removing 'experimental' tag...")
        store.remove_tag("test_task4_remove_tag", "1.0.0", "experimental")
        
        # Verify tag was removed
        artifact_data = store.download_artifact("test_task4_remove_tag", "1.0.0")
        tags = artifact_data["metadata"].get("tags", [])
        
        if "experimental" in tags:
            logger.error("❌ FAILED: Tag was not removed")
            return False
        
        if "staging" not in tags:
            logger.error("❌ FAILED: Other tags were incorrectly removed")
            return False
        
        logger.info(f"Tags after removal: {tags}")
        logger.info("✅ PASSED: Tag removal works correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FAILED: Unexpected error: {e}", exc_info=True)
        return False


def main():
    """Run all verification tests."""
    logger.info("="*80)
    logger.info("Task 4 Verification: Tagging and Deployment Workflow Support")
    logger.info("="*80)
    
    # Check if MinIO is available
    try:
        config = MinIOConfig.from_env()
        store = MinIOArtifactStore(config)
        logger.info("✅ MinIO connection successful")
    except Exception as e:
        logger.error(f"❌ MinIO connection failed: {e}")
        logger.error("Please ensure MinIO is running (docker-compose up -d)")
        return 1
    
    # Run tests
    tests = [
        ("Tag Artifact Validation", test_tag_artifact_validation),
        ("Deployment Workflow", test_deployment_workflow),
        ("Production Artifact Fallback", test_production_artifact_fallback),
        ("Artifact Lineage", test_artifact_lineage),
        ("Tag Removal", test_remove_tag)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}", exc_info=True)
            results.append((test_name, False))
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
