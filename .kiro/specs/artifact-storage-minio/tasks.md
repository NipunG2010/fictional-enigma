# Implementation Plan

- [x] 1. Create MinIOConfig and MinIOArtifactStore classes
  - Implement MinIOConfig dataclass with environment variable support
  - Create MinIOArtifactStore with Minio client initialization
  - Add _ensure_bucket_exists method creating bucket if missing
  - Implement _upload_json and _download_json helper methods
  - Add connection validation and error handling
  - _Requirements: 1.1, 1.3, 1.4, 2.4_

- [X] 2. Implement artifact upload with versioning
  - Add upload_artifact method accepting ResearchArtifact and FusionWeights
  - Create structured paths (experiment_id/version/artifact.json)
  - Upload HMM artifact, fusion weights, and metadata as separate JSON files
  - Return dictionary with uploaded object paths
  - Add retry logic with exponential backoff for failed uploads
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [X] 3. Implement artifact download and listing
  - Add download_artifact method supporting version parameter
  - Implement _get_latest_version for "latest" version resolution
  - Create list_artifacts method with filtering by experiment_id and tags
  - Add get_production_artifact for fetching production-tagged artifacts
  - Include integrity validation using artifact hash after download
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [X] 4. Add tagging and deployment workflow support
  - Implement tag_artifact method for adding tags to artifacts
  - Add validation ensuring only validated artifacts can be tagged "production"
  - Create deployment helper methods for production artifact retrieval
  - Add metadata tracking for deployment timestamps and versions
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2_

- [X] 5. Extend ExperimentTracker with MinIO integration
  - Add use_minio parameter to ExperimentTracker constructor
  - Integrate MinIOArtifactStore as optional storage backend
  - Update log_experiment to upload to both local and MinIO storage
  - Add methods for syncing between local and MinIO storage
  - Maintain backward compatibility with local-only storage
  - _Requirements: 1.5, 2.5, 4.3, 4.4, 4.5_

- [x] 6. Create comprehensive testing suite
  - Unit tests for MinIOArtifactStore with mocked Minio client
  - Integration tests with real MinIO instance from docker-compose
  - Test upload/download round-trip with artifact validation
  - Test versioning, tagging, and "latest" resolution
  - Test error handling (connection failures, missing artifacts)
  - Test production deployment workflow end-to-end
  - _Requirements: 1.4, 2.4, 3.4, 4.4_

- [x] 7. Create deployment notebook demonstrating MinIO workflow
  - Show training model and uploading to MinIO
  - Demonstrate listing and downloading artifacts by version
  - Show tagging workflow (staging → production)
  - Demonstrate production deployment artifact retrieval
  - Include troubleshooting guide for common MinIO issues
  - _Requirements: 1.5, 2.5, 3.5, 4.5_
