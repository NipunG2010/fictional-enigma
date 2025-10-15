# MinIO Artifact Storage Testing Guide

This guide describes the comprehensive testing suite for MinIO artifact storage functionality.

## Overview

The testing suite covers all aspects of MinIO integration for HMM artifact storage:

1. **Unit Tests** - Test individual components with mocked MinIO client
2. **Integration Tests** - Test full workflow with real MinIO instance
3. **Error Handling** - Test connection failures, missing artifacts, and edge cases
4. **Production Workflow** - Test complete deployment workflow from upload to production

## Test Files

### Unit Tests (Mocked MinIO)

These tests don't require a running MinIO instance and use mocked clients:

- **`test_minio_artifact_store.py`** - Tests for MinIOArtifactStore class
  - MinIOConfig configuration and environment variables
  - Bucket creation and connection validation
  - Upload/download JSON helpers with retry logic
  - Artifact upload with versioning and structured paths
  - Error handling and validation

- **`test_experiment_tracker_minio.py`** - Tests for ExperimentTracker MinIO integration
  - Backward compatibility with local-only storage
  - MinIO integration with automatic upload
  - Sync methods (sync_to_minio, sync_from_minio)
  - Graceful fallback when MinIO unavailable
  - MinIO status reporting

### Integration Tests (Real MinIO)

These tests require a running MinIO instance and test the full workflow:

- **`test_minio_integration.py`** - End-to-end integration tests
  - Connection and initialization
  - Upload/download round-trip with artifact validation
  - Versioning and "latest" resolution
  - Tagging functionality (staging, production, experimental)
  - Production deployment workflow
  - Error handling with real MinIO
  - ExperimentTracker integration

## Running Tests

### Quick Start

Run all tests (unit + integration):
```bash
python tests/run_minio_tests.py
```

### Unit Tests Only

Run only unit tests (no MinIO required):
```bash
python tests/run_minio_tests.py --unit-only
```

Or directly with pytest:
```bash
pytest tests/test_minio_artifact_store.py tests/test_experiment_tracker_minio.py -v
```

### Integration Tests Only

Run only integration tests (requires MinIO):
```bash
python tests/run_minio_tests.py --integration-only
```

Or directly with pytest:
```bash
pytest tests/test_minio_integration.py -m integration -v
```

### With Coverage

Run tests with coverage reporting:
```bash
python tests/run_minio_tests.py --coverage
```

## Setting Up MinIO for Integration Tests

Integration tests require a running MinIO instance. Use docker-compose:

### Start MinIO

```bash
# From project root
docker-compose up -d
```

This starts MinIO on `localhost:9000` with default credentials:
- Access Key: `minioadmin`
- Secret Key: `minioadmin123`
- Bucket: `hmm-artifacts` (created automatically)

### Verify MinIO is Running

```bash
# Check if MinIO container is running
docker-compose ps

# Access MinIO console
# Open browser to http://localhost:9001
# Login with minioadmin/minioadmin123
```

### Stop MinIO

```bash
docker-compose down
```

## Test Coverage

### Requirements Coverage

The test suite covers all requirements from the spec:

#### Requirement 1.4 - Upload Error Handling
- ✅ Retry logic with exponential backoff
- ✅ Clear error messages on failure
- ✅ Connection validation

#### Requirement 2.4 - Download Error Handling
- ✅ Missing artifact error messages
- ✅ Integrity validation using hash
- ✅ Graceful handling of corrupted metadata

#### Requirement 3.4 - Deployment Workflow
- ✅ Production tag validation
- ✅ Fallback to last known good version
- ✅ Deployment history tracking
- ✅ Environment-specific deployments

#### Requirement 4.4 - Metadata Tracking
- ✅ Comprehensive metadata storage
- ✅ Lineage tracking
- ✅ Tagging history
- ✅ Deployment timestamps

### Test Categories

#### Connection Tests
- Successful connection to MinIO
- Bucket creation and validation
- Invalid connection handling
- Connection failure graceful fallback

#### Upload/Download Tests
- HMM artifact upload and download
- Fusion weights upload and download
- Artifact hash validation
- Round-trip data integrity
- Missing artifact error handling

#### Versioning Tests
- Multiple version uploads
- Semantic versioning ordering
- "latest" version resolution
- Version-specific downloads

#### Tagging Tests
- Adding tags to artifacts
- Production tag validation (requires validated artifact)
- Listing artifacts by tag
- Tag removal
- Tagging history tracking

#### Deployment Tests
- Full deployment workflow (staging → production)
- Production artifact retrieval
- Fallback to last known good version
- Deployment history tracking
- Environment-specific deployments
- Artifact lineage tracking

#### Error Handling Tests
- Network error simulation
- Missing artifact error messages
- Corrupted metadata handling
- Connection failure recovery
- Upload retry logic
- Max retries exceeded

#### Integration Tests
- ExperimentTracker with MinIO enabled
- Automatic upload during log_experiment
- Sync to MinIO (local → remote)
- Sync from MinIO (remote → local)
- MinIO status reporting
- Backward compatibility

## Test Execution Flow

### Unit Tests Flow

```
1. Mock MinIO client
2. Test individual methods
3. Verify correct API calls
4. Test error conditions
5. Validate retry logic
```

### Integration Tests Flow

```
1. Check MinIO availability
2. Create test artifacts
3. Upload to real MinIO
4. Download and verify
5. Test versioning
6. Test tagging
7. Test deployment workflow
8. Clean up (optional)
```

## Troubleshooting

### Integration Tests Skipped

If integration tests are skipped with message "MinIO is not available":

1. Check if MinIO is running:
   ```bash
   docker-compose ps
   ```

2. Check MinIO logs:
   ```bash
   docker-compose logs minio
   ```

3. Verify connection:
   ```bash
   curl http://localhost:9000/minio/health/live
   ```

4. Check environment variables:
   ```bash
   echo $MINIO_ENDPOINT
   echo $MINIO_ACCESS_KEY
   echo $MINIO_SECRET_KEY
   ```

### Connection Refused Errors

If you see "Connection refused" errors:

1. Ensure MinIO is running on the correct port (9000)
2. Check if another service is using port 9000
3. Verify docker-compose configuration
4. Try restarting MinIO:
   ```bash
   docker-compose restart minio
   ```

### Test Failures

If tests fail:

1. Check MinIO logs for errors
2. Verify bucket exists and is accessible
3. Check credentials are correct
4. Ensure sufficient disk space
5. Try running tests individually to isolate issues

## Continuous Integration

For CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Start MinIO
  run: docker-compose up -d minio

- name: Wait for MinIO
  run: |
    timeout 30 bash -c 'until curl -f http://localhost:9000/minio/health/live; do sleep 1; done'

- name: Run Unit Tests
  run: pytest tests/test_minio_artifact_store.py tests/test_experiment_tracker_minio.py -v

- name: Run Integration Tests
  run: pytest tests/test_minio_integration.py -m integration -v

- name: Stop MinIO
  run: docker-compose down
```

## Adding New Tests

When adding new MinIO functionality:

1. **Add unit tests** in `test_minio_artifact_store.py` or `test_experiment_tracker_minio.py`
   - Mock the MinIO client
   - Test individual methods
   - Test error conditions

2. **Add integration tests** in `test_minio_integration.py`
   - Test with real MinIO instance
   - Test full workflow
   - Mark with `@pytest.mark.integration`

3. **Update this guide** with new test coverage

## Performance Considerations

Integration tests may be slower than unit tests because they:
- Connect to real MinIO instance
- Upload/download actual data
- Test retry logic with delays

For faster development iteration:
- Run unit tests first (`--unit-only`)
- Run integration tests before committing
- Use CI/CD for full test suite

## Test Data Cleanup

Integration tests create test artifacts in MinIO. To clean up:

```bash
# Option 1: Restart MinIO (clears all data)
docker-compose restart minio

# Option 2: Use MinIO console to delete test artifacts
# Open http://localhost:9001 and delete artifacts starting with "integration_test_"

# Option 3: Use MinIO client (mc)
mc rm --recursive --force myminio/hmm-artifacts/integration_test_*
```

## Summary

The MinIO testing suite provides comprehensive coverage of:
- ✅ All upload/download functionality
- ✅ Versioning and "latest" resolution
- ✅ Tagging and production deployment
- ✅ Error handling and retry logic
- ✅ ExperimentTracker integration
- ✅ Backward compatibility
- ✅ Production workflow end-to-end

Run tests regularly to ensure MinIO integration remains stable and reliable.
