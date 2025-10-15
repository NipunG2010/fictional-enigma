"""
Example demonstrating basic MinIO artifact store usage.

This script shows how to use the MinIOConfig and MinIOArtifactStore classes
for storing and retrieving HMM artifacts.
"""

from imp.hmm.artifact_management import MinIOConfig, MinIOArtifactStore

# Example 1: Create MinIOConfig with default values
print("Example 1: Default Configuration")
config = MinIOConfig()
print(f"  Endpoint: {config.endpoint}")
print(f"  Bucket: {config.bucket_name}")
print(f"  Secure: {config.secure}")
print()

# Example 2: Create MinIOConfig with custom values
print("Example 2: Custom Configuration")
custom_config = MinIOConfig(
    endpoint="minio.example.com:9000",
    access_key="my_access_key",
    secret_key="my_secret_key",
    secure=True,
    bucket_name="my-custom-bucket"
)
print(f"  Endpoint: {custom_config.endpoint}")
print(f"  Bucket: {custom_config.bucket_name}")
print(f"  Secure: {custom_config.secure}")
print()

# Example 3: Load configuration from environment variables
print("Example 3: Configuration from Environment")
env_config = MinIOConfig.from_env()
print(f"  Endpoint: {env_config.endpoint}")
print(f"  Bucket: {env_config.bucket_name}")
print()

# Example 4: Initialize MinIOArtifactStore (requires MinIO server)
print("Example 4: MinIOArtifactStore Initialization")
print("  Note: This requires a running MinIO server")
print("  To start MinIO: docker-compose up -d minio")
print()
print("  Example code:")
print("  ```python")
print("  store = MinIOArtifactStore()")
print("  # Upload JSON data")
print("  store._upload_json('test/data.json', {'key': 'value'})")
print("  # Download JSON data")
print("  data = store._download_json('test/data.json')")
print("  ```")
print()

# Example 5: Key Features
print("Example 5: Key Features")
print("  ✓ Environment variable configuration")
print("  ✓ Automatic bucket creation")
print("  ✓ Connection validation")
print("  ✓ Retry logic with exponential backoff")
print("  ✓ Comprehensive error handling")
print("  ✓ JSON upload/download helpers")
