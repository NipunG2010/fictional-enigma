//! S3/MinIO uploader for audit log archival
//! 
//! This module provides secure and reliable upload of audit log files to S3-compatible
//! object storage (AWS S3, MinIO) with retry logic, batch processing, and credential management.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::time::{sleep, timeout};
use tracing::{debug, info, warn, error};
use reqwest::{Client, Method, header::{HeaderMap, HeaderValue}};
use sha2::{Sha256, Digest};
use hmac::{Hmac, Mac};
use chrono::Utc;

use super::{SignalEmissionError, Result};

type HmacSha256 = Hmac<Sha256>;

/// Configuration for S3/MinIO upload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Config {
    /// S3 endpoint URL (e.g., "https://s3.amazonaws.com" or "http://localhost:9000")
    pub endpoint: String,
    
    /// S3 region (e.g., "us-east-1")
    pub region: String,
    
    /// S3 bucket name for audit logs
    pub bucket: String,
    
    /// Key prefix for audit log objects (e.g., "audit-logs/signal-emission/")
    pub key_prefix: String,
    
    /// Access key ID
    pub access_key_id: String,
    
    /// Secret access key
    pub secret_access_key: String,
    
    /// Session token (optional, for temporary credentials)
    pub session_token: Option<String>,
    
    /// Upload timeout in seconds (default: 300)
    pub upload_timeout_seconds: u64,
    
    /// Maximum retry attempts (default: 3)
    pub max_retry_attempts: u32,
    
    /// Base retry delay in milliseconds (default: 1000)
    pub retry_delay_ms: u64,
    
    /// Retry delay multiplier for exponential backoff (default: 2.0)
    pub retry_multiplier: f64,
    
    /// Upload interval in seconds (default: 300 = 5 minutes)
    pub upload_interval_seconds: u64,
    
    /// Whether to use path-style URLs (required for MinIO, default: false)
    pub path_style: bool,
    
    /// Whether to use HTTPS (default: true)
    pub use_https: bool,
    
    /// Custom metadata to add to uploaded objects
    pub metadata: HashMap<String, String>,
    
    /// Storage class for uploaded objects (e.g., "STANDARD", "GLACIER")
    pub storage_class: Option<String>,
    
    /// Server-side encryption configuration
    pub server_side_encryption: Option<String>,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            endpoint: "https://s3.amazonaws.com".to_string(),
            region: "us-east-1".to_string(),
            bucket: "audit-logs".to_string(),
            key_prefix: "signal-emission/".to_string(),
            access_key_id: String::new(),
            secret_access_key: String::new(),
            session_token: None,
            upload_timeout_seconds: 300,
            max_retry_attempts: 3,
            retry_delay_ms: 1000,
            retry_multiplier: 2.0,
            upload_interval_seconds: 300,
            path_style: false,
            use_https: true,
            metadata: HashMap::new(),
            storage_class: None,
            server_side_encryption: None,
        }
    }
}

/// Upload result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadResult {
    /// Local file path that was uploaded
    pub local_path: PathBuf,
    
    /// S3 object key
    pub s3_key: String,
    
    /// Upload timestamp
    pub uploaded_at: u64,
    
    /// File size in bytes
    pub file_size_bytes: u64,
    
    /// Upload duration in milliseconds
    pub upload_duration_ms: u64,
    
    /// Number of retry attempts made
    pub retry_attempts: u32,
    
    /// Whether upload was successful
    pub success: bool,
    
    /// Error message if upload failed
    pub error_message: Option<String>,
    
    /// ETag returned by S3
    pub etag: Option<String>,
}

/// Statistics for S3 upload operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3UploaderStats {
    /// Total number of files uploaded successfully
    pub total_files_uploaded: u64,
    
    /// Total number of upload failures
    pub total_upload_failures: u64,
    
    /// Total bytes uploaded
    pub total_bytes_uploaded: u64,
    
    /// Total upload time in milliseconds
    pub total_upload_time_ms: u64,
    
    /// Average upload speed in bytes per second
    pub average_upload_speed_bps: f64,
    
    /// Last successful upload timestamp
    pub last_upload_at: Option<u64>,
    
    /// Last upload error timestamp
    pub last_error_at: Option<u64>,
    
    /// Current retry queue size
    pub retry_queue_size: u64,
}

impl Default for S3UploaderStats {
    fn default() -> Self {
        Self {
            total_files_uploaded: 0,
            total_upload_failures: 0,
            total_bytes_uploaded: 0,
            total_upload_time_ms: 0,
            average_upload_speed_bps: 0.0,
            last_upload_at: None,
            last_error_at: None,
            retry_queue_size: 0,
        }
    }
}

/// S3/MinIO uploader for audit log files
pub struct S3Uploader {
    config: S3Config,
    client: Client,
    stats: tokio::sync::RwLock<S3UploaderStats>,
    retry_queue: tokio::sync::RwLock<Vec<PathBuf>>,
}

impl S3Uploader {
    /// Create a new S3 uploader with the given configuration
    pub fn new(config: S3Config) -> Result<Self> {
        // Validate configuration
        if config.access_key_id.is_empty() {
            return Err(SignalEmissionError::config("S3 access key ID cannot be empty"));
        }
        
        if config.secret_access_key.is_empty() {
            return Err(SignalEmissionError::config("S3 secret access key cannot be empty"));
        }
        
        if config.bucket.is_empty() {
            return Err(SignalEmissionError::config("S3 bucket name cannot be empty"));
        }
        
        let client = Client::builder()
            .timeout(Duration::from_secs(config.upload_timeout_seconds))
            .build()
            .map_err(|e| SignalEmissionError::config(format!("Failed to create HTTP client: {}", e)))?;
        
        Ok(Self {
            config,
            client,
            stats: tokio::sync::RwLock::new(S3UploaderStats::default()),
            retry_queue: tokio::sync::RwLock::new(Vec::new()),
        })
    }
    
    /// Upload a single file to S3
    pub async fn upload_file(&self, file_path: &Path) -> Result<UploadResult> {
        let start_time = SystemTime::now();
        let mut retry_attempts = 0;
        
        loop {
            match self.try_upload_file(file_path, retry_attempts).await {
                Ok(mut result) => {
                    result.retry_attempts = retry_attempts;
                    
                    // Update stats
                    {
                        let mut stats = self.stats.write().await;
                        stats.total_files_uploaded += 1;
                        stats.total_bytes_uploaded += result.file_size_bytes;
                        stats.total_upload_time_ms += result.upload_duration_ms;
                        stats.last_upload_at = Some(result.uploaded_at);
                        
                        if stats.total_upload_time_ms > 0 {
                            stats.average_upload_speed_bps = 
                                (stats.total_bytes_uploaded as f64 * 1000.0) / stats.total_upload_time_ms as f64;
                        }
                    }
                    
                    info!(
                        file_path = %file_path.display(),
                        s3_key = %result.s3_key,
                        file_size = result.file_size_bytes,
                        duration_ms = result.upload_duration_ms,
                        retry_attempts = retry_attempts,
                        "File uploaded to S3 successfully"
                    );
                    
                    return Ok(result);
                }
                Err(e) => {
                    retry_attempts += 1;
                    
                    if retry_attempts >= self.config.max_retry_attempts {
                        let duration_ms = start_time.elapsed().unwrap_or_default().as_millis() as u64;
                        
                        // Update failure stats
                        {
                            let mut stats = self.stats.write().await;
                            stats.total_upload_failures += 1;
                            stats.last_error_at = Some(current_timestamp());
                        }
                        
                        error!(
                            file_path = %file_path.display(),
                            retry_attempts = retry_attempts,
                            error = %e,
                            "Failed to upload file to S3 after all retries"
                        );
                        
                        return Ok(UploadResult {
                            local_path: file_path.to_path_buf(),
                            s3_key: self.generate_s3_key(file_path),
                            uploaded_at: current_timestamp(),
                            file_size_bytes: 0,
                            upload_duration_ms: duration_ms,
                            retry_attempts,
                            success: false,
                            error_message: Some(e.to_string()),
                            etag: None,
                        });
                    }
                    
                    // Calculate retry delay with exponential backoff
                    let delay_ms = (self.config.retry_delay_ms as f64 * 
                                   self.config.retry_multiplier.powi(retry_attempts as i32 - 1)) as u64;
                    
                    warn!(
                        file_path = %file_path.display(),
                        retry_attempts = retry_attempts,
                        delay_ms = delay_ms,
                        error = %e,
                        "Upload failed, retrying after delay"
                    );
                    
                    sleep(Duration::from_millis(delay_ms)).await;
                }
            }
        }
    }
    
    /// Try to upload a file once (internal method)
    async fn try_upload_file(&self, file_path: &Path, attempt: u32) -> Result<UploadResult> {
        let start_time = SystemTime::now();
        
        // Read file content
        let file_content = fs::read(file_path).await.map_err(|e| {
            SignalEmissionError::audit(format!("Failed to read file {}: {}", file_path.display(), e))
        })?;
        
        let file_size = file_content.len() as u64;
        let s3_key = self.generate_s3_key(file_path);
        
        debug!(
            file_path = %file_path.display(),
            s3_key = %s3_key,
            file_size = file_size,
            attempt = attempt,
            "Starting S3 upload attempt"
        );
        
        // Generate S3 URL
        let url = self.generate_s3_url(&s3_key);
        
        // Create headers
        let mut headers = HeaderMap::new();
        headers.insert("Content-Length", HeaderValue::from_str(&file_size.to_string()).unwrap());
        headers.insert("Content-Type", HeaderValue::from_static("application/json"));
        
        // Add custom metadata
        for (key, value) in &self.config.metadata {
            let header_name = format!("x-amz-meta-{}", key);
            if let Ok(header_value) = HeaderValue::from_str(value) {
                if let Ok(header_name) = reqwest::header::HeaderName::from_bytes(header_name.as_bytes()) {
                    headers.insert(header_name, header_value);
                }
            }
        }
        
        // Add storage class if specified
        if let Some(ref storage_class) = self.config.storage_class {
            headers.insert("x-amz-storage-class", HeaderValue::from_str(storage_class).unwrap());
        }
        
        // Add server-side encryption if specified
        if let Some(ref sse) = self.config.server_side_encryption {
            headers.insert("x-amz-server-side-encryption", HeaderValue::from_str(sse).unwrap());
        }
        
        // Sign the request
        let signed_headers = self.sign_request(&Method::PUT, &s3_key, &headers, &file_content).await?;
        
        // Make the request
        let response = timeout(
            Duration::from_secs(self.config.upload_timeout_seconds),
            self.client
                .put(&url)
                .headers(signed_headers)
                .body(file_content)
                .send()
        ).await
        .map_err(|_| SignalEmissionError::timeout(self.config.upload_timeout_seconds * 1000))?
        .map_err(|e| SignalEmissionError::audit(format!("S3 upload request failed: {}", e)))?;
        
        let status = response.status();
        let etag = response.headers()
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_matches('"').to_string());
        
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(SignalEmissionError::audit(format!(
                "S3 upload failed with status {}: {}",
                status, error_body
            )));
        }
        
        let duration_ms = start_time.elapsed().unwrap_or_default().as_millis() as u64;
        
        Ok(UploadResult {
            local_path: file_path.to_path_buf(),
            s3_key,
            uploaded_at: current_timestamp(),
            file_size_bytes: file_size,
            upload_duration_ms: duration_ms,
            retry_attempts: 0, // Will be set by caller
            success: true,
            error_message: None,
            etag,
        })
    }
    
    /// Upload multiple files in batch
    pub async fn upload_batch(&self, file_paths: &[PathBuf]) -> Vec<UploadResult> {
        let mut results = Vec::new();
        
        for file_path in file_paths {
            match self.upload_file(file_path).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!(
                        file_path = %file_path.display(),
                        error = %e,
                        "Failed to upload file in batch"
                    );
                    
                    results.push(UploadResult {
                        local_path: file_path.clone(),
                        s3_key: self.generate_s3_key(file_path),
                        uploaded_at: current_timestamp(),
                        file_size_bytes: 0,
                        upload_duration_ms: 0,
                        retry_attempts: 0,
                        success: false,
                        error_message: Some(e.to_string()),
                        etag: None,
                    });
                }
            }
        }
        
        results
    }
    
    /// Add files to retry queue for later upload
    pub async fn add_to_retry_queue(&self, file_paths: Vec<PathBuf>) {
        let mut queue = self.retry_queue.write().await;
        queue.extend(file_paths);
        
        let mut stats = self.stats.write().await;
        stats.retry_queue_size = queue.len() as u64;
    }
    
    /// Process retry queue
    pub async fn process_retry_queue(&self) -> Vec<UploadResult> {
        let files_to_retry = {
            let mut queue = self.retry_queue.write().await;
            let files = queue.clone();
            queue.clear();
            files
        };
        
        if files_to_retry.is_empty() {
            return Vec::new();
        }
        
        info!("Processing retry queue with {} files", files_to_retry.len());
        
        let results = self.upload_batch(&files_to_retry).await;
        
        // Re-queue failed uploads
        let failed_files: Vec<PathBuf> = results
            .iter()
            .filter(|r| !r.success)
            .map(|r| r.local_path.clone())
            .collect();
        
        if !failed_files.is_empty() {
            self.add_to_retry_queue(failed_files).await;
        }
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            let queue = self.retry_queue.read().await;
            stats.retry_queue_size = queue.len() as u64;
        }
        
        results
    }
    
    /// Generate S3 object key from file path
    fn generate_s3_key(&self, file_path: &Path) -> String {
        let filename = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        
        let timestamp = chrono::Utc::now().format("%Y/%m/%d");
        format!("{}{}/{}", self.config.key_prefix, timestamp, filename)
    }
    
    /// Generate S3 URL for the given key
    fn generate_s3_url(&self, key: &str) -> String {
        if self.config.path_style {
            format!("{}/{}/{}", self.config.endpoint, self.config.bucket, key)
        } else {
            format!("https://{}.{}/{}", self.config.bucket, 
                   self.config.endpoint.trim_start_matches("https://"), key)
        }
    }
    
    /// Sign S3 request using AWS Signature Version 4
    async fn sign_request(
        &self,
        method: &Method,
        key: &str,
        headers: &HeaderMap,
        payload: &[u8],
    ) -> Result<HeaderMap> {
        let now = Utc::now();
        let date_stamp = now.format("%Y%m%d").to_string();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        
        let mut signed_headers = headers.clone();
        signed_headers.insert("host", HeaderValue::from_str(&self.get_host()).unwrap());
        signed_headers.insert("x-amz-date", HeaderValue::from_str(&amz_date).unwrap());
        
        if let Some(ref token) = self.config.session_token {
            signed_headers.insert("x-amz-security-token", HeaderValue::from_str(token).unwrap());
        }
        
        // Create canonical request
        let canonical_uri = format!("/{}", key);
        let canonical_querystring = "";
        
        let mut canonical_headers = String::new();
        let mut signed_headers_list = Vec::new();
        
        let mut header_names: Vec<_> = signed_headers.keys().map(|k| k.as_str()).collect();
        header_names.sort();
        
        for name in &header_names {
            let value = signed_headers.get(*name).unwrap().to_str().unwrap();
            canonical_headers.push_str(&format!("{}:{}\n", name.to_lowercase(), value.trim()));
            signed_headers_list.push(name.to_lowercase());
        }
        
        let signed_headers_str = signed_headers_list.join(";");
        
        // Hash payload
        let payload_hash = format!("{:x}", Sha256::digest(payload));
        
        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method.as_str(),
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers_str,
            payload_hash
        );
        
        // Create string to sign
        let algorithm = "AWS4-HMAC-SHA256";
        let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, self.config.region);
        let string_to_sign = format!(
            "{}\n{}\n{}\n{:x}",
            algorithm,
            amz_date,
            credential_scope,
            Sha256::digest(canonical_request.as_bytes())
        );
        
        // Calculate signature
        let signing_key = self.get_signature_key(&date_stamp)?;
        let mut mac = HmacSha256::new_from_slice(&signing_key)
            .map_err(|e| SignalEmissionError::auth(format!("Failed to create HMAC: {}", e)))?;
        mac.update(string_to_sign.as_bytes());
        let signature = format!("{:x}", mac.finalize().into_bytes());
        
        // Create authorization header
        let authorization = format!(
            "{} Credential={}/{}, SignedHeaders={}, Signature={}",
            algorithm,
            self.config.access_key_id,
            credential_scope,
            signed_headers_str,
            signature
        );
        
        signed_headers.insert("authorization", HeaderValue::from_str(&authorization).unwrap());
        
        Ok(signed_headers)
    }
    
    /// Get signing key for AWS Signature Version 4
    fn get_signature_key(&self, date_stamp: &str) -> Result<Vec<u8>> {
        let k_date = self.hmac_sha256(
            format!("AWS4{}", self.config.secret_access_key).as_bytes(),
            date_stamp.as_bytes(),
        )?;
        let k_region = self.hmac_sha256(&k_date, self.config.region.as_bytes())?;
        let k_service = self.hmac_sha256(&k_region, b"s3")?;
        let k_signing = self.hmac_sha256(&k_service, b"aws4_request")?;
        Ok(k_signing)
    }
    
    /// HMAC-SHA256 helper
    fn hmac_sha256(&self, key: &[u8], data: &[u8]) -> Result<Vec<u8>> {
        let mut mac = HmacSha256::new_from_slice(key)
            .map_err(|e| SignalEmissionError::auth(format!("Failed to create HMAC: {}", e)))?;
        mac.update(data);
        Ok(mac.finalize().into_bytes().to_vec())
    }
    
    /// Get host for S3 endpoint
    fn get_host(&self) -> String {
        let endpoint = self.config.endpoint.trim_start_matches("https://").trim_start_matches("http://");
        if self.config.path_style {
            endpoint.to_string()
        } else {
            format!("{}.{}", self.config.bucket, endpoint)
        }
    }
    
    /// Get current uploader statistics
    pub async fn get_stats(&self) -> S3UploaderStats {
        self.stats.read().await.clone()
    }
    
    /// Test S3 connectivity
    pub async fn test_connection(&self) -> Result<()> {
        // Try to list objects in the bucket (HEAD request)
        let url = if self.config.path_style {
            format!("{}/{}", self.config.endpoint, self.config.bucket)
        } else {
            format!("https://{}.{}", self.config.bucket, 
                   self.config.endpoint.trim_start_matches("https://"))
        };
        
        let headers = HeaderMap::new();
        let signed_headers = self.sign_request(&Method::HEAD, "", &headers, &[]).await?;
        
        let response = timeout(
            Duration::from_secs(30),
            self.client.head(&url).headers(signed_headers).send()
        ).await
        .map_err(|_| SignalEmissionError::timeout(30000))?
        .map_err(|e| SignalEmissionError::auth(format!("S3 connection test failed: {}", e)))?;
        
        if response.status().is_success() || response.status().as_u16() == 404 {
            // 404 is OK - bucket might not exist but credentials work
            Ok(())
        } else {
            Err(SignalEmissionError::auth(format!(
                "S3 connection test failed with status: {}",
                response.status()
            )))
        }
    }
}

/// Get current timestamp in seconds since Unix epoch
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    
    fn create_test_config() -> S3Config {
        S3Config {
            endpoint: "http://localhost:9000".to_string(),
            region: "us-east-1".to_string(),
            bucket: "test-bucket".to_string(),
            key_prefix: "test/".to_string(),
            access_key_id: "test-access-key".to_string(),
            secret_access_key: "test-secret-key".to_string(),
            session_token: None,
            upload_timeout_seconds: 30,
            max_retry_attempts: 2,
            retry_delay_ms: 100,
            retry_multiplier: 2.0,
            upload_interval_seconds: 60,
            path_style: true,
            use_https: false,
            metadata: HashMap::new(),
            storage_class: None,
            server_side_encryption: None,
        }
    }
    
    #[test]
    fn test_s3_uploader_creation() {
        let config = create_test_config();
        let uploader = S3Uploader::new(config).unwrap();
        assert_eq!(uploader.config.bucket, "test-bucket");
    }
    
    #[test]
    fn test_s3_config_validation() {
        let mut config = create_test_config();
        config.access_key_id = String::new();
        
        let result = S3Uploader::new(config);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_s3_key_generation() {
        let config = create_test_config();
        let uploader = S3Uploader::new(config).unwrap();
        
        let path = PathBuf::from("/tmp/test_file.jsonl");
        let key = uploader.generate_s3_key(&path);
        
        assert!(key.starts_with("test/"));
        assert!(key.ends_with("test_file.jsonl"));
    }
    
    #[test]
    fn test_s3_url_generation() {
        let config = create_test_config();
        let uploader = S3Uploader::new(config).unwrap();
        
        let key = "test/file.jsonl";
        let url = uploader.generate_s3_url(key);
        
        assert_eq!(url, "http://localhost:9000/test-bucket/test/file.jsonl");
    }
    
    #[tokio::test]
    async fn test_retry_queue_operations() {
        let config = create_test_config();
        let uploader = S3Uploader::new(config).unwrap();
        
        let files = vec![
            PathBuf::from("/tmp/file1.jsonl"),
            PathBuf::from("/tmp/file2.jsonl"),
        ];
        
        uploader.add_to_retry_queue(files.clone()).await;
        
        let stats = uploader.get_stats().await;
        assert_eq!(stats.retry_queue_size, 2);
    }
    
    #[tokio::test]
    async fn test_stats_initialization() {
        let config = create_test_config();
        let uploader = S3Uploader::new(config).unwrap();
        
        let stats = uploader.get_stats().await;
        assert_eq!(stats.total_files_uploaded, 0);
        assert_eq!(stats.total_upload_failures, 0);
        assert_eq!(stats.retry_queue_size, 0);
    }
    
    #[test]
    fn test_host_generation() {
        let config = create_test_config();
        let uploader = S3Uploader::new(config).unwrap();
        
        let host = uploader.get_host();
        assert_eq!(host, "localhost:9000");
    }
    
    #[tokio::test]
    async fn test_signing_key_generation() {
        let config = create_test_config();
        let uploader = S3Uploader::new(config).unwrap();
        
        let date_stamp = "20231201";
        let signing_key = uploader.get_signature_key(date_stamp).unwrap();
        
        assert!(!signing_key.is_empty());
        assert_eq!(signing_key.len(), 32); // SHA256 output length
    }
    
    // Note: Integration tests with actual S3/MinIO would require running services
    // These tests focus on the logic and configuration aspects
}