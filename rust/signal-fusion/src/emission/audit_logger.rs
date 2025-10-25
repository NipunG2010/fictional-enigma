//! File-based audit logging with rotation and integrity verification
//! 
//! This module provides structured audit logging to files with configurable rotation
//! policies, integrity verification using checksums, and support for both local
//! file storage and S3/MinIO archival.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write, Read, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};
use sha2::{Sha256, Digest};

use super::{SignalEmissionError, Result};
use super::audit::{AuditEvent, SignalEmissionEvent, FeatureComputationEvent, ValidationErrorEvent, PublisherEvent, HmmWeightEvent};
use super::s3_uploader::{S3Uploader, S3Config};

/// Configuration for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Base directory for audit log files
    pub log_directory: PathBuf,
    
    /// Base filename for audit logs (without extension)
    pub log_filename: String,
    
    /// Maximum size of a single log file in bytes (default: 100MB)
    pub max_file_size_bytes: u64,
    
    /// Maximum age of a log file in seconds before rotation (default: 24 hours)
    pub max_file_age_seconds: u64,
    
    /// Maximum number of rotated log files to keep (default: 30)
    pub max_files_to_keep: u32,
    
    /// Whether to compress rotated log files (default: true)
    pub compress_rotated_files: bool,
    
    /// Whether to calculate and store file checksums (default: true)
    pub enable_integrity_verification: bool,
    
    /// Buffer size for file writes (default: 64KB)
    pub write_buffer_size: usize,
    
    /// Whether to flush after each write (default: false for performance)
    pub flush_after_write: bool,
    
    /// File permissions for log files (octal, default: 0o644)
    pub file_permissions: u32,
    
    /// Whether to create parent directories if they don't exist (default: true)
    pub create_directories: bool,
    
    /// S3/MinIO configuration for log archival (optional)
    pub s3_config: Option<S3Config>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_directory: PathBuf::from("/var/log/imp/audit"),
            log_filename: "signal_emission".to_string(),
            max_file_size_bytes: 100 * 1024 * 1024, // 100MB
            max_file_age_seconds: 24 * 60 * 60, // 24 hours
            max_files_to_keep: 30,
            compress_rotated_files: true,
            enable_integrity_verification: true,
            write_buffer_size: 64 * 1024, // 64KB
            flush_after_write: false,
            file_permissions: 0o644,
            create_directories: true,
            s3_config: None,
        }
    }
}

/// Metadata about a log file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogFileMetadata {
    /// Path to the log file
    pub file_path: PathBuf,
    
    /// Creation timestamp
    pub created_at: u64,
    
    /// Last modified timestamp
    pub modified_at: u64,
    
    /// File size in bytes
    pub size_bytes: u64,
    
    /// Number of events in the file
    pub event_count: u64,
    
    /// SHA256 checksum of the file content
    pub checksum: Option<String>,
    
    /// Whether the file is compressed
    pub compressed: bool,
    
    /// Whether the file is archived to S3/MinIO
    pub archived: bool,
}

/// Statistics about audit logging operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLoggerStats {
    /// Total number of events logged
    pub total_events_logged: u64,
    
    /// Total number of files created
    pub total_files_created: u64,
    
    /// Total number of files rotated
    pub total_files_rotated: u64,
    
    /// Total number of write errors
    pub total_write_errors: u64,
    
    /// Total number of rotation errors
    pub total_rotation_errors: u64,
    
    /// Current active log file size
    pub current_file_size_bytes: u64,
    
    /// Current active log file event count
    pub current_file_event_count: u64,
    
    /// Last rotation timestamp
    pub last_rotation_at: Option<u64>,
    
    /// Last write timestamp
    pub last_write_at: Option<u64>,
}

impl Default for AuditLoggerStats {
    fn default() -> Self {
        Self {
            total_events_logged: 0,
            total_files_created: 0,
            total_files_rotated: 0,
            total_write_errors: 0,
            total_rotation_errors: 0,
            current_file_size_bytes: 0,
            current_file_event_count: 0,
            last_rotation_at: None,
            last_write_at: None,
        }
    }
}

/// File-based audit logger with rotation and integrity verification
pub struct AuditLogger {
    config: AuditConfig,
    current_writer: Arc<Mutex<Option<BufWriter<File>>>>,
    current_file_path: Arc<RwLock<Option<PathBuf>>>,
    current_file_metadata: Arc<RwLock<LogFileMetadata>>,
    stats: Arc<RwLock<AuditLoggerStats>>,
    file_metadata_cache: Arc<RwLock<HashMap<PathBuf, LogFileMetadata>>>,
    s3_uploader: Option<S3Uploader>,
}

impl AuditLogger {
    /// Create a new audit logger with the given configuration
    pub async fn new(config: AuditConfig) -> Result<Self> {
        // Create log directory if it doesn't exist
        if config.create_directories {
            std::fs::create_dir_all(&config.log_directory).map_err(|e| {
                SignalEmissionError::config(format!(
                    "Failed to create log directory {:?}: {}",
                    config.log_directory, e
                ))
            })?;
        }
        
        // Initialize S3 uploader if configured
        let s3_uploader = if let Some(ref s3_config) = config.s3_config {
            match S3Uploader::new(s3_config.clone()) {
                Ok(uploader) => {
                    info!("S3 uploader initialized for audit log archival");
                    Some(uploader)
                }
                Err(e) => {
                    warn!("Failed to initialize S3 uploader: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let logger = Self {
            config,
            current_writer: Arc::new(Mutex::new(None)),
            current_file_path: Arc::new(RwLock::new(None)),
            current_file_metadata: Arc::new(RwLock::new(LogFileMetadata {
                file_path: PathBuf::new(),
                created_at: 0,
                modified_at: 0,
                size_bytes: 0,
                event_count: 0,
                checksum: None,
                compressed: false,
                archived: false,
            })),
            stats: Arc::new(RwLock::new(AuditLoggerStats::default())),
            file_metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            s3_uploader,
        };
        
        // Initialize the first log file
        logger.rotate_if_needed().await?;
        
        info!(
            "Audit logger initialized with directory: {:?}, max_file_size: {} bytes",
            logger.config.log_directory,
            logger.config.max_file_size_bytes
        );
        
        Ok(logger)
    }
    
    /// Log a signal emission event
    pub async fn log_signal_emission(&self, event: &SignalEmissionEvent) -> Result<()> {
        self.log_event(event).await
    }
    
    /// Log a feature computation event
    pub async fn log_feature_computation(&self, event: &FeatureComputationEvent) -> Result<()> {
        self.log_event(event).await
    }
    
    /// Log a validation error event
    pub async fn log_validation_error(&self, event: &ValidationErrorEvent) -> Result<()> {
        self.log_event(event).await
    }
    
    /// Log a publisher event
    pub async fn log_publisher_event(&self, event: &PublisherEvent) -> Result<()> {
        self.log_event(event).await
    }
    
    /// Log an HMM weight event
    pub async fn log_hmm_weight_event(&self, event: &HmmWeightEvent) -> Result<()> {
        self.log_event(event).await
    }
    
    /// Log any audit event
    async fn log_event<T>(&self, event: &T) -> Result<()>
    where
        T: AuditEvent + Serialize,
    {
        // Validate the event first
        event.validate()?;
        
        // Check if rotation is needed
        self.rotate_if_needed().await?;
        
        // Serialize the event to JSON
        let json_line = event.to_json()?;
        let log_line = format!("{}\n", json_line);
        
        // Write to the current log file
        {
            let mut writer_guard = self.current_writer.lock().map_err(|e| {
                SignalEmissionError::audit(format!("Failed to acquire writer lock: {}", e))
            })?;
            
            if let Some(ref mut writer) = *writer_guard {
                writer.write_all(log_line.as_bytes()).map_err(|e| {
                    SignalEmissionError::audit(format!("Failed to write audit event: {}", e))
                })?;
                
                if self.config.flush_after_write {
                    writer.flush().map_err(|e| {
                        SignalEmissionError::audit(format!("Failed to flush audit log: {}", e))
                    })?;
                }
            } else {
                return Err(SignalEmissionError::audit("No active log file writer"));
            }
        }
        
        // Update metadata and stats
        let log_line_size = log_line.len() as u64;
        {
            let mut metadata = self.current_file_metadata.write().await;
            metadata.size_bytes += log_line_size;
            metadata.event_count += 1;
            metadata.modified_at = current_timestamp();
        }
        
        {
            let mut stats = self.stats.write().await;
            stats.total_events_logged += 1;
            stats.current_file_size_bytes += log_line_size;
            stats.current_file_event_count += 1;
            stats.last_write_at = Some(current_timestamp());
        }
        
        debug!(
            event_type = event.event_type(),
            event_id = event.event_id(),
            correlation_id = event.correlation_id(),
            "Audit event logged successfully"
        );
        
        Ok(())
    }
    
    /// Check if log rotation is needed and perform it if necessary
    async fn rotate_if_needed(&self) -> Result<()> {
        let should_rotate = {
            let metadata = self.current_file_metadata.read().await;
            let current_time = current_timestamp();
            
            // Check if we need to rotate based on size or age
            metadata.size_bytes >= self.config.max_file_size_bytes ||
            (metadata.created_at > 0 && 
             current_time - metadata.created_at >= self.config.max_file_age_seconds) ||
            metadata.created_at == 0 // First file
        };
        
        if should_rotate {
            self.rotate_log_file().await?;
        }
        
        Ok(())
    }
    
    /// Rotate the current log file
    async fn rotate_log_file(&self) -> Result<()> {
        info!("Starting log file rotation");
        
        // Close current writer
        {
            let mut writer_guard = self.current_writer.lock().map_err(|e| {
                SignalEmissionError::audit(format!("Failed to acquire writer lock for rotation: {}", e))
            })?;
            
            if let Some(writer) = writer_guard.take() {
                drop(writer); // This will flush and close the file
            }
        }
        
        // Calculate checksum and upload to S3 if enabled
        let current_path = self.current_file_path.read().await.clone();
        if let Some(ref path) = current_path {
            if self.config.enable_integrity_verification && path.exists() {
                match self.calculate_file_checksum(path).await {
                    Ok(checksum) => {
                        let mut metadata = self.current_file_metadata.write().await;
                        metadata.checksum = Some(checksum);
                        
                        // Cache the metadata
                        let mut cache = self.file_metadata_cache.write().await;
                        cache.insert(path.clone(), metadata.clone());
                    }
                    Err(e) => {
                        warn!("Failed to calculate checksum for {}: {}", path.display(), e);
                    }
                }
            }
            
            // Upload to S3 if configured
            if let Some(ref uploader) = self.s3_uploader {
                if path.exists() {
                    match uploader.upload_file(path).await {
                        Ok(result) => {
                            if result.success {
                                info!(
                                    "Audit log uploaded to S3: {} -> {}",
                                    path.display(),
                                    result.s3_key
                                );
                                
                                // Mark as archived in metadata
                                let mut cache = self.file_metadata_cache.write().await;
                                if let Some(metadata) = cache.get_mut(path) {
                                    metadata.archived = true;
                                }
                            } else {
                                warn!(
                                    "Failed to upload audit log to S3: {} - {}",
                                    path.display(),
                                    result.error_message.unwrap_or_default()
                                );
                            }
                        }
                        Err(e) => {
                            warn!("S3 upload error for {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
        
        // Create new log file
        let new_file_path = self.generate_log_file_path();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&new_file_path)
            .map_err(|e| {
                SignalEmissionError::audit(format!(
                    "Failed to create new log file {:?}: {}",
                    new_file_path, e
                ))
            })?;
        
        // Set file permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = file.metadata().map_err(|e| {
                SignalEmissionError::audit(format!("Failed to get file metadata: {}", e))
            })?.permissions();
            perms.set_mode(self.config.file_permissions);
            std::fs::set_permissions(&new_file_path, perms).map_err(|e| {
                SignalEmissionError::audit(format!("Failed to set file permissions: {}", e))
            })?;
        }
        
        let writer = BufWriter::with_capacity(self.config.write_buffer_size, file);
        
        // Update current writer and metadata
        {
            let mut writer_guard = self.current_writer.lock().map_err(|e| {
                SignalEmissionError::audit(format!("Failed to acquire writer lock: {}", e))
            })?;
            *writer_guard = Some(writer);
        }
        
        {
            let mut path_guard = self.current_file_path.write().await;
            *path_guard = Some(new_file_path.clone());
        }
        
        let current_time = current_timestamp();
        {
            let mut metadata = self.current_file_metadata.write().await;
            *metadata = LogFileMetadata {
                file_path: new_file_path.clone(),
                created_at: current_time,
                modified_at: current_time,
                size_bytes: 0,
                event_count: 0,
                checksum: None,
                compressed: false,
                archived: false,
            };
        }
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_files_created += 1;
            stats.total_files_rotated += 1;
            stats.current_file_size_bytes = 0;
            stats.current_file_event_count = 0;
            stats.last_rotation_at = Some(current_time);
        }
        
        // Clean up old files
        if let Err(e) = self.cleanup_old_files().await {
            warn!("Failed to cleanup old log files: {}", e);
        }
        
        info!(
            "Log file rotation completed, new file: {}",
            new_file_path.display()
        );
        
        Ok(())
    }
    
    /// Generate a new log file path with timestamp
    fn generate_log_file_path(&self) -> PathBuf {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("{}_{}.jsonl", self.config.log_filename, timestamp);
        self.config.log_directory.join(filename)
    }
    
    /// Calculate SHA256 checksum of a file
    async fn calculate_file_checksum(&self, file_path: &Path) -> Result<String> {
        let file = File::open(file_path).map_err(|e| {
            SignalEmissionError::audit(format!(
                "Failed to open file for checksum calculation: {}",
                e
            ))
        })?;
        
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0; 8192];
        
        loop {
            let bytes_read = reader.read(&mut buffer).map_err(|e| {
                SignalEmissionError::audit(format!("Failed to read file for checksum: {}", e))
            })?;
            
            if bytes_read == 0 {
                break;
            }
            
            hasher.update(&buffer[..bytes_read]);
        }
        
        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }
    
    /// Clean up old log files based on retention policy
    async fn cleanup_old_files(&self) -> Result<()> {
        let log_dir = &self.config.log_directory;
        
        // Get all log files in the directory
        let entries = std::fs::read_dir(log_dir).map_err(|e| {
            SignalEmissionError::audit(format!("Failed to read log directory: {}", e))
        })?;
        
        let mut log_files: Vec<(PathBuf, SystemTime)> = Vec::new();
        
        for entry in entries {
            let entry = entry.map_err(|e| {
                SignalEmissionError::audit(format!("Failed to read directory entry: {}", e))
            })?;
            
            let path = entry.path();
            if path.is_file() && 
               path.extension().map_or(false, |ext| ext == "jsonl") &&
               path.file_name()
                   .and_then(|name| name.to_str())
                   .map_or(false, |name| name.starts_with(&self.config.log_filename)) {
                
                let metadata = entry.metadata().map_err(|e| {
                    SignalEmissionError::audit(format!("Failed to get file metadata: {}", e))
                })?;
                
                if let Ok(created) = metadata.created() {
                    log_files.push((path, created));
                }
            }
        }
        
        // Sort by creation time (oldest first)
        log_files.sort_by_key(|(_, created)| *created);
        
        // Remove files beyond the retention limit
        if log_files.len() > self.config.max_files_to_keep as usize {
            let files_to_remove = log_files.len() - self.config.max_files_to_keep as usize;
            
            for (path, _) in log_files.iter().take(files_to_remove) {
                match std::fs::remove_file(path) {
                    Ok(()) => {
                        info!("Removed old log file: {}", path.display());
                    }
                    Err(e) => {
                        warn!("Failed to remove old log file {}: {}", path.display(), e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Get current audit logger statistics
    pub async fn get_stats(&self) -> AuditLoggerStats {
        self.stats.read().await.clone()
    }
    
    /// Get metadata for all log files
    pub async fn get_file_metadata(&self) -> HashMap<PathBuf, LogFileMetadata> {
        self.file_metadata_cache.read().await.clone()
    }
    
    /// Verify the integrity of a log file using its checksum
    pub async fn verify_file_integrity(&self, file_path: &Path) -> Result<bool> {
        let cache = self.file_metadata_cache.read().await;
        
        if let Some(metadata) = cache.get(file_path) {
            if let Some(ref stored_checksum) = metadata.checksum {
                let calculated_checksum = self.calculate_file_checksum(file_path).await?;
                Ok(*stored_checksum == calculated_checksum)
            } else {
                Err(SignalEmissionError::audit(
                    "No checksum available for file integrity verification"
                ))
            }
        } else {
            Err(SignalEmissionError::audit(
                "File metadata not found in cache"
            ))
        }
    }
    
    /// Force a log file rotation
    pub async fn force_rotation(&self) -> Result<()> {
        info!("Forcing log file rotation");
        self.rotate_log_file().await
    }
    
    /// Flush the current log file
    pub async fn flush(&self) -> Result<()> {
        let mut writer_guard = self.current_writer.lock().map_err(|e| {
            SignalEmissionError::audit(format!("Failed to acquire writer lock for flush: {}", e))
        })?;
        
        if let Some(ref mut writer) = *writer_guard {
            writer.flush().map_err(|e| {
                SignalEmissionError::audit(format!("Failed to flush audit log: {}", e))
            })?;
        }
        
        Ok(())
    }
    
    /// Shutdown the audit logger gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down audit logger");
        
        // Flush and close current writer
        {
            let mut writer_guard = self.current_writer.lock().map_err(|e| {
                SignalEmissionError::audit(format!("Failed to acquire writer lock for shutdown: {}", e))
            })?;
            
            if let Some(writer) = writer_guard.take() {
                drop(writer); // This will flush and close the file
            }
        }
        
        // Calculate final checksum if enabled
        let current_path = self.current_file_path.read().await.clone();
        if let Some(ref path) = current_path {
            if self.config.enable_integrity_verification && path.exists() {
                match self.calculate_file_checksum(path).await {
                    Ok(checksum) => {
                        let mut metadata = self.current_file_metadata.write().await;
                        metadata.checksum = Some(checksum);
                        
                        // Cache the metadata
                        let mut cache = self.file_metadata_cache.write().await;
                        cache.insert(path.clone(), metadata.clone());
                    }
                    Err(e) => {
                        warn!("Failed to calculate final checksum for {}: {}", path.display(), e);
                    }
                }
            }
        }
        
        info!("Audit logger shutdown completed");
        Ok(())
    }
    
    /// Upload a specific file to S3 manually
    pub async fn upload_to_s3(&self, file_path: &Path) -> Result<bool> {
        if let Some(ref uploader) = self.s3_uploader {
            match uploader.upload_file(file_path).await {
                Ok(result) => {
                    if result.success {
                        info!(
                            "Manual S3 upload successful: {} -> {}",
                            file_path.display(),
                            result.s3_key
                        );
                        
                        // Update metadata cache
                        let mut cache = self.file_metadata_cache.write().await;
                        if let Some(metadata) = cache.get_mut(file_path) {
                            metadata.archived = true;
                        }
                        
                        Ok(true)
                    } else {
                        warn!(
                            "Manual S3 upload failed: {} - {}",
                            file_path.display(),
                            result.error_message.unwrap_or_default()
                        );
                        Ok(false)
                    }
                }
                Err(e) => {
                    error!("S3 upload error for {}: {}", file_path.display(), e);
                    Err(e)
                }
            }
        } else {
            Err(SignalEmissionError::config("S3 uploader not configured"))
        }
    }
    
    /// Upload all unarchived log files to S3
    pub async fn upload_all_to_s3(&self) -> Result<Vec<bool>> {
        if self.s3_uploader.is_none() {
            return Err(SignalEmissionError::config("S3 uploader not configured"));
        }
        
        let cache = self.file_metadata_cache.read().await;
        let unarchived_files: Vec<PathBuf> = cache
            .iter()
            .filter(|(_, metadata)| !metadata.archived && metadata.file_path.exists())
            .map(|(path, _)| path.clone())
            .collect();
        drop(cache);
        
        let mut results = Vec::new();
        for file_path in unarchived_files {
            match self.upload_to_s3(&file_path).await {
                Ok(success) => results.push(success),
                Err(_) => results.push(false),
            }
        }
        
        Ok(results)
    }
    
    /// Get S3 uploader statistics if available
    pub async fn get_s3_stats(&self) -> Option<super::s3_uploader::S3UploaderStats> {
        if let Some(ref uploader) = self.s3_uploader {
            Some(uploader.get_stats().await)
        } else {
            None
        }
    }
    
    /// Test S3 connectivity
    pub async fn test_s3_connection(&self) -> Result<()> {
        if let Some(ref uploader) = self.s3_uploader {
            uploader.test_connection().await
        } else {
            Err(SignalEmissionError::config("S3 uploader not configured"))
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
    use tempfile::TempDir;
    use crate::{SignalSide, SignalComponents, FusionWeights, TradingSignal};
    use super::super::audit::SignalEmissionEvent;
    
    fn create_test_config(temp_dir: &TempDir) -> AuditConfig {
        AuditConfig {
            log_directory: temp_dir.path().to_path_buf(),
            log_filename: "test_audit".to_string(),
            max_file_size_bytes: 1024, // Small size for testing rotation
            max_file_age_seconds: 3600,
            max_files_to_keep: 5,
            compress_rotated_files: false, // Disable for testing
            enable_integrity_verification: true,
            write_buffer_size: 1024,
            flush_after_write: true,
            file_permissions: 0o644,
            create_directories: true,
            s3_config: None, // No S3 for testing
        }
    }
    
    fn create_test_signal() -> TradingSignal {
        let components = SignalComponents {
            s_ldc: 0.5,
            s_mr: 0.3,
            s_tsmom: 0.2,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        TradingSignal::new(
            chrono::Utc::now().timestamp(),
            "BTCUSDT".to_string(),
            SignalSide::Buy,
            0.75,
            0.85,
            components,
            weights,
            "v1.0".to_string(),
            "test-correlation".to_string(),
            "test-checksum".to_string(),
            50,
        )
    }
    
    #[tokio::test]
    async fn test_audit_logger_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        
        let logger = AuditLogger::new(config).await.unwrap();
        let stats = logger.get_stats().await;
        
        assert_eq!(stats.total_files_created, 1);
        assert_eq!(stats.total_events_logged, 0);
    }
    
    #[tokio::test]
    async fn test_signal_emission_logging() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let logger = AuditLogger::new(config).await.unwrap();
        
        let signal = create_test_signal();
        let event = SignalEmissionEvent::success(
            "test-correlation".to_string(),
            signal,
            "redis".to_string(),
            50,
            0,
        );
        
        logger.log_signal_emission(&event).await.unwrap();
        
        let stats = logger.get_stats().await;
        assert_eq!(stats.total_events_logged, 1);
        assert_eq!(stats.current_file_event_count, 1);
        assert!(stats.current_file_size_bytes > 0);
    }
    
    #[tokio::test]
    async fn test_log_rotation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = create_test_config(&temp_dir);
        config.max_file_size_bytes = 100; // Very small for quick rotation
        
        let logger = AuditLogger::new(config).await.unwrap();
        
        // Log multiple events to trigger rotation
        for i in 0..10 {
            let signal = create_test_signal();
            let event = SignalEmissionEvent::success(
                format!("test-correlation-{}", i),
                signal,
                "redis".to_string(),
                50,
                0,
            );
            
            logger.log_signal_emission(&event).await.unwrap();
        }
        
        let stats = logger.get_stats().await;
        assert!(stats.total_files_rotated > 0);
        assert_eq!(stats.total_events_logged, 10);
    }
    
    #[tokio::test]
    async fn test_checksum_calculation() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let logger = AuditLogger::new(config).await.unwrap();
        
        // Write some data
        let signal = create_test_signal();
        let event = SignalEmissionEvent::success(
            "test-correlation".to_string(),
            signal,
            "redis".to_string(),
            50,
            0,
        );
        
        logger.log_signal_emission(&event).await.unwrap();
        logger.force_rotation().await.unwrap();
        
        // Check that checksum was calculated
        let metadata = logger.get_file_metadata().await;
        assert!(!metadata.is_empty());
        
        let first_file_metadata = metadata.values().next().unwrap();
        assert!(first_file_metadata.checksum.is_some());
    }
    
    #[tokio::test]
    async fn test_file_integrity_verification() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let logger = AuditLogger::new(config).await.unwrap();
        
        // Write some data and rotate
        let signal = create_test_signal();
        let event = SignalEmissionEvent::success(
            "test-correlation".to_string(),
            signal,
            "redis".to_string(),
            50,
            0,
        );
        
        logger.log_signal_emission(&event).await.unwrap();
        logger.force_rotation().await.unwrap();
        
        // Verify integrity
        let metadata = logger.get_file_metadata().await;
        let file_path = metadata.keys().next().unwrap();
        
        let is_valid = logger.verify_file_integrity(file_path).await.unwrap();
        assert!(is_valid);
    }
    
    #[tokio::test]
    async fn test_graceful_shutdown() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let logger = AuditLogger::new(config).await.unwrap();
        
        // Log an event
        let signal = create_test_signal();
        let event = SignalEmissionEvent::success(
            "test-correlation".to_string(),
            signal,
            "redis".to_string(),
            50,
            0,
        );
        
        logger.log_signal_emission(&event).await.unwrap();
        
        // Shutdown should complete without errors
        logger.shutdown().await.unwrap();
        
        let stats = logger.get_stats().await;
        assert_eq!(stats.total_events_logged, 1);
    }
    
    #[tokio::test]
    async fn test_flush_operation() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let logger = AuditLogger::new(config).await.unwrap();
        
        // Log an event
        let signal = create_test_signal();
        let event = SignalEmissionEvent::success(
            "test-correlation".to_string(),
            signal,
            "redis".to_string(),
            50,
            0,
        );
        
        logger.log_signal_emission(&event).await.unwrap();
        
        // Flush should complete without errors
        logger.flush().await.unwrap();
    }
}