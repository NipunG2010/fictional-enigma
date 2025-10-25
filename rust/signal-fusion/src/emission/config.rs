//! Unified configuration management for signal emission
//! 
//! This module provides comprehensive configuration management for the signal emission system,
//! supporting TOML files, environment variables, and configuration validation.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug};

use super::{
    SignalEmissionError, Result,
    SignalPublisherConfig, PublisherBackend,
    RedisConfig, KafkaConfig,
    BufferConfig, AuditConfig, ValidationConfig,
};

/// Complete configuration for the signal emission system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalEmissionConfig {
    /// Publisher configuration
    pub publisher: SignalPublisherConfig,
    
    /// Redis configuration (optional, required if backend includes Redis)
    pub redis: Option<RedisConfig>,
    
    /// Kafka configuration (optional, required if backend includes Kafka)
    pub kafka: Option<KafkaConfig>,
    
    /// Buffer configuration for local buffering
    pub buffer: BufferConfig,
    
    /// Audit logging configuration
    pub audit: AuditConfig,
    
    /// Signal validation configuration
    pub validation: ValidationConfig,
    
    /// Configuration metadata
    pub metadata: ConfigMetadata,
}

/// Configuration metadata for tracking and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMetadata {
    /// Configuration version for compatibility checking
    pub version: String,
    
    /// Source of the configuration (file, environment, default)
    pub source: ConfigSource,
    
    /// Path to the configuration file (if loaded from file)
    pub file_path: Option<PathBuf>,
    
    /// Timestamp when configuration was loaded
    pub loaded_at: i64,
    
    /// Environment variables that were used
    pub env_vars_used: Vec<String>,
    
    /// Validation warnings (non-fatal issues)
    pub warnings: Vec<String>,
}

/// Source of configuration values
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConfigSource {
    /// Configuration loaded from TOML file
    File,
    
    /// Configuration loaded from environment variables
    Environment,
    
    /// Configuration created from defaults
    Default,
    
    /// Configuration merged from multiple sources
    Merged,
}

impl Default for SignalEmissionConfig {
    fn default() -> Self {
        Self {
            publisher: SignalPublisherConfig::default(),
            redis: None,
            kafka: None,
            buffer: BufferConfig::default(),
            audit: AuditConfig::default(),
            validation: ValidationConfig::default(),
            metadata: ConfigMetadata {
                version: "1.0.0".to_string(),
                source: ConfigSource::Default,
                file_path: None,
                loaded_at: chrono::Utc::now().timestamp(),
                env_vars_used: Vec::new(),
                warnings: Vec::new(),
            },
        }
    }
}

impl SignalEmissionConfig {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading signal emission configuration from: {:?}", path);
        
        let content = fs::read_to_string(path)
            .map_err(|e| SignalEmissionError::config(
                format!("Failed to read config file {:?}: {}", path, e)
            ))?;
        
        let mut config: Self = toml::from_str(&content)
            .map_err(|e| SignalEmissionError::config(
                format!("Failed to parse TOML config: {}", e)
            ))?;
        
        // Update metadata
        config.metadata.source = ConfigSource::File;
        config.metadata.file_path = Some(path.to_path_buf());
        config.metadata.loaded_at = chrono::Utc::now().timestamp();
        
        // Apply environment variable overrides
        config.apply_env_overrides()?;
        
        // Validate the configuration
        config.validate()?;
        
        info!("Successfully loaded configuration from: {:?}", path);
        Ok(config)
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        info!("Loading signal emission configuration from environment variables");
        
        let mut config = Self::default();
        config.metadata.source = ConfigSource::Environment;
        config.metadata.loaded_at = chrono::Utc::now().timestamp();
        
        // Apply environment variable overrides
        config.apply_env_overrides()?;
        
        // Validate the configuration
        config.validate()?;
        
        info!("Successfully loaded configuration from environment variables");
        Ok(config)
    }
    
    /// Load configuration with fallback: file -> environment -> defaults
    pub fn load_with_fallback<P: AsRef<Path>>(config_path: Option<P>) -> Result<Self> {
        // Try to load from file first
        if let Some(path) = config_path {
            match Self::from_file(path) {
                Ok(mut config) => {
                    config.metadata.source = ConfigSource::Merged;
                    return Ok(config);
                }
                Err(e) => {
                    warn!("Failed to load config from file, falling back to environment: {}", e);
                }
            }
        }
        
        // Try environment variables
        match Self::from_env() {
            Ok(mut config) => {
                config.metadata.source = ConfigSource::Merged;
                Ok(config)
            }
            Err(e) => {
                warn!("Failed to load config from environment, using defaults: {}", e);
                let mut config = Self::default();
                config.metadata.source = ConfigSource::Merged;
                Ok(config)
            }
        }
    }
    
    /// Apply environment variable overrides to the configuration
    fn apply_env_overrides(&mut self) -> Result<()> {
        let mut env_vars_used = Vec::new();
        
        // Publisher configuration
        if let Ok(enabled) = env::var("SIGNAL_PUBLISHER_ENABLED") {
            self.publisher.enabled = enabled.parse()
                .map_err(|_| SignalEmissionError::config(
                    format!("Invalid SIGNAL_PUBLISHER_ENABLED value: {}", enabled)
                ))?;
            env_vars_used.push("SIGNAL_PUBLISHER_ENABLED".to_string());
        }
        
        if let Ok(backend) = env::var("SIGNAL_PUBLISHER_BACKEND") {
            self.publisher.backend = backend.parse()?;
            env_vars_used.push("SIGNAL_PUBLISHER_BACKEND".to_string());
        }
        
        if let Ok(timeout) = env::var("SIGNAL_PUBLISHER_TIMEOUT_MS") {
            self.publisher.publish_timeout_ms = timeout.parse()
                .map_err(|_| SignalEmissionError::config(
                    format!("Invalid SIGNAL_PUBLISHER_TIMEOUT_MS value: {}", timeout)
                ))?;
            env_vars_used.push("SIGNAL_PUBLISHER_TIMEOUT_MS".to_string());
        }
        
        if let Ok(fail_fast) = env::var("SIGNAL_PUBLISHER_FAIL_FAST") {
            self.publisher.fail_fast = fail_fast.parse()
                .map_err(|_| SignalEmissionError::config(
                    format!("Invalid SIGNAL_PUBLISHER_FAIL_FAST value: {}", fail_fast)
                ))?;
            env_vars_used.push("SIGNAL_PUBLISHER_FAIL_FAST".to_string());
        }
        
        if let Ok(max_concurrent) = env::var("SIGNAL_PUBLISHER_MAX_CONCURRENT") {
            self.publisher.max_concurrent_publishes = max_concurrent.parse()
                .map_err(|_| SignalEmissionError::config(
                    format!("Invalid SIGNAL_PUBLISHER_MAX_CONCURRENT value: {}", max_concurrent)
                ))?;
            env_vars_used.push("SIGNAL_PUBLISHER_MAX_CONCURRENT".to_string());
        }
        
        // Redis configuration
        self.apply_redis_env_overrides(&mut env_vars_used)?;
        
        // Kafka configuration
        self.apply_kafka_env_overrides(&mut env_vars_used)?;
        
        // Buffer configuration
        self.apply_buffer_env_overrides(&mut env_vars_used)?;
        
        // Audit configuration
        self.apply_audit_env_overrides(&mut env_vars_used)?;
        
        // Validation configuration
        self.apply_validation_env_overrides(&mut env_vars_used)?;
        
        self.metadata.env_vars_used = env_vars_used;
        Ok(())
    }
    
    /// Apply Redis-specific environment variable overrides
    fn apply_redis_env_overrides(&mut self, env_vars_used: &mut Vec<String>) -> Result<()> {
        if let Ok(redis_url) = env::var("REDIS_URL") {
            if self.redis.is_none() {
                self.redis = Some(RedisConfig::default());
            }
            if let Some(ref mut redis_config) = self.redis {
                redis_config.url = redis_url;
                env_vars_used.push("REDIS_URL".to_string());
            }
        }
        
        if let Ok(stream_name) = env::var("REDIS_STREAM_NAME") {
            if self.redis.is_none() {
                self.redis = Some(RedisConfig::default());
            }
            if let Some(ref mut redis_config) = self.redis {
                redis_config.stream.stream_name = stream_name;
                env_vars_used.push("REDIS_STREAM_NAME".to_string());
            }
        }
        
        if let Ok(max_len) = env::var("REDIS_STREAM_MAX_LENGTH") {
            if self.redis.is_none() {
                self.redis = Some(RedisConfig::default());
            }
            if let Some(ref mut redis_config) = self.redis {
                redis_config.stream.max_len = Some(max_len.parse()
                    .map_err(|_| SignalEmissionError::config(
                        format!("Invalid REDIS_STREAM_MAX_LENGTH value: {}", max_len)
                    ))?);
                env_vars_used.push("REDIS_STREAM_MAX_LENGTH".to_string());
            }
        }
        
        if let Ok(pool_size) = env::var("REDIS_CONNECTION_POOL_SIZE") {
            if self.redis.is_none() {
                self.redis = Some(RedisConfig::default());
            }
            if let Some(ref mut redis_config) = self.redis {
                redis_config.pool_size = pool_size.parse()
                    .map_err(|_| SignalEmissionError::config(
                        format!("Invalid REDIS_CONNECTION_POOL_SIZE value: {}", pool_size)
                    ))?;
                env_vars_used.push("REDIS_CONNECTION_POOL_SIZE".to_string());
            }
        }
        
        Ok(())
    }
    
    /// Apply Kafka-specific environment variable overrides
    fn apply_kafka_env_overrides(&mut self, env_vars_used: &mut Vec<String>) -> Result<()> {
        if let Ok(brokers) = env::var("KAFKA_BROKERS") {
            if self.kafka.is_none() {
                self.kafka = Some(KafkaConfig::default());
            }
            if let Some(ref mut kafka_config) = self.kafka {
                kafka_config.brokers = brokers;
                env_vars_used.push("KAFKA_BROKERS".to_string());
            }
        }
        
        if let Ok(topic) = env::var("KAFKA_TOPIC") {
            if self.kafka.is_none() {
                self.kafka = Some(KafkaConfig::default());
            }
            if let Some(ref mut kafka_config) = self.kafka {
                kafka_config.topic = topic;
                env_vars_used.push("KAFKA_TOPIC".to_string());
            }
        }
        
        if let Ok(compression) = env::var("KAFKA_COMPRESSION") {
            if self.kafka.is_none() {
                self.kafka = Some(KafkaConfig::default());
            }
            if let Some(ref mut kafka_config) = self.kafka {
                kafka_config.producer.compression_type = compression;
                env_vars_used.push("KAFKA_COMPRESSION".to_string());
            }
        }
        
        Ok(())
    }
    
    /// Apply buffer-specific environment variable overrides
    fn apply_buffer_env_overrides(&mut self, env_vars_used: &mut Vec<String>) -> Result<()> {
        if let Ok(max_size) = env::var("SIGNAL_BUFFER_MAX_SIZE") {
            self.buffer.max_size = max_size.parse()
                .map_err(|_| SignalEmissionError::config(
                    format!("Invalid SIGNAL_BUFFER_MAX_SIZE value: {}", max_size)
                ))?;
            env_vars_used.push("SIGNAL_BUFFER_MAX_SIZE".to_string());
        }
        
        if let Ok(persist_enabled) = env::var("SIGNAL_BUFFER_PERSIST_ENABLED") {
            let enabled = persist_enabled.parse()
                .map_err(|_| SignalEmissionError::config(
                    format!("Invalid SIGNAL_BUFFER_PERSIST_ENABLED value: {}", persist_enabled)
                ))?;
            
            if enabled && self.buffer.persistence.is_none() {
                // Create default persistence config
                self.buffer.persistence = Some(super::buffer::PersistenceConfig::default());
            } else if !enabled {
                self.buffer.persistence = None;
            }
            env_vars_used.push("SIGNAL_BUFFER_PERSIST_ENABLED".to_string());
        }
        
        if let Ok(persist_path) = env::var("SIGNAL_BUFFER_PERSIST_PATH") {
            if self.buffer.persistence.is_none() {
                self.buffer.persistence = Some(super::buffer::PersistenceConfig::default());
            }
            if let Some(ref mut persistence) = self.buffer.persistence {
                persistence.persist_path = PathBuf::from(persist_path);
                env_vars_used.push("SIGNAL_BUFFER_PERSIST_PATH".to_string());
            }
        }
        
        Ok(())
    }
    
    /// Apply audit-specific environment variable overrides
    fn apply_audit_env_overrides(&mut self, env_vars_used: &mut Vec<String>) -> Result<()> {
        // Note: AuditConfig doesn't have an enabled field - it's always enabled if configured
        
        if let Ok(log_dir) = env::var("AUDIT_LOG_DIRECTORY") {
            self.audit.log_directory = PathBuf::from(log_dir);
            env_vars_used.push("AUDIT_LOG_DIRECTORY".to_string());
        }
        
        if let Ok(max_file_size) = env::var("AUDIT_MAX_FILE_SIZE_BYTES") {
            self.audit.max_file_size_bytes = max_file_size.parse()
                .map_err(|_| SignalEmissionError::config(
                    format!("Invalid AUDIT_MAX_FILE_SIZE_BYTES value: {}", max_file_size)
                ))?;
            env_vars_used.push("AUDIT_MAX_FILE_SIZE_BYTES".to_string());
        }
        
        if let Ok(s3_bucket) = env::var("AUDIT_S3_BUCKET") {
            if self.audit.s3_config.is_none() {
                self.audit.s3_config = Some(super::s3_uploader::S3Config::default());
            }
            if let Some(ref mut s3_config) = self.audit.s3_config {
                s3_config.bucket = s3_bucket;
                env_vars_used.push("AUDIT_S3_BUCKET".to_string());
            }
        }
        
        if let Ok(s3_prefix) = env::var("AUDIT_S3_PREFIX") {
            if self.audit.s3_config.is_none() {
                self.audit.s3_config = Some(super::s3_uploader::S3Config::default());
            }
            if let Some(ref mut s3_config) = self.audit.s3_config {
                s3_config.key_prefix = s3_prefix;
                env_vars_used.push("AUDIT_S3_PREFIX".to_string());
            }
        }
        
        Ok(())
    }
    
    /// Apply validation-specific environment variable overrides
    fn apply_validation_env_overrides(&mut self, env_vars_used: &mut Vec<String>) -> Result<()> {
        if let Ok(strict_consistency) = env::var("SIGNAL_VALIDATION_STRICT_CONSISTENCY") {
            self.validation.strict_consistency_check = strict_consistency.parse()
                .map_err(|_| SignalEmissionError::config(
                    format!("Invalid SIGNAL_VALIDATION_STRICT_CONSISTENCY value: {}", strict_consistency)
                ))?;
            env_vars_used.push("SIGNAL_VALIDATION_STRICT_CONSISTENCY".to_string());
        }
        
        if let Ok(max_signal_age) = env::var("SIGNAL_VALIDATION_MAX_SIGNAL_AGE_SEC") {
            self.validation.max_signal_age_seconds = max_signal_age.parse()
                .map_err(|_| SignalEmissionError::config(
                    format!("Invalid SIGNAL_VALIDATION_MAX_SIGNAL_AGE_SEC value: {}", max_signal_age)
                ))?;
            env_vars_used.push("SIGNAL_VALIDATION_MAX_SIGNAL_AGE_SEC".to_string());
        }
        
        Ok(())
    }
    
    /// Validate the complete configuration
    pub fn validate(&mut self) -> Result<()> {
        debug!("Validating signal emission configuration");
        
        // Validate publisher configuration
        self.publisher.validate()?;
        
        // Ensure required backend configurations are present
        match self.publisher.backend {
            PublisherBackend::Redis => {
                if self.redis.is_none() {
                    return Err(SignalEmissionError::config(
                        "Redis configuration required when backend is Redis"
                    ));
                }
                self.publisher.redis = self.redis.clone();
            }
            PublisherBackend::Kafka => {
                if self.kafka.is_none() {
                    return Err(SignalEmissionError::config(
                        "Kafka configuration required when backend is Kafka"
                    ));
                }
                self.publisher.kafka = self.kafka.clone();
            }
            PublisherBackend::Both => {
                if self.redis.is_none() {
                    return Err(SignalEmissionError::config(
                        "Redis configuration required when backend includes Redis"
                    ));
                }
                if self.kafka.is_none() {
                    return Err(SignalEmissionError::config(
                        "Kafka configuration required when backend includes Kafka"
                    ));
                }
                self.publisher.redis = self.redis.clone();
                self.publisher.kafka = self.kafka.clone();
            }
            PublisherBackend::None => {
                // No backend configuration required for testing mode
                self.metadata.warnings.push(
                    "Publisher backend is set to 'none' - signals will not be published".to_string()
                );
            }
        }
        
        // Copy other configurations to publisher
        self.publisher.buffer = self.buffer.clone();
        self.publisher.audit = self.audit.clone();
        self.publisher.validation = self.validation.clone();
        
        // Individual component validation would go here if they had validate methods
        // For now, we rely on the type system and basic field validation
        
        // Check for potential configuration conflicts
        self.check_configuration_conflicts()?;
        
        debug!("Configuration validation completed successfully");
        Ok(())
    }
    
    /// Check for potential configuration conflicts and add warnings
    fn check_configuration_conflicts(&mut self) -> Result<()> {
        // Check if buffer is too small for high-throughput scenarios
        if self.buffer.max_size < 100 && self.publisher.enabled {
            self.metadata.warnings.push(
                "Buffer size is very small (< 100) which may cause signal loss under high load".to_string()
            );
        }
        
        // Check if S3 archival is not configured
        if self.audit.s3_config.is_none() && self.publisher.enabled {
            self.metadata.warnings.push(
                "S3 archival is not configured - audit logs will only be stored locally".to_string()
            );
        }
        
        // Check timeout settings
        if self.publisher.publish_timeout_ms < 1000 {
            self.metadata.warnings.push(
                "Publish timeout is very low (< 1s) which may cause frequent timeouts".to_string()
            );
        }
        
        Ok(())
    }
    
    /// Save configuration to a TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        info!("Saving signal emission configuration to: {:?}", path);
        
        let toml_content = toml::to_string_pretty(self)
            .map_err(|e| SignalEmissionError::config(
                format!("Failed to serialize config to TOML: {}", e)
            ))?;
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| SignalEmissionError::config(
                    format!("Failed to create config directory {:?}: {}", parent, e)
                ))?;
        }
        
        fs::write(path, toml_content)
            .map_err(|e| SignalEmissionError::config(
                format!("Failed to write config file {:?}: {}", path, e)
            ))?;
        
        info!("Successfully saved configuration to: {:?}", path);
        Ok(())
    }
    
    /// Get a summary of the configuration for logging
    pub fn summary(&self) -> ConfigSummary {
        ConfigSummary {
            backend: self.publisher.backend.clone(),
            enabled: self.publisher.enabled,
            redis_configured: self.redis.is_some(),
            kafka_configured: self.kafka.is_some(),
            buffer_max_size: self.buffer.max_size,
            buffer_persistence_enabled: self.buffer.persistence.is_some(),
            audit_enabled: true, // Audit is always enabled if configured
            validation_enabled: true, // Validation is always enabled if configured
            source: self.metadata.source.clone(),
            warnings_count: self.metadata.warnings.len(),
        }
    }
    
    /// Get configuration warnings
    pub fn warnings(&self) -> &[String] {
        &self.metadata.warnings
    }
    
    /// Check if configuration has any warnings
    pub fn has_warnings(&self) -> bool {
        !self.metadata.warnings.is_empty()
    }
    
    /// Get environment variables that were used
    pub fn env_vars_used(&self) -> &[String] {
        &self.metadata.env_vars_used
    }
    
    /// Get the configuration source
    pub fn source(&self) -> &ConfigSource {
        &self.metadata.source
    }
    
    /// Get the configuration version
    pub fn version(&self) -> &str {
        &self.metadata.version
    }
}

/// Configuration summary for logging and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSummary {
    pub backend: PublisherBackend,
    pub enabled: bool,
    pub redis_configured: bool,
    pub kafka_configured: bool,
    pub buffer_max_size: usize,
    pub buffer_persistence_enabled: bool,
    pub audit_enabled: bool,
    pub validation_enabled: bool,
    pub source: ConfigSource,
    pub warnings_count: usize,
}

/// Configuration hot-reloading support
pub struct ConfigWatcher {
    config_path: PathBuf,
    last_modified: Option<std::time::SystemTime>,
}

impl ConfigWatcher {
    /// Create a new configuration watcher
    pub fn new<P: AsRef<Path>>(config_path: P) -> Self {
        Self {
            config_path: config_path.as_ref().to_path_buf(),
            last_modified: None,
        }
    }
    
    /// Check if the configuration file has been modified
    pub fn check_for_changes(&mut self) -> Result<bool> {
        let metadata = fs::metadata(&self.config_path)
            .map_err(|e| SignalEmissionError::config(
                format!("Failed to read config file metadata: {}", e)
            ))?;
        
        let modified = metadata.modified()
            .map_err(|e| SignalEmissionError::config(
                format!("Failed to get file modification time: {}", e)
            ))?;
        
        let changed = match self.last_modified {
            Some(last) => modified > last,
            None => {
                self.last_modified = Some(modified);
                false
            }
        };
        
        if changed {
            self.last_modified = Some(modified);
        }
        
        Ok(changed)
    }
    
    /// Reload configuration if it has changed
    pub fn reload_if_changed(&mut self) -> Result<Option<SignalEmissionConfig>> {
        if self.check_for_changes()? {
            info!("Configuration file changed, reloading: {:?}", self.config_path);
            let config = SignalEmissionConfig::from_file(&self.config_path)?;
            Ok(Some(config))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_default_config() {
        let config = SignalEmissionConfig::default();
        assert_eq!(config.publisher.backend, PublisherBackend::None);
        assert!(config.publisher.enabled);
        assert_eq!(config.metadata.source, ConfigSource::Default);
        assert_eq!(config.metadata.version, "1.0.0");
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = SignalEmissionConfig::default();
        
        // Should validate successfully with default values
        assert!(config.validate().is_ok());
        
        // Should fail validation when Redis backend is selected but no Redis config
        config.publisher.backend = PublisherBackend::Redis;
        assert!(config.validate().is_err());
        
        // Should succeed when Redis config is provided
        config.redis = Some(RedisConfig::default());
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_env_var_overrides() {
        // Set some environment variables
        env::set_var("SIGNAL_PUBLISHER_ENABLED", "false");
        env::set_var("SIGNAL_PUBLISHER_BACKEND", "redis");
        env::set_var("REDIS_URL", "redis://test:6379");
        
        let config = SignalEmissionConfig::from_env().unwrap();
        
        assert!(!config.publisher.enabled);
        assert_eq!(config.publisher.backend, PublisherBackend::Redis);
        assert!(config.redis.is_some());
        assert_eq!(config.redis.unwrap().url, "redis://test:6379");
        
        // Clean up
        env::remove_var("SIGNAL_PUBLISHER_ENABLED");
        env::remove_var("SIGNAL_PUBLISHER_BACKEND");
        env::remove_var("REDIS_URL");
    }
    
    #[test]
    fn test_config_file_save_load() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        // Create a test configuration
        let mut original_config = SignalEmissionConfig::default();
        original_config.publisher.backend = PublisherBackend::Redis;
        original_config.redis = Some(RedisConfig::default());
        original_config.validate().unwrap();
        
        // Save to file
        original_config.save_to_file(&config_path).unwrap();
        
        // Load from file
        let loaded_config = SignalEmissionConfig::from_file(&config_path).unwrap();
        
        assert_eq!(loaded_config.publisher.backend, PublisherBackend::Redis);
        assert!(loaded_config.redis.is_some());
        assert_eq!(loaded_config.metadata.source, ConfigSource::File);
    }
    
    #[test]
    fn test_config_summary() {
        let mut config = SignalEmissionConfig::default();
        config.publisher.backend = PublisherBackend::Both;
        config.redis = Some(RedisConfig::default());
        config.kafka = Some(KafkaConfig::default());
        config.validate().unwrap();
        
        let summary = config.summary();
        assert_eq!(summary.backend, PublisherBackend::Both);
        assert!(summary.enabled);
        assert!(summary.redis_configured);
        assert!(summary.kafka_configured);
        assert!(summary.audit_enabled);
        assert!(summary.validation_enabled);
    }
    
    #[test]
    fn test_config_warnings() {
        let mut config = SignalEmissionConfig::default();
        config.publisher.backend = PublisherBackend::None;
        config.buffer.max_size = 10; // Very small buffer
        // Note: audit and validation configs don't have enabled fields in the current implementation
        // These would be controlled by the publisher.enabled field
        config.publisher.publish_timeout_ms = 500; // Very low timeout
        
        config.validate().unwrap();
        
        assert!(config.has_warnings());
        let warnings = config.warnings();
        assert!(!warnings.is_empty());
        
        // Should have warnings about small buffer and low timeout
        assert!(warnings.iter().any(|w| w.contains("Buffer size is very small")));
        assert!(warnings.iter().any(|w| w.contains("Signal validation is disabled")));
        assert!(warnings.iter().any(|w| w.contains("Publish timeout is very low")));
    }
    
    #[test]
    fn test_config_watcher() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("watch_config.toml");
        
        // Create initial config file
        let config = SignalEmissionConfig::default();
        config.save_to_file(&config_path).unwrap();
        
        let mut watcher = ConfigWatcher::new(&config_path);
        
        // First check should return false (no previous state)
        assert!(!watcher.check_for_changes().unwrap());
        
        // Modify the file
        std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamp
        config.save_to_file(&config_path).unwrap();
        
        // Should detect change
        assert!(watcher.check_for_changes().unwrap());
        
        // Subsequent check without modification should return false
        assert!(!watcher.check_for_changes().unwrap());
    }
}