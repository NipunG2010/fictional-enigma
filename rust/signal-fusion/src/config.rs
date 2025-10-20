//! Configuration management for HMM integration
//!
//! This module provides comprehensive configuration management including:
//! - Environment variable parsing
//! - TOML configuration file support
//! - Configuration validation
//! - Default values

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Duration;
use url::Url;

use crate::FusionWeights;

/// Complete configuration for HMM integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmIntegrationConfig {
    /// HMM service configuration
    #[serde(default)]
    pub service: ServiceConfig,
    
    /// Circuit breaker configuration
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    
    /// Cache configuration
    #[serde(default)]
    pub cache: CacheConfig,
    
    /// Fallback configuration
    #[serde(default)]
    pub fallback: FallbackConfig,
    
    /// Signal fusion configuration
    #[serde(default)]
    pub signal_fusion: SignalFusionConfig,
}

/// HMM service connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Base URL of the HMM service
    pub url: String,
    
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    
    /// Number of retry attempts
    pub retry_attempts: usize,
    
    /// Delay between retries in milliseconds
    pub retry_delay_ms: u64,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening circuit
    pub threshold: usize,
    
    /// Timeout in seconds before attempting recovery
    pub timeout_sec: u64,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Time-to-live for cache entries in seconds
    pub ttl_sec: u64,
    
    /// Maximum number of cache entries
    pub max_size: usize,
}

/// Fallback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackConfig {
    /// Enable fallback to static weights
    pub enabled: bool,
    
    /// Fallback weight for LDC signal
    pub w_ldc: f32,
    
    /// Fallback weight for MR signal
    pub w_mr: f32,
    
    /// Fallback weight for TSMOM signal
    pub w_tsmom: f32,
}

/// Signal fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalFusionConfig {
    /// Minimum signal strength threshold
    pub threshold: f32,
    
    /// Cooldown period between signals in seconds
    pub cooldown_sec: u64,
}

impl Default for HmmIntegrationConfig {
    fn default() -> Self {
        Self {
            service: ServiceConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            cache: CacheConfig::default(),
            fallback: FallbackConfig::default(),
            signal_fusion: SignalFusionConfig::default(),
        }
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8000".to_string(),
            timeout_ms: 5000,
            retry_attempts: 3,
            retry_delay_ms: 100,
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            threshold: 5,
            timeout_sec: 30,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            ttl_sec: 60,
            max_size: 1000,
        }
    }
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            w_ldc: 0.33,
            w_mr: 0.33,
            w_tsmom: 0.34,
        }
    }
}

impl Default for SignalFusionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            cooldown_sec: 60,
        }
    }
}

impl HmmIntegrationConfig {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path.as_ref())
            .context("Failed to read configuration file")?;
        
        let config: HmmIntegrationConfig = toml::from_str(&contents)
            .context("Failed to parse TOML configuration")?;
        
        config.validate()?;
        
        Ok(config)
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        
        // Service configuration
        if let Ok(url) = std::env::var("HMM_SERVICE_URL") {
            config.service.url = url;
        }
        if let Ok(timeout) = std::env::var("HMM_SERVICE_TIMEOUT_MS") {
            config.service.timeout_ms = timeout.parse()
                .context("Invalid HMM_SERVICE_TIMEOUT_MS")?;
        }
        if let Ok(attempts) = std::env::var("HMM_SERVICE_RETRY_ATTEMPTS") {
            config.service.retry_attempts = attempts.parse()
                .context("Invalid HMM_SERVICE_RETRY_ATTEMPTS")?;
        }
        if let Ok(delay) = std::env::var("HMM_SERVICE_RETRY_DELAY_MS") {
            config.service.retry_delay_ms = delay.parse()
                .context("Invalid HMM_SERVICE_RETRY_DELAY_MS")?;
        }
        
        // Circuit breaker configuration
        if let Ok(threshold) = std::env::var("HMM_CIRCUIT_BREAKER_THRESHOLD") {
            config.circuit_breaker.threshold = threshold.parse()
                .context("Invalid HMM_CIRCUIT_BREAKER_THRESHOLD")?;
        }
        if let Ok(timeout) = std::env::var("HMM_CIRCUIT_BREAKER_TIMEOUT_SEC") {
            config.circuit_breaker.timeout_sec = timeout.parse()
                .context("Invalid HMM_CIRCUIT_BREAKER_TIMEOUT_SEC")?;
        }
        
        // Cache configuration
        if let Ok(ttl) = std::env::var("HMM_CACHE_TTL_SEC") {
            config.cache.ttl_sec = ttl.parse()
                .context("Invalid HMM_CACHE_TTL_SEC")?;
        }
        if let Ok(max_size) = std::env::var("HMM_CACHE_MAX_SIZE") {
            config.cache.max_size = max_size.parse()
                .context("Invalid HMM_CACHE_MAX_SIZE")?;
        }
        
        // Fallback configuration
        if let Ok(enabled) = std::env::var("HMM_ENABLE_FALLBACK") {
            config.fallback.enabled = enabled.parse()
                .context("Invalid HMM_ENABLE_FALLBACK")?;
        }
        if let Ok(w_ldc) = std::env::var("HMM_FALLBACK_W_LDC") {
            config.fallback.w_ldc = w_ldc.parse()
                .context("Invalid HMM_FALLBACK_W_LDC")?;
        }
        if let Ok(w_mr) = std::env::var("HMM_FALLBACK_W_MR") {
            config.fallback.w_mr = w_mr.parse()
                .context("Invalid HMM_FALLBACK_W_MR")?;
        }
        if let Ok(w_tsmom) = std::env::var("HMM_FALLBACK_W_TSMOM") {
            config.fallback.w_tsmom = w_tsmom.parse()
                .context("Invalid HMM_FALLBACK_W_TSMOM")?;
        }
        
        // Signal fusion configuration
        if let Ok(threshold) = std::env::var("SIGNAL_FUSION_THRESHOLD") {
            config.signal_fusion.threshold = threshold.parse()
                .context("Invalid SIGNAL_FUSION_THRESHOLD")?;
        }
        if let Ok(cooldown) = std::env::var("SIGNAL_FUSION_COOLDOWN_SEC") {
            config.signal_fusion.cooldown_sec = cooldown.parse()
                .context("Invalid SIGNAL_FUSION_COOLDOWN_SEC")?;
        }
        
        config.validate()?;
        
        Ok(config)
    }
    
    /// Load configuration with priority: file > env > defaults
    /// If a file path is provided, it takes precedence over environment variables
    pub fn load(file_path: Option<&Path>) -> Result<Self> {
        if let Some(path) = file_path {
            // Load from file first
            let mut config = Self::from_file(path)?;
            
            // Override with environment variables if present
            config.merge_from_env()?;
            
            config.validate()?;
            Ok(config)
        } else {
            // Load from environment or use defaults
            Self::from_env()
        }
    }
    
    /// Merge environment variables into existing configuration
    fn merge_from_env(&mut self) -> Result<()> {
        // Service configuration
        if let Ok(url) = std::env::var("HMM_SERVICE_URL") {
            self.service.url = url;
        }
        if let Ok(timeout) = std::env::var("HMM_SERVICE_TIMEOUT_MS") {
            self.service.timeout_ms = timeout.parse()
                .context("Invalid HMM_SERVICE_TIMEOUT_MS")?;
        }
        if let Ok(attempts) = std::env::var("HMM_SERVICE_RETRY_ATTEMPTS") {
            self.service.retry_attempts = attempts.parse()
                .context("Invalid HMM_SERVICE_RETRY_ATTEMPTS")?;
        }
        if let Ok(delay) = std::env::var("HMM_SERVICE_RETRY_DELAY_MS") {
            self.service.retry_delay_ms = delay.parse()
                .context("Invalid HMM_SERVICE_RETRY_DELAY_MS")?;
        }
        
        // Circuit breaker configuration
        if let Ok(threshold) = std::env::var("HMM_CIRCUIT_BREAKER_THRESHOLD") {
            self.circuit_breaker.threshold = threshold.parse()
                .context("Invalid HMM_CIRCUIT_BREAKER_THRESHOLD")?;
        }
        if let Ok(timeout) = std::env::var("HMM_CIRCUIT_BREAKER_TIMEOUT_SEC") {
            self.circuit_breaker.timeout_sec = timeout.parse()
                .context("Invalid HMM_CIRCUIT_BREAKER_TIMEOUT_SEC")?;
        }
        
        // Cache configuration
        if let Ok(ttl) = std::env::var("HMM_CACHE_TTL_SEC") {
            self.cache.ttl_sec = ttl.parse()
                .context("Invalid HMM_CACHE_TTL_SEC")?;
        }
        if let Ok(max_size) = std::env::var("HMM_CACHE_MAX_SIZE") {
            self.cache.max_size = max_size.parse()
                .context("Invalid HMM_CACHE_MAX_SIZE")?;
        }
        
        // Fallback configuration
        if let Ok(enabled) = std::env::var("HMM_ENABLE_FALLBACK") {
            self.fallback.enabled = enabled.parse()
                .context("Invalid HMM_ENABLE_FALLBACK")?;
        }
        if let Ok(w_ldc) = std::env::var("HMM_FALLBACK_W_LDC") {
            self.fallback.w_ldc = w_ldc.parse()
                .context("Invalid HMM_FALLBACK_W_LDC")?;
        }
        if let Ok(w_mr) = std::env::var("HMM_FALLBACK_W_MR") {
            self.fallback.w_mr = w_mr.parse()
                .context("Invalid HMM_FALLBACK_W_MR")?;
        }
        if let Ok(w_tsmom) = std::env::var("HMM_FALLBACK_W_TSMOM") {
            self.fallback.w_tsmom = w_tsmom.parse()
                .context("Invalid HMM_FALLBACK_W_TSMOM")?;
        }
        
        // Signal fusion configuration
        if let Ok(threshold) = std::env::var("SIGNAL_FUSION_THRESHOLD") {
            self.signal_fusion.threshold = threshold.parse()
                .context("Invalid SIGNAL_FUSION_THRESHOLD")?;
        }
        if let Ok(cooldown) = std::env::var("SIGNAL_FUSION_COOLDOWN_SEC") {
            self.signal_fusion.cooldown_sec = cooldown.parse()
                .context("Invalid SIGNAL_FUSION_COOLDOWN_SEC")?;
        }
        
        Ok(())
    }
    
    /// Validate configuration values
    pub fn validate(&self) -> Result<()> {
        // Validate service URL
        Url::parse(&self.service.url)
            .context("Invalid service URL")?;
        
        // Validate timeouts are reasonable
        if self.service.timeout_ms == 0 {
            anyhow::bail!("Service timeout must be greater than 0");
        }
        if self.service.timeout_ms > 60000 {
            anyhow::bail!("Service timeout should not exceed 60 seconds");
        }
        
        // Validate retry configuration
        if self.service.retry_attempts > 10 {
            anyhow::bail!("Retry attempts should not exceed 10");
        }
        
        // Validate circuit breaker
        if self.circuit_breaker.threshold == 0 {
            anyhow::bail!("Circuit breaker threshold must be greater than 0");
        }
        if self.circuit_breaker.timeout_sec == 0 {
            anyhow::bail!("Circuit breaker timeout must be greater than 0");
        }
        
        // Validate cache configuration
        if self.cache.ttl_sec == 0 {
            anyhow::bail!("Cache TTL must be greater than 0");
        }
        if self.cache.max_size == 0 {
            anyhow::bail!("Cache max size must be greater than 0");
        }
        
        // Validate fallback weights
        if self.fallback.w_ldc < -1.0 || self.fallback.w_ldc > 1.0 {
            anyhow::bail!("Fallback w_ldc must be in range [-1.0, 1.0]");
        }
        if self.fallback.w_mr < -1.0 || self.fallback.w_mr > 1.0 {
            anyhow::bail!("Fallback w_mr must be in range [-1.0, 1.0]");
        }
        if self.fallback.w_tsmom < -1.0 || self.fallback.w_tsmom > 1.0 {
            anyhow::bail!("Fallback w_tsmom must be in range [-1.0, 1.0]");
        }
        
        // Validate signal fusion
        if self.signal_fusion.threshold < 0.0 || self.signal_fusion.threshold > 1.0 {
            anyhow::bail!("Signal fusion threshold must be in range [0.0, 1.0]");
        }
        
        Ok(())
    }
    
    /// Convert to HmmClientConfig for backward compatibility
    pub fn to_client_config(&self) -> Result<crate::hmm_client::HmmClientConfig> {
        Ok(crate::hmm_client::HmmClientConfig {
            base_url: Url::parse(&self.service.url)
                .context("Invalid service URL")?,
            timeout: Duration::from_millis(self.service.timeout_ms),
            retry_attempts: self.service.retry_attempts,
            retry_delay: Duration::from_millis(self.service.retry_delay_ms),
            enable_fallback: self.fallback.enabled,
            fallback_weights: FusionWeights {
                w_ldc: self.fallback.w_ldc,
                w_mr: self.fallback.w_mr,
                w_tsmom: self.fallback.w_tsmom,
            },
            circuit_breaker_threshold: self.circuit_breaker.threshold,
            circuit_breaker_timeout: Duration::from_secs(self.circuit_breaker.timeout_sec),
        })
    }
    
    /// Save configuration to TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let toml_string = toml::to_string_pretty(self)
            .context("Failed to serialize configuration to TOML")?;
        
        fs::write(path.as_ref(), toml_string)
            .context("Failed to write configuration file")?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[test]
    fn test_default_config() {
        let config = HmmIntegrationConfig::default();
        
        assert_eq!(config.service.url, "http://localhost:8000");
        assert_eq!(config.service.timeout_ms, 5000);
        assert_eq!(config.service.retry_attempts, 3);
        assert_eq!(config.circuit_breaker.threshold, 5);
        assert_eq!(config.cache.ttl_sec, 60);
        assert!(config.fallback.enabled);
        assert_eq!(config.signal_fusion.threshold, 0.3);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = HmmIntegrationConfig::default();
        
        // Valid configuration should pass
        assert!(config.validate().is_ok());
        
        // Invalid URL
        config.service.url = "not a url".to_string();
        assert!(config.validate().is_err());
        config.service.url = "http://localhost:8000".to_string();
        
        // Invalid timeout
        config.service.timeout_ms = 0;
        assert!(config.validate().is_err());
        config.service.timeout_ms = 5000;
        
        // Invalid circuit breaker threshold
        config.circuit_breaker.threshold = 0;
        assert!(config.validate().is_err());
        config.circuit_breaker.threshold = 5;
        
        // Invalid fallback weights
        config.fallback.w_ldc = 2.0;
        assert!(config.validate().is_err());
        config.fallback.w_ldc = 0.33;
        
        // Invalid signal threshold
        config.signal_fusion.threshold = 1.5;
        assert!(config.validate().is_err());
        config.signal_fusion.threshold = 0.3;
        
        // Should be valid again
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_from_env() {
        // Set environment variables
        env::set_var("HMM_SERVICE_URL", "http://test:9000");
        env::set_var("HMM_SERVICE_TIMEOUT_MS", "3000");
        env::set_var("HMM_CIRCUIT_BREAKER_THRESHOLD", "3");
        env::set_var("HMM_CACHE_TTL_SEC", "120");
        env::set_var("HMM_FALLBACK_W_LDC", "0.4");
        env::set_var("SIGNAL_FUSION_THRESHOLD", "0.5");
        
        let config = HmmIntegrationConfig::from_env().unwrap();
        
        assert_eq!(config.service.url, "http://test:9000");
        assert_eq!(config.service.timeout_ms, 3000);
        assert_eq!(config.circuit_breaker.threshold, 3);
        assert_eq!(config.cache.ttl_sec, 120);
        assert_eq!(config.fallback.w_ldc, 0.4);
        assert_eq!(config.signal_fusion.threshold, 0.5);
        
        // Clean up
        env::remove_var("HMM_SERVICE_URL");
        env::remove_var("HMM_SERVICE_TIMEOUT_MS");
        env::remove_var("HMM_CIRCUIT_BREAKER_THRESHOLD");
        env::remove_var("HMM_CACHE_TTL_SEC");
        env::remove_var("HMM_FALLBACK_W_LDC");
        env::remove_var("SIGNAL_FUSION_THRESHOLD");
    }
    
    #[test]
    fn test_to_client_config() {
        let config = HmmIntegrationConfig::default();
        let client_config = config.to_client_config().unwrap();
        
        assert_eq!(client_config.base_url.as_str(), "http://localhost:8000/");
        assert_eq!(client_config.timeout, Duration::from_millis(5000));
        assert_eq!(client_config.retry_attempts, 3);
        assert_eq!(client_config.circuit_breaker_threshold, 5);
        assert!(client_config.enable_fallback);
    }
    
    #[test]
    fn test_toml_serialization() {
        let config = HmmIntegrationConfig::default();
        let toml_string = toml::to_string(&config).unwrap();
        
        // Verify it contains expected sections
        assert!(toml_string.contains("[service]"));
        assert!(toml_string.contains("[circuit_breaker]"));
        assert!(toml_string.contains("[cache]"));
        assert!(toml_string.contains("[fallback]"));
        assert!(toml_string.contains("[signal_fusion]"));
    }
    
    #[test]
    fn test_toml_deserialization() {
        let toml_str = r#"
            [service]
            url = "http://production:8000"
            timeout_ms = 3000
            retry_attempts = 5
            retry_delay_ms = 200
            
            [circuit_breaker]
            threshold = 3
            timeout_sec = 60
            
            [cache]
            ttl_sec = 120
            max_size = 2000
            
            [fallback]
            enabled = true
            w_ldc = 0.4
            w_mr = 0.3
            w_tsmom = 0.3
            
            [signal_fusion]
            threshold = 0.4
            cooldown_sec = 30
        "#;
        
        let config: HmmIntegrationConfig = toml::from_str(toml_str).unwrap();
        
        assert_eq!(config.service.url, "http://production:8000");
        assert_eq!(config.service.timeout_ms, 3000);
        assert_eq!(config.circuit_breaker.threshold, 3);
        assert_eq!(config.cache.ttl_sec, 120);
        assert_eq!(config.fallback.w_ldc, 0.4);
        assert_eq!(config.signal_fusion.threshold, 0.4);
    }
}
