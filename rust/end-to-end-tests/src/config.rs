//! Test configuration management
//! 
//! Handles loading and validation of test configuration from TOML files,
//! environment variables, and command-line arguments.

use crate::{Result, TestFrameworkError};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main test configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    /// Pipeline integration test configuration
    pub pipeline_tests: PipelineTestConfig,
    
    /// Failure scenario test configuration
    pub failure_tests: FailureTestConfig,
    
    /// Performance validation test configuration
    pub performance_tests: PerformanceTestConfig,
    
    /// Test data generation configuration
    pub data_generation: DataGenConfig,
    
    /// Result validation configuration
    pub validation: ValidationConfig,
    
    /// Test execution configuration
    pub execution: ExecutionConfig,
}

/// Pipeline integration test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTestConfig {
    /// Symbols to test with
    pub test_symbols: Vec<String>,
    
    /// Duration of test data in hours
    pub test_duration_hours: u32,
    
    /// Data interval (e.g., "5m", "1h")
    pub data_interval: String,
    
    /// Include edge case scenarios
    pub include_edge_cases: bool,
    
    /// Validate against reference data
    pub validate_against_reference: bool,
    
    /// Reference data directory
    pub reference_data_dir: Option<String>,
}

/// Failure scenario test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureTestConfig {
    /// Test HMM service failures
    pub test_hmm_failures: bool,
    
    /// Test Redis connection failures
    pub test_redis_failures: bool,
    
    /// Test Kafka connection failures
    pub test_kafka_failures: bool,
    
    /// Test data corruption scenarios
    pub test_data_corruption: bool,
    
    /// Duration of simulated failures in seconds
    pub failure_duration_seconds: u64,
    
    /// Timeout for recovery in seconds
    pub recovery_timeout_seconds: u64,
    
    /// Number of failure scenarios to test
    pub failure_scenario_count: u32,
}

/// Performance validation test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTestConfig {
    /// Maximum allowed end-to-end latency in milliseconds
    pub max_end_to_end_latency_ms: u64,
    
    /// Minimum required throughput in signals per second
    pub min_throughput_signals_per_second: f64,
    
    /// Maximum allowed memory usage in MB
    pub max_memory_usage_mb: u64,
    
    /// Number of concurrent symbols to test
    pub concurrent_symbols: u32,
    
    /// Duration of performance tests in minutes
    pub test_duration_minutes: u32,
    
    /// Performance tolerance percentage
    pub performance_tolerance: f64,
}

/// Test data generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataGenConfig {
    /// Market scenarios to generate
    pub market_scenarios: Vec<String>,
    
    /// Include price gaps in test data
    pub include_gaps: bool,
    
    /// Include outliers in test data
    pub include_outliers: bool,
    
    /// Base price for generated data
    pub base_price: f64,
    
    /// Volatility factor for generated data
    pub volatility_factor: f64,
    
    /// Random seed for reproducible data generation
    pub random_seed: Option<u64>,
    
    /// Output directory for generated test data
    pub output_dir: String,
}

/// Result validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Tolerance for feature value comparisons
    pub feature_tolerance: f64,
    
    /// Tolerance for signal value comparisons
    pub signal_tolerance: f64,
    
    /// Tolerance for performance metric comparisons
    pub performance_tolerance: f64,
    
    /// Enable strict validation mode
    pub strict_mode: bool,
    
    /// Validation timeout in seconds
    pub validation_timeout_seconds: u64,
}

/// Test execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Maximum number of parallel test executions
    pub max_parallel_tests: u32,
    
    /// Test execution timeout in seconds
    pub test_timeout_seconds: u64,
    
    /// Enable detailed logging
    pub verbose_logging: bool,
    
    /// Output directory for test results
    pub output_dir: String,
    
    /// Generate HTML reports
    pub generate_html_reports: bool,
    
    /// Generate JSON reports
    pub generate_json_reports: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            pipeline_tests: PipelineTestConfig::default(),
            failure_tests: FailureTestConfig::default(),
            performance_tests: PerformanceTestConfig::default(),
            data_generation: DataGenConfig::default(),
            validation: ValidationConfig::default(),
            execution: ExecutionConfig::default(),
        }
    }
}

impl Default for PipelineTestConfig {
    fn default() -> Self {
        Self {
            test_symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            test_duration_hours: 24,
            data_interval: "5m".to_string(),
            include_edge_cases: true,
            validate_against_reference: true,
            reference_data_dir: Some("test_data/reference".to_string()),
        }
    }
}

impl Default for FailureTestConfig {
    fn default() -> Self {
        Self {
            test_hmm_failures: true,
            test_redis_failures: true,
            test_kafka_failures: true,
            test_data_corruption: true,
            failure_duration_seconds: 30,
            recovery_timeout_seconds: 60,
            failure_scenario_count: 5,
        }
    }
}

impl Default for PerformanceTestConfig {
    fn default() -> Self {
        Self {
            max_end_to_end_latency_ms: 100,
            min_throughput_signals_per_second: 10.0,
            max_memory_usage_mb: 512,
            concurrent_symbols: 5,
            test_duration_minutes: 10,
            performance_tolerance: 0.1,
        }
    }
}

impl Default for DataGenConfig {
    fn default() -> Self {
        Self {
            market_scenarios: vec![
                "trending_up".to_string(),
                "trending_down".to_string(),
                "sideways".to_string(),
                "high_volatility".to_string(),
            ],
            include_gaps: true,
            include_outliers: true,
            base_price: 50000.0,
            volatility_factor: 0.02,
            random_seed: Some(42),
            output_dir: "test_data/generated".to_string(),
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            feature_tolerance: 0.001,
            signal_tolerance: 0.01,
            performance_tolerance: 0.1,
            strict_mode: false,
            validation_timeout_seconds: 30,
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_parallel_tests: 4,
            test_timeout_seconds: 300,
            verbose_logging: false,
            output_dir: "test_results".to_string(),
            generate_html_reports: true,
            generate_json_reports: true,
        }
    }
}

impl TestConfig {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| TestFrameworkError::ConfigError(format!("Failed to read config file: {}", e)))?;
        
        let config: TestConfig = toml::from_str(&content)
            .map_err(|e| TestFrameworkError::ConfigError(format!("Failed to parse config: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }
    
    /// Load configuration from environment variables with optional file override
    pub fn from_env_with_file<P: AsRef<Path>>(file_path: Option<P>) -> Result<Self> {
        let mut config = if let Some(path) = file_path {
            Self::from_file(path)?
        } else {
            Self::default()
        };
        
        // Override with environment variables
        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }
    
    /// Apply environment variable overrides
    fn apply_env_overrides(&mut self) -> Result<()> {
        // Pipeline test overrides
        if let Ok(symbols) = std::env::var("TEST_SYMBOLS") {
            self.pipeline_tests.test_symbols = symbols
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
        }
        
        if let Ok(duration) = std::env::var("TEST_DURATION_HOURS") {
            self.pipeline_tests.test_duration_hours = duration.parse()
                .map_err(|e| TestFrameworkError::ConfigError(format!("Invalid TEST_DURATION_HOURS: {}", e)))?;
        }
        
        // Performance test overrides
        if let Ok(latency) = std::env::var("MAX_LATENCY_MS") {
            self.performance_tests.max_end_to_end_latency_ms = latency.parse()
                .map_err(|e| TestFrameworkError::ConfigError(format!("Invalid MAX_LATENCY_MS: {}", e)))?;
        }
        
        if let Ok(throughput) = std::env::var("MIN_THROUGHPUT") {
            self.performance_tests.min_throughput_signals_per_second = throughput.parse()
                .map_err(|e| TestFrameworkError::ConfigError(format!("Invalid MIN_THROUGHPUT: {}", e)))?;
        }
        
        // Execution overrides
        if let Ok(parallel) = std::env::var("MAX_PARALLEL_TESTS") {
            self.execution.max_parallel_tests = parallel.parse()
                .map_err(|e| TestFrameworkError::ConfigError(format!("Invalid MAX_PARALLEL_TESTS: {}", e)))?;
        }
        
        if let Ok(verbose) = std::env::var("VERBOSE_LOGGING") {
            self.execution.verbose_logging = verbose.parse()
                .map_err(|e| TestFrameworkError::ConfigError(format!("Invalid VERBOSE_LOGGING: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Validate configuration values
    pub fn validate(&self) -> Result<()> {
        // Validate pipeline test config
        if self.pipeline_tests.test_symbols.is_empty() {
            return Err(TestFrameworkError::ConfigError("test_symbols cannot be empty".to_string()).into());
        }
        
        if self.pipeline_tests.test_duration_hours == 0 {
            return Err(TestFrameworkError::ConfigError("test_duration_hours must be greater than 0".to_string()).into());
        }
        
        // Validate performance config
        if self.performance_tests.max_end_to_end_latency_ms == 0 {
            return Err(TestFrameworkError::ConfigError("max_end_to_end_latency_ms must be greater than 0".to_string()).into());
        }
        
        if self.performance_tests.min_throughput_signals_per_second <= 0.0 {
            return Err(TestFrameworkError::ConfigError("min_throughput_signals_per_second must be greater than 0".to_string()).into());
        }
        
        if self.performance_tests.concurrent_symbols == 0 {
            return Err(TestFrameworkError::ConfigError("concurrent_symbols must be greater than 0".to_string()).into());
        }
        
        // Validate execution config
        if self.execution.max_parallel_tests == 0 {
            return Err(TestFrameworkError::ConfigError("max_parallel_tests must be greater than 0".to_string()).into());
        }
        
        if self.execution.test_timeout_seconds == 0 {
            return Err(TestFrameworkError::ConfigError("test_timeout_seconds must be greater than 0".to_string()).into());
        }
        
        // Validate tolerance values
        if self.validation.feature_tolerance < 0.0 || self.validation.feature_tolerance > 1.0 {
            return Err(TestFrameworkError::ConfigError("feature_tolerance must be between 0.0 and 1.0".to_string()).into());
        }
        
        if self.validation.signal_tolerance < 0.0 || self.validation.signal_tolerance > 1.0 {
            return Err(TestFrameworkError::ConfigError("signal_tolerance must be between 0.0 and 1.0".to_string()).into());
        }
        
        Ok(())
    }
    
    /// Save configuration to a TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| TestFrameworkError::ConfigError(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path.as_ref(), content)
            .map_err(|e| TestFrameworkError::ConfigError(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_default_config_validation() {
        let config = TestConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = TestConfig::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: TestConfig = toml::from_str(&serialized).unwrap();
        assert!(deserialized.validate().is_ok());
    }
    
    #[test]
    fn test_config_file_roundtrip() {
        let config = TestConfig::default();
        let temp_file = NamedTempFile::new().unwrap();
        
        config.save_to_file(temp_file.path()).unwrap();
        let loaded_config = TestConfig::from_file(temp_file.path()).unwrap();
        
        assert_eq!(config.pipeline_tests.test_symbols, loaded_config.pipeline_tests.test_symbols);
        assert_eq!(config.performance_tests.max_end_to_end_latency_ms, loaded_config.performance_tests.max_end_to_end_latency_ms);
    }
    
    #[test]
    fn test_invalid_config_validation() {
        let mut config = TestConfig::default();
        config.pipeline_tests.test_symbols.clear();
        assert!(config.validate().is_err());
        
        config = TestConfig::default();
        config.performance_tests.max_end_to_end_latency_ms = 0;
        assert!(config.validate().is_err());
        
        config = TestConfig::default();
        config.validation.feature_tolerance = 2.0;
        assert!(config.validate().is_err());
    }
}