//! End-to-End Testing Framework for IMP Trading System
//! 
//! This crate provides comprehensive integration testing capabilities for the complete
//! signal generation pipeline, including failure scenarios and performance validation.

pub mod config;
pub mod data_generator;
pub mod harness;
pub mod performance;
pub mod reporting;
pub mod validation;

pub use config::*;
pub use data_generator::*;
pub use harness::*;
pub use performance::*;
pub use reporting::*;
pub use validation::*;

/// Re-export commonly used types
pub use anyhow::{Context, Result};
pub use chrono::{DateTime, Duration, Utc};
pub use serde::{Deserialize, Serialize};
pub use std::collections::HashMap;
pub use tokio::time::Instant;
pub use uuid::Uuid;

/// Test framework version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default test configuration file name
pub const DEFAULT_CONFIG_FILE: &str = "test_config.toml";

/// Test result status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Timeout,
}

/// Test framework error types
#[derive(Debug, thiserror::Error)]
pub enum TestFrameworkError {
    #[error("Test setup failed: {0}")]
    SetupError(String),
    
    #[error("Test data generation failed: {0}")]
    DataGenerationError(String),
    
    #[error("System under test initialization failed: {0}")]
    SystemInitError(String),
    
    #[error("Test execution timeout after {timeout_ms}ms")]
    ExecutionTimeout { timeout_ms: u64 },
    
    #[error("Validation failed: {0}")]
    ValidationError(String),
    
    #[error("Performance requirement not met: {requirement}")]
    PerformanceError { requirement: String },
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}