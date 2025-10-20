//! Configuration Management Demo
//!
//! This example demonstrates the various ways to configure the HMM integration:
//! 1. Using default configuration
//! 2. Loading from TOML file
//! 3. Loading from environment variables
//! 4. Combining file and environment variables

use anyhow::Result;
use signal_fusion::config::HmmIntegrationConfig;
use std::env;

fn main() -> Result<()> {
    println!("=== HMM Integration Configuration Demo ===\n");
    
    // 1. Default Configuration
    println!("1. Default Configuration:");
    println!("{}", "=".repeat(50));
    let default_config = HmmIntegrationConfig::default();
    println!("Service URL: {}", default_config.service.url);
    println!("Timeout: {}ms", default_config.service.timeout_ms);
    println!("Circuit Breaker Threshold: {}", default_config.circuit_breaker.threshold);
    println!("Cache TTL: {}s", default_config.cache.ttl_sec);
    println!("Fallback Enabled: {}", default_config.fallback.enabled);
    println!("Signal Threshold: {}\n", default_config.signal_fusion.threshold);
    
    // Validate default configuration
    match default_config.validate() {
        Ok(_) => println!("✓ Default configuration is valid\n"),
        Err(e) => println!("✗ Default configuration is invalid: {}\n", e),
    }
    
    // 2. Environment Variable Configuration
    println!("2. Environment Variable Configuration:");
    println!("{}", "=".repeat(50));
    
    // Set some example environment variables
    env::set_var("HMM_SERVICE_URL", "http://production:8000");
    env::set_var("HMM_SERVICE_TIMEOUT_MS", "3000");
    env::set_var("HMM_CIRCUIT_BREAKER_THRESHOLD", "3");
    env::set_var("HMM_CACHE_TTL_SEC", "120");
    env::set_var("HMM_FALLBACK_W_LDC", "0.4");
    env::set_var("HMM_FALLBACK_W_MR", "0.3");
    env::set_var("HMM_FALLBACK_W_TSMOM", "0.3");
    env::set_var("SIGNAL_FUSION_THRESHOLD", "0.5");
    
    let env_config = HmmIntegrationConfig::from_env()?;
    println!("Service URL: {}", env_config.service.url);
    println!("Timeout: {}ms", env_config.service.timeout_ms);
    println!("Circuit Breaker Threshold: {}", env_config.circuit_breaker.threshold);
    println!("Cache TTL: {}s", env_config.cache.ttl_sec);
    println!("Fallback Weights: LDC={}, MR={}, TSMOM={}", 
        env_config.fallback.w_ldc,
        env_config.fallback.w_mr,
        env_config.fallback.w_tsmom
    );
    println!("Signal Threshold: {}\n", env_config.signal_fusion.threshold);
    
    // Clean up environment variables
    env::remove_var("HMM_SERVICE_URL");
    env::remove_var("HMM_SERVICE_TIMEOUT_MS");
    env::remove_var("HMM_CIRCUIT_BREAKER_THRESHOLD");
    env::remove_var("HMM_CACHE_TTL_SEC");
    env::remove_var("HMM_FALLBACK_W_LDC");
    env::remove_var("HMM_FALLBACK_W_MR");
    env::remove_var("HMM_FALLBACK_W_TSMOM");
    env::remove_var("SIGNAL_FUSION_THRESHOLD");
    
    // 3. TOML File Configuration
    println!("3. TOML File Configuration:");
    println!("{}", "=".repeat(50));
    
    // Try to load from example file (check multiple locations)
    let file_paths = vec![
        "hmm_integration.toml",
        "rust/signal-fusion/hmm_integration.toml",
        "hmm_integration.example.toml",
    ];
    
    let mut loaded = false;
    for path in file_paths {
        match HmmIntegrationConfig::from_file(path) {
            Ok(file_config) => {
                loaded = true;
                println!("✓ Successfully loaded configuration from: {}", path);
                println!("Service URL: {}", file_config.service.url);
                println!("Timeout: {}ms", file_config.service.timeout_ms);
                println!("Circuit Breaker Threshold: {}", file_config.circuit_breaker.threshold);
                println!("Cache TTL: {}s", file_config.cache.ttl_sec);
                println!("Fallback Enabled: {}", file_config.fallback.enabled);
                println!("Signal Threshold: {}\n", file_config.signal_fusion.threshold);
                break;
            }
            Err(_) => continue,
        }
    }
    
    if !loaded {
        println!("✗ Could not load configuration file from any location");
        println!("  (This is expected if no configuration file exists)\n");
    }
    
    // 4. Convert to Client Configuration
    println!("4. Converting to HmmClientConfig:");
    println!("{}", "=".repeat(50));
    let config = HmmIntegrationConfig::default();
    let client_config = config.to_client_config()?;
    println!("✓ Successfully converted to HmmClientConfig");
    println!("Base URL: {}", client_config.base_url);
    println!("Timeout: {:?}", client_config.timeout);
    println!("Retry Attempts: {}", client_config.retry_attempts);
    println!("Circuit Breaker Threshold: {}", client_config.circuit_breaker_threshold);
    println!("Fallback Enabled: {}\n", client_config.enable_fallback);
    
    // 5. Save Configuration to File
    println!("5. Saving Configuration to File:");
    println!("{}", "=".repeat(50));
    let config = HmmIntegrationConfig::default();
    match config.save_to_file("hmm_integration_generated.toml") {
        Ok(_) => {
            println!("✓ Configuration saved to hmm_integration_generated.toml");
            println!("  You can edit this file and load it later\n");
        }
        Err(e) => {
            println!("✗ Failed to save configuration: {}\n", e);
        }
    }
    
    // 6. Configuration Validation Examples
    println!("6. Configuration Validation:");
    println!("{}", "=".repeat(50));
    
    // Test invalid URL
    let mut invalid_config = HmmIntegrationConfig::default();
    invalid_config.service.url = "not a valid url".to_string();
    match invalid_config.validate() {
        Ok(_) => println!("✗ Should have failed validation"),
        Err(e) => println!("✓ Correctly rejected invalid URL: {}", e),
    }
    
    // Test invalid timeout
    let mut invalid_config = HmmIntegrationConfig::default();
    invalid_config.service.timeout_ms = 0;
    match invalid_config.validate() {
        Ok(_) => println!("✗ Should have failed validation"),
        Err(e) => println!("✓ Correctly rejected invalid timeout: {}", e),
    }
    
    // Test invalid fallback weight
    let mut invalid_config = HmmIntegrationConfig::default();
    invalid_config.fallback.w_ldc = 2.0;
    match invalid_config.validate() {
        Ok(_) => println!("✗ Should have failed validation"),
        Err(e) => println!("✓ Correctly rejected invalid weight: {}", e),
    }
    
    println!("\n=== Demo Complete ===");
    
    Ok(())
}
