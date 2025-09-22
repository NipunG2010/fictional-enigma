// Configuration management module

pub mod settings;
#[cfg(test)]
mod tests;

pub use settings::*;

use anyhow::{Context, Result};
use chrono::Utc;
use std::fs;
use std::path::{Path, PathBuf};

pub struct ConfigManager {
    config_dir: PathBuf,
}

impl ConfigManager {
    /// Create a new ConfigManager with the specified config directory
    pub fn new<P: AsRef<Path>>(config_dir: P) -> Result<Self> {
        let config_dir = config_dir.as_ref().to_path_buf();
        
        // Create config directory if it doesn't exist
        if !config_dir.exists() {
            fs::create_dir_all(&config_dir)
                .with_context(|| format!("Failed to create config directory: {:?}", config_dir))?;
        }

        Ok(Self { config_dir })
    }

    /// Create a ConfigManager using the default config directory
    pub fn default() -> Result<Self> {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("training-data-cli");
        
        Self::new(config_dir)
    }

    /// Save a configuration with the given name
    pub fn save_config(&self, name: &str, config: &SnapshotConfig) -> Result<()> {
        self.save_config_with_description(name, config, None)
    }

    /// Save a configuration with the given name and description
    pub fn save_config_with_description(
        &self,
        name: &str,
        config: &SnapshotConfig,
        description: Option<String>,
    ) -> Result<()> {
        // Validate the configuration before saving
        config.validate().map_err(|errors| {
            anyhow::anyhow!("Configuration validation failed: {}", errors.join(", "))
        })?;

        let saved_config = SavedConfig {
            name: name.to_string(),
            config: config.clone(),
            created_at: Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description,
        };

        let config_path = self.config_path(name);
        let config_json = serde_json::to_string_pretty(&saved_config)
            .context("Failed to serialize configuration")?;

        fs::write(&config_path, config_json)
            .with_context(|| format!("Failed to write config file: {:?}", config_path))?;

        Ok(())
    }

    /// Load a configuration by name
    pub fn load_config(&self, name: &str) -> Result<SnapshotConfig> {
        let config_path = self.config_path(name);
        
        if !config_path.exists() {
            return Err(anyhow::anyhow!("Configuration '{}' not found", name));
        }

        let config_json = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

        let saved_config: SavedConfig = serde_json::from_str(&config_json)
            .with_context(|| format!("Failed to parse config file: {:?}", config_path))?;

        // Validate the loaded configuration
        saved_config.config.validate().map_err(|errors| {
            anyhow::anyhow!("Loaded configuration is invalid: {}", errors.join(", "))
        })?;

        Ok(saved_config.config)
    }

    /// List all available configurations
    pub fn list_configs(&self) -> Result<Vec<ConfigInfo>> {
        let mut configs = Vec::new();

        if !self.config_dir.exists() {
            return Ok(configs);
        }

        let entries = fs::read_dir(&self.config_dir)
            .with_context(|| format!("Failed to read config directory: {:?}", self.config_dir))?;

        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    match self.load_config_info(name) {
                        Ok(info) => configs.push(info),
                        Err(_) => {
                            // Skip invalid config files
                            continue;
                        }
                    }
                }
            }
        }

        // Sort by creation date (newest first)
        configs.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(configs)
    }

    /// Delete a configuration by name
    pub fn delete_config(&self, name: &str) -> Result<()> {
        let config_path = self.config_path(name);
        
        if !config_path.exists() {
            return Err(anyhow::anyhow!("Configuration '{}' not found", name));
        }

        fs::remove_file(&config_path)
            .with_context(|| format!("Failed to delete config file: {:?}", config_path))?;

        Ok(())
    }

    /// Check if a configuration exists
    pub fn config_exists(&self, name: &str) -> bool {
        self.config_path(name).exists()
    }

    /// Get the path for a configuration file
    fn config_path(&self, name: &str) -> PathBuf {
        self.config_dir.join(format!("{}.json", name))
    }

    /// Load configuration info without full validation
    fn load_config_info(&self, name: &str) -> Result<ConfigInfo> {
        let config_path = self.config_path(name);
        let config_json = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

        let saved_config: SavedConfig = serde_json::from_str(&config_json)
            .with_context(|| format!("Failed to parse config file: {:?}", config_path))?;

        Ok(ConfigInfo {
            name: saved_config.name,
            created_at: saved_config.created_at,
            version: saved_config.version,
            description: saved_config.description,
        })
    }
}