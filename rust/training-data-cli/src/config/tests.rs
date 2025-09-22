#[cfg(test)]
mod tests {
    use crate::config::{
        ConfigManager, SnapshotConfig, SavedConfig,
        FeatureType, LabelThresholds, ValidationLevel, DateRange, OutputFormat
    };
    use chrono::{TimeZone, Utc};
    use std::fs;
    use tempfile::TempDir;

    fn create_test_config() -> SnapshotConfig {
        SnapshotConfig {
            horizon: 24,
            features: vec![
                FeatureType::Rsi14,
                FeatureType::Sma20,
                FeatureType::Ema12,
            ],
            label_thresholds: LabelThresholds {
                buy_threshold: 0.03,
                sell_threshold: -0.03,
            },
            validation_strictness: ValidationLevel::Strict,
            date_range: Some(DateRange {
                start: Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
                end: Utc.with_ymd_and_hms(2023, 12, 31, 23, 59, 59).unwrap(),
            }),
            output_format: OutputFormat::Csv,
        }
    }

    fn create_invalid_config() -> SnapshotConfig {
        SnapshotConfig {
            horizon: 0, // Invalid: must be > 0
            features: vec![], // Invalid: must have at least one feature
            label_thresholds: LabelThresholds {
                buy_threshold: -0.01, // Invalid: must be positive
                sell_threshold: 0.01,  // Invalid: must be negative
            },
            validation_strictness: ValidationLevel::Normal,
            date_range: Some(DateRange {
                start: Utc.with_ymd_and_hms(2023, 12, 31, 0, 0, 0).unwrap(),
                end: Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(), // Invalid: start > end
            }),
            output_format: OutputFormat::Parquet,
        }
    }

    #[test]
    fn test_snapshot_config_default() {
        let config = SnapshotConfig::default();
        
        assert_eq!(config.horizon, 12);
        assert!(!config.features.is_empty());
        assert_eq!(config.label_thresholds.buy_threshold, 0.02);
        assert_eq!(config.label_thresholds.sell_threshold, -0.02);
        assert_eq!(config.validation_strictness, ValidationLevel::Normal);
        assert!(config.date_range.is_none());
        assert_eq!(config.output_format, OutputFormat::Parquet);
    }

    #[test]
    fn test_snapshot_config_validation_valid() {
        let config = create_test_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_snapshot_config_validation_invalid() {
        let config = create_invalid_config();
        let result = config.validate();
        
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.len() >= 5); // Should have multiple validation errors
        
        // Check specific error messages
        assert!(errors.iter().any(|e| e.contains("Horizon must be greater than 0")));
        assert!(errors.iter().any(|e| e.contains("At least one feature must be specified")));
        assert!(errors.iter().any(|e| e.contains("Buy threshold must be positive")));
        assert!(errors.iter().any(|e| e.contains("Sell threshold must be negative")));
        assert!(errors.iter().any(|e| e.contains("Start date must be before end date")));
    }

    #[test]
    fn test_label_thresholds_validation() {
        let mut config = create_test_config();
        
        // Test invalid thresholds where buy <= sell
        config.label_thresholds = LabelThresholds {
            buy_threshold: 0.02,
            sell_threshold: 0.02, // Same as buy threshold
        };
        assert!(config.validate().is_err());
        
        config.label_thresholds = LabelThresholds {
            buy_threshold: 0.01,
            sell_threshold: 0.02, // Greater than buy threshold
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = create_test_config();
        
        // Test serialization
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.is_empty());
        
        // Test deserialization
        let deserialized: SnapshotConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_saved_config_serialization() {
        let config = create_test_config();
        let saved_config = SavedConfig {
            name: "test_config".to_string(),
            config: config.clone(),
            created_at: Utc::now(),
            version: "1.0.0".to_string(),
            description: Some("Test configuration".to_string()),
        };
        
        // Test serialization
        let json = serde_json::to_string_pretty(&saved_config).unwrap();
        assert!(!json.is_empty());
        
        // Test deserialization
        let deserialized: SavedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(saved_config.name, deserialized.name);
        assert_eq!(saved_config.config, deserialized.config);
        assert_eq!(saved_config.version, deserialized.version);
        assert_eq!(saved_config.description, deserialized.description);
    }

    #[test]
    fn test_config_manager_new() {
        let temp_dir = TempDir::new().unwrap();
        let config_dir = temp_dir.path().join("configs");
        
        let _manager = ConfigManager::new(&config_dir).unwrap();
        assert!(config_dir.exists());
    }

    #[test]
    fn test_config_manager_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ConfigManager::new(temp_dir.path()).unwrap();
        let config = create_test_config();
        
        // Save config
        manager.save_config("test", &config).unwrap();
        
        // Load config
        let loaded_config = manager.load_config("test").unwrap();
        assert_eq!(config, loaded_config);
    }

    #[test]
    fn test_config_manager_save_with_description() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ConfigManager::new(temp_dir.path()).unwrap();
        let config = create_test_config();
        let description = "Test configuration with description".to_string();
        
        // Save config with description
        manager.save_config_with_description("test", &config, Some(description.clone())).unwrap();
        
        // Load and verify
        let loaded_config = manager.load_config("test").unwrap();
        assert_eq!(config, loaded_config);
        
        // Check that the description was saved by loading the raw file
        let config_path = temp_dir.path().join("test.json");
        let json_content = fs::read_to_string(config_path).unwrap();
        let saved_config: SavedConfig = serde_json::from_str(&json_content).unwrap();
        assert_eq!(saved_config.description, Some(description));
    }

    #[test]
    fn test_config_manager_save_invalid_config() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ConfigManager::new(temp_dir.path()).unwrap();
        let invalid_config = create_invalid_config();
        
        // Should fail to save invalid config
        let result = manager.save_config("invalid", &invalid_config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Configuration validation failed"));
    }

    #[test]
    fn test_config_manager_load_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ConfigManager::new(temp_dir.path()).unwrap();
        
        // Should fail to load non-existent config
        let result = manager.load_config("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_config_manager_list_configs() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ConfigManager::new(temp_dir.path()).unwrap();
        let config1 = create_test_config();
        let mut config2 = create_test_config();
        config2.horizon = 48;
        
        // Initially empty
        let configs = manager.list_configs().unwrap();
        assert!(configs.is_empty());
        
        // Save two configs
        manager.save_config("config1", &config1).unwrap();
        manager.save_config("config2", &config2).unwrap();
        
        // List should return both
        let configs = manager.list_configs().unwrap();
        assert_eq!(configs.len(), 2);
        
        let names: Vec<&str> = configs.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"config1"));
        assert!(names.contains(&"config2"));
    }

    #[test]
    fn test_config_manager_delete_config() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ConfigManager::new(temp_dir.path()).unwrap();
        let config = create_test_config();
        
        // Save config
        manager.save_config("test", &config).unwrap();
        assert!(manager.config_exists("test"));
        
        // Delete config
        manager.delete_config("test").unwrap();
        assert!(!manager.config_exists("test"));
        
        // Should fail to delete again
        let result = manager.delete_config("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_manager_config_exists() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ConfigManager::new(temp_dir.path()).unwrap();
        let config = create_test_config();
        
        // Initially doesn't exist
        assert!(!manager.config_exists("test"));
        
        // Save and check existence
        manager.save_config("test", &config).unwrap();
        assert!(manager.config_exists("test"));
    }

    #[test]
    fn test_feature_type_serialization() {
        let features = vec![
            FeatureType::Rsi14,
            FeatureType::Sma20,
            FeatureType::Ema12,
            FeatureType::Ema26,
            FeatureType::Macd,
            FeatureType::MacdSignal,
            FeatureType::BbUpper,
            FeatureType::BbMiddle,
            FeatureType::BbLower,
            FeatureType::Atr14,
            FeatureType::Cci14,
            FeatureType::Adx14,
        ];
        
        for feature in features {
            let json = serde_json::to_string(&feature).unwrap();
            let deserialized: FeatureType = serde_json::from_str(&json).unwrap();
            assert_eq!(feature, deserialized);
        }
    }

    #[test]
    fn test_validation_level_serialization() {
        let levels = vec![
            ValidationLevel::Strict,
            ValidationLevel::Normal,
            ValidationLevel::Lenient,
        ];
        
        for level in levels {
            let json = serde_json::to_string(&level).unwrap();
            let deserialized: ValidationLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(level, deserialized);
        }
    }

    #[test]
    fn test_output_format_serialization() {
        let formats = vec![
            OutputFormat::Parquet,
            OutputFormat::Csv,
            OutputFormat::Json,
        ];
        
        for format in formats {
            let json = serde_json::to_string(&format).unwrap();
            let deserialized: OutputFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(format, deserialized);
        }
    }

    #[test]
    fn test_date_range_serialization() {
        let date_range = DateRange {
            start: Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            end: Utc.with_ymd_and_hms(2023, 12, 31, 23, 59, 59).unwrap(),
        };
        
        let json = serde_json::to_string(&date_range).unwrap();
        let deserialized: DateRange = serde_json::from_str(&json).unwrap();
        assert_eq!(date_range, deserialized);
    }

    #[test]
    fn test_config_manager_load_corrupted_file() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ConfigManager::new(temp_dir.path()).unwrap();
        
        // Create a corrupted config file
        let config_path = temp_dir.path().join("corrupted.json");
        fs::write(&config_path, "{ invalid json }").unwrap();
        
        // Should fail to load corrupted config
        let result = manager.load_config("corrupted");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse"));
    }
}