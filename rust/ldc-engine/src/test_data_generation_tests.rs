#[cfg(test)]
mod test_data_generation_tests {
    use crate::test_data_generation::*;
    use feature_pipeline::OHLCV;
    use tempfile::TempDir;
    use std::collections::HashSet;

    #[test]
    fn test_default_config_creation() {
        let config = TestDataConfig::default();
        assert_eq!(config.sample_count, 1000);
        assert_eq!(config.base_price, 50000.0);
        assert_eq!(config.volatility, 0.02);
        assert_eq!(config.trend, 0.0);
        assert!(config.seed.is_some());
    }

    #[test]
    fn test_synthetic_data_generation_basic() {
        let mut generator = TestDataGenerator::default();
        let data = generator.create_synthetic_dataset().unwrap();
        
        // Check basic properties
        assert_eq!(data.len(), 1000);
        assert!(data.iter().all(|d| d.timestamp > 0));
        assert!(data.iter().all(|d| d.open > 0.0));
        assert!(data.iter().all(|d| d.high > 0.0));
        assert!(data.iter().all(|d| d.low > 0.0));
        assert!(data.iter().all(|d| d.close > 0.0));
        assert!(data.iter().all(|d| d.volume >= 0.0));
    }

    #[test]
    fn test_ohlc_relationships() {
        let mut generator = TestDataGenerator::default();
        let data = generator.create_synthetic_dataset().unwrap();
        
        for (i, ohlcv) in data.iter().enumerate() {
            // High should be >= max(open, close)
            assert!(
                ohlcv.high >= ohlcv.open.max(ohlcv.close),
                "Sample {}: High ({}) < max(Open ({}), Close ({}))",
                i, ohlcv.high, ohlcv.open, ohlcv.close
            );
            
            // Low should be <= min(open, close)
            assert!(
                ohlcv.low <= ohlcv.open.min(ohlcv.close),
                "Sample {}: Low ({}) > min(Open ({}), Close ({}))",
                i, ohlcv.low, ohlcv.open, ohlcv.close
            );
            
            // High should be >= Low
            assert!(
                ohlcv.high >= ohlcv.low,
                "Sample {}: High ({}) < Low ({})",
                i, ohlcv.high, ohlcv.low
            );
        }
    }

    #[test]
    fn test_timestamp_ordering() {
        let mut generator = TestDataGenerator::default();
        let data = generator.create_synthetic_dataset().unwrap();
        
        for i in 1..data.len() {
            assert!(
                data[i].timestamp > data[i-1].timestamp,
                "Timestamps not increasing at index {}: {} <= {}",
                i, data[i].timestamp, data[i-1].timestamp
            );
        }
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let config = TestDataConfig {
            sample_count: 100,
            seed: Some(12345),
            ..Default::default()
        };
        
        let mut generator1 = TestDataGenerator::new(config.clone());
        let mut generator2 = TestDataGenerator::new(config);
        
        let data1 = generator1.create_synthetic_dataset().unwrap();
        let data2 = generator2.create_synthetic_dataset().unwrap();
        
        assert_eq!(data1.len(), data2.len());
        
        for (i, (d1, d2)) in data1.iter().zip(data2.iter()).enumerate() {
            assert_eq!(d1.timestamp, d2.timestamp, "Timestamp mismatch at index {}", i);
            assert!((d1.close - d2.close).abs() < 1e-10, "Price mismatch at index {}", i);
        }
    }

    #[test]
    fn test_market_regimes() {
        let regimes = vec![
            MarketRegime::Normal,
            MarketRegime::Trending { direction: 1.0, strength: 0.5 },
            MarketRegime::Trending { direction: -1.0, strength: 0.3 },
            MarketRegime::Ranging { range_factor: 0.5 },
            MarketRegime::Crisis { volatility_multiplier: 2.0 },
        ];
        
        for regime in regimes {
            let config = TestDataConfig {
                sample_count: 200,
                market_regime: regime,
                seed: Some(42),
                ..Default::default()
            };
            
            let mut generator = TestDataGenerator::new(config);
            let data = generator.create_synthetic_dataset().unwrap();
            
            assert_eq!(data.len(), 200);
            assert!(data.iter().all(|d| d.close > 0.0));
        }
    }

    #[test]
    fn test_features_generation() {
        let mut generator = TestDataGenerator::default();
        let ohlcv_data = generator.create_synthetic_dataset().unwrap();
        let features = generator.generate_features_data(&ohlcv_data).unwrap();
        
        assert_eq!(features.len(), ohlcv_data.len());
        
        // Check that timestamps match
        for (ohlcv, feature) in ohlcv_data.iter().zip(features.iter()) {
            assert_eq!(ohlcv.timestamp, feature.timestamp);
        }
        
        // Check that later features have calculated values
        let later_features = &features[25..];
        let rsi_count = later_features.iter().filter(|f| f.rsi.is_some()).count();
        assert!(rsi_count > later_features.len() / 2, "Too few RSI values calculated");
        
        // Validate RSI ranges
        for feature in &features {
            if let Some(rsi) = feature.rsi {
                assert!(rsi >= 0.0 && rsi <= 100.0, "RSI out of range: {}", rsi);
            }
        }
    }

    #[test]
    fn test_edge_case_generation() {
        let mut generator = TestDataGenerator::default();
        let edge_cases = generator.generate_edge_case_data().unwrap();
        
        assert!(!edge_cases.is_empty());
        
        // Should include various edge cases
        let has_tiny_prices = edge_cases.iter().any(|d| d.close < 0.001);
        let has_large_prices = edge_cases.iter().any(|d| d.close > 100000.0);
        let has_zero_volume = edge_cases.iter().any(|d| d.volume == 0.0);
        let has_flat_prices = edge_cases.windows(2).any(|w| w[0].close == w[1].close);
        
        assert!(has_tiny_prices, "Should include tiny prices");
        assert!(has_large_prices, "Should include large prices");
        assert!(has_zero_volume, "Should include zero volume");
        assert!(has_flat_prices, "Should include flat price periods");
    }

    #[test]
    fn test_error_scenarios_generation() {
        let mut generator = TestDataGenerator::default();
        let scenarios = generator.generate_error_scenarios().unwrap();
        
        assert!(!scenarios.is_empty());
        
        let scenario_names: HashSet<String> = scenarios.iter().map(|s| s.name.clone()).collect();
        
        // Check that we have expected error scenarios
        assert!(scenario_names.contains("invalid_ohlc_high_low"));
        assert!(scenario_names.contains("negative_prices"));
        assert!(scenario_names.contains("invalid_timestamps"));
        
        // Each scenario should have expected error description
        for scenario in &scenarios {
            assert!(!scenario.description.is_empty());
            assert!(!scenario.data.is_empty());
        }
    }

    #[test]
    fn test_data_validation_valid_data() {
        let generator = TestDataGenerator::default();
        
        let valid_data = vec![
            OHLCV {
                timestamp: 1640995200,
                open: 50000.0,
                high: 50100.0,
                low: 49900.0,
                close: 50050.0,
                volume: 100000.0,
            },
            OHLCV {
                timestamp: 1640995500,
                open: 50050.0,
                high: 50150.0,
                low: 49950.0,
                close: 50100.0,
                volume: 120000.0,
            },
        ];
        
        let report = generator.validate_test_data(&valid_data).unwrap();
        assert!(report.is_valid(), "Valid data should pass validation");
        assert_eq!(report.errors.len(), 0);
        assert_eq!(report.total_samples, 2);
        assert!(report.quality_score() > 0.9);
    }

    #[test]
    fn test_data_validation_invalid_data() {
        let generator = TestDataGenerator::default();
        
        let invalid_data = vec![
            OHLCV {
                timestamp: 1640995200,
                open: 50000.0,
                high: 49000.0, // High < Low (invalid)
                low: 50000.0,
                close: 50050.0,
                volume: 100000.0,
            },
        ];
        
        let report = generator.validate_test_data(&invalid_data).unwrap();
        assert!(!report.is_valid(), "Invalid data should fail validation");
        assert!(!report.errors.is_empty());
        assert!(report.quality_score() < 0.5);
    }

    #[test]
    fn test_training_samples_creation() {
        let mut generator = TestDataGenerator::default();
        
        // Generate smaller dataset for testing
        generator.config.sample_count = 100;
        let ohlcv_data = generator.create_synthetic_dataset().unwrap();
        let features_data = generator.generate_features_data(&ohlcv_data).unwrap();
        
        let horizon = 4;
        let training_samples = generator.create_training_samples(&ohlcv_data, &features_data, horizon).unwrap();
        
        // Should have (total - horizon) samples
        assert_eq!(training_samples.len(), ohlcv_data.len() - horizon);
        
        // Check that all samples have valid features and labels
        for sample in &training_samples {
            assert!(sample.features.f1.is_finite());
            assert!(sample.features.f2.is_finite());
            assert!(sample.features.f3.is_finite());
            assert!(sample.features.f4.is_finite());
            assert!(sample.features.f5.is_finite());
            assert!(sample.timestamp > 0);
        }
        
        // Check label distribution (should have some variety)
        let labels: HashSet<_> = training_samples.iter().map(|s| s.label as i32).collect();
        assert!(labels.len() > 1, "Should have multiple label types");
    }

    #[test]
    fn test_data_quality_report() {
        let mut report = DataQualityReport::new();
        
        assert_eq!(report.total_samples, 0);
        assert!(report.errors.is_empty());
        assert!(report.warnings.is_empty());
        assert!(report.is_valid());
        assert_eq!(report.quality_score(), 0.0); // No samples
        
        report.total_samples = 100;
        report.add_error("Test error".to_string());
        report.add_warning("Test warning".to_string());
        
        assert!(!report.is_valid());
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.warnings.len(), 1);
        
        let score = report.quality_score();
        assert!(score < 1.0 && score > 0.0);
    }

    #[test]
    fn test_test_data_manager_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_data.json");
        
        let test_data = vec![
            OHLCV {
                timestamp: 1640995200,
                open: 50000.0,
                high: 50100.0,
                low: 49900.0,
                close: 50050.0,
                volume: 100000.0,
            },
        ];
        
        // Save data
        TestDataManager::save_dataset(&test_data, &file_path).unwrap();
        assert!(file_path.exists());
        
        // Load data
        let loaded_data = TestDataManager::load_dataset(&file_path).unwrap();
        assert_eq!(loaded_data.len(), test_data.len());
        assert_eq!(loaded_data[0].timestamp, test_data[0].timestamp);
        assert_eq!(loaded_data[0].close, test_data[0].close);
    }

    #[test]
    fn test_comprehensive_test_suite() {
        let temp_dir = TempDir::new().unwrap();
        let test_suite = TestDataManager::generate_test_suite(temp_dir.path()).unwrap();
        
        // Check directories were created
        assert!(test_suite.directories.contains_key("synthetic"));
        assert!(test_suite.directories.contains_key("edge_cases"));
        assert!(test_suite.directories.contains_key("error_scenarios"));
        
        // Check datasets were generated
        assert!(!test_suite.datasets.is_empty());
        assert!(test_suite.datasets.contains_key("small_1k"));
        assert!(test_suite.datasets.contains_key("medium_10k"));
        assert!(test_suite.datasets.contains_key("large_50k"));
        
        // Test loading datasets
        let small_dataset = test_suite.get_dataset("small_1k").unwrap();
        assert_eq!(small_dataset.len(), 1000);
        
        let medium_dataset = test_suite.get_dataset("medium_10k").unwrap();
        assert_eq!(medium_dataset.len(), 10000);
        
        let large_dataset = test_suite.get_dataset("large_50k").unwrap();
        assert_eq!(large_dataset.len(), 50000);
        
        // Test edge cases
        let edge_cases = test_suite.get_edge_cases().unwrap();
        assert!(!edge_cases.is_empty());
        
        // Test error scenarios
        let error_scenarios = test_suite.get_error_scenarios().unwrap();
        assert!(!error_scenarios.is_empty());
    }

    #[test]
    fn test_flash_crash_scenario() {
        let config = TestDataConfig {
            sample_count: 200,
            market_regime: MarketRegime::FlashCrash { 
                crash_magnitude: 0.2, 
                recovery_time: 50 
            },
            seed: Some(42),
            ..Default::default()
        };
        
        let mut generator = TestDataGenerator::new(config);
        let data = generator.create_synthetic_dataset().unwrap();
        
        assert_eq!(data.len(), 200);
        
        // Check that there's a significant price drop around sample 100
        if data.len() > 101 {
            let pre_crash_price = data[99].close;
            let crash_price = data[100].close;
            let price_drop = (pre_crash_price - crash_price) / pre_crash_price;
            
            // Should have a significant drop (though exact amount may vary due to other factors)
            assert!(price_drop > 0.05, "Flash crash should cause significant price drop");
        }
    }

    #[test]
    fn test_feature_series_conversion() {
        let features = feature_pipeline::Features {
            timestamp: 1640995200,
            rsi: Some(65.5),
            sma_20: Some(50000.0),
            ema_20: Some(50100.0),
            std_20: Some(500.0),
            zscore_20: Some(0.2),
            momentum: Some(0.001),
            wavetrend_1: Some(-10.5),
            wavetrend_2: Some(-8.2),
            cci: Some(25.0),
            adx: Some(35.0),
        };
        
        let generator = TestDataGenerator::default();
        let feature_series = generator.convert_features_to_series(&features).unwrap();
        
        assert_eq!(feature_series.f1, 65.5); // RSI
        assert_eq!(feature_series.f2, -10.5); // WT1
        assert_eq!(feature_series.f3, 25.0); // CCI
        assert_eq!(feature_series.f4, 35.0); // ADX
        assert_eq!(feature_series.f5, 0.001); // Momentum
    }

    #[test]
    fn test_label_determination() {
        let generator = TestDataGenerator::default();
        
        // Test different price changes
        assert_eq!(generator.determine_label(0.002), crate::Direction::Long); // +0.2% -> Long
        assert_eq!(generator.determine_label(-0.002), crate::Direction::Short); // -0.2% -> Short
        assert_eq!(generator.determine_label(0.0005), crate::Direction::Neutral); // +0.05% -> Neutral
        assert_eq!(generator.determine_label(-0.0005), crate::Direction::Neutral); // -0.05% -> Neutral
        assert_eq!(generator.determine_label(0.0), crate::Direction::Neutral); // 0% -> Neutral
    }
}