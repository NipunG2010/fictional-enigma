use anyhow::Result;
use ldc_engine::test_data_generation::{
    TestDataGenerator, TestDataConfig, MarketRegime, TestDataManager, DataQualityReport
};
use std::path::Path;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== LDC Engine Test Data Generation Demo ===\n");

    // Create temporary directory for demo
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    demo_synthetic_data_generation()?;
    demo_market_regimes()?;
    demo_edge_cases_and_validation()?;
    demo_features_generation()?;
    demo_training_samples_creation()?;
    demo_comprehensive_test_suite(base_path)?;
    
    println!("✅ All test data generation demos completed successfully!");
    Ok(())
}

fn demo_synthetic_data_generation() -> Result<()> {
    println!("🔄 Demo 1: Basic Synthetic Data Generation");
    
    // Create generator with default configuration
    let mut generator = TestDataGenerator::default();
    
    // Generate synthetic OHLCV data
    let ohlcv_data = generator.create_synthetic_dataset()?;
    
    println!("Generated {} OHLCV samples", ohlcv_data.len());
    println!("First sample: {:?}", ohlcv_data[0]);
    println!("Last sample: {:?}", ohlcv_data.last().unwrap());
    
    // Validate the generated data
    let quality_report = generator.validate_test_data(&ohlcv_data)?;
    println!("Data quality score: {:.2}%", quality_report.quality_score() * 100.0);
    println!("Validation errors: {}", quality_report.errors.len());
    
    if let Some(mean_price) = quality_report.mean_price {
        println!("Mean price: ${:.2}", mean_price);
    }
    
    if let Some(volatility) = quality_report.price_volatility {
        println!("Price volatility: {:.4}", volatility);
    }
    
    println!();
    Ok(())
}

fn demo_market_regimes() -> Result<()> {
    println!("🔄 Demo 2: Different Market Regimes");
    
    let regimes = vec![
        ("Normal Market", MarketRegime::Normal),
        ("Bull Trend", MarketRegime::Trending { direction: 1.0, strength: 0.5 }),
        ("Bear Trend", MarketRegime::Trending { direction: -1.0, strength: 0.3 }),
        ("Ranging Market", MarketRegime::Ranging { range_factor: 0.5 }),
        ("Crisis", MarketRegime::Crisis { volatility_multiplier: 3.0 }),
        ("Flash Crash", MarketRegime::FlashCrash { crash_magnitude: 0.2, recovery_time: 50 }),
    ];
    
    for (name, regime) in regimes {
        let config = TestDataConfig {
            sample_count: 200,
            market_regime: regime,
            seed: Some(42), // Fixed seed for reproducibility
            ..Default::default()
        };
        
        let mut generator = TestDataGenerator::new(config);
        let data = generator.create_synthetic_dataset()?;
        
        // Calculate price change statistics
        let price_changes: Vec<f64> = data.windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();
        
        let mean_change = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
        let volatility = {
            let variance = price_changes.iter()
                .map(|&x| (x - mean_change).powi(2))
                .sum::<f64>() / price_changes.len() as f64;
            variance.sqrt()
        };
        
        println!("{}: Mean return: {:.4}, Volatility: {:.4}", 
                 name, mean_change, volatility);
    }
    
    println!();
    Ok(())
}

fn demo_edge_cases_and_validation() -> Result<()> {
    println!("🔄 Demo 3: Edge Cases and Data Validation");
    
    let mut generator = TestDataGenerator::default();
    
    // Generate edge cases
    let edge_cases = generator.generate_edge_case_data()?;
    println!("Generated {} edge case samples", edge_cases.len());
    
    // Validate edge cases (some should have issues)
    let quality_report = generator.validate_test_data(&edge_cases)?;
    println!("Edge cases validation:");
    println!("  - Quality score: {:.2}%", quality_report.quality_score() * 100.0);
    println!("  - Errors found: {}", quality_report.errors.len());
    println!("  - Warnings: {}", quality_report.warnings.len());
    
    // Generate error scenarios
    let error_scenarios = generator.generate_error_scenarios()?;
    println!("\nGenerated {} error scenarios:", error_scenarios.len());
    
    for scenario in &error_scenarios {
        println!("  - {}: {}", scenario.name, scenario.description);
        
        // Validate each scenario (should fail)
        let report = generator.validate_test_data(&scenario.data)?;
        if !report.is_valid() {
            println!("    ✓ Correctly detected {} errors", report.errors.len());
        } else {
            println!("    ⚠ Unexpectedly passed validation");
        }
    }
    
    println!();
    Ok(())
}

fn demo_features_generation() -> Result<()> {
    println!("🔄 Demo 4: Technical Features Generation");
    
    let mut generator = TestDataGenerator::default();
    
    // Generate OHLCV data
    let ohlcv_data = generator.create_synthetic_dataset()?;
    
    // Generate features from OHLCV data
    let features = generator.generate_features_data(&ohlcv_data)?;
    
    println!("Generated {} feature samples", features.len());
    
    // Show some feature statistics
    let rsi_values: Vec<f64> = features.iter()
        .filter_map(|f| f.rsi)
        .collect();
    
    if !rsi_values.is_empty() {
        let mean_rsi = rsi_values.iter().sum::<f64>() / rsi_values.len() as f64;
        let min_rsi = rsi_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_rsi = rsi_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        println!("RSI Statistics:");
        println!("  - Mean: {:.2}", mean_rsi);
        println!("  - Range: {:.2} - {:.2}", min_rsi, max_rsi);
        println!("  - Valid samples: {}/{}", rsi_values.len(), features.len());
    }
    
    // Show feature availability by period
    let early_features = &features[0..20.min(features.len())];
    let late_features = &features[20.min(features.len())..];
    
    let early_rsi_count = early_features.iter().filter(|f| f.rsi.is_some()).count();
    let late_rsi_count = late_features.iter().filter(|f| f.rsi.is_some()).count();
    
    println!("Feature availability:");
    println!("  - Early periods (0-19): {}/{} have RSI", early_rsi_count, early_features.len());
    println!("  - Later periods (20+): {}/{} have RSI", late_rsi_count, late_features.len());
    
    println!();
    Ok(())
}

fn demo_training_samples_creation() -> Result<()> {
    println!("🔄 Demo 5: Training Samples Creation");
    
    let mut generator = TestDataGenerator::default();
    
    // Generate OHLCV and features data
    let ohlcv_data = generator.create_synthetic_dataset()?;
    let features_data = generator.generate_features_data(&ohlcv_data)?;
    
    // Create training samples with 4-bar horizon
    let training_samples = generator.create_training_samples(&ohlcv_data, &features_data, 4)?;
    
    println!("Created {} training samples", training_samples.len());
    
    // Analyze label distribution
    let long_count = training_samples.iter().filter(|s| matches!(s.label, ldc_engine::Direction::Long)).count();
    let short_count = training_samples.iter().filter(|s| matches!(s.label, ldc_engine::Direction::Short)).count();
    let neutral_count = training_samples.iter().filter(|s| matches!(s.label, ldc_engine::Direction::Neutral)).count();
    
    println!("Label distribution:");
    println!("  - Long: {} ({:.1}%)", long_count, long_count as f64 / training_samples.len() as f64 * 100.0);
    println!("  - Short: {} ({:.1}%)", short_count, short_count as f64 / training_samples.len() as f64 * 100.0);
    println!("  - Neutral: {} ({:.1}%)", neutral_count, neutral_count as f64 / training_samples.len() as f64 * 100.0);
    
    // Show sample feature ranges
    if let Some(sample) = training_samples.first() {
        println!("Sample features (first sample):");
        println!("  - f1 (RSI): {:.2}", sample.features.f1);
        println!("  - f2 (WT): {:.2}", sample.features.f2);
        println!("  - f3 (CCI): {:.2}", sample.features.f3);
        println!("  - f4 (ADX): {:.2}", sample.features.f4);
        println!("  - f5 (Momentum): {:.6}", sample.features.f5);
        println!("  - Label: {:?}", sample.label);
    }
    
    println!();
    Ok(())
}

fn demo_comprehensive_test_suite(base_path: &Path) -> Result<()> {
    println!("🔄 Demo 6: Comprehensive Test Suite Generation");
    
    // Generate complete test suite
    let test_suite = TestDataManager::generate_test_suite(base_path)?;
    
    println!("Generated test suite at: {}", test_suite.base_path.display());
    println!("Created directories:");
    for (name, path) in &test_suite.directories {
        println!("  - {}: {}", name, path.display());
    }
    
    println!("Generated datasets:");
    for (name, path) in &test_suite.datasets {
        println!("  - {}: {}", name, path.display());
        
        // Load and validate each dataset
        let data = test_suite.get_dataset(name)?;
        println!("    {} samples", data.len());
    }
    
    // Test loading edge cases
    let edge_cases = test_suite.get_edge_cases()?;
    println!("Edge cases: {} samples", edge_cases.len());
    
    // Test loading error scenarios
    let error_scenarios = test_suite.get_error_scenarios()?;
    println!("Error scenarios: {} scenarios", error_scenarios.len());
    
    // Demonstrate dataset usage
    println!("\nDataset usage example:");
    let small_dataset = test_suite.get_dataset("small_1k")?;
    let medium_dataset = test_suite.get_dataset("medium_10k")?;
    let large_dataset = test_suite.get_dataset("large_50k")?;
    
    println!("  - Small dataset: {} samples", small_dataset.len());
    println!("  - Medium dataset: {} samples", medium_dataset.len());
    println!("  - Large dataset: {} samples", large_dataset.len());
    
    // Performance comparison simulation
    println!("\nSimulated performance test results:");
    println!("  - Small (1k): ~0.3ms query time");
    println!("  - Medium (10k): ~0.8ms query time");
    println!("  - Large (50k): ~3.2ms query time");
    
    println!();
    Ok(())
}