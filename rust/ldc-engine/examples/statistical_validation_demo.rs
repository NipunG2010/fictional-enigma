use anyhow::Result;
use ldc_engine::{
    statistical_analysis::{StatisticalAnalyzer, StatisticalConfig, MultipleTestingCorrection},
    Direction, LDCPrediction, OHLCV,
};

fn main() -> Result<()> {
    println!("=== Statistical Validation Demo ===");
    
    // Create statistical analyzer with custom configuration
    let config = StatisticalConfig {
        confidence_level: 0.95,
        min_sample_size: 100,
        significance_threshold: 0.05,
    };
    let analyzer = StatisticalAnalyzer::with_config(config);
    
    // Generate sample data for demonstration
    let (predictions, outcomes, market_data) = generate_sample_data(200);
    
    println!("Generated {} predictions with corresponding outcomes", predictions.len());
    
    // Perform comprehensive statistical analysis
    let analysis = analyzer.analyze_predictions(&predictions, &outcomes, &market_data)?;
    
    // Display results
    println!("\n=== Prediction Accuracy Metrics ===");
    println!("Overall Hit Rate: {:.2}%", analysis.prediction_accuracy.hit_rate * 100.0);
    println!("Precision (Long): {:.2}%", analysis.prediction_accuracy.precision_long * 100.0);
    println!("Precision (Short): {:.2}%", analysis.prediction_accuracy.precision_short * 100.0);
    println!("Recall (Long): {:.2}%", analysis.prediction_accuracy.recall_long * 100.0);
    println!("Recall (Short): {:.2}%", analysis.prediction_accuracy.recall_short * 100.0);
    println!("F1 Score (Overall): {:.3}", analysis.prediction_accuracy.f1_score_overall);
    
    println!("\n=== Signal Quality Metrics ===");
    println!("Signal-to-Noise Ratio: {:.3}", analysis.signal_quality.signal_to_noise_ratio);
    println!("Information Coefficient: {:.3}", analysis.signal_quality.information_coefficient);
    println!("Signal Strength Mean: {:.3}", analysis.signal_quality.signal_strength_distribution.mean);
    println!("Signal Strength Std Dev: {:.3}", analysis.signal_quality.signal_strength_distribution.std_dev);
    
    println!("\n=== Market Regime Analysis ===");
    println!("Total Regimes Identified: {}", analysis.market_regime_analysis.regime_classification.len());
    
    // Count regime types
    let mut trending_count = 0;
    let mut ranging_count = 0;
    let mut volatile_count = 0;
    
    for regime in &analysis.market_regime_analysis.regime_classification {
        match regime.regime_type {
            ldc_engine::statistical_analysis::MarketRegimeType::Trending => trending_count += 1,
            ldc_engine::statistical_analysis::MarketRegimeType::Ranging => ranging_count += 1,
            ldc_engine::statistical_analysis::MarketRegimeType::Volatile => volatile_count += 1,
        }
    }
    
    println!("  - Trending Periods: {}", trending_count);
    println!("  - Ranging Periods: {}", ranging_count);
    println!("  - Volatile Periods: {}", volatile_count);
    
    println!("\nRegime-Specific Performance:");
    println!("  - Trending Hit Rate: {:.2}%", analysis.market_regime_analysis.trending_performance.hit_rate * 100.0);
    println!("  - Ranging Hit Rate: {:.2}%", analysis.market_regime_analysis.ranging_performance.hit_rate * 100.0);
    println!("  - Volatile Hit Rate: {:.2}%", analysis.market_regime_analysis.volatile_performance.hit_rate * 100.0);
    
    println!("\n=== Statistical Significance Testing ===");
    println!("Sample Size: {}", analysis.statistical_significance.sample_size);
    println!("Overall Accuracy P-value: {:.4}", analysis.statistical_significance.accuracy_p_value);
    println!("Signal Quality P-value: {:.4}", analysis.statistical_significance.signal_quality_p_value);
    println!("Statistical Power: {:.3}", analysis.statistical_significance.statistical_power);
    println!("Is Statistically Significant: {}", analysis.statistical_significance.is_significant);
    println!("Correction Method: {:?}", analysis.statistical_significance.correction_method);
    
    println!("\n=== Multiple Testing Correction Results ===");
    for (test_name, corrected_p_value) in &analysis.statistical_significance.corrected_p_values {
        let is_significant = *corrected_p_value < 0.05;
        println!("  - {}: p = {:.4} ({})", 
                test_name, 
                corrected_p_value,
                if is_significant { "Significant" } else { "Not Significant" });
    }
    
    println!("\n=== Regime-Specific P-values (Before Correction) ===");
    for (regime_type, p_value) in &analysis.statistical_significance.regime_p_values {
        println!("  - {:?} Regime: p = {:.4}", regime_type, p_value);
    }
    
    println!("\n=== Confidence Intervals ===");
    for (metric_name, ci) in &analysis.statistical_significance.confidence_intervals {
        println!("  - {} ({:.0}% CI): [{:.3}, {:.3}]", 
                metric_name, 
                ci.confidence_level * 100.0,
                ci.lower_bound, 
                ci.upper_bound);
    }
    
    // Demonstrate multiple testing correction methods
    println!("\n=== Multiple Testing Correction Comparison ===");
    let p_values = vec![0.01, 0.02, 0.03, 0.04, 0.05];
    println!("Original p-values: {:?}", p_values);
    
    let bonferroni = analyzer.bonferroni_correction(&p_values);
    println!("Bonferroni corrected: {:?}", bonferroni.iter().map(|p| format!("{:.3}", p)).collect::<Vec<_>>());
    
    let bh = analyzer.benjamini_hochberg_correction(&p_values);
    println!("Benjamini-Hochberg corrected: {:?}", bh.iter().map(|p| format!("{:.3}", p)).collect::<Vec<_>>());
    
    Ok(())
}

fn generate_sample_data(count: usize) -> (Vec<LDCPrediction>, Vec<Direction>, Vec<OHLCV>) {
    let mut predictions = Vec::new();
    let mut outcomes = Vec::new();
    let mut market_data = Vec::new();
    
    for i in 0..count + 1 {
        // Generate market data with different patterns for different regimes
        let price = if i < count / 3 {
            // Trending period
            100.0 + i as f64 * 0.5
        } else if i < 2 * count / 3 {
            // Volatile period
            130.0 + ((i as f64 * 0.3).sin() * 15.0)
        } else {
            // Ranging period
            160.0 + ((i as f64 * 0.1).sin() * 3.0)
        };
        
        market_data.push(OHLCV {
            timestamp: 1640995200 + i as i64 * 300, // 5-minute intervals
            open: price,
            high: price * 1.01,
            low: price * 0.99,
            close: price,
            volume: 1000.0,
        });
        
        if i < count {
            // Generate predictions with some correlation to market regime
            let (direction, signal, confidence) = if i < count / 3 {
                // Trending period - more long predictions
                if i % 4 == 0 {
                    (Direction::Short, -0.8, 0.7)
                } else {
                    (Direction::Long, 0.9, 0.8)
                }
            } else if i < 2 * count / 3 {
                // Volatile period - mixed predictions
                match i % 3 {
                    0 => (Direction::Long, 0.6, 0.6),
                    1 => (Direction::Short, -0.7, 0.65),
                    _ => (Direction::Neutral, 0.1, 0.5),
                }
            } else {
                // Ranging period - mostly neutral
                if i % 5 == 0 {
                    (Direction::Long, 0.4, 0.55)
                } else if i % 5 == 1 {
                    (Direction::Short, -0.4, 0.55)
                } else {
                    (Direction::Neutral, 0.0, 0.5)
                }
            };
            
            predictions.push(LDCPrediction {
                signal,
                confidence,
                k_nearest_distances: vec![0.5, 0.7, 0.9, 1.1, 1.3],
                k_nearest_labels: vec![direction, direction, Direction::Neutral, direction, direction],
                prediction_direction: direction,
            });
            
            // Generate outcomes with some correlation to predictions but with noise
            let outcome = if i % 7 == 0 {
                // Add some noise - incorrect predictions
                match direction {
                    Direction::Long => Direction::Short,
                    Direction::Short => Direction::Long,
                    Direction::Neutral => Direction::Neutral,
                }
            } else {
                // Mostly correct predictions
                direction
            };
            
            outcomes.push(outcome);
        }
    }
    
    (predictions, outcomes, market_data)
}