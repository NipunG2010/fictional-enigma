use anyhow::Result;
use ldc_engine::{
    statistical_analysis::{StatisticalAnalyzer, StatisticalConfig},
    Direction, LDCPrediction, FeatureSeries,
};
use feature_pipeline::OHLCV;

fn main() -> Result<()> {
    println!("=== Statistical Analysis Framework Demo ===\n");

    // Create a statistical analyzer with custom configuration
    let config = StatisticalConfig {
        confidence_level: 0.95,
        min_sample_size: 50, // Lower for demo purposes
        significance_threshold: 0.05,
    };
    let analyzer = StatisticalAnalyzer::with_config(config);

    // Generate sample predictions and outcomes for demonstration
    let (predictions, actual_outcomes, market_data) = generate_sample_data(200);

    println!("Generated {} predictions for analysis", predictions.len());
    println!("Sample prediction: {:?}", predictions[0]);
    println!("Sample actual outcome: {:?}", actual_outcomes[0]);
    println!();

    // Perform comprehensive statistical analysis
    println!("Performing statistical analysis...");
    let analysis_result = analyzer.analyze_predictions(&predictions, &actual_outcomes, &market_data)?;

    // Display accuracy metrics
    println!("=== ACCURACY METRICS ===");
    let accuracy = &analysis_result.prediction_accuracy;
    println!("Overall Hit Rate: {:.2}%", accuracy.hit_rate * 100.0);
    println!("Precision (Long): {:.2}%", accuracy.precision_long * 100.0);
    println!("Precision (Short): {:.2}%", accuracy.precision_short * 100.0);
    println!("Precision (Overall): {:.2}%", accuracy.precision_overall * 100.0);
    println!("Recall (Long): {:.2}%", accuracy.recall_long * 100.0);
    println!("Recall (Short): {:.2}%", accuracy.recall_short * 100.0);
    println!("Recall (Overall): {:.2}%", accuracy.recall_overall * 100.0);
    println!("F1 Score (Long): {:.3}", accuracy.f1_score_long);
    println!("F1 Score (Short): {:.3}", accuracy.f1_score_short);
    println!("F1 Score (Overall): {:.3}", accuracy.f1_score_overall);
    println!();

    // Display confusion matrix
    println!("=== CONFUSION MATRIX ===");
    let cm = &accuracy.confusion_matrix;
    println!("True Positives (Long): {}", cm.true_positives_long);
    println!("False Positives (Long): {}", cm.false_positives_long);
    println!("True Negatives (Short): {}", cm.true_negatives_short);
    println!("False Negatives (Short): {}", cm.false_negatives_short);
    println!("True Neutral: {}", cm.true_neutral);
    println!("False Neutral: {}", cm.false_neutral);
    println!();

    // Display signal quality metrics
    println!("=== SIGNAL QUALITY METRICS ===");
    let signal_quality = &analysis_result.signal_quality;
    println!("Signal-to-Noise Ratio: {:.3}", signal_quality.signal_to_noise_ratio);
    println!("Information Coefficient: {:.3}", signal_quality.information_coefficient);
    println!();

    println!("Signal Strength Distribution:");
    let signal_dist = &signal_quality.signal_strength_distribution;
    println!("  Mean: {:.3}", signal_dist.mean);
    println!("  Std Dev: {:.3}", signal_dist.std_dev);
    println!("  Min: {:.3}", signal_dist.min);
    println!("  Max: {:.3}", signal_dist.max);
    println!("  25th Percentile: {:.3}", signal_dist.percentile_25);
    println!("  50th Percentile: {:.3}", signal_dist.percentile_50);
    println!("  75th Percentile: {:.3}", signal_dist.percentile_75);
    println!("  95th Percentile: {:.3}", signal_dist.percentile_95);
    println!();

    println!("Average Signal Strength by Direction:");
    for (direction, avg_signal) in &signal_quality.avg_signal_strength_by_direction {
        println!("  {:?}: {:.3}", direction, avg_signal);
    }
    println!();

    // Display market regime analysis
    println!("=== MARKET REGIME ANALYSIS ===");
    let regime_analysis = &analysis_result.market_regime_analysis;
    println!("Trending Market Performance:");
    println!("  Hit Rate: {:.2}%", regime_analysis.trending_performance.hit_rate * 100.0);
    println!("  F1 Score: {:.3}", regime_analysis.trending_performance.f1_score_overall);
    
    println!("Ranging Market Performance:");
    println!("  Hit Rate: {:.2}%", regime_analysis.ranging_performance.hit_rate * 100.0);
    println!("  F1 Score: {:.3}", regime_analysis.ranging_performance.f1_score_overall);
    
    println!("Volatile Market Performance:");
    println!("  Hit Rate: {:.2}%", regime_analysis.volatile_performance.hit_rate * 100.0);
    println!("  F1 Score: {:.3}", regime_analysis.volatile_performance.f1_score_overall);
    
    println!("Market Regimes Identified: {}", regime_analysis.regime_classification.len());
    println!();

    // Display statistical significance
    println!("=== STATISTICAL SIGNIFICANCE ===");
    let significance = &analysis_result.statistical_significance;
    println!("Sample Size: {}", significance.sample_size);
    println!("Accuracy P-Value: {:.6}", significance.accuracy_p_value);
    println!("Signal Quality P-Value: {:.6}", significance.signal_quality_p_value);
    println!("Is Statistically Significant: {}", significance.is_significant);
    println!("Statistical Power: {:.3}", significance.statistical_power);
    println!();

    println!("Confidence Intervals:");
    for (metric, ci) in &significance.confidence_intervals {
        println!("  {}: [{:.3}, {:.3}] ({}% confidence)", 
                metric, ci.lower_bound, ci.upper_bound, ci.confidence_level * 100.0);
    }
    println!();

    // Demonstrate individual method calls
    println!("=== INDIVIDUAL METHOD DEMONSTRATIONS ===");
    
    // Test accuracy metrics calculation
    println!("Testing accuracy metrics calculation...");
    let accuracy_metrics = analyzer.calculate_accuracy_metrics(&predictions[..50], &actual_outcomes[..50])?;
    println!("Subset accuracy (first 50 predictions): {:.2}%", accuracy_metrics.hit_rate * 100.0);
    
    // Test signal quality calculation
    println!("Testing signal quality calculation...");
    let signal_quality_metrics = analyzer.calculate_signal_quality(&predictions[..50], &market_data[..51])?;
    println!("Subset signal-to-noise ratio: {:.3}", signal_quality_metrics.signal_to_noise_ratio);
    
    println!("\n=== Demo Complete ===");
    Ok(())
}

/// Generate sample data for demonstration purposes
fn generate_sample_data(count: usize) -> (Vec<LDCPrediction>, Vec<Direction>, Vec<OHLCV>) {
    let mut predictions = Vec::new();
    let mut actual_outcomes = Vec::new();
    let mut market_data = Vec::new();
    
    // Generate base price series
    let mut price = 50000.0;
    
    for i in 0..count + 1 {
        // Generate OHLCV data with some realistic price movement
        let price_change = (i as f64 * 0.1).sin() * 100.0 + (i as f64 * 0.05).cos() * 50.0;
        price += price_change;
        
        let high = price + (i as f64 * 0.02).abs() * 20.0;
        let low = price - (i as f64 * 0.02).abs() * 20.0;
        let volume = 1000.0 + (i as f64 * 0.1).abs() * 500.0;
        
        market_data.push(OHLCV {
            timestamp: 1640995200 + (i as i64 * 300), // 5-minute intervals
            open: price,
            high,
            low,
            close: price,
            volume,
        });
        
        // Generate predictions (skip last market data point)
        if i < count {
            // Create somewhat realistic features
            let rsi = 50.0 + (i as f32 * 0.1).sin() * 30.0;
            let wt = (i as f32 * 0.05).cos() * 50.0;
            let cci = (i as f32 * 0.02).sin() * 100.0;
            let adx = 20.0 + (i as f32 * 0.03).abs() * 30.0;
            let f5 = rsi * 0.8 + (i as f32 * 0.07).cos() * 10.0;
            
            let features = FeatureSeries {
                f1: rsi,
                f2: wt,
                f3: cci,
                f4: adx,
                f5,
            };
            
            // Generate signal with some correlation to price movement
            let signal = (price_change / 50.0) as f32 + (i as f32 * 0.1).sin() * 0.5;
            
            // Determine prediction direction based on signal
            let prediction_direction = if signal > 0.2 {
                Direction::Long
            } else if signal < -0.2 {
                Direction::Short
            } else {
                Direction::Neutral
            };
            
            // Generate confidence based on signal strength
            let confidence = (signal.abs() * 2.0).min(1.0).max(0.1);
            
            // Create k-nearest neighbors data (simplified)
            let k_nearest_distances = vec![0.5, 0.7, 0.9, 1.1, 1.3];
            let k_nearest_labels = vec![
                prediction_direction,
                if i % 3 == 0 { Direction::Long } else { Direction::Short },
                Direction::Neutral,
                prediction_direction,
                if i % 2 == 0 { Direction::Short } else { Direction::Long },
            ];
            
            predictions.push(LDCPrediction {
                signal,
                confidence,
                k_nearest_distances,
                k_nearest_labels,
                prediction_direction,
            });
            
            // Generate actual outcome with some correlation to prediction but with noise
            let future_price_change = if i + 1 < market_data.len() {
                market_data[i + 1].close - market_data[i].close
            } else {
                0.0
            };
            
            let actual_direction = if future_price_change > 50.0 {
                Direction::Long
            } else if future_price_change < -50.0 {
                Direction::Short
            } else {
                Direction::Neutral
            };
            
            // Add some noise to make it more realistic (not perfect correlation)
            let actual_outcome = if i % 7 == 0 {
                // Introduce some random incorrect predictions
                match actual_direction {
                    Direction::Long => Direction::Short,
                    Direction::Short => Direction::Long,
                    Direction::Neutral => Direction::Neutral,
                }
            } else {
                actual_direction
            };
            
            actual_outcomes.push(actual_outcome);
        }
    }
    
    (predictions, actual_outcomes, market_data)
}