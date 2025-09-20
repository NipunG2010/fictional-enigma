use ldc_engine::{LDCEngine, LDCConfig, LDCPrediction};
use feature_pipeline::{Features, OHLCV, FeaturePipeline};
use polars::prelude::*;
use std::collections::HashMap;

/// End-to-end test that demonstrates the complete pipeline:
/// 1. Generate synthetic OHLCV data
/// 2. Compute features using feature-pipeline
/// 3. Train LDC engine with historical data
/// 4. Generate predictions for all data points
/// 5. Save results to Parquet file
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting End-to-End LDC Pipeline Test");
    
    // Step 1: Generate synthetic OHLCV data
    println!("📊 Generating synthetic OHLCV data...");
    let ohlcv_data = generate_synthetic_ohlcv(1000);
    println!("   Generated {} OHLCV bars", ohlcv_data.len());
    
    // Step 2: Compute features using feature-pipeline
    println!("🔧 Computing features using feature-pipeline...");
    let pipeline = FeaturePipeline::new(20);
    let features_list = pipeline.compute_features(&ohlcv_data)?;
    println!("   Computed features for {} bars", features_list.len());
    
    // Step 3: Configure and create LDC engine
    println!("🤖 Configuring LDC engine...");
    let mut config = LDCConfig::default();
    config.max_bars_back = 500;
    config.neighbors_count = 8;
    config.feature_count = 5;
    config.use_multithreading = true;
    config.enable_debug_logging = false; // Set to true for detailed logs
    config.log_predictions = false;
    config.log_performance_metrics = true;
    
    let mut engine = LDCEngine::with_config(config);
    println!("   LDC engine configured with {} max bars back", engine.config().max_bars_back);
    
    // Step 4: Train LDC engine with historical data (first 60% for training)
    let training_size = (ohlcv_data.len() as f64 * 0.6) as usize;
    println!("📚 Training LDC engine with {} bars...", training_size);
    
    // Filter out features that don't have all required fields
    let valid_features: Vec<&Features> = features_list.iter()
        .filter(|f| f.rsi.is_some() && f.wavetrend_1.is_some() && f.cci.is_some() && f.adx.is_some())
        .collect();
    
    println!("   Found {} valid feature samples (with all required features)", valid_features.len());
    
    if valid_features.len() < 100 {
        return Err("Not enough valid features for training. Need at least 100 samples with all features computed.".into());
    }
    
    // Use the valid features for training
    let training_valid_size = (valid_features.len() as f64 * 0.6) as usize;
    let training_features: Vec<Features> = valid_features[..training_valid_size].iter().map(|f| (*f).clone()).collect();
    let training_ohlcv: Vec<OHLCV> = ohlcv_data[..training_valid_size].to_vec();
    
    engine.create_training_samples_from_ohlcv(
        &training_ohlcv,
        &training_features,
        4, // 4-bar lookahead for labeling
    )?;
    
    let training_stats = engine.get_training_stats();
    println!("   Training complete: {} spaced samples ({} long, {} short)", 
             training_stats.0, training_stats.1, training_stats.2);
    
    // Step 5: Generate predictions for all data points
    println!("🔮 Generating predictions for all {} bars...", ohlcv_data.len());
    let all_valid_features: Vec<Features> = valid_features.iter().map(|f| (*f).clone()).collect();
    let predictions = engine.batch_predict_from_features(&all_valid_features)?;
    println!("   Generated {} predictions", predictions.len());
    
    // Step 6: Create comprehensive results DataFrame
    println!("📋 Creating results DataFrame...");
    let results_df = create_results_dataframe(&ohlcv_data, &all_valid_features, &predictions)?;
    println!("   Results DataFrame created with {} rows and {} columns", 
             results_df.height(), results_df.width());
    
    // Step 7: Save results to Parquet file
    let output_file = "ldc_pipeline_results.parquet";
    println!("💾 Saving results to {}...", output_file);
    
    let mut file = std::fs::File::create(output_file)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut results_df.clone())?;
    
    println!("✅ Results saved to {}", output_file);
    
    // Step 8: Display summary statistics
    println!("\n📈 Pipeline Summary:");
    display_summary_statistics(&results_df)?;
    
    // Step 9: Display performance metrics
    let metrics = engine.get_performance_metrics();
    println!("\n⚡ Performance Metrics:");
    println!("   Total predictions: {}", metrics.total_predictions);
    println!("   Average prediction time: {:.2}ms", metrics.average_prediction_time_ms);
    println!("   Parallel predictions: {}", metrics.parallel_predictions);
    println!("   Sequential predictions: {}", metrics.sequential_predictions);
    
    println!("\n🎉 End-to-End Pipeline Test Complete!");
    println!("   Check '{}' for detailed results", output_file);
    
    Ok(())
}

/// Generate synthetic OHLCV data for testing
fn generate_synthetic_ohlcv(num_bars: usize) -> Vec<OHLCV> {
    let mut ohlcv_data = Vec::new();
    let mut price = 100.0;
    let mut timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
    
    for i in 0..num_bars {
        // Generate realistic price movement with some trend and volatility
        let trend = (i as f64 * 0.001).sin() * 0.5; // Slow trend
        let volatility = 0.02 + (i as f64 * 0.0001).sin() * 0.01; // Variable volatility
        let random_move = (rand::random::<f64>() - 0.5) * volatility;
        
        price += trend + random_move;
        
        // Generate OHLC from price
        let open = price;
        let high = price + rand::random::<f64>() * 0.01;
        let low = price - rand::random::<f64>() * 0.01;
        let close = price + (rand::random::<f64>() - 0.5) * 0.005;
        let volume = 1000.0 + rand::random::<f64>() * 500.0;
        
        ohlcv_data.push(OHLCV {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
        
        timestamp += 300; // 5-minute bars
    }
    
    ohlcv_data
}

/// Create comprehensive results DataFrame
fn create_results_dataframe(
    ohlcv_data: &[OHLCV],
    features_list: &[Features],
    predictions: &[LDCPrediction],
) -> Result<DataFrame, PolarsError> {
    let mut series_vec = Vec::new();
    
    // Use the minimum length to ensure all arrays have the same size
    let min_len = ohlcv_data.len().min(features_list.len()).min(predictions.len());
    
    // OHLCV data (truncated to min_len)
    let timestamps: Vec<i64> = ohlcv_data[..min_len].iter().map(|o| o.timestamp).collect();
    let opens: Vec<f64> = ohlcv_data[..min_len].iter().map(|o| o.open).collect();
    let highs: Vec<f64> = ohlcv_data[..min_len].iter().map(|o| o.high).collect();
    let lows: Vec<f64> = ohlcv_data[..min_len].iter().map(|o| o.low).collect();
    let closes: Vec<f64> = ohlcv_data[..min_len].iter().map(|o| o.close).collect();
    let volumes: Vec<f64> = ohlcv_data[..min_len].iter().map(|o| o.volume).collect();
    
    series_vec.push(Series::new("timestamp", timestamps));
    series_vec.push(Series::new("open", opens));
    series_vec.push(Series::new("high", highs));
    series_vec.push(Series::new("low", lows));
    series_vec.push(Series::new("close", closes));
    series_vec.push(Series::new("volume", volumes));
    
    // Features (truncated to min_len)
    let rsi: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.rsi).collect();
    let sma_20: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.sma_20).collect();
    let ema_20: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.ema_20).collect();
    let std_20: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.std_20).collect();
    let zscore_20: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.zscore_20).collect();
    let momentum: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.momentum).collect();
    let wavetrend_1: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.wavetrend_1).collect();
    let wavetrend_2: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.wavetrend_2).collect();
    let cci: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.cci).collect();
    let adx: Vec<Option<f64>> = features_list[..min_len].iter().map(|f| f.adx).collect();
    
    series_vec.push(Series::new("rsi", rsi));
    series_vec.push(Series::new("sma_20", sma_20));
    series_vec.push(Series::new("ema_20", ema_20));
    series_vec.push(Series::new("std_20", std_20));
    series_vec.push(Series::new("zscore_20", zscore_20));
    series_vec.push(Series::new("momentum", momentum));
    series_vec.push(Series::new("wavetrend_1", wavetrend_1));
    series_vec.push(Series::new("wavetrend_2", wavetrend_2));
    series_vec.push(Series::new("cci", cci));
    series_vec.push(Series::new("adx", adx));
    
    // LDC Predictions (truncated to min_len)
    let ldc_signals: Vec<f32> = predictions[..min_len].iter().map(|p| p.signal).collect();
    let ldc_confidence: Vec<f32> = predictions[..min_len].iter().map(|p| p.confidence).collect();
    let ldc_direction: Vec<String> = predictions[..min_len].iter()
        .map(|p| format!("{:?}", p.prediction_direction)).collect();
    let ldc_k_neighbors: Vec<i32> = predictions[..min_len].iter()
        .map(|p| p.k_nearest_distances.len() as i32).collect();
    
    series_vec.push(Series::new("ldc_signal", ldc_signals));
    series_vec.push(Series::new("ldc_confidence", ldc_confidence));
    series_vec.push(Series::new("ldc_direction", ldc_direction));
    series_vec.push(Series::new("ldc_k_neighbors", ldc_k_neighbors));
    
    DataFrame::new(series_vec)
}

/// Display summary statistics
fn display_summary_statistics(df: &DataFrame) -> Result<(), PolarsError> {
    println!("   Total bars processed: {}", df.height());
    
    // LDC Signal Statistics
    if let Ok(signal_col) = df.column("ldc_signal") {
        let signal_values: Vec<f32> = signal_col.f32()?.into_no_null_iter().collect();
        if !signal_values.is_empty() {
            let mean = signal_values.iter().sum::<f32>() / signal_values.len() as f32;
            let variance = signal_values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / signal_values.len() as f32;
            let std = variance.sqrt();
            let min = signal_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = signal_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            println!("   LDC Signal Statistics:");
            println!("     Mean: {:.4}", mean);
            println!("     Std: {:.4}", std);
            println!("     Min: {:.4}", min);
            println!("     Max: {:.4}", max);
        }
    }
    
    // Direction Distribution
    if let Ok(direction_col) = df.column("ldc_direction") {
        let direction_values: Vec<String> = direction_col.str()?.into_no_null_iter()
            .map(|s| s.to_string()).collect();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for direction in direction_values {
            *counts.entry(direction).or_insert(0) += 1;
        }
        
        println!("   LDC Direction Distribution:");
        for (direction, count) in counts {
            println!("     {}: {}", direction, count);
        }
    }
    
    // Confidence Statistics
    if let Ok(confidence_col) = df.column("ldc_confidence") {
        let confidence_values: Vec<f32> = confidence_col.f32()?.into_no_null_iter().collect();
        if !confidence_values.is_empty() {
            let mean = confidence_values.iter().sum::<f32>() / confidence_values.len() as f32;
            let variance = confidence_values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / confidence_values.len() as f32;
            let std = variance.sqrt();
            
            println!("   LDC Confidence Statistics:");
            println!("     Mean: {:.4}", mean);
            println!("     Std: {:.4}", std);
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_end_to_end_pipeline() {
        // This test runs the complete pipeline with a smaller dataset
        let ohlcv_data = generate_synthetic_ohlcv(100);
        let pipeline = FeaturePipeline::new(20);
        let features_list = pipeline.compute_features(&ohlcv_data).unwrap();
        
        let mut config = LDCConfig::default();
        config.max_bars_back = 50;
        config.neighbors_count = 5;
        config.use_multithreading = false; // Disable for testing
        
        let mut engine = LDCEngine::with_config(config);
        
        // Train with first 60% of data
        let training_size = (ohlcv_data.len() as f64 * 0.6) as usize;
        engine.create_training_samples_from_ohlcv(
            &ohlcv_data[..training_size],
            &features_list[..training_size],
            4,
        ).unwrap();
        
        // Generate predictions
        let predictions = engine.batch_predict_from_features(&features_list).unwrap();
        
        // Verify results
        assert_eq!(predictions.len(), features_list.len());
        assert!(predictions.iter().all(|p| p.signal.is_finite()));
        assert!(predictions.iter().all(|p| p.confidence >= 0.0));
        
        // Test DataFrame creation
        let results_df = create_results_dataframe(&ohlcv_data, &features_list, &predictions).unwrap();
        assert_eq!(results_df.height(), ohlcv_data.len());
        assert!(results_df.width() > 10); // Should have many columns
        
        println!("✅ End-to-end pipeline test passed!");
    }
}