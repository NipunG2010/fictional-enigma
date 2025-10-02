use ldc_engine::*;

// Import the testing framework from the test module
// Note: In a real application, this would be in a separate crate or module
// For demonstration purposes, we'll create a simple example here

fn main() {
    println!("=== Mathematical Accuracy Testing Framework Demo ===\n");

    // Create some test feature series
    let features1 = FeatureSeries {
        f1: 50.0,  // RSI
        f2: 10.0,  // WT
        f3: 25.0,  // CCI
        f4: 30.0,  // ADX
        f5: 65.0,  // Additional feature
    };

    let features2 = FeatureSeries {
        f1: 45.0,
        f2: 15.0,
        f3: 30.0,
        f4: 35.0,
        f5: 60.0,
    };

    println!("Testing distance calculation methods:");
    println!("Features 1: {:?}", features1);
    println!("Features 2: {:?}", features2);
    println!();

    // Test standard distance calculation
    let standard_distance = LDCEngine::lorentzian_distance(&features1, &features2, 5);
    println!("Standard Lorentzian distance: {:.6}", standard_distance);

    // Test SIMD distance calculation
    match features1.lorentzian_distance_simd(&features2) {
        Ok(simd_distance) => {
            println!("SIMD Lorentzian distance: {:.6}", simd_distance);
            let diff = (standard_distance - simd_distance).abs();
            println!("Difference (Standard vs SIMD): {:.2e}", diff);
            
            if diff < 1e-5 {
                println!("✓ SIMD calculation matches standard within tolerance");
            } else {
                println!("✗ SIMD calculation differs from standard");
            }
        }
        Err(e) => {
            println!("SIMD calculation failed: {:?}", e);
        }
    }

    // Test HNSW distance calculation
    let features1_array = features1.to_array();
    let features2_array = features2.to_array();
    let hnsw_distance = lorentzian_distance_hnsw(&features1_array, &features2_array);
    println!("HNSW Lorentzian distance: {:.6}", hnsw_distance);
    
    let hnsw_diff = (standard_distance - hnsw_distance).abs();
    println!("Difference (Standard vs HNSW): {:.2e}", hnsw_diff);
    
    if hnsw_diff < 1e-5 {
        println!("✓ HNSW calculation matches standard within tolerance");
    } else {
        println!("✗ HNSW calculation differs from standard");
    }

    println!("\n=== Edge Case Testing ===");

    // Test with identical features
    let identical_distance = LDCEngine::lorentzian_distance(&features1, &features1, 5);
    println!("Distance between identical features: {:.6}", identical_distance);
    
    if identical_distance == 0.0 {
        println!("✓ Identical features produce zero distance");
    } else {
        println!("✗ Identical features should produce zero distance");
    }

    // Test with zero features
    let zero_features = FeatureSeries {
        f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0
    };
    let zero_distance = LDCEngine::lorentzian_distance(&zero_features, &zero_features, 5);
    println!("Distance between zero features: {:.6}", zero_distance);

    // Test with extreme values
    let extreme_features = FeatureSeries {
        f1: 1e6, f2: -1e6, f3: 1e6, f4: -1e6, f5: 1e6
    };
    let extreme_distance = LDCEngine::lorentzian_distance(&features1, &extreme_features, 5);
    println!("Distance with extreme values: {:.6}", extreme_distance);
    
    if extreme_distance.is_finite() {
        println!("✓ Extreme values produce finite distance");
    } else {
        println!("✗ Extreme values should produce finite distance");
    }

    println!("\n=== Performance Comparison ===");

    // Create a larger dataset for performance testing
    let mut test_features = Vec::new();
    for i in 0..1000 {
        test_features.push(FeatureSeries {
            f1: (i as f32 * 0.1).sin() * 50.0 + 50.0,
            f2: (i as f32 * 0.05).cos() * 100.0,
            f3: (i as f32 * 0.02).sin() * 200.0,
            f4: (i as f32 * 0.01).abs() * 50.0,
            f5: (i as f32 * 0.03).tan().abs() * 30.0,
        });
    }

    let query = &test_features[0];
    let targets = &test_features[1..100]; // Use 99 targets

    // Time standard batch calculation
    let start = std::time::Instant::now();
    let standard_results = FeatureSeries::batch_lorentzian_distance_standard(query, targets);
    let standard_time = start.elapsed();

    // Time SIMD batch calculation
    let start = std::time::Instant::now();
    let simd_results = match FeatureSeries::batch_lorentzian_distance_simd(query, targets, 32) {
        Ok(results) => results,
        Err(_) => {
            println!("SIMD batch calculation failed, using standard results");
            standard_results.clone()
        }
    };
    let simd_time = start.elapsed();

    println!("Standard batch calculation time: {:?}", standard_time);
    println!("SIMD batch calculation time: {:?}", simd_time);
    
    if simd_time < standard_time {
        println!("✓ SIMD batch calculation is faster");
    } else {
        println!("→ SIMD batch calculation time is similar or slower (expected for small datasets)");
    }

    // Verify results are consistent
    let mut max_diff = 0.0f32;
    for (std_result, simd_result) in standard_results.iter().zip(simd_results.iter()) {
        let diff = (std_result - simd_result).abs();
        max_diff = max_diff.max(diff);
    }

    println!("Maximum difference between standard and SIMD batch results: {:.2e}", max_diff);
    
    if max_diff < 1e-5 {
        println!("✓ Batch calculations are consistent within tolerance");
    } else {
        println!("✗ Batch calculations differ beyond tolerance");
    }

    println!("\n=== Framework Summary ===");
    println!("The mathematical accuracy testing framework provides:");
    println!("• Comprehensive test cases covering standard, edge, extreme, and precision scenarios");
    println!("• Validation of SIMD vs standard distance calculations");
    println!("• Verification of HNSW compatibility with exact calculations");
    println!("• Detailed error reporting and statistical analysis");
    println!("• Configurable tolerance levels for different precision requirements");
    println!("• Category-based test organization for targeted validation");
}