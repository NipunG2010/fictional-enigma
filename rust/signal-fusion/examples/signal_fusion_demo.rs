use signal_fusion::{SignalComponents, FusionWeights, SignalFusion};

fn main() {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Signal Fusion Engine Demo ===\n");

    // Create a signal fusion engine with threshold and cooldown
    let mut fusion = SignalFusion::new(0.3, 60);
    println!("Created SignalFusion with threshold=0.3, cooldown=60s\n");

    // Example 1: Valid signals above threshold
    println!("Example 1: Strong BUY signal");
    let components = SignalComponents {
        s_ldc: 0.8,
        s_mr: 0.6,
        s_tsmom: 0.4,
    };
    let weights = FusionWeights {
        w_ldc: 0.5,
        w_mr: 0.3,
        w_tsmom: 0.2,
    };

    match fusion.fuse_signals(
        components, 
        weights, 
        1000, 
        "BTCUSDT", 
        "v1.0",
        "demo-correlation-1".to_string(),
        "demo-checksum-1".to_string(),
        25,
    ) {
        Ok(Some(signal)) => {
            println!("✓ Generated signal:");
            println!("  Side: {}", signal.side);
            println!("  Strength: {:.4}", signal.strength);
            println!("  Confidence: {:.4}", signal.confidence);
            println!("  Normalized weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}\n",
                signal.weights.w_ldc, signal.weights.w_mr, signal.weights.w_tsmom);
        }
        Ok(None) => println!("✗ No signal generated\n"),
        Err(e) => println!("✗ Error: {}\n", e),
    }

    // Example 2: Signal below threshold
    println!("Example 2: Weak signal below threshold");
    let components = SignalComponents {
        s_ldc: 0.1,
        s_mr: 0.1,
        s_tsmom: 0.1,
    };
    let weights = FusionWeights {
        w_ldc: 0.33,
        w_mr: 0.33,
        w_tsmom: 0.34,
    };

    match fusion.fuse_signals(
        components, 
        weights, 
        1030, 
        "BTCUSDT", 
        "v1.0",
        "demo-correlation-2".to_string(),
        "demo-checksum-2".to_string(),
        20,
    ) {
        Ok(Some(signal)) => {
            println!("✓ Generated signal: {}\n", signal.side);
        }
        Ok(None) => println!("✓ No signal generated (below threshold)\n"),
        Err(e) => println!("✗ Error: {}\n", e),
    }

    // Example 3: Cooldown suppression
    println!("Example 3: Signal suppressed by cooldown");
    let components = SignalComponents {
        s_ldc: 0.8,
        s_mr: 0.6,
        s_tsmom: 0.4,
    };
    let weights = FusionWeights {
        w_ldc: 0.5,
        w_mr: 0.3,
        w_tsmom: 0.2,
    };

    match fusion.fuse_signals(
        components, 
        weights, 
        1040, 
        "BTCUSDT", 
        "v1.0",
        "demo-correlation-3".to_string(),
        "demo-checksum-3".to_string(),
        30,
    ) {
        Ok(Some(signal)) => {
            println!("✓ Generated signal: {}\n", signal.side);
        }
        Ok(None) => println!("✓ No signal generated (cooldown active)\n"),
        Err(e) => println!("✗ Error: {}\n", e),
    }

    // Example 4: Invalid signal components
    println!("Example 4: Invalid signal components (out of range)");
    let components = SignalComponents {
        s_ldc: 2.0, // Out of valid range [-1.0, 1.0]
        s_mr: 0.6,
        s_tsmom: 0.4,
    };
    let weights = FusionWeights {
        w_ldc: 0.5,
        w_mr: 0.3,
        w_tsmom: 0.2,
    };

    match fusion.fuse_signals(
        components, 
        weights, 
        1100, 
        "BTCUSDT", 
        "v1.0",
        "demo-correlation-4".to_string(),
        "demo-checksum-4".to_string(),
        35,
    ) {
        Ok(Some(signal)) => {
            println!("✓ Generated signal: {}\n", signal.side);
        }
        Ok(None) => println!("✓ No signal generated\n"),
        Err(e) => println!("✓ Validation error caught: {}\n", e),
    }

    // Example 5: SELL signal
    println!("Example 5: Strong SELL signal");
    let components = SignalComponents {
        s_ldc: -0.7,
        s_mr: -0.5,
        s_tsmom: -0.3,
    };
    let weights = FusionWeights {
        w_ldc: 0.5,
        w_mr: 0.3,
        w_tsmom: 0.2,
    };

    match fusion.fuse_signals(
        components, 
        weights, 
        1200, 
        "BTCUSDT", 
        "v1.0",
        "demo-correlation-5".to_string(),
        "demo-checksum-5".to_string(),
        40,
    ) {
        Ok(Some(signal)) => {
            println!("✓ Generated signal:");
            println!("  Side: {}", signal.side);
            println!("  Strength: {:.4}", signal.strength);
            println!("  Confidence: {:.4}\n", signal.confidence);
        }
        Ok(None) => println!("✗ No signal generated\n"),
        Err(e) => println!("✗ Error: {}\n", e),
    }

    // Example 6: Weight normalization
    println!("Example 6: Weight normalization demonstration");
    let weights = FusionWeights {
        w_ldc: 0.6,
        w_mr: 0.3,
        w_tsmom: 0.1,
    };
    println!("Original weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}",
        weights.w_ldc, weights.w_mr, weights.w_tsmom);
    
    let normalized = weights.normalize();
    println!("Normalized weights: LDC={:.3}, MR={:.3}, TSMOM={:.3}",
        normalized.w_ldc, normalized.w_mr, normalized.w_tsmom);
    
    let sum = normalized.w_ldc.abs() + normalized.w_mr.abs() + normalized.w_tsmom.abs();
    println!("Sum of absolute values: {:.6}\n", sum);

    println!("=== Demo Complete ===");
}
