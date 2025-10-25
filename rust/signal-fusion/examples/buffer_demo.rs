//! Signal Buffer Demo
//! 
//! This example demonstrates the signal buffering system with various overflow strategies
//! and persistence capabilities.

use signal_fusion::{
    SignalBuffer, BufferConfig, OverflowStrategy, PersistenceConfig,
    TradingSignal, SignalComponents, FusionWeights, SignalSide
};
use std::time::{SystemTime, UNIX_EPOCH};

fn create_test_signal(symbol: &str, strength: f32, confidence: f32) -> TradingSignal {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
        
    TradingSignal::new(
        now,
        symbol.to_string(),
        if strength > 0.0 { SignalSide::Buy } else { SignalSide::Sell },
        strength,
        confidence,
        SignalComponents {
            s_ldc: strength * 0.5,
            s_mr: strength * 0.3,
            s_tsmom: strength * 0.2,
        },
        FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        },
        "v1.0".to_string(),
        format!("demo-correlation-{}", symbol),
        format!("checksum-{}", symbol),
        50,
    )
}

fn main() -> anyhow::Result<()> {
    println!("Signal Buffer Demo");
    println!("==================");
    
    // Demo 1: Basic buffer operations
    println!("\n1. Basic Buffer Operations");
    println!("--------------------------");
    
    let mut buffer = SignalBuffer::with_default_config();
    
    // Add some signals
    let signals = vec![
        create_test_signal("BTCUSDT", 0.8, 0.9),
        create_test_signal("ETHUSDT", -0.6, 0.7),
        create_test_signal("ADAUSDT", 0.4, 0.6),
    ];
    
    for signal in signals {
        buffer.push(signal)?;
        println!("Buffer size: {}/{}", buffer.len(), buffer.capacity());
    }
    
    // Pop signals
    while let Some(buffered_signal) = buffer.pop() {
        println!("Popped: {} (age: {}s)", 
                 buffered_signal.signal.to_compact_string(),
                 buffered_signal.age_seconds());
    }
    
    // Demo 2: Overflow strategies
    println!("\n2. Overflow Strategies");
    println!("----------------------");
    
    let config = BufferConfig {
        max_size: 2,
        overflow_strategy: OverflowStrategy::DropOldest,
        persistence: None,
        enable_metrics: true,
        warning_threshold: 0.8,
    };
    
    let mut buffer = SignalBuffer::new(config);
    
    // Fill buffer beyond capacity
    for i in 0..4 {
        let signal = create_test_signal(&format!("SYMBOL{}", i), 0.5, 0.8);
        buffer.push(signal)?;
        println!("Added SYMBOL{}, buffer size: {}", i, buffer.len());
    }
    
    // Check which signals remain
    println!("Remaining signals:");
    while let Some(buffered_signal) = buffer.pop() {
        println!("  - {}", buffered_signal.signal.symbol);
    }
    
    // Demo 3: Priority-based popping
    println!("\n3. Priority-Based Operations");
    println!("-----------------------------");
    
    let mut buffer = SignalBuffer::with_default_config();
    
    // Add signals with different confidence levels
    let signals = vec![
        create_test_signal("LOW_CONF", 0.5, 0.3),    // Low confidence
        create_test_signal("HIGH_CONF", 0.7, 0.9),   // High confidence  
        create_test_signal("MED_CONF", 0.6, 0.6),    // Medium confidence
    ];
    
    for signal in signals {
        buffer.push(signal)?;
    }
    
    // Pop by priority (highest confidence first)
    println!("Popping by priority (confidence):");
    while let Some(buffered_signal) = buffer.pop_highest_priority() {
        println!("  - {} (confidence: {:.1})", 
                 buffered_signal.signal.symbol,
                 buffered_signal.signal.confidence);
    }
    
    // Demo 4: Persistence (if temp directory is available)
    println!("\n4. Persistence Demo");
    println!("-------------------");
    
    let temp_dir = std::env::temp_dir().join("signal_buffer_demo");
    
    let persist_config = BufferConfig {
        max_size: 100,
        overflow_strategy: OverflowStrategy::DropOldest,
        persistence: Some(PersistenceConfig {
            persist_path: temp_dir.clone(),
            auto_persist: false,
            persist_interval_sec: 60,
            max_backup_files: 3,
            atomic_operations: true,
        }),
        enable_metrics: true,
        warning_threshold: 0.8,
    };
    
    let mut buffer = SignalBuffer::new(persist_config);
    
    // Add signals
    let signals = vec![
        create_test_signal("PERSIST1", 0.7, 0.8),
        create_test_signal("PERSIST2", -0.5, 0.6),
    ];
    
    for signal in signals {
        buffer.push(signal)?;
    }
    
    println!("Added {} signals to buffer", buffer.len());
    
    // Persist to disk
    buffer.persist()?;
    println!("Persisted buffer to: {:?}", buffer.persist_file_path());
    
    // Clear and restore
    let original_count = buffer.len();
    buffer.clear();
    println!("Cleared buffer (was {} signals)", original_count);
    
    buffer.restore()?;
    println!("Restored {} signals from disk", buffer.len());
    
    // Demo 5: Buffer metrics
    println!("\n5. Buffer Metrics");
    println!("-----------------");
    
    let metrics = buffer.metrics();
    println!("Current size: {}", metrics.current_size);
    println!("Max size: {}", metrics.max_size);
    println!("Utilization: {:.1}%", metrics.utilization * 100.0);
    println!("Total added: {}", metrics.total_added);
    println!("Total removed: {}", metrics.total_removed);
    println!("Total dropped: {}", metrics.total_dropped);
    println!("Overflow events: {}", metrics.overflow_events);
    
    // Clean up
    let _ = std::fs::remove_dir_all(&temp_dir);
    
    println!("\nDemo completed successfully!");
    
    Ok(())
}