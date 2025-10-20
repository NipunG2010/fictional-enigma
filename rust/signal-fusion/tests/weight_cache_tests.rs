use signal_fusion::weight_cache::WeightCache;
use signal_fusion::FusionWeights;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_cache_key_generation_consistency() {
    let cache = WeightCache::new(Duration::from_secs(60), 100);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    // Similar observations should produce cache hits
    let obs1 = [0.123456, 0.789012, -0.345678];
    let obs2 = [0.123499, 0.789049, -0.345699]; // Should round to same key
    
    cache.insert(obs1, weights.clone());
    
    // Both should hit the same cache entry
    assert!(cache.get(&obs1).is_some());
    assert!(cache.get(&obs2).is_some());
}

#[test]
fn test_basic_cache_operations() {
    let cache = WeightCache::new(Duration::from_secs(60), 100);
    let observations = [0.5, 0.3, 0.2];
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    // Initial miss
    assert!(cache.get(&observations).is_none());
    
    // Insert
    cache.insert(observations, weights.clone());
    
    // Hit after insert
    let result = cache.get(&observations);
    assert!(result.is_some());
    let cached = result.unwrap();
    assert_eq!(cached.w_ldc, weights.w_ldc);
    assert_eq!(cached.w_mr, weights.w_mr);
    assert_eq!(cached.w_tsmom, weights.w_tsmom);
}

#[test]
fn test_ttl_expiration() {
    let cache = WeightCache::new(Duration::from_millis(100), 100);
    let observations = [0.5, 0.3, 0.2];
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    cache.insert(observations, weights);
    
    // Should hit immediately
    assert!(cache.get(&observations).is_some());
    
    // Wait for expiration
    thread::sleep(Duration::from_millis(150));
    
    // Should miss after expiration
    assert!(cache.get(&observations).is_none());
}

#[test]
fn test_size_limit_eviction() {
    let cache = WeightCache::new(Duration::from_secs(60), 3);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    // Fill to capacity
    cache.insert([0.1, 0.1, 0.1], weights.clone());
    cache.insert([0.2, 0.2, 0.2], weights.clone());
    cache.insert([0.3, 0.3, 0.3], weights.clone());
    assert_eq!(cache.size(), 3);
    
    // Adding one more should evict oldest
    cache.insert([0.4, 0.4, 0.4], weights.clone());
    assert_eq!(cache.size(), 3);
    
    let stats = cache.get_stats();
    assert_eq!(stats.evictions, 1);
}

#[test]
fn test_manual_eviction() {
    let cache = WeightCache::new(Duration::from_millis(50), 100);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    // Insert multiple entries
    cache.insert([0.1, 0.1, 0.1], weights.clone());
    cache.insert([0.2, 0.2, 0.2], weights.clone());
    cache.insert([0.3, 0.3, 0.3], weights.clone());
    assert_eq!(cache.size(), 3);
    
    // Wait for expiration
    thread::sleep(Duration::from_millis(100));
    
    // Manual eviction
    cache.evict_expired();
    assert_eq!(cache.size(), 0);
}

#[test]
fn test_cache_statistics() {
    let cache = WeightCache::new(Duration::from_secs(60), 100);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    let obs1 = [0.1, 0.1, 0.1];
    let obs2 = [0.2, 0.2, 0.2];
    
    // Generate hits and misses
    cache.get(&obs1); // miss
    cache.insert(obs1, weights.clone());
    cache.get(&obs1); // hit
    cache.get(&obs1); // hit
    cache.get(&obs2); // miss
    
    let stats = cache.get_stats();
    assert_eq!(stats.hits, 2);
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.size, 1);
    assert!((stats.hit_rate - 0.5).abs() < 0.01);
}

#[test]
fn test_concurrent_access() {
    let cache = Arc::new(WeightCache::new(Duration::from_secs(60), 1000));
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    let mut handles = vec![];
    
    // Spawn multiple threads
    for i in 0..10 {
        let cache_clone = Arc::clone(&cache);
        let weights_clone = weights.clone();
        
        let handle = thread::spawn(move || {
            let obs = [i as f32 * 0.1, i as f32 * 0.1, i as f32 * 0.1];
            cache_clone.insert(obs, weights_clone);
            cache_clone.get(&obs)
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.is_some());
    }
    
    assert_eq!(cache.size(), 10);
}

#[test]
fn test_clear_cache() {
    let cache = WeightCache::new(Duration::from_secs(60), 100);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    cache.insert([0.1, 0.1, 0.1], weights.clone());
    cache.insert([0.2, 0.2, 0.2], weights.clone());
    assert_eq!(cache.size(), 2);
    
    cache.clear();
    assert_eq!(cache.size(), 0);
}

#[test]
fn test_high_precision_rounding() {
    let cache = WeightCache::new(Duration::from_secs(60), 100);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    // Test that values within 0.0005 round to same key
    let obs1 = [0.1234, 0.5678, 0.9012];
    let obs2 = [0.1234, 0.5678, 0.9014]; // Within rounding threshold
    let obs3 = [0.1234, 0.5678, 0.9020]; // Outside rounding threshold
    
    cache.insert(obs1, weights.clone());
    
    assert!(cache.get(&obs1).is_some());
    assert!(cache.get(&obs2).is_some()); // Should hit same entry
    assert!(cache.get(&obs3).is_none()); // Should miss
}

#[test]
fn test_negative_observations() {
    let cache = WeightCache::new(Duration::from_secs(60), 100);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    let obs = [-0.5, -0.3, -0.2];
    cache.insert(obs, weights.clone());
    
    let result = cache.get(&obs);
    assert!(result.is_some());
}

#[test]
fn test_mixed_sign_observations() {
    let cache = WeightCache::new(Duration::from_secs(60), 100);
    let weights = FusionWeights {
        w_ldc: 0.4,
        w_mr: 0.3,
        w_tsmom: 0.3,
    };
    
    let obs = [0.5, -0.3, 0.2];
    cache.insert(obs, weights.clone());
    
    let result = cache.get(&obs);
    assert!(result.is_some());
}
