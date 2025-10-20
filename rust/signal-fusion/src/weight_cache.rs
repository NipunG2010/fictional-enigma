use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Deserialize, Serialize};

use crate::FusionWeights;

/// Cache key based on rounded observation values for consistent lookups
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CacheKey {
    // Store observations as integer representations (rounded to 3 decimals)
    // This allows for HashMap compatibility while handling floating point precision
    obs_ldc: i32,  // observation * 1000
    obs_mr: i32,
    obs_tsmom: i32,
}

impl CacheKey {
    /// Create a cache key from observations, rounding to 3 decimal places
    fn from_observations(observations: &[f32; 3]) -> Self {
        Self {
            obs_ldc: (observations[0] * 1000.0).round() as i32,
            obs_mr: (observations[1] * 1000.0).round() as i32,
            obs_tsmom: (observations[2] * 1000.0).round() as i32,
        }
    }
}

/// Cache entry containing weights and timestamp for TTL management
#[derive(Debug, Clone)]
struct CacheEntry {
    weights: FusionWeights,
    timestamp: Instant,
}

impl CacheEntry {
    fn new(weights: FusionWeights) -> Self {
        Self {
            weights,
            timestamp: Instant::now(),
        }
    }
    
    fn is_expired(&self, ttl: Duration) -> bool {
        self.timestamp.elapsed() > ttl
    }
}

/// Statistics for cache performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub evictions: u64,
    pub hit_rate: f64,
}

/// Thread-safe weight cache with TTL-based expiration and size limits
pub struct WeightCache {
    cache: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    ttl: Duration,
    max_size: usize,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
}

impl WeightCache {
    /// Create a new weight cache with specified TTL and maximum size
    ///
    /// # Arguments
    /// * `ttl` - Time-to-live for cache entries
    /// * `max_size` - Maximum number of entries before eviction
    pub fn new(ttl: Duration, max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            ttl,
            max_size,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Get weights from cache if available and not expired
    ///
    /// # Arguments
    /// * `observations` - Signal observations [s_ldc, s_mr, s_tsmom]
    ///
    /// # Returns
    /// * `Some(FusionWeights)` if cache hit and not expired
    /// * `None` if cache miss or expired
    pub fn get(&self, observations: &[f32; 3]) -> Option<FusionWeights> {
        let key = CacheKey::from_observations(observations);
        
        // Try to read from cache
        let cache = self.cache.read().unwrap();
        
        if let Some(entry) = cache.get(&key) {
            if !entry.is_expired(self.ttl) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(entry.weights.clone());
            }
            // Entry is expired, will be cleaned up later
        }
        
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    /// Insert weights into cache
    ///
    /// # Arguments
    /// * `observations` - Signal observations [s_ldc, s_mr, s_tsmom]
    /// * `weights` - Fusion weights to cache
    pub fn insert(&self, observations: [f32; 3], weights: FusionWeights) {
        let key = CacheKey::from_observations(&observations);
        let entry = CacheEntry::new(weights);
        
        let mut cache = self.cache.write().unwrap();
        
        // Check if we need to evict entries
        if cache.len() >= self.max_size && !cache.contains_key(&key) {
            self.evict_oldest(&mut cache);
        }
        
        cache.insert(key, entry);
    }
    
    /// Evict expired entries from the cache
    ///
    /// This should be called periodically to clean up expired entries
    pub fn evict_expired(&self) {
        let mut cache = self.cache.write().unwrap();
        let ttl = self.ttl;
        
        let expired_keys: Vec<CacheKey> = cache
            .iter()
            .filter(|(_, entry)| entry.is_expired(ttl))
            .map(|(key, _)| *key)
            .collect();
        
        let evicted_count = expired_keys.len();
        for key in expired_keys {
            cache.remove(&key);
        }
        
        if evicted_count > 0 {
            self.evictions.fetch_add(evicted_count as u64, Ordering::Relaxed);
        }
    }
    
    /// Evict the oldest entry from the cache (LRU-style)
    fn evict_oldest(&self, cache: &mut HashMap<CacheKey, CacheEntry>) {
        if let Some(oldest_key) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.timestamp)
            .map(|(key, _)| *key)
        {
            cache.remove(&oldest_key);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Get cache statistics for monitoring
    pub fn get_stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        
        let cache = self.cache.read().unwrap();
        
        CacheStats {
            hits,
            misses,
            size: cache.len(),
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate,
        }
    }
    
    /// Clear all entries from the cache
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }
    
    /// Get the current size of the cache
    pub fn size(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }
}

impl Default for WeightCache {
    fn default() -> Self {
        // Default: 60 second TTL, 1000 entry max size
        Self::new(Duration::from_secs(60), 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_cache_key_generation() {
        let obs1 = [0.123456, 0.789012, -0.345678];
        let obs2 = [0.123499, 0.789049, -0.345699]; // Should round to same key
        let obs3 = [0.124000, 0.789000, -0.346000]; // Different key
        
        let key1 = CacheKey::from_observations(&obs1);
        let key2 = CacheKey::from_observations(&obs2);
        let key3 = CacheKey::from_observations(&obs3);
        
        assert_eq!(key1, key2, "Similar observations should produce same key");
        assert_ne!(key1, key3, "Different observations should produce different keys");
    }
    
    #[test]
    fn test_cache_hit_miss() {
        let cache = WeightCache::new(Duration::from_secs(60), 100);
        let observations = [0.5, 0.3, 0.2];
        let weights = FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        };
        
        // Miss on first access
        assert!(cache.get(&observations).is_none());
        let stats = cache.get_stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
        
        // Insert and hit on second access
        cache.insert(observations, weights.clone());
        let result = cache.get(&observations);
        assert!(result.is_some());
        let cached_weights = result.unwrap();
        assert_eq!(cached_weights.w_ldc, weights.w_ldc);
        
        let stats = cache.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.size, 1);
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
    fn test_size_based_eviction() {
        let cache = WeightCache::new(Duration::from_secs(60), 3);
        let weights = FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        };
        
        // Fill cache to max size
        cache.insert([0.1, 0.1, 0.1], weights.clone());
        cache.insert([0.2, 0.2, 0.2], weights.clone());
        cache.insert([0.3, 0.3, 0.3], weights.clone());
        assert_eq!(cache.size(), 3);
        
        // Adding one more should trigger eviction
        cache.insert([0.4, 0.4, 0.4], weights.clone());
        assert_eq!(cache.size(), 3);
        
        let stats = cache.get_stats();
        assert_eq!(stats.evictions, 1);
    }
    
    #[test]
    fn test_evict_expired() {
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
        
        // Evict expired entries
        cache.evict_expired();
        assert_eq!(cache.size(), 0);
        
        let stats = cache.get_stats();
        assert_eq!(stats.evictions, 3);
    }
    
    #[test]
    fn test_thread_safety() {
        let cache = Arc::new(WeightCache::new(Duration::from_secs(60), 1000));
        let weights = FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        };
        
        let mut handles = vec![];
        
        // Spawn multiple threads to insert and read
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
    fn test_cache_stats() {
        let cache = WeightCache::new(Duration::from_secs(60), 100);
        let weights = FusionWeights {
            w_ldc: 0.4,
            w_mr: 0.3,
            w_tsmom: 0.3,
        };
        
        let obs1 = [0.1, 0.1, 0.1];
        let obs2 = [0.2, 0.2, 0.2];
        
        // Generate some hits and misses
        cache.get(&obs1); // miss
        cache.insert(obs1, weights.clone());
        cache.get(&obs1); // hit
        cache.get(&obs1); // hit
        cache.get(&obs2); // miss
        
        let stats = cache.get_stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.size, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }
    
    #[test]
    fn test_clear() {
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
}
