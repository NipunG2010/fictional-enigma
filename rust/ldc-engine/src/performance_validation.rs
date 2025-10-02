use crate::{LDCEngine, FeatureSeries, TrainingSample, Direction};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use rand::prelude::*;

/// Performance validation framework for testing LDC engine performance
pub struct PerformanceValidator {
    config: PerformanceTestConfig,
    test_datasets: Vec<TestDataset>,
}

/// Configuration for performance testing with specific latency targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTestConfig {
    /// Target latency for 1k samples: 0.5ms
    pub target_latency_1k_samples_ms: f64,
    /// Target latency for 10k samples: 1.0ms
    pub target_latency_10k_samples_ms: f64,
    /// Target latency for 50k samples: 5.0ms
    pub target_latency_50k_samples_ms: f64,
    /// Target CPU utilization percentage: 90%
    pub target_cpu_utilization_percent: f64,
    /// Target HNSW accuracy percentage: 95%
    pub target_hnsw_accuracy_percent: f64,
    /// Number of test iterations for statistical significance
    pub test_iterations: usize,
    /// Number of warmup iterations before measurement
    pub warmup_iterations: usize,
}

impl Default for PerformanceTestConfig {
    fn default() -> Self {
        Self {
            target_latency_1k_samples_ms: 0.5,
            target_latency_10k_samples_ms: 1.0,
            target_latency_50k_samples_ms: 5.0,
            target_cpu_utilization_percent: 90.0,
            target_hnsw_accuracy_percent: 95.0,
            test_iterations: 100,
            warmup_iterations: 10,
        }
    }
}

/// Test dataset with synthetic trading data
#[derive(Debug, Clone)]
pub struct TestDataset {
    pub name: String,
    pub size: usize,
    pub samples: Vec<TrainingSample>,
    pub query_features: Vec<FeatureSeries>,
}

/// Individual performance test case result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTestCase {
    pub dataset_name: String,
    pub dataset_size: usize,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub target_latency_ms: f64,
    pub passed: bool,
}

/// Overall performance test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTestResult {
    pub results: Vec<PerformanceTestCase>,
}

impl PerformanceTestResult {
    /// Check if all performance tests passed
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }
    
    /// Get the number of passed tests
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }
    
    /// Get the total number of tests
    pub fn total_count(&self) -> usize {
        self.results.len()
    }
}

/// Individual HNSW accuracy test case result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWAccuracyCase {
    pub dataset_name: String,
    pub dataset_size: usize,
    pub accuracy_percent: f64,
    pub target_accuracy_percent: f64,
    pub passed: bool,
}

/// Overall HNSW accuracy test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWAccuracyResult {
    pub results: Vec<HNSWAccuracyCase>,
}

impl HNSWAccuracyResult {
    /// Check if all HNSW accuracy tests passed
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }
    
    /// Get the number of passed tests
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }
    
    /// Get the total number of tests
    pub fn total_count(&self) -> usize {
        self.results.len()
    }
}

impl PerformanceValidator {
    /// Create a new performance validator with default configuration
    pub fn new() -> Self {
        Self::with_config(PerformanceTestConfig::default())
    }
    
    /// Create a new performance validator with custom configuration
    pub fn with_config(config: PerformanceTestConfig) -> Self {
        let test_datasets = Self::generate_test_datasets();
        Self {
            config,
            test_datasets,
        }
    }
    
    /// Validate k-NN query performance meets latency targets
    pub fn validate_query_performance(&self, engine: &mut LDCEngine) -> Result<PerformanceTestResult> {
        let mut results = Vec::new();
        
        for dataset in &self.test_datasets {
            println!("Testing performance on dataset: {} ({} samples)", dataset.name, dataset.size);
            
            let mut latencies = Vec::new();
            
            // Warmup iterations to stabilize performance
            for i in 0..self.config.warmup_iterations {
                let query_idx = i % dataset.query_features.len();
                let query = &dataset.query_features[query_idx];
                let _ = engine.find_k_nearest_neighbors_optimized(query);
            }
            
            // Actual performance measurements
            for i in 0..self.config.test_iterations {
                let query_idx = i % dataset.query_features.len();
                let query = &dataset.query_features[query_idx];
                
                let start = Instant::now();
                let _results = engine.find_k_nearest_neighbors_optimized(query);
                let duration = start.elapsed();
                
                latencies.push(duration.as_secs_f64() * 1000.0); // Convert to milliseconds
            }
            
            // Calculate statistics
            let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
            let p95_latency = Self::calculate_percentile(&latencies, 95.0);
            let p99_latency = Self::calculate_percentile(&latencies, 99.0);
            
            // Determine target latency based on dataset size
            let target_latency = match dataset.size {
                size if size <= 1000 => self.config.target_latency_1k_samples_ms,
                size if size <= 10000 => self.config.target_latency_10k_samples_ms,
                _ => self.config.target_latency_50k_samples_ms,
            };
            
            let passed = avg_latency <= target_latency;
            
            println!("  Average latency: {:.3}ms (target: {:.3}ms) - {}", 
                    avg_latency, target_latency, if passed { "PASS" } else { "FAIL" });
            
            results.push(PerformanceTestCase {
                dataset_name: dataset.name.clone(),
                dataset_size: dataset.size,
                avg_latency_ms: avg_latency,
                p95_latency_ms: p95_latency,
                p99_latency_ms: p99_latency,
                target_latency_ms: target_latency,
                passed,
            });
        }
        
        Ok(PerformanceTestResult { results })
    }
    
    /// Validate HNSW accuracy vs exact search with 95% accuracy target
    pub fn validate_hnsw_accuracy(&self, engine: &mut LDCEngine) -> Result<HNSWAccuracyResult> {
        let mut accuracy_results = Vec::new();
        
        for dataset in &self.test_datasets {
            // Skip small datasets where HNSW is not beneficial
            if dataset.size < 1000 {
                continue;
            }
            
            println!("Testing HNSW accuracy on dataset: {} ({} samples)", dataset.name, dataset.size);
            
            let mut matches = 0;
            let mut total_queries = 0;
            
            // Test a subset of queries for efficiency
            let query_count = (dataset.query_features.len()).min(20);
            
            for query in dataset.query_features.iter().take(query_count) {
                // Store original HNSW setting
                let original_hnsw_setting = engine.config().use_hnsw_index;
                
                // Get exact k-NN results (disable HNSW)
                if let Ok(config) = engine.get_config_mut() {
                    config.use_hnsw_index = false;
                }
                let exact_results = engine.find_k_nearest_neighbors_optimized(query);
                
                // Get HNSW results (enable HNSW)
                if let Ok(config) = engine.get_config_mut() {
                    config.use_hnsw_index = true;
                }
                let hnsw_results = engine.find_k_nearest_neighbors_optimized(query);
                
                // Restore original HNSW setting
                if let Ok(config) = engine.get_config_mut() {
                    config.use_hnsw_index = original_hnsw_setting;
                }
                
                // Calculate overlap between exact and HNSW results
                let overlap = Self::calculate_knn_overlap(&exact_results, &hnsw_results);
                matches += overlap;
                total_queries += exact_results.len();
            }
            
            let accuracy = if total_queries > 0 {
                matches as f64 / total_queries as f64 * 100.0
            } else {
                0.0
            };
            
            let passed = accuracy >= self.config.target_hnsw_accuracy_percent;
            
            println!("  HNSW accuracy: {:.1}% (target: {:.1}%) - {}", 
                    accuracy, self.config.target_hnsw_accuracy_percent, 
                    if passed { "PASS" } else { "FAIL" });
            
            accuracy_results.push(HNSWAccuracyCase {
                dataset_name: dataset.name.clone(),
                dataset_size: dataset.size,
                accuracy_percent: accuracy,
                target_accuracy_percent: self.config.target_hnsw_accuracy_percent,
                passed,
            });
        }
        
        Ok(HNSWAccuracyResult { results: accuracy_results })
    }
    
    /// Calculate percentile from sorted values
    pub fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile / 100.0 * (sorted.len() - 1) as f64) as usize;
        sorted[index.min(sorted.len() - 1)]
    }
    
    /// Calculate overlap between k-NN results
    pub fn calculate_knn_overlap(exact: &[(f32, Direction)], hnsw: &[(f32, Direction)]) -> usize {
        use std::collections::HashSet;
        
        // For this implementation, we'll compare based on direction labels
        // In a more sophisticated version, we might compare actual sample indices
        let exact_labels: HashSet<_> = exact.iter().map(|(_, label)| label).collect();
        let hnsw_labels: HashSet<_> = hnsw.iter().map(|(_, label)| label).collect();
        
        exact_labels.intersection(&hnsw_labels).count()
    }
    
    /// Generate test datasets with synthetic trading data of various sizes
    fn generate_test_datasets() -> Vec<TestDataset> {
        vec![
            Self::create_synthetic_dataset("small_1k", 1000),
            Self::create_synthetic_dataset("medium_10k", 10000),
            Self::create_synthetic_dataset("large_50k", 50000),
        ]
    }
    
    /// Create a synthetic dataset with realistic trading data patterns
    fn create_synthetic_dataset(name: &str, size: usize) -> TestDataset {
        let mut rng = StdRng::seed_from_u64(42 + name.len() as u64);
        let mut samples = Vec::new();
        let mut query_features = Vec::new();
        
        for i in 0..size {
            // Generate realistic technical indicator values
            let time_factor = i as f32 * 0.01;
            
            let features = FeatureSeries {
                // RSI-like oscillator (0-100 range)
                f1: 50.0 + 30.0 * (time_factor * 0.1).sin() + rng.gen_range(-10.0..10.0),
                // WaveTrend-like oscillator (-100 to 100 range)
                f2: 50.0 * (time_factor * 0.05).cos() + rng.gen_range(-20.0..20.0),
                // CCI-like oscillator (-200 to 200 range)
                f3: 100.0 * (time_factor * 0.02).sin() + rng.gen_range(-50.0..50.0),
                // ADX-like trend strength (0-100 range)
                f4: 25.0 + 25.0 * (time_factor * 0.01).abs() + rng.gen_range(-5.0..5.0),
                // Additional feature (0-100 range)
                f5: 50.0 + 20.0 * (time_factor * 0.03).tan().abs().min(1.0) + rng.gen_range(-10.0..10.0),
            };
            
            // Ensure features are within reasonable bounds
            let bounded_features = FeatureSeries {
                f1: features.f1.clamp(0.0, 100.0),
                f2: features.f2.clamp(-100.0, 100.0),
                f3: features.f3.clamp(-200.0, 200.0),
                f4: features.f4.clamp(0.0, 100.0),
                f5: features.f5.clamp(0.0, 100.0),
            };
            
            // Generate label based on feature patterns
            let label = if bounded_features.f1 > 70.0 && bounded_features.f2 > 20.0 {
                Direction::Long
            } else if bounded_features.f1 < 30.0 && bounded_features.f2 < -20.0 {
                Direction::Short
            } else {
                Direction::Neutral
            };
            
            samples.push(TrainingSample {
                features: bounded_features.clone(),
                label,
                timestamp: i as i64,
                bar_index: i,
            });
            
            // Add some features as query samples (every 100th sample)
            if i % 100 == 0 {
                query_features.push(bounded_features);
            }
        }
        
        // Ensure we have at least a few query features
        if query_features.is_empty() {
            query_features.push(samples[0].features.clone());
        }
        
        TestDataset {
            name: name.to_string(),
            size,
            samples,
            query_features,
        }
    }
    
    /// Get the test datasets (for external use)
    pub fn get_test_datasets(&self) -> &[TestDataset] {
        &self.test_datasets
    }
    
    /// Get the configuration
    pub fn get_config(&self) -> &PerformanceTestConfig {
        &self.config
    }
}

impl Default for PerformanceValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_validator_creation() {
        let validator = PerformanceValidator::new();
        assert_eq!(validator.test_datasets.len(), 3);
        assert_eq!(validator.config.target_latency_1k_samples_ms, 0.5);
    }
    
    #[test]
    fn test_synthetic_dataset_generation() {
        let dataset = PerformanceValidator::create_synthetic_dataset("test", 100);
        assert_eq!(dataset.name, "test");
        assert_eq!(dataset.size, 100);
        assert_eq!(dataset.samples.len(), 100);
        assert!(!dataset.query_features.is_empty());
        
        // Verify feature bounds
        for sample in &dataset.samples {
            assert!(sample.features.f1 >= 0.0 && sample.features.f1 <= 100.0);
            assert!(sample.features.f2 >= -100.0 && sample.features.f2 <= 100.0);
            assert!(sample.features.f3 >= -200.0 && sample.features.f3 <= 200.0);
            assert!(sample.features.f4 >= 0.0 && sample.features.f4 <= 100.0);
            assert!(sample.features.f5 >= 0.0 && sample.features.f5 <= 100.0);
        }
    }
    
    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        let p50 = PerformanceValidator::calculate_percentile(&values, 50.0);
        let p95 = PerformanceValidator::calculate_percentile(&values, 95.0);
        
        // For 10 values (indices 0-9), 50th percentile should be around index 4.5, so value 5.0
        // For 95th percentile should be around index 8.55, so value 9.0
        assert!((p50 - 5.0).abs() < 0.1); // 50th percentile
        assert!((p95 - 9.0).abs() < 0.1); // 95th percentile
    }
    
    #[test]
    fn test_knn_overlap_calculation() {
        let exact = vec![
            (1.0, Direction::Long),
            (2.0, Direction::Short),
            (3.0, Direction::Neutral),
        ];
        
        let hnsw = vec![
            (1.1, Direction::Long),
            (2.1, Direction::Short),
            (4.0, Direction::Long),
        ];
        
        let overlap = PerformanceValidator::calculate_knn_overlap(&exact, &hnsw);
        assert_eq!(overlap, 2); // Long and Short directions match
    }
}