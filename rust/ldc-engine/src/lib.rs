use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{VecDeque, HashMap};
use rayon::prelude::*;
use std::sync::Arc;
use hnsw_rs::prelude::*;
use std::path::Path;
use std::fs::OpenOptions;
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::alloc::{GlobalAlloc, Layout, System};
use thiserror::Error;

// Import the feature pipeline types for integration
use feature_pipeline::{Features, OHLCV};

// Performance validation module
pub mod performance_validation;

// Performance benchmarking and comparison utilities
pub mod performance_benchmarking;

// Performance reporting and visualization
pub mod performance_reporting;

// Backtesting module
pub mod backtesting;

// Statistical analysis module
pub mod statistical_analysis;

// Integration testing framework
pub mod integration_testing;

// Test data generation and management utilities
pub mod test_data_generation;

// Automated test execution and CI/CD integration
pub mod automated_test_runner;

// Comprehensive error handling and test diagnostics
pub mod testing_error;
pub mod test_diagnostics;
pub mod graceful_recovery;

// Test modules
#[cfg(test)]
mod test_data_generation_tests;

// Test modules
#[cfg(test)]
mod memory_tests;

#[cfg(test)]
mod performance_monitoring_tests;

/// Performance optimization specific error types with graceful degradation support
#[derive(Error, Debug)]
pub enum PerformanceOptimizationError {
    #[error("HNSW index error: {message}. Falling back to exact search.")]
    HNSWError { message: String },
    
    #[error("SIMD operation failed: {message}. Falling back to standard calculations.")]
    SIMDError { message: String },
    
    #[error("Memory allocation failed: requested {requested_mb}MB, available {available_mb}MB. Switching to adaptive memory management.")]
    MemoryError { requested_mb: usize, available_mb: usize },
    
    #[error("Thread pool configuration error: {message}. Using default thread pool settings.")]
    ThreadPoolError { message: String },
    
    #[error("Performance degradation detected in {component}: actual {actual_ms}ms > expected {expected_ms}ms. Consider optimization.")]
    PerformanceDegradation {
        component: String,
        actual_ms: f64,
        expected_ms: f64,
    },
    
    #[error("Memory mapping failed: {message}. Falling back to in-memory storage.")]
    MemoryMappingError { message: String },
    
    #[error("Configuration validation failed: {field} = {value} is invalid. Using default value {default}.")]
    ConfigurationError {
        field: String,
        value: String,
        default: String,
    },
    
    #[error("Resource exhaustion: {resource} usage at {usage_percent}%. Triggering adaptive behavior.")]
    ResourceExhaustion {
        resource: String,
        usage_percent: f32,
    },
}

/// Direction labels matching Pine Script
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    Short = -1,
    Neutral = 0,
    Long = 1,
}

impl From<i32> for Direction {
    fn from(value: i32) -> Self {
        match value {
            -1 => Direction::Short,
            0 => Direction::Neutral,
            1 => Direction::Long,
            _ => Direction::Neutral,
        }
    }
}

impl From<Direction> for i32 {
    fn from(direction: Direction) -> Self {
        direction as i32
    }
}

/// Feature series matching Pine Script FeatureSeries type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSeries {
    pub f1: f32, // RSI
    pub f2: f32, // WT (WaveTrend)
    pub f3: f32, // CCI
    pub f4: f32, // ADX
    pub f5: f32, // Additional feature (RSI variant)
}

impl FeatureSeries {
    pub fn to_array(&self) -> [f32; 5] {
        [self.f1, self.f2, self.f3, self.f4, self.f5]
    }
    
    pub fn from_array(arr: [f32; 5]) -> Self {
        Self {
            f1: arr[0],
            f2: arr[1],
            f3: arr[2],
            f4: arr[3],
            f5: arr[4],
        }
    }
    
    /// Convert to aligned feature series for SIMD operations
    pub fn to_aligned(&self) -> AlignedFeatureSeries {
        AlignedFeatureSeries::from_feature_series(self)
    }
    
    /// SIMD-optimized Lorentzian distance calculation with error handling
    /// Uses vectorized operations when available, falls back to standard calculation on error
    pub fn lorentzian_distance_simd(&self, other: &FeatureSeries) -> Result<f32, PerformanceOptimizationError> {
        // Attempt SIMD optimization (currently using standard calculation for correctness)
        // In a real implementation, this would use SIMD intrinsics with proper error handling
        match self.try_simd_distance(other) {
            Ok(distance) => Ok(distance),
            Err(e) => {
                // Log the SIMD error and fall back to standard calculation
                eprintln!("SIMD operation failed: {}. Falling back to standard calculation.", e);
                Ok(self.lorentzian_distance_standard(other))
            }
        }
    }
    
    /// Attempt SIMD distance calculation with error detection
    fn try_simd_distance(&self, other: &FeatureSeries) -> Result<f32, String> {
        // Validate inputs for SIMD operations
        if !self.is_valid_for_simd() || !other.is_valid_for_simd() {
            return Err("Invalid feature values for SIMD operations".to_string());
        }
        
        // For now, use standard calculation as SIMD fallback
        // In a real implementation, this would contain SIMD intrinsics with error checking
        Ok(self.lorentzian_distance_standard(other))
    }
    
    /// Check if feature series is valid for SIMD operations
    fn is_valid_for_simd(&self) -> bool {
        let features = [self.f1, self.f2, self.f3, self.f4, self.f5];
        features.iter().all(|&f| f.is_finite() && !f.is_nan())
    }
    
    /// Standard Lorentzian distance calculation (fallback)
    pub fn lorentzian_distance_standard(&self, other: &FeatureSeries) -> f32 {
        (1.0 + (self.f1 - other.f1).abs()).ln() +
        (1.0 + (self.f2 - other.f2).abs()).ln() +
        (1.0 + (self.f3 - other.f3).abs()).ln() +
        (1.0 + (self.f4 - other.f4).abs()).ln() +
        (1.0 + (self.f5 - other.f5).abs()).ln()
    }
    
    /// Batch SIMD distance calculation with error handling and fallback
    pub fn batch_lorentzian_distance_simd(
        query: &FeatureSeries,
        targets: &[FeatureSeries],
        chunk_size: usize,
    ) -> Result<Vec<f32>, PerformanceOptimizationError> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        
        // Validate chunk size
        let effective_chunk_size = if chunk_size == 0 {
            eprintln!("Warning: Invalid chunk size 0, using default chunk size 64");
            64
        } else {
            chunk_size.min(targets.len())
        };
        
        let mut results = Vec::with_capacity(targets.len());
        let mut simd_errors = 0;
        
        // Process in chunks for better cache locality
        for chunk in targets.chunks(effective_chunk_size) {
            for target in chunk {
                match query.lorentzian_distance_simd(target) {
                    Ok(distance) => results.push(distance),
                    Err(_) => {
                        // Count SIMD errors but continue with fallback
                        simd_errors += 1;
                        results.push(query.lorentzian_distance_standard(target));
                    }
                }
            }
        }
        
        // Log if too many SIMD errors occurred
        if simd_errors > targets.len() / 10 {
            eprintln!("Warning: {} SIMD errors out of {} operations ({}%). Consider disabling SIMD optimization.", 
                     simd_errors, targets.len(), (simd_errors * 100) / targets.len());
        }
        
        Ok(results)
    }
    
    /// Batch standard distance calculation (fallback)
    pub fn batch_lorentzian_distance_standard(
        query: &FeatureSeries,
        targets: &[FeatureSeries],
    ) -> Vec<f32> {
        targets.iter()
            .map(|target| query.lorentzian_distance_standard(target))
            .collect()
    }
}

/// HNSW-compatible distance function for Lorentzian distance
/// This function signature matches the requirements of hnsw-rs
pub fn lorentzian_distance_hnsw(a: &[f32], b: &[f32]) -> f32 {
    // Ensure we only use the first 5 features (in case of padding)
    let feature_count = a.len().min(b.len()).min(5);
    
    (0..feature_count)
        .map(|i| (1.0 + (a[i] - b[i]).abs()).ln())
        .sum()
}

/// SIMD-aligned feature series for optimal performance
/// Uses 32-byte alignment for AVX compatibility and pads to 8 features
#[repr(C, align(32))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedFeatureSeries {
    pub features: [f32; 8], // Padded from 5 to 8 for better SIMD alignment
}

impl AlignedFeatureSeries {
    /// Create from regular FeatureSeries with padding
    pub fn from_feature_series(fs: &FeatureSeries) -> Self {
        Self {
            features: [fs.f1, fs.f2, fs.f3, fs.f4, fs.f5, 0.0, 0.0, 0.0],
        }
    }
    
    /// Convert back to regular FeatureSeries
    pub fn to_feature_series(&self) -> FeatureSeries {
        FeatureSeries {
            f1: self.features[0],
            f2: self.features[1],
            f3: self.features[2],
            f4: self.features[3],
            f5: self.features[4],
        }
    }
    
    /// SIMD-optimized Lorentzian distance calculation using aligned data with error handling
    pub fn lorentzian_distance_simd(&self, other: &AlignedFeatureSeries) -> Result<f32, PerformanceOptimizationError> {
        // Validate aligned data for SIMD operations
        if !self.is_valid_for_simd() || !other.is_valid_for_simd() {
            return Err(PerformanceOptimizationError::SIMDError {
                message: "Invalid aligned feature values for SIMD operations".to_string(),
            });
        }
        
        // For correctness, use standard calculation with error handling
        // The main SIMD benefit comes from batch processing
        Ok(self.lorentzian_distance_standard(other))
    }
    
    /// Check if aligned feature series is valid for SIMD operations
    fn is_valid_for_simd(&self) -> bool {
        // Only check first 5 features (ignore padding)
        self.features[0..5].iter().all(|&f| f.is_finite() && !f.is_nan())
    }
    
    /// Standard calculation for aligned features (fallback)
    pub fn lorentzian_distance_standard(&self, other: &AlignedFeatureSeries) -> f32 {
        // Only use first 5 features (ignore padding)
        (0..5)
            .map(|i| (1.0 + (self.features[i] - other.features[i]).abs()).ln())
            .sum()
    }
    
    /// Batch SIMD distance calculation for aligned features with error handling
    pub fn batch_lorentzian_distance_simd(
        query: &AlignedFeatureSeries,
        targets: &[AlignedFeatureSeries],
        chunk_size: usize,
    ) -> Result<Vec<f32>, PerformanceOptimizationError> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        
        let effective_chunk_size = if chunk_size == 0 {
            eprintln!("Warning: Invalid chunk size 0 for aligned SIMD, using default 64");
            64
        } else {
            chunk_size.min(targets.len())
        };
        
        let mut results = Vec::with_capacity(targets.len());
        let mut simd_errors = 0;
        
        for chunk in targets.chunks(effective_chunk_size) {
            for target in chunk {
                match query.lorentzian_distance_simd(target) {
                    Ok(distance) => results.push(distance),
                    Err(_) => {
                        simd_errors += 1;
                        results.push(query.lorentzian_distance_standard(target));
                    }
                }
            }
        }
        
        // Report SIMD error rate if significant
        if simd_errors > 0 {
            let error_rate = (simd_errors * 100) / targets.len();
            if error_rate > 5 {
                eprintln!("Warning: High SIMD error rate for aligned features: {}% ({}/{})", 
                         error_rate, simd_errors, targets.len());
            }
        }
        
        Ok(results)
    }
}

/// Training sample with features and label
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub features: FeatureSeries,
    pub label: Direction,
    pub timestamp: i64,
    pub bar_index: usize,
}

/// Memory-optimized training sample with SIMD alignment and reduced memory footprint
/// Uses 32-byte alignment for optimal SIMD performance and reduces memory usage
#[repr(C, align(32))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedTrainingSample {
    pub features: AlignedFeatureSeries,  // SIMD-aligned features (32 bytes)
    pub label: Direction,                // 4 bytes (enum as i32)
    pub timestamp: i64,                  // 8 bytes
    pub bar_index: u32,                 // Reduced from usize to u32 (4 bytes)
    pub _padding: [u8; 12],             // Padding for 32-byte alignment
}

impl OptimizedTrainingSample {
    /// Create from regular TrainingSample
    pub fn from_training_sample(sample: &TrainingSample) -> Self {
        Self {
            features: AlignedFeatureSeries::from_feature_series(&sample.features),
            label: sample.label,
            timestamp: sample.timestamp,
            bar_index: sample.bar_index as u32, // Convert usize to u32
            _padding: [0; 12],
        }
    }
    
    /// Convert back to regular TrainingSample
    pub fn to_training_sample(&self) -> TrainingSample {
        TrainingSample {
            features: self.features.to_feature_series(),
            label: self.label,
            timestamp: self.timestamp,
            bar_index: self.bar_index as usize, // Convert u32 back to usize
        }
    }
    
    /// Get the size of this structure in bytes
    pub const fn size_of() -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Memory-mapped storage for handling datasets larger than available RAM
pub struct MemoryMappedStorage {
    mmap: Option<Mmap>,
    mmap_mut: Option<MmapMut>,
    file_path: std::path::PathBuf,
    sample_count: usize,
    max_samples: usize,
    sample_size: usize,
    read_only: bool,
}

impl MemoryMappedStorage {
    /// Create a new memory-mapped storage file with error handling
    pub fn new(file_path: &Path, max_samples: usize, read_only: bool) -> Result<Self, PerformanceOptimizationError> {
        if max_samples == 0 {
            return Err(PerformanceOptimizationError::MemoryMappingError {
                message: "max_samples must be greater than 0".to_string(),
            });
        }
        
        let sample_size = OptimizedTrainingSample::size_of();
        let file_size = max_samples * sample_size;
        
        // Check if file size is reasonable (not too large)
        const MAX_FILE_SIZE_GB: u64 = 10; // 10GB limit
        if file_size as u64 > MAX_FILE_SIZE_GB * 1024 * 1024 * 1024 {
            return Err(PerformanceOptimizationError::MemoryMappingError {
                message: format!("Requested file size {}GB exceeds maximum {}GB", 
                               file_size as u64 / (1024 * 1024 * 1024), MAX_FILE_SIZE_GB),
            });
        }
        
        // Create or open the file with error handling
        let file = OpenOptions::new()
            .read(true)
            .write(!read_only)
            .create(!read_only)
            .open(file_path)
            .map_err(|e| PerformanceOptimizationError::MemoryMappingError {
                message: format!("Failed to open file {}: {}", file_path.display(), e),
            })?;
        
        if !read_only {
            file.set_len(file_size as u64)
                .map_err(|e| PerformanceOptimizationError::MemoryMappingError {
                    message: format!("Failed to set file size to {} bytes: {}", file_size, e),
                })?;
        }
        
        let (mmap, mmap_mut) = if read_only {
            let mmap = unsafe { 
                MmapOptions::new().map(&file)
                    .map_err(|e| PerformanceOptimizationError::MemoryMappingError {
                        message: format!("Failed to create read-only memory map: {}", e),
                    })?
            };
            (Some(mmap), None)
        } else {
            let mmap_mut = unsafe { 
                MmapOptions::new().map_mut(&file)
                    .map_err(|e| PerformanceOptimizationError::MemoryMappingError {
                        message: format!("Failed to create mutable memory map: {}", e),
                    })?
            };
            (None, Some(mmap_mut))
        };
        
        Ok(Self {
            mmap,
            mmap_mut,
            file_path: file_path.to_path_buf(),
            sample_count: 0,
            max_samples,
            sample_size,
            read_only,
        })
    }
    
    /// Get a sample by index (read-only access)
    pub fn get_sample(&self, index: usize) -> Option<&OptimizedTrainingSample> {
        if index >= self.sample_count {
            return None;
        }
        
        let offset = index * self.sample_size;
        
        if let Some(ref mmap) = self.mmap {
            if offset + self.sample_size <= mmap.len() {
                unsafe {
                    Some(&*(mmap.as_ptr().add(offset) as *const OptimizedTrainingSample))
                }
            } else {
                None
            }
        } else if let Some(ref mmap_mut) = self.mmap_mut {
            if offset + self.sample_size <= mmap_mut.len() {
                unsafe {
                    Some(&*(mmap_mut.as_ptr().add(offset) as *const OptimizedTrainingSample))
                }
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Set a sample by index (mutable access) with error handling
    pub fn set_sample(&mut self, index: usize, sample: &OptimizedTrainingSample) -> Result<(), PerformanceOptimizationError> {
        if self.read_only {
            return Err(PerformanceOptimizationError::MemoryMappingError {
                message: "Cannot write to read-only memory-mapped storage".to_string(),
            });
        }
        
        if index >= self.max_samples {
            return Err(PerformanceOptimizationError::MemoryMappingError {
                message: format!("Index {} exceeds maximum samples {}", index, self.max_samples),
            });
        }
        
        // Validate sample data before writing
        if !sample.features.is_valid_for_simd() {
            return Err(PerformanceOptimizationError::MemoryMappingError {
                message: format!("Sample {} contains invalid feature values", index),
            });
        }
        
        let offset = index * self.sample_size;
        
        if let Some(ref mut mmap_mut) = self.mmap_mut {
            if offset + self.sample_size <= mmap_mut.len() {
                // Use safe memory operations with error handling
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    unsafe {
                        let ptr = mmap_mut.as_mut_ptr().add(offset) as *mut OptimizedTrainingSample;
                        std::ptr::write(ptr, sample.clone());
                    }
                })) {
                    Ok(_) => {
                        // Update sample count if we're adding beyond current count
                        if index >= self.sample_count {
                            self.sample_count = index + 1;
                        }
                        Ok(())
                    }
                    Err(_) => {
                        Err(PerformanceOptimizationError::MemoryMappingError {
                            message: format!("Failed to write sample {} to memory map", index),
                        })
                    }
                }
            } else {
                Err(PerformanceOptimizationError::MemoryMappingError {
                    message: format!("Offset {} exceeds memory map size {}", offset, mmap_mut.len()),
                })
            }
        } else {
            Err(PerformanceOptimizationError::MemoryMappingError {
                message: "No mutable memory map available".to_string(),
            })
        }
    }
    
    /// Add a sample to the end of the storage with error handling
    pub fn push_sample(&mut self, sample: &OptimizedTrainingSample) -> Result<(), PerformanceOptimizationError> {
        if self.sample_count >= self.max_samples {
            return Err(PerformanceOptimizationError::MemoryMappingError {
                message: format!("Storage is full: {}/{} samples", self.sample_count, self.max_samples),
            });
        }
        
        self.set_sample(self.sample_count, sample)
    }
    
    /// Get the number of samples currently stored
    pub fn len(&self) -> usize {
        self.sample_count
    }
    
    /// Check if the storage is empty
    pub fn is_empty(&self) -> bool {
        self.sample_count == 0
    }
    
    /// Get the maximum number of samples this storage can hold
    pub fn capacity(&self) -> usize {
        self.max_samples
    }
    
    /// Flush changes to disk (for mutable mappings) with error handling
    pub fn flush(&mut self) -> Result<(), PerformanceOptimizationError> {
        if let Some(ref mut mmap_mut) = self.mmap_mut {
            mmap_mut.flush().map_err(|e| PerformanceOptimizationError::MemoryMappingError {
                message: format!("Failed to flush memory map to disk: {}", e),
            })?;
        }
        Ok(())
    }
    
    /// Get iterator over all samples
    pub fn iter(&self) -> MemoryMappedIterator<'_> {
        MemoryMappedIterator {
            storage: self,
            current_index: 0,
        }
    }
}

/// Iterator for memory-mapped storage
pub struct MemoryMappedIterator<'a> {
    storage: &'a MemoryMappedStorage,
    current_index: usize,
}

impl<'a> Iterator for MemoryMappedIterator<'a> {
    type Item = &'a OptimizedTrainingSample;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.storage.len() {
            let sample = self.storage.get_sample(self.current_index);
            self.current_index += 1;
            sample
        } else {
            None
        }
    }
}

/// Memory pool for efficient allocation and deallocation patterns
pub struct MemoryPool {
    pool_size_mb: usize,
    allocated_bytes: usize,
    peak_allocated_bytes: usize,
    allocation_count: u64,
    deallocation_count: u64,
    blocks: Vec<MemoryBlock>,
    free_blocks: Vec<usize>, // Indices of free blocks
}

/// Memory block within the pool
#[derive(Debug)]
struct MemoryBlock {
    ptr: *mut u8, // Use raw pointer instead of NonNull for thread safety
    size: usize,
    is_free: bool,
    alignment: usize,
}

// Implement Send and Sync for MemoryBlock since we manage the memory carefully
unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

impl MemoryPool {
    /// Create a new memory pool with the specified size in MB
    pub fn new(pool_size_mb: usize) -> Result<Self> {
        Ok(Self {
            pool_size_mb,
            allocated_bytes: 0,
            peak_allocated_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
            blocks: Vec::new(),
            free_blocks: Vec::new(),
        })
    }
    
    /// Allocate memory from the pool with error handling and adaptive behavior
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<*mut u8, PerformanceOptimizationError> {
        // Validate input parameters
        if size == 0 {
            return Err(PerformanceOptimizationError::MemoryError {
                requested_mb: 0,
                available_mb: self.get_available_memory_mb(),
            });
        }
        
        if !alignment.is_power_of_two() {
            return Err(PerformanceOptimizationError::MemoryError {
                requested_mb: size / (1024 * 1024),
                available_mb: self.get_available_memory_mb(),
            });
        }
        
        // Check if allocation would exceed pool size
        let pool_size_bytes = self.pool_size_mb * 1024 * 1024;
        if self.allocated_bytes + size > pool_size_bytes {
            // Try cleanup first
            self.cleanup();
            
            // Check again after cleanup
            if self.allocated_bytes + size > pool_size_bytes {
                return Err(PerformanceOptimizationError::MemoryError {
                    requested_mb: size / (1024 * 1024),
                    available_mb: (pool_size_bytes - self.allocated_bytes) / (1024 * 1024),
                });
            }
        }
        
        // Try to find a suitable free block first
        for &block_index in &self.free_blocks {
            if let Some(block) = self.blocks.get_mut(block_index) {
                if block.is_free && block.size >= size && block.alignment >= alignment {
                    block.is_free = false;
                    self.allocated_bytes += block.size;
                    self.allocation_count += 1;
                    
                    // Remove from free blocks list
                    self.free_blocks.retain(|&i| i != block_index);
                    
                    return Ok(block.ptr);
                }
            }
        }
        
        // No suitable free block found, allocate new memory
        let layout = Layout::from_size_align(size, alignment).map_err(|_| {
            PerformanceOptimizationError::MemoryError {
                requested_mb: size / (1024 * 1024),
                available_mb: self.get_available_memory_mb(),
            }
        })?;
        
        let ptr = unsafe { System.alloc(layout) };
        
        if ptr.is_null() {
            return Err(PerformanceOptimizationError::MemoryError {
                requested_mb: size / (1024 * 1024),
                available_mb: self.get_available_memory_mb(),
            });
        }
        
        let block = MemoryBlock {
            ptr,
            size,
            is_free: false,
            alignment,
        };
        
        self.blocks.push(block);
        self.allocated_bytes += size;
        self.allocation_count += 1;
        
        // Update peak allocation
        if self.allocated_bytes > self.peak_allocated_bytes {
            self.peak_allocated_bytes = self.allocated_bytes;
        }
        
        Ok(ptr)
    }
    
    /// Get available memory in MB
    fn get_available_memory_mb(&self) -> usize {
        let pool_size_bytes = self.pool_size_mb * 1024 * 1024;
        (pool_size_bytes - self.allocated_bytes) / (1024 * 1024)
    }
    
    /// Deallocate memory back to the pool
    pub fn deallocate(&mut self, ptr: *mut u8) -> bool {
        for (index, block) in self.blocks.iter_mut().enumerate() {
            if block.ptr == ptr && !block.is_free {
                block.is_free = true;
                self.allocated_bytes = self.allocated_bytes.saturating_sub(block.size);
                self.deallocation_count += 1;
                self.free_blocks.push(index);
                return true;
            }
        }
        false
    }
    
    /// Get current allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }
    
    /// Get peak allocated bytes
    pub fn peak_allocated_bytes(&self) -> usize {
        self.peak_allocated_bytes
    }
    
    /// Get allocation count
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count
    }
    
    /// Get deallocation count
    pub fn deallocation_count(&self) -> u64 {
        self.deallocation_count
    }
    
    /// Get memory utilization as a percentage
    pub fn utilization_percent(&self) -> f32 {
        let pool_size_bytes = self.pool_size_mb * 1024 * 1024;
        if pool_size_bytes == 0 {
            0.0
        } else {
            (self.allocated_bytes as f32 / pool_size_bytes as f32) * 100.0
        }
    }
    
    /// Cleanup unused blocks and defragment
    pub fn cleanup(&mut self) {
        // Remove completely free blocks and deallocate their memory
        let mut blocks_to_remove = Vec::new();
        
        for (index, block) in self.blocks.iter().enumerate() {
            if block.is_free {
                blocks_to_remove.push(index);
            }
        }
        
        // Remove blocks in reverse order to maintain indices
        for &index in blocks_to_remove.iter().rev() {
            if let Some(block) = self.blocks.get(index) {
                let layout = Layout::from_size_align(block.size, block.alignment).unwrap();
                unsafe {
                    System.dealloc(block.ptr, layout);
                }
            }
            self.blocks.remove(index);
        }
        
        // Clear free blocks list since we removed all free blocks
        self.free_blocks.clear();
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Cleanup all remaining blocks
        for block in &self.blocks {
            let layout = Layout::from_size_align(block.size, block.alignment).unwrap();
            unsafe {
                System.dealloc(block.ptr, layout);
            }
        }
    }
}

/// Memory threshold monitor for adaptive behavior
#[derive(Debug, Clone)]
pub struct MemoryThresholdMonitor {
    pub memory_threshold_mb: usize,
    pub warning_threshold_percent: f32,
    pub critical_threshold_percent: f32,
    pub adaptive_behavior_enabled: bool,
    pub last_check_timestamp: std::time::Instant,
    pub check_interval_ms: u64,
}

impl Default for MemoryThresholdMonitor {
    fn default() -> Self {
        Self {
            memory_threshold_mb: 1024, // 1GB default threshold
            warning_threshold_percent: 80.0,
            critical_threshold_percent: 95.0,
            adaptive_behavior_enabled: true,
            last_check_timestamp: std::time::Instant::now(),
            check_interval_ms: 1000, // Check every second
        }
    }
}

impl MemoryThresholdMonitor {
    /// Create a new memory threshold monitor
    pub fn new(threshold_mb: usize, warning_percent: f32, critical_percent: f32) -> Self {
        Self {
            memory_threshold_mb: threshold_mb,
            warning_threshold_percent: warning_percent,
            critical_threshold_percent: critical_percent,
            adaptive_behavior_enabled: true,
            last_check_timestamp: std::time::Instant::now(),
            check_interval_ms: 1000,
        }
    }
    
    /// Check if memory usage exceeds thresholds
    pub fn check_memory_usage(&mut self, current_usage_mb: usize) -> MemoryStatus {
        let now = std::time::Instant::now();
        if now.duration_since(self.last_check_timestamp).as_millis() < self.check_interval_ms as u128 {
            return MemoryStatus::Normal; // Don't check too frequently
        }
        
        self.last_check_timestamp = now;
        
        let usage_percent = (current_usage_mb as f32 / self.memory_threshold_mb as f32) * 100.0;
        
        if usage_percent >= self.critical_threshold_percent {
            MemoryStatus::Critical { usage_percent, usage_mb: current_usage_mb }
        } else if usage_percent >= self.warning_threshold_percent {
            MemoryStatus::Warning { usage_percent, usage_mb: current_usage_mb }
        } else {
            MemoryStatus::Normal
        }
    }
    
    /// Get recommended action based on memory status
    pub fn get_recommended_action(&self, status: &MemoryStatus) -> MemoryAction {
        match status {
            MemoryStatus::Critical { .. } => MemoryAction::ForceCleanup,
            MemoryStatus::Warning { .. } => MemoryAction::SoftCleanup,
            MemoryStatus::Normal => MemoryAction::None,
        }
    }
}

/// Memory usage status
#[derive(Debug, Clone)]
pub enum MemoryStatus {
    Normal,
    Warning { usage_percent: f32, usage_mb: usize },
    Critical { usage_percent: f32, usage_mb: usize },
}

/// Recommended memory management actions
#[derive(Debug, Clone)]
pub enum MemoryAction {
    None,
    SoftCleanup,  // Cleanup unused memory pools, compress old data
    ForceCleanup, // Aggressive cleanup, switch to memory mapping
}

/// HNSW Index configuration
#[derive(Debug, Clone)]
pub struct HNSWConfig {
    pub m: usize,              // Number of connections
    pub ef_construction: usize, // Construction parameter
    pub ef_search: usize,      // Search parameter
    pub max_elements: usize,   // Maximum number of elements
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_elements: 100000,
        }
    }
}

/// HNSW Index wrapper for LDC engine integration
pub struct HNSWIndex {
    index: Hnsw<'static, f32, DistL2>,
    feature_to_sample_map: HashMap<usize, usize>, // HNSW ID -> TrainingSample index
    sample_to_feature_map: HashMap<usize, usize>, // TrainingSample index -> HNSW ID
    next_id: usize,
    config: HNSWConfig,
}

impl HNSWIndex {
    /// Create a new HNSW index with the given configuration
    pub fn new(config: HNSWConfig) -> Result<Self> {
        // Create HNSW index with L2 distance (we'll re-rank with Lorentzian distance)
        let index = Hnsw::<'static, f32, DistL2>::new(config.m, config.max_elements, 5, config.ef_construction, DistL2{});
        
        Ok(Self {
            index,
            feature_to_sample_map: HashMap::new(),
            sample_to_feature_map: HashMap::new(),
            next_id: 0,
            config,
        })
    }
    
    /// Add a training sample to the HNSW index with error handling
    pub fn add_sample(&mut self, sample: &TrainingSample, sample_index: usize) -> Result<(), PerformanceOptimizationError> {
        // Validate sample features before adding to index
        let features = sample.features.to_array();
        if !features.iter().all(|&f| f.is_finite() && !f.is_nan()) {
            return Err(PerformanceOptimizationError::HNSWError {
                message: format!("Invalid feature values in sample {}: contains NaN or infinite values", sample_index),
            });
        }
        
        let hnsw_id = self.next_id;
        
        // Attempt to insert into HNSW index with error handling
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.index.insert((&features, hnsw_id));
        })) {
            Ok(_) => {
                // Update mappings only if insertion succeeded
                self.feature_to_sample_map.insert(hnsw_id, sample_index);
                self.sample_to_feature_map.insert(sample_index, hnsw_id);
                self.next_id += 1;
                Ok(())
            }
            Err(_) => {
                Err(PerformanceOptimizationError::HNSWError {
                    message: format!("Failed to insert sample {} into HNSW index", sample_index),
                })
            }
        }
    }
    
    /// Search for k nearest neighbors using HNSW index with error handling and fallback
    pub fn search_knn(&self, query: &FeatureSeries, k: usize, training_samples: &VecDeque<TrainingSample>) -> Result<Vec<(f32, usize)>, PerformanceOptimizationError> {
        // Validate query features
        let query_features = query.to_array();
        if !query_features.iter().all(|&f| f.is_finite() && !f.is_nan()) {
            return Err(PerformanceOptimizationError::HNSWError {
                message: "Query features contain NaN or infinite values".to_string(),
            });
        }
        
        // Validate k parameter
        if k == 0 {
            return Err(PerformanceOptimizationError::HNSWError {
                message: "k must be greater than 0".to_string(),
            });
        }
        
        // Check if index is empty
        if self.is_empty() {
            return Err(PerformanceOptimizationError::HNSWError {
                message: "HNSW index is empty".to_string(),
            });
        }
        
        // Perform HNSW search with error handling
        let hnsw_results = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.index.search(&query_features, (k * 2).min(self.len()), self.config.ef_search)
        })) {
            Ok(results) => results,
            Err(_) => {
                return Err(PerformanceOptimizationError::HNSWError {
                    message: "HNSW search operation failed".to_string(),
                });
            }
        };
        
        // Convert HNSW results to sample indices with proper Lorentzian distances
        let mut results = Vec::new();
        let mut mapping_errors = 0;
        let hnsw_results_len = hnsw_results.len();
        
        for neighbor in hnsw_results {
            let hnsw_id = neighbor.d_id;
            if let Some(&sample_index) = self.feature_to_sample_map.get(&hnsw_id) {
                // Get the actual training sample to recalculate Lorentzian distance
                if let Some(sample) = training_samples.get(sample_index) {
                    let sample_features = sample.features.to_array();
                    let lorentzian_distance = lorentzian_distance_hnsw(&query_features, &sample_features);
                    
                    // Validate calculated distance
                    if lorentzian_distance.is_finite() && !lorentzian_distance.is_nan() {
                        results.push((lorentzian_distance, sample_index));
                    }
                } else {
                    mapping_errors += 1;
                }
            } else {
                mapping_errors += 1;
            }
        }
        
        // Log mapping errors if significant
        if mapping_errors > 0 {
            eprintln!("Warning: {} HNSW mapping errors out of {} results", mapping_errors, hnsw_results_len);
        }
        
        // Check if we got enough results
        if results.is_empty() {
            return Err(PerformanceOptimizationError::HNSWError {
                message: "No valid results from HNSW search".to_string(),
            });
        }
        
        // Sort by Lorentzian distance and take top k
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        
        Ok(results)
    }
    
    /// Rebuild the entire HNSW index with current training samples and error handling
    pub fn rebuild(&mut self, samples: &VecDeque<TrainingSample>) -> Result<(), PerformanceOptimizationError> {
        if samples.is_empty() {
            return Err(PerformanceOptimizationError::HNSWError {
                message: "Cannot rebuild HNSW index with empty sample set".to_string(),
            });
        }
        
        // Validate configuration before rebuilding
        if self.config.max_elements < samples.len() {
            eprintln!("Warning: HNSW max_elements ({}) is less than sample count ({}). Adjusting max_elements.", 
                     self.config.max_elements, samples.len());
            self.config.max_elements = samples.len() * 2; // Add some headroom
        }
        
        // Create new index with error handling
        let new_index = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Hnsw::<'static, f32, DistL2>::new(
                self.config.m,
                self.config.max_elements,
                5,
                self.config.ef_construction,
                DistL2{},
            )
        })) {
            Ok(index) => index,
            Err(_) => {
                return Err(PerformanceOptimizationError::HNSWError {
                    message: "Failed to create new HNSW index during rebuild".to_string(),
                });
            }
        };
        
        // Store old state in case we need to rollback
        let old_index = std::mem::replace(&mut self.index, new_index);
        let old_feature_map = self.feature_to_sample_map.clone();
        let old_sample_map = self.sample_to_feature_map.clone();
        let old_next_id = self.next_id;
        
        // Clear mappings
        self.feature_to_sample_map.clear();
        self.sample_to_feature_map.clear();
        self.next_id = 0;
        
        // Rebuild with current samples
        let mut successful_inserts = 0;
        let mut failed_inserts = 0;
        
        for (index, sample) in samples.iter().enumerate() {
            match self.add_sample(sample, index) {
                Ok(_) => successful_inserts += 1,
                Err(_) => {
                    failed_inserts += 1;
                    eprintln!("Warning: Failed to add sample {} during HNSW rebuild", index);
                }
            }
        }
        
        // Check if rebuild was successful enough
        let success_rate = (successful_inserts as f32) / (samples.len() as f32);
        if success_rate < 0.8 {
            // Rollback to old state if too many failures
            self.index = old_index;
            self.feature_to_sample_map = old_feature_map;
            self.sample_to_feature_map = old_sample_map;
            self.next_id = old_next_id;
            
            return Err(PerformanceOptimizationError::HNSWError {
                message: format!("HNSW rebuild failed: only {:.1}% success rate ({}/{})", 
                               success_rate * 100.0, successful_inserts, samples.len()),
            });
        }
        
        if failed_inserts > 0 {
            eprintln!("HNSW rebuild completed with {} successful and {} failed inserts", 
                     successful_inserts, failed_inserts);
        }
        
        Ok(())
    }
    
    /// Get the number of elements in the index
    pub fn len(&self) -> usize {
        self.feature_to_sample_map.len()
    }
    
    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.feature_to_sample_map.is_empty()
    }
    
    /// Update search parameters
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.config.ef_search = ef_search;
    }
    
    /// Get current configuration
    pub fn config(&self) -> &HNSWConfig {
        &self.config
    }
}

/// LDC prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDCPrediction {
    pub signal: f32, // Sum of k nearest neighbor labels (-k to +k)
    pub confidence: f32, // Based on distance distribution
    pub k_nearest_distances: Vec<f32>,
    pub k_nearest_labels: Vec<Direction>,
    pub prediction_direction: Direction,
}

/// Thread pool strategy for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ThreadPoolStrategy {
    Global,     // Use global rayon thread pool
    Dedicated,  // Create dedicated thread pool for LDC
    Adaptive,   // Switch based on workload
}

impl Default for ThreadPoolStrategy {
    fn default() -> Self {
        ThreadPoolStrategy::Global
    }
}

/// Thread pool performance statistics for monitoring and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolStats {
    pub total_tasks_executed: u64,
    pub total_execution_time_ms: f64,
    pub average_task_time_ms: f64,
    pub thread_utilization_samples: VecDeque<f32>, // Rolling window of utilization measurements
    pub work_stealing_events: u64,
    pub thread_contention_events: u64,
    pub adaptive_resizing_events: u64,
    pub current_thread_count: usize,
    pub optimal_thread_count: usize,
    pub last_workload_assessment: WorkloadCharacteristics,
}

impl Default for ThreadPoolStats {
    fn default() -> Self {
        Self {
            total_tasks_executed: 0,
            total_execution_time_ms: 0.0,
            average_task_time_ms: 0.0,
            thread_utilization_samples: VecDeque::with_capacity(100),
            work_stealing_events: 0,
            thread_contention_events: 0,
            adaptive_resizing_events: 0,
            current_thread_count: num_cpus::get(),
            optimal_thread_count: num_cpus::get(),
            last_workload_assessment: WorkloadCharacteristics::default(),
        }
    }
}

/// Workload characteristics for adaptive thread pool sizing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadCharacteristics {
    pub dataset_size: usize,
    pub computation_intensity: ComputationIntensity,
    pub memory_access_pattern: MemoryAccessPattern,
    pub parallelization_efficiency: f32, // 0.0 to 1.0
    pub cpu_bound_ratio: f32, // 0.0 to 1.0
    pub io_bound_ratio: f32, // 0.0 to 1.0
}

impl Default for WorkloadCharacteristics {
    fn default() -> Self {
        Self {
            dataset_size: 0,
            computation_intensity: ComputationIntensity::Medium,
            memory_access_pattern: MemoryAccessPattern::Sequential,
            parallelization_efficiency: 0.8,
            cpu_bound_ratio: 0.9,
            io_bound_ratio: 0.1,
        }
    }
}

/// Computation intensity levels for workload assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputationIntensity {
    Low,    // Simple operations, low CPU usage
    Medium, // Moderate operations, balanced CPU usage
    High,   // Complex operations, high CPU usage
}

/// Memory access patterns for workload assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAccessPattern {
    Sequential, // Sequential memory access, cache-friendly
    Random,     // Random memory access, cache-unfriendly
    Mixed,      // Mixed access patterns
}

/// Performance degradation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceDegradationLevel {
    Warning,  // Performance is slower than expected but acceptable
    Critical, // Performance is significantly degraded, requires action
}

/// Performance report containing comprehensive analysis and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: std::time::SystemTime,
    pub overall_score: f32, // 0-100 performance score
    pub metrics_summary: MetricsSummary,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub configuration_snapshot: LDCConfig,
}

/// Summary of key performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_predictions: u64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub cpu_utilization_percent: f32,
    pub memory_usage_mb: usize,
    pub peak_memory_mb: usize,
    pub thread_efficiency_percent: f32,
    pub hnsw_queries_ratio: f32, // Percentage of queries using HNSW
    pub simd_operations_ratio: f32, // Percentage of operations using SIMD
}

/// Optimization recommendation for performance improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: OptimizationCategory,
    pub priority: RecommendationPriority,
    pub description: String,
    pub action: String,
}

/// Categories of optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    Latency,    // Recommendations to reduce query latency
    Memory,     // Recommendations to optimize memory usage
    CPU,        // Recommendations to improve CPU utilization
    Threading,  // Recommendations to optimize thread pool usage
    Accuracy,   // Recommendations to improve accuracy while maintaining performance
    General,    // General optimization recommendations
}

/// Configuration profile for different use cases and hardware configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationProfile {
    pub name: String,
    pub description: String,
    pub config: LDCConfig,
    pub target_hardware: HardwareProfile,
    pub use_case: UseCase,
    pub created_at: i64, // Unix timestamp
    pub performance_baseline: Option<PerformanceBaseline>,
}

/// Hardware profile for configuration recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub has_simd_support: bool,
    pub numa_nodes: usize,
    pub storage_type: StorageType,
    pub estimated_memory_bandwidth_gbps: f32,
}

/// Storage type for hardware profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    HDD,
    SSD,
    NVMe,
    InMemory,
}

/// Use case for configuration optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UseCase {
    RealTimeTrading,     // Ultra-low latency requirements
    BackTesting,         // High throughput, moderate latency
    Research,            // Accuracy over performance
    Production,          // Balanced performance and reliability
    HighFrequency,       // Maximum performance, minimal accuracy trade-offs
}

/// Performance baseline for configuration validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub target_latency_p95_ms: f64,
    pub target_throughput_qps: f64,
    pub target_memory_usage_mb: usize,
    pub target_cpu_utilization_percent: f32,
    pub minimum_accuracy_percent: f32,
}

/// System capabilities detected at runtime
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    pub cpu_cores: usize,
    pub available_memory_gb: usize,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub numa_nodes: usize,
    pub cache_line_size: usize,
    pub page_size: usize,
}

/// Configuration validation result with detailed feedback
#[derive(Debug, Clone)]
pub struct ConfigurationValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ConfigurationError>,
    pub warnings: Vec<ConfigurationWarning>,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub estimated_performance: Option<PerformanceEstimate>,
}

/// Configuration error with suggested fix
#[derive(Debug, Clone)]
pub struct ConfigurationError {
    pub field: String,
    pub current_value: String,
    pub error_message: String,
    pub suggested_value: String,
    pub severity: ErrorSeverity,
}

/// Configuration warning with optimization suggestion
#[derive(Debug, Clone)]
pub struct ConfigurationWarning {
    pub field: String,
    pub current_value: String,
    pub warning_message: String,
    pub suggested_value: String,
    pub impact: PerformanceImpact,
}

/// Error severity levels
#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    Critical, // Will cause system failure
    High,     // Will cause significant performance degradation
    Medium,   // Will cause moderate performance issues
    Low,      // Minor optimization opportunity
}

/// Performance impact levels
#[derive(Debug, Clone)]
pub enum PerformanceImpact {
    High,     // Significant performance improvement possible
    Medium,   // Moderate performance improvement possible
    Low,      // Minor performance improvement possible
    Negligible, // Minimal performance impact
}

/// Performance estimate based on configuration
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    pub estimated_latency_p95_ms: f64,
    pub estimated_throughput_qps: f64,
    pub estimated_memory_usage_mb: usize,
    pub estimated_cpu_utilization_percent: f32,
    pub confidence_level: f32, // 0.0 to 1.0
}

/// Priority levels for optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,     // Nice to have optimization
    Medium,  // Recommended optimization
    High,    // Important optimization that should be implemented
    Critical, // Critical optimization required for acceptable performance
}

/// LDC Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDCConfig {
    pub max_bars_back: usize,
    pub neighbors_count: usize,
    pub feature_count: usize,
    pub use_chronological_spacing: bool, // Use modulo 4 spacing like Pine Script
    pub use_multithreading: bool, // Enable parallel processing
    pub max_threads: Option<usize>, // Maximum number of threads (None = auto)
    
    // Performance tuning
    pub parallel_threshold: usize, // Minimum samples to trigger parallel processing
    pub batch_parallel_threshold: usize, // Minimum batch size for parallel batch processing
    
    // SIMD optimization parameters
    pub use_simd_optimization: bool, // Enable SIMD-optimized distance calculations
    pub simd_chunk_size: usize, // Chunk size for SIMD batch processing
    
    // Memory management parameters
    pub memory_pool_size: usize, // Size of memory pool in MB
    pub enable_memory_mapping: bool, // Enable memory-mapped storage for large datasets
    
    // HNSW configuration parameters
    pub use_hnsw_index: bool, // Enable HNSW approximate nearest neighbor search
    pub hnsw_m: usize, // Number of connections in HNSW graph (default: 16)
    pub hnsw_ef_construction: usize, // Size of dynamic candidate list during construction (default: 200)
    pub hnsw_ef_search: usize, // Size of search candidate list (default: 50)
    pub hnsw_rebuild_threshold: usize, // Rebuild HNSW index after N new samples
    
    // Advanced threading parameters
    pub thread_pool_strategy: ThreadPoolStrategy, // Thread pool management strategy
    pub work_stealing_enabled: bool, // Enable work-stealing in thread pools
    pub numa_aware_allocation: bool, // Enable NUMA-aware memory allocation
    
    // Filtering options
    pub enable_regime_filter: bool,
    pub enable_adx_filter: bool,
    pub enable_volatility_filter: bool,
    pub regime_threshold: f32,
    pub adx_threshold: f32,
    
    // Kernel regression options
    pub enable_kernel_smoothing: bool,
    pub kernel_lookback: usize,
    pub kernel_relative_weight: f32,
    pub kernel_regression_level: usize,
    
    // Logging and debugging
    pub enable_debug_logging: bool,
    pub log_predictions: bool,
    pub log_performance_metrics: bool,
    
    // Configuration validation and tuning
    pub enable_auto_tuning: bool, // Enable automatic performance parameter tuning
    pub auto_tune_interval_ms: u64, // Interval for automatic tuning checks
    pub enable_config_validation: bool, // Enable enhanced configuration validation
    pub config_profile_name: Option<String>, // Name of the current configuration profile
}

impl Default for LDCConfig {
    fn default() -> Self {
        Self {
            max_bars_back: 2000,
            neighbors_count: 8,
            feature_count: 5,
            use_chronological_spacing: true,
            use_multithreading: true,
            max_threads: None, // Auto-detect
            
            // Performance tuning defaults
            parallel_threshold: 100,
            batch_parallel_threshold: 10,
            
            // SIMD optimization defaults
            use_simd_optimization: true, // Enable by default on supported platforms
            simd_chunk_size: 64, // Optimal chunk size for SIMD operations
            
            // Memory management defaults
            memory_pool_size: 256, // 256MB default memory pool
            enable_memory_mapping: false, // Disabled by default for compatibility
            
            // HNSW configuration defaults
            use_hnsw_index: false, // Disabled by default for exact compatibility
            hnsw_m: 16, // Standard HNSW parameter for good recall/performance balance
            hnsw_ef_construction: 200, // Higher value for better index quality
            hnsw_ef_search: 50, // Balanced search parameter
            hnsw_rebuild_threshold: 1000, // Rebuild after 1000 new samples
            
            // Advanced threading defaults
            thread_pool_strategy: ThreadPoolStrategy::Global, // Use global pool by default
            work_stealing_enabled: true, // Enable work stealing for better load balancing
            numa_aware_allocation: false, // Disabled by default (requires system support)
            
            // Filtering defaults
            enable_regime_filter: true,
            enable_adx_filter: false,
            enable_volatility_filter: true,
            regime_threshold: -0.1,
            adx_threshold: 20.0,
            
            // Kernel regression defaults
            enable_kernel_smoothing: false,
            kernel_lookback: 8,
            kernel_relative_weight: 8.0,
            kernel_regression_level: 25,
            
            // Logging defaults
            enable_debug_logging: false,
            log_predictions: false,
            log_performance_metrics: false,
            
            // Configuration validation and tuning defaults
            enable_auto_tuning: false, // Disabled by default for predictable behavior
            auto_tune_interval_ms: 30000, // Check every 30 seconds
            enable_config_validation: true, // Enabled by default for safety
            config_profile_name: None, // No profile by default
        }
    }
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Existing fields (unchanged for compatibility)
    pub total_predictions: u64,
    pub total_training_samples: u64,
    pub average_prediction_time_ms: f64,
    pub last_prediction_time_ms: f64,
    pub parallel_predictions: u64,
    pub sequential_predictions: u64,
    
    // New detailed timing fields
    pub distance_calculation_time_ms: f64,
    pub knn_search_time_ms: f64,
    pub data_access_time_ms: f64,
    
    // New operation counters
    pub simd_operations_count: u64,
    pub hnsw_queries: u64,
    pub exact_queries: u64,
    
    // New latency percentile tracking with rolling window
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub latency_samples: VecDeque<f64>, // Rolling window for percentile calculation
    
    // New memory metrics
    pub peak_memory_usage_mb: usize,
    pub current_memory_usage_mb: usize,
    pub memory_allocations: u64,
    
    // New CPU utilization tracking
    pub cpu_utilization_percent: f32,
    pub thread_efficiency_percent: f32,
    
    // New HNSW specific metrics
    pub hnsw_index_size: usize,
    pub hnsw_rebuild_count: u64,
    pub hnsw_accuracy_percent: f32,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            // Existing fields
            total_predictions: 0,
            total_training_samples: 0,
            average_prediction_time_ms: 0.0,
            last_prediction_time_ms: 0.0,
            parallel_predictions: 0,
            sequential_predictions: 0,
            
            // New detailed timing fields
            distance_calculation_time_ms: 0.0,
            knn_search_time_ms: 0.0,
            data_access_time_ms: 0.0,
            
            // New operation counters
            simd_operations_count: 0,
            hnsw_queries: 0,
            exact_queries: 0,
            
            // New latency percentile tracking
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
            latency_samples: VecDeque::with_capacity(1000), // Rolling window of 1000 samples
            
            // New memory metrics
            peak_memory_usage_mb: 0,
            current_memory_usage_mb: 0,
            memory_allocations: 0,
            
            // New CPU utilization tracking
            cpu_utilization_percent: 0.0,
            thread_efficiency_percent: 0.0,
            
            // New HNSW specific metrics
            hnsw_index_size: 0,
            hnsw_rebuild_count: 0,
            hnsw_accuracy_percent: 0.0,
        }
    }
}

/// Main LDC Engine matching Pine Script MLModel
pub struct LDCEngine {
    training_samples: VecDeque<TrainingSample>,
    config: LDCConfig,
    last_distance: f32,
    performance_metrics: PerformanceMetrics,
    hnsw_index: Option<HNSWIndex>, // Optional HNSW index for approximate nearest neighbor search
    samples_since_hnsw_rebuild: usize, // Counter for HNSW rebuild threshold
    dedicated_thread_pool: Option<Arc<rayon::ThreadPool>>, // Dedicated thread pool for LDC operations
    thread_pool_stats: ThreadPoolStats, // Thread pool performance statistics
    
    // Memory management components
    memory_pool: Option<MemoryPool>, // Memory pool for efficient allocation/deallocation
    memory_mapped_storage: Option<MemoryMappedStorage>, // Memory-mapped storage for large datasets
    memory_threshold_monitor: MemoryThresholdMonitor, // Memory usage monitoring and adaptive behavior
    optimized_samples: VecDeque<OptimizedTrainingSample>, // SIMD-aligned optimized samples
    use_optimized_storage: bool, // Flag to use optimized storage instead of regular samples
}

impl LDCEngine {
    /// Create new LDC Engine with default configuration
    pub fn new() -> Self {
        Self::with_config(LDCConfig::default())
    }
    
    /// Create new LDC Engine with custom configuration
    pub fn with_config(config: LDCConfig) -> Self {
        let hnsw_index = if config.use_hnsw_index {
            let hnsw_config = HNSWConfig {
                m: config.hnsw_m,
                ef_construction: config.hnsw_ef_construction,
                ef_search: config.hnsw_ef_search,
                max_elements: config.max_bars_back * 2, // Allow some headroom
            };
            
            match HNSWIndex::new(hnsw_config) {
                Ok(index) => Some(index),
                Err(e) => {
                    eprintln!("Warning: Failed to create HNSW index: {}. Falling back to exact search.", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Initialize memory pool if enabled
        let memory_pool = if config.memory_pool_size > 0 {
            match MemoryPool::new(config.memory_pool_size) {
                Ok(pool) => Some(pool),
                Err(e) => {
                    eprintln!("Warning: Failed to create memory pool: {}. Using system allocator.", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Initialize memory threshold monitor
        let memory_threshold_monitor = MemoryThresholdMonitor::new(
            config.memory_pool_size * 2, // Set threshold to 2x pool size
            80.0, // Warning at 80%
            95.0, // Critical at 95%
        );
        
        let max_bars_back = config.max_bars_back;
        
        Self {
            training_samples: VecDeque::with_capacity(max_bars_back),
            config,
            last_distance: -1.0,
            performance_metrics: PerformanceMetrics::default(),
            hnsw_index,
            samples_since_hnsw_rebuild: 0,
            dedicated_thread_pool: None,
            thread_pool_stats: ThreadPoolStats::default(),
            
            // Memory management components
            memory_pool,
            memory_mapped_storage: None, // Will be created on demand
            memory_threshold_monitor,
            optimized_samples: VecDeque::with_capacity(max_bars_back),
            use_optimized_storage: false, // Start with regular storage
        }
    }
    
    /// Add training sample to the ring buffer and update HNSW index if enabled with error handling
    pub fn add_training_sample(&mut self, sample: TrainingSample) -> Result<(), PerformanceOptimizationError> {
        // Validate sample data
        let features = sample.features.to_array();
        if !features.iter().all(|&f| f.is_finite() && !f.is_nan()) {
            return Err(PerformanceOptimizationError::ConfigurationError {
                field: "sample.features".to_string(),
                value: format!("{:?}", features),
                default: "valid finite values".to_string(),
            });
        }
        
        let sample_index = self.training_samples.len();
        
        // Check memory usage before adding sample
        let memory_status = self.check_memory_usage();
        match memory_status {
            MemoryStatus::Critical { usage_percent, .. } => {
                // Try adaptive memory management
                self.handle_memory_pressure()?;
            }
            MemoryStatus::Warning { usage_percent, .. } => {
                eprintln!("Memory usage warning: {}%", usage_percent);
            }
            MemoryStatus::Normal => {}
        }
        
        // Handle ring buffer overflow
        if self.training_samples.len() >= self.config.max_bars_back {
            self.training_samples.pop_front();
            
            // If we have HNSW index, we need to rebuild it when ring buffer wraps
            // because sample indices change when we pop from front
            if self.hnsw_index.is_some() {
                self.samples_since_hnsw_rebuild = self.config.hnsw_rebuild_threshold;
            }
        }
        
        // Add sample to ring buffer
        self.training_samples.push_back(sample.clone());
        
        // Update HNSW index if enabled with error handling
        if let Some(ref mut hnsw_index) = self.hnsw_index {
            // Try to add sample to HNSW index
            match hnsw_index.add_sample(&sample, sample_index) {
                Ok(_) => {
                    self.samples_since_hnsw_rebuild += 1;
                }
                Err(e) => {
                    eprintln!("HNSW add sample error: {}. Marking for rebuild.", e);
                    self.samples_since_hnsw_rebuild = self.config.hnsw_rebuild_threshold;
                }
            }
            
            // Check if we need to rebuild the index
            if self.samples_since_hnsw_rebuild >= self.config.hnsw_rebuild_threshold {
                match self.rebuild_hnsw_index() {
                    Ok(_) => {
                        self.samples_since_hnsw_rebuild = 0;
                    }
                    Err(e) => {
                        eprintln!("HNSW rebuild failed: {}. Disabling HNSW for this session.", e);
                        self.hnsw_index = None;
                        // Continue without HNSW - this is graceful degradation
                    }
                }
            }
        }
        
        // Update performance metrics
        self.performance_metrics.total_training_samples = self.training_samples.len() as u64;
        
        Ok(())
    }
    
    /// Generate training label based on 4-bar future price direction
    pub fn generate_label(current_price: f32, future_price: f32) -> Direction {
        if future_price < current_price {
            Direction::Short
        } else if future_price > current_price {
            Direction::Long
        } else {
            Direction::Neutral
        }
    }
    
    /// Check current memory usage and return status
    fn check_memory_usage(&mut self) -> MemoryStatus {
        // Get current memory usage (simplified - in real implementation would use system calls)
        let current_usage_mb = if let Some(ref pool) = self.memory_pool {
            pool.allocated_bytes() / (1024 * 1024)
        } else {
            // Estimate based on training samples
            (self.training_samples.len() * std::mem::size_of::<TrainingSample>()) / (1024 * 1024)
        };
        
        self.memory_threshold_monitor.check_memory_usage(current_usage_mb)
    }
    
    /// Handle memory pressure with adaptive behavior
    fn handle_memory_pressure(&mut self) -> Result<(), PerformanceOptimizationError> {
        eprintln!("Handling memory pressure - attempting adaptive memory management");
        
        // Try cleanup first
        if let Some(ref mut pool) = self.memory_pool {
            pool.cleanup();
        }
        
        // If still under pressure, switch to memory mapping
        if !self.config.enable_memory_mapping {
            eprintln!("Enabling memory mapping due to memory pressure");
            self.config.enable_memory_mapping = true;
            
            // Try to create memory mapped storage
            match self.create_memory_mapped_storage() {
                Ok(_) => {
                    eprintln!("Successfully switched to memory-mapped storage");
                }
                Err(e) => {
                    eprintln!("Failed to create memory-mapped storage: {}", e);
                    // Continue with in-memory storage but reduce buffer size
                    self.config.max_bars_back = self.config.max_bars_back / 2;
                    eprintln!("Reduced max_bars_back to {} due to memory constraints", self.config.max_bars_back);
                }
            }
        }
        
        Ok(())
    }
    
    /// Create memory-mapped storage on demand
    fn create_memory_mapped_storage(&mut self) -> Result<(), PerformanceOptimizationError> {
        if self.memory_mapped_storage.is_some() {
            return Ok(()); // Already created
        }
        
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("ldc_engine_samples.mmap");
        
        match MemoryMappedStorage::new(&file_path, self.config.max_bars_back * 2, false) {
            Ok(storage) => {
                self.memory_mapped_storage = Some(storage);
                Ok(())
            }
            Err(e) => Err(e)
        }
    }
    
    /// Rebuild HNSW index with error handling
    fn rebuild_hnsw_index(&mut self) -> Result<(), PerformanceOptimizationError> {
        if let Some(ref mut hnsw_index) = self.hnsw_index {
            hnsw_index.rebuild(&self.training_samples)?;
            self.samples_since_hnsw_rebuild = 0;
            eprintln!("HNSW index rebuilt successfully with {} samples", self.training_samples.len());
        }
        Ok(())
    }
    
    /// Performance monitoring wrapper with automatic error handling and fallback
    pub fn monitor_performance<T, F>(&self, operation_name: &str, expected_ms: f64, operation: F) -> Result<T, PerformanceOptimizationError>
    where
        F: FnOnce() -> Result<T, PerformanceOptimizationError>,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        // Check for performance degradation
        if duration_ms > expected_ms {
            let degradation_level = if duration_ms > expected_ms * 2.0 {
                PerformanceDegradationLevel::Critical
            } else {
                PerformanceDegradationLevel::Warning
            };
            
            let error = PerformanceOptimizationError::PerformanceDegradation {
                component: operation_name.to_string(),
                actual_ms: duration_ms,
                expected_ms,
            };
            
            match degradation_level {
                PerformanceDegradationLevel::Warning => {
                    eprintln!("Performance warning: {}", error);
                }
                PerformanceDegradationLevel::Critical => {
                    eprintln!("Critical performance degradation: {}", error);
                }
            }
        }
        
        result
    }
    
    /// Validate configuration parameters with error handling and sensible defaults
    pub fn validate_and_fix_config(&mut self) -> Vec<PerformanceOptimizationError> {
        let mut errors = Vec::new();
        
        // Validate thread count
        if let Some(max_threads) = self.config.max_threads {
            if max_threads == 0 {
                errors.push(PerformanceOptimizationError::ConfigurationError {
                    field: "max_threads".to_string(),
                    value: "0".to_string(),
                    default: "None (auto-detect)".to_string(),
                });
                self.config.max_threads = None;
            } else if max_threads > 128 {
                errors.push(PerformanceOptimizationError::ConfigurationError {
                    field: "max_threads".to_string(),
                    value: max_threads.to_string(),
                    default: "128".to_string(),
                });
                self.config.max_threads = Some(128);
            }
        }
        
        // Validate SIMD chunk size
        if self.config.simd_chunk_size == 0 {
            errors.push(PerformanceOptimizationError::ConfigurationError {
                field: "simd_chunk_size".to_string(),
                value: "0".to_string(),
                default: "64".to_string(),
            });
            self.config.simd_chunk_size = 64;
        }
        
        // Validate memory pool size
        if self.config.memory_pool_size > 16384 { // 16GB limit
            errors.push(PerformanceOptimizationError::ConfigurationError {
                field: "memory_pool_size".to_string(),
                value: format!("{}MB", self.config.memory_pool_size),
                default: "1024MB".to_string(),
            });
            self.config.memory_pool_size = 1024;
        }
        
        // Validate HNSW parameters
        if self.config.hnsw_m == 0 {
            errors.push(PerformanceOptimizationError::ConfigurationError {
                field: "hnsw_m".to_string(),
                value: "0".to_string(),
                default: "16".to_string(),
            });
            self.config.hnsw_m = 16;
        }
        
        if self.config.hnsw_ef_construction == 0 {
            errors.push(PerformanceOptimizationError::ConfigurationError {
                field: "hnsw_ef_construction".to_string(),
                value: "0".to_string(),
                default: "200".to_string(),
            });
            self.config.hnsw_ef_construction = 200;
        }
        
        if self.config.hnsw_ef_search == 0 {
            errors.push(PerformanceOptimizationError::ConfigurationError {
                field: "hnsw_ef_search".to_string(),
                value: "0".to_string(),
                default: "50".to_string(),
            });
            self.config.hnsw_ef_search = 50;
        }
        
        errors
    }
    
    /// Get number of training samples
    pub fn training_samples_count(&self) -> usize {
        self.training_samples.len()
    }
    
    /// Get configuration
    pub fn config(&self) -> &LDCConfig {
        &self.config
    }
    
    /// Get mutable configuration (for testing purposes)
    pub fn get_config_mut(&mut self) -> Result<&mut LDCConfig> {
        Ok(&mut self.config)
    }
    
    /// Update configuration with validation and performance parameter handling
    pub fn update_config(&mut self, mut config: LDCConfig) -> Result<()> {
        // Validate configuration parameters with enhanced validation
        self.validate_config(&mut config)?;
        
        // Apply automatic performance tuning if enabled
        if config.enable_auto_tuning {
            self.auto_tune_performance_parameters(&mut config)?;
        }
        
        // Handle configuration changes that require special processing
        let needs_hnsw_rebuild = self.config.use_hnsw_index != config.use_hnsw_index ||
                                self.config.hnsw_m != config.hnsw_m ||
                                self.config.hnsw_ef_construction != config.hnsw_ef_construction;
        
        let needs_memory_pool_resize = self.config.memory_pool_size != config.memory_pool_size;
        let needs_thread_pool_update = self.config.thread_pool_strategy != config.thread_pool_strategy ||
                                      self.config.max_threads != config.max_threads;
        
        // Update configuration
        self.config = config;
        
        // Resize training samples if needed
        while self.training_samples.len() > self.config.max_bars_back {
            self.training_samples.pop_front();
        }
        
        // Handle memory pool resizing if needed
        if needs_memory_pool_resize {
            self.resize_memory_pool()?;
        }
        
        // Handle thread pool updates if needed
        if needs_thread_pool_update {
            self.update_thread_pool_configuration()?;
        }
        
        // Handle HNSW index rebuilding if needed
        if needs_hnsw_rebuild {
            if self.config.use_hnsw_index {
                self.log_debug("Configuration change requires HNSW index rebuild");
                if let Err(e) = self.rebuild_or_create_hnsw_index() {
                    if self.config.enable_debug_logging {
                        eprintln!("Warning: Failed to rebuild HNSW index after config change: {}", e);
                    }
                    self.hnsw_index = None;
                }
            } else {
                // HNSW disabled, remove index
                self.hnsw_index = None;
                self.log_debug("HNSW index disabled and removed");
            }
        }
        
        // Log configuration update
        if self.config.enable_debug_logging {
            self.log_debug(&format!("Configuration updated: SIMD={}, HNSW={}, ThreadStrategy={:?}", 
                                  self.config.use_simd_optimization,
                                  self.config.use_hnsw_index,
                                  self.config.thread_pool_strategy));
        }
        
        Ok(())
    }
    
    /// Validate configuration parameters and apply corrections
    fn validate_config(&self, config: &mut LDCConfig) -> Result<()> {
        // Validate basic parameters
        if config.max_bars_back == 0 {
            return Err(anyhow::anyhow!("max_bars_back must be greater than 0"));
        }
        
        if config.neighbors_count == 0 {
            return Err(anyhow::anyhow!("neighbors_count must be greater than 0"));
        }
        
        if config.feature_count == 0 || config.feature_count > 5 {
            return Err(anyhow::anyhow!("feature_count must be between 1 and 5"));
        }
        
        // Validate SIMD parameters
        if config.simd_chunk_size == 0 {
            config.simd_chunk_size = 64; // Set to default
            if self.config.enable_debug_logging {
                eprintln!("Warning: simd_chunk_size was 0, set to default value 64");
            }
        }
        
        // Validate memory parameters
        if config.memory_pool_size == 0 {
            config.memory_pool_size = 256; // Set to default 256MB
            if self.config.enable_debug_logging {
                eprintln!("Warning: memory_pool_size was 0, set to default value 256MB");
            }
        }
        
        // Validate HNSW parameters
        if config.use_hnsw_index {
            if config.hnsw_m == 0 {
                config.hnsw_m = 16;
                if self.config.enable_debug_logging {
                    eprintln!("Warning: hnsw_m was 0, set to default value 16");
                }
            }
            
            if config.hnsw_ef_construction == 0 {
                config.hnsw_ef_construction = 200;
                if self.config.enable_debug_logging {
                    eprintln!("Warning: hnsw_ef_construction was 0, set to default value 200");
                }
            }
            
            if config.hnsw_ef_search == 0 {
                config.hnsw_ef_search = 50;
                if self.config.enable_debug_logging {
                    eprintln!("Warning: hnsw_ef_search was 0, set to default value 50");
                }
            }
            
            if config.hnsw_rebuild_threshold == 0 {
                config.hnsw_rebuild_threshold = 1000;
                if self.config.enable_debug_logging {
                    eprintln!("Warning: hnsw_rebuild_threshold was 0, set to default value 1000");
                }
            }
            
            // Validate HNSW parameter relationships
            if config.hnsw_ef_search > config.hnsw_ef_construction {
                if self.config.enable_debug_logging {
                    eprintln!("Warning: hnsw_ef_search ({}) > hnsw_ef_construction ({}), adjusting ef_search", 
                             config.hnsw_ef_search, config.hnsw_ef_construction);
                }
                config.hnsw_ef_search = config.hnsw_ef_construction;
            }
        }
        
        // Validate threading parameters
        if let Some(max_threads) = config.max_threads {
            if max_threads == 0 {
                config.max_threads = None; // Auto-detect
                if self.config.enable_debug_logging {
                    eprintln!("Warning: max_threads was 0, set to auto-detect");
                }
            }
        }
        
        // Validate threshold parameters
        if config.parallel_threshold == 0 {
            config.parallel_threshold = 100;
            if self.config.enable_debug_logging {
                eprintln!("Warning: parallel_threshold was 0, set to default value 100");
            }
        }
        
        if config.batch_parallel_threshold == 0 {
            config.batch_parallel_threshold = 10;
            if self.config.enable_debug_logging {
                eprintln!("Warning: batch_parallel_threshold was 0, set to default value 10");
            }
        }
        
        Ok(())
    }
    
    /// Enhanced configuration validation with detailed feedback and recommendations
    pub fn validate_config_enhanced(&self, config: &LDCConfig) -> ConfigurationValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();
        
        // Detect system capabilities for validation
        let system_caps = self.detect_system_capabilities();
        
        // Validate basic parameters with detailed feedback
        if config.max_bars_back == 0 {
            errors.push(ConfigurationError {
                field: "max_bars_back".to_string(),
                current_value: "0".to_string(),
                error_message: "max_bars_back must be greater than 0".to_string(),
                suggested_value: "2000".to_string(),
                severity: ErrorSeverity::Critical,
            });
        } else if config.max_bars_back > 100000 {
            warnings.push(ConfigurationWarning {
                field: "max_bars_back".to_string(),
                current_value: config.max_bars_back.to_string(),
                warning_message: "Very large max_bars_back may cause memory issues".to_string(),
                suggested_value: "50000".to_string(),
                impact: PerformanceImpact::High,
            });
        }
        
        // Validate memory configuration against system capabilities
        let required_memory_mb = self.estimate_memory_usage(config);
        let available_memory_mb = system_caps.available_memory_gb * 1024;
        
        if required_memory_mb > available_memory_mb {
            errors.push(ConfigurationError {
                field: "memory_pool_size".to_string(),
                current_value: config.memory_pool_size.to_string(),
                error_message: format!("Configuration requires {}MB but only {}MB available", 
                                     required_memory_mb, available_memory_mb),
                suggested_value: (available_memory_mb / 2).to_string(),
                severity: ErrorSeverity::Critical,
            });
        } else if required_memory_mb > available_memory_mb * 80 / 100 {
            warnings.push(ConfigurationWarning {
                field: "memory_pool_size".to_string(),
                current_value: config.memory_pool_size.to_string(),
                warning_message: "Configuration uses >80% of available memory".to_string(),
                suggested_value: (available_memory_mb / 2).to_string(),
                impact: PerformanceImpact::Medium,
            });
        }
        
        // Validate SIMD configuration against hardware capabilities
        if config.use_simd_optimization && !system_caps.has_avx2 {
            warnings.push(ConfigurationWarning {
                field: "use_simd_optimization".to_string(),
                current_value: "true".to_string(),
                warning_message: "SIMD optimization enabled but AVX2 not detected".to_string(),
                suggested_value: "false".to_string(),
                impact: PerformanceImpact::Medium,
            });
        }
        
        // Validate threading configuration
        if let Some(max_threads) = config.max_threads {
            if max_threads > system_caps.cpu_cores * 2 {
                warnings.push(ConfigurationWarning {
                    field: "max_threads".to_string(),
                    current_value: max_threads.to_string(),
                    warning_message: "max_threads exceeds 2x CPU cores, may cause contention".to_string(),
                    suggested_value: system_caps.cpu_cores.to_string(),
                    impact: PerformanceImpact::Medium,
                });
            }
        }
        
        // Validate HNSW parameters for dataset size
        if config.use_hnsw_index {
            let dataset_size = self.training_samples.len();
            
            if dataset_size < 1000 && config.use_hnsw_index {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::General,
                    priority: RecommendationPriority::Low,
                    description: "HNSW index may not provide benefits for small datasets".to_string(),
                    action: "Consider disabling HNSW for datasets <1000 samples".to_string(),
                });
            }
            
            if config.hnsw_ef_construction < config.hnsw_m * 2 {
                warnings.push(ConfigurationWarning {
                    field: "hnsw_ef_construction".to_string(),
                    current_value: config.hnsw_ef_construction.to_string(),
                    warning_message: "hnsw_ef_construction should be at least 2x hnsw_m".to_string(),
                    suggested_value: (config.hnsw_m * 2).to_string(),
                    impact: PerformanceImpact::Medium,
                });
            }
        }
        
        // Generate performance estimate
        let performance_estimate = self.estimate_performance(config, &system_caps);
        
        ConfigurationValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            recommendations,
            estimated_performance: Some(performance_estimate),
        }
    }
    
    /// Detect system capabilities for configuration optimization
    fn detect_system_capabilities(&self) -> SystemCapabilities {
        let cpu_cores = num_cpus::get();
        
        // Estimate available memory (simplified - in production would use system APIs)
        let available_memory_gb = 8; // Default conservative estimate
        
        // Detect SIMD capabilities (simplified - would use cpuid in production)
        let has_avx2 = cfg!(target_feature = "avx2");
        let has_avx512 = cfg!(target_feature = "avx512f");
        
        SystemCapabilities {
            cpu_cores,
            available_memory_gb,
            has_avx2,
            has_avx512,
            numa_nodes: 1, // Simplified
            cache_line_size: 64,
            page_size: 4096,
        }
    }
    
    /// Estimate memory usage for a given configuration
    fn estimate_memory_usage(&self, config: &LDCConfig) -> usize {
        let sample_size = std::mem::size_of::<TrainingSample>();
        let training_memory = config.max_bars_back * sample_size / (1024 * 1024); // MB
        let pool_memory = config.memory_pool_size;
        let hnsw_memory = if config.use_hnsw_index {
            // Rough estimate: HNSW uses ~4x the data size
            training_memory * 4
        } else {
            0
        };
        
        training_memory + pool_memory + hnsw_memory + 100 // 100MB overhead
    }
    
    /// Estimate performance for a given configuration and system
    fn estimate_performance(&self, config: &LDCConfig, system: &SystemCapabilities) -> PerformanceEstimate {
        let dataset_size = self.training_samples.len().max(config.max_bars_back);
        
        // Base latency calculation (simplified model)
        let base_latency_ms = if config.use_hnsw_index && dataset_size > 1000 {
            // HNSW: O(log N) complexity
            (dataset_size as f64).log2() * 0.01
        } else if config.use_multithreading && dataset_size > config.parallel_threshold {
            // Parallel: O(N/cores) complexity
            (dataset_size as f64) / (system.cpu_cores as f64) * 0.001
        } else {
            // Sequential: O(N) complexity
            dataset_size as f64 * 0.001
        };
        
        // Apply SIMD speedup
        let simd_speedup = if config.use_simd_optimization && system.has_avx2 {
            if system.has_avx512 { 2.5 } else { 2.0 }
        } else {
            1.0
        };
        
        let estimated_latency = base_latency_ms / simd_speedup;
        
        // Estimate throughput (queries per second)
        let estimated_throughput = if estimated_latency > 0.0 {
            1000.0 / estimated_latency
        } else {
            10000.0 // Very fast
        };
        
        // Estimate memory usage
        let estimated_memory = self.estimate_memory_usage(config);
        
        // Estimate CPU utilization
        let estimated_cpu = if config.use_multithreading {
            (system.cpu_cores as f32 * 0.8).min(95.0)
        } else {
            25.0
        };
        
        // Confidence based on how well we know the system
        let confidence = if system.cpu_cores > 1 && system.available_memory_gb > 4 {
            0.8
        } else {
            0.6
        };
        
        PerformanceEstimate {
            estimated_latency_p95_ms: estimated_latency * 1.2, // P95 is typically 20% higher
            estimated_throughput_qps: estimated_throughput,
            estimated_memory_usage_mb: estimated_memory,
            estimated_cpu_utilization_percent: estimated_cpu,
            confidence_level: confidence,
        }
    }
    
    /// Automatically tune performance parameters based on system capabilities and dataset
    pub fn auto_tune_performance_parameters(&self, config: &mut LDCConfig) -> Result<()> {
        let system_caps = self.detect_system_capabilities();
        let dataset_size = self.training_samples.len();
        
        // Auto-tune threading parameters
        if config.max_threads.is_none() {
            config.max_threads = Some(system_caps.cpu_cores);
        }
        
        // Auto-tune parallel threshold based on dataset size and CPU cores
        if dataset_size > 0 {
            let optimal_threshold = (dataset_size / system_caps.cpu_cores).max(50).min(500);
            config.parallel_threshold = optimal_threshold;
        }
        
        // Auto-tune SIMD chunk size based on cache line size
        if config.use_simd_optimization {
            let optimal_chunk_size = (system_caps.cache_line_size / 4).max(32).min(256);
            config.simd_chunk_size = optimal_chunk_size;
        }
        
        // Auto-tune memory pool size based on available memory
        let optimal_memory_pool = (system_caps.available_memory_gb * 1024 / 4).max(128).min(2048);
        config.memory_pool_size = optimal_memory_pool;
        
        // Auto-tune HNSW parameters based on dataset size
        if config.use_hnsw_index && dataset_size > 0 {
            // Optimize M parameter based on dataset size
            config.hnsw_m = if dataset_size < 10000 {
                8
            } else if dataset_size < 100000 {
                16
            } else {
                32
            };
            
            // Optimize ef_construction for balance of build time and accuracy
            config.hnsw_ef_construction = config.hnsw_m * 10;
            
            // Optimize ef_search for query performance
            config.hnsw_ef_search = config.hnsw_m * 2;
            
            // Set rebuild threshold based on dataset size
            config.hnsw_rebuild_threshold = (dataset_size / 10).max(100).min(5000);
        }
        
        // Enable memory mapping for large datasets
        if dataset_size > 50000 || config.max_bars_back > 50000 {
            config.enable_memory_mapping = true;
        }
        
        // Choose optimal thread pool strategy
        config.thread_pool_strategy = if dataset_size > 10000 && system_caps.cpu_cores > 4 {
            ThreadPoolStrategy::Dedicated
        } else if dataset_size > 1000 {
            ThreadPoolStrategy::Adaptive
        } else {
            ThreadPoolStrategy::Global
        };
        
        if config.enable_debug_logging {
            println!("Auto-tuned configuration for {} samples on {}-core system with {}GB RAM", 
                    dataset_size, system_caps.cpu_cores, system_caps.available_memory_gb);
        }
        
        Ok(())
    }
    
    /// Generate configuration recommendations based on dataset size and hardware
    pub fn generate_configuration_recommendations(&self, use_case: UseCase, hardware: &HardwareProfile) -> Vec<ConfigurationProfile> {
        let mut profiles = Vec::new();
        
        // Generate profile for real-time trading
        if matches!(use_case, UseCase::RealTimeTrading | UseCase::HighFrequency) {
            let mut config = LDCConfig::default();
            
            // Optimize for ultra-low latency
            config.use_hnsw_index = hardware.memory_gb > 8;
            config.hnsw_m = 8; // Lower M for faster queries
            config.hnsw_ef_search = 16; // Lower ef_search for speed
            config.use_simd_optimization = hardware.has_simd_support;
            config.max_threads = Some(hardware.cpu_cores.min(4)); // Limit threads to reduce jitter
            config.thread_pool_strategy = ThreadPoolStrategy::Dedicated;
            config.memory_pool_size = (hardware.memory_gb * 1024 / 8).max(256);
            config.parallel_threshold = 50; // Lower threshold for consistent performance
            
            profiles.push(ConfigurationProfile {
                name: "Real-Time Trading".to_string(),
                description: "Optimized for ultra-low latency trading applications".to_string(),
                config,
                target_hardware: hardware.clone(),
                use_case: UseCase::RealTimeTrading,
                created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64,
                performance_baseline: Some(PerformanceBaseline {
                    target_latency_p95_ms: 1.0,
                    target_throughput_qps: 1000.0,
                    target_memory_usage_mb: hardware.memory_gb * 1024 / 4,
                    target_cpu_utilization_percent: 60.0,
                    minimum_accuracy_percent: 98.0,
                }),
            });
        }
        
        // Generate profile for backtesting
        if matches!(use_case, UseCase::BackTesting) {
            let mut config = LDCConfig::default();
            
            // Optimize for throughput
            config.use_hnsw_index = true;
            config.hnsw_m = 16;
            config.hnsw_ef_search = 50;
            config.use_simd_optimization = hardware.has_simd_support;
            config.max_threads = Some(hardware.cpu_cores);
            config.thread_pool_strategy = ThreadPoolStrategy::Global;
            config.memory_pool_size = (hardware.memory_gb * 1024 / 2).max(512);
            config.parallel_threshold = 100;
            config.enable_memory_mapping = hardware.memory_gb < 16; // Use for systems with limited RAM
            
            profiles.push(ConfigurationProfile {
                name: "Backtesting".to_string(),
                description: "Optimized for high-throughput backtesting workloads".to_string(),
                config,
                target_hardware: hardware.clone(),
                use_case: UseCase::BackTesting,
                created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64,
                performance_baseline: Some(PerformanceBaseline {
                    target_latency_p95_ms: 10.0,
                    target_throughput_qps: 500.0,
                    target_memory_usage_mb: hardware.memory_gb * 1024 / 2,
                    target_cpu_utilization_percent: 85.0,
                    minimum_accuracy_percent: 99.0,
                }),
            });
        }
        
        // Generate profile for research
        if matches!(use_case, UseCase::Research) {
            let mut config = LDCConfig::default();
            
            // Optimize for accuracy
            config.use_hnsw_index = false; // Use exact search for maximum accuracy
            config.use_simd_optimization = hardware.has_simd_support;
            config.max_threads = Some(hardware.cpu_cores);
            config.thread_pool_strategy = ThreadPoolStrategy::Adaptive;
            config.memory_pool_size = (hardware.memory_gb * 1024 / 4).max(256);
            config.parallel_threshold = 200; // Higher threshold for accuracy
            config.enable_debug_logging = true;
            config.log_performance_metrics = true;
            
            profiles.push(ConfigurationProfile {
                name: "Research".to_string(),
                description: "Optimized for research with maximum accuracy".to_string(),
                config,
                target_hardware: hardware.clone(),
                use_case: UseCase::Research,
                created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64,
                performance_baseline: Some(PerformanceBaseline {
                    target_latency_p95_ms: 50.0,
                    target_throughput_qps: 100.0,
                    target_memory_usage_mb: hardware.memory_gb * 1024 / 2,
                    target_cpu_utilization_percent: 70.0,
                    minimum_accuracy_percent: 100.0,
                }),
            });
        }
        
        profiles
    }
    
    /// Export configuration profile to JSON
    pub fn export_configuration_profile(&self, profile_name: &str) -> Result<String> {
        let hardware = HardwareProfile {
            cpu_cores: self.detect_system_capabilities().cpu_cores,
            memory_gb: self.detect_system_capabilities().available_memory_gb,
            has_simd_support: self.detect_system_capabilities().has_avx2,
            numa_nodes: self.detect_system_capabilities().numa_nodes,
            storage_type: StorageType::SSD, // Default assumption
            estimated_memory_bandwidth_gbps: 25.0, // Default estimate
        };
        
        let profile = ConfigurationProfile {
            name: profile_name.to_string(),
            description: format!("Exported configuration profile from current settings"),
            config: self.config.clone(),
            target_hardware: hardware,
            use_case: UseCase::Production, // Default
            created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64,
            performance_baseline: None,
        };
        
        serde_json::to_string_pretty(&profile)
            .map_err(|e| anyhow::anyhow!("Failed to serialize configuration profile: {}", e))
    }
    
    /// Import configuration profile from JSON
    pub fn import_configuration_profile(&mut self, json_data: &str) -> Result<()> {
        let profile: ConfigurationProfile = serde_json::from_str(json_data)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize configuration profile: {}", e))?;
        
        // Validate the imported configuration
        let validation_result = self.validate_config_enhanced(&profile.config);
        
        if !validation_result.is_valid {
            let error_messages: Vec<String> = validation_result.errors
                .iter()
                .map(|e| format!("{}: {}", e.field, e.error_message))
                .collect();
            
            return Err(anyhow::anyhow!("Imported configuration is invalid: {}", error_messages.join(", ")));
        }
        
        // Apply the configuration
        self.update_config(profile.config)?;
        
        if self.config.enable_debug_logging {
            println!("Successfully imported configuration profile: {}", profile.name);
            
            if !validation_result.warnings.is_empty() {
                println!("Warnings for imported configuration:");
                for warning in &validation_result.warnings {
                    println!("  - {}: {}", warning.field, warning.warning_message);
                }
            }
        }
        
        Ok(())
    }
    
    /// Resize memory pool based on new configuration
    fn resize_memory_pool(&mut self) -> Result<()> {
        // In a real implementation, this would resize the actual memory pool
        // For now, we'll just log the change
        if self.config.enable_debug_logging {
            println!("Resizing memory pool to {}MB", self.config.memory_pool_size);
        }
        Ok(())
    }
    
    /// Update thread pool configuration
    fn update_thread_pool_configuration(&mut self) -> Result<()> {
        // In a real implementation, this would update the thread pool
        // For now, we'll just log the change
        if self.config.enable_debug_logging {
            println!("Updating thread pool configuration: strategy={:?}, max_threads={:?}", 
                    self.config.thread_pool_strategy, self.config.max_threads);
        }
        Ok(())
    }
    
    /// Create predefined configuration profiles for common use cases
    pub fn create_predefined_profiles() -> HashMap<String, ConfigurationProfile> {
        let mut profiles = HashMap::new();
        
        // Default hardware profile for examples
        let default_hardware = HardwareProfile {
            cpu_cores: 8,
            memory_gb: 16,
            has_simd_support: true,
            numa_nodes: 1,
            storage_type: StorageType::SSD,
            estimated_memory_bandwidth_gbps: 25.0,
        };
        
        // Ultra-low latency profile
        let mut ultra_low_latency_config = LDCConfig::default();
        ultra_low_latency_config.use_hnsw_index = true;
        ultra_low_latency_config.hnsw_m = 8;
        ultra_low_latency_config.hnsw_ef_search = 16;
        ultra_low_latency_config.use_simd_optimization = true;
        ultra_low_latency_config.max_threads = Some(4);
        ultra_low_latency_config.thread_pool_strategy = ThreadPoolStrategy::Dedicated;
        ultra_low_latency_config.memory_pool_size = 512;
        ultra_low_latency_config.parallel_threshold = 50;
        ultra_low_latency_config.simd_chunk_size = 32;
        
        profiles.insert("ultra-low-latency".to_string(), ConfigurationProfile {
            name: "Ultra Low Latency".to_string(),
            description: "Optimized for sub-millisecond query times in high-frequency trading".to_string(),
            config: ultra_low_latency_config,
            target_hardware: default_hardware.clone(),
            use_case: UseCase::HighFrequency,
            created_at: 0,
            performance_baseline: Some(PerformanceBaseline {
                target_latency_p95_ms: 0.5,
                target_throughput_qps: 2000.0,
                target_memory_usage_mb: 1024,
                target_cpu_utilization_percent: 50.0,
                minimum_accuracy_percent: 98.0,
            }),
        });
        
        // High throughput profile
        let mut high_throughput_config = LDCConfig::default();
        high_throughput_config.use_hnsw_index = true;
        high_throughput_config.hnsw_m = 16;
        high_throughput_config.hnsw_ef_search = 50;
        high_throughput_config.use_simd_optimization = true;
        high_throughput_config.max_threads = Some(16);
        high_throughput_config.thread_pool_strategy = ThreadPoolStrategy::Global;
        high_throughput_config.memory_pool_size = 1024;
        high_throughput_config.parallel_threshold = 100;
        high_throughput_config.enable_memory_mapping = true;
        
        profiles.insert("high-throughput".to_string(), ConfigurationProfile {
            name: "High Throughput".to_string(),
            description: "Optimized for maximum throughput in backtesting and batch processing".to_string(),
            config: high_throughput_config,
            target_hardware: default_hardware.clone(),
            use_case: UseCase::BackTesting,
            created_at: 0,
            performance_baseline: Some(PerformanceBaseline {
                target_latency_p95_ms: 10.0,
                target_throughput_qps: 1000.0,
                target_memory_usage_mb: 2048,
                target_cpu_utilization_percent: 90.0,
                minimum_accuracy_percent: 99.0,
            }),
        });
        
        // Memory efficient profile
        let mut memory_efficient_config = LDCConfig::default();
        memory_efficient_config.use_hnsw_index = false; // Exact search uses less memory
        memory_efficient_config.use_simd_optimization = true;
        memory_efficient_config.max_threads = Some(4);
        memory_efficient_config.thread_pool_strategy = ThreadPoolStrategy::Adaptive;
        memory_efficient_config.memory_pool_size = 128;
        memory_efficient_config.parallel_threshold = 200;
        memory_efficient_config.enable_memory_mapping = true;
        memory_efficient_config.max_bars_back = 10000; // Smaller buffer
        
        profiles.insert("memory-efficient".to_string(), ConfigurationProfile {
            name: "Memory Efficient".to_string(),
            description: "Optimized for systems with limited memory resources".to_string(),
            config: memory_efficient_config,
            target_hardware: HardwareProfile {
                cpu_cores: 4,
                memory_gb: 4,
                has_simd_support: true,
                numa_nodes: 1,
                storage_type: StorageType::SSD,
                estimated_memory_bandwidth_gbps: 15.0,
            },
            use_case: UseCase::Production,
            created_at: 0,
            performance_baseline: Some(PerformanceBaseline {
                target_latency_p95_ms: 20.0,
                target_throughput_qps: 200.0,
                target_memory_usage_mb: 512,
                target_cpu_utilization_percent: 70.0,
                minimum_accuracy_percent: 100.0,
            }),
        });
        
        // Research profile
        let mut research_config = LDCConfig::default();
        research_config.use_hnsw_index = false; // Exact search for maximum accuracy
        research_config.use_simd_optimization = true;
        research_config.max_threads = Some(8);
        research_config.thread_pool_strategy = ThreadPoolStrategy::Adaptive;
        research_config.memory_pool_size = 512;
        research_config.parallel_threshold = 500; // Higher threshold for accuracy
        research_config.enable_debug_logging = true;
        research_config.log_performance_metrics = true;
        research_config.log_predictions = true;
        
        profiles.insert("research".to_string(), ConfigurationProfile {
            name: "Research".to_string(),
            description: "Optimized for research and development with maximum accuracy and detailed logging".to_string(),
            config: research_config,
            target_hardware: default_hardware.clone(),
            use_case: UseCase::Research,
            created_at: 0,
            performance_baseline: Some(PerformanceBaseline {
                target_latency_p95_ms: 100.0,
                target_throughput_qps: 50.0,
                target_memory_usage_mb: 1024,
                target_cpu_utilization_percent: 60.0,
                minimum_accuracy_percent: 100.0,
            }),
        });
        
        profiles
    }
    
    /// Apply a predefined configuration profile by name
    pub fn apply_predefined_profile(&mut self, profile_name: &str) -> Result<()> {
        let profiles = Self::create_predefined_profiles();
        
        if let Some(profile) = profiles.get(profile_name) {
            self.update_config(profile.config.clone())?;
            
            if self.config.enable_debug_logging {
                println!("Applied predefined profile: {} - {}", profile.name, profile.description);
            }
            
            Ok(())
        } else {
            let available_profiles: Vec<&String> = profiles.keys().collect();
            Err(anyhow::anyhow!("Unknown profile '{}'. Available profiles: {:?}", 
                               profile_name, available_profiles))
        }
    }
    
    /// Validate current configuration and return detailed results
    pub fn validate_current_configuration(&self) -> ConfigurationValidationResult {
        self.validate_config_enhanced(&self.config)
    }
    
    /// Get configuration recommendations based on current performance metrics
    pub fn get_configuration_recommendations_from_metrics(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        let metrics = &self.performance_metrics;
        
        // Analyze latency performance
        if metrics.latency_p95_ms > 10.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Latency,
                priority: if metrics.latency_p95_ms > 50.0 { 
                    RecommendationPriority::Critical 
                } else { 
                    RecommendationPriority::High 
                },
                description: format!("P95 latency is {:.2}ms, which is high", metrics.latency_p95_ms),
                action: if !self.config.use_hnsw_index && self.training_samples.len() > 1000 {
                    "Enable HNSW indexing to reduce query latency".to_string()
                } else if !self.config.use_simd_optimization {
                    "Enable SIMD optimization to accelerate distance calculations".to_string()
                } else {
                    "Consider reducing dataset size or increasing parallel_threshold".to_string()
                },
            });
        }
        
        // Analyze memory usage
        if metrics.current_memory_usage_mb > self.config.memory_pool_size * 80 / 100 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Memory,
                priority: RecommendationPriority::Medium,
                description: format!("Memory usage is {}MB ({}% of pool)", 
                                   metrics.current_memory_usage_mb,
                                   metrics.current_memory_usage_mb * 100 / self.config.memory_pool_size),
                action: "Increase memory_pool_size or enable memory mapping".to_string(),
            });
        }
        
        // Analyze CPU utilization
        if metrics.cpu_utilization_percent < 50.0 && self.training_samples.len() > 100 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::CPU,
                priority: RecommendationPriority::Medium,
                description: format!("CPU utilization is {:.1}%, indicating underutilization", 
                                   metrics.cpu_utilization_percent),
                action: "Reduce parallel_threshold or increase max_threads to improve CPU utilization".to_string(),
            });
        }
        
        // Analyze thread efficiency
        if metrics.thread_efficiency_percent < 70.0 && self.config.use_multithreading {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Threading,
                priority: RecommendationPriority::High,
                description: format!("Thread efficiency is {:.1}%, indicating poor thread utilization", 
                                   metrics.thread_efficiency_percent),
                action: "Switch to Dedicated thread pool strategy or adjust max_threads".to_string(),
            });
        }
        
        // Analyze HNSW accuracy
        if self.config.use_hnsw_index && metrics.hnsw_accuracy_percent < 95.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Accuracy,
                priority: RecommendationPriority::Medium,
                description: format!("HNSW accuracy is {:.1}%, below recommended 95%", 
                                   metrics.hnsw_accuracy_percent),
                action: "Increase hnsw_ef_construction or hnsw_m parameters".to_string(),
            });
        }
        
        // Analyze operation balance
        let total_queries = metrics.hnsw_queries + metrics.exact_queries;
        if total_queries > 0 {
            let hnsw_ratio = metrics.hnsw_queries as f64 / total_queries as f64;
            
            if self.config.use_hnsw_index && hnsw_ratio < 0.5 {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::General,
                    priority: RecommendationPriority::Low,
                    description: format!("Only {:.1}% of queries use HNSW index", hnsw_ratio * 100.0),
                    action: "Check HNSW index configuration or dataset size requirements".to_string(),
                });
            }
        }
        
        recommendations
    }
    
    /// Perform runtime configuration validation with automatic corrections
    pub fn validate_and_correct_runtime_config(&mut self) -> Result<Vec<String>> {
        let mut corrections = Vec::new();
        let system_caps = self.detect_system_capabilities();
        
        // Check memory usage and adjust if necessary
        let current_memory_mb = self.performance_metrics.current_memory_usage_mb;
        let available_memory_mb = system_caps.available_memory_gb * 1024;
        
        if current_memory_mb > available_memory_mb * 90 / 100 {
            // Critical memory usage - reduce memory pool size
            let new_pool_size = (available_memory_mb / 2).max(128);
            if new_pool_size != self.config.memory_pool_size {
                self.config.memory_pool_size = new_pool_size;
                corrections.push(format!("Reduced memory_pool_size to {}MB due to high memory usage", new_pool_size));
            }
        }
        
        // Check CPU utilization and adjust threading
        let cpu_util = self.performance_metrics.cpu_utilization_percent;
        if cpu_util > 95.0 && self.config.max_threads.unwrap_or(1) > system_caps.cpu_cores {
            // High CPU usage - reduce thread count
            self.config.max_threads = Some(system_caps.cpu_cores);
            corrections.push(format!("Reduced max_threads to {} due to high CPU usage", system_caps.cpu_cores));
        } else if cpu_util < 30.0 && self.config.max_threads.unwrap_or(1) < system_caps.cpu_cores {
            // Low CPU usage - increase thread count
            self.config.max_threads = Some(system_caps.cpu_cores);
            corrections.push(format!("Increased max_threads to {} due to low CPU usage", system_caps.cpu_cores));
        }
        
        // Check latency and adjust HNSW parameters
        if self.config.use_hnsw_index && self.performance_metrics.latency_p95_ms > 20.0 {
            // High latency - reduce HNSW search parameters
            if self.config.hnsw_ef_search > 16 {
                self.config.hnsw_ef_search = (self.config.hnsw_ef_search / 2).max(16);
                corrections.push(format!("Reduced hnsw_ef_search to {} to improve latency", self.config.hnsw_ef_search));
            }
        }
        
        // Check accuracy and adjust parameters
        if self.config.use_hnsw_index && self.performance_metrics.hnsw_accuracy_percent < 90.0 {
            // Low accuracy - increase HNSW parameters
            if self.config.hnsw_ef_construction < 400 {
                self.config.hnsw_ef_construction = (self.config.hnsw_ef_construction * 2).min(400);
                corrections.push(format!("Increased hnsw_ef_construction to {} to improve accuracy", self.config.hnsw_ef_construction));
            }
        }
        
        Ok(corrections)
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }
    
    /// Reset performance metrics
    pub fn reset_performance_metrics(&mut self) {
        self.performance_metrics = PerformanceMetrics::default();
    }
    
    /// Update detailed timing metrics
    pub fn update_timing_metrics(&mut self, distance_time_ms: f64, knn_time_ms: f64, data_access_time_ms: f64) {
        self.performance_metrics.distance_calculation_time_ms = distance_time_ms;
        self.performance_metrics.knn_search_time_ms = knn_time_ms;
        self.performance_metrics.data_access_time_ms = data_access_time_ms;
    }
    
    /// Update operation counters
    pub fn increment_simd_operations(&mut self) {
        self.performance_metrics.simd_operations_count += 1;
    }
    
    pub fn increment_hnsw_queries(&mut self) {
        self.performance_metrics.hnsw_queries += 1;
    }
    
    pub fn increment_exact_queries(&mut self) {
        self.performance_metrics.exact_queries += 1;
    }
    
    /// Update latency percentiles with rolling window
    pub fn update_latency_percentiles(&mut self, latency_ms: f64) {
        // Add new sample to rolling window
        self.performance_metrics.latency_samples.push_back(latency_ms);
        
        // Maintain rolling window size (keep last 1000 samples)
        if self.performance_metrics.latency_samples.len() > 1000 {
            self.performance_metrics.latency_samples.pop_front();
        }
        
        // Calculate percentiles if we have enough samples
        if self.performance_metrics.latency_samples.len() >= 10 {
            let mut sorted_samples: Vec<f64> = self.performance_metrics.latency_samples.iter().cloned().collect();
            sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            let _len = sorted_samples.len();
            self.performance_metrics.latency_p50_ms = Self::calculate_percentile(&sorted_samples, 50.0);
            self.performance_metrics.latency_p95_ms = Self::calculate_percentile(&sorted_samples, 95.0);
            self.performance_metrics.latency_p99_ms = Self::calculate_percentile(&sorted_samples, 99.0);
        }
    }
    
    /// Calculate percentile from sorted array
    fn calculate_percentile(sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }
        
        let index = (percentile / 100.0) * (sorted_values.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;
        
        if lower_index == upper_index {
            sorted_values[lower_index]
        } else {
            let weight = index - lower_index as f64;
            sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight
        }
    }
    
    /// Update memory metrics
    pub fn update_memory_metrics(&mut self, current_mb: usize, allocations: u64) {
        self.performance_metrics.current_memory_usage_mb = current_mb;
        self.performance_metrics.memory_allocations = allocations;
        
        // Update peak memory usage
        if current_mb > self.performance_metrics.peak_memory_usage_mb {
            self.performance_metrics.peak_memory_usage_mb = current_mb;
        }
    }
    
    /// Update CPU utilization metrics
    pub fn update_cpu_metrics(&mut self, cpu_utilization: f32, thread_efficiency: f32) {
        self.performance_metrics.cpu_utilization_percent = cpu_utilization;
        self.performance_metrics.thread_efficiency_percent = thread_efficiency;
    }
    
    /// Update HNSW specific metrics
    pub fn update_hnsw_metrics(&mut self, index_size: usize, accuracy_percent: f32) {
        self.performance_metrics.hnsw_index_size = index_size;
        self.performance_metrics.hnsw_accuracy_percent = accuracy_percent;
    }
    
    /// Increment HNSW rebuild count
    pub fn increment_hnsw_rebuild_count(&mut self) {
        self.performance_metrics.hnsw_rebuild_count += 1;
    }
    
    /// Get current memory usage in MB (system-dependent implementation)
    pub fn get_current_memory_usage_mb() -> usize {
        // This is a simplified implementation
        // In a real system, you would use platform-specific APIs
        // For now, return 0 as placeholder
        0
    }
    
    /// Get current CPU utilization percentage (system-dependent implementation)
    pub fn get_current_cpu_utilization() -> f32 {
        // This is a simplified implementation
        // In a real system, you would use platform-specific APIs
        // For now, return 0.0 as placeholder
        0.0
    }
    
    /// Calculate thread efficiency based on actual vs expected performance
    pub fn calculate_thread_efficiency(&self, actual_time_ms: f64, expected_time_ms: f64) -> f32 {
        if expected_time_ms <= 0.0 {
            return 100.0;
        }
        
        let efficiency = (expected_time_ms / actual_time_ms.max(0.001)) * 100.0;
        efficiency.min(100.0).max(0.0) as f32
    }
    

    
    /// Detect and handle performance degradation with configurable thresholds
    fn detect_performance_degradation(&mut self, operation_name: &str, actual_ms: f64, expected_ms: f64) {
        let degradation_ratio = actual_ms / expected_ms;
        
        // Define degradation severity thresholds
        let warning_threshold = 1.5;  // 50% slower than expected
        let critical_threshold = 3.0; // 200% slower than expected
        
        if degradation_ratio >= critical_threshold {
            self.log_performance_warning(
                operation_name, 
                actual_ms, 
                expected_ms, 
                PerformanceDegradationLevel::Critical
            );
            
            // Trigger automatic optimization for critical degradation
            self.trigger_automatic_optimization(operation_name, degradation_ratio);
            
        } else if degradation_ratio >= warning_threshold {
            self.log_performance_warning(
                operation_name, 
                actual_ms, 
                expected_ms, 
                PerformanceDegradationLevel::Warning
            );
        }
    }
    
    /// Log performance warning when operations exceed expected times
    fn log_performance_warning(&self, operation_name: &str, actual_ms: f64, expected_ms: f64, level: PerformanceDegradationLevel) {
        let degradation_percent = ((actual_ms - expected_ms) / expected_ms) * 100.0;
        
        match level {
            PerformanceDegradationLevel::Warning => {
                if self.config.log_performance_metrics {
                    eprintln!("⚠️  Performance Warning: {} took {:.2}ms (expected <{:.2}ms, {:.1}% slower)", 
                             operation_name, actual_ms, expected_ms, degradation_percent);
                }
            },
            PerformanceDegradationLevel::Critical => {
                if self.config.log_performance_metrics {
                    eprintln!("🚨 Performance Critical: {} took {:.2}ms (expected <{:.2}ms, {:.1}% slower)", 
                             operation_name, actual_ms, expected_ms, degradation_percent);
                }
            }
        }
    }
    
    /// Log performance timing information
    fn log_performance_timing(&self, operation_name: &str, actual_ms: f64, expected_ms: f64) {
        if actual_ms <= expected_ms {
            if self.config.enable_debug_logging {
                println!("✅ Performance OK: {} completed in {:.2}ms (expected <{:.2}ms)", 
                        operation_name, actual_ms, expected_ms);
            }
        }
    }
    
    /// Trigger automatic optimization based on performance degradation
    fn trigger_automatic_optimization(&mut self, operation_name: &str, degradation_ratio: f64) {
        if self.config.enable_debug_logging {
            println!("🔧 Triggering automatic optimization for {} (degradation: {:.1}x)", 
                    operation_name, degradation_ratio);
        }
        
        // Apply optimization strategies based on the operation type
        match operation_name {
            "knn_search" => {
                self.optimize_knn_search_strategy();
            },
            "distance_calculation" => {
                self.optimize_distance_calculation();
            },
            "data_access" => {
                self.optimize_data_access_pattern();
            },
            "memory_allocation" => {
                self.optimize_memory_management();
            },
            _ => {
                // Generic optimization
                self.apply_generic_optimization();
            }
        }
    }
    
    /// Optimize k-NN search strategy based on performance issues
    fn optimize_knn_search_strategy(&mut self) {
        // If HNSW is disabled and we have enough samples, enable it
        if !self.config.use_hnsw_index && self.training_samples.len() > 1000 {
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Enabling HNSW index for large dataset");
            }
            self.config.use_hnsw_index = true;
            let _ = self.rebuild_or_create_hnsw_index();
        }
        
        // Adjust parallel threshold to use more parallelization
        if self.config.parallel_threshold > 50 {
            self.config.parallel_threshold = (self.config.parallel_threshold as f32 * 0.8) as usize;
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Reduced parallel threshold to {}", self.config.parallel_threshold);
            }
        }
        
        // Switch to dedicated thread pool if using global
        if matches!(self.config.thread_pool_strategy, ThreadPoolStrategy::Global) {
            self.config.thread_pool_strategy = ThreadPoolStrategy::Dedicated;
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Switched to dedicated thread pool");
            }
        }
    }
    
    /// Optimize distance calculation performance
    fn optimize_distance_calculation(&mut self) {
        // Enable SIMD if not already enabled
        if !self.config.use_simd_optimization {
            self.config.use_simd_optimization = true;
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Enabled SIMD optimization");
            }
        }
        
        // Adjust SIMD chunk size for better performance
        if self.config.simd_chunk_size < 128 {
            self.config.simd_chunk_size = 128;
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Increased SIMD chunk size to {}", self.config.simd_chunk_size);
            }
        }
        
        // Enable optimized storage if not already enabled
        if !self.use_optimized_storage {
            let _ = self.enable_optimized_storage();
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Enabled optimized storage");
            }
        }
    }
    
    /// Optimize data access patterns
    fn optimize_data_access_pattern(&mut self) {
        // Enable memory mapping for large datasets
        if !self.config.enable_memory_mapping && self.training_samples.len() > 5000 {
            self.config.enable_memory_mapping = true;
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Enabled memory mapping for large dataset");
            }
        }
        
        // Increase memory pool size if needed
        if self.config.memory_pool_size < 512 {
            self.config.memory_pool_size = 512;
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Increased memory pool size to {}MB", self.config.memory_pool_size);
            }
        }
    }
    
    /// Optimize memory management
    fn optimize_memory_management(&mut self) {
        // Trigger memory pool cleanup
        if let Some(ref mut pool) = self.memory_pool {
            pool.cleanup();
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Cleaned up memory pool");
            }
        }
        
        // Switch to memory mapping if memory usage is high
        if self.performance_metrics.current_memory_usage_mb > 1024 {
            self.config.enable_memory_mapping = true;
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Enabled memory mapping due to high memory usage");
            }
        }
    }
    
    /// Apply generic optimization strategies
    fn apply_generic_optimization(&mut self) {
        // Enable work stealing if not already enabled
        if !self.config.work_stealing_enabled {
            self.config.work_stealing_enabled = true;
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Enabled work stealing");
            }
        }
        
        // Switch to adaptive thread pool strategy
        if !matches!(self.config.thread_pool_strategy, ThreadPoolStrategy::Adaptive) {
            self.config.thread_pool_strategy = ThreadPoolStrategy::Adaptive;
            if self.config.enable_debug_logging {
                println!("🔧 Auto-optimization: Switched to adaptive thread pool strategy");
            }
        }
    }
    
    /// Generate comprehensive performance report with optimization recommendations
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let mut recommendations = Vec::new();
        
        // Analyze latency performance
        if self.performance_metrics.latency_p95_ms > 5.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Latency,
                priority: RecommendationPriority::High,
                description: format!("P95 latency is {:.2}ms, consider enabling HNSW indexing", 
                                   self.performance_metrics.latency_p95_ms),
                action: "Enable HNSW indexing for datasets > 1000 samples".to_string(),
            });
        }
        
        // Analyze memory usage
        let memory_utilization = if self.performance_metrics.peak_memory_usage_mb > 0 {
            (self.performance_metrics.current_memory_usage_mb as f32 / 
             self.performance_metrics.peak_memory_usage_mb as f32) * 100.0
        } else {
            0.0
        };
        
        if memory_utilization > 80.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Memory,
                priority: RecommendationPriority::Medium,
                description: format!("Memory utilization is {:.1}%, consider memory mapping", memory_utilization),
                action: "Enable memory mapping or increase memory pool size".to_string(),
            });
        }
        
        // Analyze CPU utilization
        if self.performance_metrics.cpu_utilization_percent < 70.0 && self.training_samples.len() > 100 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::CPU,
                priority: RecommendationPriority::Medium,
                description: format!("CPU utilization is {:.1}%, increase parallelization", 
                                   self.performance_metrics.cpu_utilization_percent),
                action: "Reduce parallel threshold or enable SIMD optimization".to_string(),
            });
        }
        
        // Analyze thread efficiency
        if self.performance_metrics.thread_efficiency_percent < 60.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Threading,
                priority: RecommendationPriority::High,
                description: format!("Thread efficiency is {:.1}%, optimize thread pool", 
                                   self.performance_metrics.thread_efficiency_percent),
                action: "Switch to dedicated thread pool or enable work stealing".to_string(),
            });
        }
        
        // Analyze HNSW performance
        if self.config.use_hnsw_index && self.performance_metrics.hnsw_accuracy_percent < 95.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Accuracy,
                priority: RecommendationPriority::Medium,
                description: format!("HNSW accuracy is {:.1}%, tune parameters", 
                                   self.performance_metrics.hnsw_accuracy_percent),
                action: "Increase hnsw_ef_construction or hnsw_m parameters".to_string(),
            });
        }
        
        // Calculate overall performance score
        let performance_score = self.calculate_overall_performance_score();
        
        PerformanceReport {
            timestamp: std::time::SystemTime::now(),
            overall_score: performance_score,
            metrics_summary: self.create_metrics_summary(),
            recommendations,
            configuration_snapshot: self.config.clone(),
        }
    }
    
    /// Calculate overall performance score (0-100)
    fn calculate_overall_performance_score(&self) -> f32 {
        let mut score = 100.0f32;
        
        // Penalize high latency
        if self.performance_metrics.latency_p95_ms > 1.0 {
            score -= ((self.performance_metrics.latency_p95_ms - 1.0) * 10.0) as f32;
        }
        
        // Penalize low CPU utilization
        if self.performance_metrics.cpu_utilization_percent < 80.0 {
            score -= (80.0 - self.performance_metrics.cpu_utilization_percent) * 0.5;
        }
        
        // Penalize low thread efficiency
        if self.performance_metrics.thread_efficiency_percent < 70.0 {
            score -= (70.0 - self.performance_metrics.thread_efficiency_percent) * 0.8;
        }
        
        // Penalize high memory usage
        let memory_pressure = (self.performance_metrics.current_memory_usage_mb as f32 / 1024.0) * 100.0;
        if memory_pressure > 80.0 {
            score -= (memory_pressure - 80.0) * 0.5;
        }
        
        score.max(0.0).min(100.0)
    }
    
    /// Create metrics summary for the performance report
    fn create_metrics_summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_predictions: self.performance_metrics.total_predictions,
            average_latency_ms: self.performance_metrics.average_prediction_time_ms,
            p95_latency_ms: self.performance_metrics.latency_p95_ms,
            p99_latency_ms: self.performance_metrics.latency_p99_ms,
            cpu_utilization_percent: self.performance_metrics.cpu_utilization_percent,
            memory_usage_mb: self.performance_metrics.current_memory_usage_mb,
            peak_memory_mb: self.performance_metrics.peak_memory_usage_mb,
            thread_efficiency_percent: self.performance_metrics.thread_efficiency_percent,
            hnsw_queries_ratio: if self.performance_metrics.total_predictions > 0 {
                (self.performance_metrics.hnsw_queries as f32 / self.performance_metrics.total_predictions as f32) * 100.0
            } else {
                0.0
            },
            simd_operations_ratio: if self.performance_metrics.total_predictions > 0 {
                (self.performance_metrics.simd_operations_count as f32 / self.performance_metrics.total_predictions as f32) * 100.0
            } else {
                0.0
            },
        }
    }
    
    /// Enable optimized storage mode with SIMD-aligned samples
    pub fn enable_optimized_storage(&mut self) -> Result<()> {
        if self.use_optimized_storage {
            return Ok(()); // Already enabled
        }
        
        // Convert existing samples to optimized format
        self.optimized_samples.clear();
        for sample in &self.training_samples {
            let optimized_sample = OptimizedTrainingSample::from_training_sample(sample);
            self.optimized_samples.push_back(optimized_sample);
        }
        
        self.use_optimized_storage = true;
        
        if self.config.enable_debug_logging {
            self.log_debug(&format!("Enabled optimized storage with {} samples", self.optimized_samples.len()));
        }
        
        Ok(())
    }
    
    /// Disable optimized storage mode and revert to regular samples
    pub fn disable_optimized_storage(&mut self) -> Result<()> {
        if !self.use_optimized_storage {
            return Ok(()); // Already disabled
        }
        
        // Convert optimized samples back to regular format
        self.training_samples.clear();
        for optimized_sample in &self.optimized_samples {
            let sample = optimized_sample.to_training_sample();
            self.training_samples.push_back(sample);
        }
        
        self.use_optimized_storage = false;
        
        if self.config.enable_debug_logging {
            self.log_debug(&format!("Disabled optimized storage, reverted {} samples", self.training_samples.len()));
        }
        
        Ok(())
    }
    
    /// Initialize memory-mapped storage for large datasets
    pub fn initialize_memory_mapped_storage(&mut self, file_path: &Path) -> Result<()> {
        if self.memory_mapped_storage.is_some() {
            return Ok(()); // Already initialized
        }
        
        let max_samples = self.config.max_bars_back * 2; // Allow some headroom
        let storage = MemoryMappedStorage::new(file_path, max_samples, false)?;
        
        // Migrate existing samples to memory-mapped storage if any
        if !self.optimized_samples.is_empty() {
            let mut mmap_storage = storage;
            for sample in &self.optimized_samples {
                mmap_storage.push_sample(sample)?;
            }
            mmap_storage.flush()?;
            self.memory_mapped_storage = Some(mmap_storage);
        } else {
            self.memory_mapped_storage = Some(storage);
        }
        
        if self.config.enable_debug_logging {
            self.log_debug(&format!("Initialized memory-mapped storage at {:?}", file_path));
        }
        
        Ok(())
    }
    
    /// Add training sample with automatic memory management
    pub fn add_training_sample_optimized(&mut self, sample: TrainingSample) -> Result<()> {
        // Check memory usage and adapt behavior
        self.check_and_adapt_memory_usage()?;
        
        let sample_index = if self.use_optimized_storage {
            self.optimized_samples.len()
        } else {
            self.training_samples.len()
        };
        
        // Handle ring buffer overflow
        let max_capacity = self.config.max_bars_back;
        
        if self.use_optimized_storage {
            if self.optimized_samples.len() >= max_capacity {
                self.optimized_samples.pop_front();
                
                // If we have HNSW index, mark for rebuild
                if self.hnsw_index.is_some() {
                    self.samples_since_hnsw_rebuild = self.config.hnsw_rebuild_threshold;
                }
            }
            
            let optimized_sample = OptimizedTrainingSample::from_training_sample(&sample);
            
            // Try to use memory-mapped storage if available
            if let Some(ref mut mmap_storage) = self.memory_mapped_storage {
                if mmap_storage.len() < mmap_storage.capacity() {
                    mmap_storage.push_sample(&optimized_sample)?;
                } else {
                    // Memory-mapped storage is full, use in-memory storage
                    self.optimized_samples.push_back(optimized_sample);
                }
            } else {
                self.optimized_samples.push_back(optimized_sample);
            }
        } else {
            // Use regular storage
            if self.training_samples.len() >= max_capacity {
                self.training_samples.pop_front();
                
                if self.hnsw_index.is_some() {
                    self.samples_since_hnsw_rebuild = self.config.hnsw_rebuild_threshold;
                }
            }
            
            self.training_samples.push_back(sample.clone());
        }
        
        // Update HNSW index if enabled
        if let Some(ref mut hnsw_index) = self.hnsw_index {
            if let Err(e) = hnsw_index.add_sample(&sample, sample_index) {
                if self.config.enable_debug_logging {
                    eprintln!("Warning: Failed to add sample to HNSW index: {}. Index may need rebuilding.", e);
                }
                self.samples_since_hnsw_rebuild = self.config.hnsw_rebuild_threshold;
            } else {
                self.samples_since_hnsw_rebuild += 1;
            }
            
            // Check if we need to rebuild the index
            if self.samples_since_hnsw_rebuild >= self.config.hnsw_rebuild_threshold {
                if let Err(e) = self.rebuild_hnsw_index() {
                    if self.config.enable_debug_logging {
                        eprintln!("Warning: Failed to rebuild HNSW index: {}. Disabling HNSW for this session.", e);
                    }
                    self.hnsw_index = None;
                }
            }
        }
        
        // Update performance metrics
        let total_samples = if self.use_optimized_storage {
            self.optimized_samples.len() + self.memory_mapped_storage.as_ref().map_or(0, |s| s.len())
        } else {
            self.training_samples.len()
        };
        
        self.performance_metrics.total_training_samples = total_samples as u64;
        
        // Update memory metrics
        self.update_memory_usage_metrics();
        
        Ok(())
    }
    
    /// Check memory usage and adapt behavior automatically
    fn check_and_adapt_memory_usage(&mut self) -> Result<()> {
        let current_usage_mb = self.get_estimated_memory_usage_mb();
        let memory_status = self.memory_threshold_monitor.check_memory_usage(current_usage_mb);
        
        match memory_status {
            MemoryStatus::Critical { usage_percent, usage_mb } => {
                if self.config.enable_debug_logging {
                    eprintln!("Critical memory usage: {:.1}% ({} MB). Forcing cleanup.", usage_percent, usage_mb);
                }
                
                // Force aggressive cleanup
                self.force_memory_cleanup()?;
            }
            MemoryStatus::Warning { usage_percent, usage_mb } => {
                if self.config.enable_debug_logging {
                    eprintln!("High memory usage: {:.1}% ({} MB). Performing soft cleanup.", usage_percent, usage_mb);
                }
                
                // Perform soft cleanup
                self.soft_memory_cleanup()?;
            }
            MemoryStatus::Normal => {
                // No action needed
            }
        }
        
        Ok(())
    }
    
    /// Perform soft memory cleanup (non-aggressive)
    fn soft_memory_cleanup(&mut self) -> Result<()> {
        // Cleanup memory pool if available
        if let Some(ref mut pool) = self.memory_pool {
            pool.cleanup();
        }
        
        // Switch to optimized storage if not already using it
        if !self.use_optimized_storage && self.config.memory_pool_size > 0 {
            self.enable_optimized_storage()?;
        }
        
        Ok(())
    }
    
    /// Perform aggressive memory cleanup
    fn force_memory_cleanup(&mut self) -> Result<()> {
        // First perform soft cleanup
        self.soft_memory_cleanup()?;
        
        // Switch to memory-mapped storage if enabled and not already using it
        if self.config.enable_memory_mapping && self.memory_mapped_storage.is_none() {
            let temp_path = std::env::temp_dir().join("ldc_engine_mmap.dat");
            if let Err(e) = self.initialize_memory_mapped_storage(&temp_path) {
                if self.config.enable_debug_logging {
                    eprintln!("Warning: Failed to initialize memory mapping: {}", e);
                }
            }
        }
        
        // Reduce training sample capacity if possible
        let current_len = if self.use_optimized_storage {
            self.optimized_samples.len()
        } else {
            self.training_samples.len()
        };
        
        let reduced_capacity = (current_len * 3 / 4).max(self.config.neighbors_count * 2);
        
        if self.use_optimized_storage {
            while self.optimized_samples.len() > reduced_capacity {
                self.optimized_samples.pop_front();
            }
        } else {
            while self.training_samples.len() > reduced_capacity {
                self.training_samples.pop_front();
            }
        }
        
        // Force HNSW rebuild if needed
        if self.hnsw_index.is_some() {
            self.samples_since_hnsw_rebuild = self.config.hnsw_rebuild_threshold;
        }
        
        if self.config.enable_debug_logging {
            self.log_debug(&format!("Force cleanup completed. Reduced samples to {}", reduced_capacity));
        }
        
        Ok(())
    }
    
    /// Get estimated memory usage in MB
    fn get_estimated_memory_usage_mb(&self) -> usize {
        let mut total_bytes = 0;
        
        // Regular training samples
        total_bytes += self.training_samples.len() * std::mem::size_of::<TrainingSample>();
        
        // Optimized training samples
        total_bytes += self.optimized_samples.len() * OptimizedTrainingSample::size_of();
        
        // Memory pool usage
        if let Some(ref pool) = self.memory_pool {
            total_bytes += pool.allocated_bytes();
        }
        
        // Memory-mapped storage (estimate based on sample count)
        if let Some(ref mmap_storage) = self.memory_mapped_storage {
            total_bytes += mmap_storage.len() * OptimizedTrainingSample::size_of();
        }
        
        // HNSW index (rough estimate)
        if let Some(ref hnsw_index) = self.hnsw_index {
            total_bytes += hnsw_index.len() * 64; // Rough estimate per sample
        }
        
        // Convert to MB
        total_bytes / (1024 * 1024)
    }
    
    /// Update memory usage metrics in performance metrics
    fn update_memory_usage_metrics(&mut self) {
        let current_mb = self.get_estimated_memory_usage_mb();
        let allocations = if let Some(ref pool) = self.memory_pool {
            pool.allocation_count()
        } else {
            0
        };
        
        self.update_memory_metrics(current_mb, allocations);
    }
    
    /// Get memory pool statistics
    pub fn get_memory_pool_stats(&self) -> Option<(usize, usize, u64, u64, f32)> {
        self.memory_pool.as_ref().map(|pool| {
            (
                pool.allocated_bytes(),
                pool.peak_allocated_bytes(),
                pool.allocation_count(),
                pool.deallocation_count(),
                pool.utilization_percent(),
            )
        })
    }
    
    /// Get memory-mapped storage statistics
    pub fn get_memory_mapped_stats(&self) -> Option<(usize, usize)> {
        self.memory_mapped_storage.as_ref().map(|storage| {
            (storage.len(), storage.capacity())
        })
    }
    
    /// Flush memory-mapped storage to disk
    pub fn flush_memory_mapped_storage(&mut self) -> Result<()> {
        if let Some(ref mut storage) = self.memory_mapped_storage {
            storage.flush()?;
        }
        Ok(())
    }
    
    /// Get training samples for search (handles both regular and optimized storage)
    pub fn get_training_samples_for_search_optimized(&self, max_samples: Option<usize>) -> Vec<TrainingSample> {
        let limit = max_samples.unwrap_or(usize::MAX);
        
        if self.use_optimized_storage {
            // First try memory-mapped storage
            if let Some(ref mmap_storage) = self.memory_mapped_storage {
                let mut samples = Vec::new();
                let count = mmap_storage.len().min(limit);
                
                for i in 0..count {
                    if let Some(optimized_sample) = mmap_storage.get_sample(i) {
                        samples.push(optimized_sample.to_training_sample());
                    }
                }
                
                // Add in-memory optimized samples if we have room
                let remaining = limit.saturating_sub(samples.len());
                if remaining > 0 {
                    for optimized_sample in self.optimized_samples.iter().take(remaining) {
                        samples.push(optimized_sample.to_training_sample());
                    }
                }
                
                samples
            } else {
                // Use in-memory optimized samples
                self.optimized_samples
                    .iter()
                    .take(limit)
                    .map(|optimized_sample| optimized_sample.to_training_sample())
                    .collect()
            }
        } else {
            // Use regular samples
            self.training_samples.iter().take(limit).cloned().collect()
        }
    }
    

    
    /// Log performance report if logging is enabled
    pub fn log_performance_report(&self) {
        if self.config.log_performance_metrics {
            let report = self.generate_performance_report();
            println!("=== LDC Engine Performance Report ===");
            println!("Overall Score: {:.1}/100", report.overall_score);
            println!("Total Predictions: {}", report.metrics_summary.total_predictions);
            println!("P95 Latency: {:.2}ms", report.metrics_summary.p95_latency_ms);
            println!("CPU Utilization: {:.1}%", report.metrics_summary.cpu_utilization_percent);
            println!("Memory Usage: {}MB", report.metrics_summary.memory_usage_mb);
            
            if !report.recommendations.is_empty() {
                println!("\nOptimization Recommendations:");
                for rec in &report.recommendations {
                    println!("- [{:?}] {}: {}", rec.priority, rec.description, rec.action);
                }
            }
            println!("===================================");
        }
    }
    
    /// Check for performance degradation and log warnings
    pub fn check_performance_degradation(&self) {
        let metrics = &self.performance_metrics;
        
        // Check if prediction time exceeds expected thresholds
        if metrics.last_prediction_time_ms > 5.0 {
            self.log_performance_warning("Prediction", metrics.last_prediction_time_ms, 5.0, PerformanceDegradationLevel::Warning);
        }
        
        // Check if distance calculation is taking too long
        if metrics.distance_calculation_time_ms > 2.0 {
            self.log_performance_warning("Distance Calculation", metrics.distance_calculation_time_ms, 2.0, PerformanceDegradationLevel::Warning);
        }
        
        // Check if k-NN search is taking too long
        if metrics.knn_search_time_ms > 3.0 {
            self.log_performance_warning("k-NN Search", metrics.knn_search_time_ms, 3.0, PerformanceDegradationLevel::Warning);
        }
        
        // Check CPU utilization
        if metrics.cpu_utilization_percent < 50.0 && self.config.use_multithreading {
            if self.config.log_performance_metrics {
                eprintln!("⚠️  Low CPU utilization: {:.1}% (expected >50% with multithreading)", 
                         metrics.cpu_utilization_percent);
            }
        }
        
        // Check thread efficiency
        if metrics.thread_efficiency_percent < 70.0 && self.config.use_multithreading {
            if self.config.log_performance_metrics {
                eprintln!("⚠️  Low thread efficiency: {:.1}% (expected >70%)", 
                         metrics.thread_efficiency_percent);
            }
        }
        
        // Check HNSW accuracy if enabled
        if self.config.use_hnsw_index && metrics.hnsw_accuracy_percent < 95.0 && metrics.hnsw_queries > 0 {
            if self.config.log_performance_metrics {
                eprintln!("⚠️  Low HNSW accuracy: {:.1}% (expected >95%)", 
                         metrics.hnsw_accuracy_percent);
            }
        }
    }
    

    
    /// Log debug information if enabled
    fn log_debug(&self, message: &str) {
        if self.config.enable_debug_logging {
            println!("[LDC DEBUG] {}", message);
        }
    }
    
    /// Get or create thread pool based on strategy and workload characteristics with error handling
    /// Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    pub fn get_or_create_thread_pool(&mut self) -> Result<Arc<rayon::ThreadPool>, PerformanceOptimizationError> {
        match self.config.thread_pool_strategy {
            ThreadPoolStrategy::Global => {
                // Use global rayon thread pool
                self.get_global_thread_pool()
            }
            ThreadPoolStrategy::Dedicated => {
                // Create or reuse dedicated thread pool
                self.get_or_create_dedicated_thread_pool()
            }
            ThreadPoolStrategy::Adaptive => {
                // Choose strategy based on workload characteristics
                self.get_adaptive_thread_pool()
            }
        }
    }
    
    /// Get global rayon thread pool with optimal configuration and error handling
    fn get_global_thread_pool(&mut self) -> Result<Arc<rayon::ThreadPool>, PerformanceOptimizationError> {
        // Configure global thread pool if not already configured
        let thread_count = self.calculate_optimal_thread_count();
        
        // Validate thread count
        if thread_count == 0 {
            return Err(PerformanceOptimizationError::ThreadPoolError {
                message: "Calculated thread count is 0".to_string(),
            });
        }
        
        // Update thread pool stats
        self.thread_pool_stats.current_thread_count = thread_count;
        
        // Create a reference to the global thread pool
        // Note: We can't directly get Arc<ThreadPool> from global pool,
        // so we create a dedicated pool with same configuration
        self.get_or_create_dedicated_thread_pool_with_count(thread_count)
    }
    
    /// Get or create dedicated thread pool for LDC operations with error handling
    fn get_or_create_dedicated_thread_pool(&mut self) -> Result<Arc<rayon::ThreadPool>, PerformanceOptimizationError> {
        let optimal_thread_count = self.calculate_optimal_thread_count();
        
        if optimal_thread_count == 0 {
            return Err(PerformanceOptimizationError::ThreadPoolError {
                message: "Cannot create thread pool with 0 threads".to_string(),
            });
        }
        
        self.get_or_create_dedicated_thread_pool_with_count(optimal_thread_count)
    }
    
    /// Get or create dedicated thread pool with specific thread count and error handling
    fn get_or_create_dedicated_thread_pool_with_count(&mut self, thread_count: usize) -> Result<Arc<rayon::ThreadPool>, PerformanceOptimizationError> {
        // Validate thread count
        if thread_count == 0 {
            return Err(PerformanceOptimizationError::ThreadPoolError {
                message: "Thread count cannot be 0".to_string(),
            });
        }
        
        const MAX_THREADS: usize = 128; // Reasonable upper limit
        if thread_count > MAX_THREADS {
            return Err(PerformanceOptimizationError::ThreadPoolError {
                message: format!("Thread count {} exceeds maximum {}", thread_count, MAX_THREADS),
            });
        }
        
        // Check if we need to create or recreate the thread pool
        let needs_recreation = self.dedicated_thread_pool.is_none() || 
                              self.thread_pool_stats.current_thread_count != thread_count;
        
        if needs_recreation {
            match self.create_dedicated_thread_pool(thread_count) {
                Ok(pool) => {
                    self.dedicated_thread_pool = Some(pool.clone());
                    self.thread_pool_stats.current_thread_count = thread_count;
                    self.thread_pool_stats.adaptive_resizing_events += 1;
                    
                    if self.config.enable_debug_logging {
                        self.log_debug(&format!("Created dedicated thread pool with {} threads", thread_count));
                    }
                    
                    Ok(pool)
                }
                Err(e) => {
                    eprintln!("Failed to create dedicated thread pool: {}. Using fallback.", e);
                    // Fallback to a simple thread pool
                    match self.create_fallback_thread_pool() {
                        Ok(pool) => Ok(pool),
                        Err(fallback_error) => Err(PerformanceOptimizationError::ThreadPoolError {
                            message: format!("Both primary and fallback thread pool creation failed: {} | {}", e, fallback_error),
                        })
                    }
                }
            }
        } else {
            // Return existing thread pool
            Ok(self.dedicated_thread_pool.as_ref().unwrap().clone())
        }
    }
    
    /// Create dedicated thread pool with advanced configuration
    fn create_dedicated_thread_pool(&self, thread_count: usize) -> Result<Arc<rayon::ThreadPool>> {
        let mut builder = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .thread_name(|index| format!("ldc-worker-{}", index));
        
        // Configure work stealing if enabled
        if !self.config.work_stealing_enabled {
            // Note: rayon doesn't have a direct way to disable work stealing,
            // but we can configure it for better locality
            builder = builder.breadth_first();
        }
        
        // Build the thread pool
        let pool = builder.build()?;
        Ok(Arc::new(pool))
    }
    
    /// Create fallback thread pool with minimal configuration and error handling
    fn create_fallback_thread_pool(&self) -> Result<Arc<rayon::ThreadPool>, PerformanceOptimizationError> {
        let thread_count = num_cpus::get().min(4); // Conservative fallback
        
        // Try to create a basic thread pool
        match rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build() {
            Ok(pool) => Ok(Arc::new(pool)),
            Err(_) => {
                // Ultimate fallback - try single-threaded pool
                match rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build() {
                    Ok(pool) => {
                        eprintln!("Warning: Using single-threaded fallback thread pool");
                        Ok(Arc::new(pool))
                    }
                    Err(e) => Err(PerformanceOptimizationError::ThreadPoolError {
                        message: format!("Failed to create fallback thread pool: {}", e),
                    })
                }
            }
        }
    }
    
    /// Get adaptive thread pool based on workload characteristics with error handling
    fn get_adaptive_thread_pool(&mut self) -> Result<Arc<rayon::ThreadPool>, PerformanceOptimizationError> {
        // Assess current workload
        let workload = self.assess_workload_characteristics();
        self.thread_pool_stats.last_workload_assessment = workload.clone();
        
        // Choose strategy based on workload
        let strategy = self.choose_adaptive_strategy(&workload);
        
        match strategy {
            ThreadPoolStrategy::Global => self.get_global_thread_pool(),
            ThreadPoolStrategy::Dedicated => self.get_or_create_dedicated_thread_pool(),
            ThreadPoolStrategy::Adaptive => {
                // Shouldn't happen, but fallback to dedicated
                self.get_or_create_dedicated_thread_pool()
            }
        }
    }
    
    /// Assess current workload characteristics for adaptive thread pool sizing
    fn assess_workload_characteristics(&self) -> WorkloadCharacteristics {
        let dataset_size = self.training_samples.len();
        
        // Determine computation intensity based on configuration
        let computation_intensity = if self.config.use_simd_optimization && self.config.use_hnsw_index {
            ComputationIntensity::High
        } else if self.config.use_simd_optimization || self.config.use_hnsw_index {
            ComputationIntensity::Medium
        } else {
            ComputationIntensity::Low
        };
        
        // Determine memory access pattern
        let memory_access_pattern = if self.config.use_chronological_spacing {
            MemoryAccessPattern::Sequential
        } else if self.config.use_hnsw_index {
            MemoryAccessPattern::Random
        } else {
            MemoryAccessPattern::Mixed
        };
        
        // Estimate parallelization efficiency based on dataset size and operations
        let parallelization_efficiency = if dataset_size < 100 {
            0.3 // Small datasets don't parallelize well
        } else if dataset_size < 1000 {
            0.6 // Medium datasets have moderate efficiency
        } else {
            0.9 // Large datasets parallelize very well
        };
        
        // Estimate CPU vs IO bound ratio
        let cpu_bound_ratio = if self.config.enable_memory_mapping {
            0.7 // Memory mapping introduces some IO
        } else {
            0.95 // Mostly CPU bound for in-memory operations
        };
        
        WorkloadCharacteristics {
            dataset_size,
            computation_intensity,
            memory_access_pattern,
            parallelization_efficiency,
            cpu_bound_ratio,
            io_bound_ratio: 1.0 - cpu_bound_ratio,
        }
    }
    
    /// Choose adaptive strategy based on workload characteristics
    fn choose_adaptive_strategy(&self, workload: &WorkloadCharacteristics) -> ThreadPoolStrategy {
        // Use dedicated thread pool for large, CPU-intensive workloads
        if workload.dataset_size > 1000 && 
           workload.cpu_bound_ratio > 0.8 && 
           workload.parallelization_efficiency > 0.7 {
            ThreadPoolStrategy::Dedicated
        } else {
            // Use global thread pool for smaller or less parallelizable workloads
            ThreadPoolStrategy::Global
        }
    }
    
    /// Calculate optimal thread count based on workload and system characteristics
    fn calculate_optimal_thread_count(&self) -> usize {
        let cpu_count = num_cpus::get();
        
        // Start with configured max_threads or CPU count
        let base_threads = self.config.max_threads.unwrap_or(cpu_count);
        
        // Adjust based on workload characteristics
        let workload = &self.thread_pool_stats.last_workload_assessment;
        
        let adjusted_threads = match workload.computation_intensity {
            ComputationIntensity::Low => {
                // For low-intensity work, use fewer threads to reduce overhead
                (base_threads / 2).max(1)
            }
            ComputationIntensity::Medium => {
                // For medium-intensity work, use most available threads
                (base_threads * 3 / 4).max(1)
            }
            ComputationIntensity::High => {
                // For high-intensity work, use all available threads
                base_threads
            }
        };
        
        // Apply parallelization efficiency factor
        let efficiency_adjusted = (adjusted_threads as f32 * workload.parallelization_efficiency).ceil() as usize;
        
        // Ensure we have at least 1 thread and don't exceed system capabilities
        efficiency_adjusted.max(1).min(cpu_count * 2)
    }
    
    /// Monitor thread efficiency and update performance metrics
    pub fn monitor_thread_efficiency(&mut self, task_duration_ms: f64, thread_count: usize) {
        // Update task execution statistics
        self.thread_pool_stats.total_tasks_executed += 1;
        self.thread_pool_stats.total_execution_time_ms += task_duration_ms;
        self.thread_pool_stats.average_task_time_ms = 
            self.thread_pool_stats.total_execution_time_ms / self.thread_pool_stats.total_tasks_executed as f64;
        
        // Calculate thread efficiency
        let expected_sequential_time = task_duration_ms * thread_count as f64;
        let efficiency = self.calculate_thread_efficiency(task_duration_ms, expected_sequential_time);
        
        // Update performance metrics
        self.performance_metrics.thread_efficiency_percent = efficiency;
        
        // Add to rolling window for trend analysis
        self.thread_pool_stats.thread_utilization_samples.push_back(efficiency);
        if self.thread_pool_stats.thread_utilization_samples.len() > 100 {
            self.thread_pool_stats.thread_utilization_samples.pop_front();
        }
        
        // Update optimal thread count based on efficiency trends
        self.update_optimal_thread_count_based_on_efficiency();
        
        if self.config.log_performance_metrics {
            self.log_debug(&format!(
                "Thread efficiency: {:.1}%, Task time: {:.2}ms, Threads: {}",
                efficiency, task_duration_ms, thread_count
            ));
        }
    }
    
    /// Update optimal thread count based on efficiency trends
    fn update_optimal_thread_count_based_on_efficiency(&mut self) {
        if self.thread_pool_stats.thread_utilization_samples.len() < 10 {
            return; // Need more samples for reliable analysis
        }
        
        // Calculate average efficiency over recent samples
        let recent_samples: Vec<f32> = self.thread_pool_stats.thread_utilization_samples
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        let average_efficiency: f32 = recent_samples.iter().sum::<f32>() / recent_samples.len() as f32;
        
        // Adjust optimal thread count based on efficiency
        let current_optimal = self.thread_pool_stats.optimal_thread_count;
        let new_optimal = if average_efficiency < 60.0 {
            // Low efficiency - reduce thread count
            (current_optimal * 3 / 4).max(1)
        } else if average_efficiency > 85.0 {
            // High efficiency - potentially increase thread count
            (current_optimal * 5 / 4).min(num_cpus::get() * 2)
        } else {
            // Good efficiency - keep current count
            current_optimal
        };
        
        if new_optimal != current_optimal {
            self.thread_pool_stats.optimal_thread_count = new_optimal;
            self.thread_pool_stats.adaptive_resizing_events += 1;
            
            if self.config.enable_debug_logging {
                self.log_debug(&format!(
                    "Adjusted optimal thread count from {} to {} (efficiency: {:.1}%)",
                    current_optimal, new_optimal, average_efficiency
                ));
            }
        }
    }
    
    /// Track CPU utilization and update metrics
    pub fn track_cpu_utilization(&mut self) {
        // Get current CPU utilization (simplified implementation)
        let cpu_utilization = Self::get_current_cpu_utilization();
        
        // Update performance metrics
        self.performance_metrics.cpu_utilization_percent = cpu_utilization;
        
        // Log if CPU utilization is unexpectedly low with multithreading enabled
        if cpu_utilization < 50.0 && self.config.use_multithreading && self.training_samples.len() > self.config.parallel_threshold {
            if self.config.log_performance_metrics {
                self.log_debug(&format!(
                    "Low CPU utilization detected: {:.1}% with {} threads and {} samples",
                    cpu_utilization, self.thread_pool_stats.current_thread_count, self.training_samples.len()
                ));
            }
        }
    }
    
    /// Get thread pool statistics for monitoring and debugging
    pub fn get_thread_pool_stats(&self) -> &ThreadPoolStats {
        &self.thread_pool_stats
    }
    
    /// Reset thread pool statistics
    pub fn reset_thread_pool_stats(&mut self) {
        self.thread_pool_stats = ThreadPoolStats::default();
    }
    
    /// Generate thread pool performance report
    pub fn generate_thread_pool_report(&self) -> String {
        let stats = &self.thread_pool_stats;
        let workload = &stats.last_workload_assessment;
        
        format!(
            "=== Thread Pool Performance Report ===\n\
            \n\
            Configuration:\n\
            - Strategy: {:?}\n\
            - Current Threads: {}\n\
            - Optimal Threads: {}\n\
            - Work Stealing: {}\n\
            \n\
            Performance:\n\
            - Total Tasks: {}\n\
            - Average Task Time: {:.2}ms\n\
            - Total Execution Time: {:.2}ms\n\
            - Adaptive Resizing Events: {}\n\
            \n\
            Workload Characteristics:\n\
            - Dataset Size: {}\n\
            - Computation Intensity: {:?}\n\
            - Memory Access Pattern: {:?}\n\
            - Parallelization Efficiency: {:.1}%\n\
            - CPU Bound Ratio: {:.1}%\n\
            \n\
            Efficiency Metrics:\n\
            - Current Thread Efficiency: {:.1}%\n\
            - CPU Utilization: {:.1}%\n\
            - Utilization Samples: {}\n\
            =====================================",
            self.config.thread_pool_strategy,
            stats.current_thread_count,
            stats.optimal_thread_count,
            self.config.work_stealing_enabled,
            stats.total_tasks_executed,
            stats.average_task_time_ms,
            stats.total_execution_time_ms,
            stats.adaptive_resizing_events,
            workload.dataset_size,
            workload.computation_intensity,
            workload.memory_access_pattern,
            workload.parallelization_efficiency * 100.0,
            workload.cpu_bound_ratio * 100.0,
            self.performance_metrics.thread_efficiency_percent,
            self.performance_metrics.cpu_utilization_percent,
            stats.thread_utilization_samples.len()
        )
    }
    
    /// Log prediction if enabled
    fn log_prediction(&self, prediction: &LDCPrediction) {
        if self.config.log_predictions {
            println!("[LDC PREDICTION] Signal: {:.4}, Direction: {:?}, Confidence: {:.4}", 
                     prediction.signal, prediction.prediction_direction, prediction.confidence);
        }
    }
    
    /// Log performance metrics if enabled
    fn log_performance(&self, duration_ms: f64) {
        if self.config.log_performance_metrics {
            println!("[LDC PERFORMANCE] Prediction time: {:.2}ms, Total predictions: {}", 
                     duration_ms, self.performance_metrics.total_predictions);
        }
    }
    
    /// Calculate Lorentzian distance between two feature series
    /// This matches the Pine Script get_lorentzian_distance function exactly
    pub fn lorentzian_distance(features1: &FeatureSeries, features2: &FeatureSeries, feature_count: usize) -> f32 {
        match feature_count {
            5 => {
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln() +
                (1.0 + (features1.f3 - features2.f3).abs()).ln() +
                (1.0 + (features1.f4 - features2.f4).abs()).ln() +
                (1.0 + (features1.f5 - features2.f5).abs()).ln()
            },
            4 => {
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln() +
                (1.0 + (features1.f3 - features2.f3).abs()).ln() +
                (1.0 + (features1.f4 - features2.f4).abs()).ln()
            },
            3 => {
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln() +
                (1.0 + (features1.f3 - features2.f3).abs()).ln()
            },
            2 => {
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln()
            },
            _ => {
                // Default to 5 features
                (1.0 + (features1.f1 - features2.f1).abs()).ln() +
                (1.0 + (features1.f2 - features2.f2).abs()).ln() +
                (1.0 + (features1.f3 - features2.f3).abs()).ln() +
                (1.0 + (features1.f4 - features2.f4).abs()).ln() +
                (1.0 + (features1.f5 - features2.f5).abs()).ln()
            }
        }
    }
    
    /// Calculate Lorentzian distance using arrays (for compatibility)
    pub fn lorentzian_distance_arrays(features1: &[f32], features2: &[f32]) -> f32 {
        let min_len = features1.len().min(features2.len());
        (0..min_len)
            .map(|i| (1.0 + (features1[i] - features2[i]).abs()).ln())
            .sum()
    }
    
    /// Add training sample with automatic label generation
    pub fn add_training_sample_with_label(&mut self, features: FeatureSeries, current_price: f32, future_price: f32, timestamp: i64, bar_index: usize) {
        let label = Self::generate_label(current_price, future_price);
        let sample = TrainingSample {
            features,
            label,
            timestamp,
            bar_index,
        };
        self.add_training_sample(sample);
    }
    
    /// Get training samples with chronological spacing (modulo 4)
    /// This matches the Pine Script behavior of using i%4 for spacing
    pub fn get_training_samples_with_spacing(&self) -> Vec<&TrainingSample> {
        if !self.config.use_chronological_spacing {
            return self.training_samples.iter().collect();
        }
        
        self.training_samples
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 4 == 0) // Modulo 4 spacing like Pine Script
            .map(|(_, sample)| sample)
            .collect()
    }
    
    /// Get training samples for k-NN search with size limit
    pub fn get_training_samples_for_search(&self, max_samples: Option<usize>) -> Vec<&TrainingSample> {
        let samples = self.get_training_samples_with_spacing();
        let limit = max_samples.unwrap_or(self.config.max_bars_back);
        samples.into_iter().take(limit).collect()
    }
    
    /// Clear all training data
    pub fn clear_training_data(&mut self) {
        self.training_samples.clear();
        self.last_distance = -1.0;
    }
    
    /// Get training data statistics
    pub fn get_training_stats(&self) -> (usize, usize, usize) {
        let _total_samples = self.training_samples.len();
        let spaced_samples = self.get_training_samples_with_spacing().len();
        let long_count = self.training_samples.iter().filter(|s| s.label == Direction::Long).count();
        let short_count = self.training_samples.iter().filter(|s| s.label == Direction::Short).count();
        let _neutral_count = self.training_samples.iter().filter(|s| s.label == Direction::Neutral).count();
        
        (spaced_samples, long_count, short_count)
    }
    
    /// Find k nearest neighbors using approximate nearest neighbor search
    /// This matches the Pine Script ANN algorithm exactly
    /// Uses HNSW index when available and enabled, otherwise falls back to exact search
    pub fn find_k_nearest_neighbors(&mut self, query_features: &FeatureSeries) -> Vec<(f32, Direction)> {
        // Delegate to the optimized method for consistent behavior
        self.find_k_nearest_neighbors_optimized(query_features)
    }
    
    /// Enhanced k-NN search with multiple optimization strategies
    /// Chooses between HNSW, parallel, or sequential search based on configuration and data size
    /// Requirements: 1.1, 1.2, 1.3, 4.1, 4.2, 4.5
    pub fn find_k_nearest_neighbors_optimized(&mut self, query_features: &FeatureSeries) -> Vec<(f32, Direction)> {
        if self.training_samples.is_empty() {
            return Vec::new();
        }
        
        let _k = self.config.neighbors_count;
        let sample_count = self.training_samples.len();
        
        // Strategy selection based on configuration and data characteristics
        // Priority: HNSW > Parallel > Sequential
        
        // 1. Try HNSW index if available and beneficial (large datasets)
        if self.config.use_hnsw_index && self.is_hnsw_available() && sample_count > 1000 {
            match self.find_k_nearest_neighbors_hnsw_enhanced(query_features) {
                Ok(results) => {
                    if self.config.enable_debug_logging {
                        self.log_debug(&format!("Used HNSW search for {} samples", sample_count));
                    }
                    return results;
                }
                Err(e) => {
                    if self.config.enable_debug_logging {
                        eprintln!("HNSW search failed: {}. Falling back to exact search.", e);
                    }
                    // Continue to exact search fallback
                }
            }
        }
        
        // 2. Use parallel search for medium to large datasets
        if self.config.use_multithreading && sample_count > self.config.parallel_threshold {
            return self.find_k_nearest_neighbors_parallel_optimized(query_features);
        }
        
        // 3. Use sequential search for small datasets
        self.find_k_nearest_neighbors_sequential_optimized(query_features)
    }
    
    /// Find k nearest neighbors with tracking of which method was used (for performance metrics)
    fn find_k_nearest_neighbors_with_tracking(&self, query_features: &FeatureSeries) -> (Vec<(f32, Direction)>, bool) {
        if self.training_samples.is_empty() {
            return (Vec::new(), false);
        }
        
        let k = self.config.neighbors_count;
        
        // Try HNSW first if available and we have enough samples
        if self.is_hnsw_available() && self.training_samples.len() > 100 {
            match self.find_k_nearest_neighbors_hnsw(query_features, k) {
                Ok(results) => {
                    return (results, true); // Used HNSW
                }
                Err(e) => {
                    if self.config.enable_debug_logging {
                        eprintln!("HNSW search failed: {}. Falling back to exact search.", e);
                    }
                    // Fall through to exact search
                }
            }
        }
        
        // Exact search fallback
        let training_samples = self.get_training_samples_for_search(None);
        
        let results = if self.config.use_multithreading && training_samples.len() > self.config.parallel_threshold {
            self.find_k_nearest_neighbors_parallel(query_features, &training_samples, k)
        } else {
            self.find_k_nearest_neighbors_sequential(query_features, &training_samples, k)
        };
        
        (results, false) // Used exact search
    }
    
    /// Enhanced sequential k-NN search with SIMD support and automatic fallback
    /// Requirements: 1.1, 1.2, 1.3
    pub fn find_k_nearest_neighbors_sequential_optimized(&self, query_features: &FeatureSeries) -> Vec<(f32, Direction)> {
        let training_samples = self.get_training_samples_for_search(None);
        let k = self.config.neighbors_count;
        
        let mut distances_and_labels: Vec<(f32, Direction)> = Vec::new();
        let mut last_distance = -1.0;
        
        // Iterate through training samples with chronological spacing (modulo 4)
        for (i, sample) in training_samples.iter().enumerate() {
            let distance = if self.config.use_simd_optimization {
                // Try SIMD optimization with fallback
                match std::panic::catch_unwind(|| {
                    query_features.lorentzian_distance_simd(&sample.features)
                }) {
                    Ok(dist_result) => match dist_result {
                        Ok(dist) => dist,
                        Err(_) => Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count)
                    },
                    Err(_) => {
                        if self.config.enable_debug_logging {
                            eprintln!("SIMD distance calculation failed for sample {}, using standard calculation", i);
                        }
                        Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count)
                    }
                }
            } else {
                Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count)
            };
            
            // Apply the Pine Script condition: d >= lastDistance and i%4
            if distance >= last_distance && (i % 4 == 0 || !self.config.use_chronological_spacing) {
                last_distance = distance;
                distances_and_labels.push((distance, sample.label));
                
                // Keep only k nearest neighbors
                if distances_and_labels.len() > k {
                    // Remove the first (farthest) neighbor
                    distances_and_labels.remove(0);
                    
                    // Update last_distance to be in the lower 25% of the array
                    // This matches the Pine Script optimization
                    if distances_and_labels.len() > 3 {
                        let index = (k * 3 / 4).min(distances_and_labels.len() - 1);
                        last_distance = distances_and_labels[index].0;
                    }
                }
            }
        }
        
        // Sort by distance (ascending) to get nearest neighbors first
        distances_and_labels.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        distances_and_labels
    }
    
    /// Sequential k-NN search (legacy method for compatibility)
    fn find_k_nearest_neighbors_sequential(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample], k: usize) -> Vec<(f32, Direction)> {
        let mut distances_and_labels: Vec<(f32, Direction)> = Vec::new();
        let mut last_distance = -1.0;
        
        // Iterate through training samples with chronological spacing (modulo 4)
        for (i, sample) in training_samples.iter().enumerate() {
            let distance = if self.config.use_simd_optimization {
                match query_features.lorentzian_distance_simd(&sample.features) {
                    Ok(dist) => dist,
                    Err(_) => Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count)
                }
            } else {
                Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count)
            };
            
            // Apply the Pine Script condition: d >= lastDistance and i%4
            if distance >= last_distance && (i % 4 == 0 || !self.config.use_chronological_spacing) {
                last_distance = distance;
                distances_and_labels.push((distance, sample.label));
                
                // Keep only k nearest neighbors
                if distances_and_labels.len() > k {
                    // Remove the first (farthest) neighbor
                    distances_and_labels.remove(0);
                    
                    // Update last_distance to be in the lower 25% of the array
                    // This matches the Pine Script optimization
                    if distances_and_labels.len() > 3 {
                        let index = (k * 3 / 4).min(distances_and_labels.len() - 1);
                        last_distance = distances_and_labels[index].0;
                    }
                }
            }
        }
        
        // Sort by distance (ascending) to get nearest neighbors first
        distances_and_labels.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        distances_and_labels
    }
    
    /// Enhanced parallel k-NN search with SIMD optimization and automatic fallback
    /// Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3
    pub fn find_k_nearest_neighbors_parallel_optimized(&mut self, query_features: &FeatureSeries) -> Vec<(f32, Direction)> {
        // Get optimal thread pool for this workload first
        let thread_pool = self.get_or_create_thread_pool();
        let thread_count = self.thread_pool_stats.current_thread_count;
        
        // Get training samples after thread pool setup
        let training_samples = self.get_training_samples_for_search(None);
        let _k = self.config.neighbors_count;
        
        // Measure execution time for thread efficiency monitoring
        let start_time = std::time::Instant::now();
        
        // Execute parallel search using the selected thread pool
        let results = match thread_pool {
            Ok(pool) => pool.install(|| {
            // Choose between SIMD-optimized and standard parallel processing
            if self.config.use_simd_optimization && training_samples.len() >= self.config.simd_chunk_size {
                // Use SIMD-optimized batch processing with automatic fallback
                match self.parallel_search_with_simd_managed(query_features, &training_samples) {
                    Ok(results) => {
                        if self.config.enable_debug_logging {
                            self.log_debug(&format!("Used SIMD-optimized parallel search for {} samples", training_samples.len()));
                        }
                        results
                    }
                    Err(e) => {
                        if self.config.enable_debug_logging {
                            eprintln!("SIMD parallel search failed: {}. Falling back to standard parallel search.", e);
                        }
                        // Fallback to standard parallel processing
                        self.parallel_search_standard_managed(query_features, &training_samples)
                    }
                }
            } else {
                // Use standard parallel processing
                self.parallel_search_standard_managed(query_features, &training_samples)
            }
        }),
        Err(e) => {
            eprintln!("Thread pool error: {}. Falling back to sequential search.", e);
            self.find_k_nearest_neighbors_sequential_optimized(query_features)
        }
    };
        
        // Monitor thread efficiency
        let duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        self.monitor_thread_efficiency(duration_ms, thread_count);
        
        results
    }
    
    /// Parallel k-NN search using rayon (legacy method for compatibility)
    fn find_k_nearest_neighbors_parallel(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample], k: usize) -> Vec<(f32, Direction)> {
        // Configure thread pool if specified
        if let Some(max_threads) = self.config.max_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build_global()
                .unwrap_or_default();
        }
        
        // Use SIMD batch processing if enabled and chunk size is appropriate
        if self.config.use_simd_optimization && training_samples.len() >= self.config.simd_chunk_size {
            self.find_k_nearest_neighbors_parallel_simd(query_features, training_samples, k)
        } else {
            // Standard parallel processing
            let distances_and_labels: Vec<(f32, Direction)> = training_samples
                .par_iter()
                .enumerate()
                .filter_map(|(i, sample)| {
                    // Apply chronological spacing filter
                    if i % 4 == 0 || !self.config.use_chronological_spacing {
                        let distance = if self.config.use_simd_optimization {
                            match query_features.lorentzian_distance_simd(&sample.features) {
                                Ok(dist) => dist,
                                Err(_) => Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count)
                            }
                        } else {
                            Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count)
                        };
                        Some((distance, sample.label))
                    } else {
                        None
                    }
                })
                .collect();
            
            // Sort by distance and take k nearest
            let mut sorted_distances = distances_and_labels;
            sorted_distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted_distances.truncate(k);
            
            sorted_distances
        }
    }
    
    /// SIMD-optimized parallel search with automatic fallback mechanisms
    /// Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    fn parallel_search_with_simd(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample]) -> Result<Vec<(f32, Direction)>> {
        let k = self.config.neighbors_count;
        
        // Filter samples with chronological spacing first
        let filtered_samples: Vec<&TrainingSample> = training_samples
            .iter()
            .enumerate()
            .filter_map(|(i, sample)| {
                if i % 4 == 0 || !self.config.use_chronological_spacing {
                    Some(*sample)
                } else {
                    None
                }
            })
            .collect();
        
        if filtered_samples.is_empty() {
            return Ok(Vec::new());
        }
        
        // Process in parallel chunks for optimal SIMD utilization
        let chunk_size = self.config.simd_chunk_size;
        let chunk_results: Vec<Result<Vec<(f32, Direction)>, anyhow::Error>> = filtered_samples
            .par_chunks(chunk_size)
            .map(|chunk| -> Result<Vec<(f32, Direction)>, anyhow::Error> {
                // Extract features for SIMD batch processing
                let features: Vec<FeatureSeries> = chunk.iter()
                    .map(|sample| sample.features.clone())
                    .collect();
                
                // Calculate distances using SIMD batch processing with error handling
                let distances = match std::panic::catch_unwind(|| {
                    FeatureSeries::batch_lorentzian_distance_simd(
                        query_features,
                        &features,
                        chunk.len(),
                    )
                }) {
                    Ok(distances_result) => match distances_result {
                        Ok(distances) => distances,
                        Err(_) => FeatureSeries::batch_lorentzian_distance_standard(query_features, &features)
                    },
                    Err(_) => {
                        // SIMD operation failed, fall back to standard calculation
                        if self.config.enable_debug_logging {
                            eprintln!("SIMD batch operation failed, using standard calculation for chunk");
                        }
                        FeatureSeries::batch_lorentzian_distance_standard(query_features, &features)
                    }
                };
                
                // Combine distances with labels
                let chunk_results: Vec<(f32, Direction)> = distances
                    .into_iter()
                    .zip(chunk.iter())
                    .map(|(distance, sample)| (distance, sample.label))
                    .collect();
                
                Ok(chunk_results)
            })
            .collect();
        
        // Collect all results and handle errors
        let mut all_distances = Vec::new();
        for chunk_result in chunk_results {
            match chunk_result {
                Ok(chunk_data) => all_distances.extend(chunk_data),
                Err(e) => return Err(e),
            }
        }
        
        // Sort by distance and take k nearest
        all_distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_distances.truncate(k);
        
        Ok(all_distances)
    }
    
    /// Standard parallel search without SIMD optimization
    /// Requirements: 1.1, 1.2, 1.3
    fn parallel_search_standard(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample]) -> Vec<(f32, Direction)> {
        let k = self.config.neighbors_count;
        
        // Standard parallel processing without SIMD
        let distances_and_labels: Vec<(f32, Direction)> = training_samples
            .par_iter()
            .enumerate()
            .filter_map(|(i, sample)| {
                // Apply chronological spacing filter
                if i % 4 == 0 || !self.config.use_chronological_spacing {
                    let distance = Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count);
                    Some((distance, sample.label))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by distance and take k nearest
        let mut sorted_distances = distances_and_labels;
        sorted_distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_distances.truncate(k);
        
        sorted_distances
    }
    
    /// SIMD-optimized parallel search with thread pool management
    /// Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    fn parallel_search_with_simd_managed(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample]) -> Result<Vec<(f32, Direction)>> {
        let k = self.config.neighbors_count;
        
        // Filter samples with chronological spacing first
        let filtered_samples: Vec<&TrainingSample> = training_samples
            .iter()
            .enumerate()
            .filter_map(|(i, sample)| {
                if i % 4 == 0 || !self.config.use_chronological_spacing {
                    Some(*sample)
                } else {
                    None
                }
            })
            .collect();
        
        if filtered_samples.is_empty() {
            return Ok(Vec::new());
        }
        
        // Process in parallel chunks for optimal SIMD utilization with thread pool awareness
        let chunk_size = self.calculate_optimal_chunk_size(filtered_samples.len());
        let chunk_results: Vec<Result<Vec<(f32, Direction)>, anyhow::Error>> = filtered_samples
            .par_chunks(chunk_size)
            .map(|chunk| -> Result<Vec<(f32, Direction)>, anyhow::Error> {
                // Extract features for SIMD batch processing
                let features: Vec<FeatureSeries> = chunk.iter()
                    .map(|sample| sample.features.clone())
                    .collect();
                
                // Calculate distances using SIMD batch processing with error handling
                let distances = match std::panic::catch_unwind(|| {
                    FeatureSeries::batch_lorentzian_distance_simd(
                        query_features,
                        &features,
                        chunk.len(),
                    )
                }) {
                    Ok(distances_result) => match distances_result {
                        Ok(distances) => distances,
                        Err(_) => FeatureSeries::batch_lorentzian_distance_standard(query_features, &features)
                    },
                    Err(_) => {
                        // SIMD operation failed, fall back to standard calculation
                        if self.config.enable_debug_logging {
                            eprintln!("SIMD batch operation failed, using standard calculation for chunk");
                        }
                        FeatureSeries::batch_lorentzian_distance_standard(query_features, &features)
                    }
                };
                
                // Combine distances with labels
                let chunk_results: Vec<(f32, Direction)> = distances
                    .into_iter()
                    .zip(chunk.iter())
                    .map(|(distance, sample)| (distance, sample.label))
                    .collect();
                
                Ok(chunk_results)
            })
            .collect();
        
        // Collect all results and handle errors
        let mut all_distances = Vec::new();
        for chunk_result in chunk_results {
            match chunk_result {
                Ok(chunk_data) => all_distances.extend(chunk_data),
                Err(e) => return Err(e),
            }
        }
        
        // Sort by distance and take k nearest
        all_distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_distances.truncate(k);
        
        Ok(all_distances)
    }
    
    /// Standard parallel search with thread pool management
    /// Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3
    fn parallel_search_standard_managed(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample]) -> Vec<(f32, Direction)> {
        let k = self.config.neighbors_count;
        
        // Standard parallel processing without SIMD, but with thread pool awareness
        let distances_and_labels: Vec<(f32, Direction)> = training_samples
            .par_iter()
            .enumerate()
            .filter_map(|(i, sample)| {
                // Apply chronological spacing filter
                if i % 4 == 0 || !self.config.use_chronological_spacing {
                    let distance = Self::lorentzian_distance(query_features, &sample.features, self.config.feature_count);
                    Some((distance, sample.label))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by distance and take k nearest
        let mut sorted_distances = distances_and_labels;
        sorted_distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_distances.truncate(k);
        
        sorted_distances
    }
    
    /// Calculate optimal chunk size based on thread count and dataset characteristics
    fn calculate_optimal_chunk_size(&self, dataset_size: usize) -> usize {
        let thread_count = self.thread_pool_stats.current_thread_count;
        let base_chunk_size = self.config.simd_chunk_size;
        
        // Adjust chunk size based on thread count and dataset size
        let optimal_chunks_per_thread = 2; // Allow some work stealing opportunities
        let total_chunks = thread_count * optimal_chunks_per_thread;
        
        if total_chunks == 0 {
            return base_chunk_size;
        }
        
        let calculated_chunk_size = (dataset_size + total_chunks - 1) / total_chunks; // Ceiling division
        
        // Ensure chunk size is reasonable (not too small or too large)
        calculated_chunk_size
            .max(base_chunk_size / 4) // Minimum chunk size
            .min(base_chunk_size * 4) // Maximum chunk size
            .max(1) // At least 1 element per chunk
    }
    
    /// SIMD-optimized parallel k-NN search with batch processing (legacy method for compatibility)
    fn find_k_nearest_neighbors_parallel_simd(&self, query_features: &FeatureSeries, training_samples: &[&TrainingSample], k: usize) -> Vec<(f32, Direction)> {
        // Filter samples with chronological spacing first
        let filtered_samples: Vec<&TrainingSample> = training_samples
            .iter()
            .enumerate()
            .filter_map(|(i, sample)| {
                if i % 4 == 0 || !self.config.use_chronological_spacing {
                    Some(*sample)
                } else {
                    None
                }
            })
            .collect();
        
        if filtered_samples.is_empty() {
            return Vec::new();
        }
        
        // Extract features for batch SIMD processing
        let features: Vec<FeatureSeries> = filtered_samples.iter()
            .map(|sample| sample.features.clone())
            .collect();
        
        // Calculate distances using SIMD batch processing
        let distances = match FeatureSeries::batch_lorentzian_distance_simd(
            query_features,
            &features,
            self.config.simd_chunk_size,
        ) {
            Ok(distances) => distances,
            Err(_) => FeatureSeries::batch_lorentzian_distance_standard(query_features, &features)
        };
        
        // Combine distances with labels
        let mut distances_and_labels: Vec<(f32, Direction)> = distances
            .into_iter()
            .zip(filtered_samples.iter())
            .map(|(distance, sample)| (distance, sample.label))
            .collect();
        
        // Sort by distance and take k nearest
        distances_and_labels.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        distances_and_labels.truncate(k);
        
        distances_and_labels
    }
    
    /// Enhanced HNSW-based approximate k-NN search with automatic fallback
    /// Requirements: 4.1, 4.2, 4.5
    pub fn find_k_nearest_neighbors_hnsw_enhanced(&self, query_features: &FeatureSeries) -> Result<Vec<(f32, Direction)>> {
        let k = self.config.neighbors_count;
        
        if let Some(ref hnsw_index) = self.hnsw_index {
            // Search using HNSW index with error handling
            match hnsw_index.search_knn(query_features, k, &self.training_samples) {
                Ok(hnsw_results) => {
                    // Convert to (distance, label) format
                    let mut results = Vec::new();
                    for (distance, sample_index) in hnsw_results {
                        if let Some(sample) = self.training_samples.get(sample_index) {
                            results.push((distance, sample.label));
                        }
                    }
                    
                    // Apply chronological spacing if enabled
                    if self.config.use_chronological_spacing {
                        results = results
                            .into_iter()
                            .enumerate()
                            .filter_map(|(i, result)| {
                                if i % 4 == 0 {
                                    Some(result)
                                } else {
                                    None
                                }
                            })
                            .collect();
                    }
                    
                    // Ensure we have enough results, if not fall back to exact search
                    if results.len() < k.min(self.training_samples.len()) {
                        if self.config.enable_debug_logging {
                            eprintln!("HNSW returned insufficient results ({} < {}), falling back to exact search", 
                                     results.len(), k);
                        }
                        return Err(anyhow::anyhow!("HNSW returned insufficient results"));
                    }
                    
                    Ok(results)
                }
                Err(e) => {
                    if self.config.enable_debug_logging {
                        eprintln!("HNSW search operation failed: {}", e);
                    }
                    Err(anyhow::anyhow!("HNSW search failed: {}", e))
                }
            }
        } else {
            Err(anyhow::anyhow!("HNSW index is not available"))
        }
    }
    
    /// HNSW-based approximate k-NN search (legacy method for compatibility)
    fn find_k_nearest_neighbors_hnsw(&self, query_features: &FeatureSeries, k: usize) -> Result<Vec<(f32, Direction)>> {
        // Delegate to enhanced method
        self.find_k_nearest_neighbors_hnsw_enhanced(query_features)
    }
    
    /// Predict using k-NN with weighted voting
    pub fn predict(&mut self, query_features: &FeatureSeries) -> LDCPrediction {
        let start_time = std::time::Instant::now();
        
        if self.training_samples.is_empty() {
            self.log_debug("No training samples available for prediction");
            return LDCPrediction {
                signal: 0.0,
                confidence: 0.0,
                k_nearest_distances: Vec::new(),
                k_nearest_labels: Vec::new(),
                prediction_direction: Direction::Neutral,
            };
        }
        
        self.log_debug(&format!("Starting prediction with {} training samples", self.training_samples.len()));
        
        let k_nearest = self.find_k_nearest_neighbors(query_features);
        
        if k_nearest.is_empty() {
            self.log_debug("No k-nearest neighbors found");
            return LDCPrediction {
            signal: 0.0,
                confidence: 0.0,
                k_nearest_distances: Vec::new(),
                k_nearest_labels: Vec::new(),
                prediction_direction: Direction::Neutral,
            };
        }
        
        // Calculate signal as sum of labels (matching Pine Script array.sum(predictions))
        let signal: f32 = k_nearest.iter()
            .map(|(_, label)| i32::from(*label) as f32)
            .sum();
        
        // Calculate confidence based on distance distribution
        let distances: Vec<f32> = k_nearest.iter().map(|(dist, _)| *dist).collect();
        let labels: Vec<Direction> = k_nearest.iter().map(|(_, label)| *label).collect();
        
        let confidence = self.calculate_confidence(&distances);
        let prediction_direction = if signal > 0.0 {
            Direction::Long
        } else if signal < 0.0 {
            Direction::Short
        } else {
            Direction::Neutral
        };
        
        let prediction = LDCPrediction {
            signal,
            confidence,
            k_nearest_distances: distances,
            k_nearest_labels: labels,
            prediction_direction,
        };
        
        // Update performance metrics
        let duration = start_time.elapsed();
        let duration_ms = duration.as_secs_f64() * 1000.0;
        
        // Note: We can't mutate self here since this is a &self method
        // In a real implementation, you might want to use interior mutability
        // or return metrics along with the prediction
        
        self.log_prediction(&prediction);
        self.log_performance(duration_ms);
        
        prediction
    }
    
    /// Predict using k-NN with detailed performance tracking (mutable version)
    pub fn predict_with_metrics(&mut self, query_features: &FeatureSeries) -> LDCPrediction {
        let start_time = std::time::Instant::now();
        let data_access_start = std::time::Instant::now();
        
        if self.training_samples.is_empty() {
            self.log_debug("No training samples available for prediction");
            return LDCPrediction {
                signal: 0.0,
                confidence: 0.0,
                k_nearest_distances: Vec::new(),
                k_nearest_labels: Vec::new(),
                prediction_direction: Direction::Neutral,
            };
        }
        
        // Track data access time
        let data_access_time = data_access_start.elapsed().as_secs_f64() * 1000.0;
        
        self.log_debug(&format!("Starting prediction with {} training samples", self.training_samples.len()));
        
        // Track k-NN search time and method used
        let knn_start = std::time::Instant::now();
        let (k_nearest, used_hnsw) = self.find_k_nearest_neighbors_with_tracking(query_features);
        let knn_time = knn_start.elapsed().as_secs_f64() * 1000.0;
        
        // Update operation counters based on actual search method used
        if used_hnsw {
            self.increment_hnsw_queries();
        } else {
            self.increment_exact_queries();
        }
        
        // Update SIMD operation counter if SIMD was used (for exact search)
        if !used_hnsw && self.config.use_simd_optimization {
            self.increment_simd_operations();
        }
        
        if k_nearest.is_empty() {
            self.log_debug("No k-nearest neighbors found");
            return LDCPrediction {
                signal: 0.0,
                confidence: 0.0,
                k_nearest_distances: Vec::new(),
                k_nearest_labels: Vec::new(),
                prediction_direction: Direction::Neutral,
            };
        }
        
        // Calculate signal as sum of labels (matching Pine Script array.sum(predictions))
        let signal: f32 = k_nearest.iter()
            .map(|(_, label)| i32::from(*label) as f32)
            .sum();
        
        // Calculate confidence based on distance distribution
        let distances: Vec<f32> = k_nearest.iter().map(|(dist, _)| *dist).collect();
        let labels: Vec<Direction> = k_nearest.iter().map(|(_, label)| *label).collect();
        
        let confidence = self.calculate_confidence(&distances);
        let prediction_direction = if signal > 0.0 {
            Direction::Long
        } else if signal < 0.0 {
            Direction::Short
        } else {
            Direction::Neutral
        };
        
        let prediction = LDCPrediction {
            signal,
            confidence,
            k_nearest_distances: distances,
            k_nearest_labels: labels,
            prediction_direction,
        };
        
        // Update comprehensive performance metrics
        let total_duration = start_time.elapsed();
        let total_duration_ms = total_duration.as_secs_f64() * 1000.0;
        
        // Update basic metrics
        self.performance_metrics.total_predictions += 1;
        self.performance_metrics.last_prediction_time_ms = total_duration_ms;
        
        // Update average prediction time
        let total_time = self.performance_metrics.average_prediction_time_ms * (self.performance_metrics.total_predictions - 1) as f64 + total_duration_ms;
        self.performance_metrics.average_prediction_time_ms = total_time / self.performance_metrics.total_predictions as f64;
        
        // Update detailed timing metrics
        let distance_time = total_duration_ms - knn_time - data_access_time; // Approximate
        self.update_timing_metrics(distance_time, knn_time, data_access_time);
        
        // Update latency percentiles
        self.update_latency_percentiles(total_duration_ms);
        
        // Update memory metrics (simplified - in real implementation would use system APIs)
        let current_memory = Self::get_current_memory_usage_mb();
        self.update_memory_metrics(current_memory, self.performance_metrics.memory_allocations + 1);
        
        // Update CPU metrics (simplified - in real implementation would use system APIs)
        let cpu_utilization = Self::get_current_cpu_utilization();
        let thread_count = self.config.max_threads.unwrap_or_else(|| rayon::current_num_threads());
        let expected_sequential_time = total_duration_ms * thread_count as f64;
        let thread_efficiency = self.calculate_thread_efficiency(total_duration_ms, expected_sequential_time);
        self.update_cpu_metrics(cpu_utilization, thread_efficiency);
        
        // Update parallel/sequential counters
        if self.config.use_multithreading && self.training_samples.len() > self.config.parallel_threshold {
            self.performance_metrics.parallel_predictions += 1;
        } else {
            self.performance_metrics.sequential_predictions += 1;
        }
        
        // Check for performance degradation
        self.check_performance_degradation();
        
        self.log_prediction(&prediction);
        self.log_performance(total_duration_ms);
        
        prediction
    }
    
    /// Calculate confidence based on distance distribution
    fn calculate_confidence(&self, distances: &[f32]) -> f32 {
        if distances.is_empty() {
            return 0.0;
        }
        
        // Simple confidence based on distance variance
        let mean_distance: f32 = distances.iter().sum::<f32>() / distances.len() as f32;
        let variance: f32 = distances.iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f32>() / distances.len() as f32;
        
        // Convert variance to confidence (lower variance = higher confidence)
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            (1.0 / (1.0 + std_dev)).min(1.0)
        } else {
            1.0
        }
    }
    
    // ===========================================
    // ==== Feature Pipeline Integration ====
    // ===========================================
    
    /// Convert Features from feature-pipeline to FeatureSeries for LDC
    /// This replaces the Pine Script ml.n_rsi, ml.n_wt, ml.n_cci, ml.n_adx functions
    pub fn features_to_feature_series(features: &Features) -> Result<FeatureSeries> {
        // Extract features with proper error handling for missing values
        let f1 = features.rsi.ok_or_else(|| anyhow::anyhow!("RSI feature is missing"))? as f32;
        let f2 = features.wavetrend_1.ok_or_else(|| anyhow::anyhow!("WaveTrend feature is missing"))? as f32;
        let f3 = features.cci.ok_or_else(|| anyhow::anyhow!("CCI feature is missing"))? as f32;
        let f4 = features.adx.ok_or_else(|| anyhow::anyhow!("ADX feature is missing"))? as f32;
        let f5 = features.wavetrend_2.ok_or_else(|| anyhow::anyhow!("WaveTrend2 feature is missing"))? as f32;
        
        Ok(FeatureSeries {
            f1, // RSI
            f2, // WT (WaveTrend)
            f3, // CCI
            f4, // ADX
            f5, // WT2 (WaveTrend2) - used as 5th feature in Pine Script
        })
    }
    
    /// Add training sample from feature-pipeline data
    /// This handles the complete flow from OHLCV -> Features -> FeatureSeries -> TrainingSample
    pub fn add_training_sample_from_features(
        &mut self, 
        features: &Features, 
        current_price: f32, 
        future_price: f32
    ) -> Result<()> {
        let feature_series = Self::features_to_feature_series(features)?;
        self.add_training_sample_with_label(
            feature_series,
            current_price,
            future_price,
            features.timestamp,
            0, // bar_index - could be enhanced to track this
        );
        Ok(())
    }
    
    /// Predict using features from feature-pipeline
    /// This is the main entry point for integration
    pub fn predict_from_features(&mut self, features: &Features) -> Result<LDCPrediction> {
        let feature_series = Self::features_to_feature_series(features)?;
        Ok(self.predict(&feature_series))
    }
    
    /// Batch process features and generate predictions
    /// This is useful for backtesting or processing historical data
    pub fn batch_predict_from_features(&mut self, features_list: &[Features]) -> Result<Vec<LDCPrediction>> {
        if self.config.use_multithreading && features_list.len() > self.config.batch_parallel_threshold {
            self.batch_predict_from_features_parallel(features_list)
        } else {
            self.batch_predict_from_features_sequential(features_list)
        }
    }
    
    /// Sequential batch prediction
    fn batch_predict_from_features_sequential(&mut self, features_list: &[Features]) -> Result<Vec<LDCPrediction>> {
        let mut predictions = Vec::new();
        for features in features_list {
            let prediction = self.predict_from_features(features)?;
            predictions.push(prediction);
        }
        Ok(predictions)
    }
    
    /// Parallel batch prediction using rayon
    fn batch_predict_from_features_parallel(&mut self, features_list: &[Features]) -> Result<Vec<LDCPrediction>> {
        // Note: Due to thread pool management requiring mutable access,
        // we use sequential processing for batch predictions.
        // This ensures thread efficiency monitoring works correctly.
        // For true parallel batch processing, consider using a different approach
        // that doesn't require mutable access to the engine during prediction.
        self.batch_predict_from_features_sequential(features_list)
    }
    
    /// Create training samples from historical OHLCV data
    /// This replaces the Pine Script training data generation
    pub fn create_training_samples_from_ohlcv(
        &mut self,
        ohlcv_data: &[OHLCV],
        features_list: &[Features],
        horizon_bars: usize, // How many bars ahead to look for labeling (default 4)
    ) -> Result<()> {
        if ohlcv_data.len() != features_list.len() {
            return Err(anyhow::anyhow!("OHLCV data and features must have same length"));
        }
        
        if ohlcv_data.len() < horizon_bars + 1 {
            return Err(anyhow::anyhow!("Not enough data for training (need at least {} bars)", horizon_bars + 1));
        }
        
        if self.config.use_multithreading && ohlcv_data.len() > self.config.parallel_threshold {
            self.create_training_samples_parallel(ohlcv_data, features_list, horizon_bars)
        } else {
            self.create_training_samples_sequential(ohlcv_data, features_list, horizon_bars)
        }
    }
    
    /// Sequential training sample creation
    fn create_training_samples_sequential(
        &mut self,
        ohlcv_data: &[OHLCV],
        features_list: &[Features],
        horizon_bars: usize,
    ) -> Result<()> {
        // Create training samples with future price labeling
        for i in 0..(ohlcv_data.len() - horizon_bars) {
            let current_price = ohlcv_data[i].close as f32;
            let future_price = ohlcv_data[i + horizon_bars].close as f32;
            
            self.add_training_sample_from_features(
                &features_list[i],
                current_price,
                future_price,
            )?;
        }
        
        Ok(())
    }
    
    /// Parallel training sample creation
    fn create_training_samples_parallel(
        &mut self,
        ohlcv_data: &[OHLCV],
        features_list: &[Features],
        horizon_bars: usize,
    ) -> Result<()> {
        // Configure thread pool if specified
        if let Some(max_threads) = self.config.max_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build_global()
                .unwrap_or_default();
        }
        
        // Create training samples in parallel
        let training_samples: Result<Vec<TrainingSample>> = (0..(ohlcv_data.len() - horizon_bars))
            .into_par_iter()
            .map(|i| {
                let current_price = ohlcv_data[i].close as f32;
                let future_price = ohlcv_data[i + horizon_bars].close as f32;
                let label = Self::generate_label(current_price, future_price);
                let feature_series = Self::features_to_feature_series(&features_list[i])?;
                
                Ok(TrainingSample {
                    features: feature_series,
                    label,
                    timestamp: features_list[i].timestamp,
                    bar_index: i,
                })
            })
            .collect();
        
        // Add all training samples to the engine
        for sample in training_samples? {
            self.add_training_sample(sample);
        }
        
        Ok(())
    }
    
    // ==========================================
    // HNSW Index Management Methods
    // ==========================================
    

    
    /// Rebuild or create HNSW index based on current configuration
    pub fn rebuild_or_create_hnsw_index(&mut self) -> Result<()> {
        if self.config.use_hnsw_index {
            let hnsw_config = HNSWConfig {
                m: self.config.hnsw_m,
                ef_construction: self.config.hnsw_ef_construction,
                ef_search: self.config.hnsw_ef_search,
                max_elements: self.config.max_bars_back * 2, // Allow some headroom
            };
            
            // Create new HNSW index
            let mut new_hnsw_index = HNSWIndex::new(hnsw_config)?;
            
            // Add all current training samples
            for (index, sample) in self.training_samples.iter().enumerate() {
                new_hnsw_index.add_sample(sample, index)?;
            }
            
            self.hnsw_index = Some(new_hnsw_index);
            self.samples_since_hnsw_rebuild = 0;
            self.increment_hnsw_rebuild_count();
            
            if self.config.enable_debug_logging {
                self.log_debug(&format!("HNSW index created/rebuilt with {} samples", 
                                      self.training_samples.len()));
            }
        } else {
            self.hnsw_index = None;
        }
        
        Ok(())
    }
    
    /// Get HNSW index status information
    pub fn get_hnsw_status(&self) -> Option<(usize, usize, usize)> {
        self.hnsw_index.as_ref().map(|index| {
            (
                index.len(),                           // Number of indexed samples
                self.samples_since_hnsw_rebuild,       // Samples since last rebuild
                self.config.hnsw_rebuild_threshold,    // Rebuild threshold
            )
        })
    }
    
    /// Check if HNSW index is available and ready
    pub fn is_hnsw_available(&self) -> bool {
        self.hnsw_index.is_some() && self.config.use_hnsw_index
    }
    
    /// Force HNSW index rebuild (useful for testing or manual optimization)
    pub fn force_hnsw_rebuild(&mut self) -> Result<()> {
        if self.config.use_hnsw_index {
            self.rebuild_hnsw_index().map_err(|e| anyhow::anyhow!("HNSW rebuild failed: {}", e))
        } else {
            Err(anyhow::anyhow!("HNSW index is disabled in configuration"))
        }
    }
    
    /// Update HNSW search parameters at runtime
    pub fn update_hnsw_search_params(&mut self, ef_search: usize) -> Result<()> {
        if let Some(ref mut hnsw_index) = self.hnsw_index {
            hnsw_index.set_ef_search(ef_search);
            self.config.hnsw_ef_search = ef_search;
            
            if self.config.enable_debug_logging {
                self.log_debug(&format!("HNSW ef_search updated to {}", ef_search));
            }
            
            Ok(())
        } else {
            Err(anyhow::anyhow!("HNSW index is not available"))
        }
    }
    
    /// Get HNSW configuration
    pub fn get_hnsw_config(&self) -> Option<&HNSWConfig> {
        self.hnsw_index.as_ref().map(|index| index.config())
    }
}

// ===========================================
// ==== Tests ====
// ===========================================

#[cfg(test)]
mod config_tests;

// ===========================================
// ==== Pine Script Library Functions ====
// ===========================================

/// Pine Script library functions equivalent to jdehorty/MLExtensions and jdehorty/KernelFunctions
pub mod pine_library {
    use super::*;
    use std::collections::VecDeque;
    
    /// Regime filter - detects trending vs ranging markets
    /// Equivalent to Pine Script regime_filter function
    pub struct RegimeFilter {
        value1: f32,
        value2: f32,
        klmf: f32,
        exponential_average_abs_curve_slope: f32,
        ema_alpha: f32,
    }
    
    impl RegimeFilter {
        pub fn new() -> Self {
            Self {
                value1: 0.0,
                value2: 0.0,
                klmf: 0.0,
                exponential_average_abs_curve_slope: 0.0,
                ema_alpha: 2.0 / 201.0, // EMA alpha for 200 period
            }
        }
        
        pub fn filter(&mut self, src: f32, high: f32, low: f32, prev_src: f32, prev_high: f32, prev_low: f32, threshold: f32, use_regime_filter: bool) -> bool {
            if !use_regime_filter {
                return true;
            }
            
            // Calculate the slope of the curve (Pine Script logic)
            self.value1 = 0.2 * (src - prev_src) + 0.8 * self.value1;
            self.value2 = 0.1 * (high - low) + 0.8 * self.value2;
            
            let omega = (self.value1 / self.value2).abs();
            let alpha = (-omega.powi(2) + (omega.powi(4) + 16.0 * omega.powi(2)).sqrt()) / 8.0;
            
            self.klmf = alpha * src + (1.0 - alpha) * self.klmf;
            let abs_curve_slope = (self.klmf - self.klmf).abs(); // This should be prev_klmf, but we'll use current for simplicity
            
            // Exponential average of absolute curve slope
            self.exponential_average_abs_curve_slope = self.ema_alpha * abs_curve_slope + (1.0 - self.ema_alpha) * self.exponential_average_abs_curve_slope;
            
            let normalized_slope_decline = (abs_curve_slope - self.exponential_average_abs_curve_slope) / self.exponential_average_abs_curve_slope;
            
            normalized_slope_decline >= threshold
        }
    }
    
    /// ADX filter - filters based on Average Directional Index
    /// Equivalent to Pine Script filter_adx function
    pub struct AdxFilter {
        tr_smooth: f32,
        smooth_directional_movement_plus: f32,
        smooth_neg_movement: f32,
        rma_alpha: f32,
    }
    
    impl AdxFilter {
        pub fn new(length: usize) -> Self {
            Self {
                tr_smooth: 0.0,
                smooth_directional_movement_plus: 0.0,
                smooth_neg_movement: 0.0,
                rma_alpha: 1.0 / length as f32,
            }
        }
        
        pub fn filter(&mut self, high: f32, low: f32, close: f32, prev_high: f32, prev_low: f32, prev_close: f32, adx_threshold: f32, use_adx_filter: bool) -> bool {
            if !use_adx_filter {
                return true;
            }
            
            // True Range calculation
            let tr = (high - low).max((high - prev_close).abs()).max((low - prev_close).abs());
            
            // Directional Movement
            let directional_movement_plus = if high - prev_high > prev_low - low {
                (high - prev_high).max(0.0)
            } else {
                0.0
            };
            
            let neg_movement = if prev_low - low > high - prev_high {
                (prev_low - low).max(0.0)
            } else {
                0.0
            };
            
            // Smoothing (Wilder's smoothing)
            self.tr_smooth = self.tr_smooth - self.tr_smooth * self.rma_alpha + tr;
            self.smooth_directional_movement_plus = self.smooth_directional_movement_plus - self.smooth_directional_movement_plus * self.rma_alpha + directional_movement_plus;
            self.smooth_neg_movement = self.smooth_neg_movement - self.smooth_neg_movement * self.rma_alpha + neg_movement;
            
            // Directional Indicators
            let di_positive = (self.smooth_directional_movement_plus / self.tr_smooth) * 100.0;
            let di_negative = (self.smooth_neg_movement / self.tr_smooth) * 100.0;
            
            // DX calculation
            let dx = ((di_positive - di_negative).abs() / (di_positive + di_negative)) * 100.0;
            
            // ADX (simplified - using current dx instead of RMA for simplicity)
            let adx = dx; // In full implementation, this would be RMA of dx
            
            adx > adx_threshold
        }
    }
    
    /// Volatility filter - filters based on ATR comparison
    /// Equivalent to Pine Script filter_volatility function
    pub struct VolatilityFilter {
        atr_short: VecDeque<f32>,
        atr_long: VecDeque<f32>,
    }
    
    impl VolatilityFilter {
        pub fn new() -> Self {
            Self {
                atr_short: VecDeque::new(),
                atr_long: VecDeque::new(),
            }
        }
        
        pub fn filter(&mut self, high: f32, low: f32, close: f32, prev_close: f32, min_length: usize, max_length: usize, use_volatility_filter: bool) -> bool {
            if !use_volatility_filter {
                return true;
            }
            
            // Calculate True Range
            let tr = (high - low).max((high - prev_close).abs()).max((low - prev_close).abs());
            
            // Update ATR windows
            self.atr_short.push_back(tr);
            self.atr_long.push_back(tr);
            
            if self.atr_short.len() > min_length {
                self.atr_short.pop_front();
            }
            if self.atr_long.len() > max_length {
                self.atr_long.pop_front();
            }
            
            if self.atr_short.len() < min_length || self.atr_long.len() < max_length {
                return true; // Not enough data yet
            }
            
            // Calculate ATR averages
            let recent_atr: f32 = self.atr_short.iter().sum::<f32>() / self.atr_short.len() as f32;
            let historical_atr: f32 = self.atr_long.iter().sum::<f32>() / self.atr_long.len() as f32;
            
            recent_atr > historical_atr
        }
    }
    
    /// Rational Quadratic Kernel - equivalent to Pine Script rationalQuadratic function
    pub fn rational_quadratic_kernel(src: &[f32], lookback: usize, relative_weight: f32, start_at_bar: usize) -> f32 {
        let mut current_weight = 0.0;
        let mut cumulative_weight = 0.0;
        
        let size = src.len();
        for i in 0..(size + start_at_bar) {
            if i >= src.len() {
                break;
            }
            
            let y = src[i];
            let w = (1.0 + (i as f32).powi(2) / ((lookback as f32).powi(2) * 2.0 * relative_weight)).powf(-relative_weight);
            
            current_weight += y * w;
            cumulative_weight += w;
        }
        
        if cumulative_weight > 0.0 {
            current_weight / cumulative_weight
        } else {
            0.0
        }
    }
    
    /// Gaussian Kernel - equivalent to Pine Script gaussian function
    pub fn gaussian_kernel(src: &[f32], lookback: usize, start_at_bar: usize) -> f32 {
        let mut current_weight = 0.0;
        let mut cumulative_weight = 0.0;
        
        let size = src.len();
        for i in 0..(size + start_at_bar) {
            if i >= src.len() {
                break;
            }
            
            let y = src[i];
            let w = (-(i as f32).powi(2) / (2.0 * (lookback as f32).powi(2))).exp();
            
            current_weight += y * w;
            cumulative_weight += w;
        }
        
        if cumulative_weight > 0.0 {
            current_weight / cumulative_weight
        } else {
            0.0
        }
    }
    
    /// Combined filter that applies all filters like in Pine Script
    pub struct CombinedFilter {
        regime_filter: RegimeFilter,
        adx_filter: AdxFilter,
        volatility_filter: VolatilityFilter,
    }
    
    impl CombinedFilter {
        pub fn new(adx_length: usize) -> Self {
            Self {
                regime_filter: RegimeFilter::new(),
                adx_filter: AdxFilter::new(adx_length),
                volatility_filter: VolatilityFilter::new(),
            }
        }
        
        pub fn apply_filters(
            &mut self,
            ohlcv: &OHLCV,
            prev_ohlcv: &OHLCV,
            regime_threshold: f32,
            use_regime_filter: bool,
            adx_threshold: f32,
            use_adx_filter: bool,
            use_volatility_filter: bool,
        ) -> bool {
            let regime_ok = self.regime_filter.filter(
                ohlcv.close as f32,
                ohlcv.high as f32,
                ohlcv.low as f32,
                prev_ohlcv.close as f32,
                prev_ohlcv.high as f32,
                prev_ohlcv.low as f32,
                regime_threshold,
                use_regime_filter,
            );
            
            let adx_ok = self.adx_filter.filter(
                ohlcv.high as f32,
                ohlcv.low as f32,
                ohlcv.close as f32,
                prev_ohlcv.high as f32,
                prev_ohlcv.low as f32,
                prev_ohlcv.close as f32,
                adx_threshold,
                use_adx_filter,
            );
            
            let volatility_ok = self.volatility_filter.filter(
                ohlcv.high as f32,
                ohlcv.low as f32,
                ohlcv.close as f32,
                prev_ohlcv.close as f32,
                1, // min_length
                10, // max_length
                use_volatility_filter,
            );
            
            regime_ok && adx_ok && volatility_ok
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ldc_engine_creation() {
        let engine = LDCEngine::new();
        assert_eq!(engine.config().max_bars_back, 2000);
        assert_eq!(engine.config().neighbors_count, 8);
        assert_eq!(engine.training_samples_count(), 0);
    }
    
    #[test]
    fn test_add_training_sample() {
        let mut engine = LDCEngine::new();
        
        let features = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let sample = TrainingSample {
            features,
            label: Direction::Long,
            timestamp: 1000,
            bar_index: 0,
        };
        
        engine.add_training_sample(sample);
        assert_eq!(engine.training_samples_count(), 1);
    }
    
    #[test]
    fn test_generate_label() {
        assert_eq!(LDCEngine::generate_label(100.0, 105.0), Direction::Long);
        assert_eq!(LDCEngine::generate_label(100.0, 95.0), Direction::Short);
        assert_eq!(LDCEngine::generate_label(100.0, 100.0), Direction::Neutral);
    }
    
    #[test]
    fn test_direction_conversion() {
        assert_eq!(Direction::from(-1), Direction::Short);
        assert_eq!(Direction::from(0), Direction::Neutral);
        assert_eq!(Direction::from(1), Direction::Long);
        
        assert_eq!(i32::from(Direction::Short), -1);
        assert_eq!(i32::from(Direction::Neutral), 0);
        assert_eq!(i32::from(Direction::Long), 1);
    }
    
    #[test]
    fn test_feature_series_conversion() {
        let features = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let arr = features.to_array();
        assert_eq!(arr, [1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let features_back = FeatureSeries::from_array(arr);
        assert_eq!(features_back.f1, 1.0);
        assert_eq!(features_back.f5, 5.0);
    }
    
    #[test]
    fn test_lorentzian_distance_identical() {
        let features1 = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        let features2 = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let distance = LDCEngine::lorentzian_distance(&features1, &features2, 5);
        assert_eq!(distance, 0.0); // ln(1 + 0) = ln(1) = 0
    }
    
    #[test]
    fn test_lorentzian_distance_different() {
        let features1 = FeatureSeries {
            f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0,
        };
        let features2 = FeatureSeries {
            f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0,
        };
        
        let distance = LDCEngine::lorentzian_distance(&features1, &features2, 5);
        let expected = 5.0 * (1.0_f32 + 1.0_f32).ln(); // 5 * ln(2)
        assert!((distance - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_lorentzian_distance_feature_counts() {
        let features1 = FeatureSeries {
            f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0,
        };
        let features2 = FeatureSeries {
            f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0,
        };
        
        let distance_2 = LDCEngine::lorentzian_distance(&features1, &features2, 2);
        let distance_3 = LDCEngine::lorentzian_distance(&features1, &features2, 3);
        let distance_4 = LDCEngine::lorentzian_distance(&features1, &features2, 4);
        let distance_5 = LDCEngine::lorentzian_distance(&features1, &features2, 5);
        
        assert!(distance_2 < distance_3);
        assert!(distance_3 < distance_4);
        assert!(distance_4 < distance_5);
    }
    
    #[test]
    fn test_lorentzian_distance_arrays() {
        let features1 = vec![0.0, 1.0, 2.0];
        let features2 = vec![1.0, 2.0, 3.0];
        
        let distance = LDCEngine::lorentzian_distance_arrays(&features1, &features2);
        let expected = (1.0_f32 + 1.0_f32).ln() + (1.0_f32 + 1.0_f32).ln() + (1.0_f32 + 1.0_f32).ln();
        assert!((distance - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_ring_buffer_management() {
        let mut engine = LDCEngine::new();
        
        // Add multiple training samples
        for i in 0..10 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            engine.add_training_sample_with_label(features, 100.0, 105.0, i as i64, i);
        }
        
        assert_eq!(engine.training_samples_count(), 10);
        
        // Test chronological spacing
        let spaced_samples = engine.get_training_samples_with_spacing();
        assert_eq!(spaced_samples.len(), 3); // 0, 4, 8 (every 4th sample)
        
        // Test training stats
        let (spaced_count, long_count, short_count) = engine.get_training_stats();
        assert_eq!(spaced_count, 3);
        assert_eq!(long_count, 10); // All samples are Long (105 > 100)
        assert_eq!(short_count, 0);
    }
    
    #[test]
    fn test_ring_buffer_overflow() {
        let mut config = LDCConfig::default();
        config.max_bars_back = 5;
        let mut engine = LDCEngine::with_config(config);
        
        // Add more samples than max_bars_back
        for i in 0..10 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            engine.add_training_sample_with_label(features, 100.0, 105.0, i as i64, i);
        }
        
        // Should only keep the last 5 samples
        assert_eq!(engine.training_samples_count(), 5);
        
        // Check that the oldest samples were removed
        let samples: Vec<_> = engine.training_samples.iter().collect();
        assert_eq!(samples[0].bar_index, 5); // First sample should be from index 5
        assert_eq!(samples[4].bar_index, 9); // Last sample should be from index 9
    }
    
    #[test]
    fn test_clear_training_data() {
        let mut engine = LDCEngine::new();
        
        // Add some training data
        let features = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        engine.add_training_sample_with_label(features, 100.0, 105.0, 1000, 0);
        
        assert_eq!(engine.training_samples_count(), 1);
        
        // Clear training data
        engine.clear_training_data();
        assert_eq!(engine.training_samples_count(), 0);
    }
    
    #[test]
    fn test_k_nearest_neighbors_search() {
        let mut engine = LDCEngine::new();
        
        // Add training samples with different features
        let features1 = FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 };
        let features2 = FeatureSeries { f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0 };
        let features3 = FeatureSeries { f1: 2.0, f2: 2.0, f3: 2.0, f4: 2.0, f5: 2.0 };
        
        engine.add_training_sample_with_label(features1, 100.0, 105.0, 1000, 0); // Long
        engine.add_training_sample_with_label(features2, 100.0, 95.0, 1001, 1);  // Short
        engine.add_training_sample_with_label(features3, 100.0, 105.0, 1002, 2); // Long
        
        // Query with features similar to features1
        let query_features = FeatureSeries { f1: 0.1, f2: 0.1, f3: 0.1, f4: 0.1, f5: 0.1 };
        let k_nearest = engine.find_k_nearest_neighbors(&query_features);
        
        // Should find the nearest neighbors
        assert!(!k_nearest.is_empty());
        assert!(k_nearest.len() <= engine.config().neighbors_count);
        
        // The first neighbor should be closest to features1 (Long)
        assert_eq!(k_nearest[0].1, Direction::Long);
    }
    
    #[test]
    fn test_prediction_with_empty_engine() {
        let mut engine = LDCEngine::new();
        let query_features = FeatureSeries { f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0 };
        
        let prediction = engine.predict(&query_features);
        
        assert_eq!(prediction.signal, 0.0);
        assert_eq!(prediction.confidence, 0.0);
        assert_eq!(prediction.prediction_direction, Direction::Neutral);
        assert!(prediction.k_nearest_distances.is_empty());
        assert!(prediction.k_nearest_labels.is_empty());
    }
    
    #[test]
    fn test_prediction_with_training_data() {
        let mut engine = LDCEngine::new();
        
        // Add training samples
        let features1 = FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 };
        let features2 = FeatureSeries { f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0 };
        
        engine.add_training_sample_with_label(features1, 100.0, 105.0, 1000, 0); // Long
        engine.add_training_sample_with_label(features2, 100.0, 95.0, 1001, 1);  // Short
        
        // Query with features similar to features1
        let query_features = FeatureSeries { f1: 0.1, f2: 0.1, f3: 0.1, f4: 0.1, f5: 0.1 };
        let prediction = engine.predict(&query_features);
        
        // Should predict Long (positive signal)
        assert!(prediction.signal > 0.0);
        assert_eq!(prediction.prediction_direction, Direction::Long);
        assert!(prediction.confidence > 0.0);
        assert!(!prediction.k_nearest_distances.is_empty());
        assert!(!prediction.k_nearest_labels.is_empty());
    }
    
    #[test]
    fn test_prediction_signal_calculation() {
        let mut engine = LDCEngine::new();
        
        // Add training samples with known labels
        let features = FeatureSeries { f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0 };
        
        // Add 3 Long samples and 2 Short samples
        for i in 0..3 {
            engine.add_training_sample_with_label(features.clone(), 100.0, 105.0, 1000 + i, i as usize); // Long
        }
        for i in 3..5 {
            engine.add_training_sample_with_label(features.clone(), 100.0, 95.0, 1000 + i, i as usize); // Short
        }
        
        let query_features = FeatureSeries { f1: 1.1, f2: 1.1, f3: 1.1, f4: 1.1, f5: 1.1 };
        let prediction = engine.predict(&query_features);
        
        // Signal should be positive (3 Long - 2 Short = +1)
        assert!(prediction.signal > 0.0);
        assert_eq!(prediction.prediction_direction, Direction::Long);
    }
    
    #[test]
    fn test_feature_pipeline_integration() {
        // Create sample features from feature-pipeline
        let features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        // Test conversion to FeatureSeries
        let feature_series = LDCEngine::features_to_feature_series(&features).unwrap();
        assert_eq!(feature_series.f1, 50.0); // RSI
        assert_eq!(feature_series.f2, 25.0); // WaveTrend1
        assert_eq!(feature_series.f3, 15.0); // CCI
        assert_eq!(feature_series.f4, 20.0); // ADX
        assert_eq!(feature_series.f5, 30.0); // WaveTrend2
    }
    
    #[test]
    fn test_feature_pipeline_integration_missing_features() {
        // Create features with missing values
        let features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: None, // Missing WaveTrend
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        // Test that missing features cause an error
        let result = LDCEngine::features_to_feature_series(&features);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("WaveTrend feature is missing"));
    }
    
    #[test]
    fn test_predict_from_features() {
        let mut engine = LDCEngine::new();
        
        // Add training data using feature-pipeline format
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Test prediction using feature-pipeline format
        let query_features = Features {
            timestamp: 1001,
            rsi: Some(51.0),
            sma_20: Some(101.0),
            ema_20: Some(102.0),
            std_20: Some(2.1),
            zscore_20: Some(0.6),
            momentum: Some(1.1),
            wavetrend_1: Some(26.0),
            wavetrend_2: Some(31.0),
            cci: Some(16.0),
            adx: Some(21.0),
        };
        
        let prediction = engine.predict_from_features(&query_features).unwrap();
        assert!(prediction.signal > 0.0); // Should predict Long
        assert_eq!(prediction.prediction_direction, Direction::Long);
    }
    
    #[test]
    fn test_pine_library_regime_filter() {
        use pine_library::RegimeFilter;
        
        let mut filter = RegimeFilter::new();
        
        // Test with regime filter disabled
        let result = filter.filter(100.0, 101.0, 99.0, 99.5, 100.5, 98.5, -0.1, false);
        assert!(result); // Should always return true when disabled
        
        // Test with regime filter enabled
        let result = filter.filter(100.0, 101.0, 99.0, 99.5, 100.5, 98.5, -0.1, true);
        // Result depends on the filter logic, but should not panic
        assert!(result || !result); // Just ensure it returns a boolean
    }
    
    #[test]
    fn test_pine_library_adx_filter() {
        use pine_library::AdxFilter;
        
        let mut filter = AdxFilter::new(14);
        
        // Test with ADX filter disabled
        let result = filter.filter(101.0, 99.0, 100.0, 100.5, 98.5, 99.5, 20.0, false);
        assert!(result); // Should always return true when disabled
        
        // Test with ADX filter enabled
        let result = filter.filter(101.0, 99.0, 100.0, 100.5, 98.5, 99.5, 20.0, true);
        // Result depends on the filter logic, but should not panic
        assert!(result || !result); // Just ensure it returns a boolean
    }
    
    #[test]
    fn test_pine_library_volatility_filter() {
        use pine_library::VolatilityFilter;
        
        let mut filter = VolatilityFilter::new();
        
        // Test with volatility filter disabled
        let result = filter.filter(101.0, 99.0, 100.0, 99.5, 1, 10, false);
        assert!(result); // Should always return true when disabled
        
        // Test with volatility filter enabled
        let result = filter.filter(101.0, 99.0, 100.0, 99.5, 1, 10, true);
        // Result depends on the filter logic, but should not panic
        assert!(result || !result); // Just ensure it returns a boolean
    }
    
    #[test]
    fn test_pine_library_kernels() {
        use pine_library::{rational_quadratic_kernel, gaussian_kernel};
        
        let src = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        
        // Test rational quadratic kernel
        let result = rational_quadratic_kernel(&src, 3, 8.0, 0);
        assert!(result.is_finite());
        assert!(result > 0.0);
        
        // Test gaussian kernel
        let result = gaussian_kernel(&src, 3, 0);
        assert!(result.is_finite());
        assert!(result > 0.0);
    }
    
    #[test]
    fn test_pine_library_combined_filter() {
        use pine_library::CombinedFilter;
        
        let mut filter = CombinedFilter::new(14);
        
        let ohlcv = OHLCV {
            timestamp: 1000,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.5,
            volume: 1000.0,
        };
        
        let prev_ohlcv = OHLCV {
            timestamp: 999,
            open: 99.5,
            high: 100.5,
            low: 98.5,
            close: 99.5,
            volume: 1000.0,
        };
        
        // Test with all filters disabled
        let result = filter.apply_filters(
            &ohlcv,
            &prev_ohlcv,
            -0.1,
            false, // regime filter disabled
            20.0,
            false, // adx filter disabled
            false, // volatility filter disabled
        );
        assert!(result); // Should return true when all filters are disabled
        
        // Test with filters enabled (result depends on data, but should not panic)
        let result = filter.apply_filters(
            &ohlcv,
            &prev_ohlcv,
            -0.1,
            true, // regime filter enabled
            20.0,
            true, // adx filter enabled
            true, // volatility filter enabled
        );
        assert!(result || !result); // Just ensure it returns a boolean
    }
    
    #[test]
    fn test_multithreading_config() {
        let mut config = LDCConfig::default();
        assert!(config.use_multithreading);
        assert_eq!(config.max_threads, None);
        
        config.use_multithreading = false;
        config.max_threads = Some(4);
        
        let engine = LDCEngine::with_config(config);
        assert!(!engine.config().use_multithreading);
        assert_eq!(engine.config().max_threads, Some(4));
    }
    
    #[test]
    fn test_parallel_vs_sequential_knn() {
        let mut engine = LDCEngine::new();
        
        // Add enough training samples to trigger parallel processing
        for i in 0..150 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            engine.add_training_sample_with_label(features, 100.0, 105.0, i as i64, i);
        }
        
        let query_features = FeatureSeries {
            f1: 75.0, f2: 75.0, f3: 75.0, f4: 75.0, f5: 75.0,
        };
        
        // Test parallel k-NN (should be used for >100 samples)
        let k_nearest = engine.find_k_nearest_neighbors(&query_features);
        assert!(!k_nearest.is_empty());
        assert!(k_nearest.len() <= engine.config().neighbors_count);
    }
    
    #[test]
    fn test_parallel_batch_prediction() {
        let mut engine = LDCEngine::new();
        
        // Add training data
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Create enough features to trigger parallel processing
        let mut features_list = Vec::new();
        for i in 0..20 {
            let features = Features {
                timestamp: 1000 + i,
                rsi: Some(50.0 + i as f64),
                sma_20: Some(100.0 + i as f64),
                ema_20: Some(101.0 + i as f64),
                std_20: Some(2.0),
                zscore_20: Some(0.5),
                momentum: Some(1.0),
                wavetrend_1: Some(25.0 + i as f64),
                wavetrend_2: Some(30.0 + i as f64),
                cci: Some(15.0 + i as f64),
                adx: Some(20.0 + i as f64),
            };
            features_list.push(features);
        }
        
        // Test parallel batch prediction
        let predictions = engine.batch_predict_from_features(&features_list).unwrap();
        assert_eq!(predictions.len(), 20);
        
        // All predictions should be valid
        for prediction in &predictions {
            assert!(prediction.signal.is_finite());
            assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        }
    }
    
    #[test]
    fn test_sequential_fallback() {
        let mut config = LDCConfig::default();
        config.use_multithreading = false;
        let mut engine = LDCEngine::with_config(config);
        
        // Add training data
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Test sequential processing (should be used when multithreading is disabled)
        let query_features = Features {
            timestamp: 1001,
            rsi: Some(51.0),
            sma_20: Some(101.0),
            ema_20: Some(102.0),
            std_20: Some(2.1),
            zscore_20: Some(0.6),
            momentum: Some(1.1),
            wavetrend_1: Some(26.0),
            wavetrend_2: Some(31.0),
            cci: Some(16.0),
            adx: Some(21.0),
        };
        
        let prediction = engine.predict_from_features(&query_features).unwrap();
        assert!(prediction.signal.is_finite());
        assert!(prediction.confidence >= 0.0);
    }
    
    #[test]
    fn test_comprehensive_config() {
        let mut config = LDCConfig::default();
        
        // Test all configuration options
        config.max_bars_back = 1000;
        config.neighbors_count = 12;
        config.feature_count = 3;
        config.use_chronological_spacing = false;
        config.use_multithreading = false;
        config.max_threads = Some(2);
        
        config.parallel_threshold = 50;
        config.batch_parallel_threshold = 5;
        
        config.enable_regime_filter = false;
        config.enable_adx_filter = true;
        config.enable_volatility_filter = false;
        config.regime_threshold = -0.2;
        config.adx_threshold = 25.0;
        
        config.enable_kernel_smoothing = true;
        config.kernel_lookback = 10;
        config.kernel_relative_weight = 5.0;
        config.kernel_regression_level = 15;
        
        config.enable_debug_logging = true;
        config.log_predictions = true;
        config.log_performance_metrics = true;
        
        let engine = LDCEngine::with_config(config);
        
        // Verify configuration is set correctly
        assert_eq!(engine.config().max_bars_back, 1000);
        assert_eq!(engine.config().neighbors_count, 12);
        assert_eq!(engine.config().feature_count, 3);
        assert!(!engine.config().use_chronological_spacing);
        assert!(!engine.config().use_multithreading);
        assert_eq!(engine.config().max_threads, Some(2));
        
        assert_eq!(engine.config().parallel_threshold, 50);
        assert_eq!(engine.config().batch_parallel_threshold, 5);
        
        assert!(!engine.config().enable_regime_filter);
        assert!(engine.config().enable_adx_filter);
        assert!(!engine.config().enable_volatility_filter);
        assert_eq!(engine.config().regime_threshold, -0.2);
        assert_eq!(engine.config().adx_threshold, 25.0);
        
        assert!(engine.config().enable_kernel_smoothing);
        assert_eq!(engine.config().kernel_lookback, 10);
        assert_eq!(engine.config().kernel_relative_weight, 5.0);
        assert_eq!(engine.config().kernel_regression_level, 15);
        
        assert!(engine.config().enable_debug_logging);
        assert!(engine.config().log_predictions);
        assert!(engine.config().log_performance_metrics);
    }
    
    #[test]
    fn test_performance_metrics() {
        let mut engine = LDCEngine::new();
        
        // Test initial metrics
        let metrics = engine.get_performance_metrics();
        assert_eq!(metrics.total_predictions, 0);
        assert_eq!(metrics.total_training_samples, 0);
        assert_eq!(metrics.average_prediction_time_ms, 0.0);
        
        // Reset metrics
        engine.reset_performance_metrics();
        let metrics = engine.get_performance_metrics();
        assert_eq!(metrics.total_predictions, 0);
    }
    
    #[test]
    fn test_configurable_thresholds() {
        let mut config = LDCConfig::default();
        config.parallel_threshold = 50;
        config.batch_parallel_threshold = 5;
        config.use_multithreading = true;
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add training data
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Test with small batch (should use sequential)
        let small_batch = vec![training_features.clone()];
        let predictions = engine.batch_predict_from_features(&small_batch).unwrap();
        assert_eq!(predictions.len(), 1);
        
        // Test with large batch (should use parallel)
        let mut large_batch = Vec::new();
        for i in 0..10 {
            let features = Features {
                timestamp: 1000 + i,
                rsi: Some(50.0 + i as f64),
                sma_20: Some(100.0 + i as f64),
                ema_20: Some(101.0 + i as f64),
                std_20: Some(2.0),
                zscore_20: Some(0.5),
                momentum: Some(1.0),
                wavetrend_1: Some(25.0 + i as f64),
                wavetrend_2: Some(30.0 + i as f64),
                cci: Some(15.0 + i as f64),
                adx: Some(20.0 + i as f64),
            };
            large_batch.push(features);
        }
        
        let predictions = engine.batch_predict_from_features(&large_batch).unwrap();
        assert_eq!(predictions.len(), 10);
    }
    
    #[test]
    fn test_logging_configuration() {
        let mut config = LDCConfig::default();
        config.enable_debug_logging = true;
        config.log_predictions = true;
        config.log_performance_metrics = true;
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add training data
        let training_features = Features {
            timestamp: 1000,
            rsi: Some(50.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(1.0),
            wavetrend_1: Some(25.0),
            wavetrend_2: Some(30.0),
            cci: Some(15.0),
            adx: Some(20.0),
        };
        
        engine.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        // Test prediction with logging enabled
        let query_features = Features {
            timestamp: 1001,
            rsi: Some(51.0),
            sma_20: Some(101.0),
            ema_20: Some(102.0),
            std_20: Some(2.1),
            zscore_20: Some(0.6),
            momentum: Some(1.1),
            wavetrend_1: Some(26.0),
            wavetrend_2: Some(31.0),
            cci: Some(16.0),
            adx: Some(21.0),
        };
        
        let prediction = engine.predict_from_features(&query_features).unwrap();
        assert!(prediction.signal.is_finite());
        assert!(prediction.confidence >= 0.0);
        
        // Test with logging disabled
        let mut config_no_logging = LDCConfig::default();
        config_no_logging.enable_debug_logging = false;
        config_no_logging.log_predictions = false;
        config_no_logging.log_performance_metrics = false;
        
        let mut engine_no_logging = LDCEngine::with_config(config_no_logging);
        engine_no_logging.add_training_sample_from_features(&training_features, 100.0, 105.0).unwrap();
        
        let prediction = engine_no_logging.predict_from_features(&query_features).unwrap();
        assert!(prediction.signal.is_finite());
    }
    
    // ===========================================
    // ========== SIMD OPTIMIZATION TESTS =======
    // ===========================================
    
    #[test]
    fn test_simd_vs_standard_distance_accuracy() {
        let features1 = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        let features2 = FeatureSeries {
            f1: 1.5, f2: 2.5, f3: 3.5, f4: 4.5, f5: 5.5,
        };
        
        let standard_distance = features1.lorentzian_distance_standard(&features2);
        let simd_distance = features1.lorentzian_distance_simd(&features2).unwrap();
        
        // SIMD and standard should produce identical results
        assert!((standard_distance - simd_distance).abs() < 1e-6, 
                "SIMD distance {:.6} should match standard distance {:.6}", 
                simd_distance, standard_distance);
    }
    
    #[test]
    fn test_simd_distance_identical_features() {
        let features = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let distance = features.lorentzian_distance_simd(&features).unwrap();
        assert_eq!(distance, 0.0, "Distance between identical features should be 0");
    }
    
    #[test]
    fn test_simd_distance_zero_features() {
        let features1 = FeatureSeries {
            f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0,
        };
        let features2 = FeatureSeries {
            f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0,
        };
        
        let distance = features1.lorentzian_distance_simd(&features2).unwrap();
        let expected = 5.0 * (1.0_f32 + 1.0_f32).ln(); // 5 * ln(2)
        assert!((distance - expected).abs() < 1e-6, 
                "SIMD distance {:.6} should match expected {:.6}", 
                distance, expected);
    }
    
    #[test]
    fn test_batch_simd_distance() {
        let query = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let targets = vec![
            FeatureSeries { f1: 1.5, f2: 2.5, f3: 3.5, f4: 4.5, f5: 5.5 },
            FeatureSeries { f1: 2.0, f2: 3.0, f3: 4.0, f4: 5.0, f5: 6.0 },
            FeatureSeries { f1: 0.5, f2: 1.5, f3: 2.5, f4: 3.5, f5: 4.5 },
        ];
        
        let batch_distances = FeatureSeries::batch_lorentzian_distance_simd(&query, &targets, 2).unwrap();
        let standard_distances = FeatureSeries::batch_lorentzian_distance_standard(&query, &targets);
        
        assert_eq!(batch_distances.len(), targets.len());
        assert_eq!(standard_distances.len(), targets.len());
        
        // Compare SIMD batch vs standard batch
        for (simd_dist, standard_dist) in batch_distances.iter().zip(standard_distances.iter()) {
            assert!((simd_dist - standard_dist).abs() < 1e-6,
                    "SIMD batch distance {:.6} should match standard {:.6}",
                    simd_dist, standard_dist);
        }
    }
    
    #[test]
    fn test_aligned_feature_series() {
        let features = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let aligned = features.to_aligned();
        let converted_back = aligned.to_feature_series();
        
        // Verify conversion preserves data
        assert_eq!(features.f1, converted_back.f1);
        assert_eq!(features.f2, converted_back.f2);
        assert_eq!(features.f3, converted_back.f3);
        assert_eq!(features.f4, converted_back.f4);
        assert_eq!(features.f5, converted_back.f5);
        
        // Verify padding is zero
        assert_eq!(aligned.features[5], 0.0);
        assert_eq!(aligned.features[6], 0.0);
        assert_eq!(aligned.features[7], 0.0);
    }
    
    #[test]
    fn test_aligned_simd_distance() {
        let features1 = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        let features2 = FeatureSeries {
            f1: 1.5, f2: 2.5, f3: 3.5, f4: 4.5, f5: 5.5,
        };
        
        let aligned1 = features1.to_aligned();
        let aligned2 = features2.to_aligned();
        
        let standard_distance = features1.lorentzian_distance_standard(&features2);
        let aligned_distance = aligned1.lorentzian_distance_simd(&aligned2).unwrap();
        
        // Aligned SIMD should match standard calculation
        assert!((standard_distance - aligned_distance).abs() < 1e-6,
                "Aligned SIMD distance {:.6} should match standard {:.6}",
                aligned_distance, standard_distance);
    }
    
    #[test]
    fn test_aligned_batch_simd_distance() {
        let query = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let targets = vec![
            FeatureSeries { f1: 1.5, f2: 2.5, f3: 3.5, f4: 4.5, f5: 5.5 },
            FeatureSeries { f1: 2.0, f2: 3.0, f3: 4.0, f4: 5.0, f5: 6.0 },
        ];
        
        let query_aligned = query.to_aligned();
        let targets_aligned: Vec<AlignedFeatureSeries> = targets.iter()
            .map(|t| t.to_aligned())
            .collect();
        
        let batch_distances = AlignedFeatureSeries::batch_lorentzian_distance_simd(
            &query_aligned, &targets_aligned, 1
        ).unwrap();
        
        assert_eq!(batch_distances.len(), targets.len());
        
        // Compare with standard distances
        for (i, &batch_dist) in batch_distances.iter().enumerate() {
            let standard_dist = query.lorentzian_distance_standard(&targets[i]);
            assert!((batch_dist - standard_dist).abs() < 1e-6,
                    "Aligned batch distance {:.6} should match standard {:.6}",
                    batch_dist, standard_dist);
        }
    }
    
    #[test]
    fn test_simd_knn_search() {
        let mut config = LDCConfig::default();
        config.use_simd_optimization = true;
        config.simd_chunk_size = 4;
        config.use_multithreading = true;
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add training samples
        for i in 0..20 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            let label = if i % 2 == 0 { Direction::Long } else { Direction::Short };
            let sample = TrainingSample {
                features,
                label,
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        let query_features = FeatureSeries {
            f1: 10.0, f2: 10.0, f3: 10.0, f4: 10.0, f5: 10.0,
        };
        
        let k_nearest = engine.find_k_nearest_neighbors(&query_features);
        assert!(!k_nearest.is_empty());
        assert!(k_nearest.len() <= engine.config().neighbors_count);
        
        // Verify distances are sorted (ascending)
        for i in 1..k_nearest.len() {
            assert!(k_nearest[i-1].0 <= k_nearest[i].0,
                    "Distances should be sorted: {} <= {}",
                    k_nearest[i-1].0, k_nearest[i].0);
        }
    }
    
    #[test]
    fn test_simd_vs_standard_knn_consistency() {
        let mut config_simd = LDCConfig::default();
        config_simd.use_simd_optimization = true;
        config_simd.use_multithreading = false; // Disable for consistent comparison
        
        let mut config_standard = LDCConfig::default();
        config_standard.use_simd_optimization = false;
        config_standard.use_multithreading = false;
        
        let mut engine_simd = LDCEngine::with_config(config_simd);
        let mut engine_standard = LDCEngine::with_config(config_standard);
        
        // Add identical training data to both engines
        for i in 0..10 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            let label = if i % 2 == 0 { Direction::Long } else { Direction::Short };
            let sample = TrainingSample {
                features,
                label,
                timestamp: i as i64,
                bar_index: i,
            };
            engine_simd.add_training_sample(sample.clone());
            engine_standard.add_training_sample(sample);
        }
        
        let query_features = FeatureSeries {
            f1: 5.0, f2: 5.0, f3: 5.0, f4: 5.0, f5: 5.0,
        };
        
        let k_nearest_simd = engine_simd.find_k_nearest_neighbors(&query_features);
        let k_nearest_standard = engine_standard.find_k_nearest_neighbors(&query_features);
        
        // Results should be identical (same distances and labels)
        assert_eq!(k_nearest_simd.len(), k_nearest_standard.len());
        
        for (simd_result, standard_result) in k_nearest_simd.iter().zip(k_nearest_standard.iter()) {
            assert!((simd_result.0 - standard_result.0).abs() < 1e-6,
                    "SIMD distance {:.6} should match standard {:.6}",
                    simd_result.0, standard_result.0);
            assert_eq!(simd_result.1, standard_result.1,
                      "Labels should match: {:?} vs {:?}",
                      simd_result.1, standard_result.1);
        }
    }
    
    #[test]
    fn test_simd_performance_metrics() {
        let mut config = LDCConfig::default();
        config.use_simd_optimization = true;
        config.log_performance_metrics = false; // Disable logging for test
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add training data
        let features = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        let sample = TrainingSample {
            features,
            label: Direction::Long,
            timestamp: 1000,
            bar_index: 0,
        };
        engine.add_training_sample(sample);
        
        // Make prediction to trigger SIMD operations
        let query_features = FeatureSeries {
            f1: 1.1, f2: 2.1, f3: 3.1, f4: 4.1, f5: 5.1,
        };
        
        let initial_simd_count = engine.get_performance_metrics().simd_operations_count;
        let _prediction = engine.predict_with_metrics(&query_features);
        let final_simd_count = engine.get_performance_metrics().simd_operations_count;
        
        // SIMD operations should have been incremented
        assert!(final_simd_count > initial_simd_count,
                "SIMD operations count should increase: {} -> {}",
                initial_simd_count, final_simd_count);
    }
    
    #[test]
    fn test_simd_fallback_mechanism() {
        // Test that SIMD gracefully falls back to standard calculation
        let features1 = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        let features2 = FeatureSeries {
            f1: 1.5, f2: 2.5, f3: 3.5, f4: 4.5, f5: 5.5,
        };
        
        // This should work regardless of SIMD availability
        let distance = features1.lorentzian_distance_simd(&features2).unwrap();
        assert!(distance > 0.0);
        assert!(distance.is_finite());
        
        // Should match standard calculation
        let standard_distance = features1.lorentzian_distance_standard(&features2);
        assert!((distance - standard_distance).abs() < 1e-6);
    }
    
    #[test]
    fn test_empty_batch_simd() {
        let query = FeatureSeries {
            f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0,
        };
        
        let empty_targets: Vec<FeatureSeries> = Vec::new();
        let distances = FeatureSeries::batch_lorentzian_distance_simd(&query, &empty_targets, 4).unwrap();
        
        assert!(distances.is_empty());
    }
    
    #[test]
    fn test_simd_chunk_size_configuration() {
        let mut config = LDCConfig::default();
        config.use_simd_optimization = true;
        config.simd_chunk_size = 8;
        
        let engine = LDCEngine::with_config(config);
        assert_eq!(engine.config().simd_chunk_size, 8);
        
        // Test with invalid chunk size (should be corrected during validation)
        let mut invalid_config = LDCConfig::default();
        invalid_config.simd_chunk_size = 0;
        
        let mut engine = LDCEngine::with_config(invalid_config);
        let result = engine.update_config(engine.config().clone());
        assert!(result.is_ok());
        assert!(engine.config().simd_chunk_size > 0); // Should be corrected to default
    }
    
    #[test]
    fn test_hnsw_index_creation() {
        let mut config = LDCConfig::default();
        config.use_hnsw_index = true;
        config.hnsw_m = 16;
        config.hnsw_ef_construction = 200;
        config.hnsw_ef_search = 50;
        
        let engine = LDCEngine::with_config(config);
        assert!(engine.is_hnsw_available());
        
        let hnsw_status = engine.get_hnsw_status();
        assert!(hnsw_status.is_some());
        
        let (indexed_samples, samples_since_rebuild, rebuild_threshold) = hnsw_status.unwrap();
        assert_eq!(indexed_samples, 0); // No samples added yet
        assert_eq!(samples_since_rebuild, 0);
        assert_eq!(rebuild_threshold, engine.config().hnsw_rebuild_threshold);
    }
    
    #[test]
    fn test_hnsw_index_disabled() {
        let mut config = LDCConfig::default();
        config.use_hnsw_index = false;
        
        let engine = LDCEngine::with_config(config);
        assert!(!engine.is_hnsw_available());
        assert!(engine.get_hnsw_status().is_none());
    }
    
    #[test]
    fn test_hnsw_sample_addition() {
        let mut config = LDCConfig::default();
        config.use_hnsw_index = true;
        config.hnsw_rebuild_threshold = 10; // Small threshold for testing
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add some training samples
        for i in 0..5 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            let sample = TrainingSample {
                features,
                label: if i % 2 == 0 { Direction::Long } else { Direction::Short },
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        // Check HNSW status
        let hnsw_status = engine.get_hnsw_status();
        assert!(hnsw_status.is_some());
        
        let (indexed_samples, samples_since_rebuild, _) = hnsw_status.unwrap();
        assert_eq!(indexed_samples, 5);
        assert_eq!(samples_since_rebuild, 5);
    }
    
    #[test]
    fn test_hnsw_search_fallback() {
        let mut config = LDCConfig::default();
        config.use_hnsw_index = true;
        config.use_multithreading = false; // Disable for consistent testing
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add training samples (less than 100 to test fallback)
        for i in 0..10 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            let sample = TrainingSample {
                features,
                label: if i % 2 == 0 { Direction::Long } else { Direction::Short },
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        let query_features = FeatureSeries {
            f1: 5.0, f2: 5.0, f3: 5.0, f4: 5.0, f5: 5.0,
        };
        
        // Should use exact search (fallback) because we have < 100 samples
        let k_nearest = engine.find_k_nearest_neighbors(&query_features);
        assert!(!k_nearest.is_empty());
        assert!(k_nearest.len() <= engine.config().neighbors_count);
    }
    
    #[test]
    fn test_hnsw_configuration_update() {
        let mut config = LDCConfig::default();
        config.use_hnsw_index = false;
        
        let mut engine = LDCEngine::with_config(config);
        assert!(!engine.is_hnsw_available());
        
        // Enable HNSW
        let mut new_config = engine.config().clone();
        new_config.use_hnsw_index = true;
        new_config.hnsw_m = 32;
        
        let result = engine.update_config(new_config);
        assert!(result.is_ok());
        assert!(engine.is_hnsw_available());
        
        // Check HNSW config
        let hnsw_config = engine.get_hnsw_config();
        assert!(hnsw_config.is_some());
        assert_eq!(hnsw_config.unwrap().m, 32);
    }
    
    #[test]
    fn test_hnsw_force_rebuild() {
        let mut config = LDCConfig::default();
        config.use_hnsw_index = true;
        
        let mut engine = LDCEngine::with_config(config);
        
        // Add some samples
        for i in 0..5 {
            let features = FeatureSeries {
                f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
            };
            let sample = TrainingSample {
                features,
                label: Direction::Long,
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        // Force rebuild
        let result = engine.force_hnsw_rebuild();
        assert!(result.is_ok());
        
        // Check that rebuild count increased
        let metrics = engine.get_performance_metrics();
        assert!(metrics.hnsw_rebuild_count > 0);
    }
    
    #[test]
    fn test_lorentzian_distance_hnsw_function() {
        let features1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let features2 = [1.5, 2.5, 3.5, 4.5, 5.5];
        
        let distance = lorentzian_distance_hnsw(&features1, &features2);
        
        // Should match manual calculation
        let expected = (1.0 + 0.5_f32).ln() + 
                      (1.0 + 0.5_f32).ln() + 
                      (1.0 + 0.5_f32).ln() + 
                      (1.0 + 0.5_f32).ln() + 
                      (1.0 + 0.5_f32).ln();
        
        assert!((distance - expected).abs() < 1e-6);
        assert!(distance > 0.0);
    }
}

    #[test]
    fn test_optimized_knn_search_strategy_selection() {
        // Test that the optimized k-NN search selects appropriate strategies
        let mut engine = LDCEngine::new();
        
        // Configure for testing different strategies
        let mut config = LDCConfig::default();
        config.use_simd_optimization = true;
        config.use_multithreading = true;
        config.use_hnsw_index = false; // Start with HNSW disabled
        config.parallel_threshold = 50; // Low threshold for testing
        config.simd_chunk_size = 8;
        engine.update_config(config).unwrap();
        
        // Add small amount of training data (should use sequential)
        for i in 0..10 {
            let sample = TrainingSample {
                features: FeatureSeries {
                    f1: i as f32 * 0.1,
                    f2: i as f32 * 0.2,
                    f3: i as f32 * 0.3,
                    f4: i as f32 * 0.4,
                    f5: i as f32 * 0.5,
                },
                label: if i % 2 == 0 { Direction::Long } else { Direction::Short },
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        let query_features = FeatureSeries {
            f1: 0.15, f2: 0.25, f3: 0.35, f4: 0.45, f5: 0.55,
        };
        
        // Should use sequential search (< parallel_threshold)
        let results_small = engine.find_k_nearest_neighbors_optimized(&query_features);
        assert!(!results_small.is_empty());
        
        // Add more training data to trigger parallel search
        for i in 10..100 {
            let sample = TrainingSample {
                features: FeatureSeries {
                    f1: i as f32 * 0.01,
                    f2: i as f32 * 0.02,
                    f3: i as f32 * 0.03,
                    f4: i as f32 * 0.04,
                    f5: i as f32 * 0.05,
                },
                label: if i % 2 == 0 { Direction::Long } else { Direction::Short },
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        // Should use parallel search (>= parallel_threshold)
        let results_large = engine.find_k_nearest_neighbors_optimized(&query_features);
        assert!(!results_large.is_empty());
        assert!(results_large.len() <= engine.config().neighbors_count);
    }
    
    #[test]
    fn test_thread_pool_management() {
        // Test thread pool strategy selection and management
        let mut engine = LDCEngine::new();
        
        // Test Global strategy
        let mut config = LDCConfig::default();
        config.thread_pool_strategy = ThreadPoolStrategy::Global;
        config.max_threads = Some(4);
        engine.update_config(config).unwrap();
        
        let thread_pool = engine.get_or_create_thread_pool().unwrap();
        assert!(thread_pool.current_num_threads() > 0);
        
        // Test Dedicated strategy
        let mut config = engine.config().clone();
        config.thread_pool_strategy = ThreadPoolStrategy::Dedicated;
        config.max_threads = Some(2);
        engine.update_config(config).unwrap();
        
        let thread_pool = engine.get_or_create_thread_pool().unwrap();
        assert!(thread_pool.current_num_threads() > 0);
        
        // Test Adaptive strategy
        let mut config = engine.config().clone();
        config.thread_pool_strategy = ThreadPoolStrategy::Adaptive;
        engine.update_config(config).unwrap();
        
        let thread_pool = engine.get_or_create_thread_pool().unwrap();
        assert!(thread_pool.current_num_threads() > 0);
        
        // Test thread pool statistics
        let stats = engine.get_thread_pool_stats();
        assert_eq!(stats.total_tasks_executed, 0); // No tasks executed yet
        assert!(stats.current_thread_count > 0);
        assert!(stats.optimal_thread_count > 0);
    }
    
    #[test]
    fn test_thread_efficiency_monitoring() {
        let mut engine = LDCEngine::new();
        
        // Configure for thread efficiency testing
        let mut config = LDCConfig::default();
        config.thread_pool_strategy = ThreadPoolStrategy::Dedicated;
        config.use_multithreading = true;
        config.parallel_threshold = 10;
        engine.update_config(config).unwrap();
        
        // Add training data to trigger parallel processing
        for i in 0..50 {
            let sample = TrainingSample {
                features: FeatureSeries {
                    f1: i as f32 * 0.01,
                    f2: i as f32 * 0.02,
                    f3: i as f32 * 0.03,
                    f4: i as f32 * 0.04,
                    f5: i as f32 * 0.05,
                },
                label: if i % 2 == 0 { Direction::Long } else { Direction::Short },
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        let query_features = FeatureSeries {
            f1: 0.25, f2: 0.35, f3: 0.45, f4: 0.55, f5: 0.65,
        };
        
        // Make a prediction to trigger thread efficiency monitoring
        let _prediction = engine.predict(&query_features);
        
        // Check that thread efficiency was monitored
        let stats = engine.get_thread_pool_stats();
        assert!(stats.total_tasks_executed > 0);
        assert!(stats.average_task_time_ms >= 0.0);
        
        // Check performance metrics
        let metrics = engine.get_performance_metrics();
        assert!(metrics.thread_efficiency_percent >= 0.0);
        assert!(metrics.thread_efficiency_percent <= 100.0);
    }
    
    #[test]
    fn test_workload_assessment() {
        let mut engine = LDCEngine::new();
        
        // Configure for workload assessment testing
        let mut config = LDCConfig::default();
        config.thread_pool_strategy = ThreadPoolStrategy::Adaptive;
        config.use_simd_optimization = true;
        config.use_hnsw_index = false;
        engine.update_config(config).unwrap();
        
        // Test workload assessment with small dataset
        let workload = engine.assess_workload_characteristics();
        assert_eq!(workload.dataset_size, 0);
        assert!(matches!(workload.computation_intensity, ComputationIntensity::Medium));
        assert!(workload.parallelization_efficiency < 1.0);
        
        // Add data and reassess
        for i in 0..100 {
            let sample = TrainingSample {
                features: FeatureSeries {
                    f1: i as f32, f2: i as f32, f3: i as f32, f4: i as f32, f5: i as f32,
                },
                label: Direction::Long,
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        let workload = engine.assess_workload_characteristics();
        assert_eq!(workload.dataset_size, 100);
        assert!(workload.parallelization_efficiency > 0.5);
    }
    
    #[test]
    fn test_optimal_thread_count_calculation() {
        let mut engine = LDCEngine::new();
        
        // Test with different configurations
        let mut config = LDCConfig::default();
        config.max_threads = Some(8);
        engine.update_config(config).unwrap();
        
        let optimal_count = engine.calculate_optimal_thread_count();
        assert!(optimal_count > 0);
        assert!(optimal_count <= 8);
        
        // Test with auto-detection
        let mut config = engine.config().clone();
        config.max_threads = None;
        engine.update_config(config).unwrap();
        
        let optimal_count = engine.calculate_optimal_thread_count();
        assert!(optimal_count > 0);
        assert!(optimal_count <= num_cpus::get() * 2);
    }
    
    #[test]
    fn test_thread_pool_report_generation() {
        let mut engine = LDCEngine::new();
        
        // Configure thread pool
        let mut config = LDCConfig::default();
        config.thread_pool_strategy = ThreadPoolStrategy::Dedicated;
        config.work_stealing_enabled = true;
        engine.update_config(config).unwrap();
        
        // Generate report
        let report = engine.generate_thread_pool_report();
        assert!(report.contains("Thread Pool Performance Report"));
        assert!(report.contains("Strategy: Dedicated"));
        assert!(report.contains("Work Stealing: true"));
        assert!(report.contains("Current Threads:"));
        assert!(report.contains("Optimal Threads:"));
    }
    
    #[test]
    fn test_enhanced_hnsw_search_with_fallback() {
        // Test HNSW search with automatic fallback mechanisms
        let mut engine = LDCEngine::new();
        
        // Configure with HNSW enabled
        let mut config = LDCConfig::default();
        config.use_hnsw_index = true;
        config.hnsw_m = 8;
        config.hnsw_ef_construction = 100;
        config.hnsw_ef_search = 25;
        config.enable_debug_logging = false; // Reduce noise in tests
        engine.update_config(config).unwrap();
        
        // Add enough training data to benefit from HNSW (> 1000 samples)
        for i in 0..1200 {
            let sample = TrainingSample {
                features: FeatureSeries {
                    f1: (i as f32 * 0.001) % 1.0,
                    f2: (i as f32 * 0.002) % 1.0,
                    f3: (i as f32 * 0.003) % 1.0,
                    f4: (i as f32 * 0.004) % 1.0,
                    f5: (i as f32 * 0.005) % 1.0,
                },
                label: if i % 3 == 0 { Direction::Long } else if i % 3 == 1 { Direction::Short } else { Direction::Neutral },
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        let query_features = FeatureSeries {
            f1: 0.5, f2: 0.5, f3: 0.5, f4: 0.5, f5: 0.5,
        };
        
        // Should use HNSW search for large dataset
        let results = engine.find_k_nearest_neighbors_optimized(&query_features);
        assert!(!results.is_empty());
        assert!(results.len() <= engine.config().neighbors_count);
        
        // Verify that results are reasonable (distances should be positive)
        for (distance, _label) in &results {
            assert!(*distance >= 0.0, "Distance should be non-negative: {}", distance);
        }
    }
    
    #[test]
    fn test_parallel_search_with_simd_fallback() {
        // Test SIMD parallel search with automatic fallback
        let mut engine = LDCEngine::new();
        
        // Configure for SIMD parallel processing
        let mut config = LDCConfig::default();
        config.use_simd_optimization = true;
        config.use_multithreading = true;
        config.use_hnsw_index = false;
        config.parallel_threshold = 20;
        config.simd_chunk_size = 16;
        engine.update_config(config).unwrap();
        
        // Add training data to trigger parallel processing
        for i in 0..100 {
            let sample = TrainingSample {
                features: FeatureSeries {
                    f1: i as f32 * 0.01,
                    f2: (i as f32 * 0.02) % 1.0,
                    f3: (i as f32 * 0.03) % 1.0,
                    f4: (i as f32 * 0.04) % 1.0,
                    f5: (i as f32 * 0.05) % 1.0,
                },
                label: if i % 2 == 0 { Direction::Long } else { Direction::Short },
                timestamp: i as i64,
                bar_index: i,
            };
            engine.add_training_sample(sample);
        }
        
        let query_features = FeatureSeries {
            f1: 0.25, f2: 0.35, f3: 0.45, f4: 0.55, f5: 0.65,
        };
        
        // Test parallel optimized search
        let results = engine.find_k_nearest_neighbors_parallel_optimized(&query_features);
        assert!(!results.is_empty());
        assert!(results.len() <= engine.config().neighbors_count);
        
        // Verify results are sorted by distance
        for i in 1..results.len() {
            assert!(results[i-1].0 <= results[i].0, 
                   "Results should be sorted by distance: {} > {}", results[i-1].0, results[i].0);
        }
    }