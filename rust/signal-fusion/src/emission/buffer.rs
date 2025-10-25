//! Signal buffering system for reliable signal emission
//! 
//! This module provides in-memory buffering with configurable size limits and overflow handling,
//! plus optional disk persistence for recovery after service restarts.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn, error};
use anyhow::{Result, bail, Context};
use crate::TradingSignal;

/// Configuration for signal buffer behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Maximum number of signals to buffer in memory
    pub max_size: usize,
    
    /// Strategy to use when buffer is full
    pub overflow_strategy: OverflowStrategy,
    
    /// Optional disk persistence configuration
    pub persistence: Option<PersistenceConfig>,
    
    /// Enable buffer metrics collection
    pub enable_metrics: bool,
    
    /// Buffer utilization warning threshold (0.0 to 1.0)
    pub warning_threshold: f32,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_size: 1000,
            overflow_strategy: OverflowStrategy::DropOldest,
            persistence: None,
            enable_metrics: true,
            warning_threshold: 0.8,
        }
    }
}

/// Strategy for handling buffer overflow conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverflowStrategy {
    /// Drop the oldest signals when buffer is full
    DropOldest,
    /// Drop the newest signal when buffer is full
    DropNewest,
    /// Return an error when buffer is full
    Error,
    /// Drop signals with lowest confidence when buffer is full
    DropLowestConfidence,
}

/// Configuration for buffer persistence to disk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Directory path for buffer persistence files
    pub persist_path: PathBuf,
    
    /// Enable automatic persistence on buffer changes
    pub auto_persist: bool,
    
    /// Persistence interval in seconds (for auto_persist)
    pub persist_interval_sec: u64,
    
    /// Maximum number of backup files to keep
    pub max_backup_files: usize,
    
    /// Enable atomic file operations to prevent corruption
    pub atomic_operations: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            persist_path: PathBuf::from("/tmp/signal_buffer"),
            auto_persist: true,
            persist_interval_sec: 60,
            max_backup_files: 5,
            atomic_operations: true,
        }
    }
}

/// A buffered signal with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferedSignal {
    /// The trading signal
    pub signal: TradingSignal,
    
    /// Timestamp when signal was added to buffer
    pub buffered_at: i64,
    
    /// Number of retry attempts for this signal
    pub retry_count: u32,
    
    /// Priority for ordering (higher = more important)
    pub priority: f32,
}

impl BufferedSignal {
    /// Create a new buffered signal
    pub fn new(signal: TradingSignal) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
            
        // Calculate priority based on confidence and recency
        let age_factor = 1.0; // Newer signals get slight priority boost
        let priority = signal.confidence * age_factor;
        
        Self {
            signal,
            buffered_at: now,
            retry_count: 0,
            priority,
        }
    }
    
    /// Create a buffered signal with custom priority
    pub fn with_priority(signal: TradingSignal, priority: f32) -> Self {
        let mut buffered = Self::new(signal);
        buffered.priority = priority;
        buffered
    }
    
    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
    
    /// Get age of buffered signal in seconds
    pub fn age_seconds(&self) -> i64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        now - self.buffered_at
    }
}

/// Metrics for buffer utilization and performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferMetrics {
    /// Current number of signals in buffer
    pub current_size: usize,
    
    /// Maximum buffer size
    pub max_size: usize,
    
    /// Current utilization percentage (0.0 to 1.0)
    pub utilization: f32,
    
    /// Total signals added to buffer
    pub total_added: u64,
    
    /// Total signals removed from buffer
    pub total_removed: u64,
    
    /// Total signals dropped due to overflow
    pub total_dropped: u64,
    
    /// Total buffer overflow events
    pub overflow_events: u64,
    
    /// Average age of signals in buffer (seconds)
    pub avg_age_seconds: f32,
    
    /// Oldest signal age in buffer (seconds)
    pub oldest_signal_age: i64,
    
    /// Last persistence operation timestamp
    pub last_persist_time: Option<i64>,
    
    /// Last recovery operation timestamp
    pub last_recovery_time: Option<i64>,
}

impl Default for BufferMetrics {
    fn default() -> Self {
        Self {
            current_size: 0,
            max_size: 0,
            utilization: 0.0,
            total_added: 0,
            total_removed: 0,
            total_dropped: 0,
            overflow_events: 0,
            avg_age_seconds: 0.0,
            oldest_signal_age: 0,
            last_persist_time: None,
            last_recovery_time: None,
        }
    }
}

/// In-memory signal buffer with configurable size limits and overflow handling
pub struct SignalBuffer {
    /// Internal buffer using VecDeque for efficient FIFO operations
    buffer: VecDeque<BufferedSignal>,
    
    /// Buffer configuration
    config: BufferConfig,
    
    /// Buffer metrics
    metrics: BufferMetrics,
    
    /// Last persistence time for auto-persist
    last_persist_time: Option<Instant>,
}

impl SignalBuffer {
    /// Create a new signal buffer with the given configuration
    pub fn new(config: BufferConfig) -> Self {
        let mut metrics = BufferMetrics::default();
        metrics.max_size = config.max_size;
        
        Self {
            buffer: VecDeque::with_capacity(config.max_size),
            config,
            metrics,
            last_persist_time: None,
        }
    }
    
    /// Create a new signal buffer with default configuration
    pub fn with_default_config() -> Self {
        Self::new(BufferConfig::default())
    }
    
    /// Push a signal to the buffer, handling overflow according to strategy
    pub fn push(&mut self, signal: TradingSignal) -> Result<()> {
        let buffered_signal = BufferedSignal::new(signal);
        
        // Check if buffer is full
        if self.buffer.len() >= self.config.max_size {
            self.metrics.overflow_events += 1;
            
            match self.config.overflow_strategy {
                OverflowStrategy::DropOldest => {
                    if let Some(dropped) = self.buffer.pop_front() {
                        self.metrics.total_dropped += 1;
                        debug!(
                            "Dropped oldest signal due to buffer overflow: {} (age: {}s)",
                            dropped.signal.to_compact_string(),
                            dropped.age_seconds()
                        );
                    }
                }
                OverflowStrategy::DropNewest => {
                    self.metrics.total_dropped += 1;
                    debug!(
                        "Dropped newest signal due to buffer overflow: {}",
                        buffered_signal.signal.to_compact_string()
                    );
                    return Ok(()); // Don't add the new signal
                }
                OverflowStrategy::Error => {
                    bail!("Buffer overflow: maximum size {} exceeded", self.config.max_size);
                }
                OverflowStrategy::DropLowestConfidence => {
                    // Find signal with lowest confidence
                    if let Some((min_idx, _)) = self.buffer
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            a.signal.confidence.partial_cmp(&b.signal.confidence)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    {
                        if let Some(dropped) = self.buffer.remove(min_idx) {
                            self.metrics.total_dropped += 1;
                            debug!(
                                "Dropped signal with lowest confidence due to buffer overflow: {} (confidence: {:.3})",
                                dropped.signal.to_compact_string(),
                                dropped.signal.confidence
                            );
                        }
                    }
                }
            }
        }
        
        // Add the new signal
        self.buffer.push_back(buffered_signal.clone());
        self.metrics.total_added += 1;
        
        // Update metrics
        self.update_metrics();
        
        // Check warning threshold
        if self.metrics.utilization >= self.config.warning_threshold {
            warn!(
                "Signal buffer utilization high: {:.1}% ({}/{})",
                self.metrics.utilization * 100.0,
                self.metrics.current_size,
                self.metrics.max_size
            );
        }
        
        debug!(
            "Added signal to buffer: {} (buffer size: {}/{})",
            buffered_signal.signal.to_compact_string(),
            self.buffer.len(),
            self.config.max_size
        );
        
        // Auto-persist if enabled
        if let Some(ref persist_config) = self.config.persistence {
            if persist_config.auto_persist {
                if let Some(last_persist) = self.last_persist_time {
                    if last_persist.elapsed().as_secs() >= persist_config.persist_interval_sec {
                        if let Err(e) = self.persist() {
                            warn!("Auto-persist failed: {}", e);
                        }
                    }
                } else {
                    // First persist
                    if let Err(e) = self.persist() {
                        warn!("Initial auto-persist failed: {}", e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Pop the oldest signal from the buffer (FIFO ordering)
    pub fn pop(&mut self) -> Option<BufferedSignal> {
        if let Some(signal) = self.buffer.pop_front() {
            self.metrics.total_removed += 1;
            self.update_metrics();
            
            debug!(
                "Removed signal from buffer: {} (buffer size: {}/{})",
                signal.signal.to_compact_string(),
                self.buffer.len(),
                self.config.max_size
            );
            
            Some(signal)
        } else {
            None
        }
    }
    
    /// Pop the signal with highest priority
    pub fn pop_highest_priority(&mut self) -> Option<BufferedSignal> {
        if self.buffer.is_empty() {
            return None;
        }
        
        // Find signal with highest priority
        let (max_idx, _) = self.buffer
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.priority.partial_cmp(&b.priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
        
        if let Some(signal) = self.buffer.remove(max_idx) {
            self.metrics.total_removed += 1;
            self.update_metrics();
            
            debug!(
                "Removed highest priority signal from buffer: {} (priority: {:.3}, buffer size: {}/{})",
                signal.signal.to_compact_string(),
                signal.priority,
                self.buffer.len(),
                self.config.max_size
            );
            
            Some(signal)
        } else {
            None
        }
    }
    
    /// Peek at the oldest signal without removing it
    pub fn peek(&self) -> Option<&BufferedSignal> {
        self.buffer.front()
    }
    
    /// Get current buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.config.max_size
    }
    
    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.config.max_size
    }
    
    /// Get current buffer utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        if self.config.max_size == 0 {
            0.0
        } else {
            self.buffer.len() as f32 / self.config.max_size as f32
        }
    }
    
    /// Clear all signals from the buffer
    pub fn clear(&mut self) {
        let cleared_count = self.buffer.len();
        self.buffer.clear();
        self.update_metrics();
        
        info!("Cleared {} signals from buffer", cleared_count);
    }
    
    /// Get buffer metrics
    pub fn metrics(&self) -> &BufferMetrics {
        &self.metrics
    }
    
    /// Get buffer configuration
    pub fn config(&self) -> &BufferConfig {
        &self.config
    }
    
    /// Update buffer configuration
    pub fn update_config(&mut self, config: BufferConfig) -> Result<()> {
        // If max_size is reduced, we may need to drop signals
        if config.max_size < self.config.max_size && self.buffer.len() > config.max_size {
            let signals_to_drop = self.buffer.len() - config.max_size;
            
            match config.overflow_strategy {
                OverflowStrategy::DropOldest => {
                    for _ in 0..signals_to_drop {
                        if let Some(dropped) = self.buffer.pop_front() {
                            self.metrics.total_dropped += 1;
                            debug!(
                                "Dropped signal due to config change: {}",
                                dropped.signal.to_compact_string()
                            );
                        }
                    }
                }
                OverflowStrategy::DropNewest => {
                    for _ in 0..signals_to_drop {
                        if let Some(dropped) = self.buffer.pop_back() {
                            self.metrics.total_dropped += 1;
                            debug!(
                                "Dropped signal due to config change: {}",
                                dropped.signal.to_compact_string()
                            );
                        }
                    }
                }
                OverflowStrategy::Error => {
                    bail!(
                        "Cannot reduce buffer size from {} to {} with Error overflow strategy: {} signals would be lost",
                        self.config.max_size,
                        config.max_size,
                        signals_to_drop
                    );
                }
                OverflowStrategy::DropLowestConfidence => {
                    // Sort by confidence and drop lowest
                    let mut indices_to_remove: Vec<_> = (0..self.buffer.len()).collect();
                    indices_to_remove.sort_by(|&a, &b| {
                        self.buffer[a].signal.confidence
                            .partial_cmp(&self.buffer[b].signal.confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    
                    // Remove from highest index to lowest to maintain indices
                    for &idx in indices_to_remove.iter().take(signals_to_drop).rev() {
                        if let Some(dropped) = self.buffer.remove(idx) {
                            self.metrics.total_dropped += 1;
                            debug!(
                                "Dropped signal due to config change: {} (confidence: {:.3})",
                                dropped.signal.to_compact_string(),
                                dropped.signal.confidence
                            );
                        }
                    }
                }
            }
        }
        
        self.config = config;
        self.metrics.max_size = self.config.max_size;
        self.update_metrics();
        
        info!("Updated buffer configuration: max_size={}, strategy={:?}", 
              self.config.max_size, self.config.overflow_strategy);
        
        Ok(())
    }
    
    /// Update internal metrics
    fn update_metrics(&mut self) {
        self.metrics.current_size = self.buffer.len();
        self.metrics.utilization = self.utilization();
        
        if !self.buffer.is_empty() {
            // Calculate average age
            let total_age: i64 = self.buffer.iter().map(|s| s.age_seconds()).sum();
            self.metrics.avg_age_seconds = total_age as f32 / self.buffer.len() as f32;
            
            // Find oldest signal
            self.metrics.oldest_signal_age = self.buffer
                .iter()
                .map(|s| s.age_seconds())
                .max()
                .unwrap_or(0);
        } else {
            self.metrics.avg_age_seconds = 0.0;
            self.metrics.oldest_signal_age = 0;
        }
    }
    
    /// Persist buffer contents to disk
    pub fn persist(&mut self) -> Result<()> {
        let persist_config = match &self.config.persistence {
            Some(config) => config,
            None => {
                debug!("Persistence not configured, skipping persist operation");
                return Ok(());
            }
        };
        
        if self.buffer.is_empty() {
            debug!("Buffer is empty, skipping persist operation");
            return Ok(());
        }
        
        // Create persist directory if it doesn't exist
        fs::create_dir_all(&persist_config.persist_path)
            .with_context(|| format!("Failed to create persist directory: {:?}", persist_config.persist_path))?;
        
        let persist_file = persist_config.persist_path.join("signal_buffer.json");
        
        if persist_config.atomic_operations {
            self.persist_atomic(&persist_file)?;
        } else {
            self.persist_direct(&persist_file)?;
        }
        
        // Update metrics
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        self.metrics.last_persist_time = Some(now);
        self.last_persist_time = Some(Instant::now());
        
        // Manage backup files
        self.manage_backup_files(persist_config)?;
        
        info!(
            "Persisted {} signals to disk: {:?}",
            self.buffer.len(),
            persist_file
        );
        
        Ok(())
    }
    
    /// Persist buffer using atomic operations (write to temp file, then rename)
    fn persist_atomic(&self, persist_file: &Path) -> Result<()> {
        let temp_file = persist_file.with_extension("tmp");
        
        // Write to temporary file first
        {
            let file = File::create(&temp_file)
                .with_context(|| format!("Failed to create temp file: {:?}", temp_file))?;
            let writer = BufWriter::new(file);
            
            serde_json::to_writer_pretty(writer, &self.buffer)
                .with_context(|| "Failed to serialize buffer to JSON")?;
        }
        
        // Atomically rename temp file to final file
        fs::rename(&temp_file, persist_file)
            .with_context(|| format!("Failed to rename temp file to persist file: {:?}", persist_file))?;
        
        Ok(())
    }
    
    /// Persist buffer directly to file (non-atomic)
    fn persist_direct(&self, persist_file: &Path) -> Result<()> {
        let file = File::create(persist_file)
            .with_context(|| format!("Failed to create persist file: {:?}", persist_file))?;
        let writer = BufWriter::new(file);
        
        serde_json::to_writer_pretty(writer, &self.buffer)
            .with_context(|| "Failed to serialize buffer to JSON")?;
        
        Ok(())
    }
    
    /// Manage backup files according to configuration
    fn manage_backup_files(&self, persist_config: &PersistenceConfig) -> Result<()> {
        if persist_config.max_backup_files == 0 {
            return Ok(());
        }
        
        let persist_file = persist_config.persist_path.join("signal_buffer.json");
        
        // Create backup of current file
        if persist_file.exists() {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            let backup_file = persist_config.persist_path
                .join(format!("signal_buffer.{}.json", timestamp));
            
            if let Err(e) = fs::copy(&persist_file, &backup_file) {
                warn!("Failed to create backup file: {}", e);
            } else {
                debug!("Created backup file: {:?}", backup_file);
            }
        }
        
        // Clean up old backup files
        self.cleanup_old_backups(persist_config)?;
        
        Ok(())
    }
    
    /// Clean up old backup files, keeping only the most recent ones
    fn cleanup_old_backups(&self, persist_config: &PersistenceConfig) -> Result<()> {
        let persist_dir = &persist_config.persist_path;
        
        if !persist_dir.exists() {
            return Ok(());
        }
        
        // Find all backup files
        let mut backup_files = Vec::new();
        
        for entry in fs::read_dir(persist_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name.starts_with("signal_buffer.") && 
                   file_name.ends_with(".json") && 
                   file_name != "signal_buffer.json" {
                    
                    if let Ok(metadata) = entry.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            backup_files.push((path, modified));
                        }
                    }
                }
            }
        }
        
        // Sort by modification time (newest first)
        backup_files.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Remove excess backup files
        if backup_files.len() > persist_config.max_backup_files {
            for (path, _) in backup_files.iter().skip(persist_config.max_backup_files) {
                if let Err(e) = fs::remove_file(path) {
                    warn!("Failed to remove old backup file {:?}: {}", path, e);
                } else {
                    debug!("Removed old backup file: {:?}", path);
                }
            }
        }
        
        Ok(())
    }
    
    /// Restore buffer contents from disk
    pub fn restore(&mut self) -> Result<()> {
        let persist_config = match self.config.persistence.clone() {
            Some(config) => config,
            None => {
                debug!("Persistence not configured, skipping restore operation");
                return Ok(());
            }
        };
        
        let persist_file = persist_config.persist_path.join("signal_buffer.json");
        
        if !persist_file.exists() {
            debug!("Persist file does not exist, starting with empty buffer: {:?}", persist_file);
            return Ok(());
        }
        
        // Try to restore from main file first
        match self.restore_from_file(&persist_file) {
            Ok(()) => {
                info!(
                    "Restored {} signals from disk: {:?}",
                    self.buffer.len(),
                    persist_file
                );
                return Ok(());
            }
            Err(e) => {
                error!("Failed to restore from main file: {}", e);
                warn!("Attempting to restore from backup files");
            }
        }
        
        // Try to restore from backup files
        self.restore_from_backups(&persist_config)?;
        
        Ok(())
    }
    
    /// Restore buffer from a specific file
    fn restore_from_file(&mut self, file_path: &Path) -> Result<()> {
        let file = File::open(file_path)
            .with_context(|| format!("Failed to open persist file: {:?}", file_path))?;
        let reader = BufReader::new(file);
        
        let restored_buffer: VecDeque<BufferedSignal> = serde_json::from_reader(reader)
            .with_context(|| "Failed to deserialize buffer from JSON")?;
        
        // Validate restored signals
        let mut valid_signals = VecDeque::new();
        let mut invalid_count = 0;
        
        for buffered_signal in restored_buffer {
            if let Err(e) = buffered_signal.signal.validate() {
                warn!(
                    "Skipping invalid restored signal: {} - {}",
                    buffered_signal.signal.to_compact_string(),
                    e
                );
                invalid_count += 1;
                continue;
            }
            
            // Check if signal is too old
            if buffered_signal.age_seconds() > 3600 { // 1 hour
                debug!(
                    "Skipping old restored signal: {} (age: {}s)",
                    buffered_signal.signal.to_compact_string(),
                    buffered_signal.age_seconds()
                );
                invalid_count += 1;
                continue;
            }
            
            valid_signals.push_back(buffered_signal);
        }
        
        // Apply size limits
        if valid_signals.len() > self.config.max_size {
            let excess = valid_signals.len() - self.config.max_size;
            
            match self.config.overflow_strategy {
                OverflowStrategy::DropOldest => {
                    for _ in 0..excess {
                        valid_signals.pop_front();
                    }
                }
                OverflowStrategy::DropNewest => {
                    for _ in 0..excess {
                        valid_signals.pop_back();
                    }
                }
                OverflowStrategy::DropLowestConfidence => {
                    // Convert to Vec for sorting
                    let mut signals: Vec<_> = valid_signals.into_iter().collect();
                    signals.sort_by(|a, b| {
                        b.signal.confidence.partial_cmp(&a.signal.confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    
                    // Keep only the highest confidence signals
                    signals.truncate(self.config.max_size);
                    valid_signals = signals.into_iter().collect();
                }
                OverflowStrategy::Error => {
                    bail!(
                        "Restored buffer size {} exceeds maximum {} with Error overflow strategy",
                        valid_signals.len(),
                        self.config.max_size
                    );
                }
            }
            
            warn!(
                "Dropped {} excess signals during restore due to size limits",
                excess
            );
        }
        
        self.buffer = valid_signals;
        
        // Update metrics
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        self.metrics.last_recovery_time = Some(now);
        self.update_metrics();
        
        if invalid_count > 0 {
            warn!("Skipped {} invalid signals during restore", invalid_count);
        }
        
        Ok(())
    }
    
    /// Try to restore from backup files
    fn restore_from_backups(&mut self, persist_config: &PersistenceConfig) -> Result<()> {
        let persist_dir = &persist_config.persist_path;
        
        if !persist_dir.exists() {
            bail!("Persist directory does not exist: {:?}", persist_dir);
        }
        
        // Find all backup files
        let mut backup_files = Vec::new();
        
        for entry in fs::read_dir(persist_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name.starts_with("signal_buffer.") && 
                   file_name.ends_with(".json") && 
                   file_name != "signal_buffer.json" {
                    
                    if let Ok(metadata) = entry.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            backup_files.push((path, modified));
                        }
                    }
                }
            }
        }
        
        if backup_files.is_empty() {
            bail!("No backup files found for restore");
        }
        
        // Sort by modification time (newest first)
        backup_files.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Try each backup file until one works
        for (backup_path, _) in backup_files {
            match self.restore_from_file(&backup_path) {
                Ok(()) => {
                    info!(
                        "Successfully restored {} signals from backup: {:?}",
                        self.buffer.len(),
                        backup_path
                    );
                    return Ok(());
                }
                Err(e) => {
                    warn!("Failed to restore from backup {:?}: {}", backup_path, e);
                    continue;
                }
            }
        }
        
        bail!("Failed to restore from any backup files");
    }
    
    /// Force persistence regardless of auto-persist settings
    pub fn force_persist(&mut self) -> Result<()> {
        if self.config.persistence.is_none() {
            bail!("Persistence not configured");
        }
        
        self.persist()
    }
    
    /// Check if persistence is configured and enabled
    pub fn is_persistence_enabled(&self) -> bool {
        self.config.persistence.is_some()
    }
    
    /// Get the persist file path if persistence is configured
    pub fn persist_file_path(&self) -> Option<PathBuf> {
        self.config.persistence.as_ref()
            .map(|config| config.persist_path.join("signal_buffer.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SignalComponents, FusionWeights, SignalSide};
    
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
            format!("test-correlation-{}", symbol),
            format!("checksum-{}", symbol),
            50,
        )
    }
    
    #[test]
    fn test_buffer_creation() {
        let config = BufferConfig {
            max_size: 100,
            overflow_strategy: OverflowStrategy::DropOldest,
            persistence: None,
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let buffer = SignalBuffer::new(config.clone());
        
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 100);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.utilization(), 0.0);
        assert_eq!(buffer.config().max_size, config.max_size);
    }
    
    #[test]
    fn test_buffer_push_pop() {
        let mut buffer = SignalBuffer::with_default_config();
        
        let signal1 = create_test_signal("BTCUSDT", 0.5, 0.8);
        let signal2 = create_test_signal("ETHUSDT", -0.3, 0.6);
        
        // Push signals
        assert!(buffer.push(signal1.clone()).is_ok());
        assert!(buffer.push(signal2.clone()).is_ok());
        
        assert_eq!(buffer.len(), 2);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.utilization(), 2.0 / 1000.0);
        
        // Pop signals (FIFO order)
        let popped1 = buffer.pop().unwrap();
        assert_eq!(popped1.signal.symbol, "BTCUSDT");
        
        let popped2 = buffer.pop().unwrap();
        assert_eq!(popped2.signal.symbol, "ETHUSDT");
        
        assert!(buffer.pop().is_none());
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_buffer_overflow_drop_oldest() {
        let config = BufferConfig {
            max_size: 2,
            overflow_strategy: OverflowStrategy::DropOldest,
            persistence: None,
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let mut buffer = SignalBuffer::new(config);
        
        let signal1 = create_test_signal("BTCUSDT", 0.5, 0.8);
        let signal2 = create_test_signal("ETHUSDT", -0.3, 0.6);
        let signal3 = create_test_signal("ADAUSDT", 0.7, 0.9);
        
        // Fill buffer
        assert!(buffer.push(signal1).is_ok());
        assert!(buffer.push(signal2).is_ok());
        assert_eq!(buffer.len(), 2);
        assert!(buffer.is_full());
        
        // Push third signal should drop oldest
        assert!(buffer.push(signal3).is_ok());
        assert_eq!(buffer.len(), 2);
        
        // First signal should be ETHUSDT (BTCUSDT was dropped)
        let popped = buffer.pop().unwrap();
        assert_eq!(popped.signal.symbol, "ETHUSDT");
        
        let popped = buffer.pop().unwrap();
        assert_eq!(popped.signal.symbol, "ADAUSDT");
        
        // Check metrics
        let metrics = buffer.metrics();
        assert_eq!(metrics.total_added, 3);
        assert_eq!(metrics.total_removed, 2);
        assert_eq!(metrics.total_dropped, 1);
        assert_eq!(metrics.overflow_events, 1);
    }
    
    #[test]
    fn test_buffer_overflow_drop_newest() {
        let config = BufferConfig {
            max_size: 2,
            overflow_strategy: OverflowStrategy::DropNewest,
            persistence: None,
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let mut buffer = SignalBuffer::new(config);
        
        let signal1 = create_test_signal("BTCUSDT", 0.5, 0.8);
        let signal2 = create_test_signal("ETHUSDT", -0.3, 0.6);
        let signal3 = create_test_signal("ADAUSDT", 0.7, 0.9);
        
        // Fill buffer
        assert!(buffer.push(signal1).is_ok());
        assert!(buffer.push(signal2).is_ok());
        assert_eq!(buffer.len(), 2);
        
        // Push third signal should be dropped
        assert!(buffer.push(signal3).is_ok());
        assert_eq!(buffer.len(), 2);
        
        // Should still have original signals
        let popped = buffer.pop().unwrap();
        assert_eq!(popped.signal.symbol, "BTCUSDT");
        
        let popped = buffer.pop().unwrap();
        assert_eq!(popped.signal.symbol, "ETHUSDT");
        
        // Check metrics
        let metrics = buffer.metrics();
        assert_eq!(metrics.total_dropped, 1);
    }
    
    #[test]
    fn test_buffer_overflow_error() {
        let config = BufferConfig {
            max_size: 1,
            overflow_strategy: OverflowStrategy::Error,
            persistence: None,
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let mut buffer = SignalBuffer::new(config);
        
        let signal1 = create_test_signal("BTCUSDT", 0.5, 0.8);
        let signal2 = create_test_signal("ETHUSDT", -0.3, 0.6);
        
        // First signal should succeed
        assert!(buffer.push(signal1).is_ok());
        assert_eq!(buffer.len(), 1);
        
        // Second signal should fail
        let result = buffer.push(signal2);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Buffer overflow"));
        assert_eq!(buffer.len(), 1);
    }
    
    #[test]
    fn test_buffer_overflow_drop_lowest_confidence() {
        let config = BufferConfig {
            max_size: 2,
            overflow_strategy: OverflowStrategy::DropLowestConfidence,
            persistence: None,
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let mut buffer = SignalBuffer::new(config);
        
        let signal1 = create_test_signal("BTCUSDT", 0.5, 0.8); // High confidence
        let signal2 = create_test_signal("ETHUSDT", -0.3, 0.4); // Low confidence
        let signal3 = create_test_signal("ADAUSDT", 0.7, 0.9); // Highest confidence
        
        // Fill buffer
        assert!(buffer.push(signal1).is_ok());
        assert!(buffer.push(signal2).is_ok());
        assert_eq!(buffer.len(), 2);
        
        // Push third signal should drop lowest confidence (ETHUSDT)
        assert!(buffer.push(signal3).is_ok());
        assert_eq!(buffer.len(), 2);
        
        // Should have BTCUSDT and ADAUSDT
        let signals: Vec<_> = buffer.buffer.iter().map(|s| s.signal.symbol.as_str()).collect();
        assert!(signals.contains(&"BTCUSDT"));
        assert!(signals.contains(&"ADAUSDT"));
        assert!(!signals.contains(&"ETHUSDT"));
    }
    
    #[test]
    fn test_buffer_priority_pop() {
        let mut buffer = SignalBuffer::with_default_config();
        
        let signal1 = create_test_signal("BTCUSDT", 0.5, 0.6); // Lower confidence
        let signal2 = create_test_signal("ETHUSDT", -0.3, 0.9); // Higher confidence
        let signal3 = create_test_signal("ADAUSDT", 0.7, 0.7); // Medium confidence
        
        assert!(buffer.push(signal1).is_ok());
        assert!(buffer.push(signal2).is_ok());
        assert!(buffer.push(signal3).is_ok());
        
        // Pop highest priority (should be ETHUSDT with confidence 0.9)
        let popped = buffer.pop_highest_priority().unwrap();
        assert_eq!(popped.signal.symbol, "ETHUSDT");
        assert_eq!(popped.signal.confidence, 0.9);
        
        // Next should be ADAUSDT
        let popped = buffer.pop_highest_priority().unwrap();
        assert_eq!(popped.signal.symbol, "ADAUSDT");
        
        // Last should be BTCUSDT
        let popped = buffer.pop_highest_priority().unwrap();
        assert_eq!(popped.signal.symbol, "BTCUSDT");
        
        assert!(buffer.pop_highest_priority().is_none());
    }
    
    #[test]
    fn test_buffer_peek() {
        let mut buffer = SignalBuffer::with_default_config();
        
        assert!(buffer.peek().is_none());
        
        let signal = create_test_signal("BTCUSDT", 0.5, 0.8);
        assert!(buffer.push(signal.clone()).is_ok());
        
        // Peek should return the signal without removing it
        let peeked = buffer.peek().unwrap();
        assert_eq!(peeked.signal.symbol, "BTCUSDT");
        assert_eq!(buffer.len(), 1);
        
        // Pop should return the same signal
        let popped = buffer.pop().unwrap();
        assert_eq!(popped.signal.symbol, "BTCUSDT");
        assert_eq!(buffer.len(), 0);
    }
    
    #[test]
    fn test_buffer_clear() {
        let mut buffer = SignalBuffer::with_default_config();
        
        let signal1 = create_test_signal("BTCUSDT", 0.5, 0.8);
        let signal2 = create_test_signal("ETHUSDT", -0.3, 0.6);
        
        assert!(buffer.push(signal1).is_ok());
        assert!(buffer.push(signal2).is_ok());
        assert_eq!(buffer.len(), 2);
        
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_buffer_metrics() {
        let mut buffer = SignalBuffer::with_default_config();
        
        let signal1 = create_test_signal("BTCUSDT", 0.5, 0.8);
        let signal2 = create_test_signal("ETHUSDT", -0.3, 0.6);
        
        assert!(buffer.push(signal1).is_ok());
        assert!(buffer.push(signal2).is_ok());
        
        let metrics = buffer.metrics();
        assert_eq!(metrics.current_size, 2);
        assert_eq!(metrics.max_size, 1000);
        assert_eq!(metrics.total_added, 2);
        assert_eq!(metrics.total_removed, 0);
        assert_eq!(metrics.total_dropped, 0);
        assert_eq!(metrics.overflow_events, 0);
        assert!(metrics.avg_age_seconds >= 0.0);
        
        // Pop one signal
        buffer.pop();
        
        let metrics = buffer.metrics();
        assert_eq!(metrics.current_size, 1);
        assert_eq!(metrics.total_removed, 1);
    }
    
    #[test]
    fn test_buffered_signal() {
        let signal = create_test_signal("BTCUSDT", 0.5, 0.8);
        let buffered = BufferedSignal::new(signal.clone());
        
        assert_eq!(buffered.signal.symbol, "BTCUSDT");
        assert_eq!(buffered.retry_count, 0);
        assert_eq!(buffered.priority, 0.8); // Should match confidence
        assert!(buffered.age_seconds() >= 0);
        
        // Test with custom priority
        let buffered_custom = BufferedSignal::with_priority(signal, 0.95);
        assert_eq!(buffered_custom.priority, 0.95);
    }
    
    #[test]
    fn test_buffer_config_update() {
        let mut buffer = SignalBuffer::with_default_config();
        
        // Add some signals
        for i in 0..5 {
            let signal = create_test_signal(&format!("SYMBOL{}", i), 0.5, 0.8);
            assert!(buffer.push(signal).is_ok());
        }
        assert_eq!(buffer.len(), 5);
        
        // Reduce buffer size
        let new_config = BufferConfig {
            max_size: 3,
            overflow_strategy: OverflowStrategy::DropOldest,
            persistence: None,
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        assert!(buffer.update_config(new_config).is_ok());
        assert_eq!(buffer.len(), 3); // Should have dropped 2 oldest signals
        assert_eq!(buffer.capacity(), 3);
        
        // Check that oldest signals were dropped
        let remaining_symbols: Vec<_> = buffer.buffer
            .iter()
            .map(|s| s.signal.symbol.as_str())
            .collect();
        assert!(remaining_symbols.contains(&"SYMBOL2"));
        assert!(remaining_symbols.contains(&"SYMBOL3"));
        assert!(remaining_symbols.contains(&"SYMBOL4"));
    }
    
    #[test]
    fn test_buffer_persistence_disabled() {
        let mut buffer = SignalBuffer::with_default_config();
        
        // Persistence should be disabled by default
        assert!(!buffer.is_persistence_enabled());
        assert!(buffer.persist_file_path().is_none());
        
        // Persist should succeed but do nothing
        assert!(buffer.persist().is_ok());
        
        // Restore should succeed but do nothing
        assert!(buffer.restore().is_ok());
        
        // Force persist should fail
        assert!(buffer.force_persist().is_err());
    }
    
    #[test]
    fn test_buffer_persistence_enabled() {
        use std::env;
        
        let temp_dir = env::temp_dir().join("signal_buffer_test");
        
        let config = BufferConfig {
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
        
        let mut buffer = SignalBuffer::new(config);
        
        // Persistence should be enabled
        assert!(buffer.is_persistence_enabled());
        assert!(buffer.persist_file_path().is_some());
        
        // Add some signals
        let signal1 = create_test_signal("BTCUSDT", 0.5, 0.8);
        let signal2 = create_test_signal("ETHUSDT", -0.3, 0.6);
        
        assert!(buffer.push(signal1).is_ok());
        assert!(buffer.push(signal2).is_ok());
        assert_eq!(buffer.len(), 2);
        
        // Persist should work
        assert!(buffer.persist().is_ok());
        
        // Check that file was created
        let persist_file = temp_dir.join("signal_buffer.json");
        assert!(persist_file.exists());
        
        // Clear buffer and restore
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        
        assert!(buffer.restore().is_ok());
        assert_eq!(buffer.len(), 2);
        
        // Check that signals were restored correctly
        let symbols: Vec<_> = buffer.buffer
            .iter()
            .map(|s| s.signal.symbol.as_str())
            .collect();
        assert!(symbols.contains(&"BTCUSDT"));
        assert!(symbols.contains(&"ETHUSDT"));
        
        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
    
    #[test]
    fn test_buffer_persistence_atomic_operations() {
        use std::env;
        
        let temp_dir = env::temp_dir().join("signal_buffer_atomic_test");
        
        let config = BufferConfig {
            max_size: 100,
            overflow_strategy: OverflowStrategy::DropOldest,
            persistence: Some(PersistenceConfig {
                persist_path: temp_dir.clone(),
                auto_persist: false,
                persist_interval_sec: 60,
                max_backup_files: 2,
                atomic_operations: true,
            }),
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let mut buffer = SignalBuffer::new(config);
        
        // Add a signal
        let signal = create_test_signal("BTCUSDT", 0.5, 0.8);
        assert!(buffer.push(signal).is_ok());
        
        // Persist should work atomically
        assert!(buffer.persist().is_ok());
        
        // Check that main file exists but temp file doesn't
        let persist_file = temp_dir.join("signal_buffer.json");
        let temp_file = temp_dir.join("signal_buffer.tmp");
        
        assert!(persist_file.exists());
        assert!(!temp_file.exists());
        
        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
    
    #[test]
    fn test_buffer_persistence_non_atomic() {
        use std::env;
        
        let temp_dir = env::temp_dir().join("signal_buffer_non_atomic_test");
        
        let config = BufferConfig {
            max_size: 100,
            overflow_strategy: OverflowStrategy::DropOldest,
            persistence: Some(PersistenceConfig {
                persist_path: temp_dir.clone(),
                auto_persist: false,
                persist_interval_sec: 60,
                max_backup_files: 2,
                atomic_operations: false,
            }),
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let mut buffer = SignalBuffer::new(config);
        
        // Add a signal
        let signal = create_test_signal("BTCUSDT", 0.5, 0.8);
        assert!(buffer.push(signal).is_ok());
        
        // Persist should work
        assert!(buffer.persist().is_ok());
        
        // Check that file exists
        let persist_file = temp_dir.join("signal_buffer.json");
        assert!(persist_file.exists());
        
        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
    
    #[test]
    fn test_buffer_restore_empty_file() {
        use std::env;
        
        let temp_dir = env::temp_dir().join("signal_buffer_empty_test");
        
        let config = BufferConfig {
            max_size: 100,
            overflow_strategy: OverflowStrategy::DropOldest,
            persistence: Some(PersistenceConfig {
                persist_path: temp_dir.clone(),
                auto_persist: false,
                persist_interval_sec: 60,
                max_backup_files: 2,
                atomic_operations: true,
            }),
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let mut buffer = SignalBuffer::new(config);
        
        // Restore from non-existent file should succeed
        assert!(buffer.restore().is_ok());
        assert_eq!(buffer.len(), 0);
        
        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
    
    #[test]
    fn test_buffer_restore_with_size_limits() {
        use std::env;
        
        let temp_dir = env::temp_dir().join("signal_buffer_size_limit_test");
        
        // Create buffer with larger size first
        let large_config = BufferConfig {
            max_size: 5,
            overflow_strategy: OverflowStrategy::DropOldest,
            persistence: Some(PersistenceConfig {
                persist_path: temp_dir.clone(),
                auto_persist: false,
                persist_interval_sec: 60,
                max_backup_files: 2,
                atomic_operations: true,
            }),
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let mut buffer = SignalBuffer::new(large_config);
        
        // Add 5 signals
        for i in 0..5 {
            let signal = create_test_signal(&format!("SYMBOL{}", i), 0.5, 0.8);
            assert!(buffer.push(signal).is_ok());
        }
        assert_eq!(buffer.len(), 5);
        
        // Persist
        assert!(buffer.persist().is_ok());
        
        // Create new buffer with smaller size
        let small_config = BufferConfig {
            max_size: 3,
            overflow_strategy: OverflowStrategy::DropOldest,
            persistence: Some(PersistenceConfig {
                persist_path: temp_dir.clone(),
                auto_persist: false,
                persist_interval_sec: 60,
                max_backup_files: 2,
                atomic_operations: true,
            }),
            enable_metrics: true,
            warning_threshold: 0.8,
        };
        
        let mut small_buffer = SignalBuffer::new(small_config);
        
        // Restore should limit to 3 signals
        assert!(small_buffer.restore().is_ok());
        assert_eq!(small_buffer.len(), 3);
        
        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}