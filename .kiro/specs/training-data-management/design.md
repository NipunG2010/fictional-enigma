# Training Data Management Design Document

## Overview

The Training Data Management system provides a comprehensive CLI tool for creating, validating, and managing labeled training datasets for the LDC trading system. It integrates with the existing Rust workspace architecture, leveraging the feature-pipeline and ldc-engine components to generate high-quality training snapshots with configurable future return labels and comprehensive data quality validation.

## Architecture

### System Integration

The training data management system integrates with the existing hybrid Rust+Python architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Market    │    │  Feature Pipeline │    │ Training Data   │
│   Data (OHLCV)  │───▶│   (Rust/Polars)  │───▶│  Management     │
└─────────────────┘    └──────────────────┘    │   CLI Tool      │
                                               └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LDC Engine    │◀───│  Labeled Training│    │  Data Quality   │
│   (k-NN/Ring    │    │   Snapshots      │    │   Validation    │
│    Buffer)      │    │  (Parquet/JSON)  │    │   & Reports     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Architecture

```
training-data-cli/
├── src/
│   ├── main.rs              # CLI entry point and argument parsing
│   ├── lib.rs               # Library exports
│   ├── snapshot/            # Training snapshot creation
│   │   ├── mod.rs
│   │   ├── builder.rs       # Snapshot builder with configuration
│   │   └── labeler.rs       # Future returns label generation
│   ├── validation/          # Data quality validation
│   │   ├── mod.rs
│   │   ├── quality.rs       # Data quality checks and metrics
│   │   └── report.rs        # Validation report generation
│   ├── config/              # Configuration management
│   │   ├── mod.rs
│   │   └── settings.rs      # Configuration serialization/loading
│   └── utils/               # Utility functions
│       ├── mod.rs
│       └── progress.rs      # Progress indicators and logging
└── Cargo.toml
```

## Components and Interfaces

### 1. CLI Interface

**Primary Command Structure:**
```bash
# Create training snapshot
training-data create \
  --input data/btc_5m.parquet \
  --output snapshots/btc_training_v1.parquet \
  --horizon 12 \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --config configs/default.json

# Validate existing data
training-data validate \
  --input data/btc_5m.parquet \
  --report validation_report.json

# List and manage configurations
training-data config list
training-data config save --name "btc_12h_horizon" --file config.json
```

**CLI Arguments Interface:**
```rust
#[derive(Parser)]
#[command(name = "training-data")]
#[command(about = "Training data management for LDC trading system")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Create(CreateArgs),
    Validate(ValidateArgs),
    Config(ConfigArgs),
}

#[derive(Args)]
pub struct CreateArgs {
    #[arg(short, long)]
    pub input: PathBuf,
    #[arg(short, long)]
    pub output: PathBuf,
    #[arg(short = 'H', long, default_value = "12")]
    pub horizon: usize,
    #[arg(long)]
    pub start_date: Option<String>,
    #[arg(long)]
    pub end_date: Option<String>,
    #[arg(short, long)]
    pub config: Option<PathBuf>,
    #[arg(long, default_value = "parquet")]
    pub format: OutputFormat,
}
```

### 2. Snapshot Builder

**Core Interface:**
```rust
pub struct SnapshotBuilder {
    config: SnapshotConfig,
    feature_pipeline: FeaturePipeline,
    labeler: FutureReturnsLabeler,
    validator: DataValidator,
}

impl SnapshotBuilder {
    pub fn new(config: SnapshotConfig) -> Result<Self>;
    pub fn create_snapshot(&self, input_path: &Path, output_path: &Path) -> Result<SnapshotMetadata>;
    pub fn validate_input(&self, data: &DataFrame) -> Result<ValidationReport>;
}

pub struct SnapshotConfig {
    pub horizon: usize,
    pub features: Vec<FeatureType>,
    pub label_thresholds: LabelThresholds,
    pub validation_strictness: ValidationLevel,
    pub date_range: Option<DateRange>,
}
```

### 3. Future Returns Labeler

**Label Generation Interface:**
```rust
pub struct FutureReturnsLabeler {
    horizon: usize,
    thresholds: LabelThresholds,
}

impl FutureReturnsLabeler {
    pub fn generate_labels(&self, prices: &[f64]) -> Result<Vec<Label>>;
    pub fn calculate_returns(&self, prices: &[f64]) -> Vec<Option<f64>>;
    pub fn classify_returns(&self, returns: &[f64]) -> Vec<Label>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Label {
    Buy,    // Return > buy_threshold
    Sell,   // Return < sell_threshold  
    Hold,   // sell_threshold <= Return <= buy_threshold
}

#[derive(Debug, Clone)]
pub struct LabelThresholds {
    pub buy_threshold: f64,   // e.g., 0.02 (2%)
    pub sell_threshold: f64,  // e.g., -0.02 (-2%)
}
```

### 4. Data Quality Validator

**Validation Interface:**
```rust
pub struct DataValidator {
    config: ValidationConfig,
}

impl DataValidator {
    pub fn validate(&self, data: &DataFrame) -> Result<ValidationReport>;
    pub fn check_missing_values(&self, data: &DataFrame) -> MissingValueReport;
    pub fn detect_outliers(&self, data: &DataFrame) -> OutlierReport;
    pub fn validate_timestamps(&self, data: &DataFrame) -> TimestampReport;
    pub fn check_duplicates(&self, data: &DataFrame) -> DuplicateReport;
}

pub struct ValidationReport {
    pub overall_status: ValidationStatus,
    pub missing_values: MissingValueReport,
    pub outliers: OutlierReport,
    pub timestamps: TimestampReport,
    pub duplicates: DuplicateReport,
    pub statistics: DataStatistics,
}
```

### 5. Configuration Management

**Configuration Interface:**
```rust
pub struct ConfigManager {
    config_dir: PathBuf,
}

impl ConfigManager {
    pub fn save_config(&self, name: &str, config: &SnapshotConfig) -> Result<()>;
    pub fn load_config(&self, name: &str) -> Result<SnapshotConfig>;
    pub fn list_configs(&self) -> Result<Vec<ConfigInfo>>;
    pub fn delete_config(&self, name: &str) -> Result<()>;
}

#[derive(Serialize, Deserialize)]
pub struct SavedConfig {
    pub name: String,
    pub config: SnapshotConfig,
    pub created_at: DateTime<Utc>,
    pub version: String,
    pub description: Option<String>,
}
```

## Data Models

### Training Snapshot Schema

**Parquet Output Schema:**
```
timestamp: DateTime<Utc>           # Bar timestamp
open: f64                          # OHLCV data
high: f64
low: f64  
close: f64
volume: f64

# Technical indicators (from feature-pipeline)
rsi_14: f64
sma_20: f64
ema_12: f64
ema_26: f64
macd: f64
macd_signal: f64
bb_upper: f64
bb_middle: f64
bb_lower: f64
atr_14: f64

# Labels
future_return: Option<f64>         # (close[t+h] - close[t]) / close[t]
label: Option<Label>               # Buy/Sell/Hold classification
label_confidence: Option<f64>      # Distance from threshold
```

**Metadata JSON Schema:**
```json
{
  "snapshot_id": "btc_5m_h12_20240101_20241231",
  "created_at": "2024-01-15T10:30:00Z",
  "config": {
    "horizon": 12,
    "features": ["rsi_14", "sma_20", "ema_12", "ema_26", "macd", "bb_upper", "bb_lower", "atr_14"],
    "label_thresholds": {
      "buy_threshold": 0.02,
      "sell_threshold": -0.02
    }
  },
  "data_info": {
    "symbol": "BTCUSDT",
    "interval": "5m",
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-12-31T23:55:00Z",
    "total_bars": 105120,
    "labeled_bars": 104000
  },
  "label_distribution": {
    "buy": 34567,
    "hold": 34866,
    "sell": 34567
  },
  "validation_summary": {
    "status": "passed",
    "warnings": 2,
    "errors": 0
  }
}
```

## Error Handling

### Error Types and Recovery

```rust
#[derive(thiserror::Error, Debug)]
pub enum TrainingDataError {
    #[error("Data validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Insufficient data for horizon {horizon}: need {needed} bars, got {available}")]
    InsufficientData { horizon: usize, needed: usize, available: usize },
    
    #[error("Invalid date range: start {start} is after end {end}")]
    InvalidDateRange { start: String, end: String },
    
    #[error("Feature computation failed: {0}")]
    FeatureError(#[from] feature_pipeline::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}
```

### Graceful Degradation Strategy

1. **Missing Data Handling**: Skip bars with insufficient future data rather than failing
2. **Feature Computation Errors**: Continue with available features, log warnings
3. **Validation Failures**: Provide detailed reports but allow user override with warnings
4. **Configuration Issues**: Fall back to sensible defaults with user notification

## Testing Strategy

### Unit Tests

1. **Label Generation Tests**
   - Test future returns calculation accuracy
   - Verify threshold-based classification
   - Test edge cases (insufficient future data)

2. **Validation Tests**
   - Test missing value detection
   - Verify outlier detection algorithms
   - Test timestamp validation logic

3. **Configuration Tests**
   - Test config serialization/deserialization
   - Verify config validation logic
   - Test default value handling

### Integration Tests

1. **End-to-End Pipeline Tests**
   - Test complete snapshot creation workflow
   - Verify output format correctness
   - Test with real market data samples

2. **CLI Interface Tests**
   - Test all command-line argument combinations
   - Verify error message clarity
   - Test progress reporting

### Performance Tests

1. **Scalability Tests**
   - Test with large datasets (1M+ bars)
   - Measure memory usage patterns
   - Verify processing time targets

2. **Validation Performance**
   - Benchmark validation algorithms
   - Test parallel processing efficiency

## Implementation Considerations

### Performance Optimizations

1. **Lazy Evaluation**: Use Polars lazy frames for memory efficiency
2. **Parallel Processing**: Leverage rayon for CPU-intensive operations
3. **Streaming**: Process data in chunks for large datasets
4. **Caching**: Cache computed features to avoid recomputation

### Memory Management

1. **Chunked Processing**: Process data in configurable chunk sizes
2. **Feature Streaming**: Compute features on-demand rather than storing all in memory
3. **Garbage Collection**: Explicit cleanup of intermediate DataFrames

### Integration Points

1. **Feature Pipeline Integration**: Reuse existing feature computation logic
2. **LDC Engine Compatibility**: Ensure output format matches LDC engine expectations
3. **Storage Integration**: Support existing Parquet partitioning scheme
4. **Configuration Consistency**: Align with existing workspace configuration patterns