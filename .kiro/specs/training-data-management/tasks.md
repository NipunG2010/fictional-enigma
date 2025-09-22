# Implementation Plan

- [x] 1. Set up training-data-cli crate structure and dependencies
  - Create new Rust crate `training-data-cli` in the workspace
  - Add dependencies: clap, serde, chrono, polars, anyhow, thiserror
  - Configure workspace integration and feature-pipeline dependency
  - _Requirements: 1.1, 4.1_

- [x] 2. Implement core CLI interface and argument parsing
  - Create main.rs with clap-based CLI structure (Commands enum, Args structs)
  - Implement subcommands: create, validate, config with proper argument validation
  - Add help documentation and usage examples for all commands
  - _Requirements: 1.1, 4.1, 4.4_

- [x] 3. Create configuration management system
  - Implement SnapshotConfig struct with serialization support
  - Build ConfigManager for saving/loading configurations to JSON files
  - Add configuration validation and default value handling
  - Write unit tests for config serialization and validation
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 4. Implement future returns labeler
  - Create FutureReturnsLabeler struct with horizon and threshold configuration
  - Implement calculate_returns method using (close[t+h] - close[t]) / close[t] formula
  - Build classify_returns method for Buy/Sell/Hold label generation based on thresholds
  - Add label distribution validation to ensure reasonable class balance
  - Write comprehensive unit tests for label generation edge cases
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 5. Build data quality validation framework
  - Implement DataValidator struct with configurable validation rules
  - Create missing value detection and reporting functionality
  - Build outlier detection using statistical methods (IQR, z-score)
  - Implement timestamp validation for sequential and complete time series
  - Add duplicate detection and removal with logging
  - Write unit tests for each validation component
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 6. Create validation report generation system
  - Implement ValidationReport struct with comprehensive status tracking
  - Build report generation with statistics, warnings, and error details
  - Add JSON serialization for validation reports
  - Create human-readable report formatting for CLI output
  - _Requirements: 3.5, 3.6_

- [x] 7. Implement snapshot builder core functionality
  - Create SnapshotBuilder struct that orchestrates the entire process
  - Integrate with feature-pipeline for technical indicator computation
  - Implement data loading from Parquet files using Polars
  - Add date range filtering and data preprocessing
  - Build progress tracking and logging for long-running operations
  - _Requirements: 1.1, 1.2, 1.3, 4.5_

- [x] 8. Add snapshot creation and output functionality
  - Implement create_snapshot method that combines features, labels, and validation
  - Build Parquet output with proper schema including OHLCV, features, and labels
  - Create metadata JSON generation with snapshot info and statistics
  - Add support for multiple output formats (Parquet, CSV, JSON)
  - Implement proper error handling and cleanup on failures
  - _Requirements: 1.1, 1.3, 1.4, 1.5_

- [x] 9. Implement validate subcommand functionality
  - Create standalone validation workflow that doesn't require label generation
  - Build comprehensive data quality analysis and reporting
  - Add validation report output in both JSON and human-readable formats
  - Implement configurable validation strictness levels
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 10. Add progress indicators and user experience improvements
  - Implement progress bars for long-running operations using indicatif
  - Add detailed logging with different verbosity levels
  - Create informative error messages with suggestions for common issues
  - Build summary statistics display after successful snapshot creation
  - _Requirements: 4.5, 1.5_

- [x] 11. Create comprehensive integration tests
  - Build end-to-end tests using sample market data
  - Test complete workflow from raw OHLCV to labeled training snapshot
  - Verify output format compatibility with LDC engine expectations
  - Test error scenarios and recovery mechanisms
  - Create performance benchmarks for large dataset processing
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 3.1, 3.2_
