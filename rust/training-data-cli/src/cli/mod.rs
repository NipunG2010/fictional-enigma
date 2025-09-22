use clap::{Parser, Subcommand, Args};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "training-data")]
#[command(about = "Training data management for LDC trading system")]
#[command(version = "0.1.0")]
#[command(long_about = r#"
Training Data Management CLI for LDC Trading System

This tool helps create, validate, and manage labeled training datasets from historical 
market data. It integrates with the feature pipeline to generate technical indicators 
and creates future return labels for machine learning model training.

EXAMPLES:
    # Create a training snapshot with 12-period horizon
    training-data create -i data/btc_5m.parquet -o snapshots/btc_training.parquet -H 12

    # Validate data quality
    training-data validate -i data/btc_5m.parquet -r validation_report.json

    # Save current configuration
    training-data config save -n "btc_12h" -f config.json

For more information about each command, use: training-data <command> --help
"#)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Create a training snapshot from market data
    #[command(long_about = r#"
Create a labeled training snapshot from historical market data.

This command processes OHLCV data through the feature pipeline to generate technical 
indicators, calculates future returns based on the specified horizon, and creates 
categorical labels (Buy/Sell/Hold) for machine learning training.

The output includes:
- Original OHLCV data
- Technical indicators (RSI, SMA, EMA, MACD, Bollinger Bands, ATR)
- Future return values and categorical labels
- Metadata with data statistics and configuration

EXAMPLES:
    # Basic snapshot creation
    training-data create -i data/btc_5m.parquet -o training/btc_snapshot.parquet

    # With custom horizon and date range
    training-data create -i data/eth_1h.parquet -o training/eth_24h.parquet \
        --horizon 24 --start-date 2023-01-01 --end-date 2023-12-31

    # Using saved configuration
    training-data create -i data/btc_5m.parquet -o training/btc.parquet \
        --config configs/btc_config.json
"#)]
    Create(CreateArgs),
    
    /// Validate existing market data quality
    #[command(long_about = r#"
Validate the quality of market data without creating a training snapshot.

This command performs comprehensive data quality checks including:
- Missing value detection and reporting
- Outlier detection using statistical methods
- Timestamp validation for sequential completeness
- Duplicate detection and flagging
- Data distribution analysis

The validation report can be output in JSON format for programmatic use or 
human-readable format for manual review.

EXAMPLES:
    # Basic validation with console output
    training-data validate -i data/btc_5m.parquet

    # Generate detailed JSON report
    training-data validate -i data/eth_1h.parquet -r validation_report.json

    # Strict validation mode
    training-data validate -i data/btc_5m.parquet --strictness strict -v
"#)]
    Validate(ValidateArgs),
    
    /// Manage training configurations
    #[command(long_about = r#"
Manage saved training configurations for reproducible experiments.

Configurations store all parameters needed to recreate training snapshots, including:
- Feature selection and parameters
- Label generation settings (horizon, thresholds)
- Validation rules and strictness levels
- Output format preferences

EXAMPLES:
    # List all saved configurations
    training-data config list

    # Save current settings as a named configuration
    training-data config save -n "btc_12h_horizon" -f current_config.json

    # Load a previously saved configuration
    training-data config load btc_12h_horizon

    # Delete an old configuration
    training-data config delete old_experiment
"#)]
    Config(ConfigArgs),
}

#[derive(Args, Debug)]
pub struct CreateArgs {
    /// Input market data file (Parquet format)
    /// 
    /// The input file should contain OHLCV data with columns: timestamp, open, high, low, close, volume.
    /// Supported format: Parquet files with proper timestamp indexing.
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,
    
    /// Output path for training snapshot
    /// 
    /// The generated training snapshot will include OHLCV data, technical indicators, 
    /// future returns, and categorical labels. Directory will be created if it doesn't exist.
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,
    
    /// Future return horizon in periods
    /// 
    /// Number of periods ahead to calculate future returns. For 5-minute data, 
    /// horizon=12 means 1-hour ahead returns. Must be positive and reasonable 
    /// relative to dataset size.
    #[arg(short = 'H', long, default_value = "12", value_name = "PERIODS")]
    pub horizon: usize,
    
    /// Start date for data filtering (YYYY-MM-DD format)
    /// 
    /// Only process data from this date onwards. If not specified, uses all available data.
    /// Must be in ISO date format (YYYY-MM-DD).
    #[arg(long, value_name = "DATE")]
    pub start_date: Option<String>,
    
    /// End date for data filtering (YYYY-MM-DD format)
    /// 
    /// Only process data up to this date. If not specified, uses all available data.
    /// Must be in ISO date format (YYYY-MM-DD) and after start-date.
    #[arg(long, value_name = "DATE")]
    pub end_date: Option<String>,
    
    /// Configuration file path
    /// 
    /// JSON file containing feature selection, label thresholds, and validation settings.
    /// If not provided, uses default configuration.
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    
    /// Output format for the training snapshot
    /// 
    /// Parquet is recommended for large datasets and ML workflows.
    /// CSV is useful for manual inspection. JSON is for small datasets only.
    #[arg(long, default_value = "parquet", value_name = "FORMAT")]
    pub format: OutputFormat,
    
    /// Buy threshold for label classification
    /// 
    /// Future returns above this threshold are labeled as 'Buy'. 
    /// Should be positive (e.g., 0.02 for 2%).
    #[arg(long, value_name = "PERCENT")]
    pub buy_threshold: Option<f64>,
    
    /// Sell threshold for label classification
    /// 
    /// Future returns below this threshold are labeled as 'Sell'.
    /// Should be negative (e.g., -0.02 for -2%).
    #[arg(long, value_name = "PERCENT")]
    pub sell_threshold: Option<f64>,
    
    /// Enable verbose output with detailed progress information
    #[arg(short, long)]
    pub verbose: bool,
    
    /// Skip data validation (not recommended)
    /// 
    /// Bypasses quality checks for faster processing. Use only with trusted, 
    /// pre-validated data sources.
    #[arg(long)]
    pub skip_validation: bool,
}

#[derive(Args, Debug, Clone)]
pub struct ValidateArgs {
    /// Input market data file to validate
    /// 
    /// Parquet file containing OHLCV market data. The file should have proper 
    /// timestamp indexing and standard OHLCV column names.
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,
    
    /// Output path for validation report
    /// 
    /// If specified, saves detailed validation report in JSON format.
    /// If not provided, displays summary on console only.
    #[arg(short, long, value_name = "FILE")]
    pub report: Option<PathBuf>,
    
    /// Validation strictness level
    /// 
    /// - strict: Fails on any data quality issues
    /// - normal: Warns on issues but continues processing  
    /// - lenient: Only fails on critical errors
    #[arg(long, default_value = "normal", value_name = "LEVEL")]
    pub strictness: ValidationLevel,
    
    /// Check for specific data quality issues
    /// 
    /// Comma-separated list of checks to perform: missing, outliers, duplicates, timestamps.
    /// If not specified, runs all available checks.
    #[arg(long, value_delimiter = ',', value_name = "CHECKS")]
    pub checks: Option<Vec<String>>,
    
    /// Maximum acceptable missing value percentage
    /// 
    /// Validation fails if missing values exceed this threshold (0.0-100.0).
    #[arg(long, value_name = "PERCENT")]
    pub max_missing_percent: Option<f64>,
    
    /// Enable verbose output with detailed statistics
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Args, Debug)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub action: ConfigAction,
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// List all saved configurations with details
    /// 
    /// Shows configuration names, creation dates, and brief descriptions.
    /// Use --verbose for full configuration details.
    List {
        /// Show full configuration details
        #[arg(short, long)]
        verbose: bool,
    },
    
    /// Save a new configuration from file or current settings
    /// 
    /// Saves configuration parameters for later reuse. Configurations include
    /// feature selection, label thresholds, validation rules, and output preferences.
    Save {
        /// Unique name for the configuration
        /// 
        /// Use descriptive names like 'btc_12h_horizon' or 'eth_strict_validation'.
        /// Names must be unique and contain only alphanumeric characters, hyphens, and underscores.
        #[arg(short, long, value_name = "NAME")]
        name: String,
        
        /// Configuration file to save from
        /// 
        /// JSON file containing configuration parameters. If not provided,
        /// creates a template configuration file.
        #[arg(short, long, value_name = "FILE")]
        file: Option<PathBuf>,
        
        /// Optional description for the configuration
        #[arg(short, long, value_name = "TEXT")]
        description: Option<String>,
    },
    
    /// Load a previously saved configuration
    /// 
    /// Loads configuration parameters and displays them. Use the output
    /// with other commands by saving to a file first.
    Load {
        /// Configuration name to load
        /// 
        /// Use 'config list' to see available configuration names.
        #[arg(value_name = "NAME")]
        name: String,
        
        /// Output file for loaded configuration
        /// 
        /// If specified, saves the loaded configuration to this file
        /// for use with create/validate commands.
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    
    /// Delete a saved configuration
    /// 
    /// Permanently removes a saved configuration. This action cannot be undone.
    Delete {
        /// Configuration name to delete
        /// 
        /// Use 'config list' to see available configuration names.
        #[arg(value_name = "NAME")]
        name: String,
        
        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },
    
    /// Show a template configuration file
    /// 
    /// Displays a complete configuration template with all available options
    /// and their descriptions. Useful for creating custom configurations.
    Template {
        /// Output file for template (if not specified, prints to stdout)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    /// Apache Parquet format (recommended for large datasets and ML workflows)
    Parquet,
    /// Comma-separated values format (good for manual inspection and Excel)
    Csv,
    /// JSON format (suitable for small datasets and debugging)
    Json,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum ValidationLevel {
    /// Strict validation - fails on any data quality issues
    Strict,
    /// Normal validation - warns on issues but continues processing
    Normal,
    /// Lenient validation - only fails on critical errors that prevent processing
    Lenient,
}

impl CreateArgs {
    /// Validate the create command arguments
    pub fn validate(&self) -> Result<(), String> {
        // Validate horizon is reasonable
        if self.horizon == 0 {
            return Err("Horizon must be greater than 0".to_string());
        }
        if self.horizon > 1000 {
            return Err("Horizon is too large (max 1000 periods)".to_string());
        }

        // Validate thresholds if provided
        if let Some(buy_threshold) = self.buy_threshold {
            if buy_threshold <= 0.0 {
                return Err("Buy threshold must be positive".to_string());
            }
            if buy_threshold > 1.0 {
                return Err("Buy threshold should be reasonable (typically < 1.0 or 100%)".to_string());
            }
        }

        if let Some(sell_threshold) = self.sell_threshold {
            if sell_threshold >= 0.0 {
                return Err("Sell threshold must be negative".to_string());
            }
            if sell_threshold < -1.0 {
                return Err("Sell threshold should be reasonable (typically > -1.0 or -100%)".to_string());
            }
        }

        // Validate that buy threshold > absolute value of sell threshold if both provided
        if let (Some(buy), Some(sell)) = (self.buy_threshold, self.sell_threshold) {
            if buy <= sell.abs() {
                return Err("Buy threshold must be greater than absolute value of sell threshold".to_string());
            }
        }

        // Validate date format if provided
        if let Some(ref start_date) = self.start_date {
            if !is_valid_date_format(start_date) {
                return Err("Start date must be in YYYY-MM-DD format".to_string());
            }
        }

        if let Some(ref end_date) = self.end_date {
            if !is_valid_date_format(end_date) {
                return Err("End date must be in YYYY-MM-DD format".to_string());
            }
        }

        // Validate date range if both provided
        if let (Some(ref start), Some(ref end)) = (&self.start_date, &self.end_date) {
            if start >= end {
                return Err("Start date must be before end date".to_string());
            }
        }

        // Validate input file extension
        if let Some(ext) = self.input.extension() {
            if ext != "parquet" {
                return Err("Input file must be in Parquet format (.parquet extension)".to_string());
            }
        } else {
            return Err("Input file must have .parquet extension".to_string());
        }

        Ok(())
    }
}

impl ValidateArgs {
    /// Validate the validate command arguments
    pub fn validate(&self) -> Result<(), String> {
        // Validate input file extension
        if let Some(ext) = self.input.extension() {
            if ext != "parquet" {
                return Err("Input file must be in Parquet format (.parquet extension)".to_string());
            }
        } else {
            return Err("Input file must have .parquet extension".to_string());
        }

        // Validate max missing percent if provided
        if let Some(max_missing) = self.max_missing_percent {
            if max_missing < 0.0 || max_missing > 100.0 {
                return Err("Max missing percentage must be between 0.0 and 100.0".to_string());
            }
        }

        // Validate check names if provided
        if let Some(ref checks) = self.checks {
            let valid_checks = ["missing", "outliers", "duplicates", "timestamps"];
            for check in checks {
                if !valid_checks.contains(&check.as_str()) {
                    return Err(format!(
                        "Invalid check '{}'. Valid checks are: {}",
                        check,
                        valid_checks.join(", ")
                    ));
                }
            }
        }

        Ok(())
    }
}

impl ConfigArgs {
    /// Validate the config command arguments
    pub fn validate(&self) -> Result<(), String> {
        match &self.action {
            ConfigAction::Save { name, .. } => {
                if name.is_empty() {
                    return Err("Configuration name cannot be empty".to_string());
                }
                
                // Validate name contains only allowed characters
                if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
                    return Err("Configuration name can only contain alphanumeric characters, hyphens, and underscores".to_string());
                }
                
                if name.len() > 50 {
                    return Err("Configuration name must be 50 characters or less".to_string());
                }
            }
            ConfigAction::Load { name, .. } | ConfigAction::Delete { name, .. } => {
                if name.is_empty() {
                    return Err("Configuration name cannot be empty".to_string());
                }
            }
            _ => {} // List and Template don't need validation
        }
        
        Ok(())
    }
}

/// Helper function to validate date format (YYYY-MM-DD)
fn is_valid_date_format(date_str: &str) -> bool {
    // Basic regex-like validation for YYYY-MM-DD format
    if date_str.len() != 10 {
        return false;
    }
    
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() != 3 {
        return false;
    }
    
    // Check year (4 digits)
    if parts[0].len() != 4 || !parts[0].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    
    // Check month (2 digits, 01-12)
    if parts[1].len() != 2 || !parts[1].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let month: u32 = parts[1].parse().unwrap_or(0);
    if month < 1 || month > 12 {
        return false;
    }
    
    // Check day (2 digits, 01-31)
    if parts[2].len() != 2 || !parts[2].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let day: u32 = parts[2].parse().unwrap_or(0);
    if day < 1 || day > 31 {
        return false;
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_date_format() {
        assert!(is_valid_date_format("2023-01-01"));
        assert!(is_valid_date_format("2023-12-31"));
        assert!(!is_valid_date_format("2023-1-1"));
        assert!(!is_valid_date_format("23-01-01"));
        assert!(!is_valid_date_format("2023-13-01"));
        assert!(!is_valid_date_format("2023-01-32"));
        assert!(!is_valid_date_format("not-a-date"));
    }

    #[test]
    fn test_create_args_validation() {
        let mut args = CreateArgs {
            input: PathBuf::from("test.parquet"),
            output: PathBuf::from("output.parquet"),
            horizon: 12,
            start_date: None,
            end_date: None,
            config: None,
            format: OutputFormat::Parquet,
            buy_threshold: None,
            sell_threshold: None,
            verbose: false,
            skip_validation: false,
        };

        // Valid args should pass
        assert!(args.validate().is_ok());

        // Invalid horizon
        args.horizon = 0;
        assert!(args.validate().is_err());
        args.horizon = 12;

        // Invalid thresholds
        args.buy_threshold = Some(-0.1);
        assert!(args.validate().is_err());
        args.buy_threshold = Some(0.02);
        
        args.sell_threshold = Some(0.1);
        assert!(args.validate().is_err());
        args.sell_threshold = Some(-0.02);

        // Invalid threshold relationship
        args.buy_threshold = Some(0.01);
        args.sell_threshold = Some(-0.02);
        assert!(args.validate().is_err());
    }
}