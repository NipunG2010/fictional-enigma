use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use std::time::Duration;
use log::{info, warn, error};

/// Progress tracker for long-running operations
pub struct ProgressTracker {
    multi_progress: MultiProgress,
    main_bar: Option<ProgressBar>,
    current_step: usize,
    total_steps: usize,
}

impl ProgressTracker {
    /// Create a new progress tracker with the specified number of steps
    pub fn new(total_steps: usize) -> Self {
        let multi_progress = MultiProgress::new();
        
        Self {
            multi_progress,
            main_bar: None,
            current_step: 0,
            total_steps,
        }
    }

    /// Start tracking progress with a main progress bar
    pub fn start(&mut self, message: &str) {
        let pb = self.multi_progress.add(ProgressBar::new(self.total_steps as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(Duration::from_millis(100));
        
        self.main_bar = Some(pb);
        info!("Started: {}", message);
    }

    /// Update progress to the next step
    pub fn next_step(&mut self, step_message: &str) {
        if let Some(ref pb) = self.main_bar {
            self.current_step += 1;
            pb.set_position(self.current_step as u64);
            pb.set_message(format!("Step {}/{}: {}", self.current_step, self.total_steps, step_message));
            info!("Step {}/{}: {}", self.current_step, self.total_steps, step_message);
        }
    }

    /// Create a sub-progress bar for detailed operations
    pub fn create_sub_progress(&self, length: u64, message: &str) -> ProgressBar {
        let pb = self.multi_progress.add(ProgressBar::new(length));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner:.yellow} [{elapsed_precise}] [{bar:30.yellow/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-")
        );
        pb.set_message(message.to_string());
        pb
    }

    /// Finish the main progress bar with a completion message
    pub fn finish(&mut self, message: &str) {
        if let Some(ref pb) = self.main_bar {
            pb.finish_with_message(message.to_string());
            info!("Completed: {}", message);
        }
    }

    /// Finish with an error message
    pub fn finish_with_error(&mut self, error_message: &str) {
        if let Some(ref pb) = self.main_bar {
            pb.finish_with_message(format!("❌ {}", error_message));
            error!("Failed: {}", error_message);
        }
    }

    /// Get the current step number
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get the total number of steps
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
}

/// Initialize logging with the specified verbosity level
pub fn init_logging(verbose: bool) {
    let log_level = if verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };

    env_logger::Builder::from_default_env()
        .filter_level(log_level)
        .format_timestamp_secs()
        .init();
}

/// Display summary statistics after successful snapshot creation
pub fn display_summary_statistics(
    total_rows: usize,
    labeled_rows: usize,
    features_count: usize,
    label_distribution: Option<&crate::snapshot::builder::LabelDistributionInfo>,
    processing_time: Duration,
) {
    println!("\n🎉 Snapshot Creation Summary");
    println!("═══════════════════════════════════════");
    println!("📊 Data Statistics:");
    println!("   • Total rows processed: {}", format_number(total_rows));
    println!("   • Labeled rows: {}", format_number(labeled_rows));
    println!("   • Features computed: {}", features_count);
    println!("   • Processing time: {:.2}s", processing_time.as_secs_f64());
    
    if let Some(dist) = label_distribution {
        println!("\n🏷️  Label Distribution:");
        println!("   • Buy signals: {} ({:.1}%)", format_number(dist.buy_count), dist.buy_percentage);
        println!("   • Sell signals: {} ({:.1}%)", format_number(dist.sell_count), dist.sell_percentage);
        println!("   • Hold signals: {} ({:.1}%)", format_number(dist.hold_count), dist.hold_percentage);
        
        // Check for balanced distribution
        let max_percentage = dist.buy_percentage.max(dist.sell_percentage).max(dist.hold_percentage);
        let min_percentage = dist.buy_percentage.min(dist.sell_percentage).min(dist.hold_percentage);
        
        if max_percentage - min_percentage > 20.0 {
            println!("   ⚠️  Warning: Label distribution is imbalanced");
        } else {
            println!("   ✅ Label distribution is reasonably balanced");
        }
    }
    
    println!("\n✨ Snapshot created successfully!");
}

/// Format large numbers with thousands separators
fn format_number(num: usize) -> String {
    let num_str = num.to_string();
    let mut result = String::new();
    
    for (i, c) in num_str.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    
    result.chars().rev().collect()
}

/// Create informative error messages with suggestions for common issues
pub fn create_error_message(error: &anyhow::Error) -> String {
    let error_str = error.to_string().to_lowercase();
    
    let suggestion = if error_str.contains("no such file") || error_str.contains("not found") {
        "\n💡 Suggestions:\n   • Check that the input file path is correct\n   • Ensure the file exists and is readable\n   • Use absolute paths if relative paths aren't working"
    } else if error_str.contains("insufficient data") {
        "\n💡 Suggestions:\n   • Reduce the horizon parameter\n   • Use a larger dataset with more historical data\n   • Check the date range filter settings"
    } else if error_str.contains("validation failed") {
        "\n💡 Suggestions:\n   • Use --skip-validation to bypass data quality checks\n   • Set validation strictness to 'lenient'\n   • Check the validation report for specific issues"
    } else if error_str.contains("permission denied") {
        "\n💡 Suggestions:\n   • Check file permissions\n   • Ensure the output directory is writable\n   • Try running with appropriate permissions"
    } else if error_str.contains("invalid date") || error_str.contains("parse") {
        "\n💡 Suggestions:\n   • Use ISO 8601 date format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)\n   • Check that start date is before end date\n   • Ensure dates are within the data range"
    } else if error_str.contains("memory") || error_str.contains("allocation") {
        "\n💡 Suggestions:\n   • Process data in smaller chunks\n   • Reduce the date range\n   • Close other memory-intensive applications"
    } else if error_str.contains("column") || error_str.contains("schema") {
        "\n💡 Suggestions:\n   • Verify the input file has required OHLCV columns\n   • Check column names match expected format\n   • Ensure the file is a valid Parquet format"
    } else {
        "\n💡 For more help:\n   • Use --verbose for detailed logging\n   • Check the documentation\n   • Verify input data format and parameters"
    };
    
    format!("❌ Error: {}{}", error, suggestion)
}

/// Display a warning message with formatting
pub fn display_warning(message: &str) {
    println!("⚠️  Warning: {}", message);
    warn!("{}", message);
}

/// Display an info message with formatting
pub fn display_info(message: &str) {
    println!("ℹ️  {}", message);
    info!("{}", message);
}

/// Display a success message with formatting
pub fn display_success(message: &str) {
    println!("✅ {}", message);
    info!("{}", message);
}