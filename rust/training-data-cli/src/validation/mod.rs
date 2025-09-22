// Data validation module

pub mod quality;
pub mod report;

#[cfg(test)]
mod tests;

pub use quality::{
    DataValidator, ValidationConfig, ValidationResult, ValidationStatus,
    MissingValueReport, OutlierReport, TimestampReport, DuplicateReport,
    DataStatistics, OutlierMethod, OutlierStats,
};
pub use report::{ValidationReport, ValidationSummary};