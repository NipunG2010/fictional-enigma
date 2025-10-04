"""
Data validation utilities for HMM training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from scipy import stats as scipy_stats


class ValidationReport(BaseModel):
    """Report of data validation results."""
    
    is_valid: bool = Field(description="Overall validation status")
    timestamp: datetime = Field(default_factory=datetime.now)
    checks_passed: List[str] = Field(default_factory=list)
    checks_failed: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    statistics: Dict = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def print_summary(self):
        """Print formatted validation summary."""
        print("\n" + "="*60)
        print("📋 DATA VALIDATION REPORT")
        print("="*60)
        print(f"Status: {'✓ PASSED' if self.is_valid else '✗ FAILED'}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        if self.checks_passed:
            print(f"✓ Passed Checks ({len(self.checks_passed)}):")
            for check in self.checks_passed:
                print(f"  • {check}")
            print()
        
        if self.checks_failed:
            print(f"✗ Failed Checks ({len(self.checks_failed)}):")
            for check in self.checks_failed:
                print(f"  • {check}")
            print()
        
        if self.warnings:
            print(f"⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()
        
        if self.recommendations:
            print(f"💡 Recommendations:")
            for rec in self.recommendations:
                print(f"  • {rec}")
            print()
        
        print("="*60)


class DataValidator:
    """
    Validate signal data quality for HMM training.
    
    Performs checks for:
    - Data completeness
    - Numerical stability
    - Temporal consistency
    - Statistical properties
    - HMM-specific requirements
    """
    
    def __init__(self, 
                 min_samples: int = 100,
                 max_missing_pct: float = 10.0,
                 check_stationarity: bool = True,
                 check_multicollinearity: bool = True):
        """
        Initialize data validator.
        
        Args:
            min_samples: Minimum number of samples required
            max_missing_pct: Maximum allowed missing data percentage
            check_stationarity: Check for stationarity
            check_multicollinearity: Check for multicollinearity
        """
        self.min_samples = min_samples
        self.max_missing_pct = max_missing_pct
        self.check_stationarity = check_stationarity
        self.check_multicollinearity = check_multicollinearity
    
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Perform comprehensive data validation.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            ValidationReport with results
        """
        report = ValidationReport(is_valid=True)
        
        print("🔍 Validating data quality...")
        
        # Run all validation checks
        self._check_shape(df, report)
        self._check_missing_data(df, report)
        self._check_numerical_stability(df, report)
        self._check_temporal_consistency(df, report)
        self._check_statistical_properties(df, report)
        
        if self.check_stationarity:
            self._check_stationarity(df, report)
        
        if self.check_multicollinearity:
            self._check_multicollinearity(df, report)
        
        self._check_hmm_requirements(df, report)
        
        # Compute overall statistics
        report.statistics = self._compute_statistics(df)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(df, report)
        
        # Determine overall validity
        report.is_valid = len(report.checks_failed) == 0
        
        print(f"✓ Validation complete: {len(report.checks_passed)} passed, "
              f"{len(report.checks_failed)} failed, {len(report.warnings)} warnings")
        
        return report
    
    def _check_shape(self, df: pd.DataFrame, report: ValidationReport):
        """Check data shape requirements."""
        n_samples, n_features = df.shape
        
        if n_samples < self.min_samples:
            report.checks_failed.append(
                f"Insufficient samples: {n_samples} < {self.min_samples}"
            )
        else:
            report.checks_passed.append(
                f"Sufficient samples: {n_samples} >= {self.min_samples}"
            )
        
        if n_features < 1:
            report.checks_failed.append("No features found")
        else:
            report.checks_passed.append(f"Features present: {n_features}")
    
    def _check_missing_data(self, df: pd.DataFrame, report: ValidationReport):
        """Check for missing data."""
        missing_counts = df.isna().sum()
        missing_pct = (missing_counts / len(df)) * 100
        
        max_missing = missing_pct.max()
        
        if max_missing > self.max_missing_pct:
            report.checks_failed.append(
                f"Excessive missing data: {max_missing:.1f}% > {self.max_missing_pct}%"
            )
        elif max_missing > 0:
            report.warnings.append(
                f"Missing data present: {max_missing:.1f}% (within threshold)"
            )
            report.checks_passed.append("Missing data within acceptable range")
        else:
            report.checks_passed.append("No missing data")
    
    def _check_numerical_stability(self, df: pd.DataFrame, report: ValidationReport):
        """Check for numerical stability issues."""
        issues = []
        
        for col in df.columns:
            # Check for infinite values
            if np.isinf(df[col]).any():
                issues.append(f"{col}: contains infinite values")
            
            # Check for constant columns
            if df[col].nunique() == 1:
                issues.append(f"{col}: constant column (no variance)")
            
            # Check for extreme values
            if df[col].dtype in [np.float32, np.float64]:
                if df[col].abs().max() > 1e10:
                    issues.append(f"{col}: extremely large values (>{1e10})")
                if (df[col].abs() < 1e-10).any() and (df[col] != 0).any():
                    report.warnings.append(f"{col}: extremely small values (<{1e-10})")
        
        if issues:
            for issue in issues:
                report.checks_failed.append(f"Numerical instability: {issue}")
        else:
            report.checks_passed.append("Numerical stability OK")
    
    def _check_temporal_consistency(self, df: pd.DataFrame, report: ValidationReport):
        """Check temporal consistency of time series."""
        if not isinstance(df.index, pd.DatetimeIndex):
            report.warnings.append("Index is not DatetimeIndex - skipping temporal checks")
            return
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            n_duplicates = df.index.duplicated().sum()
            report.checks_failed.append(f"Duplicate timestamps: {n_duplicates}")
        else:
            report.checks_passed.append("No duplicate timestamps")
        
        # Check for monotonic index
        if not df.index.is_monotonic_increasing:
            report.checks_failed.append("Index is not monotonically increasing")
        else:
            report.checks_passed.append("Index is monotonically increasing")
        
        # Check for regular intervals
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            if time_diffs.nunique() > 1:
                report.warnings.append(
                    f"Irregular time intervals detected "
                    f"(min: {time_diffs.min()}, max: {time_diffs.max()})"
                )
    
    def _check_statistical_properties(self, df: pd.DataFrame, report: ValidationReport):
        """Check statistical properties of the data."""
        for col in df.columns:
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            # Check variance
            variance = data.var()
            if variance < 1e-10:
                report.warnings.append(f"{col}: very low variance ({variance:.2e})")
            
            # Check for extreme skewness
            skewness = data.skew()
            if abs(skewness) > 5:
                report.warnings.append(
                    f"{col}: high skewness ({skewness:.2f}) - consider transformation"
                )
            
            # Check for extreme kurtosis
            kurtosis = data.kurtosis()
            if abs(kurtosis) > 10:
                report.warnings.append(
                    f"{col}: high kurtosis ({kurtosis:.2f}) - heavy tails detected"
                )
        
        report.checks_passed.append("Statistical properties checked")
    
    def _check_stationarity(self, df: pd.DataFrame, report: ValidationReport):
        """Check for stationarity using augmented Dickey-Fuller test."""
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            report.warnings.append("statsmodels not available - skipping stationarity test")
            return
        
        non_stationary = []
        
        for col in df.columns:
            data = df[col].dropna()
            
            if len(data) < 20:
                continue
            
            try:
                result = adfuller(data, autolag='AIC')
                p_value = result[1]
                
                if p_value > 0.05:
                    non_stationary.append(f"{col} (p={p_value:.3f})")
            except Exception as e:
                report.warnings.append(f"Stationarity test failed for {col}: {str(e)}")
        
        if non_stationary:
            report.warnings.append(
                f"Non-stationary series detected: {', '.join(non_stationary)}. "
                f"Consider differencing or detrending."
            )
        else:
            report.checks_passed.append("All series appear stationary")
    
    def _check_multicollinearity(self, df: pd.DataFrame, report: ValidationReport):
        """Check for multicollinearity between features."""
        if df.shape[1] < 2:
            return
        
        # Compute correlation matrix
        corr_matrix = df.corr().abs()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append(
                        f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]} "
                        f"({corr_matrix.iloc[i, j]:.3f})"
                    )
        
        if high_corr_pairs:
            report.warnings.append(
                f"High correlation detected: {', '.join(high_corr_pairs)}. "
                f"Consider removing redundant features."
            )
        else:
            report.checks_passed.append("No severe multicollinearity detected")
    
    def _check_hmm_requirements(self, df: pd.DataFrame, report: ValidationReport):
        """Check HMM-specific requirements."""
        
        # Check minimum sequence length for HMM training
        if len(df) < 50:
            report.warnings.append(
                f"Short sequence ({len(df)} samples). "
                f"HMM training may be unstable with <50 samples."
            )
        
        # Check for sufficient variability
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.1:
                report.warnings.append(
                    f"{col}: low variability ({unique_ratio:.1%} unique values). "
                    f"May not provide useful regime information."
                )
        
        report.checks_passed.append("HMM-specific requirements checked")
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute comprehensive statistics."""
        stats_dict = {
            'shape': df.shape,
            'n_samples': len(df),
            'n_features': len(df.columns),
            'features': df.columns.tolist(),
            'missing_values': df.isna().sum().to_dict(),
            'missing_pct': ((df.isna().sum() / len(df)) * 100).to_dict(),
            'descriptive_stats': df.describe().to_dict(),
        }
        
        if isinstance(df.index, pd.DatetimeIndex):
            stats_dict['temporal'] = {
                'start': str(df.index.min()),
                'end': str(df.index.max()),
                'duration': str(df.index.max() - df.index.min()),
                'frequency': str(pd.infer_freq(df.index)) if len(df) > 2 else 'unknown'
            }
        
        # Correlation matrix
        if df.shape[1] > 1:
            stats_dict['correlation_matrix'] = df.corr().to_dict()
        
        return stats_dict
    
    def _generate_recommendations(self, 
                                 df: pd.DataFrame, 
                                 report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Based on failed checks
        if any('missing data' in check.lower() for check in report.checks_failed):
            recommendations.append(
                "Apply missing value imputation using SignalPreprocessor "
                "(forward_fill, interpolate, or drop)"
            )
        
        if any('numerical instability' in check.lower() for check in report.checks_failed):
            recommendations.append(
                "Address numerical instability: remove infinite values, "
                "clip extreme values, or apply robust scaling"
            )
        
        # Based on warnings
        if any('skewness' in warning.lower() for warning in report.warnings):
            recommendations.append(
                "Apply log transformation or use robust scaling to handle skewed distributions"
            )
        
        if any('non-stationary' in warning.lower() for warning in report.warnings):
            recommendations.append(
                "Apply differencing or detrending to achieve stationarity"
            )
        
        if any('correlation' in warning.lower() for warning in report.warnings):
            recommendations.append(
                "Consider feature selection or PCA to reduce multicollinearity"
            )
        
        # General recommendations
        if len(df) < 200:
            recommendations.append(
                "Consider collecting more data for robust HMM training (recommended: >200 samples)"
            )
        
        if not recommendations:
            recommendations.append(
                "Data quality is good. Proceed with HMM training."
            )
        
        return recommendations
    
    def quick_check(self, df: pd.DataFrame) -> bool:
        """
        Quick validation check returning boolean.
        
        Args:
            df: DataFrame to check
        
        Returns:
            True if data passes basic checks
        """
        # Basic checks only
        if len(df) < self.min_samples:
            return False
        
        if df.isna().sum().sum() / df.size * 100 > self.max_missing_pct:
            return False
        
        if np.isinf(df.values).any():
            return False
        
        return True
