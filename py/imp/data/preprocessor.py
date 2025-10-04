"""
Signal preprocessing for HMM training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, validator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats as scipy_stats
import warnings


class PreprocessingConfig(BaseModel):
    """Configuration for signal preprocessing."""
    
    scaling_method: Literal['standardize', 'normalize', 'robust', 'none'] = Field(
        default='standardize',
        description="Scaling method to apply"
    )
    handle_missing: Literal['forward_fill', 'backward_fill', 'interpolate', 'drop', 'mean'] = Field(
        default='forward_fill',
        description="Method for handling missing values"
    )
    outlier_method: Literal['zscore', 'iqr', 'none'] = Field(
        default='zscore',
        description="Outlier detection method"
    )
    outlier_threshold: float = Field(
        default=3.0,
        description="Threshold for outlier detection (z-score or IQR multiplier)"
    )
    outlier_action: Literal['clip', 'remove', 'none'] = Field(
        default='clip',
        description="Action to take on detected outliers"
    )
    remove_trend: bool = Field(
        default=False,
        description="Remove linear trend from signals"
    )
    apply_log_transform: bool = Field(
        default=False,
        description="Apply log transformation (for positive signals)"
    )
    window_size: Optional[int] = Field(
        default=None,
        description="Window size for rolling operations"
    )
    
    @validator('outlier_threshold')
    def validate_threshold(cls, v):
        if v <= 0:
            raise ValueError("Outlier threshold must be positive")
        return v


class SignalPreprocessor:
    """
    Preprocess LDC signals for HMM training.
    
    Handles:
    - Missing value imputation
    - Outlier detection and treatment
    - Scaling and normalization
    - Trend removal
    - Feature transformations
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize signal preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self._scaler = None
        self._preprocessing_stats: Dict = {}
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Input DataFrame with signals
        
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        print("🔧 Preprocessing signals...")
        print(f"   Input shape: {df_processed.shape}")
        
        # Track preprocessing steps
        self._preprocessing_stats = {
            'input_shape': df_processed.shape,
            'steps': []
        }
        
        # Step 1: Handle missing values
        df_processed, missing_stats = self._handle_missing_values(df_processed)
        self._preprocessing_stats['steps'].append({
            'step': 'handle_missing',
            'method': self.config.handle_missing,
            'stats': missing_stats
        })
        
        # Step 2: Apply log transform if requested
        if self.config.apply_log_transform:
            df_processed, log_stats = self._apply_log_transform(df_processed)
            self._preprocessing_stats['steps'].append({
                'step': 'log_transform',
                'stats': log_stats
            })
        
        # Step 3: Remove trend if requested
        if self.config.remove_trend:
            df_processed, trend_stats = self._remove_trend(df_processed)
            self._preprocessing_stats['steps'].append({
                'step': 'remove_trend',
                'stats': trend_stats
            })
        
        # Step 4: Detect and handle outliers
        if self.config.outlier_method != 'none':
            df_processed, outlier_stats = self._handle_outliers(df_processed)
            self._preprocessing_stats['steps'].append({
                'step': 'handle_outliers',
                'method': self.config.outlier_method,
                'action': self.config.outlier_action,
                'stats': outlier_stats
            })
        
        # Step 5: Apply scaling
        if self.config.scaling_method != 'none':
            df_processed, scaling_stats = self._apply_scaling(df_processed)
            self._preprocessing_stats['steps'].append({
                'step': 'scaling',
                'method': self.config.scaling_method,
                'stats': scaling_stats
            })
        
        self._preprocessing_stats['output_shape'] = df_processed.shape
        
        print(f"   Output shape: {df_processed.shape}")
        print(f"✓ Preprocessing complete ({len(self._preprocessing_stats['steps'])} steps)")
        
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        if self._scaler is None and self.config.scaling_method != 'none':
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df_processed = df.copy()
        
        # Apply same transformations (without fitting)
        df_processed, _ = self._handle_missing_values(df_processed)
        
        if self.config.apply_log_transform:
            df_processed, _ = self._apply_log_transform(df_processed)
        
        if self.config.remove_trend:
            df_processed, _ = self._remove_trend(df_processed)
        
        if self.config.outlier_method != 'none':
            df_processed, _ = self._handle_outliers(df_processed)
        
        if self.config.scaling_method != 'none' and self._scaler is not None:
            df_processed[df_processed.columns] = self._scaler.transform(df_processed.values)
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values in the data."""
        missing_before = df.isna().sum().sum()
        
        if missing_before == 0:
            return df, {'missing_before': 0, 'missing_after': 0}
        
        print(f"   Handling {missing_before} missing values...")
        
        if self.config.handle_missing == 'forward_fill':
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')  # Handle leading NaNs
        elif self.config.handle_missing == 'backward_fill':
            df = df.fillna(method='bfill')
            df = df.fillna(method='ffill')  # Handle trailing NaNs
        elif self.config.handle_missing == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both')
        elif self.config.handle_missing == 'mean':
            df = df.fillna(df.mean())
        elif self.config.handle_missing == 'drop':
            df = df.dropna()
        
        missing_after = df.isna().sum().sum()
        
        stats = {
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'method': self.config.handle_missing
        }
        
        if missing_after > 0:
            print(f"   ⚠️  {missing_after} missing values remain after imputation")
        
        return df, stats
    
    def _apply_log_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Apply log transformation to positive signals."""
        print("   Applying log transformation...")
        
        transformed_cols = []
        for col in df.columns:
            if (df[col] > 0).all():
                df[col] = np.log1p(df[col])
                transformed_cols.append(col)
            else:
                print(f"   ⚠️  Skipping {col} (contains non-positive values)")
        
        stats = {
            'transformed_columns': transformed_cols,
            'skipped_columns': [c for c in df.columns if c not in transformed_cols]
        }
        
        return df, stats
    
    def _remove_trend(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Remove linear trend from signals."""
        print("   Removing linear trends...")
        
        trends = {}
        for col in df.columns:
            # Fit linear trend
            x = np.arange(len(df))
            y = df[col].values
            
            # Remove NaN for trend fitting
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                continue
            
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            trend = slope * x + intercept
            
            # Detrend
            df[col] = df[col] - trend
            
            trends[col] = {'slope': float(slope), 'intercept': float(intercept)}
        
        stats = {'trends': trends}
        return df, stats
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle outliers."""
        print(f"   Detecting outliers ({self.config.outlier_method})...")
        
        outlier_counts = {}
        
        for col in df.columns:
            if self.config.outlier_method == 'zscore':
                # Z-score method
                z_scores = np.abs(scipy_stats.zscore(df[col], nan_policy='omit'))
                outliers = z_scores > self.config.outlier_threshold
            elif self.config.outlier_method == 'iqr':
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            else:
                continue
            
            outlier_count = outliers.sum()
            outlier_counts[col] = int(outlier_count)
            
            if outlier_count > 0:
                print(f"   Found {outlier_count} outliers in {col}")
                
                if self.config.outlier_action == 'clip':
                    # Clip to threshold
                    if self.config.outlier_method == 'zscore':
                        mean = df[col].mean()
                        std = df[col].std()
                        lower = mean - self.config.outlier_threshold * std
                        upper = mean + self.config.outlier_threshold * std
                        df[col] = df[col].clip(lower, upper)
                    else:  # IQR
                        df[col] = df[col].clip(lower_bound, upper_bound)
                elif self.config.outlier_action == 'remove':
                    # Set to NaN and re-impute
                    df.loc[outliers, col] = np.nan
        
        # Re-impute if outliers were removed
        if self.config.outlier_action == 'remove' and sum(outlier_counts.values()) > 0:
            df, _ = self._handle_missing_values(df)
        
        stats = {
            'outlier_counts': outlier_counts,
            'total_outliers': sum(outlier_counts.values())
        }
        
        return df, stats
    
    def _apply_scaling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Apply scaling to the data."""
        print(f"   Applying {self.config.scaling_method} scaling...")
        
        if self.config.scaling_method == 'standardize':
            self._scaler = StandardScaler()
        elif self.config.scaling_method == 'normalize':
            self._scaler = MinMaxScaler()
        elif self.config.scaling_method == 'robust':
            self._scaler = RobustScaler()
        else:
            return df, {}
        
        # Fit and transform
        df[df.columns] = self._scaler.fit_transform(df.values)
        
        stats = {
            'method': self.config.scaling_method,
            'feature_means': self._scaler.mean_.tolist() if hasattr(self._scaler, 'mean_') else None,
            'feature_scales': self._scaler.scale_.tolist() if hasattr(self._scaler, 'scale_') else None
        }
        
        return df, stats
    
    def get_preprocessing_stats(self) -> Dict:
        """Get statistics about preprocessing steps."""
        return self._preprocessing_stats.copy()
    
    def get_recommendations(self, df: pd.DataFrame) -> List[str]:
        """
        Analyze data and provide preprocessing recommendations.
        
        Args:
            df: Input DataFrame
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for missing values
        missing_pct = (df.isna().sum() / len(df)) * 100
        if missing_pct.max() > 5:
            recommendations.append(
                f"High missing data detected (up to {missing_pct.max():.1f}%). "
                f"Consider 'interpolate' or 'drop' for handle_missing."
            )
        
        # Check for outliers
        for col in df.columns:
            z_scores = np.abs(scipy_stats.zscore(df[col].dropna()))
            outlier_pct = (z_scores > 3).sum() / len(z_scores) * 100
            if outlier_pct > 5:
                recommendations.append(
                    f"High outlier rate in {col} ({outlier_pct:.1f}%). "
                    f"Consider outlier_action='clip' or 'remove'."
                )
        
        # Check for skewness
        for col in df.columns:
            skewness = df[col].skew()
            if abs(skewness) > 2:
                recommendations.append(
                    f"High skewness in {col} ({skewness:.2f}). "
                    f"Consider apply_log_transform=True or robust scaling."
                )
        
        # Check for trends
        for col in df.columns:
            x = np.arange(len(df))
            y = df[col].dropna().values
            if len(y) > 10:
                slope, _ = np.polyfit(x[:len(y)], y, 1)
                if abs(slope) > 0.01:
                    recommendations.append(
                        f"Linear trend detected in {col}. "
                        f"Consider remove_trend=True for stationarity."
                    )
        
        # Check scale differences
        ranges = df.max() - df.min()
        if ranges.max() / ranges.min() > 10:
            recommendations.append(
                "Large scale differences between features. "
                "Scaling is recommended (standardize or normalize)."
            )
        
        if not recommendations:
            recommendations.append("Data quality looks good. Current config should work well.")
        
        return recommendations
