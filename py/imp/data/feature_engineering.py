"""
Feature engineering pipeline for multivariate HMM observations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""
    
    add_returns: bool = Field(
        default=False,
        description="Add return features (percentage change)"
    )
    add_rolling_stats: bool = Field(
        default=False,
        description="Add rolling statistics (mean, std, etc.)"
    )
    rolling_windows: List[int] = Field(
        default=[5, 10, 20],
        description="Window sizes for rolling statistics"
    )
    add_momentum: bool = Field(
        default=False,
        description="Add momentum indicators"
    )
    add_volatility: bool = Field(
        default=False,
        description="Add volatility measures"
    )
    add_pca: bool = Field(
        default=False,
        description="Add PCA components"
    )
    n_pca_components: Optional[int] = Field(
        default=None,
        description="Number of PCA components (None for auto)"
    )
    add_lags: bool = Field(
        default=False,
        description="Add lagged features"
    )
    lag_periods: List[int] = Field(
        default=[1, 2, 3],
        description="Lag periods to add"
    )
    smooth_signals: bool = Field(
        default=False,
        description="Apply smoothing filter"
    )
    smoothing_window: int = Field(
        default=5,
        description="Window size for smoothing"
    )


class FeatureEngineer:
    """
    Engineer features for multivariate HMM observations.
    
    Creates additional features from raw signals to improve
    regime detection and model performance.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureConfig()
        self._pca = None
        self._feature_names: List[str] = []
        self._original_features: List[str] = []
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit feature engineer and transform data.
        
        Args:
            df: Input DataFrame with signals
        
        Returns:
            DataFrame with engineered features
        """
        self._original_features = df.columns.tolist()
        df_features = df.copy()
        
        print("🔧 Engineering features...")
        print(f"   Input features: {len(df.columns)}")
        
        # Apply smoothing first if requested
        if self.config.smooth_signals:
            df_features = self._add_smoothed_features(df_features)
        
        # Add various feature types
        if self.config.add_returns:
            df_features = self._add_returns(df_features)
        
        if self.config.add_rolling_stats:
            df_features = self._add_rolling_statistics(df_features)
        
        if self.config.add_momentum:
            df_features = self._add_momentum_features(df_features)
        
        if self.config.add_volatility:
            df_features = self._add_volatility_features(df_features)
        
        if self.config.add_lags:
            df_features = self._add_lagged_features(df_features)
        
        if self.config.add_pca:
            df_features = self._add_pca_features(df_features)
        
        # Remove any rows with NaN created by feature engineering
        initial_len = len(df_features)
        df_features = df_features.dropna()
        dropped = initial_len - len(df_features)
        
        if dropped > 0:
            print(f"   Dropped {dropped} rows with NaN from feature engineering")
        
        self._feature_names = df_features.columns.tolist()
        
        print(f"   Output features: {len(df_features.columns)}")
        print(f"   Output samples: {len(df_features)}")
        print(f"✓ Feature engineering complete")
        
        return df_features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature engineer.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Apply same transformations
        if self.config.smooth_signals:
            df_features = self._add_smoothed_features(df_features)
        
        if self.config.add_returns:
            df_features = self._add_returns(df_features)
        
        if self.config.add_rolling_stats:
            df_features = self._add_rolling_statistics(df_features)
        
        if self.config.add_momentum:
            df_features = self._add_momentum_features(df_features)
        
        if self.config.add_volatility:
            df_features = self._add_volatility_features(df_features)
        
        if self.config.add_lags:
            df_features = self._add_lagged_features(df_features)
        
        if self.config.add_pca and self._pca is not None:
            # Use fitted PCA
            pca_features = self._pca.transform(df[self._original_features].dropna())
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'pca_{i}' for i in range(pca_features.shape[1])],
                index=df[self._original_features].dropna().index
            )
            df_features = df_features.join(pca_df, how='inner')
        
        df_features = df_features.dropna()
        
        return df_features
    
    def _add_smoothed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing filter to signals."""
        print(f"   Adding smoothed features (window={self.config.smoothing_window})...")
        
        for col in self._original_features:
            if col in df.columns:
                # Use Savitzky-Golay filter for smoothing
                window = min(self.config.smoothing_window, len(df) // 2)
                if window % 2 == 0:
                    window += 1  # Must be odd
                
                if window >= 3:
                    try:
                        df[f'{col}_smooth'] = savgol_filter(
                            df[col].fillna(method='ffill').fillna(method='bfill'),
                            window_length=window,
                            polyorder=2
                        )
                    except Exception:
                        # Fallback to simple moving average
                        df[f'{col}_smooth'] = df[col].rolling(window=window, center=True).mean()
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features (percentage change)."""
        print("   Adding return features...")
        
        for col in self._original_features:
            if col in df.columns:
                df[f'{col}_return'] = df[col].pct_change()
                df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
        
        return df
    
    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        print(f"   Adding rolling statistics (windows={self.config.rolling_windows})...")
        
        for col in self._original_features:
            if col not in df.columns:
                continue
            
            for window in self.config.rolling_windows:
                if window >= len(df):
                    continue
                
                # Rolling mean
                df[f'{col}_ma{window}'] = df[col].rolling(window=window).mean()
                
                # Rolling std
                df[f'{col}_std{window}'] = df[col].rolling(window=window).std()
                
                # Rolling z-score
                rolling_mean = df[col].rolling(window=window).mean()
                rolling_std = df[col].rolling(window=window).std()
                df[f'{col}_zscore{window}'] = (df[col] - rolling_mean) / rolling_std
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        print("   Adding momentum features...")
        
        for col in self._original_features:
            if col not in df.columns:
                continue
            
            # Rate of change
            df[f'{col}_roc5'] = df[col].pct_change(periods=5)
            df[f'{col}_roc10'] = df[col].pct_change(periods=10)
            
            # Momentum (difference)
            df[f'{col}_momentum5'] = df[col] - df[col].shift(5)
            df[f'{col}_momentum10'] = df[col] - df[col].shift(10)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures."""
        print("   Adding volatility features...")
        
        for col in self._original_features:
            if col not in df.columns:
                continue
            
            # Historical volatility (rolling std of returns)
            returns = df[col].pct_change()
            df[f'{col}_vol10'] = returns.rolling(window=10).std()
            df[f'{col}_vol20'] = returns.rolling(window=20).std()
            
            # Parkinson volatility (using high-low range if available)
            # For now, use rolling range as proxy
            df[f'{col}_range10'] = df[col].rolling(window=10).max() - df[col].rolling(window=10).min()
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        print(f"   Adding lagged features (lags={self.config.lag_periods})...")
        
        for col in self._original_features:
            if col not in df.columns:
                continue
            
            for lag in self.config.lag_periods:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return df
    
    def _add_pca_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add PCA components."""
        print("   Adding PCA features...")
        
        # Use only original features for PCA
        original_cols = [col for col in self._original_features if col in df.columns]
        
        if len(original_cols) < 2:
            print("   ⚠️  Skipping PCA (need at least 2 features)")
            return df
        
        # Prepare data for PCA
        pca_data = df[original_cols].dropna()
        
        if len(pca_data) < 10:
            print("   ⚠️  Skipping PCA (insufficient samples)")
            return df
        
        # Determine number of components
        n_components = self.config.n_pca_components
        if n_components is None:
            # Use explained variance threshold
            n_components = min(len(original_cols), len(pca_data) // 10)
        
        # Fit PCA
        self._pca = PCA(n_components=n_components)
        pca_features = self._pca.fit_transform(pca_data)
        
        # Add PCA features
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'pca_{i}' for i in range(pca_features.shape[1])],
            index=pca_data.index
        )
        
        df = df.join(pca_df, how='left')
        
        explained_var = self._pca.explained_variance_ratio_.sum()
        print(f"   Added {n_components} PCA components (explained variance: {explained_var:.2%})")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names after engineering."""
        return self._feature_names.copy()
    
    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute feature importance based on variance and correlation with original signals.
        
        Args:
            df: DataFrame with engineered features
        
        Returns:
            DataFrame with feature importance metrics
        """
        importance = pd.DataFrame({
            'feature': df.columns,
            'variance': df.var().values,
            'mean_abs': df.abs().mean().values,
            'non_zero_pct': (df != 0).mean().values * 100
        })
        
        # Add correlation with original features
        if self._original_features:
            max_corr = []
            for col in df.columns:
                corrs = [abs(df[col].corr(df[orig])) 
                        for orig in self._original_features 
                        if orig in df.columns]
                max_corr.append(max(corrs) if corrs else 0)
            
            importance['max_corr_original'] = max_corr
        
        # Sort by variance
        importance = importance.sort_values('variance', ascending=False)
        
        return importance
    
    def select_top_features(self, 
                          df: pd.DataFrame, 
                          n_features: int,
                          method: Literal['variance', 'correlation'] = 'variance') -> pd.DataFrame:
        """
        Select top N features based on importance metric.
        
        Args:
            df: DataFrame with engineered features
            n_features: Number of features to select
            method: Selection method ('variance' or 'correlation')
        
        Returns:
            DataFrame with selected features
        """
        importance = self.get_feature_importance(df)
        
        if method == 'variance':
            top_features = importance.nlargest(n_features, 'variance')['feature'].tolist()
        elif method == 'correlation':
            top_features = importance.nlargest(n_features, 'max_corr_original')['feature'].tolist()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ Selected top {n_features} features by {method}")
        
        return df[top_features]
