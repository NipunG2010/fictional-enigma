"""
Data loading utilities for HMM research environment.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

def load_ldc_signals(data_path: Union[str, Path], 
                    signals: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load LDC signal data from parquet files.
    
    Args:
        data_path: Path to the data directory or specific parquet file
        signals: List of signal names to load (default: all available)
    
    Returns:
        DataFrame with timestamp index and signal columns
    """
    
    data_path = Path(data_path)
    
    if data_path.is_file() and data_path.suffix == '.parquet':
        # Load single parquet file
        df = pd.read_parquet(data_path)
    elif data_path.is_dir():
        # Look for parquet files in directory
        parquet_files = list(data_path.glob('*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")
        
        # Load and combine all parquet files
        dfs = []
        for file in parquet_files:
            df_temp = pd.read_parquet(file)
            dfs.append(df_temp)
        
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    # Ensure timestamp column is datetime and set as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif df.index.name != 'timestamp':
        # Try to convert index to datetime if it's not already
        try:
            df.index = pd.to_datetime(df.index)
            df.index.name = 'timestamp'
        except:
            print("⚠️  Warning: Could not convert index to timestamp")
    
    # Filter signals if specified
    if signals:
        available_signals = [col for col in signals if col in df.columns]
        missing_signals = [col for col in signals if col not in df.columns]
        
        if missing_signals:
            print(f"⚠️  Warning: Missing signals: {missing_signals}")
        
        if available_signals:
            df = df[available_signals]
        else:
            raise ValueError("None of the specified signals found in data")
    
    print(f"✓ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
    print(f"✓ Signals: {list(df.columns)}")
    
    return df


def load_sample_data(n_samples: int = 1000, 
                    n_features: int = 3,
                    random_state: int = 42) -> pd.DataFrame:
    """
    Generate sample data for testing when real LDC data is not available.
    
    Args:
        n_samples: Number of time steps
        n_features: Number of signal features
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic signal data
    """
    
    np.random.seed(random_state)
    
    # Generate timestamps
    timestamps = pd.date_range(
        start='2024-01-01', 
        periods=n_samples, 
        freq='5T'  # 5-minute intervals
    )
    
    # Generate synthetic signals with regime-like behavior
    data = {}
    
    # Create regime switching behavior
    regime_changes = np.random.choice(n_samples, size=5, replace=False)
    regime_changes = np.sort(regime_changes)
    
    for i in range(n_features):
        signal = np.zeros(n_samples)
        current_regime = 0
        
        for j, change_point in enumerate(np.append(regime_changes, n_samples)):
            start_idx = regime_changes[j-1] if j > 0 else 0
            end_idx = change_point
            
            # Different regime characteristics
            if current_regime % 3 == 0:  # Low volatility regime
                signal[start_idx:end_idx] = np.random.normal(0, 0.5, end_idx - start_idx)
            elif current_regime % 3 == 1:  # High volatility regime
                signal[start_idx:end_idx] = np.random.normal(0, 2.0, end_idx - start_idx)
            else:  # Trending regime
                trend = np.linspace(0, 2, end_idx - start_idx)
                signal[start_idx:end_idx] = trend + np.random.normal(0, 0.8, end_idx - start_idx)
            
            current_regime += 1
        
        # Add some autocorrelation
        for k in range(1, len(signal)):
            signal[k] += 0.3 * signal[k-1]
        
        data[f's_signal_{i+1}'] = signal
    
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = 'timestamp'
    
    print(f"✓ Generated sample data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def preprocess_signals(df: pd.DataFrame, 
                      method: str = 'standardize',
                      handle_missing: str = 'forward_fill',
                      outlier_threshold: float = 3.0) -> pd.DataFrame:
    """
    Preprocess signal data for HMM training.
    
    Args:
        df: Input DataFrame with signal data
        method: Preprocessing method ('standardize', 'normalize', 'none')
        handle_missing: How to handle missing values ('forward_fill', 'drop', 'interpolate')
        outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
        Preprocessed DataFrame
    """
    
    df_processed = df.copy()
    
    print(f"🔧 Preprocessing signals...")
    print(f"   Input shape: {df_processed.shape}")
    
    # Handle missing values
    if handle_missing == 'forward_fill':
        df_processed = df_processed.fillna(method='ffill')
        df_processed = df_processed.fillna(method='bfill')  # Handle leading NaNs
    elif handle_missing == 'drop':
        df_processed = df_processed.dropna()
    elif handle_missing == 'interpolate':
        df_processed = df_processed.interpolate(method='linear')
    
    # Remove outliers
    if outlier_threshold > 0:
        for col in df_processed.columns:
            z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
            outliers = z_scores > outlier_threshold
            if outliers.sum() > 0:
                print(f"   Removing {outliers.sum()} outliers from {col}")
                df_processed.loc[outliers, col] = np.nan
        
        # Fill outlier-induced NaNs
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
    # Apply preprocessing method
    if method == 'standardize':
        for col in df_processed.columns:
            df_processed[col] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
        print("   ✓ Applied standardization (z-score)")
    elif method == 'normalize':
        for col in df_processed.columns:
            min_val = df_processed[col].min()
            max_val = df_processed[col].max()
            df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val)
        print("   ✓ Applied min-max normalization")
    elif method == 'none':
        print("   ✓ No scaling applied")
    
    print(f"   Output shape: {df_processed.shape}")
    
    return df_processed


def create_multivariate_observations(df: pd.DataFrame) -> np.ndarray:
    """
    Convert DataFrame to numpy array suitable for HMM training.
    
    Args:
        df: DataFrame with signal data
    
    Returns:
        2D numpy array (n_samples, n_features)
    """
    
    observations = df.values
    
    print(f"✓ Created observations array: {observations.shape}")
    print(f"   Features: {list(df.columns)}")
    
    return observations


def save_processed_data(df: pd.DataFrame, 
                       output_path: Union[str, Path],
                       metadata: Optional[Dict] = None):
    """
    Save processed data with metadata.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the data
        metadata: Optional metadata dictionary
    """
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    if output_path.suffix == '.parquet':
        df.to_parquet(output_path)
    elif output_path.suffix == '.csv':
        df.to_csv(output_path)
    else:
        # Default to parquet
        output_path = output_path.with_suffix('.parquet')
        df.to_parquet(output_path)
    
    # Save metadata
    if metadata:
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    print(f"✓ Saved processed data to {output_path}")


def load_rust_ldc_output(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load output from Rust LDC engine.
    
    Args:
        data_dir: Directory containing LDC engine output files
    
    Returns:
        DataFrame with LDC signals
    """
    
    data_dir = Path(data_dir)
    
    # Look for common LDC output files
    signal_files = {
        'features': data_dir / 'features.parquet',
        'signals': data_dir / 'signals.parquet',
        'ohlcv': data_dir / 'ohlcv.parquet'
    }
    
    dfs = {}
    for name, file_path in signal_files.items():
        if file_path.exists():
            dfs[name] = pd.read_parquet(file_path)
            print(f"✓ Loaded {name}: {dfs[name].shape}")
    
    if not dfs:
        raise FileNotFoundError(f"No LDC output files found in {data_dir}")
    
    # Combine dataframes if multiple files exist
    if len(dfs) == 1:
        df = list(dfs.values())[0]
    else:
        # Merge on timestamp/index
        df = None
        for name, data in dfs.items():
            if df is None:
                df = data
            else:
                df = df.join(data, how='outer', rsuffix=f'_{name}')
    
    # Ensure timestamp index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    return df