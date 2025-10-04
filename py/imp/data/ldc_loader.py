"""
LDC signal data loader for Rust LDC engine output integration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime
import json


class LDCSignalConfig(BaseModel):
    """Configuration for LDC signal loading."""
    
    signals: List[str] = Field(
        default=['s_LDC', 's_MR', 's_TSMOM'],
        description="List of signal names to load"
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="Additional feature columns to load"
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Start date for data filtering"
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="End date for data filtering"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Trading symbol filter"
    )
    interval: Optional[str] = Field(
        default=None,
        description="Time interval filter (e.g., '5m', '1h')"
    )
    
    class Config:
        arbitrary_types_allowed = True


class LDCDataLoader:
    """
    Load and integrate LDC signal data from Rust LDC engine output.
    
    Supports loading from:
    - Parquet files (features.parquet, signals.parquet, ohlcv.parquet)
    - Partitioned data directories (symbol/date/interval structure)
    - CSV files (legacy format)
    """
    
    def __init__(self, config: Optional[LDCSignalConfig] = None):
        """
        Initialize LDC data loader.
        
        Args:
            config: Configuration for signal loading
        """
        self.config = config or LDCSignalConfig()
        self._loaded_data: Optional[pd.DataFrame] = None
        self._metadata: Dict = {}
    
    def load_from_directory(self, 
                           data_dir: Union[str, Path],
                           auto_detect: bool = True) -> pd.DataFrame:
        """
        Load LDC signals from a directory.
        
        Args:
            data_dir: Path to directory containing LDC output
            auto_detect: Automatically detect file structure
        
        Returns:
            DataFrame with loaded signals
        """
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Check for partitioned data structure
        if auto_detect and self._is_partitioned_structure(data_dir):
            return self._load_partitioned_data(data_dir)
        
        # Load from flat directory structure
        return self._load_flat_directory(data_dir)
    
    def load_from_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load LDC signals from a single file.
        
        Args:
            file_path: Path to parquet or CSV file
        
        Returns:
            DataFrame with loaded signals
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self._process_loaded_data(df, str(file_path))
    
    def load_rust_ldc_output(self, 
                            output_dir: Union[str, Path],
                            file_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load output from Rust LDC engine with standard file naming.
        
        Args:
            output_dir: Directory containing LDC engine output
            file_types: Types of files to load ('features', 'signals', 'ohlcv')
        
        Returns:
            Combined DataFrame with all requested data
        """
        output_dir = Path(output_dir)
        file_types = file_types or ['features', 'signals', 'ohlcv']
        
        dfs = {}
        for file_type in file_types:
            file_path = output_dir / f'{file_type}.parquet'
            if file_path.exists():
                dfs[file_type] = pd.read_parquet(file_path)
                print(f"✓ Loaded {file_type}: {dfs[file_type].shape}")
            else:
                print(f"⚠️  File not found: {file_path}")
        
        if not dfs:
            raise FileNotFoundError(
                f"No LDC output files found in {output_dir}. "
                f"Expected: {', '.join(f'{ft}.parquet' for ft in file_types)}"
            )
        
        # Combine dataframes
        df = self._combine_dataframes(dfs)
        return self._process_loaded_data(df, str(output_dir))
    
    def _is_partitioned_structure(self, data_dir: Path) -> bool:
        """Check if directory has partitioned structure (symbol=X/date=Y/interval=Z)."""
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        return any(d.name.startswith('symbol=') for d in subdirs)
    
    def _load_partitioned_data(self, data_dir: Path) -> pd.DataFrame:
        """Load data from partitioned directory structure."""
        print(f"📂 Loading partitioned data from {data_dir}")
        
        dfs = []
        
        # Walk through partitioned structure
        for symbol_dir in data_dir.glob('symbol=*'):
            symbol = symbol_dir.name.split('=')[1]
            
            # Apply symbol filter if specified
            if self.config.symbol and symbol != self.config.symbol:
                continue
            
            for date_dir in symbol_dir.glob('date=*'):
                date = date_dir.name.split('=')[1]
                
                for interval_dir in date_dir.glob('interval=*'):
                    interval = interval_dir.name.split('=')[1]
                    
                    # Apply interval filter if specified
                    if self.config.interval and interval != self.config.interval:
                        continue
                    
                    # Load all parquet files in this partition
                    for parquet_file in interval_dir.glob('*.parquet'):
                        df_partition = pd.read_parquet(parquet_file)
                        
                        # Add partition columns
                        df_partition['symbol'] = symbol
                        df_partition['date'] = date
                        df_partition['interval'] = interval
                        
                        dfs.append(df_partition)
                        print(f"  ✓ Loaded {symbol}/{date}/{interval}: {df_partition.shape}")
        
        if not dfs:
            raise ValueError(
                f"No data found matching filters: "
                f"symbol={self.config.symbol}, interval={self.config.interval}"
            )
        
        # Combine all partitions
        df = pd.concat(dfs, ignore_index=True)
        print(f"✓ Combined {len(dfs)} partitions: {df.shape}")
        
        return self._process_loaded_data(df, str(data_dir))
    
    def _load_flat_directory(self, data_dir: Path) -> pd.DataFrame:
        """Load data from flat directory structure."""
        print(f"📂 Loading from flat directory: {data_dir}")
        
        # Look for standard LDC output files
        file_patterns = {
            'features': ['features.parquet', 'features_*.parquet'],
            'signals': ['signals.parquet', 'signals_*.parquet'],
            'ohlcv': ['ohlcv.parquet', 'ohlcv_*.parquet']
        }
        
        dfs = {}
        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                files = list(data_dir.glob(pattern))
                if files:
                    # Load all matching files
                    type_dfs = [pd.read_parquet(f) for f in files]
                    dfs[file_type] = pd.concat(type_dfs, ignore_index=True) if len(type_dfs) > 1 else type_dfs[0]
                    print(f"  ✓ Loaded {file_type}: {dfs[file_type].shape}")
                    break
        
        if not dfs:
            raise FileNotFoundError(f"No LDC data files found in {data_dir}")
        
        # Combine dataframes
        df = self._combine_dataframes(dfs)
        return self._process_loaded_data(df, str(data_dir))
    
    def _combine_dataframes(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple dataframes on timestamp."""
        if len(dfs) == 1:
            return list(dfs.values())[0]
        
        # Start with first dataframe
        df = None
        for name, data in dfs.items():
            if df is None:
                df = data.copy()
            else:
                # Merge on timestamp or index
                if 'timestamp' in df.columns and 'timestamp' in data.columns:
                    df = pd.merge(df, data, on='timestamp', how='outer', suffixes=('', f'_{name}'))
                else:
                    df = df.join(data, how='outer', rsuffix=f'_{name}')
        
        return df
    
    def _process_loaded_data(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Process loaded data: set index, filter columns, apply date filters."""
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            if df['timestamp'].isna().any():
                # Try parsing as string
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True)
        elif df.index.name != 'timestamp':
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
                if df.index.isna().any():
                    df.index = pd.to_datetime(df.index, errors='coerce')
                df.index.name = 'timestamp'
            except Exception as e:
                print(f"⚠️  Warning: Could not convert index to timestamp: {e}")
        
        # Apply date filters
        if self.config.start_date:
            df = df[df.index >= self.config.start_date]
        if self.config.end_date:
            df = df[df.index <= self.config.end_date]
        
        # Filter columns based on config
        available_columns = df.columns.tolist()
        requested_columns = []
        
        # Add requested signals
        for signal in self.config.signals:
            if signal in available_columns:
                requested_columns.append(signal)
            else:
                print(f"⚠️  Signal not found: {signal}")
        
        # Add requested features
        if self.config.features:
            for feature in self.config.features:
                if feature in available_columns and feature not in requested_columns:
                    requested_columns.append(feature)
                elif feature not in available_columns:
                    print(f"⚠️  Feature not found: {feature}")
        
        # If no specific columns requested or found, use all numeric columns
        if not requested_columns:
            requested_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"ℹ️  Using all numeric columns: {len(requested_columns)} columns")
        
        df_filtered = df[requested_columns].copy()
        
        # Store metadata
        self._metadata = {
            'source': source,
            'shape': df_filtered.shape,
            'columns': requested_columns,
            'date_range': {
                'start': str(df_filtered.index.min()),
                'end': str(df_filtered.index.max())
            },
            'loaded_at': datetime.now().isoformat()
        }
        
        self._loaded_data = df_filtered
        
        print(f"\n✓ Data loaded successfully")
        print(f"  Shape: {df_filtered.shape}")
        print(f"  Columns: {requested_columns}")
        print(f"  Date range: {df_filtered.index.min()} to {df_filtered.index.max()}")
        
        return df_filtered
    
    def get_metadata(self) -> Dict:
        """Get metadata about loaded data."""
        return self._metadata.copy()
    
    def save_metadata(self, output_path: Union[str, Path]):
        """Save metadata to JSON file."""
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(self._metadata, f, indent=2)
        print(f"✓ Metadata saved to {output_path}")
    
    def get_signal_statistics(self) -> pd.DataFrame:
        """Get basic statistics for loaded signals."""
        if self._loaded_data is None:
            raise ValueError("No data loaded. Call load_* method first.")
        
        stats = self._loaded_data.describe().T
        stats['missing'] = self._loaded_data.isna().sum()
        stats['missing_pct'] = (stats['missing'] / len(self._loaded_data)) * 100
        
        return stats
