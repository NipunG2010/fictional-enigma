"""
Data loading and validation for backtesting framework.

This module provides comprehensive data loading capabilities for signal data,
market data (OHLCV), and data quality validation. It supports both partitioned
and flat file structures with robust error handling and validation.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date
import logging
from dataclasses import dataclass
import warnings

from .config import DataSourceConfig

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality issues and statistics."""
    
    total_records: int
    missing_values: Dict[str, int]
    missing_percentages: Dict[str, float]
    anomalies: Dict[str, List[str]]
    date_range: Tuple[datetime, datetime]
    symbols: List[str]
    data_gaps: List[Dict[str, Any]]
    quality_score: float
    warnings: List[str]
    errors: List[str]


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataLoader:
    """
    Load and validate signal and market data for backtesting.
    
    Supports:
    - Partitioned Parquet files (symbol/date/interval structure)
    - Flat directory structure
    - Date range filtering
    - Data quality validation
    - Missing value handling
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Data source configuration
        """
        self.config = config
        self._signal_data: Optional[pd.DataFrame] = None
        self._market_data: Optional[pd.DataFrame] = None
        self._quality_report: Optional[DataQualityReport] = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def load_signal_data(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load signal data from Parquet files with date filtering.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            validate: Whether to perform data validation
            
        Returns:
            DataFrame with signal data
            
        Raises:
            FileNotFoundError: If signal files not found
            DataValidationError: If validation fails
        """
        logger.info(f"Loading signal data for symbols: {symbols}")
        
        if not self.config.signals_path.exists():
            raise FileNotFoundError(f"Signals path not found: {self.config.signals_path}")
        
        # Determine if we have partitioned or flat structure
        if self._is_partitioned_structure(self.config.signals_path):
            df = self._load_partitioned_signals(symbols, start_date, end_date)
        else:
            df = self._load_flat_signals(symbols, start_date, end_date)
        
        # Validate data if requested
        if validate:
            self._validate_signal_data(df)
        
        self._signal_data = df
        logger.info(f"Loaded signal data: {df.shape} records")
        
        return df
    
    def load_market_data(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load market data (OHLCV) from Parquet files.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            validate: Whether to perform data validation
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            FileNotFoundError: If market data files not found
            DataValidationError: If validation fails
        """
        logger.info(f"Loading market data for symbols: {symbols}")
        
        if not self.config.market_data_path.exists():
            raise FileNotFoundError(f"Market data path not found: {self.config.market_data_path}")
        
        # Determine if we have partitioned or flat structure
        if self._is_partitioned_structure(self.config.market_data_path):
            df = self._load_partitioned_ohlcv(symbols, start_date, end_date)
        else:
            df = self._load_flat_ohlcv(symbols, start_date, end_date)
        
        # Validate data if requested
        if validate:
            self._validate_market_data(df)
        
        self._market_data = df
        logger.info(f"Loaded market data: {df.shape} records")
        
        return df
    
    def load_combined_data(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both signal and market data with alignment.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            validate: Whether to perform data validation
            
        Returns:
            Tuple of (signal_data, market_data) DataFrames
        """
        logger.info("Loading combined signal and market data")
        
        # Load both datasets
        signal_df = self.load_signal_data(symbols, start_date, end_date, validate=False)
        market_df = self.load_market_data(symbols, start_date, end_date, validate=False)
        
        # Align timestamps and symbols
        signal_df, market_df = self._align_datasets(signal_df, market_df)
        
        # Validate combined data if requested
        if validate:
            self._validate_combined_data(signal_df, market_df)
        
        return signal_df, market_df
    
    def validate_data_quality(
        self,
        signal_df: Optional[pd.DataFrame] = None,
        market_df: Optional[pd.DataFrame] = None
    ) -> DataQualityReport:
        """
        Perform comprehensive data quality validation.
        
        Args:
            signal_df: Signal data to validate (uses loaded data if None)
            market_df: Market data to validate (uses loaded data if None)
            
        Returns:
            Data quality report
        """
        signal_df = signal_df if signal_df is not None else self._signal_data
        market_df = market_df if market_df is not None else self._market_data
        
        if signal_df is None and market_df is None:
            raise ValueError("No data available for validation. Load data first.")
        
        logger.info("Performing data quality validation")
        
        # Initialize report components
        total_records = 0
        missing_values = {}
        missing_percentages = {}
        anomalies = {}
        warnings_list = []
        errors_list = []
        symbols = []
        
        # Analyze signal data
        if signal_df is not None:
            total_records += len(signal_df)
            symbols.extend(signal_df['symbol'].unique() if 'symbol' in signal_df.columns else [])
            
            # Check for missing values
            for col in signal_df.columns:
                if signal_df[col].dtype in ['float64', 'int64']:
                    missing = signal_df[col].isna().sum()
                    missing_values[f"signal_{col}"] = missing
                    missing_percentages[f"signal_{col}"] = (missing / len(signal_df)) * 100
            
            # Check for signal anomalies
            signal_anomalies = self._detect_signal_anomalies(signal_df)
            anomalies.update({f"signal_{k}": v for k, v in signal_anomalies.items()})
        
        # Analyze market data
        if market_df is not None:
            total_records += len(market_df)
            if 'symbol' in market_df.columns:
                symbols.extend(market_df['symbol'].unique())
            
            # Check for missing values
            for col in market_df.columns:
                if market_df[col].dtype in ['float64', 'int64']:
                    missing = market_df[col].isna().sum()
                    missing_values[f"market_{col}"] = missing
                    missing_percentages[f"market_{col}"] = (missing / len(market_df)) * 100
            
            # Check for market data anomalies
            market_anomalies = self._detect_market_anomalies(market_df)
            anomalies.update({f"market_{k}": v for k, v in market_anomalies.items()})
        
        # Detect data gaps
        data_gaps = []
        if signal_df is not None:
            gaps = self._detect_data_gaps(signal_df)
            data_gaps.extend(gaps)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            missing_percentages, anomalies, data_gaps
        )
        
        # Generate warnings and errors
        for col, pct in missing_percentages.items():
            if pct > 10:
                errors_list.append(f"High missing data in {col}: {pct:.1f}%")
            elif pct > 5:
                warnings_list.append(f"Moderate missing data in {col}: {pct:.1f}%")
        
        if quality_score < 0.7:
            errors_list.append(f"Low data quality score: {quality_score:.2f}")
        elif quality_score < 0.8:
            warnings_list.append(f"Moderate data quality score: {quality_score:.2f}")
        
        # Determine date range
        date_range = self._get_date_range(signal_df, market_df)
        
        # Create report
        report = DataQualityReport(
            total_records=total_records,
            missing_values=missing_values,
            missing_percentages=missing_percentages,
            anomalies=anomalies,
            date_range=date_range,
            symbols=list(set(symbols)),
            data_gaps=data_gaps,
            quality_score=quality_score,
            warnings=warnings_list,
            errors=errors_list
        )
        
        self._quality_report = report
        
        # Log summary
        logger.info(f"Data quality validation complete:")
        logger.info(f"  Quality score: {quality_score:.2f}")
        logger.info(f"  Total records: {total_records:,}")
        logger.info(f"  Symbols: {len(report.symbols)}")
        logger.info(f"  Warnings: {len(warnings_list)}")
        logger.info(f"  Errors: {len(errors_list)}")
        
        return report
    
    def _is_partitioned_structure(self, path: Path) -> bool:
        """Check if directory has partitioned structure (symbol=X/date=Y/interval=Z)."""
        if not path.is_dir():
            return False
        
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        return any(d.name.startswith('symbol=') for d in subdirs)
    
    def _load_partitioned_signals(
        self,
        symbols: List[str],
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> pd.DataFrame:
        """Load signal data from partitioned structure."""
        logger.info("Loading signals from partitioned structure")
        
        dfs = []
        
        for symbol in symbols:
            symbol_path = self.config.signals_path / f"symbol={symbol}"
            if not symbol_path.exists():
                logger.warning(f"No data found for symbol: {symbol}")
                continue
            
            # Iterate through date partitions
            for date_dir in symbol_path.glob("date=*"):
                date_str = date_dir.name.split('=')[1]
                try:
                    partition_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Invalid date format: {date_str}")
                    continue
                
                # Apply date filtering
                if start_date and partition_date < start_date:
                    continue
                if end_date and partition_date > end_date:
                    continue
                
                # Load signal files from all intervals
                for interval_dir in date_dir.glob("interval=*"):
                    signal_file = interval_dir / "signals.parquet"
                    if signal_file.exists():
                        try:
                            df_partition = pd.read_parquet(signal_file)
                            
                            # Add partition metadata
                            df_partition['symbol'] = symbol
                            df_partition['date'] = date_str
                            df_partition['interval'] = interval_dir.name.split('=')[1]
                            
                            dfs.append(df_partition)
                            
                        except Exception as e:
                            logger.error(f"Error loading {signal_file}: {e}")
        
        if not dfs:
            raise FileNotFoundError(f"No signal data found for symbols: {symbols}")
        
        # Combine all partitions
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(dfs)} partitions with {len(df)} total records")
        
        return df
    
    def _load_flat_signals(
        self,
        symbols: List[str],
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> pd.DataFrame:
        """Load signal data from flat directory structure."""
        logger.info("Loading signals from flat structure")
        
        # Look for signal files
        signal_files = list(self.config.signals_path.glob("*signals*.parquet"))
        
        if not signal_files:
            raise FileNotFoundError(f"No signal files found in {self.config.signals_path}")
        
        dfs = []
        for file_path in signal_files:
            try:
                df = pd.read_parquet(file_path)
                
                # Filter by symbols if symbol column exists
                if 'symbol' in df.columns:
                    df = df[df['symbol'].isin(symbols)]
                
                if len(df) > 0:
                    dfs.append(df)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not dfs:
            raise FileNotFoundError(f"No valid signal data found for symbols: {symbols}")
        
        # Combine all files
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df.set_index('timestamp', inplace=True)
        
        # Apply date filtering
        if start_date:
            df = df[df.index.date >= start_date]
        if end_date:
            df = df[df.index.date <= end_date]
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def _load_partitioned_ohlcv(
        self,
        symbols: List[str],
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> pd.DataFrame:
        """Load OHLCV data from partitioned structure."""
        logger.info("Loading OHLCV from partitioned structure")
        
        dfs = []
        
        for symbol in symbols:
            symbol_path = self.config.market_data_path / f"symbol={symbol}"
            if not symbol_path.exists():
                logger.warning(f"No market data found for symbol: {symbol}")
                continue
            
            # Iterate through date partitions
            for date_dir in symbol_path.glob("date=*"):
                date_str = date_dir.name.split('=')[1]
                try:
                    partition_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Invalid date format: {date_str}")
                    continue
                
                # Apply date filtering
                if start_date and partition_date < start_date:
                    continue
                if end_date and partition_date > end_date:
                    continue
                
                # Load OHLCV files from all intervals
                for interval_dir in date_dir.glob("interval=*"):
                    ohlcv_file = interval_dir / "ohlcv.parquet"
                    if ohlcv_file.exists():
                        try:
                            df_partition = pd.read_parquet(ohlcv_file)
                            
                            # Add partition metadata
                            df_partition['symbol'] = symbol
                            df_partition['date'] = date_str
                            df_partition['interval'] = interval_dir.name.split('=')[1]
                            
                            dfs.append(df_partition)
                            
                        except Exception as e:
                            logger.error(f"Error loading {ohlcv_file}: {e}")
        
        if not dfs:
            raise FileNotFoundError(f"No market data found for symbols: {symbols}")
        
        # Combine all partitions
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(dfs)} partitions with {len(df)} total records")
        
        return df
    
    def _load_flat_ohlcv(
        self,
        symbols: List[str],
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> pd.DataFrame:
        """Load OHLCV data from flat directory structure."""
        logger.info("Loading OHLCV from flat structure")
        
        # Look for OHLCV files
        ohlcv_files = list(self.config.market_data_path.glob("*ohlcv*.parquet"))
        
        if not ohlcv_files:
            raise FileNotFoundError(f"No OHLCV files found in {self.config.market_data_path}")
        
        dfs = []
        for file_path in ohlcv_files:
            try:
                df = pd.read_parquet(file_path)
                
                # Filter by symbols if symbol column exists
                if 'symbol' in df.columns:
                    df = df[df['symbol'].isin(symbols)]
                
                if len(df) > 0:
                    dfs.append(df)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not dfs:
            raise FileNotFoundError(f"No valid OHLCV data found for symbols: {symbols}")
        
        # Combine all files
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df.set_index('timestamp', inplace=True)
        
        # Apply date filtering
        if start_date:
            df = df[df.index.date >= start_date]
        if end_date:
            df = df[df.index.date <= end_date]
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def _align_datasets(
        self,
        signal_df: pd.DataFrame,
        market_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align signal and market data on timestamp and symbol."""
        logger.info("Aligning signal and market datasets")
        
        # Ensure both have symbol columns
        if 'symbol' not in signal_df.columns or 'symbol' not in market_df.columns:
            logger.warning("Symbol column missing, alignment may be incomplete")
            return signal_df, market_df
        
        # Find common symbols
        signal_symbols = set(signal_df['symbol'].unique())
        market_symbols = set(market_df['symbol'].unique())
        common_symbols = signal_symbols.intersection(market_symbols)
        
        if not common_symbols:
            raise DataValidationError("No common symbols found between signal and market data")
        
        # Filter to common symbols
        signal_df = signal_df[signal_df['symbol'].isin(common_symbols)]
        market_df = market_df[market_df['symbol'].isin(common_symbols)]
        
        # Find common time range
        signal_start = signal_df.index.min()
        signal_end = signal_df.index.max()
        market_start = market_df.index.min()
        market_end = market_df.index.max()
        
        common_start = max(signal_start, market_start)
        common_end = min(signal_end, market_end)
        
        # Filter to common time range
        signal_df = signal_df[(signal_df.index >= common_start) & (signal_df.index <= common_end)]
        market_df = market_df[(market_df.index >= common_start) & (market_df.index <= common_end)]
        
        logger.info(f"Aligned datasets:")
        logger.info(f"  Common symbols: {len(common_symbols)}")
        logger.info(f"  Common time range: {common_start} to {common_end}")
        logger.info(f"  Signal records: {len(signal_df)}")
        logger.info(f"  Market records: {len(market_df)}")
        
        return signal_df, market_df
    
    def _validate_signal_data(self, df: pd.DataFrame) -> None:
        """Validate signal data structure and content."""
        logger.info("Validating signal data")
        
        # Check required columns
        expected_signal_cols = ['s_ldc', 's_mr', 's_tsmom']
        missing_cols = [col for col in expected_signal_cols if col not in df.columns]
        
        if missing_cols:
            raise DataValidationError(f"Missing required signal columns: {missing_cols}")
        
        # Check for reasonable signal ranges
        for col in expected_signal_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    if values.min() < -2 or values.max() > 2:
                        logger.warning(f"Signal {col} has extreme values: [{values.min():.3f}, {values.max():.3f}]")
        
        # Check timestamp index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("Signal data must have datetime index")
        
        # Check for duplicate timestamps per symbol
        if 'symbol' in df.columns:
            duplicates = df.groupby('symbol').apply(lambda x: x.index.duplicated().sum())
            if duplicates.sum() > 0:
                logger.warning(f"Found {duplicates.sum()} duplicate timestamps in signal data")
    
    def _validate_market_data(self, df: pd.DataFrame) -> None:
        """Validate market data structure and content."""
        logger.info("Validating market data")
        
        # Check required OHLCV columns
        expected_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in expected_ohlcv_cols if col not in df.columns]
        
        if missing_cols:
            raise DataValidationError(f"Missing required OHLCV columns: {missing_cols}")
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            
            if invalid_ohlc.sum() > 0:
                logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC relationships")
        
        # Check for negative prices or volumes
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                negative_prices = (df[col] <= 0).sum()
                if negative_prices > 0:
                    logger.warning(f"Found {negative_prices} non-positive prices in {col}")
        
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                logger.warning(f"Found {negative_volume} negative volumes")
        
        # Check timestamp index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("Market data must have datetime index")
    
    def _validate_combined_data(
        self,
        signal_df: pd.DataFrame,
        market_df: pd.DataFrame
    ) -> None:
        """Validate alignment between signal and market data."""
        logger.info("Validating combined data alignment")
        
        # Check timestamp alignment
        if 'symbol' in signal_df.columns and 'symbol' in market_df.columns:
            for symbol in signal_df['symbol'].unique():
                signal_times = signal_df[signal_df['symbol'] == symbol].index
                market_times = market_df[market_df['symbol'] == symbol].index
                
                # Check coverage
                signal_coverage = len(signal_times.intersection(market_times)) / len(signal_times)
                if signal_coverage < 0.8:
                    logger.warning(f"Low timestamp coverage for {symbol}: {signal_coverage:.1%}")
    
    def _detect_signal_anomalies(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect anomalies in signal data."""
        anomalies = {}
        
        signal_cols = [col for col in df.columns if col.startswith('s_')]
        
        for col in signal_cols:
            col_anomalies = []
            values = df[col].dropna()
            
            if len(values) > 0:
                # Check for extreme values (beyond 3 standard deviations)
                mean_val = values.mean()
                std_val = values.std()
                extreme_count = ((values - mean_val).abs() > 3 * std_val).sum()
                
                if extreme_count > 0:
                    col_anomalies.append(f"{extreme_count} extreme values (>3σ)")
                
                # Check for constant values
                if values.nunique() == 1:
                    col_anomalies.append("All values are identical")
                
                # Check for NaN spikes
                nan_pct = df[col].isna().mean() * 100
                if nan_pct > 20:
                    col_anomalies.append(f"High NaN percentage: {nan_pct:.1f}%")
            
            if col_anomalies:
                anomalies[col] = col_anomalies
        
        return anomalies
    
    def _detect_market_anomalies(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect anomalies in market data."""
        anomalies = {}
        
        # Check price gaps
        if 'close' in df.columns and 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_index()
                if len(symbol_data) > 1:
                    price_changes = symbol_data['close'].pct_change().dropna()
                    large_gaps = (price_changes.abs() > 0.1).sum()  # >10% price changes
                    
                    if large_gaps > 0:
                        if symbol not in anomalies:
                            anomalies[symbol] = []
                        anomalies[symbol].append(f"{large_gaps} large price gaps (>10%)")
        
        # Check volume spikes
        if 'volume' in df.columns:
            volumes = df['volume'].dropna()
            if len(volumes) > 0:
                volume_median = volumes.median()
                volume_spikes = (volumes > volume_median * 10).sum()
                
                if volume_spikes > 0:
                    anomalies['volume'] = [f"{volume_spikes} volume spikes (>10x median)"]
        
        return anomalies
    
    def _detect_data_gaps(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect gaps in time series data."""
        gaps = []
        
        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_index()
                
                if len(symbol_data) > 1:
                    # Detect gaps larger than expected interval
                    time_diffs = symbol_data.index.to_series().diff().dropna()
                    expected_interval = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(minutes=5)
                    
                    large_gaps = time_diffs[time_diffs > expected_interval * 2]
                    
                    for gap_start, gap_duration in large_gaps.items():
                        gaps.append({
                            'symbol': symbol,
                            'gap_start': gap_start,
                            'gap_duration': str(gap_duration),
                            'expected_interval': str(expected_interval)
                        })
        
        return gaps
    
    def _calculate_quality_score(
        self,
        missing_percentages: Dict[str, float],
        anomalies: Dict[str, List[str]],
        data_gaps: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall data quality score (0-1)."""
        score = 1.0
        
        # Penalize missing data
        if missing_percentages:
            avg_missing = sum(missing_percentages.values()) / len(missing_percentages)
            score -= min(avg_missing / 100, 0.3)  # Max 30% penalty for missing data
        
        # Penalize anomalies
        total_anomalies = sum(len(v) for v in anomalies.values())
        score -= min(total_anomalies * 0.05, 0.3)  # Max 30% penalty for anomalies
        
        # Penalize data gaps
        score -= min(len(data_gaps) * 0.02, 0.2)  # Max 20% penalty for gaps
        
        return max(score, 0.0)
    
    def _get_date_range(
        self,
        signal_df: Optional[pd.DataFrame],
        market_df: Optional[pd.DataFrame]
    ) -> Tuple[datetime, datetime]:
        """Get overall date range from available data."""
        start_dates = []
        end_dates = []
        
        if signal_df is not None and len(signal_df) > 0:
            start_dates.append(signal_df.index.min())
            end_dates.append(signal_df.index.max())
        
        if market_df is not None and len(market_df) > 0:
            start_dates.append(market_df.index.min())
            end_dates.append(market_df.index.max())
        
        if not start_dates:
            now = datetime.now()
            return now, now
        
        return min(start_dates), max(end_dates)
    
    def get_loaded_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Get currently loaded signal and market data."""
        return self._signal_data, self._market_data
    
    def get_quality_report(self) -> Optional[DataQualityReport]:
        """Get the latest data quality report."""
        return self._quality_report
    
    def save_quality_report(self, output_path: Path) -> None:
        """Save data quality report to file."""
        if self._quality_report is None:
            raise ValueError("No quality report available. Run validate_data_quality() first.")
        
        import json
        
        # Convert report to dict
        report_dict = {
            'total_records': self._quality_report.total_records,
            'missing_values': self._quality_report.missing_values,
            'missing_percentages': self._quality_report.missing_percentages,
            'anomalies': self._quality_report.anomalies,
            'date_range': {
                'start': self._quality_report.date_range[0].isoformat(),
                'end': self._quality_report.date_range[1].isoformat()
            },
            'symbols': self._quality_report.symbols,
            'data_gaps': self._quality_report.data_gaps,
            'quality_score': self._quality_report.quality_score,
            'warnings': self._quality_report.warnings,
            'errors': self._quality_report.errors,
            'generated_at': datetime.now().isoformat()
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Quality report saved to {output_path}")