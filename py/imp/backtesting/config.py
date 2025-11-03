"""
Configuration schema for backtesting framework using Pydantic.

This module defines comprehensive configuration classes for all aspects
of the backtesting system, including data sources, signal processing,
position sizing, cost modeling, and performance analysis.
"""

from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Any
from enum import Enum
import os

import yaml
from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict


class PositionSizingMethod(str, Enum):
    """Available position sizing methods."""
    FIXED_SIZE = "fixed_size"
    PERCENTAGE = "percentage"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"


class FusionMethod(str, Enum):
    """Signal fusion methods."""
    HMM_WEIGHTED = "hmm_weighted"
    STATIC_WEIGHTED = "static_weighted"


class CostStructureType(str, Enum):
    """Asset class types for cost modeling."""
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITY = "equity"
    FUTURES = "futures"


class DataSourceConfig(BaseModel):
    """Configuration for data sources and paths."""
    model_config = ConfigDict(extra="forbid")
    
    signals_path: Path = Field(
        ..., 
        description="Path to Parquet signal files (supports partitioning)"
    )
    market_data_path: Path = Field(
        ..., 
        description="Path to OHLCV market data files"
    )
    minio_endpoint: Optional[str] = Field(
        None, 
        description="MinIO endpoint for HMM artifacts (optional)"
    )
    minio_access_key: Optional[str] = Field(
        None, 
        description="MinIO access key"
    )
    minio_secret_key: Optional[str] = Field(
        None, 
        description="MinIO secret key"
    )
    minio_bucket: str = Field(
        "artifacts", 
        description="MinIO bucket name for artifacts"
    )
    local_cache_path: Path = Field(
        Path("./cache"), 
        description="Local cache directory for artifacts"
    )


class SignalProcessingConfig(BaseModel):
    """Configuration for signal processing and filtering."""
    model_config = ConfigDict(extra="forbid")
    
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"ldc": 0.6, "mr": 0.4, "tsmom": 0.5},
        description="Signal thresholds for trade generation"
    )
    filters: Dict[str, Union[float, int]] = Field(
        default_factory=lambda: {
            "min_signal_strength": 0.3,
            "max_trades_per_day": 10,
            "cooldown_periods": 5
        },
        description="Signal filtering rules"
    )
    fusion_method: FusionMethod = Field(
        FusionMethod.HMM_WEIGHTED,
        description="Method for combining signals"
    )
    static_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Static weights when not using HMM (required if fusion_method=static_weighted)"
    )
    
    @validator('static_weights')
    def validate_static_weights(cls, v, values):
        if values.get('fusion_method') == FusionMethod.STATIC_WEIGHTED and v is None:
            raise ValueError("static_weights required when fusion_method is static_weighted")
        return v


class PositionSizingConfig(BaseModel):
    """Configuration for position sizing and risk management."""
    model_config = ConfigDict(extra="forbid")
    
    method: PositionSizingMethod = Field(
        PositionSizingMethod.PERCENTAGE,
        description="Position sizing method"
    )
    fixed_size: Optional[float] = Field(
        None,
        description="Fixed dollar amount per trade (required if method=fixed_size)"
    )
    percentage: Optional[float] = Field(
        0.02,  # 2% of portfolio
        description="Percentage of portfolio per trade (required if method=percentage)",
        ge=0.001,
        le=1.0
    )
    volatility_lookback: int = Field(
        20,
        description="Lookback period for volatility calculation",
        ge=5,
        le=100
    )
    volatility_target: Optional[float] = Field(
        0.15,  # 15% annualized volatility target
        description="Target volatility for volatility-adjusted sizing",
        ge=0.01,
        le=1.0
    )
    max_position_size: float = Field(
        0.1,  # 10% max position
        description="Maximum position size as fraction of portfolio",
        ge=0.001,
        le=1.0
    )
    max_total_exposure: float = Field(
        1.0,  # 100% max total exposure
        description="Maximum total exposure across all positions",
        ge=0.1,
        le=3.0
    )
    data_frequency_minutes: int = Field(
        5,
        description="Data frequency in minutes for annualization calculations",
        ge=1,
        le=1440
    )
    
    @validator('fixed_size')
    def validate_fixed_size(cls, v, values):
        if values.get('method') == PositionSizingMethod.FIXED_SIZE and v is None:
            raise ValueError("fixed_size required when method is fixed_size")
        return v


class CostModelConfig(BaseModel):
    """Configuration for transaction costs and slippage modeling."""
    model_config = ConfigDict(extra="forbid")
    
    commission_rate: float = Field(
        0.001,  # 0.1%
        description="Commission rate as fraction of trade value",
        ge=0.0,
        le=0.1
    )
    min_commission: float = Field(
        1.0,
        description="Minimum commission per trade",
        ge=0.0
    )
    spread_costs: Dict[CostStructureType, float] = Field(
        default_factory=lambda: {
            CostStructureType.CRYPTO: 0.0005,
            CostStructureType.FOREX: 0.00002,
            CostStructureType.EQUITY: 0.0001,
            CostStructureType.FUTURES: 0.0002
        },
        description="Half-spread costs by asset class"
    )
    slippage_linear_impact: float = Field(
        0.0001,
        description="Linear market impact coefficient",
        ge=0.0,
        le=0.01
    )
    slippage_sqrt_impact: float = Field(
        0.001,
        description="Square-root market impact coefficient",
        ge=0.0,
        le=0.1
    )
    asset_class: CostStructureType = Field(
        CostStructureType.CRYPTO,
        description="Asset class for cost modeling"
    )


class PerformanceConfig(BaseModel):
    """Configuration for performance analysis and metrics."""
    model_config = ConfigDict(extra="forbid")
    
    benchmark_symbol: Optional[str] = Field(
        None,
        description="Symbol to use as benchmark (e.g., 'BTCUSDT' for buy-and-hold)"
    )
    risk_free_rate: float = Field(
        0.02,  # 2% annual risk-free rate
        description="Annual risk-free rate for Sharpe ratio calculation",
        ge=0.0,
        le=0.2
    )
    var_confidence: float = Field(
        0.95,
        description="Confidence level for Value at Risk calculation",
        ge=0.8,
        le=0.99
    )
    regime_analysis: bool = Field(
        True,
        description="Enable regime-specific performance analysis"
    )
    attribution_analysis: bool = Field(
        True,
        description="Enable performance attribution by signal source"
    )


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward validation."""
    model_config = ConfigDict(extra="forbid")
    
    enabled: bool = Field(
        False,
        description="Enable walk-forward validation"
    )
    train_period: str = Field(
        "6M",
        description="Training period (e.g., '6M', '1Y')"
    )
    test_period: str = Field(
        "1M",
        description="Testing period (e.g., '1M', '2W')"
    )
    step_size: str = Field(
        "2W",
        description="Step size between validation windows (e.g., '1W', '2W')"
    )
    min_train_samples: int = Field(
        1000,
        description="Minimum number of samples required for training",
        ge=100
    )
    retrain_threshold: float = Field(
        0.05,
        description="Performance degradation threshold for retraining",
        ge=0.01,
        le=0.5
    )


class BacktestConfig(BaseModel):
    """Main configuration class for backtesting framework."""
    model_config = ConfigDict(extra="forbid")
    
    # Basic settings
    name: str = Field(
        "backtest",
        description="Name for this backtest configuration"
    )
    description: Optional[str] = Field(
        None,
        description="Description of the backtest"
    )
    
    # Date range
    start_date: date = Field(
        ...,
        description="Start date for backtesting"
    )
    end_date: date = Field(
        ...,
        description="End date for backtesting"
    )
    
    # Symbols to test
    symbols: List[str] = Field(
        ...,
        description="List of symbols to backtest",
        min_items=1
    )
    
    # Initial capital
    initial_capital: float = Field(
        100000.0,
        description="Initial capital for backtesting",
        gt=0
    )
    
    # Component configurations
    data_source: DataSourceConfig = Field(
        ...,
        description="Data source configuration"
    )
    signal_processing: SignalProcessingConfig = Field(
        default_factory=SignalProcessingConfig,
        description="Signal processing configuration"
    )
    position_sizing: PositionSizingConfig = Field(
        default_factory=PositionSizingConfig,
        description="Position sizing configuration"
    )
    cost_model: CostModelConfig = Field(
        default_factory=CostModelConfig,
        description="Cost modeling configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance analysis configuration"
    )
    walk_forward: WalkForwardConfig = Field(
        default_factory=WalkForwardConfig,
        description="Walk-forward validation configuration"
    )
    
    # Output settings
    output_path: Path = Field(
        Path("./backtest_results"),
        description="Directory for backtest output files"
    )
    save_trades: bool = Field(
        True,
        description="Save individual trade records"
    )
    save_portfolio_state: bool = Field(
        True,
        description="Save portfolio state over time"
    )
    generate_report: bool = Field(
        True,
        description="Generate comprehensive backtest report"
    )
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        start_date = values.get('start_date')
        if start_date and v <= start_date:
            raise ValueError("end_date must be after start_date")
        return v
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], env_override: bool = True) -> "BacktestConfig":
        """
        Load configuration from YAML file with optional environment variable overrides.
        
        Args:
            yaml_path: Path to YAML configuration file
            env_override: If True, override config values with environment variables
            
        Returns:
            BacktestConfig instance
            
        Environment variable format:
            BACKTEST_<SECTION>_<KEY> (e.g., BACKTEST_INITIAL_CAPITAL=200000)
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Apply environment variable overrides
        if env_override:
            config_dict = cls._apply_env_overrides(config_dict)
        
        return cls(**config_dict)
    
    @staticmethod
    def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables should be prefixed with BACKTEST_ and use
        uppercase with underscores (e.g., BACKTEST_INITIAL_CAPITAL).
        """
        # Top-level overrides
        env_mappings = {
            'BACKTEST_NAME': 'name',
            'BACKTEST_INITIAL_CAPITAL': ('initial_capital', float),
            'BACKTEST_START_DATE': ('start_date', lambda x: datetime.strptime(x, '%Y-%m-%d').date()),
            'BACKTEST_END_DATE': ('end_date', lambda x: datetime.strptime(x, '%Y-%m-%d').date()),
        }
        
        for env_var, mapping in env_mappings.items():
            if env_var in os.environ:
                if isinstance(mapping, tuple):
                    key, converter = mapping
                    config_dict[key] = converter(os.environ[env_var])
                else:
                    config_dict[mapping] = os.environ[env_var]
        
        # Data source overrides
        if 'data_source' in config_dict:
            ds_mappings = {
                'BACKTEST_SIGNALS_PATH': 'signals_path',
                'BACKTEST_MARKET_DATA_PATH': 'market_data_path',
                'BACKTEST_MINIO_ENDPOINT': 'minio_endpoint',
                'BACKTEST_MINIO_ACCESS_KEY': 'minio_access_key',
                'BACKTEST_MINIO_SECRET_KEY': 'minio_secret_key',
                'BACKTEST_MINIO_BUCKET': 'minio_bucket',
            }
            
            for env_var, key in ds_mappings.items():
                if env_var in os.environ:
                    config_dict['data_source'][key] = os.environ[env_var]
        
        # Position sizing overrides
        if 'position_sizing' in config_dict:
            ps_mappings = {
                'BACKTEST_POSITION_SIZING_METHOD': 'method',
                'BACKTEST_POSITION_SIZING_PERCENTAGE': ('percentage', float),
                'BACKTEST_MAX_POSITION_SIZE': ('max_position_size', float),
            }
            
            for env_var, mapping in ps_mappings.items():
                if env_var in os.environ:
                    if isinstance(mapping, tuple):
                        key, converter = mapping
                        config_dict['position_sizing'][key] = converter(os.environ[env_var])
                    else:
                        config_dict['position_sizing'][mapping] = os.environ[env_var]
        
        return config_dict
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and handle special types
        config_dict = self.model_dump(mode='json')
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def validate_paths(self) -> None:
        """
        Validate that all required paths exist.
        
        Raises:
            FileNotFoundError: If required paths don't exist
        """
        if not self.data_source.signals_path.exists():
            raise FileNotFoundError(f"Signals path not found: {self.data_source.signals_path}")
        
        if not self.data_source.market_data_path.exists():
            raise FileNotFoundError(f"Market data path not found: {self.data_source.market_data_path}")
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create cache directory if it doesn't exist
        self.data_source.local_cache_path.mkdir(parents=True, exist_ok=True)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Perform comprehensive configuration validation.
        
        Returns:
            Dictionary with validation results including warnings and errors
            
        Requirements: 1.1, 6.5
        """
        errors = []
        warnings = []
        
        # Validate date range
        if self.end_date <= self.start_date:
            errors.append(f"end_date ({self.end_date}) must be after start_date ({self.start_date})")
        
        date_range_days = (self.end_date - self.start_date).days
        if date_range_days < 7:
            warnings.append(f"Short date range: only {date_range_days} days")
        
        # Validate symbols
        if not self.symbols:
            errors.append("No symbols specified")
        
        # Validate initial capital
        if self.initial_capital <= 0:
            errors.append(f"Initial capital must be positive: {self.initial_capital}")
        elif self.initial_capital < 1000:
            warnings.append(f"Low initial capital: ${self.initial_capital:,.2f}")
        
        # Validate position sizing
        if self.position_sizing.method == PositionSizingMethod.FIXED_SIZE:
            if self.position_sizing.fixed_size is None:
                errors.append("fixed_size required when method is fixed_size")
            elif self.position_sizing.fixed_size > self.initial_capital:
                errors.append(f"fixed_size ({self.position_sizing.fixed_size}) exceeds initial capital")
        
        if self.position_sizing.method == PositionSizingMethod.PERCENTAGE:
            if self.position_sizing.percentage is None:
                errors.append("percentage required when method is percentage")
            elif self.position_sizing.percentage > 0.2:
                warnings.append(f"High position sizing percentage: {self.position_sizing.percentage:.1%}")
        
        # Validate max position size
        if self.position_sizing.max_position_size > 0.5:
            warnings.append(f"High max position size: {self.position_sizing.max_position_size:.1%}")
        
        # Validate signal processing
        if self.signal_processing.fusion_method == FusionMethod.STATIC_WEIGHTED:
            if self.signal_processing.static_weights is None:
                errors.append("static_weights required when fusion_method is static_weighted")
            else:
                # Validate weights sum to approximately 1.0
                weight_sum = sum(self.signal_processing.static_weights.values())
                if abs(weight_sum - 1.0) > 0.01:
                    warnings.append(f"Static weights sum to {weight_sum:.3f}, expected ~1.0")
        
        # Validate cost model
        if self.cost_model.commission_rate > 0.01:
            warnings.append(f"High commission rate: {self.cost_model.commission_rate:.2%}")
        
        # Validate walk-forward settings
        if self.walk_forward.enabled:
            if self.walk_forward.min_train_samples < 100:
                warnings.append(f"Low min_train_samples: {self.walk_forward.min_train_samples}")
        
        # Validate paths
        try:
            self.validate_paths()
        except FileNotFoundError as e:
            errors.append(str(e))
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'num_errors': len(errors),
            'num_warnings': len(warnings)
        }
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of the configuration.
        
        Returns:
            Formatted configuration summary
        """
        summary_lines = [
            f"Backtest Configuration: {self.name}",
            f"{'=' * 60}",
            f"Date Range: {self.start_date} to {self.end_date} ({(self.end_date - self.start_date).days} days)",
            f"Symbols: {', '.join(self.symbols)}",
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"",
            f"Position Sizing:",
            f"  Method: {self.position_sizing.method.value}",
            f"  Max Position: {self.position_sizing.max_position_size:.1%}",
            f"  Max Exposure: {self.position_sizing.max_total_exposure:.1%}",
            f"",
            f"Signal Processing:",
            f"  Fusion Method: {self.signal_processing.fusion_method.value}",
            f"  Thresholds: {self.signal_processing.thresholds}",
            f"",
            f"Cost Model:",
            f"  Asset Class: {self.cost_model.asset_class.value}",
            f"  Commission Rate: {self.cost_model.commission_rate:.3%}",
            f"  Min Commission: ${self.cost_model.min_commission:.2f}",
            f"",
            f"Performance Analysis:",
            f"  Risk-Free Rate: {self.performance.risk_free_rate:.2%}",
            f"  Regime Analysis: {'Enabled' if self.performance.regime_analysis else 'Disabled'}",
            f"  Attribution Analysis: {'Enabled' if self.performance.attribution_analysis else 'Disabled'}",
        ]
        
        if self.walk_forward.enabled:
            summary_lines.extend([
                f"",
                f"Walk-Forward Validation:",
                f"  Train Period: {self.walk_forward.train_period}",
                f"  Test Period: {self.walk_forward.test_period}",
                f"  Step Size: {self.walk_forward.step_size}",
            ])
        
        summary_lines.extend([
            f"",
            f"Output:",
            f"  Path: {self.output_path}",
            f"  Save Trades: {self.save_trades}",
            f"  Save Portfolio State: {self.save_portfolio_state}",
            f"  Generate Report: {self.generate_report}",
        ])
        
        return "\n".join(summary_lines)


# Example configuration factory functions
def create_crypto_config(
    symbols: List[str],
    start_date: date,
    end_date: date,
    signals_path: Path,
    market_data_path: Path
) -> BacktestConfig:
    """Create a default configuration for cryptocurrency backtesting."""
    return BacktestConfig(
        name="crypto_backtest",
        description="Cryptocurrency backtesting with HMM regime detection",
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        initial_capital=100000.0,
        data_source=DataSourceConfig(
            signals_path=signals_path,
            market_data_path=market_data_path
        ),
        cost_model=CostModelConfig(
            asset_class=CostStructureType.CRYPTO,
            commission_rate=0.001,  # 0.1% for crypto
            min_commission=0.1
        )
    )


def create_forex_config(
    symbols: List[str],
    start_date: date,
    end_date: date,
    signals_path: Path,
    market_data_path: Path
) -> BacktestConfig:
    """Create a default configuration for forex backtesting."""
    return BacktestConfig(
        name="forex_backtest",
        description="Forex backtesting with tight spreads",
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        initial_capital=100000.0,
        data_source=DataSourceConfig(
            signals_path=signals_path,
            market_data_path=market_data_path
        ),
        cost_model=CostModelConfig(
            asset_class=CostStructureType.FOREX,
            commission_rate=0.0002,  # 0.02% for forex
            min_commission=0.0
        ),
        position_sizing=PositionSizingConfig(
            method=PositionSizingMethod.VOLATILITY_ADJUSTED,
            volatility_target=0.10  # Lower volatility target for forex
        )
    )