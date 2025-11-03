"""
Advanced slippage modeling system for backtesting framework.

This module implements sophisticated slippage models including linear and
square-root market impact models, volatility-based adjustments, and detailed
cost breakdown analysis and reporting.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from .trade_generator import Order
from .config import CostModelConfig


logger = logging.getLogger(__name__)


class SlippageModel(str, Enum):
    """Available slippage models."""
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    COMBINED = "combined"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class SlippageEstimate:
    """Detailed slippage estimate for a trade."""
    order_id: str
    symbol: str
    timestamp: datetime
    
    # Order details
    order_size: float
    order_price: float
    notional_value: float
    
    # Market conditions
    volume: Optional[float]
    participation_rate: float
    volatility: float
    
    # Slippage components
    linear_impact: float
    sqrt_impact: float
    volatility_adjustment: float
    
    # Total slippage
    total_slippage: float
    slippage_bps: float
    
    # Execution price
    execution_price: float
    price_impact_pct: float
    
    # Model used
    model: str
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'order_size': self.order_size,
            'order_price': self.order_price,
            'notional_value': self.notional_value,
            'volume': self.volume,
            'participation_rate': self.participation_rate,
            'volatility': self.volatility,
            'linear_impact': self.linear_impact,
            'sqrt_impact': self.sqrt_impact,
            'volatility_adjustment': self.volatility_adjustment,
            'total_slippage': self.total_slippage,
            'slippage_bps': self.slippage_bps,
            'execution_price': self.execution_price,
            'price_impact_pct': self.price_impact_pct,
            'model': self.model,
            **self.metadata
        }


class SlippageModelEngine:
    """
    Advanced slippage modeling engine with multiple impact models.
    
    This class implements linear and square-root market impact models,
    volatility-based adjustments, and provides detailed cost breakdown
    analysis and reporting.
    
    Requirements: 3.2, 3.5
    """
    
    def __init__(
        self,
        config: CostModelConfig,
        model: SlippageModel = SlippageModel.COMBINED
    ):
        """
        Initialize SlippageModelEngine.
        
        Args:
            config: Cost model configuration
            model: Slippage model to use
        """
        self.config = config
        self.model = model
        
        # Track all slippage estimates
        self._estimates: List[SlippageEstimate] = []
        
        # Cache for market data
        self._volatility_cache: Dict[str, Tuple[datetime, float]] = {}
        self._volume_cache: Dict[str, Tuple[datetime, float]] = {}
        
        logger.info(f"Initialized SlippageModelEngine with {model.value} model")
    
    def estimate_slippage(
        self,
        order: Order,
        market_data: Optional[pd.DataFrame] = None,
        volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> SlippageEstimate:
        """
        Estimate slippage for an order.
        
        Args:
            order: Order to estimate slippage for
            market_data: Market data for calculations (optional)
            volume: Trading volume (optional, overrides market_data)
            volatility: Market volatility (optional, overrides calculation)
            
        Returns:
            SlippageEstimate with detailed breakdown
            
        Requirements: 3.2, 3.5
        """
        # Get market conditions
        if volume is None and market_data is not None:
            volume = self._get_volume(order.symbol, order.timestamp, market_data)
        
        if volatility is None and market_data is not None:
            volatility = self._calculate_volatility(
                order.symbol, market_data, order.timestamp
            )
        elif volatility is None:
            volatility = 0.15  # Default typical volatility
        
        # Calculate participation rate
        participation_rate = self._calculate_participation_rate(
            order.quantity, volume
        )
        
        # Calculate slippage components based on model
        if self.model == SlippageModel.LINEAR:
            linear_impact, sqrt_impact, vol_adjustment = self._linear_model(
                participation_rate, volatility
            )
        elif self.model == SlippageModel.SQUARE_ROOT:
            linear_impact, sqrt_impact, vol_adjustment = self._sqrt_model(
                participation_rate, volatility
            )
        elif self.model == SlippageModel.COMBINED:
            linear_impact, sqrt_impact, vol_adjustment = self._combined_model(
                participation_rate, volatility
            )
        else:  # VOLATILITY_ADJUSTED
            linear_impact, sqrt_impact, vol_adjustment = self._volatility_adjusted_model(
                participation_rate, volatility
            )
        
        # Calculate total slippage impact percentage
        total_impact_pct = linear_impact + sqrt_impact + vol_adjustment
        
        # Convert to dollar slippage
        notional_value = order.notional_value
        total_slippage = notional_value * total_impact_pct
        
        # Calculate slippage in basis points
        slippage_bps = total_impact_pct * 10000
        
        # Calculate execution price
        price_impact = total_slippage / abs(order.quantity) if order.quantity != 0 else 0
        if order.side == 'buy':
            execution_price = order.price + price_impact
        else:
            execution_price = order.price - price_impact
        
        # Calculate price impact percentage
        price_impact_pct = (price_impact / order.price * 100) if order.price > 0 else 0
        
        estimate = SlippageEstimate(
            order_id=order.order_id,
            symbol=order.symbol,
            timestamp=order.timestamp,
            order_size=order.quantity,
            order_price=order.price,
            notional_value=notional_value,
            volume=volume,
            participation_rate=participation_rate,
            volatility=volatility,
            linear_impact=linear_impact * notional_value,
            sqrt_impact=sqrt_impact * notional_value,
            volatility_adjustment=vol_adjustment * notional_value,
            total_slippage=total_slippage,
            slippage_bps=slippage_bps,
            execution_price=execution_price,
            price_impact_pct=price_impact_pct,
            model=self.model.value,
            metadata={
                'side': order.side,
                'asset_class': self.config.asset_class.value
            }
        )
        
        self._estimates.append(estimate)
        
        logger.debug(f"Estimated slippage for {order.symbol}: ${total_slippage:.2f} "
                    f"({slippage_bps:.2f} bps)")
        
        return estimate
    
    def _calculate_participation_rate(
        self,
        order_size: float,
        volume: Optional[float]
    ) -> float:
        """
        Calculate participation rate (order size as fraction of volume).
        
        Args:
            order_size: Size of the order
            volume: Trading volume
            
        Returns:
            Participation rate
        """
        if volume is None or volume == 0:
            # Conservative estimate if no volume data
            return 0.01  # Assume 1% of typical volume
        
        return abs(order_size) / volume
    
    def _linear_model(
        self,
        participation_rate: float,
        volatility: float
    ) -> Tuple[float, float, float]:
        """
        Linear market impact model.
        
        Impact = linear_coefficient * participation_rate
        
        Args:
            participation_rate: Order size as fraction of volume
            volatility: Market volatility
            
        Returns:
            Tuple of (linear_impact, sqrt_impact, volatility_adjustment)
            
        Requirements: 3.2
        """
        linear_impact = self.config.slippage_linear_impact * participation_rate
        sqrt_impact = 0.0
        vol_adjustment = 0.0
        
        return linear_impact, sqrt_impact, vol_adjustment
    
    def _sqrt_model(
        self,
        participation_rate: float,
        volatility: float
    ) -> Tuple[float, float, float]:
        """
        Square-root market impact model.
        
        Impact = sqrt_coefficient * sqrt(participation_rate)
        
        This model captures the non-linear relationship between order size
        and market impact, where larger orders have proportionally less impact.
        
        Args:
            participation_rate: Order size as fraction of volume
            volatility: Market volatility
            
        Returns:
            Tuple of (linear_impact, sqrt_impact, volatility_adjustment)
            
        Requirements: 3.2
        """
        linear_impact = 0.0
        sqrt_impact = self.config.slippage_sqrt_impact * np.sqrt(participation_rate)
        vol_adjustment = 0.0
        
        return linear_impact, sqrt_impact, vol_adjustment
    
    def _combined_model(
        self,
        participation_rate: float,
        volatility: float
    ) -> Tuple[float, float, float]:
        """
        Combined linear and square-root model.
        
        Impact = linear_coefficient * participation_rate + 
                 sqrt_coefficient * sqrt(participation_rate)
        
        This combines both models to capture different aspects of market impact.
        
        Args:
            participation_rate: Order size as fraction of volume
            volatility: Market volatility
            
        Returns:
            Tuple of (linear_impact, sqrt_impact, volatility_adjustment)
            
        Requirements: 3.2
        """
        linear_impact = self.config.slippage_linear_impact * participation_rate
        sqrt_impact = self.config.slippage_sqrt_impact * np.sqrt(participation_rate)
        vol_adjustment = 0.0
        
        return linear_impact, sqrt_impact, vol_adjustment
    
    def _volatility_adjusted_model(
        self,
        participation_rate: float,
        volatility: float
    ) -> Tuple[float, float, float]:
        """
        Volatility-adjusted combined model.
        
        Adjusts slippage based on market volatility. Higher volatility
        leads to higher slippage due to wider spreads and less liquidity.
        
        Args:
            participation_rate: Order size as fraction of volume
            volatility: Market volatility
            
        Returns:
            Tuple of (linear_impact, sqrt_impact, volatility_adjustment)
            
        Requirements: 3.2, 3.5
        """
        # Base impacts
        linear_impact = self.config.slippage_linear_impact * participation_rate
        sqrt_impact = self.config.slippage_sqrt_impact * np.sqrt(participation_rate)
        
        # Volatility adjustment
        # Normalize to typical volatility of 0.15 (15%)
        vol_multiplier = max(1.0, volatility / 0.15)
        
        # Apply volatility adjustment to both components
        linear_impact *= vol_multiplier
        sqrt_impact *= vol_multiplier
        
        # Calculate additional volatility-based slippage
        vol_adjustment = (vol_multiplier - 1.0) * (linear_impact + sqrt_impact)
        
        return linear_impact, sqrt_impact, vol_adjustment
    
    def _get_volume(
        self,
        symbol: str,
        timestamp: datetime,
        market_data: pd.DataFrame
    ) -> Optional[float]:
        """
        Get trading volume for a symbol at a given timestamp.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            market_data: Market data DataFrame
            
        Returns:
            Trading volume or None if not available
        """
        # Check cache first
        cache_key = f"{symbol}_{timestamp}"
        if cache_key in self._volume_cache:
            cache_time, cached_volume = self._volume_cache[cache_key]
            return cached_volume
        
        # Filter market data
        symbol_data = market_data[
            (market_data['symbol'] == symbol) &
            (market_data['timestamp'] == timestamp)
        ]
        
        if len(symbol_data) == 0:
            logger.debug(f"No volume data found for {symbol} at {timestamp}")
            return None
        
        volume = symbol_data['volume'].iloc[0]
        
        # Cache the result
        self._volume_cache[cache_key] = (timestamp, volume)
        
        return volume
    
    def _detect_data_frequency_minutes(self, market_data: pd.DataFrame) -> int:
        """
        Detect data frequency in minutes from timestamps.
        
        Args:
            market_data: Market data DataFrame with timestamp column
            
        Returns:
            Frequency in minutes, defaults to 5 if cannot detect
        """
        if len(market_data) < 2:
            return 5  # Default to 5 minutes
        
        # Sort by timestamp and calculate differences
        sorted_data = market_data.sort_values('timestamp')
        time_diffs = sorted_data['timestamp'].diff().dropna()
        
        if len(time_diffs) == 0:
            return 5
        
        # Get the most common time difference
        mode_diff = time_diffs.mode()
        if len(mode_diff) == 0:
            return 5
        
        # Convert to minutes
        frequency_minutes = int(mode_diff.iloc[0].total_seconds() / 60)
        
        # Validate reasonable range
        if frequency_minutes < 1 or frequency_minutes > 1440:
            return 5
        
        return frequency_minutes

    def _calculate_volatility(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        timestamp: datetime,
        lookback: int = 20
    ) -> float:
        """
        Calculate recent volatility for slippage adjustment.
        
        Args:
            symbol: Trading symbol
            market_data: Market data DataFrame
            timestamp: Current timestamp
            lookback: Number of periods to look back
            
        Returns:
            Annualized volatility
        """
        # Check cache first
        cache_key = f"{symbol}_{timestamp}"
        if cache_key in self._volatility_cache:
            cache_time, cached_vol = self._volatility_cache[cache_key]
            # Use cached value if less than 1 hour old
            if (timestamp - cache_time).total_seconds() < 3600:
                return cached_vol
        
        # Filter market data for the symbol
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        if len(symbol_data) < lookback:
            # Return typical volatility if insufficient data
            return 0.15  # 15% typical volatility
        
        # Sort by timestamp and get recent data
        symbol_data = symbol_data.sort_values('timestamp')
        recent_data = symbol_data[symbol_data['timestamp'] <= timestamp].tail(lookback)
        
        if len(recent_data) < 2:
            return 0.15
        
        # Calculate returns
        recent_data['returns'] = recent_data['close'].pct_change()
        
        # Calculate volatility (standard deviation of returns)
        volatility = recent_data['returns'].std()
        
        # Calculate annualization factor based on data frequency
        frequency_minutes = self._detect_data_frequency_minutes(symbol_data)
        bars_per_day = 1440 / frequency_minutes  # 1440 minutes in a day
        bars_per_year = bars_per_day * 365
        annualized_volatility = volatility * np.sqrt(bars_per_year)
        
        # Cache the result
        self._volatility_cache[cache_key] = (timestamp, annualized_volatility)
        
        return annualized_volatility
    
    def estimate_batch_slippage(
        self,
        orders: List[Order],
        market_data: Optional[pd.DataFrame] = None
    ) -> List[SlippageEstimate]:
        """
        Estimate slippage for multiple orders efficiently.
        
        Args:
            orders: List of orders
            market_data: Market data for calculations
            
        Returns:
            List of SlippageEstimate objects
        """
        estimates = []
        
        for order in orders:
            try:
                estimate = self.estimate_slippage(order, market_data)
                estimates.append(estimate)
            except Exception as e:
                logger.error(f"Error estimating slippage for order {order.order_id}: {e}")
                continue
        
        logger.info(f"Estimated slippage for {len(estimates)} orders")
        return estimates
    
    def get_slippage_breakdown(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get detailed slippage breakdown with optional filtering.
        
        Args:
            symbol: Filter by symbol (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            
        Returns:
            Dictionary with slippage breakdown analysis
            
        Requirements: 3.5
        """
        # Filter estimates
        filtered_estimates = self._estimates
        
        if symbol:
            filtered_estimates = [e for e in filtered_estimates if e.symbol == symbol]
        
        if start_time:
            filtered_estimates = [e for e in filtered_estimates if e.timestamp >= start_time]
        
        if end_time:
            filtered_estimates = [e for e in filtered_estimates if e.timestamp <= end_time]
        
        if not filtered_estimates:
            return {
                'num_trades': 0,
                'total_slippage': 0.0,
                'avg_slippage_bps': 0.0,
                'components': {}
            }
        
        # Calculate aggregates
        total_slippage = sum(e.total_slippage for e in filtered_estimates)
        total_notional = sum(e.notional_value for e in filtered_estimates)
        
        total_linear = sum(e.linear_impact for e in filtered_estimates)
        total_sqrt = sum(e.sqrt_impact for e in filtered_estimates)
        total_vol_adj = sum(e.volatility_adjustment for e in filtered_estimates)
        
        avg_slippage_bps = (total_slippage / total_notional * 10000) if total_notional > 0 else 0
        
        # Component breakdown
        components = {
            'linear_impact': {
                'total': total_linear,
                'percentage': (total_linear / total_slippage * 100) if total_slippage > 0 else 0,
                'avg_per_trade': total_linear / len(filtered_estimates)
            },
            'sqrt_impact': {
                'total': total_sqrt,
                'percentage': (total_sqrt / total_slippage * 100) if total_slippage > 0 else 0,
                'avg_per_trade': total_sqrt / len(filtered_estimates)
            },
            'volatility_adjustment': {
                'total': total_vol_adj,
                'percentage': (total_vol_adj / total_slippage * 100) if total_slippage > 0 else 0,
                'avg_per_trade': total_vol_adj / len(filtered_estimates)
            }
        }
        
        # Participation rate statistics
        participation_rates = [e.participation_rate for e in filtered_estimates]
        volatilities = [e.volatility for e in filtered_estimates]
        
        return {
            'num_trades': len(filtered_estimates),
            'total_slippage': total_slippage,
            'total_notional': total_notional,
            'avg_slippage_bps': avg_slippage_bps,
            'avg_slippage_per_trade': total_slippage / len(filtered_estimates),
            'components': components,
            'avg_participation_rate': np.mean(participation_rates),
            'max_participation_rate': max(participation_rates),
            'avg_volatility': np.mean(volatilities),
            'max_volatility': max(volatilities),
            'model': self.model.value
        }
    
    def get_slippage_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        """
        Get slippage breakdown by symbol.
        
        Returns:
            Dictionary with slippage breakdown for each symbol
        """
        symbols = set(e.symbol for e in self._estimates)
        
        breakdown = {}
        for symbol in symbols:
            breakdown[symbol] = self.get_slippage_breakdown(symbol=symbol)
        
        return breakdown
    
    def get_slippage_time_series(self) -> pd.DataFrame:
        """
        Get slippage data as a time series DataFrame.
        
        Returns:
            DataFrame with slippage data over time
        """
        if not self._estimates:
            return pd.DataFrame()
        
        data = [estimate.to_dict() for estimate in self._estimates]
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def compare_models(
        self,
        order: Order,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, SlippageEstimate]:
        """
        Compare slippage estimates across different models.
        
        Args:
            order: Order to estimate slippage for
            market_data: Market data for calculations
            
        Returns:
            Dictionary mapping model names to estimates
        """
        original_model = self.model
        estimates = {}
        
        for model in SlippageModel:
            self.model = model
            estimate = self.estimate_slippage(order, market_data)
            estimates[model.value] = estimate
        
        # Restore original model
        self.model = original_model
        
        return estimates
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall slippage statistics.
        
        Returns:
            Dictionary with slippage statistics
        """
        if not self._estimates:
            return {
                'num_trades': 0,
                'total_slippage': 0.0,
                'avg_slippage_bps': 0.0
            }
        
        breakdown = self.get_slippage_breakdown()
        
        # Add additional statistics
        slippage_bps_list = [e.slippage_bps for e in self._estimates]
        price_impact_list = [e.price_impact_pct for e in self._estimates]
        
        breakdown.update({
            'min_slippage_bps': min(slippage_bps_list),
            'max_slippage_bps': max(slippage_bps_list),
            'median_slippage_bps': np.median(slippage_bps_list),
            'std_slippage_bps': np.std(slippage_bps_list),
            'avg_price_impact_pct': np.mean(price_impact_list),
            'max_price_impact_pct': max(price_impact_list)
        })
        
        return breakdown
    
    def reset_state(self) -> None:
        """Reset all internal state (useful for testing)."""
        self._estimates.clear()
        self._volatility_cache.clear()
        self._volume_cache.clear()
        logger.info("Reset SlippageModelEngine state")
