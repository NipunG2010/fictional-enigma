"""
Transaction cost and slippage modeling for backtesting framework.

This module implements the CostModel class that applies realistic transaction
costs including commissions, spreads, and market impact to simulated trades.
It supports different cost structures for various asset classes and provides
detailed cost breakdown analysis.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from .trade_generator import Order
from .config import CostModelConfig, CostStructureType


logger = logging.getLogger(__name__)


@dataclass
class TradeCost:
    """Detailed breakdown of costs for a single trade."""
    order_id: str
    symbol: str
    timestamp: datetime
    notional_value: float
    
    # Cost components
    commission: float
    spread_cost: float
    market_impact: float
    slippage: float
    
    # Total costs
    total_cost: float
    cost_bps: float  # Cost in basis points
    
    # Execution details
    order_price: float
    execution_price: float
    price_impact_pct: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'notional_value': self.notional_value,
            'commission': self.commission,
            'spread_cost': self.spread_cost,
            'market_impact': self.market_impact,
            'slippage': self.slippage,
            'total_cost': self.total_cost,
            'cost_bps': self.cost_bps,
            'order_price': self.order_price,
            'execution_price': self.execution_price,
            'price_impact_pct': self.price_impact_pct,
            **self.metadata
        }


class CostModel:
    """
    Applies realistic transaction costs and slippage to trades.
    
    This class implements commission calculation with configurable rate structures,
    spread cost modeling for different asset classes, and market impact estimation
    based on order size and volume.
    
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    
    def __init__(self, config: CostModelConfig):
        """
        Initialize CostModel.
        
        Args:
            config: Cost model configuration
        """
        self.config = config
        
        # Track all calculated costs
        self._trade_costs: List[TradeCost] = []
        
        # Cache for volume data
        self._volume_cache: Dict[str, Tuple[datetime, float]] = {}
        
        logger.info(f"Initialized CostModel for {config.asset_class.value} with "
                   f"commission rate: {config.commission_rate*100:.3f}%")
    
    def calculate_trade_cost(
        self,
        order: Order,
        market_data: Optional[pd.DataFrame] = None,
        volume: Optional[float] = None
    ) -> TradeCost:
        """
        Calculate total transaction cost for an order.
        
        Args:
            order: Order to calculate costs for
            market_data: Market data for volume-based calculations (optional)
            volume: Trading volume for the period (optional, overrides market_data)
            
        Returns:
            TradeCost object with detailed cost breakdown
            
        Requirements: 3.1, 3.2, 3.3, 3.4
        """
        notional_value = order.notional_value
        
        # Calculate commission
        commission = self._calculate_commission(notional_value)
        
        # Calculate spread cost
        spread_cost = self._calculate_spread_cost(
            symbol=order.symbol,
            notional_value=notional_value
        )
        
        # Calculate market impact
        if volume is None and market_data is not None:
            volume = self._get_volume(order.symbol, order.timestamp, market_data)
        
        market_impact = self._calculate_market_impact(
            order_size=order.quantity,
            price=order.price,
            volume=volume
        )
        
        # Calculate slippage (combination of spread and market impact)
        slippage = self._calculate_slippage(
            order_size=order.quantity,
            price=order.price,
            volume=volume,
            market_data=market_data,
            symbol=order.symbol,
            timestamp=order.timestamp
        )
        
        # Calculate total cost
        total_cost = commission + spread_cost + market_impact + slippage
        
        # Calculate execution price (order price + slippage impact)
        price_impact = (market_impact + slippage) / abs(order.quantity) if order.quantity != 0 else 0
        if order.side == 'buy':
            execution_price = order.price + price_impact
        else:
            execution_price = order.price - price_impact
        
        # Calculate cost in basis points
        cost_bps = (total_cost / notional_value * 10000) if notional_value > 0 else 0
        
        # Calculate price impact percentage
        price_impact_pct = (price_impact / order.price * 100) if order.price > 0 else 0
        
        trade_cost = TradeCost(
            order_id=order.order_id,
            symbol=order.symbol,
            timestamp=order.timestamp,
            notional_value=notional_value,
            commission=commission,
            spread_cost=spread_cost,
            market_impact=market_impact,
            slippage=slippage,
            total_cost=total_cost,
            cost_bps=cost_bps,
            order_price=order.price,
            execution_price=execution_price,
            price_impact_pct=price_impact_pct,
            metadata={
                'side': order.side,
                'quantity': order.quantity,
                'asset_class': self.config.asset_class.value,
                'volume': volume
            }
        )
        
        self._trade_costs.append(trade_cost)
        
        logger.debug(f"Calculated trade cost for {order.symbol}: ${total_cost:.2f} "
                    f"({cost_bps:.2f} bps)")
        
        return trade_cost
    
    def _calculate_commission(self, notional_value: float) -> float:
        """
        Calculate commission with configurable rate structure.
        
        Args:
            notional_value: Notional value of the trade
            
        Returns:
            Commission amount
            
        Requirements: 3.1, 3.4
        """
        # Calculate percentage-based commission
        commission = notional_value * self.config.commission_rate
        
        # Apply minimum commission
        commission = max(commission, self.config.min_commission)
        
        return commission
    
    def _calculate_spread_cost(
        self,
        symbol: str,
        notional_value: float
    ) -> float:
        """
        Calculate spread cost for different asset classes.
        
        The spread cost represents the cost of crossing the bid-ask spread.
        We use half-spread as the cost since we assume execution at mid-price.
        
        Args:
            symbol: Trading symbol
            notional_value: Notional value of the trade
            
        Returns:
            Spread cost amount
            
        Requirements: 3.1, 3.3
        """
        # Get spread rate for the asset class
        spread_rate = self.config.spread_costs.get(
            self.config.asset_class,
            0.0001  # Default 1 bps
        )
        
        # Calculate spread cost
        spread_cost = notional_value * spread_rate
        
        return spread_cost
    
    def _calculate_market_impact(
        self,
        order_size: float,
        price: float,
        volume: Optional[float] = None
    ) -> float:
        """
        Calculate market impact based on order size and volume.
        
        Market impact is the price movement caused by the order itself.
        We use a linear model: impact = coefficient * (order_size / volume)
        
        Args:
            order_size: Size of the order
            price: Order price
            volume: Trading volume for the period
            
        Returns:
            Market impact cost
            
        Requirements: 3.2, 3.3
        """
        if volume is None or volume == 0:
            # If no volume data, use a conservative estimate
            # Assume order is 1% of typical volume
            participation_rate = 0.01
        else:
            # Calculate participation rate (order size as fraction of volume)
            participation_rate = abs(order_size) / volume
        
        # Calculate market impact using linear model
        # impact_pct = linear_coefficient * participation_rate
        impact_pct = self.config.slippage_linear_impact * participation_rate
        
        # Convert to dollar impact
        notional_value = abs(order_size * price)
        market_impact = notional_value * impact_pct
        
        return market_impact
    
    def _calculate_slippage(
        self,
        order_size: float,
        price: float,
        volume: Optional[float] = None,
        market_data: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Calculate slippage based on order size and market volatility.
        
        Slippage represents the difference between expected and actual execution price
        due to market conditions. We use both linear and square-root impact models.
        
        Args:
            order_size: Size of the order
            price: Order price
            volume: Trading volume for the period
            market_data: Market data for volatility calculations
            symbol: Trading symbol
            timestamp: Order timestamp
            
        Returns:
            Slippage cost
            
        Requirements: 3.2, 3.5
        """
        notional_value = abs(order_size * price)
        
        # Calculate participation rate
        if volume is None or volume == 0:
            participation_rate = 0.01  # Conservative estimate
        else:
            participation_rate = abs(order_size) / volume
        
        # Linear impact component
        linear_impact = self.config.slippage_linear_impact * participation_rate
        
        # Square-root impact component (captures non-linear market impact)
        sqrt_impact = self.config.slippage_sqrt_impact * np.sqrt(participation_rate)
        
        # Volatility adjustment (if market data available)
        volatility_multiplier = 1.0
        if market_data is not None and symbol is not None and timestamp is not None:
            volatility = self._calculate_volatility(symbol, market_data, timestamp)
            # Increase slippage in high volatility environments
            # Normalize to typical volatility of 0.15 (15%)
            volatility_multiplier = max(1.0, volatility / 0.15)
        
        # Combine impact components
        total_impact_pct = (linear_impact + sqrt_impact) * volatility_multiplier
        
        # Convert to dollar slippage
        slippage = notional_value * total_impact_pct
        
        return slippage
    
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
        
        # Filter market data for the symbol and timestamp
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
        # Filter market data for the symbol
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        if len(symbol_data) < lookback:
            # Return typical volatility if insufficient data
            return 0.15  # 15% typical volatility
        
        # Sort by timestamp and get recent data
        symbol_data = symbol_data.sort_values('timestamp')
        
        # Get data up to the current timestamp
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
        
        return annualized_volatility
    
    def calculate_batch_costs(
        self,
        orders: List[Order],
        market_data: Optional[pd.DataFrame] = None
    ) -> List[TradeCost]:
        """
        Calculate costs for multiple orders efficiently.
        
        Args:
            orders: List of orders
            market_data: Market data for calculations
            
        Returns:
            List of TradeCost objects
        """
        trade_costs = []
        
        for order in orders:
            try:
                cost = self.calculate_trade_cost(order, market_data)
                trade_costs.append(cost)
            except Exception as e:
                logger.error(f"Error calculating cost for order {order.order_id}: {e}")
                continue
        
        logger.info(f"Calculated costs for {len(trade_costs)} orders")
        return trade_costs
    
    def get_cost_breakdown(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get detailed cost breakdown with optional filtering.
        
        Args:
            symbol: Filter by symbol (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            
        Returns:
            Dictionary with cost breakdown analysis
            
        Requirements: 3.4, 3.5
        """
        # Filter trade costs
        filtered_costs = self._trade_costs
        
        if symbol:
            filtered_costs = [c for c in filtered_costs if c.symbol == symbol]
        
        if start_time:
            filtered_costs = [c for c in filtered_costs if c.timestamp >= start_time]
        
        if end_time:
            filtered_costs = [c for c in filtered_costs if c.timestamp <= end_time]
        
        if not filtered_costs:
            return {
                'num_trades': 0,
                'total_cost': 0.0,
                'avg_cost_bps': 0.0,
                'cost_components': {}
            }
        
        # Calculate aggregates
        total_cost = sum(c.total_cost for c in filtered_costs)
        total_notional = sum(c.notional_value for c in filtered_costs)
        
        total_commission = sum(c.commission for c in filtered_costs)
        total_spread = sum(c.spread_cost for c in filtered_costs)
        total_impact = sum(c.market_impact for c in filtered_costs)
        total_slippage = sum(c.slippage for c in filtered_costs)
        
        avg_cost_bps = (total_cost / total_notional * 10000) if total_notional > 0 else 0
        
        # Cost component breakdown
        cost_components = {
            'commission': {
                'total': total_commission,
                'percentage': (total_commission / total_cost * 100) if total_cost > 0 else 0,
                'avg_per_trade': total_commission / len(filtered_costs)
            },
            'spread': {
                'total': total_spread,
                'percentage': (total_spread / total_cost * 100) if total_cost > 0 else 0,
                'avg_per_trade': total_spread / len(filtered_costs)
            },
            'market_impact': {
                'total': total_impact,
                'percentage': (total_impact / total_cost * 100) if total_cost > 0 else 0,
                'avg_per_trade': total_impact / len(filtered_costs)
            },
            'slippage': {
                'total': total_slippage,
                'percentage': (total_slippage / total_cost * 100) if total_cost > 0 else 0,
                'avg_per_trade': total_slippage / len(filtered_costs)
            }
        }
        
        return {
            'num_trades': len(filtered_costs),
            'total_cost': total_cost,
            'total_notional': total_notional,
            'avg_cost_bps': avg_cost_bps,
            'avg_cost_per_trade': total_cost / len(filtered_costs),
            'cost_components': cost_components,
            'cost_to_notional_ratio': (total_cost / total_notional) if total_notional > 0 else 0
        }
    
    def get_cost_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cost breakdown by symbol.
        
        Returns:
            Dictionary with cost breakdown for each symbol
        """
        symbols = set(c.symbol for c in self._trade_costs)
        
        breakdown = {}
        for symbol in symbols:
            breakdown[symbol] = self.get_cost_breakdown(symbol=symbol)
        
        return breakdown
    
    def get_cost_time_series(self) -> pd.DataFrame:
        """
        Get cost data as a time series DataFrame.
        
        Returns:
            DataFrame with cost data over time
        """
        if not self._trade_costs:
            return pd.DataFrame()
        
        data = [cost.to_dict() for cost in self._trade_costs]
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall cost statistics.
        
        Returns:
            Dictionary with cost statistics
        """
        if not self._trade_costs:
            return {
                'num_trades': 0,
                'total_cost': 0.0,
                'avg_cost_bps': 0.0
            }
        
        breakdown = self.get_cost_breakdown()
        
        # Add additional statistics
        cost_bps_list = [c.cost_bps for c in self._trade_costs]
        price_impact_list = [c.price_impact_pct for c in self._trade_costs]
        
        breakdown.update({
            'min_cost_bps': min(cost_bps_list),
            'max_cost_bps': max(cost_bps_list),
            'median_cost_bps': np.median(cost_bps_list),
            'std_cost_bps': np.std(cost_bps_list),
            'avg_price_impact_pct': np.mean(price_impact_list),
            'max_price_impact_pct': max(price_impact_list)
        })
        
        return breakdown
    
    def reset_state(self) -> None:
        """Reset all internal state (useful for testing)."""
        self._trade_costs.clear()
        self._volume_cache.clear()
        logger.info("Reset CostModel state")
