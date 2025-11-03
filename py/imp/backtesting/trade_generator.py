"""
Trade generation and position sizing for backtesting framework.

This module implements the TradeGenerator class that converts trade signals
into sized orders with proper risk management, position limits, and exposure controls.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

import pandas as pd
import numpy as np

from .trade_signal_generator import TradeSignal, TradeAction
from .config import PositionSizingConfig, PositionSizingMethod


logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Generated order with position sizing and metadata."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str  # 'market', 'limit', etc.
    trade_signal: TradeSignal
    position_size_method: str
    risk_checks_passed: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of the order."""
        return abs(self.quantity * self.price)
    
    def __post_init__(self):
        """Validate order after initialization."""
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"Order price must be positive, got {self.price}")


class TradeGenerator:
    """
    Generates sized trades with risk management and position limits.
    
    This class implements multiple position sizing methods (fixed size, percentage,
    volatility-adjusted, Kelly criterion) and enforces risk management rules for
    position limits and exposure controls.
    
    Requirements: 2.1, 2.2, 2.4
    """
    
    def __init__(
        self,
        config: PositionSizingConfig,
        initial_capital: float
    ):
        """
        Initialize TradeGenerator.
        
        Args:
            config: Position sizing configuration
            initial_capital: Initial capital for position sizing calculations
        """
        self.config = config
        self.initial_capital = initial_capital
        
        # Track generated orders
        self._orders: List[Order] = []
        
        # Cache for volatility calculations
        self._volatility_cache: Dict[str, Tuple[datetime, float]] = {}
        
        logger.info(f"Initialized TradeGenerator with method: {config.method.value}")
    
    def generate_orders(
        self,
        trade_signals: List[TradeSignal],
        market_data: pd.DataFrame,
        portfolio_value: float,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[Order]:
        """
        Generate sized orders from trade signals.
        
        Args:
            trade_signals: List of trade signals to convert to orders
            market_data: Market data for volatility calculations
            portfolio_value: Current portfolio value
            current_positions: Current position sizes by symbol
            current_prices: Current prices by symbol
            
        Returns:
            List of generated orders
            
        Requirements: 2.1, 2.2, 2.4
        """
        if not trade_signals:
            return []
        
        orders = []
        
        for signal in trade_signals:
            try:
                # Get current price for the symbol
                if signal.symbol not in current_prices:
                    logger.warning(f"No price available for {signal.symbol}, skipping order generation")
                    continue
                
                current_price = current_prices[signal.symbol]
                current_position = current_positions.get(signal.symbol, 0.0)
                
                # Calculate position size
                position_size = self._calculate_position_size(
                    signal=signal,
                    market_data=market_data,
                    portfolio_value=portfolio_value,
                    current_price=current_price,
                    current_position=current_position
                )
                
                if position_size == 0:
                    logger.debug(f"Position size is zero for {signal.symbol}, skipping order")
                    continue
                
                # Apply risk management checks
                risk_checks = self._apply_risk_checks(
                    symbol=signal.symbol,
                    position_size=position_size,
                    current_price=current_price,
                    portfolio_value=portfolio_value,
                    current_positions=current_positions,
                    current_prices=current_prices
                )
                
                if not all(risk_checks.values()):
                    logger.warning(f"Risk checks failed for {signal.symbol}: {risk_checks}")
                    continue
                
                # Determine order side based on trade action
                side = self._determine_order_side(signal.action, current_position)
                
                # Calculate order quantity
                quantity = self._calculate_order_quantity(
                    signal=signal,
                    position_size=position_size,
                    current_position=current_position,
                    current_price=current_price
                )
                
                if quantity <= 0:
                    logger.debug(f"Order quantity is zero or negative for {signal.symbol}, skipping")
                    continue
                
                # Generate order
                order = self._create_order(
                    signal=signal,
                    side=side,
                    quantity=quantity,
                    price=current_price,
                    risk_checks=risk_checks
                )
                
                orders.append(order)
                self._orders.append(order)
                
                logger.info(f"Generated order: {side} {quantity:.4f} {signal.symbol} @ {current_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error generating order for {signal.symbol}: {e}")
                continue
        
        logger.info(f"Generated {len(orders)} orders from {len(trade_signals)} trade signals")
        return orders
    
    def _calculate_position_size(
        self,
        signal: TradeSignal,
        market_data: pd.DataFrame,
        portfolio_value: float,
        current_price: float,
        current_position: float
    ) -> float:
        """
        Calculate position size based on configured method.
        
        Args:
            signal: Trade signal
            market_data: Market data for calculations
            portfolio_value: Current portfolio value
            current_price: Current price of the symbol
            current_position: Current position size
            
        Returns:
            Position size in dollars
            
        Requirements: 2.1, 2.2
        """
        method = self.config.method
        
        if method == PositionSizingMethod.FIXED_SIZE:
            return self._fixed_size_position(signal)
        
        elif method == PositionSizingMethod.PERCENTAGE:
            return self._percentage_position(signal, portfolio_value)
        
        elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted_position(
                signal, market_data, portfolio_value, current_price
            )
        
        elif method == PositionSizingMethod.KELLY_CRITERION:
            return self._kelly_criterion_position(
                signal, portfolio_value
            )
        
        else:
            logger.error(f"Unknown position sizing method: {method}")
            return 0.0
    
    def _fixed_size_position(self, signal: TradeSignal) -> float:
        """
        Calculate fixed dollar amount position size.
        
        Requirements: 2.1
        """
        if self.config.fixed_size is None:
            logger.error("Fixed size not configured")
            return 0.0
        
        # Adjust by signal confidence
        base_size = self.config.fixed_size
        adjusted_size = base_size * signal.confidence
        
        return adjusted_size
    
    def _percentage_position(
        self,
        signal: TradeSignal,
        portfolio_value: float
    ) -> float:
        """
        Calculate percentage of portfolio position size.
        
        Requirements: 2.1
        """
        if self.config.percentage is None:
            logger.error("Percentage not configured")
            return 0.0
        
        # Base position size as percentage of portfolio
        base_size = portfolio_value * self.config.percentage
        
        # Adjust by signal confidence
        adjusted_size = base_size * signal.confidence
        
        return adjusted_size
    
    def _volatility_adjusted_position(
        self,
        signal: TradeSignal,
        market_data: pd.DataFrame,
        portfolio_value: float,
        current_price: float
    ) -> float:
        """
        Calculate volatility-adjusted position size.
        
        Uses ATR (Average True Range) to adjust position size based on
        market volatility, targeting a specific volatility level.
        
        Requirements: 2.1
        """
        if self.config.volatility_target is None:
            logger.error("Volatility target not configured")
            return 0.0
        
        # Calculate volatility for the symbol
        volatility = self._calculate_volatility(
            symbol=signal.symbol,
            market_data=market_data,
            timestamp=signal.timestamp
        )
        
        if volatility == 0:
            logger.warning(f"Zero volatility for {signal.symbol}, using percentage method")
            return self._percentage_position(signal, portfolio_value)
        
        # Calculate position size to achieve target volatility
        # position_size = (portfolio_value * target_volatility) / asset_volatility
        target_vol = self.config.volatility_target
        position_size = (portfolio_value * target_vol) / volatility
        
        # Adjust by signal confidence
        adjusted_size = position_size * signal.confidence
        
        return adjusted_size
    
    def _kelly_criterion_position(
        self,
        signal: TradeSignal,
        portfolio_value: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly fraction = (p * b - q) / b
        where p = probability of win, q = probability of loss, b = win/loss ratio
        
        We use signal confidence as a proxy for win probability.
        
        Requirements: 2.1
        """
        # Use signal confidence as win probability
        win_prob = signal.confidence
        loss_prob = 1.0 - win_prob
        
        # Assume 2:1 reward/risk ratio (can be made configurable)
        win_loss_ratio = 2.0
        
        # Calculate Kelly fraction
        kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        
        # Apply fractional Kelly (use 25% of full Kelly for safety)
        fractional_kelly = 0.25
        kelly_fraction *= fractional_kelly
        
        # Ensure non-negative
        kelly_fraction = max(0.0, kelly_fraction)
        
        # Calculate position size
        position_size = portfolio_value * kelly_fraction
        
        return position_size
    
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
        timestamp: datetime
    ) -> float:
        """
        Calculate volatility (ATR) for a symbol.
        
        Args:
            symbol: Symbol to calculate volatility for
            market_data: Market data DataFrame
            timestamp: Current timestamp
            
        Returns:
            Annualized volatility
        """
        # Check cache first
        if symbol in self._volatility_cache:
            cache_time, cached_vol = self._volatility_cache[symbol]
            # Use cached value if less than 1 hour old
            if (timestamp - cache_time).total_seconds() < 3600:
                return cached_vol
        
        # Filter market data for the symbol
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        if len(symbol_data) < self.config.volatility_lookback:
            logger.warning(f"Insufficient data for volatility calculation for {symbol}")
            return 0.0
        
        # Sort by timestamp and get recent data
        symbol_data = symbol_data.sort_values('timestamp')
        recent_data = symbol_data.tail(self.config.volatility_lookback).copy()
        
        # Calculate True Range
        recent_data['high_low'] = recent_data['high'] - recent_data['low']
        recent_data['high_close'] = abs(recent_data['high'] - recent_data['close'].shift(1))
        recent_data['low_close'] = abs(recent_data['low'] - recent_data['close'].shift(1))
        
        recent_data['true_range'] = recent_data[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate ATR (Average True Range)
        atr = recent_data['true_range'].mean()
        
        # Get current price for normalization
        current_price = recent_data['close'].iloc[-1]
        
        # Calculate percentage volatility
        pct_volatility = atr / current_price if current_price > 0 else 0.0
        
        # Calculate annualization factor based on data frequency
        if hasattr(self.config, 'data_frequency_minutes'):
            frequency_minutes = self.config.data_frequency_minutes
        else:
            # Fallback: detect from data
            frequency_minutes = self._detect_data_frequency_minutes(symbol_data)
        
        bars_per_day = 1440 / frequency_minutes  # 1440 minutes in a day
        bars_per_year = bars_per_day * 365
        annualized_volatility = pct_volatility * np.sqrt(bars_per_year)
        
        # Cache the result
        self._volatility_cache[symbol] = (timestamp, annualized_volatility)
        
        return annualized_volatility
    
    def _apply_risk_checks(
        self,
        symbol: str,
        position_size: float,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Apply risk management checks to position size.
        
        Args:
            symbol: Symbol being traded
            position_size: Proposed position size in dollars
            current_price: Current price
            portfolio_value: Current portfolio value
            current_positions: Current positions by symbol
            current_prices: Current prices by symbol
            
        Returns:
            Dictionary of risk check results
            
        Requirements: 2.2, 2.4
        """
        checks = {}
        
        # Check maximum position size
        max_position_value = portfolio_value * self.config.max_position_size
        checks['max_position_size'] = position_size <= max_position_value
        
        if not checks['max_position_size']:
            logger.debug(f"Position size {position_size:.2f} exceeds max {max_position_value:.2f} for {symbol}")
        
        # Check maximum total exposure
        current_exposure = sum(
            abs(pos * current_prices.get(sym, 0))
            for sym, pos in current_positions.items()
        )
        
        new_exposure = current_exposure + position_size
        max_exposure = portfolio_value * self.config.max_total_exposure
        checks['max_total_exposure'] = new_exposure <= max_exposure
        
        if not checks['max_total_exposure']:
            logger.debug(f"New exposure {new_exposure:.2f} exceeds max {max_exposure:.2f}")
        
        # Check minimum position size (avoid dust trades)
        min_position_value = 10.0  # $10 minimum
        checks['min_position_size'] = position_size >= min_position_value
        
        if not checks['min_position_size']:
            logger.debug(f"Position size {position_size:.2f} below minimum {min_position_value:.2f} for {symbol}")
        
        # Check portfolio value is positive
        checks['positive_portfolio_value'] = portfolio_value > 0
        
        return checks
    
    def _determine_order_side(
        self,
        action: TradeAction,
        current_position: float
    ) -> str:
        """
        Determine order side (buy/sell) based on action and current position.
        
        Args:
            action: Trade action
            current_position: Current position size
            
        Returns:
            Order side ('buy' or 'sell')
        """
        if action == TradeAction.BUY:
            return 'buy'
        elif action == TradeAction.SELL:
            return 'sell'
        elif action == TradeAction.CLOSE_LONG:
            return 'sell'  # Sell to close long
        elif action == TradeAction.CLOSE_SHORT:
            return 'buy'  # Buy to close short
        else:
            # Default to buy for HOLD (shouldn't happen)
            return 'buy'
    
    def _calculate_order_quantity(
        self,
        signal: TradeSignal,
        position_size: float,
        current_position: float,
        current_price: float
    ) -> float:
        """
        Calculate order quantity based on position size and current position.
        
        Args:
            signal: Trade signal
            position_size: Target position size in dollars
            current_position: Current position quantity
            current_price: Current price
            
        Returns:
            Order quantity
        """
        # Convert position size to quantity
        target_quantity = position_size / current_price
        
        # Handle different trade actions
        if signal.action == TradeAction.BUY:
            # Opening or adding to long position
            quantity = target_quantity
        
        elif signal.action == TradeAction.SELL:
            # Opening or adding to short position
            quantity = target_quantity
        
        elif signal.action == TradeAction.CLOSE_LONG:
            # Closing long position - sell current position
            quantity = abs(current_position)
        
        elif signal.action == TradeAction.CLOSE_SHORT:
            # Closing short position - buy back current position
            quantity = abs(current_position)
        
        else:
            quantity = 0.0
        
        return quantity
    
    def _get_regime_state_safely(self, signal: TradeSignal) -> Optional[str]:
        """
        Safely retrieve regime_state from signal, handling missing attributes.
        
        Args:
            signal: Trade signal that may or may not have processed_signal.regime_state
            
        Returns:
            Regime state string if available, None otherwise
        """
        try:
            # First check if processed_signal exists
            processed_signal = getattr(signal, 'processed_signal', None)
            if processed_signal is None:
                return None
            
            # Then check if regime_state exists on processed_signal
            regime_state = getattr(processed_signal, 'regime_state', None)
            return regime_state
            
        except (AttributeError, TypeError):
            # Handle any unexpected attribute access issues
            return None
    
    def _create_order(
        self,
        signal: TradeSignal,
        side: str,
        quantity: float,
        price: float,
        risk_checks: Dict[str, bool]
    ) -> Order:
        """
        Create an order with proper timestamps and metadata.
        
        Args:
            signal: Trade signal
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Order price
            risk_checks: Risk check results
            
        Returns:
            Generated order
            
        Requirements: 2.1, 2.4
        """
        order_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        order = Order(
            order_id=order_id,
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type='market',  # Default to market orders
            trade_signal=signal,
            position_size_method=self.config.method.value,
            risk_checks_passed=risk_checks,
            metadata={
                'trade_id': signal.trade_id,
                'signal_confidence': signal.confidence,
                'signal_source': signal.signal_source,
                'regime_state': self._get_regime_state_safely(signal),
                'notional_value': quantity * price
            }
        )
        
        return order
    
    def get_orders(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Order]:
        """
        Get generated orders with optional filtering.
        
        Args:
            symbol: Filter by symbol (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            
        Returns:
            List of orders
        """
        filtered_orders = self._orders
        
        if symbol:
            filtered_orders = [o for o in filtered_orders if o.symbol == symbol]
        
        if start_time:
            filtered_orders = [o for o in filtered_orders if o.timestamp >= start_time]
        
        if end_time:
            filtered_orders = [o for o in filtered_orders if o.timestamp <= end_time]
        
        return filtered_orders
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated orders.
        
        Returns:
            Dictionary with order statistics
        """
        if not self._orders:
            return {
                'total_orders': 0,
                'buy_orders': 0,
                'sell_orders': 0,
                'total_notional': 0.0,
                'avg_order_size': 0.0,
                'symbols_traded': []
            }
        
        buy_orders = [o for o in self._orders if o.side == 'buy']
        sell_orders = [o for o in self._orders if o.side == 'sell']
        
        total_notional = sum(o.notional_value for o in self._orders)
        avg_order_size = total_notional / len(self._orders)
        
        symbols_traded = list(set(o.symbol for o in self._orders))
        
        return {
            'total_orders': len(self._orders),
            'buy_orders': len(buy_orders),
            'sell_orders': len(sell_orders),
            'total_notional': total_notional,
            'avg_order_size': avg_order_size,
            'symbols_traded': symbols_traded,
            'position_sizing_method': self.config.method.value
        }
    
    def reset_state(self) -> None:
        """Reset all internal state (useful for testing)."""
        self._orders.clear()
        self._volatility_cache.clear()
        logger.info("Reset TradeGenerator state")
