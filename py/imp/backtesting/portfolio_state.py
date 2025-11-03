"""
Portfolio state management and position tracking for backtesting framework.

This module implements the PortfolioState class that tracks cash and positions
over time, handles position carry-forward logic for multi-period backtests, and
provides position reconciliation and validation methods.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from copy import deepcopy

import pandas as pd
import numpy as np

from .trade_generator import Order


logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a single symbol."""
    symbol: str
    quantity: float  # Positive for long, negative for short
    avg_entry_price: float
    current_price: float
    timestamp: datetime
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of the position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Calculate cost basis of the position."""
        return self.quantity * self.avg_entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / abs(self.cost_basis)) * 100
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    def __repr__(self) -> str:
        direction = "LONG" if self.is_long else "SHORT"
        return (f"Position({self.symbol} {direction} {abs(self.quantity):.4f} @ "
                f"{self.avg_entry_price:.2f}, PnL: {self.unrealized_pnl:.2f})")


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    timestamp: datetime
    cash: float
    positions: Dict[str, Position]
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    
    @property
    def market_value(self) -> float:
        """Calculate total market value of positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def num_positions(self) -> int:
        """Get number of open positions."""
        return len(self.positions)
    
    @property
    def leverage(self) -> float:
        """Calculate leverage (total exposure / total value)."""
        if self.total_value == 0:
            return 0.0
        total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        return total_exposure / self.total_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cash': self.cash,
            'market_value': self.market_value,
            'total_value': self.total_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'num_positions': self.num_positions,
            'leverage': self.leverage,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_entry_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                }
                for symbol, pos in self.positions.items()
            }
        }


class PortfolioState:
    """
    Tracks portfolio state including cash and positions over time.
    
    This class maintains the current state of the portfolio, handles position
    updates from executed orders, tracks realized and unrealized P&L, and
    provides position carry-forward logic for multi-period backtests.
    
    Requirements: 1.2, 1.3
    """
    
    def __init__(
        self,
        initial_capital: float,
        start_time: datetime
    ):
        """
        Initialize PortfolioState.
        
        Args:
            initial_capital: Initial cash balance
            start_time: Start time of the backtest
        """
        self.initial_capital = initial_capital
        self.start_time = start_time
        
        # Current state
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._realized_pnl = 0.0
        
        # Historical snapshots
        self._snapshots: List[PortfolioSnapshot] = []
        
        # Trade history for reconciliation
        self._executed_orders: List[Order] = []
        
        # Current timestamp
        self._current_time = start_time
        
        logger.info(f"Initialized PortfolioState with ${initial_capital:,.2f} capital")
    
    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._cash
    
    @property
    def positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return deepcopy(self._positions)
    
    @property
    def position_quantities(self) -> Dict[str, float]:
        """Get current position quantities by symbol."""
        return {symbol: pos.quantity for symbol, pos in self._positions.items()}
    
    @property
    def market_value(self) -> float:
        """Calculate total market value of positions."""
        return sum(pos.market_value for pos in self._positions.values())
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value (cash + positions)."""
        return self._cash + self.market_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self._positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._realized_pnl
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self._realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        """Calculate return percentage."""
        if self.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.initial_capital) * 100
    
    def update_from_order(
        self,
        order: Order,
        execution_price: Optional[float] = None,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> None:
        """
        Update portfolio state from an executed order.
        
        Args:
            order: Executed order
            execution_price: Actual execution price (defaults to order price)
            commission: Commission paid
            slippage: Slippage cost
            
        Requirements: 1.2, 1.3
        """
        if execution_price is None:
            execution_price = order.price
        
        symbol = order.symbol
        quantity = order.quantity if order.side == 'buy' else -order.quantity
        
        # Calculate total cost including fees
        notional_value = abs(quantity * execution_price)
        total_cost = commission + slippage
        
        # Update cash
        if order.side == 'buy':
            self._cash -= (notional_value + total_cost)
        else:  # sell
            self._cash += (notional_value - total_cost)
        
        # Update position
        self._update_position(
            symbol=symbol,
            quantity=quantity,
            price=execution_price,
            timestamp=order.timestamp
        )
        
        # Track executed order
        self._executed_orders.append(order)
        self._current_time = order.timestamp
        
        logger.debug(f"Updated portfolio from order: {order.side} {abs(quantity):.4f} {symbol} @ {execution_price:.2f}")
    
    def _update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime
    ) -> None:
        """
        Update position for a symbol.
        
        Handles position opening, closing, and averaging.
        """
        if symbol not in self._positions:
            # Open new position
            if quantity != 0:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=price,
                    current_price=price,
                    timestamp=timestamp
                )
                logger.debug(f"Opened new position: {self._positions[symbol]}")
        else:
            current_pos = self._positions[symbol]
            new_quantity = current_pos.quantity + quantity
            
            if new_quantity == 0:
                # Close position completely
                realized_pnl = self._calculate_realized_pnl(
                    current_pos.quantity,
                    current_pos.avg_entry_price,
                    -quantity,
                    price
                )
                self._realized_pnl += realized_pnl
                
                logger.debug(f"Closed position {symbol}: realized P&L = ${realized_pnl:.2f}")
                del self._positions[symbol]
            
            elif (current_pos.quantity > 0 and new_quantity > 0) or \
                 (current_pos.quantity < 0 and new_quantity < 0):
                # Adding to position - calculate new average price
                total_cost = (current_pos.quantity * current_pos.avg_entry_price) + (quantity * price)
                new_avg_price = total_cost / new_quantity
                
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=new_quantity,
                    avg_entry_price=new_avg_price,
                    current_price=price,
                    timestamp=timestamp
                )
                logger.debug(f"Added to position: {self._positions[symbol]}")
            
            else:
                # Reducing or reversing position
                # First, close the existing position
                close_quantity = min(abs(current_pos.quantity), abs(quantity))
                realized_pnl = self._calculate_realized_pnl(
                    current_pos.quantity,
                    current_pos.avg_entry_price,
                    -close_quantity if current_pos.quantity > 0 else close_quantity,
                    price
                )
                self._realized_pnl += realized_pnl
                
                logger.debug(f"Reduced position {symbol}: realized P&L = ${realized_pnl:.2f}")
                
                # If there's remaining quantity, open new position in opposite direction
                if abs(new_quantity) > 0:
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        quantity=new_quantity,
                        avg_entry_price=price,
                        current_price=price,
                        timestamp=timestamp
                    )
                    logger.debug(f"Reversed position: {self._positions[symbol]}")
                else:
                    del self._positions[symbol]
    
    def _calculate_realized_pnl(
        self,
        position_quantity: float,
        entry_price: float,
        close_quantity: float,
        exit_price: float
    ) -> float:
        """
        Calculate realized P&L for a position close.
        
        Args:
            position_quantity: Current position quantity
            entry_price: Average entry price
            close_quantity: Quantity being closed (negative for long, positive for short)
            exit_price: Exit price
            
        Returns:
            Realized P&L
        """
        # Determine if closing long or short
        if position_quantity > 0:
            # Closing long position
            pnl = close_quantity * (exit_price - entry_price)
        else:
            # Closing short position
            pnl = -close_quantity * (entry_price - exit_price)
        
        return pnl
    
    def update_prices(
        self,
        prices: Dict[str, float],
        timestamp: datetime
    ) -> None:
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary of current prices by symbol
            timestamp: Current timestamp
            
        Requirements: 1.2, 1.3
        """
        for symbol, position in self._positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                position.timestamp = timestamp
        
        self._current_time = timestamp
    
    def take_snapshot(self, timestamp: Optional[datetime] = None) -> PortfolioSnapshot:
        """
        Take a snapshot of current portfolio state.
        
        Args:
            timestamp: Timestamp for the snapshot (defaults to current time)
            
        Returns:
            Portfolio snapshot
            
        Requirements: 1.2, 1.3
        """
        if timestamp is None:
            timestamp = self._current_time
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self._cash,
            positions=deepcopy(self._positions),
            total_value=self.total_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self._realized_pnl,
            total_pnl=self.total_pnl
        )
        
        self._snapshots.append(snapshot)
        return snapshot
    
    def get_snapshots(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PortfolioSnapshot]:
        """
        Get historical portfolio snapshots with optional filtering.
        
        Args:
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            
        Returns:
            List of portfolio snapshots
        """
        filtered_snapshots = self._snapshots
        
        if start_time:
            filtered_snapshots = [s for s in filtered_snapshots if s.timestamp >= start_time]
        
        if end_time:
            filtered_snapshots = [s for s in filtered_snapshots if s.timestamp <= end_time]
        
        return filtered_snapshots
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as a DataFrame.
        
        Returns:
            DataFrame with timestamp and portfolio values
        """
        if not self._snapshots:
            return pd.DataFrame()
        
        data = []
        for snapshot in self._snapshots:
            data.append({
                'timestamp': snapshot.timestamp,
                'cash': snapshot.cash,
                'market_value': snapshot.market_value,
                'total_value': snapshot.total_value,
                'unrealized_pnl': snapshot.unrealized_pnl,
                'realized_pnl': snapshot.realized_pnl,
                'total_pnl': snapshot.total_pnl,
                'num_positions': snapshot.num_positions,
                'leverage': snapshot.leverage
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def validate_state(self) -> Dict[str, Any]:
        """
        Validate portfolio state for consistency.
        
        Returns:
            Dictionary with validation results
            
        Requirements: 1.3
        """
        issues = []
        warnings = []
        
        # Check cash balance
        if self._cash < 0:
            issues.append(f"Negative cash balance: ${self._cash:.2f}")
        
        # Check position consistency
        for symbol, position in self._positions.items():
            if position.quantity == 0:
                issues.append(f"Zero quantity position for {symbol}")
            
            if position.avg_entry_price <= 0:
                issues.append(f"Invalid entry price for {symbol}: {position.avg_entry_price}")
            
            if position.current_price <= 0:
                issues.append(f"Invalid current price for {symbol}: {position.current_price}")
        
        # Check total value
        if self.total_value <= 0:
            warnings.append(f"Portfolio value is non-positive: ${self.total_value:.2f}")
        
        # Check for excessive leverage
        if self._positions:
            total_exposure = sum(abs(pos.market_value) for pos in self._positions.values())
            leverage = total_exposure / self.total_value if self.total_value > 0 else 0
            
            if leverage > 3.0:
                warnings.append(f"High leverage detected: {leverage:.2f}x")
        
        is_valid = len(issues) == 0
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings,
            'cash': self._cash,
            'total_value': self.total_value,
            'num_positions': len(self._positions),
            'total_pnl': self.total_pnl
        }
    
    def reconcile_positions(
        self,
        expected_positions: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Reconcile current positions with expected positions.
        
        Args:
            expected_positions: Expected position quantities by symbol
            current_prices: Current prices by symbol
            
        Returns:
            Dictionary with reconciliation results
            
        Requirements: 1.3
        """
        discrepancies = []
        
        # Check all expected positions
        for symbol, expected_qty in expected_positions.items():
            actual_qty = self._positions.get(symbol, Position(
                symbol=symbol,
                quantity=0.0,
                avg_entry_price=0.0,
                current_price=0.0,
                timestamp=self._current_time
            )).quantity
            
            if abs(actual_qty - expected_qty) > 1e-6:  # Allow small floating point differences
                discrepancies.append({
                    'symbol': symbol,
                    'expected_quantity': expected_qty,
                    'actual_quantity': actual_qty,
                    'difference': actual_qty - expected_qty
                })
        
        # Check for unexpected positions
        for symbol in self._positions:
            if symbol not in expected_positions:
                discrepancies.append({
                    'symbol': symbol,
                    'expected_quantity': 0.0,
                    'actual_quantity': self._positions[symbol].quantity,
                    'difference': self._positions[symbol].quantity
                })
        
        is_reconciled = len(discrepancies) == 0
        
        return {
            'is_reconciled': is_reconciled,
            'discrepancies': discrepancies,
            'num_discrepancies': len(discrepancies)
        }
    
    def get_position_history(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get position history for a symbol from snapshots.
        
        Args:
            symbol: Symbol to get history for
            
        Returns:
            List of position states over time
        """
        history = []
        
        for snapshot in self._snapshots:
            if symbol in snapshot.positions:
                pos = snapshot.positions[symbol]
                history.append({
                    'timestamp': snapshot.timestamp,
                    'quantity': pos.quantity,
                    'avg_entry_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                })
        
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get portfolio statistics.
        
        Returns:
            Dictionary with portfolio statistics
        """
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self._cash,
            'market_value': self.market_value,
            'total_value': self.total_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self._realized_pnl,
            'total_pnl': self.total_pnl,
            'return_pct': self.return_pct,
            'num_positions': len(self._positions),
            'num_snapshots': len(self._snapshots),
            'num_executed_orders': len(self._executed_orders),
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_entry_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for symbol, pos in self._positions.items()
            }
        }
    
    def reset_state(self) -> None:
        """Reset portfolio to initial state (useful for testing)."""
        self._cash = self.initial_capital
        self._positions.clear()
        self._realized_pnl = 0.0
        self._snapshots.clear()
        self._executed_orders.clear()
        self._current_time = self.start_time
        logger.info("Reset PortfolioState to initial state")
