"""
Regime-specific performance analysis for backtesting framework.

This module implements regime-aware performance analysis that breaks down
performance by HMM-detected market regimes and provides attribution analysis
by signal source.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import numpy as np

from .portfolio_state import PortfolioSnapshot
from .trade_generator import Order
from .performance_analyzer import PerformanceMetrics, PerformanceAnalyzer
from .config import PerformanceConfig


logger = logging.getLogger(__name__)


@dataclass
class RegimeMetrics:
    """Performance metrics for a specific regime."""
    regime_state: int
    regime_name: str
    
    # Time in regime
    num_periods: int
    time_pct: float
    
    # Performance
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    
    # Trade statistics
    num_trades: int
    win_rate: float
    avg_trade_pnl: float
    
    # Contribution to overall performance
    contribution_to_return: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime_state': self.regime_state,
            'regime_name': self.regime_name,
            'num_periods': self.num_periods,
            'time_pct': self.time_pct,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'avg_trade_pnl': self.avg_trade_pnl,
            'contribution_to_return': self.contribution_to_return
        }


@dataclass
class SignalAttribution:
    """Performance attribution by signal source."""
    signal_source: str
    
    # Trade statistics
    num_trades: int
    trade_pct: float
    
    # Performance
    total_pnl: float
    avg_pnl: float
    win_rate: float
    
    # Contribution
    contribution_to_return: float
    contribution_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'signal_source': self.signal_source,
            'num_trades': self.num_trades,
            'trade_pct': self.trade_pct,
            'total_pnl': self.total_pnl,
            'avg_pnl': self.avg_pnl,
            'win_rate': self.win_rate,
            'contribution_to_return': self.contribution_to_return,
            'contribution_pct': self.contribution_pct
        }


@dataclass
class TimeSeriesPerformance:
    """Time-series performance tracking."""
    timestamps: List[datetime]
    equity_values: List[float]
    returns: List[float]
    cumulative_returns: List[float]
    drawdowns: List[float]
    regime_states: List[int]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'equity_value': self.equity_values,
            'returns': self.returns,
            'cumulative_returns': self.cumulative_returns,
            'drawdowns': self.drawdowns,
            'regime_state': self.regime_states
        })


class RegimePerformanceAnalyzer:
    """
    Analyze performance by market regime and signal source.
    
    This class provides regime-specific performance breakdown, attribution
    analysis by signal source, and time-series performance tracking.
    
    Requirements: 4.3, 4.4
    """
    
    def __init__(
        self,
        config: PerformanceConfig,
        regime_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize RegimePerformanceAnalyzer.
        
        Args:
            config: Performance configuration
            regime_names: Optional mapping of regime states to names
        """
        self.config = config
        self.regime_names = regime_names or {}
        
        # Base performance analyzer for metrics calculation
        self.base_analyzer = PerformanceAnalyzer(config)
        
        logger.info("Initialized RegimePerformanceAnalyzer")
    
    def analyze_by_regime(
        self,
        snapshots: List[PortfolioSnapshot],
        orders: List[Order],
        initial_capital: float,
        regime_data: pd.DataFrame
    ) -> Dict[int, RegimeMetrics]:
        """
        Analyze performance broken down by market regime.
        
        Args:
            snapshots: Portfolio snapshots
            orders: Executed orders
            initial_capital: Initial capital
            regime_data: DataFrame with 'timestamp' and 'regime_state' columns
            
        Returns:
            Dictionary mapping regime state to metrics
            
        Requirements: 4.3
        """
        if not snapshots or regime_data.empty:
            logger.warning("Insufficient data for regime analysis")
            return {}
        
        # Convert snapshots to DataFrame
        equity_df = self._snapshots_to_dataframe(snapshots)
        
        # Merge with regime data
        equity_df = equity_df.set_index('timestamp')
        regime_data = regime_data.set_index('timestamp')
        
        merged = equity_df.join(regime_data[['regime_state']], how='left')
        merged = merged.reset_index()
        
        # Forward fill regime states for missing values
        merged['regime_state'] = merged['regime_state'].ffill()
        merged['regime_state'] = merged['regime_state'].bfill()
        
        # Get unique regimes
        unique_regimes = merged['regime_state'].dropna().unique()
        
        regime_metrics = {}
        total_periods = len(merged)
        
        for regime_state in sorted(unique_regimes):
            regime_state = int(regime_state)
            
            # Filter data for this regime
            regime_mask = merged['regime_state'] == regime_state
            regime_snapshots_df = merged[regime_mask].copy()
            
            if len(regime_snapshots_df) < 2:
                logger.debug(f"Insufficient data for regime {regime_state}")
                continue
            
            # Calculate metrics for this regime
            metrics = self._calculate_regime_metrics(
                regime_snapshots_df=regime_snapshots_df,
                regime_state=regime_state,
                orders=orders,
                initial_capital=initial_capital,
                total_periods=total_periods
            )
            
            regime_metrics[regime_state] = metrics
            
            logger.info(f"Regime {regime_state} ({metrics.regime_name}): "
                       f"Return={metrics.total_return:.2%}, "
                       f"Sharpe={metrics.sharpe_ratio:.2f}, "
                       f"Time={metrics.time_pct:.1%}")
        
        return regime_metrics
    
    def _calculate_regime_metrics(
        self,
        regime_snapshots_df: pd.DataFrame,
        regime_state: int,
        orders: List[Order],
        initial_capital: float,
        total_periods: int
    ) -> RegimeMetrics:
        """Calculate metrics for a specific regime."""
        # Get regime name
        regime_name = self.regime_names.get(regime_state, f"Regime {regime_state}")
        
        # Time in regime
        num_periods = len(regime_snapshots_df)
        time_pct = num_periods / total_periods if total_periods > 0 else 0.0
        
        # Calculate returns
        regime_snapshots_df['returns'] = regime_snapshots_df['total_value'].pct_change()
        period_returns = regime_snapshots_df['returns'].dropna()
        
        if len(period_returns) > 0:
            # Total return for this regime
            start_value = regime_snapshots_df['total_value'].iloc[0]
            end_value = regime_snapshots_df['total_value'].iloc[-1]
            total_return = (end_value - start_value) / start_value
            
            # Annualized return
            start_date = regime_snapshots_df['timestamp'].iloc[0]
            end_date = regime_snapshots_df['timestamp'].iloc[-1]
            days = (end_date - start_date).days
            years = days / 365.25
            
            if years > 0:
                annualized_return = (1 + total_return) ** (1 / years) - 1
            else:
                annualized_return = 0.0
            
            # Volatility
            volatility = period_returns.std()
            
            # Sharpe ratio
            periods_per_year = 288 * 365  # Adjust based on data frequency
            risk_free_rate = self.config.risk_free_rate
            excess_returns = period_returns - (risk_free_rate / periods_per_year)
            
            if volatility > 0:
                sharpe_ratio = excess_returns.mean() / volatility * np.sqrt(periods_per_year)
            else:
                sharpe_ratio = 0.0
        else:
            total_return = 0.0
            annualized_return = 0.0
            volatility = 0.0
            sharpe_ratio = 0.0
        
        # Filter orders for this regime
        regime_timestamps = set(regime_snapshots_df['timestamp'])
        regime_orders = [
            o for o in orders
            if o.timestamp in regime_timestamps
        ]
        
        # Trade statistics
        if regime_orders:
            trades = self._group_trades_by_regime(regime_orders, regime_snapshots_df)
            num_trades = len(trades)
            
            if trades:
                winning_trades = [t for t in trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(trades)
                avg_trade_pnl = np.mean([t['pnl'] for t in trades])
            else:
                win_rate = 0.0
                avg_trade_pnl = 0.0
        else:
            num_trades = 0
            win_rate = 0.0
            avg_trade_pnl = 0.0
        
        # Contribution to overall return
        contribution_to_return = total_return * time_pct
        
        return RegimeMetrics(
            regime_state=regime_state,
            regime_name=regime_name,
            num_periods=num_periods,
            time_pct=time_pct,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            contribution_to_return=contribution_to_return
        )
    
    def analyze_signal_attribution(
        self,
        orders: List[Order],
        snapshots: List[PortfolioSnapshot],
        initial_capital: float
    ) -> Dict[str, SignalAttribution]:
        """
        Analyze performance attribution by signal source.
        
        Args:
            orders: Executed orders
            snapshots: Portfolio snapshots
            initial_capital: Initial capital
            
        Returns:
            Dictionary mapping signal source to attribution metrics
            
        Requirements: 4.4
        """
        if not orders:
            logger.warning("No orders for signal attribution analysis")
            return {}
        
        # Group orders by signal source
        signal_orders = defaultdict(list)
        for order in orders:
            signal_source = order.metadata.get('signal_source', 'unknown')
            signal_orders[signal_source].append(order)
        
        # Calculate attribution for each signal source
        attributions = {}
        total_trades = len(orders)
        
        # Calculate total P&L for percentage contributions
        total_pnl = sum(
            self._estimate_order_pnl(order, snapshots)
            for order in orders
        )
        
        for signal_source, source_orders in signal_orders.items():
            attribution = self._calculate_signal_attribution(
                signal_source=signal_source,
                orders=source_orders,
                snapshots=snapshots,
                total_trades=total_trades,
                total_pnl=total_pnl,
                initial_capital=initial_capital
            )
            
            attributions[signal_source] = attribution
            
            logger.info(f"Signal {signal_source}: "
                       f"Trades={attribution.num_trades}, "
                       f"PnL=${attribution.total_pnl:.2f}, "
                       f"WinRate={attribution.win_rate:.1%}")
        
        return attributions
    
    def _calculate_signal_attribution(
        self,
        signal_source: str,
        orders: List[Order],
        snapshots: List[PortfolioSnapshot],
        total_trades: int,
        total_pnl: float,
        initial_capital: float
    ) -> SignalAttribution:
        """Calculate attribution metrics for a signal source."""
        num_trades = len(orders)
        trade_pct = num_trades / total_trades if total_trades > 0 else 0.0
        
        # Estimate P&L for each order
        order_pnls = [
            self._estimate_order_pnl(order, snapshots)
            for order in orders
        ]
        
        total_signal_pnl = sum(order_pnls)
        avg_pnl = total_signal_pnl / num_trades if num_trades > 0 else 0.0
        
        # Win rate
        winning_orders = [pnl for pnl in order_pnls if pnl > 0]
        win_rate = len(winning_orders) / num_trades if num_trades > 0 else 0.0
        
        # Contribution to return
        contribution_to_return = total_signal_pnl / initial_capital if initial_capital > 0 else 0.0
        contribution_pct = total_signal_pnl / total_pnl if total_pnl != 0 else 0.0
        
        return SignalAttribution(
            signal_source=signal_source,
            num_trades=num_trades,
            trade_pct=trade_pct,
            total_pnl=total_signal_pnl,
            avg_pnl=avg_pnl,
            win_rate=win_rate,
            contribution_to_return=contribution_to_return,
            contribution_pct=contribution_pct
        )
    
    def create_time_series_performance(
        self,
        snapshots: List[PortfolioSnapshot],
        regime_data: pd.DataFrame,
        initial_capital: float
    ) -> TimeSeriesPerformance:
        """
        Create time-series performance tracking with regime information.
        
        Args:
            snapshots: Portfolio snapshots
            regime_data: DataFrame with regime states
            initial_capital: Initial capital
            
        Returns:
            Time-series performance data
            
        Requirements: 4.4
        """
        if not snapshots:
            logger.warning("No snapshots for time-series performance")
            return TimeSeriesPerformance(
                timestamps=[],
                equity_values=[],
                returns=[],
                cumulative_returns=[],
                drawdowns=[],
                regime_states=[]
            )
        
        # Convert to DataFrame
        equity_df = self._snapshots_to_dataframe(snapshots)
        
        # Merge with regime data
        equity_df = equity_df.set_index('timestamp')
        regime_data = regime_data.set_index('timestamp')
        
        merged = equity_df.join(regime_data[['regime_state']], how='left')
        merged = merged.reset_index()
        
        # Forward fill regime states
        merged['regime_state'] = merged['regime_state'].ffill()
        merged['regime_state'] = merged['regime_state'].bfill()
        merged['regime_state'] = merged['regime_state'].fillna(0).astype(int)
        
        # Calculate returns
        merged['returns'] = merged['total_value'].pct_change().fillna(0)
        
        # Calculate cumulative returns
        merged['cumulative_returns'] = (merged['total_value'] / initial_capital) - 1
        
        # Calculate drawdowns
        merged['running_max'] = merged['total_value'].cummax()
        merged['drawdowns'] = (merged['total_value'] - merged['running_max']) / merged['running_max']
        
        return TimeSeriesPerformance(
            timestamps=merged['timestamp'].tolist(),
            equity_values=merged['total_value'].tolist(),
            returns=merged['returns'].tolist(),
            cumulative_returns=merged['cumulative_returns'].tolist(),
            drawdowns=merged['drawdowns'].tolist(),
            regime_states=merged['regime_state'].tolist()
        )
    
    def _snapshots_to_dataframe(
        self,
        snapshots: List[PortfolioSnapshot]
    ) -> pd.DataFrame:
        """Convert portfolio snapshots to DataFrame."""
        data = []
        for snapshot in snapshots:
            data.append({
                'timestamp': snapshot.timestamp,
                'total_value': snapshot.total_value,
                'cash': snapshot.cash,
                'market_value': snapshot.market_value,
                'unrealized_pnl': snapshot.unrealized_pnl,
                'realized_pnl': snapshot.realized_pnl,
                'total_pnl': snapshot.total_pnl
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def _group_trades_by_regime(
        self,
        orders: List[Order],
        regime_snapshots_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Group orders into trades for a specific regime.
        
        This is a simplified implementation for regime-specific analysis.
        """
        trades = []
        
        # Group by symbol
        symbol_orders = defaultdict(list)
        for order in orders:
            symbol_orders[order.symbol].append(order)
        
        # For each symbol, match buy/sell pairs
        for symbol, orders_list in symbol_orders.items():
            orders_list = sorted(orders_list, key=lambda x: x.timestamp)
            
            position = 0.0
            entry_price = 0.0
            
            for order in orders_list:
                if order.side == 'buy':
                    if position <= 0:
                        entry_price = order.price
                        position = order.quantity
                    else:
                        total_cost = (position * entry_price) + (order.quantity * order.price)
                        position += order.quantity
                        entry_price = total_cost / position
                
                elif order.side == 'sell':
                    if position > 0:
                        pnl = order.quantity * (order.price - entry_price)
                        
                        trades.append({
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'exit_price': order.price,
                            'quantity': order.quantity,
                            'pnl': pnl
                        })
                        
                        position -= order.quantity
        
        return trades
    
    def _estimate_order_pnl(
        self,
        order: Order,
        snapshots: List[PortfolioSnapshot]
    ) -> float:
        """
        Estimate P&L contribution of an order.
        
        This is a simplified estimation that looks at portfolio value changes
        around the order execution time.
        """
        # Find snapshots before and after the order
        order_time = order.timestamp
        
        before_snapshot = None
        after_snapshot = None
        
        for i, snapshot in enumerate(snapshots):
            if snapshot.timestamp <= order_time:
                before_snapshot = snapshot
            elif snapshot.timestamp > order_time and after_snapshot is None:
                after_snapshot = snapshot
                break
        
        if before_snapshot and after_snapshot:
            # Estimate P&L as the change in total P&L
            pnl_change = after_snapshot.total_pnl - before_snapshot.total_pnl
            return pnl_change
        
        return 0.0
    
    def generate_regime_report(
        self,
        regime_metrics: Dict[int, RegimeMetrics],
        signal_attributions: Dict[str, SignalAttribution],
        overall_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """
        Generate comprehensive regime analysis report.
        
        Args:
            regime_metrics: Metrics by regime
            signal_attributions: Attribution by signal source
            overall_metrics: Overall performance metrics
            
        Returns:
            Comprehensive report dictionary
            
        Requirements: 4.3, 4.4
        """
        report = {
            'overall_performance': overall_metrics.to_dict(),
            'regime_analysis': {
                'by_regime': {
                    state: metrics.to_dict()
                    for state, metrics in regime_metrics.items()
                },
                'summary': self._summarize_regime_performance(regime_metrics)
            },
            'signal_attribution': {
                'by_signal': {
                    source: attr.to_dict()
                    for source, attr in signal_attributions.items()
                },
                'summary': self._summarize_signal_attribution(signal_attributions)
            }
        }
        
        return report
    
    def _summarize_regime_performance(
        self,
        regime_metrics: Dict[int, RegimeMetrics]
    ) -> Dict[str, Any]:
        """Summarize regime performance."""
        if not regime_metrics:
            return {}
        
        # Find best and worst regimes
        best_regime = max(regime_metrics.values(), key=lambda x: x.total_return)
        worst_regime = min(regime_metrics.values(), key=lambda x: x.total_return)
        
        # Calculate weighted average metrics
        total_time = sum(m.num_periods for m in regime_metrics.values())
        
        if total_time > 0:
            weighted_return = sum(
                m.total_return * (m.num_periods / total_time)
                for m in regime_metrics.values()
            )
            weighted_sharpe = sum(
                m.sharpe_ratio * (m.num_periods / total_time)
                for m in regime_metrics.values()
            )
        else:
            weighted_return = 0.0
            weighted_sharpe = 0.0
        
        return {
            'num_regimes': len(regime_metrics),
            'best_regime': {
                'state': best_regime.regime_state,
                'name': best_regime.regime_name,
                'return': best_regime.total_return
            },
            'worst_regime': {
                'state': worst_regime.regime_state,
                'name': worst_regime.regime_name,
                'return': worst_regime.total_return
            },
            'weighted_avg_return': weighted_return,
            'weighted_avg_sharpe': weighted_sharpe
        }
    
    def _summarize_signal_attribution(
        self,
        signal_attributions: Dict[str, SignalAttribution]
    ) -> Dict[str, Any]:
        """Summarize signal attribution."""
        if not signal_attributions:
            return {}
        
        # Find best and worst signals
        best_signal = max(signal_attributions.values(), key=lambda x: x.total_pnl)
        worst_signal = min(signal_attributions.values(), key=lambda x: x.total_pnl)
        
        # Total statistics
        total_trades = sum(a.num_trades for a in signal_attributions.values())
        total_pnl = sum(a.total_pnl for a in signal_attributions.values())
        
        return {
            'num_signals': len(signal_attributions),
            'best_signal': {
                'source': best_signal.signal_source,
                'pnl': best_signal.total_pnl,
                'win_rate': best_signal.win_rate
            },
            'worst_signal': {
                'source': worst_signal.signal_source,
                'pnl': worst_signal.total_pnl,
                'win_rate': worst_signal.win_rate
            },
            'total_trades': total_trades,
            'total_pnl': total_pnl
        }
