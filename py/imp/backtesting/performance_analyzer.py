"""
Performance analysis and metrics calculation for backtesting framework.

This module implements the PerformanceAnalyzer class that calculates comprehensive
performance metrics including returns, risk-adjusted metrics, drawdowns, and
benchmark comparisons.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from scipy import stats

from .portfolio_state import PortfolioSnapshot
from .trade_generator import Order
from .config import PerformanceConfig


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int  # in periods
    avg_drawdown: float
    
    # Risk measures
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    
    # Trade statistics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Time-based metrics
    start_date: datetime
    end_date: datetime
    num_periods: int
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            'volatility': self.volatility,
            'annualized_volatility': self.annualized_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_drawdown': self.avg_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'num_periods': self.num_periods,
            **self.metadata
        }


@dataclass
class BenchmarkComparison:
    """Comparison metrics against a benchmark."""
    strategy_return: float
    benchmark_return: float
    excess_return: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    correlation: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_return': self.strategy_return,
            'benchmark_return': self.benchmark_return,
            'excess_return': self.excess_return,
            'alpha': self.alpha,
            'beta': self.beta,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error,
            'correlation': self.correlation
        }


class PerformanceAnalyzer:
    """
    Calculate comprehensive performance metrics and analysis.
    
    This class computes standard performance metrics (returns, Sharpe, drawdown),
    risk-adjusted metrics (Sortino, Calmar, VaR), and benchmark comparisons.
    
    Requirements: 4.1, 4.2, 4.5
    """
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize PerformanceAnalyzer.
        
        Args:
            config: Performance analysis configuration
        """
        self.config = config
        
        # Cache for computed metrics
        self._metrics_cache: Dict[str, PerformanceMetrics] = {}
        
        logger.info("Initialized PerformanceAnalyzer")
    
    def calculate_metrics(
        self,
        snapshots: List[PortfolioSnapshot],
        orders: List[Order],
        initial_capital: float
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            snapshots: Portfolio snapshots over time
            orders: Executed orders
            initial_capital: Initial capital
            
        Returns:
            Performance metrics
            
        Requirements: 4.1, 4.2
        """
        if not snapshots:
            logger.warning("No snapshots provided for performance calculation")
            return self._empty_metrics()
        
        # Convert snapshots to DataFrame for easier analysis
        equity_curve = self._snapshots_to_dataframe(snapshots)
        
        # Calculate returns
        returns_metrics = self._calculate_returns(equity_curve, initial_capital)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(equity_curve, returns_metrics)
        
        # Calculate drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(equity_curve)
        
        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics(orders, snapshots)
        
        # Combine all metrics
        metrics = PerformanceMetrics(
            # Returns
            total_return=returns_metrics['total_return'],
            annualized_return=returns_metrics['annualized_return'],
            cumulative_return=returns_metrics['cumulative_return'],
            
            # Risk
            volatility=risk_metrics['volatility'],
            annualized_volatility=risk_metrics['annualized_volatility'],
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            sortino_ratio=risk_metrics['sortino_ratio'],
            calmar_ratio=risk_metrics['calmar_ratio'],
            
            # Drawdown
            max_drawdown=drawdown_metrics['max_drawdown'],
            max_drawdown_duration=drawdown_metrics['max_drawdown_duration'],
            avg_drawdown=drawdown_metrics['avg_drawdown'],
            
            # Risk measures
            var_95=risk_metrics['var_95'],
            cvar_95=risk_metrics['cvar_95'],
            
            # Trade stats
            num_trades=trade_stats['num_trades'],
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            
            # Time-based
            start_date=snapshots[0].timestamp,
            end_date=snapshots[-1].timestamp,
            num_periods=len(snapshots)
        )
        
        logger.info(f"Calculated performance metrics: Return={metrics.total_return:.2%}, "
                   f"Sharpe={metrics.sharpe_ratio:.2f}, MaxDD={metrics.max_drawdown:.2%}")
        
        return metrics
    
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
    
    def _calculate_returns(
        self,
        equity_curve: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, float]:
        """
        Calculate return metrics.
        
        Requirements: 4.1
        """
        if len(equity_curve) < 2:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'cumulative_return': 0.0,
                'period_returns': pd.Series()
            }
        
        # Calculate period returns
        equity_curve['returns'] = equity_curve['total_value'].pct_change()
        period_returns = equity_curve['returns'].dropna()
        
        # Total return
        final_value = equity_curve['total_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Cumulative return
        cumulative_return = (equity_curve['total_value'] / initial_capital - 1).iloc[-1]
        
        # Annualized return
        start_date = equity_curve['timestamp'].iloc[0]
        end_date = equity_curve['timestamp'].iloc[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_return,
            'period_returns': period_returns
        }
    
    def _calculate_risk_metrics(
        self,
        equity_curve: pd.DataFrame,
        returns_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted metrics.
        
        Requirements: 4.2
        """
        period_returns = returns_metrics['period_returns']
        
        if len(period_returns) < 2:
            return {
                'volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0
            }
        
        # Volatility
        volatility = period_returns.std()
        
        # Annualize volatility (assuming 5-minute bars, 288 per day)
        # Adjust based on your data frequency
        periods_per_day = 288
        periods_per_year = periods_per_day * 365
        annualized_volatility = volatility * np.sqrt(periods_per_year)
        
        # Sharpe Ratio
        risk_free_rate = self.config.risk_free_rate
        excess_returns = period_returns - (risk_free_rate / periods_per_year)
        
        if volatility > 0:
            sharpe_ratio = excess_returns.mean() / volatility * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio (uses downside deviation)
        downside_returns = period_returns[period_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std()
            if downside_deviation > 0:
                sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year)
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = sharpe_ratio  # No downside, use Sharpe
        
        # Calmar Ratio (return / max drawdown)
        drawdown_metrics = self._calculate_drawdown_metrics(equity_curve)
        max_drawdown = abs(drawdown_metrics['max_drawdown'])
        
        if max_drawdown > 0:
            calmar_ratio = returns_metrics['annualized_return'] / max_drawdown
        else:
            calmar_ratio = 0.0
        
        # Value at Risk (VaR) at 95% confidence
        var_95 = np.percentile(period_returns, (1 - self.config.var_confidence) * 100)
        
        # Conditional VaR (CVaR) - average of returns below VaR
        returns_below_var = period_returns[period_returns <= var_95]
        if len(returns_below_var) > 0:
            cvar_95 = returns_below_var.mean()
        else:
            cvar_95 = var_95
        
        return {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_drawdown_metrics(
        self,
        equity_curve: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate drawdown metrics.
        
        Requirements: 4.1
        """
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'avg_drawdown': 0.0,
                'drawdown_series': pd.Series()
            }
        
        # Calculate running maximum
        equity_curve['running_max'] = equity_curve['total_value'].cummax()
        
        # Calculate drawdown
        equity_curve['drawdown'] = (equity_curve['total_value'] - equity_curve['running_max']) / equity_curve['running_max']
        
        # Max drawdown
        max_drawdown = equity_curve['drawdown'].min()
        
        # Calculate drawdown durations
        in_drawdown = equity_curve['drawdown'] < 0
        drawdown_periods = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0
        
        # Add final drawdown if still in one
        if current_duration > 0:
            drawdown_periods.append(current_duration)
        
        # Max drawdown duration
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Average drawdown (only negative values)
        drawdowns = equity_curve['drawdown'][equity_curve['drawdown'] < 0]
        avg_drawdown = drawdowns.mean() if len(drawdowns) > 0 else 0.0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown': avg_drawdown,
            'drawdown_series': equity_curve['drawdown']
        }
    
    def _calculate_trade_statistics(
        self,
        orders: List[Order],
        snapshots: List[PortfolioSnapshot]
    ) -> Dict[str, float]:
        """
        Calculate trade statistics.
        
        Requirements: 4.1
        """
        if not orders:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        # Group orders into round-trip trades
        trades = self._group_round_trip_trades(orders, snapshots)
        
        if not trades:
            return {
                'num_trades': len(orders),
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        # Separate winning and losing trades
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        # Win rate
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # Average win/loss
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
        
        # Profit factor (gross profit / gross loss)
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        
        return {
            'num_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def _group_round_trip_trades(
        self,
        orders: List[Order],
        snapshots: List[PortfolioSnapshot]
    ) -> List[Dict[str, Any]]:
        """
        Group orders into round-trip trades for P&L calculation.
        
        This is a simplified implementation that matches buy/sell pairs.
        """
        trades = []
        
        # Group orders by symbol
        symbol_orders = {}
        for order in orders:
            if order.symbol not in symbol_orders:
                symbol_orders[order.symbol] = []
            symbol_orders[order.symbol].append(order)
        
        # For each symbol, match buy/sell pairs
        for symbol, orders_list in symbol_orders.items():
            orders_list = sorted(orders_list, key=lambda x: x.timestamp)
            
            position = 0.0
            entry_price = 0.0
            entry_time = None
            
            for order in orders_list:
                if order.side == 'buy':
                    if position <= 0:
                        # Opening or reversing to long
                        entry_price = order.price
                        entry_time = order.timestamp
                        position = order.quantity
                    else:
                        # Adding to long
                        total_cost = (position * entry_price) + (order.quantity * order.price)
                        position += order.quantity
                        entry_price = total_cost / position
                
                elif order.side == 'sell':
                    if position > 0:
                        # Closing long position
                        pnl = order.quantity * (order.price - entry_price)
                        
                        trades.append({
                            'symbol': symbol,
                            'entry_time': entry_time,
                            'exit_time': order.timestamp,
                            'entry_price': entry_price,
                            'exit_price': order.price,
                            'quantity': order.quantity,
                            'pnl': pnl,
                            'side': 'long'
                        })
                        
                        position -= order.quantity
                        if position <= 0:
                            position = 0.0
                            entry_price = 0.0
                    else:
                        # Opening short
                        entry_price = order.price
                        entry_time = order.timestamp
                        position = -order.quantity
        
        return trades
    
    def compare_to_benchmark(
        self,
        strategy_snapshots: List[PortfolioSnapshot],
        benchmark_data: pd.DataFrame,
        initial_capital: float
    ) -> BenchmarkComparison:
        """
        Compare strategy performance to a benchmark.
        
        Args:
            strategy_snapshots: Strategy portfolio snapshots
            benchmark_data: Benchmark price data with 'timestamp' and 'close' columns
            initial_capital: Initial capital
            
        Returns:
            Benchmark comparison metrics
            
        Requirements: 4.5
        """
        if not strategy_snapshots or benchmark_data.empty:
            logger.warning("Insufficient data for benchmark comparison")
            return self._empty_benchmark_comparison()
        
        # Convert strategy snapshots to DataFrame
        strategy_df = self._snapshots_to_dataframe(strategy_snapshots)
        
        # Align timestamps
        strategy_df = strategy_df.set_index('timestamp')
        benchmark_data = benchmark_data.set_index('timestamp')
        
        # Merge on timestamp
        merged = strategy_df.join(benchmark_data[['close']], how='inner')
        
        if len(merged) < 2:
            logger.warning("Insufficient overlapping data for benchmark comparison")
            return self._empty_benchmark_comparison()
        
        # Calculate returns
        strategy_returns = merged['total_value'].pct_change().dropna()
        benchmark_returns = merged['close'].pct_change().dropna()
        
        # Align returns
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        if len(strategy_returns) < 2:
            return self._empty_benchmark_comparison()
        
        # Calculate cumulative returns
        strategy_return = (merged['total_value'].iloc[-1] / initial_capital) - 1
        benchmark_return = (merged['close'].iloc[-1] / merged['close'].iloc[0]) - 1
        
        # Excess return
        excess_return = strategy_return - benchmark_return
        
        # Calculate beta and alpha using linear regression
        # CAPM: R_strategy = alpha + beta * R_benchmark
        if len(strategy_returns) > 1 and benchmark_returns.std() > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                benchmark_returns, strategy_returns
            )
            beta = slope
            alpha = intercept
            correlation = r_value
        else:
            beta = 0.0
            alpha = 0.0
            correlation = 0.0
        
        # Information ratio (excess return / tracking error)
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std()
        
        if tracking_error > 0:
            # Annualize information ratio
            periods_per_year = 288 * 365  # Adjust based on data frequency
            information_ratio = (excess_returns.mean() / tracking_error) * np.sqrt(periods_per_year)
        else:
            information_ratio = 0.0
        
        comparison = BenchmarkComparison(
            strategy_return=strategy_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            correlation=correlation
        )
        
        logger.info(f"Benchmark comparison: Strategy={strategy_return:.2%}, "
                   f"Benchmark={benchmark_return:.2%}, Alpha={alpha:.4f}, Beta={beta:.2f}")
        
        return comparison
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            cumulative_return=0.0,
            volatility=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            avg_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            num_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            num_periods=0
        )
    
    def _empty_benchmark_comparison(self) -> BenchmarkComparison:
        """Return empty benchmark comparison."""
        return BenchmarkComparison(
            strategy_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            alpha=0.0,
            beta=0.0,
            information_ratio=0.0,
            tracking_error=0.0,
            correlation=0.0
        )
