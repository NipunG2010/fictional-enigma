"""
Example demonstrating position sizing and portfolio state management.

This script shows how to use the TradeGenerator and PortfolioState classes
to generate sized orders and track portfolio state over time.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imp.backtesting import (
    TradeGenerator,
    PortfolioState,
    TradeSignal,
    TradeAction,
    ProcessedSignal,
    SignalDirection,
    PositionSizingConfig,
    PositionSizingMethod
)


def create_sample_trade_signal(
    symbol: str,
    timestamp: datetime,
    action: TradeAction,
    confidence: float
) -> TradeSignal:
    """Create a sample trade signal for testing."""
    processed_signal = ProcessedSignal(
        timestamp=timestamp,
        symbol=symbol,
        direction=SignalDirection.BUY if action == TradeAction.BUY else SignalDirection.SELL,
        strength=confidence,
        signal_source='ldc',
        raw_signals={'ldc': 0.7, 'mr': 0.3, 'tsmom': 0.5},
        regime_state=1,
        regime_weights={'ldc': 0.5, 'mr': 0.3, 'tsmom': 0.2}
    )
    
    return TradeSignal(
        trade_id=f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
        timestamp=timestamp,
        symbol=symbol,
        action=action,
        confidence=confidence,
        signal_source='ldc',
        reasoning=f"Test {action.value} signal",
        processed_signal=processed_signal
    )


def create_sample_market_data(symbols: list, periods: int = 100) -> pd.DataFrame:
    """Create sample market data for testing."""
    data = []
    base_time = datetime(2024, 1, 1, 9, 0)
    
    for symbol in symbols:
        base_price = 100.0 if symbol == 'BTCUSDT' else 50.0
        
        for i in range(periods):
            timestamp = base_time + timedelta(minutes=5 * i)
            
            # Generate random OHLCV data
            close = base_price * (1 + np.random.randn() * 0.02)
            high = close * (1 + abs(np.random.randn()) * 0.01)
            low = close * (1 - abs(np.random.randn()) * 0.01)
            open_price = close * (1 + np.random.randn() * 0.005)
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
    
    return pd.DataFrame(data)


def demonstrate_position_sizing():
    """Demonstrate different position sizing methods."""
    print("=" * 80)
    print("Position Sizing Methods Demonstration")
    print("=" * 80)
    
    initial_capital = 100000.0
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    # Create sample market data
    market_data = create_sample_market_data(symbols)
    
    # Test different position sizing methods
    methods = [
        (PositionSizingMethod.FIXED_SIZE, {'fixed_size': 10000.0}),
        (PositionSizingMethod.PERCENTAGE, {'percentage': 0.02}),
        (PositionSizingMethod.VOLATILITY_ADJUSTED, {'volatility_target': 0.15}),
        (PositionSizingMethod.KELLY_CRITERION, {})
    ]
    
    for method, extra_config in methods:
        print(f"\n{'-' * 80}")
        print(f"Method: {method.value}")
        print(f"{'-' * 80}")
        
        # Create configuration
        config = PositionSizingConfig(
            method=method,
            **extra_config
        )
        
        # Create trade generator
        generator = TradeGenerator(config, initial_capital)
        
        # Create sample trade signals
        base_time = datetime(2024, 1, 1, 10, 0)
        signals = [
            create_sample_trade_signal('BTCUSDT', base_time, TradeAction.BUY, 0.8),
            create_sample_trade_signal('ETHUSDT', base_time + timedelta(minutes=5), TradeAction.BUY, 0.6)
        ]
        
        # Generate orders
        current_prices = {'BTCUSDT': 100.0, 'ETHUSDT': 50.0}
        orders = generator.generate_orders(
            trade_signals=signals,
            market_data=market_data,
            portfolio_value=initial_capital,
            current_positions={},
            current_prices=current_prices
        )
        
        # Display results
        for order in orders:
            print(f"\nSymbol: {order.symbol}")
            print(f"  Side: {order.side}")
            print(f"  Quantity: {order.quantity:.4f}")
            print(f"  Price: ${order.price:.2f}")
            print(f"  Notional Value: ${order.notional_value:,.2f}")
            print(f"  Confidence: {order.trade_signal.confidence:.2f}")
        
        # Get statistics
        stats = generator.get_order_statistics()
        print(f"\nStatistics:")
        print(f"  Total Orders: {stats['total_orders']}")
        print(f"  Total Notional: ${stats['total_notional']:,.2f}")
        print(f"  Avg Order Size: ${stats['avg_order_size']:,.2f}")


def demonstrate_portfolio_tracking():
    """Demonstrate portfolio state tracking."""
    print("\n" + "=" * 80)
    print("Portfolio State Tracking Demonstration")
    print("=" * 80)
    
    initial_capital = 100000.0
    start_time = datetime(2024, 1, 1, 9, 0)
    
    # Initialize portfolio
    portfolio = PortfolioState(initial_capital, start_time)
    
    print(f"\nInitial State:")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Total Value: ${portfolio.total_value:,.2f}")
    
    # Create configuration and generator
    config = PositionSizingConfig(
        method=PositionSizingMethod.PERCENTAGE,
        percentage=0.02
    )
    generator = TradeGenerator(config, initial_capital)
    
    # Simulate a series of trades
    base_time = start_time + timedelta(hours=1)
    current_prices = {'BTCUSDT': 100.0, 'ETHUSDT': 50.0}
    
    # Trade 1: Buy BTCUSDT
    print(f"\n{'-' * 80}")
    print("Trade 1: Buy BTCUSDT")
    print(f"{'-' * 80}")
    
    signal1 = create_sample_trade_signal('BTCUSDT', base_time, TradeAction.BUY, 0.8)
    market_data = create_sample_market_data(['BTCUSDT', 'ETHUSDT'])
    
    orders1 = generator.generate_orders(
        trade_signals=[signal1],
        market_data=market_data,
        portfolio_value=portfolio.total_value,
        current_positions=portfolio.position_quantities,
        current_prices=current_prices
    )
    
    if orders1:
        order = orders1[0]
        portfolio.update_from_order(order, commission=10.0, slippage=5.0)
        print(f"  Executed: {order.side} {order.quantity:.4f} @ ${order.price:.2f}")
        print(f"  Cash: ${portfolio.cash:,.2f}")
        print(f"  Positions: {len(portfolio.positions)}")
    
    # Update prices and take snapshot
    current_prices['BTCUSDT'] = 105.0  # Price increased
    portfolio.update_prices(current_prices, base_time + timedelta(minutes=30))
    snapshot1 = portfolio.take_snapshot()
    
    print(f"\nAfter price update:")
    print(f"  Total Value: ${snapshot1.total_value:,.2f}")
    print(f"  Unrealized P&L: ${snapshot1.unrealized_pnl:,.2f}")
    
    # Trade 2: Buy ETHUSDT
    print(f"\n{'-' * 80}")
    print("Trade 2: Buy ETHUSDT")
    print(f"{'-' * 80}")
    
    signal2 = create_sample_trade_signal('ETHUSDT', base_time + timedelta(hours=1), TradeAction.BUY, 0.7)
    
    orders2 = generator.generate_orders(
        trade_signals=[signal2],
        market_data=market_data,
        portfolio_value=portfolio.total_value,
        current_positions=portfolio.position_quantities,
        current_prices=current_prices
    )
    
    if orders2:
        order = orders2[0]
        portfolio.update_from_order(order, commission=10.0, slippage=5.0)
        print(f"  Executed: {order.side} {order.quantity:.4f} @ ${order.price:.2f}")
        print(f"  Cash: ${portfolio.cash:,.2f}")
        print(f"  Positions: {len(portfolio.positions)}")
    
    # Update prices again
    current_prices['BTCUSDT'] = 110.0
    current_prices['ETHUSDT'] = 52.0
    portfolio.update_prices(current_prices, base_time + timedelta(hours=2))
    snapshot2 = portfolio.take_snapshot()
    
    print(f"\nAfter second price update:")
    print(f"  Total Value: ${snapshot2.total_value:,.2f}")
    print(f"  Unrealized P&L: ${snapshot2.unrealized_pnl:,.2f}")
    print(f"  Return: {portfolio.return_pct:.2f}%")
    
    # Trade 3: Close BTCUSDT position
    print(f"\n{'-' * 80}")
    print("Trade 3: Close BTCUSDT position")
    print(f"{'-' * 80}")
    
    signal3 = create_sample_trade_signal('BTCUSDT', base_time + timedelta(hours=3), TradeAction.CLOSE_LONG, 1.0)
    
    orders3 = generator.generate_orders(
        trade_signals=[signal3],
        market_data=market_data,
        portfolio_value=portfolio.total_value,
        current_positions=portfolio.position_quantities,
        current_prices=current_prices
    )
    
    if orders3:
        order = orders3[0]
        portfolio.update_from_order(order, execution_price=110.0, commission=10.0, slippage=5.0)
        print(f"  Executed: {order.side} {order.quantity:.4f} @ ${order.price:.2f}")
        print(f"  Cash: ${portfolio.cash:,.2f}")
        print(f"  Realized P&L: ${portfolio.realized_pnl:,.2f}")
        print(f"  Positions: {len(portfolio.positions)}")
    
    # Final snapshot
    snapshot3 = portfolio.take_snapshot()
    
    print(f"\n{'-' * 80}")
    print("Final Portfolio State")
    print(f"{'-' * 80}")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Market Value: ${portfolio.market_value:,.2f}")
    print(f"  Total Value: ${portfolio.total_value:,.2f}")
    print(f"  Realized P&L: ${portfolio.realized_pnl:,.2f}")
    print(f"  Unrealized P&L: ${portfolio.unrealized_pnl:,.2f}")
    print(f"  Total P&L: ${portfolio.total_pnl:,.2f}")
    print(f"  Return: {portfolio.return_pct:.2f}%")
    
    # Validate portfolio state
    validation = portfolio.validate_state()
    print(f"\nValidation:")
    print(f"  Is Valid: {validation['is_valid']}")
    if validation['issues']:
        print(f"  Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    # Get equity curve
    equity_curve = portfolio.get_equity_curve()
    print(f"\nEquity Curve:")
    print(f"  Number of snapshots: {len(equity_curve)}")
    if not equity_curve.empty:
        print(f"  Starting value: ${equity_curve['total_value'].iloc[0]:,.2f}")
        print(f"  Ending value: ${equity_curve['total_value'].iloc[-1]:,.2f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("Position Sizing and Portfolio State Management Examples")
    print("=" * 80)
    
    try:
        demonstrate_position_sizing()
        demonstrate_portfolio_tracking()
        
        print("\n" + "=" * 80)
        print("All demonstrations completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
