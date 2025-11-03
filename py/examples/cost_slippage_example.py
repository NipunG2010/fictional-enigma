"""
Example demonstrating transaction cost and slippage modeling.

This example shows how to use the CostModel and SlippageModelEngine
to calculate realistic transaction costs and slippage for backtesting.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from imp.backtesting import (
    CostModel,
    SlippageModelEngine,
    SlippageModel,
    Order,
    TradeSignal,
    TradeAction,
    ProcessedSignal,
    SignalDirection,
    CostModelConfig,
    CostStructureType,
)


def create_sample_order(
    symbol: str = "BTCUSDT",
    side: str = "buy",
    quantity: float = 1.0,
    price: float = 50000.0,
    timestamp: datetime = None
) -> Order:
    """Create a sample order for testing."""
    if timestamp is None:
        timestamp = datetime.now()
    
    # Create a mock trade signal
    processed_signal = ProcessedSignal(
        timestamp=timestamp,
        symbol=symbol,
        direction=SignalDirection.BUY if side == "buy" else SignalDirection.SELL,
        strength=0.65,
        signal_source="ldc",
        raw_signals={'ldc': 0.7, 'mr': 0.5, 'tsmom': 0.6, 'fusion': 0.65},
        regime_state=1,
        regime_weights={'ldc': 0.4, 'mr': 0.3, 'tsmom': 0.3},
        metadata={}
    )
    
    trade_signal = TradeSignal(
        trade_id=f"trade_{timestamp.strftime('%Y%m%d_%H%M%S')}",
        timestamp=timestamp,
        symbol=symbol,
        action=TradeAction.BUY if side == "buy" else TradeAction.SELL,
        confidence=0.65,
        signal_source="ldc",
        reasoning="Sample trade signal for cost/slippage demonstration",
        processed_signal=processed_signal,
        metadata={}
    )
    
    order = Order(
        order_id=f"order_{timestamp.strftime('%Y%m%d_%H%M%S')}",
        timestamp=timestamp,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_type="market",
        trade_signal=trade_signal,
        position_size_method="percentage",
        risk_checks_passed={'all': True},
        metadata={}
    )
    
    return order


def create_sample_market_data(
    symbol: str = "BTCUSDT",
    num_bars: int = 100,
    base_price: float = 50000.0,
    base_volume: float = 1000.0
) -> pd.DataFrame:
    """Create sample market data for testing."""
    timestamps = [datetime.now() - timedelta(minutes=5*i) for i in range(num_bars)]
    timestamps.reverse()
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.001, num_bars)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate volume with some randomness
    volumes = base_volume * (1 + np.random.normal(0, 0.2, num_bars))
    volumes = np.abs(volumes)
    
    data = []
    for i, ts in enumerate(timestamps):
        high = prices[i] * (1 + abs(np.random.normal(0, 0.002)))
        low = prices[i] * (1 - abs(np.random.normal(0, 0.002)))
        
        data.append({
            'timestamp': ts,
            'symbol': symbol,
            'open': prices[i],
            'high': high,
            'low': low,
            'close': prices[i],
            'volume': volumes[i]
        })
    
    return pd.DataFrame(data)


def demonstrate_cost_model():
    """Demonstrate the CostModel functionality."""
    print("=" * 80)
    print("COST MODEL DEMONSTRATION")
    print("=" * 80)
    
    # Create cost model configuration for crypto
    config = CostModelConfig(
        commission_rate=0.001,  # 0.1%
        min_commission=1.0,
        asset_class=CostStructureType.CRYPTO,
        slippage_linear_impact=0.0001,
        slippage_sqrt_impact=0.001
    )
    
    # Initialize cost model
    cost_model = CostModel(config)
    
    # Create sample orders
    orders = [
        create_sample_order("BTCUSDT", "buy", 1.0, 50000.0),
        create_sample_order("BTCUSDT", "sell", 0.5, 51000.0),
        create_sample_order("ETHUSDT", "buy", 10.0, 3000.0),
    ]
    
    # Create sample market data
    market_data = create_sample_market_data()
    
    print("\nCalculating costs for sample orders...")
    print("-" * 80)
    
    # Calculate costs for each order
    for order in orders:
        cost = cost_model.calculate_trade_cost(order, market_data)
        
        print(f"\nOrder: {order.side.upper()} {order.quantity} {order.symbol} @ ${order.price:,.2f}")
        print(f"  Notional Value: ${cost.notional_value:,.2f}")
        print(f"  Commission: ${cost.commission:.2f}")
        print(f"  Spread Cost: ${cost.spread_cost:.2f}")
        print(f"  Market Impact: ${cost.market_impact:.2f}")
        print(f"  Slippage: ${cost.slippage:.2f}")
        print(f"  Total Cost: ${cost.total_cost:.2f} ({cost.cost_bps:.2f} bps)")
        print(f"  Execution Price: ${cost.execution_price:,.2f} (impact: {cost.price_impact_pct:.4f}%)")
    
    # Get cost breakdown
    print("\n" + "=" * 80)
    print("COST BREAKDOWN ANALYSIS")
    print("=" * 80)
    
    breakdown = cost_model.get_cost_breakdown()
    
    print(f"\nTotal Trades: {breakdown['num_trades']}")
    print(f"Total Cost: ${breakdown['total_cost']:,.2f}")
    print(f"Total Notional: ${breakdown['total_notional']:,.2f}")
    print(f"Average Cost: {breakdown['avg_cost_bps']:.2f} bps")
    print(f"Cost to Notional Ratio: {breakdown['cost_to_notional_ratio']:.4%}")
    
    print("\nCost Components:")
    for component, details in breakdown['cost_components'].items():
        print(f"  {component.replace('_', ' ').title()}:")
        print(f"    Total: ${details['total']:,.2f}")
        print(f"    Percentage: {details['percentage']:.2f}%")
        print(f"    Avg per Trade: ${details['avg_per_trade']:.2f}")
    
    # Get statistics
    print("\n" + "=" * 80)
    print("COST STATISTICS")
    print("=" * 80)
    
    stats = cost_model.get_statistics()
    print(f"\nMin Cost: {stats['min_cost_bps']:.2f} bps")
    print(f"Max Cost: {stats['max_cost_bps']:.2f} bps")
    print(f"Median Cost: {stats['median_cost_bps']:.2f} bps")
    print(f"Std Dev: {stats['std_cost_bps']:.2f} bps")
    print(f"Avg Price Impact: {stats['avg_price_impact_pct']:.4f}%")
    print(f"Max Price Impact: {stats['max_price_impact_pct']:.4f}%")


def demonstrate_slippage_model():
    """Demonstrate the SlippageModelEngine functionality."""
    print("\n\n" + "=" * 80)
    print("SLIPPAGE MODEL DEMONSTRATION")
    print("=" * 80)
    
    # Create cost model configuration
    config = CostModelConfig(
        commission_rate=0.001,
        min_commission=1.0,
        asset_class=CostStructureType.CRYPTO,
        slippage_linear_impact=0.0001,
        slippage_sqrt_impact=0.001
    )
    
    # Create sample order
    order = create_sample_order("BTCUSDT", "buy", 2.0, 50000.0)
    market_data = create_sample_market_data()
    
    print("\nComparing different slippage models...")
    print("-" * 80)
    
    # Test different slippage models
    models = [
        SlippageModel.LINEAR,
        SlippageModel.SQUARE_ROOT,
        SlippageModel.COMBINED,
        SlippageModel.VOLATILITY_ADJUSTED
    ]
    
    for model in models:
        engine = SlippageModelEngine(config, model)
        estimate = engine.estimate_slippage(order, market_data)
        
        print(f"\n{model.value.upper().replace('_', ' ')} Model:")
        print(f"  Total Slippage: ${estimate.total_slippage:.2f} ({estimate.slippage_bps:.2f} bps)")
        print(f"  Linear Impact: ${estimate.linear_impact:.2f}")
        print(f"  Sqrt Impact: ${estimate.sqrt_impact:.2f}")
        print(f"  Volatility Adj: ${estimate.volatility_adjustment:.2f}")
        print(f"  Execution Price: ${estimate.execution_price:,.2f}")
        print(f"  Price Impact: {estimate.price_impact_pct:.4f}%")
        print(f"  Participation Rate: {estimate.participation_rate:.4%}")
        print(f"  Volatility: {estimate.volatility:.2%}")
    
    # Detailed analysis with combined model
    print("\n" + "=" * 80)
    print("DETAILED SLIPPAGE ANALYSIS (Combined Model)")
    print("=" * 80)
    
    engine = SlippageModelEngine(config, SlippageModel.COMBINED)
    
    # Create multiple orders with different sizes
    test_orders = [
        create_sample_order("BTCUSDT", "buy", 0.5, 50000.0),
        create_sample_order("BTCUSDT", "buy", 1.0, 50000.0),
        create_sample_order("BTCUSDT", "buy", 2.0, 50000.0),
        create_sample_order("BTCUSDT", "buy", 5.0, 50000.0),
    ]
    
    print("\nSlippage vs Order Size:")
    print("-" * 80)
    
    for test_order in test_orders:
        estimate = engine.estimate_slippage(test_order, market_data)
        print(f"Order Size: {test_order.quantity:5.1f} BTC | "
              f"Slippage: ${estimate.total_slippage:8.2f} ({estimate.slippage_bps:6.2f} bps) | "
              f"Participation: {estimate.participation_rate:6.2%}")
    
    # Get slippage breakdown
    print("\n" + "=" * 80)
    print("SLIPPAGE BREAKDOWN")
    print("=" * 80)
    
    breakdown = engine.get_slippage_breakdown()
    
    print(f"\nTotal Trades: {breakdown['num_trades']}")
    print(f"Total Slippage: ${breakdown['total_slippage']:,.2f}")
    print(f"Average Slippage: {breakdown['avg_slippage_bps']:.2f} bps")
    print(f"Avg Participation Rate: {breakdown['avg_participation_rate']:.4%}")
    print(f"Max Participation Rate: {breakdown['max_participation_rate']:.4%}")
    print(f"Avg Volatility: {breakdown['avg_volatility']:.2%}")
    
    print("\nSlippage Components:")
    for component, details in breakdown['components'].items():
        print(f"  {component.replace('_', ' ').title()}:")
        print(f"    Total: ${details['total']:,.2f}")
        print(f"    Percentage: {details['percentage']:.2f}%")
        print(f"    Avg per Trade: ${details['avg_per_trade']:.2f}")


def demonstrate_integrated_costs():
    """Demonstrate integrated cost and slippage calculation."""
    print("\n\n" + "=" * 80)
    print("INTEGRATED COST & SLIPPAGE ANALYSIS")
    print("=" * 80)
    
    # Create configuration
    config = CostModelConfig(
        commission_rate=0.001,
        min_commission=1.0,
        asset_class=CostStructureType.CRYPTO,
        slippage_linear_impact=0.0001,
        slippage_sqrt_impact=0.001
    )
    
    # Initialize both models
    cost_model = CostModel(config)
    slippage_engine = SlippageModelEngine(config, SlippageModel.VOLATILITY_ADJUSTED)
    
    # Create sample order
    order = create_sample_order("BTCUSDT", "buy", 1.5, 50000.0)
    market_data = create_sample_market_data()
    
    # Calculate costs
    trade_cost = cost_model.calculate_trade_cost(order, market_data)
    slippage_estimate = slippage_engine.estimate_slippage(order, market_data)
    
    print(f"\nOrder: {order.side.upper()} {order.quantity} {order.symbol} @ ${order.price:,.2f}")
    print(f"Notional Value: ${order.notional_value:,.2f}")
    print("-" * 80)
    
    print("\nCost Breakdown:")
    print(f"  Commission: ${trade_cost.commission:8.2f}")
    print(f"  Spread Cost: ${trade_cost.spread_cost:8.2f}")
    print(f"  Market Impact: ${trade_cost.market_impact:8.2f}")
    print(f"  Slippage: ${trade_cost.slippage:8.2f}")
    print(f"  {'─' * 30}")
    print(f"  Total Cost: ${trade_cost.total_cost:8.2f} ({trade_cost.cost_bps:.2f} bps)")
    
    print("\nSlippage Detail:")
    print(f"  Linear Impact: ${slippage_estimate.linear_impact:8.2f}")
    print(f"  Sqrt Impact: ${slippage_estimate.sqrt_impact:8.2f}")
    print(f"  Vol Adjustment: ${slippage_estimate.volatility_adjustment:8.2f}")
    print(f"  {'─' * 30}")
    print(f"  Total Slippage: ${slippage_estimate.total_slippage:8.2f} ({slippage_estimate.slippage_bps:.2f} bps)")
    
    print("\nExecution Details:")
    print(f"  Order Price: ${order.price:,.2f}")
    print(f"  Execution Price: ${trade_cost.execution_price:,.2f}")
    print(f"  Price Impact: {trade_cost.price_impact_pct:.4f}%")
    print(f"  Participation Rate: {slippage_estimate.participation_rate:.4%}")
    print(f"  Market Volatility: {slippage_estimate.volatility:.2%}")
    
    # Calculate net proceeds/cost
    if order.side == 'buy':
        total_cost = order.notional_value + trade_cost.total_cost
        print(f"\nTotal Cost to Buy: ${total_cost:,.2f}")
        print(f"Effective Price: ${total_cost / order.quantity:,.2f} per unit")
    else:
        net_proceeds = order.notional_value - trade_cost.total_cost
        print(f"\nNet Proceeds from Sale: ${net_proceeds:,.2f}")
        print(f"Effective Price: ${net_proceeds / order.quantity:,.2f} per unit")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_cost_model()
    demonstrate_slippage_model()
    demonstrate_integrated_costs()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
