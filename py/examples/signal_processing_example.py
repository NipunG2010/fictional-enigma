#!/usr/bin/env python3
"""
Example demonstrating signal processing and trade signal generation.

This script shows how to use the SignalProcessor and TradeSignalGenerator
classes to convert raw signals into actionable trading decisions.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the parent directory to the path so we can import imp
sys.path.append(str(Path(__file__).parent.parent))

from imp.backtesting import (
    SignalProcessor,
    TradeSignalGenerator,
    SignalProcessingConfig,
    FusionMethod
)


def create_sample_signal_data() -> pd.DataFrame:
    """Create sample signal data for demonstration."""
    np.random.seed(42)  # For reproducible results
    
    # Generate 100 time periods (5-minute intervals)
    timestamps = [
        datetime(2024, 1, 1, 9, 0) + timedelta(minutes=5*i) 
        for i in range(100)
    ]
    
    # Generate sample signals with some correlation
    n_periods = len(timestamps)
    
    # LDC signals (trend following)
    ldc_trend = np.cumsum(np.random.normal(0, 0.01, n_periods))
    s_ldc = np.tanh(ldc_trend) + np.random.normal(0, 0.1, n_periods)
    
    # Mean reversion signals (opposite to trend)
    s_mr = -0.5 * s_ldc + np.random.normal(0, 0.15, n_periods)
    
    # Time series momentum
    s_tsmom = 0.3 * s_ldc + np.random.normal(0, 0.12, n_periods)
    
    # Simple regime states (0, 1, 2)
    regime_states = np.random.choice([0, 1, 2], n_periods, p=[0.4, 0.4, 0.2])
    
    signal_data = pd.DataFrame({
        'timestamp': timestamps,
        's_ldc': s_ldc,
        's_mr': s_mr,
        's_tsmom': s_tsmom,
        'regime_state': regime_states
    })
    
    return signal_data


def main():
    """Demonstrate signal processing and trade generation."""
    print("Signal Processing and Trade Generation Example")
    print("=" * 50)
    
    # Create configuration
    config = SignalProcessingConfig(
        thresholds={'ldc': 0.4, 'mr': 0.3, 'tsmom': 0.35},
        filters={
            'min_signal_strength': 0.2,
            'max_trades_per_day': 8,
            'cooldown_periods': 3,
            'max_consecutive_signals': 2,
            'position_change_threshold': 0.4
        },
        fusion_method=FusionMethod.STATIC_WEIGHTED,
        static_weights={'ldc': 0.4, 'mr': 0.3, 'tsmom': 0.3}
    )
    
    # Create sample data
    signal_data = create_sample_signal_data()
    print(f"Created {len(signal_data)} sample signal periods")
    print(f"Signal data columns: {list(signal_data.columns)}")
    print()
    
    # Initialize signal processor
    signal_processor = SignalProcessor(config)
    
    # Process signals
    print("Processing signals...")
    processed_signals = signal_processor.process_signals(
        signal_data, 
        symbol="BTCUSDT"
    )
    
    print(f"Generated {len(processed_signals)} processed signals")
    
    # Show some processed signals
    if processed_signals:
        print("\nSample processed signals:")
        for i, signal in enumerate(processed_signals[:5]):
            print(f"  {i+1}. {signal.timestamp.strftime('%H:%M')} - "
                  f"{signal.direction.value.upper()} "
                  f"(strength: {signal.strength:.3f}, "
                  f"source: {signal.signal_source})")
    
    # Initialize trade signal generator
    trade_generator = TradeSignalGenerator(config)
    
    # Generate trade signals
    print("\nGenerating trade signals...")
    trade_signals = trade_generator.generate_trade_signals(processed_signals)
    
    print(f"Generated {len(trade_signals)} trade signals")
    
    # Show trade signals
    if trade_signals:
        print("\nTrade signals:")
        for i, trade in enumerate(trade_signals):
            print(f"  {i+1}. {trade.timestamp.strftime('%H:%M')} - "
                  f"{trade.action.value.upper()} "
                  f"(confidence: {trade.confidence:.3f}) "
                  f"- {trade.reasoning}")
    
    # Show statistics
    stats = trade_generator.get_trading_statistics()
    print(f"\nTrading Statistics:")
    print(f"  Total signals processed: {stats['total_signals_processed']}")
    print(f"  Trade signals generated: {stats['trade_signals_generated']}")
    print(f"  Signals filtered: {stats['signals_filtered']}")
    print(f"  Filter rate: {stats['filter_rate']:.1%}")
    
    if stats['filter_reasons']:
        print(f"  Filter reasons:")
        for reason, count in stats['filter_reasons'].items():
            print(f"    {reason}: {count}")
    
    # Show audit trail sample
    audit_trail = trade_generator.get_audit_trail()
    print(f"\nAudit trail entries: {len(audit_trail)}")
    
    if audit_trail:
        print("\nSample audit trail entries:")
        for entry in audit_trail[:3]:
            event_type = entry.get('event_type', 'unknown')
            timestamp = entry.get('timestamp', 'unknown')
            symbol = entry.get('symbol', 'unknown')
            print(f"  {timestamp} - {event_type} for {symbol}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()