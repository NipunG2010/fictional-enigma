#!/usr/bin/env python3
"""
Simple test to verify signal processing classes work correctly.
This test uses minimal dependencies to validate the core functionality.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Test the imports
try:
    from imp.backtesting.config import SignalProcessingConfig, FusionMethod
    from imp.backtesting.signal_processor import SignalProcessor, ProcessedSignal, SignalDirection
    from imp.backtesting.trade_signal_generator import TradeSignalGenerator, TradeAction
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_configuration():
    """Test configuration creation."""
    try:
        config = SignalProcessingConfig(
            thresholds={'ldc': 0.5, 'mr': 0.4, 'tsmom': 0.3},
            filters={'min_signal_strength': 0.2, 'max_trades_per_day': 5},
            fusion_method=FusionMethod.STATIC_WEIGHTED,
            static_weights={'ldc': 0.4, 'mr': 0.3, 'tsmom': 0.3}
        )
        print("✓ Configuration creation successful")
        return config
    except Exception as e:
        print(f"✗ Configuration creation failed: {e}")
        return None


def test_signal_processor(config):
    """Test SignalProcessor initialization."""
    try:
        processor = SignalProcessor(config)
        print("✓ SignalProcessor initialization successful")
        return processor
    except Exception as e:
        print(f"✗ SignalProcessor initialization failed: {e}")
        return None


def test_trade_generator(config):
    """Test TradeSignalGenerator initialization."""
    try:
        generator = TradeSignalGenerator(config)
        print("✓ TradeSignalGenerator initialization successful")
        return generator
    except Exception as e:
        print(f"✗ TradeSignalGenerator initialization failed: {e}")
        return None


def test_processed_signal():
    """Test ProcessedSignal creation."""
    try:
        signal = ProcessedSignal(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            direction=SignalDirection.BUY,
            strength=0.75,
            signal_source="ldc",
            raw_signals={'ldc': 0.8, 'mr': -0.2, 'tsmom': 0.5}
        )
        print("✓ ProcessedSignal creation successful")
        return signal
    except Exception as e:
        print(f"✗ ProcessedSignal creation failed: {e}")
        return None


def main():
    """Run basic functionality tests."""
    print("Signal Processing Basic Functionality Test")
    print("=" * 45)
    
    # Test configuration
    config = test_configuration()
    if not config:
        return
    
    # Test signal processor
    processor = test_signal_processor(config)
    if not processor:
        return
    
    # Test trade generator
    generator = test_trade_generator(config)
    if not generator:
        return
    
    # Test data structures
    signal = test_processed_signal()
    if not signal:
        return
    
    # Test basic methods
    try:
        # Test signal processor methods
        weights = processor._get_regime_weights(1)
        print(f"✓ Regime weights retrieval: {weights}")
        
        # Test fusion
        fused = processor._apply_fusion(
            {'ldc': 0.6, 'mr': -0.3, 'tsmom': 0.4},
            {'ldc': 0.5, 'mr': 0.3, 'tsmom': 0.2}
        )
        print(f"✓ Signal fusion: {fused:.3f}")
        
        # Test trade generator methods
        stats = generator.get_trading_statistics()
        print(f"✓ Trading statistics: {len(stats)} fields")
        
        audit = generator.get_audit_trail()
        print(f"✓ Audit trail: {len(audit)} entries")
        
    except Exception as e:
        print(f"✗ Method testing failed: {e}")
        return
    
    print("\n🎉 All basic functionality tests passed!")
    print("\nThe signal processing and trade generation system is ready for use.")


if __name__ == "__main__":
    main()