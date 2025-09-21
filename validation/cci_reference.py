#!/usr/bin/env python3
"""
Reference CCI implementation to understand the correct formula
"""

import pandas as pd
import numpy as np

def reference_cci(high, low, close, period=20):
    """
    Reference CCI implementation based on the standard formula:
    
    CCI = (Typical Price - SMA of Typical Price) / (0.015 * Mean Absolute Deviation)
    
    Where Mean Absolute Deviation is the mean of |TP - SMA(TP)| over the period
    """
    # Typical Price
    tp = (high + low + close) / 3.0
    
    # Simple Moving Average of Typical Price
    tp_sma = tp.rolling(window=period, min_periods=period).mean()
    
    # Mean Absolute Deviation
    # This is the rolling mean of the absolute deviations from the SMA
    mad = (tp - tp_sma).abs().rolling(window=period, min_periods=period).mean()
    
    # CCI calculation
    cci = (tp - tp_sma) / (0.015 * mad)
    
    return cci, tp, tp_sma, mad

# Test with our validation data
validation_data = {
    'high': [100.087, 100.114, 100.158, 100.218, 100.292, 100.376, 100.467, 100.561, 100.654, 100.744,
             100.827, 100.899, 100.958, 101.003, 101.030, 101.040, 101.031, 101.004, 100.960, 100.899],
    'low': [99.957, 99.979, 100.020, 100.077, 100.147, 100.229, 100.318, 100.411, 100.504, 100.595,
            100.679, 100.754, 100.817, 100.866, 100.898, 100.913, 100.911, 100.890, 100.853, 100.799],
    'close': [100.007, 100.035, 100.081, 100.143, 100.218, 100.304, 100.398, 100.494, 100.591, 100.684,
              100.770, 100.845, 100.908, 100.957, 100.988, 101.001, 100.996, 100.973, 100.932, 100.875]
}

df = pd.DataFrame(validation_data)

print("Validation Data (first 10 rows):")
print(df.head(10))

# Calculate CCI with period=20 (but we only have 20 data points)
cci, tp, tp_sma, mad = reference_cci(df['high'], df['low'], df['close'], period=5)

print(f"\nTypical Price (last 10): {tp.tail(10).values}")
print(f"TP SMA (last 10): {tp_sma.tail(10).values}")
print(f"MAD (last 10): {mad.tail(10).values}")
print(f"Reference CCI (last 10): {cci.tail(10).values}")

# Compare with pandas_ta
try:
    import pandas_ta as ta
    cci_ta = ta.cci(df['high'], df['low'], df['close'], length=5)
    print(f"pandas_ta CCI (last 10): {cci_ta.tail(10).values if cci_ta is not None else 'None'}")
    
    if cci_ta is not None:
        diff = cci_ta - cci
        print(f"Difference (last 10): {diff.tail(10).values}")
        
        # Check if the ratio is consistent (maybe pandas_ta uses a different constant)
        ratio = cci_ta / cci
        print(f"Ratio pandas_ta/reference (last 10): {ratio.tail(10).values}")
        
except ImportError:
    print("pandas_ta not available")

# Let's also test with the exact data from our Rust output
print("\n" + "="*60)
print("Testing with exact Rust validation data...")

rust_data = [
    {"high": 100.08698805087815, "low": 99.95697897367377, "close": 100.00698246490623},
    {"high": 100.11353360083459, "low": 99.97915511988965, "close": 100.03465652084947},
    {"high": 100.15807089985852, "low": 100.01969460901708, "close": 100.08060472524602},
    {"high": 100.21850865311158, "low": 100.07660157010045, "close": 100.14273157899713},
    {"high": 100.29219670550481, "low": 100.14731564329946, "close": 100.2183803397517},
]

rust_df = pd.DataFrame(rust_data)
rust_cci, rust_tp, rust_tp_sma, rust_mad = reference_cci(rust_df['high'], rust_df['low'], rust_df['close'], period=3)

print("Rust data CCI calculation:")
print(f"Typical Price: {rust_tp.values}")
print(f"TP SMA: {rust_tp_sma.values}")
print(f"MAD: {rust_mad.values}")
print(f"Reference CCI: {rust_cci.values}")

# What our Rust code should produce for the same data
expected_rust_cci = [0.0, 133.33333333333334, 149.41118281146973, 157.89547551072917, 162.32114628995873]
print(f"Rust output: {expected_rust_cci}")

print(f"Difference: {rust_cci.values - np.array(expected_rust_cci)}")