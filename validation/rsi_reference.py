#!/usr/bin/env python3
"""
Reference RSI implementation to understand the correct formula
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

def reference_rsi(close, period=14):
    """
    Reference RSI implementation using the standard Wilder's smoothing method:
    
    1. Calculate price changes
    2. Separate gains and losses
    3. Calculate initial average gain and loss (SMA for first period)
    4. Use Wilder's smoothing for subsequent values: new_avg = (old_avg * (period-1) + current_value) / period
    5. RS = avg_gain / avg_loss
    6. RSI = 100 - (100 / (1 + RS))
    """
    # Price changes
    delta = close.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    
    # Calculate RSI using Wilder's smoothing
    avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi, gains, losses, avg_gains, avg_losses, rs

def pandas_rsi_method(close, period=14):
    """
    Try to replicate pandas_ta RSI method
    """
    return ta.rsi(close, length=period)

# Test with our validation data
rust_data = [
    100.00698246490623, 100.03465652084947, 100.08060472524602, 100.14273157899713, 100.2183803397517,
    100.30446172031856, 100.39758716798637, 100.49420044530792, 100.59070301947172, 100.683570595204,
    100.7694597131385, 100.84530446216979, 100.9084038933649, 100.95650065500791, 100.98785077958154,
    101.0012836181371, 100.99624986670699, 100.97285471227939, 100.93187256866722, 100.8747398425263,
    100.80352274858349, 100.7208583647996, 100.6298687719777, 100.53405007053588, 100.43714007443823,
    100.34297030163104, 100.25530929008353, 100.17770510435051, 100.11333507470512, 100.06487033334288
]

close_series = pd.Series(rust_data)

print("Close prices (last 10):")
print(close_series.tail(10).values)

# Calculate RSI using our reference method
ref_rsi, gains, losses, avg_gains, avg_losses, rs = reference_rsi(close_series, period=14)

print(f"\nReference RSI (last 10): {ref_rsi.tail(10).values}")

# Calculate using pandas_ta
pandas_rsi = pandas_rsi_method(close_series, period=14)
print(f"pandas_ta RSI (last 10): {pandas_rsi.tail(10).values}")

# Our Rust output for comparison
rust_rsi = [99.00990099009901] * 30  # This is what Rust is producing (constant)
print(f"Rust RSI (last 10): {rust_rsi[-10:]}")

# Differences
if pandas_rsi is not None:
    diff_pandas = pandas_rsi - ref_rsi
    print(f"\nDifference pandas_ta - reference (last 10): {diff_pandas.tail(10).values}")
    
    diff_rust = np.array(rust_rsi[-10:]) - pandas_rsi.tail(10).values
    print(f"Difference Rust - pandas_ta (last 10): {diff_rust}")

# Let's also check the intermediate values
print(f"\nGains (last 10): {gains.tail(10).values}")
print(f"Losses (last 10): {losses.tail(10).values}")
print(f"Avg Gains (last 10): {avg_gains.tail(10).values}")
print(f"Avg Losses (last 10): {avg_losses.tail(10).values}")
print(f"RS (last 10): {rs.tail(10).values}")

# Check if the issue is with our EMA calculation
print(f"\nFirst few values:")
print(f"Reference RSI (first 20): {ref_rsi.head(20).values}")
print(f"pandas_ta RSI (first 20): {pandas_rsi.head(20).values}")

# The issue might be that our Rust implementation is not handling the initial values correctly
# Let's see what happens with a simple test case
simple_data = [100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 102, 101, 100]
simple_series = pd.Series(simple_data)
simple_ref_rsi, _, _, _, _, _ = reference_rsi(simple_series, period=14)
simple_pandas_rsi = pandas_rsi_method(simple_series, period=14)

print(f"\nSimple test data RSI:")
print(f"Reference: {simple_ref_rsi.values}")
print(f"pandas_ta: {simple_pandas_rsi.values if simple_pandas_rsi is not None else 'None'}")