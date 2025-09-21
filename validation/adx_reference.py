#!/usr/bin/env python3
"""
Reference ADX implementation to understand the differences
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

def reference_adx(high, low, close, period=14):
    """
    Reference ADX implementation following the standard formula
    """
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    up_move = high - prev_high
    down_move = prev_low - low
    
    # +DM and -DM calculation
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Wilder's smoothing (approximated with EWM)
    alpha = 1.0 / period
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate +DI and -DI
    di_plus = 100 * (plus_dm_s / tr_s)
    di_minus = 100 * (minus_dm_s / tr_s)
    
    # Calculate DX
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    
    # Calculate ADX (smoothed DX)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    return adx, dx, di_plus, di_minus, tr_s, plus_dm_s, minus_dm_s

# Test with our validation data (last 10 points)
rust_data = [
    {"high": 100.82437163316823, "low": 100.73190077467709, "close": 100.80352274858349},
    {"high": 100.73848765330244, "low": 100.65329017084326, "close": 100.7208583647996},
    {"high": 100.6444784493145, "low": 100.56639091459043, "close": 100.6298687719777},
    {"high": 100.54586713913255, "low": 100.4745960051626, "close": 100.53405007053588},
    {"high": 100.44641613252993, "low": 100.38154324636265, "close": 100.43714007443823},
    {"high": 100.34997893046366, "low": 100.29096954709776, "close": 100.34297030163104},
    {"high": 100.26034331573479, "low": 100.20655632862437, "close": 100.25530929008353},
    {"high": 100.18107379242957, "low": 100.1317738647958, "close": 100.17770510435051},
    {"high": 100.1153613048175, "low": 100.06973257618232, "close": 100.11333507470512},
    {"high": 100.06588778628809, "low": 100.02304883665802, "close": 100.06487033334288},
]

# Add more data points for better calculation
extended_data = []
base_price = 101.0
for i in range(20):
    price = base_price - i * 0.02 + np.sin(i * 0.3) * 0.01
    extended_data.append({
        "high": price + 0.005,
        "low": price - 0.005,
        "close": price + np.sin(i * 0.1) * 0.002
    })

extended_data.extend(rust_data)

df = pd.DataFrame(extended_data)

print("Test data (last 10 rows):")
print(df.tail(10))

# Calculate using our reference method
ref_adx, ref_dx, ref_di_plus, ref_di_minus, ref_tr_s, ref_plus_dm_s, ref_minus_dm_s = reference_adx(
    df['high'], df['low'], df['close'], period=20
)

print(f"\nReference ADX (last 10): {ref_adx.tail(10).values}")
print(f"Reference DX (last 10): {ref_dx.tail(10).values}")
print(f"Reference +DI (last 10): {ref_di_plus.tail(10).values}")
print(f"Reference -DI (last 10): {ref_di_minus.tail(10).values}")

# Calculate using pandas_ta
pandas_adx = ta.adx(df['high'], df['low'], df['close'], length=20)
if pandas_adx is not None:
    adx_col = f'ADX_{20}'
    if adx_col in pandas_adx.columns:
        print(f"pandas_ta ADX (last 10): {pandas_adx[adx_col].tail(10).values}")
    else:
        print(f"Available columns: {pandas_adx.columns.tolist()}")
        print(f"pandas_ta ADX (last 10): {pandas_adx.iloc[:, -1].tail(10).values}")  # Last column
else:
    print("pandas_ta ADX returned None")

# Our Rust values for comparison (from the validation output)
rust_adx = [5.625372, 7.022000, 7.603284, 7.500408, 7.404384, 7.773342, 8.495738, 9.474720, 10.626550, 11.877955]
print(f"Rust ADX (last 10): {rust_adx}")

# Compare differences
if pandas_adx is not None and adx_col in pandas_adx.columns:
    pandas_values = pandas_adx[adx_col].tail(10).values
    ref_values = ref_adx.tail(10).values
    
    print(f"\nDifference pandas_ta - reference: {pandas_values - ref_values}")
    print(f"Difference Rust - reference: {np.array(rust_adx) - ref_values}")
    print(f"Difference Rust - pandas_ta: {np.array(rust_adx) - pandas_values}")

# Let's also check intermediate calculations
print(f"\nIntermediate values (last 5):")
print(f"TR smoothed: {ref_tr_s.tail(5).values}")
print(f"+DM smoothed: {ref_plus_dm_s.tail(5).values}")
print(f"-DM smoothed: {ref_minus_dm_s.tail(5).values}")