#!/usr/bin/env python3
"""
Debug CCI calculation differences
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

# Create simple test data
data = {
    'high': [101, 102, 103, 102, 101, 100, 99, 100, 101, 102],
    'low': [99, 100, 101, 100, 99, 98, 97, 98, 99, 100],
    'close': [100, 101, 102, 101, 100, 99, 98, 99, 100, 101]
}

df = pd.DataFrame(data)

print("Test Data:")
print(df)

# Calculate CCI manually
def manual_cci(df, period=20):
    # Typical Price
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    print(f"\nTypical Price: {tp.values}")
    
    # SMA of Typical Price
    tp_sma = tp.rolling(window=period).mean()
    print(f"TP SMA: {tp_sma.values}")
    
    # Mean Absolute Deviation
    mad = (tp - tp_sma).abs().rolling(window=period).mean()
    print(f"MAD: {mad.values}")
    
    # CCI
    cci = (tp - tp_sma) / (0.015 * mad)
    print(f"Manual CCI: {cci.values}")
    
    return cci

# Calculate using pandas_ta with smaller period first
cci_ta_3 = ta.cci(df['high'], df['low'], df['close'], length=3)
print(f"\npandas_ta CCI (period=3): {cci_ta_3}")

# Calculate manually with period=3
manual_cci_3 = manual_cci(df, 3)

if cci_ta_3 is not None:
    print(f"Difference (period=3): {(cci_ta_3 - manual_cci_3).values}")
else:
    print("pandas_ta CCI returned None")

# Try with the full dataset from our validation
print("\n" + "="*50)
print("Let's check what pandas_ta.cci expects...")
print("Available parameters:", ta.cci.__doc__ if hasattr(ta.cci, '__doc__') else "No docs")

# Try different approach
try:
    # Create a proper DataFrame with index
    df_indexed = df.copy()
    df_indexed.index = pd.RangeIndex(len(df_indexed))
    
    cci_result = ta.cci(high=df_indexed['high'], low=df_indexed['low'], close=df_indexed['close'], length=3)
    print(f"CCI with explicit parameters: {cci_result}")
except Exception as e:
    print(f"Error: {e}")

# Let's also check what happens with our actual validation data
print("\nTesting with validation data sample...")
validation_data = {
    'high': [100.087, 100.114, 100.158, 100.218, 100.292],
    'low': [99.957, 99.979, 100.020, 100.077, 100.147],
    'close': [100.007, 100.035, 100.081, 100.143, 100.218]
}
val_df = pd.DataFrame(validation_data)
val_cci = ta.cci(val_df['high'], val_df['low'], val_df['close'], length=3)
print(f"Validation CCI: {val_cci}")