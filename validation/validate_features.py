#!/usr/bin/env python3
"""
Feature Validation Script

This script compares our Rust feature calculations with pandas_ta (Python)
to ensure accuracy and correctness of our implementations.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import subprocess
import sys
from pathlib import Path

def generate_test_data_from_rust(rust_ohlcv):
    """Convert Rust OHLCV data to pandas DataFrame"""
    if rust_ohlcv is None:
        return None
    
    return pd.DataFrame(rust_ohlcv)

def compute_python_features(df, rsi_period=14, ma_period=20):
    """Compute features using pandas_ta"""
    print("🐍 Computing features with pandas_ta...")
    
    # Create a copy for pandas_ta (it modifies the dataframe)
    df_ta = df.copy()
    
    features = {}
    
    # RSI
    features['rsi'] = ta.rsi(df_ta['close'], length=rsi_period)
    
    # Moving Averages
    features['sma_20'] = ta.sma(df_ta['close'], length=ma_period)
    features['ema_20'] = ta.ema(df_ta['close'], length=ma_period)
    
    # Standard Deviation (rolling)
    features['std_20'] = df_ta['close'].rolling(window=ma_period).std()
    
    # Z-Score (manual calculation)
    rolling_mean = df_ta['close'].rolling(window=ma_period).mean()
    rolling_std = df_ta['close'].rolling(window=ma_period).std()
    features['zscore_20'] = (df_ta['close'] - rolling_mean) / rolling_std
    
    # Momentum (percentage change)
    features['momentum'] = df_ta['close'].pct_change()
    
    # WaveTrend (custom implementation similar to our Rust version)
    features['wavetrend_1'], features['wavetrend_2'] = compute_wavetrend(df_ta, n1=ma_period, n2=rsi_period)
    
    # CCI
    features['cci'] = ta.cci(df_ta['high'], df_ta['low'], df_ta['close'], length=ma_period)
    
    # ADX
    features['adx'] = ta.adx(df_ta['high'], df_ta['low'], df_ta['close'], length=ma_period)['ADX_' + str(ma_period)]
    
    return features

def compute_wavetrend(df, n1=10, n2=21):
    """Custom WaveTrend implementation to match our Rust version"""
    # Typical Price
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    
    # ESA (Exponential Smoothing Average)
    esa = tp.ewm(span=n1, adjust=False).mean()
    
    # D (Absolute difference smoothed)
    d = (tp - esa).abs().ewm(span=n1, adjust=False).mean()
    
    # CI (Channel Index)
    ci = (tp - esa) / (0.015 * d)
    ci = ci.fillna(0.0)  # Handle division by zero
    
    # WT1 (WaveTrend 1)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    
    # WT2 (WaveTrend 2 - signal line)
    wt2 = wt1.ewm(span=4, adjust=False).mean()
    
    return wt1, wt2

def run_rust_features():
    """Run our Rust feature computation and return results"""
    print("🦀 Running Rust feature computation...")
    
    rust_dir = Path(__file__).parent.parent / "rust" / "feature-pipeline"
    
    try:
        # Run our Rust validation output example
        result = subprocess.run(
            ["cargo", "run", "--example", "validation_output"],
            cwd=rust_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"❌ Rust execution failed: {result.stderr}")
            return None, None
            
        # Parse JSON output
        try:
            data = json.loads(result.stdout)
            return data.get('ohlcv_data'), data.get('features')
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Rust JSON output: {e}")
            return None, None
        
    except subprocess.TimeoutExpired:
        print("❌ Rust execution timed out")
        return None, None
    except Exception as e:
        print(f"❌ Error running Rust: {e}")
        return None, None



def compare_features(python_features, rust_features, tolerance=1e-2):
    """Compare Python and Rust feature calculations"""
    print("\n📊 Feature Comparison Results:")
    print("=" * 60)
    
    if rust_features is None:
        print("❌ Cannot compare - Rust features not available")
        return
    
    comparison_results = {}
    
    feature_mapping = {
        'rsi': 'rsi',
        'sma_20': 'sma_20', 
        'ema_20': 'ema_20',
        'std_20': 'std_20',
        'zscore_20': 'zscore_20',
        'momentum': 'momentum',
        'wavetrend_1': 'wavetrend_1',
        'wavetrend_2': 'wavetrend_2',
        'cci': 'cci',
        'adx': 'adx'
    }
    
    for py_name, rust_field in feature_mapping.items():
        print(f"\n🔍 {py_name.upper()}:")
        
        # Get Python values (drop NaN values)
        py_series = python_features[py_name]
        py_valid_indices = py_series.dropna().index
        py_values = py_series.dropna().values
        
        if len(py_values) == 0:
            print(f"   ⚠️  No valid Python values")
            continue
        
        # Get corresponding Rust values
        rust_values = []
        rust_indices = []
        for i, feature in enumerate(rust_features):
            if rust_field in feature and feature[rust_field] is not None:
                rust_values.append(feature[rust_field])
                rust_indices.append(i)
        
        if len(rust_values) == 0:
            print(f"   ⚠️  No valid Rust values")
            continue
        
        # Find overlapping indices for comparison
        # Take the last N values where both have data
        n_compare = min(len(py_values), len(rust_values), 10)
        
        py_vals = py_values[-n_compare:]
        rust_vals = rust_values[-n_compare:]
        
        # Calculate differences
        diffs = np.abs(np.array(py_vals) - np.array(rust_vals))
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)
        
        # Calculate relative differences for percentage-based tolerance
        rel_diffs = diffs / (np.abs(py_vals) + 1e-10)  # Avoid division by zero
        max_rel_diff = np.max(rel_diffs)
        
        # Determine if values match within tolerance (absolute or relative)
        matches = (max_diff < tolerance) or (max_rel_diff < 0.05)  # 5% relative tolerance
        
        print(f"   Python:    {[f'{v:.6f}' for v in py_vals]}")
        print(f"   Rust:      {[f'{v:.6f}' for v in rust_vals]}")
        print(f"   Max Diff:  {max_diff:.6f}")
        print(f"   Mean Diff: {mean_diff:.6f}")
        print(f"   Max Rel:   {max_rel_diff:.2%}")
        print(f"   Status:    {'✅ MATCH' if matches else '❌ MISMATCH'}")
        
        comparison_results[py_name] = {
            'matches': matches,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'max_rel_diff': max_rel_diff,
            'python_values': py_vals.tolist(),
            'rust_values': rust_vals,
            'n_compared': n_compare
        }
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print("=" * 30)
    matches = sum(1 for r in comparison_results.values() if r['matches'])
    total = len(comparison_results)
    print(f"Features matching: {matches}/{total}")
    
    if matches == total:
        print("🎉 All features match! Rust implementation is accurate.")
    elif matches > total * 0.7:
        print("✅ Most features match. Minor differences may be due to implementation details.")
    else:
        print("⚠️  Significant differences found. Review implementation.")
        
    return comparison_results

def main():
    print("🧪 Feature Validation: Rust vs Python (pandas_ta)")
    print("=" * 60)
    
    # Run Rust features first to get the exact same data
    print("🦀 Getting Rust data and features...")
    rust_ohlcv, rust_features = run_rust_features()
    
    if rust_ohlcv is None or rust_features is None:
        print("❌ Failed to get Rust data. Exiting.")
        return
    
    # Convert Rust OHLCV to DataFrame
    df = generate_test_data_from_rust(rust_ohlcv)
    print(f"📊 Using {len(df)} bars of OHLCV data from Rust")
    
    # Compute Python features on the same data
    python_features = compute_python_features(df)
    
    # Compare results
    results = compare_features(python_features, rust_features)
    
    # Save detailed results
    output_file = Path(__file__).parent / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'test_data_shape': df.shape,
            'comparison_results': results,
            'python_feature_stats': {
                name: {
                    'count': int(series.count()),
                    'mean': float(series.mean()) if series.count() > 0 else None,
                    'std': float(series.std()) if series.count() > 0 else None,
                    'min': float(series.min()) if series.count() > 0 else None,
                    'max': float(series.max()) if series.count() > 0 else None,
                }
                for name, series in python_features.items()
            }
        }, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()