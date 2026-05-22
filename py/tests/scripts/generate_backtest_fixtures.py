#!/usr/bin/env python3
"""
Generate sample Parquet fixture files for backtesting end-to-end tests.

Produces:
  - backtest_fixtures/signals/signals.parquet    (signal data for 3 symbols)
  - backtest_fixtures/market_data/market.parquet  (OHLCV data for 3 symbols)

The data is deterministic (fixed seed) and represents ~30 days of 1-hour bars
for BTCUSDT, ETHUSDT, and SOLUSDT.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "backtest_fixtures"

RNG = np.random.RandomState(42)

N_DAYS = 30
HOURS_PER_DAY = 24
N_PERIODS = N_DAYS * HOURS_PER_DAY  # 720

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BASE_TIMESTAMP = pd.Timestamp("2024-01-01")


def _base_price(symbol: str) -> float:
    return {"BTCUSDT": 42000.0, "ETHUSDT": 2200.0, "SOLUSDT": 100.0}[symbol]


def _generate_symbol_data(symbol: str, seed_offset: int):
    """Generate signal + OHLCV data for one symbol."""
    local_rng = np.random.RandomState(42 + seed_offset)

    timestamps = pd.date_range(BASE_TIMESTAMP, periods=N_PERIODS, freq="1h", inclusive="left")

    base = _base_price(symbol)
    returns = local_rng.randn(N_PERIODS) * 0.002    # ~0.2% hourly vol
    price = base * np.exp(np.cumsum(returns))

    # Synthetic signals
    s_ldc = 0.3 * np.sin(np.linspace(0, 8 * np.pi, N_PERIODS)) + 0.1 * local_rng.randn(N_PERIODS)
    s_mr = 0.2 * np.cos(np.linspace(0, 6 * np.pi, N_PERIODS)) + 0.1 * local_rng.randn(N_PERIODS)
    s_tsmom = 0.4 * np.tanh(np.cumsum(returns) * 50) + 0.05 * local_rng.randn(N_PERIODS)

    regime_state = (np.abs(np.fft.fft(local_rng.randn(N_PERIODS))).cumsum() % 3).astype(int)

    ohlc = {
        "open":  price * (1 + 0.001 * local_rng.randn(N_PERIODS)),
        "high":  price * (1 + 0.005 * np.abs(local_rng.randn(N_PERIODS))),
        "low":   price * (1 - 0.005 * np.abs(local_rng.randn(N_PERIODS))),
        "close": price,
        "volume": np.abs(local_rng.exponential(scale=2000, size=N_PERIODS)),
    }

    df_signal = pd.DataFrame({
        "timestamp":   timestamps,
        "symbol":      symbol,
        "s_ldc":       s_ldc,
        "s_mr":        s_mr,
        "s_tsmom":     s_tsmom,
        "regime_state": regime_state,
        "close":       ohlc["close"],
        "volume":      ohlc["volume"],
        "high":        ohlc["high"],
        "low":         ohlc["low"],
    })

    df_market = pd.DataFrame({
        "timestamp": timestamps,
        "symbol":    symbol,
        **ohlc,
    })

    return df_signal, df_market


def main():
    print(f"Generating backtest fixtures in {FIXTURE_DIR}...")

    signal_dir  = FIXTURE_DIR / "signals"
    market_dir  = FIXTURE_DIR / "market_data"
    signal_dir.mkdir(parents=True, exist_ok=True)
    market_dir.mkdir(parents=True, exist_ok=True)

    all_signals = []
    all_markets = []

    for i, sym in enumerate(SYMBOLS):
        df_sig, df_mkt = _generate_symbol_data(sym, i)
        all_signals.append(df_sig)
        all_markets.append(df_mkt)

    signals_df  = pd.concat(all_signals, ignore_index=True)
    market_df   = pd.concat(all_markets, ignore_index=True)

    sig_path  = signal_dir / "signals.parquet"
    mkt_path  = market_dir / "market_ohlcv.parquet"

    signals_df.to_parquet(sig_path, index=False)
    market_df.to_parquet(mkt_path, index=False)

    print(f"  Signals  -> {sig_path}  ({signals_df.shape[0]} rows, {signals_df.shape[1]} cols)")
    print(f"  Market   -> {mkt_path}  ({market_df.shape[0]} rows, {market_df.shape[1]} cols)")
    print(f"  Symbols: {SYMBOLS}")
    print(f"  Periods: {N_PERIODS} per symbol (1h bars, {N_DAYS} days)")
    print("Done.")


if __name__ == "__main__":
    main()
