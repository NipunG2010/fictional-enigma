# Cross-language integration: Python backtester consumes Rust runtime output (P5)

**Date:** 2026-06-13
**Goal:** First end-to-end run where the Python backtesting framework consumes the Rust inference-engine's canonical JSONL output, validating the output contract for real.

## Problem

The Rust runtime emits canonical JSONL (`imp.runtime.output.v1`) and the Python backtester consumes parquet signal/market files — but nothing connects them. There is no proof the canonical output contract is consumable, and no way to backtest what the runtime actually emits.

## Design

### New Rust fixture config
`rust/inference-engine/fixtures/cross-language.toml` — `fallback_only` mode, deterministic, reading the **git-tracked** input `rust/partitioned_data/symbol=BTCUSDT/date=2025-09-19/interval=5m/ohlcv.parquet` (1000 rows, epoch-seconds timestamps starting 1700000000), `max_rows = 256`. Output written to `py/tests/backtest_fixtures/rust_runtime/cross-language.jsonl` + `.summary.json`, which are **committed** so Python tests run without a Rust toolchain.

### New Python module
`py/imp/backtesting/rust_runtime_loader.py`:

- `load_runtime_run(jsonl_path, summary_path=None, verify=True) -> RuntimeRunArtifacts`
  Parses JSONL, asserts `schema_version == "imp.runtime.output.v1"` per record. When the summary is given, verifies `canonical_output_sha256` (SHA-256 of the raw JSONL bytes, lines joined with `\n`) and `output_rows`.
- `to_backtest_frames(artifacts, symbol=None) -> (signal_df, market_df)`
  `market_df`: one row per record — `timestamp` (datetime from epoch seconds), `symbol`, `open/high/low/close/volume`.
  `signal_df`: rows where `intermediate_signals.fusion_inputs` is present — `timestamp`, `symbol`, `s_ldc`, `s_mr`, `s_tsmom`, plus `fused_score`, `signal_generated`, `actionable_side` for analysis.
- `export_backtest_fixtures(artifacts, output_dir)` — writes `signals/signals.parquet` + `market_data/market_ohlcv.parquet` in the layout `DataLoader` flat-structure loading expects (`*signals*.parquet` / `*ohlcv*.parquet` globs).

### Integration test
`py/tests/test_rust_runtime_integration.py`:
1. Contract tests — schema version, checksum integrity vs summary, row counts, side enum values (`BUY`/`SELL`/`HOLD`).
2. Frame conversion tests — required columns, dtypes, signal range `[-1, 1]`.
3. Full `BacktestEngine.run_backtest()` on exported fixtures (date range 2023-11-13 → 2023-11-17 matching the fixture timestamps), asserting the 6-step pipeline completes and produces metrics.
4. `@pytest.mark.integration` regeneration test that invokes `cargo run -p inference-engine -- run-runtime` and asserts fresh output matches the committed fixture (skipped when cargo is unavailable).

### Out of scope
Real market data stress-testing (separate P5 item), HMM-service mode, Redis/Kafka emission paths.

## Verification
- `pytest tests/test_rust_runtime_integration.py -v` passes from `py/`.
- The first run: `cargo run -p inference-engine -- run-runtime --config fixtures/cross-language.toml` produces the fixture consumed by the passing test.
- `docs/status.md` P5 row updated: cross-language integration first run complete.
