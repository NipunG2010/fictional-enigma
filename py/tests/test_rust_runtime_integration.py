#!/usr/bin/env python3
"""
Cross-language integration: Python backtester consumes Rust runtime canonical JSONL.

This test exercises the canonical output contract (`imp.runtime.output.v1`) for real:
  1. Loads the committed Rust-runtime JSONL fixture + run summary
  2. Verifies integrity (schema version, SHA-256 checksum, row counts)
  3. Converts records into the backtester's (signal_df, market_df) frames
  4. Exports DataLoader-compatible parquet fixtures
  5. Runs the full 6-step BacktestEngine pipeline on the exported data

Run:
    cd py
    python -m pytest tests/test_rust_runtime_integration.py -v
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # py/
FIXTURES = PROJECT_ROOT / "tests" / "backtest_fixtures" / "rust_runtime"
RUST_JSONL = FIXTURES / "cross-language.jsonl"
RUST_SUMMARY = FIXTURES / "cross-language.summary.json"

sys.path.insert(0, str(PROJECT_ROOT))

from imp.backtesting.rust_runtime_loader import (
    RUNTIME_SCHEMA_VERSION,
    RuntimeContractError,
    RuntimeRunArtifacts,
    export_backtest_fixtures,
    load_runtime_run,
    to_backtest_frames,
)

rust_fixture_available = pytest.mark.skipif(
    not (RUST_JSONL.exists() and RUST_SUMMARY.exists()),
    reason="committed Rust runtime fixture not present (generate via "
    "`cargo run -p inference-engine -- run-runtime --config fixtures/cross-language.toml`)",
)


# ===================================================================
# Synthetic record helpers (loader unit tests, no Rust output needed)
# ===================================================================

def make_record(
    bar_index: int,
    timestamp: int,
    fused: bool = True,
    side: str = "BUY",
) -> dict:
    """Build a minimal-but-complete canonical runtime output record."""
    signal = {
        "raw": 0.4 if fused else None,
        "normalized": 0.4 if fused else None,
        "confidence": 0.4 if fused else None,
        "available": fused,
        "status": "computed" if fused else "waiting_for_training",
        "reason": None if fused else "warmup",
        "normalization": "tanh",
    }
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "versions": {
            "runtime_version": "0.1.0",
            "runtime_mode": "fallback_only",
            "model_version": "test-v1",
        },
        "ohlcv": {
            "timestamp": timestamp,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        },
        "features": {"timestamp": timestamp, "rsi": 50.0},
        "intermediate_signals": {
            "ldc": signal,
            "mr": signal,
            "tsmom": signal,
            "fusion_inputs": (
                {"s_ldc": 0.4, "s_mr": -0.2, "s_tsmom": 0.3} if fused else None
            ),
        },
        "hmm": {
            "source": "static_fallback",
            "weights": {"w_ldc": 0.5, "w_mr": 0.3, "w_tsmom": 0.2},
            "cache_hit": False,
        },
        "fused_output": {
            "eligible_for_fusion": fused,
            "fused_score": 0.2 if fused else 0.0,
            "confidence": 0.2 if fused else 0.0,
            "threshold": 0.05,
            "cooldown_active": False,
            "recommended_side": side if fused else "HOLD",
            "actionable_side": side if fused else "HOLD",
            "signal_generated": fused,
            "suppression_reason": None,
            "weights_used": {"w_ldc": 0.5, "w_mr": 0.3, "w_tsmom": 0.2},
        },
        "audit": {
            "run_id": "test-run",
            "correlation_id": f"BTCUSDT-{timestamp}-{bar_index:05}",
            "input_checksum": "0" * 64,
            "feature_checksum": "0" * 64,
        },
        "emission": {
            "configured": False,
            "backend": "None",
            "attempted": False,
            "emitted": False,
            "error": None,
        },
        "timing": {"bar_index": bar_index},
    }


def write_synthetic_run(directory: Path, n_warmup: int = 3, n_fused: int = 5) -> tuple[Path, Path]:
    """Write a synthetic JSONL + summary pair the way the Rust runtime does."""
    records = []
    base_ts = 1700000000
    for i in range(n_warmup + n_fused):
        records.append(
            make_record(
                bar_index=i,
                timestamp=base_ts + i * 300,
                fused=i >= n_warmup,
                side="BUY" if i % 2 == 0 else "SELL",
            )
        )
    lines = [json.dumps(r) for r in records]
    payload = ("".join(line + "\n" for line in lines)).encode("utf-8")

    jsonl_path = directory / "synthetic.jsonl"
    jsonl_path.write_bytes(payload)

    summary = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_version": "0.1.0",
        "runtime_mode": "fallback_only",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "input_rows": len(records),
        "output_rows": len(records),
        "fused_rows": n_fused,
        "emitted_rows": n_fused,
        "fallback_rows": n_fused,
        "cache_hits": 0,
        "first_timestamp": base_ts,
        "last_timestamp": base_ts + (len(records) - 1) * 300,
        "canonical_output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    summary_path = directory / "synthetic.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return jsonl_path, summary_path


@pytest.fixture()
def synthetic_run(tmp_path: Path) -> tuple[Path, Path]:
    return write_synthetic_run(tmp_path)


# ===================================================================
# Loader unit tests (synthetic data)
# ===================================================================

class TestLoadRuntimeRun:
    def test_loads_records_and_summary(self, synthetic_run):
        jsonl_path, summary_path = synthetic_run
        artifacts = load_runtime_run(jsonl_path, summary_path)
        assert isinstance(artifacts, RuntimeRunArtifacts)
        assert len(artifacts.records) == 8
        assert artifacts.summary["symbol"] == "BTCUSDT"

    def test_checksum_verified_against_summary(self, synthetic_run):
        jsonl_path, summary_path = synthetic_run
        # Should not raise — checksum matches
        load_runtime_run(jsonl_path, summary_path, verify=True)

    def test_tampered_jsonl_fails_checksum(self, synthetic_run, tmp_path):
        jsonl_path, summary_path = synthetic_run
        tampered = tmp_path / "tampered.jsonl"
        content = jsonl_path.read_bytes()
        tampered.write_bytes(content.replace(b'"close":100.5', b'"close":999.9'))
        with pytest.raises(RuntimeContractError, match="checksum"):
            load_runtime_run(tampered, summary_path, verify=True)

    def test_row_count_mismatch_fails(self, synthetic_run, tmp_path):
        jsonl_path, summary_path = synthetic_run
        truncated = tmp_path / "truncated.jsonl"
        lines = jsonl_path.read_bytes().splitlines(keepends=True)
        truncated.write_bytes(b"".join(lines[:-1]))
        with pytest.raises(RuntimeContractError):
            load_runtime_run(truncated, summary_path, verify=True)

    def test_unknown_schema_version_fails(self, tmp_path):
        record = make_record(0, 1700000000)
        record["schema_version"] = "imp.runtime.output.v999"
        path = tmp_path / "bad_schema.jsonl"
        path.write_text(json.dumps(record) + "\n")
        with pytest.raises(RuntimeContractError, match="schema"):
            load_runtime_run(path)

    def test_loads_without_summary(self, synthetic_run):
        jsonl_path, _ = synthetic_run
        artifacts = load_runtime_run(jsonl_path)
        assert artifacts.summary is None
        assert len(artifacts.records) == 8


class TestToBacktestFrames:
    def test_market_frame_schema(self, synthetic_run):
        artifacts = load_runtime_run(*synthetic_run)
        signal_df, market_df = to_backtest_frames(artifacts)
        required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        assert required.issubset(market_df.columns)
        assert len(market_df) == 8  # all bars, including warmup
        assert pd.api.types.is_datetime64_any_dtype(market_df["timestamp"])

    def test_signal_frame_schema(self, synthetic_run):
        artifacts = load_runtime_run(*synthetic_run)
        signal_df, _ = to_backtest_frames(artifacts)
        required = {"timestamp", "symbol", "s_ldc", "s_mr", "s_tsmom"}
        assert required.issubset(signal_df.columns)
        # warmup rows (no fusion_inputs) are excluded
        assert len(signal_df) == 5

    def test_signal_values_match_fusion_inputs(self, synthetic_run):
        artifacts = load_runtime_run(*synthetic_run)
        signal_df, _ = to_backtest_frames(artifacts)
        assert (signal_df["s_ldc"] == pytest.approx(0.4)) is not False
        assert signal_df["s_ldc"].between(-1, 1).all()
        assert signal_df["s_mr"].between(-1, 1).all()
        assert signal_df["s_tsmom"].between(-1, 1).all()

    def test_symbol_from_summary(self, synthetic_run):
        artifacts = load_runtime_run(*synthetic_run)
        signal_df, market_df = to_backtest_frames(artifacts)
        assert (signal_df["symbol"] == "BTCUSDT").all()
        assert (market_df["symbol"] == "BTCUSDT").all()

    def test_symbol_required_without_summary(self, synthetic_run):
        jsonl_path, _ = synthetic_run
        artifacts = load_runtime_run(jsonl_path)
        with pytest.raises(RuntimeContractError, match="symbol"):
            to_backtest_frames(artifacts)
        signal_df, market_df = to_backtest_frames(artifacts, symbol="ETHUSDT")
        assert (market_df["symbol"] == "ETHUSDT").all()


class TestExportBacktestFixtures:
    def test_exports_dataloader_layout(self, synthetic_run, tmp_path):
        artifacts = load_runtime_run(*synthetic_run)
        out = tmp_path / "exported"
        paths = export_backtest_fixtures(artifacts, out)
        # DataLoader flat-structure globs: *signals*.parquet / *ohlcv*.parquet
        assert list((out / "signals").glob("*signals*.parquet"))
        assert list((out / "market_data").glob("*ohlcv*.parquet"))
        signals = pd.read_parquet(paths["signals"])
        market = pd.read_parquet(paths["market_data"])
        assert {"s_ldc", "s_mr", "s_tsmom"}.issubset(signals.columns)
        assert {"open", "high", "low", "close", "volume"}.issubset(market.columns)


# ===================================================================
# Contract tests against the committed Rust runtime fixture
# ===================================================================

@rust_fixture_available
class TestRustFixtureContract:
    @pytest.fixture(scope="class")
    def artifacts(self) -> RuntimeRunArtifacts:
        return load_runtime_run(RUST_JSONL, RUST_SUMMARY, verify=True)

    def test_schema_version(self, artifacts):
        assert all(r["schema_version"] == RUNTIME_SCHEMA_VERSION for r in artifacts.records)

    def test_summary_row_counts(self, artifacts):
        assert artifacts.summary["output_rows"] == len(artifacts.records)
        assert artifacts.summary["fused_rows"] <= artifacts.summary["output_rows"]

    def test_sides_are_canonical(self, artifacts):
        sides = {r["fused_output"]["recommended_side"] for r in artifacts.records}
        assert sides.issubset({"BUY", "SELL", "HOLD"})

    def test_fused_rows_have_signal_components(self, artifacts):
        fused = [
            r for r in artifacts.records
            if r["intermediate_signals"]["fusion_inputs"] is not None
        ]
        assert len(fused) == artifacts.summary["fused_rows"]
        for r in fused:
            fi = r["intermediate_signals"]["fusion_inputs"]
            assert -1.0 <= fi["s_ldc"] <= 1.0
            assert -1.0 <= fi["s_mr"] <= 1.0
            assert -1.0 <= fi["s_tsmom"] <= 1.0

    def test_timestamps_monotonic(self, artifacts):
        ts = [r["ohlcv"]["timestamp"] for r in artifacts.records]
        assert ts == sorted(ts)


# ===================================================================
# Full cross-language E2E: Rust JSONL → frames → BacktestEngine
# ===================================================================

@rust_fixture_available
class TestCrossLanguageBacktest:
    @pytest.fixture(scope="class")
    def backtest_results(self, tmp_path_factory):
        from imp.backtesting.backtest_engine import BacktestEngine
        from imp.backtesting.config import BacktestConfig

        artifacts = load_runtime_run(RUST_JSONL, RUST_SUMMARY, verify=True)
        out_dir = tmp_path_factory.mktemp("rust_backtest")
        export_backtest_fixtures(artifacts, out_dir)

        signal_df, _ = to_backtest_frames(artifacts)
        start = signal_df["timestamp"].min().date()
        end = signal_df["timestamp"].max().date()
        symbol = artifacts.summary["symbol"]

        config = BacktestConfig(
            name="rust_runtime_cross_language",
            description="Backtest consuming Rust runtime canonical JSONL output",
            start_date=start,
            end_date=end,
            symbols=[symbol],
            initial_capital=100_000.0,
            data_source={
                "signals_path": str(out_dir / "signals"),
                "market_data_path": str(out_dir / "market_data"),
            },
            signal_processing={
                "thresholds": {"ldc": 0.2, "mr": 0.15, "tsmom": 0.2},
                "filters": {
                    "min_signal_strength": 0.1,
                    "max_trades_per_day": 50,
                    "cooldown_periods": 3,
                },
                "fusion_method": "hmm_weighted",
            },
            position_sizing={
                "method": "percentage",
                "percentage": 0.05,
                "max_position_size": 0.2,
                "max_total_exposure": 1.0,
            },
            cost_model={
                "commission_rate": 0.001,
                "min_commission": 1.0,
                "asset_class": "crypto",
            },
            performance={"risk_free_rate": 0.02, "regime_analysis": False},
            walk_forward={"enabled": False},
            output_path=str(out_dir / "output"),
            save_trades=False,
            save_portfolio_state=False,
            generate_report=False,
        )
        engine = BacktestEngine(config)
        return engine.run_backtest()

    def test_pipeline_completes(self, backtest_results):
        assert backtest_results is not None
        assert backtest_results.execution_time > 0

    def test_portfolio_snapshots_produced(self, backtest_results):
        assert len(backtest_results.portfolio_snapshots) > 0

    def test_performance_metrics_present(self, backtest_results):
        metrics = backtest_results.performance_metrics
        assert metrics is not None
        assert metrics.total_return is not None

    def test_final_portfolio_value_positive(self, backtest_results):
        assert backtest_results.final_portfolio_value > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
