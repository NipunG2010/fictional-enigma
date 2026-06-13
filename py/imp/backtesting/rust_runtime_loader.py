"""
Loader for the Rust inference-engine's canonical JSONL output.

This is the cross-language bridge between the Rust runtime and the Python
backtesting framework. The Rust runtime (`cargo run -p inference-engine --
run-runtime`) writes one JSON record per OHLCV bar in the canonical
`imp.runtime.output.v1` schema plus a run summary JSON. This module:

  1. Parses and integrity-checks that output (schema version, SHA-256 of the
     raw JSONL bytes against `canonical_output_sha256`, row counts)
  2. Converts records into the (signal_df, market_df) frames the
     BacktestEngine consumes
  3. Exports DataLoader-compatible parquet fixtures (flat layout with
     `*signals*.parquet` / `*ohlcv*.parquet` naming)

The signal columns `s_ldc` / `s_mr` / `s_tsmom` are taken from
`intermediate_signals.fusion_inputs` — the exact values the Rust fusion
engine consumed — so a backtest over these frames evaluates what the runtime
actually emitted, not a Python re-derivation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

RUNTIME_SCHEMA_VERSION = "imp.runtime.output.v1"


class RuntimeContractError(Exception):
    """Raised when Rust runtime output violates the canonical contract."""


@dataclass
class RuntimeRunArtifacts:
    """Parsed canonical runtime output: JSONL records plus optional summary."""

    records: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]] = None
    jsonl_path: Optional[Path] = None


def load_runtime_run(
    jsonl_path: Path,
    summary_path: Optional[Path] = None,
    verify: bool = True,
) -> RuntimeRunArtifacts:
    """
    Load a Rust runtime canonical JSONL run, optionally verifying integrity.

    Args:
        jsonl_path: Path to the canonical JSONL output
        summary_path: Path to the run summary JSON (enables checksum/row checks)
        verify: When True and a summary is provided, verify the SHA-256 of the
            raw JSONL bytes against `canonical_output_sha256` and the record
            count against `output_rows`

    Raises:
        RuntimeContractError: On schema version, checksum, or row count violations
    """
    jsonl_path = Path(jsonl_path)
    raw = jsonl_path.read_bytes()

    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeContractError(
                f"{jsonl_path}:{line_no}: invalid JSON: {exc}"
            ) from exc
        version = record.get("schema_version")
        if version != RUNTIME_SCHEMA_VERSION:
            raise RuntimeContractError(
                f"{jsonl_path}:{line_no}: unexpected schema_version {version!r}, "
                f"expected {RUNTIME_SCHEMA_VERSION!r}"
            )
        records.append(record)

    summary: Optional[Dict[str, Any]] = None
    if summary_path is not None:
        summary = json.loads(Path(summary_path).read_text())
        if verify:
            _verify_against_summary(raw, records, summary, jsonl_path)

    logger.info("Loaded %d canonical runtime records from %s", len(records), jsonl_path)
    return RuntimeRunArtifacts(records=records, summary=summary, jsonl_path=jsonl_path)


def _verify_against_summary(
    raw: bytes,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    jsonl_path: Path,
) -> None:
    """Verify JSONL bytes and record count against the Rust run summary."""
    expected_sha = summary.get("canonical_output_sha256")
    # The Rust runtime hashes each serialized line followed by b"\n", which is
    # exactly the raw bytes of the file it writes.
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha != actual_sha:
        raise RuntimeContractError(
            f"{jsonl_path}: checksum mismatch — summary canonical_output_sha256="
            f"{expected_sha}, actual={actual_sha}"
        )

    expected_rows = summary.get("output_rows")
    if expected_rows != len(records):
        raise RuntimeContractError(
            f"{jsonl_path}: row count mismatch — summary output_rows={expected_rows}, "
            f"actual={len(records)}"
        )


def to_backtest_frames(
    artifacts: RuntimeRunArtifacts,
    symbol: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert canonical runtime records into BacktestEngine input frames.

    Returns:
        (signal_df, market_df) where market_df has one row per record with
        OHLCV columns, and signal_df has one row per fused record with the
        s_ldc/s_mr/s_tsmom components the Rust fusion engine consumed, plus
        the runtime's fused_score / signal_generated / actionable_side for
        downstream analysis. Timestamps are converted from epoch seconds to
        pandas datetimes.

    Raises:
        RuntimeContractError: If no symbol is available (no summary and no
            explicit symbol argument)
    """
    if symbol is None:
        if artifacts.summary is None or "symbol" not in artifacts.summary:
            raise RuntimeContractError(
                "symbol is required: pass symbol= explicitly or load the run summary"
            )
        symbol = artifacts.summary["symbol"]

    market_rows = []
    signal_rows = []
    for record in artifacts.records:
        ohlcv = record["ohlcv"]
        timestamp = pd.to_datetime(ohlcv["timestamp"], unit="s")
        market_rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "open": ohlcv["open"],
                "high": ohlcv["high"],
                "low": ohlcv["low"],
                "close": ohlcv["close"],
                "volume": ohlcv["volume"],
            }
        )

        fusion_inputs = record["intermediate_signals"].get("fusion_inputs")
        if fusion_inputs is None:
            continue
        fused_output = record["fused_output"]
        signal_rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "s_ldc": fusion_inputs["s_ldc"],
                "s_mr": fusion_inputs["s_mr"],
                "s_tsmom": fusion_inputs["s_tsmom"],
                "fused_score": fused_output["fused_score"],
                "signal_generated": fused_output["signal_generated"],
                "actionable_side": fused_output["actionable_side"],
            }
        )

    market_df = pd.DataFrame(market_rows)
    signal_df = pd.DataFrame(signal_rows)
    logger.info(
        "Converted %d records → %d market rows, %d signal rows (symbol=%s)",
        len(artifacts.records), len(market_df), len(signal_df), symbol,
    )
    return signal_df, market_df


def export_backtest_fixtures(
    artifacts: RuntimeRunArtifacts,
    output_dir: Path,
    symbol: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Write parquet files in the flat layout DataLoader expects.

    Produces `<output_dir>/signals/signals.parquet` and
    `<output_dir>/market_data/market_ohlcv.parquet`, matching DataLoader's
    flat-structure globs (`*signals*.parquet` / `*ohlcv*.parquet`).

    Returns:
        Mapping of {"signals": path, "market_data": path}
    """
    signal_df, market_df = to_backtest_frames(artifacts, symbol=symbol)

    output_dir = Path(output_dir)
    signals_dir = output_dir / "signals"
    market_dir = output_dir / "market_data"
    signals_dir.mkdir(parents=True, exist_ok=True)
    market_dir.mkdir(parents=True, exist_ok=True)

    signals_path = signals_dir / "signals.parquet"
    market_path = market_dir / "market_ohlcv.parquet"
    signal_df.to_parquet(signals_path, index=False)
    market_df.to_parquet(market_path, index=False)

    logger.info("Exported backtest fixtures to %s", output_dir)
    return {"signals": signals_path, "market_data": market_path}
