# Architecture

Describes the system as it exists today. For what is runnable vs. not, see [`status.md`](status.md).

---

## One-paragraph summary

IMP is a hybrid Rust/Python repository with real implementations for feature generation, LDC, HMM research, HMM service endpoints, signal fusion, and backtesting. The repository ships a real Rust offline batch runtime and a daemon mode in `rust/inference-engine`, but is still a **partially integrated system** — the always-on service story and repo-wide non-mock E2E proof are incomplete.

---

## Intended data flow

```
OHLCV data
  → feature-pipeline       (technical indicators)
  → LDC / MR / TSMOM       (three independent trading signals)
  → HMM regime inference   (market regime → per-signal weights)
  → signal fusion           (weighted combination → fused score + side)
  → emission                (optional: Redis / Kafka)
  → downstream consumers
```

---

## Rust workspace (`rust/`)

Crates are defined in `rust/Cargo.toml` (workspace resolver v2).

### `inference-engine`
The single CLI entrypoint and runtime orchestrator. All supported execution paths go through here.

**Subcommands:**
- `run-runtime` — offline batch: reads OHLCV, runs full pipeline, writes canonical JSONL + summary JSON
- `smoke` — deterministic fixture mode: runs batch and compares output byte-for-byte to expected fixtures
- `serve` — daemon mode: runs pipeline on a configurable interval, exposes `GET /health`
- `compute-features` — feature-only CLI: reads OHLCV parquet, writes features parquet
- `describe-modes` — prints runtime mode descriptions

**Source files:**
- `src/main.rs` — CLI dispatch + inline `#[cfg(test)]` integration tests
- `src/runtime.rs` — batch orchestration logic
- `src/daemon.rs` — daemon/serve mode: loop + health check HTTP server + graceful shutdown
- `src/config.rs` — TOML config loading and validation
- `src/hmm.rs` — HMM client calls and fallback resolution
- `src/schema.rs` — canonical output schema (JSONL record structure)

**Config fixtures** (`fixtures/`):
- `local-smoke.toml` — canonical local smoke config (paired with `.expected.jsonl` + `.expected.summary.json`)
- `integration-test.toml` — non-mock integration test config (fallback_only, 16 rows max)
- `integration-hmm.example.toml` — template for live HMM service integration
- `fallback-only.example.toml` — template for static-weights-only runs

### `feature-pipeline`
Technical indicator computation from OHLCV data: RSI, EMA, Bollinger Bands, ATR, momentum, mean-reversion prep. Partitioned parquet read/write helpers. Feature validation. This is the strongest fully runnable Rust path.

### `ldc-engine`
Lorentzian Distance Classification library. Implements the LDC k-NN variant using Lorentzian (log-metric) distance. Substantial tests under `tests/`, performance utilities, and test helpers. Wired into the batch runtime as a signal source.

### `signal-fusion`
Combines LDC/MR/TSMOM signals using HMM-derived weights into a single fused score. Also contains:
- `hmm_client.rs` — HTTP client for the Python HMM service (`POST /inference/predict`)
- `signal_emitter.rs` — Redis and Kafka emission backends
- Prometheus metrics, circuit-breaker logic, validation

### `end-to-end-tests`
Mock-based integration harness. Real dependencies (`polars`, `reqwest`, `ldc-engine`, `signal-fusion`) are commented out in `Cargo.toml`. Do not treat passing tests here as proof of real E2E integration.

### `training-data-cli`
Utilities for preparing HMM training data.

---

## Python packages (`py/`)

Installed as a single package (`imp-python`) via `py/pyproject.toml`. Extras: `dev`, `optimization`, `research`.

### `py/imp/hmm/`
HMM research and training core.
- `trainer.py` — HMM training workflows (note: `PomegranateTrainer` is a prototype due to current API changes)
- `inference.py` — regime inference logic
- `weight_optimizer.py` — per-regime fusion weight optimization
- `artifact_management.py` — `HMMArtifact` and `FusionWeights` Pydantic models + MinIO artifact store
- `regime_analysis.py` — regime characterization and analysis utilities

### `py/hmm_service/`
FastAPI microservice. The Rust runtime's `integration_hmm` mode calls this service.

**Startup sequence:**
1. Logging configuration
2. Performance manager (connection pooling, concurrency limits)
3. Cache manager (inference result caching)
4. Model loader (MinIO connection, fallback setup)
5. Default model load from MinIO (or fallback)
6. Inference engine initialization (precomputes inverse covariances, log-determinants)
7. Ready to serve

**Key endpoints:** `POST /inference/predict`, `GET /inference/fusion-weights`, `GET /inference/state-probabilities`, `GET /health`, `GET /docs`

**HMM ↔ Rust contract:**
- Request: `{observations: [s_ldc, s_mr, s_tsmom], timestamp, request_id}`
- Response fields used by Rust: `state_probabilities`, `most_likely_state`, `confidence`, `fusion_weights`, `model_version`, `processing_time_ms`
- Artifact interfaces (`HMMArtifact`, `FusionWeights`) are defined in Python and structurally mirrored in `rust/signal-fusion`

### `py/imp/backtesting/`
13-module backtesting framework covering: data loading, signal processing, trade generation, portfolio state, cost/slippage models, performance analysis, walk-forward validation, MinIO artifact management, tagging/deployment workflows. 96 tests across 5 files.

### `py/imp/tuning/`, `py/imp/visualization/`
Hyperparameter optimization (scikit-optimize) and research visualization utilities.

### `notebooks/`
Research notebooks at the repo root level (not under `py/`). Generated outputs are gitignored.

---

## Canonical output format (JSONL per row)

Each processed OHLCV row produces one record:

```
ohlcv                          raw bar
features                       computed technical indicators
intermediate_signals
  ldc / mr / tsmom             raw signal values
  fusion_inputs                pre-fusion representation
hmm
  source                       "service" | "cache" | "fallback"
  weights / probabilities      per-state weights and state probs
  request metadata             request_id, model_version, timing
fused_output
  fused_score                  combined signal value
  recommended_side             BUY | SELL | HOLD
  actionable_side              after suppression logic
  suppression_reason           why signal was suppressed (if any)
  weights_used                 weights applied during fusion
audit
  run_id / correlation_id      trace identifiers
  input_checksum               SHA-256 of OHLCV input
  feature_checksum             SHA-256 of computed features
emission
  configured / attempted / succeeded / error
timing
versions
```

A batch run also writes a summary JSON with row counts and SHA-256 of the JSONL output.

---

## Runtime modes (configured via `[runtime].mode`)

| Mode | Description |
|---|---|
| `offline_batch` | Primary mode. Reads OHLCV file, full pipeline, canonical JSONL + summary output. |
| `local_smoke` | Deterministic fixture mode. Static fallback weights, no emission, byte-for-byte comparison to expected output. |
| `integration_hmm` | Batch mode with live HMM service HTTP call. Falls back to cached/static weights when service is unavailable. |
| `fallback_only` | No HMM service call. Uses configured static weights only. Useful for local dev and failure isolation. |

## HMM fallback behavior (config-driven)

When the HMM service is unavailable, behavior depends on `[hmm].on_failure`:
- `use_cache_then_fallback` — use cached weights if available, otherwise static fallback
- `use_fallback` — always use static fallback weights
- `fail` — abort the run

The chosen path is recorded in the output's `hmm.source` field.

## Emission backends

Configured via `[emission].backend`: `none` | `redis` | `kafka` | `both`. Emission failure behavior depends on `[failure].emission_failure`: `continue` | `fail_run`.
