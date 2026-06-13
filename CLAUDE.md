# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

IMP (Intelligent Market Prediction) is a **hybrid Rust + Python** system for feature generation, Lorentzian Distance Classification (LDC), HMM-based regime modeling, signal fusion, and backtesting. It is **not production-ready**; treat it as a partially integrated research/prototype platform. The authoritative status is in [`docs/status.md`](docs/status.md). Before describing any component as "complete" or "production-ready", check that file.

## Commands

### Rust (run from `rust/`)

```bash
# Build all crates
cargo build --all

# Run all tests
cargo test --all

# Format check / lint
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings

# Offline batch runtime
cargo run -p inference-engine -- run-runtime --config inference-engine/fixtures/local-smoke.toml

# Deterministic smoke validation (compares output to fixture byte-for-byte)
cargo run -p inference-engine -- smoke --config inference-engine/fixtures/local-smoke.toml

# Daemon mode (runs pipeline every N seconds, exposes GET /health on port)
cargo run -p inference-engine -- serve --port 9090 --interval 60 --config inference-engine/fixtures/local-smoke.toml

# Feature-only CLI
cargo run -p inference-engine -- compute-features --input sample/ohlcv.parquet --output sample/features_cli.parquet

# Non-mock integration test (full pipeline: feature pipeline → LDC → HMM/fallback → fusion → output)
cargo test -p inference-engine -- --nocapture
```

### Python (run from `py/`)

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,optimization,research]"

# All tests
pytest tests/ -v

# Single test file
pytest tests/test_canonical_backtest.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# By marker: research | integration | notebook | slow
pytest tests/ -v -m research

# With coverage
pytest tests/ --cov=imp --cov-report=html

# Format and lint
make format        # black + isort
make lint          # flake8
make type-check    # mypy

# Canonical backtesting E2E run
python -m pytest tests/test_canonical_backtest.py -v
```

### HMM microservice (run from `py/hmm_service/`)

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Endpoints: /docs  /health  /inference/predict  /inference/fusion-weights
```

### Local infra (MinIO + Redis + Kafka)

```bash
docker compose up -d minio redis kafka
# MinIO: 9000 (API), 9001 (console)  Redis: 6379  Kafka: 9092
```

## Architecture

### Data flow (intended)

```
OHLCV → feature-pipeline → LDC / MR / TSMOM signals → HMM regime inference → signal fusion → emission → downstream
```

### Rust workspace (`rust/`)

| Crate | Role |
|---|---|
| `inference-engine` | CLI entrypoint + offline batch orchestrator + daemon mode. Source of truth for the runnable path. |
| `feature-pipeline` | Technical indicator computation, parquet I/O, feature validation |
| `ldc-engine` | Lorentzian Distance Classification library and testing utilities |
| `signal-fusion` | HMM client (HTTP), signal fusion, Redis/Kafka emission, metrics |
| `training-data-cli` | Training data utilities |
| `end-to-end-tests` | Mock-heavy integration scaffold — **not** proof of real E2E |

The workspace `Cargo.toml` pins shared dependencies (polars 0.51, tokio, serde, reqwest, ndarray, rayon, prometheus).

### Python packages (`py/`)

| Path | Role |
|---|---|
| `py/imp/hmm/` | HMM training, artifact management, weight optimization, regime analysis |
| `py/imp/backtesting/` | 13-module backtesting framework (data load → simulation → metrics → walk-forward) |
| `py/hmm_service/` | FastAPI service wrapping HMM inference; integration target for `integration_hmm` runtime mode |
| `py/imp/tuning/`, `visualization/` | Hyperparameter tuning and research visualizations |
| `notebooks/` | Research notebooks (top-level, not under `py/`) |

### Inference engine runtime modes

Configured via `[runtime].mode` in TOML configs under `rust/inference-engine/fixtures/`:

- `offline_batch` — primary supported mode; reads OHLCV file, writes canonical JSONL + summary JSON
- `local_smoke` — uses bundled fixture + expected output for byte-for-byte comparison
- `integration_hmm` — batch mode with live HMM service HTTP call
- `fallback_only` — no HMM service call, static weights only

### HMM service ↔ Rust runtime contract

The Rust runtime calls `POST /inference/predict` with `{observations, timestamp, request_id}`. The response fields it uses are `state_probabilities`, `most_likely_state`, `confidence`, `fusion_weights`, `model_version`. Artifact interfaces `HMMArtifact` / `FusionWeights` are defined in Python and structurally mirrored in `rust/signal-fusion`.

### Canonical output format

Each processed OHLCV row produces one JSONL record with: `ohlcv`, `features`, `intermediate_signals` (ldc/mr/tsmom/fusion_inputs), `hmm` (weights + source metadata), `fused_output` (score + side + suppression), `audit` (checksums), `emission` status, `timing`, `versions`. A summary JSON with row counts and SHA-256 of the JSONL is also written.

## Key files to know

- `rust/inference-engine/src/main.rs` — CLI dispatch, integration tests (inline `#[cfg(test)]`)
- `rust/inference-engine/src/runtime.rs` — batch orchestration logic
- `rust/inference-engine/src/daemon.rs` — daemon/serve mode
- `rust/inference-engine/fixtures/local-smoke.toml` — canonical local smoke config
- `py/hmm_service/app.py` — FastAPI app with documented startup sequence
- `py/imp/hmm/` — HMM trainer, artifact management, weight optimizer
- `py/tests/test_canonical_backtest.py` — 22-test canonical E2E backtest (real fixture data)
- `py/tests/backtest_fixtures/` — deterministic parquet fixtures (3 symbols, 720 hourly bars each)
- `docs/status.md` — canonical status matrix, runtime truth, stubs inventory; update this first before marking anything complete
