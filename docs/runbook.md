# Runbook

All supported local execution paths. For what these paths actually prove (vs. what they don't), see [`status.md`](status.md).

## Prerequisites

- Rust stable toolchain
- Python 3.9+
- Docker / Docker Compose (for MinIO, Redis, Kafka)

---

## Rust runtime (run from `rust/`)

### Offline batch runtime
Full pipeline: OHLCV → features → MR/TSMOM/LDC → HMM/fallback → fusion → optional emission → JSONL output.

```bash
cd rust
cargo run -p inference-engine -- run-runtime \
  --config inference-engine/fixtures/local-smoke.toml
```

### Deterministic smoke validation
Runs the batch path and compares output byte-for-byte against bundled expected fixtures.

```bash
cd rust
cargo run -p inference-engine -- smoke \
  --config inference-engine/fixtures/local-smoke.toml
```

Smoke fixtures: `inference-engine/fixtures/local-smoke.{toml,expected.jsonl,expected.summary.json}`. Input data: `rust/sample/ohlcv.parquet`.

### Daemon mode
Runs the pipeline on a configurable interval and exposes a health check endpoint.

```bash
cd rust
cargo run -p inference-engine -- serve --port 9090 --interval 60 \
  --config inference-engine/fixtures/local-smoke.toml
```

Health check: `GET http://localhost:9090/health` → `{"status":"ok"}`. Graceful shutdown on Ctrl+C / SIGTERM (completes the current pipeline run, flushes emitter, saves HMM cache).

### Feature-only CLI

```bash
cd rust
cargo run -p inference-engine -- compute-features \
  --input sample/ohlcv.parquet \
  --output sample/features_cli.parquet
```

### Non-mock integration test
Validates the full pipeline (feature pipeline → LDC → HMM/fallback → fusion → output) against sample parquet data using real components. Integration tests are inline `#[cfg(test)]` modules in `main.rs`.

```bash
cd rust
cargo test -p inference-engine -- --nocapture
```

### All Rust tests + lint

```bash
cd rust
cargo test --all
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

---

## Python research environment (run from `py/`)

### Setup

```bash
cd py
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,optimization,research]"
```

### Tests

```bash
pytest tests/ -v                           # all tests
pytest tests/ -v -m "not slow"            # skip slow tests
pytest tests/test_canonical_backtest.py -v # canonical E2E backtest (22 tests)
pytest tests/ -v -m research              # research marker only
pytest tests/ --cov=imp --cov-report=html # with coverage
```

### Lint and format

```bash
make format       # black + isort (auto-fix)
make lint         # flake8
make type-check   # mypy
make pre-commit   # all pre-commit hooks
```

### Jupyter

```bash
jupyter lab   # notebooks live at repo root in notebooks/
```

---

## Python HMM microservice (run from `py/hmm_service/`)

```bash
cd py/hmm_service
pip install -r requirements.txt
cp .env.example .env          # configure MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Key endpoints: `/health`, `/docs`, `/inference/predict`, `/inference/fusion-weights`, `/inference/state-probabilities`.

This service is the integration target for the Rust runtime's `integration_hmm` mode.

---

## Local infrastructure

```bash
docker compose up -d minio redis kafka
```

| Service | Ports |
|---|---|
| MinIO | 9000 (API), 9001 (console) |
| Redis | 6379 |
| Kafka | 9092 |

This does **not** start the Python HMM service or the Rust runtime — both have separate startup paths above.

---

## Runtime config

All runtime configs are TOML files under `rust/inference-engine/fixtures/`:

| File | Purpose |
|---|---|
| `local-smoke.toml` | Canonical deterministic smoke run |
| `integration-test.toml` | Non-mock integration test (fallback_only, 16 rows max) |
| `integration-hmm.example.toml` | Template for live HMM service integration |
| `fallback-only.example.toml` | Template for static-weights-only runs |

Key config sections: `[runtime].mode`, `[hmm].on_failure`, `[failure].invalid_features`, `[failure].emission_failure`, `[emission].backend`.

---

## Failure modes

| Scenario | Behavior |
|---|---|
| HMM service unavailable | Config-driven: `use_cache_then_fallback` / `use_fallback` / `fail`. Source recorded in output `hmm.source`. |
| Bad config | Run fails at startup before any output. |
| Missing input file or columns | Run fails before feature computation. |
| Invalid features | Config-driven: `fail` or `skip_row`. Warmup-period NaNs handled explicitly. |
| Emission failure | Config-driven: `continue` (records error in output) or `fail_run`. |

---

## Healthy batch run checklist

- Canonical JSONL output file written
- Summary JSON with row counts and SHA-256 written
- Optional feature parquet output (if configured)
- Optional weight-cache JSON (if configured)
- No startup TODO/stub errors in `rust/inference-engine`
