# Runtime Truth

This document answers one question: **what can someone reasonably run from this repository today, and under what caveats?**

It is intentionally narrower than the architecture vision.

## Runnable today

### 1. Rust feature-generation CLI
This is the cleanest supported Rust execution path currently exposed by the repo.

```bash
cd rust
cargo run -p inference-engine -- compute-features \
  --input sample/ohlcv.parquet \
  --output sample/features_cli.parquet
```

Why this is considered runnable:
- `rust/inference-engine/src/main.rs` implements the `ComputeFeatures` subcommand.
- That path delegates to the implemented `feature-pipeline` library.

### 2. Python research environment
The Python research stack is substantial enough to document as a runnable local workflow.

Documented baseline:

```bash
cd py
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,optimization,research]"
```

Why this is considered runnable:
- `py/pyproject.toml` defines the package and extras.
- notebooks and research modules exist under `py/imp/` and `../notebooks/`.

### 3. Python HMM microservice in isolation
The HMM service can be treated as a runnable service prototype.

```bash
cd py/hmm_service
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Why this is considered runnable:
- `py/hmm_service/app.py` creates the FastAPI app.
- `py/hmm_service/routers/inference.py` exposes real inference endpoints.

### 4. Local MinIO + Redis helper services
The root compose file is useful for local dependencies, but only for a subset of the intended system.

```bash
docker compose up -d minio redis
```

What this actually provides:
- MinIO
- Redis

What it does **not** provide:
- Kafka
- the HMM service
- the Rust inference runtime
- a full repo-wide pipeline

## Runnable with caveats

### Signal-fusion crate workflows
`rust/signal-fusion` contains substantial code and tests for HMM client behavior, fusion, emission, and metrics.

Caveat: this does **not** mean the repo as a whole provides a fully wired production runtime. It means the crate itself contains meaningful implementation and crate-level test/demo paths.

### Backtesting workflows
`py/imp/backtesting` is substantial enough that local experiments and targeted runs are reasonable.

Caveats:
- data/config preparation is still on the user,
- validation is incomplete,
- the repo does not currently provide a single canonical blessed backtest run proving readiness.

## Not runnable as a truthful claim today

### 1. Full Rust inference runtime service
Do **not** claim that the repository currently ships a complete production-like inference service.

Reason:
- `rust/inference-engine/src/main.rs` still has TODO placeholders for config loading, component initialization, server startup, and the signal loop.

### 2. True end-to-end integrated pipeline without mocks
Do **not** claim that the repository currently proves a real end-to-end path from OHLCV input to emitted final signal across the real component stack.

Reasons:
- `rust/end-to-end-tests/Cargo.toml` has real dependencies commented out.
- `rust/end-to-end-tests/src/harness.rs` uses mock signal-fusion and LDC components.

### 3. Production readiness
Do **not** describe the repository as production-ready.

Reasons include:
- incomplete runtime orchestration,
- mock-heavy repo-wide E2E evidence,
- tracked generated artifacts that blur fixture vs output vs temporary residue,
- inconsistent historical status language still present outside the new source-of-truth docs.

## Canonical supported path for now

If you need one honest sentence for the repo today, use this:

> IMP currently supports a **research/prototype workflow** centered on Rust feature generation plus Python HMM research/service components, with substantial library code for LDC and signal fusion, but it does **not** yet provide a fully proven production runtime or non-mock full-system end-to-end pipeline.
