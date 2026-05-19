# Runtime Truth

This document answers one question: **what can someone reasonably run from this repository today, and under what caveats?**

It is intentionally narrower than the architecture vision.

## Runnable today

### 1. Rust offline batch runtime
This is now the primary supported Rust execution path.

```bash
cd rust
cargo run -p inference-engine -- run-runtime \
  --config inference-engine/fixtures/local-smoke.toml
```

Why this is considered runnable:
- `rust/inference-engine` now loads real config, bootstraps components, runs a batch orchestration loop, and shuts down cleanly.
- The runtime loads market input, computes features, generates MR/TSMOM/LDC signals, resolves HMM-or-fallback weights, fuses signals, optionally emits them, and writes canonical JSONL output.
- A deterministic smoke fixture and expected output are provided for local verification.

### 2. Rust deterministic smoke validation

```bash
cd rust
cargo run -p inference-engine -- smoke \
  --config inference-engine/fixtures/local-smoke.toml
```

Why this is considered runnable:
- it uses a known-good fixture input and config,
- it compares generated output to a tracked expected artifact,
- and it exercises the real runtime path rather than a mock harness.

### 3. Rust feature-generation CLI
The feature CLI remains supported as a narrower utility path.

```bash
cd rust
cargo run -p inference-engine -- compute-features \
  --input sample/ohlcv.parquet \
  --output sample/features_cli.parquet
```

Why this is considered runnable:
- `rust/inference-engine/src/main.rs` still implements the `ComputeFeatures` subcommand.
- That path delegates to the implemented `feature-pipeline` library.

### 4. Python research environment
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

### 5. Python HMM microservice in isolation
The HMM service can be treated as a runnable service prototype and is the integration target for `integration_hmm` batch mode.

```bash
cd py/hmm_service
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Why this is considered runnable:
- `py/hmm_service/app.py` creates the FastAPI app.
- `py/hmm_service/routers/inference.py` exposes real inference endpoints.

### 6. Local MinIO + Redis + Kafka helper services
The root compose file is useful for local dependencies, but still does not stand up the full system on its own.

```bash
docker compose up -d minio redis kafka
```

What this actually provides:
- MinIO
- Redis
- Kafka

What it does **not** provide:
- the Python HMM service
- the Rust runtime process itself
- a full repo-wide production deployment

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

### 1. Full always-on Rust inference service
Do **not** claim that the repository currently ships a production-like always-on service.

Reason:
- the implemented runtime is a real **batch orchestrator**, not a daemonized live market-data service with production operational semantics.

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

> IMP currently supports a **real Rust offline batch runtime** plus Python HMM research/service components, with deterministic local smoke validation and optional HMM/Redis/Kafka integrations, but it still does **not** provide a production-ready always-on deployment or repo-wide proof that every historical integration surface is fully validated end-to-end.
