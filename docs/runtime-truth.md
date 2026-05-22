# Runtime Truth

This document answers one question: **what can someone reasonably run from this repository today, and under what caveats?**

It is intentionally narrower than the architecture vision.

## Runnable today

### 1. Rust offline batch runtime (+ daemon mode)
This is the primary supported Rust execution path, now with **both batch and daemon modes**.

#### Batch mode
```bash
cd rust
cargo run -p inference-engine -- run-runtime \
  --config inference-engine/fixtures/local-smoke.toml
```

#### Daemon (long-running) mode
```bash
cd rust
cargo run -p inference-engine -- serve --port 9090 --interval 60 \
  --config inference-engine/fixtures/local-smoke.toml
```

Why this is considered runnable:
- `rust/inference-engine` loads real config, bootstraps components, runs the orchestration loop, and shuts down cleanly.
- The **daemon mode** (`serve` subcommand) adds: periodic pipeline execution at configurable intervals, a health check HTTP endpoint, and graceful shutdown on SIGINT/SIGTERM.
- The runtime loads market input, computes features, generates MR/TSMOM/LDC signals, resolves HMM-or-fallback weights, fuses signals, optionally emits them, and writes canonical JSONL output.
- A deterministic smoke fixture and expected output are provided for local verification.
- A non-mock integration test validates the full pipeline path end-to-end with real components.


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

### 5. Python HMM microservice — canonical startup and artifact-loading paths
The HMM service has a documented canonical startup path and artifact-loading workflow, and is the integration target for `integration_hmm` batch mode.

**Canonical service startup path:**

```bash
cd py/hmm_service
pip install -r requirements.txt
cp .env.example .env  # or set env vars directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Service startup follows this sequence:
1. Logging configuration
2. Performance manager initialization (connection pooling, concurrency limits)
3. Cache manager initialization (inference result caching)
4. Model loader initialization (MinIO connection, fallback setup)
5. Default model load (from MinIO or fallback)
6. Inference engine initialization with loaded model
7. Ready to serve inference endpoints

**Canonical artifact-loading path:**
1. `MinIOArtifactStore` connects to MinIO (configurable via `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`)
2. `ModelLoader.load_model_from_minio()` downloads artifact by `experiment_id`/`version`
3. `HMMArtifact(**artifact_data["hmm_artifact"])` reconstructs the Pydantic model
4. `FusionWeights(**artifact_data["fusion_weights"])` reconstructs optional per-state weights
5. `ArtifactValidator.run_all_validations()` validates structural integrity
6. `HMMInferenceEngine.load_model()` preprocesses parameters (precomputes inverse covariances, log-determinants) for fast inference
7. Service is ready for `/inference/*` requests

Why this is considered runnable:
- `py/hmm_service/app.py` creates the FastAPI app with documented component initialization order.
- Artifact interfaces (`HMMArtifact`/`FusionWeights`) are compatible with the Rust `signal-fusion` crate.
- All 30 HMM core tests pass; basic service test passes.
- The model loader supports hot-reloading and periodic update checks.

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

## Not runnable as a truthful claim today

### 1. Full always-on Rust inference service

### 2. True end-to-end integrated pipeline without mocks

### 3. Production readiness

## Changes since the previous truth

### P4 — Runtime orchestration (newly runnable)

The inference engine now supports **daemon mode** (`serve` subcommand):
- Periodic pipeline execution at configurable intervals (default: 60s)
- Health check HTTP endpoint on configurable port (default: 9090)
- Graceful shutdown via SIGINT/SIGTERM
- Non-mock integration test in `tests/` validates the full pipeline path

This is the first step toward production-style always-on operation. What remains:
- Prometheus metrics integration
- Structured logging for production consumption
- Container orchestration (Kubernetes deployment)
- Config hot-reload without service restart

## Runnable with caveats

### Signal-fusion crate workflows
`rust/signal-fusion` contains substantial code and tests for HMM client behavior, fusion, emission, and metrics.

Caveat: this does **not** mean the repo as a whole provides a fully wired production runtime. It means the crate itself contains meaningful implementation and crate-level test/demo paths.

### Backtesting workflows
`py/imp/backtesting` is substantial enough that local experiments and targeted runs are reasonable. A canonical end-to-end backtest run now exists and passes with real deterministic fixture data.

```bash
cd py
python -m pytest tests/test_canonical_backtest.py -v
```

Caveats:
- data/config preparation is still on the user,
- validation is substantially improved (96 tests across 5 files), but full stress-testing with real market data remains as future work.

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
