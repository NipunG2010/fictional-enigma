# Status

Single source of truth for what is implemented, what is runnable, what is a stub, and what the status vocabulary means.

**Rule:** before marking any component, phase, or task as complete, update this file first. If another README, plan, or spec disagrees with this file, this file wins.

---

## Status vocabulary

| Label | Meaning |
|---|---|
| **implemented** | Real code exists, more than a placeholder, basic test evidence present. Does NOT imply production deployment or repo-wide integration. |
| **partially integrated** | Wired to some adjacent components, but real E2E path has gaps, mocks, disabled dependencies, or missing orchestration. |
| **prototype** | Real and useful code that is intentionally exploratory or provisional — not the final production approach. |
| **test scaffold** | Exists to support future validation; mocks/stubs stand in for the real production path. |
| **not implemented** | Claimed runtime or behavior does not actually exist yet, even if partial setup or TODOs exist. |
| **not production-ready** | Code exists but does not meet the full bar: real runtime path + non-mock integration + failure handling + observability + docs. |

Evidence ladder (weakest → strongest): idea → placeholder/TODO → code exists → code + unit tests → adjacent integration → real E2E non-mock path → operationally supported.

---

## Implementation status matrix

| Component | Code exists? | Adjacent integration? | Runnable today? | Test evidence | Status |
|---|---|---|---|---|---|
| Feature pipeline | Yes | Yes — library + CLI + runtime | Yes | Unit tests in `rust/feature-pipeline` | **implemented** |
| LDC engine | Yes | Partial | Partial | Extensive tests under `rust/ldc-engine/tests/` | **implemented** |
| HMM research package | Yes | Partial | Yes | 30 unit tests passing | **partially integrated** |
| HMM service | Yes | Partial | Yes, in isolation | Service tests pass | **partially integrated** |
| Signal fusion / emission | Yes | Partial | Partial | Tests under `rust/signal-fusion/tests/` | **partially integrated** |
| Inference-engine runtime | Yes | Yes — feature pipeline, LDC, HMM/fallback, fusion, optional emission, **batch + daemon modes** | Yes | Deterministic smoke fixture + non-mock integration test + daemon compile-verified | **partially integrated** |
| End-to-end test suite | Yes | No — core components are mocks | Yes, as scaffold | Scaffold behavior only | **test scaffold** |
| Backtesting framework | Yes | Partial | Yes | 96 tests across 5 files (32 unit + 21 MinIO + 16 tagging + 22 canonical E2E + 5 walk-forward) | **partially integrated** |
| Production hardening | Partial | No | No | No repo-wide proof | **not production-ready** |

---

## What is runnable today

### 1. Rust offline batch runtime + daemon mode
Primary supported Rust path. Runs the full pipeline: OHLCV → features → MR/TSMOM/LDC → HMM/fallback weights → fusion → optional emission → canonical JSONL output.

Also supports **daemon mode** (`serve` subcommand): periodic pipeline execution, health check HTTP endpoint (`GET /health → {"status":"ok"}`), graceful shutdown on SIGINT/SIGTERM.

### 2. Rust deterministic smoke validation
Compares generated output byte-for-byte against bundled expected fixtures. Uses real runtime path, not a mock.

### 3. Rust feature-generation CLI
`compute-features` subcommand — reads OHLCV parquet, writes features parquet.

### 4. Python research environment
`py/imp/` research modules + notebooks. Install via `pip install -e ".[dev,optimization,research]"`.

### 5. Python HMM microservice
FastAPI service at `py/hmm_service/`. Canonical startup: logging → performance manager → cache manager → model loader (MinIO) → inference engine → ready. All 30 HMM core tests pass.

### 6. Canonical backtesting E2E run
`pytest tests/test_canonical_backtest.py` — 22 tests, real deterministic fixture data (3 symbols, 720 hourly bars each), full 6-step pipeline.

### 7. Local infra
`docker compose up -d minio redis kafka` — does NOT start the Python HMM service or Rust runtime.

---

## What is NOT runnable as a truthful claim

- **Full always-on live trading service** — the runtime is a batch orchestrator, not a daemonized live-market-data service with production semantics.
- **True non-mock E2E** — `rust/end-to-end-tests` has real deps commented out and uses mock `SignalPipeline` + mock `LDCEngine`.
- **Production-ready** — no production hardening, deployment automation, observability, or operational SLOs.

---

## Known stubs, mocks, and example artifacts

| Path | Classification | Notes |
|---|---|---|
| `rust/end-to-end-tests/src/harness.rs` | **test-only** | Mock `SignalPipeline`, mock `LDCEngine` — scaffold behavior only |
| `rust/end-to-end-tests/Cargo.toml` | **blocked** | `polars`, `reqwest`, `ldc-engine`, `signal-fusion` commented out |
| `py/imp/hmm/trainer.py` (`PomegranateTrainer`) | **prototype** | Simplified placeholder due to current pomegranate API changes |
| `rust/inference-engine/fixtures/local-smoke.expected.jsonl` | **example output** | Canonical smoke fixture for deterministic validation — not proof of production readiness |

### Generated artifacts (gitignored, not tracked)
`hmm_integration_generated.toml`, `notebooks/*_results/`, `py/processed_data/`, `py/temp_*/`, `rust/sample/*.parquet`, `rust/partitioned_data/`, `rust/ldc-engine/ldc_pipeline_results.parquet`, `validation_output/`.

---

## Hygiene rule

A task is not "done" for repo-status purposes because code exists or a checkbox is checked. To claim a repo-level capability: implementation + documented runtime path + non-mock evidence + placeholder/artifact cleanup.

---

## Roadmap

> This section describes planned work. The status matrix above is the source of truth for what exists today.

### Phase summary

| Phase | Theme | Current truth | Remaining work |
|---|---|---|---|
| P0 | Truth reset | Canonical status, runtime-truth, placeholder labeling, hygiene policy, and definition-of-done language established | Keep docs aligned with actual repo state |
| P1 | Feature pipeline | **Implemented** — strongest runnable Rust path | Maintain and validate |
| P2 | LDC engine | **Implemented** as a library | Complete adjacent runtime integration; reduce ambiguity between library success and runtime success |
| P3 | HMM research + service | **Partially integrated** — substantial prototype and service code | Stabilize artifact interfaces; harden canonical startup path |
| P4 | Runtime integration | **Partially integrated** — real offline batch + daemon mode; non-mock integration test passes | Production hardening: Prometheus metrics, structured logging, container orchestration, config hot-reload |
| P5 | Backtesting + validation | **Partially integrated** — 96 tests, canonical E2E run with real fixture data | Stress-test with real market data; extend cross-language integration |
| P6 | Production hardening | **Not production-ready** | Only claim after: real always-on runtime, non-mock full-stack E2E, observability, operational evidence |

### P4: What remains for runtime

Daemon mode is implemented and the non-mock integration test passes. What's still needed for production-grade operation:

- Prometheus metrics integration (scaffolding exists in `rust/signal-fusion`)
- Structured logging for production consumption
- Container orchestration (Kubernetes deployment manifests)
- Config hot-reload without service restart
- Repo-wide proof that every optional integration path (HMM service, Redis, Kafka) is validated end-to-end without mocks

### P5: What remains for backtesting

- Canonical E2E test passes (22 tests, real fixture data) — done
- Walk-forward validation — done (5 tests)
- Stress-testing with real market data — not done
- Cross-language integration with Rust runtime output — not done

### P6: Production hardening exit criteria

Do not mark this complete until the repository has all of:

- A real supported always-on runtime (not just offline batch)
- Non-mock full-stack integration evidence (every integration surface validated without mocks)
- Failure handling evidence (circuit breaker, retry, alerting)
- Observability (metrics, structured logs, health checks beyond `/health`)
- Deployment automation
- Documentation that matches shipped behavior

### Planning rule

Every phase update must answer two questions separately:
1. **What is planned?**
2. **What is already true in the repository today?**

If those answers differ, the roadmap must say so explicitly — and the status matrix above must be updated first.
