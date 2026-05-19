# IMP - Intelligent Market Prediction

IMP is a hybrid Rust + Python trading-systems repository with substantial research and library code for feature generation, Lorentzian Distance Classification (LDC), HMM-based regime modeling, signal fusion, and backtesting.

## Current state

**This repository is not currently a production-ready end-to-end trading runtime.**

What is true today:
- `rust/inference-engine` now provides a real **offline batch runtime** that orchestrates OHLCV input, feature generation, MR/TSMOM/LDC signals, HMM-or-fallback weights, fusion, optional emission, and canonical JSONL output.
- The Rust feature pipeline remains implemented and exposed through both the runtime and the `compute-features` CLI path.
- The Rust `ldc-engine` and `signal-fusion` crates contain substantial library code and tests and are now wired into the runtime's batch path.
- The Python HMM research package and FastAPI HMM service contain real prototype/service code that can be used by the Rust runtime in integration mode.
- The Python backtesting package is substantial, but its validation story is incomplete.

What is **not** true today:
- The repository is still **not** a production-ready always-on trading platform.
- `rust/end-to-end-tests` is still **not** proof of a real non-mock full-system pipeline; it currently uses mocked components and temporarily disabled real dependencies.
- The repository should **not** be described as production-ready, Phase 6 complete, or operationally hardened end-to-end.

For the authoritative status view, start here:
- [`docs/implementation-status.md`](docs/implementation-status.md)
- [`docs/runtime-truth.md`](docs/runtime-truth.md)
- [`docs/definition-of-done.md`](docs/definition-of-done.md)

## What works today

### Runnable paths
- **Rust offline batch runtime** via `cargo run -p inference-engine -- run-runtime --config ...`
- **Rust deterministic smoke runtime** via `cargo run -p inference-engine -- smoke --config ...`
- **Rust feature generation CLI** via `cargo run -p inference-engine -- compute-features ...`
- **Python research environment** for HMM experimentation and notebooks
- **Python HMM microservice** in isolation
- **Local infrastructure helpers** in `docker-compose.yml` for MinIO, Redis, and Kafka

### Substantial implemented code
- `rust/feature-pipeline`: feature computation, validation, partitioned parquet helpers
- `rust/ldc-engine`: LDC library, performance utilities, testing helpers
- `rust/signal-fusion`: HMM client, signal fusion, emission infrastructure, validation, metrics
- `py/imp/hmm`: HMM training, artifact management, weight optimization, regime analysis
- `py/hmm_service`: FastAPI service and supporting modules
- `py/imp/backtesting`: data loading, signal processing, trade generation, performance analysis, walk-forward components

## What does not yet have repo-wide proof

- A fully wired **always-on** Rust inference service with a live market-data loop and production operational semantics
- Repo-wide proof that every optional integration path is validated under real infrastructure without mock-heavy gaps elsewhere in the repository
- Production hardening claims such as deployment readiness, operational support, or validated latency/availability guarantees

## Repository layout

```text
.
├── docs/                     # Source-of-truth status, runtime, hygiene, and roadmap docs
├── notebooks/                # Research notebooks and notebook-generated artifacts
├── py/
│   ├── imp/                  # Python research and backtesting packages
│   ├── hmm_service/          # FastAPI HMM inference service prototype
│   └── tests/                # Python tests and research-validation tests
├── rust/
│   ├── feature-pipeline/     # Feature computation library
│   ├── ldc-engine/           # Lorentzian Distance Classification library
│   ├── signal-fusion/        # Signal fusion and emission library
│   ├── inference-engine/     # CLI + offline batch runtime orchestrator
│   ├── end-to-end-tests/     # Mock-heavy integration test scaffold
│   └── training-data-cli/    # Training data utilities
├── docker-compose.yml        # Local MinIO + Redis + Kafka helpers
├── validation/               # Validation scripts and references
└── validation_output/        # Example/generated validation output currently tracked
```

## Getting started

Use the setup guide and runtime runbook for the supported local paths:
- [`docs/setup.md`](docs/setup.md)
- [`docs/runtime-runbook.md`](docs/runtime-runbook.md)

Key caveats before you start:
- The documented baseline is a **real batch runtime**, not a production always-on deployment.
- `docker-compose.yml` brings up **MinIO, Redis, and Kafka**; the Python HMM service still has a separate startup path.
- The repo-level runtime truth is documented in [`docs/runtime-truth.md`](docs/runtime-truth.md).

## Documentation map

### Current-state documentation
- [`docs/implementation-status.md`](docs/implementation-status.md) - canonical implementation matrix
- [`docs/runtime-truth.md`](docs/runtime-truth.md) - what is runnable today, with caveats
- [`docs/placeholder-inventory.md`](docs/placeholder-inventory.md) - stubs, placeholders, mocks, empty files, and example outputs
- [`docs/repo-hygiene.md`](docs/repo-hygiene.md) - tracked artifact policy and cleanup baseline
- [`docs/definition-of-done.md`](docs/definition-of-done.md) - shared status vocabulary

### Supporting documentation
- [`docs/architecture.md`](docs/architecture.md) - architecture as implemented vs intended
- [`docs/runtime-runbook.md`](docs/runtime-runbook.md) - runtime modes, commands, outputs, infra, and failure handling
- [`docs/plan.md`](docs/plan.md) - roadmap, explicitly separated from current-state claims
- `docs/ldc-engine/` - LDC-specific documentation

## Definition-of-done language

This repo now distinguishes between:
- **implemented**
- **partially integrated**
- **prototype**
- **test scaffold**
- **not implemented**
- **not production-ready**

See [`docs/definition-of-done.md`](docs/definition-of-done.md) before marking a phase, task, or subsystem as complete.

## Repo notes

- Historical task checkboxes under `.kiro/specs/` record implementation work, not automatic proof of integrated runtime readiness.
- Generated outputs and temporary artifacts are still tracked in several places; see [`docs/repo-hygiene.md`](docs/repo-hygiene.md) for the current policy and cleanup baseline.
- Rust and Python package metadata declare MIT licensing, but the repository does not currently include a top-level `LICENSE` file.
