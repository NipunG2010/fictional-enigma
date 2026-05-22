# Implementation Status

This document is the **canonical source of truth** for repository implementation status.

If another README, plan, spec, notebook summary, or task list disagrees with this document, treat **this document as authoritative** until the discrepancy is resolved.

Status labels are defined in [`definition-of-done.md`](definition-of-done.md).

## Status matrix

| Area | Code exists? | Integrated with adjacent components? | Runnable today? | Test evidence present? | Status label | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Feature pipeline | Yes | Yes, as a library and CLI path | Yes | Yes | **implemented** | `rust/feature-pipeline/src/lib.rs`, `rust/inference-engine/src/main.rs` (`ComputeFeatures`), tests in `rust/feature-pipeline/src/lib.rs` |
| LDC engine | Yes | Partial | Partial | Yes | **implemented** | `rust/ldc-engine/src/lib.rs`, extensive tests under `rust/ldc-engine/tests/` |
| HMM research package | Yes | Partial | Yes | Yes (30 unit tests passing) | **partially integrated** | `py/imp/hmm/{trainer.py,inference.py,weight_optimizer.py,artifact_management.py,models.py,regime_analysis.py}`, notebooks, tests under `py/tests/` |
| HMM service | Yes | Partial | Yes, in isolation | Yes (service tests pass) | **partially integrated** | `py/hmm_service/app.py`, `py/hmm_service/routers/inference.py`, `py/hmm_service/core/*`, tests in `py/hmm_service/test_*.py` |
| Signal fusion and emission | Yes | Partial | Partial | Yes | **partially integrated** | `rust/signal-fusion/src/{lib.rs,config.rs,hmm_client.rs,signal_pipeline.rs,signal_emitter.rs}`, tests under `rust/signal-fusion/tests/` |
| Inference-engine runtime | Yes | Yes, across feature pipeline, LDC, HMM/fallback, fusion, optional emission in batch mode, and **daemon (serve) mode** with health check endpoint and graceful shutdown | Yes, as both **offline batch runtime** and **long-running daemon** | Yes, via deterministic smoke fixture/output validation + **non-mock integration test** + compile-verified daemon | **partially integrated** | `rust/inference-engine/src/{main.rs,config.rs,runtime.rs,hmm.rs,schema.rs,daemon.rs}`, `rust/inference-engine/fixtures/{local-smoke.toml,integration-test.toml}`, `docs/runtime-runbook.md` |
| End-to-end test suite | Yes | No, core components replaced by mocks | Yes, as a scaffold | Yes, for scaffold behavior | **test scaffold** | `rust/end-to-end-tests/Cargo.toml` has disabled real deps, `rust/end-to-end-tests/src/harness.rs` defines mock `SignalPipeline` and mock `LDCEngine` |
| Backtesting framework | Yes | Partial | Yes | Yes (96 tests: 32 core + 21 MinIO + 16 tagging + 22 canonical E2E + 5 walk-forward) | **partially integrated** | `py/imp/backtesting/*.py` (13 modules), `py/test_backtest_engine_basic.py` (32 tests), `py/tests/test_task3_minio_download_listing.py` (21 tests), `py/tests/test_task4_tagging_deployment.py` (16 tests), `py/tests/test_canonical_backtest.py` (22 E2E tests with real fixture data), `py/tests/backtest_fixtures/*` (deterministic parquet fixtures) |
| Production hardening | Partial | No | No | No repo-wide proof | **not production-ready** | scattered metrics/config/deployment code exists, but no repo-wide evidence of integrated runtime readiness or operational validation |

## Interpretation notes

### Feature pipeline
The feature pipeline is the strongest fully runnable path in the repository today. It has substantive implementation, local tests, and a CLI path exposed through `inference-engine`.

### LDC engine
The LDC crate is substantial and heavily tested, but the repository does not currently prove that it is wired into a real orchestrated runtime path without gaps or mocks.

### HMM research and HMM service
There is real Python implementation for both training/research and serving inference. Artifact interfaces (`HMMArtifact`/`FusionWeights`) are standardized and compatible with the Rust `signal-fusion` crate. The service implements a canonical startup path from MinIO storage through model loading to inference endpoints. Documentation now covers both the canonical service startup path and the artifact-loading workflow. Remaining gaps are production hardening, not missing implementation.

### Signal fusion
The signal-fusion crate contains real library code for HMM integration, fusion, emission, validation, and metrics. The gap is at **repo-wide runtime integration**, not absence of implementation.

### Inference runtime
`rust/inference-engine` now ships a real offline batch orchestrator **and** a long-running daemon mode with health check endpoint and graceful shutdown. The daemon mode (`serve` subcommand) is the first step toward production-style always-on operation. A non-mock integration test validates the full pipeline path (feature pipeline → LDC → HMM/fallback → fusion → output). The remaining gap is production hardening: monitoring, alerting, deployment automation, and repo-wide proof around every optional integration surface.

### End-to-end testing
The end-to-end testing project is valuable infrastructure, but it is currently a **mock-based harness** rather than proof of real full-system integration.

### Backtesting
Backtesting is now backed by **96 unit tests** across 5 dedicated test files covering the full framework: configuration, signal processing, trade generation, portfolio state management, cost/slippage models, performance analysis, walk-forward validation, engine orchestration, MinIO artifact management, tagging/deployment workflows, **and a canonical end-to-end run with real fixture data**. The canonical test (`tests/test_canonical_backtest.py` — 22 tests) loads deterministic parquet fixtures (3 symbols, 720 hourly bars each), runs the full 6-step pipeline, and validates structural invariants on the results. Fixtures are generated by `tests/scripts/generate_backtest_fixtures.py`.

To run the canonical backtest:
```bash
cd py
python -m pytest tests/test_canonical_backtest.py -v
```

## Usage rule

When updating a README, roadmap, task checklist, or milestone summary:
1. update this matrix first,
2. link back to it from the changed document,
3. avoid stronger language than the status label justified here.
