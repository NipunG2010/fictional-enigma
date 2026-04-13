# Implementation Status

This document is the **canonical source of truth** for repository implementation status.

If another README, plan, spec, notebook summary, or task list disagrees with this document, treat **this document as authoritative** until the discrepancy is resolved.

Status labels are defined in [`definition-of-done.md`](definition-of-done.md).

## Status matrix

| Area | Code exists? | Integrated with adjacent components? | Runnable today? | Test evidence present? | Status label | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Feature pipeline | Yes | Yes, as a library and CLI path | Yes | Yes | **implemented** | `rust/feature-pipeline/src/lib.rs`, `rust/inference-engine/src/main.rs` (`ComputeFeatures`), tests in `rust/feature-pipeline/src/lib.rs` |
| LDC engine | Yes | Partial | Partial | Yes | **implemented** | `rust/ldc-engine/src/lib.rs`, extensive tests under `rust/ldc-engine/tests/` |
| HMM research package | Yes | Partial | Yes | Yes | **prototype** | `py/imp/hmm/{trainer.py,inference.py,weight_optimizer.py,artifact_management.py}`, notebooks, tests under `py/tests/` |
| HMM service | Yes | Partial | Yes, in isolation | Yes | **prototype** | `py/hmm_service/app.py`, `py/hmm_service/routers/inference.py`, `py/hmm_service/core/*`, tests in `py/hmm_service/test_*.py` |
| Signal fusion and emission | Yes | Partial | Partial | Yes | **partially integrated** | `rust/signal-fusion/src/{lib.rs,config.rs,hmm_client.rs,signal_pipeline.rs,signal_emitter.rs}`, tests under `rust/signal-fusion/tests/` |
| Inference-engine runtime | Partial | No | No, as a full runtime | No meaningful runtime proof | **not implemented** | `rust/inference-engine/src/main.rs` still contains TODOs for config load, component init, server start, and signal loop |
| End-to-end test suite | Yes | No, core components replaced by mocks | Yes, as a scaffold | Yes, for scaffold behavior | **test scaffold** | `rust/end-to-end-tests/Cargo.toml` has disabled real deps, `rust/end-to-end-tests/src/harness.rs` defines mock `SignalPipeline` and mock `LDCEngine` |
| Backtesting framework | Yes | Partial | Partial | Partial | **partially integrated** | `py/imp/backtesting/*.py`, open items in `.kiro/specs/backtesting-framework/tasks.md`, empty tracked file `py/test_backtest_engine_basic.py` |
| Production hardening | Partial | No | No | No repo-wide proof | **not production-ready** | scattered metrics/config/deployment code exists, but no repo-wide evidence of integrated runtime readiness or operational validation |

## Interpretation notes

### Feature pipeline
The feature pipeline is the strongest fully runnable path in the repository today. It has substantive implementation, local tests, and a CLI path exposed through `inference-engine`.

### LDC engine
The LDC crate is substantial and heavily tested, but the repository does not currently prove that it is wired into a real orchestrated runtime path without gaps or mocks.

### HMM research and HMM service
There is real Python implementation for both training/research and serving inference. The correct framing is **prototype/service prototype**, not empty shell and not production-ready deployment.

### Signal fusion
The signal-fusion crate contains real library code for HMM integration, fusion, emission, validation, and metrics. The gap is at **repo-wide runtime integration**, not absence of implementation.

### Inference runtime
The top-level runtime claim is the biggest mismatch the repo had before this reset. `rust/inference-engine` is currently a CLI plus startup skeleton, not a complete running service.

### End-to-end testing
The end-to-end testing project is valuable infrastructure, but it is currently a **mock-based harness** rather than proof of real full-system integration.

### Backtesting
Backtesting is more implemented than a "not started" label would suggest. At the same time, it should not be described as fully validated because dedicated testing and validation work remains incomplete.

## Usage rule

When updating a README, roadmap, task checklist, or milestone summary:
1. update this matrix first,
2. link back to it from the changed document,
3. avoid stronger language than the status label justified here.
