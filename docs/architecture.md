# Architecture

This document describes the repository architecture **as it exists today**, while also distinguishing the intended end-state from the currently implemented runtime.

For current implementation truth, see:
- [`implementation-status.md`](implementation-status.md)
- [`runtime-truth.md`](runtime-truth.md)

## High-level shape

The repository is a hybrid system with:
- **Rust** for feature computation, LDC logic, signal fusion, and runtime-oriented components
- **Python** for HMM research, model/training workflows, a FastAPI inference service, and backtesting

## Component map

| Component | Primary paths | Current role |
| --- | --- | --- |
| Feature pipeline | `rust/feature-pipeline` | Implemented technical indicator and signal-prep library |
| Inference entrypoint | `rust/inference-engine` | Feature CLI plus incomplete runtime stub |
| LDC engine | `rust/ldc-engine` | Implemented LDC-focused library with extensive testing/perf utilities |
| Signal fusion | `rust/signal-fusion` | Implemented fusion/emission library with HMM client and metrics |
| End-to-end tests | `rust/end-to-end-tests` | Mock-heavy integration scaffold |
| HMM research | `py/imp/hmm` | Implemented research/prototype package |
| HMM service | `py/hmm_service` | FastAPI service prototype for inference and weights |
| Backtesting | `py/imp/backtesting` | Substantial framework implementation with incomplete validation |
| Local infra helpers | `docker-compose.yml` | MinIO + Redis only |

## Intended end-state data flow

```text
OHLCV data
  -> feature pipeline
  -> LDC / MR / TSMOM components
  -> HMM regime inference
  -> signal fusion
  -> signal emission
  -> downstream consumers
```

## Current implemented reality

Today, the architecture is better described as a set of **substantial component implementations** than as one fully proven runtime.

### Stronger implemented areas
- feature computation in Rust
- LDC library code and testing infrastructure
- HMM research/training code in Python
- HMM inference service prototype in Python
- signal fusion/emission library code in Rust
- backtesting modules in Python

### Weaker integration areas
- repo-level runtime orchestration in `rust/inference-engine`
- non-mock full-stack end-to-end testing
- production hardening claims across the whole repository

## Supported architecture statement

A truthful one-paragraph description of the repo today is:

> IMP is a hybrid Rust/Python repository with real implementations for feature generation, LDC, HMM research, HMM service endpoints, signal fusion, and backtesting components. The repository currently behaves more like a collection of substantial libraries and prototypes than a fully proven production runtime, because the main Rust orchestration path is still incomplete and the repo-wide end-to-end test harness still relies on mocks.

## Operational boundary

Do not treat the following as currently guaranteed by the repo:
- a complete always-on inference service,
- fully validated end-to-end pipeline latency claims,
- production message-bus deployment proof,
- production availability/SLO claims.

Those remain part of the **target architecture**, not the **current implementation truth**.
