# Roadmap

> This document is a **roadmap**, not a statement of current implementation truth.
>
> For what is actually implemented today, see [`implementation-status.md`](implementation-status.md) and [`runtime-truth.md`](runtime-truth.md).

## Purpose

The original plan in this repository mixed future-state architecture with present-state claims. This rewritten roadmap separates the two.

## Phase summary

| Phase | Theme | Current truth | Roadmap intent |
| --- | --- | --- | --- |
| P0 | Truth reset | This document set now establishes canonical status, runtime truth, placeholder labeling, hygiene policy, and definition-of-done language | keep docs and specs aligned with actual repo state |
| P1 | Feature pipeline | mostly implemented | maintain and validate the strongest runnable path |
| P2 | LDC engine | implemented as a library | complete adjacent runtime integration and reduce ambiguity between library success and runtime success |
| P3 | HMM research and service | substantial prototype/service prototype | stabilize artifacts and supported inference path |
| P4 | Runtime integration | incomplete | finish the real Rust runtime orchestration and remove current TODO/stub gaps |
| P5 | Backtesting and validation | substantial framework, canonical run complete (96 tests across 5 files) | stress-test with real market data and extend cross-language integration |
| P6 | Production hardening | not achieved | only claim after real runtime, non-mock integration, observability, and operational evidence exist |

## Immediate P0 priorities

1. keep `docs/implementation-status.md` as the canonical status matrix,
2. keep `docs/runtime-truth.md` aligned with actual runnable paths,
3. keep placeholder and generated-artifact labeling explicit,
4. stop using checked task boxes as proof of production or integration readiness,
5. ensure README and doc links match the real file layout.

## Follow-on roadmap

### P1: Strengthen the feature pipeline baseline
- preserve the feature CLI path as the most reliable documented Rust entrypoint
- keep feature computation validation current
- document sample data and fixture policy clearly

### P2: Finish the LDC integration story
- connect the implemented LDC library to a real runtime path
- clarify example/benchmark code versus supported runtime code
- retain performance tooling, but separate benchmark aspiration from service guarantees

### P3: Stabilize HMM research and service boundaries
- keep Python research workflows explicit as research/prototype paths
- standardize artifact interfaces between Python and Rust
- document one canonical service startup path and one canonical artifact-loading path

### P4: Complete runtime orchestration
Exit criteria should include all of the following:
- configuration loading is implemented,
- components initialize from real config,
- the server/runtime entrypoint actually starts,
- the signal loop exists,
- runtime docs describe the supported path,
- non-mock integration evidence exists.

### P5: Finish backtesting validation
Exit criteria:
- [x] open backtesting test tasks are closed,
- [x] empty placeholder tests are removed or implemented,
- [x] one documented canonical backtest run exists (`py/tests/test_canonical_backtest.py`, 22 tests with real fixture data),
- [x] validation outputs are clearly separated from example/demo outputs (fixtures in `py/tests/backtest_fixtures/`, generated artifacts gitignored).

### P6: Production hardening
Do not mark this phase complete until the repository has:
- a real supported runtime path,
- non-mock full-stack integration evidence,
- observability and failure-handling evidence,
- repo hygiene around generated outputs,
- documentation that matches the shipped behavior.

## Planning rule

Every future phase update should answer two separate questions:
1. **What is planned?**
2. **What is already true in the repository today?**

If those answers are different, the roadmap must say so explicitly.
