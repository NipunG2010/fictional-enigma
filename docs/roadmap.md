# Roadmap

> This is a **roadmap** — not a statement of current implementation truth. For what exists today, see [`status.md`](status.md).

---

## Phase summary

| Phase | Theme | Current truth | Remaining work |
|---|---|---|---|
| P0 | Truth reset | Canonical status, runtime-truth, placeholder labeling, hygiene policy, and definition-of-done language established | Keep docs aligned with actual repo state |
| P1 | Feature pipeline | **Implemented** — strongest runnable Rust path | Maintain and validate |
| P2 | LDC engine | **Implemented** as a library | Complete adjacent runtime integration; reduce ambiguity between library success and runtime success |
| P3 | HMM research + service | **Partially integrated** — substantial prototype and service code | Stabilize artifact interfaces; harden canonical startup path |
| P4 | Runtime integration | **Partially integrated** — real offline batch + daemon mode; non-mock integration test passes | Production hardening: Prometheus metrics, structured logging, container orchestration, config hot-reload |
| P5 | Backtesting + validation | **Partially integrated** — 96 tests, canonical E2E run with real fixture data | Stress-test with real market data; extend cross-language integration |
| P6 | Production hardening | **Not production-ready** | Only claim after: real always-on runtime, non-mock full-stack E2E, observability, operational evidence |

---

## P4: What remains for runtime

Daemon mode is implemented and the non-mock integration test passes. What's still needed for production-grade operation:

- Prometheus metrics integration (scaffolding exists in `rust/signal-fusion`)
- Structured logging for production consumption
- Container orchestration (Kubernetes deployment manifests)
- Config hot-reload without service restart
- Repo-wide proof that every optional integration path (HMM service, Redis, Kafka) is validated end-to-end without mocks

## P5: What remains for backtesting

- Canonical E2E test passes (22 tests, real fixture data) — done
- Walk-forward validation — done (5 tests)
- Stress-testing with real market data — not done
- Cross-language integration with Rust runtime output — not done

## P6: Production hardening exit criteria

Do not mark this complete until the repository has all of:

- A real supported always-on runtime (not just offline batch)
- Non-mock full-stack integration evidence (every integration surface validated without mocks)
- Failure handling evidence (circuit breaker, retry, alerting)
- Observability (metrics, structured logs, health checks beyond `/health`)
- Deployment automation
- Documentation that matches shipped behavior

---

## Planning rule

Every phase update must answer two questions separately:
1. **What is planned?**
2. **What is already true in the repository today?**

If those answers differ, the roadmap must say so explicitly — and [`status.md`](status.md) must be updated first.
