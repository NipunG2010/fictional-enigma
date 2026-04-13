# Definition of Done

This repository uses the following status language consistently across READMEs, docs, plans, and specs.

## Status vocabulary

### implemented
Use this when:
- code exists in the repository,
- the code is more than a placeholder,
- and at least basic local or automated evidence exists.

Do **not** use this to imply production deployment, repo-wide integration, or operational readiness.

### partially integrated
Use this when:
- a component is wired to some adjacent components,
- but a real end-to-end path still has gaps, mocks, disabled dependencies, or missing runtime orchestration.

### prototype
Use this when:
- the code is real and useful,
- but it is still exploratory, simplified, or intentionally provisional,
- or the implementation is not yet the intended final production approach.

### test scaffold
Use this when:
- the code primarily exists to support future validation,
- and mocks, stubs, or simulated components stand in for the real production path.

### not implemented
Use this when:
- the claimed runtime or behavior does not actually exist yet,
- even if partial setup code, TODOs, or dependency wiring exists.

### tested
Use this only when:
- automated tests exercise the relevant implementation path,
- and the tests match the level of claim being made.

Examples:
- mock-based tests can justify **tested scaffold behavior**,
- they cannot justify **tested full-system integration**.

### production-ready
Use this only when all of the following are true:
- the real runtime path exists,
- the relevant integrations are non-mock,
- failure handling is in place,
- observability/operability is present,
- docs describe the supported path,
- and the repo has evidence to support the claim.

This repository does **not** currently meet that bar as a whole.

## Evidence ladder

From weakest to strongest:
1. idea or plan
2. placeholder or TODO
3. code exists
4. code exists with unit/local tests
5. adjacent integration works
6. real end-to-end non-mock path works
7. operationally supported and production-ready

Do not skip levels in documentation language.

## Rules for phase and task language

### Safe phrases
- "implemented as a library"
- "service prototype"
- "partially integrated"
- "mock-based test harness"
- "validation incomplete"
- "runnable in isolation"

### Phrases to avoid unless proven
- "complete"
- "end-to-end validated"
- "production-grade"
- "production-ready"
- "fully integrated"
- "operationally hardened"

## Repo-wide done checklist

Before upgrading a subsystem status in README or roadmap, confirm:
- [ ] the code path exists
- [ ] the runtime path is documented
- [ ] the evidence matches the strength of the claim
- [ ] mocks/placeholders are called out
- [ ] generated outputs are labeled or ignored
- [ ] the canonical status docs were updated first
