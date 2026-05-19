# Repo Hygiene Baseline

This document records the current cleanup baseline for tracked artifacts, generated outputs, and status drift.

## Current hygiene problems

### 1. Generated outputs are tracked without a clear policy
Examples currently present in the repo include:
- `validation_output/validation_report.json`
- `hmm_integration_generated.toml`
- notebook result directories under `notebooks/`
- `py/processed_data/**`
- `py/temp_requirements_test/**`
- `py/temp_test_output/**`
- `rust/sample/*.parquet`
- `rust/partitioned_data/**`
- `rust/ldc-engine/ldc_pipeline_results.parquet`

### 2. Empty tracked files blur real work vs placeholders
See [`placeholder-inventory.md`](placeholder-inventory.md).

### 3. Historical docs and specs overstated maturity
This reset introduces a canonical status matrix and runtime-truth doc so the repo stops using unchecked task completion as proof of integrated readiness.

### 4. Repository-level license file is absent
Rust and Python package metadata declare MIT, but the repository root does not currently include a top-level `LICENSE` file.

## Retention policy

### Keep in git only if the file is one of these
- source code
- hand-written documentation
- intentional small fixtures/examples that are clearly labeled
- config templates or test assets that are necessary to understand or exercise the repo

### Move, ignore, or remove if the file is one of these
- notebook execution output
- temporary scratch output
- validation/example reports that can be misread as proof of readiness
- generated config files that can be reproduced
- large or ambiguous parquet/json/png outputs without fixture labeling
- empty placeholder files

## First-pass decisions for current tracked artifacts

| Path or pattern | Keep now? | Intended long-term policy |
| --- | --- | --- |
| `rust/sample/*.parquet` | Temporarily yes | either relabel as fixtures/examples or move out of normal source paths |
| `validation_output/validation_report.json` | Temporarily yes | relabel as example output or remove |
| `hmm_integration_generated.toml` | No, going forward | generate locally; do not track regenerated copies |
| `notebooks/*_results/**` | No, going forward | ignore generated notebook outputs |
| `notebooks/processed_data/**` | No, going forward | ignore generated notebook data products |
| `py/processed_data/**` | No, going forward | ignore generated research outputs |
| `py/temp_*` | No | ignore or remove temp outputs |
| `rust/partitioned_data/**` | No | ignore generated partitions |
| `rust/ldc-engine/ldc_pipeline_results.parquet` | No, going forward | ignore generated result files |

## `.gitignore` policy

The root `.gitignore` now includes repo-specific patterns for generated research outputs, temp directories, validation output, and Rust sample/result parquet files.

Important nuance:
- adding ignore rules does **not** remove already tracked files from git history,
- it only prevents future drift until a dedicated cleanup PR removes or relocates the tracked artifacts.

## Guardrails

Before calling something complete, update:
1. [`implementation-status.md`](implementation-status.md)
2. [`runtime-truth.md`](runtime-truth.md) if runnable behavior changed
3. [`placeholder-inventory.md`](placeholder-inventory.md) if a stub was added or removed
4. relevant README/spec links

## Definition-of-done hygiene rule

A task is not "done" for repo-status purposes just because code exists or a checkbox is checked.

To claim a repo-level capability, the repo should have:
- the implementation,
- the documented runtime path,
- non-mock evidence for the claimed integration level,
- and cleanup/labeling for any placeholder or generated artifacts involved.
