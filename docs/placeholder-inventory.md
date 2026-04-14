# Placeholder and Stub Inventory

This inventory exists so placeholders, scaffolds, mocks, and example outputs are not mistaken for finished product behavior.

## Classification legend

- **prototype**: intentional early implementation that works for exploration but is not the final intended approach
- **incomplete**: real code path with explicit missing work
- **test-only**: exists to support tests, not production/runtime behavior
- **blocked**: work is limited by a dependency or environment issue called out in the repo
- **example output**: generated output kept for illustration, not validation proof
- **empty residue**: tracked empty file with no current implementation value proven in-repo

## High-impact placeholders and stubs

| Path | Classification | Why it is called out | Current handling |
| --- | --- | --- | --- |
| `rust/inference-engine/fixtures/local-smoke.expected.jsonl` and `.summary.json` | **example output** | These are canonical smoke fixtures for deterministic validation, not proof of production readiness by themselves | Treat as runtime fixtures paired with the real batch runtime and smoke command |
| `rust/end-to-end-tests/src/harness.rs` | **test-only** | Defines local mock `SignalPipeline`, signals, and mock `LDCEngine` implementations | Treat as scaffold behavior only |
| `rust/end-to-end-tests/Cargo.toml` | **blocked** | `polars`, `reqwest`, `ldc-engine`, and `signal-fusion` are commented out due to dependency/build issues | Treat as partial harness, not integrated E2E proof |
| `py/imp/hmm/trainer.py` (`PomegranateTrainer`) | **prototype** | Explicitly uses a simplified placeholder implementation because of current pomegranate API changes | Keep labeled as prototype until replaced or validated |
| `validation_output/validation_report.json` | **example output** | Contains mostly null/zero metrics and should not be interpreted as proof of validated readiness | Keep only as example output unless replaced with real validated evidence |

## Empty tracked files

These should not be mistaken for implemented work.

| Path | Classification | Suggested interpretation |
| --- | --- | --- |
| `notebooks/NOTEBOOK_08_FIXES.md` | **empty residue** | incomplete notebook follow-up note |
| `notebooks/TASK_6_FUSION_WEIGHT_OPTIMIZATION_SUMMARY.md` | **empty residue** | incomplete summary placeholder |
| `notebooks/gen_nb.py` | **empty residue** | incomplete helper script placeholder |
| `py/imp/data/IMPLEMENTATION_SUMMARY.md` | **empty residue** | incomplete summary placeholder |
| `py/imp/tuning/TASK_11_IMPLEMENTATION_SUMMARY.md` | **empty residue** | incomplete summary placeholder |
| `py/scripts/verify_task7_notebook.py` | **empty residue** | incomplete verification script |
| `py/test_backtest_engine_basic.py` | **empty residue** | missing backtesting test implementation |
| `py/tests/TASK_9_IMPLEMENTATION_SUMMARY.md` | **empty residue** | incomplete summary placeholder |
| `py/tests/test_task3_minio_download_listing.py` | **empty residue** | incomplete test placeholder |
| `py/tests/test_task4_tagging_deployment.py` | **empty residue** | incomplete test placeholder |
| `rust/ldc-engine/examples/csv_data_test.rs` | **empty residue** | incomplete example placeholder |
| `rust/ldc-engine/examples/test_configurations.rs` | **empty residue** | incomplete example placeholder |
| `rust/signal-fusion/tests/config_integration_tests.rs` | **empty residue** | incomplete test placeholder |

## Generated or example artifacts that need explicit labeling

| Path or pattern | Classification | Why it matters |
| --- | --- | --- |
| `hmm_integration_generated.toml` | **example output** | generated config currently tracked without clear policy |
| `notebooks/*_results/**` and `notebooks/processed_data/**` | **example output** | notebook outputs can look like validated artifacts if unlabeled |
| `py/processed_data/**` | **example output** | generated research outputs should not be read as canonical data assets |
| `py/temp_requirements_test/**`, `py/temp_test_output/**` | **empty residue** | temporary outputs should not stay in the normal repo surface |
| `rust/sample/*.parquet` | **example output** | useful as sample fixtures only if explicitly treated that way |
| `rust/partitioned_data/**` | **example output** | generated partition output, not source code |
| `rust/ldc-engine/ldc_pipeline_results.parquet` | **example output** | generated result file currently tracked in crate root |

## Rule going forward

If a file is a stub, placeholder, mock, temp file, generated artifact, or historical note, it must satisfy **one** of these conditions:
1. be clearly labeled in the file and docs,
2. live in a fixture/example location,
3. be ignored and left out of version control,
4. or be removed.
