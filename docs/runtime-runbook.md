# Runtime Runbook

This document is the operational guide for the Rust runtime in `rust/inference-engine`.

## What the runtime now does

The runtime provides a deterministic, documented batch path from:

```text
OHLCV input
  -> feature computation
  -> MR / TSMOM / LDC signal generation
  -> HMM weights or deterministic fallback weights
  -> signal fusion
  -> optional emission
  -> canonical JSONL output
```

The first-class MVP mode is **offline batch**. The runtime modes are still explicit so the repo has one coherent entrypoint for local smoke, HMM integration, and fallback-only runs.

## Supported runtime modes

The runtime mode is configured in `[runtime].mode`.

### 1. `offline_batch`
The main supported mode.

- Processes a finite OHLCV file end-to-end.
- Produces canonical JSONL output and a summary JSON.
- Can use HMM service integration, cached weights, or fallback weights depending on config.
- Can optionally emit fused signals.

### 2. `local_smoke`
Deterministic fixture mode.

- Uses the sample OHLCV fixture.
- Disables wall-clock timing noise.
- Uses static fallback weights.
- Disables bus emission.
- Has an expected output fixture for byte-for-byte comparison.

### 3. `integration_hmm`
Batch mode with live HMM service integration.

- Attempts `py/hmm_service` HTTP inference.
- Preserves request/response metadata in the final output.
- Uses cached or static fallback behavior when configured.

### 4. `fallback_only`
Deterministic no-HMM mode.

- Never calls the HMM service.
- Uses configured static weights only.
- Useful for local development and failure isolation.

## Canonical output

The canonical output format is **JSONL**.

One record is written per processed input row. Each record includes:

- `ohlcv`
- `features`
- `intermediate_signals`
  - `ldc`
  - `mr`
  - `tsmom`
  - `fusion_inputs`
- `hmm`
  - service/cache/fallback source
  - weights
  - probabilities when available
  - request/fallback metadata
- `fused_output`
  - fused score
  - recommended side
  - actionable side
  - suppression reason
  - weights used
- `audit`
  - run id
  - correlation id
  - input checksum
  - feature checksum
- `emission`
  - whether emission was configured
  - whether it was attempted
  - whether it succeeded
  - error text when emission failed
- `timing`
- `versions`

The batch run also writes a summary JSON with row counts and the SHA-256 of the canonical JSONL output.

## Config files

Bundled configs live under `rust/inference-engine/fixtures/`.

- `local-smoke.toml` — deterministic smoke fixture config
- `integration-hmm.example.toml` — template for HMM service integration
- `fallback-only.example.toml` — template for static-fallback-only runs
- `integration-test.toml` — non-mock integration test config (fallback_only mode, 16 rows max)

## Daemon (long-running) mode

The inference engine now supports a **daemon mode** (`serve` subcommand) that runs the pipeline periodically and exposes a health check HTTP endpoint. This is the first step toward production-style always-on operation.

### How it works

- The pipeline runs in a loop at a configurable interval (default: 60 seconds).
- A minimal health check HTTP server listens on a configurable port (default: 9090).
- SIGINT (Ctrl+C) triggers graceful shutdown — the current pipeline run completes, then the process exits cleanly.

### Health check endpoint

```
GET /health
```

Response:
```json
{"status":"ok"}
```

Status: **200 OK**

### Run the daemon

```bash
cd rust

# Default: port 9090, interval 60s
cargo run -p inference-engine -- serve

# Custom port and interval
cargo run -p inference-engine -- serve --port 8080 --interval 120 \
  --config inference-engine/fixtures/local-smoke.toml

# Shorthand --serve flag (equivalent to `serve` subcommand)
cargo run -p inference-engine -- --serve --serve-port 9090 --serve-interval 60
```

### What happens on shutdown

1. The signal handler receives SIGINT (Ctrl+C) or SIGTERM.
2. The daemon loop exits after the current pipeline run completes.
3. The health check server shuts down.
4. All resources are cleaned up (HMM cache saved, emitter flushed).

### Daemon config

The daemon mode reuses the same TOML config files as batch mode. Key settings:

- `--port` / `--serve-port`: Health check HTTP endpoint port (default: 9090)
- `--interval` / `--serve-interval`: Seconds between pipeline runs (default: 60)
- `--config`: Path to the runtime configuration TOML (default: `inference-engine/fixtures/local-smoke.toml`)

## Commands

Run commands from the Rust workspace root:

```bash
cd rust
```

### Describe the runtime modes

```bash
cargo run -p inference-engine -- describe-modes
```

### Run the runtime with a config

```bash
cargo run -p inference-engine -- run-runtime \
  --config inference-engine/fixtures/local-smoke.toml
```

### Run the deterministic smoke test

```bash
cargo run -p inference-engine -- smoke \
  --config inference-engine/fixtures/local-smoke.toml
```

### Run as a long-running daemon

```bash
cargo run -p inference-engine -- serve --port 9090 --interval 60 \
  --config inference-engine/fixtures/local-smoke.toml
```

### Run the non-mock integration test

```bash
cd rust
cargo test -p inference-engine -- --nocapture
```

This runs the real pipeline (feature pipeline → LDC → HMM/fallback → fusion → output) against sample data without mocked components. The integration tests live as inline `#[cfg(test)]` modules inside `main.rs`.

### Keep using the feature-only CLI

```bash
cargo run -p inference-engine -- compute-features \
  --input sample/ohlcv.parquet \
  --output sample/features_cli.parquet
```

## Local smoke path

The known-good smoke path uses:

- config: `rust/inference-engine/fixtures/local-smoke.toml`
- input fixture: `rust/sample/ohlcv.parquet`
- expected canonical output: `rust/inference-engine/fixtures/local-smoke.expected.jsonl`
- expected summary: `rust/inference-engine/fixtures/local-smoke.expected.summary.json`

A successful smoke run means:

1. the runtime completes,
2. the generated JSONL matches the expected fixture byte-for-byte,
3. the generated summary matches the expected summary byte-for-byte.

## HMM integration contract

The runtime expects the HMM service contract exposed by `py/hmm_service`:

- `POST /inference/predict`
- request body:
  - `observations: [s_ldc, s_mr, s_tsmom]`
  - `timestamp`
  - `request_id`
- response fields used by the runtime:
  - `state_probabilities`
  - `most_likely_state`
  - `confidence`
  - `fusion_weights`
  - `model_version`
  - `timestamp`
  - `processing_time_ms`

If the service is unavailable, the runtime behavior is explicit and config-driven:

- use cached weights when available,
- otherwise use static fallback weights,
- or fail the run if configured to do so.

## Optional emission

Emission is optional.

Supported backends:

- `none`
- `redis`
- `kafka`
- `both`

Runtime emission is wired through `rust/signal-fusion`'s `SignalEmitter`.

The runtime does **not** require bus backends for successful local smoke or fallback-only runs.

## Local infrastructure story

The root `docker-compose.yml` now covers local infrastructure needed by the runtime's optional integrations:

- MinIO
- Redis
- Kafka

Start them with:

```bash
docker compose up -d minio redis kafka
```

### HMM service

The root compose file does **not** start the Python HMM service. That service still has its own startup path because it depends on the Python research/service environment and model-loading setup.

Start it separately when needed:

```bash
cd py/hmm_service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Failure-mode behavior

### HMM unavailable

Behavior depends on config:

- `use_cache_then_fallback`
- `use_fallback`
- `fail`

The chosen path is recorded in the output's `hmm` metadata.

### Bad config

The run fails during startup before any output is emitted.

### Missing data file or missing required columns

The run fails before feature computation.

### Invalid features

Behavior depends on `failure.invalid_features`:

- `fail`
- `skip_row`

Warmup-related missing values are handled explicitly rather than silently disguised as valid signals.

### Emission failure

Behavior depends on `failure.emission_failure`:

- `continue`
- `fail_run`

Failures are recorded in the output's `emission.error` field.

## Verification checklist

A healthy batch run should give you:

- a canonical JSONL output file,
- a summary JSON with row counts and SHA-256,
- optional feature parquet output,
- optional weight-cache JSON,
- no placeholder startup TODO path in `rust/inference-engine`.
