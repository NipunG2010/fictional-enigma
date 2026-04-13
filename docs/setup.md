# Setup Guide

This setup guide documents the **supported local paths that are truthful today**.

It does **not** promise a full production runtime.

For the current runtime boundary, see [`runtime-truth.md`](runtime-truth.md).

## Prerequisites

- Rust toolchain
- Python 3.9+
- `pip`
- Docker / Docker Compose if you want local MinIO and Redis

## Option 1: Rust offline batch runtime

This is the primary supported Rust path.

```bash
cd rust
cargo run -p inference-engine -- run-runtime \
  --config inference-engine/fixtures/local-smoke.toml
```

What this gives you:
- OHLCV loading
- feature computation through `feature-pipeline`
- MR / TSMOM / LDC signal generation
- HMM or fallback weights
- signal fusion
- canonical JSONL output
- optional emission wiring

What this does not give you:
- a production always-on inference service
- a fully validated repo-wide non-mock E2E proof for every optional integration path

## Option 2: Python research environment

Manual setup is the most reliable documented baseline.

```bash
cd py
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,optimization,research]"
```

Typical next steps:

```bash
jupyter lab
```

Notes:
- notebooks live at the repository top level in `notebooks/`
- research code lives under `py/imp/`

## Option 3: HMM microservice prototype

```bash
cd py/hmm_service
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Useful endpoints once the service is up:
- `/docs`
- `/health`
- `/inference/state-probabilities`
- `/inference/fusion-weights`
- `/inference/predict`

## Option 4: Deterministic local smoke validation

```bash
cd rust
cargo run -p inference-engine -- smoke \
  --config inference-engine/fixtures/local-smoke.toml
```

This compares the generated runtime output against the bundled expected fixture.

## Option 5: Local dependency services

```bash
docker compose up -d minio redis kafka
```

This compose file currently starts:
- MinIO on ports `9000` and `9001`
- Redis on port `6379`
- Kafka on port `9092`

It does **not** currently start:
- the Python HMM service
- the Rust runtime process itself

## What not to expect from setup today

After following this guide, you should **not** assume you now have:
- a production-ready deployment,
- a continuously running live trading service,
- or repo-wide proof that every historical scaffold and optional integration path is now fully validated.

The supported truth is a real batch runtime plus explicit optional integrations, not full production hardening.
