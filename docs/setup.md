# Setup Guide

This setup guide documents the **supported local paths that are truthful today**.

It does **not** promise a full production runtime.

For the current runtime boundary, see [`runtime-truth.md`](runtime-truth.md).

## Prerequisites

- Rust toolchain
- Python 3.9+
- `pip`
- Docker / Docker Compose if you want local MinIO and Redis

## Option 1: Rust feature-generation CLI

This is the clearest Rust path currently exposed by the repository.

```bash
cd rust
cargo run -p inference-engine -- compute-features \
  --input sample/ohlcv.parquet \
  --output sample/features_cli.parquet
```

What this gives you:
- feature computation through the implemented `feature-pipeline` library

What this does not give you:
- a full always-on inference runtime
- a server process
- a live end-to-end trading pipeline

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

## Option 4: Local dependency services

```bash
docker compose up -d minio redis
```

This compose file currently starts:
- MinIO on ports `9000` and `9001`
- Redis on port `6379`

It does **not** currently start:
- Kafka
- the HMM service
- the Rust runtime

## What not to expect from setup today

After following this guide, you should **not** assume you now have:
- a full repo-wide end-to-end runtime,
- a production-ready deployment,
- or a non-mock E2E validation story.

Those remain future integration/hardening work.
