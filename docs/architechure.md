Nice — below is a **production-ready, high-level architecture** for a hybrid Rust+Python system that uses Rust/Polars for high-throughput feature pipelines & inference (LDC / MR / TSMOM / ANN) and Python for research, flexible model training (HMMs, hyperparameter search) and tooling. I include concrete components, data & model artifact formats, how services interact, latency/operational notes, and an ordered rollout you can follow.

I’ll start with the top-level architecture diagram and then drill into components, data flows, interfaces, ops, and a step-by-step rollout plan.

---

# Top-level architecture (summary)

```
+----------------------+        +----------------------+        +--------------------+
|  Research & Train    | <----> |  Artifact Store      | <----> |  Rust Inference    |
|  (Python notebooks)  |        |  (S3/MinIO + Parquet)|        |  Engine (Polars)   |
|  - vectorbt, PyTorch |        |                      |        |  - LDC (k-NN)      |
|  - hmmlearn/pomegranate |     |                      |        |  - MR, TSMOM       |
+----------------------+        +----------------------+        +--------------------+
         |                                   ^                         |
         | model artifacts, metrics          |                         |
         v                                   |                         v
+----------------------+        +----------------------+        +--------------------+
|  Model Training      |        |  Feature Store       |        |  Signal Bus &      |
|  Service (Python)    | -----> |  (Parquet, Arrow IPC)| -----> |  Execution (Redis/ |
|  - HMM training      | pushes |                      |  feeds |  Kafka)            |
|  - weight optimizer  |        +----------------------+        +--------------------+
+----------------------+                                               |
                                                                       v
                                                              +------------------+
                                                              |  Monitoring &    |
                                                              |  Ops (Prom/Graf) |
                                                              +------------------+
```

Key ideas:

* Rust/Polars is the **production data & inference core** for speed and reliability. Polars’ Rust implementation is the recommended DataFrame engine. ([Docs.rs][1], [Polars][2])
* Python is the **research and model-training environment** (fast iteration, mature libraries for HMMs and probabilistic models). Use `hmmlearn` or `pomegranate` for regime modeling and prototyping. ([hmmlearn.readthedocs.io][3], [pomegranate.readthedocs.io][4])
* Storage & interchange format: **Parquet** (and Arrow IPC for zero-copy) for features and historical data; model artifacts as JSON / joblib / NPZ / serialized protobuf.
* Communication between parts: REST/gRPC for model/weight fetching, or file-based pulls from the artifact store for deterministic deployments.

---

# Components — responsibilities & tech choices

## 1) Data Lake / Artifact Store

* **What**: S3/MinIO object store for Parquet files (per-symbol, per-interval), plus a model-artifact bucket.
* **Why**: Columnar Parquet + Arrow IPC is efficient, language-agnostic, and fast to read/write from both Rust and Python.
* **Artifacts**:

  * Raw OHLCV Parquet partitions (symbol/date/interval).
  * Feature Parquet (zscore, RSI, WT, CCI, ADX, mom, vol, etc.).
  * Model artifacts: LDC training snapshot (feature vectors + labels) in flat binary, HNSW index file (if used), HMM params (JSON or joblib), and fusion weight sets (JSON).

## 2) Research & Model Training (Python)

* **Stack**: Jupyter + vectorbt / pandas / numpy for exploration; scikit-learn & PyTorch for prototyping (scikit-learn is good for k-NN baselines; PyTorch for learned embeddings); `pomegranate` or `hmmlearn` for HMM training. ([pomegranate.readthedocs.io][4], [hmmlearn.readthedocs.io][3])
* **Responsibilities**:

  * Experimentation: feature design, hyperparameter search, ablation.
  * Train HMMs, evaluate #states, evaluate state interpretability (using AIC/BIC or held-out likelihood).
  * Compute state-conditioned fusion weights (one weight vector per HMM state).
  * Produce model artifacts (HMM params + per-state weights + metadata), store them in Artifact Store.
* **Why Python**: Mature ecosystem for statistical HMMs and rapid prototyping of experiments.

## 3) Model Training Service (Python microservice)

* **Purpose**: Scheduleable service (cron/Airflow) to:

  * Re-train HMMs on rolling windows (Baum–Welch / EM) using `pomegranate` / `hmmlearn`.
  * Derive `w^{(state)}` per regime by optimizing weights on state slices.
  * Validate candidates via walk-forward; publish best artifact to Artifact Store (with versioning).
* **Outputs**: `hmm_vX.json` (A, μ, Σ), `weights_vX.json`, training metrics & health report.

## 4) Feature Pipeline & Inference Engine (Rust)

* **Stack**: Rust service using `polars` (lazy API) for columnar, multithreaded feature building; custom LDC k-NN implementation (fast Rust), optional ANN via `hnsw_rs` for large training sets. ([Docs.rs][1])
* **Responsibilities**:

  * Read Parquet or Arrow IPC from Artifact Store.
  * Build/maintain rolling feature windows per symbol (RSI, WaveTrend, CCI, ADX, MA, std, zscore, TSMOM, etc.) using Polars lazy expressions. ([docs.pola.rs][5])
  * Maintain ring-buffer of labeled feature vectors for LDC; expose k-NN query engine with Lorentzian distance (optimized, multithreaded).
  * Provide MR and TSMOM scorers (pure, deterministic, Rust-native).
  * Optionally hold an in-memory HNSW index (from `hnsw_rs`) for fast ANN queries if N\_train large. ([Crates.io][6], [GitHub][7])
* **APIs**:

  * Internal library: `infer(features) -> { s_LDC, s_MR, s_TSMOM }`
  * Service endpoints:

    * `/signal` (REST or gRPC): returns fusion `S_t`, weights, side, confidence, and components (for audit).
    * `/metrics` for internal telemetry.

## 5) HMM Inference & Weighting

Two deployment options (pick one, both possible):

**Option A — Python HMM microservice (recommended initial):**

* The Python Model Training Service also exposes a **lightweight HMM inference endpoint** (FastAPI / Uvicorn) that:

  * Loads the latest HMM artifact (A, μ, Σ) and per-state weights `w^{(j)}`.
  * Accepts observation vectors `[s_LDC, s_MR, s_TSMOM]` (or feature vector) and returns filtered state probabilities `P(S_t=j | O_{1:t})` or Viterbi path on a short window.
  * Returns computed `w_t = Σ_j P_j * w^{(j)}` to Rust on request or returns `P_j` and lets Rust compute weighted sum.
* **Why**: fast to develop and test using mature HMM libraries (`pomegranate` or `hmmlearn`). Low-latency is acceptable for intraday (1m) workflows. ([pomegranate.readthedocs.io][4], [hmmlearn.readthedocs.io][3])

**Option B — Port HMM inference to Rust (long-term):**

* After proving HMM usefulness, port the filter/Viterbi implementation to Rust for a single-process deployment. Baum–Welch can remain in Python (training), but online filtering (forward probabilities) implemented in Rust for minimal dependence.
* This port ensures fewer cross-language RPC calls and reduces runtime dependencies.

## 6) Fusion & Signal Emitter (Rust)

* **Inputs**: `s_LDC`, `s_MR`, `s_TSMOM` (from Rust inference), and `w_t` (from HMM microservice or precomputed static).
* **Logic**:

  * Compute `S_t = dot(w_t, [s_LDC, s_MR, s_TSMOM])`.
  * Apply threshold `τ`, cooldown, stop-loss/time-stop logic (emit signals only if |S\_t| > τ and other risk filters pass).
* **Output**: Publish signal JSON lines to Signal Bus (Redis stream / Kafka topic) consumed by execution engine.

## 7) Execution layer (external)

* Execution is out-of-scope for you (you mentioned you'll handle it). Provide a clear contract:

  * Signal schema: timestamp, symbol, side, size fraction, confidence, breakdown of components + model version.
  * Acks & fills: execution system returns fill events back to the system for tracking PnL and updating training labels (useful for labeling in LDC).

## 8) Signal Bus & Observability

* **Signal stream**: Redis Streams or Kafka for reliable, ordered delivery.
* **Observability**: Prometheus metrics from Rust & Python services; Grafana dashboards (feature distributions, state probabilities, LDC distance histogram, latency, errors).
* **Audit logs**: Persist signals + features + model-version to object store for replay & forensic analysis.

---

# Data & model artifact formats (recommended)

* **Time-series & features**: Parquet partitioned by `symbol=BTCUSDT/interval=5m/date=YYYY-MM-DD`. Use Arrow IPC for in-memory handoffs if needed.
* **LDC snapshot**: Compact binary (flat arrays of `f32`) plus JSON meta (feature order, normalization medians/IQRs).
* **HNSW index**: native `hnsw_rs` on-disk file (if used); loadable by Rust inference engine.
* **HMM artifact**: JSON with `{A, pi, mus, covariances, state_weights}`, and a version string + train-window metadata.
* **Signals**: JSONL with full breakdown and model versions.

---

# Latency & performance considerations

* Rust inference (Polars + LDC brute force up to N≈50k) should handle 1m bars for tens of symbols on a single beefy core machine; use `rayon` and `select_nth_unstable` for speed. Polars provides multi-threaded lazy evaluation. ([docs.pola.rs][5], [Docs.rs][1])
* If N\_train grows large or you're serving many tickers, use `hnsw_rs` for ANN sub-linear queries. ([Crates.io][6])
* Python HMM inference service is lightweight: model-level forward filtering or `predict_proba` on a 3-element observation is cheap. Use Uvicorn + Gunicorn for concurrency.
* Keep I/O efficient: use Parquet + Arrow; avoid repeatedly loading whole datasets every bar.

---

# Security, CI/CD & Deployment

* Containerize services (Docker): `rust-inference:prod:v1`, `python-hmm:prod:v1`.
* CI pipeline: build & test Rust crates (clippy, cargo test), run Python unit tests + HMM validation notebooks, build Docker images, push to registry.
* Use Git tags for model & code versioning; model artifacts reference commit SHA.
* Deploy with Kubernetes (or Docker Compose for MVP). Expose health-checks and read-only endpoints for metadata.

---

# Rollout plan — granular milestones

## Sprint 0 — repo & infra (1 week)

* Create mono-repo two-workspaces: `rust/` and `py/`.
* Stand up local MinIO (S3-compatible), start storing raw CSV → Parquet.
* Add CI skeleton.

## Sprint 1 — MVP feature + MR + TSMOM in Rust (2 weeks)

* Implement Rust Polars pipeline to read CSV/Parquet and compute RSI, MA, std, zscore, momentum.
* Implement MR + TSMOM scorers and output per-bar components.
* Unit tests & small dataset run.

**Deliverable**: Rust binary that outputs `s_MR` & `s_TSMOM` JSON per bar.

## Sprint 2 — LDC engine in Rust (2–3 weeks)

* Implement ring-buffer labeled store, Lorentzian distance kernel, parallel k-NN query, simple weighted voting to produce `s_LDC`.
* Add small CLI to build training snapshots (labels by horizon h) and test classifier on historical test windows.
* Optionally implement HNSW switch for large N.

**Deliverable**: `rust-inference` that returns `s_LDC`, `s_MR`, `s_TSMOM`.

## Sprint 3 — Python research & HMM prototyping (2–3 weeks, in parallel)

* In Python notebook, implement HMM training (pomegranate / hmmlearn) on the observation `[s_LDC,s_MR,s_TSMOM]`.
* Train HMM for M=2..4 and derive per-state fusion weights (optimize per-state Sharpe).
* Save artifact to MinIO.

**Deliverable**: working `train_hmm.py` that writes `hmm_v1.json` & `weights_v1.json`.

## Sprint 4 — HMM microservice + integration (1–2 weeks)

* Build Python HMM microservice (FastAPI) that loads artifact, serves `POST /filter` to return `p_states` or `w_t`.
* Integrate in Rust inference: call HMM service per bar (or per N bars) to get `w_t`. Add caching and dwell-time smoothing.
* Implement fusion `S_t` and signal emission to Redis.

**Deliverable**: end-to-end signal emitter with HMM-weighting.

## Sprint 5 — Backtesting, walk-forward & validation (2 weeks)

* Implement Python backtest harness (vectorbt) to validate full pipeline using exported signals and offline simulation.
* Implement walk-forward retrain + selection of HMMs; compare static vs HMM fusion.

**Deliverable**: OOS metrics and backtest report.

## Sprint 6 — Hardening, port HMM filter to Rust (optional, 3–4 weeks)

* If needed for latency or dependency reduction, port HMM filtering (forward probabilities) into Rust and replace Python online service.
* Keep Python for heavy training only.

**Deliverable**: single-process Rust inference with embedded HMM inference.

---

# When to use Python vs Rust (rules of thumb)

* Use **Rust** for high-throughput, deterministic, latency-sensitive code: feature computation (Polars), k-NN Lorentzian kernel, ANN HNSW, production signal emitter. Polars in Rust is a first-class choice. ([Docs.rs][1], [Polars][2])
* Use **Python** for rapid research, model training, numerical experiments, HMM training/visualization, vectorbt walk-forwards and plotting. HMM libraries (`pomegranate`, `hmmlearn`) make regime modeling straightforward. ([pomegranate.readthedocs.io][4], [hmmlearn.readthedocs.io][3])
* Bridge: use artifact store + small RPC (FastAPI/gRPC) or PyO3/maturin bindings if you prefer embedding Python in Rust or vice-versa (PyO3 + maturin makes packaging Rust extensions for Python easy). ([pyo3.rs][8], [Reddit][9])

---

# Failure modes & safeguards

* **Data drift**: log feature distributions; set retrain triggers when KS distance exceeds threshold.
* **Degenerate HMM covariances**: regularize Σ with `λI` and fallback to static weights if degeneracy detected.
* **LDC overfitting**: limit train history, purge overlaps, use rolling windows and validate with walk-forward.
* **RPC failure**: if Python HMM service fails, Rust should fallback to latest persisted `weights_vX.json`.
* **Runtime cost leak**: cap per-bar compute time; batch heavy tasks (rebuilding HNSW) off the critical path.

---

# Quick resource pointers (useful docs)

* Polars (Rust) docs: fast lazy API & Arrow-backed DataFrames. ([Docs.rs][1], [docs.pola.rs][5])
* PyO3 / maturin: packaging Rust code for Python or embedding Python in Rust. ([pyo3.rs][8], [Reddit][9])
* ANN in Rust: `hnsw_rs` / rust-cv/hnsw for scalable approximate NN. ([Crates.io][6], [GitHub][7])
* HMM libraries in Python: `pomegranate` (flexible & fast) and `hmmlearn` (scikit-like API) for Gaussian HMMs. ([pomegranate.readthedocs.io][4], [hmmlearn.readthedocs.io][3])
* Rust ML ecosystem: `linfa`, `smartcore` for classical ML algorithms (useful if you want to port more modeling into Rust). ([GitHub][10])

---

# Final notes / recommended next action

1. **Start MVP**: implement Rust Polars feature pipeline + MR/TSMOM + LDC brute force for a single symbol & 5m bars. Get deterministic signals. (Sprint 1–2)
2. **Parallel research**: in Python, prototype HMM over the exported Rust signals and derive state-conditioned weights. (Sprint 3)
3. **Integrate**: expose the HMM online via FastAPI and have Rust call it for weight vectors; finalize fusion & backtests. (Sprint 4–5)
4. **Decide porting**: if HMM proves valuable and latency/ops require it, port the inference to Rust. (Sprint 6)

[1]: https://docs.rs/polars/latest/polars/?utm_source=chatgpt.com "polars - Rust"
[2]: https://pola.rs/?utm_source=chatgpt.com "Polars — DataFrames for the new era"
[3]: https://hmmlearn.readthedocs.io/en/latest/tutorial.html?utm_source=chatgpt.com "Tutorial — hmmlearn 0.3.3.post1+ge01a10e documentation"
[4]: https://pomegranate.readthedocs.io/en/latest/tutorials/B_Model_Tutorial_4_Hidden_Markov_Models.html?utm_source=chatgpt.com "Hidden Markov Models — pomegranate 1.0.0 documentation"
[5]: https://docs.pola.rs/api/rust/dev/polars_lazy/index.html?utm_source=chatgpt.com "Crate polars_lazy - Rust"
[6]: https://crates.io/crates/hnsw_rs?utm_source=chatgpt.com "hnsw_rs - crates.io: Rust Package Registry"
[7]: https://github.com/rust-cv/hnsw?utm_source=chatgpt.com "rust-cv/hnsw"
[8]: https://pyo3.rs/?utm_source=chatgpt.com "Introduction - PyO3 user guide"
[9]: https://www.reddit.com/r/Python/comments/125q9vo/writing_python_extensions_never_been_easier_with/?utm_source=chatgpt.com "Writing python extensions never been easier… with Rust ..."
[10]: https://github.com/rust-ml/linfa?utm_source=chatgpt.com "rust-ml/linfa: A Rust machine learning framework."
