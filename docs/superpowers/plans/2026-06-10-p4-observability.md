# P4 Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire structured JSON logging and a Prometheus `/metrics` endpoint into the inference-engine daemon so production deployments can be observed without custom tooling.

**Architecture:** Add `--log-format` flag (text/json) by enabling the `json` feature on `tracing-subscriber`. Introduce a `RuntimeMetrics` struct in `inference-engine` that wraps four prometheus counters/histograms using a private `Registry`. Extend the daemon's raw-TCP health check server to parse the HTTP request path and route `/metrics` → Prometheus text export, `/health` → `{"status":"ok"}`. Wire metrics into the daemon's pipeline loop.

**Tech Stack:** Rust, tracing-subscriber 0.3 (json feature), prometheus 0.13 (already in workspace deps), tokio (already in workspace deps).

---

## Subsystem note

P4 has four independent chunks. This plan covers **Observability** (logging + metrics) only. The other three should each get their own plan:
- **Non-mock Redis/Kafka integration test validation** — verify `rust/signal-fusion/tests/redis_integration_tests.rs` + `kafka_integration_tests.rs` actually compile and pass against real containers.
- **Config hot-reload** — add `POST /reload` to the health check server; daemon re-reads TOML on next run.
- **Container orchestration** — multi-stage Dockerfile + docker-compose service entry for the Rust runtime.

---

## File map

| Action | Path |
|--------|------|
| Modify | `rust/Cargo.toml` |
| Modify | `rust/inference-engine/src/main.rs` |
| Create | `rust/inference-engine/src/metrics.rs` |
| Modify | `rust/inference-engine/src/daemon.rs` |

---

### Task 1: Enable `--log-format json` CLI flag

**Files:**
- Modify: `rust/Cargo.toml` — add `json` feature to `tracing-subscriber`
- Modify: `rust/inference-engine/src/main.rs` — add `log_format` arg, update `init_tracing`

- [ ] **Step 1: Write the failing tests**

Add inside the existing `#[cfg(test)]` block at the bottom of `rust/inference-engine/src/main.rs`:

```rust
#[cfg(test)]
mod log_format_tests {
    use super::*;

    #[test]
    fn test_log_format_arg_defaults_to_text() {
        let args = Args::try_parse_from(["inference-engine"]).unwrap();
        assert_eq!(args.log_format, "text");
    }

    #[test]
    fn test_log_format_arg_accepts_json() {
        let args = Args::try_parse_from(["inference-engine", "--log-format", "json"]).unwrap();
        assert_eq!(args.log_format, "json");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd rust
cargo test -p inference-engine log_format_tests -- --nocapture
```

Expected output: compile error — `Args` has no field `log_format`.

- [ ] **Step 3: Implement the flag and update `init_tracing`**

In `rust/Cargo.toml`, replace:
```toml
tracing-subscriber = "0.3"
```
with:
```toml
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter", "json"] }
```

In `rust/inference-engine/src/main.rs`, add `log_format` to the `Args` struct after the `log_level` field:
```rust
/// Log output format.  "text" (default) is human-readable; "json" emits one JSON object per
/// line, suitable for structured log aggregators (Loki, Datadog, CloudWatch, etc.).
#[arg(long, default_value = "text")]
log_format: String,
```

Replace the `init_tracing` function:
```rust
fn init_tracing(level: &str, format: &str) -> Result<()> {
    let log_level = match level {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };
    if format == "json" {
        tracing_subscriber::fmt().json().with_max_level(log_level).init();
    } else {
        tracing_subscriber::fmt().with_max_level(log_level).init();
    }
    Ok(())
}
```

Update the `init_tracing` call in `main()` (line ~99):
```rust
init_tracing(&args.log_level, &args.log_format)?;
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd rust
cargo test -p inference-engine log_format_tests -- --nocapture
```

Expected: PASS (2 tests). Also verify full build:
```bash
cargo build -p inference-engine
```

- [ ] **Step 5: Commit**

```bash
git add rust/Cargo.toml rust/inference-engine/src/main.rs
git commit -m "feat(inference-engine): add --log-format json flag for structured logging"
```

---

### Task 2: Create `RuntimeMetrics` struct

**Files:**
- Create: `rust/inference-engine/src/metrics.rs`
- Modify: `rust/inference-engine/src/main.rs` — add `mod metrics;`

- [ ] **Step 1: Write the failing tests**

Create `rust/inference-engine/src/metrics.rs` with tests only (no implementation yet):

```rust
// rust/inference-engine/src/metrics.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_counters_are_zero() {
        let m = RuntimeMetrics::new().unwrap();
        let text = m.gather_text();
        assert!(text.contains("imp_pipeline_runs_total 0"));
        assert!(text.contains("imp_pipeline_errors_total 0"));
        assert!(text.contains("imp_pipeline_rows_total 0"));
    }

    #[test]
    fn test_record_run_success_increments_counters() {
        let m = RuntimeMetrics::new().unwrap();
        m.record_run_success(42, 1.5);
        let text = m.gather_text();
        assert!(text.contains("imp_pipeline_runs_total 1"));
        assert!(text.contains("imp_pipeline_rows_total 42"));
    }

    #[test]
    fn test_record_run_error_increments_error_counter() {
        let m = RuntimeMetrics::new().unwrap();
        m.record_run_error();
        let text = m.gather_text();
        assert!(text.contains("imp_pipeline_errors_total 1"));
    }

    #[test]
    fn test_gather_text_is_valid_prometheus_format() {
        let m = RuntimeMetrics::new().unwrap();
        let text = m.gather_text();
        assert!(text.contains("# HELP imp_pipeline_runs_total"));
        assert!(text.contains("# TYPE imp_pipeline_runs_total counter"));
        assert!(text.contains("# HELP imp_pipeline_duration_seconds"));
        assert!(text.contains("# TYPE imp_pipeline_duration_seconds histogram"));
    }
}
```

Add `mod metrics;` to the top of `rust/inference-engine/src/main.rs` alongside the other `mod` declarations.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd rust
cargo test -p inference-engine test_initial_counters -- --nocapture
```

Expected: compile error — `RuntimeMetrics` not defined.

- [ ] **Step 3: Implement `RuntimeMetrics`**

Add the full implementation to `rust/inference-engine/src/metrics.rs` above the `#[cfg(test)]` block:

```rust
use anyhow::Result;
use prometheus::{Histogram, HistogramOpts, IntCounter, Registry, TextEncoder};
use std::sync::Arc;

#[derive(Clone)]
pub struct RuntimeMetrics {
    pub pipeline_runs: IntCounter,
    pub pipeline_errors: IntCounter,
    pub pipeline_rows: IntCounter,
    pub pipeline_duration_seconds: Histogram,
    registry: Arc<Registry>,
}

impl RuntimeMetrics {
    pub fn new() -> Result<Self> {
        let registry = Arc::new(Registry::new());

        let pipeline_runs = IntCounter::new(
            "imp_pipeline_runs_total",
            "Total pipeline runs completed successfully",
        )?;
        let pipeline_errors = IntCounter::new(
            "imp_pipeline_errors_total",
            "Total pipeline run errors",
        )?;
        let pipeline_rows = IntCounter::new(
            "imp_pipeline_rows_total",
            "Total OHLCV rows processed across all runs",
        )?;
        let pipeline_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "imp_pipeline_duration_seconds",
                "Pipeline run duration in seconds",
            )
            .buckets(vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]),
        )?;

        registry.register(Box::new(pipeline_runs.clone()))?;
        registry.register(Box::new(pipeline_errors.clone()))?;
        registry.register(Box::new(pipeline_rows.clone()))?;
        registry.register(Box::new(pipeline_duration_seconds.clone()))?;

        Ok(Self {
            pipeline_runs,
            pipeline_errors,
            pipeline_rows,
            pipeline_duration_seconds,
            registry,
        })
    }

    /// Record a successful run. `rows` is the number of OHLCV rows processed;
    /// `duration_seconds` is the wall-clock elapsed time for the run.
    pub fn record_run_success(&self, rows: u64, duration_seconds: f64) {
        self.pipeline_runs.inc();
        self.pipeline_rows.inc_by(rows);
        self.pipeline_duration_seconds.observe(duration_seconds);
    }

    pub fn record_run_error(&self) {
        self.pipeline_errors.inc();
    }

    /// Return Prometheus text-format exposition of all registered metrics.
    pub fn gather_text(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode_to_string(&metric_families).unwrap_or_default()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd rust
cargo test -p inference-engine -- --nocapture
```

Expected: all existing tests pass plus the 4 new metrics tests.

- [ ] **Step 5: Commit**

```bash
git add rust/inference-engine/src/metrics.rs rust/inference-engine/src/main.rs
git commit -m "feat(inference-engine): add RuntimeMetrics struct with prometheus counters and text export"
```

---

### Task 3: Wire `RuntimeMetrics` into the daemon pipeline loop

**Files:**
- Modify: `rust/inference-engine/src/daemon.rs` — `run_daemon` signature takes `Arc<RuntimeMetrics>`
- Modify: `rust/inference-engine/src/main.rs` — create metrics instance, pass to `run_daemon`

- [ ] **Step 1: Write the failing test**

Add inside `daemon.rs` at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::RuntimeMetrics;
    use std::sync::Arc;

    #[test]
    fn test_run_daemon_accepts_metrics_param() {
        // Compile-check: ensure run_daemon's new signature is callable.
        let metrics = Arc::new(RuntimeMetrics::new().unwrap());
        let _config = DaemonConfig {
            port: 9090,
            interval: std::time::Duration::from_secs(60),
        };
        // We don't call run_daemon (it blocks), just verify the types compose.
        assert_eq!(metrics.pipeline_runs.get(), 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd rust
cargo test -p inference-engine test_run_daemon_accepts -- --nocapture
```

Expected: compile error — `crate::metrics` not in scope in daemon.rs (metrics module not imported there yet).

- [ ] **Step 3: Update `run_daemon` signature and loop**

In `rust/inference-engine/src/daemon.rs`, add to imports:
```rust
use crate::metrics::RuntimeMetrics;
use std::sync::Arc;
```

Update `run_daemon` signature:
```rust
pub async fn run_daemon(
    config_path: PathBuf,
    daemon_config: DaemonConfig,
    metrics: Arc<RuntimeMetrics>,
) -> Result<()> {
```

Inside `run_daemon`, after the health check server spawn (to pass metrics to it), add:
```rust
let health_metrics = metrics.clone();
tokio::spawn(async move {
    if let Err(e) = serve_health_check(daemon_config.port, health_rx, health_metrics).await {
        error!(error = %e, "health check server terminated with error");
    }
});
```

Inside the daemon loop, replace the `match run_single_pipeline(...)` block:
```rust
match run_single_pipeline(&config_path).await {
    Ok(summary) => {
        metrics.record_run_success(
            summary.output_rows as u64,
            run_start.elapsed().as_secs_f64(),
        );
        info!(
            run = run_count,
            output_rows = summary.output_rows,
            fused_rows = summary.fused_rows,
            fallback_rows = summary.fallback_rows,
            elapsed_ms = run_start.elapsed().as_millis() as u64,
            "pipeline run completed"
        );
    }
    Err(e) => {
        metrics.record_run_error();
        error!(run = run_count, error = %e, "pipeline run failed");
    }
}
```

In `rust/inference-engine/src/main.rs`, in the `Command::Serve` match arm, update the `run_daemon` call:
```rust
Command::Serve { port, interval, config } => {
    let daemon_config = DaemonConfig {
        port,
        interval: Duration::from_secs(interval),
    };
    let metrics = Arc::new(crate::metrics::RuntimeMetrics::new()?);
    daemon::run_daemon(PathBuf::from(&config), daemon_config, metrics).await?;
    Ok(())
}
```

Add `use std::sync::Arc;` to the imports at the top of `main.rs` if not already present.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd rust
cargo test -p inference-engine -- --nocapture
cargo build -p inference-engine
```

Expected: all tests pass, binary builds without errors.

- [ ] **Step 5: Commit**

```bash
git add rust/inference-engine/src/daemon.rs rust/inference-engine/src/main.rs
git commit -m "feat(inference-engine): wire RuntimeMetrics into daemon pipeline loop"
```

---

### Task 4: Add `/metrics` HTTP route to daemon health check server

**Files:**
- Modify: `rust/inference-engine/src/daemon.rs` — `serve_health_check` takes `Arc<RuntimeMetrics>`, parses HTTP path

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `daemon.rs`:

```rust
#[tokio::test]
async fn test_health_endpoint_returns_ok_json() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;
    use std::sync::Arc;
    use crate::metrics::RuntimeMetrics;

    let metrics = Arc::new(RuntimeMetrics::new().unwrap());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    tokio::spawn(async move {
        serve_health_check(19201, shutdown_rx, metrics).await.ok();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let mut stream = TcpStream::connect("127.0.0.1:19201").await.unwrap();
    stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").await.unwrap();
    let mut buf = vec![0u8; 1024];
    let n = stream.read(&mut buf).await.unwrap();
    let response = std::str::from_utf8(&buf[..n]).unwrap();

    assert!(response.contains("200 OK"), "expected 200, got: {}", response);
    assert!(response.contains(r#"{"status":"ok"}"#));

    shutdown_tx.send(true).ok();
}

#[tokio::test]
async fn test_metrics_endpoint_returns_prometheus_text() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;
    use std::sync::Arc;
    use crate::metrics::RuntimeMetrics;

    let metrics = Arc::new(RuntimeMetrics::new().unwrap());
    metrics.record_run_success(100, 2.0);

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let metrics_clone = metrics.clone();

    tokio::spawn(async move {
        serve_health_check(19202, shutdown_rx, metrics_clone).await.ok();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let mut stream = TcpStream::connect("127.0.0.1:19202").await.unwrap();
    stream.write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n").await.unwrap();
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await.unwrap();
    let response = std::str::from_utf8(&buf[..n]).unwrap();

    assert!(response.contains("200 OK"), "expected 200, got: {}", response);
    assert!(response.contains("imp_pipeline_runs_total 1"));
    assert!(response.contains("imp_pipeline_rows_total 100"));
    assert!(response.contains("# HELP imp_pipeline_runs_total"));

    shutdown_tx.send(true).ok();
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd rust
cargo test -p inference-engine test_health_endpoint test_metrics_endpoint -- --nocapture
```

Expected: compile error — `serve_health_check` still has the old signature (no `metrics` param).

- [ ] **Step 3: Update `serve_health_check`**

Replace the current `serve_health_check` function in `daemon.rs`:

```rust
async fn serve_health_check(
    port: u16,
    mut shutdown_rx: watch::Receiver<bool>,
    metrics: Arc<RuntimeMetrics>,
) -> Result<()> {
    let bind_addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&bind_addr).await.map_err(|e| {
        anyhow::anyhow!("failed to bind health check on {}: {}", bind_addr, e)
    })?;

    info!(addr = %bind_addr, "health check server listening");

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((mut stream, peer_addr)) => {
                        let mut buf = [0u8; 1024];
                        let n = stream.read(&mut buf).await.unwrap_or(0);
                        let request = std::str::from_utf8(&buf[..n]).unwrap_or("");

                        // Parse path from request line: "GET /metrics HTTP/1.1"
                        let path = request.lines().next().and_then(|line| {
                            let mut parts = line.split_whitespace();
                            parts.next(); // skip method
                            parts.next() // path
                        }).unwrap_or("/health");

                        let (body, content_type) = match path {
                            "/metrics" => (
                                metrics.gather_text(),
                                "text/plain; version=0.0.4; charset=utf-8",
                            ),
                            _ => (
                                r#"{"status":"ok"}"#.to_string(),
                                "application/json",
                            ),
                        };

                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            content_type,
                            body.len(),
                            body,
                        );
                        if let Err(e) = stream.write_all(response.as_bytes()).await {
                            warn!(peer = %peer_addr, error = %e, "failed to write response");
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "health check accept error");
                    }
                }
            }
            _ = shutdown_rx.changed() => {
                info!("shutdown signal received, stopping health check server");
                break;
            }
        }
    }

    Ok(())
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd rust
cargo test -p inference-engine -- --nocapture
```

Expected: all tests pass including the 2 new routing tests. Verify binary still builds and smoke still works:

```bash
cargo run -p inference-engine -- smoke --config inference-engine/fixtures/local-smoke.toml
```

Expected: `smoke: output matches expected fixture` (byte-for-byte comparison passes).

- [ ] **Step 5: Commit**

```bash
git add rust/inference-engine/src/daemon.rs
git commit -m "feat(inference-engine): add GET /metrics Prometheus endpoint to daemon health server"
```

---

## Self-review

**Spec coverage against P4 observability goals:**

| P4 item | Covered? |
|---------|---------|
| Prometheus metrics integration | Yes — Tasks 2–4 implement `/metrics` endpoint |
| Structured logging for production | Yes — Task 1 adds `--log-format json` |
| Container orchestration | No — separate plan |
| Config hot-reload | No — separate plan |
| Non-mock integration path validation | No — separate plan |

**Placeholder scan:** No TBD, TODO, or "implement later" entries found.

**Type consistency check:**
- `RuntimeMetrics::new() -> Result<Self>` — used in Tasks 2, 3, 4 consistently.
- `record_run_success(rows: u64, duration_seconds: f64)` — called in Task 3 with `summary.output_rows as u64` and `run_start.elapsed().as_secs_f64()`.
- `serve_health_check(port: u16, shutdown_rx: watch::Receiver<bool>, metrics: Arc<RuntimeMetrics>)` — updated in Task 3 spawn call and tested in Task 4.
- `run_daemon` takes `Arc<RuntimeMetrics>` — introduced in Task 3, called from `main.rs` in Task 3.
