//! Daemon (long-running) mode for the inference engine.
//!
//! Provides:
//! - A `--serve` loop that runs the pipeline at a configurable interval
//! - A health check HTTP endpoint (GET /health returns `{"status":"ok"}`)
//! - Graceful shutdown on SIGINT/SIGTERM via tokio signal handling

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::config::RuntimeConfig;
use crate::runtime::InferenceRuntime;
use crate::schema::RuntimeRunSummary;

/// Configuration for daemon (long-running) mode.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// TCP port for the health check HTTP endpoint.
    pub port: u16,
    /// Interval between successive pipeline runs.
    pub interval: Duration,
}

/// Run the inference engine as a long-running daemon process.
///
/// 1. Starts a minimal health check HTTP server on the configured port.
/// 2. Registers signal handlers (SIGINT / SIGTERM) for graceful shutdown.
/// 3. Runs the full pipeline in a loop, waiting `interval` between runs.
/// 4. On a shutdown signal, completes the current run (or wait) and exits cleanly.
pub async fn run_daemon(config_path: PathBuf, daemon_config: DaemonConfig) -> Result<()> {
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

    // Signal handler owns a clone of the sender
    let signal_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        handle_shutdown_signals(signal_tx).await;
    });

    // Health check server gets its own receiver
    let health_rx = shutdown_rx.clone();
    tokio::spawn(async move {
        if let Err(e) = serve_health_check(daemon_config.port, health_rx).await {
            error!(error = %e, "health check server terminated with error");
        }
    });

    let daemon_start = Instant::now();
    let mut run_count: u64 = 0;

    info!(
        port = daemon_config.port,
        interval_s = daemon_config.interval.as_secs(),
        config = %config_path.display(),
        "daemon mode started"
    );

    loop {
        // Fast check — has a shutdown already been requested?
        if *shutdown_rx.borrow() {
            info!("shutdown signal received, stopping daemon loop");
            break;
        }

        run_count += 1;
        let run_start = Instant::now();
        info!(run = run_count, "starting pipeline run");

        match run_single_pipeline(&config_path).await {
            Ok(summary) => {
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
                error!(run = run_count, error = %e, "pipeline run failed");
            }
        }

        // Calculate remaining time before the next run
        let elapsed = run_start.elapsed();
        let remaining = if elapsed < daemon_config.interval {
            daemon_config.interval - elapsed
        } else {
            Duration::ZERO
        };

        if remaining > Duration::ZERO {
            tokio::select! {
                _ = tokio::time::sleep(remaining) => {}
                _ = shutdown_rx.changed() => {
                    info!("shutdown signal received during sleep interval");
                    break;
                }
            }
        }
    }

    let uptime = daemon_start.elapsed();
    info!(
        runs_completed = run_count,
        uptime_s = uptime.as_secs(),
        "daemon shut down gracefully"
    );

    Ok(())
}

/// Bootstrap, run, and shut down a single pipeline pass.
async fn run_single_pipeline(config_path: &std::path::Path) -> Result<RuntimeRunSummary> {
    let config = RuntimeConfig::load(config_path)?;
    let mut runtime = InferenceRuntime::bootstrap(config).await?;
    let summary = runtime.run().await?;
    runtime.shutdown().await?;
    Ok(summary)
}

/// Serve a minimal health check HTTP endpoint on `port`.
///
/// Responds to any HTTP request with:
/// ```json
/// {"status":"ok"}
/// ```
/// and a 200 status code.
async fn serve_health_check(port: u16, mut shutdown_rx: watch::Receiver<bool>) -> Result<()> {
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
                        // Drain the request (we don't care about its content)
                        let mut buf = [0u8; 1024];
                        let _ = stream.read(&mut buf).await;

                        let body = r#"{"status":"ok"}"#;
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body,
                        );
                        if let Err(e) = stream.write_all(response.as_bytes()).await {
                            warn!(peer = %peer_addr, error = %e, "failed to write health check response");
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

/// Block until a shutdown signal (SIGINT or SIGTERM) is received, then notify
/// the daemon main loop via the watch channel.
///
/// - SIGINT (Ctrl+C) is handled on all platforms.
/// - SIGTERM is handled on Unix (Linux/macOS) for container/process-manager
///   graceful shutdown where SIGTERM is sent before SIGKILL.
async fn handle_shutdown_signals(shutdown_tx: watch::Sender<bool>) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                error!(error = %e, "failed to register SIGTERM handler");
                // Fall back to SIGINT-only
                let _ = tokio::signal::ctrl_c().await;
                info!("received shutdown signal");
                let _ = shutdown_tx.send(true);
                return;
            }
        };

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("received SIGINT (Ctrl+C), initiating graceful shutdown...");
            }
            _ = sigterm.recv() => {
                info!("received SIGTERM, initiating graceful shutdown...");
            }
        }
    }

    #[cfg(not(unix))]
    {
        if let Err(e) = tokio::signal::ctrl_c().await {
            error!(error = %e, "failed to register SIGINT handler");
        }
        info!("received shutdown signal, notifying daemon loop");
    }

    let _ = shutdown_tx.send(true);
}
