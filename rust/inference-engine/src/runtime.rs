use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use feature_pipeline::{FeaturePipeline, Features, OHLCV, Signals};
use ldc_engine::{Direction, LDCEngine, LDCPrediction};
use polars::prelude::*;
use sha2::{Digest, Sha256};
use signal_fusion::{SignalComponents, SignalEmitter, SignalFusion, SignalSide};
use tracing::info;

use crate::config::{EmissionFailureBehavior, InvalidFeatureBehavior, RuntimeConfig};
use crate::hmm::HmmWeightResolver;
use crate::schema::{
    AuditMetadata, EmissionMetadata, FusedOutput, HmmMetadata, HmmWeightSource, IntermediateSignals,
    RecordTiming, RuntimeOutputRecord, RuntimeRunSummary, SignalObservation, VersionMetadata,
    CANONICAL_SCHEMA_VERSION,
};

pub struct InferenceRuntime {
    config: RuntimeConfig,
    feature_pipeline: FeaturePipeline,
    ldc_engine: LDCEngine,
    hmm_resolver: HmmWeightResolver,
    fusion_engine: SignalFusion,
    emitter: Option<SignalEmitter>,
    run_id: String,
}

impl InferenceRuntime {
    pub async fn bootstrap(config: RuntimeConfig) -> Result<Self> {
        let feature_pipeline = FeaturePipeline::with_periods(
            config.features.window_size,
            config.features.rsi_period,
            config.features.ma_period,
        );
        let ldc_engine = LDCEngine::with_config(config.ldc.engine.clone());
        let hmm_resolver = HmmWeightResolver::from_config(&config)?;
        let fusion_engine = config.build_signal_fusion();
        let emitter = if config.emission.enabled {
            let mut emitter_config = config.build_emitter_config();
            emitter_config.fail_on_emission_error = true;
            Some(SignalEmitter::new(emitter_config).await?)
        } else {
            None
        };

        let run_id = deterministic_sha256(&serde_json::json!({
            "mode": config.runtime.mode.to_string(),
            "symbol": &config.runtime.symbol,
            "interval": &config.runtime.interval,
            "input": config.input.market_data.to_string_lossy(),
            "threshold": config.fusion.threshold,
        }))?;

        info!(
            mode = %config.runtime.mode,
            symbol = %config.runtime.symbol,
            input = %config.input.market_data.display(),
            "runtime bootstrapped"
        );

        Ok(Self {
            config,
            feature_pipeline,
            ldc_engine,
            hmm_resolver,
            fusion_engine,
            emitter,
            run_id,
        })
    }

    pub async fn run(&mut self) -> Result<RuntimeRunSummary> {
        self.prepare_output_paths()?;

        let market_data = self.load_market_data()?;
        self.feature_pipeline
            .validate_ohlcv_data(&market_data)
            .context("market input validation failed")?;

        let features = self
            .feature_pipeline
            .compute_features_safe(&market_data)
            .context("feature computation failed")?;

        let mr_signals = self
            .feature_pipeline
            .generate_mr_signal(&features)
            .context("MR signal generation failed")?;
        let tsmom_signals = self
            .feature_pipeline
            .generate_tsmom_signal(&features)
            .context("TSMOM signal generation failed")?;

        if self.config.features.persist_feature_parquet {
            if let Some(path) = &self.config.output.feature_parquet {
                write_features_parquet(&features, path)?;
            }
        }

        let output_temp_path = partial_path(&self.config.output.canonical_jsonl);
        let summary_temp_path = partial_path(&self.config.output.summary_json);
        let output_file = File::create(&output_temp_path)
            .with_context(|| format!("failed to create {}", output_temp_path.display()))?;
        let mut writer = BufWriter::new(output_file);
        let mut hasher = Sha256::new();

        let input_rows = market_data.len();
        let warmup_bars = self.config.warmup_bars();
        let mut output_rows = 0usize;
        let mut fused_rows = 0usize;
        let mut emitted_rows = 0usize;
        let mut fallback_rows = 0usize;
        let mut cache_hits = 0usize;
        let mut first_timestamp = None;
        let mut last_timestamp = None;

        for index in 0..market_data.len() {
            self.maybe_add_training_sample(index, &market_data, &features, warmup_bars)?;

            let signal_start = Instant::now();
            let ohlcv = market_data[index].clone();
            let feature_row = features[index].clone();
            let mr_signal = &mr_signals[index];
            let tsmom_signal = &tsmom_signals[index];

            let mr_observation = build_mr_observation(mr_signal, index, warmup_bars);
            let tsmom_observation = build_tsmom_observation(tsmom_signal, index, warmup_bars);
            let ldc_observation = self.build_ldc_observation(index, &feature_row, warmup_bars)?;

            let input_checksum = deterministic_sha256(&serde_json::json!({
                "ohlcv": &ohlcv,
                "features": &feature_row,
            }))?;
            let correlation_id = format!(
                "{}-{}-{:05}",
                self.config.runtime.symbol,
                market_data[index].timestamp,
                index
            );

            let mut hmm_metadata = HmmMetadata {
                source: HmmWeightSource::NotInvoked,
                model_version: None,
                cache_key: None,
                cache_hit: false,
                state_probabilities: None,
                most_likely_state: None,
                confidence: None,
                weights: None,
                request_error: None,
                fallback_reason: None,
                service_url: None,
                service_latency_ms: None,
            };
            let mut fused_output = FusedOutput {
                eligible_for_fusion: false,
                fused_score: 0.0,
                confidence: 0.0,
                threshold: self.config.fusion.threshold,
                cooldown_active: false,
                recommended_side: SignalSide::Hold,
                actionable_side: SignalSide::Hold,
                signal_generated: false,
                suppression_reason: Some("missing_signal_components".to_string()),
                weights_used: None,
            };
            let mut emission = EmissionMetadata {
                configured: self.emitter.is_some(),
                backend: format!("{:?}", self.config.emission.backend),
                attempted: false,
                emitted: false,
                error: None,
            };

            let feature_checksum = deterministic_sha256(&serde_json::json!({
                "input_checksum": input_checksum,
                "ldc": ldc_observation.normalized,
                "mr": mr_observation.normalized,
                "tsmom": tsmom_observation.normalized,
            }))?;

            let fusion_inputs = if let (Some(s_ldc), Some(s_mr), Some(s_tsmom)) = (
                ldc_observation.normalized,
                mr_observation.normalized,
                tsmom_observation.normalized,
            ) {
                Some(SignalComponents {
                    s_ldc,
                    s_mr,
                    s_tsmom,
                })
            } else {
                None
            };

            let mut hmm_latency_ms = None;
            let mut emission_latency_ms = None;

            if let Some(components) = fusion_inputs.clone() {
                fused_rows += 1;
                let hmm_start = Instant::now();
                let resolved = self
                    .hmm_resolver
                    .resolve(&components, ohlcv.timestamp)
                    .await
                    .with_context(|| format!("failed to resolve HMM weights for bar {}", index))?;
                hmm_latency_ms = self
                    .config
                    .runtime
                    .capture_wall_clock_timing
                    .then_some(hmm_start.elapsed().as_millis() as u64);
                if resolved.metadata.source == HmmWeightSource::StaticFallback {
                    fallback_rows += 1;
                }
                if resolved.metadata.cache_hit {
                    cache_hits += 1;
                }
                hmm_metadata = resolved.metadata.clone();
                let normalized_weights = if self.config.fusion.normalize_weights {
                    resolved.weights.normalize()
                } else {
                    resolved.weights.clone()
                };
                let fused_score = components.s_ldc * normalized_weights.w_ldc
                    + components.s_mr * normalized_weights.w_mr
                    + components.s_tsmom * normalized_weights.w_tsmom;
                let recommended_side = if fused_score > 0.0 {
                    SignalSide::Buy
                } else if fused_score < 0.0 {
                    SignalSide::Sell
                } else {
                    SignalSide::Hold
                };
                let confidence = fused_score.abs().min(1.0);
                let cooldown_active = self.fusion_engine.is_in_cooldown(ohlcv.timestamp);
                let below_threshold = fused_score.abs() < self.config.fusion.threshold;

                let maybe_signal = self.fusion_engine.fuse_signals_sync(
                    components.clone(),
                    resolved.weights.clone(),
                    ohlcv.timestamp,
                    &self.config.runtime.symbol,
                    &resolved.model_version,
                    correlation_id.clone(),
                    feature_checksum.clone(),
                    signal_start.elapsed().as_millis() as u64,
                )?;

                let suppression_reason = if maybe_signal.is_none() {
                    if below_threshold {
                        Some("below_threshold".to_string())
                    } else if cooldown_active {
                        Some("cooldown_active".to_string())
                    } else {
                        Some("suppressed".to_string())
                    }
                } else {
                    None
                };

                let actionable_side = maybe_signal
                    .as_ref()
                    .map(|signal| signal.side.clone())
                    .unwrap_or(SignalSide::Hold);
                let signal_generated = maybe_signal.is_some();

                if let Some(mut signal) = maybe_signal {
                    if let Some(state_probabilities) = resolved.metadata.state_probabilities.clone() {
                        let fallback_used = resolved.metadata.source == HmmWeightSource::StaticFallback;
                        signal = signal.with_hmm_data(state_probabilities, fallback_used);
                    }

                    if let Some(emitter) = &self.emitter {
                        let emission_start = Instant::now();
                        emission.attempted = true;
                        match emitter.emit_signal(signal.clone()).await {
                            Ok(emitted) => {
                                emission.emitted = emitted;
                                if emitted {
                                    emitted_rows += 1;
                                }
                            }
                            Err(error) => {
                                emission.error = Some(error.to_string());
                                if self.config.failure.emission_failure == EmissionFailureBehavior::FailRun {
                                    return Err(anyhow::anyhow!(error.to_string()));
                                }
                            }
                        }
                        emission_latency_ms = self
                            .config
                            .runtime
                            .capture_wall_clock_timing
                            .then_some(emission_start.elapsed().as_millis() as u64);
                    }
                }

                fused_output = FusedOutput {
                    eligible_for_fusion: true,
                    fused_score,
                    confidence,
                    threshold: self.config.fusion.threshold,
                    cooldown_active,
                    recommended_side,
                    actionable_side,
                    signal_generated,
                    suppression_reason,
                    weights_used: Some(normalized_weights),
                };
            } else {
                self.validate_missing_feature_row(index, warmup_bars, &mr_observation, &tsmom_observation, &ldc_observation)?;
            }

            let record = RuntimeOutputRecord {
                schema_version: CANONICAL_SCHEMA_VERSION.to_string(),
                versions: VersionMetadata {
                    runtime_version: env!("CARGO_PKG_VERSION").to_string(),
                    runtime_mode: self.config.runtime.mode.clone(),
                    model_version: hmm_metadata
                        .model_version
                        .clone()
                        .unwrap_or_else(|| self.config.fusion.model_version.clone()),
                },
                ohlcv: ohlcv.clone(),
                features: feature_row.clone(),
                intermediate_signals: IntermediateSignals {
                    ldc: ldc_observation,
                    mr: mr_observation,
                    tsmom: tsmom_observation,
                    fusion_inputs,
                },
                hmm: hmm_metadata,
                fused_output,
                audit: AuditMetadata {
                    run_id: self.run_id.clone(),
                    correlation_id,
                    input_checksum,
                    feature_checksum,
                },
                emission,
                timing: RecordTiming {
                    bar_index: index,
                    signal_generation_latency_ms: self
                        .config
                        .runtime
                        .capture_wall_clock_timing
                        .then_some(signal_start.elapsed().as_millis() as u64),
                    hmm_latency_ms,
                    emission_latency_ms,
                },
            };

            let line = serde_json::to_string(&record).context("failed to serialize runtime output record")?;
            writer
                .write_all(line.as_bytes())
                .context("failed to write runtime output line")?;
            writer.write_all(b"\n").context("failed to write newline")?;
            hasher.update(line.as_bytes());
            hasher.update(b"\n");

            output_rows += 1;
            first_timestamp.get_or_insert(record.ohlcv.timestamp);
            last_timestamp = Some(record.ohlcv.timestamp);
        }

        writer.flush().context("failed to flush runtime output")?;
        let summary = RuntimeRunSummary {
            schema_version: CANONICAL_SCHEMA_VERSION.to_string(),
            runtime_version: env!("CARGO_PKG_VERSION").to_string(),
            runtime_mode: self.config.runtime.mode.clone(),
            symbol: self.config.runtime.symbol.clone(),
            interval: self.config.runtime.interval.clone(),
            input_rows,
            output_rows,
            fused_rows,
            emitted_rows,
            fallback_rows,
            cache_hits,
            first_timestamp,
            last_timestamp,
            canonical_output_sha256: format!("{:x}", hasher.finalize()),
        };

        if let Some(parent) = self.config.output.summary_json.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(&summary_temp_path, serde_json::to_vec_pretty(&summary)?)
            .with_context(|| format!("failed to write {}", summary_temp_path.display()))?;

        self.hmm_resolver.save_cache()?;
        rename_overwrite(&output_temp_path, &self.config.output.canonical_jsonl)?;
        rename_overwrite(&summary_temp_path, &self.config.output.summary_json)?;

        Ok(summary)
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        self.hmm_resolver.save_cache()?;
        if let Some(emitter) = &self.emitter {
            emitter.shutdown().await?;
        }
        Ok(())
    }

    fn prepare_output_paths(&self) -> Result<()> {
        ensure_parent(&self.config.output.canonical_jsonl)?;
        ensure_parent(&self.config.output.summary_json)?;
        if let Some(path) = &self.config.output.feature_parquet {
            ensure_parent(path)?;
        }
        if let Some(path) = &self.config.output.weight_cache_json {
            ensure_parent(path)?;
        }
        if !self.config.output.overwrite {
            if self.config.output.canonical_jsonl.exists() {
                bail!(
                    "canonical output already exists and overwrite=false: {}",
                    self.config.output.canonical_jsonl.display()
                );
            }
            if self.config.output.summary_json.exists() {
                bail!(
                    "summary output already exists and overwrite=false: {}",
                    self.config.output.summary_json.display()
                );
            }
        }
        Ok(())
    }

    fn load_market_data(&self) -> Result<Vec<OHLCV>> {
        let path = &self.config.input.market_data;
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();

        let dataframe = match extension.as_str() {
            "csv" => self.feature_pipeline.read_csv(&path.to_string_lossy())?,
            "parquet" => read_parquet(path)?,
            other => bail!("unsupported market input extension: {}", other),
        };

        let mut market_data = dataframe_to_ohlcv(&dataframe)?;
        if let Some(max_rows) = self.config.runtime.max_rows {
            market_data.truncate(max_rows.min(market_data.len()));
        }
        Ok(market_data)
    }

    fn maybe_add_training_sample(
        &mut self,
        index: usize,
        market_data: &[OHLCV],
        features: &[Features],
        warmup_bars: usize,
    ) -> Result<()> {
        if index < self.config.ldc.training_horizon_bars {
            return Ok(());
        }

        let anchor_index = index - self.config.ldc.training_horizon_bars;
        let anchor_features = &features[anchor_index];
        let current_price = market_data[anchor_index].close as f32;
        let future_price = market_data[index].close as f32;

        match self
            .ldc_engine
            .add_training_sample_from_features(anchor_features, current_price, future_price)
        {
            Ok(()) => Ok(()),
            Err(error) => {
                if anchor_index < warmup_bars
                    || self.config.failure.invalid_features == InvalidFeatureBehavior::SkipRow
                {
                    Ok(())
                } else {
                    Err(error).context(format!(
                        "failed to add LDC training sample at anchor index {}",
                        anchor_index
                    ))
                }
            }
        }
    }

    fn build_ldc_observation(
        &mut self,
        index: usize,
        features: &Features,
        warmup_bars: usize,
    ) -> Result<SignalObservation> {
        if self.ldc_engine.training_samples_count() < self.config.ldc.min_training_samples {
            return Ok(SignalObservation {
                raw: None,
                normalized: None,
                confidence: None,
                available: false,
                status: "waiting_for_training".to_string(),
                reason: Some(format!(
                    "need at least {} training samples, have {}",
                    self.config.ldc.min_training_samples,
                    self.ldc_engine.training_samples_count()
                )),
                normalization: "signal / neighbors_count".to_string(),
            });
        }

        match self.ldc_engine.predict_from_features(features) {
            Ok(prediction) => Ok(build_ldc_signal(&prediction, self.config.ldc.engine.neighbors_count)),
            Err(error) => {
                if index < warmup_bars || self.config.failure.invalid_features == InvalidFeatureBehavior::SkipRow {
                    Ok(SignalObservation {
                        raw: None,
                        normalized: None,
                        confidence: None,
                        available: false,
                        status: "feature_unavailable".to_string(),
                        reason: Some(error.to_string()),
                        normalization: "signal / neighbors_count".to_string(),
                    })
                } else {
                    Err(error).context(format!("failed to compute LDC signal at bar {}", index))
                }
            }
        }
    }

    fn validate_missing_feature_row(
        &self,
        index: usize,
        warmup_bars: usize,
        mr: &SignalObservation,
        tsmom: &SignalObservation,
        ldc: &SignalObservation,
    ) -> Result<()> {
        if index < warmup_bars || self.config.failure.invalid_features == InvalidFeatureBehavior::SkipRow {
            return Ok(());
        }

        let missing = [
            (!mr.available).then_some("mr"),
            (!tsmom.available).then_some("tsmom"),
            (!ldc.available).then_some("ldc"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        if missing.is_empty() {
            Ok(())
        } else {
            bail!("missing or invalid signal inputs after warmup at bar {}: {:?}", index, missing);
        }
    }
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    Ok(())
}

fn partial_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output");
    path.with_file_name(format!("{}.partial", file_name))
}

fn rename_overwrite(from: &Path, to: &Path) -> Result<()> {
    if to.exists() {
        fs::remove_file(to)
            .with_context(|| format!("failed to remove existing {}", to.display()))?;
    }
    fs::rename(from, to)
        .with_context(|| format!("failed to rename {} -> {}", from.display(), to.display()))?;
    Ok(())
}

fn read_parquet(path: &Path) -> Result<DataFrame> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = ParquetReader::new(file);
    reader
        .finish()
        .with_context(|| format!("failed to read parquet {}", path.display()))
}

fn dataframe_to_ohlcv(dataframe: &DataFrame) -> Result<Vec<OHLCV>> {
    let timestamps = series_to_i64_vec(dataframe.column("timestamp")?)?;
    let opens = series_to_f64_vec(dataframe.column("open")?)?;
    let highs = series_to_f64_vec(dataframe.column("high")?)?;
    let lows = series_to_f64_vec(dataframe.column("low")?)?;
    let closes = series_to_f64_vec(dataframe.column("close")?)?;
    let volumes = series_to_f64_vec(dataframe.column("volume")?)?;

    let mut records = Vec::with_capacity(dataframe.height());
    for index in 0..dataframe.height() {
        records.push(OHLCV {
            timestamp: timestamps[index],
            open: opens[index],
            high: highs[index],
            low: lows[index],
            close: closes[index],
            volume: volumes[index],
        });
    }
    Ok(records)
}

fn series_to_i64_vec(series: &Column) -> Result<Vec<i64>> {
    let casted = series.cast(&DataType::Int64)?;
    casted
        .i64()?
        .into_iter()
        .map(|value| value.context("null value found in required integer column"))
        .collect()
}

fn series_to_f64_vec(series: &Column) -> Result<Vec<f64>> {
    let casted = series.cast(&DataType::Float64)?;
    casted
        .f64()?
        .into_iter()
        .map(|value| value.context("null value found in required float column"))
        .collect()
}

fn write_features_parquet(features: &[Features], path: &Path) -> Result<()> {
    ensure_parent(path)?;

    let timestamps: Vec<i64> = features.iter().map(|feature| feature.timestamp).collect();
    let rsi: Vec<Option<f64>> = features.iter().map(|feature| feature.rsi).collect();
    let sma_20: Vec<Option<f64>> = features.iter().map(|feature| feature.sma_20).collect();
    let ema_20: Vec<Option<f64>> = features.iter().map(|feature| feature.ema_20).collect();
    let std_20: Vec<Option<f64>> = features.iter().map(|feature| feature.std_20).collect();
    let zscore_20: Vec<Option<f64>> = features.iter().map(|feature| feature.zscore_20).collect();
    let momentum: Vec<Option<f64>> = features.iter().map(|feature| feature.momentum).collect();
    let wavetrend_1: Vec<Option<f64>> = features.iter().map(|feature| feature.wavetrend_1).collect();
    let wavetrend_2: Vec<Option<f64>> = features.iter().map(|feature| feature.wavetrend_2).collect();
    let cci: Vec<Option<f64>> = features.iter().map(|feature| feature.cci).collect();
    let adx: Vec<Option<f64>> = features.iter().map(|feature| feature.adx).collect();

    let mut dataframe = df! {
        "timestamp" => timestamps,
        "rsi" => rsi,
        "sma_20" => sma_20,
        "ema_20" => ema_20,
        "std_20" => std_20,
        "zscore_20" => zscore_20,
        "momentum" => momentum,
        "wavetrend_1" => wavetrend_1,
        "wavetrend_2" => wavetrend_2,
        "cci" => cci,
        "adx" => adx,
    }?;

    let file = File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    ParquetWriter::new(file)
        .finish(&mut dataframe)
        .with_context(|| format!("failed to write features parquet {}", path.display()))?;
    Ok(())
}

fn build_mr_observation(signal: &Signals, index: usize, warmup_bars: usize) -> SignalObservation {
    build_signal_observation(
        signal.s_mr.map(|value| value as f32),
        index,
        warmup_bars,
        "raw / (1 + |raw|)",
    )
}

fn build_tsmom_observation(signal: &Signals, index: usize, warmup_bars: usize) -> SignalObservation {
    build_signal_observation(
        signal.s_tsmom.map(|value| value as f32),
        index,
        warmup_bars,
        "raw / (1 + |raw|)",
    )
}

fn build_signal_observation(
    raw: Option<f32>,
    index: usize,
    warmup_bars: usize,
    normalization: &str,
) -> SignalObservation {
    match raw.filter(|value| value.is_finite()) {
        Some(value) => SignalObservation {
            raw: Some(value),
            normalized: Some(normalize_signed_value(value)),
            confidence: Some(value.abs().min(1.0)),
            available: true,
            status: "ready".to_string(),
            reason: None,
            normalization: normalization.to_string(),
        },
        None => SignalObservation {
            raw: None,
            normalized: None,
            confidence: None,
            available: false,
            status: if index < warmup_bars {
                "warmup".to_string()
            } else {
                "missing".to_string()
            },
            reason: Some(if index < warmup_bars {
                "feature warmup window not satisfied".to_string()
            } else {
                "required feature value was null or non-finite".to_string()
            }),
            normalization: normalization.to_string(),
        },
    }
}

fn build_ldc_signal(prediction: &LDCPrediction, neighbors_count: usize) -> SignalObservation {
    let denominator = neighbors_count.max(1) as f32;
    let normalized = (prediction.signal / denominator).clamp(-1.0, 1.0);
    SignalObservation {
        raw: Some(prediction.signal),
        normalized: Some(normalized),
        confidence: Some(prediction.confidence.clamp(0.0, 1.0)),
        available: true,
        status: match prediction.prediction_direction {
            Direction::Long => "long".to_string(),
            Direction::Short => "short".to_string(),
            Direction::Neutral => "neutral".to_string(),
        },
        reason: None,
        normalization: "signal / neighbors_count".to_string(),
    }
}

fn normalize_signed_value(value: f32) -> f32 {
    (value / (1.0 + value.abs())).clamp(-1.0, 1.0)
}

fn deterministic_sha256(value: &serde_json::Value) -> Result<String> {
    let encoded = serde_json::to_vec(value).context("failed to encode JSON for checksum")?;
    let digest = Sha256::digest(encoded);
    Ok(format!("{:x}", digest))
}
