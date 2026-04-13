use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use ldc_engine::LDCConfig;
use serde::{Deserialize, Serialize};
use signal_fusion::emission::{AuditConfig, KafkaConfig, PublisherBackend, RedisConfig, ValidationConfig};
use signal_fusion::{FusionWeights, SignalEmitterConfig, SignalFusion};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    OfflineBatch,
    LocalSmoke,
    IntegrationHmm,
    FallbackOnly,
}

impl Default for RuntimeMode {
    fn default() -> Self {
        Self::OfflineBatch
    }
}

impl std::fmt::Display for RuntimeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeMode::OfflineBatch => write!(f, "offline_batch"),
            RuntimeMode::LocalSmoke => write!(f, "local_smoke"),
            RuntimeMode::IntegrationHmm => write!(f, "integration_hmm"),
            RuntimeMode::FallbackOnly => write!(f, "fallback_only"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HmmMode {
    ServicePreferred,
    ServiceRequired,
    StaticFallback,
}

impl Default for HmmMode {
    fn default() -> Self {
        Self::ServicePreferred
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HmmUnavailableBehavior {
    UseCacheThenFallback,
    UseFallback,
    Fail,
}

impl Default for HmmUnavailableBehavior {
    fn default() -> Self {
        Self::UseCacheThenFallback
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InvalidFeatureBehavior {
    Fail,
    SkipRow,
}

impl Default for InvalidFeatureBehavior {
    fn default() -> Self {
        Self::Fail
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EmissionFailureBehavior {
    Continue,
    FailRun,
}

impl Default for EmissionFailureBehavior {
    fn default() -> Self {
        Self::Continue
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EmissionBackendMode {
    None,
    Redis,
    Kafka,
    Both,
}

impl Default for EmissionBackendMode {
    fn default() -> Self {
        Self::None
    }
}

impl From<&EmissionBackendMode> for PublisherBackend {
    fn from(value: &EmissionBackendMode) -> Self {
        match value {
            EmissionBackendMode::None => PublisherBackend::None,
            EmissionBackendMode::Redis => PublisherBackend::Redis,
            EmissionBackendMode::Kafka => PublisherBackend::Kafka,
            EmissionBackendMode::Both => PublisherBackend::Both,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    pub runtime: RuntimeSection,
    pub input: InputConfig,
    pub output: OutputConfig,
    pub features: FeatureConfig,
    pub ldc: LdcRuntimeConfig,
    pub hmm: HmmConfig,
    pub fusion: FusionConfig,
    pub emission: EmissionConfig,
    pub failure: FailureConfig,
    pub validation: ValidationTargetConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeSection::default(),
            input: InputConfig::default(),
            output: OutputConfig::default(),
            features: FeatureConfig::default(),
            ldc: LdcRuntimeConfig::default(),
            hmm: HmmConfig::default(),
            fusion: FusionConfig::default(),
            emission: EmissionConfig::default(),
            failure: FailureConfig::default(),
            validation: ValidationTargetConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeSection {
    pub mode: RuntimeMode,
    pub symbol: String,
    pub interval: String,
    pub deterministic_output: bool,
    pub capture_wall_clock_timing: bool,
    pub max_rows: Option<usize>,
}

impl Default for RuntimeSection {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::OfflineBatch,
            symbol: "BTCUSDT".to_string(),
            interval: "5m".to_string(),
            deterministic_output: false,
            capture_wall_clock_timing: true,
            max_rows: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InputConfig {
    pub market_data: PathBuf,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            market_data: PathBuf::from("../sample/ohlcv.parquet"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub canonical_jsonl: PathBuf,
    pub summary_json: PathBuf,
    pub feature_parquet: Option<PathBuf>,
    pub weight_cache_json: Option<PathBuf>,
    pub overwrite: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            canonical_jsonl: PathBuf::from("runtime_output/runtime_output.jsonl"),
            summary_json: PathBuf::from("runtime_output/runtime_summary.json"),
            feature_parquet: Some(PathBuf::from("runtime_output/features.parquet")),
            weight_cache_json: Some(PathBuf::from("runtime_output/weight_cache.json")),
            overwrite: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureConfig {
    pub window_size: usize,
    pub rsi_period: usize,
    pub ma_period: usize,
    pub persist_feature_parquet: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            rsi_period: 14,
            ma_period: 20,
            persist_feature_parquet: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LdcRuntimeConfig {
    pub training_horizon_bars: usize,
    pub min_training_samples: usize,
    pub engine: LDCConfig,
}

impl Default for LdcRuntimeConfig {
    fn default() -> Self {
        Self {
            training_horizon_bars: 4,
            min_training_samples: 8,
            engine: LDCConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HmmConfig {
    pub mode: HmmMode,
    pub unavailable_behavior: HmmUnavailableBehavior,
    pub service_url: String,
    pub timeout_ms: u64,
    pub retry_attempts: usize,
    pub retry_delay_ms: u64,
    pub cache_ttl_seconds: u64,
    pub max_cache_entries: usize,
    pub fallback_weights: FusionWeights,
}

impl Default for HmmConfig {
    fn default() -> Self {
        Self {
            mode: HmmMode::ServicePreferred,
            unavailable_behavior: HmmUnavailableBehavior::UseCacheThenFallback,
            service_url: "http://localhost:8000".to_string(),
            timeout_ms: 2_000,
            retry_attempts: 2,
            retry_delay_ms: 100,
            cache_ttl_seconds: 300,
            max_cache_entries: 1_000,
            fallback_weights: FusionWeights {
                w_ldc: 0.4,
                w_mr: 0.3,
                w_tsmom: 0.3,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FusionConfig {
    pub threshold: f32,
    pub cooldown_seconds: u64,
    pub normalize_weights: bool,
    pub model_version: String,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.15,
            cooldown_seconds: 0,
            normalize_weights: true,
            model_version: "runtime-mvp-v1".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmissionConfig {
    pub enabled: bool,
    pub backend: EmissionBackendMode,
    pub fail_on_error: bool,
    pub redis: Option<RedisConfig>,
    pub kafka: Option<KafkaConfig>,
    pub audit_directory: PathBuf,
    pub validation_max_signal_age_seconds: i64,
    pub validation_max_future_seconds: i64,
}

impl Default for EmissionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: EmissionBackendMode::None,
            fail_on_error: false,
            redis: None,
            kafka: None,
            audit_directory: PathBuf::from("runtime_output/audit"),
            validation_max_signal_age_seconds: 31_536_000,
            validation_max_future_seconds: 300,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FailureConfig {
    pub invalid_features: InvalidFeatureBehavior,
    pub emission_failure: EmissionFailureBehavior,
}

impl Default for FailureConfig {
    fn default() -> Self {
        Self {
            invalid_features: InvalidFeatureBehavior::Fail,
            emission_failure: EmissionFailureBehavior::Continue,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ValidationTargetConfig {
    pub expected_output_jsonl: Option<PathBuf>,
    pub expected_summary_json: Option<PathBuf>,
}

impl RuntimeConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read runtime config {}", path.display()))?;
        let mut config: RuntimeConfig = toml::from_str(&raw)
            .with_context(|| format!("failed to parse runtime config {}", path.display()))?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        config.apply_mode_presets();
        config.resolve_relative_paths(base_dir);
        config.validate()?;
        Ok(config)
    }

    pub fn apply_mode_presets(&mut self) {
        match self.runtime.mode {
            RuntimeMode::OfflineBatch => {}
            RuntimeMode::LocalSmoke => {
                self.runtime.deterministic_output = true;
                self.runtime.capture_wall_clock_timing = false;
                self.runtime.max_rows = self.runtime.max_rows.or(Some(64));
                self.hmm.mode = HmmMode::StaticFallback;
                self.hmm.unavailable_behavior = HmmUnavailableBehavior::UseFallback;
                self.fusion.cooldown_seconds = 0;
                self.emission.enabled = false;
                self.emission.backend = EmissionBackendMode::None;
                self.failure.invalid_features = InvalidFeatureBehavior::SkipRow;
            }
            RuntimeMode::IntegrationHmm => {
                self.hmm.mode = HmmMode::ServiceRequired;
                if self.hmm.unavailable_behavior == HmmUnavailableBehavior::Fail {
                    self.hmm.unavailable_behavior = HmmUnavailableBehavior::UseCacheThenFallback;
                }
            }
            RuntimeMode::FallbackOnly => {
                self.hmm.mode = HmmMode::StaticFallback;
                self.hmm.unavailable_behavior = HmmUnavailableBehavior::UseFallback;
            }
        }
    }

    fn resolve_relative_paths(&mut self, base_dir: &Path) {
        resolve_path(base_dir, &mut self.input.market_data);
        resolve_path(base_dir, &mut self.output.canonical_jsonl);
        resolve_path(base_dir, &mut self.output.summary_json);
        if let Some(path) = &mut self.output.feature_parquet {
            resolve_path(base_dir, path);
        }
        if let Some(path) = &mut self.output.weight_cache_json {
            resolve_path(base_dir, path);
        }
        resolve_path(base_dir, &mut self.emission.audit_directory);
        if let Some(path) = &mut self.validation.expected_output_jsonl {
            resolve_path(base_dir, path);
        }
        if let Some(path) = &mut self.validation.expected_summary_json {
            resolve_path(base_dir, path);
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.runtime.symbol.is_empty() {
            bail!("runtime.symbol must not be empty");
        }
        if !self.runtime.symbol.chars().all(|ch| ch.is_ascii_alphanumeric()) {
            bail!("runtime.symbol must be uppercase alphanumeric");
        }
        if self.runtime.symbol != self.runtime.symbol.to_uppercase() {
            bail!("runtime.symbol must be uppercase");
        }
        if !self.input.market_data.exists() {
            bail!("input.market_data does not exist: {}", self.input.market_data.display());
        }
        if self.features.window_size == 0 || self.features.ma_period == 0 || self.features.rsi_period == 0 {
            bail!("feature periods must be greater than zero");
        }
        if self.ldc.training_horizon_bars == 0 {
            bail!("ldc.training_horizon_bars must be greater than zero");
        }
        if self.ldc.min_training_samples == 0 {
            bail!("ldc.min_training_samples must be greater than zero");
        }
        if self.fusion.threshold < 0.0 || self.fusion.threshold > 1.0 {
            bail!("fusion.threshold must be between 0.0 and 1.0");
        }
        self.hmm.fallback_weights.validate()?;
        if matches!(self.hmm.mode, HmmMode::ServicePreferred | HmmMode::ServiceRequired)
            && self.hmm.service_url.trim().is_empty()
        {
            bail!("hmm.service_url must not be empty when HMM service mode is enabled");
        }
        if self.hmm.cache_ttl_seconds == 0 {
            bail!("hmm.cache_ttl_seconds must be greater than zero");
        }
        if self.hmm.max_cache_entries == 0 {
            bail!("hmm.max_cache_entries must be greater than zero");
        }
        if self.emission.enabled {
            match self.emission.backend {
                EmissionBackendMode::None => {}
                EmissionBackendMode::Redis if self.emission.redis.is_none() => {
                    bail!("emission.redis config is required when backend=redis")
                }
                EmissionBackendMode::Kafka if self.emission.kafka.is_none() => {
                    bail!("emission.kafka config is required when backend=kafka")
                }
                EmissionBackendMode::Both => {
                    if self.emission.redis.is_none() {
                        bail!("emission.redis config is required when backend=both");
                    }
                    if self.emission.kafka.is_none() {
                        bail!("emission.kafka config is required when backend=both");
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn warmup_bars(&self) -> usize {
        self.features
            .window_size
            .max(self.features.ma_period)
            .max(self.features.rsi_period)
    }

    pub fn build_signal_fusion(&self) -> SignalFusion {
        SignalFusion::with_normalization(
            self.fusion.threshold,
            self.fusion.cooldown_seconds,
            self.fusion.normalize_weights,
        )
    }

    pub fn build_emitter_config(&self) -> SignalEmitterConfig {
        let mut validation = ValidationConfig::default();
        validation.max_signal_age_seconds = self.emission.validation_max_signal_age_seconds;
        validation.max_future_seconds = self.emission.validation_max_future_seconds;

        let audit = AuditConfig {
            log_directory: self.emission.audit_directory.clone(),
            log_filename: "runtime_signal_emission".to_string(),
            ..AuditConfig::default()
        };

        let publisher = signal_fusion::emission::SignalPublisherConfig {
            backend: PublisherBackend::from(&self.emission.backend),
            enabled: self.emission.enabled,
            redis: self.emission.redis.clone(),
            kafka: self.emission.kafka.clone(),
            audit,
            validation,
            ..signal_fusion::emission::SignalPublisherConfig::default()
        };

        SignalEmitterConfig {
            enabled: self.emission.enabled,
            publisher,
            fail_on_emission_error: self.emission.fail_on_error,
            model_version: self.fusion.model_version.clone(),
            audit_feature_computation: true,
        }
    }
}

fn resolve_path(base_dir: &Path, path: &mut PathBuf) {
    if path.is_relative() {
        *path = base_dir.join(&*path);
    }
}
