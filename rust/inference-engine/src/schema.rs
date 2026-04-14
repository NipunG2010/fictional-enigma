use feature_pipeline::{Features, OHLCV};
use serde::{Deserialize, Serialize};
use signal_fusion::{FusionWeights, SignalComponents, SignalSide};

use crate::config::RuntimeMode;

pub const CANONICAL_SCHEMA_VERSION: &str = "imp.runtime.output.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeOutputRecord {
    pub schema_version: String,
    pub versions: VersionMetadata,
    pub ohlcv: OHLCV,
    pub features: Features,
    pub intermediate_signals: IntermediateSignals,
    pub hmm: HmmMetadata,
    pub fused_output: FusedOutput,
    pub audit: AuditMetadata,
    pub emission: EmissionMetadata,
    pub timing: RecordTiming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    pub runtime_version: String,
    pub runtime_mode: RuntimeMode,
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateSignals {
    pub ldc: SignalObservation,
    pub mr: SignalObservation,
    pub tsmom: SignalObservation,
    pub fusion_inputs: Option<SignalComponents>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalObservation {
    pub raw: Option<f32>,
    pub normalized: Option<f32>,
    pub confidence: Option<f32>,
    pub available: bool,
    pub status: String,
    pub reason: Option<String>,
    pub normalization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmMetadata {
    pub source: HmmWeightSource,
    pub model_version: Option<String>,
    pub cache_key: Option<String>,
    pub cache_hit: bool,
    pub state_probabilities: Option<Vec<f32>>,
    pub most_likely_state: Option<usize>,
    pub confidence: Option<f32>,
    pub weights: Option<FusionWeights>,
    pub request_error: Option<String>,
    pub fallback_reason: Option<String>,
    pub service_url: Option<String>,
    pub service_latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HmmWeightSource {
    Service,
    Cache,
    StaticFallback,
    NotInvoked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOutput {
    pub eligible_for_fusion: bool,
    pub fused_score: f32,
    pub confidence: f32,
    pub threshold: f32,
    pub cooldown_active: bool,
    pub recommended_side: SignalSide,
    pub actionable_side: SignalSide,
    pub signal_generated: bool,
    pub suppression_reason: Option<String>,
    pub weights_used: Option<FusionWeights>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditMetadata {
    pub run_id: String,
    pub correlation_id: String,
    pub input_checksum: String,
    pub feature_checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionMetadata {
    pub configured: bool,
    pub backend: String,
    pub attempted: bool,
    pub emitted: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordTiming {
    pub bar_index: usize,
    pub signal_generation_latency_ms: Option<u64>,
    pub hmm_latency_ms: Option<u64>,
    pub emission_latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRunSummary {
    pub schema_version: String,
    pub runtime_version: String,
    pub runtime_mode: RuntimeMode,
    pub symbol: String,
    pub interval: String,
    pub input_rows: usize,
    pub output_rows: usize,
    pub fused_rows: usize,
    pub emitted_rows: usize,
    pub fallback_rows: usize,
    pub cache_hits: usize,
    pub first_timestamp: Option<i64>,
    pub last_timestamp: Option<i64>,
    pub canonical_output_sha256: String,
}
