use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use signal_fusion::hmm_client::{HmmClient, HmmClientConfig};
use signal_fusion::{FusionWeights, SignalComponents};

use crate::config::{HmmMode, HmmUnavailableBehavior, RuntimeConfig};
use crate::schema::{HmmMetadata, HmmWeightSource};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedHmmResponse {
    cache_key: String,
    weights: FusionWeights,
    state_probabilities: Option<Vec<f32>>,
    most_likely_state: Option<usize>,
    confidence: Option<f32>,
    model_version: String,
    cached_at: i64,
    source: HmmWeightSource,
}

#[derive(Debug, Clone)]
pub struct ResolvedHmmWeights {
    pub weights: FusionWeights,
    pub model_version: String,
    pub metadata: HmmMetadata,
}

pub struct HmmWeightResolver {
    mode: HmmMode,
    unavailable_behavior: HmmUnavailableBehavior,
    service_url: String,
    cache_ttl_seconds: u64,
    max_cache_entries: usize,
    fallback_weights: FusionWeights,
    cache_path: Option<PathBuf>,
    client: Option<HmmClient>,
    cache: BTreeMap<String, CachedHmmResponse>,
}

impl HmmWeightResolver {
    pub fn from_config(config: &RuntimeConfig) -> Result<Self> {
        let client = match config.hmm.mode {
            HmmMode::StaticFallback => None,
            HmmMode::ServicePreferred | HmmMode::ServiceRequired => {
                let client_config = HmmClientConfig {
                    base_url: Url::parse(&config.hmm.service_url)
                        .with_context(|| format!("invalid HMM service URL: {}", config.hmm.service_url))?,
                    timeout: Duration::from_millis(config.hmm.timeout_ms),
                    retry_attempts: config.hmm.retry_attempts,
                    retry_delay: Duration::from_millis(config.hmm.retry_delay_ms),
                    enable_fallback: false,
                    fallback_weights: config.hmm.fallback_weights.clone(),
                    circuit_breaker_threshold: 5,
                    circuit_breaker_timeout: Duration::from_secs(30),
                };
                Some(HmmClient::with_config(client_config)?)
            }
        };

        let mut resolver = Self {
            mode: config.hmm.mode.clone(),
            unavailable_behavior: config.hmm.unavailable_behavior.clone(),
            service_url: config.hmm.service_url.clone(),
            cache_ttl_seconds: config.hmm.cache_ttl_seconds,
            max_cache_entries: config.hmm.max_cache_entries,
            fallback_weights: config.hmm.fallback_weights.clone(),
            cache_path: config.output.weight_cache_json.clone(),
            client,
            cache: BTreeMap::new(),
        };
        resolver.load_cache()?;
        Ok(resolver)
    }

    pub fn save_cache(&self) -> Result<()> {
        let Some(path) = &self.cache_path else {
            return Ok(());
        };

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create HMM cache directory {}", parent.display()))?;
        }

        let values: Vec<CachedHmmResponse> = self.cache.values().cloned().collect();
        let payload = serde_json::to_vec_pretty(&values).context("failed to serialize HMM cache")?;
        fs::write(path, payload)
            .with_context(|| format!("failed to write HMM cache {}", path.display()))?;
        Ok(())
    }

    pub async fn resolve(&mut self, signals: &SignalComponents, timestamp: i64) -> Result<ResolvedHmmWeights> {
        let observations = [signals.s_ldc, signals.s_mr, signals.s_tsmom];
        let cache_key = cache_key(observations);

        if let Some(cached) = self.get_cached(&cache_key, timestamp) {
            return Ok(self.from_cached(cache_key, cached, true));
        }

        match self.mode {
            HmmMode::StaticFallback => Ok(self.static_fallback(cache_key, None)),
            HmmMode::ServicePreferred | HmmMode::ServiceRequired => {
                if let Some(client) = &self.client {
                    let start = Instant::now();
                    match client.predict(observations, Some(cache_key.clone())).await {
                        Ok(response) => {
                            let latency_ms = start.elapsed().as_millis() as u64;
                            let source = if response.model_version == "fallback" {
                                HmmWeightSource::StaticFallback
                            } else {
                                HmmWeightSource::Service
                            };

                            let cached = CachedHmmResponse {
                                cache_key: cache_key.clone(),
                                weights: response.fusion_weights.clone(),
                                state_probabilities: Some(response.state_probabilities.clone()),
                                most_likely_state: Some(response.most_likely_state),
                                confidence: Some(response.confidence),
                                model_version: response.model_version.clone(),
                                cached_at: timestamp,
                                source: source.clone(),
                            };
                            self.insert_cache(cached.clone());

                            Ok(ResolvedHmmWeights {
                                weights: response.fusion_weights,
                                model_version: response.model_version.clone(),
                                metadata: HmmMetadata {
                                    source: source.clone(),
                                    model_version: Some(response.model_version.clone()),
                                    cache_key: Some(cache_key),
                                    cache_hit: false,
                                    state_probabilities: Some(response.state_probabilities),
                                    most_likely_state: Some(response.most_likely_state),
                                    confidence: Some(response.confidence),
                                    weights: Some(cached.weights.clone()),
                                    request_error: None,
                                    fallback_reason: (source == HmmWeightSource::StaticFallback)
                                        .then(|| "service returned fallback response".to_string()),
                                    service_url: Some(self.service_url.clone()),
                                    service_latency_ms: Some(latency_ms),
                                },
                            })
                        }
                        Err(error) => self.handle_service_failure(cache_key, error.to_string(), timestamp),
                    }
                } else {
                    self.handle_service_failure(cache_key, "HMM client was not initialized".to_string(), timestamp)
                }
            }
        }
    }

    fn handle_service_failure(
        &mut self,
        cache_key: String,
        error: String,
        timestamp: i64,
    ) -> Result<ResolvedHmmWeights> {
        if matches!(self.unavailable_behavior, HmmUnavailableBehavior::UseCacheThenFallback) {
            if let Some(cached) = self.get_cached(&cache_key, timestamp) {
                let mut resolved = self.from_cached(cache_key, cached, true);
                resolved.metadata.request_error = Some(error);
                return Ok(resolved);
            }
        }

        match self.unavailable_behavior {
            HmmUnavailableBehavior::UseCacheThenFallback | HmmUnavailableBehavior::UseFallback => {
                Ok(self.static_fallback(cache_key, Some(error)))
            }
            HmmUnavailableBehavior::Fail => Err(anyhow::anyhow!(error)),
        }
    }

    fn static_fallback(&mut self, cache_key: String, request_error: Option<String>) -> ResolvedHmmWeights {
        let cached = CachedHmmResponse {
            cache_key: cache_key.clone(),
            weights: self.fallback_weights.clone(),
            state_probabilities: None,
            most_likely_state: None,
            confidence: None,
            model_version: "fallback-static".to_string(),
            cached_at: chrono::Utc::now().timestamp(),
            source: HmmWeightSource::StaticFallback,
        };
        self.insert_cache(cached);

        ResolvedHmmWeights {
            weights: self.fallback_weights.clone(),
            model_version: "fallback-static".to_string(),
            metadata: HmmMetadata {
                source: HmmWeightSource::StaticFallback,
                model_version: Some("fallback-static".to_string()),
                cache_key: Some(cache_key),
                cache_hit: false,
                state_probabilities: None,
                most_likely_state: None,
                confidence: None,
                weights: Some(self.fallback_weights.clone()),
                request_error,
                fallback_reason: Some("static fallback weights configured in runtime config".to_string()),
                service_url: Some(self.service_url.clone()),
                service_latency_ms: None,
            },
        }
    }

    fn from_cached(
        &self,
        cache_key: String,
        cached: CachedHmmResponse,
        cache_hit: bool,
    ) -> ResolvedHmmWeights {
        let fallback_reason = (cached.source == HmmWeightSource::StaticFallback)
            .then(|| "cached fallback weights reused".to_string());

        ResolvedHmmWeights {
            weights: cached.weights.clone(),
            model_version: cached.model_version.clone(),
            metadata: HmmMetadata {
                source: cached.source.clone(),
                model_version: Some(cached.model_version.clone()),
                cache_key: Some(cache_key),
                cache_hit,
                state_probabilities: cached.state_probabilities.clone(),
                most_likely_state: cached.most_likely_state,
                confidence: cached.confidence,
                weights: Some(cached.weights),
                request_error: None,
                fallback_reason,
                service_url: Some(self.service_url.clone()),
                service_latency_ms: None,
            },
        }
    }

    fn get_cached(&self, cache_key: &str, timestamp: i64) -> Option<CachedHmmResponse> {
        let cached = self.cache.get(cache_key)?;
        let age = timestamp.saturating_sub(cached.cached_at) as u64;
        if age <= self.cache_ttl_seconds {
            Some(cached.clone())
        } else {
            None
        }
    }

    fn insert_cache(&mut self, entry: CachedHmmResponse) {
        self.cache.insert(entry.cache_key.clone(), entry);
        while self.cache.len() > self.max_cache_entries {
            let first_key = self.cache.keys().next().cloned();
            if let Some(key) = first_key {
                self.cache.remove(&key);
            }
        }
    }

    fn load_cache(&mut self) -> Result<()> {
        let Some(path) = &self.cache_path else {
            return Ok(());
        };
        if !path.exists() {
            return Ok(());
        }
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read HMM cache {}", path.display()))?;
        let entries: Vec<CachedHmmResponse> = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse HMM cache {}", path.display()))?;
        self.cache = entries
            .into_iter()
            .map(|entry| (entry.cache_key.clone(), entry))
            .collect();
        Ok(())
    }
}

fn cache_key(observations: [f32; 3]) -> String {
    format!(
        "{:.3}|{:.3}|{:.3}",
        observations[0], observations[1], observations[2]
    )
}
