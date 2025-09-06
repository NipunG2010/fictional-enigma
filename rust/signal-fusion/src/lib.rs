use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalComponents {
    pub s_ldc: f32,
    pub s_mr: f32,
    pub s_tsmom: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionWeights {
    pub w_ldc: f32,
    pub w_mr: f32,
    pub w_tsmom: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub timestamp: i64,
    pub symbol: String,
    pub side: String, // "BUY", "SELL", "HOLD"
    pub strength: f32, // -1.0 to 1.0
    pub confidence: f32,
    pub components: SignalComponents,
    pub weights: FusionWeights,
    pub model_version: String,
}

pub struct SignalFusion {
    threshold: f32,
    cooldown_period: u64, // seconds
    last_signal_time: Option<i64>,
}

impl SignalFusion {
    pub fn new(threshold: f32, cooldown_period: u64) -> Self {
        Self {
            threshold,
            cooldown_period,
            last_signal_time: None,
        }
    }
    
    pub fn fuse_signals(
        &mut self,
        components: SignalComponents,
        weights: FusionWeights,
        timestamp: i64,
        symbol: &str,
        model_version: &str,
    ) -> Result<Option<TradingSignal>> {
        // Check cooldown
        if let Some(last_time) = self.last_signal_time {
            if timestamp - last_time < self.cooldown_period as i64 {
                return Ok(None);
            }
        }
        
        // Compute fused signal
        let fused_signal = 
            components.s_ldc * weights.w_ldc +
            components.s_mr * weights.w_mr +
            components.s_tsmom * weights.w_tsmom;
        
        // Apply threshold
        if fused_signal.abs() < self.threshold {
            return Ok(None);
        }
        
        // Determine side
        let side = if fused_signal > 0.0 { "BUY" } else { "SELL" };
        
        // Calculate confidence (simplified)
        let confidence = fused_signal.abs().min(1.0);
        
        let signal = TradingSignal {
            timestamp,
            symbol: symbol.to_string(),
            side: side.to_string(),
            strength: fused_signal,
            confidence,
            components,
            weights,
            model_version: model_version.to_string(),
        };
        
        self.last_signal_time = Some(timestamp);
        
        Ok(Some(signal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_fusion_creation() {
        let fusion = SignalFusion::new(0.5, 60);
        assert_eq!(fusion.threshold, 0.5);
        assert_eq!(fusion.cooldown_period, 60);
    }
    
    #[test]
    fn test_fuse_signals_above_threshold() {
        let mut fusion = SignalFusion::new(0.3, 0);
        
        let components = SignalComponents {
            s_ldc: 0.8,
            s_mr: 0.2,
            s_tsmom: 0.1,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.5,
            w_mr: 0.3,
            w_tsmom: 0.2,
        };
        
        let result = fusion.fuse_signals(
            components,
            weights,
            1000,
            "BTCUSDT",
            "v1.0",
        ).unwrap();
        
        assert!(result.is_some());
        let signal = result.unwrap();
        assert_eq!(signal.symbol, "BTCUSDT");
        assert_eq!(signal.side, "BUY");
    }
    
    #[test]
    fn test_fuse_signals_below_threshold() {
        let mut fusion = SignalFusion::new(0.5, 0);
        
        let components = SignalComponents {
            s_ldc: 0.1,
            s_mr: 0.1,
            s_tsmom: 0.1,
        };
        
        let weights = FusionWeights {
            w_ldc: 0.33,
            w_mr: 0.33,
            w_tsmom: 0.34,
        };
        
        let result = fusion.fuse_signals(
            components,
            weights,
            1000,
            "BTCUSDT",
            "v1.0",
        ).unwrap();
        
        assert!(result.is_none());
    }
}
