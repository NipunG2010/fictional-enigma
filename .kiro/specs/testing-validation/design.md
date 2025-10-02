# Testing & Validation Design Document

## Overview

The Testing & Validation design provides a comprehensive testing framework for the LDC trading system, building upon the existing performance-optimized LDC engine. The design includes unit tests for mathematical accuracy, a historical backtesting framework for strategy validation, performance validation tests, integration testing, and statistical analysis tools to ensure the system meets production requirements for real-time trading.

## Architecture

### Testing Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Testing & Validation Framework              │
├─────────────────────────────────────────────────────────────────┤
│  Test Categories                     │  Supporting Infrastructure │
├─────────────────────────────────────┼─────────────────────────────┤
│  • Unit Tests (Mathematical)        │  • Test Data Generators     │
│  • Performance Validation Tests     │  • Market Data Simulators   │
│  • Integration Tests                │  • Statistical Analysis     │
│  • Backtesting Framework           │  • Report Generation        │
│  • Statistical Validation          │  • CI/CD Integration        │
└─────────────────────────────────────┴─────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Test Execution Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Unit Tests (Mathematical Accuracy)                   │
│  Stage 2: Performance Validation (Timing & Resource)           │
│  Stage 3: Integration Tests (Component Interaction)            │
│  Stage 4: Backtesting (Historical Strategy Validation)         │
│  Stage 5: Statistical Analysis (Predictive Quality)            │
└─────────────────────────────────────────────────────────────────┘
```

### Integration with Existing LDC Engine

The testing framework integrates with the existing enhanced LDC engine:

```rust
// Existing LDC Engine (Enhanced)
pub struct LDCEngine {
    training_samples: VecDeque<TrainingSample>,
    config: LDCConfig,                    // Enhanced with performance options
    performance_metrics: PerformanceMetrics, // Enhanced with detailed metrics
    hnsw_index: Option<HNSWIndex>,       // Optional HNSW indexing
    // ... other enhanced fields
}

// New Testing Framework Components
pub struct TestingFramework {
    ldc_engine: LDCEngine,
    test_data_generator: TestDataGenerator,
    backtesting_engine: BacktestingEngine,
    performance_validator: PerformanceValidator,
    statistical_analyzer: StatisticalAnalyzer,
}
```

## Components and Interfaces

### 1. Unit Testing Framework

**Mathematical Accuracy Testing Interface:**
```rust
pub struct MathematicalTestSuite {
    tolerance: f64,
    test_cases: Vec<DistanceTestCase>,
}

#[derive(Debug, Clone)]
pub struct DistanceTestCase {
    pub name: String,
    pub features1: FeatureSeries,
    pub features2: FeatureSeries,
    pub expected_distance: f64,
    pub test_category: TestCategory,
}

#[derive(Debug, Clone)]
pub enum TestCategory {
    Standard,           // Normal feature ranges
    EdgeCases,         // Zero, NaN, infinity
    ExtremeValues,     // Very large/small values
    Precision,         // Floating-point precision tests
}

impl MathematicalTestSuite {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-6,
            test_cases: Self::generate_test_cases(),
        }
    }
    
    /// Test standard vs SIMD distance calculation accuracy
    pub fn test_simd_accuracy(&self) -> TestResult {
        let mut results = Vec::new();
        
        for test_case in &self.test_cases {
            let standard_distance = LDCEngine::lorentzian_distance(
                &test_case.features1, 
                &test_case.features2, 
                5
            );
            
            let simd_distance = test_case.features1.lorentzian_distance_simd(
                &test_case.features2
            );
            
            let diff = (standard_distance - simd_distance).abs();
            let passed = diff < self.tolerance as f32;
            
            results.push(UnitTestResult {
                test_name: format!("SIMD_vs_Standard_{}", test_case.name),
                passed,
                expected: standard_distance as f64,
                actual: simd_distance as f64,
                difference: diff as f64,
                tolerance: self.tolerance,
            });
        }
        
        TestResult::from_unit_results(results)
    }
    
    /// Test HNSW distance calculation compatibility
    pub fn test_hnsw_compatibility(&self) -> TestResult {
        let mut results = Vec::new();
        
        for test_case in &self.test_cases {
            let rust_distance = LDCEngine::lorentzian_distance(
                &test_case.features1, 
                &test_case.features2, 
                5
            );
            
            let features1_array = test_case.features1.to_array();
            let features2_array = test_case.features2.to_array();
            let hnsw_distance = lorentzian_distance_hnsw(&features1_array, &features2_array);
            
            let diff = (rust_distance - hnsw_distance).abs();
            let passed = diff < self.tolerance as f32;
            
            results.push(UnitTestResult {
                test_name: format!("HNSW_vs_Standard_{}", test_case.name),
                passed,
                expected: rust_distance as f64,
                actual: hnsw_distance as f64,
                difference: diff as f64,
                tolerance: self.tolerance,
            });
        }
        
        TestResult::from_unit_results(results)
    }
    
    /// Generate comprehensive test cases
    fn generate_test_cases() -> Vec<DistanceTestCase> {
        let mut cases = Vec::new();
        
        // Standard test cases
        cases.push(DistanceTestCase {
            name: "identical_features".to_string(),
            features1: FeatureSeries { f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0 },
            features2: FeatureSeries { f1: 1.0, f2: 2.0, f3: 3.0, f4: 4.0, f5: 5.0 },
            expected_distance: 0.0,
            test_category: TestCategory::Standard,
        });
        
        // Edge cases
        cases.push(DistanceTestCase {
            name: "zero_features".to_string(),
            features1: FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 },
            features2: FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 },
            expected_distance: 0.0,
            test_category: TestCategory::EdgeCases,
        });
        
        // Extreme values
        cases.push(DistanceTestCase {
            name: "extreme_values".to_string(),
            features1: FeatureSeries { f1: f32::MAX, f2: f32::MIN, f3: 1e10, f4: -1e10, f5: 0.0 },
            features2: FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 },
            expected_distance: f64::INFINITY, // Will be calculated
            test_category: TestCategory::ExtremeValues,
        });
        
        // Add more test cases...
        cases
    }
}

#[derive(Debug, Clone)]
pub struct UnitTestResult {
    pub test_name: String,
    pub passed: bool,
    pub expected: f64,
    pub actual: f64,
    pub difference: f64,
    pub tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f64,
    pub results: Vec<UnitTestResult>,
}
```

### 2. Performance Validation Framework

**Performance Testing Interface:**
```rust
pub struct PerformanceValidator {
    config: PerformanceTestConfig,
    test_datasets: Vec<TestDataset>,
}

#[derive(Debug, Clone)]
pub struct PerformanceTestConfig {
    pub target_latency_1k_samples_ms: f64,   // 0.5ms
    pub target_latency_10k_samples_ms: f64,  // 1.0ms
    pub target_latency_50k_samples_ms: f64,  // 5.0ms
    pub target_cpu_utilization_percent: f64, // 90%
    pub target_hnsw_accuracy_percent: f64,   // 95%
    pub test_iterations: usize,               // 100
    pub warmup_iterations: usize,             // 10
}

#[derive(Debug, Clone)]
pub struct TestDataset {
    pub name: String,
    pub size: usize,
    pub samples: Vec<TrainingSample>,
    pub query_features: Vec<FeatureSeries>,
}

impl PerformanceValidator {
    pub fn new(config: PerformanceTestConfig) -> Self {
        Self {
            config,
            test_datasets: Self::generate_test_datasets(),
        }
    }
    
    /// Validate k-NN query performance meets targets
    pub fn validate_query_performance(&self, engine: &LDCEngine) -> PerformanceTestResult {
        let mut results = Vec::new();
        
        for dataset in &self.test_datasets {
            let mut latencies = Vec::new();
            
            // Warmup
            for _ in 0..self.config.warmup_iterations {
                let _ = engine.find_k_nearest_neighbors_optimized(&dataset.query_features[0]);
            }
            
            // Actual measurements
            for i in 0..self.config.test_iterations {
                let query_idx = i % dataset.query_features.len();
                let query = &dataset.query_features[query_idx];
                
                let start = std::time::Instant::now();
                let _ = engine.find_k_nearest_neighbors_optimized(query);
                let duration = start.elapsed();
                
                latencies.push(duration.as_secs_f64() * 1000.0); // Convert to ms
            }
            
            let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
            let p95_latency = Self::calculate_percentile(&latencies, 95.0);
            let p99_latency = Self::calculate_percentile(&latencies, 99.0);
            
            let target_latency = match dataset.size {
                size if size <= 1000 => self.config.target_latency_1k_samples_ms,
                size if size <= 10000 => self.config.target_latency_10k_samples_ms,
                _ => self.config.target_latency_50k_samples_ms,
            };
            
            results.push(PerformanceTestCase {
                dataset_name: dataset.name.clone(),
                dataset_size: dataset.size,
                avg_latency_ms: avg_latency,
                p95_latency_ms: p95_latency,
                p99_latency_ms: p99_latency,
                target_latency_ms: target_latency,
                passed: avg_latency <= target_latency,
            });
        }
        
        PerformanceTestResult { results }
    }
    
    /// Validate HNSW accuracy vs exact search
    pub fn validate_hnsw_accuracy(&self, engine: &LDCEngine) -> HNSWAccuracyResult {
        let mut accuracy_results = Vec::new();
        
        for dataset in &self.test_datasets {
            if dataset.size < 1000 { continue; } // HNSW only useful for larger datasets
            
            let mut matches = 0;
            let mut total_queries = 0;
            
            for query in &dataset.query_features {
                // Get exact k-NN results
                let mut exact_engine = engine.clone();
                exact_engine.config.use_hnsw_index = false;
                let exact_results = exact_engine.find_k_nearest_neighbors_optimized(query);
                
                // Get HNSW results
                let mut hnsw_engine = engine.clone();
                hnsw_engine.config.use_hnsw_index = true;
                let hnsw_results = hnsw_engine.find_k_nearest_neighbors_optimized(query);
                
                // Calculate overlap
                let overlap = Self::calculate_knn_overlap(&exact_results, &hnsw_results);
                matches += overlap;
                total_queries += exact_results.len();
            }
            
            let accuracy = matches as f64 / total_queries as f64 * 100.0;
            let passed = accuracy >= self.config.target_hnsw_accuracy_percent;
            
            accuracy_results.push(HNSWAccuracyCase {
                dataset_name: dataset.name.clone(),
                dataset_size: dataset.size,
                accuracy_percent: accuracy,
                target_accuracy_percent: self.config.target_hnsw_accuracy_percent,
                passed,
            });
        }
        
        HNSWAccuracyResult { results: accuracy_results }
    }
    
    fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile / 100.0 * (sorted.len() - 1) as f64) as usize;
        sorted[index]
    }
    
    fn calculate_knn_overlap(exact: &[(f32, Direction)], hnsw: &[(f32, Direction)]) -> usize {
        let exact_labels: std::collections::HashSet<_> = exact.iter().map(|(_, label)| label).collect();
        let hnsw_labels: std::collections::HashSet<_> = hnsw.iter().map(|(_, label)| label).collect();
        exact_labels.intersection(&hnsw_labels).count()
    }
    
    fn generate_test_datasets() -> Vec<TestDataset> {
        vec![
            Self::create_synthetic_dataset("small_1k", 1000),
            Self::create_synthetic_dataset("medium_10k", 10000),
            Self::create_synthetic_dataset("large_50k", 50000),
        ]
    }
    
    fn create_synthetic_dataset(name: &str, size: usize) -> TestDataset {
        // Generate realistic synthetic trading data
        let mut samples = Vec::new();
        let mut query_features = Vec::new();
        
        for i in 0..size {
            let features = FeatureSeries {
                f1: (i as f32 * 0.1).sin() * 50.0 + 50.0, // RSI-like
                f2: (i as f32 * 0.05).cos() * 100.0,      // WT-like
                f3: (i as f32 * 0.02).sin() * 200.0,      // CCI-like
                f4: (i as f32 * 0.01).abs() * 50.0,       // ADX-like
                f5: (i as f32 * 0.03).tan().abs() * 30.0, // Additional feature
            };
            
            let label = if i % 3 == 0 {
                Direction::Long
            } else if i % 3 == 1 {
                Direction::Short
            } else {
                Direction::Neutral
            };
            
            samples.push(TrainingSample {
                features: features.clone(),
                label,
                timestamp: i as i64,
                bar_index: i,
            });
            
            // Add some features as query samples
            if i % 100 == 0 {
                query_features.push(features);
            }
        }
        
        TestDataset {
            name: name.to_string(),
            size,
            samples,
            query_features,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceTestCase {
    pub dataset_name: String,
    pub dataset_size: usize,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub target_latency_ms: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct PerformanceTestResult {
    pub results: Vec<PerformanceTestCase>,
}

#[derive(Debug, Clone)]
pub struct HNSWAccuracyCase {
    pub dataset_name: String,
    pub dataset_size: usize,
    pub accuracy_percent: f64,
    pub target_accuracy_percent: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct HNSWAccuracyResult {
    pub results: Vec<HNSWAccuracyCase>,
}
```

### 3. Historical Backtesting Framework

**Backtesting Engine Interface:**
```rust
pub struct BacktestingEngine {
    config: BacktestConfig,
    ldc_engine: LDCEngine,
    performance_calculator: PerformanceCalculator,
}

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub position_size: f64,
    pub transaction_cost: f64,
    pub slippage: f64,
    pub signal_threshold: f32,
    pub max_positions: usize,
    pub rebalance_frequency: Duration,
}

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub total_trades: usize,
    pub profitable_trades: usize,
    pub average_trade_return: f64,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<EquityPoint>,
    pub performance_attribution: PerformanceAttribution,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub direction: Direction,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub signal_strength: f32,
    pub holding_period: Duration,
}

#[derive(Debug, Clone)]
pub struct EquityPoint {
    pub timestamp: DateTime<Utc>,
    pub equity: f64,
    pub drawdown: f64,
    pub position_value: f64,
}

impl BacktestingEngine {
    pub fn new(config: BacktestConfig, ldc_config: LDCConfig) -> Self {
        Self {
            config,
            ldc_engine: LDCEngine::with_config(ldc_config),
            performance_calculator: PerformanceCalculator::new(),
        }
    }
    
    /// Run historical backtest on OHLCV data
    pub fn run_backtest(
        &mut self,
        ohlcv_data: &[OHLCV],
        features_data: &[Features],
    ) -> Result<BacktestResult> {
        if ohlcv_data.len() != features_data.len() {
            return Err(anyhow::anyhow!("OHLCV and features data length mismatch"));
        }
        
        let mut trades = Vec::new();
        let mut equity_curve = Vec::new();
        let mut current_equity = self.config.initial_capital;
        let mut current_position: Option<Position> = None;
        let mut max_equity = current_equity;
        let mut max_drawdown = 0.0;
        
        // Build training data for LDC engine
        self.build_training_data(ohlcv_data, features_data)?;
        
        // Walk through historical data
        for (i, (ohlcv, features)) in ohlcv_data.iter().zip(features_data.iter()).enumerate() {
            let timestamp = DateTime::from_timestamp(ohlcv.timestamp, 0)
                .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?;
            
            // Generate LDC prediction
            let prediction = self.ldc_engine.predict_from_features(features)?;
            
            // Check for position exit
            if let Some(ref position) = current_position {
                if self.should_exit_position(position, &prediction, ohlcv) {
                    let trade = self.close_position(position, ohlcv, timestamp)?;
                    current_equity += trade.pnl;
                    trades.push(trade);
                    current_position = None;
                }
            }
            
            // Check for new position entry
            if current_position.is_none() && self.should_enter_position(&prediction) {
                current_position = Some(Position {
                    direction: prediction.prediction_direction,
                    entry_price: ohlcv.close,
                    entry_time: timestamp,
                    quantity: self.calculate_position_size(current_equity, ohlcv.close),
                    signal_strength: prediction.signal,
                });
            }
            
            // Update equity curve
            let position_value = current_position
                .as_ref()
                .map(|p| p.calculate_unrealized_pnl(ohlcv.close))
                .unwrap_or(0.0);
            
            let total_equity = current_equity + position_value;
            max_equity = max_equity.max(total_equity);
            let current_drawdown = (max_equity - total_equity) / max_equity;
            max_drawdown = max_drawdown.max(current_drawdown);
            
            equity_curve.push(EquityPoint {
                timestamp,
                equity: total_equity,
                drawdown: current_drawdown,
                position_value,
            });
            
            // Update training data (rolling window)
            if i > 100 { // Start updating after initial training period
                self.update_training_data(ohlcv, features, i)?;
            }
        }
        
        // Close any remaining position
        if let Some(position) = current_position {
            let last_ohlcv = ohlcv_data.last().unwrap();
            let last_timestamp = DateTime::from_timestamp(last_ohlcv.timestamp, 0).unwrap();
            let trade = self.close_position(&position, last_ohlcv, last_timestamp)?;
            current_equity += trade.pnl;
            trades.push(trade);
        }
        
        // Calculate performance metrics
        let performance_metrics = self.performance_calculator.calculate_metrics(
            &trades,
            &equity_curve,
            self.config.initial_capital,
        );
        
        Ok(BacktestResult {
            total_return: (current_equity - self.config.initial_capital) / self.config.initial_capital,
            sharpe_ratio: performance_metrics.sharpe_ratio,
            max_drawdown,
            win_rate: performance_metrics.win_rate,
            total_trades: trades.len(),
            profitable_trades: trades.iter().filter(|t| t.pnl > 0.0).count(),
            average_trade_return: trades.iter().map(|t| t.pnl).sum::<f64>() / trades.len() as f64,
            trades,
            equity_curve,
            performance_attribution: performance_metrics.attribution,
        })
    }
    
    fn build_training_data(&mut self, ohlcv_data: &[OHLCV], features_data: &[Features]) -> Result<()> {
        // Use first portion of data for initial training
        let training_size = (ohlcv_data.len() * 0.2).min(2000); // 20% or max 2000 samples
        
        self.ldc_engine.create_training_samples_from_ohlcv(
            &ohlcv_data[..training_size],
            &features_data[..training_size],
            4, // 4-bar horizon as per Pine Script
        )?;
        
        Ok(())
    }
    
    fn should_enter_position(&self, prediction: &LDCPrediction) -> bool {
        prediction.signal.abs() >= self.config.signal_threshold &&
        prediction.confidence > 0.5
    }
    
    fn should_exit_position(&self, position: &Position, prediction: &LDCPrediction, ohlcv: &OHLCV) -> bool {
        // Exit on opposite signal or stop loss
        match position.direction {
            Direction::Long => prediction.prediction_direction == Direction::Short || 
                              ohlcv.close < position.entry_price * 0.95, // 5% stop loss
            Direction::Short => prediction.prediction_direction == Direction::Long ||
                               ohlcv.close > position.entry_price * 1.05, // 5% stop loss
            Direction::Neutral => true, // Exit neutral positions immediately
        }
    }
    
    fn calculate_position_size(&self, equity: f64, price: f64) -> f64 {
        (equity * self.config.position_size) / price
    }
    
    fn close_position(&self, position: &Position, ohlcv: &OHLCV, timestamp: DateTime<Utc>) -> Result<Trade> {
        let pnl = position.calculate_realized_pnl(ohlcv.close, self.config.transaction_cost);
        
        Ok(Trade {
            entry_time: position.entry_time,
            exit_time: timestamp,
            direction: position.direction,
            entry_price: position.entry_price,
            exit_price: ohlcv.close,
            quantity: position.quantity,
            pnl,
            signal_strength: position.signal_strength,
            holding_period: timestamp - position.entry_time,
        })
    }
    
    fn update_training_data(&mut self, ohlcv: &OHLCV, features: &Features, index: usize) -> Result<()> {
        // Add new training sample with future price (if available)
        if index >= 4 { // Need 4 bars ahead for labeling
            // This would need access to future data in a real implementation
            // For backtesting, we can look ahead since we have historical data
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Position {
    direction: Direction,
    entry_price: f64,
    entry_time: DateTime<Utc>,
    quantity: f64,
    signal_strength: f32,
}

impl Position {
    fn calculate_unrealized_pnl(&self, current_price: f64) -> f64 {
        match self.direction {
            Direction::Long => (current_price - self.entry_price) * self.quantity,
            Direction::Short => (self.entry_price - current_price) * self.quantity,
            Direction::Neutral => 0.0,
        }
    }
    
    fn calculate_realized_pnl(&self, exit_price: f64, transaction_cost: f64) -> f64 {
        let gross_pnl = match self.direction {
            Direction::Long => (exit_price - self.entry_price) * self.quantity,
            Direction::Short => (self.entry_price - exit_price) * self.quantity,
            Direction::Neutral => 0.0,
        };
        
        gross_pnl - (transaction_cost * self.quantity * 2.0) // Entry + exit costs
    }
}
```

### 4. Statistical Analysis Framework

**Statistical Validation Interface:**
```rust
pub struct StatisticalAnalyzer {
    config: StatisticalConfig,
}

#[derive(Debug, Clone)]
pub struct StatisticalConfig {
    pub confidence_level: f64,        // 0.95 for 95% confidence
    pub min_sample_size: usize,       // Minimum samples for statistical significance
    pub significance_threshold: f64,   // p-value threshold (0.05)
}

#[derive(Debug, Clone)]
pub struct StatisticalAnalysisResult {
    pub prediction_accuracy: AccuracyMetrics,
    pub signal_quality: SignalQualityMetrics,
    pub market_regime_analysis: MarketRegimeAnalysis,
    pub statistical_significance: SignificanceTest,
}

#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    pub hit_rate: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: ConfusionMatrix,
}

#[derive(Debug, Clone)]
pub struct SignalQualityMetrics {
    pub signal_to_noise_ratio: f64,
    pub information_coefficient: f64,
    pub signal_strength_distribution: Vec<f64>,
    pub confidence_distribution: Vec<f64>,
}

impl StatisticalAnalyzer {
    pub fn analyze_predictions(
        &self,
        predictions: &[LDCPrediction],
        actual_outcomes: &[Direction],
        market_data: &[OHLCV],
    ) -> StatisticalAnalysisResult {
        let accuracy_metrics = self.calculate_accuracy_metrics(predictions, actual_outcomes);
        let signal_quality = self.calculate_signal_quality(predictions, market_data);
        let regime_analysis = self.analyze_market_regimes(predictions, market_data);
        let significance = self.test_statistical_significance(predictions, actual_outcomes);
        
        StatisticalAnalysisResult {
            prediction_accuracy: accuracy_metrics,
            signal_quality,
            market_regime_analysis: regime_analysis,
            statistical_significance: significance,
        }
    }
    
    fn calculate_accuracy_metrics(
        &self,
        predictions: &[LDCPrediction],
        actual_outcomes: &[Direction],
    ) -> AccuracyMetrics {
        let mut tp = 0; // True positives
        let mut fp = 0; // False positives
        let mut tn = 0; // True negatives
        let mut fn_count = 0; // False negatives
        
        for (pred, actual) in predictions.iter().zip(actual_outcomes.iter()) {
            match (pred.prediction_direction, *actual) {
                (Direction::Long, Direction::Long) => tp += 1,
                (Direction::Long, _) => fp += 1,
                (Direction::Short, Direction::Short) => tn += 1,
                (Direction::Short, _) => fn_count += 1,
                _ => {} // Neutral cases
            }
        }
        
        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / (tp + fn_count) as f64;
        let f1_score = 2.0 * (precision * recall) / (precision + recall);
        let hit_rate = (tp + tn) as f64 / predictions.len() as f64;
        
        AccuracyMetrics {
            hit_rate,
            precision,
            recall,
            f1_score,
            confusion_matrix: ConfusionMatrix { tp, fp, tn, fn_count },
        }
    }
    
    fn calculate_signal_quality(
        &self,
        predictions: &[LDCPrediction],
        market_data: &[OHLCV],
    ) -> SignalQualityMetrics {
        // Calculate information coefficient (correlation between signal and future returns)
        let signals: Vec<f64> = predictions.iter().map(|p| p.signal as f64).collect();
        let returns: Vec<f64> = market_data.windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();
        
        let ic = self.calculate_correlation(&signals[..returns.len()], &returns);
        
        // Calculate signal-to-noise ratio
        let signal_mean = signals.iter().sum::<f64>() / signals.len() as f64;
        let signal_variance = signals.iter()
            .map(|s| (s - signal_mean).powi(2))
            .sum::<f64>() / signals.len() as f64;
        let signal_std = signal_variance.sqrt();
        let snr = signal_mean.abs() / signal_std;
        
        SignalQualityMetrics {
            signal_to_noise_ratio: snr,
            information_coefficient: ic,
            signal_strength_distribution: signals,
            confidence_distribution: predictions.iter().map(|p| p.confidence as f64).collect(),
        }
    }
    
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len()) as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
        let sum_x2 = x.iter().map(|a| a * a).sum::<f64>();
        let sum_y2 = y.iter().map(|b| b * b).sum::<f64>();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 { 0.0 } else { numerator / denominator }
    }
}

#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    pub tp: usize, // True positives
    pub fp: usize, // False positives
    pub tn: usize, // True negatives
    pub fn_count: usize, // False negatives
}
```

## Error Handling

### Testing-Specific Error Management

```rust
#[derive(thiserror::Error, Debug)]
pub enum TestingError {
    #[error("Test data generation failed: {0}")]
    TestDataError(String),
    
    #[error("Performance test failed: {component} took {actual_ms}ms (target: {target_ms}ms)")]
    PerformanceTestFailed {
        component: String,
        actual_ms: f64,
        target_ms: f64,
    },
    
    #[error("Statistical test failed: insufficient sample size {actual} (minimum: {required})")]
    InsufficientSampleSize { actual: usize, required: usize },
    
    #[error("Backtest validation failed: {0}")]
    BacktestError(String),
    
    #[error("Mathematical accuracy test failed: difference {difference} exceeds tolerance {tolerance}")]
    AccuracyTestFailed { difference: f64, tolerance: f64 },
}
```

## Testing Strategy

### Comprehensive Test Execution Pipeline

```rust
pub struct TestExecutionPipeline {
    unit_tests: MathematicalTestSuite,
    performance_validator: PerformanceValidator,
    backtesting_engine: BacktestingEngine,
    statistical_analyzer: StatisticalAnalyzer,
}

impl TestExecutionPipeline {
    pub fn run_all_tests(&self, engine: &LDCEngine) -> ComprehensiveTestResult {
        let mut results = ComprehensiveTestResult::new();
        
        // Stage 1: Unit Tests
        results.unit_test_results = self.run_unit_tests();
        
        // Stage 2: Performance Validation
        results.performance_results = self.performance_validator.validate_query_performance(engine);
        results.hnsw_accuracy_results = self.performance_validator.validate_hnsw_accuracy(engine);
        
        // Stage 3: Integration Tests
        results.integration_results = self.run_integration_tests(engine);
        
        // Stage 4: Backtesting
        results.backtest_results = self.run_backtests();
        
        // Stage 5: Statistical Analysis
        results.statistical_results = self.run_statistical_analysis();
        
        results
    }
}
```

## Implementation Considerations

### Test Data Management

1. **Synthetic Data Generation**: Create realistic market data for testing
2. **Historical Data Integration**: Use real market data for backtesting
3. **Edge Case Coverage**: Test boundary conditions and error scenarios
4. **Performance Data**: Generate datasets of various sizes for performance testing

### CI/CD Integration

1. **Parallel Execution**: Run independent test suites in parallel
2. **Test Reporting**: Generate machine-readable reports for CI systems
3. **Performance Regression Detection**: Track performance metrics over time
4. **Automated Test Selection**: Run relevant tests based on code changes

### Statistical Rigor

1. **Sample Size Validation**: Ensure sufficient data for statistical significance
2. **Multiple Testing Correction**: Adjust p-values for multiple comparisons
3. **Cross-Validation**: Use proper train/test splits for validation
4. **Confidence Intervals**: Report uncertainty in performance metrics