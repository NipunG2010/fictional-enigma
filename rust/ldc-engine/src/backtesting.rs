use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};


use crate::{LDCEngine, LDCConfig, Direction, FeatureSeries, TrainingSample};
use feature_pipeline::{OHLCV, Features};

/// Configuration for backtesting parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital for backtesting
    pub initial_capital: f64,
    /// Position size as a fraction of equity (0.0 to 1.0)
    pub position_size: f64,
    /// Transaction cost per trade as a fraction of trade value
    pub transaction_cost: f64,
    /// Slippage as a fraction of price
    pub slippage: f64,
    /// Minimum signal strength required to enter a position
    pub signal_threshold: f32,
    /// Maximum number of concurrent positions
    pub max_positions: usize,
    /// How often to rebalance positions
    pub rebalance_frequency: Duration,
    /// Stop loss percentage (0.05 = 5%)
    pub stop_loss_percent: f64,
    /// Take profit percentage (0.10 = 10%)
    pub take_profit_percent: f64,
    /// Minimum holding period before allowing exit
    pub min_holding_period: Duration,
    /// Maximum holding period before forced exit
    pub max_holding_period: Duration,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1, // 10% of equity per position
            transaction_cost: 0.001, // 0.1% transaction cost
            slippage: 0.0005, // 0.05% slippage
            signal_threshold: 0.5,
            max_positions: 1,
            rebalance_frequency: Duration::hours(1),
            stop_loss_percent: 0.05, // 5% stop loss
            take_profit_percent: 0.10, // 10% take profit
            min_holding_period: Duration::minutes(15),
            max_holding_period: Duration::hours(24),
        }
    }
}

/// Represents a trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Direction of the position (Long/Short)
    pub direction: Direction,
    /// Entry price
    pub entry_price: f64,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Position quantity (number of shares/units)
    pub quantity: f64,
    /// Signal strength that triggered this position
    pub signal_strength: f32,
    /// Confidence level of the prediction
    pub confidence: f32,
    /// Stop loss price
    pub stop_loss_price: f64,
    /// Take profit price
    pub take_profit_price: f64,
    /// Position ID for tracking
    pub position_id: u64,
}

impl Position {
    /// Create a new position
    pub fn new(
        direction: Direction,
        entry_price: f64,
        entry_time: DateTime<Utc>,
        quantity: f64,
        signal_strength: f32,
        confidence: f32,
        config: &BacktestConfig,
        position_id: u64,
    ) -> Self {
        let (stop_loss_price, take_profit_price) = match direction {
            Direction::Long => (
                entry_price * (1.0 - config.stop_loss_percent),
                entry_price * (1.0 + config.take_profit_percent),
            ),
            Direction::Short => (
                entry_price * (1.0 + config.stop_loss_percent),
                entry_price * (1.0 - config.take_profit_percent),
            ),
            Direction::Neutral => (entry_price, entry_price), // No stop/take profit for neutral
        };

        Self {
            direction,
            entry_price,
            entry_time,
            quantity,
            signal_strength,
            confidence,
            stop_loss_price,
            take_profit_price,
            position_id,
        }
    }

    /// Calculate unrealized PnL for the position
    pub fn calculate_unrealized_pnl(&self, current_price: f64) -> f64 {
        match self.direction {
            Direction::Long => (current_price - self.entry_price) * self.quantity,
            Direction::Short => (self.entry_price - current_price) * self.quantity,
            Direction::Neutral => 0.0,
        }
    }

    /// Calculate realized PnL when closing the position
    pub fn calculate_realized_pnl(&self, exit_price: f64, transaction_cost: f64, slippage: f64) -> f64 {
        let effective_exit_price = match self.direction {
            Direction::Long => exit_price * (1.0 - slippage),
            Direction::Short => exit_price * (1.0 + slippage),
            Direction::Neutral => exit_price,
        };

        let gross_pnl = match self.direction {
            Direction::Long => (effective_exit_price - self.entry_price) * self.quantity,
            Direction::Short => (self.entry_price - effective_exit_price) * self.quantity,
            Direction::Neutral => 0.0,
        };

        // Subtract transaction costs (entry + exit)
        let total_transaction_cost = (self.entry_price + effective_exit_price) * self.quantity * transaction_cost;
        gross_pnl - total_transaction_cost
    }

    /// Check if position should be closed due to stop loss
    pub fn should_stop_loss(&self, current_price: f64) -> bool {
        match self.direction {
            Direction::Long => current_price <= self.stop_loss_price,
            Direction::Short => current_price >= self.stop_loss_price,
            Direction::Neutral => false,
        }
    }

    /// Check if position should be closed due to take profit
    pub fn should_take_profit(&self, current_price: f64) -> bool {
        match self.direction {
            Direction::Long => current_price >= self.take_profit_price,
            Direction::Short => current_price <= self.take_profit_price,
            Direction::Neutral => false,
        }
    }

    /// Check if position has exceeded maximum holding period
    pub fn should_force_exit(&self, current_time: DateTime<Utc>, max_holding_period: Duration) -> bool {
        current_time - self.entry_time >= max_holding_period
    }

    /// Check if position has met minimum holding period
    pub fn can_exit(&self, current_time: DateTime<Utc>, min_holding_period: Duration) -> bool {
        current_time - self.entry_time >= min_holding_period
    }
}

/// Represents a completed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp
    pub exit_time: DateTime<Utc>,
    /// Direction of the trade
    pub direction: Direction,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Trade quantity
    pub quantity: f64,
    /// Realized profit/loss
    pub pnl: f64,
    /// Signal strength that triggered entry
    pub signal_strength: f32,
    /// Confidence level of the prediction
    pub confidence: f32,
    /// Holding period
    pub holding_period: Duration,
    /// Exit reason
    pub exit_reason: ExitReason,
    /// Position ID for tracking
    pub position_id: u64,
}

/// Reason for exiting a position
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ExitReason {
    /// Opposite signal triggered exit
    OppositeSignal,
    /// Stop loss triggered
    StopLoss,
    /// Take profit triggered
    TakeProfit,
    /// Maximum holding period exceeded
    MaxHoldingPeriod,
    /// End of backtest period
    EndOfData,
    /// Manual exit (for testing)
    Manual,
}

/// Point in the equity curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Total equity (cash + position value)
    pub equity: f64,
    /// Current drawdown from peak
    pub drawdown: f64,
    /// Value of open positions
    pub position_value: f64,
    /// Available cash
    pub cash: f64,
    /// Number of open positions
    pub open_positions: usize,
}

/// LDC prediction result
#[derive(Debug, Clone)]
pub struct LDCPrediction {
    /// Predicted direction
    pub prediction_direction: Direction,
    /// Signal strength (-1.0 to 1.0)
    pub signal: f32,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Distance to nearest neighbors
    pub distances: Vec<f32>,
}

/// Core backtesting engine
pub struct BacktestingEngine {
    /// Backtesting configuration
    config: BacktestConfig,
    /// LDC engine for predictions
    ldc_engine: LDCEngine,
    /// Performance calculator
    performance_calculator: PerformanceCalculator,
    /// Next position ID
    next_position_id: u64,
}

impl BacktestingEngine {
    /// Create a new backtesting engine
    pub fn new(config: BacktestConfig, ldc_config: LDCConfig) -> Self {
        Self {
            config,
            ldc_engine: LDCEngine::with_config(ldc_config),
            performance_calculator: PerformanceCalculator::new(),
            next_position_id: 1,
        }
    }

    /// Run historical backtest on OHLCV and features data
    pub fn run_backtest(
        &mut self,
        ohlcv_data: &[OHLCV],
        features_data: &[Features],
    ) -> Result<BacktestResult> {
        if ohlcv_data.len() != features_data.len() {
            return Err(anyhow::anyhow!(
                "OHLCV and features data length mismatch: {} vs {}",
                ohlcv_data.len(),
                features_data.len()
            ));
        }

        if ohlcv_data.is_empty() {
            return Err(anyhow::anyhow!("No data provided for backtesting"));
        }

        // Initialize backtest state
        let mut trades = Vec::new();
        let mut equity_curve = Vec::new();
        let mut current_equity = self.config.initial_capital;
        let mut current_cash = self.config.initial_capital;
        let mut open_positions: Vec<Position> = Vec::new();
        let mut max_equity = current_equity;
        let mut max_drawdown: f64 = 0.0;

        // Build initial training data for LDC engine
        self.build_initial_training_data(ohlcv_data, features_data)
            .context("Failed to build initial training data")?;

        println!("Starting backtest with {} data points", ohlcv_data.len());

        // Walk through historical data
        for (i, (ohlcv, features)) in ohlcv_data.iter().zip(features_data.iter()).enumerate() {
            let timestamp = DateTime::from_timestamp(ohlcv.timestamp, 0)
                .ok_or_else(|| anyhow::anyhow!("Invalid timestamp at index {}: {}", i, ohlcv.timestamp))?;

            // Convert features to FeatureSeries for LDC prediction
            let feature_series = self.convert_features_to_series(features)?;

            // Generate LDC prediction
            let prediction = self.generate_prediction(&feature_series)
                .context(format!("Failed to generate prediction at index {}", i))?;

            // Process position exits first
            let mut positions_to_close = Vec::new();
            for (pos_idx, position) in open_positions.iter().enumerate() {
                if self.should_exit_position(position, &prediction, ohlcv, timestamp) {
                    positions_to_close.push(pos_idx);
                }
            }

            // Close positions (in reverse order to maintain indices)
            for &pos_idx in positions_to_close.iter().rev() {
                let position = open_positions.remove(pos_idx);
                let exit_reason = self.determine_exit_reason(&position, &prediction, ohlcv, timestamp);
                let trade = self.close_position(position, ohlcv, timestamp, exit_reason)?;
                current_cash += self.config.initial_capital * self.config.position_size + trade.pnl;
                trades.push(trade);
            }

            // Check for new position entry
            if open_positions.len() < self.config.max_positions && self.should_enter_position(&prediction, current_cash) {
                let position = self.open_position(&prediction, ohlcv, timestamp, current_cash)?;
                current_cash -= self.config.initial_capital * self.config.position_size;
                open_positions.push(position);
            }

            // Calculate current equity and position values
            let total_position_value: f64 = open_positions
                .iter()
                .map(|p| p.calculate_unrealized_pnl(ohlcv.close) + (p.quantity * p.entry_price))
                .sum();

            let total_equity = current_cash + total_position_value;
            max_equity = max_equity.max(total_equity);
            let current_drawdown = if max_equity > 0.0 {
                (max_equity - total_equity) / max_equity
            } else {
                0.0
            };
            max_drawdown = max_drawdown.max(current_drawdown);

            // Record equity point
            equity_curve.push(EquityPoint {
                timestamp,
                equity: total_equity,
                drawdown: current_drawdown,
                position_value: total_position_value,
                cash: current_cash,
                open_positions: open_positions.len(),
            });

            // Update training data periodically (every 100 bars after initial training)
            if i > 200 && i % 100 == 0 {
                self.update_training_data(ohlcv, features, i)
                    .context(format!("Failed to update training data at index {}", i))?;
            }

            current_equity = total_equity;
        }

        // Close any remaining positions at the end
        let final_ohlcv = ohlcv_data.last().unwrap();
        let final_timestamp = DateTime::from_timestamp(final_ohlcv.timestamp, 0).unwrap();
        for position in open_positions {
            let trade = self.close_position(position, final_ohlcv, final_timestamp, ExitReason::EndOfData)?;
            current_cash += self.config.initial_capital * self.config.position_size + trade.pnl;
            trades.push(trade);
        }

        // Calculate comprehensive performance metrics
        let performance_metrics = self.performance_calculator.calculate_metrics(
            &trades,
            &equity_curve,
            self.config.initial_capital,
        );

        println!("Backtest completed: {} trades, final equity: ${:.2}", trades.len(), current_cash);

        Ok(BacktestResult {
            total_return: performance_metrics.total_returns,
            sharpe_ratio: performance_metrics.sharpe_ratio,
            max_drawdown: performance_metrics.max_drawdown,
            win_rate: performance_metrics.win_rate,
            total_trades: trades.len(),
            profitable_trades: trades.iter().filter(|t| t.pnl > 0.0).count(),
            average_trade_return: if trades.is_empty() {
                0.0
            } else {
                trades.iter().map(|t| t.pnl).sum::<f64>() / trades.len() as f64
            },
            trades,
            equity_curve,
            performance_attribution: performance_metrics.attribution,
            trade_analysis: performance_metrics.trade_analysis,
            drawdown_analysis: performance_metrics.drawdown_analysis,
        })
    }

    /// Build initial training data from the first portion of historical data
    fn build_initial_training_data(&mut self, ohlcv_data: &[OHLCV], features_data: &[Features]) -> Result<()> {
        // Use first 20% of data for initial training, minimum 100 samples, maximum 2000
        let training_size = (ohlcv_data.len() * 20 / 100).clamp(100, 2000).min(ohlcv_data.len());
        
        println!("Building initial training data with {} samples", training_size);

        for i in 0..training_size {
            if i + 4 >= ohlcv_data.len() {
                break; // Need at least 4 bars ahead for labeling
            }

            let current_features = self.convert_features_to_series(&features_data[i])?;
            
            // Look ahead 4 bars to determine label (as per Pine Script logic)
            let current_price = ohlcv_data[i].close;
            let future_price = ohlcv_data[i + 4].close;
            let price_change = (future_price - current_price) / current_price;

            // Determine label based on price change threshold
            let label = if price_change > 0.01 {
                Direction::Long
            } else if price_change < -0.01 {
                Direction::Short
            } else {
                Direction::Neutral
            };

            let training_sample = TrainingSample {
                features: current_features,
                label,
                timestamp: ohlcv_data[i].timestamp,
                bar_index: i,
            };

            let _ = self.ldc_engine.add_training_sample(training_sample);
        }

        println!("Initial training data built with {} samples", self.ldc_engine.training_samples.len());
        Ok(())
    }

    /// Convert Features struct to FeatureSeries
    fn convert_features_to_series(&self, features: &Features) -> Result<FeatureSeries> {
        Ok(FeatureSeries {
            f1: features.rsi.unwrap_or(50.0) as f32, // Default RSI to 50 if missing
            f2: features.wavetrend_1.unwrap_or(0.0) as f32, // Default WT to 0 if missing
            f3: features.cci.unwrap_or(0.0) as f32, // Default CCI to 0 if missing
            f4: features.adx.unwrap_or(25.0) as f32, // Default ADX to 25 if missing
            f5: features.momentum.unwrap_or(0.0) as f32, // Default momentum to 0 if missing
        })
    }

    /// Generate LDC prediction from features
    fn generate_prediction(&mut self, features: &FeatureSeries) -> Result<LDCPrediction> {
        // Get k-nearest neighbors from LDC engine
        let neighbors = self.ldc_engine.find_k_nearest_neighbors_optimized(features);
        
        if neighbors.is_empty() {
            return Ok(LDCPrediction {
                prediction_direction: Direction::Neutral,
                signal: 0.0,
                confidence: 0.0,
                distances: vec![],
            });
        }

        // Calculate prediction based on neighbor labels and distances
        let mut long_weight = 0.0;
        let mut short_weight = 0.0;
        let mut total_weight = 0.0;
        let distances: Vec<f32> = neighbors.iter().map(|(dist, _)| *dist).collect();

        for (distance, direction) in &neighbors {
            // Use inverse distance as weight (closer neighbors have more influence)
            let weight = 1.0 / (1.0 + distance);
            total_weight += weight;

            match direction {
                Direction::Long => long_weight += weight,
                Direction::Short => short_weight += weight,
                Direction::Neutral => {}, // Neutral doesn't contribute to directional bias
            }
        }

        // Determine prediction direction and signal strength
        let (prediction_direction, signal) = if long_weight > short_weight {
            (Direction::Long, (long_weight - short_weight) / total_weight)
        } else if short_weight > long_weight {
            (Direction::Short, -(short_weight - long_weight) / total_weight)
        } else {
            (Direction::Neutral, 0.0)
        };

        // Calculate confidence based on consistency of neighbors and distance
        let avg_distance = distances.iter().sum::<f32>() / distances.len() as f32;
        let confidence = (1.0 / (1.0 + avg_distance)).min(1.0);

        Ok(LDCPrediction {
            prediction_direction,
            signal,
            confidence,
            distances,
        })
    }

    /// Check if we should enter a new position
    fn should_enter_position(&self, prediction: &LDCPrediction, available_cash: f64) -> bool {
        // Check signal strength threshold
        if prediction.signal.abs() < self.config.signal_threshold {
            return false;
        }

        // Check confidence threshold
        if prediction.confidence < 0.3 {
            return false;
        }

        // Check if we have enough cash
        let required_capital = self.config.initial_capital * self.config.position_size;
        if available_cash < required_capital {
            return false;
        }

        // Don't enter neutral positions
        prediction.prediction_direction != Direction::Neutral
    }

    /// Check if we should exit an existing position
    fn should_exit_position(
        &self,
        position: &Position,
        prediction: &LDCPrediction,
        ohlcv: &OHLCV,
        current_time: DateTime<Utc>,
    ) -> bool {
        // Check stop loss
        if position.should_stop_loss(ohlcv.close) {
            return true;
        }

        // Check take profit
        if position.should_take_profit(ohlcv.close) {
            return true;
        }

        // Check maximum holding period
        if position.should_force_exit(current_time, self.config.max_holding_period) {
            return true;
        }

        // Check for opposite signal (only if minimum holding period is met)
        if position.can_exit(current_time, self.config.min_holding_period) {
            match (position.direction, prediction.prediction_direction) {
                (Direction::Long, Direction::Short) => {
                    return prediction.signal.abs() >= self.config.signal_threshold;
                }
                (Direction::Short, Direction::Long) => {
                    return prediction.signal.abs() >= self.config.signal_threshold;
                }
                _ => {}
            }
        }

        false
    }

    /// Determine the reason for exiting a position
    fn determine_exit_reason(
        &self,
        position: &Position,
        prediction: &LDCPrediction,
        ohlcv: &OHLCV,
        current_time: DateTime<Utc>,
    ) -> ExitReason {
        if position.should_stop_loss(ohlcv.close) {
            ExitReason::StopLoss
        } else if position.should_take_profit(ohlcv.close) {
            ExitReason::TakeProfit
        } else if position.should_force_exit(current_time, self.config.max_holding_period) {
            ExitReason::MaxHoldingPeriod
        } else {
            ExitReason::OppositeSignal
        }
    }

    /// Open a new position
    fn open_position(
        &mut self,
        prediction: &LDCPrediction,
        ohlcv: &OHLCV,
        timestamp: DateTime<Utc>,
        _available_cash: f64,
    ) -> Result<Position> {
        let position_value = self.config.initial_capital * self.config.position_size;
        let entry_price = ohlcv.close * (1.0 + self.config.slippage); // Add slippage
        let quantity = position_value / entry_price;

        let position = Position::new(
            prediction.prediction_direction,
            entry_price,
            timestamp,
            quantity,
            prediction.signal,
            prediction.confidence,
            &self.config,
            self.next_position_id,
        );

        self.next_position_id += 1;
        Ok(position)
    }

    /// Close an existing position
    fn close_position(
        &self,
        position: Position,
        ohlcv: &OHLCV,
        timestamp: DateTime<Utc>,
        exit_reason: ExitReason,
    ) -> Result<Trade> {
        let exit_price = ohlcv.close;
        let pnl = position.calculate_realized_pnl(exit_price, self.config.transaction_cost, self.config.slippage);
        let holding_period = timestamp - position.entry_time;

        Ok(Trade {
            entry_time: position.entry_time,
            exit_time: timestamp,
            direction: position.direction,
            entry_price: position.entry_price,
            exit_price,
            quantity: position.quantity,
            pnl,
            signal_strength: position.signal_strength,
            confidence: position.confidence,
            holding_period,
            exit_reason,
            position_id: position.position_id,
        })
    }

    /// Update training data with new sample
    fn update_training_data(&mut self, ohlcv: &OHLCV, features: &Features, index: usize) -> Result<()> {
        // This is a simplified version - in a real implementation, we would need
        // access to future data to properly label the training sample
        let feature_series = self.convert_features_to_series(features)?;
        
        // For now, use a simple labeling based on recent price action
        // In practice, this would use future price data that's available during backtesting
        let label = Direction::Neutral; // Placeholder - would need proper forward-looking labeling

        let training_sample = TrainingSample {
            features: feature_series,
            label,
            timestamp: ohlcv.timestamp,
            bar_index: index,
        };

        let _ = self.ldc_engine.add_training_sample(training_sample);
        Ok(())
    }
}

/// Performance calculator for backtesting results with comprehensive metrics
pub struct PerformanceCalculator {
    /// Risk-free rate for Sharpe ratio calculation (annualized)
    risk_free_rate: f64,
    /// Trading days per year for annualization
    trading_days_per_year: f64,
}

impl PerformanceCalculator {
    /// Create a new performance calculator
    pub fn new() -> Self {
        Self {
            risk_free_rate: 0.02, // 2% annual risk-free rate
            trading_days_per_year: 252.0, // Standard trading days per year
        }
    }

    /// Create with custom risk-free rate
    pub fn with_risk_free_rate(risk_free_rate: f64) -> Self {
        Self {
            risk_free_rate,
            trading_days_per_year: 252.0,
        }
    }

    /// Calculate comprehensive performance metrics
    pub fn calculate_metrics(
        &self,
        trades: &[Trade],
        equity_curve: &[EquityPoint],
        initial_capital: f64,
    ) -> PerformanceMetrics {
        let win_rate = self.calculate_win_rate(trades);
        let sharpe_ratio = self.calculate_sharpe_ratio(equity_curve, initial_capital);
        let max_drawdown = self.calculate_maximum_drawdown(equity_curve);
        let total_returns = self.calculate_total_returns(equity_curve, initial_capital);
        let attribution = self.calculate_performance_attribution(trades);
        let trade_analysis = self.calculate_trade_analysis(trades);
        let drawdown_analysis = self.calculate_drawdown_analysis(equity_curve);

        PerformanceMetrics {
            win_rate,
            sharpe_ratio,
            max_drawdown,
            total_returns,
            attribution,
            trade_analysis,
            drawdown_analysis,
        }
    }

    /// Calculate win rate (percentage of profitable trades)
    fn calculate_win_rate(&self, trades: &[Trade]) -> f64 {
        if trades.is_empty() {
            return 0.0;
        }
        trades.iter().filter(|t| t.pnl > 0.0).count() as f64 / trades.len() as f64
    }

    /// Calculate Sharpe ratio from equity curve with proper risk-free rate adjustment
    fn calculate_sharpe_ratio(&self, equity_curve: &[EquityPoint], initial_capital: f64) -> f64 {
        if equity_curve.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| {
                if w[0].equity > 0.0 {
                    (w[1].equity - w[0].equity) / w[0].equity
                } else {
                    0.0
                }
            })
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let return_variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let return_std = return_variance.sqrt();

        if return_std == 0.0 {
            return 0.0;
        }

        // Annualize the returns and adjust for risk-free rate
        let annualized_return = mean_return * self.trading_days_per_year;
        let annualized_std = return_std * self.trading_days_per_year.sqrt();
        let excess_return = annualized_return - self.risk_free_rate;

        excess_return / annualized_std
    }

    /// Calculate maximum drawdown from equity curve
    fn calculate_maximum_drawdown(&self, equity_curve: &[EquityPoint]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_equity = equity_curve[0].equity;
        let mut max_drawdown = 0.0;

        for point in equity_curve {
            if point.equity > max_equity {
                max_equity = point.equity;
            }
            let current_drawdown = if max_equity > 0.0 {
                (max_equity - point.equity) / max_equity
            } else {
                0.0
            };
            if current_drawdown > max_drawdown {
                max_drawdown = current_drawdown;
            }
        }

        max_drawdown
    }

    /// Calculate total returns from equity curve
    fn calculate_total_returns(&self, equity_curve: &[EquityPoint], initial_capital: f64) -> f64 {
        if equity_curve.is_empty() || initial_capital <= 0.0 {
            return 0.0;
        }

        let final_equity = equity_curve.last().unwrap().equity;
        (final_equity - initial_capital) / initial_capital
    }

    /// Calculate comprehensive performance attribution analysis
    fn calculate_performance_attribution(&self, trades: &[Trade]) -> PerformanceAttribution {
        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        
        // Direction-based attribution
        let long_trades: Vec<&Trade> = trades.iter().filter(|t| t.direction == Direction::Long).collect();
        let short_trades: Vec<&Trade> = trades.iter().filter(|t| t.direction == Direction::Short).collect();
        
        let long_pnl: f64 = long_trades.iter().map(|t| t.pnl).sum();
        let short_pnl: f64 = short_trades.iter().map(|t| t.pnl).sum();

        // Signal strength attribution
        let high_signal_trades: Vec<&Trade> = trades.iter().filter(|t| t.signal_strength.abs() > 0.7).collect();
        let medium_signal_trades: Vec<&Trade> = trades.iter().filter(|t| t.signal_strength.abs() > 0.3 && t.signal_strength.abs() <= 0.7).collect();
        let low_signal_trades: Vec<&Trade> = trades.iter().filter(|t| t.signal_strength.abs() <= 0.3).collect();

        let high_signal_pnl: f64 = high_signal_trades.iter().map(|t| t.pnl).sum();
        let medium_signal_pnl: f64 = medium_signal_trades.iter().map(|t| t.pnl).sum();
        let low_signal_pnl: f64 = low_signal_trades.iter().map(|t| t.pnl).sum();

        // Exit reason attribution
        let mut exit_reason_pnl = std::collections::HashMap::new();
        let mut exit_reason_counts = std::collections::HashMap::new();
        
        for trade in trades {
            let pnl_sum = exit_reason_pnl.entry(trade.exit_reason.clone()).or_insert(0.0);
            *pnl_sum += trade.pnl;
            let count = exit_reason_counts.entry(trade.exit_reason.clone()).or_insert(0);
            *count += 1;
        }

        // Holding period attribution
        let short_term_trades: Vec<&Trade> = trades.iter().filter(|t| t.holding_period.num_hours() < 4).collect();
        let medium_term_trades: Vec<&Trade> = trades.iter().filter(|t| t.holding_period.num_hours() >= 4 && t.holding_period.num_hours() < 24).collect();
        let long_term_trades: Vec<&Trade> = trades.iter().filter(|t| t.holding_period.num_hours() >= 24).collect();

        let short_term_pnl: f64 = short_term_trades.iter().map(|t| t.pnl).sum();
        let medium_term_pnl: f64 = medium_term_trades.iter().map(|t| t.pnl).sum();
        let long_term_pnl: f64 = long_term_trades.iter().map(|t| t.pnl).sum();

        PerformanceAttribution {
            total_pnl,
            long_pnl,
            short_pnl,
            long_trades: long_trades.len(),
            short_trades: short_trades.len(),
            high_signal_pnl,
            medium_signal_pnl,
            low_signal_pnl,
            high_signal_trades: high_signal_trades.len(),
            medium_signal_trades: medium_signal_trades.len(),
            low_signal_trades: low_signal_trades.len(),
            exit_reason_pnl,
            exit_reason_counts,
            short_term_pnl,
            medium_term_pnl,
            long_term_pnl,
            short_term_trades: short_term_trades.len(),
            medium_term_trades: medium_term_trades.len(),
            long_term_trades: long_term_trades.len(),
        }
    }

    /// Calculate detailed trade-by-trade analysis
    fn calculate_trade_analysis(&self, trades: &[Trade]) -> TradeAnalysis {
        if trades.is_empty() {
            return TradeAnalysis::default();
        }

        // Basic statistics
        let total_trades = trades.len();
        let profitable_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = trades.iter().filter(|t| t.pnl < 0.0).count();
        let breakeven_trades = total_trades - profitable_trades - losing_trades;

        // PnL statistics
        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let average_pnl = total_pnl / total_trades as f64;
        let profitable_pnl: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let losing_pnl: f64 = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl).sum();

        let average_winning_trade = if profitable_trades > 0 {
            profitable_pnl / profitable_trades as f64
        } else {
            0.0
        };

        let average_losing_trade = if losing_trades > 0 {
            losing_pnl / losing_trades as f64
        } else {
            0.0
        };

        // Find largest winner and loser
        let largest_winner = trades.iter().map(|t| t.pnl).fold(f64::NEG_INFINITY, f64::max);
        let largest_loser = trades.iter().map(|t| t.pnl).fold(f64::INFINITY, f64::min);

        // Profit factor (gross profit / gross loss)
        let profit_factor = if losing_pnl.abs() > 0.0 {
            profitable_pnl / losing_pnl.abs()
        } else if profitable_pnl > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Holding period analysis
        let holding_periods: Vec<i64> = trades.iter().map(|t| t.holding_period.num_minutes()).collect();
        let average_holding_period = holding_periods.iter().sum::<i64>() as f64 / total_trades as f64;
        let min_holding_period = *holding_periods.iter().min().unwrap_or(&0);
        let max_holding_period = *holding_periods.iter().max().unwrap_or(&0);

        // Signal strength correlation with PnL
        let signal_pnl_correlation = self.calculate_correlation(
            &trades.iter().map(|t| t.signal_strength as f64).collect::<Vec<_>>(),
            &trades.iter().map(|t| t.pnl).collect::<Vec<_>>(),
        );

        // Confidence correlation with PnL
        let confidence_pnl_correlation = self.calculate_correlation(
            &trades.iter().map(|t| t.confidence as f64).collect::<Vec<_>>(),
            &trades.iter().map(|t| t.pnl).collect::<Vec<_>>(),
        );

        // Consecutive wins/losses analysis
        let (max_consecutive_wins, max_consecutive_losses) = self.calculate_consecutive_streaks(trades);

        TradeAnalysis {
            total_trades,
            profitable_trades,
            losing_trades,
            breakeven_trades,
            win_rate: profitable_trades as f64 / total_trades as f64,
            total_pnl,
            average_pnl,
            profitable_pnl,
            losing_pnl,
            average_winning_trade,
            average_losing_trade,
            largest_winner,
            largest_loser,
            profit_factor,
            average_holding_period_minutes: average_holding_period,
            min_holding_period_minutes: min_holding_period,
            max_holding_period_minutes: max_holding_period,
            signal_pnl_correlation,
            confidence_pnl_correlation,
            max_consecutive_wins,
            max_consecutive_losses,
        }
    }

    /// Calculate drawdown analysis from equity curve
    fn calculate_drawdown_analysis(&self, equity_curve: &[EquityPoint]) -> DrawdownAnalysis {
        if equity_curve.len() < 2 {
            return DrawdownAnalysis::default();
        }

        let mut max_equity = equity_curve[0].equity;
        let mut current_drawdown_start: Option<DateTime<Utc>> = None;
        let mut drawdown_periods = Vec::new();
        let mut max_drawdown = 0.0;
        let mut max_drawdown_duration = Duration::zero();

        for (i, point) in equity_curve.iter().enumerate() {
            if point.equity > max_equity {
                // New equity high - end any current drawdown period
                if let Some(start_time) = current_drawdown_start {
                    let duration = point.timestamp - start_time;
                    drawdown_periods.push(DrawdownPeriod {
                        start_time,
                        end_time: point.timestamp,
                        duration,
                        max_drawdown_in_period: equity_curve[i-1].drawdown,
                        recovery_time: duration,
                    });
                    current_drawdown_start = None;
                }
                max_equity = point.equity;
            } else if point.equity < max_equity {
                // In drawdown
                if current_drawdown_start.is_none() {
                    current_drawdown_start = Some(point.timestamp);
                }
                
                let current_dd = (max_equity - point.equity) / max_equity;
                if current_dd > max_drawdown {
                    max_drawdown = current_dd;
                }
            }
        }

        // Handle case where backtest ends in drawdown
        if let Some(start_time) = current_drawdown_start {
            let end_time = equity_curve.last().unwrap().timestamp;
            let duration = end_time - start_time;
            drawdown_periods.push(DrawdownPeriod {
                start_time,
                end_time,
                duration,
                max_drawdown_in_period: equity_curve.last().unwrap().drawdown,
                recovery_time: duration, // Still in drawdown
            });
        }

        // Calculate statistics
        let average_drawdown_duration = if !drawdown_periods.is_empty() {
            drawdown_periods.iter().map(|p| p.duration.num_minutes()).sum::<i64>() as f64 / drawdown_periods.len() as f64
        } else {
            0.0
        };

        max_drawdown_duration = drawdown_periods.iter().map(|p| p.duration).max().unwrap_or(Duration::zero());

        let average_recovery_time = if !drawdown_periods.is_empty() {
            drawdown_periods.iter().map(|p| p.recovery_time.num_minutes()).sum::<i64>() as f64 / drawdown_periods.len() as f64
        } else {
            0.0
        };

        DrawdownAnalysis {
            max_drawdown,
            max_drawdown_duration_minutes: max_drawdown_duration.num_minutes(),
            average_drawdown_duration_minutes: average_drawdown_duration,
            average_recovery_time_minutes: average_recovery_time,
            drawdown_periods_count: drawdown_periods.len(),
            drawdown_periods,
        }
    }

    /// Calculate correlation between two series
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|b| b * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate maximum consecutive wins and losses
    fn calculate_consecutive_streaks(&self, trades: &[Trade]) -> (usize, usize) {
        if trades.is_empty() {
            return (0, 0);
        }

        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.pnl > 0.0 {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else if trade.pnl < 0.0 {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            } else {
                // Breakeven trade resets both streaks
                current_wins = 0;
                current_losses = 0;
            }
        }

        (max_wins, max_losses)
    }
}

/// Comprehensive performance metrics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,
    /// Sharpe ratio (risk-adjusted returns)
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Total returns
    pub total_returns: f64,
    /// Performance attribution analysis
    pub attribution: PerformanceAttribution,
    /// Detailed trade analysis
    pub trade_analysis: TradeAnalysis,
    /// Drawdown analysis
    pub drawdown_analysis: DrawdownAnalysis,
}

/// Comprehensive performance attribution breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAttribution {
    /// Total PnL across all trades
    pub total_pnl: f64,
    /// PnL from long positions
    pub long_pnl: f64,
    /// PnL from short positions
    pub short_pnl: f64,
    /// Number of long trades
    pub long_trades: usize,
    /// Number of short trades
    pub short_trades: usize,
    /// PnL from high signal strength trades (>0.7)
    pub high_signal_pnl: f64,
    /// PnL from medium signal strength trades (0.3-0.7)
    pub medium_signal_pnl: f64,
    /// PnL from low signal strength trades (<0.3)
    pub low_signal_pnl: f64,
    /// Number of high signal strength trades
    pub high_signal_trades: usize,
    /// Number of medium signal strength trades
    pub medium_signal_trades: usize,
    /// Number of low signal strength trades
    pub low_signal_trades: usize,
    /// PnL breakdown by exit reason
    pub exit_reason_pnl: std::collections::HashMap<ExitReason, f64>,
    /// Trade count breakdown by exit reason
    pub exit_reason_counts: std::collections::HashMap<ExitReason, usize>,
    /// PnL from short-term trades (<4 hours)
    pub short_term_pnl: f64,
    /// PnL from medium-term trades (4-24 hours)
    pub medium_term_pnl: f64,
    /// PnL from long-term trades (>24 hours)
    pub long_term_pnl: f64,
    /// Number of short-term trades
    pub short_term_trades: usize,
    /// Number of medium-term trades
    pub medium_term_trades: usize,
    /// Number of long-term trades
    pub long_term_trades: usize,
}

/// Detailed trade-by-trade analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeAnalysis {
    /// Total number of trades
    pub total_trades: usize,
    /// Number of profitable trades
    pub profitable_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Number of breakeven trades
    pub breakeven_trades: usize,
    /// Win rate (profitable trades / total trades)
    pub win_rate: f64,
    /// Total PnL across all trades
    pub total_pnl: f64,
    /// Average PnL per trade
    pub average_pnl: f64,
    /// Total PnL from profitable trades
    pub profitable_pnl: f64,
    /// Total PnL from losing trades
    pub losing_pnl: f64,
    /// Average PnL from winning trades
    pub average_winning_trade: f64,
    /// Average PnL from losing trades
    pub average_losing_trade: f64,
    /// Largest single winning trade
    pub largest_winner: f64,
    /// Largest single losing trade
    pub largest_loser: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average holding period in minutes
    pub average_holding_period_minutes: f64,
    /// Minimum holding period in minutes
    pub min_holding_period_minutes: i64,
    /// Maximum holding period in minutes
    pub max_holding_period_minutes: i64,
    /// Correlation between signal strength and PnL
    pub signal_pnl_correlation: f64,
    /// Correlation between confidence and PnL
    pub confidence_pnl_correlation: f64,
    /// Maximum consecutive winning trades
    pub max_consecutive_wins: usize,
    /// Maximum consecutive losing trades
    pub max_consecutive_losses: usize,
}

impl Default for TradeAnalysis {
    fn default() -> Self {
        Self {
            total_trades: 0,
            profitable_trades: 0,
            losing_trades: 0,
            breakeven_trades: 0,
            win_rate: 0.0,
            total_pnl: 0.0,
            average_pnl: 0.0,
            profitable_pnl: 0.0,
            losing_pnl: 0.0,
            average_winning_trade: 0.0,
            average_losing_trade: 0.0,
            largest_winner: 0.0,
            largest_loser: 0.0,
            profit_factor: 0.0,
            average_holding_period_minutes: 0.0,
            min_holding_period_minutes: 0,
            max_holding_period_minutes: 0,
            signal_pnl_correlation: 0.0,
            confidence_pnl_correlation: 0.0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
        }
    }
}

/// Drawdown analysis with detailed periods and recovery times
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownAnalysis {
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Duration of maximum drawdown in minutes
    pub max_drawdown_duration_minutes: i64,
    /// Average drawdown duration in minutes
    pub average_drawdown_duration_minutes: f64,
    /// Average recovery time in minutes
    pub average_recovery_time_minutes: f64,
    /// Number of drawdown periods
    pub drawdown_periods_count: usize,
    /// Detailed drawdown periods
    pub drawdown_periods: Vec<DrawdownPeriod>,
}

impl Default for DrawdownAnalysis {
    fn default() -> Self {
        Self {
            max_drawdown: 0.0,
            max_drawdown_duration_minutes: 0,
            average_drawdown_duration_minutes: 0.0,
            average_recovery_time_minutes: 0.0,
            drawdown_periods_count: 0,
            drawdown_periods: Vec::new(),
        }
    }
}

/// Individual drawdown period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownPeriod {
    /// Start time of drawdown
    pub start_time: DateTime<Utc>,
    /// End time of drawdown (recovery to new high)
    pub end_time: DateTime<Utc>,
    /// Duration of drawdown period
    pub duration: Duration,
    /// Maximum drawdown percentage during this period
    pub max_drawdown_in_period: f64,
    /// Time to recover from this drawdown
    pub recovery_time: Duration,
}

/// Complete backtesting result with comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return as a percentage
    pub total_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Number of profitable trades
    pub profitable_trades: usize,
    /// Average trade return
    pub average_trade_return: f64,
    /// All trades executed
    pub trades: Vec<Trade>,
    /// Equity curve over time
    pub equity_curve: Vec<EquityPoint>,
    /// Comprehensive performance attribution analysis
    pub performance_attribution: PerformanceAttribution,
    /// Detailed trade analysis
    pub trade_analysis: TradeAnalysis,
    /// Drawdown analysis with recovery times
    pub drawdown_analysis: DrawdownAnalysis,
}

impl BacktestResult {
    /// Print a comprehensive summary of the backtest results
    pub fn print_summary(&self) {
        println!("\n=== Comprehensive Backtest Results Summary ===");
        
        // Basic Performance Metrics
        println!("📊 Basic Performance:");
        println!("  Total Return: {:.2}%", self.total_return * 100.0);
        println!("  Sharpe Ratio: {:.2}", self.sharpe_ratio);
        println!("  Max Drawdown: {:.2}%", self.max_drawdown * 100.0);
        println!("  Win Rate: {:.2}%", self.win_rate * 100.0);
        
        // Trade Statistics
        println!("\n📈 Trade Statistics:");
        println!("  Total Trades: {}", self.total_trades);
        println!("  Profitable Trades: {} ({:.1}%)", 
                 self.profitable_trades, 
                 self.profitable_trades as f64 / self.total_trades as f64 * 100.0);
        println!("  Losing Trades: {} ({:.1}%)", 
                 self.trade_analysis.losing_trades,
                 self.trade_analysis.losing_trades as f64 / self.total_trades as f64 * 100.0);
        println!("  Breakeven Trades: {}", self.trade_analysis.breakeven_trades);
        println!("  Average Trade Return: ${:.2}", self.average_trade_return);
        println!("  Profit Factor: {:.2}", self.trade_analysis.profit_factor);
        println!("  Largest Winner: ${:.2}", self.trade_analysis.largest_winner);
        println!("  Largest Loser: ${:.2}", self.trade_analysis.largest_loser);
        
        // Direction Attribution
        println!("\n🎯 Direction Attribution:");
        println!("  Long Trades: {} (PnL: ${:.2})", 
                 self.performance_attribution.long_trades, 
                 self.performance_attribution.long_pnl);
        println!("  Short Trades: {} (PnL: ${:.2})", 
                 self.performance_attribution.short_trades, 
                 self.performance_attribution.short_pnl);
        
        // Signal Strength Attribution
        println!("\n🔍 Signal Strength Attribution:");
        println!("  High Signal (>0.7): {} trades (PnL: ${:.2})", 
                 self.performance_attribution.high_signal_trades,
                 self.performance_attribution.high_signal_pnl);
        println!("  Medium Signal (0.3-0.7): {} trades (PnL: ${:.2})", 
                 self.performance_attribution.medium_signal_trades,
                 self.performance_attribution.medium_signal_pnl);
        println!("  Low Signal (<0.3): {} trades (PnL: ${:.2})", 
                 self.performance_attribution.low_signal_trades,
                 self.performance_attribution.low_signal_pnl);
        
        // Holding Period Attribution
        println!("\n⏱️ Holding Period Attribution:");
        println!("  Short-term (<4h): {} trades (PnL: ${:.2})", 
                 self.performance_attribution.short_term_trades,
                 self.performance_attribution.short_term_pnl);
        println!("  Medium-term (4-24h): {} trades (PnL: ${:.2})", 
                 self.performance_attribution.medium_term_trades,
                 self.performance_attribution.medium_term_pnl);
        println!("  Long-term (>24h): {} trades (PnL: ${:.2})", 
                 self.performance_attribution.long_term_trades,
                 self.performance_attribution.long_term_pnl);
        println!("  Average Holding Period: {:.1} minutes", 
                 self.trade_analysis.average_holding_period_minutes);
        
        // Drawdown Analysis
        println!("\n📉 Drawdown Analysis:");
        println!("  Maximum Drawdown: {:.2}%", self.drawdown_analysis.max_drawdown * 100.0);
        println!("  Max Drawdown Duration: {:.1} hours", 
                 self.drawdown_analysis.max_drawdown_duration_minutes as f64 / 60.0);
        println!("  Average Drawdown Duration: {:.1} hours", 
                 self.drawdown_analysis.average_drawdown_duration_minutes / 60.0);
        println!("  Average Recovery Time: {:.1} hours", 
                 self.drawdown_analysis.average_recovery_time_minutes / 60.0);
        println!("  Number of Drawdown Periods: {}", 
                 self.drawdown_analysis.drawdown_periods_count);
        
        // Correlation Analysis
        println!("\n🔗 Correlation Analysis:");
        println!("  Signal Strength vs PnL: {:.3}", 
                 self.trade_analysis.signal_pnl_correlation);
        println!("  Confidence vs PnL: {:.3}", 
                 self.trade_analysis.confidence_pnl_correlation);
        
        // Streak Analysis
        println!("\n🔥 Streak Analysis:");
        println!("  Max Consecutive Wins: {}", 
                 self.trade_analysis.max_consecutive_wins);
        println!("  Max Consecutive Losses: {}", 
                 self.trade_analysis.max_consecutive_losses);
        
        // Exit Reason Breakdown
        println!("\n🚪 Exit Reason Breakdown:");
        for (reason, count) in &self.performance_attribution.exit_reason_counts {
            if let Some(pnl) = self.performance_attribution.exit_reason_pnl.get(reason) {
                println!("  {:?}: {} trades (PnL: ${:.2})", reason, count, pnl);
            }
        }
        
        println!("==============================================\n");
    }

    /// Print a detailed trade-by-trade report
    pub fn print_detailed_trades(&self, limit: Option<usize>) {
        println!("\n=== Detailed Trade Report ===");
        
        let trades_to_show = if let Some(limit) = limit {
            &self.trades[..limit.min(self.trades.len())]
        } else {
            &self.trades
        };
        
        println!("{:<20} {:<8} {:<10} {:<10} {:<10} {:<12} {:<8} {:<8} {:<15}", 
                 "Entry Time", "Dir", "Entry $", "Exit $", "PnL $", "Hold (min)", "Signal", "Conf", "Exit Reason");
        println!("{}", "-".repeat(120));
        
        for trade in trades_to_show {
            println!("{:<20} {:<8} {:<10.2} {:<10.2} {:<10.2} {:<12} {:<8.2} {:<8.2} {:<15?}", 
                     trade.entry_time.format("%Y-%m-%d %H:%M"),
                     format!("{:?}", trade.direction),
                     trade.entry_price,
                     trade.exit_price,
                     trade.pnl,
                     trade.holding_period.num_minutes(),
                     trade.signal_strength,
                     trade.confidence,
                     trade.exit_reason);
        }
        
        if let Some(limit) = limit {
            if self.trades.len() > limit {
                println!("... and {} more trades", self.trades.len() - limit);
            }
        }
        
        println!("==============================\n");
    }

    /// Generate a performance report in JSON format
    pub fn to_json_report(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow::anyhow!("Failed to serialize backtest result: {}", e))
    }

    /// Save performance report to file
    pub fn save_report(&self, file_path: &str) -> Result<()> {
        let json_report = self.to_json_report()?;
        std::fs::write(file_path, json_report)
            .map_err(|e| anyhow::anyhow!("Failed to write report to {}: {}", file_path, e))?;
        println!("Performance report saved to: {}", file_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LDCConfig, Direction};
    use feature_pipeline::{OHLCV, Features};
    use chrono::{DateTime, Utc};

    /// Create sample OHLCV data for testing
    fn create_sample_ohlcv_data(count: usize) -> Vec<OHLCV> {
        let mut data = Vec::new();
        let base_timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
        let mut price = 100.0;

        for i in 0..count {
            // Simulate some price movement
            let price_change = (i as f64 * 0.1).sin() * 2.0;
            price += price_change;
            
            let high = price + (i as f64 * 0.05).abs();
            let low = price - (i as f64 * 0.05).abs();
            let volume = 1000.0 + (i as f64 * 10.0);

            data.push(OHLCV {
                timestamp: base_timestamp + (i as i64 * 3600), // 1 hour intervals
                open: price - price_change,
                high,
                low,
                close: price,
                volume,
            });
        }

        data
    }

    /// Create sample features data for testing
    fn create_sample_features_data(ohlcv_data: &[OHLCV]) -> Vec<Features> {
        ohlcv_data
            .iter()
            .enumerate()
            .map(|(i, ohlcv)| {
                let rsi = 50.0 + (i as f64 * 0.1).sin() * 30.0; // RSI between 20-80
                let momentum = (ohlcv.close - 100.0) / 100.0; // Simple momentum
                
                Features {
                    timestamp: ohlcv.timestamp,
                    rsi: Some(rsi),
                    sma_20: Some(ohlcv.close * 0.99), // Slightly below close
                    ema_20: Some(ohlcv.close * 1.01), // Slightly above close
                    std_20: Some(2.0),
                    zscore_20: Some((ohlcv.close - 100.0) / 2.0),
                    momentum: Some(momentum),
                    wavetrend_1: Some((i as f64 * 0.2).sin() * 50.0),
                    wavetrend_2: Some((i as f64 * 0.2).cos() * 50.0),
                    cci: Some((i as f64 * 0.15).sin() * 100.0),
                    adx: Some(25.0 + (i as f64 * 0.1).abs() * 25.0),
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        
        assert_eq!(config.initial_capital, 100_000.0);
        assert_eq!(config.position_size, 0.1);
        assert_eq!(config.transaction_cost, 0.001);
        assert_eq!(config.signal_threshold, 0.5);
        assert_eq!(config.max_positions, 1);
    }

    #[test]
    fn test_position_creation() {
        let config = BacktestConfig::default();
        let entry_time = Utc::now();
        
        let position = Position::new(
            Direction::Long,
            100.0,
            entry_time,
            10.0,
            0.8,
            0.9,
            &config,
            1,
        );

        assert_eq!(position.direction, Direction::Long);
        assert_eq!(position.entry_price, 100.0);
        assert_eq!(position.quantity, 10.0);
        assert_eq!(position.signal_strength, 0.8);
        assert_eq!(position.confidence, 0.9);
        assert_eq!(position.position_id, 1);
        
        // Check stop loss and take profit prices (with floating point tolerance)
        assert!((position.stop_loss_price - 95.0).abs() < 0.01); // 5% stop loss
        assert!((position.take_profit_price - 110.0).abs() < 0.01); // 10% take profit
    }

    #[test]
    fn test_position_pnl_calculation() {
        let config = BacktestConfig::default();
        let entry_time = Utc::now();
        
        let long_position = Position::new(
            Direction::Long,
            100.0,
            entry_time,
            10.0,
            0.8,
            0.9,
            &config,
            1,
        );

        // Test unrealized PnL
        assert_eq!(long_position.calculate_unrealized_pnl(105.0), 50.0); // (105-100) * 10
        assert_eq!(long_position.calculate_unrealized_pnl(95.0), -50.0); // (95-100) * 10

        // Test realized PnL with transaction costs
        let realized_pnl = long_position.calculate_realized_pnl(105.0, 0.001, 0.0005);
        // Gross PnL: (105 * 0.9995 - 100) * 10 = 49.475
        // Transaction costs: (100 + 104.9475) * 10 * 0.001 = 2.049475
        // Net PnL: 49.475 - 2.049475 = 47.425525
        assert!((realized_pnl - 47.425525).abs() < 0.01);
    }

    #[test]
    fn test_position_exit_conditions() {
        let config = BacktestConfig::default();
        let entry_time = Utc::now();
        
        let long_position = Position::new(
            Direction::Long,
            100.0,
            entry_time,
            10.0,
            0.8,
            0.9,
            &config,
            1,
        );

        // Test stop loss
        assert!(long_position.should_stop_loss(94.0)); // Below 95.0 stop loss
        assert!(!long_position.should_stop_loss(96.0)); // Above 95.0 stop loss

        // Test take profit
        assert!(long_position.should_take_profit(111.0)); // Above 110.0 take profit
        assert!(!long_position.should_take_profit(109.0)); // Below 110.0 take profit

        // Test holding period
        let future_time = entry_time + Duration::hours(25);
        assert!(long_position.should_force_exit(future_time, Duration::hours(24)));
        
        let near_future = entry_time + Duration::minutes(10);
        assert!(!long_position.can_exit(near_future, Duration::minutes(15)));
    }

    #[test]
    fn test_backtesting_engine_creation() {
        let backtest_config = BacktestConfig::default();
        let ldc_config = LDCConfig::default();
        
        let engine = BacktestingEngine::new(backtest_config, ldc_config);
        
        // Just verify it was created successfully
        assert_eq!(engine.next_position_id, 1);
    }

    #[test]
    fn test_features_to_series_conversion() {
        let backtest_config = BacktestConfig::default();
        let ldc_config = LDCConfig::default();
        let engine = BacktestingEngine::new(backtest_config, ldc_config);

        let features = Features {
            timestamp: 1640995200,
            rsi: Some(60.0),
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: Some(0.01),
            wavetrend_1: Some(10.0),
            wavetrend_2: Some(-5.0),
            cci: Some(50.0),
            adx: Some(30.0),
        };

        let feature_series = engine.convert_features_to_series(&features).unwrap();
        
        assert_eq!(feature_series.f1, 60.0); // RSI
        assert_eq!(feature_series.f2, 10.0); // Wavetrend 1
        assert_eq!(feature_series.f3, 50.0); // CCI
        assert_eq!(feature_series.f4, 30.0); // ADX
        assert_eq!(feature_series.f5, 0.01); // Momentum
    }

    #[test]
    fn test_features_to_series_with_missing_values() {
        let backtest_config = BacktestConfig::default();
        let ldc_config = LDCConfig::default();
        let engine = BacktestingEngine::new(backtest_config, ldc_config);

        let features = Features {
            timestamp: 1640995200,
            rsi: None, // Missing RSI
            sma_20: Some(100.0),
            ema_20: Some(101.0),
            std_20: Some(2.0),
            zscore_20: Some(0.5),
            momentum: None, // Missing momentum
            wavetrend_1: None, // Missing wavetrend
            wavetrend_2: Some(-5.0),
            cci: None, // Missing CCI
            adx: Some(30.0),
        };

        let feature_series = engine.convert_features_to_series(&features).unwrap();
        
        assert_eq!(feature_series.f1, 50.0); // Default RSI
        assert_eq!(feature_series.f2, 0.0); // Default Wavetrend 1
        assert_eq!(feature_series.f3, 0.0); // Default CCI
        assert_eq!(feature_series.f4, 30.0); // ADX
        assert_eq!(feature_series.f5, 0.0); // Default momentum
    }

    #[test]
    fn test_performance_calculator() {
        let calculator = PerformanceCalculator::new();
        
        // Create sample trades
        let trades = vec![
            Trade {
                entry_time: Utc::now(),
                exit_time: Utc::now() + Duration::hours(1),
                direction: Direction::Long,
                entry_price: 100.0,
                exit_price: 105.0,
                quantity: 10.0,
                pnl: 50.0,
                signal_strength: 0.8,
                confidence: 0.9,
                holding_period: Duration::hours(1),
                exit_reason: ExitReason::TakeProfit,
                position_id: 1,
            },
            Trade {
                entry_time: Utc::now(),
                exit_time: Utc::now() + Duration::hours(2),
                direction: Direction::Short,
                entry_price: 105.0,
                exit_price: 102.0,
                quantity: 10.0,
                pnl: 30.0,
                signal_strength: 0.7,
                confidence: 0.8,
                holding_period: Duration::hours(2),
                exit_reason: ExitReason::OppositeSignal,
                position_id: 2,
            },
            Trade {
                entry_time: Utc::now(),
                exit_time: Utc::now() + Duration::hours(1),
                direction: Direction::Long,
                entry_price: 102.0,
                exit_price: 98.0,
                quantity: 10.0,
                pnl: -40.0,
                signal_strength: 0.6,
                confidence: 0.7,
                holding_period: Duration::hours(1),
                exit_reason: ExitReason::StopLoss,
                position_id: 3,
            },
        ];

        // Create sample equity curve
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc::now(),
                equity: 10000.0,
                drawdown: 0.0,
                position_value: 0.0,
                cash: 10000.0,
                open_positions: 0,
            },
            EquityPoint {
                timestamp: Utc::now() + Duration::hours(1),
                equity: 10050.0,
                drawdown: 0.0,
                position_value: 0.0,
                cash: 10050.0,
                open_positions: 0,
            },
            EquityPoint {
                timestamp: Utc::now() + Duration::hours(2),
                equity: 10080.0,
                drawdown: 0.0,
                position_value: 0.0,
                cash: 10080.0,
                open_positions: 0,
            },
            EquityPoint {
                timestamp: Utc::now() + Duration::hours(3),
                equity: 10040.0,
                drawdown: 0.004, // 0.4% drawdown
                position_value: 0.0,
                cash: 10040.0,
                open_positions: 0,
            },
        ];

        let metrics = calculator.calculate_metrics(&trades, &equity_curve, 10000.0);
        
        // Verify metrics
        assert_eq!(metrics.win_rate, 2.0 / 3.0); // 2 out of 3 trades profitable
        assert!(metrics.sharpe_ratio.is_finite());
        
        // Verify attribution
        assert_eq!(metrics.attribution.total_pnl, 40.0); // 50 + 30 - 40
        assert_eq!(metrics.attribution.long_pnl, 10.0); // 50 - 40
        assert_eq!(metrics.attribution.short_pnl, 30.0);
        assert_eq!(metrics.attribution.long_trades, 2);
        assert_eq!(metrics.attribution.short_trades, 1);
    }
}

// Include performance calculation tests
#[cfg(test)]
#[path = "backtesting_performance_tests.rs"]
mod backtesting_performance_tests;