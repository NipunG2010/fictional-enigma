"""
Signal processing and filtering system for backtesting framework.

This module implements the SignalProcessor class that converts raw signals
into actionable trading decisions by applying thresholds, filters, and
regime-aware fusion weights.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from .config import SignalProcessingConfig, FusionMethod
from .artifact_loader import ArtifactLoader


logger = logging.getLogger(__name__)


def _cast_regime_state(val: Any) -> Optional[int]:
    """Safely cast a raw value to an optional integer regime state."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


class SignalDirection(str, Enum):
    """Signal direction enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class ProcessedSignal:
    """Processed signal with trading decision."""
    timestamp: datetime
    symbol: str
    direction: SignalDirection
    strength: float
    signal_source: str
    raw_signals: Dict[str, float]
    regime_state: Optional[int] = None
    regime_weights: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SignalConflict:
    """Represents a signal conflict that needs resolution."""
    timestamp: datetime
    symbol: str
    conflicting_signals: List[ProcessedSignal]
    resolution: Optional[ProcessedSignal] = None
    reason: str = ""


class SignalProcessor:
    """
    Processes raw signals into actionable trading decisions.
    
    This class applies thresholds, filters, and regime-aware fusion to convert
    raw signal data into buy/sell/hold decisions with appropriate strength values.
    """
    
    def __init__(
        self,
        config: SignalProcessingConfig,
        artifact_loader: Optional[ArtifactLoader] = None
    ):
        """
        Initialize SignalProcessor.
        
        Args:
            config: Signal processing configuration
            artifact_loader: Optional artifact loader for HMM weights
        """
        self.config = config
        self.artifact_loader = artifact_loader
        
        # Track recent trades for filtering
        self._recent_trades: Dict[str, List[datetime]] = {}
        
        # Cache for HMM weights
        self._hmm_weights_cache: Optional[Dict[str, float]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)  # Cache HMM weights for 1 hour
        
        logger.info(f"Initialized SignalProcessor with fusion method: {config.fusion_method}")
    
    def process_signals(
        self,
        signal_df: pd.DataFrame,
        symbol: str,
        current_time: Optional[datetime] = None
    ) -> List[ProcessedSignal]:
        """
        Process raw signals into trading decisions.
        
        Args:
            signal_df: DataFrame with signal data
            symbol: Symbol being processed
            current_time: Current time for filtering (optional)
            
        Returns:
            List of processed signals with trading decisions
            
        Requirements: 2.1, 2.2, 2.4, 2.5
        """
        if signal_df.empty:
            logger.warning(f"Empty signal DataFrame for symbol {symbol}")
            return []
        
        processed_signals = []
        
        for _, row in signal_df.iterrows():
            try:
                # Resolve timestamp from index or column
                row_timestamp = row.name if isinstance(row.name, pd.Timestamp) else pd.to_datetime(row.get('timestamp', None))
                
                # Extract raw signals
                raw_signals = {
                    'ldc': float(row.get('s_ldc', 0.0)),
                    'mr': float(row.get('s_mr', 0.0)),
                    'tsmom': float(row.get('s_tsmom', 0.0))
                }
                
                # Get regime information if available
                regime_state = _cast_regime_state(row.get('regime_state'))
                regime_weights = self._get_regime_weights(regime_state)
                
                # Apply regime-aware fusion
                fused_signal = self._apply_fusion(raw_signals, regime_weights)
                
                # Apply thresholds to determine direction
                direction, strength = self._apply_thresholds(fused_signal, raw_signals)
                
                # Create processed signal
                signal = ProcessedSignal(
                    timestamp=row_timestamp,
                    symbol=symbol,
                    direction=direction,
                    strength=strength,
                    signal_source=self._determine_signal_source(raw_signals, regime_weights),
                    raw_signals=raw_signals,
                    regime_state=regime_state,
                    regime_weights=regime_weights,
                    metadata={
                        'fused_signal': fused_signal,
                        'fusion_method': self.config.fusion_method.value
                    }
                )
                
                # Apply filters
                if self._passes_filters(signal, current_time):
                    processed_signals.append(signal)
                    
                    # Update trade tracking for filtering
                    if direction != SignalDirection.HOLD:
                        self._update_trade_tracking(symbol, row_timestamp)
                
            except Exception as e:
                logger.error(f"Error processing signal for {symbol} at {row_timestamp}: {e}")
                continue
        
        logger.info(f"Processed {len(signal_df)} raw signals into {len(processed_signals)} trading signals for {symbol}")
        return processed_signals
    
    def _get_regime_weights(self, regime_state: Optional[int]) -> Optional[Dict[str, float]]:
        """
        Get regime-specific weights for signal fusion.
        
        Args:
            regime_state: Current regime state from HMM
            
        Returns:
            Dictionary of signal weights or None if not available
        """
        if self.config.fusion_method == FusionMethod.STATIC_WEIGHTED:
            return self.config.static_weights
        
        if regime_state is None or self.artifact_loader is None:
            logger.debug("No regime state or artifact loader available, using equal weights")
            return {'ldc': 1/3, 'mr': 1/3, 'tsmom': 1/3}
        
        # Check cache first
        if (self._hmm_weights_cache is not None and 
            self._cache_timestamp is not None and
            datetime.now() - self._cache_timestamp < self._cache_ttl):
            return self._hmm_weights_cache
        
        try:
            # Load HMM weights from artifact loader
            weights_artifact = self.artifact_loader.load_fusion_weights()
            if weights_artifact and 'weights' in weights_artifact:
                # Extract weights for the current regime state
                state_weights = weights_artifact['weights'].get(f'state_{regime_state}', {})
                if state_weights:
                    self._hmm_weights_cache = state_weights
                    self._cache_timestamp = datetime.now()
                    logger.debug(f"Loaded HMM weights for regime {regime_state}: {state_weights}")
                    return state_weights
        
        except Exception as e:
            logger.warning(f"Failed to load HMM weights: {e}")
        
        # Fallback to equal weights
        fallback_weights = {'ldc': 1/3, 'mr': 1/3, 'tsmom': 1/3}
        logger.debug(f"Using fallback equal weights for regime {regime_state}")
        return fallback_weights
    
    def _apply_fusion(
        self,
        raw_signals: Dict[str, float],
        regime_weights: Optional[Dict[str, float]]
    ) -> float:
        """
        Apply signal fusion using regime-aware weights.
        
        Args:
            raw_signals: Dictionary of raw signal values
            regime_weights: Regime-specific weights
            
        Returns:
            Fused signal value
        """
        if regime_weights is None:
            # Equal weighting fallback
            weights = {'ldc': 1/3, 'mr': 1/3, 'tsmom': 1/3}
        else:
            weights = regime_weights
        
        fused_signal = 0.0
        total_weight = 0.0
        
        for signal_name, signal_value in raw_signals.items():
            if signal_name in weights:
                weight = weights[signal_name]
                fused_signal += signal_value * weight
                total_weight += weight
        
        # Normalize by total weight to handle missing signals
        if total_weight > 0:
            fused_signal /= total_weight
        
        return fused_signal
    
    def _apply_thresholds(
        self,
        fused_signal: float,
        raw_signals: Dict[str, float]
    ) -> Tuple[SignalDirection, float]:
        """
        Apply thresholds to determine trading direction and strength.
        
        Args:
            fused_signal: Fused signal value
            raw_signals: Raw signal values for validation
            
        Returns:
            Tuple of (direction, strength)
        """
        # Check minimum signal strength
        min_strength = self.config.filters.get('min_signal_strength', 0.3)
        abs_signal = abs(fused_signal)
        
        if abs_signal < min_strength:
            return SignalDirection.HOLD, 0.0
        
        # Determine direction based on fused signal
        if fused_signal > 0:
            direction = SignalDirection.BUY
        elif fused_signal < 0:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD
        
        # Validate against individual signal thresholds
        if not self._validate_individual_thresholds(raw_signals, direction):
            return SignalDirection.HOLD, 0.0
        
        return direction, abs_signal
    
    def _validate_individual_thresholds(
        self,
        raw_signals: Dict[str, float],
        direction: SignalDirection
    ) -> bool:
        """
        Validate that individual signals meet their thresholds.
        
        Args:
            raw_signals: Raw signal values
            direction: Proposed trading direction
            
        Returns:
            True if validation passes
        """
        if direction == SignalDirection.HOLD:
            return True
        
        # Check if at least one signal exceeds its threshold in the right direction
        threshold_met = False
        
        for signal_name, threshold in self.config.thresholds.items():
            if signal_name in raw_signals:
                signal_value = raw_signals[signal_name]
                
                if direction == SignalDirection.BUY and signal_value > threshold:
                    threshold_met = True
                    break
                elif direction == SignalDirection.SELL and signal_value < -threshold:
                    threshold_met = True
                    break
        
        return threshold_met
    
    def _determine_signal_source(
        self,
        raw_signals: Dict[str, float],
        regime_weights: Optional[Dict[str, float]]
    ) -> str:
        """
        Determine the primary signal source for attribution.
        
        Args:
            raw_signals: Raw signal values
            regime_weights: Regime-specific weights
            
        Returns:
            Name of the primary signal source
        """
        if regime_weights is None:
            # Find signal with highest absolute value
            max_signal = max(raw_signals.items(), key=lambda x: abs(x[1]))
            return max_signal[0]
        
        # Find signal with highest weighted contribution
        weighted_contributions = {}
        for signal_name, signal_value in raw_signals.items():
            if signal_name in regime_weights:
                weighted_contributions[signal_name] = abs(signal_value * regime_weights[signal_name])
        
        if weighted_contributions:
            max_contribution = max(weighted_contributions.items(), key=lambda x: x[1])
            return max_contribution[0]
        
        return 'unknown'
    
    def _passes_filters(
        self,
        signal: ProcessedSignal,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Apply signal filtering rules.
        
        Args:
            signal: Processed signal to validate
            current_time: Current time for filtering
            
        Returns:
            True if signal passes all filters
        """
        if signal.direction == SignalDirection.HOLD:
            return True
        
        # Check maximum trades per day
        max_trades_per_day = self.config.filters.get('max_trades_per_day', 10)
        if self._exceeds_daily_trade_limit(signal.symbol, signal.timestamp, max_trades_per_day):
            logger.debug(f"Signal filtered: exceeds daily trade limit for {signal.symbol}")
            return False
        
        # Check cooldown period
        cooldown_periods = self.config.filters.get('cooldown_periods', 5)
        if self._in_cooldown_period(signal.symbol, signal.timestamp, cooldown_periods):
            logger.debug(f"Signal filtered: in cooldown period for {signal.symbol}")
            return False
        
        return True
    
    def _exceeds_daily_trade_limit(
        self,
        symbol: str,
        timestamp: datetime,
        max_trades: int
    ) -> bool:
        """Check if daily trade limit would be exceeded."""
        if symbol not in self._recent_trades:
            return False
        
        # Count trades on the same day
        day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        daily_trades = [
            t for t in self._recent_trades[symbol]
            if day_start <= t < day_end
        ]
        
        return len(daily_trades) >= max_trades
    
    def _in_cooldown_period(
        self,
        symbol: str,
        timestamp: datetime,
        cooldown_periods: int
    ) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol not in self._recent_trades or not self._recent_trades[symbol]:
            return False
        
        # Check if last trade was within cooldown period
        last_trade = max(self._recent_trades[symbol])
        # Assuming 5-minute periods, adjust as needed
        cooldown_duration = timedelta(minutes=5 * cooldown_periods)
        
        return timestamp - last_trade < cooldown_duration
    
    def _update_trade_tracking(self, symbol: str, timestamp: datetime) -> None:
        """Update trade tracking for filtering."""
        if symbol not in self._recent_trades:
            self._recent_trades[symbol] = []
        
        self._recent_trades[symbol].append(timestamp)
        
        # Keep only recent trades (last 24 hours)
        cutoff = timestamp - timedelta(hours=24)
        self._recent_trades[symbol] = [
            t for t in self._recent_trades[symbol] if t > cutoff
        ]
    
    def resolve_conflicts(
        self,
        signals: List[ProcessedSignal]
    ) -> List[ProcessedSignal]:
        """
        Resolve signal conflicts using configured rules.
        
        Args:
            signals: List of potentially conflicting signals
            
        Returns:
            List of resolved signals
            
        Requirements: 2.4, 2.5
        """
        if len(signals) <= 1:
            return signals
        
        # Group signals by symbol and timestamp
        signal_groups = {}
        for signal in signals:
            key = (signal.symbol, signal.timestamp)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        resolved_signals = []
        conflicts = []
        
        for (symbol, timestamp), group_signals in signal_groups.items():
            if len(group_signals) == 1:
                resolved_signals.extend(group_signals)
                continue
            
            # Check for conflicts (opposing directions)
            directions = {s.direction for s in group_signals}
            
            if len(directions) > 1 and SignalDirection.HOLD not in directions:
                # We have a conflict between BUY and SELL
                conflict = SignalConflict(
                    timestamp=timestamp,
                    symbol=symbol,
                    conflicting_signals=group_signals
                )
                
                # Resolve by choosing signal with highest strength
                best_signal = max(group_signals, key=lambda s: s.strength)
                conflict.resolution = best_signal
                conflict.reason = f"Chose signal with highest strength: {best_signal.strength:.3f}"
                
                conflicts.append(conflict)
                resolved_signals.append(best_signal)
                
                logger.warning(f"Resolved signal conflict for {symbol} at {timestamp}: {conflict.reason}")
            else:
                # No conflict or only HOLD signals
                # Choose the strongest non-HOLD signal, or HOLD if all are HOLD
                non_hold_signals = [s for s in group_signals if s.direction != SignalDirection.HOLD]
                if non_hold_signals:
                    best_signal = max(non_hold_signals, key=lambda s: s.strength)
                    resolved_signals.append(best_signal)
                else:
                    # All are HOLD, just take the first one
                    resolved_signals.append(group_signals[0])
        
        if conflicts:
            logger.info(f"Resolved {len(conflicts)} signal conflicts")
        
        return resolved_signals
    
    def validate_signal_strength(
        self,
        signals: List[ProcessedSignal]
    ) -> List[ProcessedSignal]:
        """
        Validate and adjust signal strengths.
        
        Args:
            signals: List of signals to validate
            
        Returns:
            List of validated signals
            
        Requirements: 2.5
        """
        validated_signals = []
        
        for signal in signals:
            # Ensure strength is within valid range [0, 1]
            adjusted_strength = max(0.0, min(1.0, signal.strength))
            
            if adjusted_strength != signal.strength:
                logger.debug(f"Adjusted signal strength for {signal.symbol} from {signal.strength:.3f} to {adjusted_strength:.3f}")
                
                # Create new signal with adjusted strength
                validated_signal = ProcessedSignal(
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    direction=signal.direction,
                    strength=adjusted_strength,
                    signal_source=signal.signal_source,
                    raw_signals=signal.raw_signals,
                    regime_state=signal.regime_state,
                    regime_weights=signal.regime_weights,
                    metadata={**signal.metadata, 'strength_adjusted': True}
                )
                validated_signals.append(validated_signal)
            else:
                validated_signals.append(signal)
        
        return validated_signals
    
    def get_audit_trail(
        self,
        signals: List[ProcessedSignal]
    ) -> List[Dict[str, Any]]:
        """
        Generate audit trail for signal processing.
        
        Args:
            signals: List of processed signals
            
        Returns:
            List of audit trail entries
            
        Requirements: 2.5
        """
        audit_trail = []
        
        for signal in signals:
            audit_entry = {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'direction': signal.direction.value,
                'strength': signal.strength,
                'signal_source': signal.signal_source,
                'raw_signals': signal.raw_signals,
                'regime_state': signal.regime_state,
                'regime_weights': signal.regime_weights,
                'fusion_method': self.config.fusion_method.value,
                'thresholds_applied': self.config.thresholds,
                'filters_applied': self.config.filters,
                'metadata': signal.metadata
            }
            audit_trail.append(audit_entry)
        
        return audit_trail