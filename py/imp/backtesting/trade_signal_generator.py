"""
Trade signal generation logic for backtesting framework.

This module implements the TradeSignalGenerator class that converts processed
signals into buy/sell/hold decisions with configurable filtering rules and
comprehensive audit trail logging.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

import pandas as pd
import numpy as np

from .signal_processor import ProcessedSignal, SignalDirection
from .config import SignalProcessingConfig


logger = logging.getLogger(__name__)


class TradeAction(str, Enum):
    """Trade action enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class TradeSignal:
    """Generated trade signal with decision logic."""
    trade_id: str
    timestamp: datetime
    symbol: str
    action: TradeAction
    confidence: float  # 0.0 to 1.0
    signal_source: str
    reasoning: str
    processed_signal: ProcessedSignal
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate trade signal after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class TradingState:
    """Current trading state for a symbol."""
    symbol: str
    position: float = 0.0  # Positive for long, negative for short, 0 for flat
    last_trade_time: Optional[datetime] = None
    last_trade_action: Optional[TradeAction] = None
    consecutive_signals: int = 0
    signal_history: List[ProcessedSignal] = field(default_factory=list)
    
    @property
    def is_long(self) -> bool:
        """Check if currently in long position."""
        return self.position > 0
    
    @property
    def is_short(self) -> bool:
        """Check if currently in short position."""
        return self.position < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if currently flat (no position)."""
        return self.position == 0


class TradeSignalGenerator:
    """
    Generates trade signals from processed signals with filtering and audit trails.
    
    This class implements buy/sell/hold decision logic, applies configurable
    filtering rules to prevent overtrading, and maintains comprehensive audit
    trails for all signal-to-trade conversions.
    """
    
    def __init__(self, config: SignalProcessingConfig):
        """
        Initialize TradeSignalGenerator.
        
        Args:
            config: Signal processing configuration
        """
        self.config = config
        
        # Track trading state for each symbol
        self._trading_states: Dict[str, TradingState] = {}
        
        # Audit trail storage
        self._audit_trail: List[Dict[str, Any]] = []
        
        # Filter tracking
        self._daily_trade_counts: Dict[str, Dict[str, int]] = {}  # symbol -> date -> count
        self._last_signal_times: Dict[str, datetime] = {}  # symbol -> last_signal_time
        
        logger.info("Initialized TradeSignalGenerator")
    
    def generate_trade_signals(
        self,
        processed_signals: List[ProcessedSignal],
        current_positions: Optional[Dict[str, float]] = None
    ) -> List[TradeSignal]:
        """
        Generate trade signals from processed signals.
        
        Args:
            processed_signals: List of processed signals
            current_positions: Current position sizes by symbol (optional)
            
        Returns:
            List of trade signals with decisions
            
        Requirements: 2.1, 2.3, 2.5
        """
        # Update trading states with current positions first, regardless of whether we have signals
        # Always call _update_trading_states, even for empty dict, to handle position closures
        # Convert None to empty dict to ensure consistent behavior
        positions = current_positions if current_positions is not None else {}
        self._update_trading_states(positions)
        
        if not processed_signals:
            return []
        
        trade_signals = []
        
        for signal in processed_signals:
            try:
                # Get or create trading state for symbol
                state = self._get_trading_state(signal.symbol)
                
                # Apply overtrading filters
                if not self._passes_overtrading_filters(signal, state):
                    self._log_filtered_signal(signal, "overtrading_filter")
                    continue
                
                # Generate trade decision
                trade_signal = self._generate_trade_decision(signal, state)
                
                if trade_signal:
                    trade_signals.append(trade_signal)
                    
                    # Update trading state
                    self._update_state_after_signal(state, signal, trade_signal)
                    
                    # Log to audit trail
                    self._log_trade_signal(trade_signal)
                
            except Exception as e:
                logger.error(f"Error generating trade signal for {signal.symbol}: {e}")
                self._log_error(signal, str(e))
                continue
        
        logger.info(f"Generated {len(trade_signals)} trade signals from {len(processed_signals)} processed signals")
        return trade_signals
    
    def _get_trading_state(self, symbol: str) -> TradingState:
        """Get or create trading state for symbol."""
        if symbol not in self._trading_states:
            self._trading_states[symbol] = TradingState(symbol=symbol)
        return self._trading_states[symbol]
    
    def _update_trading_states(self, current_positions: Dict[str, float]) -> None:
        """
        Update trading states with current positions.
        
        Handles both position updates and position closures for symbols that
        disappear from the position snapshot.
        """
        # Get all currently tracked symbols
        tracked_symbols = set(self._trading_states.keys())
        current_symbols = set(current_positions.keys())
        
        # Update positions for symbols in the current snapshot
        for symbol, position in current_positions.items():
            state = self._get_trading_state(symbol)
            state.position = position
        
        # Zero out positions for symbols that were tracked but are missing from current snapshot
        missing_symbols = tracked_symbols - current_symbols
        for symbol in missing_symbols:
            if symbol in self._trading_states:
                self._trading_states[symbol].position = 0.0
                logger.debug(f"Zeroed out position for missing symbol: {symbol}")
    
    def _passes_overtrading_filters(
        self,
        signal: ProcessedSignal,
        state: TradingState
    ) -> bool:
        """
        Check if signal passes overtrading prevention filters.
        
        Args:
            signal: Processed signal to check
            state: Current trading state
            
        Returns:
            True if signal passes filters
            
        Requirements: 2.3
        """
        # Check maximum trades per day
        if not self._check_daily_trade_limit(signal):
            return False
        
        # Check cooldown period
        if not self._check_cooldown_period(signal):
            return False
        
        # Check consecutive signal limit
        if not self._check_consecutive_signals(signal, state):
            return False
        
        # Check signal strength requirements
        if not self._check_signal_strength_requirements(signal, state):
            return False
        
        return True
    
    def _check_daily_trade_limit(self, signal: ProcessedSignal) -> bool:
        """Check daily trade limit filter."""
        max_trades = self.config.filters.get('max_trades_per_day', 10)
        
        date_str = signal.timestamp.date().isoformat()
        symbol = signal.symbol
        
        if symbol not in self._daily_trade_counts:
            self._daily_trade_counts[symbol] = {}
        
        current_count = self._daily_trade_counts[symbol].get(date_str, 0)
        return current_count < max_trades
    
    def _check_cooldown_period(self, signal: ProcessedSignal) -> bool:
        """Check cooldown period filter."""
        cooldown_periods = self.config.filters.get('cooldown_periods', 5)
        
        if signal.symbol not in self._last_signal_times:
            return True
        
        last_time = self._last_signal_times[signal.symbol]
        # Assuming 5-minute periods, adjust as needed
        cooldown_duration = timedelta(minutes=5 * cooldown_periods)
        
        return signal.timestamp - last_time >= cooldown_duration
    
    def _check_consecutive_signals(
        self,
        signal: ProcessedSignal,
        state: TradingState
    ) -> bool:
        """Check consecutive signals limit."""
        max_consecutive = self.config.filters.get('max_consecutive_signals', 3)
        
        # Count consecutive signals in the same direction
        if (state.last_trade_action and 
            self._same_direction(signal.direction, state.last_trade_action)):
            return state.consecutive_signals < max_consecutive
        
        return True
    
    def _check_signal_strength_requirements(
        self,
        signal: ProcessedSignal,
        state: TradingState
    ) -> bool:
        """Check signal strength requirements based on current position."""
        min_strength = self.config.filters.get('min_signal_strength', 0.3)
        
        # Higher threshold for position changes
        if not state.is_flat:
            position_change_threshold = self.config.filters.get('position_change_threshold', 0.5)
            return signal.strength >= position_change_threshold
        
        return signal.strength >= min_strength
    
    def _same_direction(
        self,
        signal_direction: SignalDirection,
        trade_action: TradeAction
    ) -> bool:
        """Check if signal direction matches trade action direction."""
        if signal_direction == SignalDirection.BUY:
            return trade_action in [TradeAction.BUY]
        elif signal_direction == SignalDirection.SELL:
            return trade_action in [TradeAction.SELL]
        return False
    
    def _generate_trade_decision(
        self,
        signal: ProcessedSignal,
        state: TradingState
    ) -> Optional[TradeSignal]:
        """
        Generate trade decision based on signal and current state.
        
        Args:
            signal: Processed signal
            state: Current trading state
            
        Returns:
            Trade signal or None if no action needed
            
        Requirements: 2.1, 2.3
        """
        if signal.direction == SignalDirection.HOLD:
            return None
        
        # Determine trade action based on signal and current position
        action, reasoning = self._determine_trade_action(signal, state)
        
        if action == TradeAction.HOLD:
            return None
        
        # Calculate confidence based on signal strength and consistency
        confidence = self._calculate_confidence(signal, state)
        
        # Generate unique trade ID
        trade_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        trade_signal = TradeSignal(
            trade_id=trade_id,
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            action=action,
            confidence=confidence,
            signal_source=signal.signal_source,
            reasoning=reasoning,
            processed_signal=signal,
            metadata={
                'current_position': state.position,
                'consecutive_signals': state.consecutive_signals,
                'signal_strength': signal.strength,
                'regime_state': signal.regime_state
            }
        )
        
        return trade_signal
    
    def _determine_trade_action(
        self,
        signal: ProcessedSignal,
        state: TradingState
    ) -> Tuple[TradeAction, str]:
        """
        Determine the appropriate trade action.
        
        Args:
            signal: Processed signal
            state: Current trading state
            
        Returns:
            Tuple of (action, reasoning)
        """
        current_pos = state.position
        
        if signal.direction == SignalDirection.BUY:
            if state.is_flat:
                return TradeAction.BUY, "Open long position on buy signal"
            elif state.is_short:
                return TradeAction.CLOSE_SHORT, "Close short position on buy signal"
            elif state.is_long:
                # Already long, could add to position or hold
                if signal.strength > 0.7:  # High confidence threshold
                    return TradeAction.BUY, "Add to long position on strong buy signal"
                else:
                    return TradeAction.HOLD, "Already long, signal not strong enough to add"
        
        elif signal.direction == SignalDirection.SELL:
            if state.is_flat:
                return TradeAction.SELL, "Open short position on sell signal"
            elif state.is_long:
                return TradeAction.CLOSE_LONG, "Close long position on sell signal"
            elif state.is_short:
                # Already short, could add to position or hold
                if signal.strength > 0.7:  # High confidence threshold
                    return TradeAction.SELL, "Add to short position on strong sell signal"
                else:
                    return TradeAction.HOLD, "Already short, signal not strong enough to add"
        
        return TradeAction.HOLD, "No clear action determined"
    
    def _calculate_confidence(
        self,
        signal: ProcessedSignal,
        state: TradingState
    ) -> float:
        """
        Calculate confidence level for trade signal.
        
        Args:
            signal: Processed signal
            state: Current trading state
            
        Returns:
            Confidence level between 0.0 and 1.0
        """
        base_confidence = signal.strength
        
        # Adjust based on signal consistency
        if len(state.signal_history) > 0:
            recent_signals = state.signal_history[-3:]  # Last 3 signals
            consistent_direction = all(
                s.direction == signal.direction for s in recent_signals
            )
            if consistent_direction:
                base_confidence *= 1.1  # Boost for consistency
            else:
                base_confidence *= 0.9  # Reduce for inconsistency
        
        # Adjust based on regime confidence
        if signal.regime_state is not None and signal.regime_weights:
            # Higher confidence if regime weights are concentrated
            weight_concentration = max(signal.regime_weights.values()) if signal.regime_weights else 0.33
            regime_boost = (weight_concentration - 0.33) / 0.67  # Normalize to 0-1
            base_confidence *= (1.0 + 0.2 * regime_boost)  # Up to 20% boost
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, base_confidence))
    
    def _update_state_after_signal(
        self,
        state: TradingState,
        signal: ProcessedSignal,
        trade_signal: TradeSignal
    ) -> None:
        """Update trading state after generating trade signal."""
        # Update signal history
        state.signal_history.append(signal)
        if len(state.signal_history) > 10:  # Keep last 10 signals
            state.signal_history = state.signal_history[-10:]
        
        # Update consecutive signals count
        if (state.last_trade_action and 
            self._same_direction(signal.direction, trade_signal.action)):
            state.consecutive_signals += 1
        else:
            state.consecutive_signals = 1
        
        # Update last trade info
        state.last_trade_time = signal.timestamp
        state.last_trade_action = trade_signal.action
        
        # Update daily trade count
        date_str = signal.timestamp.date().isoformat()
        if state.symbol not in self._daily_trade_counts:
            self._daily_trade_counts[state.symbol] = {}
        
        current_count = self._daily_trade_counts[state.symbol].get(date_str, 0)
        self._daily_trade_counts[state.symbol][date_str] = current_count + 1
        
        # Update last signal time
        self._last_signal_times[state.symbol] = signal.timestamp
    
    def _log_trade_signal(self, trade_signal: TradeSignal) -> None:
        """
        Log trade signal to audit trail.
        
        Requirements: 2.5
        """
        audit_entry = {
            'timestamp': trade_signal.timestamp.isoformat(),
            'trade_id': trade_signal.trade_id,
            'symbol': trade_signal.symbol,
            'action': trade_signal.action.value,
            'confidence': trade_signal.confidence,
            'signal_source': trade_signal.signal_source,
            'reasoning': trade_signal.reasoning,
            'signal_strength': self._get_processed_signal_attr(trade_signal, 'strength'),
            'signal_direction': self._get_processed_signal_direction(trade_signal),
            'raw_signals': self._get_processed_signal_attr(trade_signal, 'raw_signals', {}),
            'regime_state': self._get_processed_signal_attr(trade_signal, 'regime_state'),
            'regime_weights': self._get_processed_signal_attr(trade_signal, 'regime_weights'),
            'metadata': trade_signal.metadata,
            'filters_applied': self.config.filters,
            'event_type': 'trade_signal_generated'
        }
        
        self._audit_trail.append(audit_entry)
        logger.debug(f"Generated trade signal: {trade_signal.action.value} {trade_signal.symbol} "
                    f"(confidence: {trade_signal.confidence:.3f})")
    
    def _get_processed_signal_attr(self, trade_signal: TradeSignal, attr_name: str, default=None):
        """
        Safely retrieve an attribute from trade_signal.processed_signal.
        
        Args:
            trade_signal: Trade signal that may or may not have processed_signal
            attr_name: Name of the attribute to retrieve
            default: Default value if attribute is not found
            
        Returns:
            Attribute value if available, default otherwise
        """
        try:
            processed_signal = getattr(trade_signal, 'processed_signal', None)
            if processed_signal is None:
                return default
            
            return getattr(processed_signal, attr_name, default)
            
        except (AttributeError, TypeError):
            return default
    
    def _get_processed_signal_direction(self, trade_signal: TradeSignal):
        """
        Safely retrieve direction.value from trade_signal.processed_signal.
        
        Args:
            trade_signal: Trade signal that may or may not have processed_signal.direction
            
        Returns:
            Direction value if available, None otherwise
        """
        try:
            processed_signal = getattr(trade_signal, 'processed_signal', None)
            if processed_signal is None:
                return None
            
            direction = getattr(processed_signal, 'direction', None)
            if direction is None:
                return None
            
            return getattr(direction, 'value', None)
            
        except (AttributeError, TypeError):
            return None
    
    def _log_filtered_signal(self, signal: ProcessedSignal, filter_reason: str) -> None:
        """Log filtered signal to audit trail."""
        audit_entry = {
            'timestamp': signal.timestamp.isoformat(),
            'symbol': signal.symbol,
            'signal_direction': signal.direction.value,
            'signal_strength': signal.strength,
            'signal_source': signal.signal_source,
            'filter_reason': filter_reason,
            'raw_signals': signal.raw_signals,
            'regime_state': signal.regime_state,
            'event_type': 'signal_filtered'
        }
        
        self._audit_trail.append(audit_entry)
        logger.debug(f"Filtered signal for {signal.symbol}: {filter_reason}")
    
    def _log_error(self, signal: ProcessedSignal, error_message: str) -> None:
        """Log error to audit trail."""
        audit_entry = {
            'timestamp': signal.timestamp.isoformat(),
            'symbol': signal.symbol,
            'error_message': error_message,
            'signal_data': {
                'direction': signal.direction.value,
                'strength': signal.strength,
                'source': signal.signal_source
            },
            'event_type': 'error'
        }
        
        self._audit_trail.append(audit_entry)
    
    def get_audit_trail(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit trail entries with optional filtering.
        
        Args:
            symbol: Filter by symbol (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            
        Returns:
            List of audit trail entries
            
        Requirements: 2.5
        """
        filtered_trail = self._audit_trail
        
        if symbol:
            filtered_trail = [
                entry for entry in filtered_trail
                if entry.get('symbol') == symbol
            ]
        
        if start_time:
            start_iso = start_time.isoformat()
            filtered_trail = [
                entry for entry in filtered_trail
                if entry.get('timestamp', '') >= start_iso
            ]
        
        if end_time:
            end_iso = end_time.isoformat()
            filtered_trail = [
                entry for entry in filtered_trail
                if entry.get('timestamp', '') <= end_iso
            ]
        
        return filtered_trail
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """
        Get trading statistics and filter effectiveness.
        
        Returns:
            Dictionary with trading statistics
        """
        total_signals = len([e for e in self._audit_trail if e.get('event_type') == 'signal_filtered']) + \
                       len([e for e in self._audit_trail if e.get('event_type') == 'trade_signal_generated'])
        
        generated_signals = len([e for e in self._audit_trail if e.get('event_type') == 'trade_signal_generated'])
        filtered_signals = len([e for e in self._audit_trail if e.get('event_type') == 'signal_filtered'])
        
        filter_reasons = {}
        for entry in self._audit_trail:
            if entry.get('event_type') == 'signal_filtered':
                reason = entry.get('filter_reason', 'unknown')
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
        
        return {
            'total_signals_processed': total_signals,
            'trade_signals_generated': generated_signals,
            'signals_filtered': filtered_signals,
            'filter_rate': filtered_signals / total_signals if total_signals > 0 else 0.0,
            'filter_reasons': filter_reasons,
            'symbols_traded': list(self._trading_states.keys()),
            'daily_trade_counts': dict(self._daily_trade_counts)
        }
    
    def reset_state(self) -> None:
        """Reset all internal state (useful for testing)."""
        self._trading_states.clear()
        self._audit_trail.clear()
        self._daily_trade_counts.clear()
        self._last_signal_times.clear()
        logger.info("Reset TradeSignalGenerator state")