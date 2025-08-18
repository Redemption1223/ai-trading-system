"""
Multi-Timeframe Ensemble Trading Engine
Purpose: Coordinate multiple trading strategies across different timeframes for optimal signal generation

Features:
- Multi-timeframe analysis (M1, M5, M15, H1, H4, D1)
- Ensemble voting mechanisms
- Dynamic weight adjustment based on performance
- Cross-timeframe confirmation systems
- Advanced signal correlation analysis
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

class Timeframe(Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

class VotingMethod(Enum):
    SIMPLE_MAJORITY = "SIMPLE_MAJORITY"
    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"
    CONFIDENCE_BASED = "CONFIDENCE_BASED"
    PERFORMANCE_WEIGHTED = "PERFORMANCE_WEIGHTED"
    ADAPTIVE_CONSENSUS = "ADAPTIVE_CONSENSUS"

@dataclass
class TimeframeSignal:
    timeframe: Timeframe
    symbol: str
    direction: str  # BUY/SELL/HOLD
    confidence: float
    strength: float
    strategy_name: str
    timestamp: datetime
    indicators: Dict[str, float]
    market_conditions: Dict[str, Any]

@dataclass
class EnsembleSignal:
    symbol: str
    final_direction: str
    ensemble_confidence: float
    ensemble_strength: float
    contributing_signals: List[TimeframeSignal]
    voting_method: str
    timestamp: datetime
    correlation_analysis: Dict[str, float]

class MultiTimeframeEnsemble:
    """Multi-timeframe ensemble trading engine"""
    
    def __init__(self, symbols: List[str] = None):
        self.name = "ENSEMBLE_TRADING_ENGINE"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        
        # Configuration
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY']
        self.active_timeframes = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
        
        # Ensemble configuration
        self.voting_method = VotingMethod.ADAPTIVE_CONSENSUS
        self.min_signals_required = 2
        self.correlation_threshold = 0.3
        
        # Timeframe weights (dynamically adjusted)
        self.timeframe_weights = {
            Timeframe.M1: 0.1,
            Timeframe.M5: 0.15,
            Timeframe.M15: 0.2,
            Timeframe.H1: 0.25,
            Timeframe.H4: 0.2,
            Timeframe.D1: 0.1
        }
        
        # Strategy registry
        self.strategies = {}
        self.strategy_performance = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'avg_confidence': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0
        })
        
        # Signal storage
        self.timeframe_signals = defaultdict(lambda: deque(maxlen=100))
        self.ensemble_signals = deque(maxlen=200)
        self.signal_correlation_matrix = {}
        
        # Performance tracking
        self.ensemble_performance = {
            'signals_generated': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'avg_ensemble_confidence': 0.0,
            'best_performing_timeframe': None,
            'worst_performing_timeframe': None
        }
        
        # Real-time processing
        self.processing_thread = None
        self.is_processing = False
        self.processing_interval = 30  # seconds
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize the ensemble trading engine"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Initialize correlation matrix
            self._initialize_correlation_matrix()
            
            # Setup voting mechanisms
            self._setup_voting_mechanisms()
            
            # Initialize adaptive weight system
            self._initialize_adaptive_weights()
            
            self.status = "INITIALIZED"
            self.logger.info("Ensemble Trading Engine initialized successfully")
            
            return {
                "status": "initialized",
                "agent": "ENSEMBLE_ENGINE",
                "active_timeframes": [tf.value for tf in self.active_timeframes],
                "symbols": self.symbols,
                "voting_method": self.voting_method.value,
                "strategies_registered": len(self.strategies)
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "error": str(e)}
    
    def register_strategy(self, strategy_name: str, timeframes: List[Timeframe], 
                         weight: float = 1.0, enabled: bool = True):
        """Register a trading strategy for specific timeframes"""
        try:
            self.strategies[strategy_name] = {
                'timeframes': timeframes,
                'weight': weight,
                'enabled': enabled,
                'last_signal_time': None,
                'total_signals': 0,
                'performance_score': 0.5  # Initial neutral score
            }
            
            self.logger.info(f"Registered strategy: {strategy_name} for timeframes: {[tf.value for tf in timeframes]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Strategy registration failed: {e}")
            return False
    
    def add_timeframe_signal(self, signal: TimeframeSignal):
        """Add a signal from a specific timeframe strategy"""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return False
            
            # Store signal by timeframe and symbol
            key = f"{signal.timeframe.value}_{signal.symbol}"
            self.timeframe_signals[key].append(signal)
            
            # Update strategy performance tracking
            self._update_strategy_tracking(signal)
            
            # Trigger ensemble processing
            self._trigger_ensemble_processing(signal.symbol)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding timeframe signal: {e}")
            return False
    
    def generate_ensemble_signal(self, symbol: str) -> Optional[EnsembleSignal]:
        """Generate ensemble signal for a symbol using all available timeframe signals"""
        try:
            # Collect recent signals for this symbol
            recent_signals = self._collect_recent_signals(symbol)
            
            if len(recent_signals) < self.min_signals_required:
                return None
            
            # Perform correlation analysis
            correlation_analysis = self._analyze_signal_correlation(recent_signals)
            
            # Apply voting mechanism
            ensemble_result = self._apply_voting_mechanism(recent_signals, correlation_analysis)
            
            if not ensemble_result:
                return None
            
            # Create ensemble signal
            ensemble_signal = EnsembleSignal(
                symbol=symbol,
                final_direction=ensemble_result['direction'],
                ensemble_confidence=ensemble_result['confidence'],
                ensemble_strength=ensemble_result['strength'],
                contributing_signals=recent_signals,
                voting_method=self.voting_method.value,
                timestamp=datetime.now(),
                correlation_analysis=correlation_analysis
            )
            
            # Store ensemble signal
            self.ensemble_signals.append(ensemble_signal)
            
            # Update performance metrics
            self._update_ensemble_performance(ensemble_signal)
            
            self.logger.info(f"Generated ensemble signal: {symbol} {ensemble_result['direction']} "
                           f"(confidence: {ensemble_result['confidence']:.2f})")
            
            return ensemble_signal
            
        except Exception as e:
            self.logger.error(f"Ensemble signal generation failed: {e}")
            return None
    
    def _collect_recent_signals(self, symbol: str, lookback_minutes: int = 60) -> List[TimeframeSignal]:
        """Collect recent signals for a symbol from all timeframes"""
        recent_signals = []
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        
        for timeframe in self.active_timeframes:
            key = f"{timeframe.value}_{symbol}"
            if key in self.timeframe_signals:
                for signal in self.timeframe_signals[key]:
                    if signal.timestamp >= cutoff_time:
                        recent_signals.append(signal)
        
        # Sort by timestamp (most recent first)
        recent_signals.sort(key=lambda x: x.timestamp, reverse=True)
        
        return recent_signals
    
    def _analyze_signal_correlation(self, signals: List[TimeframeSignal]) -> Dict[str, float]:
        """Analyze correlation between signals from different timeframes"""
        try:
            correlation_analysis = {
                'directional_agreement': 0.0,
                'confidence_coherence': 0.0,
                'strength_alignment': 0.0,
                'temporal_consistency': 0.0
            }
            
            if len(signals) < 2:
                return correlation_analysis
            
            # Directional agreement
            buy_count = sum(1 for s in signals if s.direction == 'BUY')
            sell_count = sum(1 for s in signals if s.direction == 'SELL')
            total_directional = buy_count + sell_count
            
            if total_directional > 0:
                correlation_analysis['directional_agreement'] = max(buy_count, sell_count) / total_directional
            
            # Confidence coherence (low variance in confidence = higher coherence)
            confidences = [s.confidence for s in signals if s.direction != 'HOLD']
            if confidences:
                confidence_std = np.std(confidences)
                correlation_analysis['confidence_coherence'] = max(0, 1 - confidence_std)
            
            # Strength alignment
            strengths = [s.strength for s in signals if s.direction != 'HOLD']
            if strengths:
                strength_std = np.std(strengths)
                correlation_analysis['strength_alignment'] = max(0, 1 - (strength_std / 100))
            
            # Temporal consistency (signals closer in time are more consistent)
            if len(signals) >= 2:
                time_diffs = []
                for i in range(len(signals) - 1):
                    diff = abs((signals[i].timestamp - signals[i+1].timestamp).total_seconds())
                    time_diffs.append(diff)
                
                avg_time_diff = np.mean(time_diffs)
                # Normalize: closer signals (lower diff) = higher consistency
                correlation_analysis['temporal_consistency'] = max(0, 1 - (avg_time_diff / 3600))  # 1 hour normalization
            
            return correlation_analysis
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            return correlation_analysis
    
    def _apply_voting_mechanism(self, signals: List[TimeframeSignal], 
                               correlation_analysis: Dict[str, float]) -> Optional[Dict]:
        """Apply the selected voting mechanism to determine ensemble decision"""
        try:
            if self.voting_method == VotingMethod.SIMPLE_MAJORITY:
                return self._simple_majority_vote(signals)
            elif self.voting_method == VotingMethod.WEIGHTED_AVERAGE:
                return self._weighted_average_vote(signals)
            elif self.voting_method == VotingMethod.CONFIDENCE_BASED:
                return self._confidence_based_vote(signals)
            elif self.voting_method == VotingMethod.PERFORMANCE_WEIGHTED:
                return self._performance_weighted_vote(signals)
            elif self.voting_method == VotingMethod.ADAPTIVE_CONSENSUS:
                return self._adaptive_consensus_vote(signals, correlation_analysis)
            else:
                return self._simple_majority_vote(signals)  # Default fallback
                
        except Exception as e:
            self.logger.error(f"Voting mechanism failed: {e}")
            return None
    
    def _simple_majority_vote(self, signals: List[TimeframeSignal]) -> Optional[Dict]:
        """Simple majority voting"""
        vote_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for signal in signals:
            vote_counts[signal.direction] += 1
        
        # Find majority
        max_votes = max(vote_counts.values())
        if max_votes == 0:
            return None
        
        winning_direction = [k for k, v in vote_counts.items() if v == max_votes][0]
        
        if winning_direction == 'HOLD':
            return None
        
        # Calculate ensemble metrics
        contributing_signals = [s for s in signals if s.direction == winning_direction]
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        avg_strength = np.mean([s.strength for s in contributing_signals])
        
        return {
            'direction': winning_direction,
            'confidence': avg_confidence,
            'strength': avg_strength
        }
    
    def _weighted_average_vote(self, signals: List[TimeframeSignal]) -> Optional[Dict]:
        """Weighted average voting based on timeframe weights"""
        vote_weights = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        for signal in signals:
            weight = self.timeframe_weights.get(signal.timeframe, 1.0)
            vote_weights[signal.direction] += weight * signal.confidence
        
        # Find weighted winner
        max_weight = max(vote_weights.values())
        if max_weight == 0:
            return None
        
        winning_direction = [k for k, v in vote_weights.items() if v == max_weight][0]
        
        if winning_direction == 'HOLD':
            return None
        
        # Calculate weighted ensemble metrics
        contributing_signals = [s for s in signals if s.direction == winning_direction]
        weighted_confidence = sum(s.confidence * self.timeframe_weights.get(s.timeframe, 1.0) 
                                for s in contributing_signals)
        weighted_confidence /= sum(self.timeframe_weights.get(s.timeframe, 1.0) 
                                 for s in contributing_signals)
        
        weighted_strength = sum(s.strength * self.timeframe_weights.get(s.timeframe, 1.0) 
                              for s in contributing_signals)
        weighted_strength /= sum(self.timeframe_weights.get(s.timeframe, 1.0) 
                               for s in contributing_signals)
        
        return {
            'direction': winning_direction,
            'confidence': weighted_confidence,
            'strength': weighted_strength
        }
    
    def _confidence_based_vote(self, signals: List[TimeframeSignal]) -> Optional[Dict]:
        """Confidence-based voting - higher confidence signals have more weight"""
        confidence_weights = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        for signal in signals:
            confidence_weights[signal.direction] += signal.confidence ** 2  # Square for emphasis
        
        max_weight = max(confidence_weights.values())
        if max_weight == 0:
            return None
        
        winning_direction = [k for k, v in confidence_weights.items() if v == max_weight][0]
        
        if winning_direction == 'HOLD':
            return None
        
        contributing_signals = [s for s in signals if s.direction == winning_direction]
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        avg_strength = np.mean([s.strength for s in contributing_signals])
        
        return {
            'direction': winning_direction,
            'confidence': avg_confidence,
            'strength': avg_strength
        }
    
    def _performance_weighted_vote(self, signals: List[TimeframeSignal]) -> Optional[Dict]:
        """Performance-weighted voting based on strategy historical performance"""
        performance_weights = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        for signal in signals:
            strategy_perf = self.strategy_performance.get(signal.strategy_name, {})
            perf_score = strategy_perf.get('win_rate', 0.5)  # Default neutral performance
            
            performance_weights[signal.direction] += perf_score * signal.confidence
        
        max_weight = max(performance_weights.values())
        if max_weight == 0:
            return None
        
        winning_direction = [k for k, v in performance_weights.items() if v == max_weight][0]
        
        if winning_direction == 'HOLD':
            return None
        
        contributing_signals = [s for s in signals if s.direction == winning_direction]
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        avg_strength = np.mean([s.strength for s in contributing_signals])
        
        return {
            'direction': winning_direction,
            'confidence': avg_confidence,
            'strength': avg_strength
        }
    
    def _adaptive_consensus_vote(self, signals: List[TimeframeSignal], 
                               correlation_analysis: Dict[str, float]) -> Optional[Dict]:
        """Adaptive consensus voting that considers correlation and adapts weights"""
        try:
            # Base weights from multiple factors
            vote_scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
            
            # Overall correlation strength
            correlation_strength = np.mean(list(correlation_analysis.values()))
            
            for signal in signals:
                # Get base weight factors
                timeframe_weight = self.timeframe_weights.get(signal.timeframe, 1.0)
                strategy_perf = self.strategy_performance.get(signal.strategy_name, {})
                performance_weight = strategy_perf.get('win_rate', 0.5)
                
                # Adaptive weight calculation
                adaptive_weight = (
                    timeframe_weight * 0.3 +
                    performance_weight * 0.3 +
                    signal.confidence * 0.2 +
                    correlation_strength * 0.2
                )
                
                # Apply directional agreement bonus
                directional_agreement = correlation_analysis.get('directional_agreement', 0.5)
                if directional_agreement > 0.7:  # Strong agreement
                    adaptive_weight *= 1.2
                
                vote_scores[signal.direction] += adaptive_weight * signal.strength
            
            # Find adaptive consensus winner
            max_score = max(vote_scores.values())
            if max_score == 0:
                return None
            
            winning_direction = [k for k, v in vote_scores.items() if v == max_score][0]
            
            if winning_direction == 'HOLD':
                return None
            
            # Calculate adaptive ensemble metrics
            contributing_signals = [s for s in signals if s.direction == winning_direction]
            
            # Confidence adjusted by correlation
            base_confidence = np.mean([s.confidence for s in contributing_signals])
            adjusted_confidence = base_confidence * (0.5 + 0.5 * correlation_strength)
            
            # Strength adjusted by signal count and agreement
            base_strength = np.mean([s.strength for s in contributing_signals])
            count_bonus = min(0.2, len(contributing_signals) * 0.05)  # Bonus for more signals
            adjusted_strength = base_strength * (1 + count_bonus)
            
            return {
                'direction': winning_direction,
                'confidence': min(1.0, adjusted_confidence),
                'strength': min(100.0, adjusted_strength)
            }
            
        except Exception as e:
            self.logger.error(f"Adaptive consensus voting failed: {e}")
            return self._simple_majority_vote(signals)  # Fallback
    
    def update_strategy_performance(self, strategy_name: str, trade_result: Dict):
        """Update strategy performance metrics"""
        try:
            if strategy_name not in self.strategy_performance:
                return
            
            perf = self.strategy_performance[strategy_name]
            
            # Update basic metrics
            perf['total_trades'] += 1
            
            if trade_result.get('profit_loss', 0) > 0:
                perf['winning_trades'] += 1
            
            perf['total_profit'] += trade_result.get('profit_loss', 0)
            
            # Calculate derived metrics
            perf['win_rate'] = perf['winning_trades'] / max(1, perf['total_trades'])
            
            # Update drawdown
            if trade_result.get('profit_loss', 0) < 0:
                perf['max_drawdown'] = max(perf['max_drawdown'], 
                                         abs(trade_result.get('profit_loss', 0)))
            
            # Update strategy weight in registry
            if strategy_name in self.strategies:
                self.strategies[strategy_name]['performance_score'] = perf['win_rate']
            
            self.logger.debug(f"Updated performance for {strategy_name}: "
                            f"Win rate: {perf['win_rate']:.2%}")
            
        except Exception as e:
            self.logger.error(f"Performance update failed: {e}")
    
    def start_real_time_processing(self) -> Dict:
        """Start real-time ensemble processing"""
        if self.is_processing:
            return {"status": "already_running"}
        
        try:
            self.is_processing = True
            
            def processing_loop():
                self.logger.info("Starting real-time ensemble processing")
                
                while self.is_processing:
                    try:
                        # Process ensemble signals for all symbols
                        for symbol in self.symbols:
                            ensemble_signal = self.generate_ensemble_signal(symbol)
                            if ensemble_signal:
                                self.logger.debug(f"Generated ensemble signal for {symbol}")
                        
                        # Update adaptive weights
                        self._update_adaptive_weights()
                        
                        # Clean old signals
                        self._cleanup_old_signals()
                        
                        time.sleep(self.processing_interval)
                        
                    except Exception as e:
                        self.logger.error(f"Processing loop error: {e}")
                        time.sleep(10)
            
            self.processing_thread = threading.Thread(target=processing_loop, daemon=True)
            self.processing_thread.start()
            
            return {"status": "started"}
            
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            self.is_processing = False
            return {"status": "failed", "error": str(e)}
    
    def stop_real_time_processing(self) -> Dict:
        """Stop real-time ensemble processing"""
        try:
            self.is_processing = False
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=10)
            
            return {"status": "stopped"}
            
        except Exception as e:
            self.logger.error(f"Error stopping processing: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_ensemble_status(self) -> Dict:
        """Get current ensemble status and statistics"""
        try:
            # Calculate timeframe signal counts
            timeframe_counts = {}
            for timeframe in self.active_timeframes:
                count = 0
                for symbol in self.symbols:
                    key = f"{timeframe.value}_{symbol}"
                    if key in self.timeframe_signals:
                        count += len(self.timeframe_signals[key])
                timeframe_counts[timeframe.value] = count
            
            # Find best performing strategy
            best_strategy = None
            best_performance = 0
            for strategy_name, perf in self.strategy_performance.items():
                if perf['win_rate'] > best_performance:
                    best_performance = perf['win_rate']
                    best_strategy = strategy_name
            
            return {
                'status': self.status,
                'is_processing': self.is_processing,
                'active_timeframes': [tf.value for tf in self.active_timeframes],
                'voting_method': self.voting_method.value,
                'strategies_registered': len(self.strategies),
                'timeframe_signal_counts': timeframe_counts,
                'ensemble_signals_generated': len(self.ensemble_signals),
                'best_performing_strategy': best_strategy,
                'best_strategy_performance': best_performance,
                'ensemble_performance': self.ensemble_performance
            }
            
        except Exception as e:
            self.logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}
    
    def get_recent_ensemble_signals(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get recent ensemble signals"""
        try:
            signals = list(self.ensemble_signals)
            
            if symbol:
                signals = [s for s in signals if s.symbol == symbol]
            
            # Sort by timestamp (most recent first)
            signals.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Convert to dict format
            result = []
            for signal in signals[:limit]:
                result.append({
                    'symbol': signal.symbol,
                    'direction': signal.final_direction,
                    'confidence': signal.ensemble_confidence,
                    'strength': signal.ensemble_strength,
                    'voting_method': signal.voting_method,
                    'timestamp': signal.timestamp.isoformat(),
                    'contributing_signals_count': len(signal.contributing_signals),
                    'correlation_analysis': signal.correlation_analysis
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Recent signals retrieval failed: {e}")
            return []
    
    def shutdown(self):
        """Clean shutdown of ensemble engine"""
        try:
            self.logger.info("Shutting down Ensemble Trading Engine...")
            
            # Stop processing
            self.stop_real_time_processing()
            
            # Clear signal storage
            self.timeframe_signals.clear()
            self.ensemble_signals.clear()
            
            self.status = "SHUTDOWN"
            self.logger.info("Ensemble Trading Engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    # Helper methods
    
    def _validate_signal(self, signal: TimeframeSignal) -> bool:
        """Validate incoming timeframe signal"""
        try:
            if not isinstance(signal.timeframe, Timeframe):
                return False
            if signal.direction not in ['BUY', 'SELL', 'HOLD']:
                return False
            if not 0 <= signal.confidence <= 1:
                return False
            if not 0 <= signal.strength <= 100:
                return False
            return True
        except:
            return False
    
    def _update_strategy_tracking(self, signal: TimeframeSignal):
        """Update strategy tracking metrics"""
        strategy_name = signal.strategy_name
        if strategy_name in self.strategies:
            self.strategies[strategy_name]['last_signal_time'] = signal.timestamp
            self.strategies[strategy_name]['total_signals'] += 1
    
    def _trigger_ensemble_processing(self, symbol: str):
        """Trigger ensemble processing for a symbol (if not in real-time mode)"""
        if not self.is_processing:
            # Manual trigger for single symbol
            ensemble_signal = self.generate_ensemble_signal(symbol)
            if ensemble_signal:
                self.logger.debug(f"Manual ensemble signal generated for {symbol}")
    
    def _initialize_correlation_matrix(self):
        """Initialize signal correlation tracking matrix"""
        self.signal_correlation_matrix = {}
        for tf1 in self.active_timeframes:
            for tf2 in self.active_timeframes:
                if tf1 != tf2:
                    key = f"{tf1.value}_{tf2.value}"
                    self.signal_correlation_matrix[key] = {
                        'correlation_coefficient': 0.0,
                        'sample_count': 0,
                        'last_updated': datetime.now()
                    }
    
    def _setup_voting_mechanisms(self):
        """Setup voting mechanism configurations"""
        self.voting_configs = {
            VotingMethod.SIMPLE_MAJORITY: {'min_signals': 2},
            VotingMethod.WEIGHTED_AVERAGE: {'min_signals': 2, 'use_timeframe_weights': True},
            VotingMethod.CONFIDENCE_BASED: {'min_signals': 2, 'confidence_threshold': 0.6},
            VotingMethod.PERFORMANCE_WEIGHTED: {'min_signals': 2, 'min_performance_history': 10},
            VotingMethod.ADAPTIVE_CONSENSUS: {'min_signals': 2, 'adaptation_rate': 0.1}
        }
    
    def _initialize_adaptive_weights(self):
        """Initialize adaptive weight adjustment system"""
        self.weight_adaptation = {
            'learning_rate': 0.05,
            'performance_window': 50,  # Number of recent signals to consider
            'min_weight': 0.05,
            'max_weight': 0.4
        }
    
    def _update_adaptive_weights(self):
        """Update timeframe weights based on recent performance"""
        try:
            if len(self.ensemble_signals) < 10:
                return  # Need minimum history
            
            # Analyze recent performance by timeframe
            recent_signals = list(self.ensemble_signals)[-self.weight_adaptation['performance_window']:]
            
            timeframe_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
            
            for ensemble_signal in recent_signals:
                for tf_signal in ensemble_signal.contributing_signals:
                    if tf_signal.direction == ensemble_signal.final_direction:
                        timeframe_performance[tf_signal.timeframe]['correct'] += 1
                    timeframe_performance[tf_signal.timeframe]['total'] += 1
            
            # Update weights based on performance
            for timeframe in self.active_timeframes:
                if timeframe in timeframe_performance:
                    perf = timeframe_performance[timeframe]
                    if perf['total'] > 0:
                        accuracy = perf['correct'] / perf['total']
                        
                        # Adaptive weight adjustment
                        current_weight = self.timeframe_weights[timeframe]
                        target_weight = accuracy  # Target weight based on accuracy
                        
                        # Gradual adjustment
                        learning_rate = self.weight_adaptation['learning_rate']
                        new_weight = current_weight + learning_rate * (target_weight - current_weight)
                        
                        # Apply bounds
                        new_weight = max(self.weight_adaptation['min_weight'], 
                                       min(self.weight_adaptation['max_weight'], new_weight))
                        
                        self.timeframe_weights[timeframe] = new_weight
            
            # Normalize weights to sum to 1
            total_weight = sum(self.timeframe_weights.values())
            if total_weight > 0:
                for timeframe in self.timeframe_weights:
                    self.timeframe_weights[timeframe] /= total_weight
            
        except Exception as e:
            self.logger.error(f"Adaptive weight update failed: {e}")
    
    def _update_ensemble_performance(self, ensemble_signal: EnsembleSignal):
        """Update ensemble performance tracking"""
        try:
            self.ensemble_performance['signals_generated'] += 1
            
            # Update average confidence
            current_avg = self.ensemble_performance['avg_ensemble_confidence']
            count = self.ensemble_performance['signals_generated']
            new_avg = ((current_avg * (count - 1)) + ensemble_signal.ensemble_confidence) / count
            self.ensemble_performance['avg_ensemble_confidence'] = new_avg
            
        except Exception as e:
            self.logger.error(f"Ensemble performance update failed: {e}")
    
    def _cleanup_old_signals(self, max_age_hours: int = 24):
        """Clean up old signals to manage memory"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Clean timeframe signals
            for key in list(self.timeframe_signals.keys()):
                signals = self.timeframe_signals[key]
                # Keep only recent signals
                recent_signals = deque([s for s in signals if s.timestamp >= cutoff_time], maxlen=100)
                self.timeframe_signals[key] = recent_signals
            
            # Clean ensemble signals
            recent_ensemble = deque([s for s in self.ensemble_signals if s.timestamp >= cutoff_time], maxlen=200)
            self.ensemble_signals = recent_ensemble
            
        except Exception as e:
            self.logger.error(f"Signal cleanup failed: {e}")

# Test the ensemble engine
if __name__ == "__main__":
    print("Testing Multi-Timeframe Ensemble Trading Engine")
    print("=" * 50)
    
    # Create ensemble engine
    ensemble = MultiTimeframeEnsemble(['EURUSD', 'GBPUSD'])
    result = ensemble.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Register test strategies
        ensemble.register_strategy("MA_Cross", [Timeframe.H1, Timeframe.H4])
        ensemble.register_strategy("RSI_Divergence", [Timeframe.M15, Timeframe.H1])
        ensemble.register_strategy("Trend_Following", [Timeframe.H4, Timeframe.D1])
        
        # Add sample signals
        test_signals = [
            TimeframeSignal(
                timeframe=Timeframe.H1,
                symbol='EURUSD',
                direction='BUY',
                confidence=0.75,
                strength=80,
                strategy_name='MA_Cross',
                timestamp=datetime.now(),
                indicators={'ma_fast': 1.0950, 'ma_slow': 1.0945},
                market_conditions={'trend': 'up', 'volatility': 'medium'}
            ),
            TimeframeSignal(
                timeframe=Timeframe.H4,
                symbol='EURUSD',
                direction='BUY',
                confidence=0.68,
                strength=75,
                strategy_name='Trend_Following',
                timestamp=datetime.now(),
                indicators={'trend_strength': 0.7},
                market_conditions={'trend': 'up', 'volatility': 'low'}
            )
        ]
        
        # Add signals and generate ensemble
        for signal in test_signals:
            ensemble.add_timeframe_signal(signal)
        
        # Test ensemble signal generation
        ensemble_signal = ensemble.generate_ensemble_signal('EURUSD')
        if ensemble_signal:
            print(f"\\nEnsemble signal generated:")
            print(f"Direction: {ensemble_signal.final_direction}")
            print(f"Confidence: {ensemble_signal.ensemble_confidence:.2f}")
            print(f"Strength: {ensemble_signal.ensemble_strength:.1f}")
        
        # Test status
        status = ensemble.get_ensemble_status()
        print(f"\\nEnsemble Status: {status}")
        
        # Test recent signals
        recent = ensemble.get_recent_ensemble_signals('EURUSD')
        print(f"\\nRecent signals: {len(recent)}")
        
        print("\\nShutting down...")
        ensemble.shutdown()
        
    print("Ensemble Trading Engine test completed")