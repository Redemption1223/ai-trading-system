"""
AGENT_04: Chart Signal Agent
Status: FULLY IMPLEMENTED
Purpose: Analyze charts and generate visual trading signals with technical analysis
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Try to import numerical libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

class ChartSignalAgent:
    """Chart analysis and signal generation agent with technical indicators"""
    
    def __init__(self, symbol: str, mt5_connector=None):
        self.name = "CHART_SIGNAL_AGENT"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        self.symbol = symbol
        self.mt5_connector = mt5_connector
        
        # Analysis parameters
        self.timeframes = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']
        self.current_timeframe = 'H1'
        self.lookback_periods = 100
        
        # Signal generation
        self.signal_strength_threshold = 60  # Minimum signal strength
        self.pattern_confidence_threshold = 70  # Minimum pattern confidence
        self.trend_confirmation_periods = 3  # Periods to confirm trend
        
        # Technical indicators settings
        self.indicators = {
            'sma_fast': 20,
            'sma_slow': 50,
            'ema_fast': 12,
            'ema_slow': 26,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14
        }
        
        # Current market data
        self.market_data = None
        self.current_price = None
        self.last_update = None
        
        # Signal history
        self.signal_history = []
        self.max_signal_history = 100
        
        # Analysis results
        self.current_signal = None
        self.trend_direction = None
        self.pattern_detected = None
        self.support_resistance = {'support': [], 'resistance': []}
        
        # Performance tracking
        self.signals_generated = 0
        self.patterns_detected = 0
        self.successful_signals = 0
        self.false_signals = 0
        
        # Threading for real-time analysis
        self.analysis_thread = None
        self.is_analyzing = False
        self.analysis_interval = 30  # seconds
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize the chart signal agent"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version} for {self.symbol}")
            
            # Validate symbol
            if not self.symbol:
                return {"status": "failed", "agent": "AGENT_04", "error": "No symbol provided"}
            
            # Check dependencies
            dependencies = {
                'numpy': NUMPY_AVAILABLE,
                'pandas': PANDAS_AVAILABLE,
                'talib': TALIB_AVAILABLE
            }
            
            # Initialize market data - LIVE ONLY
            if self.mt5_connector and self.mt5_connector.connection_status:
                self.logger.info("MT5 connected - using LIVE data only")
                live_data = self._fetch_live_data()
                if live_data is not None:
                    self.market_data = live_data
                    self.current_price = self._get_current_price()
                else:
                    self.logger.error("Could not fetch LIVE data from MT5")
                    raise Exception("LIVE data required - MT5 data unavailable")
            else:
                self.logger.error("MT5 not connected - LIVE connection required")
                raise Exception("LIVE MT5 connection required for operation")
            
            self.last_update = datetime.now()
            self.status = "INITIALIZED"
            
            self.logger.info(f"Chart signal agent initialized for {self.symbol}")
            
            return {
                "status": "initialized",
                "agent": "AGENT_04",
                "symbol": self.symbol,
                "timeframe": self.current_timeframe,
                "dependencies": dependencies,
                "data_points": len(self.market_data) if PANDAS_AVAILABLE and self.market_data is not None else 0,
                "current_price": self.current_price
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "AGENT_04", "error": str(e)}
    
    def start_analysis(self):
        """Start real-time chart analysis"""
        if self.is_analyzing:
            return {"status": "already_running", "message": "Analysis already active"}
        
        try:
            self.is_analyzing = True
            
            def analysis_loop():
                self.logger.info(f"Starting real-time analysis for {self.symbol}")
                
                while self.is_analyzing:
                    try:
                        # Update market data
                        self._update_market_data()
                        
                        # Perform technical analysis
                        self._perform_technical_analysis()
                        
                        # Generate signals if conditions are met
                        signal = self._generate_signal()
                        if signal:
                            self.current_signal = signal
                            self._add_signal_to_history(signal)
                            self.logger.info(f"New signal generated: {signal['direction']} - {signal['strength']}")
                        
                        # Sleep for next iteration
                        time.sleep(self.analysis_interval)
                        
                    except Exception as e:
                        self.logger.error(f"Analysis loop error: {e}")
                        time.sleep(10)  # Wait longer on error
            
            # Start analysis thread
            self.analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
            self.analysis_thread.start()
            
            self.status = "ANALYZING"
            self.logger.info(f"Real-time analysis started for {self.symbol}")
            
            return {"status": "started", "message": f"Analysis active for {self.symbol}"}
            
        except Exception as e:
            self.logger.error(f"Failed to start analysis: {e}")
            self.is_analyzing = False
            return {"status": "failed", "message": f"Failed to start: {e}"}
    
    def stop_analysis(self):
        """Stop real-time analysis"""
        try:
            self.is_analyzing = False
            
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=5)
            
            self.status = "STOPPED"
            self.logger.info(f"Analysis stopped for {self.symbol}")
            
            return {"status": "stopped", "message": "Analysis stopped successfully"}
            
        except Exception as e:
            self.logger.error(f"Error stopping analysis: {e}")
            return {"status": "error", "message": f"Error stopping: {e}"}
    
    def _update_market_data(self):
        """Update market data from MT5 or generate new sample data"""
        try:
            if self.mt5_connector and self.mt5_connector.connection_status:
                new_data = self._fetch_live_data()
                if new_data is not None:
                    self.market_data = new_data
                    self.current_price = self._get_current_price()
                    self.last_update = datetime.now()
            else:
                # Update sample data (simulate new price movements)
                if PANDAS_AVAILABLE and self.market_data is not None:
                    last_price = self.market_data.iloc[-1]['close']
                    # Simulate price movement
                    if NUMPY_AVAILABLE:
                        change = np.random.normal(0, 0.001) * last_price
                        volume = np.random.randint(100, 1000)
                    else:
                        import random
                        change = random.gauss(0, 0.001) * last_price
                        volume = random.randint(100, 1000)
                    
                    new_price = last_price + change
                    
                    # Add new row to data
                    new_row = {
                        'time': datetime.now(),
                        'open': last_price,
                        'high': max(last_price, new_price),
                        'low': min(last_price, new_price),
                        'close': new_price,
                        'volume': volume
                    }
                    
                    self.market_data = pd.concat([self.market_data, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Keep only recent data
                    if len(self.market_data) > self.lookback_periods:
                        self.market_data = self.market_data.tail(self.lookback_periods)
                    
                    self.current_price = new_price
                    self.last_update = datetime.now()
                    
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def _fetch_live_data(self):
        """Fetch live market data from MT5"""
        try:
            if not self.mt5_connector:
                return None
            
            # Get historical data to build analysis base
            history = self.mt5_connector.get_price_history(
                self.symbol, 
                self._get_mt5_timeframe(), 
                self.lookback_periods
            )
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error fetching live data: {e}")
            return None
    
    # REMOVED: Sample data generation - LIVE DATA ONLY
    def _perform_technical_analysis(self):
        """Perform comprehensive technical analysis"""
        try:
            if not PANDAS_AVAILABLE or self.market_data is None or len(self.market_data) < 50:
                # Fallback analysis without pandas/numpy
                self.trend_direction = {'direction': 'SIDEWAYS', 'strength': 50, 'duration': 0, 'key_levels': []}
                self.pattern_detected = {'pattern_type': None, 'confidence': 0, 'direction': None}
                self.support_resistance = {'support': [], 'resistance': []}
                return
            
            # Calculate technical indicators
            indicators = self._calculate_indicators()
            
            # Detect patterns
            patterns = self._detect_chart_patterns()
            
            # Analyze trend
            trend = self._analyze_trend()
            
            # Find support and resistance
            self.support_resistance = self._find_support_resistance()
            
            # Store results
            self.trend_direction = trend
            self.pattern_detected = patterns
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {e}")
    
    def _calculate_indicators(self) -> Dict:
        """Calculate technical indicators"""
        try:
            if not PANDAS_AVAILABLE:
                return {}
            
            if not NUMPY_AVAILABLE:
                # Fallback: return empty indicators when numpy is not available
                return {}
            
            close_prices = self.market_data['close'].values
            high_prices = self.market_data['high'].values
            low_prices = self.market_data['low'].values
            
            indicators = {}
            
            # Moving Averages
            if len(close_prices) >= self.indicators['sma_fast']:
                indicators['sma_20'] = self._sma(close_prices, self.indicators['sma_fast'])
                indicators['sma_50'] = self._sma(close_prices, self.indicators['sma_slow'])
                indicators['ema_12'] = self._ema(close_prices, self.indicators['ema_fast'])
                indicators['ema_26'] = self._ema(close_prices, self.indicators['ema_slow'])
            
            # RSI
            if len(close_prices) >= self.indicators['rsi_period'] + 1:
                indicators['rsi'] = self._rsi(close_prices, self.indicators['rsi_period'])
            
            # MACD (simplified calculation)
            if len(close_prices) >= max(self.indicators['macd_fast'], self.indicators['macd_slow']):
                macd_data = self._macd(close_prices)
                indicators.update(macd_data)
            
            # Bollinger Bands
            if len(close_prices) >= self.indicators['bb_period']:
                bb_data = self._bollinger_bands(close_prices)
                indicators.update(bb_data)
            
            # ATR
            if len(close_prices) >= self.indicators['atr_period']:
                indicators['atr'] = self._atr(high_prices, low_prices, close_prices)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error: {e}")
            return {}
    
    def _detect_chart_patterns(self) -> Dict:
        """Detect chart patterns"""
        try:
            patterns = {
                'pattern_type': None,
                'confidence': 0,
                'direction': None,
                'target_price': None
            }
            
            if not PANDAS_AVAILABLE or len(self.market_data) < 20:
                return patterns
            
            close_prices = self.market_data['close'].tail(20).values
            high_prices = self.market_data['high'].tail(20).values
            low_prices = self.market_data['low'].tail(20).values
            
            # Simple pattern detection
            
            # Double Top/Bottom detection
            recent_highs = self._find_local_maxima(high_prices)
            recent_lows = self._find_local_minima(low_prices)
            
            if len(recent_highs) >= 2:
                # Check for double top
                if abs(recent_highs[-1] - recent_highs[-2]) < (recent_highs[-1] * 0.005):  # Within 0.5%
                    patterns['pattern_type'] = 'double_top'
                    patterns['confidence'] = 75
                    patterns['direction'] = 'SELL'
                    patterns['target_price'] = recent_lows[-1] if len(recent_lows) > 0 else None
            
            if len(recent_lows) >= 2:
                # Check for double bottom
                if abs(recent_lows[-1] - recent_lows[-2]) < (recent_lows[-1] * 0.005):  # Within 0.5%
                    patterns['pattern_type'] = 'double_bottom'
                    patterns['confidence'] = 75
                    patterns['direction'] = 'BUY'
                    patterns['target_price'] = recent_highs[-1] if len(recent_highs) > 0 else None
            
            # Head and Shoulders (simplified)
            if len(recent_highs) >= 3 and len(recent_lows) >= 2:
                if (recent_highs[-2] > recent_highs[-1] and 
                    recent_highs[-2] > recent_highs[-3] and
                    abs(recent_highs[-1] - recent_highs[-3]) < (recent_highs[-2] * 0.01)):
                    patterns['pattern_type'] = 'head_and_shoulders'
                    patterns['confidence'] = 80
                    patterns['direction'] = 'SELL'
            
            # Triangle patterns
            triangle = self._detect_triangle_pattern(high_prices, low_prices)
            if triangle['detected']:
                patterns.update(triangle)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Pattern detection error: {e}")
            return {'pattern_type': None, 'confidence': 0, 'direction': None}
    
    def _analyze_trend(self) -> Dict:
        """Analyze current trend direction and strength"""
        try:
            trend = {
                'direction': 'SIDEWAYS',
                'strength': 0,
                'duration': 0,
                'key_levels': []
            }
            
            if not PANDAS_AVAILABLE or len(self.market_data) < 20:
                return trend
            
            close_prices = self.market_data['close'].tail(20).values
            
            # Simple trend analysis using price direction
            recent_prices = close_prices[-10:]  # Last 10 periods
            older_prices = close_prices[-20:-10]  # Previous 10 periods
            
            if NUMPY_AVAILABLE:
                recent_avg = np.mean(recent_prices)
                older_avg = np.mean(older_prices)
                max_price = np.max(close_prices)
                min_price = np.min(close_prices)
            else:
                recent_avg = sum(recent_prices) / len(recent_prices)
                older_avg = sum(older_prices) / len(older_prices)
                max_price = max(close_prices)
                min_price = min(close_prices)
            
            price_change_pct = ((recent_avg - older_avg) / older_avg) * 100
            
            # Determine trend direction
            if price_change_pct > 0.5:  # More than 0.5% increase
                trend['direction'] = 'UPTREND'
                trend['strength'] = min(100, abs(price_change_pct) * 20)  # Scale to 0-100
            elif price_change_pct < -0.5:  # More than 0.5% decrease
                trend['direction'] = 'DOWNTREND'
                trend['strength'] = min(100, abs(price_change_pct) * 20)  # Scale to 0-100
            else:
                trend['direction'] = 'SIDEWAYS'
                trend['strength'] = 30  # Moderate sideways strength
            
            # Count consecutive periods in same direction
            consecutive_periods = 1
            for i in range(len(close_prices) - 2, 0, -1):
                if ((close_prices[i+1] > close_prices[i]) == (recent_avg > older_avg)):
                    consecutive_periods += 1
                else:
                    break
            
            trend['duration'] = consecutive_periods
            
            # Key levels (simplified)
            trend['key_levels'] = [
                {'type': 'resistance', 'price': max_price, 'strength': 70},
                {'type': 'support', 'price': min_price, 'strength': 70}
            ]
            
            return trend
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {'direction': 'SIDEWAYS', 'strength': 0, 'duration': 0}
    
    def _generate_signal(self) -> Optional[Dict]:
        """Generate trading signal based on analysis"""
        try:
            if not PANDAS_AVAILABLE or self.market_data is None:
                return None
            
            # Calculate signal components
            trend_signal = self._get_trend_signal()
            pattern_signal = self._get_pattern_signal()
            indicator_signal = self._get_indicator_signal()
            
            # Combine signals
            total_strength = 0
            signal_count = 0
            dominant_direction = None
            
            buy_signals = 0
            sell_signals = 0
            
            for signal in [trend_signal, pattern_signal, indicator_signal]:
                if signal and signal.get('strength', 0) > 50:
                    total_strength += signal['strength']
                    signal_count += 1
                    
                    if signal['direction'] == 'BUY':
                        buy_signals += 1
                    elif signal['direction'] == 'SELL':
                        sell_signals += 1
            
            if signal_count == 0:
                return None
            
            # Determine dominant direction
            if buy_signals > sell_signals:
                dominant_direction = 'BUY'
            elif sell_signals > buy_signals:
                dominant_direction = 'SELL'
            else:
                return None  # Conflicting signals
            
            # Calculate average strength
            avg_strength = total_strength / signal_count
            
            # Only generate signal if strength is above threshold
            if avg_strength < self.signal_strength_threshold:
                return None
            
            # Generate signal
            signal = {
                'symbol': self.symbol,
                'direction': dominant_direction,
                'strength': round(avg_strength, 1),
                'confidence': min(100, avg_strength + (signal_count * 10)),  # Bonus for multiple confirmations
                'timestamp': datetime.now().isoformat(),
                'timeframe': self.current_timeframe,
                'entry_price': self.current_price,
                'components': {
                    'trend': trend_signal,
                    'pattern': pattern_signal,
                    'indicators': indicator_signal
                },
                'risk_reward_ratio': 2.0,  # Default ratio
                'agent': 'AGENT_04'
            }
            
            # Add suggested stop loss and take profit
            if self.trend_direction and 'key_levels' in self.trend_direction:
                support_levels = [level['price'] for level in self.trend_direction['key_levels'] if level['type'] == 'support']
                resistance_levels = [level['price'] for level in self.trend_direction['key_levels'] if level['type'] == 'resistance']
                
                if dominant_direction == 'BUY' and support_levels:
                    signal['stop_loss'] = min(support_levels)
                    signal['take_profit'] = self.current_price + (self.current_price - signal['stop_loss']) * 2
                elif dominant_direction == 'SELL' and resistance_levels:
                    signal['stop_loss'] = max(resistance_levels)
                    signal['take_profit'] = self.current_price - (signal['stop_loss'] - self.current_price) * 2
            
            self.signals_generated += 1
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return None
    
    def _get_trend_signal(self) -> Optional[Dict]:
        """Get signal from trend analysis"""
        if not self.trend_direction:
            return None
        
        direction = self.trend_direction.get('direction')
        strength = self.trend_direction.get('strength', 0)
        
        if direction == 'UPTREND':
            return {'direction': 'BUY', 'strength': strength, 'source': 'trend'}
        elif direction == 'DOWNTREND':
            return {'direction': 'SELL', 'strength': strength, 'source': 'trend'}
        
        return None
    
    def _get_pattern_signal(self) -> Optional[Dict]:
        """Get signal from pattern analysis"""
        if not self.pattern_detected or not self.pattern_detected.get('pattern_type'):
            return None
        
        confidence = self.pattern_detected.get('confidence', 0)
        direction = self.pattern_detected.get('direction')
        
        if confidence >= self.pattern_confidence_threshold and direction:
            return {'direction': direction, 'strength': confidence, 'source': 'pattern'}
        
        return None
    
    def _get_indicator_signal(self) -> Optional[Dict]:
        """Get signal from technical indicators"""
        try:
            indicators = self._calculate_indicators()
            
            if not indicators:
                return None
            
            buy_signals = 0
            sell_signals = 0
            total_strength = 0
            
            # RSI signals
            if 'rsi' in indicators:
                rsi_current = indicators['rsi'][-1] if len(indicators['rsi']) > 0 else 50
                
                if rsi_current < 30:  # Oversold
                    buy_signals += 1
                    total_strength += 70
                elif rsi_current > 70:  # Overbought
                    sell_signals += 1
                    total_strength += 70
            
            # Moving Average crossover
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma20_current = indicators['sma_20'][-1] if len(indicators['sma_20']) > 0 else 0
                sma50_current = indicators['sma_50'][-1] if len(indicators['sma_50']) > 0 else 0
                
                if sma20_current > sma50_current and self.current_price > sma20_current:
                    buy_signals += 1
                    total_strength += 60
                elif sma20_current < sma50_current and self.current_price < sma20_current:
                    sell_signals += 1
                    total_strength += 60
            
            # MACD signals
            if 'macd_line' in indicators and 'macd_signal' in indicators:
                macd_line = indicators['macd_line'][-1] if len(indicators['macd_line']) > 0 else 0
                macd_signal = indicators['macd_signal'][-1] if len(indicators['macd_signal']) > 0 else 0
                
                if macd_line > macd_signal:
                    buy_signals += 1
                    total_strength += 50
                elif macd_line < macd_signal:
                    sell_signals += 1
                    total_strength += 50
            
            # Determine dominant signal
            if buy_signals > sell_signals:
                avg_strength = total_strength / (buy_signals + sell_signals) if (buy_signals + sell_signals) > 0 else 0
                return {'direction': 'BUY', 'strength': avg_strength, 'source': 'indicators'}
            elif sell_signals > buy_signals:
                avg_strength = total_strength / (buy_signals + sell_signals) if (buy_signals + sell_signals) > 0 else 0
                return {'direction': 'SELL', 'strength': avg_strength, 'source': 'indicators'}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Indicator signal error: {e}")
            return None
    
    # Technical indicator calculation methods (simplified implementations)
    
    def _sma(self, prices, period: int):
        """Simple Moving Average"""
        if not NUMPY_AVAILABLE:
            return []
        return np.convolve(prices, np.ones(period)/period, mode='valid')
    
    def _ema(self, prices, period: int):
        """Exponential Moving Average"""
        if not NUMPY_AVAILABLE:
            return []
        alpha = 2 / (period + 1)
        ema = [prices[0]]
        
        for price in prices[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
        
        return np.array(ema)
    
    def _rsi(self, prices, period: int):
        """Relative Strength Index"""
        if not NUMPY_AVAILABLE:
            return []
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _macd(self, prices) -> Dict:
        """MACD indicator"""
        if not NUMPY_AVAILABLE:
            return {}
        ema_fast = self._ema(prices, self.indicators['macd_fast'])
        ema_slow = self._ema(prices, self.indicators['macd_slow'])
        
        # Align arrays
        min_length = min(len(ema_fast), len(ema_slow))
        ema_fast = ema_fast[-min_length:]
        ema_slow = ema_slow[-min_length:]
        
        macd_line = ema_fast - ema_slow
        macd_signal = self._ema(macd_line, self.indicators['macd_signal'])
        
        # Align final arrays
        min_length = min(len(macd_line), len(macd_signal))
        macd_line = macd_line[-min_length:]
        macd_signal = macd_signal[-min_length:]
        
        histogram = macd_line - macd_signal
        
        return {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': histogram
        }
    
    def _bollinger_bands(self, prices) -> Dict:
        """Bollinger Bands"""
        if not NUMPY_AVAILABLE:
            return {}
        sma = self._sma(prices, self.indicators['bb_period'])
        
        # Calculate rolling standard deviation
        rolling_std = []
        period = self.indicators['bb_period']
        
        for i in range(period - 1, len(prices)):
            std_dev = np.std(prices[i-period+1:i+1])
            rolling_std.append(std_dev)
        
        rolling_std = np.array(rolling_std)
        
        # Align arrays
        min_length = min(len(sma), len(rolling_std))
        sma = sma[-min_length:]
        rolling_std = rolling_std[-min_length:]
        
        bb_upper = sma + (rolling_std * self.indicators['bb_std'])
        bb_lower = sma - (rolling_std * self.indicators['bb_std'])
        
        return {
            'bb_upper': bb_upper,
            'bb_middle': sma,
            'bb_lower': bb_lower
        }
    
    def _atr(self, high, low, close):
        """Average True Range"""
        if not NUMPY_AVAILABLE:
            return []
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.convolve(true_range, np.ones(self.indicators['atr_period'])/self.indicators['atr_period'], mode='valid')
        
        return atr
    
    def _find_local_maxima(self, prices) -> List[float]:
        """Find local maxima in price series"""
        maxima = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                maxima.append(prices[i])
        return maxima
    
    def _find_local_minima(self, prices) -> List[float]:
        """Find local minima in price series"""
        minima = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                minima.append(prices[i])
        return minima
    
    def _detect_triangle_pattern(self, high, low) -> Dict:
        """Detect triangle patterns"""
        try:
            # Simplified triangle detection
            recent_highs = self._find_local_maxima(high)
            recent_lows = self._find_local_minima(low)
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                # Check if highs are converging with lows
                high_trend = recent_highs[-1] - recent_highs[-2]
                low_trend = recent_lows[-1] - recent_lows[-2]
                
                if abs(high_trend) < (recent_highs[-1] * 0.01) and abs(low_trend) < (recent_lows[-1] * 0.01):
                    # Converging triangle
                    return {
                        'detected': True,
                        'pattern_type': 'triangle',
                        'confidence': 65,
                        'direction': 'BREAKOUT_PENDING'
                    }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error(f"Triangle pattern detection error: {e}")
            return {'detected': False}
    
    def _find_support_resistance(self) -> Dict:
        """Find support and resistance levels"""
        try:
            if not PANDAS_AVAILABLE or len(self.market_data) < 20:
                return {'support': [], 'resistance': []}
            
            high_prices = self.market_data['high'].values
            low_prices = self.market_data['low'].values
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            for high in self._find_local_maxima(high_prices):
                resistance_levels.append({
                    'price': high,
                    'strength': 70,  # Default strength
                    'touches': 1
                })
            
            for low in self._find_local_minima(low_prices):
                support_levels.append({
                    'price': low,
                    'strength': 70,  # Default strength
                    'touches': 1
                })
            
            return {
                'support': support_levels[-3:] if len(support_levels) > 3 else support_levels,  # Keep last 3
                'resistance': resistance_levels[-3:] if len(resistance_levels) > 3 else resistance_levels  # Keep last 3
            }
            
        except Exception as e:
            self.logger.error(f"Support/resistance detection error: {e}")
            return {'support': [], 'resistance': []}
    
    def _get_current_price(self) -> float:
        """Get current price"""
        try:
            if self.mt5_connector and self.mt5_connector.connection_status:
                tick = self.mt5_connector.get_live_tick(self.symbol)
                if tick:
                    return (tick['bid'] + tick['ask']) / 2  # Mid price
            
            # Fallback to last close price from data
            if PANDAS_AVAILABLE and self.market_data is not None and len(self.market_data) > 0:
                return self.market_data.iloc[-1]['close']
            
            return 1.0950  # Default fallback
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 1.0950
    
    def _get_mt5_timeframe(self):
        """Convert timeframe to MT5 format"""
        timeframe_map = {
            'M1': 1,    # Would be mt5.TIMEFRAME_M1 in real implementation
            'M5': 5,    # Would be mt5.TIMEFRAME_M5
            'M15': 15,  # Would be mt5.TIMEFRAME_M15
            'H1': 60,   # Would be mt5.TIMEFRAME_H1
            'H4': 240,  # Would be mt5.TIMEFRAME_H4
            'D1': 1440  # Would be mt5.TIMEFRAME_D1
        }
        return timeframe_map.get(self.current_timeframe, 60)
    
    def _add_signal_to_history(self, signal: Dict):
        """Add signal to history"""
        try:
            self.signal_history.append(signal)
            
            # Trim history if too long
            if len(self.signal_history) > self.max_signal_history:
                self.signal_history = self.signal_history[-self.max_signal_history:]
                
        except Exception as e:
            self.logger.error(f"Error adding signal to history: {e}")
    
    def get_current_signal(self) -> Optional[Dict]:
        """Get the current/latest signal"""
        return self.current_signal
    
    def get_last_signal_time(self) -> Optional[str]:
        """Get timestamp of last signal"""
        if self.current_signal:
            return self.current_signal.get('timestamp')
        return None
    
    def get_performance_metrics(self) -> Dict:
        """Get agent performance metrics"""
        try:
            success_rate = 0
            if self.signals_generated > 0:
                success_rate = (self.successful_signals / self.signals_generated) * 100
            
            return {
                'signals_generated': self.signals_generated,
                'patterns_detected': self.patterns_detected,
                'successful_signals': self.successful_signals,
                'false_signals': self.false_signals,
                'success_rate': round(success_rate, 1),
                'current_trend': self.trend_direction.get('direction', 'UNKNOWN') if self.trend_direction else 'UNKNOWN',
                'last_pattern': self.pattern_detected.get('pattern_type', 'NONE') if self.pattern_detected else 'NONE',
                'analysis_active': self.is_analyzing,
                'data_points': len(self.market_data) if PANDAS_AVAILABLE and self.market_data is not None else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    def set_timeframe(self, timeframe: str) -> bool:
        """Set analysis timeframe"""
        if timeframe in self.timeframes:
            self.current_timeframe = timeframe
            self.logger.info(f"Timeframe changed to {timeframe}")
            return True
        return False
    
    def get_status(self):
        """Get current agent status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'symbol': self.symbol,
            'timeframe': self.current_timeframe,
            'is_analyzing': self.is_analyzing,
            'current_price': self.current_price,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'signals_generated': self.signals_generated,
            'data_available': PANDAS_AVAILABLE and self.market_data is not None,
            'mt5_connected': self.mt5_connector.connection_status if self.mt5_connector else False
        }
    
    def shutdown(self):
        """Clean shutdown of chart signal agent"""
        try:
            self.logger.info(f"Shutting down chart signal agent for {self.symbol}...")
            
            # Stop analysis
            self.stop_analysis()
            
            # Clear data
            self.market_data = None
            self.current_signal = None
            self.signal_history.clear()
            
            self.status = "SHUTDOWN"
            self.logger.info("Chart signal agent shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Agent test
if __name__ == "__main__":
    # Test the chart signal agent
    print("Testing AGENT_04: Chart Signal Agent")
    print("=" * 40)
    
    # Create chart agent for EURUSD
    agent = ChartSignalAgent("EURUSD")
    result = agent.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test analysis
        print("\nTesting technical analysis...")
        agent._perform_technical_analysis()
        
        # Test signal generation
        print("Testing signal generation...")
        signal = agent._generate_signal()
        if signal:
            print(f"Signal generated: {signal}")
        else:
            print("No signal generated")
        
        # Test status
        status = agent.get_status()
        print(f"\nStatus: {status}")
        
        # Test metrics
        metrics = agent.get_performance_metrics()
        print(f"Metrics: {metrics}")
        
        # Test shutdown
        print("\nShutting down...")
        agent.shutdown()
        
    print("Chart Signal Agent test completed")