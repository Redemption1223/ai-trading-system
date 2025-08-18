"""
AGENT_06: Technical Analyst
Status: FULLY IMPLEMENTED
Purpose: Advanced technical analysis engine with comprehensive indicator calculations and trend analysis
"""

import logging
import math
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Required libraries for live trading
import numpy as np
import pandas as pd

class TrendDirection(Enum):
    """Enum for trend directions"""
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    SIDEWAYS = "SIDEWAYS"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"

class SignalStrength(Enum):
    """Enum for signal strength levels"""
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    VERY_WEAK = "VERY_WEAK"

class TechnicalAnalyst:
    """Advanced technical analysis engine for comprehensive market analysis"""
    
    def __init__(self, symbol: str = "EURUSD"):
        self.name = "TECHNICAL_ANALYST"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        self.symbol = symbol
        
        # Market data storage
        self.market_data = None
        self.data_history = []
        self.max_history = 500
        self.min_data_points = 50
        
        # Indicator configurations
        self.indicator_config = {
            # Moving Averages
            'sma_periods': [5, 10, 20, 50, 100, 200],
            'ema_periods': [8, 13, 21, 34, 55],
            'wma_periods': [10, 20, 30],
            
            # Momentum Indicators
            'rsi_period': 14,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'williams_r_period': 14,
            'roc_period': 12,
            'momentum_period': 10,
            
            # Trend Indicators
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'aroon_period': 25,
            'cci_period': 20,
            
            # Volatility Indicators
            'bb_period': 20,
            'bb_std': 2.0,
            'atr_period': 14,
            'donchian_period': 20,
            
            # Volume Indicators
            'obv_period': 10,
            'ad_period': 21,
            'mfi_period': 14,
            
            # Support/Resistance
            'pivot_type': 'standard',
            'support_resistance_periods': 50
        }
        
        # Analysis results cache
        self.indicators = {}
        self.trend_analysis = {}
        self.support_resistance = {'support': [], 'resistance': []}
        self.signals = []
        self.signal_history = []
        self.max_signal_history = 200
        
        # Performance tracking
        self.analysis_count = 0
        self.signals_generated = 0
        self.last_analysis_time = None
        
        # Real-time analysis
        self.analysis_thread = None
        self.is_analyzing = False
        self.analysis_interval = 60  # seconds
        
        # Signal generation settings
        self.signal_threshold = 60  # Minimum signal strength
        self.confluence_weight = 1.5  # Weight for confluent signals
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize the technical analyst"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version} for {self.symbol}")
            
            # NO SAMPLE DATA - LIVE DATA ONLY
            self.logger.info("Technical Analyst initialized for LIVE data only - no simulation")
            
            # Perform initial analysis if we have data
            if self.market_data is not None and len(self.market_data) >= self.min_data_points:
                self._perform_full_analysis()
                self.logger.info("Initial technical analysis completed")
            
            self.status = "INITIALIZED"
            self.logger.info("Technical Analyst initialized successfully")
            
            return {
                "status": "initialized",
                "agent": "AGENT_06",
                "symbol": self.symbol,
                "data_points": len(self.market_data) if self.market_data is not None else 0,
                "indicators_configured": len(self.indicator_config),
                "numpy_available": True,
                "pandas_available": True,
                "analysis_ready": self.market_data is not None and len(self.market_data) >= self.min_data_points
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "AGENT_06", "error": str(e)}
    
    # REMOVED: _generate_sample_data() - LIVE DATA ONLY
    
    def update_market_data(self, market_data):
        """Update market data for analysis"""
        try:
            self.market_data = market_data
            
            # Add to history
            self.data_history.append({
                'timestamp': datetime.now().isoformat(),
                'data_points': len(market_data) if hasattr(market_data, '__len__') else 0,
                'last_price': self._get_last_price()
            })
            
            # Trim history
            if len(self.data_history) > 50:
                self.data_history = self.data_history[-50:]
            
            # Trigger analysis if we have enough data
            if hasattr(market_data, '__len__') and len(market_data) >= self.min_data_points:
                self._perform_full_analysis()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return False
    
    def _get_last_price(self):
        """Get the last close price from market data"""
        try:
            if self.market_data is None:
                return None
            
            if isinstance(self.market_data, pd.DataFrame):
                return self.market_data.iloc[-1]['close']
            elif isinstance(self.market_data, list) and len(self.market_data) > 0:
                return self.market_data[-1]['close']
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting last price: {e}")
            return None
    
    def _perform_full_analysis(self):
        """Perform complete technical analysis"""
        try:
            self.logger.debug("Performing full technical analysis")
            
            # Calculate all indicators
            self.indicators = self._calculate_all_indicators()
            
            # Perform trend analysis
            self.trend_analysis = self._analyze_trends()
            
            # Find support and resistance levels
            self.support_resistance = self._find_support_resistance()
            
            # Generate signals
            self.signals = self._generate_technical_signals()
            
            # Update performance tracking
            self.analysis_count += 1
            self.last_analysis_time = datetime.now()
            
            if self.signals:
                self.signals_generated += len(self.signals)
                
                # Add to signal history
                for signal in self.signals:
                    signal['analysis_timestamp'] = self.last_analysis_time.isoformat()
                    self.signal_history.append(signal)
                
                # Trim signal history
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history = self.signal_history[-self.max_signal_history:]
            
            self.logger.debug(f"Analysis completed - {len(self.signals)} signals generated")
            
        except Exception as e:
            self.logger.error(f"Full analysis error: {e}")
    
    def _calculate_all_indicators(self) -> Dict:
        """Calculate all technical indicators"""
        try:
            indicators = {}
            
            if self.market_data is None:
                return indicators
            
            # Get price arrays
            if isinstance(self.market_data, pd.DataFrame):
                high = self.market_data['high'].values
                low = self.market_data['low'].values
                close = self.market_data['close'].values
                volume = self.market_data['volume'].values
                open_prices = self.market_data['open'].values
            else:
                high = [candle['high'] for candle in self.market_data]
                low = [candle['low'] for candle in self.market_data]
                close = [candle['close'] for candle in self.market_data]
                volume = [candle['volume'] for candle in self.market_data]
                open_prices = [candle['open'] for candle in self.market_data]
            
            # Moving Averages
            for period in self.indicator_config['sma_periods']:
                if len(close) >= period:
                    indicators[f'sma_{period}'] = self._sma(close, period)
            
            for period in self.indicator_config['ema_periods']:
                if len(close) >= period:
                    indicators[f'ema_{period}'] = self._ema(close, period)
            
            # Momentum Indicators
            if len(close) >= self.indicator_config['rsi_period']:
                indicators['rsi'] = self._rsi(close, self.indicator_config['rsi_period'])
            
            if len(close) >= max(self.indicator_config['macd_fast'], self.indicator_config['macd_slow']):
                macd_data = self._macd(close)
                indicators.update(macd_data)
            
            if len(high) >= self.indicator_config['stoch_k_period']:
                stoch_data = self._stochastic(high, low, close)
                indicators.update(stoch_data)
            
            if len(close) >= self.indicator_config['williams_r_period']:
                indicators['williams_r'] = self._williams_r(high, low, close, self.indicator_config['williams_r_period'])
            
            # Trend Indicators
            if len(close) >= self.indicator_config['adx_period']:
                adx_data = self._adx(high, low, close)
                indicators.update(adx_data)
            
            if len(high) >= self.indicator_config['aroon_period']:
                aroon_data = self._aroon(high, low, self.indicator_config['aroon_period'])
                indicators.update(aroon_data)
            
            # Volatility Indicators
            if len(close) >= self.indicator_config['bb_period']:
                bb_data = self._bollinger_bands(close)
                indicators.update(bb_data)
            
            if len(high) >= self.indicator_config['atr_period']:
                indicators['atr'] = self._atr(high, low, close)
            
            # Volume Indicators
            if len(close) >= 2 and len(volume) >= 2:
                indicators['obv'] = self._obv(close, volume)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error: {e}")
            return {}
    
    # Technical Indicator Implementations
    
    def _sma(self, prices: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return []
        
        sma = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
        
        return sma
    
    def _ema(self, prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if len(prices) < period:
            return []
        
        alpha = 2.0 / (period + 1)
        ema = [sum(prices[:period]) / period]  # Start with SMA
        
        for i in range(period, len(prices)):
            ema_value = alpha * prices[i] + (1 - alpha) * ema[-1]
            ema.append(ema_value)
        
        return ema
    
    def _rsi(self, prices: List[float], period: int) -> List[float]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(delta, 0) for delta in deltas]
        losses = [-min(delta, 0) for delta in deltas]
        
        rsi = []
        
        # Initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
        
        # Calculate remaining RSI values
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
        
        return rsi
    
    def _macd(self, prices: List[float]) -> Dict[str, List[float]]:
        """MACD Indicator"""
        try:
            fast_ema = self._ema(prices, self.indicator_config['macd_fast'])
            slow_ema = self._ema(prices, self.indicator_config['macd_slow'])
            
            if not fast_ema or not slow_ema:
                return {}
            
            # Align arrays to same length
            min_len = min(len(fast_ema), len(slow_ema))
            fast_ema = fast_ema[-min_len:]
            slow_ema = slow_ema[-min_len:]
            
            # MACD line
            macd_line = [fast_ema[i] - slow_ema[i] for i in range(len(fast_ema))]
            
            # Signal line
            signal_line = self._ema(macd_line, self.indicator_config['macd_signal'])
            
            # Histogram
            if signal_line:
                min_len = min(len(macd_line), len(signal_line))
                macd_line = macd_line[-min_len:]
                histogram = [macd_line[i] - signal_line[i] for i in range(len(signal_line))]
            else:
                histogram = []
            
            return {
                'macd_line': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': histogram
            }
            
        except Exception as e:
            self.logger.error(f"MACD calculation error: {e}")
            return {}
    
    def _stochastic(self, high: List[float], low: List[float], close: List[float]) -> Dict[str, List[float]]:
        """Stochastic Oscillator"""
        try:
            if len(high) < self.indicator_config['stoch_k_period']:
                return {}
            
            k_percent = []
            
            for i in range(self.indicator_config['stoch_k_period'] - 1, len(close)):
                highest_high = max(high[i - self.indicator_config['stoch_k_period'] + 1:i + 1])
                lowest_low = min(low[i - self.indicator_config['stoch_k_period'] + 1:i + 1])
                
                if highest_high == lowest_low:
                    k_value = 50  # Default value when range is 0
                else:
                    k_value = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
                
                k_percent.append(k_value)
            
            # %D is SMA of %K
            d_percent = self._sma(k_percent, self.indicator_config['stoch_d_period'])
            
            return {
                'stoch_k': k_percent,
                'stoch_d': d_percent
            }
            
        except Exception as e:
            self.logger.error(f"Stochastic calculation error: {e}")
            return {}
    
    def _williams_r(self, high: List[float], low: List[float], close: List[float], period: int) -> List[float]:
        """Williams %R"""
        try:
            if len(high) < period:
                return []
            
            williams = []
            
            for i in range(period - 1, len(close)):
                highest_high = max(high[i - period + 1:i + 1])
                lowest_low = min(low[i - period + 1:i + 1])
                
                if highest_high == lowest_low:
                    wr_value = -50  # Default value
                else:
                    wr_value = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
                
                williams.append(wr_value)
            
            return williams
            
        except Exception as e:
            self.logger.error(f"Williams %R calculation error: {e}")
            return []
    
    def _adx(self, high: List[float], low: List[float], close: List[float]) -> Dict[str, List[float]]:
        """Average Directional Index"""
        try:
            period = self.indicator_config['adx_period']
            if len(high) < period + 1:
                return {}
            
            # Calculate True Range and Directional Movement
            tr = []
            plus_dm = []
            minus_dm = []
            
            for i in range(1, len(high)):
                # True Range
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr.append(max(tr1, tr2, tr3))
                
                # Directional Movement
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
                minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
            
            # Smooth the values
            smoothed_tr = []
            smoothed_plus_dm = []
            smoothed_minus_dm = []
            
            # First value is sum of first period
            smoothed_tr.append(sum(tr[:period]))
            smoothed_plus_dm.append(sum(plus_dm[:period]))
            smoothed_minus_dm.append(sum(minus_dm[:period]))
            
            # Subsequent values use Wilder's smoothing
            for i in range(period, len(tr)):
                smoothed_tr.append(smoothed_tr[-1] - smoothed_tr[-1]/period + tr[i])
                smoothed_plus_dm.append(smoothed_plus_dm[-1] - smoothed_plus_dm[-1]/period + plus_dm[i])
                smoothed_minus_dm.append(smoothed_minus_dm[-1] - smoothed_minus_dm[-1]/period + minus_dm[i])
            
            # Calculate DI+ and DI-
            plus_di = [(smoothed_plus_dm[i] / smoothed_tr[i] * 100) if smoothed_tr[i] != 0 else 0 
                      for i in range(len(smoothed_tr))]
            minus_di = [(smoothed_minus_dm[i] / smoothed_tr[i] * 100) if smoothed_tr[i] != 0 else 0 
                       for i in range(len(smoothed_tr))]
            
            # Calculate DX and ADX
            dx = []
            adx = []
            
            for i in range(len(plus_di)):
                di_sum = plus_di[i] + minus_di[i]
                if di_sum != 0:
                    dx.append(abs(plus_di[i] - minus_di[i]) / di_sum * 100)
                else:
                    dx.append(0)
            
            if len(dx) >= period:
                # First ADX is average of first period DX values
                adx.append(sum(dx[:period]) / period)
                
                # Subsequent ADX values use Wilder's smoothing
                for i in range(period, len(dx)):
                    adx.append((adx[-1] * (period - 1) + dx[i]) / period)
            
            return {
                'adx': adx,
                'plus_di': plus_di[len(plus_di) - len(adx):] if adx else [],
                'minus_di': minus_di[len(minus_di) - len(adx):] if adx else []
            }
            
        except Exception as e:
            self.logger.error(f"ADX calculation error: {e}")
            return {}
    
    def _aroon(self, high: List[float], low: List[float], period: int) -> Dict[str, List[float]]:
        """Aroon Indicator"""
        try:
            if len(high) < period:
                return {}
            
            aroon_up = []
            aroon_down = []
            
            for i in range(period - 1, len(high)):
                # Find highest high and lowest low positions in the period
                period_high = high[i - period + 1:i + 1]
                period_low = low[i - period + 1:i + 1]
                
                # Handle both list and numpy array cases
                if hasattr(period_high, 'index'):
                    # For regular Python lists
                    high_pos = period_high.index(max(period_high))
                    low_pos = period_low.index(min(period_low))
                else:
                    # For numpy arrays or pandas Series
                    if isinstance(period_high, np.ndarray):
                        high_pos = np.argmax(period_high)
                        low_pos = np.argmin(period_low)
                    else:
                        # Fallback: convert to list
                        high_list = list(period_high)
                        low_list = list(period_low)
                        high_pos = high_list.index(max(high_list))
                        low_pos = low_list.index(min(low_list))
                
                aroon_up.append(((period - 1 - high_pos) / (period - 1)) * 100)
                aroon_down.append(((period - 1 - low_pos) / (period - 1)) * 100)
            
            return {
                'aroon_up': aroon_up,
                'aroon_down': aroon_down,
                'aroon_oscillator': [aroon_up[i] - aroon_down[i] for i in range(len(aroon_up))]
            }
            
        except Exception as e:
            self.logger.error(f"Aroon calculation error: {e}")
            return {}
    
    def _bollinger_bands(self, prices: List[float]) -> Dict[str, List[float]]:
        """Bollinger Bands"""
        try:
            period = self.indicator_config['bb_period']
            std_dev = self.indicator_config['bb_std']
            
            if len(prices) < period:
                return {}
            
            sma = self._sma(prices, period)
            upper_band = []
            lower_band = []
            
            for i in range(len(sma)):
                # Calculate standard deviation for the period
                period_prices = prices[i:i + period]
                mean = sma[i]
                variance = sum((x - mean) ** 2 for x in period_prices) / period
                std = math.sqrt(variance)
                
                upper_band.append(mean + (std_dev * std))
                lower_band.append(mean - (std_dev * std))
            
            return {
                'bb_upper': upper_band,
                'bb_middle': sma,
                'bb_lower': lower_band,
                'bb_width': [upper_band[i] - lower_band[i] for i in range(len(upper_band))],
                'bb_percent': [((prices[i + period - 1] - lower_band[i]) / (upper_band[i] - lower_band[i])) * 100 
                              if upper_band[i] != lower_band[i] else 50 
                              for i in range(len(upper_band))]
            }
            
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation error: {e}")
            return {}
    
    def _atr(self, high: List[float], low: List[float], close: List[float]) -> List[float]:
        """Average True Range"""
        try:
            period = self.indicator_config['atr_period']
            if len(high) < period + 1:
                return []
            
            true_ranges = []
            
            for i in range(1, len(high)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                true_ranges.append(max(tr1, tr2, tr3))
            
            # Calculate ATR using Wilder's smoothing
            atr = []
            atr.append(sum(true_ranges[:period]) / period)  # First ATR is simple average
            
            for i in range(period, len(true_ranges)):
                atr_value = (atr[-1] * (period - 1) + true_ranges[i]) / period
                atr.append(atr_value)
            
            return atr
            
        except Exception as e:
            self.logger.error(f"ATR calculation error: {e}")
            return []
    
    def _obv(self, prices: List[float], volume: List[float]) -> List[float]:
        """On-Balance Volume"""
        try:
            if len(prices) != len(volume) or len(prices) < 2:
                return []
            
            obv = [volume[0]]  # Start with first volume
            
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    obv.append(obv[-1] + volume[i])
                elif prices[i] < prices[i-1]:
                    obv.append(obv[-1] - volume[i])
                else:
                    obv.append(obv[-1])
            
            return obv
            
        except Exception as e:
            self.logger.error(f"OBV calculation error: {e}")
            return []
    
    def _analyze_trends(self) -> Dict:
        """Comprehensive trend analysis"""
        try:
            trend_analysis = {
                'primary_trend': TrendDirection.SIDEWAYS,
                'secondary_trend': TrendDirection.SIDEWAYS,
                'trend_strength': 0,
                'trend_duration': 0,
                'trend_signals': []
            }
            
            if not self.indicators or not self.market_data:
                return trend_analysis
            
            # Get current price
            current_price = self._get_last_price()
            if not current_price:
                return trend_analysis
            
            trend_signals = []
            trend_score = 0
            
            # Moving Average Analysis
            ma_signals = self._analyze_moving_averages()
            trend_signals.extend(ma_signals)
            trend_score += sum(signal.get('score', 0) for signal in ma_signals)
            
            # MACD Analysis
            if 'macd_line' in self.indicators and 'macd_signal' in self.indicators:
                macd_signals = self._analyze_macd()
                trend_signals.extend(macd_signals)
                trend_score += sum(signal.get('score', 0) for signal in macd_signals)
            
            # ADX Analysis
            if 'adx' in self.indicators:
                adx_signals = self._analyze_adx()
                trend_signals.extend(adx_signals)
                trend_score += sum(signal.get('score', 0) for signal in adx_signals)
            
            # Determine primary trend
            if trend_score > 60:
                trend_analysis['primary_trend'] = TrendDirection.STRONG_UPTREND
            elif trend_score > 30:
                trend_analysis['primary_trend'] = TrendDirection.UPTREND
            elif trend_score < -60:
                trend_analysis['primary_trend'] = TrendDirection.STRONG_DOWNTREND
            elif trend_score < -30:
                trend_analysis['primary_trend'] = TrendDirection.DOWNTREND
            else:
                trend_analysis['primary_trend'] = TrendDirection.SIDEWAYS
            
            trend_analysis['trend_strength'] = abs(trend_score)
            trend_analysis['trend_signals'] = trend_signals
            
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {'primary_trend': TrendDirection.SIDEWAYS, 'trend_strength': 0, 'trend_signals': []}
    
    def _analyze_moving_averages(self) -> List[Dict]:
        """Analyze moving average crossovers and positions"""
        signals = []
        
        try:
            current_price = self._get_last_price()
            
            # Check price position relative to MAs
            for period in [20, 50, 200]:
                ma_key = f'sma_{period}'
                if ma_key in self.indicators and self.indicators[ma_key]:
                    ma_value = self.indicators[ma_key][-1]
                    
                    if current_price > ma_value:
                        signals.append({
                            'type': 'ma_position',
                            'indicator': ma_key,
                            'signal': 'bullish',
                            'score': 10,
                            'description': f'Price above SMA({period})'
                        })
                    else:
                        signals.append({
                            'type': 'ma_position',
                            'indicator': ma_key,
                            'signal': 'bearish',
                            'score': -10,
                            'description': f'Price below SMA({period})'
                        })
            
            # Check for MA crossovers
            if 'sma_20' in self.indicators and 'sma_50' in self.indicators:
                sma20 = self.indicators['sma_20']
                sma50 = self.indicators['sma_50']
                
                if len(sma20) >= 2 and len(sma50) >= 2:
                    if sma20[-1] > sma50[-1] and sma20[-2] <= sma50[-2]:
                        signals.append({
                            'type': 'ma_crossover',
                            'indicator': 'sma_20_50',
                            'signal': 'golden_cross',
                            'score': 25,
                            'description': 'SMA(20) crossed above SMA(50)'
                        })
                    elif sma20[-1] < sma50[-1] and sma20[-2] >= sma50[-2]:
                        signals.append({
                            'type': 'ma_crossover',
                            'indicator': 'sma_20_50',
                            'signal': 'death_cross',
                            'score': -25,
                            'description': 'SMA(20) crossed below SMA(50)'
                        })
            
        except Exception as e:
            self.logger.error(f"MA analysis error: {e}")
        
        return signals
    
    def _analyze_macd(self) -> List[Dict]:
        """Analyze MACD signals"""
        signals = []
        
        try:
            macd_line = self.indicators.get('macd_line', [])
            macd_signal = self.indicators.get('macd_signal', [])
            macd_histogram = self.indicators.get('macd_histogram', [])
            
            if len(macd_line) >= 2 and len(macd_signal) >= 2:
                # MACD line crossover
                if macd_line[-1] > macd_signal[-1] and macd_line[-2] <= macd_signal[-2]:
                    signals.append({
                        'type': 'macd_crossover',
                        'signal': 'bullish',
                        'score': 20,
                        'description': 'MACD line crossed above signal line'
                    })
                elif macd_line[-1] < macd_signal[-1] and macd_line[-2] >= macd_signal[-2]:
                    signals.append({
                        'type': 'macd_crossover',
                        'signal': 'bearish',
                        'score': -20,
                        'description': 'MACD line crossed below signal line'
                    })
                
                # Zero line crossover
                if macd_line[-1] > 0 and macd_line[-2] <= 0:
                    signals.append({
                        'type': 'macd_zero_cross',
                        'signal': 'bullish',
                        'score': 15,
                        'description': 'MACD crossed above zero line'
                    })
                elif macd_line[-1] < 0 and macd_line[-2] >= 0:
                    signals.append({
                        'type': 'macd_zero_cross',
                        'signal': 'bearish',
                        'score': -15,
                        'description': 'MACD crossed below zero line'
                    })
            
        except Exception as e:
            self.logger.error(f"MACD analysis error: {e}")
        
        return signals
    
    def _analyze_adx(self) -> List[Dict]:
        """Analyze ADX for trend strength"""
        signals = []
        
        try:
            adx = self.indicators.get('adx', [])
            plus_di = self.indicators.get('plus_di', [])
            minus_di = self.indicators.get('minus_di', [])
            
            if adx and plus_di and minus_di:
                current_adx = adx[-1]
                current_plus_di = plus_di[-1]
                current_minus_di = minus_di[-1]
                
                # ADX trend strength
                if current_adx > 25:
                    if current_plus_di > current_minus_di:
                        signals.append({
                            'type': 'adx_trend',
                            'signal': 'strong_uptrend',
                            'score': 20,
                            'description': f'Strong uptrend (ADX: {current_adx:.1f})'
                        })
                    else:
                        signals.append({
                            'type': 'adx_trend',
                            'signal': 'strong_downtrend',
                            'score': -20,
                            'description': f'Strong downtrend (ADX: {current_adx:.1f})'
                        })
                elif current_adx < 20:
                    signals.append({
                        'type': 'adx_trend',
                        'signal': 'weak_trend',
                        'score': 0,
                        'description': f'Weak/sideways trend (ADX: {current_adx:.1f})'
                    })
            
        except Exception as e:
            self.logger.error(f"ADX analysis error: {e}")
        
        return signals
    
    def _find_support_resistance(self) -> Dict:
        """Find key support and resistance levels"""
        try:
            support_resistance = {'support': [], 'resistance': []}
            
            if self.market_data is None:
                return support_resistance
            
            # Get price data
            if isinstance(self.market_data, pd.DataFrame):
                high = self.market_data['high'].values[-self.indicator_config['support_resistance_periods']:]
                low = self.market_data['low'].values[-self.indicator_config['support_resistance_periods']:]
                close = self.market_data['close'].values[-self.indicator_config['support_resistance_periods']:]
            else:
                high = [candle['high'] for candle in self.market_data[-self.indicator_config['support_resistance_periods']:]]
                low = [candle['low'] for candle in self.market_data[-self.indicator_config['support_resistance_periods']:]]
                close = [candle['close'] for candle in self.market_data[-self.indicator_config['support_resistance_periods']:]]
            
            # Find local maxima (resistance) and minima (support)
            resistance_levels = self._find_local_extrema(high, 'max')
            support_levels = self._find_local_extrema(low, 'min')
            
            # Add pivot points if available
            if len(high) >= 3 and len(low) >= 3 and len(close) >= 3:
                pivot_data = self._calculate_pivot_points(high[-1], low[-1], close[-1])
                support_resistance['pivot_points'] = pivot_data
            
            support_resistance['support'] = support_levels
            support_resistance['resistance'] = resistance_levels
            
            return support_resistance
            
        except Exception as e:
            self.logger.error(f"Support/Resistance detection error: {e}")
            return {'support': [], 'resistance': []}
    
    def _find_local_extrema(self, prices: List[float], extrema_type: str, window: int = 5) -> List[Dict]:
        """Find local extrema (maxima or minima) in price data"""
        extrema = []
        
        try:
            for i in range(window, len(prices) - window):
                if extrema_type == 'max':
                    if prices[i] == max(prices[i-window:i+window+1]):
                        extrema.append({
                            'price': prices[i],
                            'position': i,
                            'strength': self._calculate_level_strength(prices, i, extrema_type),
                            'touches': 1
                        })
                elif extrema_type == 'min':
                    if prices[i] == min(prices[i-window:i+window+1]):
                        extrema.append({
                            'price': prices[i],
                            'position': i,
                            'strength': self._calculate_level_strength(prices, i, extrema_type),
                            'touches': 1
                        })
            
            # Sort by strength and return top levels
            extrema.sort(key=lambda x: x['strength'], reverse=True)
            return extrema[:5]  # Return top 5 levels
            
        except Exception as e:
            self.logger.error(f"Local extrema detection error: {e}")
            return []
    
    def _calculate_level_strength(self, prices: List[float], position: int, extrema_type: str) -> float:
        """Calculate the strength of a support/resistance level"""
        try:
            strength = 1.0
            price = prices[position]
            
            # Count touches within a tolerance
            tolerance = price * 0.001  # 0.1% tolerance
            
            for i, p in enumerate(prices):
                if abs(p - price) <= tolerance and abs(i - position) > 3:
                    strength += 1.0
            
            # Recent levels are more important
            recency_factor = max(0.5, 1.0 - (len(prices) - position) / len(prices))
            strength *= recency_factor
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Level strength calculation error: {e}")
            return 1.0
    
    def _calculate_pivot_points(self, high: float, low: float, close: float) -> Dict:
        """Calculate pivot points for support/resistance"""
        try:
            pivot = (high + low + close) / 3
            
            return {
                'pivot': pivot,
                'r1': 2 * pivot - low,
                'r2': pivot + (high - low),
                'r3': high + 2 * (pivot - low),
                's1': 2 * pivot - high,
                's2': pivot - (high - low),
                's3': low - 2 * (high - pivot)
            }
            
        except Exception as e:
            self.logger.error(f"Pivot points calculation error: {e}")
            return {}
    
    def _generate_technical_signals(self) -> List[Dict]:
        """Generate comprehensive technical analysis signals"""
        try:
            signals = []
            
            if not self.indicators or not self.trend_analysis:
                return signals
            
            # Momentum signals
            momentum_signals = self._generate_momentum_signals()
            signals.extend(momentum_signals)
            
            # Trend following signals
            trend_signals = self._generate_trend_signals()
            signals.extend(trend_signals)
            
            # Mean reversion signals
            reversion_signals = self._generate_mean_reversion_signals()
            signals.extend(reversion_signals)
            
            # Breakout signals
            breakout_signals = self._generate_breakout_signals()
            signals.extend(breakout_signals)
            
            # Filter and rank signals
            filtered_signals = self._filter_and_rank_signals(signals)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return []
    
    def _generate_momentum_signals(self) -> List[Dict]:
        """Generate momentum-based signals"""
        signals = []
        
        try:
            # RSI signals
            if 'rsi' in self.indicators and self.indicators['rsi']:
                rsi = self.indicators['rsi'][-1]
                
                if rsi < 30:
                    signals.append({
                        'type': 'momentum',
                        'indicator': 'rsi',
                        'direction': 'BUY',
                        'strength': 70,
                        'description': f'RSI oversold ({rsi:.1f})',
                        'entry_reason': 'oversold_bounce'
                    })
                elif rsi > 70:
                    signals.append({
                        'type': 'momentum',
                        'indicator': 'rsi',
                        'direction': 'SELL',
                        'strength': 70,
                        'description': f'RSI overbought ({rsi:.1f})',
                        'entry_reason': 'overbought_reversal'
                    })
            
            # Stochastic signals
            if 'stoch_k' in self.indicators and 'stoch_d' in self.indicators:
                stoch_k = self.indicators['stoch_k']
                stoch_d = self.indicators['stoch_d']
                
                if len(stoch_k) >= 2 and len(stoch_d) >= 2:
                    if stoch_k[-1] > stoch_d[-1] and stoch_k[-2] <= stoch_d[-2] and stoch_k[-1] < 20:
                        signals.append({
                            'type': 'momentum',
                            'indicator': 'stochastic',
                            'direction': 'BUY',
                            'strength': 65,
                            'description': 'Stochastic bullish crossover in oversold area',
                            'entry_reason': 'oversold_crossover'
                        })
                    elif stoch_k[-1] < stoch_d[-1] and stoch_k[-2] >= stoch_d[-2] and stoch_k[-1] > 80:
                        signals.append({
                            'type': 'momentum',
                            'indicator': 'stochastic',
                            'direction': 'SELL',
                            'strength': 65,
                            'description': 'Stochastic bearish crossover in overbought area',
                            'entry_reason': 'overbought_crossover'
                        })
            
        except Exception as e:
            self.logger.error(f"Momentum signal generation error: {e}")
        
        return signals
    
    def _generate_trend_signals(self) -> List[Dict]:
        """Generate trend-following signals"""
        signals = []
        
        try:
            # Use trend analysis results
            if self.trend_analysis['primary_trend'] in [TrendDirection.STRONG_UPTREND, TrendDirection.UPTREND]:
                signals.append({
                    'type': 'trend',
                    'indicator': 'trend_analysis',
                    'direction': 'BUY',
                    'strength': min(self.trend_analysis['trend_strength'], 90),
                    'description': f"Strong uptrend detected (strength: {self.trend_analysis['trend_strength']})",
                    'entry_reason': 'trend_following'
                })
            elif self.trend_analysis['primary_trend'] in [TrendDirection.STRONG_DOWNTREND, TrendDirection.DOWNTREND]:
                signals.append({
                    'type': 'trend',
                    'indicator': 'trend_analysis',
                    'direction': 'SELL',
                    'strength': min(self.trend_analysis['trend_strength'], 90),
                    'description': f"Strong downtrend detected (strength: {self.trend_analysis['trend_strength']})",
                    'entry_reason': 'trend_following'
                })
            
        except Exception as e:
            self.logger.error(f"Trend signal generation error: {e}")
        
        return signals
    
    def _generate_mean_reversion_signals(self) -> List[Dict]:
        """Generate mean reversion signals"""
        signals = []
        
        try:
            # Bollinger Bands mean reversion
            if all(key in self.indicators for key in ['bb_upper', 'bb_lower', 'bb_percent']):
                bb_percent = self.indicators['bb_percent']
                if bb_percent:
                    current_bb_percent = bb_percent[-1]
                    
                    if current_bb_percent < 5:  # Near lower band
                        signals.append({
                            'type': 'mean_reversion',
                            'indicator': 'bollinger_bands',
                            'direction': 'BUY',
                            'strength': 60,
                            'description': f'Price near lower Bollinger Band ({current_bb_percent:.1f}%)',
                            'entry_reason': 'oversold_reversion'
                        })
                    elif current_bb_percent > 95:  # Near upper band
                        signals.append({
                            'type': 'mean_reversion',
                            'indicator': 'bollinger_bands',
                            'direction': 'SELL',
                            'strength': 60,
                            'description': f'Price near upper Bollinger Band ({current_bb_percent:.1f}%)',
                            'entry_reason': 'overbought_reversion'
                        })
            
        except Exception as e:
            self.logger.error(f"Mean reversion signal generation error: {e}")
        
        return signals
    
    def _generate_breakout_signals(self) -> List[Dict]:
        """Generate breakout signals"""
        signals = []
        
        try:
            current_price = self._get_last_price()
            if not current_price:
                return signals
            
            # Support/Resistance breakouts
            for resistance in self.support_resistance.get('resistance', []):
                if current_price > resistance['price'] * 1.001:  # 0.1% buffer
                    signals.append({
                        'type': 'breakout',
                        'indicator': 'resistance_breakout',
                        'direction': 'BUY',
                        'strength': min(80, resistance['strength'] * 20),
                        'description': f"Resistance breakout at {resistance['price']:.5f}",
                        'entry_reason': 'resistance_breakout'
                    })
            
            for support in self.support_resistance.get('support', []):
                if current_price < support['price'] * 0.999:  # 0.1% buffer
                    signals.append({
                        'type': 'breakout',
                        'indicator': 'support_breakdown',
                        'direction': 'SELL',
                        'strength': min(80, support['strength'] * 20),
                        'description': f"Support breakdown at {support['price']:.5f}",
                        'entry_reason': 'support_breakdown'
                    })
            
        except Exception as e:
            self.logger.error(f"Breakout signal generation error: {e}")
        
        return signals
    
    def _filter_and_rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """Filter and rank signals by strength and confluence"""
        try:
            # Filter by minimum strength
            filtered = [s for s in signals if s.get('strength', 0) >= self.signal_threshold]
            
            # Check for confluence (multiple signals in same direction)
            buy_signals = [s for s in filtered if s.get('direction') == 'BUY']
            sell_signals = [s for s in filtered if s.get('direction') == 'SELL']
            
            # Boost strength for confluent signals
            if len(buy_signals) > 1:
                for signal in buy_signals:
                    signal['strength'] = min(100, signal['strength'] * self.confluence_weight)
                    signal['confluence'] = len(buy_signals)
            
            if len(sell_signals) > 1:
                for signal in sell_signals:
                    signal['strength'] = min(100, signal['strength'] * self.confluence_weight)
                    signal['confluence'] = len(sell_signals)
            
            # Add metadata
            for signal in filtered:
                signal['timestamp'] = datetime.now().isoformat()
                signal['symbol'] = self.symbol
                signal['agent'] = 'AGENT_06'
                signal['technical_confidence'] = signal.get('strength', 0)  # For signal coordinator
            
            # Sort by strength
            filtered.sort(key=lambda x: x.get('strength', 0), reverse=True)
            
            return filtered[:10]  # Return top 10 signals
            
        except Exception as e:
            self.logger.error(f"Signal filtering error: {e}")
            return signals
    
    def start_real_time_analysis(self):
        """Start real-time technical analysis"""
        if self.is_analyzing:
            return {"status": "already_running", "message": "Analysis already active"}
        
        try:
            self.is_analyzing = True
            
            def analysis_loop():
                self.logger.info("Starting real-time technical analysis")
                
                while self.is_analyzing:
                    try:
                        if self.market_data is not None:
                            self._perform_full_analysis()
                        
                        time.sleep(self.analysis_interval)
                        
                    except Exception as e:
                        self.logger.error(f"Analysis loop error: {e}")
                        time.sleep(10)
            
            self.analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
            self.analysis_thread.start()
            
            self.status = "ANALYZING"
            return {"status": "started", "message": "Real-time analysis started"}
            
        except Exception as e:
            self.logger.error(f"Failed to start real-time analysis: {e}")
            self.is_analyzing = False
            return {"status": "failed", "message": str(e)}
    
    def stop_real_time_analysis(self):
        """Stop real-time technical analysis"""
        try:
            self.is_analyzing = False
            
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=5)
            
            self.status = "INITIALIZED"
            return {"status": "stopped", "message": "Real-time analysis stopped"}
            
        except Exception as e:
            self.logger.error(f"Error stopping analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_current_analysis(self) -> Dict:
        """Get current technical analysis results"""
        return {
            'indicators': self.indicators,
            'trend_analysis': self.trend_analysis,
            'support_resistance': self.support_resistance,
            'signals': self.signals,
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'current_price': self._get_last_price()
        }
    
    def analyze_current_market(self) -> Dict:
        """Analyze current market conditions and return comprehensive results"""
        try:
            # Perform full analysis
            self._perform_full_analysis()
            
            # Return comprehensive analysis results
            return {
                'status': 'success',
                'symbol': self.symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'indicators': self.indicators,
                'trend_analysis': self.trend_analysis,
                'support_resistance': self.support_resistance,
                'signals': self.signals,
                'current_price': self._get_last_price(),
                'data_points': len(self.market_data) if self.market_data else 0,
                'analysis_summary': {
                    'trend_direction': self.trend_analysis.get('primary_trend', TrendDirection.SIDEWAYS).value if self.trend_analysis else 'UNKNOWN',
                    'trend_strength': self.trend_analysis.get('trend_strength', 0) if self.trend_analysis else 0,
                    'signal_count': len(self.signals) if self.signals else 0,
                    'strongest_signal': max(self.signals, key=lambda x: x.get('strength', 0)) if self.signals else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Market analysis error: {e}")
            return {
                'status': 'error',
                'symbol': self.symbol,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def get_signal_summary(self) -> Dict:
        """Get a summary of current signals"""
        try:
            if not self.signals:
                return {"message": "No signals available"}
            
            buy_signals = [s for s in self.signals if s.get('direction') == 'BUY']
            sell_signals = [s for s in self.signals if s.get('direction') == 'SELL']
            
            strongest_signal = max(self.signals, key=lambda x: x.get('strength', 0))
            
            return {
                'total_signals': len(self.signals),
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'strongest_signal': strongest_signal,
                'avg_strength': sum(s.get('strength', 0) for s in self.signals) / len(self.signals),
                'consensus': 'BULLISH' if len(buy_signals) > len(sell_signals) else 'BEARISH' if len(sell_signals) > len(buy_signals) else 'NEUTRAL'
            }
            
        except Exception as e:
            self.logger.error(f"Signal summary error: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """Get technical analyst performance metrics"""
        return {
            'analysis_count': self.analysis_count,
            'signals_generated': self.signals_generated,
            'signal_history_size': len(self.signal_history),
            'indicators_calculated': len(self.indicators),
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'is_analyzing': self.is_analyzing,
            'data_points': len(self.market_data) if self.market_data is not None else 0,
            'trend_direction': self.trend_analysis.get('primary_trend', TrendDirection.SIDEWAYS).value if self.trend_analysis else 'UNKNOWN',
            'trend_strength': self.trend_analysis.get('trend_strength', 0) if self.trend_analysis else 0
        }
    
    def get_status(self):
        """Get current technical analyst status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'symbol': self.symbol,
            'is_analyzing': self.is_analyzing,
            'analysis_count': self.analysis_count,
            'signals_generated': self.signals_generated,
            'current_signals': len(self.signals),
            'data_points': len(self.market_data) if self.market_data is not None else 0,
            'indicators_available': list(self.indicators.keys()) if self.indicators else [],
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'trend_direction': self.trend_analysis.get('primary_trend', TrendDirection.SIDEWAYS).value if self.trend_analysis else 'UNKNOWN'
        }
    
    def shutdown(self):
        """Clean shutdown of technical analyst"""
        try:
            self.logger.info("Shutting down Technical Analyst...")
            
            # Stop real-time analysis
            self.stop_real_time_analysis()
            
            # Save final metrics
            metrics = self.get_performance_metrics()
            self.logger.info(f"Final metrics: {metrics}")
            
            # Clear memory
            self.indicators.clear()
            self.trend_analysis.clear()
            self.support_resistance = {'support': [], 'resistance': []}
            self.signals.clear()
            self.signal_history.clear()
            self.data_history.clear()
            
            self.status = "SHUTDOWN"
            self.logger.info("Technical Analyst shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Agent test
if __name__ == "__main__":
    # Test the technical analyst
    print("Testing AGENT_06: Technical Analyst")
    print("=" * 40)
    
    # Create technical analyst for EURUSD
    analyst = TechnicalAnalyst("EURUSD")
    result = analyst.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test analysis
        print("\nTesting technical analysis...")
        analysis = analyst.get_current_analysis()
        print(f"Indicators calculated: {len(analysis['indicators'])}")
        print(f"Signals generated: {len(analysis['signals'])}")
        
        # Test signal summary
        print("\nTesting signal summary...")
        summary = analyst.get_signal_summary()
        print(f"Signal summary: {summary}")
        
        # Test performance metrics
        metrics = analyst.get_performance_metrics()
        print(f"\nPerformance metrics: {metrics}")
        
        # Test status
        status = analyst.get_status()
        print(f"\nStatus: {status}")
        
        # Test shutdown
        print("\nShutting down...")
        analyst.shutdown()
        
    print("Technical Analyst test completed")