from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import asyncio
import logging
from dataclasses import dataclass
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced AI Trading System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trading pairs to monitor
FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", 
    "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "CADCHF",
    "CADJPY", "CHFJPY", "EURAUD", "EURCAD", "EURCHF"
]

# Global storage
market_data_store = {pair: [] for pair in FOREX_PAIRS}
account_info_store = {}
ai_predictions_store = {pair: [] for pair in FOREX_PAIRS}
trading_signals_store = []
pattern_analysis_store = {pair: {} for pair in FOREX_PAIRS}
active_trades = []

@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY, SELL, CLOSE
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    timestamp: str

class AdvancedNeuralAI:
    def __init__(self):
        self.models = {}
        self.feature_window = 50
        self.min_confidence_threshold = 0.75
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        
    def calculate_advanced_indicators(self, prices: List[float], volumes: List[float] = None) -> Dict:
        """Calculate advanced technical indicators"""
        if len(prices) < self.feature_window:
            return {}
            
        df = pd.DataFrame({'price': prices, 'volume': volumes or [1000] * len(prices)})
        
        try:
            # Price-based indicators
            df['returns'] = df['price'].pct_change()
            df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
            
            # Moving averages (multiple timeframes)
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['price'].rolling(period).mean()
                df[f'ema_{period}'] = df['price'].ewm(span=period).mean()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['price'].rolling(bb_period).mean()
            bb_std_val = df['price'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['price'].ewm(span=12).mean()
            exp2 = df['price'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Stochastic Oscillator
            low_14 = df['price'].rolling(14).min()
            high_14 = df['price'].rolling(14).max()
            df['stoch_k'] = 100 * (df['price'] - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Average True Range (ATR)
            df['high'] = df['price'] * 1.001  # Simulated high
            df['low'] = df['price'] * 0.999   # Simulated low
            df['prev_close'] = df['price'].shift(1)
            
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['prev_close'])
            tr3 = abs(df['low'] - df['prev_close'])
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price patterns
            df['higher_highs'] = (df['price'] > df['price'].shift(1)) & (df['price'].shift(1) > df['price'].shift(2))
            df['lower_lows'] = (df['price'] < df['price'].shift(1)) & (df['price'].shift(1) < df['price'].shift(2))
            
            # Volatility
            df['volatility'] = df['returns'].rolling(20).std()
            df['price_velocity'] = df['price'].diff(5) / 5  # 5-period velocity
            
            return df.iloc[-1].to_dict()
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return {}
    
    def detect_patterns(self, prices: List[float], symbol: str) -> Dict:
        """Detect chart patterns"""
        if len(prices) < 20:
            return {"pattern": "insufficient_data", "strength": 0}
        
        try:
            recent_prices = prices[-20:]
            
            # Support and Resistance levels
            highs = []
            lows = []
            
            for i in range(2, len(recent_prices) - 2):
                # Local highs
                if (recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i-2] and
                    recent_prices[i] > recent_prices[i+1] and recent_prices[i] > recent_prices[i+2]):
                    highs.append(recent_prices[i])
                
                # Local lows
                if (recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i-2] and
                    recent_prices[i] < recent_prices[i+1] and recent_prices[i] < recent_prices[i+2]):
                    lows.append(recent_prices[i])
            
            current_price = recent_prices[-1]
            
            # Trend detection
            short_trend = np.polyfit(range(5), recent_prices[-5:], 1)[0]
            medium_trend = np.polyfit(range(10), recent_prices[-10:], 1)[0]
            long_trend = np.polyfit(range(20), recent_prices[-20:], 1)[0]
            
            # Pattern classification
            if short_trend > 0 and medium_trend > 0 and long_trend > 0:
                pattern = "strong_uptrend"
                strength = min(0.9, abs(short_trend) * 10000)
            elif short_trend < 0 and medium_trend < 0 and long_trend < 0:
                pattern = "strong_downtrend"
                strength = min(0.9, abs(short_trend) * 10000)
            elif abs(short_trend) < 0.0001:
                pattern = "sideways"
                strength = 0.3
            elif short_trend > 0 > medium_trend:
                pattern = "reversal_up"
                strength = 0.7
            elif short_trend < 0 < medium_trend:
                pattern = "reversal_down"
                strength = 0.7
            else:
                pattern = "mixed_signals"
                strength = 0.4
            
            return {
                "pattern": pattern,
                "strength": strength,
                "short_trend": short_trend,
                "medium_trend": medium_trend,
                "long_trend": long_trend,
                "support_levels": lows[-3:] if lows else [],
                "resistance_levels": highs[-3:] if highs else [],
                "current_price": current_price
            }
            
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            return {"pattern": "error", "strength": 0}
    
    def neural_prediction(self, symbol: str, indicators: Dict, pattern: Dict) -> Dict:
        """Advanced neural network prediction"""
        try:
            if not indicators or not pattern:
                return {"action": "HOLD", "confidence": 0.5, "reasoning": "Insufficient data"}
            
            # Neural network simulation using weighted features
            features = []
            
            # Technical indicator signals
            rsi = indicators.get('rsi', 50)
            macd_histogram = indicators.get('macd_histogram', 0)
            bb_position = indicators.get('bb_position', 0.5)
            stoch_k = indicators.get('stoch_k', 50)
            atr = indicators.get('atr', 0)
            
            # Pattern signals
            pattern_strength = pattern.get('strength', 0)
            pattern_type = pattern.get('pattern', 'mixed_signals')
            
            # Feature engineering (simulating neural network layers)
            momentum_signal = 0
            if rsi < 30 and stoch_k < 20:  # Oversold
                momentum_signal += 0.3
            elif rsi > 70 and stoch_k > 80:  # Overbought
                momentum_signal -= 0.3
            
            trend_signal = 0
            if pattern_type == "strong_uptrend":
                trend_signal += pattern_strength
            elif pattern_type == "strong_downtrend":
                trend_signal -= pattern_strength
            elif pattern_type == "reversal_up":
                trend_signal += pattern_strength * 0.8
            elif pattern_type == "reversal_down":
                trend_signal -= pattern_strength * 0.8
            
            volatility_adjustment = min(1.0, atr * 1000) if atr else 0.5
            
            # Combined neural score
            neural_score = (momentum_signal * 0.4 + trend_signal * 0.6) * volatility_adjustment
            
            # Decision logic
            if neural_score > 0.3:
                action = "BUY"
                confidence = min(0.95, 0.6 + abs(neural_score))
            elif neural_score < -0.3:
                action = "SELL"
                confidence = min(0.95, 0.6 + abs(neural_score))
            else:
                action = "HOLD"
                confidence = 0.5
            
            reasoning = f"Neural: {neural_score:.3f}, Pattern: {pattern_type}, RSI: {rsi:.1f}, Volatility: {volatility_adjustment:.3f}"
            
            return {
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "neural_score": neural_score,
                "technical_signals": {
                    "rsi": rsi,
                    "macd_histogram": macd_histogram,
                    "bb_position": bb_position,
                    "pattern": pattern_type,
                    "pattern_strength": pattern_strength
                }
            }
            
        except Exception as e:
            logger.error(f"Neural prediction error: {e}")
            return {"action": "HOLD", "confidence": 0.5, "reasoning": f"Error: {str(e)}"}
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, account_balance: float) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            if not all([entry_price, stop_loss, account_balance]) or account_balance <= 0:
                return 0.01  # Minimum size
            
            # Risk per trade (2% of account)
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Distance to stop loss
            stop_distance = abs(entry_price - stop_loss)
            
            if stop_distance == 0:
                return 0.01
            
            # Position size calculation
            # For forex: Position Size = Risk Amount / (Stop Distance * Pip Value)
            # Simplified: assuming 1 pip = 0.0001 for major pairs
            pip_value = 10  # Approximate for 1 lot
            position_size = risk_amount / (stop_distance * pip_value)
            
            # Ensure position size is within reasonable bounds
            position_size = max(0.01, min(1.0, position_size))
            
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.01
    
    def generate_trading_signal(self, symbol: str, market_data: List[Dict], account_balance: float) -> Optional[TradingSignal]:
        """Generate complete trading signal with entry, stop loss, and take profit"""
        try:
            if len(market_data) < self.feature_window:
                return None
            
            prices = [float(d['price']) for d in market_data[-self.feature_window:]]
            volumes = [int(d.get('volume', 1000)) for d in market_data[-self.feature_window:]]
            
            # Calculate indicators and patterns
            indicators = self.calculate_advanced_indicators(prices, volumes)
            pattern = self.detect_patterns(prices, symbol)
            
            # Store pattern analysis
            pattern_analysis_store[symbol] = {
                "pattern": pattern,
                "indicators": indicators,
                "timestamp": datetime.now().isoformat()
            }
            
            # Get neural prediction
            prediction = self.neural_prediction(symbol, indicators, pattern)
            
            if prediction['confidence'] < self.min_confidence_threshold:
                return None
            
            current_price = prices[-1]
            atr = indicators.get('atr', current_price * 0.001)
            
            if prediction['action'] in ['BUY', 'SELL']:
                # Calculate stop loss and take profit
                if prediction['action'] == 'BUY':
                    stop_loss = current_price - (atr * 2)
                    take_profit = current_price + (atr * 3)
                else:  # SELL
                    stop_loss = current_price + (atr * 2)
                    take_profit = current_price - (atr * 3)
                
                # Calculate position size
                position_size = self.calculate_position_size(symbol, current_price, stop_loss, account_balance)
                
                return TradingSignal(
                    symbol=symbol,
                    action=prediction['action'],
                    confidence=prediction['confidence'],
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    reasoning=prediction['reasoning'],
                    timestamp=datetime.now().isoformat()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None

# Initialize AI engine
ai_engine = AdvancedNeuralAI()

# Auto-trading configuration
AUTO_TRADING_ENABLED = False  # Start disabled for safety
DEMO_MODE = True  # Always start in demo mode

@app.get("/")
async def root():
    return {
        "message": "Advanced AI Trading System",
        "status": "running",
        "version": "2.0",
        "supported_pairs": len(FOREX_PAIRS),
        "auto_trading": AUTO_TRADING_ENABLED,
        "demo_mode": DEMO_MODE
    }

@app.post("/webhook")
async def receive_mt5_data(request: Request):
    """Receive data from MT5 EA"""
    try:
        data = await request.json()
        
        if data.get("type") == "market_data":
            symbol = data.get("symbol", "EURUSD").upper()
            
            # Store market data for supported pairs
            if symbol in FOREX_PAIRS:
                market_entry = {
                    "symbol": symbol,
                    "price": float(data.get("price", 0)),
                    "bid": float(data.get("bid", 0)),
                    "ask": float(data.get("ask", 0)),
                    "volume": int(data.get("volume", 0)),
                    "timestamp": data.get("timestamp", datetime.now().isoformat())
                }
                
                market_data_store[symbol].append(market_entry)
                
                # Keep only last 200 records per pair
                if len(market_data_store[symbol]) > 200:
                    market_data_store[symbol].pop(0)
                
                # Generate AI prediction and trading signal
                account_balance = account_info_store.get('balance', 10000)
                
                # Make prediction
                if len(market_data_store[symbol]) >= 10:
                    prices = [d['price'] for d in market_data_store[symbol][-50:]]
                    indicators = ai_engine.calculate_advanced_indicators(prices)
                    pattern = ai_engine.detect_patterns(prices, symbol)
                    prediction = ai_engine.neural_prediction(symbol, indicators, pattern)
                    
                    ai_predictions_store[symbol].append({
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "prediction": prediction,
                        "pattern": pattern
                    })
                    
                    # Keep only last 50 predictions per pair
                    if len(ai_predictions_store[symbol]) > 50:
                        ai_predictions_store[symbol].pop(0)
                    
                    # Generate trading signal if confidence is high enough
                    trading_signal = ai_engine.generate_trading_signal(symbol, market_data_store[symbol], account_balance)
                    if trading_signal and AUTO_TRADING_ENABLED:
                        trading_signals_store.append(trading_signal.__dict__)
                        
                        # Keep only last 100 signals
                        if len(trading_signals_store) > 100:
                            trading_signals_store.pop(0)
                        
                        logger.info(f"Trading signal generated: {symbol} {trading_signal.action} @ {trading_signal.entry_price}")
                
        elif data.get("type") == "account_info":
            account_info_store.update({
                "balance": float(data.get("balance", 0)),
                "equity": float(data.get("equity", 0)),
                "profit": float(data.get("profit", 0)),
                "free_margin": float(data.get("free_margin", 0)),
                "margin": float(data.get("margin", 0)),
                "timestamp": data.get("timestamp", datetime.now().isoformat())
            })
        
        return {"status": "success", "message": "Data processed"}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/market-data/{symbol}")
async def get_symbol_data(symbol: str):
    """Get market data for specific symbol"""
    symbol = symbol.upper()
    return market_data_store.get(symbol, [])[-50:]

@app.get("/market-data")
async def get_all_market_data():
    """Get market data for all symbols"""
    result = {}
    for symbol in FOREX_PAIRS:
        if market_data_store[symbol]:
            result[symbol] = market_data_store[symbol][-20:]
    return result

@app.get("/ai-predictions/{symbol}")
async def get_symbol_predictions(symbol: str):
    """Get AI predictions for specific symbol"""
    symbol = symbol.upper()
    return ai_predictions_store.get(symbol, [])[-20:]

@app.get("/ai-predictions")
async def get_all_predictions():
    """Get AI predictions for all symbols"""
    result = {}
    for symbol in FOREX_PAIRS:
        if ai_predictions_store[symbol]:
            result[symbol] = ai_predictions_store[symbol][-5:]
    return result

@app.get("/trading-signals")
async def get_trading_signals():
    """Get recent trading signals"""
    return trading_signals_store[-20:]

@app.get("/pattern-analysis")
async def get_pattern_analysis():
    """Get pattern analysis for all symbols"""
    return pattern_analysis_store

@app.get("/pattern-analysis/{symbol}")
async def get_symbol_pattern(symbol: str):
    """Get pattern analysis for specific symbol"""
    symbol = symbol.upper()
    return pattern_analysis_store.get(symbol, {})

@app.get("/account-info")
async def get_account_info():
    return account_info_store

@app.get("/system-status")
async def get_system_status():
    active_pairs = sum(1 for symbol in FOREX_PAIRS if market_data_store[symbol])
    total_predictions = sum(len(predictions) for predictions in ai_predictions_store.values())
    
    return {
        "ai_active": True,
        "auto_trading_enabled": AUTO_TRADING_ENABLED,
        "demo_mode": DEMO_MODE,
        "active_pairs": active_pairs,
        "total_pairs": len(FOREX_PAIRS),
        "total_predictions": total_predictions,
        "trading_signals_count": len(trading_signals_store),
        "last_update": datetime.now().isoformat()
    }

@app.post("/toggle-auto-trading")
async def toggle_auto_trading(request: Request):
    """Toggle auto-trading on/off"""
    global AUTO_TRADING_ENABLED
    try:
        data = await request.json()
        AUTO_TRADING_ENABLED = data.get("enabled", False)
        
        logger.info(f"Auto-trading {'enabled' if AUTO_TRADING_ENABLED else 'disabled'}")
        
        return {
            "status": "success",
            "auto_trading_enabled": AUTO_TRADING_ENABLED,
            "message": f"Auto-trading {'enabled' if AUTO_TRADING_ENABLED else 'disabled'}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/supported-pairs")
async def get_supported_pairs():
    """Get list of supported forex pairs"""
    return {"pairs": FOREX_PAIRS}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
