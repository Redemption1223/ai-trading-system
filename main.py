from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Trading System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for real-time data
market_data_store = []
account_info_store = {}
ai_predictions_store = []

class SimpleAIEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_trained = False
        self.feature_window = 20
        
    def calculate_technical_indicators(self, prices):
        """Calculate basic technical indicators"""
        if len(prices) < self.feature_window:
            return None
            
        df = pd.DataFrame({'price': prices})
        
        # Price changes
        df['returns'] = df['price'].pct_change()
        df['price_change'] = df['price'].diff()
        
        # Moving averages
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_10'] = df['price'].rolling(10).mean()
        df['sma_20'] = df['price'].rolling(20).mean()
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(20).mean()
        bb_std = df['price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume trend (if available)
        df['volume_avg'] = df.get('volume', pd.Series([1000]*len(df))).rolling(10).mean()
        
        return df
    
    def extract_features(self, df):
        """Extract features for ML model"""
        if df is None or len(df) < self.feature_window:
            return None
            
        features = []
        latest_idx = len(df) - 1
        
        if latest_idx >= self.feature_window:
            row = df.iloc[latest_idx]
            features = [
                row.get('returns', 0),
                row.get('rsi', 50),
                row.get('bb_position', 0.5),
                (row.get('sma_5', 0) - row.get('sma_20', 0)) / row.get('price', 1),
                row.get('price_change', 0) / row.get('price', 1)
            ]
            
        return np.array(features).reshape(1, -1) if features else None
    
    def train_model(self, historical_data):
        """Train the model with synthetic data initially"""
        try:
            # Create synthetic training data
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic features
            X = np.random.randn(n_samples, 5)
            
            # Generate synthetic labels (0=SELL, 1=HOLD, 2=BUY)
            # Simple logic: if RSI > 70 → SELL, if RSI < 30 → BUY, else HOLD
            y = np.array([
                2 if x[1] < -1 else 0 if x[1] > 1 else 1 
                for x in X
            ])
            
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("AI model trained successfully")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
    
    def make_prediction(self, market_data):
        """Make trading prediction"""
        try:
            if len(market_data) < 5:
                return {
                    "action": "HOLD", 
                    "confidence": 0.5, 
                    "reasoning": "Insufficient data"
                }
            
            # Extract prices
            prices = [float(d['price']) for d in market_data[-50:]]
            
            # Calculate indicators
            df = self.calculate_technical_indicators(prices)
            features = self.extract_features(df)
            
            if features is None:
                return {
                    "action": "HOLD", 
                    "confidence": 0.5, 
                    "reasoning": "Feature extraction failed"
                }
            
            # Train model if not trained
            if not self.is_trained:
                self.train_model(market_data)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            actions = ["SELL", "HOLD", "BUY"]
            current_price = prices[-1]
            
            # Add reasoning
            rsi = df.iloc[-1]['rsi'] if 'rsi' in df.columns else 50
            sma_signal = "bullish" if df.iloc[-1]['sma_5'] > df.iloc[-1]['sma_20'] else "bearish"
            
            reasoning = f"RSI: {rsi:.1f}, Trend: {sma_signal}, Price: {current_price:.5f}"
            
            return {
                "action": actions[prediction],
                "confidence": float(confidence),
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "action": "HOLD", 
                "confidence": 0.5, 
                "reasoning": f"Error: {str(e)}"
            }

# Initialize AI engine
ai_engine = SimpleAIEngine()

@app.get("/")
async def root():
    return {
        "message": "AI Trading System API", 
        "status": "running",
        "ai_trained": ai_engine.is_trained
    }

@app.post("/webhook")
async def receive_mt5_data(request: Request):
    """Receive data from MT5 EA"""
    try:
        data = await request.json()
        logger.info(f"Received data: {data}")
        
        if data.get("type") == "market_data":
            # Store market data
            market_entry = {
                "symbol": data.get("symbol", "EURUSD"),
                "price": float(data.get("price", 0)),
                "volume": int(data.get("volume", 0)),
                "timestamp": data.get("timestamp", datetime.now().isoformat())
            }
            
            market_data_store.append(market_entry)
            
            # Keep only last 200 records
            if len(market_data_store) > 200:
                market_data_store.pop(0)
            
            # Make AI prediction
            if len(market_data_store) >= 5:
                prediction = ai_engine.make_prediction(market_data_store)
                ai_predictions_store.append({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": market_entry["symbol"],
                    "prediction": prediction
                })
                
                # Keep only last 50 predictions
                if len(ai_predictions_store) > 50:
                    ai_predictions_store.pop(0)
                    
        elif data.get("type") == "account_info":
            # Store account info
            account_info_store.update({
                "balance": float(data.get("balance", 0)),
                "equity": float(data.get("equity", 0)),
                "profit": float(data.get("profit", 0)),
                "timestamp": data.get("timestamp", datetime.now().isoformat())
            })
        
        return {"status": "success", "message": "Data received"}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/market-data")
async def get_market_data():
    """Get recent market data"""
    return market_data_store[-50:]  # Last 50 records

@app.get("/account-info")
async def get_account_info():
    """Get current account information"""
    return account_info_store

@app.get("/ai-predictions")
async def get_ai_predictions():
    """Get recent AI predictions"""
    return ai_predictions_store[-20:]  # Last 20 predictions

@app.get("/system-status")
async def get_system_status():
    """Get system status"""
    return {
        "ai_trained": ai_engine.is_trained,
        "market_data_count": len(market_data_store),
        "predictions_count": len(ai_predictions_store),
        "last_update": market_data_store[-1]["timestamp"] if market_data_store else None
    }

@app.get("/news")
async def get_news():
    """Get market news (mock data for now)"""
    # Mock news data
    mock_news = [
        {
            "title": "ECB Keeps Interest Rates Unchanged",
            "description": "European Central Bank maintains current monetary policy",
            "source": {"name": "Financial Times"},
            "publishedAt": datetime.now().isoformat(),
            "sentiment": "neutral"
        },
        {
            "title": "USD Strengthens Against Major Currencies",
            "description": "Dollar gains on positive economic indicators",
            "source": {"name": "Reuters"},
            "publishedAt": datetime.now().isoformat(),
            "sentiment": "bullish"
        },
        {
            "title": "Oil Prices Rise on Supply Concerns",
            "description": "Crude oil futures up 2% in early trading",
            "source": {"name": "Bloomberg"},
            "publishedAt": datetime.now().isoformat(),
            "sentiment": "bullish"
        }
    ]
    
    return mock_news

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
