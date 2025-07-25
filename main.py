from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
from datetime import datetime
import random

app = FastAPI(title="AI Trading System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
market_data_store = []
account_info_store = {}
ai_predictions_store = []

class SimpleAI:
    def __init__(self):
        self.confidence = 0.75
        
    def make_prediction(self, market_data):
        if len(market_data) < 3:
            return {"action": "HOLD", "confidence": 0.5, "reasoning": "Not enough data"}
        
        # Get last few prices
        prices = [float(d['price']) for d in market_data[-5:]]
        current_price = prices[-1]
        
        # Simple trend analysis
        if len(prices) >= 3:
            trend = (prices[-1] - prices[-3]) / prices[-3]
            
            if trend > 0.0001:  # Rising trend
                action = "BUY"
                confidence = min(0.9, 0.6 + abs(trend) * 1000)
                reasoning = f"Upward trend detected: {trend:.4%}"
            elif trend < -0.0001:  # Falling trend
                action = "SELL" 
                confidence = min(0.9, 0.6 + abs(trend) * 1000)
                reasoning = f"Downward trend detected: {trend:.4%}"
            else:
                action = "HOLD"
                confidence = 0.5
                reasoning = "Sideways movement"
        else:
            action = "HOLD"
            confidence = 0.5
            reasoning = "Insufficient price history"
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning
        }

ai_engine = SimpleAI()

@app.get("/")
async def root():
    return {
        "message": "AI Trading System API", 
        "status": "running",
        "version": "1.0"
    }

@app.post("/webhook")
async def receive_mt5_data(request: Request):
    try:
        data = await request.json()
        
        if data.get("type") == "market_data":
            market_entry = {
                "symbol": data.get("symbol", "EURUSD"),
                "price": float(data.get("price", 1.0850)),
                "volume": int(data.get("volume", 1000)),
                "timestamp": data.get("timestamp", datetime.now().isoformat())
            }
            
            market_data_store.append(market_entry)
            
            # Keep only last 100 records
            if len(market_data_store) > 100:
                market_data_store.pop(0)
            
            # Make AI prediction
            prediction = ai_engine.make_prediction(market_data_store)
            ai_predictions_store.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": market_entry["symbol"],
                "prediction": prediction
            })
            
            # Keep only last 30 predictions
            if len(ai_predictions_store) > 30:
                ai_predictions_store.pop(0)
                
        elif data.get("type") == "account_info":
            account_info_store.update({
                "balance": float(data.get("balance", 10000)),
                "equity": float(data.get("equity", 10000)),
                "profit": float(data.get("profit", 0)),
                "timestamp": data.get("timestamp", datetime.now().isoformat())
            })
        
        return {"status": "success", "message": "Data received"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/market-data")
async def get_market_data():
    return market_data_store[-30:]

@app.get("/account-info") 
async def get_account_info():
    return account_info_store

@app.get("/ai-predictions")
async def get_ai_predictions():
    return ai_predictions_store[-10:]

@app.get("/system-status")
async def get_system_status():
    return {
        "ai_active": True,
        "market_data_count": len(market_data_store),
        "predictions_count": len(ai_predictions_store),
        "last_update": market_data_store[-1]["timestamp"] if market_data_store else None
    }

@app.get("/news")
async def get_news():
    # Simple mock news
    news_items = [
        {
            "title": "Fed Considers Interest Rate Changes",
            "description": "Federal Reserve meeting scheduled this week",
            "source": {"name": "Economic Times"},
            "publishedAt": datetime.now().isoformat(),
            "sentiment": "neutral"
        },
        {
            "title": "EUR/USD Shows Volatility", 
            "description": "Euro strengthens against dollar in morning trading",
            "source": {"name": "Forex News"},
            "publishedAt": datetime.now().isoformat(),
            "sentiment": "bullish"
        }
    ]
    return news_items

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
