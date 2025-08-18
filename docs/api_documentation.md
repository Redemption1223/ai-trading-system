# AGI Trading System - API Documentation

## 🔗 API Overview

The AGI Trading System exposes RESTful APIs for integration and control.

**Base URL**: `http://localhost:5000/api`  
**Current Status**: Phase 1 - Template endpoints defined  
**Implementation**: Phase 2 development

---

## 🎯 Signal Endpoints

### GET /api/signals/recent
Get recent trading signals

**Response**:
```json
{
  "signals": [
    {
      "id": "signal_001",
      "timestamp": "2024-01-15T10:30:00Z",
      "symbol": "EURUSD",
      "type": "buy",
      "confidence": 0.85,
      "agent": "NEURAL_BRAIN",
      "price": 1.0845,
      "target": 1.0865,
      "stop_loss": 1.0825
    }
  ]
}
```

### GET /api/signals/live
WebSocket endpoint for real-time signals

**Connection**: `ws://localhost:5000/signals`

**Message Format**:
```json
{
  "type": "signal",
  "data": {
    "symbol": "GBPUSD",
    "action": "sell",
    "confidence": 0.72,
    "timestamp": "2024-01-15T10:31:00Z"
  }
}
```

---

## 🤖 Agent Endpoints

### GET /api/agents/status
Get status of all agents

**Response**:
```json
{
  "agents": [
    {
      "id": "AGENT_01",
      "name": "MT5_CONNECTOR",
      "status": "online",
      "last_signal": "2024-01-15T10:30:00Z",
      "signals_generated": 45,
      "accuracy": 0.67
    }
  ]
}
```

### POST /api/agents/{agent_id}/restart
Restart specific agent

**Response**:
```json
{
  "success": true,
  "message": "Agent restarted successfully",
  "agent_id": "AGENT_05"
}
```

---

## 📊 Market Data Endpoints

### GET /api/chart/{symbol}/{timeframe}
Get chart data for symbol and timeframe

**Parameters**:
- `symbol`: Trading symbol (e.g., EURUSD)
- `timeframe`: M1, M5, M15, H1, H4, D1

**Response**:
```json
{
  "symbol": "EURUSD",
  "timeframe": "H1", 
  "data": {
    "timestamps": ["2024-01-15T09:00:00Z", ...],
    "open": [1.0840, ...],
    "high": [1.0850, ...],
    "low": [1.0835, ...],
    "close": [1.0845, ...],
    "volume": [1200, ...]
  }
}
```

### GET /api/market/symbols
Get available trading symbols

**Response**:
```json
{
  "symbols": [
    {
      "symbol": "EURUSD",
      "description": "Euro vs US Dollar",
      "type": "forex",
      "spread": 0.8,
      "active": true
    }
  ]
}
```

---

## 📈 Performance Endpoints

### GET /api/metrics
Get system performance metrics

**Response**:
```json
{
  "totalSignals": 234,
  "accuracy": 0.68,
  "profit": 1250.50,
  "activeAgents": 10,
  "uptime": 86400,
  "cpu_usage": 25.5,
  "memory_usage": 1024
}
```

### GET /api/performance/detailed
Get detailed performance breakdown

**Response**:
```json
{
  "daily_stats": {
    "signals": 45,
    "trades": 12,
    "win_rate": 0.75,
    "profit_loss": 234.50
  },
  "agent_performance": [
    {
      "agent": "NEURAL_BRAIN",
      "signals": 15,
      "accuracy": 0.80
    }
  ]
}
```

---

## ⚙️ Configuration Endpoints

### GET /api/config
Get current system configuration

**Response**:
```json
{
  "risk_settings": {
    "max_risk_per_trade": 0.02,
    "max_daily_loss": 0.05
  },
  "mt5_settings": {
    "auto_connect": true,
    "timeout": 30
  }
}
```

### POST /api/config
Update system configuration

**Request Body**:
```json
{
  "risk_settings": {
    "max_risk_per_trade": 0.015
  }
}
```

---

## 🔧 System Control Endpoints

### POST /api/system/start
Start the trading system

**Response**:
```json
{
  "success": true,
  "message": "System started successfully",
  "agents_started": 12
}
```

### POST /api/system/stop
Stop the trading system

**Response**:
```json
{
  "success": true,
  "message": "System stopped successfully"
}
```

### GET /api/system/health
Get system health status

**Response**:
```json
{
  "status": "healthy",
  "mt5_connection": "connected",
  "agents_online": 12,
  "last_update": "2024-01-15T10:30:00Z"
}
```

---

## 🧪 Testing Endpoints

### POST /api/backtest
Run backtesting simulation

**Request Body**:
```json
{
  "symbol": "EURUSD",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "initial_balance": 10000,
  "strategy": "neural_signals"
}
```

**Response**:
```json
{
  "backtest_id": "bt_001",
  "status": "running",
  "estimated_duration": "5 minutes"
}
```

### GET /api/backtest/{test_id}/results
Get backtesting results

**Response**:
```json
{
  "test_id": "bt_001",
  "status": "completed",
  "results": {
    "total_trades": 45,
    "win_rate": 0.67,
    "total_return": 0.15,
    "max_drawdown": 0.08,
    "sharpe_ratio": 1.25
  }
}
```

---

## 🔒 Authentication (Phase 2)

### POST /api/auth/login
User authentication

**Request Body**:
```json
{
  "username": "trader1",
  "password": "secure_password"
}
```

**Response**:
```json
{
  "token": "jwt_token_here",
  "expires_at": "2024-01-15T18:00:00Z"
}
```

---

## 📡 WebSocket Events

### Connection
```javascript
const socket = new WebSocket('ws://localhost:5000/signals');
```

### Event Types
- `signal`: New trading signal
- `status`: Agent status update
- `metric`: Performance metric update
- `alert`: System alert/warning

### Example Usage
```javascript
socket.onmessage = function(event) {
  const data = JSON.parse(event.data);
  switch(data.type) {
    case 'signal':
      handleNewSignal(data.signal);
      break;
    case 'status':
      updateAgentStatus(data.agent);
      break;
  }
};
```

---

## 🛠️ Error Handling

### Error Response Format
```json
{
  "error": true,
  "code": "AGENT_OFFLINE",
  "message": "Neural Brain agent is currently offline",
  "details": {
    "agent": "AGENT_05",
    "last_seen": "2024-01-15T09:45:00Z"
  }
}
```

### HTTP Status Codes
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `404`: Not Found  
- `500`: Internal Server Error
- `503`: Service Unavailable

---

## 📋 Implementation Status

**Phase 1**: ✅ API endpoints defined  
**Phase 2**: 🚧 Implementation in progress  
**Phase 3**: ⏳ Advanced features  

### Current Limitations
- All endpoints return template responses
- WebSocket connections not yet implemented
- Authentication system pending
- Rate limiting not implemented

### Next Phase Development
1. Implement core signal endpoints
2. Add MT5 integration APIs
3. Create real-time WebSocket handling
4. Add authentication and security
5. Implement performance monitoring

---

## 🧪 Testing the API

### Using curl
```bash
# Get system health
curl http://localhost:5000/api/system/health

# Get recent signals  
curl http://localhost:5000/api/signals/recent

# Get agent status
curl http://localhost:5000/api/agents/status
```

### Using Python
```python
import requests

# Get metrics
response = requests.get('http://localhost:5000/api/metrics')
metrics = response.json()
print(f"Total signals: {metrics['totalSignals']}")
```

**Ready for Phase 2 API Implementation!**