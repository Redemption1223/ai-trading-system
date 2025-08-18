# AGI Trading System - User Guide

## 🚀 Getting Started

Welcome to the AGI Trading System - an artificial intelligence-powered trading platform that combines machine learning, technical analysis, and risk management for automated forex trading.

**Current Status**: Phase 1 Complete - Foundation Ready  
**Next Phase**: Full functionality implementation

---

## 📋 Quick Start Checklist

### Prerequisites Setup
- [ ] Windows OS with MetaTrader 5 installed
- [ ] Python 3.8+ environment
- [ ] Broker account (demo recommended for testing)
- [ ] Stable internet connection

### Installation Steps
1. **Run Setup**: Execute `scripts\setup_environment.bat`
2. **Install Dependencies**: Run `scripts\install_dependencies.bat` 
3. **Configure Settings**: Edit configuration files in `/config/`
4. **Test System**: Run `python scripts\verify_setup.py`
5. **Start System**: Execute `scripts\start_system.bat`

---

## 🖥️ Dashboard Overview

### Main Interface Components

#### 1. Signal Panel (Left)
- **Purpose**: Displays real-time trading signals
- **Information**: Signal type, confidence, source agent
- **Actions**: Review signal history, filter by criteria
- **Status**: Template ready - awaiting Phase 2

#### 2. Chart View (Center) 
- **Purpose**: Market analysis and visualization
- **Features**: Multi-timeframe charts, technical indicators
- **Controls**: Symbol selection, timeframe switching
- **Status**: Template ready - awaiting Phase 2

#### 3. System Monitor (Right)
- **Purpose**: System health and performance tracking
- **Displays**: Agent status, system metrics, alerts
- **Monitoring**: CPU usage, memory, connection status
- **Status**: Template ready - awaiting Phase 2

### Header Metrics
- **Signal Count**: Total signals generated today
- **Accuracy**: Overall signal accuracy percentage  
- **P&L**: Profit/Loss for current session
- **Connection**: MT5 connection status indicator

---

## ⚙️ Configuration Guide

### Risk Management Settings
Location: `/config/risk_parameters.yaml`

```yaml
risk_settings:
  max_risk_per_trade: 0.02      # 2% of account per trade
  max_daily_loss: 0.05          # 5% daily loss limit
  max_open_positions: 5         # Maximum concurrent trades
```

**Important**: Always set conservative risk limits, especially when starting.

### MT5 Connection Setup
Location: `/config/mt5_windows_config.yaml`

```yaml
broker_settings:
  name: "FxPro"                 # Your broker name
  server: "auto"                # Auto-detect server
  # Add login/password in .env file
```

**Security**: Never store credentials in config files. Use `.env` file:
```
MT5_LOGIN=your_login_number
MT5_PASSWORD=your_password
MT5_SERVER=your_server_name
```

### Trading Symbols
Location: `/config/chart_selection.yaml`

```yaml
default_symbols:
  - "EURUSD"    # Major pair - low spread
  - "GBPUSD"    # Volatile major pair
  - "USDJPY"    # Trending pair
```

**Recommendation**: Start with major currency pairs for better liquidity and lower spreads.

---

## 🤖 Understanding the Agents

### Core Trading Agents

#### 🔌 AGENT_01: MT5 Connector
- **Function**: Manages MetaTrader 5 connection
- **Status**: Connected/Disconnected indicator
- **Actions**: Auto-reconnect, health monitoring

#### 🎯 AGENT_02: Signal Coordinator  
- **Function**: Combines signals from all agents
- **Logic**: Weighted decision making
- **Output**: Final BUY/SELL/HOLD decisions

#### ⚠️ AGENT_03: Risk Calculator
- **Function**: Position sizing and risk management
- **Calculations**: Stop loss, take profit, position size
- **Protection**: Prevents over-leveraging

#### 📊 AGENT_04: Chart Analyzer
- **Function**: Technical chart pattern recognition
- **Analysis**: Trends, support/resistance, patterns
- **Signals**: Visual-based trading opportunities

### Intelligence Layer

#### 🧠 AGENT_05: Neural Brain
- **Function**: Machine learning pattern recognition
- **Technology**: Neural networks, deep learning
- **Capability**: Complex pattern detection

#### 📈 AGENT_06: Technical Analyst
- **Function**: Traditional technical analysis
- **Indicators**: RSI, MACD, Moving Averages
- **Signals**: TA-based recommendations

#### 📰 AGENT_07: News Sentiment
- **Function**: News and sentiment analysis
- **Sources**: Economic calendars, news feeds
- **Impact**: Fundamental analysis integration

---

## 📊 Reading Trading Signals

### Signal Components

#### Signal Types
- 🟢 **BUY**: Bullish signal - price expected to rise
- 🔴 **SELL**: Bearish signal - price expected to fall  
- 🟡 **HOLD**: Neutral signal - no clear direction

#### Confidence Levels
- **90-100%**: Very High - Strong conviction
- **70-89%**: High - Good probability
- **50-69%**: Medium - Moderate confidence
- **Below 50%**: Low - Weak signal

#### Signal Information
```
Symbol: EURUSD
Type: BUY
Confidence: 75%
Agent: Neural Brain
Entry: 1.0845
Target: 1.0875
Stop Loss: 1.0820
```

### Signal Quality Indicators
- **Multi-Agent Consensus**: Multiple agents agree
- **High Confidence**: Above 70% confidence
- **Good Risk/Reward**: Minimum 1:2 ratio
- **Trend Alignment**: Signal matches overall trend

---

## 🔧 System Operations

### Starting the System

#### Method 1: Batch Script
```batch
# Double-click or run in command prompt
scripts\start_system.bat
```

#### Method 2: Manual Start
```bash
# Verify setup first
python scripts\verify_setup.py

# Start main application  
python main.py
```

### Monitoring System Health

#### Dashboard Indicators
- **Connection Status**: Green = Connected, Red = Disconnected
- **Agent Status**: Shows which agents are online/offline
- **System Alerts**: Warnings and important notifications

#### Performance Metrics
- **CPU Usage**: Should stay below 50%
- **Memory Usage**: Typically under 2GB
- **Signal Frequency**: Varies by market conditions

### Troubleshooting Common Issues

#### MT5 Connection Problems
1. Check MT5 terminal is running
2. Verify credentials in `.env` file
3. Ensure broker allows API access
4. Run connection test: `python scripts\test_connection.py`

#### Agent Offline Issues
1. Check system resources (CPU/memory)
2. Review error logs in `/logs/` directory
3. Restart specific agent if needed
4. Verify configuration files

#### Performance Issues
1. Close unnecessary applications
2. Check available RAM
3. Verify internet connection stability
4. Review log files for errors

---

## 📈 Best Practices

### Risk Management
- **Start Small**: Use minimum position sizes initially
- **Demo First**: Test thoroughly on demo account
- **Set Limits**: Never risk more than you can afford to lose
- **Monitor Closely**: Watch system performance regularly

### System Monitoring  
- **Daily Checks**: Review system health daily
- **Log Analysis**: Check logs for warnings/errors
- **Performance Review**: Analyze signal accuracy weekly
- **Backup Settings**: Keep configuration backups

### Trading Guidelines
- **Market Hours**: Best performance during active market hours
- **News Events**: Monitor high-impact economic events
- **Volatility**: Adjust risk during high volatility periods
- **Correlation**: Avoid highly correlated pairs

---

## 🧪 Testing and Validation

### Backtesting (Phase 2)
- **Purpose**: Test strategies on historical data
- **Duration**: Minimum 3 months of data
- **Metrics**: Win rate, profit factor, drawdown
- **Validation**: Forward testing after backtesting

### Paper Trading
- **Recommendation**: Start with demo account
- **Duration**: At least 1 month live testing
- **Criteria**: Consistent profitability before live trading
- **Monitoring**: Document all trades and results

---

## 📊 Performance Analysis

### Key Metrics to Track
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit ÷ Gross loss
- **Maximum Drawdown**: Largest losing streak
- **Sharpe Ratio**: Risk-adjusted returns

### Weekly Review Process
1. **Signal Analysis**: Review signal quality and frequency
2. **Performance Metrics**: Check key performance indicators
3. **System Health**: Verify all agents functioning properly
4. **Risk Assessment**: Ensure risk parameters followed

---

## 🔄 Current Status: Phase 1

### What's Working Now
✅ Complete project structure  
✅ All 12 agent templates created  
✅ Configuration system ready  
✅ Web dashboard templates  
✅ Setup and verification scripts  

### Phase 2 Development (Coming Next)
🚧 Actual agent functionality  
🚧 Real-time signal generation  
🚧 MT5 integration  
🚧 Neural network implementation  
🚧 Live trading capabilities  

### Getting Ready for Phase 2
- Familiarize yourself with the dashboard layout
- Review and customize configuration files
- Test MT5 connection setup
- Understand risk management settings
- Prepare demo account for testing

---

## 📞 Support and Resources

### Log Files Location
- **System Logs**: `/logs/system.log`
- **Trading Logs**: `/logs/trading.log` 
- **Error Logs**: `/logs/errors.log`

### Configuration Files
- **Main Config**: `/config/` directory
- **Environment**: `.env` file
- **Agent Settings**: Individual YAML files

### Getting Help
- Check verification script output for setup issues
- Review logs for error messages
- Ensure all prerequisites are met
- Verify MT5 terminal accessibility

**Ready to begin your AGI trading journey!** 🚀