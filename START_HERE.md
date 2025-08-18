# AGI Trading System v2.0

Advanced AI-Powered Forex Trading Platform with 12 Specialized Agents

## 🚀 Quick Start Guide

### Prerequisites
1. **MetaTrader 5** installed and running
2. **Python 3.7+** installed
3. MT5 configured to **allow DLL imports** (Tools > Options > Expert Advisors > Allow DLL imports)

### Installation

1. **Install Required Packages:**
   ```bash
   # Run the installer script
   install_requirements.bat
   
   # Or install manually:
   pip install MetaTrader5 psutil pandas numpy
   ```

2. **Configure MT5:**
   - Open MetaTrader 5
   - Go to Tools > Options > Expert Advisors
   - Check "Allow DLL imports"
   - Check "Allow WebRequest for listed URLs" (optional)
   - Login to your trading account (demo or live)

### Starting the System

```bash
# Navigate to the system directory
cd "path\to\agi_trading_system"

# Start the trading system
python start_trading_system.py
```

## 📊 System Components

### Core Agents
- **MT5 Connector**: Direct integration with MetaTrader 5
- **Signal Coordinator**: Multi-threaded signal processing
- **Risk Calculator**: Advanced position sizing with 4 risk models
- **Chart Signal Agent**: Real-time technical analysis
- **Neural Signal Brain**: AI-powered signal generation
- **Technical Analyst**: 31 technical indicators
- **Market Data Manager**: Real-time data streaming
- **Trade Execution Engine**: Advanced order management
- **Portfolio Manager**: Automated portfolio rebalancing
- **Performance Analytics**: Comprehensive performance tracking
- **Alert System**: Multi-channel notifications
- **Configuration Manager**: Hot-reloadable configuration

## ⚙️ Configuration

The system starts with default settings:
- **Symbols**: EURUSD, GBPUSD, USDJPY, AUDUSD
- **Risk per Trade**: 2%
- **Execution Mode**: DEMO (change to LIVE for real trading)
- **Auto Trading**: Enabled

## 🎮 System Commands

When running, you can use these commands:
- `status` - Show system status
- `stop` - Stop trading operations
- `quit` - Shutdown entire system
- `help` - Show available commands

## 🔧 Troubleshooting

### Common Issues

1. **"MetaTrader5 package not available"**
   - Run: `pip install MetaTrader5`
   - Ensure MT5 is running

2. **"MT5 Connection failed"**
   - Check MT5 is running and logged in
   - Enable "Allow DLL imports" in MT5 settings
   - Restart MT5 and try again

3. **"Access denied" errors**
   - Run command prompt as Administrator
   - Check Windows firewall settings

### Testing the System

To run comprehensive tests:
```bash
# Test all agents
python test_comprehensive_suite.py

# Test integration
python test_integration_workflow.py
```

## 📈 Trading Modes

### Demo Mode (Default)
- Safe testing environment
- No real money at risk
- Full functionality

### Live Mode
- Real money trading
- Change `execution_mode` to `ExecutionMode.LIVE` in `start_trading_system.py`
- **Use with caution!**

## 🔐 Safety Features

- Comprehensive risk management
- Position size limits
- Stop loss automation
- Portfolio rebalancing
- Real-time monitoring
- Alert system

## 📞 Support

The system includes comprehensive logging and error handling. Check the console output for detailed information about system status and any issues.

## ⚠️ Disclaimer

This trading system is for educational and research purposes. Trading forex involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results.