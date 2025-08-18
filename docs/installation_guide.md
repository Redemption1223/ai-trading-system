# AGI Trading System - Installation Guide

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (Recommended: Python 3.10)
- **Windows OS** (Required for MetaTrader 5)
- **MetaTrader 5** terminal installed
- **Git** for version control

### Installation Steps

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd agi_trading_system
```

#### 2. Run Setup Script
```batch
# Windows
scripts\setup_environment.bat
```

#### 3. Install Dependencies
```batch
# Alternative installation
scripts\install_dependencies.bat
```

#### 4. Configure Environment
1. Copy `.env.template` to `.env`
2. Fill in your MT5 credentials:
   ```
   MT5_LOGIN=your_login
   MT5_PASSWORD=your_password
   MT5_SERVER=your_server
   ```

#### 5. Verify Installation
```bash
python scripts\verify_setup.py
```

## 🔧 Manual Installation

### Python Environment
```bash
# Create virtual environment
python -m venv agi_env
agi_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Required Packages
- **Core**: numpy, pandas, scipy
- **ML**: tensorflow, scikit-learn, torch
- **Trading**: MetaTrader5, TA-Lib  
- **Web**: flask, plotly, dash
- **Data**: requests, beautifulsoup4, yfinance

### TA-Lib Installation (Windows)
```bash
# Download TA-Lib wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

pip install TA_Lib-0.6.0-cp310-cp310-win_amd64.whl
```

## 🧪 Testing Installation

### Connection Test
```bash
python scripts\test_connection.py
```

### Agent Template Test
```bash
python -c "from core.mt5_windows_connector import MT5WindowsConnector; print('✅ Import successful')"
```

### Start System
```batch
scripts\start_system.bat
```

## 🛠️ Troubleshooting

### Common Issues

#### MetaTrader5 Package Error
```bash
pip install --upgrade MetaTrader5
```

#### TA-Lib Installation Failed
- Download appropriate wheel file
- Use pip install with local file path

#### Import Errors
- Check Python version compatibility
- Verify virtual environment activation
- Run `pip list` to confirm package installation

### System Requirements
- **RAM**: Minimum 8GB, Recommended 16GB
- **Storage**: 5GB free space
- **Network**: Stable internet connection
- **MT5**: Valid broker account (demo/live)

## 📁 Directory Structure After Installation
```
agi_trading_system/
├── 🏗️ Phase 1: COMPLETE (Templates Ready)
├── core/          # 4 core agents
├── ml/            # 1 neural agent  
├── data/          # 4 data agents
├── ui/            # 1 dashboard agent
├── utils/         # 1 monitor agent
├── validation/    # 1 validator agent
├── config/        # 7 YAML configs
├── static/        # Web assets
├── templates/     # HTML templates
└── scripts/       # Setup & verification
```

## ✅ Installation Success Criteria
- [ ] All dependencies installed
- [ ] All 12 agent templates created
- [ ] Configuration files ready
- [ ] Verification script passes 100%
- [ ] Ready for Phase 2 development

## 🚀 Next Steps
Once installation is complete:
1. Review configuration files in `/config/`
2. Test MT5 connection
3. Start Phase 2 implementation
4. Begin AGI agent development

## 📞 Support
For installation issues:
- Check verification script output
- Review logs in `/logs/` directory  
- Ensure all prerequisites are met
- Verify MT5 terminal is running