@echo off
echo 🚀 AGI Trading System - System Startup
echo =====================================

echo 🔍 Checking system status...
python scripts\verify_setup.py

echo.
echo 🚀 Starting AGI Trading System...
python main.py

pause