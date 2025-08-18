@echo off
echo Installing AGI Trading System Requirements...
echo ==========================================

echo.
echo Installing MetaTrader5 package...
pip install MetaTrader5

echo.
echo Installing additional packages...
pip install psutil
pip install pandas
pip install numpy

echo.
echo Installation complete!
echo.
echo To start the trading system, run:
echo   python start_trading_system.py
echo.
pause