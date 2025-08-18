"""
AGENT_01: MT5 Windows Connector
Status: FULLY IMPLEMENTED
Purpose: MetaTrader 5 connection and Windows integration with auto-reconnection
"""

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

import subprocess
import time
import threading
import os
import logging
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

class MT5WindowsConnector:
    """Windows-native MT5 terminal interface with auto-connection"""
    
    def __init__(self):
        self.name = "MT5_WINDOWS_CONNECTOR"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        self.connection_status = False
        self.account_info = None
        self.auto_reconnect = True
        self.connection_monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Setup logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
    def initialize(self):
        """Initialize MT5 connection with retry logic"""
        self.logger.info(f"Initializing {self.name} v{self.version}")
        
        # Check for required dependencies
        if not MT5_AVAILABLE:
            self.logger.error("MetaTrader5 package not available")
            self.status = "FAILED"
            return {
                "status": "failed", 
                "agent": "AGENT_01", 
                "error": "MetaTrader5 package not installed"
            }
        
        max_retries = 5
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                # Check if MT5 is running
                if not self.is_mt5_running():
                    self.logger.info("MT5 not running, attempting to start...")
                    if self.start_mt5_terminal():
                        time.sleep(15)  # Wait for MT5 to fully load
                    else:
                        self.logger.warning("Could not start MT5 terminal")
                
                # Initialize connection
                if mt5.initialize():
                    self.account_info = mt5.account_info()
                    if self.account_info:
                        self.connection_status = True
                        self.status = "CONNECTED"
                        
                        self.logger.info(f"Connected to MT5 Account: {self.account_info.login}")
                        self.logger.info(f"Broker: {self.account_info.company}")
                        self.logger.info(f"Balance: ${self.account_info.balance}")
                        
                        # Start connection monitoring
                        self.start_connection_monitor()
                        
                        return {
                            "status": "connected", 
                            "agent": "AGENT_01",
                            "account": self.account_info.login,
                            "balance": self.account_info.balance,
                            "broker": self.account_info.company
                        }
                    else:
                        self.logger.warning(f"MT5 initialized but no account info on attempt {attempt + 1}")
                else:
                    self.logger.warning(f"MT5 initialization failed on attempt {attempt + 1}")
                    
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                        
            except Exception as e:
                self.logger.error(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                
        self.logger.error("Failed to connect after all attempts")
        self.status = "FAILED"
        return {"status": "failed", "agent": "AGENT_01", "error": "Connection failed"}
    
    def is_mt5_running(self):
        """Check if MT5 terminal is running"""
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available - cannot check MT5 process")
            return False
            
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and 'terminal64.exe' in proc.info['name'].lower():
                    return True
        except Exception as e:
            self.logger.error(f"Error checking MT5 process: {e}")
        return False
    
    def start_mt5_terminal(self):
        """Start MT5 terminal if not running"""
        mt5_paths = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            r"C:\Users\{}\AppData\Roaming\MetaQuotes\Terminal\*.exe".format(os.getenv('USERNAME', ''))
        ]
        
        for path in mt5_paths:
            if '*' in path:
                # Handle wildcard paths
                import glob
                matches = glob.glob(path)
                for match in matches:
                    if os.path.exists(match):
                        try:
                            subprocess.Popen([match])
                            self.logger.info(f"Starting MT5 from: {match}")
                            return True
                        except Exception as e:
                            self.logger.error(f"Failed to start MT5 from {match}: {e}")
            else:
                if os.path.exists(path):
                    try:
                        subprocess.Popen([path])
                        self.logger.info(f"Starting MT5 from: {path}")
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to start MT5 from {path}: {e}")
                
        self.logger.error("MT5 terminal not found in standard locations")
        return False
    
    def get_live_tick(self, symbol):
        """Get real-time tick data"""
        if not self.connection_status:
            self.logger.warning("Not connected to MT5")
            return None
            
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    'symbol': symbol,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'time': tick.time,
                    'volume': tick.volume,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                self.logger.warning(f"No tick data for {symbol}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting tick for {symbol}: {e}")
            return None
    
    def get_price_history(self, symbol, timeframe, count=100):
        """Get historical price data"""
        if not self.connection_status:
            self.logger.warning("Not connected to MT5")
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is not None and len(rates) > 0:
                if PANDAS_AVAILABLE:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    return df
                else:
                    # Return raw numpy array if pandas not available
                    return rates
            else:
                self.logger.warning(f"No historical data for {symbol}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting history for {symbol}: {e}")
            return None
    
    def start_connection_monitor(self):
        """Start background connection monitoring"""
        def monitor():
            while self.auto_reconnect:
                try:
                    time.sleep(60)  # Check every minute
                    if not self.is_connected():
                        self.logger.warning("Connection lost, attempting reconnect...")
                        self.connection_status = False
                        self.status = "RECONNECTING"
                        
                        # Attempt reconnection
                        if mt5.initialize():
                            account_info = mt5.account_info()
                            if account_info:
                                self.account_info = account_info
                                self.connection_status = True
                                self.status = "CONNECTED"
                                self.logger.info("Reconnection successful")
                            else:
                                self.logger.error("Reconnection failed - no account info")
                        else:
                            self.logger.error("Reconnection failed - MT5 initialize failed")
                except Exception as e:
                    self.logger.error(f"Connection monitor error: {e}")
                    
        self.connection_monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.connection_monitor_thread.start()
        self.logger.info("Connection monitor started")
    
    def is_connected(self):
        """Check if connection is active"""
        try:
            account_info = mt5.account_info()
            return account_info is not None
        except:
            return False
    
    def get_account_balance(self):
        """Get current account balance"""
        try:
            if self.connection_status:
                current_account = mt5.account_info()
                if current_account:
                    return current_account.balance
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 0.0
    
    def get_account_info(self):
        """Get complete account information"""
        try:
            if self.connection_status:
                account = mt5.account_info()
                if account:
                    return {
                        'login': account.login,
                        'balance': account.balance,
                        'equity': account.equity,
                        'margin': account.margin,
                        'free_margin': account.margin_free,
                        'margin_level': account.margin_level,
                        'company': account.company,
                        'server': account.server,
                        'currency': account.currency,
                        'leverage': account.leverage
                    }
            return None
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol):
        """Get symbol information"""
        try:
            if self.connection_status:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    return {
                        'symbol': symbol,
                        'bid': symbol_info.bid,
                        'ask': symbol_info.ask,
                        'spread': symbol_info.spread,
                        'digits': symbol_info.digits,
                        'point': symbol_info.point,
                        'trade_mode': symbol_info.trade_mode,
                        'min_lot': symbol_info.volume_min,
                        'max_lot': symbol_info.volume_max,
                        'lot_step': symbol_info.volume_step
                    }
            return None
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def check_connection(self):
        """Check MT5 connection health"""
        try:
            if not self.connection_status:
                return {"status": "disconnected", "message": "Not connected to MT5"}
            
            # Test connection by getting account info
            account = mt5.account_info()
            if account:
                return {
                    "status": "connected",
                    "message": "Connection healthy",
                    "account": account.login,
                    "server": account.server
                }
            else:
                return {"status": "error", "message": "Connected but no account info"}
                
        except Exception as e:
            return {"status": "error", "message": f"Connection check failed: {e}"}
    
    def get_status(self):
        """Get current agent status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'connected': self.connection_status,
            'account_balance': self.get_account_balance() if self.connection_status else 0,
            'mt5_running': self.is_mt5_running(),
            'monitor_active': self.connection_monitor_thread.is_alive() if self.connection_monitor_thread else False
        }
    
    def shutdown(self):
        """Clean shutdown"""
        self.logger.info("Shutting down MT5 connector...")
        self.auto_reconnect = False
        
        if self.connection_monitor_thread and self.connection_monitor_thread.is_alive():
            self.connection_monitor_thread.join(timeout=5)
        
        if self.connection_status:
            mt5.shutdown()
            self.connection_status = False
            
        self.status = "SHUTDOWN"
        self.logger.info("MT5 connector shutdown complete")

# Agent test
if __name__ == "__main__":
    # Test the MT5 connector
    connector = MT5WindowsConnector()
    result = connector.initialize()
    print(f"Initialization result: {result}")
    
    if result['status'] == 'connected':
        print("Testing functionality...")
        
        # Test tick data
        tick = connector.get_live_tick("EURUSD")
        print(f"EURUSD tick: {tick}")
        
        # Test historical data
        history = connector.get_price_history("EURUSD", mt5.TIMEFRAME_H1, 10)
        print(f"Historical data shape: {history.shape if history is not None else 'None'}")
        
        # Test account info
        account = connector.get_account_info()
        print(f"Account info: {account}")
        
        # Test status
        status = connector.get_status()
        print(f"Status: {status}")
        
    print("MT5 Connector test completed")