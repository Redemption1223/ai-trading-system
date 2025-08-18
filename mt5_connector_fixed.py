"""
Fixed MT5 Connector - Simple and Reliable
Addresses intermittent connection issues
"""

import MetaTrader5 as mt5

import time
import logging
from datetime import datetime

class MT5ConnectorFixed:
    """Simplified, reliable MT5 connector"""
    
    def __init__(self):
        self.name = "MT5_CONNECTOR_FIXED"
        self.status = "DISCONNECTED"
        self.version = "1.1.0"
        self.account_info = None
        self.terminal_info = None
        self.connection_status = False  # For compatibility with Chart Signal Agent
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize MT5 connection - simplified approach"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # MT5 is required - no fallbacks
            
            # Single initialization attempt with proper cleanup
            try:
                # Ensure clean state
                mt5.shutdown()
                time.sleep(0.5)
                
                # Initialize
                self.logger.info("Connecting to MetaTrader 5...")
                if not mt5.initialize():
                    error = mt5.last_error()
                    self.logger.error(f"MT5 initialization failed: {error}")
                    return {
                        "status": "failed", 
                        "agent": "MT5_FIXED", 
                        "error": f"MT5 init failed: {error}",
                        "fix": "Check MT5 is running, logged in, and 'Allow DLL imports' is enabled"
                    }
                
                # Verify connection by getting account info
                account_info = mt5.account_info()
                if not account_info:
                    self.logger.error("Connected but no account info - not logged in?")
                    mt5.shutdown()
                    return {
                        "status": "failed", 
                        "agent": "MT5_FIXED", 
                        "error": "No account logged in",
                        "fix": "Login to your MT5 account first"
                    }
                
                # Get terminal info
                terminal_info = mt5.terminal_info()
                if not terminal_info:
                    self.logger.warning("Could not get terminal info")
                
                # Store connection info
                self.account_info = {
                    'login': account_info.login,
                    'server': account_info.server,
                    'currency': account_info.currency,
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'leverage': account_info.leverage,
                    'company': account_info.company
                }
                
                if terminal_info:
                    self.terminal_info = {
                        'name': terminal_info.name,
                        'company': terminal_info.company,
                        'path': terminal_info.path,
                        'build': terminal_info.build
                    }
                
                # Test symbol access
                test_tick = mt5.symbol_info_tick("EURUSD")
                if not test_tick:
                    self.logger.warning("Cannot access EURUSD - may have limited symbol access")
                
                self.status = "CONNECTED"
                self.connection_status = True  # Update compatibility flag
                
                self.logger.info(f"✓ Connected to MT5!")
                self.logger.info(f"✓ Account: {account_info.login}")
                self.logger.info(f"✓ Server: {account_info.server}")
                self.logger.info(f"✓ Balance: ${account_info.balance:,.2f}")
                if terminal_info:
                    self.logger.info(f"✓ Terminal: {terminal_info.name} (Build {terminal_info.build})")
                
                return {
                    "status": "initialized",
                    "agent": "MT5_FIXED",
                    "account": account_info.login,
                    "server": account_info.server,
                    "balance": account_info.balance,
                    "currency": account_info.currency,
                    "terminal": terminal_info.name if terminal_info else "Unknown"
                }
                
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
                try:
                    mt5.shutdown()
                except:
                    pass
                return {"status": "failed", "agent": "MT5_FIXED", "error": str(e)}
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "MT5_FIXED", "error": str(e)}
    
    def get_current_price(self, symbol):
        """Get current price for symbol"""
        try:
            if self.status != "CONNECTED":
                return {"status": "error", "message": "Not connected to MT5"}
            
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    "status": "success",
                    "symbol": symbol,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "time": datetime.fromtimestamp(tick.time)
                }
            else:
                return {"status": "error", "message": f"Cannot get price for {symbol}"}
        
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_account_info(self):
        """Get current account information"""
        try:
            if self.status != "CONNECTED":
                return {"status": "error", "message": "Not connected"}
            
            account_info = mt5.account_info()
            if account_info:
                return {
                    "status": "success",
                    "login": account_info.login,
                    "server": account_info.server,
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "margin_free": account_info.margin_free,
                    "currency": account_info.currency,
                    "leverage": account_info.leverage
                }
            else:
                return {"status": "error", "message": "Cannot get account info"}
        
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {"status": "error", "message": str(e)}
    
    def test_connection(self):
        """Test if connection is still working"""
        try:
            if self.status != "CONNECTED":
                return False
            
            # Test by getting account info
            account_info = mt5.account_info()
            if account_info:
                # Update stored info
                self.account_info['balance'] = account_info.balance
                self.account_info['equity'] = account_info.equity
                return True
            else:
                self.logger.warning("Connection test failed - no account info")
                self.status = "DISCONNECTED"
                self.connection_status = False  # Update compatibility flag
                return False
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            self.status = "DISCONNECTED"
            self.connection_status = False  # Update compatibility flag
            return False
    
    def get_status(self):
        """Get connector status"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "account_info": self.account_info,
            "terminal_info": self.terminal_info,
            "connection_active": self.test_connection() if self.status == "CONNECTED" else False
        }
    
    def get_live_tick(self, symbol):
        """Get live tick data - compatibility method"""
        return self.get_current_price(symbol)
    
    def get_price_history(self, symbol, timeframe='H1', count=100):
        """Get price history - basic implementation"""
        try:
            if self.status != "CONNECTED":
                return {"status": "error", "message": "Not connected"}
            
            # Convert timeframe string to MT5 constant
            timeframe_map = {
                'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                'H1': 16385, 'H4': 16388, 'D1': 16408
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, 16385)  # Default to H1
            
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is not None and len(rates) > 0:
                return {
                    "status": "success",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "count": len(rates),
                    "data": rates
                }
            else:
                return {"status": "error", "message": f"No price history for {symbol}"}
        
        except Exception as e:
            self.logger.error(f"Error getting price history for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_positions(self):
        """Get all open positions"""
        try:
            if not self.connection_status or not mt5.connected():
                self.logger.error("MT5 not connected - cannot get positions")
                return {"status": "error", "message": "Not connected to MT5"}
            
            # Get all open positions
            positions = mt5.positions_get()
            
            if positions is None:
                return {"status": "success", "positions": []}
            
            # Convert to list of dictionaries
            position_list = []
            for position in positions:
                position_list.append({
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'type': position.type,
                    'volume': position.volume,
                    'price_open': position.price_open,
                    'price_current': position.price_current,
                    'profit': position.profit,
                    'swap': position.swap,
                    'commission': position.commission,
                    'time': position.time,
                    'comment': position.comment
                })
            
            return {
                "status": "success",
                "positions": position_list,
                "count": len(position_list)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"status": "error", "message": str(e)}

    def shutdown(self):
        """Clean shutdown"""
        try:
            self.logger.info("Shutting down MT5 connector...")
            mt5.shutdown()
            self.status = "SHUTDOWN"
            self.connection_status = False
            self.logger.info("MT5 connector shutdown complete")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

def test_fixed_connector():
    """Test the fixed connector"""
    print("Testing Fixed MT5 Connector...")
    print("=" * 50)
    
    connector = MT5ConnectorFixed()
    
    # Test initialization
    result = connector.initialize()
    print(f"\nInitialization: {result['status']}")
    
    if result['status'] == 'initialized':
        print(f"✓ Account: {result['account']}")
        print(f"✓ Server: {result['server']}")
        print(f"✓ Balance: ${result['balance']:,.2f}")
        
        # Test price retrieval
        print("\nTesting price retrieval...")
        price = connector.get_current_price("EURUSD")
        if price['status'] == 'success':
            print(f"✓ EURUSD: {price['bid']:.5f}")
        else:
            print(f"✗ Price error: {price['message']}")
        
        # Test connection
        print("\nTesting connection stability...")
        if connector.test_connection():
            print("✓ Connection stable")
        else:
            print("✗ Connection unstable")
        
        print("\n🎉 FIXED CONNECTOR TEST PASSED!")
        return True
    else:
        print(f"✗ Connection failed: {result.get('error', 'Unknown error')}")
        if 'fix' in result:
            print(f"💡 Fix: {result['fix']}")
        return False

if __name__ == "__main__":
    test_fixed_connector()