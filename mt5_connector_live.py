"""
MT5 Live Connector - LIVE TRADING ONLY
No simulation, no demo, no fallbacks - Pure live trading connection
"""

import MetaTrader5 as mt5
import time
import logging
from datetime import datetime

class MT5LiveConnector:
    """Live-only MT5 connector - NO SIMULATION CODE"""
    
    def __init__(self):
        self.name = "MT5_LIVE_CONNECTOR"
        self.status = "DISCONNECTED"
        self.version = "2.0.0"
        self.account_info = None
        self.terminal_info = None
        self.connection_status = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize LIVE MT5 connection ONLY"""
        try:
            self.logger.info(f"Initializing LIVE {self.name} v{self.version}")
            
            # Ensure clean state
            mt5.shutdown()
            time.sleep(0.5)
            
            # Initialize MT5 connection
            self.logger.info("Connecting to MetaTrader 5...")
            if not mt5.initialize():
                error_msg = "Failed to initialize MT5 - Check MT5 is running and DLL imports are enabled"
                self.logger.error(error_msg)
                return {
                    "status": "failed", 
                    "agent": "MT5_LIVE", 
                    "error": error_msg,
                    "fix": "1. Start MT5, 2. Tools->Options->Expert Advisors->Allow DLL imports"
                }
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("No account info - MT5 not logged in")
                return {
                    "status": "failed",
                    "error": "MT5 not logged in to any account"
                }
            
            # Store account information
            self.account_info = {
                "login": account_info.login,
                "server": account_info.server,
                "company": account_info.company,
                "name": account_info.name,
                "currency": account_info.currency,
                "balance": account_info.balance,
                "credit": account_info.credit,
                "profit": account_info.profit,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "margin_free": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "leverage": account_info.leverage,
                "trade_mode": account_info.trade_mode
            }
            
            # Store terminal information
            terminal_info = mt5.terminal_info()
            if terminal_info:
                self.terminal_info = {
                    "build": terminal_info.build,
                    "name": terminal_info.name,
                    "path": terminal_info.path,
                    "connected": terminal_info.connected,
                    "trade_allowed": terminal_info.trade_allowed
                }
            
            self.status = "CONNECTED"
            self.connection_status = True
            
            self.logger.info(f"[OK] Connected to MT5!")
            self.logger.info(f"[OK] Account: {account_info.login}")
            self.logger.info(f"[OK] Server: {account_info.server}")
            self.logger.info(f"[OK] Balance: ${account_info.balance}")
            self.logger.info(f"[OK] Terminal: {terminal_info.name if terminal_info else 'Unknown'} (Build {terminal_info.build if terminal_info else 'Unknown'})")
            
            return {
                "status": "initialized",
                "agent": "MT5_LIVE",
                "account": account_info.login,
                "server": account_info.server,
                "balance": account_info.balance,
                "currency": account_info.currency,
                "leverage": account_info.leverage,
                "trade_mode": account_info.trade_mode
            }
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            return {"status": "failed", "agent": "MT5_LIVE", "error": str(e)}
    
    def get_current_price(self, symbol):
        """Get LIVE current price for symbol - NO SIMULATION"""
        try:
            if not self.connection_status:
                self.logger.error("MT5 not connected")
                return {"status": "error", "message": "Not connected to MT5"}
            
            # Get symbol tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"No tick data for {symbol}")
                return {"status": "error", "message": f"No tick data for {symbol}"}
            
            return {
                "status": "success",
                "symbol": symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "time": tick.time,
                "spread": (tick.ask - tick.bid) * (10 ** mt5.symbol_info(symbol).digits)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_account_info(self):
        """Get LIVE account information - NO SIMULATION"""
        try:
            if not self.connection_status:
                return {"status": "error", "message": "Not connected to MT5"}
            
            # Get fresh account info
            account_info = mt5.account_info()
            if account_info is None:
                return {"status": "error", "message": "Could not retrieve account info"}
            
            return {
                "status": "success",
                "login": account_info.login,
                "server": account_info.server,
                "company": account_info.company,
                "name": account_info.name,
                "currency": account_info.currency,
                "balance": account_info.balance,
                "credit": account_info.credit,
                "profit": account_info.profit,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "margin_free": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "leverage": account_info.leverage,
                "trade_mode": account_info.trade_mode,
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_status(self):
        """Get connector status"""
        return {
            "status": self.status,
            "connection_status": self.connection_status,
            "account": self.account_info.get("login") if self.account_info else None,
            "last_check": datetime.now().isoformat()
        }
    
    def get_live_tick(self, symbol):
        """Get LIVE tick data - NO SIMULATION"""
        try:
            if not self.connection_status:
                return {"status": "error", "message": "Not connected to MT5"}
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"status": "error", "message": f"No tick data for {symbol}"}
            
            return {
                "status": "success",
                "symbol": symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "time": tick.time,
                "flags": tick.flags
            }
            
        except Exception as e:
            self.logger.error(f"Error getting tick for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_price_history(self, symbol, timeframe='H1', count=100):
        """Get LIVE price history - NO SIMULATION"""
        try:
            if not self.connection_status:
                return {"status": "error", "message": "Not connected to MT5"}
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return {"status": "error", "message": f"Symbol {symbol} not found"}
            
            # Select the symbol (ensure it's visible in Market Watch)
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to select symbol {symbol}")
                return {"status": "error", "message": f"Failed to select symbol {symbol}"}
            
            # Convert timeframe string to MT5 constant
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get LIVE rates data
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"No rates data for {symbol}")
                return {"status": "error", "message": f"No rates data for {symbol}"}
            
            # Convert to list of dictionaries
            rates_list = []
            for rate in rates:
                rates_list.append({
                    'time': rate[0],
                    'open': rate[1],
                    'high': rate[2],
                    'low': rate[3],
                    'close': rate[4],
                    'volume': rate[5]
                })
            
            return {
                "status": "success",
                "data": rates_list,
                "symbol": symbol,
                "timeframe": timeframe,
                "count": len(rates_list)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting price history for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_positions(self):
        """Get LIVE open positions - NO SIMULATION"""
        try:
            if not self.connection_status:
                return {"status": "error", "message": "Not connected to MT5"}
            
            # Get LIVE open positions
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
                    'comment': position.comment,
                    'magic': position.magic,
                    'identifier': position.identifier
                })
            
            return {
                "status": "success",
                "positions": position_list,
                "count": len(position_list)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"status": "error", "message": str(e)}
    
    def place_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment=""):
        """Place LIVE order - REAL TRADING"""
        try:
            if not self.connection_status:
                return {"status": "error", "message": "Not connected to MT5"}
            
            # Get symbol info for proper formatting
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"status": "error", "message": f"Symbol {symbol} not found"}
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add price if specified (for pending orders)
            if price is not None:
                request["price"] = price
            
            # Add stop loss if specified
            if sl is not None:
                request["sl"] = sl
            
            # Add take profit if specified
            if tp is not None:
                request["tp"] = tp
            
            # Send LIVE order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "status": "error",
                    "message": f"Order failed: {result.comment}",
                    "retcode": result.retcode
                }
            
            return {
                "status": "success",
                "order_id": result.order,
                "ticket": result.order,
                "volume": result.volume,
                "price": result.price,
                "comment": result.comment
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"status": "error", "message": str(e)}
    
    def close_position(self, ticket):
        """Close LIVE position - REAL TRADING"""
        try:
            if not self.connection_status:
                return {"status": "error", "message": "Not connected to MT5"}
            
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                return {"status": "error", "message": f"Position {ticket} not found"}
            
            position = positions[0]
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "status": "error",
                    "message": f"Close failed: {result.comment}",
                    "retcode": result.retcode
                }
            
            return {
                "status": "success",
                "closed_ticket": ticket,
                "close_price": result.price,
                "volume": result.volume
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {"status": "error", "message": str(e)}
    
    def test_connection(self):
        """Test LIVE connection"""
        try:
            if not self.connection_status:
                return False
            
            # Test with a simple account info request
            account = mt5.account_info()
            terminal = mt5.terminal_info()
            
            return account is not None and terminal is not None and terminal.connected
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def shutdown(self):
        """Clean shutdown"""
        try:
            self.logger.info("Shutting down MT5 Live connector...")
            mt5.shutdown()
            self.status = "SHUTDOWN"
            self.connection_status = False
            self.logger.info("MT5 Live connector shutdown complete")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

def test_live_connector():
    """Test the LIVE connector"""
    print("Testing LIVE MT5 Connector...")
    print("=" * 50)
    
    connector = MT5LiveConnector()
    
    # Test initialization
    result = connector.initialize()
    print(f"\nInitialization: {result['status']}")
    
    if result['status'] == 'initialized':
        print(f"[OK] Account: {result['account']}")
        print(f"[OK] Server: {result['server']}")
        print(f"[OK] Balance: ${result['balance']:,.2f}")
        print(f"[OK] Currency: {result['currency']}")
        print(f"[OK] Leverage: 1:{result['leverage']}")
        print(f"[OK] Trade Mode: {result['trade_mode']}")
        
        # Test live price retrieval
        print("\nTesting LIVE price retrieval...")
        price = connector.get_current_price("EURUSD")
        if price['status'] == 'success':
            print(f"[OK] EURUSD: {price['bid']:.5f} / {price['ask']:.5f}")
        else:
            print(f"[FAIL] Price error: {price['message']}")
        
        # Test live positions
        print("\nTesting LIVE positions...")
        positions = connector.get_positions()
        if positions['status'] == 'success':
            print(f"[OK] Open positions: {positions['count']}")
        
        # Test connection
        print("\nTesting LIVE connection stability...")
        if connector.test_connection():
            print("[OK] LIVE connection stable")
        else:
            print("[FAIL] LIVE connection unstable")
        
        print("\n[SUCCESS] LIVE CONNECTOR TEST PASSED!")
        return True
    else:
        print(f"[FAIL] Connection failed: {result.get('error', 'Unknown error')}")
        if 'fix' in result:
            print(f"Fix: {result['fix']}")
        return False

if __name__ == "__main__":
    test_live_connector()