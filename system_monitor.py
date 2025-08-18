"""
AGI Trading System Monitor
Real-time system status and health monitoring
"""

import sys
import os
import time
import threading
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt5_connector_fixed import MT5ConnectorFixed

class SystemMonitor:
    """Real-time system monitoring and diagnostics"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.mt5_connector = None
        self.start_time = None
        self.stats = {
            'connection_checks': 0,
            'connection_failures': 0,
            'price_updates': 0,
            'price_failures': 0,
            'uptime': 0
        }
    
    def start_monitoring(self):
        """Start system monitoring"""
        print("\n" + "="*60)
        print("  AGI TRADING SYSTEM MONITOR")
        print("="*60)
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Initialize MT5 connection
        print("\n[INIT] Initializing MT5 Connection...")
        self.mt5_connector = MT5ConnectorFixed()
        
        result = self.mt5_connector.initialize()
        
        if result['status'] != 'initialized':
            print(f"❌ MT5 Connection Failed: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ MT5 Connected Successfully!")
        print(f"   Account: {result['account']} on {result['server']}")
        print(f"   Balance: ${result['balance']:,.2f} {result.get('currency', 'USD')}")
        
        # Start monitoring loop
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        return True
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        last_price_update = time.time()
        last_connection_check = time.time()
        
        print("\n[MONITOR] Starting real-time monitoring...")
        print("Commands: 'status', 'prices', 'stats', 'quit'")
        print("-" * 60)
        
        while self.monitoring:
            current_time = time.time()
            
            # Update uptime
            self.stats['uptime'] = current_time - self.start_time
            
            # Connection health check every 30 seconds
            if current_time - last_connection_check >= 30:
                self.stats['connection_checks'] += 1
                
                if self.mt5_connector.test_connection():
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Connection healthy")
                else:
                    self.stats['connection_failures'] += 1
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Connection lost!")
                    
                    # Try to reconnect
                    print("  Attempting reconnection...")
                    reconnect_result = self.mt5_connector.initialize()
                    if reconnect_result['status'] == 'initialized':
                        print("  ✓ Reconnected successfully")
                    else:
                        print(f"  ❌ Reconnection failed: {reconnect_result.get('error', 'Unknown')}")
                
                last_connection_check = current_time
            
            # Price updates every 10 seconds
            if current_time - last_price_update >= 10:
                self.stats['price_updates'] += 1
                
                prices_ok = True
                current_prices = {}
                
                for symbol in symbols:
                    price = self.mt5_connector.get_current_price(symbol)
                    if price['status'] == 'success':
                        current_prices[symbol] = price['bid']
                    else:
                        prices_ok = False
                        self.stats['price_failures'] += 1
                
                if prices_ok:
                    price_str = " | ".join([f"{sym}: {price:.5f}" for sym, price in current_prices.items()])
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 {price_str}")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Price update failed")
                
                last_price_update = current_time
            
            time.sleep(1)
    
    def show_status(self):
        """Show current system status"""
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)
        
        if self.mt5_connector:
            status = self.mt5_connector.get_status()
            print(f"MT5 Status: {status['status']}")
            
            if status['account_info']:
                acc = status['account_info']
                print(f"Account: {acc['login']} ({acc['server']})")
                print(f"Balance: ${acc['balance']:,.2f}")
                print(f"Equity: ${acc['equity']:,.2f}")
                print(f"Free Margin: ${acc.get('margin_free', 0):,.2f}")
                print(f"Leverage: 1:{acc['leverage']}")
            
            if status['connection_active']:
                print("Connection: ✅ Active")
            else:
                print("Connection: ❌ Inactive")
        
        uptime_str = str(timedelta(seconds=int(self.stats['uptime'])))
        print(f"Uptime: {uptime_str}")
        print(f"Connection Checks: {self.stats['connection_checks']}")
        print(f"Connection Failures: {self.stats['connection_failures']}")
        print(f"Price Updates: {self.stats['price_updates']}")
        print(f"Price Failures: {self.stats['price_failures']}")
        
        if self.stats['connection_checks'] > 0:
            reliability = ((self.stats['connection_checks'] - self.stats['connection_failures']) / self.stats['connection_checks']) * 100
            print(f"Connection Reliability: {reliability:.1f}%")
        
        print("="*60)
    
    def show_prices(self):
        """Show current prices"""
        print("\nCURRENT PRICES")
        print("-" * 30)
        
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD']
        
        for symbol in symbols:
            price = self.mt5_connector.get_current_price(symbol)
            if price['status'] == 'success':
                print(f"{symbol:8} {price['bid']:.5f} / {price['ask']:.5f}")
            else:
                print(f"{symbol:8} ❌ Not available")
    
    def show_stats(self):
        """Show detailed statistics"""
        print("\nSYSTEM STATISTICS")
        print("-" * 40)
        
        uptime_str = str(timedelta(seconds=int(self.stats['uptime'])))
        print(f"System Uptime: {uptime_str}")
        print(f"Total Connection Checks: {self.stats['connection_checks']}")
        print(f"Failed Connections: {self.stats['connection_failures']}")
        print(f"Successful Price Updates: {self.stats['price_updates']}")
        print(f"Failed Price Updates: {self.stats['price_failures']}")
        
        if self.stats['connection_checks'] > 0:
            success_rate = ((self.stats['connection_checks'] - self.stats['connection_failures']) / self.stats['connection_checks']) * 100
            print(f"Connection Success Rate: {success_rate:.2f}%")
        
        if self.stats['price_updates'] > 0:
            price_success_rate = ((self.stats['price_updates'] - self.stats['price_failures']) / self.stats['price_updates']) * 100
            print(f"Price Update Success Rate: {price_success_rate:.2f}%")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        print("\n[MONITOR] Stopping system monitor...")
        
        if self.mt5_connector:
            self.mt5_connector.shutdown()
        
        print("Monitor stopped.")

def main():
    """Main monitoring interface"""
    monitor = SystemMonitor()
    
    if not monitor.start_monitoring():
        print("Failed to start monitoring")
        return
    
    try:
        while monitor.monitoring:
            try:
                cmd = input("\nMonitor> ").strip().lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                elif cmd == 'status':
                    monitor.show_status()
                elif cmd == 'prices':
                    monitor.show_prices()
                elif cmd == 'stats':
                    monitor.show_stats()
                elif cmd == 'help':
                    print("\nAvailable commands:")
                    print("  status - Show system status")
                    print("  prices - Show current prices")
                    print("  stats  - Show detailed statistics")
                    print("  quit   - Exit monitor")
                elif cmd == '':
                    continue
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nInterrupt received...")
                break
            except EOFError:
                print("\nEOF received...")
                break
    
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()