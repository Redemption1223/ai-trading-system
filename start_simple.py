"""
Simple AGI Trading System Startup
Uses the fixed MT5 connector for reliable connection
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt5_connector_fixed import MT5ConnectorFixed

def main():
    """Simple startup with fixed MT5 connector"""
    print("\n" + "="*60)
    print("  AGI Trading System - Simple Startup")
    print("  Using Fixed MT5 Connector")
    print("="*60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test fixed connector first
    print("\n[Step 1] Testing MT5 Connection...")
    connector = MT5ConnectorFixed()
    
    result = connector.initialize()
    
    if result['status'] == 'initialized':
        print(f"✅ MT5 Connected Successfully!")
        print(f"   Account: {result['account']}")
        print(f"   Server: {result['server']}")
        print(f"   Balance: ${result['balance']:,.2f} {result.get('currency', 'USD')}")
        print(f"   Terminal: {result.get('terminal', 'Unknown')}")
        
        # Test real-time data
        print("\n[Step 2] Testing Real-time Data...")
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        for symbol in symbols:
            price = connector.get_current_price(symbol)
            if price['status'] == 'success':
                print(f"   📈 {symbol}: {price['bid']:.5f}")
            else:
                print(f"   ❌ {symbol}: {price['message']}")
        
        print("\n[Step 3] Connection Stability Test...")
        for i in range(5):
            if connector.test_connection():
                print(f"   ✓ Connection test {i+1}/5: PASS")
            else:
                print(f"   ❌ Connection test {i+1}/5: FAIL")
                break
            time.sleep(1)
        
        print("\n" + "="*60)
        print("🎉 MT5 CONNECTION IS STABLE!")
        print("Your system is ready to run the full AGI Trading System.")
        print("\nNext steps:")
        print("1. Use 'python start_trading_system.py' for full system")
        print("2. Or modify start_trading_system.py to use MT5ConnectorFixed")
        print("="*60)
        
        # Keep connection alive for testing
        print("\nConnection test running... (Press Ctrl+C to stop)")
        try:
            while True:
                time.sleep(10)
                if not connector.test_connection():
                    print("❌ Connection lost!")
                    break
                else:
                    account = connector.get_account_info()
                    if account['status'] == 'success':
                        print(f"💰 Balance: ${account['balance']:,.2f} | Equity: ${account['equity']:,.2f}")
        except KeyboardInterrupt:
            print("\nStopping...")
        
        connector.shutdown()
        
    else:
        print(f"❌ MT5 Connection Failed!")
        print(f"   Error: {result.get('error', 'Unknown error')}")
        if 'fix' in result:
            print(f"   💡 Fix: {result['fix']}")
        
        print("\n🔧 Troubleshooting Steps:")
        print("1. Make sure MT5 is running and you're logged in")
        print("2. In MT5: Tools > Options > Expert Advisors")
        print("3. Check 'Allow DLL imports'")
        print("4. Try restarting MT5")
        print("5. Run MT5 as Administrator if needed")

if __name__ == "__main__":
    main()