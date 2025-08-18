"""
Quick LIVE Status Check - LIVE TRADING ONLY
Shows your real trading information with NO SIMULATION
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt5_connector_live import MT5LiveConnector
from datetime import datetime

def quick_live_status():
    """Quick LIVE status check of your trading account - NO SIMULATION"""
    print("\n" + "="*60)
    print("LIVE TRADING STATUS - NO SIMULATION")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize LIVE MT5 connector
        mt5 = MT5LiveConnector()
        result = mt5.initialize()
        
        if result['status'] == 'initialized':
            print("\n[OK] LIVE MT5 Connection: SUCCESS")
            
            # Get LIVE account information
            account = mt5.get_account_info()
            if account['status'] == 'success':
                print(f"\nYOUR LIVE ACCOUNT DETAILS:")
                print(f"  Account Number: {account.get('login', 'Unknown')}")
                print(f"  Server: {account.get('server', 'Unknown')}")
                print(f"  Company: {account.get('company', 'Unknown')}")
                print(f"  Currency: {account.get('currency', 'USD')}")
                print(f"  Leverage: 1:{account.get('leverage', 100)}")
                print(f"  Trade Mode: {account.get('trade_mode', 'UNKNOWN')}")
                
                print(f"\nYOUR LIVE BALANCE:")
                print(f"  Balance: ${account.get('balance', 0):,.2f}")
                print(f"  Equity: ${account.get('equity', 0):,.2f}")
                print(f"  Margin: ${account.get('margin', 0):,.2f}")
                print(f"  Free Margin: ${account.get('margin_free', 0):,.2f}")
                print(f"  Margin Level: {account.get('margin_level', 0):.2f}%")
                
                # Calculate P&L
                balance = account.get('balance', 0)
                equity = account.get('equity', 0)
                if balance > 0:
                    pnl = equity - balance
                    pnl_percent = (pnl / balance) * 100
                    status_icon = "[UP]" if pnl >= 0 else "[DOWN]"
                    print(f"  Profit/Loss: {status_icon} ${pnl:,.2f} ({pnl_percent:+.2f}%)")
            
            # Get LIVE positions
            positions = mt5.get_positions()
            if positions['status'] == 'success' and positions['positions']:
                print(f"\nYOUR LIVE OPEN POSITIONS ({len(positions['positions'])}):")
                print(f"  {'Symbol':<8} {'Type':<6} {'Volume':<8} {'Price':<10} {'P&L':<12}")
                print("  " + "-" * 50)
                
                total_pnl = 0
                for pos in positions['positions'][:10]:  # Show first 10
                    symbol = pos.get('symbol', 'Unknown')
                    pos_type = 'BUY' if pos.get('type', 0) == 0 else 'SELL'
                    volume = pos.get('volume', 0)
                    price = pos.get('price_open', 0)
                    profit = pos.get('profit', 0)
                    total_pnl += profit
                    
                    print(f"  {symbol:<8} {pos_type:<6} {volume:<8.2f} {price:<10.5f} ${profit:<11.2f}")
                
                if len(positions['positions']) > 10:
                    print(f"  ... and {len(positions['positions']) - 10} more positions")
                
                print(f"\n  Total Position P&L: ${total_pnl:,.2f}")
            else:
                print(f"\nYOUR LIVE OPEN POSITIONS: None")
            
            # Get LIVE market prices
            print(f"\nLIVE MARKET PRICES:")
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
            print(f"  {'Symbol':<8} {'Bid':<10} {'Ask':<10} {'Spread':<8}")
            print("  " + "-" * 40)
            
            for symbol in symbols:
                price = mt5.get_current_price(symbol)
                if price['status'] == 'success':
                    bid = price.get('bid', 0)
                    ask = price.get('ask', 0)
                    spread = price.get('spread', 0)
                    print(f"  {symbol:<8} {bid:<10.5f} {ask:<10.5f} {spread:<8.1f}")
            
        else:
            print("\n[FAIL] LIVE MT5 Connection: FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
            print("\nRequired for LIVE trading:")
            print("1. MetaTrader 5 must be running")
            print("2. Enable 'Allow DLL imports' in MT5 settings")
            print("3. Must be logged into a LIVE trading account")
            print("4. Account must have sufficient funds")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\nLIVE Trading Requirements:")
        print("1. MetaTrader 5 installed and running")
        print("2. Valid LIVE trading account")
        print("3. Sufficient account balance")
        print("4. Proper risk management settings")
    
    print("\n" + "="*60)
    print("LIVE TRADING MODE - NO SIMULATION")
    print("For full LIVE system: python start_trading_system.py")
    print("="*60)

if __name__ == "__main__":
    quick_live_status()