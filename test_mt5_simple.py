"""
Simple MT5 Connection Test
Quick test to verify MT5 connectivity
"""

def test_mt5_connection():
    """Test MT5 connection step by step"""
    print("Testing MT5 Connection...")
    print("=" * 40)
    
    # Step 1: Import
    try:
        import MetaTrader5 as mt5
        print("✓ MetaTrader5 package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MetaTrader5: {e}")
        print("\nFix: Run 'pip install MetaTrader5'")
        return False
    
    # Step 2: Initialize
    try:
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"✗ MT5 initialization failed: {error}")
            print("\nPossible fixes:")
            print("- Make sure MT5 is running and logged in")
            print("- Enable 'Allow DLL imports' in MT5 settings")
            print("- Try running as Administrator")
            return False
        else:
            print("✓ MT5 initialized successfully")
    except Exception as e:
        print(f"✗ MT5 initialization error: {e}")
        return False
    
    # Step 3: Get terminal info
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"✓ Terminal: {terminal_info.name}")
            print(f"  Path: {terminal_info.path}")
        else:
            print("⚠ Could not get terminal info")
    except Exception as e:
        print(f"⚠ Terminal info error: {e}")
    
    # Step 4: Get account info
    try:
        account_info = mt5.account_info()
        if account_info:
            print(f"✓ Account: {account_info.login}")
            print(f"  Server: {account_info.server}")
            print(f"  Balance: ${account_info.balance:,.2f}")
        else:
            print("✗ Not logged into any account")
            print("\nFix: Login to your MT5 account")
            return False
    except Exception as e:
        print(f"✗ Account info error: {e}")
        return False
    
    # Step 5: Test symbol access
    try:
        symbols = mt5.symbols_get()
        if symbols:
            print(f"✓ Can access {len(symbols)} symbols")
            
            # Test specific symbol
            eurusd = mt5.symbol_info("EURUSD")
            if eurusd:
                print("✓ EURUSD data available")
            else:
                print("⚠ EURUSD not available (broker may not offer it)")
        else:
            print("✗ No symbols available")
    except Exception as e:
        print(f"⚠ Symbol access error: {e}")
    
    # Step 6: Test price data
    try:
        tick = mt5.symbol_info_tick("EURUSD")
        if tick:
            print(f"✓ Live price data: EURUSD = {tick.bid:.5f}")
        else:
            print("⚠ Could not get live prices")
    except Exception as e:
        print(f"⚠ Price data error: {e}")
    
    # Shutdown
    mt5.shutdown()
    
    print("\n" + "=" * 40)
    print("🎉 MT5 CONNECTION TEST PASSED!")
    print("You can now start the trading system.")
    return True

if __name__ == "__main__":
    try:
        if test_mt5_connection():
            print("\n✅ Ready to start trading system!")
        else:
            print("\n❌ Fix the issues above first.")
    except Exception as e:
        print(f"Test failed: {e}")
    
    input("\nPress Enter to exit...")