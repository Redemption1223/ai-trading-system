"""
MT5 Connection Diagnostic Tool
Helps identify and fix MT5 connection issues
"""

import sys
import os
import subprocess

def check_mt5_process():
    """Check if MT5 is running"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name']):
            if 'terminal64' in proc.info['name'].lower() or 'metatrader' in proc.info['name'].lower():
                print(f"✓ Found MT5 process: {proc.info['name']} (PID: {proc.info['pid']})")
                return True
        print("✗ MT5 process not found")
        return False
    except ImportError:
        print("⚠ psutil not available - cannot check MT5 process")
        return None

def check_mt5_package():
    """Check MT5 Python package"""
    try:
        import MetaTrader5 as mt5
        print("✓ MetaTrader5 package is installed")
        
        # Try to initialize
        if mt5.initialize():
            print("✓ MT5 initialization successful")
            
            # Get terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"✓ Terminal info: {terminal_info.name}")
                print(f"  Company: {terminal_info.company}")
                print(f"  Path: {terminal_info.path}")
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                print(f"✓ Account connected: {account_info.login}")
                print(f"  Server: {account_info.server}")
                print(f"  Balance: ${account_info.balance}")
            else:
                print("✗ No account logged in")
            
            # Test getting symbol info
            symbol_info = mt5.symbol_info("EURUSD")
            if symbol_info:
                print("✓ Can access symbol data (EURUSD)")
            else:
                print("✗ Cannot access symbol data")
            
            mt5.shutdown()
            return True
        else:
            error = mt5.last_error()
            print(f"✗ MT5 initialization failed: {error}")
            return False
            
    except ImportError:
        print("✗ MetaTrader5 package not installed")
        return False
    except Exception as e:
        print(f"✗ MT5 package error: {e}")
        return False

def check_python_architecture():
    """Check Python architecture matches MT5"""
    import platform
    arch = platform.architecture()[0]
    print(f"Python architecture: {arch}")
    
    if arch == "64bit":
        print("✓ Python is 64-bit (compatible with MT5)")
        return True
    else:
        print("✗ Python is 32-bit (may cause issues with MT5)")
        return False

def check_windows_version():
    """Check Windows version"""
    import platform
    print(f"Windows version: {platform.platform()}")

def run_diagnostics():
    """Run all diagnostic checks"""
    print("=" * 60)
    print("MT5 CONNECTION DIAGNOSTIC")
    print("=" * 60)
    
    print("\n1. Checking Windows environment...")
    check_windows_version()
    
    print("\n2. Checking Python architecture...")
    python_ok = check_python_architecture()
    
    print("\n3. Checking MT5 process...")
    process_ok = check_mt5_process()
    
    print("\n4. Checking MetaTrader5 Python package...")
    package_ok = check_mt5_package()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if package_ok:
        print("🎉 SUCCESS: MT5 connection is working!")
        print("\nYour system is ready. Try starting the trading system again.")
    else:
        print("❌ ISSUES FOUND:")
        
        if not python_ok:
            print("  - Install 64-bit Python")
        
        if process_ok is False:
            print("  - Start MetaTrader 5")
        
        if not package_ok:
            print("  - Fix MetaTrader5 package installation")
        
        print("\n🔧 RECOMMENDED FIXES:")
        
        if process_ok is False:
            print("1. Open MetaTrader 5 and login to your account")
        
        print("2. In MT5: Tools > Options > Expert Advisors")
        print("   - Check 'Allow DLL imports'")
        print("   - Check 'Enable Expert Advisors'")
        
        print("3. Reinstall MetaTrader5 package:")
        print("   pip uninstall MetaTrader5 -y")
        print("   pip install MetaTrader5")
        
        if not python_ok:
            print("4. Install 64-bit Python from python.org")

def main():
    """Main function"""
    try:
        run_diagnostics()
    except Exception as e:
        print(f"Diagnostic error: {e}")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()