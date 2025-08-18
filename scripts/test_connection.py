"""Test MT5 connection and system readiness"""

import sys
import os

def test_mt5_availability():
    """Test if MT5 can be imported and initialized"""
    print("🔌 Testing MT5 connection...")
    
    try:
        import MetaTrader5 as mt5
        
        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialize() failed")
            return False
            
        print("✅ MT5 connection successful")
        
        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info is not None:
            print(f"✅ Terminal: {terminal_info.name}")
            print(f"✅ Company: {terminal_info.company}")
            
        mt5.shutdown()
        return True
        
    except ImportError:
        print("❌ MetaTrader5 package not installed")
        return False
    except Exception as e:
        print(f"❌ MT5 connection failed: {e}")
        return False

def test_system_components():
    """Test core system components"""
    print("\n🧪 Testing system components...")
    
    try:
        # Test agent imports
        sys.path.append('.')
        from core.mt5_windows_connector import MT5WindowsConnector
        from core.signal_coordinator import SignalCoordinator
        from ml.neural_signal_brain import NeuralSignalBrain
        
        print("✅ Core agents importable")
        return True
        
    except ImportError as e:
        print(f"❌ Agent import failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 AGI Trading System - Connection Test")
    print("=" * 40)
    
    mt5_ok = test_mt5_availability()
    system_ok = test_system_components()
    
    print("\n" + "=" * 40)
    if mt5_ok and system_ok:
        print("🎉 All systems ready!")
    else:
        print("⚠️ Some components need attention")
        
    print("🚀 System test complete")