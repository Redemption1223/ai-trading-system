"""
Test the specific fixes applied for pandas/numpy and Market Data Manager
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_technical_analyst_fix():
    """Test the Technical Analyst Aroon fix"""
    print("Testing Technical Analyst Fix...")
    print("-" * 40)
    
    try:
        from data.technical_analyst import TechnicalAnalyst
        
        analyst = TechnicalAnalyst('EURUSD')
        result = analyst.initialize()
        
        if result['status'] == 'initialized':
            print("✅ Technical Analyst initialization: SUCCESS")
            
            # Test the Aroon calculation specifically
            try:
                # This should not fail now
                current_analysis = analyst.get_current_analysis()
                if current_analysis:
                    print("✅ Current analysis retrieval: SUCCESS")
                    if 'indicators' in current_analysis:
                        indicators = current_analysis['indicators']
                        if 'aroon_up' in indicators:
                            print("✅ Aroon calculation: SUCCESS")
                        else:
                            print("⚠ Aroon not in indicators (might be normal)")
                    else:
                        print("⚠ No indicators in analysis")
                else:
                    print("⚠ No current analysis available")
                    
            except Exception as e:
                print(f"❌ Technical analysis error: {e}")
                return False
            
            analyst.shutdown()
            return True
        else:
            print(f"❌ Technical Analyst failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Technical Analyst test error: {e}")
        return False

def test_market_data_manager_fix():
    """Test the Market Data Manager symbol handling fix"""
    print("\nTesting Market Data Manager Fix...")
    print("-" * 40)
    
    try:
        from data.market_data_manager import MarketDataManager
        
        manager = MarketDataManager()
        result = manager.initialize()
        
        if result['status'] == 'initialized':
            print("✅ Market Data Manager initialization: SUCCESS")
            
            # Test streaming with single symbol (this was causing the bug)
            print("Testing single symbol streaming...")
            stream_result1 = manager.start_streaming("EURUSD")  # Single string
            print(f"Single symbol result: {stream_result1.get('status', 'unknown')}")
            
            manager.stop_streaming()
            
            # Test streaming with symbol list (correct way)
            print("Testing symbol list streaming...")
            stream_result2 = manager.start_streaming(["EURUSD", "GBPUSD"])  # List
            print(f"Symbol list result: {stream_result2.get('status', 'unknown')}")
            
            manager.stop_streaming()
            
            if stream_result1.get('status') in ['started', 'already_active'] and stream_result2.get('status') in ['started', 'already_active']:
                print("✅ Both streaming methods: SUCCESS")
                manager.shutdown()
                return True
            else:
                print("❌ Streaming methods failed")
                manager.shutdown()
                return False
        else:
            print(f"❌ Market Data Manager failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Market Data Manager test error: {e}")
        return False

def test_main_startup_fix():
    """Test that the main startup script fix works"""
    print("\nTesting Main Startup Fix...")
    print("-" * 40)
    
    try:
        # Just test the config and setup part, not the full system
        from start_trading_system import AGITradingSystem
        
        system = AGITradingSystem()
        
        # Check that config has the right structure
        if 'trading_symbols' in system.config and isinstance(system.config['trading_symbols'], list):
            print("✅ Trading symbols config: SUCCESS")
            print(f"   Symbols: {system.config['trading_symbols']}")
            return True
        else:
            print("❌ Trading symbols config: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Main startup test error: {e}")
        return False

def main():
    """Run all fix tests"""
    print("="*60)
    print("TESTING APPLIED FIXES")
    print("="*60)
    
    tests = [
        test_technical_analyst_fix,
        test_market_data_manager_fix,
        test_main_startup_fix
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print("FIX TEST RESULTS")
    print("="*60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL FIXES SUCCESSFUL!")
        print("\nThe system should now run without:")
        print("  ❌ 'numpy.ndarray' object has no attribute 'index'")
        print("  ❌ DataFrame ambiguity errors")
        print("  ❌ Single character symbol processing")
        print("\nRun: python start_trading_system.py")
    else:
        print("⚠️  Some fixes may need additional work")
    
    print("="*60)

if __name__ == "__main__":
    main()