"""
Quick test of the fixed AGI Trading System
Tests the key components that were failing
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fixed_connector():
    """Test the fixed MT5 connector"""
    print("Testing Fixed MT5 Connector...")
    print("-" * 40)
    
    try:
        from mt5_connector_fixed import MT5ConnectorFixed
        
        connector = MT5ConnectorFixed()
        result = connector.initialize()
        
        if result['status'] == 'initialized':
            print(f"✅ MT5 Connection: SUCCESS")
            print(f"   Account: {result['account']}")
            print(f"   Balance: ${result['balance']:,.2f}")
            
            # Test compatibility attributes
            print(f"   Connection Status: {connector.connection_status}")
            
            # Test compatibility methods
            price = connector.get_live_tick("EURUSD")
            if price['status'] == 'success':
                print(f"   EURUSD Price: {price['bid']:.5f}")
            
            connector.shutdown()
            return True
        else:
            print(f"❌ MT5 Connection Failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ MT5 Test Error: {e}")
        return False

def test_chart_signal_agent():
    """Test Chart Signal Agent with fixed connector"""
    print("\nTesting Chart Signal Agent...")
    print("-" * 40)
    
    try:
        from mt5_connector_fixed import MT5ConnectorFixed
        from core.chart_signal_agent import ChartSignalAgent
        
        # Initialize MT5
        mt5_connector = MT5ConnectorFixed()
        mt5_result = mt5_connector.initialize()
        
        if mt5_result['status'] != 'initialized':
            print("❌ MT5 required for Chart Signal Agent")
            return False
        
        # Test Chart Signal Agent
        chart_agent = ChartSignalAgent('EURUSD', mt5_connector)
        chart_result = chart_agent.initialize()
        
        if chart_result['status'] == 'initialized':
            print("✅ Chart Signal Agent: SUCCESS")
            print(f"   Symbol: {chart_result.get('symbol', 'Unknown')}")
            chart_agent.shutdown()
            mt5_connector.shutdown()
            return True
        else:
            print(f"❌ Chart Signal Agent Failed: {chart_result.get('error')}")
            mt5_connector.shutdown()
            return False
            
    except Exception as e:
        print(f"❌ Chart Signal Test Error: {e}")
        return False

def test_technical_analyst_methods():
    """Test Technical Analyst method names"""
    print("\nTesting Technical Analyst Methods...")
    print("-" * 40)
    
    try:
        from data.technical_analyst import TechnicalAnalyst
        
        analyst = TechnicalAnalyst('EURUSD')
        result = analyst.initialize()
        
        if result['status'] == 'initialized':
            print("✅ Technical Analyst: SUCCESS")
            
            # Check method names
            if hasattr(analyst, 'start_realtime_analysis'):
                print("   ✅ start_realtime_analysis: Available")
            elif hasattr(analyst, 'start_real_time_analysis'):
                print("   ✅ start_real_time_analysis: Available")
            else:
                print("   ❌ Real-time analysis method not found")
            
            if hasattr(analyst, 'stop_realtime_analysis'):
                print("   ✅ stop_realtime_analysis: Available")
            elif hasattr(analyst, 'stop_real_time_analysis'):
                print("   ✅ stop_real_time_analysis: Available")
            else:
                print("   ❌ Stop analysis method not found")
            
            analyst.shutdown()
            return True
        else:
            print(f"❌ Technical Analyst Failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Technical Analyst Test Error: {e}")
        return False

def test_neural_brain_accuracy():
    """Test Neural Brain accuracy reporting"""
    print("\nTesting Neural Brain Accuracy...")
    print("-" * 40)
    
    try:
        from ml.neural_signal_brain import NeuralSignalBrain
        
        neural = NeuralSignalBrain()
        result = neural.initialize()
        
        if result['status'] == 'initialized':
            print("✅ Neural Signal Brain: SUCCESS")
            
            # Check accuracy reporting
            accuracy = result.get('model_accuracy', result.get('accuracy', 0))
            print(f"   Raw Accuracy: {accuracy}")
            
            if accuracy > 1:
                print(f"   Formatted: {accuracy:.1f}%")
            elif accuracy > 0:
                print(f"   Formatted: {accuracy*100:.1f}%")
            else:
                print(f"   Status: Training in progress")
            
            neural.shutdown()
            return True
        else:
            print(f"❌ Neural Brain Failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Neural Brain Test Error: {e}")
        return False

def main():
    """Run all fixed system tests"""
    print("="*60)
    print("AGI TRADING SYSTEM - FIXED COMPONENTS TEST")
    print("="*60)
    
    tests = [
        test_fixed_connector,
        test_chart_signal_agent,
        test_technical_analyst_methods,
        test_neural_brain_accuracy
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL FIXES SUCCESSFUL!")
        print("Your system should now start without errors.")
        print("\nRun: python start_trading_system.py")
    else:
        print("⚠️  Some issues remain. Check the output above.")
    
    print("="*60)

if __name__ == "__main__":
    main()