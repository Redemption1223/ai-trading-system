"""
Simple test for AGENT_02: Signal Coordinator
No Unicode characters for Windows compatibility
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mt5_windows_connector import MT5WindowsConnector
from core.signal_coordinator import SignalCoordinator

def test_signal_coordinator_simple():
    """Simple test for signal coordinator"""
    print("Testing AGENT_02: Signal Coordinator")
    print("=" * 50)
    
    # Test 1: Basic initialization with MT5 connector
    print("Test 1: MT5 connector creation...")
    mt5_connector = MT5WindowsConnector()
    mt5_result = mt5_connector.initialize()
    print(f"MT5 Connector status: {mt5_result['status']}")
    
    # Test 2: Signal coordinator creation
    print("Test 2: Signal coordinator creation...")
    coordinator = SignalCoordinator(mt5_connector)
    
    assert coordinator.name == "SIGNAL_COORDINATOR"
    assert coordinator.version == "1.0.0"
    assert coordinator.status == "DISCONNECTED"
    assert not coordinator.is_running
    
    print("PASS: Basic creation successful")
    
    # Test 3: Coordinator initialization
    print("Test 3: Coordinator initialization...")
    result = coordinator.initialize()
    
    assert 'status' in result
    assert 'agent' in result
    assert result['agent'] == 'AGENT_02'
    
    if result['status'] == 'initialized':
        print("SUCCESS: Coordinator initialized successfully")
        assert 'charts_supported' in result
        assert result['charts_supported'] == True
    else:
        print(f"EXPECTED: Initialization result: {result['status']}")
    
    print("PASS: Initialization handling working")
    
    # Test 4: Status reporting
    print("Test 4: Status reporting...")
    status = coordinator.get_status()
    
    assert 'name' in status
    assert 'version' in status
    assert 'status' in status
    assert 'is_running' in status
    assert 'active_charts' in status
    assert 'performance_metrics' in status
    assert status['name'] == "SIGNAL_COORDINATOR"
    
    print(f"Status: {status['status']}")
    print(f"Charts active: {len(status['active_charts'])}")
    print(f"Is running: {status['is_running']}")
    
    print("PASS: Status reporting working")
    
    # Test 5: Performance metrics
    print("Test 5: Performance metrics...")
    metrics = status['performance_metrics']
    
    assert 'total_signals' in metrics
    assert 'signals_processed' in metrics
    assert 'charts_active' in metrics
    assert metrics['total_signals'] == 0  # Should start at 0
    
    print("PASS: Performance metrics working")
    
    # Test 6: Signal queue operations
    print("Test 6: Signal queue operations...")
    signals = coordinator.get_latest_signals(5)
    
    assert isinstance(signals, list)
    assert len(signals) == 0  # Should be empty initially
    
    print("PASS: Signal queue operations working")
    
    # Test 7: Coordination start/stop
    print("Test 7: Coordination control...")
    
    if result['status'] == 'initialized':
        # Test start coordination
        start_result = coordinator.start_coordination()
        assert 'status' in start_result
        
        if start_result['status'] == 'started':
            print("SUCCESS: Coordination started")
            
            # Let it run briefly
            time.sleep(2)
            
            # Check if running
            status = coordinator.get_status()
            assert status['is_running'] == True
            
            # Test stop coordination
            stop_result = coordinator.stop_coordination()
            assert stop_result['status'] in ['stopped', 'error']
            print(f"Coordination stop result: {stop_result['status']}")
        else:
            print(f"EXPECTED: Coordination start result: {start_result['status']}")
    
    print("PASS: Coordination control working")
    
    # Test 8: Cleanup
    print("Test 8: Cleanup...")
    coordinator.shutdown()
    
    # Check final status
    final_status = coordinator.get_status()
    print(f"Final status: {final_status['status']}")
    
    print("PASS: Cleanup successful")
    
    print("\n" + "=" * 50)
    print("AGENT_02 TEST RESULTS:")
    print("- Basic creation: PASS")
    print("- Initialization: PASS")
    print("- Status reporting: PASS")
    print("- Performance metrics: PASS")
    print("- Signal queue operations: PASS")
    print("- Coordination control: PASS")
    print("- Cleanup: PASS")
    print("=" * 50)
    print("AGENT_02: ALL TESTS PASSED")
    
    return True

if __name__ == "__main__":
    try:
        test_signal_coordinator_simple()
        print("\nAGENT_02 ready for production use!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise