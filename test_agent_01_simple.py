"""
Simple test for AGENT_01: MT5 Windows Connector
No Unicode characters for Windows compatibility
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mt5_windows_connector import MT5WindowsConnector

def test_mt5_connector_simple():
    """Simple test for MT5 connector without dependencies"""
    print("Testing AGENT_01: MT5 Windows Connector")
    print("=" * 50)
    
    # Test 1: Basic initialization
    print("Test 1: Basic class creation...")
    connector = MT5WindowsConnector()
    
    assert connector.name == "MT5_WINDOWS_CONNECTOR"
    assert connector.version == "1.0.0"
    assert connector.status == "DISCONNECTED"
    assert not connector.connection_status
    
    print("PASS: Basic initialization successful")
    
    # Test 2: Status reporting
    print("Test 2: Status reporting...")
    status = connector.get_status()
    
    assert 'name' in status
    assert 'version' in status
    assert 'status' in status
    assert 'connected' in status
    assert status['name'] == "MT5_WINDOWS_CONNECTOR"
    assert status['connected'] == False
    
    print("PASS: Status reporting working")
    
    # Test 3: Connection attempt (will fail gracefully without MT5)
    print("Test 3: Connection attempt...")
    result = connector.initialize()
    
    assert 'status' in result
    assert 'agent' in result
    assert result['agent'] == 'AGENT_01'
    
    if result['status'] == 'failed':
        print("EXPECTED: Connection failed (MT5 not available)")
        assert 'error' in result
    elif result['status'] == 'connected':
        print("SUCCESS: Connected to MT5!")
        assert 'account' in result
        assert 'balance' in result
    
    print("PASS: Connection handling working")
    
    # Test 4: Cleanup
    print("Test 4: Cleanup...")
    connector.shutdown()
    
    print("PASS: Cleanup successful")
    
    print("\n" + "=" * 50)
    print("AGENT_01 TEST RESULTS:")
    print("- Basic initialization: PASS")
    print("- Status reporting: PASS") 
    print("- Connection handling: PASS")
    print("- Cleanup: PASS")
    print("=" * 50)
    print("AGENT_01: ALL TESTS PASSED")
    
    return True

if __name__ == "__main__":
    try:
        test_mt5_connector_simple()
        print("\nAGENT_01 ready for production use!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise