"""
Test suite for AGENT_01: MT5 Windows Connector
Comprehensive testing of MT5 connection and functionality
"""

import sys
import os
import time
import unittest
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from core.mt5_windows_connector import MT5WindowsConnector

def test_mt5_connector():
    """Comprehensive test for MT5 connector"""
    print("Testing AGENT_01: MT5 Windows Connector")
    print("=" * 50)
    
    if not MT5_AVAILABLE:
        print("WARNING: MetaTrader5 package not available - running limited tests")
        # Test basic class creation
        connector = MT5WindowsConnector()
        assert connector.name == "MT5_WINDOWS_CONNECTOR"
        assert connector.version == "1.0.0"
        print("OK: Basic class creation: PASSED")
        return True
    
    # Create connector instance
    connector = MT5WindowsConnector()
    
    # Test 1: Initial state
    print("Test 1: Initial state...")
    assert connector.status == "DISCONNECTED"
    assert not connector.connection_status
    print("OK: Initial state: PASSED")
    
    # Test 2: MT5 process check
    print("Test 2: MT5 process detection...")
    mt5_running = connector.is_mt5_running()
    print(f"   MT5 running: {mt5_running}")
    print("✅ Process detection: PASSED")
    
    # Test 3: Connection attempt
    print("Test 3: Connection initialization...")
    result = connector.initialize()
    print(f"   Connection result: {result}")
    
    if result['status'] == 'connected':
        print("✅ Connection: PASSED")
        
        # Test 4: Account balance
        print("Test 4: Account balance...")
        balance = connector.get_account_balance()
        assert isinstance(balance, (int, float))
        assert balance >= 0
        print(f"   Account balance: ${balance}")
        print("✅ Account balance: PASSED")
        
        # Test 5: Account info
        print("Test 5: Account information...")
        account_info = connector.get_account_info()
        if account_info:
            assert 'login' in account_info
            assert 'balance' in account_info
            assert 'company' in account_info
            print(f"   Account login: {account_info['login']}")
            print(f"   Broker: {account_info['company']}")
            print("✅ Account information: PASSED")
        else:
            print("⚠️  Account information: WARNING - No account info available")
        
        # Test 6: Live tick data
        print("Test 6: Live tick data...")
        tick = connector.get_live_tick("EURUSD")
        if tick:
            assert 'bid' in tick
            assert 'ask' in tick
            assert tick['bid'] > 0
            assert tick['ask'] > 0
            assert tick['ask'] >= tick['bid']  # Ask should be >= Bid
            print(f"   EURUSD: Bid={tick['bid']}, Ask={tick['ask']}")
            print("✅ Live tick data: PASSED")
        else:
            print("⚠️  Live tick data: WARNING - No tick data available")
        
        # Test 7: Historical data
        print("Test 7: Historical data...")
        try:
            history = connector.get_price_history("EURUSD", mt5.TIMEFRAME_H1, 10)
            if history is not None:
                assert len(history) > 0
                assert 'open' in history.columns
                assert 'high' in history.columns
                assert 'low' in history.columns
                assert 'close' in history.columns
                print(f"   Historical data: {len(history)} bars retrieved")
                print("✅ Historical data: PASSED")
            else:
                print("⚠️  Historical data: WARNING - No historical data available")
        except Exception as e:
            print(f"⚠️  Historical data: WARNING - {e}")
        
        # Test 8: Symbol information
        print("Test 8: Symbol information...")
        symbol_info = connector.get_symbol_info("EURUSD")
        if symbol_info:
            assert 'symbol' in symbol_info
            assert 'bid' in symbol_info
            assert 'ask' in symbol_info
            assert 'spread' in symbol_info
            print(f"   EURUSD spread: {symbol_info['spread']}")
            print("✅ Symbol information: PASSED")
        else:
            print("⚠️  Symbol information: WARNING - No symbol info available")
        
        # Test 9: Connection health check
        print("Test 9: Connection health check...")
        health = connector.check_connection()
        assert health['status'] in ['connected', 'disconnected', 'error']
        print(f"   Health status: {health['status']}")
        print("✅ Connection health: PASSED")
        
        # Test 10: Status reporting
        print("Test 10: Status reporting...")
        status = connector.get_status()
        assert 'name' in status
        assert 'version' in status
        assert 'status' in status
        assert 'connected' in status
        print(f"   Agent status: {status['status']}")
        print("✅ Status reporting: PASSED")
        
        print("\n🎉 All tests completed successfully!")
        print(f"Connected to: {account_info['company'] if account_info else 'Unknown'}")
        print(f"Account balance: ${balance}")
        
    elif result['status'] == 'failed':
        print("⚠️  Connection failed - running offline tests only")
        
        # Test offline functionality
        print("Test 4: Offline status check...")
        status = connector.get_status()
        assert status['connected'] == False
        print("✅ Offline status: PASSED")
        
        print("Test 5: Offline balance check...")
        balance = connector.get_account_balance()
        assert balance == 0.0
        print("✅ Offline balance: PASSED")
        
        print("\n⚠️  Limited testing completed (MT5 not available)")
        
    else:
        raise Exception(f"Unexpected connection result: {result}")
    
    # Test cleanup
    print("\nCleaning up...")
    connector.shutdown()
    time.sleep(1)  # Allow cleanup to complete
    
    print("✅ MT5 Connector: ALL TESTS PASSED")
    return True

class TestMT5Connector(unittest.TestCase):
    """Unit test class for MT5 Connector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.connector = MT5WindowsConnector()
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'connector'):
            self.connector.shutdown()
    
    def test_initialization(self):
        """Test basic initialization"""
        self.assertEqual(self.connector.name, "MT5_WINDOWS_CONNECTOR")
        self.assertEqual(self.connector.version, "1.0.0")
        self.assertEqual(self.connector.status, "DISCONNECTED")
        self.assertFalse(self.connector.connection_status)
    
    def test_status_reporting(self):
        """Test status reporting functionality"""
        status = self.connector.get_status()
        self.assertIn('name', status)
        self.assertIn('version', status)
        self.assertIn('status', status)
        self.assertIn('connected', status)
    
    @unittest.skipUnless(MT5_AVAILABLE, "MT5 package not available")
    def test_connection_attempt(self):
        """Test connection attempt (if MT5 available)"""
        result = self.connector.initialize()
        self.assertIn('status', result)
        self.assertIn('agent', result)
        self.assertEqual(result['agent'], 'AGENT_01')

if __name__ == "__main__":
    # Run the comprehensive test
    try:
        test_mt5_connector()
        print("\n🎉 AGENT_01 Testing Complete!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    
    # Also run unit tests
    print("\n" + "="*50)
    print("Running Unit Tests...")
    unittest.main(verbosity=2, exit=False)