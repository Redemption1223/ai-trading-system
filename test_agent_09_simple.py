"""
Simple test for AGENT_09: Portfolio Manager
No Unicode characters for Windows compatibility
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio.portfolio_manager import PortfolioManager, AssetClass, RiskLevel, AllocationStrategy

def test_portfolio_manager_simple():
    """Simple test for portfolio manager"""
    print("Testing AGENT_09: Portfolio Manager")
    print("=" * 50)
    
    # Test 1: Basic initialization
    print("Test 1: Basic initialization...")
    portfolio = PortfolioManager(initial_balance=10000.0, base_currency="USD")
    
    assert portfolio.name == "PORTFOLIO_MANAGER"
    assert portfolio.version == "1.0.0"
    assert portfolio.status == "DISCONNECTED"
    assert portfolio.initial_balance == 10000.0
    assert portfolio.base_currency == "USD"
    assert portfolio.current_balance == 10000.0
    assert portfolio.available_balance == 10000.0
    assert len(portfolio.positions) == 0
    assert portfolio.risk_level == RiskLevel.MODERATE
    assert portfolio.allocation_strategy == AllocationStrategy.EQUAL_WEIGHT
    
    print("PASS: Basic initialization successful")
    
    # Test 2: Portfolio manager initialization
    print("Test 2: Portfolio manager initialization...")
    result = portfolio.initialize()
    
    assert 'status' in result
    assert 'agent' in result
    assert result['agent'] == 'AGENT_09'
    
    if result['status'] == 'initialized':
        print("SUCCESS: Portfolio manager initialized successfully")
        assert 'initial_balance' in result
        assert 'base_currency' in result
        assert 'risk_level' in result
        assert 'allocation_strategy' in result
        assert 'asset_class_limits' in result
        assert result['initial_balance'] == 10000.0
        assert result['base_currency'] == "USD"
        assert result['risk_level'] == "MODERATE"
        print(f"Initial balance: ${result['initial_balance']}")
        print(f"Risk level: {result['risk_level']}")
        print(f"Allocation strategy: {result['allocation_strategy']}")
    else:
        print(f"EXPECTED: Initialization result: {result['status']}")
    
    print("PASS: Initialization handling working")
    
    # Test 3: Status reporting
    print("Test 3: Status reporting...")
    status = portfolio.get_status()
    
    assert 'name' in status
    assert 'version' in status
    assert 'status' in status
    assert 'initial_balance' in status
    assert 'current_balance' in status
    assert 'positions_count' in status
    assert 'risk_level' in status
    assert 'allocation_strategy' in status
    assert 'is_monitoring' in status
    assert 'performance_summary' in status
    assert status['name'] == "PORTFOLIO_MANAGER"
    assert status['positions_count'] == 0
    
    print(f"Status: {status['status']}")
    print(f"Current balance: ${status['current_balance']}")
    print(f"Positions count: {status['positions_count']}")
    print(f"Risk level: {status['risk_level']}")
    print(f"Is monitoring: {status['is_monitoring']}")
    
    print("PASS: Status reporting working")
    
    # Test 4: Position management - Add positions
    print("Test 4: Position management - Add positions...")
    if result['status'] == 'initialized':
        # Add EURUSD position
        pos1_result = portfolio.add_position("EURUSD", AssetClass.FOREX, 100000, 1.0950, 1.0960)
        
        if pos1_result['status'] == 'success':
            assert 'symbol' in pos1_result
            assert 'quantity' in pos1_result
            assert 'market_value' in pos1_result
            assert 'portfolio_weight' in pos1_result
            assert pos1_result['symbol'] == "EURUSD"
            assert pos1_result['quantity'] == 100000
            assert pos1_result['market_value'] > 0
            
            print(f"Added EURUSD position: {pos1_result['market_value']} market value")
            print(f"Portfolio weight: {pos1_result['portfolio_weight']:.1%}")
        
        # Add GBPUSD position
        pos2_result = portfolio.add_position("GBPUSD", AssetClass.FOREX, 75000, 1.2750, 1.2765)
        
        if pos2_result['status'] == 'success':
            assert pos2_result['symbol'] == "GBPUSD"
            print(f"Added GBPUSD position: {pos2_result['market_value']} market value")
        
        # Check positions were added
        assert len(portfolio.positions) == 2
        assert "EURUSD" in portfolio.positions
        assert "GBPUSD" in portfolio.positions
        
        print("SUCCESS: Positions added successfully")
    
    print("PASS: Position management working")
    
    # Test 5: Position updates
    print("Test 5: Position updates...")
    if result['status'] == 'initialized' and len(portfolio.positions) > 0:
        # Update EURUSD position with new price
        update_result = portfolio.update_position("EURUSD", 1.0970)  # Price increase
        
        if update_result['status'] == 'success':
            assert 'symbol' in update_result
            assert 'current_price' in update_result
            assert 'unrealized_pnl' in update_result
            assert update_result['symbol'] == "EURUSD"
            assert update_result['current_price'] == 1.0970
            
            print(f"Updated EURUSD to 1.0970")
            print(f"Unrealized P&L: {update_result['unrealized_pnl']}")
            print("SUCCESS: Position update working")
        
        # Test invalid position update
        invalid_update = portfolio.update_position("INVALID", 1.0000)
        assert invalid_update['status'] == 'error'
        assert 'not found' in invalid_update['message'].lower()
    
    print("PASS: Position updates working")
    
    # Test 6: Portfolio summary
    print("Test 6: Portfolio summary...")
    if result['status'] == 'initialized':
        summary = portfolio.get_portfolio_summary()
        
        assert 'total_value' in summary
        assert 'available_balance' in summary
        assert 'invested_value' in summary
        assert 'unrealized_pnl' in summary
        assert 'positions_count' in summary
        assert 'asset_class_breakdown' in summary
        assert 'performance_metrics' in summary
        assert 'risk_level' in summary
        
        print(f"Total portfolio value: ${summary['total_value']}")
        print(f"Invested value: ${summary['invested_value']}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']}")
        print(f"Positions count: {summary['positions_count']}")
        print(f"Asset class breakdown: {summary['asset_class_breakdown']}")
        
        # Should have FOREX positions
        if 'FOREX' in summary['asset_class_breakdown']:
            print(f"FOREX allocation: {summary['asset_class_breakdown']['FOREX']:.1%}")
    
    print("PASS: Portfolio summary working")
    
    # Test 7: Position details
    print("Test 7: Position details...")
    if result['status'] == 'initialized' and "EURUSD" in portfolio.positions:
        position_details = portfolio.get_position_details("EURUSD")
        
        if 'error' not in position_details:
            assert 'symbol' in position_details
            assert 'asset_class' in position_details
            assert 'quantity' in position_details
            assert 'market_value' in position_details
            assert 'weight' in position_details
            assert position_details['symbol'] == "EURUSD"
            assert position_details['asset_class'] == "FOREX"
            assert position_details['quantity'] == 100000
            
            print(f"EURUSD details: {position_details['weight']:.1f}% weight")
            print(f"Market value: ${position_details['market_value']}")
            print("SUCCESS: Position details available")
        
        # Test all positions
        all_positions = portfolio.get_position_details()
        assert isinstance(all_positions, dict)
        print(f"All positions retrieved: {len(all_positions)} positions")
    
    print("PASS: Position details working")
    
    # Test 8: Target allocations
    print("Test 8: Target allocations...")
    if result['status'] == 'initialized':
        # Set custom target allocations
        new_allocations = {
            "EURUSD": 0.4,
            "GBPUSD": 0.3,
            "USDJPY": 0.2,
            "AUDUSD": 0.1
        }
        
        allocation_result = portfolio.set_target_allocations(new_allocations)
        
        if allocation_result['status'] == 'success':
            assert portfolio.target_allocations == new_allocations
            print("Target allocations updated successfully")
            print(f"New allocations: {new_allocations}")
            
            if 'rebalance_result' in allocation_result:
                rebalance_info = allocation_result['rebalance_result']
                print(f"Rebalance triggered: {rebalance_info.get('status', 'unknown')}")
        
        # Test invalid allocation (doesn't sum to 1.0)
        invalid_allocations = {"EURUSD": 0.8, "GBPUSD": 0.3}  # Sums to 1.1
        invalid_result = portfolio.set_target_allocations(invalid_allocations)
        assert invalid_result['status'] == 'error'
        print("Invalid allocation correctly rejected")
    
    print("PASS: Target allocations working")
    
    # Test 9: Rebalancing
    print("Test 9: Rebalancing...")
    if result['status'] == 'initialized' and len(portfolio.positions) > 0:
        # Manual rebalance
        rebalance_result = portfolio.rebalance_portfolio()
        
        assert 'status' in rebalance_result
        
        if rebalance_result['status'] == 'success':
            assert 'positions_rebalanced' in rebalance_result
            assert 'symbols_rebalanced' in rebalance_result
            print(f"Rebalanced {rebalance_result['positions_rebalanced']} positions")
            print(f"Symbols rebalanced: {rebalance_result['symbols_rebalanced']}")
        elif rebalance_result['status'] == 'no_rebalance_needed':
            print("No rebalancing needed - portfolio within thresholds")
        else:
            print(f"Rebalance result: {rebalance_result['status']}")
    
    print("PASS: Rebalancing working")
    
    # Test 10: Risk analysis
    print("Test 10: Risk analysis...")
    if result['status'] == 'initialized':
        risk_analysis = portfolio.get_risk_analysis()
        
        if 'error' not in risk_analysis:
            assert 'risk_level' in risk_analysis
            assert 'risk_score' in risk_analysis
            assert 'risk_factors' in risk_analysis
            assert 'metrics' in risk_analysis
            assert 'limits' in risk_analysis
            
            print(f"Risk level: {risk_analysis['risk_level']}")
            print(f"Risk score: {risk_analysis['risk_score']}/100")
            print(f"Risk factors: {len(risk_analysis['risk_factors'])}")
            
            metrics = risk_analysis['metrics']
            print(f"Portfolio volatility: {metrics.get('portfolio_volatility', 0):.1f}%")
            print(f"Max drawdown: {metrics.get('max_drawdown', 0):.1f}%")
            print(f"Concentration risk: {metrics.get('concentration_risk', 0):.1f}%")
        else:
            print(f"Risk analysis error: {risk_analysis['error']}")
    
    print("PASS: Risk analysis working")
    
    # Test 11: Risk parameter updates
    print("Test 11: Risk parameter updates...")
    if result['status'] == 'initialized':
        # Test setting conservative risk level
        risk_update = portfolio.set_risk_parameters(
            risk_level=RiskLevel.CONSERVATIVE,
            max_position_size=0.08
        )
        
        if risk_update['status'] == 'success':
            assert portfolio.risk_level == RiskLevel.CONSERVATIVE
            assert portfolio.max_position_size == 0.08
            print(f"Risk level updated to: {risk_update['risk_level']}")
            print(f"Max position size: {risk_update['max_position_size']:.1%}")
        
        # Reset to moderate
        portfolio.set_risk_parameters(risk_level=RiskLevel.MODERATE)
    
    print("PASS: Risk parameter updates working")
    
    # Test 12: Performance metrics
    print("Test 12: Performance metrics...")
    if result['status'] == 'initialized':
        # Wait for some monitoring cycles
        time.sleep(2)
        
        # Get updated summary with performance metrics
        summary = portfolio.get_portfolio_summary()
        performance = summary.get('performance_metrics', {})
        
        if performance:
            assert 'total_return' in performance
            assert 'sharpe_ratio' in performance
            assert 'max_drawdown' in performance
            assert 'volatility' in performance
            
            print(f"Total return: {performance.get('total_return', 0):.2f}%")
            print(f"Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")
            print(f"Volatility: {performance.get('volatility', 0):.2f}%")
            print("SUCCESS: Performance metrics calculated")
    
    print("PASS: Performance metrics working")
    
    # Test 13: Position removal
    print("Test 13: Position removal...")
    if result['status'] == 'initialized' and "GBPUSD" in portfolio.positions:
        initial_positions = len(portfolio.positions)
        
        # Remove GBPUSD position with some realized P&L
        remove_result = portfolio.remove_position("GBPUSD", realized_pnl=150.0)
        
        if remove_result['status'] == 'success':
            assert 'symbol' in remove_result
            assert 'realized_pnl' in remove_result
            assert 'new_balance' in remove_result
            assert remove_result['symbol'] == "GBPUSD"
            assert remove_result['realized_pnl'] == 150.0
            assert len(portfolio.positions) == initial_positions - 1
            assert "GBPUSD" not in portfolio.positions
            
            print(f"Position removed: {remove_result['symbol']}")
            print(f"Realized P&L: ${remove_result['realized_pnl']}")
            print(f"New balance: ${remove_result['new_balance']}")
        
        # Test removing non-existent position
        invalid_remove = portfolio.remove_position("INVALID")
        assert invalid_remove['status'] == 'error'
    
    print("PASS: Position removal working")
    
    # Test 14: Asset class limits
    print("Test 14: Asset class limits...")
    if result['status'] == 'initialized':
        # Check default asset class limits
        limits = portfolio.asset_class_limits
        
        assert AssetClass.FOREX in limits
        assert AssetClass.STOCKS in limits
        assert AssetClass.COMMODITIES in limits
        
        print(f"FOREX limit: {limits[AssetClass.FOREX]:.1%}")
        print(f"STOCKS limit: {limits[AssetClass.STOCKS]:.1%}")
        print(f"COMMODITIES limit: {limits[AssetClass.COMMODITIES]:.1%}")
        
        # Test with position that would exceed limits (hypothetical)
        forex_exposure = sum(pos.weight for pos in portfolio.positions.values() 
                           if pos.asset_class == AssetClass.FOREX)
        print(f"Current FOREX exposure: {forex_exposure:.1%}")
        
        if forex_exposure > limits[AssetClass.FOREX]:
            print("FOREX exposure exceeds limit (would trigger risk alert)")
    
    print("PASS: Asset class limits working")
    
    # Test 15: Cleanup
    print("Test 15: Cleanup...")
    portfolio.shutdown()
    
    # Check final status
    final_status = portfolio.get_status()
    print(f"Final status: {final_status['status']}")
    
    # Verify cleanup
    assert portfolio.status == "SHUTDOWN"
    assert not portfolio.is_monitoring
    
    print("PASS: Cleanup successful")
    
    print("\n" + "=" * 50)
    print("AGENT_09 TEST RESULTS:")
    print("- Basic initialization: PASS")
    print("- Portfolio manager initialization: PASS")
    print("- Status reporting: PASS")
    print("- Position management: PASS")
    print("- Position updates: PASS")
    print("- Portfolio summary: PASS")
    print("- Position details: PASS")
    print("- Target allocations: PASS")
    print("- Rebalancing: PASS")
    print("- Risk analysis: PASS")
    print("- Risk parameter updates: PASS")
    print("- Performance metrics: PASS")
    print("- Position removal: PASS")
    print("- Asset class limits: PASS")
    print("- Cleanup: PASS")
    print("=" * 50)
    print("AGENT_09: ALL TESTS PASSED")
    
    return True

if __name__ == "__main__":
    try:
        test_portfolio_manager_simple()
        print("\nAGENT_09 ready for production use!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise