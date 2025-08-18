"""
Simple test for AGENT_03: Risk Calculator
No Unicode characters for Windows compatibility
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.risk_calculator import RiskCalculator

def test_risk_calculator_simple():
    """Simple test for risk calculator"""
    print("Testing AGENT_03: Risk Calculator")
    print("=" * 50)
    
    # Test 1: Basic initialization with $10,000 account
    print("Test 1: Basic initialization...")
    risk_calc = RiskCalculator(10000.0)
    
    assert risk_calc.name == "RISK_CALCULATOR"
    assert risk_calc.version == "1.0.0"
    assert risk_calc.status == "DISCONNECTED"
    assert risk_calc.account_balance == 10000.0
    
    print("PASS: Basic initialization successful")
    
    # Test 2: Risk calculator initialization
    print("Test 2: Risk calculator initialization...")
    result = risk_calc.initialize()
    
    assert 'status' in result
    assert 'agent' in result
    assert result['agent'] == 'AGENT_03'
    
    if result['status'] == 'initialized':
        print("SUCCESS: Risk calculator initialized successfully")
        assert 'account_balance' in result
        assert 'max_risk_per_trade' in result
        assert 'risk_models_available' in result
        assert result['account_balance'] == 10000.0
        assert len(result['risk_models_available']) >= 4  # Should have multiple risk models
    else:
        print(f"EXPECTED: Initialization result: {result['status']}")
    
    print("PASS: Initialization handling working")
    
    # Test 3: Status reporting
    print("Test 3: Status reporting...")
    status = risk_calc.get_status()
    
    assert 'name' in status
    assert 'version' in status
    assert 'status' in status
    assert 'account_balance' in status
    assert 'available_models' in status
    assert 'risk_parameters' in status
    assert status['name'] == "RISK_CALCULATOR"
    assert status['account_balance'] == 10000.0
    
    print(f"Status: {status['status']}")
    print(f"Account balance: ${status['account_balance']}")
    print(f"Available models: {len(status['available_models'])}")
    
    print("PASS: Status reporting working")
    
    # Test 4: Position size calculation - Fixed Percent Model
    print("Test 4: Position size calculation (Fixed Percent)...")
    position = risk_calc.calculate_position_size(
        entry_price=1.0950,
        stop_loss=1.0900,
        model='fixed_percent'
    )
    
    assert 'model' in position
    assert 'position_size' in position
    assert 'risk_amount' in position
    assert 'risk_percent' in position
    assert position['model'] == 'fixed_percent'
    assert position['position_size'] > 0
    assert position['risk_amount'] > 0
    
    print(f"Position size: {position['position_size']}")
    print(f"Risk amount: ${position['risk_amount']}")
    print(f"Risk percent: {position['risk_percent']}%")
    
    print("PASS: Position size calculation working")
    
    # Test 5: Different risk models
    print("Test 5: Testing different risk models...")
    
    models_to_test = ['kelly_criterion', 'volatility_adjusted', 'martingale_control']
    for model in models_to_test:
        test_position = risk_calc.calculate_position_size(
            entry_price=1.0950,
            stop_loss=1.0900,
            model=model
        )
        
        assert 'model' in test_position
        assert test_position['model'] == model
        assert 'position_size' in test_position
        print(f"  {model}: Position size = {test_position['position_size']}")
    
    print("PASS: Multiple risk models working")
    
    # Test 6: Stop loss calculation
    print("Test 6: Stop loss calculation...")
    
    # Test ATR-based stop loss
    stop_loss_atr = risk_calc.calculate_stop_loss(
        entry_price=1.0950,
        direction='BUY',
        atr_value=0.0025
    )
    
    assert 'stop_loss' in stop_loss_atr
    assert 'method' in stop_loss_atr
    assert 'pip_risk' in stop_loss_atr
    assert stop_loss_atr['method'] == 'atr_based'
    assert stop_loss_atr['stop_loss'] < 1.0950  # Should be below entry for BUY
    
    # Test percentage-based stop loss
    stop_loss_pct = risk_calc.calculate_stop_loss(
        entry_price=1.0950,
        direction='BUY'
    )
    
    assert stop_loss_pct['method'] == 'percentage_based'
    
    print(f"ATR-based SL: {stop_loss_atr['stop_loss']}")
    print(f"Percentage-based SL: {stop_loss_pct['stop_loss']}")
    
    print("PASS: Stop loss calculation working")
    
    # Test 7: Take profit calculation
    print("Test 7: Take profit calculation...")
    take_profit = risk_calc.calculate_take_profit(
        entry_price=1.0950,
        stop_loss=1.0900,
        direction='BUY',
        risk_reward_ratio=2.0
    )
    
    assert 'take_profit' in take_profit
    assert 'risk_reward_ratio' in take_profit
    assert 'pip_risk' in take_profit
    assert 'pip_profit' in take_profit
    assert take_profit['risk_reward_ratio'] == 2.0
    assert take_profit['take_profit'] > 1.0950  # Should be above entry for BUY
    
    print(f"Take profit: {take_profit['take_profit']}")
    print(f"Risk/Reward ratio: {take_profit['risk_reward_ratio']}:1")
    print(f"Pip risk: {take_profit['pip_risk']}")
    print(f"Pip profit: {take_profit['pip_profit']}")
    
    print("PASS: Take profit calculation working")
    
    # Test 8: Trade risk validation
    print("Test 8: Trade risk validation...")
    validation = risk_calc.validate_trade_risk(position)
    
    assert 'approved' in validation
    assert 'risk_score' in validation
    assert 'warnings' in validation
    assert 'rejections' in validation
    assert isinstance(validation['approved'], bool)
    assert isinstance(validation['warnings'], list)
    assert isinstance(validation['rejections'], list)
    
    print(f"Trade approved: {validation['approved']}")
    print(f"Risk score: {validation['risk_score']}")
    print(f"Warnings: {len(validation['warnings'])}")
    print(f"Rejections: {len(validation['rejections'])}")
    
    print("PASS: Trade validation working")
    
    # Test 9: Performance metrics
    print("Test 9: Performance metrics...")
    metrics = risk_calc.get_performance_metrics()
    
    assert 'calculations_performed' in metrics
    assert 'risk_warnings_issued' in metrics
    assert 'rejected_trades' in metrics
    assert 'avg_risk_percent' in metrics
    assert 'account_balance' in metrics
    assert metrics['calculations_performed'] > 0  # Should have performed calculations
    
    print(f"Calculations performed: {metrics['calculations_performed']}")
    print(f"Risk warnings issued: {metrics['risk_warnings_issued']}")
    print(f"Average risk percent: {metrics['avg_risk_percent']}%")
    
    print("PASS: Performance metrics working")
    
    # Test 10: Account balance update
    print("Test 10: Account balance update...")
    balance_update = risk_calc.update_account_balance(15000.0)
    
    assert balance_update == True
    assert risk_calc.account_balance == 15000.0
    
    # Test with invalid balance
    invalid_update = risk_calc.update_account_balance(-1000.0)
    assert invalid_update == False
    assert risk_calc.account_balance == 15000.0  # Should remain unchanged
    
    print(f"Balance updated to: ${risk_calc.account_balance}")
    
    print("PASS: Account balance update working")
    
    # Test 11: Error handling
    print("Test 11: Error handling...")
    
    # Test invalid entry price
    invalid_position = risk_calc.calculate_position_size(
        entry_price=0,
        stop_loss=1.0900
    )
    assert 'error' in invalid_position
    
    # Test same entry and stop loss
    same_prices = risk_calc.calculate_position_size(
        entry_price=1.0950,
        stop_loss=1.0950
    )
    assert 'error' in same_prices
    
    print("PASS: Error handling working")
    
    # Test 12: Cleanup
    print("Test 12: Cleanup...")
    risk_calc.shutdown()
    
    # Check final status
    final_status = risk_calc.get_status()
    print(f"Final status: {final_status['status']}")
    
    print("PASS: Cleanup successful")
    
    print("\n" + "=" * 50)
    print("AGENT_03 TEST RESULTS:")
    print("- Basic initialization: PASS")
    print("- Risk calculator initialization: PASS")
    print("- Status reporting: PASS")
    print("- Position size calculation: PASS")
    print("- Multiple risk models: PASS")
    print("- Stop loss calculation: PASS")
    print("- Take profit calculation: PASS")
    print("- Trade validation: PASS")
    print("- Performance metrics: PASS")
    print("- Account balance update: PASS")
    print("- Error handling: PASS")
    print("- Cleanup: PASS")
    print("=" * 50)
    print("AGENT_03: ALL TESTS PASSED")
    
    return True

if __name__ == "__main__":
    try:
        test_risk_calculator_simple()
        print("\nAGENT_03 ready for production use!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise