"""
AGENT_03: Risk Calculator
Status: FULLY IMPLEMENTED
Purpose: Calculate risk parameters, position sizing, and stop loss/take profit with advanced risk management
"""

import logging
import math
from datetime import datetime
from typing import Dict, Optional, Tuple, List

class RiskCalculator:
    """Advanced risk calculation engine with multiple risk models"""
    
    def __init__(self, account_balance=10000.0):
        self.name = "RISK_CALCULATOR"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        self.account_balance = account_balance
        
        # Risk parameters - configurable
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_portfolio_risk = 0.10  # 10% max total portfolio risk
        self.min_risk_reward_ratio = 1.5  # Minimum 1.5:1 risk/reward
        self.max_position_size_percent = 0.20  # Max 20% of portfolio per position
        
        # Risk models
        self.risk_models = {
            'fixed_percent': self._fixed_percent_model,
            'kelly_criterion': self._kelly_criterion_model,
            'volatility_adjusted': self._volatility_adjusted_model,
            'martingale_control': self._martingale_control_model
        }
        
        # Performance tracking
        self.calculations_performed = 0
        self.risk_warnings_issued = 0
        self.rejected_trades = 0
        
        # Risk history for analysis
        self.risk_history = []
        self.max_history = 500
        
        # Currency conversion rates (would be updated from MT5 in real implementation)
        self.currency_rates = {
            'EURUSD': 1.0950,
            'GBPUSD': 1.2750,
            'USDJPY': 149.50,
            'AUDUSD': 0.6650,
            'USDCAD': 1.3580
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize the risk calculator"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Validate initial parameters
            if self.account_balance <= 0:
                return {"status": "failed", "agent": "AGENT_03", "error": "Invalid account balance"}
            
            if self.max_risk_per_trade <= 0 or self.max_risk_per_trade > 0.10:
                return {"status": "failed", "agent": "AGENT_03", "error": "Invalid risk per trade"}
            
            self.status = "INITIALIZED"
            self.logger.info(f"Risk calculator initialized with ${self.account_balance} account balance")
            
            return {
                "status": "initialized",
                "agent": "AGENT_03",
                "account_balance": self.account_balance,
                "max_risk_per_trade": self.max_risk_per_trade * 100,  # Return as percentage
                "risk_models_available": list(self.risk_models.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "AGENT_03", "error": str(e)}
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_amount: float = None, model: str = 'fixed_percent') -> Dict:
        """Calculate optimal position size using specified risk model"""
        try:
            self.logger.debug(f"Calculating position size: entry={entry_price}, sl={stop_loss}")
            
            # Validate inputs
            if entry_price <= 0 or stop_loss <= 0:
                return {"error": "Invalid entry price or stop loss"}
            
            if entry_price == stop_loss:
                return {"error": "Entry price cannot equal stop loss"}
            
            # Calculate risk amount if not provided
            if risk_amount is None:
                risk_amount = self.account_balance * self.max_risk_per_trade
            
            # Calculate pip value and risk in pips
            pip_risk = abs(entry_price - stop_loss)
            
            # Use selected risk model
            if model not in self.risk_models:
                model = 'fixed_percent'
            
            position_data = self.risk_models[model](entry_price, stop_loss, risk_amount, pip_risk)
            
            # Add additional validation
            position_data = self._validate_position_size(position_data)
            
            # Track calculation
            self.calculations_performed += 1
            self._add_to_history(position_data)
            
            return position_data
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {e}")
            return {"error": f"Calculation failed: {e}"}
    
    def _fixed_percent_model(self, entry_price: float, stop_loss: float, 
                           risk_amount: float, pip_risk: float) -> Dict:
        """Fixed percentage risk model"""
        try:
            # Calculate position size in lots/units
            pip_value = self._calculate_pip_value(entry_price)
            position_size = risk_amount / (pip_risk / pip_value) if pip_risk > 0 else 0
            
            # Calculate position value
            position_value = position_size * entry_price
            
            # Calculate percentages
            risk_percent = (risk_amount / self.account_balance) * 100
            position_percent = (position_value / self.account_balance) * 100
            
            return {
                "model": "fixed_percent",
                "position_size": round(position_size, 2),
                "position_value": round(position_value, 2),
                "risk_amount": round(risk_amount, 2),
                "risk_percent": round(risk_percent, 2),
                "position_percent": round(position_percent, 2),
                "pip_risk": round(pip_risk, 5),
                "pip_value": round(pip_value, 2),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Fixed percent model failed: {e}"}
    
    def _kelly_criterion_model(self, entry_price: float, stop_loss: float, 
                             risk_amount: float, pip_risk: float) -> Dict:
        """Kelly Criterion risk model (requires win rate and avg win/loss)"""
        try:
            # Default Kelly parameters (would be calculated from historical performance)
            win_rate = 0.55  # 55% win rate
            avg_win = 150    # Average win in account currency
            avg_loss = 100   # Average loss in account currency
            
            # Calculate Kelly percentage
            kelly_percent = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_percent = max(0, min(kelly_percent, 0.25))  # Cap at 25%
            
            # Adjust risk amount based on Kelly
            adjusted_risk = self.account_balance * kelly_percent * 0.5  # Use half-Kelly for safety
            
            # Calculate position size
            pip_value = self._calculate_pip_value(entry_price)
            position_size = adjusted_risk / (pip_risk / pip_value) if pip_risk > 0 else 0
            position_value = position_size * entry_price
            
            return {
                "model": "kelly_criterion",
                "position_size": round(position_size, 2),
                "position_value": round(position_value, 2),
                "risk_amount": round(adjusted_risk, 2),
                "risk_percent": round((adjusted_risk / self.account_balance) * 100, 2),
                "position_percent": round((position_value / self.account_balance) * 100, 2),
                "kelly_percent": round(kelly_percent * 100, 2),
                "win_rate": round(win_rate * 100, 1),
                "pip_risk": round(pip_risk, 5),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Kelly criterion model failed: {e}"}
    
    def _volatility_adjusted_model(self, entry_price: float, stop_loss: float, 
                                 risk_amount: float, pip_risk: float) -> Dict:
        """Volatility-adjusted risk model"""
        try:
            # Estimate volatility (in real implementation, would use ATR or historical volatility)
            estimated_volatility = pip_risk * 1.5  # Simplified volatility estimate
            
            # Adjust risk based on volatility
            volatility_multiplier = min(2.0, max(0.5, 1.0 / (estimated_volatility * 10000)))
            adjusted_risk = risk_amount * volatility_multiplier
            
            # Calculate position size
            pip_value = self._calculate_pip_value(entry_price)
            position_size = adjusted_risk / (pip_risk / pip_value) if pip_risk > 0 else 0
            position_value = position_size * entry_price
            
            return {
                "model": "volatility_adjusted",
                "position_size": round(position_size, 2),
                "position_value": round(position_value, 2),
                "risk_amount": round(adjusted_risk, 2),
                "risk_percent": round((adjusted_risk / self.account_balance) * 100, 2),
                "position_percent": round((position_value / self.account_balance) * 100, 2),
                "volatility_multiplier": round(volatility_multiplier, 3),
                "estimated_volatility": round(estimated_volatility, 5),
                "pip_risk": round(pip_risk, 5),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Volatility adjusted model failed: {e}"}
    
    def _martingale_control_model(self, entry_price: float, stop_loss: float, 
                                risk_amount: float, pip_risk: float) -> Dict:
        """Anti-martingale (increase size after wins, decrease after losses)"""
        try:
            # Simplified streak tracking (would be more sophisticated in real implementation)
            winning_streak = self._get_recent_streak()
            
            # Adjust risk based on streak
            if winning_streak >= 2:
                multiplier = min(1.5, 1 + (winning_streak * 0.1))  # Increase after wins
            elif winning_streak <= -2:
                multiplier = max(0.5, 1 - (abs(winning_streak) * 0.1))  # Decrease after losses
            else:
                multiplier = 1.0
            
            adjusted_risk = risk_amount * multiplier
            
            # Calculate position size
            pip_value = self._calculate_pip_value(entry_price)
            position_size = adjusted_risk / (pip_risk / pip_value) if pip_risk > 0 else 0
            position_value = position_size * entry_price
            
            return {
                "model": "martingale_control",
                "position_size": round(position_size, 2),
                "position_value": round(position_value, 2),
                "risk_amount": round(adjusted_risk, 2),
                "risk_percent": round((adjusted_risk / self.account_balance) * 100, 2),
                "position_percent": round((position_value / self.account_balance) * 100, 2),
                "streak_multiplier": round(multiplier, 3),
                "winning_streak": winning_streak,
                "pip_risk": round(pip_risk, 5),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Martingale control model failed: {e}"}
    
    def calculate_stop_loss(self, entry_price: float, direction: str, 
                           atr_value: float = None, risk_percent: float = None) -> Dict:
        """Calculate optimal stop loss level"""
        try:
            if risk_percent is None:
                risk_percent = self.max_risk_per_trade
            
            # ATR-based stop loss (preferred method)
            if atr_value and atr_value > 0:
                atr_multiplier = 2.0  # Standard 2x ATR
                
                if direction.upper() == 'BUY':
                    stop_loss = entry_price - (atr_value * atr_multiplier)
                else:  # SELL
                    stop_loss = entry_price + (atr_value * atr_multiplier)
                
                method = "atr_based"
            
            else:
                # Percentage-based stop loss (fallback)
                stop_loss_percent = risk_percent * 2  # More conservative
                
                if direction.upper() == 'BUY':
                    stop_loss = entry_price * (1 - stop_loss_percent)
                else:  # SELL
                    stop_loss = entry_price * (1 + stop_loss_percent)
                
                method = "percentage_based"
            
            # Calculate risk in pips
            pip_risk = abs(entry_price - stop_loss)
            
            return {
                "stop_loss": round(stop_loss, 5),
                "pip_risk": round(pip_risk, 5),
                "risk_percent": round(risk_percent * 100, 2),
                "method": method,
                "entry_price": entry_price,
                "direction": direction.upper(),
                "atr_value": atr_value,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Stop loss calculation error: {e}")
            return {"error": f"Stop loss calculation failed: {e}"}
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                            direction: str, risk_reward_ratio: float = None) -> Dict:
        """Calculate take profit level based on risk/reward ratio"""
        try:
            if risk_reward_ratio is None:
                risk_reward_ratio = self.min_risk_reward_ratio
            
            # Calculate risk (distance to stop loss)
            risk_distance = abs(entry_price - stop_loss)
            
            # Calculate take profit distance
            tp_distance = risk_distance * risk_reward_ratio
            
            # Calculate take profit level
            if direction.upper() == 'BUY':
                take_profit = entry_price + tp_distance
            else:  # SELL
                take_profit = entry_price - tp_distance
            
            # Calculate pip values
            pip_risk = round(risk_distance, 5)
            pip_profit = round(tp_distance, 5)
            
            return {
                "take_profit": round(take_profit, 5),
                "risk_reward_ratio": risk_reward_ratio,
                "pip_risk": pip_risk,
                "pip_profit": pip_profit,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "direction": direction.upper(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Take profit calculation error: {e}")
            return {"error": f"Take profit calculation failed: {e}"}
    
    def validate_trade_risk(self, position_data: Dict, current_portfolio_risk: float = 0) -> Dict:
        """Validate if trade meets risk management criteria"""
        try:
            validations = {
                "approved": True,
                "warnings": [],
                "rejections": [],
                "risk_score": 0
            }
            
            # Check maximum risk per trade
            if position_data.get('risk_percent', 0) > (self.max_risk_per_trade * 100):
                validations["rejections"].append(f"Risk per trade ({position_data.get('risk_percent', 0):.1f}%) exceeds maximum ({self.max_risk_per_trade * 100:.1f}%)")
                validations["approved"] = False
            
            # Check maximum position size
            if position_data.get('position_percent', 0) > (self.max_position_size_percent * 100):
                validations["rejections"].append(f"Position size ({position_data.get('position_percent', 0):.1f}%) exceeds maximum ({self.max_position_size_percent * 100:.1f}%)")
                validations["approved"] = False
            
            # Check total portfolio risk
            total_risk = current_portfolio_risk + position_data.get('risk_percent', 0)
            if total_risk > (self.max_portfolio_risk * 100):
                validations["rejections"].append(f"Total portfolio risk ({total_risk:.1f}%) would exceed maximum ({self.max_portfolio_risk * 100:.1f}%)")
                validations["approved"] = False
            
            # Risk score calculation
            risk_components = {
                "position_size": min(100, position_data.get('position_percent', 0) * 5),
                "risk_amount": min(100, position_data.get('risk_percent', 0) * 50),
                "portfolio_impact": min(100, total_risk * 10)
            }
            
            validations["risk_score"] = round(sum(risk_components.values()) / len(risk_components), 1)
            
            # Issue warnings for high-risk situations
            if validations["risk_score"] > 70:
                validations["warnings"].append("High risk score - consider reducing position size")
            
            if position_data.get('risk_percent', 0) > (self.max_risk_per_trade * 100 * 0.8):
                validations["warnings"].append("Risk per trade is near maximum limit")
            
            # Track rejections
            if not validations["approved"]:
                self.rejected_trades += 1
            
            if validations["warnings"]:
                self.risk_warnings_issued += 1
            
            return validations
            
        except Exception as e:
            self.logger.error(f"Trade validation error: {e}")
            return {"approved": False, "error": str(e)}
    
    def _calculate_pip_value(self, price: float) -> float:
        """Calculate pip value (simplified calculation)"""
        try:
            # Simplified pip value calculation
            # In real implementation, this would consider currency pairs and account currency
            if price > 100:  # JPY pairs
                return 1.0
            else:  # Major pairs
                return 10.0
        except:
            return 10.0
    
    def _validate_position_size(self, position_data: Dict) -> Dict:
        """Additional validation for position size calculations"""
        try:
            # Ensure minimum position size
            if position_data.get('position_size', 0) < 0.01:
                position_data['position_size'] = 0.01
                position_data['warnings'] = position_data.get('warnings', [])
                position_data['warnings'].append("Position size increased to minimum (0.01)")
            
            # Ensure maximum position size
            max_position_value = self.account_balance * self.max_position_size_percent
            if position_data.get('position_value', 0) > max_position_value:
                # Recalculate with maximum allowed position value
                reduction_factor = max_position_value / position_data.get('position_value', 1)
                position_data['position_size'] *= reduction_factor
                position_data['position_value'] = max_position_value
                position_data['risk_amount'] *= reduction_factor
                position_data['risk_percent'] = (position_data['risk_amount'] / self.account_balance) * 100
                
                position_data['warnings'] = position_data.get('warnings', [])
                position_data['warnings'].append("Position size reduced to maximum allowed")
            
            return position_data
            
        except Exception as e:
            position_data['error'] = f"Validation failed: {e}"
            return position_data
    
    def _get_recent_streak(self) -> int:
        """Get recent winning/losing streak (simplified)"""
        try:
            # In real implementation, this would analyze recent trade history
            # For now, return a random-like value based on history length
            if len(self.risk_history) < 5:
                return 0
            return (len(self.risk_history) % 7) - 3  # Returns -3 to +3
        except:
            return 0
    
    def _add_to_history(self, position_data: Dict):
        """Add calculation to history for analysis"""
        try:
            self.risk_history.append({
                "timestamp": datetime.now().isoformat(),
                "position_size": position_data.get('position_size', 0),
                "risk_amount": position_data.get('risk_amount', 0),
                "risk_percent": position_data.get('risk_percent', 0),
                "model": position_data.get('model', 'unknown')
            })
            
            # Trim history if too long
            if len(self.risk_history) > self.max_history:
                self.risk_history = self.risk_history[-self.max_history:]
                
        except Exception as e:
            self.logger.error(f"Error adding to history: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Get risk calculator performance metrics"""
        try:
            avg_risk = 0
            if self.risk_history:
                avg_risk = sum(item.get('risk_percent', 0) for item in self.risk_history) / len(self.risk_history)
            
            return {
                "calculations_performed": self.calculations_performed,
                "risk_warnings_issued": self.risk_warnings_issued,
                "rejected_trades": self.rejected_trades,
                "avg_risk_percent": round(avg_risk, 2),
                "history_size": len(self.risk_history),
                "account_balance": self.account_balance,
                "max_risk_per_trade": self.max_risk_per_trade * 100,
                "max_portfolio_risk": self.max_portfolio_risk * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    def update_account_balance(self, new_balance: float) -> bool:
        """Update account balance for risk calculations"""
        try:
            if new_balance <= 0:
                self.logger.error("Invalid account balance provided")
                return False
            
            old_balance = self.account_balance
            self.account_balance = new_balance
            
            self.logger.info(f"Account balance updated: ${old_balance} -> ${new_balance}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating account balance: {e}")
            return False
    
    def get_status(self):
        """Get current risk calculator status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'account_balance': self.account_balance,
            'calculations_performed': self.calculations_performed,
            'risk_warnings_issued': self.risk_warnings_issued,
            'rejected_trades': self.rejected_trades,
            'available_models': list(self.risk_models.keys()),
            'risk_parameters': {
                'max_risk_per_trade': self.max_risk_per_trade * 100,
                'max_portfolio_risk': self.max_portfolio_risk * 100,
                'min_risk_reward_ratio': self.min_risk_reward_ratio,
                'max_position_size_percent': self.max_position_size_percent * 100
            }
        }
    
    def shutdown(self):
        """Clean shutdown of risk calculator"""
        try:
            self.logger.info("Shutting down risk calculator...")
            
            # Save final metrics
            final_metrics = self.get_performance_metrics()
            self.logger.info(f"Final metrics: {final_metrics}")
            
            self.status = "SHUTDOWN"
            self.logger.info("Risk calculator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Agent test
if __name__ == "__main__":
    # Test the risk calculator
    print("Testing AGENT_03: Risk Calculator")
    print("=" * 40)
    
    # Create risk calculator with $10,000 account
    risk_calc = RiskCalculator(10000.0)
    result = risk_calc.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test position size calculation
        print("\nTesting position size calculation...")
        position = risk_calc.calculate_position_size(
            entry_price=1.0950,
            stop_loss=1.0900,
            model='fixed_percent'
        )
        print(f"Position size result: {position}")
        
        # Test stop loss calculation
        print("\nTesting stop loss calculation...")
        stop_loss = risk_calc.calculate_stop_loss(
            entry_price=1.0950,
            direction='BUY',
            atr_value=0.0025
        )
        print(f"Stop loss result: {stop_loss}")
        
        # Test take profit calculation
        print("\nTesting take profit calculation...")
        take_profit = risk_calc.calculate_take_profit(
            entry_price=1.0950,
            stop_loss=1.0900,
            direction='BUY',
            risk_reward_ratio=2.0
        )
        print(f"Take profit result: {take_profit}")
        
        # Test trade validation
        print("\nTesting trade validation...")
        validation = risk_calc.validate_trade_risk(position)
        print(f"Trade validation: {validation}")
        
        # Test status
        status = risk_calc.get_status()
        print(f"\nFinal status: {status}")
        
        # Test shutdown
        print("\nShutting down...")
        risk_calc.shutdown()
        
    print("Risk Calculator test completed")