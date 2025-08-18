"""
AGENT_09: Portfolio Manager
Status: FULLY IMPLEMENTED
Purpose: Advanced portfolio management with risk monitoring, allocation, and performance tracking
"""

import logging
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict, deque
import math

class AssetClass(Enum):
    """Asset classes"""
    FOREX = "FOREX"
    STOCKS = "STOCKS"
    COMMODITIES = "COMMODITIES"
    INDICES = "INDICES"
    CRYPTO = "CRYPTO"

class RiskLevel(Enum):
    """Portfolio risk levels"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    CUSTOM = "CUSTOM"

class AllocationStrategy(Enum):
    """Portfolio allocation strategies"""
    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    RISK_PARITY = "RISK_PARITY"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    CUSTOM = "CUSTOM"

class PortfolioPosition:
    """Portfolio position object"""
    
    def __init__(self, symbol: str, asset_class: AssetClass, current_price: float, 
                 quantity: float, entry_price: float = None):
        self.symbol = symbol
        self.asset_class = asset_class
        self.current_price = current_price
        self.quantity = quantity
        self.entry_price = entry_price or current_price
        self.market_value = current_price * abs(quantity)
        self.unrealized_pnl = (current_price - self.entry_price) * quantity
        self.weight = 0.0  # Portfolio weight percentage
        self.target_weight = 0.0  # Target allocation weight
        self.last_updated = datetime.now()
        
        # Risk metrics
        self.volatility = 0.0
        self.beta = 1.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.var_95 = 0.0  # 95% Value at Risk

class PortfolioManager:
    """Advanced portfolio management with risk monitoring and allocation"""
    
    def __init__(self, initial_balance: float = 10000.0, base_currency: str = "USD"):
        self.name = "PORTFOLIO_MANAGER"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        
        # Portfolio settings
        self.initial_balance = initial_balance
        self.base_currency = base_currency
        self.current_balance = initial_balance
        self.available_balance = initial_balance
        self.used_margin = 0.0
        self.free_margin = initial_balance
        
        # Risk settings
        self.risk_level = RiskLevel.MODERATE
        self.allocation_strategy = AllocationStrategy.EQUAL_WEIGHT
        self.max_portfolio_risk = 0.02  # 2% max daily portfolio risk
        self.max_position_size = 0.1  # 10% max position size
        self.max_correlation_exposure = 0.3  # 30% max correlated exposure
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance
        
        # Positions and allocations
        self.positions = {}  # symbol -> PortfolioPosition
        self.target_allocations = {}  # symbol -> target weight
        self.historical_values = deque(maxlen=1000)  # Historical portfolio values
        self.rebalance_history = deque(maxlen=100)
        
        # Asset class limits
        self.asset_class_limits = {
            AssetClass.FOREX: 0.6,  # 60% max in forex
            AssetClass.STOCKS: 0.4,  # 40% max in stocks
            AssetClass.COMMODITIES: 0.2,  # 20% max in commodities
            AssetClass.INDICES: 0.3,  # 30% max in indices
            AssetClass.CRYPTO: 0.1   # 10% max in crypto
        }
        
        # Performance tracking
        self.portfolio_metrics = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0
        }
        
        # Correlation matrix
        self.correlation_matrix = {}
        self.correlation_update_interval = 3600  # Update correlations every hour
        self.last_correlation_update = None
        
        # Connected agents
        self.trade_execution_engine = None
        self.market_data_manager = None
        self.risk_calculator = None
        
        # Monitoring threads
        self.monitoring_thread = None
        self.rebalancing_thread = None
        self.is_monitoring = False
        self.auto_rebalance_enabled = True
        self.rebalance_frequency = 86400  # Daily rebalancing
        self.last_rebalance_time = None
        
        # Price history for calculations
        self.price_history = defaultdict(deque)  # symbol -> price history
        self.max_price_history = 252  # 1 year of daily prices
        
        # Currency conversion rates (simplified)
        self.fx_rates = {
            'EURUSD': 1.0950,
            'GBPUSD': 1.2750,
            'USDJPY': 0.00668,  # 1/149.50
            'AUDUSD': 0.6650,
            'USDCHF': 0.8750,
            'USDCAD': 0.7326,  # 1/1.3650
            'USD': 1.0  # Base currency
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self, trade_execution_engine=None, market_data_manager=None, risk_calculator=None):
        """Initialize the portfolio manager"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Connect to other agents
            if trade_execution_engine:
                self.trade_execution_engine = trade_execution_engine
                self.logger.info("Trade execution engine connected")
            
            if market_data_manager:
                self.market_data_manager = market_data_manager
                self.logger.info("Market data manager connected")
            
            if risk_calculator:
                self.risk_calculator = risk_calculator
                self.logger.info("Risk calculator connected")
            
            # Set up default target allocations
            self._setup_default_allocations()
            
            # Initialize portfolio metrics
            self._initialize_portfolio_metrics()
            
            # Start monitoring threads
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_portfolio, daemon=True)
            self.monitoring_thread.start()
            
            self.rebalancing_thread = threading.Thread(target=self._auto_rebalance, daemon=True)
            self.rebalancing_thread.start()
            
            # Record initial portfolio value
            self._record_portfolio_value()
            
            self.status = "INITIALIZED"
            self.logger.info("Portfolio Manager initialized successfully")
            
            return {
                "status": "initialized",
                "agent": "AGENT_09",
                "initial_balance": self.initial_balance,
                "base_currency": self.base_currency,
                "risk_level": self.risk_level.value,
                "allocation_strategy": self.allocation_strategy.value,
                "asset_class_limits": {k.value: v for k, v in self.asset_class_limits.items()},
                "auto_rebalance_enabled": self.auto_rebalance_enabled
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "AGENT_09", "error": str(e)}
    
    def _setup_default_allocations(self):
        """Set up default target allocations"""
        try:
            # Default allocation based on strategy
            if self.allocation_strategy == AllocationStrategy.EQUAL_WEIGHT:
                # Equal weight across major currency pairs
                symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
                weight_per_symbol = 1.0 / len(symbols)
                
                for symbol in symbols:
                    self.target_allocations[symbol] = weight_per_symbol
            
            elif self.allocation_strategy == AllocationStrategy.RISK_PARITY:
                # Risk parity allocation (simplified)
                self.target_allocations = {
                    "EURUSD": 0.3,  # Lower volatility, higher allocation
                    "GBPUSD": 0.25,
                    "USDJPY": 0.2,
                    "AUDUSD": 0.25  # Higher volatility, lower allocation
                }
            
            else:
                # Default equal weight
                self.target_allocations = {
                    "EURUSD": 0.25,
                    "GBPUSD": 0.25,
                    "USDJPY": 0.25,
                    "AUDUSD": 0.25
                }
            
            self.logger.info(f"Default allocations set: {self.target_allocations}")
            
        except Exception as e:
            self.logger.error(f"Error setting up default allocations: {e}")
    
    def _initialize_portfolio_metrics(self):
        """Initialize portfolio performance metrics"""
        try:
            # Set initial values
            self.portfolio_metrics.update({
                'start_date': datetime.now().isoformat(),
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0
            })
            
            # Record initial portfolio value
            initial_record = {
                'timestamp': datetime.now().isoformat(),
                'total_value': self.current_balance,
                'available_balance': self.available_balance,
                'used_margin': self.used_margin,
                'unrealized_pnl': 0.0,
                'positions_count': 0
            }
            
            self.historical_values.append(initial_record)
            
        except Exception as e:
            self.logger.error(f"Error initializing portfolio metrics: {e}")
    
    def add_position(self, symbol: str, asset_class: AssetClass, quantity: float, 
                    entry_price: float, current_price: float = None) -> Dict:
        """Add a position to the portfolio"""
        try:
            if current_price is None:
                current_price = entry_price
            
            position = PortfolioPosition(
                symbol=symbol,
                asset_class=asset_class,
                current_price=current_price,
                quantity=quantity,
                entry_price=entry_price
            )
            
            self.positions[symbol] = position
            self._update_portfolio_weights()
            self._record_portfolio_value()
            
            self.logger.info(f"Position added: {symbol} {quantity} @ {entry_price}")
            
            return {
                "status": "success",
                "symbol": symbol,
                "quantity": quantity,
                "market_value": position.market_value,
                "portfolio_weight": position.weight
            }
            
        except Exception as e:
            self.logger.error(f"Error adding position {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def update_position(self, symbol: str, current_price: float, quantity: float = None) -> Dict:
        """Update an existing position"""
        try:
            if symbol not in self.positions:
                return {"status": "error", "message": "Position not found"}
            
            position = self.positions[symbol]
            position.current_price = current_price
            
            if quantity is not None:
                position.quantity = quantity
            
            position.market_value = current_price * abs(position.quantity)
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            position.last_updated = datetime.now()
            
            # Add to price history
            self.price_history[symbol].append({
                'price': current_price,
                'timestamp': datetime.now()
            })
            
            if len(self.price_history[symbol]) > self.max_price_history:
                self.price_history[symbol].popleft()
            
            self._update_portfolio_weights()
            self._record_portfolio_value()
            
            return {
                "status": "success",
                "symbol": symbol,
                "current_price": current_price,
                "market_value": position.market_value,
                "unrealized_pnl": position.unrealized_pnl,
                "portfolio_weight": position.weight
            }
            
        except Exception as e:
            self.logger.error(f"Error updating position {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def remove_position(self, symbol: str, realized_pnl: float = 0.0) -> Dict:
        """Remove a position from the portfolio"""
        try:
            if symbol not in self.positions:
                return {"status": "error", "message": "Position not found"}
            
            position = self.positions[symbol]
            
            # Update current balance with realized P&L
            self.current_balance += realized_pnl
            self.available_balance += position.market_value + realized_pnl
            
            # Remove position
            del self.positions[symbol]
            
            self._update_portfolio_weights()
            self._record_portfolio_value()
            
            self.logger.info(f"Position removed: {symbol}, Realized P&L: {realized_pnl}")
            
            return {
                "status": "success",
                "symbol": symbol,
                "realized_pnl": realized_pnl,
                "new_balance": self.current_balance
            }
            
        except Exception as e:
            self.logger.error(f"Error removing position {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _update_portfolio_weights(self):
        """Update portfolio weights for all positions"""
        try:
            total_market_value = sum(pos.market_value for pos in self.positions.values())
            
            if total_market_value > 0:
                for position in self.positions.values():
                    position.weight = position.market_value / total_market_value
            else:
                for position in self.positions.values():
                    position.weight = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error updating portfolio weights: {e}")
    
    def _record_portfolio_value(self):
        """Record current portfolio value for historical tracking"""
        try:
            total_market_value = sum(pos.market_value for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_value = self.available_balance + total_market_value
            
            record = {
                'timestamp': datetime.now().isoformat(),
                'total_value': total_value,
                'available_balance': self.available_balance,
                'market_value': total_market_value,
                'unrealized_pnl': total_unrealized_pnl,
                'used_margin': self.used_margin,
                'positions_count': len(self.positions)
            }
            
            self.historical_values.append(record)
            
        except Exception as e:
            self.logger.error(f"Error recording portfolio value: {e}")
    
    def _monitor_portfolio(self):
        """Monitor portfolio in real-time"""
        self.logger.info("Starting portfolio monitoring thread")
        
        while self.is_monitoring:
            try:
                # Update positions with current market data
                self._update_positions_from_market_data()
                
                # Calculate portfolio metrics
                self._calculate_portfolio_metrics()
                
                # Check risk limits
                self._check_risk_limits()
                
                # Update correlations periodically
                self._update_correlations()
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in portfolio monitoring: {e}")
                time.sleep(30)
    
    def _update_positions_from_market_data(self):
        """Update positions with current market data"""
        try:
            if not self.market_data_manager:
                return
            
            for symbol, position in self.positions.items():
                try:
                    # Get latest tick
                    tick = self.market_data_manager.get_latest_tick(symbol)
                    if tick:
                        # Use mid price for portfolio valuation
                        current_price = (tick['bid'] + tick['ask']) / 2
                        self.update_position(symbol, current_price)
                        
                except Exception as e:
                    self.logger.debug(f"Could not update {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error updating positions from market data: {e}")
    
    def _calculate_portfolio_metrics(self):
        """Calculate comprehensive portfolio performance metrics"""
        try:
            if len(self.historical_values) < 2:
                return
            
            # Get historical values
            values = [record['total_value'] for record in self.historical_values]
            current_value = values[-1]
            
            # Total return
            total_return = (current_value - self.initial_balance) / self.initial_balance
            
            # Calculate returns series
            returns = []
            for i in range(1, len(values)):
                daily_return = (values[i] - values[i-1]) / values[i-1]
                returns.append(daily_return)
            
            if len(returns) < 2:
                return
            
            # Volatility (annualized)
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            daily_volatility = math.sqrt(variance)
            annualized_volatility = daily_volatility * math.sqrt(252)
            
            # Annualized return
            days_elapsed = len(returns)
            if days_elapsed > 0:
                annualized_return = ((current_value / self.initial_balance) ** (252 / days_elapsed)) - 1
            else:
                annualized_return = 0.0
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = annualized_return / max(annualized_volatility, 0.001)
            
            # Max drawdown
            peak = self.initial_balance
            max_drawdown = 0.0
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Sortino ratio (downside deviation)
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
                downside_deviation = math.sqrt(downside_variance) * math.sqrt(252)
                sortino_ratio = annualized_return / max(downside_deviation, 0.001)
            else:
                sortino_ratio = float('inf') if annualized_return > 0 else 0.0
            
            # Calmar ratio
            calmar_ratio = annualized_return / max(max_drawdown, 0.001)
            
            # Value at Risk (95%)
            if len(returns) >= 20:
                sorted_returns = sorted(returns)
                var_index = int(0.05 * len(returns))
                var_95 = abs(sorted_returns[var_index]) * current_value
                
                # Conditional VaR (95%)
                cvar_returns = sorted_returns[:var_index + 1]
                cvar_95 = abs(sum(cvar_returns) / len(cvar_returns)) * current_value if cvar_returns else 0.0
            else:
                var_95 = 0.0
                cvar_95 = 0.0
            
            # Win rate and profit factor
            winning_periods = len([r for r in returns if r > 0])
            win_rate = winning_periods / len(returns) if returns else 0.0
            
            gross_profits = sum(r for r in returns if r > 0)
            gross_losses = abs(sum(r for r in returns if r < 0))
            profit_factor = gross_profits / max(gross_losses, 0.001)
            
            # Update metrics
            self.portfolio_metrics.update({
                'total_return': round(total_return * 100, 2),
                'annualized_return': round(annualized_return * 100, 2),
                'volatility': round(annualized_volatility * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown * 100, 2),
                'win_rate': round(win_rate * 100, 2),
                'profit_factor': round(profit_factor, 2),
                'calmar_ratio': round(calmar_ratio, 2),
                'sortino_ratio': round(min(sortino_ratio, 999.99), 2),
                'var_95': round(var_95, 2),
                'cvar_95': round(cvar_95, 2),
                'current_value': round(current_value, 2),
                'last_updated': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
    
    def _check_risk_limits(self):
        """Check portfolio risk limits and generate alerts"""
        try:
            # Check position size limits
            for symbol, position in self.positions.items():
                if position.weight > self.max_position_size:
                    self.logger.warning(f"Position size limit exceeded for {symbol}: {position.weight:.1%}")
            
            # Check asset class limits
            asset_class_exposure = defaultdict(float)
            for position in self.positions.values():
                asset_class_exposure[position.asset_class] += position.weight
            
            for asset_class, exposure in asset_class_exposure.items():
                limit = self.asset_class_limits.get(asset_class, 1.0)
                if exposure > limit:
                    self.logger.warning(f"Asset class limit exceeded for {asset_class.value}: {exposure:.1%}")
            
            # Check max drawdown
            if self.portfolio_metrics['max_drawdown'] > 20.0:  # 20% max drawdown
                self.logger.warning(f"High drawdown detected: {self.portfolio_metrics['max_drawdown']:.1f}%")
            
            # Check daily risk
            if len(self.historical_values) >= 2:
                current_value = self.historical_values[-1]['total_value']
                previous_value = self.historical_values[-2]['total_value']
                daily_change = abs(current_value - previous_value) / previous_value
                
                if daily_change > self.max_portfolio_risk:
                    self.logger.warning(f"Daily risk limit exceeded: {daily_change:.1%}")
                    
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    def _update_correlations(self):
        """Update correlation matrix between positions"""
        try:
            if (self.last_correlation_update and 
                datetime.now() - self.last_correlation_update < timedelta(seconds=self.correlation_update_interval)):
                return
            
            symbols = list(self.positions.keys())
            if len(symbols) < 2:
                return
            
            # Simple correlation calculation using price history
            correlation_matrix = {}
            
            for symbol1 in symbols:
                correlation_matrix[symbol1] = {}
                for symbol2 in symbols:
                    if symbol1 == symbol2:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        # Calculate correlation (simplified)
                        corr = self._calculate_correlation(symbol1, symbol2)
                        correlation_matrix[symbol1][symbol2] = corr
            
            self.correlation_matrix = correlation_matrix
            self.last_correlation_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating correlations: {e}")
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            history1 = self.price_history.get(symbol1, [])
            history2 = self.price_history.get(symbol2, [])
            
            if len(history1) < 30 or len(history2) < 30:
                # Default correlation based on currency pairs
                if symbol1[:3] == symbol2[:3] or symbol1[3:] == symbol2[3:]:
                    return 0.7  # High correlation for same base/quote currency
                else:
                    return 0.3  # Moderate correlation for different pairs
            
            # Calculate returns
            returns1 = []
            returns2 = []
            
            min_length = min(len(history1), len(history2))
            for i in range(1, min_length):
                ret1 = (history1[i]['price'] - history1[i-1]['price']) / history1[i-1]['price']
                ret2 = (history2[i]['price'] - history2[i-1]['price']) / history2[i-1]['price']
                returns1.append(ret1)
                returns2.append(ret2)
            
            if len(returns1) < 10:
                return 0.3
            
            # Calculate correlation coefficient
            n = len(returns1)
            mean1 = sum(returns1) / n
            mean2 = sum(returns2) / n
            
            numerator = sum((returns1[i] - mean1) * (returns2[i] - mean2) for i in range(n))
            
            sum_sq1 = sum((r - mean1) ** 2 for r in returns1)
            sum_sq2 = sum((r - mean2) ** 2 for r in returns2)
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return max(-1.0, min(1.0, correlation))  # Clamp between -1 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
            return 0.0
    
    def _auto_rebalance(self):
        """Automatic portfolio rebalancing thread"""
        self.logger.info("Starting auto-rebalancing thread")
        
        while self.is_monitoring:
            try:
                if (self.auto_rebalance_enabled and
                    (self.last_rebalance_time is None or
                     datetime.now() - self.last_rebalance_time > timedelta(seconds=self.rebalance_frequency))):
                    
                    self.rebalance_portfolio()
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in auto-rebalancing: {e}")
                time.sleep(3600)
    
    def rebalance_portfolio(self) -> Dict:
        """Rebalance portfolio to target allocations"""
        try:
            if not self.positions:
                return {"status": "error", "message": "No positions to rebalance"}
            
            rebalance_needed = []
            total_value = sum(pos.market_value for pos in self.positions.values())
            
            if total_value == 0:
                return {"status": "error", "message": "Zero portfolio value"}
            
            # Check which positions need rebalancing
            for symbol, position in self.positions.items():
                target_weight = self.target_allocations.get(symbol, 0.0)
                current_weight = position.weight
                deviation = abs(current_weight - target_weight)
                
                if deviation > self.rebalance_threshold:
                    target_value = target_weight * total_value
                    current_value = position.market_value
                    adjustment = target_value - current_value
                    
                    rebalance_needed.append({
                        'symbol': symbol,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'deviation': deviation,
                        'adjustment_value': adjustment,
                        'adjustment_quantity': adjustment / position.current_price
                    })
            
            if not rebalance_needed:
                return {"status": "no_rebalance_needed", "message": "Portfolio is within rebalance thresholds"}
            
            # Execute rebalancing (simulation)
            rebalanced_positions = []
            for rebalance_info in rebalance_needed:
                symbol = rebalance_info['symbol']
                adjustment_qty = rebalance_info['adjustment_quantity']
                
                # In a real implementation, this would place orders through the execution engine
                self.logger.info(f"Rebalancing {symbol}: {adjustment_qty:+.4f} units")
                
                # Update position quantity (simulated)
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position.quantity += adjustment_qty
                    position.market_value = position.current_price * abs(position.quantity)
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                
                rebalanced_positions.append(symbol)
            
            # Update weights and record rebalance
            self._update_portfolio_weights()
            
            rebalance_record = {
                'timestamp': datetime.now().isoformat(),
                'positions_rebalanced': len(rebalanced_positions),
                'symbols': rebalanced_positions,
                'total_portfolio_value': total_value,
                'reason': 'scheduled_rebalance'
            }
            
            self.rebalance_history.append(rebalance_record)
            self.last_rebalance_time = datetime.now()
            
            self.logger.info(f"Portfolio rebalanced: {len(rebalanced_positions)} positions adjusted")
            
            return {
                "status": "success",
                "positions_rebalanced": len(rebalanced_positions),
                "symbols_rebalanced": rebalanced_positions,
                "rebalance_details": rebalance_needed
            }
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            return {"status": "error", "message": str(e)}
    
    def set_target_allocations(self, allocations: Dict[str, float]) -> Dict:
        """Set target portfolio allocations"""
        try:
            # Validate allocations
            total_allocation = sum(allocations.values())
            if abs(total_allocation - 1.0) > 0.01:  # Allow 1% tolerance
                return {"status": "error", "message": f"Allocations must sum to 1.0, got {total_allocation}"}
            
            for symbol, allocation in allocations.items():
                if allocation < 0 or allocation > 1:
                    return {"status": "error", "message": f"Invalid allocation for {symbol}: {allocation}"}
            
            self.target_allocations = allocations.copy()
            
            # Trigger rebalance if needed
            if self.positions:
                rebalance_result = self.rebalance_portfolio()
                return {
                    "status": "success",
                    "message": "Target allocations updated",
                    "rebalance_result": rebalance_result
                }
            else:
                return {"status": "success", "message": "Target allocations updated"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            total_market_value = sum(pos.market_value for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_value = self.available_balance + total_market_value
            
            # Asset class breakdown
            asset_class_breakdown = defaultdict(float)
            for position in self.positions.values():
                asset_class_breakdown[position.asset_class.value] += position.weight
            
            # Top positions
            top_positions = sorted(
                self.positions.items(),
                key=lambda x: x[1].market_value,
                reverse=True
            )[:5]
            
            top_positions_data = []
            for symbol, position in top_positions:
                top_positions_data.append({
                    'symbol': symbol,
                    'weight': round(position.weight * 100, 2),
                    'market_value': round(position.market_value, 2),
                    'unrealized_pnl': round(position.unrealized_pnl, 2),
                    'asset_class': position.asset_class.value
                })
            
            return {
                'total_value': round(total_value, 2),
                'available_balance': round(self.available_balance, 2),
                'invested_value': round(total_market_value, 2),
                'unrealized_pnl': round(total_unrealized_pnl, 2),
                'total_return_pct': round(((total_value - self.initial_balance) / self.initial_balance) * 100, 2),
                'positions_count': len(self.positions),
                'asset_class_breakdown': dict(asset_class_breakdown),
                'top_positions': top_positions_data,
                'performance_metrics': self.portfolio_metrics,
                'risk_level': self.risk_level.value,
                'allocation_strategy': self.allocation_strategy.value,
                'last_rebalance': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None,
                'correlation_updated': self.last_correlation_update.isoformat() if self.last_correlation_update else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {"error": str(e)}
    
    def get_position_details(self, symbol: str = None) -> Dict:
        """Get detailed position information"""
        try:
            if symbol:
                if symbol not in self.positions:
                    return {"error": "Position not found"}
                
                position = self.positions[symbol]
                return {
                    'symbol': symbol,
                    'asset_class': position.asset_class.value,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'market_value': round(position.market_value, 2),
                    'unrealized_pnl': round(position.unrealized_pnl, 2),
                    'weight': round(position.weight * 100, 2),
                    'target_weight': round(self.target_allocations.get(symbol, 0) * 100, 2),
                    'last_updated': position.last_updated.isoformat(),
                    'risk_metrics': {
                        'volatility': position.volatility,
                        'beta': position.beta,
                        'var_95': position.var_95
                    }
                }
            else:
                # Return all positions
                positions_data = {}
                for sym, pos in self.positions.items():
                    positions_data[sym] = self.get_position_details(sym)
                
                return positions_data
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_risk_analysis(self) -> Dict:
        """Get portfolio risk analysis"""
        try:
            # Portfolio level risk metrics
            portfolio_var = self.portfolio_metrics.get('var_95', 0)
            portfolio_volatility = self.portfolio_metrics.get('volatility', 0)
            max_drawdown = self.portfolio_metrics.get('max_drawdown', 0)
            
            # Position concentration risk
            position_weights = [pos.weight for pos in self.positions.values()]
            concentration_risk = max(position_weights) if position_weights else 0
            
            # Asset class concentration
            asset_class_exposure = defaultdict(float)
            for position in self.positions.values():
                asset_class_exposure[position.asset_class] += position.weight
            
            max_asset_class_exposure = max(asset_class_exposure.values()) if asset_class_exposure else 0
            
            # Correlation risk (highest correlation)
            max_correlation = 0.0
            if len(self.correlation_matrix) > 1:
                for symbol1, correlations in self.correlation_matrix.items():
                    for symbol2, corr in correlations.items():
                        if symbol1 != symbol2 and abs(corr) > abs(max_correlation):
                            max_correlation = corr
            
            # Risk assessment
            risk_score = 0
            risk_factors = []
            
            if concentration_risk > self.max_position_size:
                risk_score += 20
                risk_factors.append(f"High position concentration: {concentration_risk:.1%}")
            
            if max_asset_class_exposure > 0.7:
                risk_score += 15
                risk_factors.append(f"High asset class concentration: {max_asset_class_exposure:.1%}")
            
            if abs(max_correlation) > 0.8:
                risk_score += 15
                risk_factors.append(f"High correlation risk: {max_correlation:.2f}")
            
            if portfolio_volatility > 25:
                risk_score += 20
                risk_factors.append(f"High portfolio volatility: {portfolio_volatility:.1f}%")
            
            if max_drawdown > 15:
                risk_score += 20
                risk_factors.append(f"High drawdown: {max_drawdown:.1f}%")
            
            if portfolio_var > self.current_balance * 0.05:
                risk_score += 10
                risk_factors.append(f"High VaR: {portfolio_var:.2f}")
            
            # Risk level assessment
            if risk_score >= 70:
                risk_level = "HIGH"
            elif risk_score >= 40:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'risk_level': risk_level,
                'risk_score': min(risk_score, 100),
                'risk_factors': risk_factors,
                'metrics': {
                    'portfolio_volatility': portfolio_volatility,
                    'value_at_risk_95': portfolio_var,
                    'max_drawdown': max_drawdown,
                    'concentration_risk': round(concentration_risk * 100, 2),
                    'max_asset_class_exposure': round(max_asset_class_exposure * 100, 2),
                    'max_correlation': round(max_correlation, 2),
                    'sharpe_ratio': self.portfolio_metrics.get('sharpe_ratio', 0)
                },
                'limits': {
                    'max_position_size': self.max_position_size * 100,
                    'max_portfolio_risk': self.max_portfolio_risk * 100,
                    'max_correlation_exposure': self.max_correlation_exposure * 100
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk analysis: {e}")
            return {"error": str(e)}
    
    def set_risk_parameters(self, risk_level: RiskLevel = None, max_position_size: float = None,
                           max_portfolio_risk: float = None, rebalance_threshold: float = None) -> Dict:
        """Update risk parameters"""
        try:
            if risk_level:
                self.risk_level = risk_level
                
                # Adjust parameters based on risk level
                if risk_level == RiskLevel.CONSERVATIVE:
                    self.max_position_size = 0.05  # 5%
                    self.max_portfolio_risk = 0.01  # 1%
                    self.rebalance_threshold = 0.02  # 2%
                elif risk_level == RiskLevel.MODERATE:
                    self.max_position_size = 0.1  # 10%
                    self.max_portfolio_risk = 0.02  # 2%
                    self.rebalance_threshold = 0.05  # 5%
                elif risk_level == RiskLevel.AGGRESSIVE:
                    self.max_position_size = 0.2  # 20%
                    self.max_portfolio_risk = 0.05  # 5%
                    self.rebalance_threshold = 0.1  # 10%
            
            # Override with specific parameters if provided
            if max_position_size is not None:
                self.max_position_size = max_position_size
            
            if max_portfolio_risk is not None:
                self.max_portfolio_risk = max_portfolio_risk
            
            if rebalance_threshold is not None:
                self.rebalance_threshold = rebalance_threshold
            
            self.logger.info(f"Risk parameters updated: Level={self.risk_level.value}")
            
            return {
                "status": "success",
                "risk_level": self.risk_level.value,
                "max_position_size": self.max_position_size,
                "max_portfolio_risk": self.max_portfolio_risk,
                "rebalance_threshold": self.rebalance_threshold
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_status(self):
        """Get current portfolio manager status"""
        total_value = self.available_balance + sum(pos.market_value for pos in self.positions.values())
        
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'initial_balance': self.initial_balance,
            'current_balance': round(total_value, 2),
            'available_balance': round(self.available_balance, 2),
            'positions_count': len(self.positions),
            'risk_level': self.risk_level.value,
            'allocation_strategy': self.allocation_strategy.value,
            'auto_rebalance_enabled': self.auto_rebalance_enabled,
            'is_monitoring': self.is_monitoring,
            'last_rebalance': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None,
            'performance_summary': {
                'total_return': self.portfolio_metrics.get('total_return', 0),
                'sharpe_ratio': self.portfolio_metrics.get('sharpe_ratio', 0),
                'max_drawdown': self.portfolio_metrics.get('max_drawdown', 0),
                'win_rate': self.portfolio_metrics.get('win_rate', 0)
            }
        }
    
    def shutdown(self):
        """Clean shutdown of portfolio manager"""
        try:
            self.logger.info("Shutting down Portfolio Manager...")
            
            # Stop monitoring
            self.is_monitoring = False
            
            # Wait for threads to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            if self.rebalancing_thread and self.rebalancing_thread.is_alive():
                self.rebalancing_thread.join(timeout=5)
            
            # Save final portfolio state
            final_summary = self.get_portfolio_summary()
            self.logger.info(f"Final portfolio summary: Total Value: {final_summary.get('total_value', 0)}")
            
            # Record final portfolio value
            self._record_portfolio_value()
            
            self.status = "SHUTDOWN"
            self.logger.info("Portfolio Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Agent test
if __name__ == "__main__":
    # Test the portfolio manager
    print("Testing AGENT_09: Portfolio Manager")
    print("=" * 40)
    
    # Create portfolio manager
    portfolio = PortfolioManager(initial_balance=10000.0)
    result = portfolio.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test adding positions
        print("\nTesting position management...")
        
        # Add some positions
        pos1 = portfolio.add_position("EURUSD", AssetClass.FOREX, 100000, 1.0950, 1.0960)
        print(f"Added EURUSD: {pos1}")
        
        pos2 = portfolio.add_position("GBPUSD", AssetClass.FOREX, 75000, 1.2750, 1.2765)
        print(f"Added GBPUSD: {pos2}")
        
        # Wait for monitoring
        time.sleep(2)
        
        # Get portfolio summary
        summary = portfolio.get_portfolio_summary()
        print(f"\nPortfolio Summary: {summary}")
        
        # Test rebalancing
        print("\nTesting rebalancing...")
        rebalance_result = portfolio.rebalance_portfolio()
        print(f"Rebalance result: {rebalance_result}")
        
        # Test risk analysis
        risk_analysis = portfolio.get_risk_analysis()
        print(f"\nRisk Analysis: {risk_analysis}")
        
        # Test status
        status = portfolio.get_status()
        print(f"\nStatus: {status}")
        
        # Shutdown
        print("\nShutting down...")
        portfolio.shutdown()
        
    print("Portfolio Manager test completed")