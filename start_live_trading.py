"""
AGI Trading System - LIVE TRADING ONLY
NO SIMULATION, NO DEMO, NO FALLBACKS - PURE LIVE TRADING
"""

import sys
import os
import time
import threading
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required libraries - NO FALLBACKS
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Import all core agents - LIVE ONLY
from mt5_connector_fixed import MT5ConnectorFixed as MT5WindowsConnector
from core.signal_coordinator import SignalCoordinator
from core.risk_calculator import RiskCalculator
from core.chart_signal_agent import ChartSignalAgent
from ml.neural_signal_brain import NeuralSignalBrain
from data.technical_analyst import TechnicalAnalyst
from data.market_data_manager import MarketDataManager
from execution.trade_execution_engine import TradeExecutionEngine, ExecutionMode
from portfolio.portfolio_manager import PortfolioManager
from analytics.performance_analytics import PerformanceAnalytics
from alerts.alert_system import AlertSystem, AlertType, AlertLevel
from config.configuration_manager import ConfigurationManager

class LiveTradingSystem:
    """LIVE TRADING SYSTEM - NO SIMULATION MODE"""
    
    def __init__(self):
        self.system_name = "AGI Trading System - LIVE"
        self.version = "3.0.0"
        self.start_time = None
        self.agents = {}
        self.system_running = False
        self.main_loop_thread = None
        
        # LIVE TRADING CONFIGURATION ONLY
        self.config = {
            'trading_symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDJPY', 'USDCAD'],
            'primary_symbol': 'EURUSD',
            'risk_per_trade': 0.01,  # 1% risk per trade for live
            'execution_mode': ExecutionMode.LIVE,  # LIVE ONLY
            'auto_trading': True,
            'signal_threshold': 0.75,  # Higher threshold for live
            'max_daily_trades': 10,
            'max_open_positions': 5,
            'stop_loss_multiplier': 1.5,
            'take_profit_multiplier': 2.0
        }
    
    def print_banner(self):
        """Print live trading banner"""
        print("\n" + "="*80)
        print(f"  {self.system_name} v{self.version}")
        print("  LIVE TRADING SYSTEM - REAL MONEY")
        print("="*80)
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Mode: LIVE TRADING ONLY")
        print(f"  Execution: REAL MONEY")
        print(f"  Primary Symbol: {self.config['primary_symbol']}")
        print(f"  Risk per Trade: {self.config['risk_per_trade']*100:.1f}%")
        print(f"  Max Daily Trades: {self.config['max_daily_trades']}")
        print(f"  Signal Threshold: {self.config['signal_threshold']*100:.0f}%")
        print("="*80)
        print("  [WARNING] THIS SYSTEM TRADES WITH REAL MONEY")
        print("  [WARNING] ENSURE YOU UNDERSTAND THE RISKS")
        print("="*80)
    
    def verify_live_prerequisites(self):
        """Verify all prerequisites for live trading"""
        print("\n[LIVE TRADING PREREQUISITE CHECK]")
        print("-" * 60)
        
        prerequisites_met = True
        
        # Check MT5 availability
        try:
            if not mt5.initialize():
                print("[FAIL] MetaTrader 5 not available or not running")
                prerequisites_met = False
            else:
                account_info = mt5.account_info()
                if not account_info:
                    print("[FAIL] Cannot access MT5 account information")
                    prerequisites_met = False
                else:
                    print(f"[OK] MT5 Account: {account_info.login}")
                    print(f"[OK] Server: {account_info.server}")
                    print(f"[OK] Balance: ${account_info.balance:,.2f}")
                    print(f"[OK] Currency: {account_info.currency}")
                    print(f"[OK] Leverage: 1:{account_info.leverage}")
                    
                    # Check if account has sufficient balance
                    if account_info.balance < 100:
                        print("[WARN] Account balance is very low for live trading")
                    
                    # Check if account allows auto trading
                    if not account_info.trade_allowed:
                        print("[FAIL] Auto trading not allowed on this account")
                        prerequisites_met = False
                
        except Exception as e:
            print(f"[FAIL] MT5 Error: {e}")
            prerequisites_met = False
        
        # Check required libraries
        try:
            import numpy
            print("[OK] NumPy available")
        except ImportError:
            print("[FAIL] NumPy not available - required for live trading")
            prerequisites_met = False
        
        try:
            import pandas
            print("[OK] Pandas available")
        except ImportError:
            print("[FAIL] Pandas not available - required for live trading")
            prerequisites_met = False
        
        # Check symbol access
        try:
            for symbol in self.config['trading_symbols']:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    print(f"[OK] Symbol {symbol} accessible")
                else:
                    print(f"[WARN] Symbol {symbol} not accessible")
        except Exception as e:
            print(f"[WARN] Symbol check error: {e}")
        
        print("-" * 60)
        if prerequisites_met:
            print("[SUCCESS] All prerequisites met for live trading")
            return True
        else:
            print("[FAIL] Prerequisites not met - cannot start live trading")
            return False
    
    def initialize_live_system(self):
        """Initialize all system components for LIVE trading"""
        self.print_banner()
        
        # Verify prerequisites first
        if not self.verify_live_prerequisites():
            return False
        
        print("\n[LIVE SYSTEM INITIALIZATION]")
        
        try:
            # Phase 1: Core Infrastructure
            print("\nPhase 1: Live Infrastructure...")
            
            # Configuration Manager
            print("  -> Initializing Configuration Manager...")
            self.agents['config'] = ConfigurationManager()
            config_result = self.agents['config'].initialize()
            if config_result['status'] == 'initialized':
                print("     [OK] Configuration Manager ready")
            else:
                print("     [FAIL] Configuration Manager failed")
                return False
            
            # Alert System
            print("  -> Initializing Alert System...")
            self.agents['alerts'] = AlertSystem()
            alert_result = self.agents['alerts'].initialize()
            if alert_result['status'] == 'initialized':
                print("     [OK] Alert System ready")
            else:
                print("     [FAIL] Alert System failed")
                return False
            
            # Phase 2: Live Market Connection
            print("\nPhase 2: Live Market Connection...")
            
            # MT5 Connector - LIVE ONLY
            print("  -> Connecting to LIVE MetaTrader 5...")
            self.agents['mt5'] = MT5WindowsConnector()
            mt5_result = self.agents['mt5'].initialize()
            if mt5_result['status'] == 'initialized':
                print("     [OK] LIVE MT5 Connection established")
                print(f"       LIVE Account: {mt5_result.get('account', 'Unknown')}")
                print(f"       Server: {mt5_result.get('server', 'Unknown')}")
                print(f"       LIVE Balance: ${mt5_result.get('balance', 0):,.2f}")
                
                # Create live connection alert
                self.agents['alerts'].create_alert(
                    AlertType.CONNECTION,
                    AlertLevel.INFO,
                    "LIVE MT5 Connected",
                    f"Connected to LIVE MT5 Account: {mt5_result.get('account', 'Unknown')}",
                    "LIVE_SYSTEM"
                )
            else:
                print("     [FAIL] LIVE MT5 Connection failed")
                self.agents['alerts'].create_alert(
                    AlertType.CONNECTION,
                    AlertLevel.ERROR,
                    "LIVE MT5 Connection Failed",
                    "Could not connect to LIVE MetaTrader 5",
                    "LIVE_SYSTEM"
                )
                return False
            
            # Market Data Manager - LIVE DATA ONLY
            print("  -> Initializing LIVE Market Data Manager...")
            self.agents['market_data'] = MarketDataManager()
            data_result = self.agents['market_data'].initialize()
            if data_result['status'] == 'initialized':
                print("     [OK] LIVE Market Data Manager ready")
            else:
                print("     [FAIL] LIVE Market Data Manager failed")
                return False
            
            # Phase 3: Analysis Engines - LIVE DATA
            print("\nPhase 3: LIVE Analysis Engines...")
            
            # Technical Analyst - LIVE DATA ONLY
            print("  -> Initializing LIVE Technical Analyst...")
            self.agents['technical'] = TechnicalAnalyst(self.config['primary_symbol'])
            tech_result = self.agents['technical'].initialize()
            if tech_result['status'] == 'initialized':
                print("     [OK] LIVE Technical Analyst ready")
            else:
                print("     [FAIL] LIVE Technical Analyst failed")
                return False
            
            # Neural Signal Brain - LIVE LEARNING
            print("  -> Initializing LIVE AI Neural Network...")
            self.agents['neural'] = NeuralSignalBrain()
            neural_result = self.agents['neural'].initialize()
            if neural_result['status'] == 'initialized':
                print("     [OK] LIVE AI Neural Network ready")
                accuracy = neural_result.get('model_accuracy', neural_result.get('accuracy', 0))
                if accuracy > 1:
                    print(f"       Model Accuracy: {accuracy:.1f}%")
                elif accuracy > 0:
                    print(f"       Model Accuracy: {accuracy*100:.1f}%")
                else:
                    print("       Model Accuracy: Training in progress...")
            else:
                print("     [FAIL] LIVE AI Neural Network failed")
                return False
            
            # Chart Signal Agent - LIVE CHARTS
            print("  -> Initializing LIVE Chart Signal Agent...")
            self.agents['chart'] = ChartSignalAgent(self.config['primary_symbol'], self.agents['mt5'])
            chart_result = self.agents['chart'].initialize()
            if chart_result['status'] == 'initialized':
                print("     [OK] LIVE Chart Signal Agent ready")
            else:
                print("     [FAIL] LIVE Chart Signal Agent failed")
                return False
            
            # Phase 4: LIVE Risk & Execution
            print("\nPhase 4: LIVE Risk & Execution...")
            
            # Risk Calculator - LIVE MONEY
            print("  -> Initializing LIVE Risk Calculator...")
            self.agents['risk'] = RiskCalculator()
            risk_result = self.agents['risk'].initialize()
            if risk_result['status'] == 'initialized':
                print("     [OK] LIVE Risk Calculator ready")
                # Update with live account balance
                live_balance = mt5_result.get('balance', 10000.0)
                self.agents['risk'].update_account_balance(live_balance)
                print(f"       Live Account Balance: ${live_balance:,.2f}")
            else:
                print("     [FAIL] LIVE Risk Calculator failed")
                return False
            
            # Signal Coordinator - LIVE SIGNALS
            print("  -> Initializing LIVE Signal Coordinator...")
            self.agents['coordinator'] = SignalCoordinator(self.agents['mt5'])
            coord_result = self.agents['coordinator'].initialize()
            if coord_result['status'] == 'initialized':
                print("     [OK] LIVE Signal Coordinator ready")
            else:
                print("     [FAIL] LIVE Signal Coordinator failed")
                return False
            
            # Trade Execution Engine - LIVE TRADES
            print("  -> Initializing LIVE Trade Execution Engine...")
            self.agents['execution'] = TradeExecutionEngine()
            exec_result = self.agents['execution'].initialize()
            if exec_result['status'] == 'initialized':
                print(f"     [OK] LIVE Trade Execution Engine ready")
                print(f"       [WARNING] THIS WILL EXECUTE REAL TRADES")
            else:
                print("     [FAIL] LIVE Trade Execution Engine failed")
                return False
            
            # Phase 5: Portfolio & Analytics - LIVE MONEY
            print("\nPhase 5: LIVE Portfolio & Analytics...")
            
            # Portfolio Manager - LIVE PORTFOLIO
            print("  -> Initializing LIVE Portfolio Manager...")
            self.agents['portfolio'] = PortfolioManager()
            port_result = self.agents['portfolio'].initialize()
            if port_result['status'] == 'initialized':
                print("     [OK] LIVE Portfolio Manager ready")
            else:
                print("     [FAIL] LIVE Portfolio Manager failed")
                return False
            
            # Performance Analytics - LIVE PERFORMANCE
            print("  -> Initializing LIVE Performance Analytics...")
            self.agents['analytics'] = PerformanceAnalytics()
            analytics_result = self.agents['analytics'].initialize()
            if analytics_result['status'] == 'initialized':
                print("     [OK] LIVE Performance Analytics ready")
            else:
                print("     [FAIL] LIVE Performance Analytics failed")
                return False
            
            print("\n" + "="*80)
            print("  [SUCCESS] LIVE SYSTEM INITIALIZATION COMPLETE")
            print(f"  [OK] All {len(self.agents)} agents operational")
            print("  [WARNING] READY FOR LIVE TRADING WITH REAL MONEY")
            print("="*80)
            
            # Create system ready alert
            self.agents['alerts'].create_alert(
                AlertType.SYSTEM_ERROR,
                AlertLevel.INFO,
                "LIVE System Online",
                "AGI Trading System is fully operational for LIVE trading with real money",
                "LIVE_SYSTEM"
            )
            
            return True
            
        except Exception as e:
            print(f"\n[FAIL] LIVE SYSTEM INITIALIZATION FAILED: {e}")
            return False
    
    def start_live_trading(self):
        """Start LIVE trading operations"""
        if not self.system_running:
            print("\n[STARTING LIVE TRADING OPERATIONS]")
            print("[WARNING] TRADING WITH REAL MONEY")
            
            # Start live market data streaming
            print("  -> Starting LIVE market data streaming...")
            stream_result = self.agents['market_data'].start_streaming(self.config['trading_symbols'])
            if stream_result.get('status') in ['started', 'already_active']:
                print(f"     [OK] LIVE streaming started for {len(self.config['trading_symbols'])} symbols")
            else:
                print(f"     [WARN] LIVE streaming issue: {stream_result.get('message', 'Unknown')}")
            
            # Start signal coordination
            print("  -> Starting LIVE signal coordination...")
            coord_start = self.agents['coordinator'].start_coordination()
            if coord_start.get('status') == 'started':
                print("     [OK] LIVE signal coordination active")
            
            # Start chart analysis
            print("  -> Starting LIVE chart analysis...")
            analysis_start = self.agents['chart'].start_analysis()
            if analysis_start.get('status') == 'started':
                print("     [OK] LIVE chart analysis active")
            
            # Start technical analysis
            print("  -> Starting LIVE technical analysis...")
            if hasattr(self.agents['technical'], 'start_realtime_analysis'):
                tech_start = self.agents['technical'].start_realtime_analysis()
            elif hasattr(self.agents['technical'], 'start_real_time_analysis'):
                tech_start = self.agents['technical'].start_real_time_analysis()
            else:
                tech_start = {"status": "not_available"}
            
            if tech_start.get('status') == 'started':
                print("     [OK] LIVE technical analysis active")
            else:
                print("     [WARN] LIVE technical analysis method not available")
            
            # Enable LIVE trading
            if self.config['auto_trading']:
                print("  -> Enabling LIVE auto-trading...")
                print("     [WARNING] LIVE AUTO-TRADING ENABLED")
                print("     [WARNING] SYSTEM WILL TRADE WITH REAL MONEY")
                self.agents['execution'].enable_trading()
                print("     [OK] LIVE auto-trading enabled")
            
            self.system_running = True
            self.start_time = time.time()
            
            # Start main loop
            self.main_loop_thread = threading.Thread(target=self._live_trading_loop, daemon=True)
            self.main_loop_thread.start()
            
            print("\n" + "="*80)
            print("  [LAUNCH] LIVE TRADING SYSTEM IS ACTIVE!")
            print("  [ANALYTICS] Monitoring LIVE market conditions...")
            print("  [AI] AI analysis processing LIVE data...")
            print("  [WARNING] LIVE AUTO-TRADING ENABLED")
            print("  [WARNING] TRADING WITH REAL MONEY")
            print("="*80)
            
            return True
        
        return False
    
    def _live_trading_loop(self):
        """Main LIVE trading loop - trades with real money"""
        loop_counter = 0
        
        while self.system_running:
            try:
                loop_counter += 1
                
                # Every 10 seconds, check system health
                if loop_counter % 10 == 0:
                    self._live_health_check()
                
                # Every 30 seconds, process signals
                if loop_counter % 30 == 0:
                    self._process_live_trading_signals()
                
                # Every 2 minutes, update analytics
                if loop_counter % 120 == 0:
                    self._update_live_analytics()
                
                # Every 5 minutes, check rebalancing
                if loop_counter % 300 == 0:
                    self._check_live_rebalancing()
                
                time.sleep(1)  # 1 second loop
                
            except Exception as e:
                print(f"LIVE trading loop error: {e}")
                self.agents['alerts'].create_alert(
                    AlertType.SYSTEM_ERROR,
                    AlertLevel.ERROR,
                    "LIVE Trading Loop Error",
                    f"Error in LIVE trading loop: {str(e)}",
                    "LIVE_SYSTEM"
                )
    
    def _live_health_check(self):
        """Check LIVE system health"""
        try:
            # Check MT5 connection
            mt5_status = self.agents['mt5'].get_status()
            if mt5_status['status'] != 'CONNECTED':
                print("[WARN] LIVE MT5 connection lost - attempting reconnect...")
                self.agents['mt5'].initialize()
            
            # Check account balance
            account_info = self.agents['mt5'].get_account_info()
            if account_info.get('status') == 'success':
                balance = account_info.get('balance', 0)
                equity = account_info.get('equity', 0)
                margin_level = account_info.get('margin_level', 0)
                
                # Alert if margin level is low
                if margin_level > 0 and margin_level < 200:
                    self.agents['alerts'].create_alert(
                        AlertType.PORTFOLIO,
                        AlertLevel.WARNING,
                        "Low Margin Level",
                        f"Margin level: {margin_level:.1f}%",
                        "LIVE_SYSTEM"
                    )
                
                print(f"[LIVE] Balance: ${balance:,.2f} | Equity: ${equity:,.2f} | Margin: {margin_level:.1f}%")
        
        except Exception as e:
            print(f"LIVE health check error: {e}")
    
    def _process_live_trading_signals(self):
        """Process LIVE trading signals - TRADES WITH REAL MONEY"""
        try:
            if not self.config['auto_trading']:
                return
            
            # Get current analysis
            tech_analysis = self.agents['technical'].get_current_analysis()
            
            if tech_analysis and len(tech_analysis.get('signals', [])) > 0:
                signals = tech_analysis['signals']
                strongest_signal = max(signals, key=lambda x: x.get('strength', 0))
                
                if strongest_signal.get('strength', 0) >= self.config['signal_threshold'] * 100:
                    print(f"[LIVE SIGNAL] {strongest_signal.get('action')} - {strongest_signal.get('strength', 0):.1f}%")
                    
                    # Calculate risk for LIVE trade
                    current_price = self.agents['mt5'].get_current_price(self.config['primary_symbol'])
                    if current_price.get('status') == 'success':
                        
                        risk_calc = self.agents['risk'].calculate_position_size(
                            symbol=self.config['primary_symbol'],
                            direction=strongest_signal.get('action'),
                            entry_price=current_price.get('bid', 0),
                            stop_loss=current_price.get('bid', 0) * (1 - self.config['stop_loss_multiplier']/100),
                            take_profit=current_price.get('bid', 0) * (1 + self.config['take_profit_multiplier']/100)
                        )
                        
                        if risk_calc.get('status') == 'success':
                            # Create LIVE trading signal
                            live_signal = {
                                'symbol': self.config['primary_symbol'],
                                'direction': strongest_signal.get('action'),
                                'volume': risk_calc.get('position_size', 0.01),
                                'entry_price': current_price.get('bid', 0),
                                'stop_loss': current_price.get('bid', 0) * (1 - self.config['stop_loss_multiplier']/100),
                                'take_profit': current_price.get('bid', 0) * (1 + self.config['take_profit_multiplier']/100),
                                'source': 'LIVE_TECHNICAL_ANALYSIS'
                            }
                            
                            # Execute LIVE trade
                            execution_result = self.agents['execution'].execute_signal(live_signal)
                            
                            if execution_result.get('status') == 'success':
                                print(f"[LIVE TRADE] EXECUTED: {execution_result.get('order_id')}")
                                
                                self.agents['alerts'].create_alert(
                                    AlertType.TRADE,
                                    AlertLevel.INFO,
                                    "LIVE Trade Executed",
                                    f"LIVE {strongest_signal.get('action')} trade on {self.config['primary_symbol']}",
                                    "LIVE_TRADER"
                                )
                            else:
                                print(f"[ERROR] LIVE trade execution failed: {execution_result.get('message')}")
        
        except Exception as e:
            print(f"LIVE signal processing error: {e}")
    
    def _update_live_analytics(self):
        """Update LIVE performance analytics"""
        try:
            analysis = self.agents['analytics'].analyze_performance()
            if analysis.get('status') == 'success':
                print(f"[LIVE ANALYTICS] Return: {analysis.get('total_return', 0):.2f}% | Sharpe: {analysis.get('sharpe_ratio', 0):.2f}")
        
        except Exception as e:
            print(f"LIVE analytics update error: {e}")
    
    def _check_live_rebalancing(self):
        """Check if LIVE portfolio rebalancing is needed"""
        try:
            portfolio_summary = self.agents['portfolio'].get_portfolio_summary()
            
            # Rebalancing check for LIVE trading
            for position in portfolio_summary.get('positions', []):
                if position.get('weight', 0) > 0.4:
                    print("[REBALANCE] LIVE portfolio rebalancing recommended")
                    rebalance_result = self.agents['portfolio'].rebalance_portfolio()
                    if rebalance_result.get('status') == 'success':
                        print("[SUCCESS] LIVE portfolio rebalanced")
                    break
        
        except Exception as e:
            print(f"LIVE rebalancing check error: {e}")
    
    def stop_live_trading(self):
        """Stop LIVE trading operations"""
        print("\n[STOPPING LIVE TRADING OPERATIONS]")
        
        self.system_running = False
        
        # Stop all analysis
        if 'chart' in self.agents:
            self.agents['chart'].stop_analysis()
        if 'technical' in self.agents:
            if hasattr(self.agents['technical'], 'stop_realtime_analysis'):
                self.agents['technical'].stop_realtime_analysis()
            elif hasattr(self.agents['technical'], 'stop_real_time_analysis'):
                self.agents['technical'].stop_real_time_analysis()
        if 'coordinator' in self.agents:
            self.agents['coordinator'].stop_coordination()
        
        # Disable trading
        if 'execution' in self.agents:
            self.agents['execution'].disable_trading()
        
        print("  [OK] LIVE trading operations stopped")
    
    def shutdown_live_system(self):
        """Shutdown entire LIVE system"""
        print("\n[LIVE SYSTEM SHUTDOWN]")
        
        self.stop_live_trading()
        
        # Shutdown all agents
        shutdown_order = [
            'coordinator', 'execution', 'chart', 'market_data',
            'portfolio', 'analytics', 'neural', 'technical', 'risk',
            'alerts', 'config', 'mt5'
        ]
        
        for agent_name in shutdown_order:
            if agent_name in self.agents:
                try:
                    print(f"  -> Shutting down {agent_name}...")
                    self.agents[agent_name].shutdown()
                    print(f"     [OK] {agent_name} shutdown complete")
                except Exception as e:
                    print(f"     [FAIL] {agent_name} shutdown error: {e}")
        
        if self.start_time:
            runtime = time.time() - self.start_time
            print(f"\n  LIVE system runtime: {runtime/3600:.2f} hours")
        
        print("\n" + "="*80)
        print("  [SUCCESS] LIVE SYSTEM SHUTDOWN COMPLETE")
        print("="*80)

def main():
    """Main LIVE trading entry point"""
    live_system = LiveTradingSystem()
    
    try:
        # Initialize LIVE system
        if live_system.initialize_live_system():
            # Confirm LIVE trading
            print("\n" + "="*80)
            print("  [FINAL WARNING] THIS SYSTEM TRADES WITH REAL MONEY")
            print("  [CONFIRMATION] Type 'START LIVE TRADING' to begin:")
            print("="*80)
            
            user_input = input("\nConfirmation> ").strip()
            
            if user_input == "START LIVE TRADING":
                # Start LIVE trading
                if live_system.start_live_trading():
                    print("\nLIVE TRADING SYSTEM IS ACTIVE. Commands:")
                    print("  'status' - Show system status")
                    print("  'balance' - Show account balance")
                    print("  'positions' - Show open positions")
                    print("  'stop' - Stop trading")
                    print("  'shutdown' - Shutdown system")
                    print("  'help' - Show this help")
                    
                    # Interactive command loop
                    while live_system.system_running:
                        try:
                            cmd = input("\nLIVE> ").strip().lower()
                            
                            if cmd == 'shutdown' or cmd == 'exit':
                                break
                            elif cmd == 'stop':
                                live_system.stop_live_trading()
                            elif cmd == 'status':
                                status = live_system.agents['mt5'].get_status()
                                print(f"\nLIVE System Status: {status.get('status', 'UNKNOWN')}")
                            elif cmd == 'balance':
                                account = live_system.agents['mt5'].get_account_info()
                                if account.get('status') == 'success':
                                    print(f"\nLIVE Balance: ${account.get('balance', 0):,.2f}")
                                    print(f"LIVE Equity: ${account.get('equity', 0):,.2f}")
                            elif cmd == 'positions':
                                positions = live_system.agents['mt5'].get_positions()
                                if positions.get('status') == 'success':
                                    pos_list = positions.get('positions', [])
                                    print(f"\nOpen Positions: {len(pos_list)}")
                                    for pos in pos_list[:5]:  # Show first 5
                                        print(f"  {pos.get('symbol', 'Unknown')}: {pos.get('profit', 0):.2f}")
                            elif cmd == 'help':
                                print("\nLIVE Trading Commands:")
                                print("  status - System status")
                                print("  balance - Account balance")
                                print("  positions - Open positions")
                                print("  stop - Stop trading")
                                print("  shutdown - Shutdown system")
                            elif cmd == '':
                                continue
                            else:
                                print(f"Unknown command: {cmd}. Type 'help' for commands.")
                        
                        except KeyboardInterrupt:
                            print("\nInterrupt received. Shutting down LIVE system...")
                            break
                        except EOFError:
                            print("\nEOF received. Shutting down LIVE system...")
                            break
                else:
                    print("Failed to start LIVE trading operations")
            else:
                print("LIVE trading NOT started. User confirmation not received.")
        else:
            print("Failed to initialize LIVE system")
    
    except KeyboardInterrupt:
        print("\nInterrupt received during startup. Shutting down...")
    
    finally:
        # Always shutdown gracefully
        live_system.shutdown_live_system()

if __name__ == "__main__":
    main()