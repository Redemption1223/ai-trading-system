"""
AGI Trading System - Main Startup Script
Initializes and starts the complete trading system
"""

import sys
import os
import time
import threading
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all core agents - LIVE TRADING ONLY
from mt5_connector_live import MT5LiveConnector
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

class AGITradingSystem:
    """Main AGI Trading System Controller"""
    
    def __init__(self):
        self.system_name = "AGI Trading System"
        self.version = "2.0.0"
        self.start_time = None
        self.agents = {}
        self.system_running = False
        self.main_loop_thread = None
        
        # System configuration - LIVE TRADING ONLY
        self.config = {
            'trading_symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
            'primary_symbol': 'EURUSD',
            'risk_per_trade': 0.02,  # 2% risk per trade
            'execution_mode': ExecutionMode.LIVE,  # LIVE TRADING MODE
            'auto_trading': True,
            'signal_threshold': 0.7
        }
    
    def print_banner(self):
        """Print system banner"""
        print("\n" + "="*80)
        print(f"  {self.system_name} v{self.version}")
        print("  Advanced AI-Powered Trading Platform")
        print("="*80)
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Mode: {'AUTO TRADING' if self.config['auto_trading'] else 'MANUAL TRADING'}")
        print(f"  Execution: {self.config['execution_mode'].value}")
        print(f"  Primary Symbol: {self.config['primary_symbol']}")
        print(f"  Risk per Trade: {self.config['risk_per_trade']*100:.1f}%")
        print("="*80)
    
    def initialize_system(self):
        """Initialize all system components"""
        self.print_banner()
        print("\n[SYSTEM INITIALIZATION]")
        
        try:
            # Phase 1: Core Infrastructure
            print("\nPhase 1: Core Infrastructure...")
            
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
            
            # Phase 2: Market Connection
            print("\nPhase 2: Market Connection...")
            
            # MT5 Live Connector
            print("  -> Connecting to LIVE MetaTrader 5...")
            self.agents['mt5'] = MT5LiveConnector()
            mt5_result = self.agents['mt5'].initialize()
            if mt5_result['status'] == 'initialized':
                print("     [OK] MT5 Connection established")
                print(f"       Account: {mt5_result.get('account', 'Unknown')}")
                print(f"       Server: {mt5_result.get('server', 'Unknown')}")
                print(f"       Balance: ${mt5_result.get('balance', 0):,.2f}")
                # Create success alert
                self.agents['alerts'].create_alert(
                    AlertType.CONNECTION,
                    AlertLevel.INFO,
                    "MT5 Connected",
                    f"Connected to MT5 Account: {mt5_result.get('account', 'Unknown')}",
                    "SYSTEM"
                )
            else:
                print("     [FAIL] MT5 Connection failed - Check MT5 is running and allow DLL imports")
                self.agents['alerts'].create_alert(
                    AlertType.CONNECTION,
                    AlertLevel.ERROR,
                    "MT5 Connection Failed",
                    "Could not connect to MetaTrader 5. Check MT5 is running.",
                    "SYSTEM"
                )
                return False
            
            # Market Data Manager
            print("  -> Initializing Market Data Manager...")
            self.agents['market_data'] = MarketDataManager()
            data_result = self.agents['market_data'].initialize()
            if data_result['status'] == 'initialized':
                print("     [OK] Market Data Manager ready")
            else:
                print("     [FAIL] Market Data Manager failed")
                return False
            
            # Phase 3: Analysis Engines
            print("\nPhase 3: Analysis Engines...")
            
            # Technical Analyst
            print("  -> Initializing Technical Analyst...")
            self.agents['technical'] = TechnicalAnalyst(self.config['primary_symbol'])
            tech_result = self.agents['technical'].initialize()
            if tech_result['status'] == 'initialized':
                print("     [OK] Technical Analyst ready")
            else:
                print("     [FAIL] Technical Analyst failed")
                return False
            
            # Neural Signal Brain
            print("  -> Initializing AI Neural Network...")
            self.agents['neural'] = NeuralSignalBrain()
            neural_result = self.agents['neural'].initialize()
            if neural_result['status'] == 'initialized':
                print("     [OK] AI Neural Network ready")
                # Get accuracy from result or status
                accuracy = neural_result.get('model_accuracy', neural_result.get('accuracy', 0))
                if accuracy > 1:  # If it's a percentage already
                    print(f"       Model Accuracy: {accuracy:.1f}%")
                elif accuracy > 0:  # If it's a decimal
                    print(f"       Model Accuracy: {accuracy*100:.1f}%")
                else:
                    print("       Model Accuracy: Training in progress...")
            else:
                print("     [FAIL] AI Neural Network failed")
                return False
            
            # Chart Signal Agent
            print("  -> Initializing Chart Signal Agent...")
            self.agents['chart'] = ChartSignalAgent(self.config['primary_symbol'], self.agents['mt5'])
            chart_result = self.agents['chart'].initialize()
            if chart_result['status'] == 'initialized':
                print("     [OK] Chart Signal Agent ready")
            else:
                print("     [FAIL] Chart Signal Agent failed")
                return False
            
            # Phase 4: Risk & Execution
            print("\nPhase 4: Risk & Execution...")
            
            # Risk Calculator
            print("  -> Initializing Risk Calculator...")
            self.agents['risk'] = RiskCalculator()
            risk_result = self.agents['risk'].initialize()
            if risk_result['status'] == 'initialized':
                print("     [OK] Risk Calculator ready")
                # Update risk parameters
                self.agents['risk'].update_account_balance(10000.0)  # Set initial balance
            else:
                print("     [FAIL] Risk Calculator failed")
                return False
            
            # Signal Coordinator
            print("  -> Initializing Signal Coordinator...")
            self.agents['coordinator'] = SignalCoordinator(self.agents['mt5'])
            coord_result = self.agents['coordinator'].initialize()
            if coord_result['status'] == 'initialized':
                print("     [OK] Signal Coordinator ready")
            else:
                print("     [FAIL] Signal Coordinator failed")
                return False
            
            # Trade Execution Engine
            print("  -> Initializing Trade Execution Engine...")
            self.agents['execution'] = TradeExecutionEngine()
            exec_result = self.agents['execution'].initialize()
            if exec_result['status'] == 'initialized':
                print(f"     [OK] Trade Execution Engine ready ({self.config['execution_mode'].value} mode)")
            else:
                print("     [FAIL] Trade Execution Engine failed")
                return False
            
            # Phase 5: Portfolio & Analytics
            print("\nPhase 5: Portfolio & Analytics...")
            
            # Portfolio Manager
            print("  -> Initializing Portfolio Manager...")
            self.agents['portfolio'] = PortfolioManager()
            port_result = self.agents['portfolio'].initialize()
            if port_result['status'] == 'initialized':
                print("     [OK] Portfolio Manager ready")
            else:
                print("     [FAIL] Portfolio Manager failed")
                return False
            
            # Performance Analytics
            print("  -> Initializing Performance Analytics...")
            self.agents['analytics'] = PerformanceAnalytics()
            analytics_result = self.agents['analytics'].initialize()
            if analytics_result['status'] == 'initialized':
                print("     [OK] Performance Analytics ready")
            else:
                print("     [FAIL] Performance Analytics failed")
                return False
            
            print("\n" + "="*80)
            print("  [OK] SYSTEM INITIALIZATION COMPLETE")
            print(f"  [OK] All {len(self.agents)} agents operational")
            print("  [OK] Ready for trading operations")
            print("="*80)
            
            # Create system ready alert
            self.agents['alerts'].create_alert(
                AlertType.SYSTEM_ERROR,
                AlertLevel.INFO,
                "System Online",
                "AGI Trading System is fully operational and ready for trading",
                "SYSTEM"
            )
            
            return True
            
        except Exception as e:
            print(f"\n[FAIL] SYSTEM INITIALIZATION FAILED: {e}")
            return False
    
    def start_trading(self):
        """Start the main trading loop"""
        if not self.system_running:
            print("\n[STARTING TRADING OPERATIONS]")
            
            # Start market data streaming
            print("  -> Starting market data streaming...")
            # Pass the entire list of symbols, not individual symbols
            stream_result = self.agents['market_data'].start_streaming(self.config['trading_symbols'])
            if stream_result.get('status') in ['started', 'already_active']:
                print(f"     [OK] Streaming started for {len(self.config['trading_symbols'])} symbols")
            else:
                print(f"     [WARN] Streaming issue: {stream_result.get('message', 'Unknown')}")
            
            # Start signal coordination
            print("  -> Starting signal coordination...")
            coord_start = self.agents['coordinator'].start_coordination()
            if coord_start.get('status') == 'started':
                print("     [OK] Signal coordination active")
            
            # Start chart analysis
            print("  -> Starting chart analysis...")
            analysis_start = self.agents['chart'].start_analysis()
            if analysis_start.get('status') == 'started':
                print("     [OK] Chart analysis active")
            
            # Start technical analysis
            print("  -> Starting technical analysis...")
            # Try both method names for compatibility
            if hasattr(self.agents['technical'], 'start_realtime_analysis'):
                tech_start = self.agents['technical'].start_realtime_analysis()
            elif hasattr(self.agents['technical'], 'start_real_time_analysis'):
                tech_start = self.agents['technical'].start_real_time_analysis()
            else:
                tech_start = {"status": "not_available"}
            
            if tech_start.get('status') == 'started':
                print("     [OK] Technical analysis active")
            else:
                print("     [WARN] Technical analysis method not available")
            
            # Enable trading
            if self.config['auto_trading']:
                print("  -> Enabling auto-trading...")
                self.agents['execution'].enable_trading()
                print("     [OK] Auto-trading enabled")
            
            self.system_running = True
            self.start_time = time.time()
            
            # Start main loop
            self.main_loop_thread = threading.Thread(target=self._main_trading_loop, daemon=True)
            self.main_loop_thread.start()
            
            print("\n" + "="*80)
            print("  [LAUNCH] TRADING SYSTEM IS LIVE!")
            print("  [ANALYTICS] Monitoring market conditions...")
            print("  [AI] AI analysis in progress...")
            if self.config['auto_trading']:
                print("  [FAST] Auto-trading ENABLED")
            else:
                print("  [MANUAL] Manual trading mode")
            print("="*80)
            
            return True
        
        return False
    
    def _main_trading_loop(self):
        """Main trading loop - runs in separate thread"""
        loop_counter = 0
        
        while self.system_running:
            try:
                loop_counter += 1
                
                # Every 30 seconds, check system status
                if loop_counter % 30 == 0:
                    self._system_health_check()
                
                # Every 60 seconds, process signals
                if loop_counter % 60 == 0:
                    self._process_trading_signals()
                
                # Every 5 minutes, update analytics
                if loop_counter % 300 == 0:
                    self._update_analytics()
                
                # Every 10 minutes, rebalance if needed
                if loop_counter % 600 == 0:
                    self._check_rebalancing()
                
                time.sleep(1)  # 1 second loop
                
            except Exception as e:
                print(f"Main loop error: {e}")
                self.agents['alerts'].create_alert(
                    AlertType.SYSTEM_ERROR,
                    AlertLevel.ERROR,
                    "Main Loop Error",
                    f"Error in main trading loop: {str(e)}",
                    "SYSTEM"
                )
    
    def _system_health_check(self):
        """Check system health"""
        try:
            # Check MT5 connection
            mt5_status = self.agents['mt5'].get_status()
            if mt5_status['status'] != 'CONNECTED':
                print("[WARN]  MT5 connection lost - attempting reconnect...")
                self.agents['mt5'].initialize()
            
            # Check if we have recent data
            current_price = self.agents['mt5'].get_current_price(self.config['primary_symbol'])
            if current_price.get('status') == 'success':
                print(f"[UP] {self.config['primary_symbol']}: {current_price.get('bid', 0):.5f}")
        
        except Exception as e:
            print(f"Health check error: {e}")
    
    def _process_trading_signals(self):
        """Process trading signals"""
        try:
            if not self.config['auto_trading']:
                return
            
            # Get current market analysis
            tech_analysis = self.agents['technical'].get_current_analysis()
            
            if tech_analysis and len(tech_analysis.get('signals', [])) > 0:
                # Get the strongest signal
                signals = tech_analysis['signals']
                strongest_signal = max(signals, key=lambda x: x.get('strength', 0))
                
                if strongest_signal.get('strength', 0) >= self.config['signal_threshold'] * 100:
                    print(f"[TARGET] Strong signal detected: {strongest_signal.get('action')} - {strongest_signal.get('strength', 0):.1f}%")
                    
                    # Calculate risk
                    current_price = self.agents['mt5'].get_current_price(self.config['primary_symbol'])
                    if current_price.get('status') == 'success':
                        
                        risk_calc = self.agents['risk'].calculate_position_size(
                            symbol=self.config['primary_symbol'],
                            direction=strongest_signal.get('action'),
                            entry_price=current_price.get('bid', 0),
                            stop_loss=current_price.get('bid', 0) * 0.995,  # 0.5% SL
                            take_profit=current_price.get('bid', 0) * 1.01   # 1% TP
                        )
                        
                        if risk_calc.get('status') == 'success':
                            # Create trading signal
                            trading_signal = {
                                'symbol': self.config['primary_symbol'],
                                'direction': strongest_signal.get('action'),
                                'volume': risk_calc.get('position_size', 0.01),
                                'entry_price': current_price.get('bid', 0),
                                'stop_loss': current_price.get('bid', 0) * 0.995,
                                'take_profit': current_price.get('bid', 0) * 1.01,
                                'source': 'TECHNICAL_ANALYSIS'
                            }
                            
                            # Execute trade
                            execution_result = self.agents['execution'].execute_signal(trading_signal)
                            
                            if execution_result.get('status') == 'success':
                                print(f"[SUCCESS] Trade executed: {execution_result.get('order_id')}")
                                
                                self.agents['alerts'].create_alert(
                                    AlertType.TRADE,
                                    AlertLevel.INFO,
                                    "Trade Executed",
                                    f"Executed {strongest_signal.get('action')} trade on {self.config['primary_symbol']}",
                                    "AUTO_TRADER"
                                )
                            else:
                                print(f"[ERROR] Trade execution failed: {execution_result.get('message')}")
        
        except Exception as e:
            print(f"Signal processing error: {e}")
    
    def _update_analytics(self):
        """Update performance analytics"""
        try:
            analysis = self.agents['analytics'].analyze_performance()
            if analysis.get('status') == 'success':
                print(f"[ANALYTICS] Portfolio: {analysis.get('total_return', 0):.2f}% return, Sharpe: {analysis.get('sharpe_ratio', 0):.2f}")
        
        except Exception as e:
            print(f"Analytics update error: {e}")
    
    def _check_rebalancing(self):
        """Check if portfolio rebalancing is needed"""
        try:
            portfolio_summary = self.agents['portfolio'].get_portfolio_summary()
            
            # Simple rebalancing check - if any position > 40% of portfolio
            for position in portfolio_summary.get('positions', []):
                if position.get('weight', 0) > 0.4:
                    print("[REBALANCE]  Portfolio rebalancing recommended")
                    rebalance_result = self.agents['portfolio'].rebalance_portfolio()
                    if rebalance_result.get('status') == 'success':
                        print("[SUCCESS] Portfolio rebalanced")
                    break
        
        except Exception as e:
            print(f"Rebalancing check error: {e}")
    
    def stop_trading(self):
        """Stop trading operations"""
        print("\n[STOPPING TRADING OPERATIONS]")
        
        self.system_running = False
        
        # Stop analysis
        if 'chart' in self.agents:
            self.agents['chart'].stop_analysis()
        if 'technical' in self.agents:
            # Try both method names for compatibility
            if hasattr(self.agents['technical'], 'stop_realtime_analysis'):
                self.agents['technical'].stop_realtime_analysis()
            elif hasattr(self.agents['technical'], 'stop_real_time_analysis'):
                self.agents['technical'].stop_real_time_analysis()
        if 'coordinator' in self.agents:
            self.agents['coordinator'].stop_coordination()
        
        # Disable trading
        if 'execution' in self.agents:
            self.agents['execution'].disable_trading()
        
        print("  [OK] Trading operations stopped")
    
    def shutdown_system(self):
        """Shutdown entire system"""
        print("\n[SYSTEM SHUTDOWN]")
        
        self.stop_trading()
        
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
            print(f"\n  System runtime: {runtime/3600:.2f} hours")
        
        print("\n" + "="*80)
        print("  [RED] SYSTEM SHUTDOWN COMPLETE")
        print("="*80)
    
    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            'system_running': self.system_running,
            'runtime': time.time() - self.start_time if self.start_time else 0,
            'agents_status': {}
        }
        
        for name, agent in self.agents.items():
            try:
                status['agents_status'][name] = agent.get_status()
            except:
                status['agents_status'][name] = {'status': 'ERROR'}
        
        return status

def main():
    """Main entry point"""
    trading_system = AGITradingSystem()
    
    try:
        # Initialize system
        if trading_system.initialize_system():
            # Start trading
            if trading_system.start_trading():
                print("\nSystem is running. Commands:")
                print("  'status' - Show system status")
                print("  'stop' - Stop trading")
                print("  'quit' - Shutdown system")
                print("  'help' - Show this help")
                
                # Interactive command loop
                while trading_system.system_running:
                    try:
                        cmd = input("\nAGI> ").strip().lower()
                        
                        if cmd == 'quit' or cmd == 'exit':
                            break
                        elif cmd == 'stop':
                            trading_system.stop_trading()
                        elif cmd == 'status':
                            status = trading_system.get_system_status()
                            print(f"\nSystem Status: {'RUNNING' if status['system_running'] else 'STOPPED'}")
                            print(f"Runtime: {status['runtime']/3600:.2f} hours")
                            print("Agent Status:")
                            for name, agent_status in status['agents_status'].items():
                                print(f"  {name}: {agent_status.get('status', 'UNKNOWN')}")
                        elif cmd == 'help':
                            print("\nAvailable commands:")
                            print("  status - Show system status")
                            print("  stop - Stop trading operations")
                            print("  quit - Shutdown entire system")
                        elif cmd == '':
                            continue
                        else:
                            print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                    
                    except KeyboardInterrupt:
                        print("\nInterrupt received. Shutting down...")
                        break
                    except EOFError:
                        print("\nEOF received. Shutting down...")
                        break
            else:
                print("Failed to start trading operations")
        else:
            print("Failed to initialize system")
    
    except KeyboardInterrupt:
        print("\nInterrupt received during startup. Shutting down...")
    
    finally:
        # Always shutdown gracefully
        trading_system.shutdown_system()

if __name__ == "__main__":
    main()