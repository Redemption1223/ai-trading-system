"""
AGI Trading System - Real-time Dashboard
Shows live trading information, system status, and performance metrics
"""

import sys
import os
import time
import threading
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import system components
from mt5_connector_fixed import MT5ConnectorFixed
from core.signal_coordinator import SignalCoordinator
from core.risk_calculator import RiskCalculator
from data.technical_analyst import TechnicalAnalyst
from ml.neural_signal_brain import NeuralSignalBrain
from portfolio.portfolio_manager import PortfolioManager
from analytics.performance_analytics import PerformanceAnalytics

class TradingDashboard:
    """Real-time trading dashboard for monitoring system status"""
    
    def __init__(self):
        self.running = False
        self.agents = {}
        self.dashboard_data = {}
        self.update_thread = None
        
        # Initialize core agents for monitoring
        self.initialize_monitoring_agents()
    
    def initialize_monitoring_agents(self):
        """Initialize agents for monitoring purposes"""
        print("Initializing dashboard monitoring agents...")
        
        try:
            # MT5 Connector for real account data
            self.agents['mt5'] = MT5ConnectorFixed()
            mt5_result = self.agents['mt5'].initialize()
            
            if mt5_result['status'] == 'initialized':
                print("✓ MT5 Connection established")
                
                # Get real account information
                account_info = self.agents['mt5'].get_account_info()
                if account_info['status'] == 'success':
                    self.dashboard_data['account'] = account_info
                    print(f"✓ Account: {account_info.get('login', 'Unknown')}")
                    print(f"✓ Server: {account_info.get('server', 'Unknown')}")
                    print(f"✓ Balance: ${account_info.get('balance', 0):,.2f}")
                    print(f"✓ Equity: ${account_info.get('equity', 0):,.2f}")
                    print(f"✓ Margin: ${account_info.get('margin', 0):,.2f}")
            else:
                print("! MT5 Connection failed - Dashboard will run in demo mode")
                self.dashboard_data['account'] = {
                    'status': 'demo',
                    'balance': 10000.0,
                    'equity': 10000.0,
                    'margin': 0.0
                }
            
            # Technical Analyst for market analysis
            self.agents['technical'] = TechnicalAnalyst('EURUSD')
            self.agents['technical'].initialize()
            
            # Neural Brain for AI insights
            self.agents['neural'] = NeuralSignalBrain()
            self.agents['neural'].initialize()
            
            # Portfolio Manager for portfolio tracking
            self.agents['portfolio'] = PortfolioManager()
            self.agents['portfolio'].initialize()
            
            # Performance Analytics
            self.agents['analytics'] = PerformanceAnalytics()
            self.agents['analytics'].initialize()
            
            print("✓ All monitoring agents initialized")
            
        except Exception as e:
            print(f"Error initializing agents: {e}")
    
    def start_dashboard(self):
        """Start the real-time dashboard"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        print("\n" + "="*80)
        print("AGI TRADING SYSTEM - REAL-TIME DASHBOARD")
        print("="*80)
        
        self._run_interactive_dashboard()
    
    def _update_loop(self):
        """Background update loop for real-time data"""
        while self.running:
            try:
                # Update market data
                if 'mt5' in self.agents:
                    current_prices = {}
                    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
                    
                    for symbol in symbols:
                        price = self.agents['mt5'].get_current_price(symbol)
                        if price['status'] == 'success':
                            current_prices[symbol] = {
                                'bid': price.get('bid', 0),
                                'ask': price.get('ask', 0),
                                'spread': price.get('spread', 0),
                                'timestamp': datetime.now().isoformat()
                            }
                    
                    self.dashboard_data['prices'] = current_prices
                
                # Update technical analysis
                if 'technical' in self.agents:
                    analysis = self.agents['technical'].get_current_analysis()
                    if analysis:
                        self.dashboard_data['technical'] = {
                            'signals': len(analysis.get('signals', [])),
                            'indicators': len(analysis.get('indicators', {})),
                            'trend': analysis.get('trend_direction', 'UNKNOWN'),
                            'strength': analysis.get('trend_strength', 0)
                        }
                
                # Update AI neural network status
                if 'neural' in self.agents:
                    neural_status = self.agents['neural'].get_status()
                    self.dashboard_data['neural'] = {
                        'accuracy': neural_status.get('model_accuracy', 0),
                        'predictions': neural_status.get('predictions_made', 0),
                        'training_samples': neural_status.get('training_samples', 0),
                        'is_trained': neural_status.get('is_trained', False)
                    }
                
                # Update portfolio information
                if 'portfolio' in self.agents:
                    portfolio = self.agents['portfolio'].get_portfolio_summary()
                    self.dashboard_data['portfolio'] = {
                        'total_value': portfolio.get('total_value', 0),
                        'positions': len(portfolio.get('positions', [])),
                        'unrealized_pnl': portfolio.get('unrealized_pnl', 0),
                        'risk_level': portfolio.get('risk_level', 'MODERATE')
                    }
                
                # Update performance metrics
                if 'analytics' in self.agents:
                    performance = self.agents['analytics'].analyze_performance()
                    if performance.get('status') == 'success':
                        self.dashboard_data['performance'] = {
                            'total_return': performance.get('total_return', 0),
                            'sharpe_ratio': performance.get('sharpe_ratio', 0),
                            'max_drawdown': performance.get('max_drawdown', 0),
                            'win_rate': performance.get('win_rate', 0)
                        }
                
                self.dashboard_data['last_update'] = datetime.now().isoformat()
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"Dashboard update error: {e}")
                time.sleep(10)
    
    def _run_interactive_dashboard(self):
        """Run interactive dashboard with commands"""
        print("\nDashboard Commands:")
        print("  'status' - Show system status")
        print("  'prices' - Show current market prices")
        print("  'account' - Show account information")
        print("  'portfolio' - Show portfolio summary")
        print("  'signals' - Show trading signals")
        print("  'ai' - Show AI neural network status")
        print("  'performance' - Show performance metrics")
        print("  'live' - Start live monitoring mode")
        print("  'export' - Export data to JSON")
        print("  'quit' - Exit dashboard")
        
        while self.running:
            try:
                cmd = input("\nDashboard> ").strip().lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                elif cmd == 'status':
                    self.show_system_status()
                elif cmd == 'prices':
                    self.show_current_prices()
                elif cmd == 'account':
                    self.show_account_info()
                elif cmd == 'portfolio':
                    self.show_portfolio()
                elif cmd == 'signals':
                    self.show_trading_signals()
                elif cmd == 'ai':
                    self.show_ai_status()
                elif cmd == 'performance':
                    self.show_performance()
                elif cmd == 'live':
                    self.live_monitoring_mode()
                elif cmd == 'export':
                    self.export_dashboard_data()
                elif cmd == '':
                    continue
                else:
                    print(f"Unknown command: {cmd}. Type a command or 'quit' to exit.")
            
            except KeyboardInterrupt:
                print("\nExiting dashboard...")
                break
            except EOFError:
                break
        
        self.running = False
        print("Dashboard stopped.")
    
    def show_system_status(self):
        """Show overall system status"""
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)
        
        # Account status
        account = self.dashboard_data.get('account', {})
        print(f"Account: {account.get('login', 'Demo')}")
        print(f"Server: {account.get('server', 'Simulation')}")
        print(f"Balance: ${account.get('balance', 0):,.2f}")
        print(f"Equity: ${account.get('equity', 0):,.2f}")
        print(f"Free Margin: ${account.get('margin_free', 0):,.2f}")
        
        # Technical analysis status
        technical = self.dashboard_data.get('technical', {})
        print(f"\nTechnical Analysis:")
        print(f"  Signals: {technical.get('signals', 0)}")
        print(f"  Indicators: {technical.get('indicators', 0)}")
        print(f"  Trend: {technical.get('trend', 'UNKNOWN')}")
        
        # AI status
        neural = self.dashboard_data.get('neural', {})
        print(f"\nAI Neural Network:")
        print(f"  Model Accuracy: {neural.get('accuracy', 0):.1f}%")
        print(f"  Predictions Made: {neural.get('predictions', 0)}")
        print(f"  Training Samples: {neural.get('training_samples', 0)}")
        
        # Last update
        last_update = self.dashboard_data.get('last_update', 'Never')
        print(f"\nLast Update: {last_update}")
    
    def show_current_prices(self):
        """Show current market prices"""
        print("\n" + "="*60)
        print("CURRENT MARKET PRICES")
        print("="*60)
        
        prices = self.dashboard_data.get('prices', {})
        if not prices:
            print("No price data available. Check MT5 connection.")
            return
        
        print(f"{'Symbol':<8} {'Bid':<10} {'Ask':<10} {'Spread':<8}")
        print("-" * 40)
        
        for symbol, data in prices.items():
            bid = data.get('bid', 0)
            ask = data.get('ask', 0)
            spread = data.get('spread', 0)
            print(f"{symbol:<8} {bid:<10.5f} {ask:<10.5f} {spread:<8.1f}")
    
    def show_account_info(self):
        """Show detailed account information"""
        print("\n" + "="*60)
        print("ACCOUNT INFORMATION")
        print("="*60)
        
        account = self.dashboard_data.get('account', {})
        
        print(f"Account Number: {account.get('login', 'Demo Account')}")
        print(f"Server: {account.get('server', 'Simulation')}")
        print(f"Company: {account.get('company', 'Demo Broker')}")
        print(f"Currency: {account.get('currency', 'USD')}")
        print(f"Leverage: 1:{account.get('leverage', 100)}")
        print(f"Trade Mode: {account.get('trade_mode', 'DEMO')}")
        
        print(f"\nBalance: ${account.get('balance', 0):,.2f}")
        print(f"Equity: ${account.get('equity', 0):,.2f}")
        print(f"Margin: ${account.get('margin', 0):,.2f}")
        print(f"Free Margin: ${account.get('margin_free', 0):,.2f}")
        print(f"Margin Level: {account.get('margin_level', 0):.2f}%")
        
        # Calculate additional metrics
        balance = account.get('balance', 0)
        equity = account.get('equity', 0)
        if balance > 0:
            pnl = equity - balance
            pnl_percent = (pnl / balance) * 100
            print(f"\nUnrealized P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)")
    
    def show_portfolio(self):
        """Show portfolio summary"""
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        
        portfolio = self.dashboard_data.get('portfolio', {})
        
        print(f"Total Value: ${portfolio.get('total_value', 0):,.2f}")
        print(f"Open Positions: {portfolio.get('positions', 0)}")
        print(f"Unrealized P&L: ${portfolio.get('unrealized_pnl', 0):,.2f}")
        print(f"Risk Level: {portfolio.get('risk_level', 'MODERATE')}")
    
    def show_trading_signals(self):
        """Show current trading signals"""
        print("\n" + "="*60)
        print("TRADING SIGNALS")
        print("="*60)
        
        if 'technical' in self.agents:
            analysis = self.agents['technical'].get_current_analysis()
            if analysis and 'signals' in analysis:
                signals = analysis['signals']
                
                if signals:
                    print(f"{'Action':<6} {'Strength':<8} {'Reason':<30}")
                    print("-" * 50)
                    
                    for signal in signals:
                        action = signal.get('action', 'UNKNOWN')
                        strength = signal.get('strength', 0)
                        reason = signal.get('reason', 'No reason provided')[:28]
                        print(f"{action:<6} {strength:<8.1f} {reason:<30}")
                else:
                    print("No trading signals available.")
            else:
                print("Technical analysis not available.")
        else:
            print("Technical analyst not initialized.")
    
    def show_ai_status(self):
        """Show AI neural network status"""
        print("\n" + "="*60)
        print("AI NEURAL NETWORK STATUS")
        print("="*60)
        
        neural = self.dashboard_data.get('neural', {})
        
        print(f"Model Accuracy: {neural.get('accuracy', 0):.1f}%")
        print(f"Is Trained: {'Yes' if neural.get('is_trained', False) else 'No'}")
        print(f"Training Samples: {neural.get('training_samples', 0)}")
        print(f"Predictions Made: {neural.get('predictions', 0)}")
        
        if 'neural' in self.agents:
            # Get a live prediction
            try:
                # Use dummy features for demonstration
                dummy_features = [0.5] * 10  # 10 features
                prediction = self.agents['neural'].predict_signal(dummy_features)
                
                if isinstance(prediction, dict) and 'probabilities' in prediction:
                    probs = prediction['probabilities']
                    print(f"\nLive AI Prediction:")
                    print(f"  Buy Probability: {probs.get('BUY', 0):.1%}")
                    print(f"  Hold Probability: {probs.get('HOLD', 0):.1%}")
                    print(f"  Sell Probability: {probs.get('SELL', 0):.1%}")
                    print(f"  Confidence: {prediction.get('confidence', 0):.1%}")
                    print(f"  Recommended Action: {prediction.get('action', 'UNKNOWN')}")
            except Exception as e:
                print(f"Could not get live prediction: {e}")
    
    def show_performance(self):
        """Show performance metrics"""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        performance = self.dashboard_data.get('performance', {})
        
        print(f"Total Return: {performance.get('total_return', 0):.2f}%")
        print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {performance.get('max_drawdown', 0):.2f}%")
        print(f"Win Rate: {performance.get('win_rate', 0):.1f}%")
    
    def live_monitoring_mode(self):
        """Enter live monitoring mode with continuous updates"""
        print("\n" + "="*60)
        print("LIVE MONITORING MODE (Press Ctrl+C to exit)")
        print("="*60)
        
        try:
            while True:
                # Clear screen (works on Windows)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("AGI TRADING SYSTEM - LIVE MONITOR")
                print("="*60)
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Show key metrics
                account = self.dashboard_data.get('account', {})
                print(f"\nAccount: ${account.get('equity', 0):,.2f} | ", end="")
                
                neural = self.dashboard_data.get('neural', {})
                print(f"AI: {neural.get('accuracy', 0):.1f}% | ", end="")
                
                technical = self.dashboard_data.get('technical', {})
                print(f"Signals: {technical.get('signals', 0)} | ", end="")
                print(f"Trend: {technical.get('trend', 'UNKNOWN')}")
                
                # Show prices
                prices = self.dashboard_data.get('prices', {})
                if prices:
                    print(f"\nLive Prices:")
                    for symbol, data in list(prices.items())[:4]:  # Show first 4
                        bid = data.get('bid', 0)
                        print(f"  {symbol}: {bid:.5f}")
                
                print(f"\nLast Update: {self.dashboard_data.get('last_update', 'Never')}")
                print("\nPress Ctrl+C to exit live mode...")
                
                time.sleep(5)
        
        except KeyboardInterrupt:
            print("\nExited live monitoring mode.")
    
    def export_dashboard_data(self):
        """Export dashboard data to JSON file"""
        try:
            filename = f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.dashboard_data, f, indent=2, default=str)
            print(f"Dashboard data exported to: {filename}")
        except Exception as e:
            print(f"Export failed: {e}")
    
    def stop_dashboard(self):
        """Stop the dashboard"""
        self.running = False
        print("Stopping dashboard...")

def main():
    """Main dashboard entry point"""
    dashboard = TradingDashboard()
    
    try:
        dashboard.start_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard interrupted by user.")
    finally:
        dashboard.stop_dashboard()

if __name__ == "__main__":
    main()