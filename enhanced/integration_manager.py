"""
Enhanced Integration Manager
Integrates all MQL5-enhanced components into the AGI Trading System
"""

import sys
import os
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add enhanced modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced modules
from market_microstructure_analyzer import EnhancedMarketMicrostructureAnalyzer
from harmonic_pattern_analyzer import AdvancedHarmonicPatternAnalyzer
from advanced_ui_dashboard import AdvancedTradingDashboard
from enhanced_error_recovery import EnhancedErrorRecoverySystem

class EnhancedIntegrationManager:
    """Manages integration of all enhanced MQL5 components"""
    
    def __init__(self, trading_system=None):
        self.name = "ENHANCED_INTEGRATION_MANAGER"
        self.version = "1.0.0"
        self.trading_system = trading_system
        
        # Enhanced components
        self.microstructure_analyzer = None
        self.pattern_analyzer = None
        self.ui_dashboard = None
        self.error_recovery = None
        
        # Integration state
        self.is_initialized = False
        self.is_running = False
        self.components_status = {}
        
        # Data flow
        self.data_update_thread = None
        self.update_interval = 1.0  # seconds
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self, symbol: str = "EURUSD") -> Dict:
        """Initialize all enhanced components"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Initialize Error Recovery System first
            self.logger.info("Initializing Enhanced Error Recovery System...")
            self.error_recovery = EnhancedErrorRecoverySystem()
            recovery_result = self.error_recovery.initialize()
            self.components_status['error_recovery'] = recovery_result
            
            if recovery_result['status'] != 'initialized':
                raise Exception(f"Error Recovery initialization failed: {recovery_result.get('error')}")
            
            # Initialize Market Microstructure Analyzer
            self.logger.info("Initializing Enhanced Market Microstructure Analyzer...")
            self.microstructure_analyzer = EnhancedMarketMicrostructureAnalyzer(symbol)
            micro_result = self.microstructure_analyzer.initialize()
            self.components_status['microstructure'] = micro_result
            
            if micro_result['status'] != 'initialized':
                self.error_recovery.report_error(
                    "ComponentInitializationError",
                    f"Microstructure Analyzer initialization failed: {micro_result.get('error')}",
                    "INTEGRATION_MANAGER"
                )
            
            # Initialize Harmonic Pattern Analyzer
            self.logger.info("Initializing Advanced Harmonic Pattern Analyzer...")
            self.pattern_analyzer = AdvancedHarmonicPatternAnalyzer(symbol)
            pattern_result = self.pattern_analyzer.initialize()
            self.components_status['pattern_analyzer'] = pattern_result
            
            if pattern_result['status'] != 'initialized':
                self.error_recovery.report_error(
                    "ComponentInitializationError",
                    f"Pattern Analyzer initialization failed: {pattern_result.get('error')}",
                    "INTEGRATION_MANAGER"
                )
            
            # Initialize UI Dashboard
            self.logger.info("Initializing Advanced UI Dashboard...")
            self.ui_dashboard = AdvancedTradingDashboard(self.trading_system)
            ui_result = self.ui_dashboard.initialize()
            self.components_status['ui_dashboard'] = ui_result
            
            if ui_result['status'] != 'initialized':
                self.error_recovery.report_error(
                    "ComponentInitializationError",
                    f"UI Dashboard initialization failed: {ui_result.get('error')}",
                    "INTEGRATION_MANAGER"
                )
            
            # Start data flow
            self.is_running = True
            self.data_update_thread = threading.Thread(target=self._data_flow_loop, daemon=True)
            self.data_update_thread.start()
            
            self.is_initialized = True
            
            # Success summary
            successful_components = sum(1 for status in self.components_status.values() 
                                      if status['status'] == 'initialized')
            total_components = len(self.components_status)
            
            self.logger.info(f"Enhanced Integration Manager initialized successfully")
            self.logger.info(f"Components initialized: {successful_components}/{total_components}")
            
            return {
                "status": "initialized",
                "agent": "ENHANCED_INTEGRATION_MANAGER",
                "symbol": symbol,
                "components_initialized": successful_components,
                "total_components": total_components,
                "components_status": self.components_status,
                "features_enabled": [
                    "Market Microstructure Analysis",
                    "Harmonic Pattern Recognition",
                    "Advanced UI Dashboard",
                    "Enhanced Error Recovery",
                    "Real-time Data Integration",
                    "Multi-component Coordination"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced Integration Manager initialization failed: {e}")
            if self.error_recovery:
                self.error_recovery.report_error(
                    "SystemInitializationError",
                    f"Integration Manager initialization failed: {str(e)}",
                    "INTEGRATION_MANAGER"
                )
            return {"status": "failed", "agent": "ENHANCED_INTEGRATION_MANAGER", "error": str(e)}
    
    def update_market_data(self, bid: float, ask: float, volume: float, timestamp: datetime = None):
        """Update market data across all components"""
        if not self.is_initialized:
            return
        
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Update microstructure analyzer
            if self.microstructure_analyzer:
                self.microstructure_analyzer.update_tick_data(bid, ask, volume, timestamp)
            
            # Get current price for pattern analyzer
            current_price = (bid + ask) / 2
            
            # Note: Pattern analyzer needs price history, not individual ticks
            # This would be handled by maintaining a price buffer
            
        except Exception as e:
            self.logger.error(f"Market data update error: {e}")
            if self.error_recovery:
                self.error_recovery.report_error(
                    "DataUpdateError",
                    f"Market data update failed: {str(e)}",
                    "INTEGRATION_MANAGER"
                )
    
    def update_price_history(self, prices: List[float], timestamps: List[datetime] = None):
        """Update price history for pattern analysis"""
        if not self.is_initialized:
            return
        
        try:
            # Update pattern analyzer
            if self.pattern_analyzer:
                self.pattern_analyzer.update_price_data(prices, timestamps)
            
        except Exception as e:
            self.logger.error(f"Price history update error: {e}")
            if self.error_recovery:
                self.error_recovery.report_error(
                    "DataUpdateError",
                    f"Price history update failed: {str(e)}",
                    "INTEGRATION_MANAGER"
                )
    
    def update_account_data(self, account_info: Dict):
        """Update account data across components"""
        if not self.is_initialized:
            return
        
        try:
            # Update UI dashboard
            if self.ui_dashboard:
                self.ui_dashboard.update_account_data(account_info)
            
            # Check for risk alerts
            if self.error_recovery:
                if 'margin_level' in account_info:
                    margin = account_info['margin_level']
                    if margin < 100:  # Low margin alert
                        self.error_recovery.check_alert_conditions('low_margin', margin)
                
                if 'balance' in account_info and 'equity' in account_info:
                    balance = account_info['balance']
                    equity = account_info['equity']
                    if balance > 0:
                        drawdown = ((balance - equity) / balance) * 100
                        self.error_recovery.check_alert_conditions('high_drawdown', drawdown)
            
        except Exception as e:
            self.logger.error(f"Account data update error: {e}")
            if self.error_recovery:
                self.error_recovery.report_error(
                    "DataUpdateError",
                    f"Account data update failed: {str(e)}",
                    "INTEGRATION_MANAGER"
                )
    
    def update_trading_stats(self, stats: Dict):
        """Update trading statistics"""
        if not self.is_initialized:
            return
        
        try:
            # Update UI dashboard
            if self.ui_dashboard:
                self.ui_dashboard.update_trading_stats(stats)
            
        except Exception as e:
            self.logger.error(f"Trading stats update error: {e}")
            if self.error_recovery:
                self.error_recovery.report_error(
                    "DataUpdateError",
                    f"Trading stats update failed: {str(e)}",
                    "INTEGRATION_MANAGER"
                )
    
    def update_positions(self, positions: List[Dict]):
        """Update position information"""
        if not self.is_initialized:
            return
        
        try:
            # Update UI dashboard
            if self.ui_dashboard:
                self.ui_dashboard.update_positions(positions)
            
        except Exception as e:
            self.logger.error(f"Positions update error: {e}")
            if self.error_recovery:
                self.error_recovery.report_error(
                    "DataUpdateError",
                    f"Positions update failed: {str(e)}",
                    "INTEGRATION_MANAGER"
                )
    
    def get_microstructure_analysis(self) -> Optional[Dict]:
        """Get current microstructure analysis"""
        if self.microstructure_analyzer:
            try:
                return self.microstructure_analyzer.get_microstructure_analysis()
            except Exception as e:
                self.logger.error(f"Microstructure analysis error: {e}")
                if self.error_recovery:
                    self.error_recovery.report_error(
                        "AnalysisError",
                        f"Microstructure analysis failed: {str(e)}",
                        "MICROSTRUCTURE_ANALYZER"
                    )
        return None
    
    def get_pattern_analysis(self) -> Optional[Dict]:
        """Get current pattern analysis"""
        if self.pattern_analyzer:
            try:
                return self.pattern_analyzer.get_pattern_analysis()
            except Exception as e:
                self.logger.error(f"Pattern analysis error: {e}")
                if self.error_recovery:
                    self.error_recovery.report_error(
                        "AnalysisError",
                        f"Pattern analysis failed: {str(e)}",
                        "PATTERN_ANALYZER"
                    )
        return None
    
    def get_comprehensive_analysis(self) -> Dict:
        """Get comprehensive analysis from all components"""
        try:
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "microstructure": None,
                "patterns": None,
                "system_health": None,
                "alerts": []
            }
            
            # Get microstructure analysis
            micro_analysis = self.get_microstructure_analysis()
            if micro_analysis and micro_analysis.get('status') == 'success':
                analysis['microstructure'] = micro_analysis
            
            # Get pattern analysis
            pattern_analysis = self.get_pattern_analysis()
            if pattern_analysis and pattern_analysis.get('status') == 'success':
                analysis['patterns'] = pattern_analysis
            
            # Get system health
            if self.error_recovery:
                analysis['system_health'] = self.error_recovery.get_system_status()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis error: {e}")
            if self.error_recovery:
                self.error_recovery.report_error(
                    "AnalysisError",
                    f"Comprehensive analysis failed: {str(e)}",
                    "INTEGRATION_MANAGER"
                )
            return {"error": str(e)}
    
    def add_signal_to_ui(self, signal_text: str, signal_type: str = "INFO"):
        """Add signal to UI dashboard"""
        if self.ui_dashboard:
            try:
                self.ui_dashboard.add_signal(signal_text, signal_type)
            except Exception as e:
                self.logger.error(f"UI signal update error: {e}")
    
    def report_error(self, error_type: str, error_message: str, error_source: str):
        """Report error through error recovery system"""
        if self.error_recovery:
            return self.error_recovery.report_error(error_type, error_message, error_source)
        else:
            self.logger.error(f"Error reported but no recovery system: {error_type} - {error_message}")
            return None
    
    def _data_flow_loop(self):
        """Main data flow coordination loop"""
        while self.is_running:
            try:
                # Get comprehensive analysis
                analysis = self.get_comprehensive_analysis()
                
                # Skip UI updates to avoid threading issues
                # UI can be started separately if needed
                
                # Log analysis results instead
                if 'microstructure' in analysis and analysis['microstructure']:
                    micro_data = analysis['microstructure']
                    if 'market_regime' in micro_data:
                        self.logger.info(f"Market Regime: {micro_data['market_regime']}")
                
                if 'patterns' in analysis and analysis['patterns']:
                    pattern_data = analysis['patterns']
                    if pattern_data.get('total_patterns', 0) > 0:
                        best_pattern = pattern_data.get('best_pattern')
                        if best_pattern:
                            self.logger.info(f"Pattern: {best_pattern['type']} ({best_pattern['confidence']:.1f}%)")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                # Only log critical errors to avoid spam
                if "main thread is not in main loop" not in str(e):
                    self.logger.error(f"Data flow loop error: {e}")
                    if self.error_recovery:
                        self.error_recovery.report_error(
                            "DataFlowError",
                            f"Data flow loop error: {str(e)}",
                            "INTEGRATION_MANAGER"
                        )
                time.sleep(5)  # Wait longer on error
    
    def start_ui_dashboard(self):
        """Start the UI dashboard in a separate thread"""
        if self.ui_dashboard:
            try:
                # Run dashboard in separate thread
                ui_thread = threading.Thread(target=self.ui_dashboard.run, daemon=False)
                ui_thread.start()
                self.logger.info("UI Dashboard started in separate thread")
                return ui_thread
            except Exception as e:
                self.logger.error(f"Failed to start UI dashboard: {e}")
                if self.error_recovery:
                    self.error_recovery.report_error(
                        "UIStartupError",
                        f"UI dashboard startup failed: {str(e)}",
                        "INTEGRATION_MANAGER"
                    )
                return None
        return None
    
    def get_integration_status(self) -> Dict:
        """Get integration status"""
        return {
            "name": self.name,
            "version": self.version,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "components_status": self.components_status,
            "active_components": {
                "microstructure_analyzer": self.microstructure_analyzer is not None,
                "pattern_analyzer": self.pattern_analyzer is not None,
                "ui_dashboard": self.ui_dashboard is not None,
                "error_recovery": self.error_recovery is not None
            },
            "system_health": self.error_recovery.get_system_status() if self.error_recovery else None
        }
    
    def shutdown(self):
        """Shutdown all enhanced components"""
        try:
            self.logger.info("Shutting down Enhanced Integration Manager...")
            self.is_running = False
            
            # Stop data flow thread
            if self.data_update_thread and self.data_update_thread.is_alive():
                self.data_update_thread.join(timeout=5)
            
            # Shutdown components in reverse order
            if self.ui_dashboard:
                self.ui_dashboard.shutdown()
            
            if self.pattern_analyzer:
                self.pattern_analyzer.shutdown()
            
            if self.microstructure_analyzer:
                self.microstructure_analyzer.shutdown()
            
            if self.error_recovery:
                self.error_recovery.shutdown()
            
            self.logger.info("Enhanced Integration Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Integration Manager shutdown error: {e}")

def test_integration_manager():
    """Test the integration manager"""
    print("Testing Enhanced Integration Manager...")
    print("=" * 60)
    
    integration_manager = EnhancedIntegrationManager()
    
    # Test initialization
    result = integration_manager.initialize("EURUSD")
    print(f"Initialization: {result['status']}")
    
    if result['status'] == 'initialized':
        print(f"Components initialized: {result['components_initialized']}/{result['total_components']}")
        print("Features enabled:")
        for feature in result['features_enabled']:
            print(f"  - {feature}")
        
        # Test data updates
        print("\\nTesting data updates...")
        integration_manager.update_market_data(1.17000, 1.17002, 1.5)
        integration_manager.update_account_data({
            'balance': 10000.0,
            'equity': 10150.0,
            'margin_level': 250.0
        })
        
        # Test analysis
        analysis = integration_manager.get_comprehensive_analysis()
        print(f"Analysis timestamp: {analysis.get('timestamp', 'N/A')}")
        
        # Get status
        status = integration_manager.get_integration_status()
        print(f"\\nIntegration Status:")
        print(f"  Running: {status['is_running']}")
        print(f"  Active Components: {sum(status['active_components'].values())}/4")
        
        print("\\n[OK] Enhanced Integration Manager Test PASSED!")
        
        # Note: UI dashboard would be started separately for actual use
        return integration_manager
    else:
        print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
        return None

if __name__ == "__main__":
    integration_manager = test_integration_manager()
    if integration_manager:
        print("\\nIntegration Manager is ready for use!")
        print("To start the UI dashboard, call: integration_manager.start_ui_dashboard()")
        
        # Keep running for demonstration
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\\nShutting down...")
        finally:
            integration_manager.shutdown()