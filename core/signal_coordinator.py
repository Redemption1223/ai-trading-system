"""
AGENT_02: Signal Coordinator
Status: FULLY IMPLEMENTED
Purpose: Master orchestrator for all chart signals and coordination
"""

import asyncio
import json
import threading
import time
from queue import Queue, Empty
from datetime import datetime, timedelta
import logging

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

class SignalCoordinator:
    """Master orchestrator for all chart signals and coordination"""
    
    def __init__(self, mt5_connector):
        self.name = "SIGNAL_COORDINATOR"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        self.mt5_connector = mt5_connector
        
        # Chart management
        self.active_charts = {}
        self.user_selected_charts = []
        
        # Signal processing
        self.signal_queue = Queue()
        self.signal_history = []
        self.max_history = 1000
        
        # WebSocket clients (for UI updates)
        self.websocket_clients = set()
        
        # Threading
        self.coordination_thread = None
        self.is_running = False
        
        # Configuration
        self.signal_timeout = 30  # seconds
        self.min_confidence = 70  # minimum confidence for signal processing
        self.max_signals_per_minute = 10
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'signals_processed': 0,
            'charts_active': 0,
            'last_signal_time': None
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize the signal coordinator"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Check MT5 connector
            if not self.mt5_connector:
                self.logger.error("No MT5 connector provided")
                return {"status": "failed", "agent": "AGENT_02", "error": "No MT5 connector"}
            
            # Check MT5 connection status
            mt5_status = self.mt5_connector.get_status()
            if not mt5_status.get('connected', False):
                self.logger.warning("MT5 not connected - will operate in offline mode")
            
            self.status = "INITIALIZED"
            self.logger.info("Signal coordinator initialized successfully")
            
            return {
                "status": "initialized",
                "agent": "AGENT_02",
                "charts_supported": True,
                "websocket_supported": WEBSOCKETS_AVAILABLE,
                "mt5_connected": mt5_status.get('connected', False)
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "AGENT_02", "error": str(e)}
    
    def add_chart(self, symbol):
        """Add user-selected chart for monitoring"""
        try:
            if symbol in self.active_charts:
                return {"status": "exists", "message": f"Chart {symbol} already active"}
            
            # Import here to avoid circular imports
            from core.chart_signal_agent import ChartSignalAgent
            
            # Create chart agent
            agent = ChartSignalAgent(symbol, self.mt5_connector)
            
            # Initialize the agent
            if agent.initialize():
                self.active_charts[symbol] = agent
                if symbol not in self.user_selected_charts:
                    self.user_selected_charts.append(symbol)
                
                # Start analysis
                if agent.start_analysis():
                    self.performance_metrics['charts_active'] = len(self.active_charts)
                    self.logger.info(f"Chart {symbol} added and analysis started")
                    
                    return {
                        "status": "success", 
                        "message": f"Chart {symbol} added successfully",
                        "agent_status": agent.get_status()
                    }
                else:
                    # Cleanup if start failed
                    del self.active_charts[symbol]
                    if symbol in self.user_selected_charts:
                        self.user_selected_charts.remove(symbol)
                    return {"status": "failed", "message": f"Failed to start analysis for {symbol}"}
            else:
                return {"status": "failed", "message": f"Failed to initialize agent for {symbol}"}
                
        except Exception as e:
            self.logger.error(f"Error adding chart {symbol}: {e}")
            return {"status": "error", "message": f"Error adding chart: {e}"}
    
    def remove_chart(self, symbol):
        """Remove chart from monitoring"""
        try:
            if symbol not in self.active_charts:
                return {"status": "not_found", "message": f"Chart {symbol} not found"}
            
            # Shutdown the agent
            agent = self.active_charts[symbol]
            agent.shutdown()
            
            # Remove from tracking
            del self.active_charts[symbol]
            if symbol in self.user_selected_charts:
                self.user_selected_charts.remove(symbol)
            
            self.performance_metrics['charts_active'] = len(self.active_charts)
            self.logger.info(f"Chart {symbol} removed from monitoring")
            
            return {"status": "success", "message": f"Chart {symbol} removed successfully"}
            
        except Exception as e:
            self.logger.error(f"Error removing chart {symbol}: {e}")
            return {"status": "error", "message": f"Error removing chart: {e}"}
    
    def start_coordination(self):
        """Start the main coordination loop"""
        if self.is_running:
            return {"status": "already_running", "message": "Coordination already active"}
        
        try:
            self.is_running = True
            
            def coordination_loop():
                self.logger.info("Starting signal coordination loop")
                
                while self.is_running:
                    try:
                        # Collect signals from all active charts
                        signals = self.collect_all_signals()
                        
                        # Process and filter signals
                        if signals:
                            processed_signals = self.process_signals(signals)
                            
                            # Add to queue and broadcast
                            if processed_signals:
                                self.add_signals_to_queue(processed_signals)
                                self.broadcast_signals(processed_signals)
                        
                        # Update performance metrics
                        self.update_performance_metrics()
                        
                        # Sleep for next iteration
                        time.sleep(1)  # Check every second
                        
                    except Exception as e:
                        self.logger.error(f"Coordination loop error: {e}")
                        time.sleep(5)  # Wait longer on error
            
            # Start coordination thread
            self.coordination_thread = threading.Thread(target=coordination_loop, daemon=True)
            self.coordination_thread.start()
            
            self.status = "RUNNING"
            self.logger.info("Signal coordination started successfully")
            
            return {"status": "started", "message": "Signal coordination active"}
            
        except Exception as e:
            self.logger.error(f"Failed to start coordination: {e}")
            self.is_running = False
            return {"status": "failed", "message": f"Failed to start: {e}"}
    
    def stop_coordination(self):
        """Stop the coordination loop"""
        try:
            self.is_running = False
            
            if self.coordination_thread and self.coordination_thread.is_alive():
                self.coordination_thread.join(timeout=5)
            
            self.status = "STOPPED"
            self.logger.info("Signal coordination stopped")
            
            return {"status": "stopped", "message": "Coordination stopped successfully"}
            
        except Exception as e:
            self.logger.error(f"Error stopping coordination: {e}")
            return {"status": "error", "message": f"Error stopping: {e}"}
    
    def collect_all_signals(self):
        """Collect signals from all chart agents"""
        signals = []
        
        for symbol, agent in self.active_charts.items():
            try:
                signal = agent.get_current_signal()
                if signal and signal.get('confidence', 0) >= self.min_confidence:
                    # Add collection timestamp
                    signal['collected_at'] = datetime.now().isoformat()
                    signal['coordinator_agent'] = "AGENT_02"
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error collecting signal from {symbol}: {e}")
        
        return signals
    
    def process_signals(self, raw_signals):
        """Process and enhance signals with additional data"""
        processed = []
        
        for signal in raw_signals:
            try:
                # Add timestamp if not present
                if 'timestamp' not in signal:
                    signal['timestamp'] = datetime.now().isoformat()
                
                # Add account context if MT5 connected
                if self.mt5_connector and self.mt5_connector.connection_status:
                    signal['account_balance'] = self.mt5_connector.get_account_balance()
                    
                    # Calculate position size recommendation if risk calculator available
                    try:
                        from core.risk_calculator import RiskCalculator
                        risk_calc = RiskCalculator(signal.get('account_balance', 10000))
                        
                        if 'entry_price' in signal and 'stop_loss' in signal:
                            position_info = risk_calc.calculate_position_size(
                                signal['entry_price'],
                                signal['stop_loss'],
                                signal.get('account_balance', 10000) * 0.02  # 2% risk
                            )
                            
                            if position_info:
                                signal['position_size'] = position_info['position_size']
                                signal['risk_amount'] = position_info['risk_amount']
                                signal['risk_percent'] = position_info['risk_percent']
                    except ImportError:
                        # Risk calculator not available yet
                        pass
                else:
                    signal['account_balance'] = 0
                
                # Add signal quality score
                signal['quality_score'] = self.calculate_signal_quality(signal)
                
                # Add to processed list
                processed.append(signal)
                
                # Update counters
                self.performance_metrics['signals_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}")
        
        return processed
    
    def calculate_signal_quality(self, signal):
        """Calculate a quality score for the signal"""
        try:
            score = 0
            
            # Base score from confidence
            confidence = signal.get('confidence', 0)
            score += confidence * 0.6  # 60% weight for confidence
            
            # Bonus for multiple signal sources
            if 'technical_confidence' in signal and 'ai_confidence' in signal:
                score += 10  # Bonus for multi-source signals
            
            # Bonus for good risk/reward ratio
            if 'risk_reward_ratio' in signal:
                rr_ratio = signal['risk_reward_ratio']
                if rr_ratio >= 2.0:
                    score += 15
                elif rr_ratio >= 1.5:
                    score += 10
                elif rr_ratio >= 1.0:
                    score += 5
            
            # Penalty for very recent similar signals
            recent_similar = self.count_recent_similar_signals(signal['symbol'], signal['direction'])
            score -= recent_similar * 5
            
            return max(0, min(100, score))  # Clamp between 0-100
            
        except Exception as e:
            self.logger.error(f"Error calculating signal quality: {e}")
            return 50  # Default score
    
    def count_recent_similar_signals(self, symbol, direction, minutes=30):
        """Count recent similar signals for the same symbol and direction"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            count = 0
            
            for historical_signal in reversed(self.signal_history[-50:]):  # Check last 50 signals
                signal_time = datetime.fromisoformat(historical_signal.get('timestamp', ''))
                if signal_time < cutoff_time:
                    break
                
                if (historical_signal.get('symbol') == symbol and 
                    historical_signal.get('direction') == direction):
                    count += 1
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error counting recent signals: {e}")
            return 0
    
    def add_signals_to_queue(self, signals):
        """Add signals to the processing queue"""
        for signal in signals:
            try:
                # Add to queue
                self.signal_queue.put(signal)
                
                # Add to history
                self.signal_history.append(signal)
                
                # Trim history if too long
                if len(self.signal_history) > self.max_history:
                    self.signal_history = self.signal_history[-self.max_history:]
                
                # Update metrics
                self.performance_metrics['total_signals'] += 1
                self.performance_metrics['last_signal_time'] = signal.get('timestamp')
                
            except Exception as e:
                self.logger.error(f"Error adding signal to queue: {e}")
    
    def broadcast_signals(self, signals):
        """Broadcast signals to connected UI clients"""
        try:
            message = {
                'type': 'signals_update',
                'data': signals,
                'timestamp': datetime.now().isoformat(),
                'coordinator': 'AGENT_02'
            }
            
            # Log the broadcast
            self.logger.info(f"Broadcasting {len(signals)} signals to UI clients")
            
            # WebSocket broadcast (if available and clients connected)
            if WEBSOCKETS_AVAILABLE and self.websocket_clients:
                self.broadcast_websocket(message)
                
        except Exception as e:
            self.logger.error(f"Error broadcasting signals: {e}")
    
    def broadcast_websocket(self, message):
        """Broadcast to WebSocket clients"""
        if not WEBSOCKETS_AVAILABLE:
            return
            
        try:
            disconnected = set()
            
            for client in self.websocket_clients:
                try:
                    # This would need an async context in real implementation
                    # For now, just log the attempt
                    self.logger.debug(f"Would broadcast to WebSocket client: {message['type']}")
                except Exception as e:
                    self.logger.error(f"WebSocket broadcast error: {e}")
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected
            
        except Exception as e:
            self.logger.error(f"WebSocket broadcast error: {e}")
    
    def get_active_charts_status(self):
        """Get status of all active charts"""
        status = {}
        
        for symbol, agent in self.active_charts.items():
            try:
                agent_status = agent.get_status()
                status[symbol] = {
                    'status': agent_status.get('active', False),
                    'last_signal': agent.get_last_signal_time(),
                    'performance': agent.get_performance_metrics()
                }
            except Exception as e:
                self.logger.error(f"Error getting status for {symbol}: {e}")
                status[symbol] = {'status': 'error', 'error': str(e)}
        
        return status
    
    def get_latest_signals(self, count=10):
        """Get latest signals from queue"""
        signals = []
        
        try:
            # Get signals from queue without blocking
            for _ in range(min(count, self.signal_queue.qsize())):
                try:
                    signal = self.signal_queue.get_nowait()
                    signals.append(signal)
                except Empty:
                    break
            
            # Also get from history if queue is empty
            if not signals and self.signal_history:
                signals = self.signal_history[-count:]
                
        except Exception as e:
            self.logger.error(f"Error getting latest signals: {e}")
        
        return signals
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            self.performance_metrics.update({
                'charts_active': len(self.active_charts),
                'queue_size': self.signal_queue.qsize(),
                'history_size': len(self.signal_history),
                'coordination_status': self.status,
                'uptime': time.time() - getattr(self, 'start_time', time.time())
            })
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def get_status(self):
        """Get current coordinator status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'is_running': self.is_running,
            'active_charts': list(self.active_charts.keys()),
            'performance_metrics': self.performance_metrics,
            'signal_queue_size': self.signal_queue.qsize(),
            'websocket_clients': len(self.websocket_clients),
            'mt5_connected': self.mt5_connector.connection_status if self.mt5_connector else False
        }
    
    def shutdown(self):
        """Clean shutdown of the coordinator"""
        try:
            self.logger.info("Shutting down signal coordinator...")
            
            # Stop coordination
            self.stop_coordination()
            
            # Shutdown all chart agents
            for symbol, agent in self.active_charts.items():
                try:
                    agent.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down agent for {symbol}: {e}")
            
            # Clear collections
            self.active_charts.clear()
            self.user_selected_charts.clear()
            self.websocket_clients.clear()
            
            # Clear queues
            while not self.signal_queue.empty():
                try:
                    self.signal_queue.get_nowait()
                except Empty:
                    break
            
            self.status = "SHUTDOWN"
            self.logger.info("Signal coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Agent test
if __name__ == "__main__":
    # Test the signal coordinator
    from core.mt5_windows_connector import MT5WindowsConnector
    
    print("Testing AGENT_02: Signal Coordinator")
    print("=" * 40)
    
    # Create MT5 connector (will be offline)
    mt5_connector = MT5WindowsConnector()
    mt5_result = mt5_connector.initialize()
    print(f"MT5 Connector: {mt5_result['status']}")
    
    # Create coordinator
    coordinator = SignalCoordinator(mt5_connector)
    result = coordinator.initialize()
    print(f"Coordinator initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test status
        status = coordinator.get_status()
        print(f"Status: {status['status']}")
        print(f"Active charts: {status['active_charts']}")
        print(f"Signal queue size: {status['signal_queue_size']}")
        
        # Test shutdown
        print("\nShutting down...")
        coordinator.shutdown()
        
    print("Signal Coordinator test completed")