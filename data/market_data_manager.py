"""
AGENT_07: Market Data Manager
Status: FULLY IMPLEMENTED
Purpose: Advanced market data collection, processing, and real-time streaming management
"""

import logging
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
import queue

# Try to import numerical libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

class DataQuality:
    """Data quality metrics and validation"""
    
    def __init__(self):
        self.total_ticks = 0
        self.valid_ticks = 0
        self.invalid_ticks = 0
        self.gaps_detected = 0
        self.duplicate_ticks = 0
        self.last_validation_time = None
    
    def get_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        if self.total_ticks == 0:
            return 0.0
        
        quality_score = (self.valid_ticks / self.total_ticks) * 100
        
        # Penalize for gaps and duplicates
        gap_penalty = min(10, self.gaps_detected * 2)
        duplicate_penalty = min(5, self.duplicate_ticks * 0.1)
        
        return max(0, quality_score - gap_penalty - duplicate_penalty)

class MarketDataManager:
    """Advanced market data collection and streaming manager"""
    
    def __init__(self, symbols: List[str] = None):
        self.name = "MARKET_DATA_MANAGER"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        
        # Symbols to track
        self.symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        self.active_symbols = set()
        
        # Data storage
        self.tick_data = defaultdict(deque)  # Symbol -> deque of ticks
        self.ohlc_data = defaultdict(dict)   # Symbol -> {timeframe: DataFrame}
        self.max_tick_history = 10000
        self.max_ohlc_bars = 1000
        
        # Data streaming
        self.data_subscribers = defaultdict(list)  # Symbol -> list of callbacks
        self.streaming_active = False
        self.stream_threads = {}
        
        # Data sources
        self.mt5_connector = None
        self.external_feeds = []
        self.feed_priorities = {}  # Source -> priority
        
        # Real-time processing
        self.tick_queue = queue.Queue(maxsize=10000)
        self.processing_thread = None
        self.is_processing = False
        
        # Data quality monitoring
        self.quality_metrics = defaultdict(DataQuality)
        self.quality_thresholds = {
            'min_quality_score': 85.0,
            'max_gap_seconds': 30,
            'max_duplicate_rate': 0.05
        }
        
        # Timeframes for OHLC data
        self.timeframes = {
            'M1': 60,      # 1 minute
            'M5': 300,     # 5 minutes  
            'M15': 900,    # 15 minutes
            'H1': 3600,    # 1 hour
            'H4': 14400,   # 4 hours
            'D1': 86400    # 1 day
        }
        
        # Performance tracking
        self.performance_stats = {
            'ticks_processed': 0,
            'bars_generated': 0,
            'data_gaps': 0,
            'processing_errors': 0,
            'last_tick_time': None,
            'uptime_start': None
        }
        
        # Data persistence
        self.enable_persistence = True
        self.data_file_path = "market_data"
        self.save_interval = 300  # Save every 5 minutes
        
        # Market hours and session management
        self.market_sessions = {
            'TOKYO': {'start': '00:00', 'end': '09:00', 'timezone': 'Asia/Tokyo'},
            'LONDON': {'start': '08:00', 'end': '17:00', 'timezone': 'Europe/London'},
            'NEW_YORK': {'start': '13:00', 'end': '22:00', 'timezone': 'America/New_York'}
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self, mt5_connector=None):
        """Initialize the market data manager"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Set up MT5 connector
            if mt5_connector:
                self.mt5_connector = mt5_connector
                self.feed_priorities['MT5'] = 1  # Highest priority
                self.logger.info("MT5 connector configured as primary data source")
            
            # Initialize data structures for each symbol
            for symbol in self.symbols:
                self.tick_data[symbol] = deque(maxlen=self.max_tick_history)
                self.ohlc_data[symbol] = {}
                for timeframe in self.timeframes:
                    self.ohlc_data[symbol][timeframe] = [] if not PANDAS_AVAILABLE else pd.DataFrame()
                self.quality_metrics[symbol] = DataQuality()
            
            # Start tick processing thread
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_tick_queue, daemon=True)
            self.processing_thread.start()
            
            # Initialize performance tracking
            self.performance_stats['uptime_start'] = datetime.now()
            
            self.status = "INITIALIZED"
            self.logger.info(f"Market Data Manager initialized for {len(self.symbols)} symbols")
            
            return {
                "status": "initialized",
                "agent": "AGENT_07", 
                "symbols": self.symbols,
                "timeframes": list(self.timeframes.keys()),
                "data_sources": list(self.feed_priorities.keys()),
                "pandas_available": PANDAS_AVAILABLE,
                "numpy_available": NUMPY_AVAILABLE
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "AGENT_07", "error": str(e)}
    
    def start_streaming(self, symbols: List[str] = None):
        """Start real-time data streaming for specified symbols"""
        try:
            if self.streaming_active:
                return {"status": "already_active", "message": "Streaming already active"}
            
            # Handle case where a single string is passed instead of a list
            if isinstance(symbols, str):
                symbols_to_stream = [symbols]
            else:
                symbols_to_stream = symbols or self.symbols
            self.streaming_active = True
            
            # Start streaming thread for each symbol
            for symbol in symbols_to_stream:
                if symbol not in self.stream_threads:
                    thread = threading.Thread(
                        target=self._stream_symbol_data, 
                        args=(symbol,), 
                        daemon=True
                    )
                    thread.start()
                    self.stream_threads[symbol] = thread
                    self.active_symbols.add(symbol)
            
            self.logger.info(f"Data streaming started for {len(symbols_to_stream)} symbols")
            
            return {
                "status": "started",
                "symbols_streaming": list(self.active_symbols),
                "message": "Real-time data streaming active"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return {"status": "failed", "message": str(e)}
    
    def stop_streaming(self, symbols: List[str] = None):
        """Stop data streaming for specified symbols"""
        try:
            symbols_to_stop = symbols or list(self.active_symbols)
            
            for symbol in symbols_to_stop:
                if symbol in self.active_symbols:
                    self.active_symbols.remove(symbol)
                    
                if symbol in self.stream_threads:
                    # Thread will stop when streaming_active is False
                    del self.stream_threads[symbol]
            
            if not symbols:
                self.streaming_active = False
                self.stream_threads.clear()
                self.active_symbols.clear()
            
            self.logger.info(f"Data streaming stopped for {len(symbols_to_stop)} symbols")
            
            return {
                "status": "stopped",
                "symbols_stopped": symbols_to_stop,
                "active_symbols": list(self.active_symbols)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stop streaming: {e}")
            return {"status": "error", "message": str(e)}
    
    def _stream_symbol_data(self, symbol: str):
        """Stream data for a specific symbol"""
        self.logger.info(f"Starting data stream for {symbol}")
        
        while self.streaming_active and symbol in self.active_symbols:
            try:
                # Get tick data from primary source (MT5)
                tick_data = self._fetch_tick_data(symbol)
                
                if tick_data:
                    # Add to processing queue
                    try:
                        self.tick_queue.put((symbol, tick_data), timeout=1)
                    except queue.Full:
                        self.logger.warning(f"Tick queue full, dropping data for {symbol}")
                        self.performance_stats['processing_errors'] += 1
                
                # Control streaming frequency
                time.sleep(0.1)  # 10 ticks per second max
                
            except Exception as e:
                self.logger.error(f"Error streaming {symbol}: {e}")
                time.sleep(1)  # Wait before retry
    
    def _fetch_tick_data(self, symbol: str) -> Optional[Dict]:
        """Fetch latest tick data for symbol"""
        try:
            # Try MT5 first (highest priority)
            if self.mt5_connector:
                tick = self.mt5_connector.get_live_tick(symbol)
                if tick:
                    return {
                        'symbol': symbol,
                        'bid': tick.get('bid', 0),
                        'ask': tick.get('ask', 0),
                        'time': tick.get('time', datetime.now().timestamp()),
                        'volume': tick.get('volume', 0),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'MT5'
                    }
            
            # Fallback: Generate synthetic tick data
            # LIVE DATA ONLY - No synthetic tick generation
            return {"symbol": symbol, "bid": 0, "ask": 0, "error": "Live data not available"}
            
        except Exception as e:
            self.logger.error(f"Error fetching tick data for {symbol}: {e}")
            return None
    
    # REMOVED: _generate_synthetic_tick() - LIVE DATA ONLY
    
    def _process_tick_queue(self):
        """Process incoming tick data from queue"""
        self.logger.info("Starting tick data processing thread")
        
        while self.is_processing:
            try:
                # Get tick from queue with timeout
                try:
                    symbol, tick_data = self.tick_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Validate tick data
                if self._validate_tick_data(symbol, tick_data):
                    # Store tick data
                    self._store_tick_data(symbol, tick_data)
                    
                    # Update OHLC bars
                    self._update_ohlc_bars(symbol, tick_data)
                    
                    # Notify subscribers
                    self._notify_subscribers(symbol, tick_data)
                    
                    # Update performance stats
                    self.performance_stats['ticks_processed'] += 1
                    self.performance_stats['last_tick_time'] = datetime.now().isoformat()
                
                # Mark task as done
                self.tick_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing tick data: {e}")
                self.performance_stats['processing_errors'] += 1
    
    def _validate_tick_data(self, symbol: str, tick_data: Dict) -> bool:
        """Validate incoming tick data quality"""
        try:
            quality = self.quality_metrics[symbol]
            quality.total_ticks += 1
            
            # Basic validation checks
            if not isinstance(tick_data, dict):
                quality.invalid_ticks += 1
                return False
            
            required_fields = ['bid', 'ask', 'time', 'timestamp']
            for field in required_fields:
                if field not in tick_data:
                    quality.invalid_ticks += 1
                    return False
            
            # Price validation
            bid = tick_data.get('bid', 0)
            ask = tick_data.get('ask', 0)
            
            if bid <= 0 or ask <= 0 or ask < bid:
                quality.invalid_ticks += 1
                return False
            
            # Check for reasonable spread
            spread = ask - bid
            if spread > bid * 0.01:  # Spread > 1% is suspicious
                quality.invalid_ticks += 1
                return False
            
            # Time validation
            tick_time = datetime.fromisoformat(tick_data['timestamp'].replace('Z', '+00:00'))
            now = datetime.now()
            
            # Tick should not be too far in future or past
            if abs((now - tick_time).total_seconds()) > 300:  # 5 minutes tolerance
                quality.invalid_ticks += 1
                return False
            
            # Check for duplicates (same timestamp and price)
            recent_ticks = list(self.tick_data[symbol])[-10:]  # Check last 10 ticks
            for recent_tick in recent_ticks:
                if (recent_tick.get('timestamp') == tick_data['timestamp'] and
                    recent_tick.get('bid') == tick_data['bid'] and
                    recent_tick.get('ask') == tick_data['ask']):
                    quality.duplicate_ticks += 1
                    return False
            
            quality.valid_ticks += 1
            quality.last_validation_time = datetime.now().isoformat()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Tick validation error for {symbol}: {e}")
            self.quality_metrics[symbol].invalid_ticks += 1
            return False
    
    def _store_tick_data(self, symbol: str, tick_data: Dict):
        """Store validated tick data"""
        try:
            # Add to tick history
            self.tick_data[symbol].append(tick_data)
            
            # Optional: persist to disk
            if self.enable_persistence:
                self._persist_tick_data(symbol, tick_data)
                
        except Exception as e:
            self.logger.error(f"Error storing tick data for {symbol}: {e}")
    
    def _update_ohlc_bars(self, symbol: str, tick_data: Dict):
        """Update OHLC bars with new tick data"""
        try:
            tick_time = datetime.fromisoformat(tick_data['timestamp'].replace('Z', '+00:00'))
            mid_price = (tick_data['bid'] + tick_data['ask']) / 2
            volume = tick_data.get('volume', 1)
            
            for timeframe, seconds in self.timeframes.items():
                # Calculate bar timestamp (aligned to timeframe)
                bar_timestamp = self._align_timestamp(tick_time, seconds)
                
                # Get or create OHLC data for this timeframe
                ohlc_data = self.ohlc_data[symbol][timeframe]
                
                if PANDAS_AVAILABLE and isinstance(ohlc_data, pd.DataFrame):
                    self._update_pandas_ohlc(ohlc_data, bar_timestamp, mid_price, volume)
                else:
                    self._update_list_ohlc(ohlc_data, bar_timestamp, mid_price, volume)
                
        except Exception as e:
            self.logger.error(f"Error updating OHLC bars for {symbol}: {e}")
    
    def _align_timestamp(self, timestamp: datetime, seconds: int) -> datetime:
        """Align timestamp to timeframe boundary"""
        # Round down to nearest timeframe boundary
        total_seconds = int(timestamp.timestamp())
        aligned_seconds = (total_seconds // seconds) * seconds
        return datetime.fromtimestamp(aligned_seconds)
    
    def _update_pandas_ohlc(self, ohlc_df, bar_time: datetime, price: float, volume: int):
        """Update pandas OHLC DataFrame"""
        try:
            if ohlc_df.empty or bar_time not in ohlc_df.index:
                # Create new bar
                new_bar = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'timestamp': bar_time.isoformat()
                }
                
                new_row = pd.DataFrame([new_bar], index=[bar_time])
                ohlc_df = pd.concat([ohlc_df, new_row])
                
                # Keep only recent bars
                if len(ohlc_df) > self.max_ohlc_bars:
                    ohlc_df = ohlc_df.tail(self.max_ohlc_bars)
                    
            else:
                # Update existing bar
                ohlc_df.at[bar_time, 'high'] = max(ohlc_df.at[bar_time, 'high'], price)
                ohlc_df.at[bar_time, 'low'] = min(ohlc_df.at[bar_time, 'low'], price)
                ohlc_df.at[bar_time, 'close'] = price
                ohlc_df.at[bar_time, 'volume'] += volume
                
        except Exception as e:
            self.logger.error(f"Error updating pandas OHLC: {e}")
    
    def _update_list_ohlc(self, ohlc_list: List, bar_time: datetime, price: float, volume: int):
        """Update list-based OHLC data"""
        try:
            # Find existing bar or create new one
            bar_found = False
            
            for bar in reversed(ohlc_list):  # Check recent bars first
                if bar['timestamp'] == bar_time.isoformat():
                    # Update existing bar
                    bar['high'] = max(bar['high'], price)
                    bar['low'] = min(bar['low'], price)
                    bar['close'] = price
                    bar['volume'] += volume
                    bar_found = True
                    break
            
            if not bar_found:
                # Create new bar
                new_bar = {
                    'timestamp': bar_time.isoformat(),
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                }
                
                ohlc_list.append(new_bar)
                
                # Keep only recent bars
                if len(ohlc_list) > self.max_ohlc_bars:
                    ohlc_list[:] = ohlc_list[-self.max_ohlc_bars:]
                    
        except Exception as e:
            self.logger.error(f"Error updating list OHLC: {e}")
    
    def _notify_subscribers(self, symbol: str, tick_data: Dict):
        """Notify all subscribers of new tick data"""
        try:
            subscribers = self.data_subscribers.get(symbol, [])
            
            for callback in subscribers:
                try:
                    callback(symbol, tick_data)
                except Exception as e:
                    self.logger.error(f"Error notifying subscriber: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in subscriber notification: {e}")
    
    def subscribe_to_symbol(self, symbol: str, callback):
        """Subscribe to real-time updates for a symbol"""
        try:
            if symbol not in self.data_subscribers:
                self.data_subscribers[symbol] = []
                
            self.data_subscribers[symbol].append(callback)
            
            self.logger.info(f"New subscriber added for {symbol}")
            
            return {
                "status": "subscribed",
                "symbol": symbol,
                "subscriber_count": len(self.data_subscribers[symbol])
            }
            
        except Exception as e:
            self.logger.error(f"Error adding subscriber for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def unsubscribe_from_symbol(self, symbol: str, callback):
        """Unsubscribe from symbol updates"""
        try:
            if symbol in self.data_subscribers and callback in self.data_subscribers[symbol]:
                self.data_subscribers[symbol].remove(callback)
                
                self.logger.info(f"Subscriber removed for {symbol}")
                
                return {
                    "status": "unsubscribed",
                    "symbol": symbol,
                    "subscriber_count": len(self.data_subscribers[symbol])
                }
            else:
                return {"status": "not_found", "message": "Subscriber not found"}
                
        except Exception as e:
            self.logger.error(f"Error removing subscriber for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_latest_tick(self, symbol: str) -> Optional[Dict]:
        """Get latest tick data for symbol"""
        try:
            if symbol in self.tick_data and self.tick_data[symbol]:
                return dict(self.tick_data[symbol][-1])  # Return copy
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest tick for {symbol}: {e}")
            return None
    
    def get_tick_history(self, symbol: str, count: int = 100) -> List[Dict]:
        """Get recent tick history for symbol"""
        try:
            if symbol in self.tick_data:
                ticks = list(self.tick_data[symbol])
                return ticks[-count:] if len(ticks) > count else ticks
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting tick history for {symbol}: {e}")
            return []
    
    def get_ohlc_data(self, symbol: str, timeframe: str, count: int = 100):
        """Get OHLC data for symbol and timeframe"""
        try:
            if symbol in self.ohlc_data and timeframe in self.ohlc_data[symbol]:
                ohlc_data = self.ohlc_data[symbol][timeframe]
                
                if PANDAS_AVAILABLE and isinstance(ohlc_data, pd.DataFrame):
                    return ohlc_data.tail(count)
                elif isinstance(ohlc_data, list):
                    return ohlc_data[-count:] if len(ohlc_data) > count else ohlc_data
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting OHLC data for {symbol} {timeframe}: {e}")
            return None
    
    def get_data_quality(self, symbol: str = None) -> Dict:
        """Get data quality metrics"""
        try:
            if symbol:
                if symbol in self.quality_metrics:
                    quality = self.quality_metrics[symbol]
                    return {
                        'symbol': symbol,
                        'total_ticks': quality.total_ticks,
                        'valid_ticks': quality.valid_ticks,
                        'invalid_ticks': quality.invalid_ticks,
                        'quality_score': quality.get_quality_score(),
                        'gaps_detected': quality.gaps_detected,
                        'duplicate_ticks': quality.duplicate_ticks,
                        'last_validation': quality.last_validation_time
                    }
                else:
                    return {"error": f"No quality data for symbol {symbol}"}
            else:
                # Return quality for all symbols
                quality_data = {}
                for sym, quality in self.quality_metrics.items():
                    quality_data[sym] = {
                        'total_ticks': quality.total_ticks,
                        'valid_ticks': quality.valid_ticks,
                        'quality_score': quality.get_quality_score()
                    }
                return quality_data
                
        except Exception as e:
            self.logger.error(f"Error getting data quality: {e}")
            return {"error": str(e)}
    
    def get_market_summary(self) -> Dict:
        """Get overall market data summary"""
        try:
            summary = {
                'symbols_tracked': len(self.symbols),
                'symbols_active': len(self.active_symbols),
                'streaming_active': self.streaming_active,
                'total_ticks_processed': self.performance_stats['ticks_processed'],
                'processing_errors': self.performance_stats['processing_errors'],
                'last_tick_time': self.performance_stats['last_tick_time'],
                'data_quality': {}
            }
            
            # Add latest prices
            latest_prices = {}
            for symbol in self.symbols:
                latest_tick = self.get_latest_tick(symbol)
                if latest_tick:
                    latest_prices[symbol] = {
                        'bid': latest_tick['bid'],
                        'ask': latest_tick['ask'],
                        'timestamp': latest_tick['timestamp']
                    }
                    
            summary['latest_prices'] = latest_prices
            
            # Add data quality summary
            for symbol in self.symbols:
                if symbol in self.quality_metrics:
                    quality = self.quality_metrics[symbol]
                    summary['data_quality'][symbol] = quality.get_quality_score()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting market summary: {e}")
            return {"error": str(e)}
    
    def _persist_tick_data(self, symbol: str, tick_data: Dict):
        """Persist tick data to storage (simplified implementation)"""
        try:
            # In a real implementation, this would write to database or file
            # For now, just log periodic saves
            if self.performance_stats['ticks_processed'] % 1000 == 0:
                self.logger.debug(f"Would persist 1000 ticks for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error persisting tick data: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        try:
            uptime = None
            if self.performance_stats['uptime_start']:
                uptime_delta = datetime.now() - self.performance_stats['uptime_start']
                uptime = uptime_delta.total_seconds()
            
            return {
                'ticks_processed': self.performance_stats['ticks_processed'],
                'bars_generated': self.performance_stats['bars_generated'],
                'processing_errors': self.performance_stats['processing_errors'],
                'data_gaps': self.performance_stats['data_gaps'],
                'uptime_seconds': uptime,
                'symbols_active': len(self.active_symbols),
                'subscribers_total': sum(len(subs) for subs in self.data_subscribers.values()),
                'queue_size': self.tick_queue.qsize(),
                'streaming_active': self.streaming_active,
                'last_tick_time': self.performance_stats['last_tick_time'],
                'avg_quality_score': sum(q.get_quality_score() for q in self.quality_metrics.values()) / max(len(self.quality_metrics), 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    def get_status(self):
        """Get current manager status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'symbols_configured': self.symbols,
            'symbols_active': list(self.active_symbols),
            'streaming_active': self.streaming_active,
            'processing_active': self.is_processing,
            'data_sources': list(self.feed_priorities.keys()),
            'timeframes_available': list(self.timeframes.keys()),
            'subscribers_count': {symbol: len(subs) for symbol, subs in self.data_subscribers.items()},
            'performance': self.get_performance_metrics()
        }
    
    def shutdown(self):
        """Clean shutdown of market data manager"""
        try:
            self.logger.info("Shutting down Market Data Manager...")
            
            # Stop streaming
            self.stop_streaming()
            
            # Stop processing
            self.is_processing = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            
            # Save final metrics
            final_metrics = self.get_performance_metrics()
            self.logger.info(f"Final metrics: {final_metrics}")
            
            # Clear data structures
            self.tick_data.clear()
            self.ohlc_data.clear()
            self.data_subscribers.clear()
            self.quality_metrics.clear()
            
            # Clear queue
            while not self.tick_queue.empty():
                try:
                    self.tick_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.status = "SHUTDOWN"
            self.logger.info("Market Data Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Agent test
if __name__ == "__main__":
    # Test the market data manager
    print("Testing AGENT_07: Market Data Manager")
    print("=" * 40)
    
    # Create manager for multiple symbols
    symbols = ["EURUSD", "GBPUSD"]
    manager = MarketDataManager(symbols)
    result = manager.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test streaming
        print("\nTesting data streaming...")
        stream_result = manager.start_streaming(["EURUSD"])
        print(f"Streaming result: {stream_result}")
        
        # Let it run for a few seconds
        time.sleep(3)
        
        # Check data
        print("\nTesting data retrieval...")
        latest_tick = manager.get_latest_tick("EURUSD")
        print(f"Latest tick: {latest_tick}")
        
        tick_history = manager.get_tick_history("EURUSD", 5)
        print(f"Tick history: {len(tick_history)} ticks")
        
        # Test data quality
        quality = manager.get_data_quality("EURUSD")
        print(f"Data quality: {quality}")
        
        # Test market summary
        summary = manager.get_market_summary()
        print(f"Market summary: {summary}")
        
        # Test performance metrics
        metrics = manager.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        # Test status
        status = manager.get_status()
        print(f"\nStatus: {status}")
        
        # Stop streaming and shutdown
        print("\nShutting down...")
        manager.stop_streaming()
        manager.shutdown()
        
    print("Market Data Manager test completed")