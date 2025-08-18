"""
Enhanced Market Microstructure Analyzer
Integrated from MQL5 Expert Advisor - Advanced Microstructure Analysis
"""

import numpy as np
import pandas as pd
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    QUIET = "QUIET"
    TRENDING = "TRENDING"
    VOLATILE = "VOLATILE"
    REVERSAL = "REVERSAL"
    BREAKOUT = "BREAKOUT"

@dataclass
class MarketMicrostructure:
    """Enhanced Market Microstructure Data"""
    # Basic microstructure
    bid_ask_spread: float = 0.0
    market_depth: float = 0.0
    volume_imbalance: float = 0.0
    tick_direction: float = 0.0
    slippage_tracking: float = 0.0
    avg_spread: float = 0.0
    last_update: datetime = None
    
    # Advanced microstructure analysis
    order_book_depth: List[float] = None
    volume_at_price: List[float] = None
    market_impact: float = 0.0
    liquidity_provision: float = 0.0
    high_frequency_activity: float = 0.0
    algorithmic_trading_detection: float = 0.0
    dark_pool_activity: float = 0.0
    iceberg_order_detection: float = 0.0
    manipulation_detection: float = 0.0
    front_running_detection: float = 0.0
    latency_arbitrage: float = 0.0
    market_efficiency: float = 0.0
    price_discovery: float = 0.0
    trading_cost_analysis: float = 0.0
    execution_quality: float = 0.0
    market_resilience: float = 0.0
    toxic_flow: float = 0.0
    flash_crash_risk: bool = False
    market_stress: float = 0.0
    
    def __post_init__(self):
        if self.order_book_depth is None:
            self.order_book_depth = [0.0] * 10
        if self.volume_at_price is None:
            self.volume_at_price = [0.0] * 20
        if self.last_update is None:
            self.last_update = datetime.now()

@dataclass
class IntermarketAnalysis:
    """Enhanced Intermarket Analysis"""
    dxy_correlation: float = 0.0
    gold_correlation: float = 0.0
    oil_correlation: float = 0.0
    bond_yield_correlation: float = 0.0
    stock_market_correlation: float = 0.0
    last_update: datetime = None
    
    # Enhanced correlations
    commodity_correlations: List[float] = None
    bond_spread_analysis: List[float] = None
    yield_curve_analysis: List[float] = None
    sector_rotation_analysis: List[float] = None
    global_equity_correlations: List[float] = None
    crypto_correlations: List[float] = None
    real_estate_correlations: List[float] = None
    inflation_correlations: List[float] = None
    central_bank_policy: List[float] = None
    economic_surprise_index: List[float] = None
    global_risk_appetite: float = 0.0
    carry_trade_analysis: List[float] = None
    flow_of_funds_analysis: List[float] = None
    cross_asset_momentum: List[float] = None
    macro_regime_detection: float = 0.0
    systemic_risk_indicators: List[float] = None
    market_contagion_risk: bool = False
    cross_asset_volatility: List[float] = None
    intermarket_signals: List[str] = None
    
    def __post_init__(self):
        if self.commodity_correlations is None:
            self.commodity_correlations = [0.0] * 10
        if self.bond_spread_analysis is None:
            self.bond_spread_analysis = [0.0] * 5
        if self.yield_curve_analysis is None:
            self.yield_curve_analysis = [0.0] * 10
        if self.sector_rotation_analysis is None:
            self.sector_rotation_analysis = [0.0] * 12
        if self.global_equity_correlations is None:
            self.global_equity_correlations = [0.0] * 20
        if self.crypto_correlations is None:
            self.crypto_correlations = [0.0] * 5
        if self.real_estate_correlations is None:
            self.real_estate_correlations = [0.0] * 3
        if self.inflation_correlations is None:
            self.inflation_correlations = [0.0] * 5
        if self.central_bank_policy is None:
            self.central_bank_policy = [0.0] * 8
        if self.economic_surprise_index is None:
            self.economic_surprise_index = [0.0] * 8
        if self.carry_trade_analysis is None:
            self.carry_trade_analysis = [0.0] * 10
        if self.flow_of_funds_analysis is None:
            self.flow_of_funds_analysis = [0.0] * 8
        if self.cross_asset_momentum is None:
            self.cross_asset_momentum = [0.0] * 15
        if self.systemic_risk_indicators is None:
            self.systemic_risk_indicators = [0.0] * 10
        if self.cross_asset_volatility is None:
            self.cross_asset_volatility = [0.0] * 15
        if self.intermarket_signals is None:
            self.intermarket_signals = [""] * 20
        if self.last_update is None:
            self.last_update = datetime.now()

class EnhancedMarketMicrostructureAnalyzer:
    """Enhanced Market Microstructure Analyzer based on MQL5 Expert Advisor"""
    
    def __init__(self, symbol: str):
        self.name = "ENHANCED_MICROSTRUCTURE_ANALYZER"
        self.version = "1.0.0"
        self.symbol = symbol
        self.microstructure = MarketMicrostructure()
        self.intermarket = IntermarketAnalysis()
        self.market_regime = MarketRegime.QUIET
        
        # Historical data storage
        self.tick_history = []
        self.volume_history = []
        self.spread_history = []
        self.hft_activity_tracker = []
        
        # Analysis parameters
        self.analysis_window = 1000  # Number of ticks to analyze
        self.hft_threshold = 100.0   # Ticks per second threshold for HFT
        self.manipulation_threshold = 3.0  # Standard deviations for manipulation
        
        # Threading
        self.analysis_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self) -> Dict:
        """Initialize the enhanced microstructure analyzer"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version} for {self.symbol}")
            
            # Initialize historical data structures
            self.tick_history = []
            self.volume_history = []
            self.spread_history = []
            self.hft_activity_tracker = []
            
            # Start analysis thread
            self.is_running = True
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.analysis_thread.start()
            
            self.logger.info("Enhanced Market Microstructure Analyzer initialized successfully")
            return {
                "status": "initialized",
                "agent": "ENHANCED_MICROSTRUCTURE_ANALYZER",
                "symbol": self.symbol,
                "analysis_window": self.analysis_window,
                "features": [
                    "High-Frequency Activity Detection",
                    "Dark Pool Analysis",
                    "Iceberg Order Detection",
                    "Market Manipulation Detection",
                    "Liquidity Provision Analysis",
                    "Execution Quality Metrics",
                    "Intermarket Analysis",
                    "Algorithmic Trading Detection"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return {"status": "failed", "agent": "ENHANCED_MICROSTRUCTURE_ANALYZER", "error": str(e)}
    
    def update_tick_data(self, bid: float, ask: float, volume: float, timestamp: datetime = None) -> None:
        """Update with new tick data for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            # Store tick data
            tick_data = {
                'timestamp': timestamp,
                'bid': bid,
                'ask': ask,
                'spread': ask - bid,
                'volume': volume,
                'mid_price': (bid + ask) / 2
            }
            
            self.tick_history.append(tick_data)
            
            # Maintain analysis window
            if len(self.tick_history) > self.analysis_window:
                self.tick_history.pop(0)
            
            # Update basic microstructure data
            self.microstructure.bid_ask_spread = ask - bid
            self.microstructure.last_update = timestamp
            
            # Calculate running averages
            if len(self.tick_history) > 10:
                recent_spreads = [t['spread'] for t in self.tick_history[-10:]]
                self.microstructure.avg_spread = np.mean(recent_spreads)
    
    def monitor_high_frequency_activity(self) -> float:
        """Monitor for High Frequency Trading Activity - Enhanced from MQL5"""
        if len(self.tick_history) < 100:
            return 0.0
        
        try:
            # Calculate tick frequency over recent period
            recent_ticks = self.tick_history[-100:]
            if len(recent_ticks) < 2:
                return 0.0
            
            time_span = (recent_ticks[-1]['timestamp'] - recent_ticks[0]['timestamp']).total_seconds()
            if time_span <= 0:
                return 0.0
            
            ticks_per_second = len(recent_ticks) / time_span
            
            # Detect rapid quote changes (HFT signature)
            rapid_quote_changes = 0
            for i in range(1, len(recent_ticks)):
                prev_tick = recent_ticks[i-1]
                curr_tick = recent_ticks[i]
                
                bid_changed = abs(curr_tick['bid'] - prev_tick['bid']) > 0
                ask_changed = abs(curr_tick['ask'] - prev_tick['ask']) > 0
                
                if bid_changed or ask_changed:
                    rapid_quote_changes += 1
            
            # Calculate HFT activity score
            hft_score = 0.0
            
            # High tick frequency
            if ticks_per_second > 10.0:
                hft_score += 2.0
            
            # Rapid quote changes
            if rapid_quote_changes > 50:
                hft_score += 5.0
            
            # Microsecond-level activity patterns
            if ticks_per_second > self.hft_threshold:
                hft_score += 3.0
            
            # Apply decay
            self.microstructure.high_frequency_activity = min(100.0, 
                self.microstructure.high_frequency_activity * 0.95 + hft_score)
            
            if self.microstructure.high_frequency_activity > 50.0:
                self.logger.info(f"High frequency trading activity detected: "
                               f"{self.microstructure.high_frequency_activity:.1f}%")
                self.logger.info(f"   Ticks per second: {ticks_per_second:.2f}")
                self.logger.info(f"   Rapid quote changes: {rapid_quote_changes}")
            
            return self.microstructure.high_frequency_activity
            
        except Exception as e:
            self.logger.error(f"HFT monitoring error: {e}")
            return 0.0
    
    def analyze_dark_pool_activity(self) -> float:
        """Analyze Dark Pool Activity - Enhanced from MQL5"""
        if len(self.tick_history) < 50:
            return 0.0
        
        try:
            recent_ticks = self.tick_history[-50:]
            
            # Detect dark pool signatures
            dark_pool_score = 0.0
            
            # Large volume without significant price movement
            volumes = [t['volume'] for t in recent_ticks]
            prices = [t['mid_price'] for t in recent_ticks]
            
            avg_volume = np.mean(volumes)
            price_volatility = np.std(prices) if len(prices) > 1 else 0.0
            
            # High volume, low volatility indicates dark pool activity
            if avg_volume > 0 and price_volatility > 0:
                volume_to_volatility_ratio = avg_volume / price_volatility
                
                if volume_to_volatility_ratio > 1000:  # Threshold for dark pool activity
                    dark_pool_score += 8.0
                elif volume_to_volatility_ratio > 500:
                    dark_pool_score += 5.0
                elif volume_to_volatility_ratio > 200:
                    dark_pool_score += 6.0
            
            # Apply decay and bounds
            self.microstructure.dark_pool_activity = min(100.0,
                self.microstructure.dark_pool_activity * 0.92 + dark_pool_score)
            
            if self.microstructure.dark_pool_activity > 30.0:
                self.logger.info(f"Dark pool activity detected: "
                               f"{self.microstructure.dark_pool_activity:.1f}%")
            
            return self.microstructure.dark_pool_activity
            
        except Exception as e:
            self.logger.error(f"Dark pool analysis error: {e}")
            return 0.0
    
    def detect_iceberg_orders(self) -> float:
        """Detect Iceberg Orders - Enhanced from MQL5"""
        if len(self.tick_history) < 100:
            return 0.0
        
        try:
            recent_ticks = self.tick_history[-100:]
            
            # Iceberg order detection
            iceberg_score = 0.0
            
            # Look for repeated volume at same price levels
            price_volume_map = {}
            for tick in recent_ticks:
                price_key = round(tick['mid_price'], 5)
                if price_key not in price_volume_map:
                    price_volume_map[price_key] = []
                price_volume_map[price_key].append(tick['volume'])
            
            # Detect consistent volume patterns (iceberg signature)
            for price, volumes in price_volume_map.items():
                if len(volumes) > 5:  # Multiple hits at same price
                    volume_consistency = 1.0 - (np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0.0
                    
                    if volume_consistency > 0.8:  # High consistency indicates iceberg
                        iceberg_score += len(volumes) * volume_consistency
            
            # Apply decay and update
            self.microstructure.iceberg_order_detection = min(100.0,
                self.microstructure.iceberg_order_detection * 0.95 + iceberg_score * 0.1)
            
            if self.microstructure.iceberg_order_detection > 40.0:
                self.logger.info(f"Iceberg order activity detected: "
                               f"{self.microstructure.iceberg_order_detection:.1f}%")
            
            return self.microstructure.iceberg_order_detection
            
        except Exception as e:
            self.logger.error(f"Iceberg order detection error: {e}")
            return 0.0
    
    def calculate_liquidity_metrics(self) -> Dict:
        """Calculate Liquidity Metrics - Enhanced from MQL5"""
        if len(self.tick_history) < 20:
            return {"status": "insufficient_data"}
        
        try:
            recent_ticks = self.tick_history[-20:]
            
            # Calculate relative spread
            spreads = [t['spread'] for t in recent_ticks]
            prices = [t['mid_price'] for t in recent_ticks]
            
            avg_spread = np.mean(spreads)
            avg_price = np.mean(prices)
            relative_spread = (avg_spread / avg_price) * 10000 if avg_price > 0 else 0.0  # in basis points
            
            # Volume analysis
            volumes = [t['volume'] for t in recent_ticks]
            avg_volume = np.mean(volumes)
            volume_volatility = np.std(volumes) if len(volumes) > 1 else 1.0
            
            # Market depth analysis (simulated from available data)
            market_depth_score = 0.0
            for i, depth in enumerate(self.microstructure.order_book_depth):
                liquidity_decay = 0.9 ** i
                market_depth_score += depth * liquidity_decay
            
            # Price impact estimation
            if len(recent_ticks) > 1:
                price_changes = [abs(recent_ticks[i]['mid_price'] - recent_ticks[i-1]['mid_price']) 
                               for i in range(1, len(recent_ticks))]
                avg_price_change = np.mean(price_changes)
                price_impact = avg_price_change / avg_volume if avg_volume > 0 else 0.0
            else:
                price_impact = 0.0
            
            # Liquidity provision score calculation
            spread_score = max(0.0, 100.0 - relative_spread)
            volume_score = min(100.0, 50.0 + (avg_volume / volume_volatility)) if volume_volatility > 0 else 50.0
            impact_score = max(0.0, 100.0 - (price_impact * 1000000.0)) if price_impact > 0 else 100.0
            
            self.microstructure.liquidity_provision = (spread_score * 0.4 + volume_score * 0.3 + 
                                                     market_depth_score * 0.2 + impact_score * 0.1)
            self.microstructure.liquidity_provision = max(0.0, min(100.0, self.microstructure.liquidity_provision))
            
            # Market impact and trading cost analysis
            self.microstructure.market_impact = price_impact * 1000000.0
            self.microstructure.trading_cost_analysis = relative_spread + (self.microstructure.market_impact * 0.1)
            
            # Execution quality metrics
            execution_score = 100.0 - self.microstructure.trading_cost_analysis
            self.microstructure.execution_quality = max(0.0, min(100.0, execution_score))
            
            self.logger.info(f"Liquidity metrics calculated:")
            self.logger.info(f"   Relative spread: {relative_spread:.2f} bps")
            self.logger.info(f"   Volume volatility: {volume_volatility:.2f}")
            self.logger.info(f"   Price impact: {price_impact * 1000000:.4f}")
            self.logger.info(f"   Liquidity score: {self.microstructure.liquidity_provision:.1f}%")
            self.logger.info(f"   Execution quality: {self.microstructure.execution_quality:.1f}%")
            
            return {
                "status": "success",
                "relative_spread_bps": relative_spread,
                "volume_volatility": volume_volatility,
                "price_impact": price_impact,
                "liquidity_score": self.microstructure.liquidity_provision,
                "execution_quality": self.microstructure.execution_quality,
                "trading_cost": self.microstructure.trading_cost_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Liquidity metrics calculation error: {e}")
            return {"status": "error", "message": str(e)}
    
    def detect_market_manipulation(self) -> float:
        """Detect Market Manipulation - Enhanced from MQL5"""
        if len(self.tick_history) < 50:
            return 0.0
        
        try:
            recent_ticks = self.tick_history[-50:]
            
            manipulation_score = 0.0
            
            # Detect unusual price patterns
            prices = [t['mid_price'] for t in recent_ticks]
            volumes = [t['volume'] for t in recent_ticks]
            
            # Calculate price z-score
            if len(prices) > 1:
                price_mean = np.mean(prices)
                price_std = np.std(prices)
                
                if price_std > 0:
                    latest_price_zscore = abs(prices[-1] - price_mean) / price_std
                    
                    # Unusual price movements
                    if latest_price_zscore > self.manipulation_threshold:
                        manipulation_score += latest_price_zscore * 10
            
            # Detect volume anomalies
            if len(volumes) > 1:
                volume_mean = np.mean(volumes)
                volume_std = np.std(volumes)
                
                if volume_std > 0:
                    latest_volume_zscore = abs(volumes[-1] - volume_mean) / volume_std
                    
                    # Unusual volume spikes
                    if latest_volume_zscore > self.manipulation_threshold:
                        manipulation_score += latest_volume_zscore * 5
            
            # Check for price-volume divergence
            if len(prices) > 10 and len(volumes) > 10:
                price_correlation = np.corrcoef(prices[-10:], volumes[-10:])[0, 1]
                
                # Unusual price-volume relationships
                if abs(price_correlation) > 0.8:  # Either very high positive or negative correlation
                    manipulation_score += abs(price_correlation) * 15
            
            # Apply decay and update
            self.microstructure.manipulation_detection = min(100.0,
                self.microstructure.manipulation_detection * 0.90 + manipulation_score * 0.1)
            
            if self.microstructure.manipulation_detection > 30.0:
                self.logger.warning(f"Market manipulation detected: "
                                  f"{self.microstructure.manipulation_detection:.1f}%")
            
            return self.microstructure.manipulation_detection
            
        except Exception as e:
            self.logger.error(f"Market manipulation detection error: {e}")
            return 0.0
    
    def analyze_algorithmic_trading(self) -> float:
        """Analyze Algorithmic Trading Activity - Enhanced from MQL5"""
        if len(self.tick_history) < 100:
            return 0.0
        
        try:
            recent_ticks = self.tick_history[-100:]
            
            algo_score = 0.0
            
            # Time-based patterns (algorithmic signature)
            timestamps = [t['timestamp'] for t in recent_ticks]
            time_intervals = []
            
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_intervals.append(interval)
            
            if time_intervals:
                # Regular time intervals suggest algorithmic trading
                interval_std = np.std(time_intervals)
                interval_mean = np.mean(time_intervals)
                
                if interval_mean > 0:
                    regularity_score = 1.0 - (interval_std / interval_mean)
                    
                    if regularity_score > 0.8:  # High regularity
                        algo_score += regularity_score * 20
            
            # Volume patterns
            volumes = [t['volume'] for t in recent_ticks]
            
            # Check for consistent volume sizes (algorithmic signature)
            unique_volumes = len(set(volumes))
            total_volumes = len(volumes)
            
            if total_volumes > 0:
                volume_diversity = unique_volumes / total_volumes
                
                if volume_diversity < 0.3:  # Low diversity suggests algo trading
                    algo_score += (1.0 - volume_diversity) * 25
            
            # Price increment patterns
            spreads = [t['spread'] for t in recent_ticks]
            
            # Consistent spread patterns
            if spreads:
                spread_consistency = 1.0 - (np.std(spreads) / np.mean(spreads)) if np.mean(spreads) > 0 else 0.0
                
                if spread_consistency > 0.9:  # Very consistent spreads
                    algo_score += spread_consistency * 15
            
            # Update algorithmic trading detection
            self.microstructure.algorithmic_trading_detection = min(100.0,
                self.microstructure.algorithmic_trading_detection * 0.95 + algo_score * 0.1)
            
            if self.microstructure.algorithmic_trading_detection > 40.0:
                self.logger.info(f"Algorithmic trading activity detected: "
                               f"{self.microstructure.algorithmic_trading_detection:.1f}%")
            
            return self.microstructure.algorithmic_trading_detection
            
        except Exception as e:
            self.logger.error(f"Algorithmic trading analysis error: {e}")
            return 0.0
    
    def _analysis_loop(self) -> None:
        """Main analysis loop"""
        while self.is_running:
            try:
                if len(self.tick_history) > 10:
                    # Run all microstructure analyses
                    self.monitor_high_frequency_activity()
                    self.analyze_dark_pool_activity()
                    self.detect_iceberg_orders()
                    self.calculate_liquidity_metrics()
                    self.detect_market_manipulation()
                    self.analyze_algorithmic_trading()
                    
                    # Update market regime
                    self._update_market_regime()
                
                time.sleep(1)  # Analysis every second
                
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                time.sleep(5)
    
    def _update_market_regime(self) -> None:
        """Update market regime based on microstructure analysis"""
        try:
            # Determine market regime based on microstructure indicators
            hft_activity = self.microstructure.high_frequency_activity
            volatility = np.std([t['mid_price'] for t in self.tick_history[-20:]]) if len(self.tick_history) >= 20 else 0.0
            manipulation = self.microstructure.manipulation_detection
            
            if manipulation > 50.0:
                self.market_regime = MarketRegime.VOLATILE
            elif hft_activity > 70.0:
                self.market_regime = MarketRegime.TRENDING
            elif volatility > 0.001:  # High volatility threshold
                self.market_regime = MarketRegime.VOLATILE
            elif hft_activity > 30.0:
                self.market_regime = MarketRegime.BREAKOUT
            else:
                self.market_regime = MarketRegime.QUIET
                
        except Exception as e:
            self.logger.error(f"Market regime update error: {e}")
    
    def get_microstructure_analysis(self) -> Dict:
        """Get comprehensive microstructure analysis"""
        return {
            "status": "success",
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "market_regime": self.market_regime.value,
            "microstructure": {
                "bid_ask_spread": self.microstructure.bid_ask_spread,
                "avg_spread": self.microstructure.avg_spread,
                "liquidity_provision": self.microstructure.liquidity_provision,
                "execution_quality": self.microstructure.execution_quality,
                "market_impact": self.microstructure.market_impact,
                "trading_cost": self.microstructure.trading_cost_analysis
            },
            "activity_detection": {
                "high_frequency_activity": self.microstructure.high_frequency_activity,
                "algorithmic_trading": self.microstructure.algorithmic_trading_detection,
                "dark_pool_activity": self.microstructure.dark_pool_activity,
                "iceberg_orders": self.microstructure.iceberg_order_detection,
                "market_manipulation": self.microstructure.manipulation_detection
            },
            "market_quality": {
                "market_efficiency": self.microstructure.market_efficiency,
                "price_discovery": self.microstructure.price_discovery,
                "market_resilience": self.microstructure.market_resilience,
                "flash_crash_risk": self.microstructure.flash_crash_risk,
                "market_stress": self.microstructure.market_stress
            }
        }
    
    def get_status(self) -> Dict:
        """Get analyzer status"""
        return {
            "name": self.name,
            "version": self.version,
            "symbol": self.symbol,
            "is_running": self.is_running,
            "tick_count": len(self.tick_history),
            "market_regime": self.market_regime.value,
            "last_update": self.microstructure.last_update.isoformat() if self.microstructure.last_update else None
        }
    
    def shutdown(self) -> None:
        """Shutdown the analyzer"""
        try:
            self.logger.info("Shutting down Enhanced Market Microstructure Analyzer...")
            self.is_running = False
            
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=5)
            
            self.logger.info("Enhanced Market Microstructure Analyzer shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

def test_microstructure_analyzer():
    """Test the enhanced microstructure analyzer"""
    print("Testing Enhanced Market Microstructure Analyzer...")
    print("=" * 60)
    
    analyzer = EnhancedMarketMicrostructureAnalyzer("EURUSD")
    
    # Test initialization
    result = analyzer.initialize()
    print(f"Initialization: {result['status']}")
    
    if result['status'] == 'initialized':
        # Simulate some tick data
        import random
        base_price = 1.17000
        
        for i in range(100):
            bid = base_price + random.uniform(-0.0001, 0.0001)
            ask = bid + random.uniform(0.00001, 0.00005)
            volume = random.uniform(0.1, 10.0)
            
            analyzer.update_tick_data(bid, ask, volume)
            time.sleep(0.01)  # Small delay
        
        # Get analysis
        analysis = analyzer.get_microstructure_analysis()
        print(f"\nMarket Regime: {analysis['market_regime']}")
        print(f"Liquidity Score: {analysis['microstructure']['liquidity_provision']:.1f}%")
        print(f"HFT Activity: {analysis['activity_detection']['high_frequency_activity']:.1f}%")
        print(f"Execution Quality: {analysis['microstructure']['execution_quality']:.1f}%")
        
        print("\n[OK] Enhanced Microstructure Analyzer Test PASSED!")
        return True
    else:
        print(f"[FAIL] Test failed: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    test_microstructure_analyzer()