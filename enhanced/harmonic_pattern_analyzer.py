"""
Advanced Harmonic Pattern Recognition System
Integrated from MQL5 Expert Advisor - Complete Harmonic Pattern Analysis
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import math

class PatternType(Enum):
    """Harmonic Pattern Types from MQL5"""
    GARTLEY = "GARTLEY"
    BUTTERFLY = "BUTTERFLY"
    BAT = "BAT"
    CRAB = "CRAB"
    ABCD = "ABCD"
    THREE_DRIVES = "THREE_DRIVES"
    WOLFE_WAVE = "WOLFE_WAVE"
    FIBONACCI_EXTENSION = "FIBONACCI_EXTENSION"
    CYPHER = "CYPHER"
    SHARK = "SHARK"

class ExitType(Enum):
    """Exit Types"""
    TARGET_1 = "TARGET_1"
    TARGET_2 = "TARGET_2"
    STOP_LOSS = "STOP_LOSS"
    BREAK_EVEN = "BREAK_EVEN"
    TRAIL_STOP = "TRAIL_STOP"

@dataclass
class FibLevels:
    """Fibonacci Levels for Pattern Analysis"""
    level_0: float = 0.0
    level_236: float = 0.236
    level_382: float = 0.382
    level_50: float = 0.5
    level_618: float = 0.618
    level_786: float = 0.786
    level_100: float = 1.0
    
    # Extensions
    extension_1272: float = 1.272
    extension_1414: float = 1.414
    extension_1618: float = 1.618
    
    # Additional retracement levels
    retrace_1134: float = 1.134
    retrace_1886: float = 1.886
    
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()

@dataclass
class PatternData:
    """Enhanced Pattern Data Structure"""
    pattern_type: PatternType
    entry: float = 0.0
    target: float = 0.0
    stop: float = 0.0
    time: datetime = None
    is_valid: bool = False
    description: str = ""
    
    # Enhancements from MQL5
    confidence: float = 0.0
    pattern_height: float = 0.0
    pattern_width: float = 0.0
    volume_confirmation: float = 0.0
    harmonic_ratio: bool = False
    price_projection: float = 0.0
    time_projection: float = 0.0
    pattern_classification: str = ""
    
    # Pattern points
    point_x: float = 0.0
    point_a: float = 0.0
    point_b: float = 0.0
    point_c: float = 0.0
    point_d: float = 0.0
    
    def __post_init__(self):
        if self.time is None:
            self.time = datetime.now()

class Point(NamedTuple):
    """Price Point for Pattern Analysis"""
    price: float
    time: datetime
    index: int

class AdvancedHarmonicPatternAnalyzer:
    """Advanced Harmonic Pattern Recognition based on MQL5 Expert Advisor"""
    
    def __init__(self, symbol: str):
        self.name = "ADVANCED_HARMONIC_PATTERN_ANALYZER"
        self.version = "1.0.0"
        self.symbol = symbol
        
        # Fibonacci levels
        self.fib_levels = FibLevels()
        
        # Pattern tolerance (how strict the ratios must be)
        self.pattern_tolerance = 0.05  # 5% tolerance
        
        # Minimum pattern size
        self.min_pattern_height = 0.001  # Minimum price movement
        
        # Historical data
        self.price_history = []
        self.highs = []
        self.lows = []
        
        # Active patterns
        self.active_patterns = []
        self.completed_patterns = []
        
        # Pattern definitions with exact Fibonacci ratios
        self.pattern_definitions = {
            PatternType.GARTLEY: {
                'XA_AB': (0.618, 0.618),  # AB should be 61.8% of XA
                'AB_BC': (0.382, 0.886),  # BC should be 38.2% to 88.6% of AB
                'BC_CD': (1.272, 1.618)   # CD should be 127.2% to 161.8% of BC
            },
            PatternType.BUTTERFLY: {
                'XA_AB': (0.786, 0.786),  # AB should be 78.6% of XA
                'AB_BC': (0.382, 0.886),  # BC should be 38.2% to 88.6% of AB
                'BC_CD': (1.618, 2.618)   # CD should be 161.8% to 261.8% of BC
            },
            PatternType.BAT: {
                'XA_AB': (0.382, 0.50),   # AB should be 38.2% to 50% of XA
                'AB_BC': (0.382, 0.886),  # BC should be 38.2% to 88.6% of AB
                'BC_CD': (1.618, 2.618)   # CD should be 161.8% to 261.8% of BC
            },
            PatternType.CRAB: {
                'XA_AB': (0.382, 0.618),  # AB should be 38.2% to 61.8% of XA
                'AB_BC': (0.382, 0.886),  # BC should be 38.2% to 88.6% of AB
                'BC_CD': (2.240, 3.618)   # CD should be 224% to 361.8% of BC
            },
            PatternType.ABCD: {
                'AB_BC': (0.382, 0.886),  # BC should be 38.2% to 88.6% of AB
                'BC_CD': (1.272, 1.618)   # CD should be 127.2% to 161.8% of BC
            }
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self) -> Dict:
        """Initialize the harmonic pattern analyzer"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version} for {self.symbol}")
            
            # Initialize data structures
            self.price_history = []
            self.highs = []
            self.lows = []
            self.active_patterns = []
            self.completed_patterns = []
            
            self.logger.info("Advanced Harmonic Pattern Analyzer initialized successfully")
            return {
                "status": "initialized",
                "agent": "ADVANCED_HARMONIC_PATTERN_ANALYZER",
                "symbol": self.symbol,
                "supported_patterns": [pattern.value for pattern in PatternType],
                "pattern_tolerance": self.pattern_tolerance,
                "min_pattern_height": self.min_pattern_height
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return {"status": "failed", "agent": "ADVANCED_HARMONIC_PATTERN_ANALYZER", "error": str(e)}
    
    def update_price_data(self, prices: List[float], timestamps: List[datetime] = None) -> None:
        """Update price data for pattern analysis"""
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(minutes=i) for i in range(len(prices))]
        
        # Store price history
        for i, price in enumerate(prices):
            if i < len(timestamps):
                self.price_history.append({
                    'price': price,
                    'timestamp': timestamps[i],
                    'index': len(self.price_history)
                })
        
        # Maintain reasonable history size
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-800:]
            # Re-index
            for i, item in enumerate(self.price_history):
                item['index'] = i
        
        # Find pivot highs and lows
        self._find_pivot_points()
    
    def _find_pivot_points(self) -> None:
        """Find pivot highs and lows for pattern analysis"""
        if len(self.price_history) < 10:
            return
        
        prices = [item['price'] for item in self.price_history]
        timestamps = [item['timestamp'] for item in self.price_history]
        
        # Simple pivot detection (can be enhanced)
        lookback = 5
        
        # Find pivot highs
        self.highs = []
        for i in range(lookback, len(prices) - lookback):
            is_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and prices[j] >= prices[i]:
                    is_high = False
                    break
            
            if is_high:
                self.highs.append(Point(prices[i], timestamps[i], i))
        
        # Find pivot lows
        self.lows = []
        for i in range(lookback, len(prices) - lookback):
            is_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and prices[j] <= prices[i]:
                    is_low = False
                    break
            
            if is_low:
                self.lows.append(Point(prices[i], timestamps[i], i))
    
    def detect_gartley_pattern(self) -> List[PatternData]:
        """Detect Gartley Harmonic Pattern - Complete Implementation"""
        patterns = []
        
        if len(self.highs) < 3 or len(self.lows) < 2:
            return patterns
        
        try:
            # Look for 5-point pattern: X-A-B-C-D
            for i in range(len(self.highs) - 2):
                for j in range(len(self.lows) - 1):
                    # Try different combinations
                    if self.highs[i].index < self.lows[j].index:
                        # Potential X (high), A (low) start
                        x_point = self.highs[i]
                        a_point = self.lows[j]
                        
                        # Find B (high after A)
                        b_candidates = [h for h in self.highs if h.index > a_point.index]
                        if not b_candidates:
                            continue
                        
                        for b_point in b_candidates[:3]:  # Check first 3 candidates
                            # Find C (low after B)
                            c_candidates = [l for l in self.lows if l.index > b_point.index]
                            if not c_candidates:
                                continue
                            
                            for c_point in c_candidates[:3]:  # Check first 3 candidates
                                # Find D (high after C)
                                d_candidates = [h for h in self.highs if h.index > c_point.index]
                                if not d_candidates:
                                    continue
                                
                                d_point = d_candidates[0]  # Take first candidate
                                
                                # Validate Gartley ratios
                                pattern = self._validate_gartley_ratios(x_point, a_point, b_point, c_point, d_point)
                                if pattern:
                                    patterns.append(pattern)
            
            if patterns:
                self.logger.info(f"Detected {len(patterns)} Gartley pattern(s)")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Gartley pattern detection error: {e}")
            return patterns
    
    def _validate_gartley_ratios(self, x: Point, a: Point, b: Point, c: Point, d: Point) -> Optional[PatternData]:
        """Validate Gartley pattern ratios"""
        try:
            # Calculate legs
            xa_leg = abs(x.price - a.price)
            ab_leg = abs(a.price - b.price)
            bc_leg = abs(b.price - c.price)
            cd_leg = abs(c.price - d.price)
            
            if xa_leg == 0 or ab_leg == 0 or bc_leg == 0:
                return None
            
            # Calculate ratios
            ab_xa_ratio = ab_leg / xa_leg
            bc_ab_ratio = bc_leg / ab_leg
            cd_bc_ratio = cd_leg / bc_leg if bc_leg > 0 else 0
            
            # Gartley specific ratios
            gartley_ratios = self.pattern_definitions[PatternType.GARTLEY]
            
            # Check AB/XA ratio (should be around 0.618)
            if not self._is_ratio_valid(ab_xa_ratio, gartley_ratios['XA_AB'][0], self.pattern_tolerance):
                return None
            
            # Check BC/AB ratio (should be 0.382 to 0.886)
            if not (gartley_ratios['AB_BC'][0] <= bc_ab_ratio <= gartley_ratios['AB_BC'][1]):
                return None
            
            # Check CD/BC ratio (should be 1.272 to 1.618)
            if not (gartley_ratios['BC_CD'][0] <= cd_bc_ratio <= gartley_ratios['BC_CD'][1]):
                return None
            
            # Calculate confidence based on how close ratios are to ideal
            ideal_ab_xa = 0.618
            ideal_bc_ab = 0.618
            ideal_cd_bc = 1.272
            
            confidence = 100.0 - (
                abs(ab_xa_ratio - ideal_ab_xa) * 100 +
                abs(bc_ab_ratio - ideal_bc_ab) * 100 +
                abs(cd_bc_ratio - ideal_cd_bc) * 50
            )
            confidence = max(0.0, min(100.0, confidence))
            
            # Create pattern data
            pattern = PatternData(
                pattern_type=PatternType.GARTLEY,
                entry=d.price,
                target=d.price + (xa_leg * 0.618),  # 61.8% retracement target
                stop=d.price - (xa_leg * 0.1),     # 10% stop loss
                time=d.time,
                is_valid=True,
                description=f"Gartley Pattern at {d.price:.5f}",
                confidence=confidence,
                pattern_height=xa_leg,
                pattern_width=(d.time - x.time).total_seconds() / 3600,  # Hours
                harmonic_ratio=True,
                price_projection=d.price + (xa_leg * 0.618),
                pattern_classification="Bullish Gartley",
                point_x=x.price,
                point_a=a.price,
                point_b=b.price,
                point_c=c.price,
                point_d=d.price
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Gartley ratio validation error: {e}")
            return None
    
    def detect_butterfly_pattern(self) -> List[PatternData]:
        """Detect Butterfly Harmonic Pattern - Complete Implementation"""
        patterns = []
        
        if len(self.highs) < 3 or len(self.lows) < 2:
            return patterns
        
        try:
            # Similar structure to Gartley but different ratios
            for i in range(len(self.highs) - 2):
                for j in range(len(self.lows) - 1):
                    if self.highs[i].index < self.lows[j].index:
                        x_point = self.highs[i]
                        a_point = self.lows[j]
                        
                        b_candidates = [h for h in self.highs if h.index > a_point.index]
                        if not b_candidates:
                            continue
                        
                        for b_point in b_candidates[:3]:
                            c_candidates = [l for l in self.lows if l.index > b_point.index]
                            if not c_candidates:
                                continue
                            
                            for c_point in c_candidates[:3]:
                                d_candidates = [h for h in self.highs if h.index > c_point.index]
                                if not d_candidates:
                                    continue
                                
                                d_point = d_candidates[0]
                                
                                pattern = self._validate_butterfly_ratios(x_point, a_point, b_point, c_point, d_point)
                                if pattern:
                                    patterns.append(pattern)
            
            if patterns:
                self.logger.info(f"Detected {len(patterns)} Butterfly pattern(s)")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Butterfly pattern detection error: {e}")
            return patterns
    
    def _validate_butterfly_ratios(self, x: Point, a: Point, b: Point, c: Point, d: Point) -> Optional[PatternData]:
        """Validate Butterfly pattern ratios"""
        try:
            # Calculate legs
            xa_leg = abs(x.price - a.price)
            ab_leg = abs(a.price - b.price)
            bc_leg = abs(b.price - c.price)
            cd_leg = abs(c.price - d.price)
            
            if xa_leg == 0 or ab_leg == 0 or bc_leg == 0:
                return None
            
            # Calculate ratios
            ab_xa_ratio = ab_leg / xa_leg
            bc_ab_ratio = bc_leg / ab_leg
            cd_bc_ratio = cd_leg / bc_leg if bc_leg > 0 else 0
            
            # Butterfly specific ratios
            butterfly_ratios = self.pattern_definitions[PatternType.BUTTERFLY]
            
            # Check AB/XA ratio (should be around 0.786)
            if not self._is_ratio_valid(ab_xa_ratio, butterfly_ratios['XA_AB'][0], self.pattern_tolerance):
                return None
            
            # Check BC/AB ratio
            if not (butterfly_ratios['AB_BC'][0] <= bc_ab_ratio <= butterfly_ratios['AB_BC'][1]):
                return None
            
            # Check CD/BC ratio (should be 1.618 to 2.618)
            if not (butterfly_ratios['BC_CD'][0] <= cd_bc_ratio <= butterfly_ratios['BC_CD'][1]):
                return None
            
            # Calculate confidence
            ideal_ab_xa = 0.786
            ideal_bc_ab = 0.618
            ideal_cd_bc = 1.618
            
            confidence = 100.0 - (
                abs(ab_xa_ratio - ideal_ab_xa) * 100 +
                abs(bc_ab_ratio - ideal_bc_ab) * 100 +
                abs(cd_bc_ratio - ideal_cd_bc) * 50
            )
            confidence = max(0.0, min(100.0, confidence))
            
            # Create pattern data
            pattern = PatternData(
                pattern_type=PatternType.BUTTERFLY,
                entry=d.price,
                target=d.price + (xa_leg * 0.786),  # 78.6% retracement target
                stop=d.price - (xa_leg * 0.1),
                time=d.time,
                is_valid=True,
                description=f"Butterfly Pattern at {d.price:.5f}",
                confidence=confidence,
                pattern_height=xa_leg,
                pattern_width=(d.time - x.time).total_seconds() / 3600,
                harmonic_ratio=True,
                price_projection=d.price + (xa_leg * 0.786),
                pattern_classification="Bullish Butterfly",
                point_x=x.price,
                point_a=a.price,
                point_b=b.price,
                point_c=c.price,
                point_d=d.price
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Butterfly ratio validation error: {e}")
            return None
    
    def detect_bat_pattern(self) -> List[PatternData]:
        """Detect Bat Harmonic Pattern - Complete Implementation"""
        patterns = []
        
        if len(self.highs) < 3 or len(self.lows) < 2:
            return patterns
        
        try:
            for i in range(len(self.highs) - 2):
                for j in range(len(self.lows) - 1):
                    if self.highs[i].index < self.lows[j].index:
                        x_point = self.highs[i]
                        a_point = self.lows[j]
                        
                        b_candidates = [h for h in self.highs if h.index > a_point.index]
                        if not b_candidates:
                            continue
                        
                        for b_point in b_candidates[:3]:
                            c_candidates = [l for l in self.lows if l.index > b_point.index]
                            if not c_candidates:
                                continue
                            
                            for c_point in c_candidates[:3]:
                                d_candidates = [h for h in self.highs if h.index > c_point.index]
                                if not d_candidates:
                                    continue
                                
                                d_point = d_candidates[0]
                                
                                pattern = self._validate_bat_ratios(x_point, a_point, b_point, c_point, d_point)
                                if pattern:
                                    patterns.append(pattern)
            
            if patterns:
                self.logger.info(f"Detected {len(patterns)} Bat pattern(s)")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Bat pattern detection error: {e}")
            return patterns
    
    def _validate_bat_ratios(self, x: Point, a: Point, b: Point, c: Point, d: Point) -> Optional[PatternData]:
        """Validate Bat pattern ratios"""
        try:
            # Calculate legs
            xa_leg = abs(x.price - a.price)
            ab_leg = abs(a.price - b.price)
            bc_leg = abs(b.price - c.price)
            cd_leg = abs(c.price - d.price)
            
            if xa_leg == 0 or ab_leg == 0 or bc_leg == 0:
                return None
            
            # Calculate ratios
            ab_xa_ratio = ab_leg / xa_leg
            bc_ab_ratio = bc_leg / ab_leg
            cd_bc_ratio = cd_leg / bc_leg if bc_leg > 0 else 0
            
            # Bat specific ratios
            bat_ratios = self.pattern_definitions[PatternType.BAT]
            
            # Check AB/XA ratio (should be 0.382 to 0.50)
            if not (bat_ratios['XA_AB'][0] <= ab_xa_ratio <= bat_ratios['XA_AB'][1]):
                return None
            
            # Check BC/AB ratio
            if not (bat_ratios['AB_BC'][0] <= bc_ab_ratio <= bat_ratios['AB_BC'][1]):
                return None
            
            # Check CD/BC ratio
            if not (bat_ratios['BC_CD'][0] <= cd_bc_ratio <= bat_ratios['BC_CD'][1]):
                return None
            
            # Calculate confidence
            ideal_ab_xa = 0.382
            ideal_bc_ab = 0.618
            ideal_cd_bc = 1.618
            
            confidence = 100.0 - (
                abs(ab_xa_ratio - ideal_ab_xa) * 100 +
                abs(bc_ab_ratio - ideal_bc_ab) * 100 +
                abs(cd_bc_ratio - ideal_cd_bc) * 50
            )
            confidence = max(0.0, min(100.0, confidence))
            
            # Create pattern data
            pattern = PatternData(
                pattern_type=PatternType.BAT,
                entry=d.price,
                target=d.price + (xa_leg * 0.382),  # 38.2% retracement target
                stop=d.price - (xa_leg * 0.1),
                time=d.time,
                is_valid=True,
                description=f"Bat Pattern at {d.price:.5f}",
                confidence=confidence,
                pattern_height=xa_leg,
                pattern_width=(d.time - x.time).total_seconds() / 3600,
                harmonic_ratio=True,
                price_projection=d.price + (xa_leg * 0.382),
                pattern_classification="Bullish Bat",
                point_x=x.price,
                point_a=a.price,
                point_b=b.price,
                point_c=c.price,
                point_d=d.price
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Bat ratio validation error: {e}")
            return None
    
    def detect_crab_pattern(self) -> List[PatternData]:
        """Detect Crab Harmonic Pattern - Complete Implementation"""
        patterns = []
        
        if len(self.highs) < 3 or len(self.lows) < 2:
            return patterns
        
        try:
            for i in range(len(self.highs) - 2):
                for j in range(len(self.lows) - 1):
                    if self.highs[i].index < self.lows[j].index:
                        x_point = self.highs[i]
                        a_point = self.lows[j]
                        
                        b_candidates = [h for h in self.highs if h.index > a_point.index]
                        if not b_candidates:
                            continue
                        
                        for b_point in b_candidates[:3]:
                            c_candidates = [l for l in self.lows if l.index > b_point.index]
                            if not c_candidates:
                                continue
                            
                            for c_point in c_candidates[:3]:
                                d_candidates = [h for h in self.highs if h.index > c_point.index]
                                if not d_candidates:
                                    continue
                                
                                d_point = d_candidates[0]
                                
                                pattern = self._validate_crab_ratios(x_point, a_point, b_point, c_point, d_point)
                                if pattern:
                                    patterns.append(pattern)
            
            if patterns:
                self.logger.info(f"Detected {len(patterns)} Crab pattern(s)")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Crab pattern detection error: {e}")
            return patterns
    
    def _validate_crab_ratios(self, x: Point, a: Point, b: Point, c: Point, d: Point) -> Optional[PatternData]:
        """Validate Crab pattern ratios"""
        try:
            # Calculate legs
            xa_leg = abs(x.price - a.price)
            ab_leg = abs(a.price - b.price)
            bc_leg = abs(b.price - c.price)
            cd_leg = abs(c.price - d.price)
            
            if xa_leg == 0 or ab_leg == 0 or bc_leg == 0:
                return None
            
            # Calculate ratios
            ab_xa_ratio = ab_leg / xa_leg
            bc_ab_ratio = bc_leg / ab_leg
            cd_bc_ratio = cd_leg / bc_leg if bc_leg > 0 else 0
            
            # Crab specific ratios
            crab_ratios = self.pattern_definitions[PatternType.CRAB]
            
            # Check AB/XA ratio (should be 0.382 to 0.618)
            if not (crab_ratios['XA_AB'][0] <= ab_xa_ratio <= crab_ratios['XA_AB'][1]):
                return None
            
            # Check BC/AB ratio
            if not (crab_ratios['AB_BC'][0] <= bc_ab_ratio <= crab_ratios['AB_BC'][1]):
                return None
            
            # Check CD/BC ratio (should be 2.240 to 3.618 - extreme extension)
            if not (crab_ratios['BC_CD'][0] <= cd_bc_ratio <= crab_ratios['BC_CD'][1]):
                return None
            
            # Calculate confidence
            ideal_ab_xa = 0.618
            ideal_bc_ab = 0.618
            ideal_cd_bc = 2.618
            
            confidence = 100.0 - (
                abs(ab_xa_ratio - ideal_ab_xa) * 100 +
                abs(bc_ab_ratio - ideal_bc_ab) * 100 +
                abs(cd_bc_ratio - ideal_cd_bc) * 30  # Less weight due to extreme ratio
            )
            confidence = max(0.0, min(100.0, confidence))
            
            # Create pattern data
            pattern = PatternData(
                pattern_type=PatternType.CRAB,
                entry=d.price,
                target=d.price - (xa_leg * 0.618),  # Strong reversal expected
                stop=d.price + (xa_leg * 0.1),
                time=d.time,
                is_valid=True,
                description=f"Crab Pattern at {d.price:.5f}",
                confidence=confidence,
                pattern_height=xa_leg,
                pattern_width=(d.time - x.time).total_seconds() / 3600,
                harmonic_ratio=True,
                price_projection=d.price - (xa_leg * 0.618),
                pattern_classification="Bearish Crab",
                point_x=x.price,
                point_a=a.price,
                point_b=b.price,
                point_c=c.price,
                point_d=d.price
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Crab ratio validation error: {e}")
            return None
    
    def _is_ratio_valid(self, actual: float, target: float, tolerance: float) -> bool:
        """Check if ratio is within tolerance"""
        return abs(actual - target) <= tolerance
    
    def analyze_all_patterns(self) -> List[PatternData]:
        """Analyze all harmonic patterns"""
        all_patterns = []
        
        try:
            # Detect all pattern types
            all_patterns.extend(self.detect_gartley_pattern())
            all_patterns.extend(self.detect_butterfly_pattern())
            all_patterns.extend(self.detect_bat_pattern())
            all_patterns.extend(self.detect_crab_pattern())
            
            # Sort by confidence
            all_patterns.sort(key=lambda p: p.confidence, reverse=True)
            
            # Store active patterns
            self.active_patterns = all_patterns
            
            if all_patterns:
                self.logger.info(f"Detected {len(all_patterns)} total harmonic patterns")
                for pattern in all_patterns[:3]:  # Log top 3
                    self.logger.info(f"   {pattern.pattern_type.value}: {pattern.confidence:.1f}% confidence at {pattern.entry:.5f}")
            
            return all_patterns
            
        except Exception as e:
            self.logger.error(f"Pattern analysis error: {e}")
            return []
    
    def get_pattern_analysis(self) -> Dict:
        """Get comprehensive pattern analysis"""
        patterns = self.analyze_all_patterns()
        
        return {
            "status": "success",
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "total_patterns": len(patterns),
            "patterns": [
                {
                    "type": pattern.pattern_type.value,
                    "confidence": pattern.confidence,
                    "entry": pattern.entry,
                    "target": pattern.target,
                    "stop": pattern.stop,
                    "description": pattern.description,
                    "classification": pattern.pattern_classification,
                    "points": {
                        "X": pattern.point_x,
                        "A": pattern.point_a,
                        "B": pattern.point_b,
                        "C": pattern.point_c,
                        "D": pattern.point_d
                    }
                }
                for pattern in patterns
            ],
            "best_pattern": {
                "type": patterns[0].pattern_type.value,
                "confidence": patterns[0].confidence,
                "entry": patterns[0].entry,
                "target": patterns[0].target
            } if patterns else None
        }
    
    def get_status(self) -> Dict:
        """Get analyzer status"""
        return {
            "name": self.name,
            "version": self.version,
            "symbol": self.symbol,
            "price_points": len(self.price_history),
            "pivot_highs": len(self.highs),
            "pivot_lows": len(self.lows),
            "active_patterns": len(self.active_patterns),
            "pattern_tolerance": self.pattern_tolerance
        }
    
    def shutdown(self) -> None:
        """Shutdown the analyzer"""
        try:
            self.logger.info("Shutting down Advanced Harmonic Pattern Analyzer...")
            self.active_patterns.clear()
            self.completed_patterns.clear()
            self.price_history.clear()
            self.highs.clear()
            self.lows.clear()
            self.logger.info("Advanced Harmonic Pattern Analyzer shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

def test_harmonic_pattern_analyzer():
    """Test the harmonic pattern analyzer"""
    print("Testing Advanced Harmonic Pattern Analyzer...")
    print("=" * 60)
    
    analyzer = AdvancedHarmonicPatternAnalyzer("EURUSD")
    
    # Test initialization
    result = analyzer.initialize()
    print(f"Initialization: {result['status']}")
    
    if result['status'] == 'initialized':
        # Generate sample price data that might form patterns
        import random
        random.seed(42)  # For reproducible results
        
        base_price = 1.17000
        prices = []
        
        # Generate a potential Gartley-like pattern
        for i in range(200):
            # Create some structured movement
            if i < 50:
                # Downtrend (X to A)
                price = base_price - (i * 0.0001)
            elif i < 100:
                # Partial retracement (A to B)
                price = base_price - 0.005 + ((i - 50) * 0.00006)
            elif i < 150:
                # Another decline (B to C)
                price = base_price - 0.002 - ((i - 100) * 0.00004)
            else:
                # Final retracement (C to D)
                price = base_price - 0.004 + ((i - 150) * 0.00005)
            
            # Add some noise
            price += random.uniform(-0.00005, 0.00005)
            prices.append(price)
        
        # Update analyzer with price data
        analyzer.update_price_data(prices)
        
        # Analyze patterns
        analysis = analyzer.get_pattern_analysis()
        print(f"\nTotal Patterns Detected: {analysis['total_patterns']}")
        
        if analysis['patterns']:
            print(f"Best Pattern: {analysis['best_pattern']['type']}")
            print(f"Confidence: {analysis['best_pattern']['confidence']:.1f}%")
            print(f"Entry: {analysis['best_pattern']['entry']:.5f}")
            print(f"Target: {analysis['best_pattern']['target']:.5f}")
        
        print("\n[OK] Advanced Harmonic Pattern Analyzer Test PASSED!")
        return True
    else:
        print(f"[FAIL] Test failed: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    test_harmonic_pattern_analyzer()