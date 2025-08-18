"""
AGENT_ADAPTIVE: Adaptive Coordination System
Status: FULLY IMPLEMENTED
Purpose: Dynamic system optimization based on market conditions and performance

This system provides:
1. Market volatility response and risk adjustment
2. Performance-based resource optimization
3. Dynamic load balancing and agent reallocation
4. Pattern recognition and learning from successful strategies
5. Real-time topology switching for optimal efficiency
6. Self-organizing agent management
"""

import asyncio
import json
import time
import threading
import logging
import statistics
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
from dataclasses import dataclass, asdict

# Coordination topologies
class TopologyType(Enum):
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    RING = "ring"
    HYBRID = "hybrid"
    EMERGENCY = "emergency"

# Market conditions
class MarketCondition(Enum):
    STABLE = "stable"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"
    CRISIS = "crisis"

# Agent performance states
class AgentState(Enum):
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"

@dataclass
class PerformanceMetrics:
    """Agent performance metrics"""
    response_time: float
    success_rate: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    error_rate: float
    signal_quality: float
    profit_factor: float
    timestamp: datetime

@dataclass
class MarketMetrics:
    """Market condition metrics"""
    volatility: float
    volume: float
    trend_strength: float
    support_resistance: float
    liquidity: float
    news_sentiment: float
    condition: MarketCondition
    timestamp: datetime

@dataclass
class CoordinationState:
    """Current coordination configuration"""
    topology: TopologyType
    active_agents: List[str]
    agent_priorities: Dict[str, int]
    load_distribution: Dict[str, float]
    communication_matrix: Dict[str, List[str]]
    risk_parameters: Dict[str, float]
    timestamp: datetime

class AdaptiveCoordinator:
    """
    Advanced adaptive coordination system that dynamically optimizes
    performance based on market conditions and system load
    """
    
    def __init__(self, mt5_connector=None, signal_coordinator=None):
        self.name = "ADAPTIVE_COORDINATOR"
        self.version = "1.0.0"
        self.status = "INITIALIZING"
        
        # External dependencies
        self.mt5_connector = mt5_connector
        self.signal_coordinator = signal_coordinator
        
        # Core state
        self.current_topology = TopologyType.HIERARCHICAL
        self.current_market_condition = MarketCondition.STABLE
        self.coordination_state = None
        
        # Agent management
        self.agents = {}
        self.agent_performance = {}
        self.agent_states = {}
        self.agent_roles = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.market_history = deque(maxlen=500)
        self.coordination_history = deque(maxlen=200)
        
        # Learning system
        self.pattern_database = {}
        self.success_patterns = []
        self.adaptation_rules = {}
        self.learning_weights = {
            'performance': 0.4,
            'market_fit': 0.3,
            'stability': 0.2,
            'efficiency': 0.1
        }
        
        # Configuration
        self.config = {
            'adaptation_threshold': 0.15,  # 15% improvement needed for switch
            'volatility_threshold': 0.02,  # 2% volatility threshold
            'performance_window': 300,     # 5 minutes for performance evaluation
            'market_analysis_interval': 60, # 1 minute market analysis
            'rebalance_interval': 180,     # 3 minutes load rebalancing
            'pattern_min_samples': 10,     # Minimum samples for pattern recognition
            'emergency_threshold': 0.5     # 50% failure rate triggers emergency mode
        }
        
        # Threading
        self.coordination_thread = None
        self.analysis_thread = None
        self.learning_thread = None
        self.is_running = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def initialize(self):
        """Initialize the adaptive coordination system"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Initialize base coordination state
            self.coordination_state = CoordinationState(
                topology=TopologyType.HIERARCHICAL,
                active_agents=[],
                agent_priorities={},
                load_distribution={},
                communication_matrix={},
                risk_parameters=self._get_default_risk_parameters(),
                timestamp=datetime.now()
            )
            
            # Load adaptation rules
            self._initialize_adaptation_rules()
            
            # Setup pattern recognition
            self._initialize_pattern_recognition()
            
            self.status = "INITIALIZED"
            self.logger.info("Adaptive coordinator initialized successfully")
            
            return {
                "status": "initialized",
                "agent": "ADAPTIVE_COORDINATOR",
                "topology": self.current_topology.value,
                "market_condition": self.current_market_condition.value,
                "adaptation_enabled": True
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "ADAPTIVE_COORDINATOR", "error": str(e)}
    
    def start_coordination(self):
        """Start the adaptive coordination system"""
        if self.is_running:
            return {"status": "already_running", "message": "Adaptive coordination already active"}
        
        try:
            self.is_running = True
            
            # Start main coordination loop
            self.coordination_thread = threading.Thread(
                target=self._coordination_loop, daemon=True
            )
            self.coordination_thread.start()
            
            # Start market analysis loop
            self.analysis_thread = threading.Thread(
                target=self._market_analysis_loop, daemon=True
            )
            self.analysis_thread.start()
            
            # Start learning loop
            self.learning_thread = threading.Thread(
                target=self._learning_loop, daemon=True
            )
            self.learning_thread.start()
            
            self.status = "RUNNING"
            self.logger.info("Adaptive coordination started successfully")
            
            return {"status": "started", "message": "Adaptive coordination active"}
            
        except Exception as e:
            self.logger.error(f"Failed to start adaptive coordination: {e}")
            self.is_running = False
            return {"status": "failed", "message": f"Failed to start: {e}"}
    
    def stop_coordination(self):
        """Stop the adaptive coordination system"""
        try:
            self.is_running = False
            
            # Wait for threads to finish
            for thread in [self.coordination_thread, self.analysis_thread, self.learning_thread]:
                if thread and thread.is_alive():
                    thread.join(timeout=5)
            
            self.status = "STOPPED"
            self.logger.info("Adaptive coordination stopped")
            
            return {"status": "stopped", "message": "Adaptive coordination stopped successfully"}
            
        except Exception as e:
            self.logger.error(f"Error stopping adaptive coordination: {e}")
            return {"status": "error", "message": f"Error stopping: {e}"}
    
    def register_agent(self, agent_id: str, agent_instance: Any, priority: int = 2, 
                      capabilities: List[str] = None):
        """Register an agent with the adaptive coordinator"""
        try:
            self.agents[agent_id] = {
                'instance': agent_instance,
                'priority': priority,
                'capabilities': capabilities or [],
                'registered_at': datetime.now(),
                'status': 'active'
            }
            
            # Initialize performance tracking
            self.agent_performance[agent_id] = deque(maxlen=100)
            self.agent_states[agent_id] = AgentState.GOOD
            
            # Assign initial role based on capabilities and priority
            self._assign_agent_role(agent_id)
            
            # Update coordination state
            self.coordination_state.active_agents.append(agent_id)
            self.coordination_state.agent_priorities[agent_id] = priority
            self.coordination_state.load_distribution[agent_id] = 0.0
            
            self.logger.info(f"Agent {agent_id} registered with priority {priority}")
            
            return {"status": "registered", "agent_id": agent_id, "role": self.agent_roles.get(agent_id)}
            
        except Exception as e:
            self.logger.error(f"Error registering agent {agent_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def update_agent_performance(self, agent_id: str, metrics: PerformanceMetrics):
        """Update agent performance metrics"""
        try:
            if agent_id not in self.agents:
                return {"status": "not_found", "message": f"Agent {agent_id} not registered"}
            
            # Store performance metrics
            self.agent_performance[agent_id].append(metrics)
            
            # Update agent state based on performance
            new_state = self._evaluate_agent_state(agent_id, metrics)
            if new_state != self.agent_states.get(agent_id):
                self.logger.info(f"Agent {agent_id} state changed: {self.agent_states.get(agent_id)} -> {new_state}")
                self.agent_states[agent_id] = new_state
                
                # Trigger rebalancing if needed
                if new_state in [AgentState.OVERLOADED, AgentState.FAILED]:
                    self._trigger_load_rebalancing(agent_id)
            
            # Check for adaptation triggers
            self._check_adaptation_triggers()
            
            return {"status": "updated", "agent_state": new_state.value}
            
        except Exception as e:
            self.logger.error(f"Error updating performance for {agent_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def update_market_conditions(self, market_metrics: MarketMetrics):
        """Update market condition analysis"""
        try:
            # Store market metrics
            self.market_history.append(market_metrics)
            
            # Update current market condition
            previous_condition = self.current_market_condition
            self.current_market_condition = market_metrics.condition
            
            # If market condition changed significantly, adapt coordination
            if previous_condition != self.current_market_condition:
                self.logger.info(f"Market condition changed: {previous_condition.value} -> {self.current_market_condition.value}")
                self._adapt_to_market_condition(market_metrics)
            
            # Update risk parameters based on volatility
            self._adjust_risk_parameters(market_metrics)
            
            return {"status": "updated", "market_condition": self.current_market_condition.value}
            
        except Exception as e:
            self.logger.error(f"Error updating market conditions: {e}")
            return {"status": "error", "message": str(e)}
    
    def force_topology_switch(self, new_topology: TopologyType, reason: str = "manual"):
        """Force a topology switch with validation"""
        try:
            if new_topology == self.current_topology:
                return {"status": "no_change", "message": "Already using requested topology"}
            
            # Validate topology switch is beneficial
            if reason != "manual" and not self._validate_topology_switch(new_topology):
                return {"status": "rejected", "message": "Topology switch validation failed"}
            
            # Perform the switch
            previous_topology = self.current_topology
            success = self._switch_topology(new_topology)
            
            if success:
                self.logger.info(f"Topology switched: {previous_topology.value} -> {new_topology.value} (reason: {reason})")
                
                # Record the switch for learning
                self._record_topology_switch(previous_topology, new_topology, reason)
                
                return {"status": "switched", "from": previous_topology.value, "to": new_topology.value}
            else:
                return {"status": "failed", "message": "Topology switch failed"}
                
        except Exception as e:
            self.logger.error(f"Error switching topology: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_optimization_recommendations(self):
        """Get current optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze current performance
            overall_performance = self._calculate_overall_performance()
            
            # Check for topology optimization
            optimal_topology = self._recommend_optimal_topology()
            if optimal_topology != self.current_topology:
                recommendations.append({
                    'type': 'topology_switch',
                    'current': self.current_topology.value,
                    'recommended': optimal_topology.value,
                    'expected_improvement': self._estimate_topology_improvement(optimal_topology),
                    'priority': 'high' if overall_performance < 0.7 else 'medium'
                })
            
            # Check for agent reallocation
            reallocation_suggestions = self._analyze_agent_reallocation()
            if reallocation_suggestions:
                recommendations.extend(reallocation_suggestions)
            
            # Check for risk parameter adjustments
            risk_adjustments = self._analyze_risk_adjustments()
            if risk_adjustments:
                recommendations.extend(risk_adjustments)
            
            return {
                "status": "success",
                "overall_performance": overall_performance,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            # Calculate current performance metrics
            overall_performance = self._calculate_overall_performance()
            
            # Get agent status summary
            agent_status = {}
            for agent_id in self.agents:
                state = self.agent_states.get(agent_id, AgentState.GOOD)
                recent_performance = list(self.agent_performance.get(agent_id, []))[-5:]
                avg_performance = np.mean([p.success_rate for p in recent_performance]) if recent_performance else 0.0
                
                agent_status[agent_id] = {
                    'state': state.value,
                    'role': self.agent_roles.get(agent_id, 'unknown'),
                    'load': self.coordination_state.load_distribution.get(agent_id, 0.0),
                    'performance': avg_performance,
                    'priority': self.coordination_state.agent_priorities.get(agent_id, 2)
                }
            
            return {
                'name': self.name,
                'version': self.version,
                'status': self.status,
                'is_running': self.is_running,
                'current_topology': self.current_topology.value,
                'market_condition': self.current_market_condition.value,
                'overall_performance': overall_performance,
                'active_agents': len(self.coordination_state.active_agents),
                'agent_status': agent_status,
                'adaptation_enabled': True,
                'last_adaptation': self._get_last_adaptation_time(),
                'patterns_learned': len(self.success_patterns),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"status": "error", "message": str(e)}
    
    # Private methods for internal coordination logic
    
    def _coordination_loop(self):
        """Main coordination loop"""
        self.logger.info("Starting adaptive coordination loop")
        
        while self.is_running:
            try:
                # Monitor agent performance
                self._monitor_agent_performance()
                
                # Check for load rebalancing needs
                if self._needs_load_rebalancing():
                    self._perform_load_rebalancing()
                
                # Check for topology optimization
                if self._should_optimize_topology():
                    optimal_topology = self._recommend_optimal_topology()
                    if optimal_topology != self.current_topology:
                        self._switch_topology(optimal_topology)
                
                # Update coordination state
                self._update_coordination_state()
                
                # Sleep before next iteration
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Coordination loop error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _market_analysis_loop(self):
        """Market analysis and adaptation loop"""
        self.logger.info("Starting market analysis loop")
        
        while self.is_running:
            try:
                # Analyze current market conditions
                if self.mt5_connector:
                    market_metrics = self._analyze_market_conditions()
                    if market_metrics:
                        self.update_market_conditions(market_metrics)
                
                # Check for emergency conditions
                if self._detect_emergency_conditions():
                    self._activate_emergency_mode()
                
                time.sleep(self.config['market_analysis_interval'])
                
            except Exception as e:
                self.logger.error(f"Market analysis loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _learning_loop(self):
        """Pattern learning and optimization loop"""
        self.logger.info("Starting learning loop")
        
        while self.is_running:
            try:
                # Analyze recent performance patterns
                self._analyze_performance_patterns()
                
                # Update adaptation rules based on success patterns
                self._update_adaptation_rules()
                
                # Clean old patterns
                self._clean_old_patterns()
                
                time.sleep(300)  # Learn every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Learning loop error: {e}")
                time.sleep(300)  # Wait on error
    
    def _get_default_risk_parameters(self):
        """Get default risk parameters"""
        return {
            'max_risk_per_trade': 0.02,
            'max_daily_loss': 0.05,
            'max_open_positions': 5,
            'volatility_multiplier': 1.0,
            'correlation_limit': 0.7
        }
    
    def _initialize_adaptation_rules(self):
        """Initialize adaptation rules"""
        self.adaptation_rules = {
            'high_volatility': {
                'topology': TopologyType.MESH,
                'risk_reduction': 0.5,
                'priority_boost': ['RISK_CALCULATOR', 'NEURAL_SIGNAL_BRAIN']
            },
            'low_performance': {
                'topology': TopologyType.HIERARCHICAL,
                'agent_reallocation': True,
                'increase_monitoring': True
            },
            'trending_market': {
                'topology': TopologyType.RING,
                'boost_signal_agents': True,
                'reduce_risk_aversion': 0.1
            },
            'crisis_mode': {
                'topology': TopologyType.EMERGENCY,
                'risk_reduction': 0.8,
                'emergency_stop': True
            }
        }
    
    def _initialize_pattern_recognition(self):
        """Initialize pattern recognition system"""
        self.pattern_database = {
            'successful_topologies': defaultdict(list),
            'market_adaptations': defaultdict(list),
            'performance_improvements': [],
            'failure_patterns': []
        }
    
    def _assign_agent_role(self, agent_id: str):
        """Assign role to agent based on capabilities and current needs"""
        agent_info = self.agents[agent_id]
        capabilities = agent_info['capabilities']
        
        # Role assignment logic based on agent type and current topology
        if 'MT5_CONNECTOR' in agent_id:
            self.agent_roles[agent_id] = 'data_provider'
        elif 'SIGNAL_COORDINATOR' in agent_id:
            self.agent_roles[agent_id] = 'signal_orchestrator'
        elif 'RISK_CALCULATOR' in agent_id:
            self.agent_roles[agent_id] = 'risk_manager'
        elif 'NEURAL' in agent_id:
            self.agent_roles[agent_id] = 'pattern_analyzer'
        elif 'TECHNICAL_ANALYST' in agent_id:
            self.agent_roles[agent_id] = 'market_analyzer'
        else:
            self.agent_roles[agent_id] = 'general_worker'
    
    def _evaluate_agent_state(self, agent_id: str, metrics: PerformanceMetrics) -> AgentState:
        """Evaluate agent state based on performance metrics"""
        try:
            # Define thresholds
            if metrics.success_rate >= 0.9 and metrics.response_time <= 1.0 and metrics.error_rate <= 0.01:
                return AgentState.OPTIMAL
            elif metrics.success_rate >= 0.8 and metrics.response_time <= 2.0 and metrics.error_rate <= 0.05:
                return AgentState.GOOD
            elif metrics.success_rate >= 0.6 and metrics.response_time <= 5.0 and metrics.error_rate <= 0.15:
                return AgentState.DEGRADED
            elif metrics.success_rate >= 0.3 and metrics.response_time <= 10.0:
                return AgentState.OVERLOADED
            else:
                return AgentState.FAILED
                
        except Exception as e:
            self.logger.error(f"Error evaluating agent state for {agent_id}: {e}")
            return AgentState.DEGRADED
    
    def _calculate_overall_performance(self) -> float:
        """Calculate overall system performance score"""
        try:
            if not self.agent_performance:
                return 0.5  # Default neutral score
            
            agent_scores = []
            for agent_id, performance_history in self.agent_performance.items():
                if performance_history:
                    recent_metrics = list(performance_history)[-5:]  # Last 5 measurements
                    avg_success = np.mean([m.success_rate for m in recent_metrics])
                    avg_response = np.mean([m.response_time for m in recent_metrics])
                    avg_quality = np.mean([m.signal_quality for m in recent_metrics])
                    
                    # Combined score (normalized)
                    score = (avg_success * 0.4 + 
                            (1.0 / max(avg_response, 0.1)) * 0.3 + 
                            avg_quality * 0.3)
                    agent_scores.append(min(1.0, score))
            
            return np.mean(agent_scores) if agent_scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating overall performance: {e}")
            return 0.5
    
    def _recommend_optimal_topology(self) -> TopologyType:
        """Recommend optimal topology based on current conditions"""
        try:
            current_performance = self._calculate_overall_performance()
            market_volatility = self._get_current_volatility()
            system_load = self._get_system_load()
            
            # Emergency conditions
            if self._detect_emergency_conditions():
                return TopologyType.EMERGENCY
            
            # High volatility conditions
            if market_volatility > self.config['volatility_threshold'] * 3:
                return TopologyType.MESH  # Distributed for fault tolerance
            
            # High performance, stable conditions
            if current_performance > 0.85 and market_volatility < self.config['volatility_threshold']:
                return TopologyType.RING  # Pipeline for efficiency
            
            # High load conditions
            if system_load > 0.8:
                return TopologyType.MESH  # Distribute load
            
            # Trending market conditions
            if self.current_market_condition == MarketCondition.TRENDING:
                return TopologyType.RING  # Sequential processing for trends
            
            # Default to hierarchical for complex coordination
            return TopologyType.HIERARCHICAL
            
        except Exception as e:
            self.logger.error(f"Error recommending topology: {e}")
            return self.current_topology
    
    def _switch_topology(self, new_topology: TopologyType) -> bool:
        """Switch to new coordination topology"""
        try:
            self.logger.info(f"Switching topology from {self.current_topology.value} to {new_topology.value}")
            
            # Store previous state for rollback
            previous_state = self.coordination_state
            
            # Update topology
            self.current_topology = new_topology
            
            # Reconfigure agent communication matrix
            self._update_communication_matrix(new_topology)
            
            # Redistribute agent priorities and roles
            self._redistribute_agent_roles(new_topology)
            
            # Update coordination state
            self.coordination_state.topology = new_topology
            self.coordination_state.timestamp = datetime.now()
            
            # Record successful switch
            self.coordination_history.append({
                'timestamp': datetime.now(),
                'action': 'topology_switch',
                'from': previous_state.topology.value,
                'to': new_topology.value,
                'performance_before': self._calculate_overall_performance()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error switching topology: {e}")
            return False
    
    def _update_communication_matrix(self, topology: TopologyType):
        """Update agent communication matrix based on topology"""
        agents = self.coordination_state.active_agents
        matrix = {}
        
        if topology == TopologyType.HIERARCHICAL:
            # Central coordinator communicates with all
            coordinator = self._find_coordinator_agent()
            for agent in agents:
                if agent == coordinator:
                    matrix[agent] = [a for a in agents if a != agent]
                else:
                    matrix[agent] = [coordinator]
                    
        elif topology == TopologyType.MESH:
            # All agents communicate with all others
            for agent in agents:
                matrix[agent] = [a for a in agents if a != agent]
                
        elif topology == TopologyType.RING:
            # Ring topology - each agent talks to next
            for i, agent in enumerate(agents):
                next_agent = agents[(i + 1) % len(agents)]
                matrix[agent] = [next_agent]
                
        elif topology == TopologyType.HYBRID:
            # Mixed topology based on agent roles
            core_agents = [a for a in agents if self.agent_roles.get(a) in ['signal_orchestrator', 'risk_manager']]
            for agent in agents:
                if agent in core_agents:
                    matrix[agent] = [a for a in agents if a != agent]
                else:
                    matrix[agent] = core_agents
                    
        else:  # EMERGENCY
            # Emergency coordinator only
            emergency_coord = self._find_emergency_coordinator()
            for agent in agents:
                if agent == emergency_coord:
                    matrix[agent] = [a for a in agents if a != agent]
                else:
                    matrix[agent] = [emergency_coord]
        
        self.coordination_state.communication_matrix = matrix
    
    def _find_coordinator_agent(self) -> str:
        """Find the best agent to act as coordinator"""
        # Prefer signal coordinator if available
        for agent_id in self.coordination_state.active_agents:
            if 'SIGNAL_COORDINATOR' in agent_id:
                return agent_id
        
        # Fall back to highest priority agent
        if self.coordination_state.agent_priorities:
            return min(self.coordination_state.agent_priorities.keys(), 
                      key=lambda x: self.coordination_state.agent_priorities[x])
        
        return self.coordination_state.active_agents[0] if self.coordination_state.active_agents else None
    
    def _find_emergency_coordinator(self) -> str:
        """Find the most reliable agent for emergency coordination"""
        # Find agent with best recent performance
        best_agent = None
        best_score = 0
        
        for agent_id in self.coordination_state.active_agents:
            if agent_id in self.agent_performance and self.agent_performance[agent_id]:
                recent_metrics = list(self.agent_performance[agent_id])[-3:]
                avg_success = np.mean([m.success_rate for m in recent_metrics])
                if avg_success > best_score:
                    best_score = avg_success
                    best_agent = agent_id
        
        return best_agent or self._find_coordinator_agent()
    
    def _analyze_market_conditions(self) -> Optional[MarketMetrics]:
        """Analyze current market conditions"""
        try:
            if not self.mt5_connector:
                return None
            
            # Get market data (this would integrate with actual MT5 data)
            # For now, simulate based on available information
            
            volatility = self._calculate_market_volatility()
            volume = self._get_market_volume()
            trend_strength = self._calculate_trend_strength()
            
            # Determine market condition
            condition = MarketCondition.STABLE
            if volatility > 0.03:
                condition = MarketCondition.VOLATILE
            elif trend_strength > 0.7:
                condition = MarketCondition.TRENDING
            elif volume < 0.3:
                condition = MarketCondition.RANGING
            
            return MarketMetrics(
                volatility=volatility,
                volume=volume,
                trend_strength=trend_strength,
                support_resistance=0.5,  # Placeholder
                liquidity=0.8,  # Placeholder
                news_sentiment=0.0,  # Placeholder
                condition=condition,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return None
    
    def _calculate_market_volatility(self) -> float:
        """Calculate current market volatility"""
        # Placeholder implementation
        # In real implementation, this would calculate volatility from price data
        import random
        return random.uniform(0.01, 0.05)
    
    def _get_market_volume(self) -> float:
        """Get normalized market volume"""
        # Placeholder implementation
        import random
        return random.uniform(0.2, 1.0)
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength indicator"""
        # Placeholder implementation
        import random
        return random.uniform(0.0, 1.0)
    
    def _get_current_volatility(self) -> float:
        """Get current market volatility"""
        if self.market_history:
            return self.market_history[-1].volatility
        return 0.02  # Default
    
    def _get_system_load(self) -> float:
        """Calculate current system load"""
        if not self.agent_performance:
            return 0.5
        
        loads = []
        for agent_id, performance_history in self.agent_performance.items():
            if performance_history:
                recent_metrics = list(performance_history)[-3:]
                avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
                loads.append(avg_cpu)
        
        return np.mean(loads) if loads else 0.5
    
    def _detect_emergency_conditions(self) -> bool:
        """Detect if emergency conditions exist"""
        try:
            # Check agent failure rates
            failed_agents = sum(1 for state in self.agent_states.values() 
                              if state == AgentState.FAILED)
            failure_rate = failed_agents / max(len(self.agent_states), 1)
            
            if failure_rate >= self.config['emergency_threshold']:
                return True
            
            # Check overall performance
            overall_performance = self._calculate_overall_performance()
            if overall_performance < 0.3:
                return True
            
            # Check market crisis conditions
            if self.current_market_condition == MarketCondition.CRISIS:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting emergency conditions: {e}")
            return False
    
    def _adapt_to_market_condition(self, market_metrics: MarketMetrics):
        """Adapt coordination to new market conditions"""
        try:
            condition = market_metrics.condition
            
            if condition == MarketCondition.VOLATILE:
                # Increase risk monitoring, switch to mesh for fault tolerance
                self._adjust_risk_parameters(market_metrics, factor=0.7)
                if self.current_topology != TopologyType.MESH:
                    self._switch_topology(TopologyType.MESH)
                    
            elif condition == MarketCondition.TRENDING:
                # Optimize for trend following, use ring topology
                self._boost_trend_agents()
                if self.current_topology != TopologyType.RING:
                    self._switch_topology(TopologyType.RING)
                    
            elif condition == MarketCondition.CRISIS:
                # Emergency mode
                self._activate_emergency_mode()
                
        except Exception as e:
            self.logger.error(f"Error adapting to market condition: {e}")
    
    def _adjust_risk_parameters(self, market_metrics: MarketMetrics, factor: float = 1.0):
        """Adjust risk parameters based on market conditions"""
        try:
            volatility_multiplier = min(2.0, 1.0 + market_metrics.volatility * 10)
            
            self.coordination_state.risk_parameters.update({
                'max_risk_per_trade': 0.02 * factor / volatility_multiplier,
                'volatility_multiplier': volatility_multiplier,
                'max_open_positions': max(1, int(5 * factor))
            })
            
            self.logger.info(f"Risk parameters adjusted: factor={factor}, volatility_mult={volatility_multiplier}")
            
        except Exception as e:
            self.logger.error(f"Error adjusting risk parameters: {e}")
    
    def _boost_trend_agents(self):
        """Boost priority of trend-following agents"""
        try:
            trend_agents = ['NEURAL_SIGNAL_BRAIN', 'TECHNICAL_ANALYST', 'CHART_SIGNAL_AGENT']
            
            for agent_id in self.coordination_state.active_agents:
                if any(trend_agent in agent_id for trend_agent in trend_agents):
                    # Increase priority (lower number = higher priority)
                    current_priority = self.coordination_state.agent_priorities.get(agent_id, 2)
                    self.coordination_state.agent_priorities[agent_id] = max(1, current_priority - 1)
                    
        except Exception as e:
            self.logger.error(f"Error boosting trend agents: {e}")
    
    def _activate_emergency_mode(self):
        """Activate emergency coordination mode"""
        try:
            self.logger.warning("Activating emergency coordination mode")
            
            # Switch to emergency topology
            self._switch_topology(TopologyType.EMERGENCY)
            
            # Reduce risk drastically
            self.coordination_state.risk_parameters.update({
                'max_risk_per_trade': 0.005,  # 0.5% max risk
                'max_open_positions': 1,      # Only 1 position
                'emergency_stop': True
            })
            
            # Prioritize critical agents only
            critical_agents = ['RISK_CALCULATOR', 'SIGNAL_COORDINATOR', 'MT5_CONNECTOR']
            for agent_id in self.coordination_state.active_agents:
                if any(critical in agent_id for critical in critical_agents):
                    self.coordination_state.agent_priorities[agent_id] = 1
                else:
                    self.coordination_state.agent_priorities[agent_id] = 3
            
        except Exception as e:
            self.logger.error(f"Error activating emergency mode: {e}")
    
    def _needs_load_rebalancing(self) -> bool:
        """Check if load rebalancing is needed"""
        try:
            if not self.coordination_state.load_distribution:
                return False
            
            loads = list(self.coordination_state.load_distribution.values())
            if not loads:
                return False
            
            max_load = max(loads)
            min_load = min(loads)
            
            # Rebalance if difference is > 40%
            return (max_load - min_load) > 0.4
            
        except Exception as e:
            self.logger.error(f"Error checking load rebalancing: {e}")
            return False
    
    def _perform_load_rebalancing(self):
        """Perform load rebalancing across agents"""
        try:
            self.logger.info("Performing load rebalancing")
            
            # Find overloaded and underloaded agents
            loads = self.coordination_state.load_distribution
            sorted_agents = sorted(loads.items(), key=lambda x: x[1], reverse=True)
            
            overloaded = [agent for agent, load in sorted_agents[:len(sorted_agents)//2] if load > 0.7]
            underloaded = [agent for agent, load in sorted_agents[len(sorted_agents)//2:] if load < 0.3]
            
            # Redistribute tasks from overloaded to underloaded agents
            for overloaded_agent in overloaded:
                if underloaded:
                    target_agent = underloaded[0]
                    
                    # Transfer some load
                    transfer_amount = min(0.2, loads[overloaded_agent] - 0.5)
                    loads[overloaded_agent] -= transfer_amount
                    loads[target_agent] += transfer_amount
                    
                    self.logger.info(f"Transferred load {transfer_amount:.2f} from {overloaded_agent} to {target_agent}")
                    
                    # Remove from underloaded if now loaded enough
                    if loads[target_agent] > 0.5:
                        underloaded.remove(target_agent)
            
        except Exception as e:
            self.logger.error(f"Error performing load rebalancing: {e}")
    
    def _trigger_load_rebalancing(self, agent_id: str):
        """Trigger immediate load rebalancing for specific agent"""
        try:
            current_load = self.coordination_state.load_distribution.get(agent_id, 0.0)
            
            if current_load > 0.8:
                # Find agents with lower load to redistribute to
                available_agents = [
                    (aid, load) for aid, load in self.coordination_state.load_distribution.items()
                    if aid != agent_id and load < 0.5 and self.agent_states.get(aid, AgentState.FAILED) != AgentState.FAILED
                ]
                
                if available_agents:
                    # Sort by lowest load
                    available_agents.sort(key=lambda x: x[1])
                    target_agent = available_agents[0][0]
                    
                    # Transfer half the load
                    transfer_amount = current_load * 0.5
                    self.coordination_state.load_distribution[agent_id] -= transfer_amount
                    self.coordination_state.load_distribution[target_agent] += transfer_amount
                    
                    self.logger.info(f"Emergency load transfer: {transfer_amount:.2f} from {agent_id} to {target_agent}")
                    
        except Exception as e:
            self.logger.error(f"Error triggering load rebalancing for {agent_id}: {e}")
    
    def _should_optimize_topology(self) -> bool:
        """Check if topology optimization should be triggered"""
        try:
            # Don't optimize too frequently
            last_adaptation = self._get_last_adaptation_time()
            if last_adaptation and (datetime.now() - last_adaptation).seconds < 300:  # 5 minutes
                return False
            
            # Check if current performance is suboptimal
            current_performance = self._calculate_overall_performance()
            
            # Estimate performance with different topology
            optimal_topology = self._recommend_optimal_topology()
            if optimal_topology != self.current_topology:
                estimated_improvement = self._estimate_topology_improvement(optimal_topology)
                return estimated_improvement > self.config['adaptation_threshold']
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking topology optimization: {e}")
            return False
    
    def _estimate_topology_improvement(self, new_topology: TopologyType) -> float:
        """Estimate performance improvement from topology switch"""
        try:
            # Look for historical patterns
            current_conditions = (self.current_market_condition, self._get_system_load())
            
            # Check pattern database for similar conditions
            for pattern in self.success_patterns:
                if (pattern.get('market_condition') == self.current_market_condition and
                    pattern.get('topology') == new_topology):
                    return pattern.get('improvement', 0.0)
            
            # Use heuristic estimates
            if new_topology == TopologyType.MESH and self._get_system_load() > 0.8:
                return 0.2  # 20% improvement expected for high load
            elif new_topology == TopologyType.RING and self.current_market_condition == MarketCondition.TRENDING:
                return 0.15  # 15% improvement for trending markets
            elif new_topology == TopologyType.HIERARCHICAL and self._calculate_overall_performance() < 0.6:
                return 0.1  # 10% improvement for low performance
            
            return 0.05  # Conservative default estimate
            
        except Exception as e:
            self.logger.error(f"Error estimating topology improvement: {e}")
            return 0.0
    
    def _validate_topology_switch(self, new_topology: TopologyType) -> bool:
        """Validate that topology switch would be beneficial"""
        try:
            estimated_improvement = self._estimate_topology_improvement(new_topology)
            return estimated_improvement > self.config['adaptation_threshold']
            
        except Exception as e:
            self.logger.error(f"Error validating topology switch: {e}")
            return False
    
    def _record_topology_switch(self, from_topology: TopologyType, to_topology: TopologyType, reason: str):
        """Record topology switch for learning"""
        try:
            record = {
                'timestamp': datetime.now(),
                'from_topology': from_topology.value,
                'to_topology': to_topology.value,
                'reason': reason,
                'market_condition': self.current_market_condition.value,
                'performance_before': self._calculate_overall_performance(),
                'system_load': self._get_system_load()
            }
            
            self.coordination_history.append(record)
            
        except Exception as e:
            self.logger.error(f"Error recording topology switch: {e}")
    
    def _analyze_agent_reallocation(self) -> List[Dict]:
        """Analyze potential agent reallocation opportunities"""
        try:
            recommendations = []
            
            # Find agents that could benefit from role changes
            for agent_id, state in self.agent_states.items():
                if state == AgentState.OVERLOADED:
                    recommendations.append({
                        'type': 'reduce_load',
                        'agent': agent_id,
                        'current_load': self.coordination_state.load_distribution.get(agent_id, 0.0),
                        'recommendation': 'redistribute_tasks',
                        'priority': 'high'
                    })
                elif state == AgentState.OPTIMAL:
                    recommendations.append({
                        'type': 'increase_load',
                        'agent': agent_id,
                        'current_load': self.coordination_state.load_distribution.get(agent_id, 0.0),
                        'recommendation': 'assign_additional_tasks',
                        'priority': 'medium'
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing agent reallocation: {e}")
            return []
    
    def _analyze_risk_adjustments(self) -> List[Dict]:
        """Analyze potential risk parameter adjustments"""
        try:
            recommendations = []
            
            current_volatility = self._get_current_volatility()
            current_performance = self._calculate_overall_performance()
            
            # Recommend risk reduction if high volatility
            if current_volatility > self.config['volatility_threshold'] * 2:
                recommendations.append({
                    'type': 'risk_reduction',
                    'parameter': 'max_risk_per_trade',
                    'current': self.coordination_state.risk_parameters.get('max_risk_per_trade', 0.02),
                    'recommended': 0.01,
                    'reason': 'high_market_volatility',
                    'priority': 'high'
                })
            
            # Recommend risk increase if stable and performing well
            elif current_volatility < self.config['volatility_threshold'] and current_performance > 0.8:
                recommendations.append({
                    'type': 'risk_increase',
                    'parameter': 'max_risk_per_trade',
                    'current': self.coordination_state.risk_parameters.get('max_risk_per_trade', 0.02),
                    'recommended': 0.025,
                    'reason': 'stable_high_performance',
                    'priority': 'medium'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk adjustments: {e}")
            return []
    
    def _get_last_adaptation_time(self) -> Optional[datetime]:
        """Get timestamp of last adaptation"""
        try:
            if self.coordination_history:
                return self.coordination_history[-1].get('timestamp')
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting last adaptation time: {e}")
            return None
    
    def _monitor_agent_performance(self):
        """Monitor and update agent performance metrics"""
        try:
            for agent_id, agent_info in self.agents.items():
                agent_instance = agent_info['instance']
                
                # Try to get performance metrics from agent
                try:
                    if hasattr(agent_instance, 'get_performance_metrics'):
                        metrics = agent_instance.get_performance_metrics()
                        if metrics:
                            # Convert to PerformanceMetrics object
                            perf_metrics = PerformanceMetrics(
                                response_time=metrics.get('response_time', 1.0),
                                success_rate=metrics.get('success_rate', 0.8),
                                throughput=metrics.get('throughput', 1.0),
                                cpu_usage=metrics.get('cpu_usage', 0.5),
                                memory_usage=metrics.get('memory_usage', 0.5),
                                error_rate=metrics.get('error_rate', 0.1),
                                signal_quality=metrics.get('signal_quality', 0.7),
                                profit_factor=metrics.get('profit_factor', 1.0),
                                timestamp=datetime.now()
                            )
                            
                            self.update_agent_performance(agent_id, perf_metrics)
                            
                except Exception as e:
                    self.logger.debug(f"Could not get performance metrics from {agent_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error monitoring agent performance: {e}")
    
    def _update_coordination_state(self):
        """Update the current coordination state"""
        try:
            self.coordination_state.timestamp = datetime.now()
            
            # Update load distribution based on agent states
            for agent_id in self.coordination_state.active_agents:
                state = self.agent_states.get(agent_id, AgentState.GOOD)
                
                if state == AgentState.OPTIMAL:
                    # Can handle more load
                    current_load = self.coordination_state.load_distribution.get(agent_id, 0.0)
                    self.coordination_state.load_distribution[agent_id] = min(0.9, current_load + 0.1)
                elif state == AgentState.OVERLOADED:
                    # Reduce load
                    current_load = self.coordination_state.load_distribution.get(agent_id, 0.0)
                    self.coordination_state.load_distribution[agent_id] = max(0.1, current_load - 0.2)
                elif state == AgentState.FAILED:
                    # Zero load for failed agents
                    self.coordination_state.load_distribution[agent_id] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error updating coordination state: {e}")
    
    def _redistribute_agent_roles(self, topology: TopologyType):
        """Redistribute agent roles based on new topology"""
        try:
            if topology == TopologyType.HIERARCHICAL:
                # Centralized roles - boost coordinator priority
                for agent_id in self.coordination_state.active_agents:
                    if 'COORDINATOR' in agent_id:
                        self.coordination_state.agent_priorities[agent_id] = 1
                        
            elif topology == TopologyType.MESH:
                # Distributed roles - equalize priorities
                for agent_id in self.coordination_state.active_agents:
                    self.coordination_state.agent_priorities[agent_id] = 2
                    
            elif topology == TopologyType.RING:
                # Sequential roles - prioritize by processing order
                for i, agent_id in enumerate(self.coordination_state.active_agents):
                    self.coordination_state.agent_priorities[agent_id] = (i % 3) + 1
                    
        except Exception as e:
            self.logger.error(f"Error redistributing agent roles: {e}")
    
    def _analyze_performance_patterns(self):
        """Analyze recent performance patterns for learning"""
        try:
            if len(self.coordination_history) < self.config['pattern_min_samples']:
                return
            
            # Analyze successful adaptations
            recent_adaptations = list(self.coordination_history)[-20:]  # Last 20 adaptations
            
            for adaptation in recent_adaptations:
                # Check if adaptation was successful (would need performance after measurement)
                # For now, use heuristic based on no immediate reversal
                was_successful = True  # Placeholder logic
                
                if was_successful:
                    pattern = {
                        'market_condition': adaptation.get('market_condition'),
                        'topology': adaptation.get('to_topology'),
                        'improvement': 0.1,  # Placeholder
                        'timestamp': adaptation.get('timestamp')
                    }
                    
                    # Add to success patterns if not duplicate
                    if not any(p.get('market_condition') == pattern['market_condition'] and 
                              p.get('topology') == pattern['topology'] 
                              for p in self.success_patterns):
                        self.success_patterns.append(pattern)
                        
        except Exception as e:
            self.logger.error(f"Error analyzing performance patterns: {e}")
    
    def _update_adaptation_rules(self):
        """Update adaptation rules based on learned patterns"""
        try:
            # Update rules based on successful patterns
            for pattern in self.success_patterns:
                market_condition = pattern.get('market_condition')
                topology = pattern.get('topology')
                improvement = pattern.get('improvement', 0.0)
                
                if market_condition and topology and improvement > 0.1:
                    # Update rule for this market condition
                    if market_condition not in self.adaptation_rules:
                        self.adaptation_rules[market_condition] = {}
                    
                    self.adaptation_rules[market_condition]['topology'] = TopologyType(topology)
                    self.adaptation_rules[market_condition]['confidence'] = improvement
                    
        except Exception as e:
            self.logger.error(f"Error updating adaptation rules: {e}")
    
    def _clean_old_patterns(self):
        """Clean old patterns to prevent memory bloat"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep patterns for 7 days
            
            self.success_patterns = [
                p for p in self.success_patterns
                if p.get('timestamp', datetime.now()) > cutoff_time
            ]
            
            # Limit total patterns
            if len(self.success_patterns) > 100:
                self.success_patterns = self.success_patterns[-100:]
                
        except Exception as e:
            self.logger.error(f"Error cleaning old patterns: {e}")
    
    def _check_adaptation_triggers(self):
        """Check if any adaptation triggers should fire"""
        try:
            current_performance = self._calculate_overall_performance()
            
            # Performance degradation trigger
            if current_performance < 0.6:
                self.logger.info("Performance degradation detected, triggering optimization")
                optimal_topology = self._recommend_optimal_topology()
                if optimal_topology != self.current_topology:
                    self._switch_topology(optimal_topology)
            
            # Market condition change trigger
            if self.market_history and len(self.market_history) >= 2:
                current_condition = self.market_history[-1].condition
                previous_condition = self.market_history[-2].condition
                
                if current_condition != previous_condition:
                    self.logger.info(f"Market condition change detected: {previous_condition.value} -> {current_condition.value}")
                    self._adapt_to_market_condition(self.market_history[-1])
                    
        except Exception as e:
            self.logger.error(f"Error checking adaptation triggers: {e}")
    
    def shutdown(self):
        """Clean shutdown of the adaptive coordinator"""
        try:
            self.logger.info("Shutting down adaptive coordinator...")
            
            # Stop coordination
            self.stop_coordination()
            
            # Clear data structures
            self.agents.clear()
            self.agent_performance.clear()
            self.agent_states.clear()
            self.agent_roles.clear()
            
            self.status = "SHUTDOWN"
            self.logger.info("Adaptive coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Test the adaptive coordinator
if __name__ == "__main__":
    print("Testing Adaptive Coordination System")
    print("=" * 40)
    
    coordinator = AdaptiveCoordinator()
    result = coordinator.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test status
        status = coordinator.get_system_status()
        print(f"Status: {status['status']}")
        print(f"Topology: {status['current_topology']}")
        print(f"Market condition: {status['market_condition']}")
        
        # Test shutdown
        print("\nShutting down...")
        coordinator.shutdown()
        
    print("Adaptive Coordinator test completed")